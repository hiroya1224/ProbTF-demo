# 失敗 bag の自動 episode 抽出と実効 parameter 推定

## 1. 目的と初期版の範囲

この実装は、開始時刻と duration を人が指定する代わりに、ROS 1 bag 内の
`flight_state`、command、odometry、IMU から解析可能な episode を抽出する。
絶対 z、正常飛行を学習した異常分布、複雑な hidden-state model は使わない。

処理は次の二つを混同しない。

1. flight controller が有効であること
2. 自由飛行の parameter fit に使えること

controller-active だが支持面に載っている sample、force landing、command 欠損、
持続的な model mismatch は消去せず、失敗診断用の interval として保持する。

## 2. 入力と時刻

各 bag から次の5 topic だけを直接読む。

- `/gimbalrotor/flight_state`
- `/gimbalrotor/four_axes/command`
- `/gimbalrotor/gimbals_ctrl`
- `/gimbalrotor/sensor_plugin/imu1/ros_converted`
- `/gimbalrotor/uav/cog/odom`

Header stamp が正なら event time、Header がなければ bag record time を使い、
すべて bag 開始からの秒へ変換する。重複時刻は安定 sort 後の先頭 sample に
正規化する。入力 bag は変更せず、実際に読んだ bag の SHA-256 を結果へ残す。

## 3. controller-active episode

既定の active state は次である。

| value | state |
|---:|---|
| 3 | TAKEOFF |
| 4 | LAND |
| 5 | HOVER |
| 17 | FORCE_LANDING |

odometry 時刻へ flight state を zero-order hold し、active state が連続する
run を episode とする。非常に短い run だけを除く。command topic の一時欠損で
episode 自体は分割しない。flight state は controller の意図を表す一次情報で
あり、物理的に浮上済みであることまでは意味しないからである。

command の有効性は、bag 内で観測された command の中央値 sample period の
5倍以内に新しい command があるかで別途判定する。欠損中に最後の command を
無制限に保持して fit することはしない。

## 4. episode-local support と liftoff

各 episode の開始直前 `baseline_window_s` にある非 active odometry を候補とし、
速度 norm の中央値と MAD から静止 sample を選ぶ。支持面高さは、その静止
sample の z 中央値

\[
z_\mathrm{support}=\mathrm{median}(z_i)
\]

とする。ばらつきは

\[
\sigma_z=1.4826\,\mathrm{median}
\left|z_i-\mathrm{median}(z_i)\right|
\]

で求める。これを episode ごとに再推定するため、台の追加や mocap 原点の違いに
依存しない。

liftoff は、相対高度が

\[
z-z_\mathrm{support}>
\max(h_\mathrm{min}, k\sigma_z)
\]

を満たし、同じ persistence window 内で上向き速度が静止時の robust scale
より有意に大きい状態が継続した最初の時刻とする。既定値は
`h_min=0.02 m`、`k=4`、persistence `0.10 s` である。`0.02 m` は絶対高度では
なく、mocap の微小 jitter を物理的な浮上と誤認しないための相対変位 floor
である。

支持面を復元できない、liftoff がない、または浮上後の有効時間が短い episode は
`not_identifiable` とする。bag が空中から開始した場合も、初期版は推測で
支持面を補わず fail closed とする。

## 5. command-response model

rotor thrust command と gimbal angle は source に固定した Grape geometry で
body 6軸 command wrench \(u\) へ変換する。command は未校正なので物理 wrench
ではない。

並進3軸では IMU specific force、回転3軸では平滑化 gyro の時間微分を response
\(y\) とし、各軸を独立に

\[
y_t=b+g\,u_{t-\tau}+d\,x_t+\epsilon_t
\]

で fit する。\(b\) は bias、\(g\) は effective gain、\(d\) は velocity または
angular-velocity feedback である。共通 lag \(\tau\) は候補 grid の normalized
RMSE 平均が最小となる値を選ぶ。

係数は Huber IRLS で求め、最終95%区間には時系列相関を完全に独立 sample と
みなさない moving-block bootstrap を使う。これは calibrated physical
mass/inertia の推定でも、Bayesian posterior でもない。

## 6. fit mask と failure diagnostic mask

まず、liftoff 後で command heartbeat が有効かつ diagnostic flight state でない
sample から preliminary robust fit を作る。既定では FORCE_LANDING `17` を
diagnostic state とし、controller-active episode には残すが fit には使わない。

preliminary residual は軸ごとに中央値を引き、MAD と標準偏差から robust scale
を作る。6軸の standardized residual の RMS が設定値を超えた状態が persistence
時間以上続く run を `persistent_model_mismatch` とする。その sample を外して
一度再 fit し、mask が変わればもう一度だけ fit する。これは正常データ集合を
別途学習する異常検知ではなく、同じ episode 内の力学 model に対する持続的な
不整合の表示である。

mask 探索中の preliminary fit は点推定だけを計算する。mask が確定した後に
moving-block bootstrap を一度だけ実行するため、途中の候補 mask ごとに同じ
不確実性計算を繰り返さない。

最終的な分類は次である。

- `fit_intervals`: parameter fit に実際に使用した sample
- `controlled_supported`: controller-active だが liftoff 前
- `diagnostic_flight_state_17`: force landing
- `missing_command`: command heartbeat が無効
- `persistent_model_mismatch`: model residual が持続
- `outside_command_covered_model_interval`: fit に必要な command coverage 外

不整合な sample を入れたときに Huber fit や正規化 posterior が自然に
「潰える」ことには依存しない。モデルが誤っていても係数が最もましな値へ集中し、
狭い区間を返す場合があるためである。

## 7. parameter の時間推移

最終 `fit_mask` の sample を時刻順に累積し、一定間隔ごとに同じ Huber model を
再 fit する。GUI に表示する trace はこの cumulative estimate であり、各点で
block bootstrap を繰り返すものではない。最終値だけが block-bootstrap 95%
interval を持つ。長時間 episode では再計算が二次的に増えないよう、設定した
step を下限としつつ trace を最大120点に制限する。これは最終 fit の sample を
間引く処理ではなく、GUI用の途中経過を評価する時刻だけを減らす処理である。

複数 bag の時刻は入力順に連結して GUI に表示する。ただし各 episode は独立に
fit する。機体構成、台、controller 設定が異なる試行を、同一 parameter を持つ
ものとして暗黙に同化しないためである。

## 8. 対話 GUI と累積セッション

通常の操作入口は Tkinter GUI `failure_analysis_gui.py` とした。起動後の操作は
ファイル選択と解析実行だけであり、bag path、出力 path、固定時刻をコマンドライン
へ入力しない。推定は worker thread で行うため、処理中も window の event loop と
進捗バーは応答を続ける。

時系列 plot と parameter trace は Matplotlib の Tk canvas を直接埋め込む。
したがって HTML や静的 SVG を介さず、標準 toolbar による pan、zoom、表示履歴、
画像保存を使用できる。別 tab の表には episode の判定理由と最終 parameter、
block-bootstrap 95% 区間、information grade を表示する。

GUI session は選択された path を次の3状態で管理する。

- `pending`: 選択済みだが未解析
- `completed`: 累積結果へ追加済み
- `error`: 読み込みまたは解析に失敗し、次回に再試行可能

二回目以降の実行では `pending` と `error` だけを解析する。新しい bag は独立に
episode 抽出・fit した後、既存の `sequence_duration_s` を新しい
`sequence_offset_s` と trace 時刻へ加算する。旧 bag の bootstrap を再実行
しないため、追加操作の計算量は新しい bag の分だけである。この結合でも episode
間の parameter は pool しない。

各 bag が完了するたび、累積 `analysis.json` を一時ファイルへ書いて `fsync` 後に
rename する。途中の bag が失敗した場合や GUI を閉じた場合にも、それ以前に完了
した結果を壊さない。既定の保存先は次であり、実 path と保存先を開く button は
GUI 内に表示する。

```text
${ROS_HOME:-~/.ros}/grape_param_estim/failure_analysis/YYYYMMDD-HHMMSS/
```

同じ config hash と result schema の結果だけを結合する。結合後は bag index、
sequence time、bag count、duration、result hash を再計算する。

## 9. browser report と再現性

`analyze_failure_bags.py` は次を生成する。

- `analysis.json`: 全入力 hash、config hash、episode、mask、推定値、trace
- `report.html`: matplotlib SVG を inline で格納した自己完結 report

browser report は外部 CDN や実行中の web server を必要としない。時系列には
高度、速度、specific-force norm、angular-rate norm、vertical command、
flight state と、fit/diagnostic/liftoff を重ねる。parameter trace は並進と
回転を別 scale で表示し、episode の詳細には最終係数、区間、残差 score を
表示する。

GUI が主操作経路になった後も、この CLI と browser report は headless batch、
比較、監査用として残す。

## 10. 既知の制約

- flight state topic が欠けた bag は自動抽出できない。
- 空中から収録開始した bag は支持面を復元できず `not_identifiable` になる。
- command-to-force calibration がないため、残差を物理的な外力 N と呼べない。
- 対角 model は軸間 coupling を表現しない。
- closed-loop command は内生変数なので、gain と lag は診断的な関連量である。
- 線形 velocity feedback は実際の空力、controller、未モデル化 coupling を
  分離しない。
- residual threshold と persistence は少数の明示的な設定として残る。

この初期版で不足が確認されるまでは、学習済み異常分布、particle filter による
見かけの posterior、複雑な hidden semi-Markov model は追加しない。
