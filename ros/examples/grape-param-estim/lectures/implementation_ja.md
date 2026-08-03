# Grape 実機同化・PID 候補評価・GUI 実装

## 1. 実装の範囲

`grape_param_estim` は、Grape の記録済み飛行を検査し、選択された一つ以上の飛行窓を一回の joint smoothing で同化し、その raw posterior ensemble から次回飛行の exact PID gain 候補を導出・検証するパッケージである。変更対象はこのパッケージ内に限定している。

処理は次の成果物を境界として分離した。

1. inspection bundle: rosbag の topic 契約、完全な飛行 episode、連続区間候補、controller snapshot、構成 fingerprint、軽量 preview
2. assimilation run: shared static posterior、bag-local trajectory、residual wrench、correction-transform path、ridge・Q・反復診断
3. PID proposal evaluation: raw PID proposal ensemble、評価した exact candidate、candidate × bag × physical member の full closed-loop forecast、物理単位別 metric、Pareto 判定、YAML 表示用テキスト
4. project archive: project JSON、GUI state、生の全登録 rosbag、上記成果物を含む標準 ZIP / ZIP64

各 loader は schema、status、dtype、shape、member ID、quaternion、有限性、成果物間の参照を検証する。`writing` または `cancelled` の bundle は完成結果として読み込まない。旧形式を推測して読み替える処理は置いていない。

## 2. 推定問題

尤度へ入れる観測は、各時刻の position と orientation のみである。twist、IMU、加速度、actuator command、PID debug は、初期状態、既知 controller の復元、時刻整合、Q の事前校正、診断には使うが、観測成分へ追加しない。

forecast operator は次の閉ループ全体を毎 member で積分する。

```text
recorded reference
  -> stateful full controller
  -> allocation and actuator dynamics
  -> continuous constant delay
  -> full six-DoF plant
  -> pose feedback
```

遅れは非負の連続値として shared parameter chart に含める。controller command は timestamp 付き履歴へ格納し、`u(t - delay)` を zero-order hold で取得する。publish period 未満の遅れも整数 sample へ丸めない。

構成 fingerprint が一致する bag では、次の 19 成分を全 bag で共有する。

- mass 1
- full symmetric positive-definite inertia 6
- CoG offset 3
- rotor force effectiveness 4
- rotor torque effectiveness 4
- continuous constant delay 1

初期 pose 補正、初期 velocity / angular velocity、controller integral state、actuator state、OU residual-wrench innovation、latent trajectory は bag ごとに独立である。bag ごとの posterior を後から平均せず、全 bag の pose residual を一つの likelihood vector に連結して一回だけ ensemble-space IEnKS-Q を解く。アンサンブル数が augmented control dimension より小さい場合も、実際の prior ensemble span 内で解く。

要求 prior ensemble の forecast が数値的に有限でない場合は、その member だけを除外したり、
有限な罰 residual へ置換したりしない。全 member の prior center からの偏差を同じ dyadic
係数 `1/2, 1/4, ...` で縮め、全 forecast が有限になる最初の scale を実効 prior とする。
この global radial backoff は観測への fit の良し悪しを使わず、member ID、center、方向、
rank を保存する。要求 / 実効 ensemble、試行 scale、例外、採用 scale は artifact に保存し、
scale が 1 未満なら GUI に warning を出す。これは数値 feasibility のために prior 自体を
狭める操作なので、警告を消して元の prior と同一視してはならない。

## 3. rosbag inspection と controller snapshot

inspection worker は rosbag record time を正本にし、takeoff から stop まで完結した episode のみを列挙する。強制着陸 state を含む episode も stop まで同じ飛行として扱う。平常飛行 state の最長連続区間を優先し、存在しない場合は control-active な連続区間を明示的な警告付きで返す。

PID の current 値は、各 bag の記録済み dynamic-reconfigure update から group ごとに復元する。数値を別ファイルから補完しない。複数 bag の snapshot が異なる場合、GUI は baseline bag の明示選択を要求し、平均しない。snapshot の bag ID、topic、record time、source kind は PID evaluation に保存する。

dynamic_reconfigure は、利用者が決定した gain を controller へ書き込む機構に限る。parameter
推定、PID 候補生成、候補選択は行わず、GUI も自動書込みをしない。

構成 fingerprint は payload、rotor / propeller、geometry、robot model revision、actuator wiring、hardware revision から作る。情報が不足する inspection は `needs_configuration_confirmation` となり、GUI から識別情報を確認して再 inspection するまで joint run の対象にできない。

## 4. raw member と不定座標変換

shared posterior の member ID は、全 bag の次の値で共通である。

- physical parameter と constant delay
- initial state、controller state、actuator state
- posterior / prior trajectory
- residual-wrench knot / interval path
- nominal から member trajectory への correction translation / rotation vector
- PID proposal の source member
- candidate forecast と metric

観測 correction は `nominal^-1 * observed`、posterior correction は `nominal^-1 * member` として保存する。parameter の周辺平均だけを正本にはしない。GUI の member 選択は Master、Bag browser、Next experiment の全画面で同じ raw ID を指す。

## 5. PID 候補の生成と検証

PID 自体は過去飛行の同化変数ではない。各 physical posterior member について、固定した controller nominal mass、full inertia、geometry、allocation と、member の mass、full inertia、CoG、force / torque effectiveness から局所的な並進・回転 acceleration response を組み立てる。`xy`、`z`、`roll_pitch`、`yaw` の group response を最小二乗で合わせ、記録済み current P/I/D を同じ group scale で拡大・縮小する。したがって各 member-derived proposal は source member ID を持つ exact 4 × 3 gain 行列であり、P:I:D 比を保つ。PID limit と controller nominal model は変更しない。

評価対象は必ず current を含み、ユーザが明示した member-derived candidate または exact user candidate を追加する。全 candidate を、全 selected bag と全 raw physical member の組合せで同じ full closed-loop operatorへ通す。static parameter と constant delay は member の値を使い、bag-local initial state と posterior residual-wrench path は同じ member ID の値を使う。residual policy は `posterior_replay` または `zero` として成果物へ保存する。

評価量は次を分離して保持する。

- time-integrated position RMSE [m]
- time-integrated orientation RMSE [rad]
- maximum position error [m]
- maximum orientation error [rad]
- forecast completion
- numerical failure count と member ごとの reason
- correction path の zero-coverage 診断
- current からの log gain change

各 metric は先に bag ごとの raw member lawから mean と upper CVaR を求め、その後 bag を等重みで集約する。position と orientation を足した score は作らない。設定済みの独立指標、completion、failure、gain change を使って Pareto dominated / non-dominated を判定する。位置・姿勢 threshold は利用者が値を指定した場合だけ別々に評価し、未設定なら `Not configured` のまま Pareto 判定から外す。

raw proposal ensemble から代表 member を自動選択しない。候補が明示選択され、current に対して全独立指標で悪化せず少なくとも一つで改善し、かつ Pareto non-dominated の場合だけ recommendation を成立させる。条件を満たさなければ YAML 表示は current、diff は空となる。

## 6. Desktop GUI と project

GUI は `Master`、`Bag browser`、`Next experiment` の三画面を持つ。

- Master: joint run へ入れる bag、構成 fingerprint、AUTO / MODIFIED / LOCKED 区間、raw shared member、ridge、bag contribution、収束・Q 警告
- Bag browser: native file selection、実 preview、trajectory 上の区間編集、observed / reference / nominal / selected member、residual と correction path、3D pose
- Next experiment: 明示 member からの PID evaluation 実行、current / proposed / difference / ratio / raw range、Pareto・推薦状態、aggregate / bag / member metric、current / candidate correction path、3D comparison、read-only YAML / diff

inspection、assimilation、PID evaluation は GUI process 内で計算せず、ROS Python worker を別 process として起動する。worker stdout は typed JSON Lines progress 専用、stderr は log 専用である。進捗 fraction は単調で、member / bag、反復、ETA を表示する。cancel は cooperative signal を送り、一定時間後に terminate、最後に kill する。cancelled bundle は GUI へ load しない。

project の freshness fingerprint は selected bag IDs、各 bag SHA256、選択区間、controller snapshot、構成 fingerprint、estimator settings を含む。いずれかを変えると既存 run は `STALE` となる。

Save Project は working project 全体を `allowZip64=True` の標準 ZIP へ atomic replace で保存する。Load Project は absolute path、`..`、drive prefix、symlink、重複 entry、暗号化 entry、過大展開を拒否し、同梱 raw bag の SHA256 を再検証する。同じ project ID が存在する場合は別 ID の working project として復元する。

## 7. 実行環境

ROS worker は catkin workspace の `/usr/bin/python3` と ROS message package を使う。GUI は PySide6、pyqtgraph、PyVistaQt、VTK を持つ Python 3.10 以上の環境を使い、worker interpreter とは分離する。GUI launcher は source package と installed worker の双方を探索し、worker interpreter は `GRAPE_PARAM_ESTIM_WORKER_PYTHON` で明示変更できる。

本節の GUI 記述は実装された code contract である。本ホストは Python 3.8.10 で、Python 3.10
以上の GUI interpreter と PySide6、pyqtgraph、PyVista、PyVistaQt がないため、Qt widget / 3D
描画を起動した視覚的な受入確認は未実施である。Qt 非依存 contract test と artifact loader
test の結果、およびこの制約は実飛行検証報告に記録する。

機械的な artifact validation の成功は、推定の科学的妥当性、posterior coverage、次回実機飛行の安全を保証しない。synthetic truth、held-out flight、Q resolution、coverage、prior predictive stability を別々に確認する必要がある。実 rosbag を使った今回の確認結果は [実飛行検証報告](real_flight_validation_ja.md) に分離した。

## 8. 成功飛行の完全 hold-out validation

成功飛行の pose を失敗飛行の同化へ混入させず、完成した assimilation run から
`member_id`、mass、full inertia、CoG、force / torque effectiveness、constant delay だけを
移送する validation worker を用意した。source bag の initial state、controller / actuator
state、reference、Q path は移送しない。

held-out bag では記録済み reference と controller snapshot を既知条件とし、先頭 pose samples
だけから初期 pose / velocity anchor を作る。全 member と固定 nominal baseline を zero residual
wrench で全区間 closed-loop forecast し、観測による途中 reset は行わない。forecast API は
target observations を保持せず、scoring API で初めて observed pose と reference の双方に
対する position / orientation の RMSE と maximum error を別々に計算する。

request は source run status、bag SHA256、configuration fingerprint、interval を検査する。
artifact は全 raw trajectory、error path、metric、failure reason と provenance を保存し、strict
loader は quaternion、member alignment、success / NaN 契約を検査したうえで error と metric を
再計算する。検証結果は [実飛行検証報告](real_flight_validation_ja.md) に記録した。
