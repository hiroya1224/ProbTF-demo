# 接触除外、PID・物理パラメータのリッジ、初回調整提案

## 1. この更新で解いた問題

案3では、次の四点を同時に扱う必要があった。

1. 空中 episode の中へ接地 sample が混ざり、自由飛行の parameter を汚さないこと。
2. command-response の実効係数だけでなく、実験者が実際に変更する PID 値へ
   つながる情報を出すこと。
3. 位置・姿勢だけでは分離できない actuator scale と質量・慣性を一点推定せず、
   識別不能リッジとして不確実性を残すこと。
4. 多数の bag を選びやすくし、処理の残り時間を表示すること。

実装は複雑な異常分布や hidden-state model を追加していない。episode ごとの
支持面、記録された controller 内部量、ロバスト回帰、moving-block bootstrap
だけを使う。

## 2. 接地 sample は posterior 任せにせず除外する

episode ごとに推定済みの支持面高さを \(h_0\)、そのロバスト標準偏差を
\(\sigma_h\) とする。自由飛行とみなす最小 clearance は

\[
\Delta h =
\max(0.02\ {\rm m},\,4\sigma_h)
\]

である。liftoff 後であっても、

\[
z(t) \le h_0 + \Delta h
\]

が 0.10 秒以上持続した sample は `support_contact` とする。この mask は
`fit_mask` から明示的に引き、`failure_diagnostic_intervals` へ残す。

これは「接地データを混ぜれば posterior が自然に消す」という設計ではない。
不整合 sample を混ぜたロバスト推定は、分布を広げるとは限らず、別の誤った係数へ
集中することもあるためである。接触らしいデータは推定に使わず、失敗原因を読む
ためのデータとして保存する。

通常は controller-active episode 直前の静止区間から \(h_0\) を得る。bag が
制御有効状態から始まり直前区間がない場合だけ、episode 冒頭の静止区間を使う。
出所はそれぞれ `pre_control_stationary`、
`initial_controlled_stationary` として JSON に残る。

## 3. bag から読む controller 情報

次の topic を command、IMU、odometry と同じ rosbag pass で読む。

- `/gimbalrotor/debug/pose/pid`
  - x, y, z, roll, pitch, yaw の `total`, `p_term`, `i_term`, `d_term`
- `/gimbalrotor/controller/xy/parameter_updates`
- `/gimbalrotor/controller/z/parameter_updates`
- `/gimbalrotor/controller/roll_pitch/parameter_updates`
- `/gimbalrotor/controller/yaw/parameter_updates`
  - 各 group の `p_gain`, `i_gain`, `d_gain`

Gimbalrotor controller の source では、並進 PID total は world 座標の desired
acceleration、roll/pitch/yaw total は CoG 座標の desired angular acceleration
として使われる。並進値は odometry quaternion で body / CoG 座標へ回し、
実効推定器と同じ座標に合わせる。

実効推定で選択された lag を \(\tau\) とし、応答時刻 \(t\) に対して PID と
command は \(t-\tau\) から読む。これにより、応答だけを遅らせて比較する。

## 4. 応答倍率と controller model 相当値

fit には `fit_mask` の自由飛行 sample だけを使う。軸ごとに、記録 PID total
\(u(t)\)、観測応答 \(y(t)\)、速度または角速度 \(v(t)\) から

\[
y(t) = b + r u(t) + c v(t) + \epsilon(t)
\]

を Huber IRLS で fit する。並進の \(y\) は IMU specific force、回転の \(y\) は
gyro を平滑化・微分した angular acceleration である。係数 \(r\) は
`actual response / recorded PID desired acceleration` という実効応答倍率になる。

さらに、source geometry で復元した command force / torque を \(q(t)\) として

\[
q(t) = b_q + M_c u(t) + \epsilon_q(t)
\]

を fit する。並進の \(M_c\) は controller mass 相当値、回転の \(M_c\) は
controller inertia 相当値である。これは bag と固定 source geometry から復元した
値であり、秤や CAD で独立測定した真値ではない。

両方の回帰に moving-block bootstrap を適用し、時系列相関を残した empirical
distribution と 95%区間を得る。これは厳密な Bayesian posterior ではないが、
一点推定へ潰さず、後段でリッジの幅を伝播できる。

x/y と roll/pitch は、それぞれの正の bootstrap sample の幾何平均で
`xy`、`roll_pitch` group にまとめる。z と yaw は単軸 group である。

## 5. 識別不能リッジを保持する

現在 controller model を基準にした物理 parameter 比を \(\rho\)、実 actuator
scale を \(\alpha\) とする。観測できる応答倍率は

\[
r = \frac{\alpha}{\rho}
\]

だけである。たとえば \(r\) が一定なら、質量を大きくして actuator scale も
同じだけ大きくした組はすべて同じ位置・姿勢応答を作る。したがって
\(\alpha\) と \(\rho\) を独立な一点として報告してはいけない。

JSON の `non_identifiability_ridge` は複数の \(\rho\) に対して

\[
\alpha = r\rho
\]

を保存する。各点の `actuator_scale_ci95` は \(r\) の bootstrap distribution を
push-forward した区間である。GUI では group 行を選ぶと、この式とリッジ上の点を
確認できる。

## 6. 「最小の修正」から初回候補を作る

PID の P/I/D を同じ倍率 \(s_{\rm pid}\)、controller model parameter を
\(s_{\rm model}\) 倍するとする。P/I/D を別々に識別する情報はこの閉ループ log
だけでは不足するため、三項間の形は変えない。

名目応答を1へ戻す理想条件は

\[
r\,s_{\rm pid}\,s_{\rm model}=1
\]

である。PID group と model parameter を二つの変更単位とみなし、

\[
\min\left[
(\log s_{\rm pid})^2+(\log s_{\rm model})^2
\right]
\]

を上の制約下で解くと、

\[
s_{\rm pid}=s_{\rm model}=\frac{1}{\sqrt{r}}
\]

となる。controller model 相当値が得られない場合だけ、PID 単独の
\(s_{\rm pid}=1/r\) を候補にする。

初回試行で大きく動かさないため、それぞれの倍率を `[0.8, 1.2]` に clamp する。
さらに次の保守的な判定を行う。

- \(r\) の95%区間が1を含む: `nominal_within_uncertainty` とし、現在値を維持。
- 入力励起、決定係数、区間幅、符号の根拠が不足: `weak_evidence` とし、現在値を維持。
- PID total と P+I+D の差である feedforward の相対 RMS が 0.5 を超える:
  P/I/D 同率変更の仮定が弱いため、現在値を維持。
- それ以外: `proposal_available` として clamp 後の初回候補を表示。

feedforward 自体は変更しない。この提案は安定性を保証する auto-tuning ではなく、
次の係留試験で一項目ずつ検証する候補である。実装は controller や bag へ値を
書き戻さない。

## 7. GUI の選択と ETA

bag 一覧には独立した `解析` 列を設け、未解析 bag を `[x]` / `[ ]` で選ぶ。
複数選択 dialog に加え、`フォルダを追加…` は選択フォルダ直下の `.bag` を
一括登録する。解析済み bag は `済` となり、累積結果には残すが再解析しない。

進捗は次の実作業から作る determinate 値である。

- rosbag message 時刻による読み込み位置
- file byte 数による SHA-256 進捗
- alignment delay 候補
- 6軸の moving-block bootstrap
- cumulative parameter trace
- controller ridge の6軸 bootstrap

複数 bag の全体進捗は file size で重み付けする。開始直後は 8 MiB/s の読み込みを
初期 prior とした所要時間を出し、進捗が得られたら
`elapsed / completed_fraction` による実測全時間へ滑らかに切り替える。そのため
従来の左右へ動くだけの bar と違い、`残り約` の時間が解析中に更新される。

## 8. 実 bag での確認

`20260612_grape_hovering_4_2026-06-12-17-33-59.bag` では、支持面は bag-local
z 約 0.1333 m と推定され、再接触した約 17.79–18.11 s が
`support_contact` として fit から除外された。PID topic と gain snapshot も読み、
4 group のリッジを生成できた。

この例では xy と roll/pitch は初回候補を出し、z と yaw は95%区間が1を含むため
現状維持となった。これは中央値だけを見て全軸を変更せず、bootstrap の不確実性を
実際の提案判断へ使っていることを示す。

## 9. 出力と限界

結果 schema は `grape_failure_automatic_result/v2` である。各 estimated episode
の `controller_advice` に、軸別 fit、group 別 response scale、現在 PID、
controller model 相当値、リッジ、初回判断と候補値を保存する。GUI と HTML report
は同じ値を表示する。

残る主な限界は次である。

- 閉ループデータなので、入力と状態・外乱の相関を完全には分離できない。
- command wrench は固定した source geometry に依存する。
- mass / inertia と actuator scale のリッジは意図的に解消していない。
- P/I/D の相対比、位相余裕、飽和・anti-windup はこの提案だけでは再設計しない。
- 初回候補を適用した後の安定性は、新しい安全な試験と bag で確認する必要がある。
