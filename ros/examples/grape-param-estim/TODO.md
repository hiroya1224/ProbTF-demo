# Grape rosbag Bayesian counterfactual analysis TODO

最終更新: 2026-07-24

## 目的

この解析の目的は真の質量や真の推力係数を一意に当てることではない。失敗・成功 rosbag から

1. controller が期待した軌道と実際の軌道の probabilistic gap、
2. その gap を説明する有効応答 parameter の posterior、
3. PID、controller 内 mass/inertia、allocation/推力 scale、delay compensation などを変えたとき、
   目標軌道の許容 tube に入る posterior probability、
4. 観測済みデータの support 内で「この範囲なら成功可能性が高い」といえる joint parameter region

を返すことである。出力は単一の「正解 gain」ではなく、

\[
q(\kappa)=P(\text{target tube を満たす}\mid D,r,\operatorname{do}(\kappa))
\]

と

\[
\mathcal K_\gamma=\{\kappa:q(\kappa)\ge\gamma\}
\]

である。未観測領域の候補は成功と断定せず `UNSUPPORTED` にする。この TODO は解析・提案までを対象とし、
実機 parameter の自動書換えや自動飛行試験は含めない。

ProbTF core の model-based temporal evaluation は
[`../../core/probtf_core/TODO.md`](../../core/probtf_core/TODO.md) に分離する。core 完成を待たず、
この package 内の明示的 adapter で MVP を作り、後から core API へ接続する。

## 固定する設計判断

- mocap pose の数値二階微分を独立な acceleration observation として尤度へ掛けない。
- mocap、IMU、ESC/RPM、PWM、gimbal、command、PID debug、battery を event time で融合する。
- gain 候補を評価するときは記録済み command を固定せず、PC 側と MCU 側を含む controller を閉ループ再計算する。
- 質量と推力 scale のような非識別な量は、個別の真値ではなく識別可能な比・有効係数・joint posterior として報告する。
- 時刻ごとの ProbTF marginal は可視化・時刻 query に使い、軌道全体の成功確率には時間相関を保つ trajectory sample を使う。
- retrospective diagnosis と、失敗前 prefix だけを用いる causal prediction を別 score として報告する。

## 候補の状態

`DEFAULT_CANDIDATE`、`OPTIONAL_CANDIDATE`、`OPTIONAL`、`PRUNE`、`EXPERIMENTAL` を用いる。
正式な default は Phase 2 の frozen evaluation 後にのみ決める。閾値を結果に合わせて動かさないよう、
bag hash、split、metric、seed、判定閾値を先に `config/selection_protocol.yaml` へ保存する。

---

## Phase 1: 実装そのものの TODO

### G0. データ manifest と再現性を固定する

- [ ] 全 rosbag の絶対 path、SHA-256、収録日時、duration、topic/type/count、開始終了 event time を
  `config/bag_manifest.yaml` に記録する。
- [ ] bag 内 `/rosparam`、controller debug、launch 設定、source commit から、当時の PID、機体 parameter、
  mixer/allocation、filter、mode を復元する。
- [ ] 復元できた値、一定と仮定した値、bag から本質的に欠落している値を区別する。
  欠落値をゼロや現在値で黙って補わず、latent parameter または `UNKNOWN` にする。
- [ ] Header stamp、bag record time、MCU time の対応と clock offset/drift を推定し、採用した event-time 規則を保存する。
- [ ] propeller 交換、battery、機体構成、software version、成功/失敗、dropout、gain change を episode metadata にする。
- [ ] 評価を見る前に episode 単位の train/validation/test split を固定する。同一 bag の隣接時刻を
  train と test に分けない。
- [ ] 現時点の仮ラベルである bag 3（dropout）、bag 4（attitude failure）、bag 7（success）、
  bag 8（gain sweep）を一次資料と照合して確定する。

### G1. event-time dataset と品質診断を実装する

- [ ] `src/grape_param_estim/episode.py` に topic reader、座標系/単位変換、motor/gimbal channel 対応、
  mode segmentation を集約する。
- [ ] mocap gap、IMU saturation、timestamp jump、PWM/ESC 欠測、mode switch を mask と event として保持する。
- [ ] desired pose、mocap pose、IMU、controller state、command、actuator telemetry を一つの
  episode timeline から query できるようにする。
- [ ] 生 topic を上書きせず、normalized dataset の config/source hash を保存する。
- [ ] Savitzky–Golay 等の微分値は可視化・初期診断 baseline としてのみ出し、主 likelihood には使わない。

### G2. actual trajectory posterior を作る

非線形性と dropout 時の差が実データ依存なので、次の backend を共通 interface で比較可能にする。

| Backend | Phase 1 の位置づけ | 特徴 |
|---|---|---|
| error-state EKF + RTS smoother | `DEFAULT_CANDIDATE` | 軽量、依存追加が少ない |
| factor graph + IMU preintegration | `OPTIONAL_CANDIDATE` | dropout、非同期 sensor、cross-time covariance に強い可能性 |

- [ ] `state_smoother.py` に共通の input/output contract を定義する。
- [ ] state は少なくとも pose、body velocity、angular velocity、IMU bias を持つ。
- [ ] mocap pose、gyro、accelerometer をそれぞれの timestamp と noise model で融合する。
- [ ] smoothed marginal だけでなく、同じ `sample_id` が全 timestamp を通る trajectory sample を出す。
- [ ] online-prefix mode では filter output だけを用い、未来を使う RTS/factor smoothing を禁止する。
- [ ] mocap dropout 前後、orientation wrap、stationary interval、aggressive attitude の test を作る。

### G3. 当時の controller を再現する

| 実装 | Phase 1 の位置づけ | 用途 |
|---|---|---|
| 既存 PC/MCU C++ controller の library 化・exact replay | `DEFAULT_CANDIDATE` かつ oracle | 正しい反実仮想 |
| vectorized Python surrogate | `OPTIONAL_CANDIDATE` | 大量 candidate の高速 sweep |

- [ ] `controller_replay.py` に `teacher_forced` と `free_run` を実装する。
- [ ] PC 側 `GimbalrotorController` と MCU 側 `AttitudeController` の P/I/D、saturation、
  anti-windup、mode reset、mixer/allocation、rate 差を含める。
- [ ] bag 内の当時の parameter と初期 integrator state を復元する。復元不能な内部 state は posterior に含める。
- [ ] \(\kappa_0\) で P/I/D debug、four-axes command、vectoring force、PWM が bag と一致する replay test を作る。
- [ ] Python surrogate は exact replay と同じ interface にし、候補ごとに backend を選べるようにする。
- [ ] gain を変えた rollout では controller state と command を最初から再計算する。

### G4. desired / nominal / actual の三軌道を明示する

- [ ] `desired`: controller に与えた reference から復元する。
- [ ] `nominal`: 当時の controller parameter と nominal 機体情報で `free_run` して復元する。
- [ ] `actual_posterior`: G2 の sensor-fused trajectory とする。
- [ ] \(\Delta T_{\rm track}=T_{\rm desired}^{-1}T_{\rm actual}\) と
  \(\Delta T_{\rm model}=T_{\rm nominal}^{-1}T_{\rm actual}\) を同一 sample 内で計算する。
- [ ] nominal と actual の共通初期状態・noise を保ち、covariance を独立と仮定して単純加算しない。

### G5. 有効機体応答モデルを実装する

第一候補は物理真値ではなく、制御に効く低次元の有効モデルである。

\[
x_{t+1}=F_\Delta(x_t,u_t;\eta)+w_t
\]

の \(\eta\) に、軸別 effectiveness、cross-coupling、motor/gimbal delay と時定数、角速度 damping、
bias、battery/RPM dependence、episode random effect を含める。

| Model | Phase 1 の位置づけ | 実装方針 |
|---|---|---|
| low-dimensional effective response | `DEFAULT_CANDIDATE` | 最初に全 bag で fit |
| structured 6-DoF mechanics | `OPTIONAL_CANDIDATE` | mass/inertia/thrust の gauge を明示した比較 baseline |
| effective response + sparse GP residual | 条件付き `OPTIONAL_CANDIDATE` | parametric held-out residual に構造が残る場合だけ実装 |

- [ ] `effective_response.py` に全 model の共通 transition/log-likelihood interface を作る。
- [ ] episode 間を \(\eta_e\sim N(\bar\eta,\Sigma_{\rm episode})\) とする階層モデルを入れる。
- [ ] pose を \(\eta\) で一段積分した予測と mocap/IMU observation を比較し、heavy-tail 外れ値 model を持たせる。
- [ ] posterior correlation、Fisher/Hessian または sample rank から非識別方向を検出し、
  個別値でなく比・積・joint region を報告する。
- [ ] structured mechanics は calibrated wrench が復元可能な区間と不可能な区間を分ける。
- [ ] sparse GP は state/action 全域へ自由外挿せず、training support 外で variance を広げる。

### G6. Bayesian inference backend を実装する

| Backend | Phase 1 の位置づけ | 実装範囲 |
|---|---|---|
| modular smoother + tempered resample-move SMC | `DEFAULT_CANDIDATE` | MVP と全 bag |
| joint PMCMC | `OPTIONAL_CANDIDATE` | synthetic と代表的な失敗/成功 bag の最小比較 |
| likelihood-free BayesSim | 条件付き `OPTIONAL_CANDIDATE` | likelihood が妥当化できず、black-box simulator が検証済みの場合だけ |

- [ ] 現在の particle filter から parameter transform、prior、ESS、resample、MCMC rejuvenation を再利用可能にする。
- [ ] trajectory uncertainty を point estimate に潰さず、trajectory sample または likelihood marginalization で渡す。
- [ ] seed 別 convergence、ESS、R-hat/chain mixing 相当、posterior predictive check を保存する。
- [ ] synthetic known-truth で recovery だけでなく credible interval coverage を検証する。
- [ ] PMCMC は全機能を作る前に、一つの共通 model で modular SMC との差を測れる vertical slice を完成させる。
- [ ] BayesSim は simulator 自体が held-out 実 bag を再現できるという前提 test を通った後だけ着手する。

### G7. closed-loop counterfactual evaluator を実装する

- [ ] `counterfactual.py` に candidate \(\kappa\)、posterior \(\eta\)、初期 trajectory sample、
  process noise を受け取る closed-loop rollout を実装する。
- [ ] candidate には PID だけでなく controller 内 mass/inertia、allocation/thrust scale、
  delay compensation を joint に指定できるようにする。
- [ ] target tube を位置、姿勢、速度、角速度、saturation 継続時間、ground/contact、安全限界で明示する。
- [ ] 最初は grid/Sobol で候補空間を覆い、\(q(\kappa)\)、credible interval、
  \(\mathcal K_\gamma\) を計算する。
- [ ] 観測 input/state/parameter support からの距離、importance weight ESS、posterior predictive uncertainty により
  `SUPPORTED` / `EXTRAPOLATIVE` / `UNSUPPORTED` を付ける。
- [ ] 推奨値は posterior mean 最大点だけでなく、成功確率の lower credible bound が高い連結領域として返す。
- [ ] offline candidate 数が大きく、Python surrogate を使っても計算量が支配的な場合だけ
  Bayesian optimization を optional accelerator として追加する。
- [ ] 実機への次回値書込みは行わず、human-reviewed proposal artifact までにする。

### G8. ProbTF と解析 artifact を出力する

- [ ] 時刻ごとに `world -> desired/cog`、`world -> nominal/cog`、
  `world -> actual_posterior/cog`、`world -> counterfactual/<candidate>/cog` を出す。
- [ ] `nominal/cog -> actual_posterior/cog` と、その SE(3) log residual の uncertainty を出す。
- [ ] application message として次を追加する。
  - `TrajectoryParticleSet.msg`: trajectory/sample ID、timestamped transforms、weight、candidate、provenance
  - `ModelMismatch.msg`: tracking/model residual、credible interval、diagnostic
  - `CounterfactualCandidate.msg`: candidate、\(q\)、credible interval、support、constraint violations
- [ ] 元 bag を不変に保ち、解析 topic を record-time 順に merge した新しい analysis bag を生成する。
- [ ] machine-readable CSV/Parquet/JSON と、図・採否理由を含む report を同じ run ID で保存する。
- [ ] source bag hash、topic、time interval、config hash、source commit、model version、seed を全 artifact に持たせる。

### G9. 最初の vertical slice を完成する

- [ ] bag 4 で desired / nominal / actual posterior と \(\Delta T_{\rm model}(t)\) を出す。
- [ ] pitch/roll effectiveness、delay、cross-coupling の posterior を出す。
- [ ] roll/pitch P–D と allocation parameter の小さな joint grid を exact controller で閉ループ評価する。
- [ ] target-tube probability と `UNSUPPORTED` 領域を一枚の contour/report にする。
- [ ] 同じ pipeline を bag 7、8 に設定変更だけで適用する。

---

## Phase 2: 実装後の比較・選定 TODO

### V0. 比較 protocol を凍結する

- [ ] `config/selection_protocol.yaml` に bag hash、episode split、failure time、prefix cutoff、
  candidate bounds、seed、metrics、初期判定閾値を保存する。
- [ ] synthetic、成功、制御失敗、sensor dropout、gain sweep を別 strata として集計する。
- [ ] leave-one-bag-out を基本とし、同一 flight の時刻点を独立 sample 数として数えない。
- [ ] 全候補へ同じ trajectory samples、candidate grid、random numbers を可能な限り用いる。
- [ ] episode/bag 単位 bootstrap 95% CI と、最良との差の 1-standard-error を出す。

### V1. 共通 hard gate

一つでも満たさない候補は性能が高く見えても default にしない。

- [ ] online-prefix 評価で cutoff 後の message、smoother result、failure label を使わない。
- [ ] frame、単位、motor order、timestamp、controller mode が bag と一致する。
- [ ] NaN、非物理的 quaternion、負の分散、weight collapse を未診断のまま返さない。
- [ ] synthetic 95% credible interval の empirical coverage について binomial 95% CI が 0.95 を含む。
- [ ] 実 bag の 50/80/95% predictive interval が単調かつ概ね calibration される。
- [ ] support 外候補を高確率の「推奨」として返さない。
- [ ] 同じ run が hash と seed から再現できる。

### V2. trajectory smoother を選ぶ

評価値:

- held-out mocap gap の position/orientation RMSE、log predictive density、50/80/95% coverage
- IMU innovation の bias と whiteness
- dropout 後の復帰時間、failure 前 prefix の drift
- p50/p95 runtime、peak memory、依存 package と保守量

判定:

- [ ] hard gate 通過後、held-out score が最良の 1 standard error 以内なら単純な EKF+RTS を default にする。
- [ ] factor graph が dropout または aggressive-flight strata で log score/RMSE を 10% 以上改善し、
  bootstrap CI が 0 を跨がなければ optional に残す。
- [ ] factor graph が calibration を改善しなくても常に遅い、または依存関係に対する固有の利点がなければ prune する。
- [ ] 両者が用途別に勝つなら EKF+RTS を default、factor graph を
  `backend=factor_graph` の明示 option とし、自動切替条件は dropout 長など観測可能量で定義する。

### V3. controller replay 実装を選ぶ

exact C++ replay は oracle なので、bag 再現 gate を通らなければ解析全体を先へ進めない。

- [ ] \(\kappa_0\) で P/I/D 項、command、gimbal force、PWM の normalized RMSE、最大誤差、
  saturation/mode-event 一致率を測る。
- [ ] 初期閾値として continuous output の normalized RMSE 1% 以下、最大誤差 3% 以下、
  discrete event 100% 一致を `selection_protocol.yaml` に登録し、実験前に単位別許容値へ確定する。
- [ ] Python surrogate が上記 gate を通り、candidate sweep を 5 倍以上高速化するなら optional に残す。
- [ ] surrogate が gate を通らない場合は counterfactual 用から prune し、図示/初期探索にも
  「近似」と明記できない限り使わない。
- [ ] surrogate と exact が同等でない領域では、surrogate で候補を絞っても最終 \(q(\kappa)\) は exact で再評価する。

### V4. 機体応答モデルを選ぶ

評価値:

- held-out 1-step / multi-step trajectory log predictive density と RMSE
- predictive interval coverage と innovation whiteness
- 成功/失敗および target-tube event の Brier score、calibration curve
- episode、battery、propeller 交換を跨ぐ transfer
- posterior の非識別 rank、support 外 rollout の安定性

判定:

- [ ] low-dimensional effective model が最良の 1 standard error 以内なら、解釈・同定容易性から default にする。
- [ ] structured mechanics が二つ以上の独立 held-out episode で multi-step log score または
  counterfactual Brier score を 10% 以上改善するか、propeller/battery を跨ぐ transfer でのみ明確に勝つなら
  optional に残す。
- [ ] mechanics の個別 mass/thrust 値が非識別で prediction も改善しない場合、物理値推定を prune し、
  識別可能な有効比だけを残す。
- [ ] parametric model の held-out residual に対する whiteness/independence test が棄却された場合だけ GP 比較を開始する。
- [ ] GP が二つ以上の held-out episode で log score または Brier score を 10% 以上改善し、
  calibration を悪化させず、support 外で uncertainty を増やすなら optional に残す。
- [ ] GP の改善が train 内だけ、または effectiveness と discrepancy の gauge を悪化させるなら prune する。

### V5. inference backend を選ぶ

評価値:

- synthetic parameter/識別可能比の bias と credible interval coverage
- seed 間の posterior 距離、ESS、mixing、failure rate
- real held-out predictive log score、coverage、counterfactual Brier score
- wall time、peak memory、1 candidate 当たりの計算量

判定:

- [ ] modular SMC が最良の 1 standard error 以内なら default にする。
- [ ] PMCMC が posterior coverage または held-out score を 10% 以上改善するなら offline high-accuracy option に残す。
- [ ] PMCMC が 10 倍以上高コストで品質差が 1 standard error 未満なら public workflow から prune し、
  小規模 validation oracle としてだけ残す。
- [ ] likelihood-based 候補が全て calibration gate を満たさず、検証済み simulator がある場合だけ BayesSim を比較する。
- [ ] BayesSim が held-out real score を改善しない、または simulator bias を狭い posterior に隠す場合は prune する。

### V6. counterfactual の有効性を判定する

反実仮想には未試行候補の直接真値がないため、次の順で検証する。

- [ ] **factual replay:** 観測された \(\kappa_0\) の軌道と成功/失敗を posterior predictive が覆う。
- [ ] **held-out episode:** 一つの bag を完全に除外して fit し、その軌道 log score、target-tube event、
  success probability を予測する。
- [ ] **gain-sweep check:** bag 8 等の実際に gain が変わった区間/episode を隠し、候補順位と実結果の
  Spearman rank、Brier score、calibration を測る。
- [ ] **failure-prefix check:** failure 前 prefix のみから、警告時刻、false alarm/flight、
  sensor dropout と dynamics mismatch の識別を測る。
- [ ] climatology、nominal deterministic model、least-squares effective model、現在の PID を baseline にする。
- [ ] Bayesian evaluator が held-out Brier/log score を baseline より改善し、その bootstrap CI が 0 を跨がず、
  target-tube coverage gate を通った場合だけ「次回候補の確率的提案に有用」と判定する。
- [ ] 候補順位は合うが絶対確率が未 calibration なら、順位付け tool として optional に残し、
  成功確率という表示は prune する。
- [ ] factual/held-out の両方で baseline を改善しなければ counterfactual recommendation を prune し、
  model-mismatch diagnosis だけを成果として残す。
- [ ] 遠い未試行候補は、性能が良く見えても `UNSUPPORTED` のままにし、実 flight の成功主張に使わない。

### V7. 最終 default と optional を確定する

- [ ] `SELECTION_RESULTS.md` に commit、bag/config hash、候補ごとの metric/CI、hard-gate 結果、
  default/optional/prune の理由を残す。
- [ ] hard gate 後は一標準誤差則で、同等なら実装・依存・runtime が小さい候補を default にする。
- [ ] 特定 strata で統計的かつ実用的な改善がある候補は optional に残し、適用条件を config で選べるようにする。
- [ ] 全 strata で Pareto dominated な候補は公開 config、launch、docs から prune する。
- [ ] 判別不能なら `EXPERIMENTAL` とし、default を変えず、必要な追加 flight 条件と sample 数を記録する。
- [ ] 二つの候補が共に固有の利点を持つ場合、より広い strata で勝ち、単純な方を default にする。
- [ ] 選定後、使われない分岐を残したままにせず、CI matrix と保守担当を default/optional のみに絞る。

## 推奨する実装順

依存関係と検証可能性から、次の順を default とする。

1. G0–G1: manifest、event time、split を凍結する。
2. G2 と G3: actual posterior と exact controller replay を成立させる。
3. G4: desired / nominal / actual と model gap を bag 4 で可視化する。
4. G5–G6: 低次元 effective response + modular SMC を MVP とする。
5. G7–G9: 小さい joint grid で closed-loop counterfactual を end-to-end にする。
6. V0–V7: frozen held-out 評価で候補を選ぶ。
7. core temporal API が完成したら adapter を置換し、同じ artifact hash/metric が保たれることを確認する。

## 完了条件

- [ ] 失敗 bag について「期待軌道との差」と uncertainty を ProbTF 表現で追える。
- [ ] 成功・失敗・gain-sweep の held-out bag に対し、予測確率の calibration を測れる。
- [ ] 次回候補を点ではなく joint region、credible interval、support label として出せる。
- [ ] どの option を default、optional、prune にしたかが、再実行可能な Phase 2 結果から説明できる。
- [ ] 真値の非識別性、missing telemetry、counterfactual extrapolation を成功確率の中に隠さない。
