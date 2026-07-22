# Grape 動力学パラメータ推定の設計と実装

この文書は `grape_param_estim` パッケージが実装しているオフライン推定を、数式、データ処理、ROS bag の入出力、ProbTF 表現まで含めて説明する。実行手順だけを知りたい場合はパッケージ直下の `README.md` を参照すること。本書は、結果をどこまで物理パラメータとして解釈できるかを判断するための資料である。

## 1. 目的と現在の実装範囲

推定対象は、Grape の `fc` 座標系で表した一つの剛体の慣性パラメータである。

\[
\theta =
[m,c_x,c_y,c_z,I_{xx},I_{xy},I_{xz},I_{yy},I_{yz},I_{zz}]^\top .
\]

- \(m\) は質量である。
- \(c=[c_x,c_y,c_z]^\top\) は `fc` 原点から重心へのベクトルである。
- \(I_{ij}\) は重心まわり、かつ `fc` 軸で表した慣性テンソル \(J_C\) の成分である。

推定器は ROS master や `rosbag play` を必要としない。入力 bag を直接走査し、mocap pose と actuator wrench を event time で同期し、静的な \(\theta\) に対する particle filter を実行する。元の全 message と推定途中の message は record time 順に merge され、別の解析 bag になる。

処理の全体像は次のとおりである。

```text
input ROS bag
  ├─ mocap world_from_fc pose ── resample / SG derivative ── s, omega, alpha
  ├─ calibrated wrench ───────────────────────────────────────┐
  └─ command + measured gimbal angle ── wrench reconstruction ┤
                                                               v
                                      tempered resample-move particle filter
                                                               |
                         posterior / particles / residual / diagnostics / CoG ProbTF
                                                               |
                         original messages と record-time merge
                                                               v
                                                       analysis ROS bag
```

これはまず「広い bounded-uniform prior から、十分に励起された既知の剛体を回収できるか」を確認する sanity check である。実 bag については、actuator calibration と励起の不足から、出力を必ずしも URDF nominal と一致する真の物理量とは解釈しない。

## 2. 座標系、時刻、符号規約

### 2.1 pose と body frame

`/gimbalrotor/mocap/pose` は `world` から `gimbalrotor/fc` への pose、すなわち位置 \(p_{WF}\) と回転 \(R_{WF}\) を与えるものとして読む。quaternion の配列順は ROS と SciPy に合わせて `xyzw` である。全 pose sample の `header.frame_id` は、先頭の `/` を除いて `world` でなければならず、空 frame も拒否する。

calibrated actuator wrench は `gimbalrotor/fc` 原点まわり、同じ軸で表した

\[
w_a=[F_x,F_y,F_z,\tau_x,\tau_y,\tau_z]^\top
\]

でなければならない。calibrated wrench が存在する場合、その全 sample の `WrenchStamped.header.frame_id` は、先頭の `/` を除いて `gimbalrotor/fc` でなければならない。空または異なる frame を暗黙に変換しない。

### 2.2 event time と bag record time

- Header を持ち、有効な stamp を持つ pose、wrench、joint state は header stamp を event time とする。
- Header のない `/gimbalrotor/four_axes/command` と `/gimbalrotor/flight_state` は bag record time を使う。
- 解析 message の Header は推定対象となった event time を持つ。
- 解析 message の bag record time は、mocap の event time と record time の対応を補間して決める。

この分離により、解析上の物理時刻を Header に保ちながら、元 bag と同じ再生時系列へ message を配置する。

## 3. 剛体力学モデル

重心一次モーメントと `fc` 原点まわりの慣性を

\[
h=mc,
\qquad
J_O=J_C+m\left((c^\top c)I-cc^\top\right)
\]

とする。mocap から得る `fc` 原点の world 加速度を \(\ddot p_{WF}\)、world 重力を \(g_W=[0,0,-9.80665]^\top\) とすると、body specific acceleration は

\[
s=R_{WF}^\top(\ddot p_{WF}-g_W)
\]

である。body 座標で表した角速度、角加速度をそれぞれ \(\omega,\alpha\) とすると、`fc` 原点まわりに actuator が与えるべき wrench は

\[
F_a=ms+\alpha\times h+\omega\times(\omega\times h),
\]

\[
\tau_a=J_O\alpha+\omega\times(J_O\omega)+h\times s.
\]

静止・水平時には \(s\simeq[0,0,g]^\top\) であり、\(F_{a,z}\simeq mg\) となる。重心が `fc` 原点からずれていれば \(h\times s\) により hover 中にも roll/pitch moment が必要になる。

各 particle は次の物理制約を満たす場合だけ support に入る。

- \(m>0\)
- \(J_C\) が正定値
- 主慣性モーメントが厳密な三角不等式を満たす
- 全成分が有限値

既定 prior は URDF 値を中心にした Gaussian ではない。設定ファイルの有限範囲から一様に生成し、上の物理制約で rejection sampling した分布である。既定範囲は質量 `0.5--5.0 kg`、CoG の x/y が `-0.2--0.2 m`、z が `-0.2--0.15 m`、対角慣性が `0.01--0.25 kg m^2`、非対角成分が `-0.06--0.06 kg m^2` である。

## 4. command から actuator wrench を再構成する場合

calibrated wrench topic がない実 bag では、`FourAxisCommand.base_thrust` を「N 単位の実推力」と仮定した effective model を用いる。これは force sensor の観測ではなく、controller の目標値に対する明示的な近似である。

rotor \(i\) の arm yaw を \(\psi_i\)、実測 gimbal angle を \(q_i\)、目標 thrust を \(\lambda_i\) とする。arm 座標での force は

\[
f_i^{arm}=
[0,-\lambda_i\sin q_i,\lambda_i\cos q_i]^\top,
\qquad
f_i=R_z(\psi_i)f_i^{arm}.
\]

`fc` から thrust point へのベクトルは

\[
p_i=p_{gimbal,i}^{fc}+R_z(\psi_i)R_x(q_i)[0,0,0.056]^\top .
\]

全 wrench は

\[
F_a=\sum_i f_i,
\qquad
\tau_a=\sum_i\left(p_i\times f_i+\sigma_i k_m f_i\right)
\]

で再構成する。実装に固定されている監査済み Grape geometry は次のとおりである。

| 項目 | 値 |
|---|---|
| main body 上の gimbal 原点 | `(-0.22309,-0.22309,0)`, `(0.22309,-0.22309,0)`, `(0.22309,0.22309,0)`, `(-0.22309,0.22309,0)` m |
| main body から `fc` | `(-0.0172999969,-0.00110000084,0.0570609990)` m |
| arm yaw | `[-2.3562,-0.7854,0.7854,2.3562]` rad |
| rotor direction \(\sigma_i\) | `[-1,+1,-1,+1]` |
| thrust offset | `0.056 m` |
| moment/force rate \(k_m\) | `-0.0181 m` |

command は zero-order hold、実測 gimbal angle は線形補間する。既定の許容 age は command `0.10 s`、joint `0.05 s` である。motor の立ち上がり遅れや command-to-force calibration は現在のモデルに含まれない。

## 5. mocap 運動学の前処理

### 5.1 resampling と微分

pose は event time で重複を除去し、quaternion の符号を連続化した後、既定 `50 Hz` の一様 grid へ変換する。

- 位置は補間後、51 点・3 次の Savitzky--Golay filter で平滑化し、同じ局所多項式の二階微分から world 加速度を得る。
- 姿勢は SLERP で resampling し、quaternion 成分を SG 平滑化した後に再正規化する。
- 隣接する回転の \(R_i^\top R_{i+1}\) の rotation vector から角速度を求める。
- world 表現の角速度をさらに局所多項式で平滑化・微分し、角加速度を求めて body 座標へ戻す。
- SG window の半幅に回転微分 stencil の余白を加えた `window_length // 2 + 2` sample は無効とする。

pose の最近傍 age が既定 `0.03 s` を超える grid point は使わない。calibrated wrench も別の `max_wrench_age_s=0.03` で gate する。推定用 sample は 50 Hz grid から 5 sample ごとに選ぶため、既定の尤度更新入力は 10 Hz である。

### 5.2 1 cm / 1 deg の意味

設定値 `mocap_position_sigma_m=0.01` と `mocap_orientation_sigma_deg=1.0` は、一つの raw mocap sample に対する等方的標準偏差を表す。orientation は接空間の小角度 Gaussian として解釈する。`kinematics.py` は SG 係数から、独立 sample 近似で acceleration、angular velocity、angular acceleration の noise scale も計算する。

ただし、現在の particle likelihood はこの導関数 covariance を parameter ごとの wrench covariance へ伝播していない。尤度には固定の residual scale `0.80 N` と `0.06 N m` を使う。この点は重要な近似であり、「1 cm / 1 deg を完全な観測 covariance として Bayesian update に組み込んだ」実装ではない。

## 6. 観測の選択と gate

推定器は次の優先順位で actuator 観測を決める。

1. 選択区間に calibrated actuator wrench が 10 sample 以上あれば、その topic を使う。
2. それ以外は `base_thrust` と実測 gimbal angle から effective wrench を再構成する。

command mode では `flight_state=5`、すなわち `HOVER_STATE` だけを既定で採用する。`TAKEOFF_STATE=3` には、機体が床上にあるまま thrust を ramp する長い区間が含まれ、床反力を持たない剛体式が成立しないためである。許容 state は `real_bag.allowed_flight_states` で変更できる。

各 batch の採用前には、既存 history と pending batch の whitened finite-difference Jacobian を連結して excitation rank と condition number を計算する。既定値は `minimum_excitation_rank=1` なので、一部の parameter combination だけが観測可能な rank-deficient batch も更新に使う。診断はそれを `updated_rank_deficient` と表示する。rank が最小値未満、または full-rank 時の condition number が上限を超えた batch は posterior に入れず、diagnostic-only message を残す。

## 7. tempered resample-move particle filter

観測 residual の各成分には、自由度 \(\nu=5\) の独立 Student-t likelihood を用いる。residual を \(r_j\)、scale を \(\sigma_j\) とすると、一成分の log likelihood は定数項を除いて

\[
\log p(r_j\mid\theta)
=-\log\sigma_j-
\frac{\nu+1}{2}
\log\left(1+\frac{r_j^2}{\nu\sigma_j^2}\right)
\]

である。既定では 5 observation を一 batch とする。

急峻な likelihood を一度に掛けて particle が一つへ collapse するのを避けるため、batch likelihood を power \(\beta:0\rightarrow1\) で tempering する。次の power increment は ESS が particle 数の 70%程度になるよう二分探索し、必要に応じて systematic resampling を行う。resampling threshold は 50%である。

resample 後の rejuvenation は二種類を混ぜる。

- 高頻度の局所 proposal: particle covariance に基づく Gaussian random walk。有限 bounds では反射する。
- 低頻度の大域 proposal: 物理制約付き bounded-uniform prior から再度生成する。既定確率は 3%である。

どちらも、それまでの全 observation と現在の tempered batch を対象に Metropolis acceptance を計算する。既定 particle 数は 1024、MCMC は resampling 一回につき 2 step である。posterior summary は weighted mean、support 内の MAP particle、標準偏差、parameter ごとの 2.5/97.5 percentile、10 x 10 covariance、ESS を含む。

この「局所/大域」は同一 offline filter 内の proposal mixture であり、別周期の二つの ROS node ではない。

## 8. synthetic sanity bag

`generate_sanity_bag.py` は ROS master を使わず、既知の 6-DoF multisine trajectory を 24 秒・50 Hz で生成する。clean trajectory から力学 wrench を作り、別途 noisy pose を bag へ書く。

- position noise: 各軸 `1 cm` standard deviation
- orientation noise: 接空間各軸 `1 deg` standard deviation
- calibrated force noise: 各軸 `0.02 N`
- calibrated torque noise: 各軸 `0.002 N m`

generator は全 trajectory の 10-parameter Jacobian rank が 10 であることを検査する。truth は `config/sanity_truth.yaml` から読み、現在の q=0 Grape composite を `fc` で表した次の値である。

| parameter | truth |
|---|---:|
| mass | 2.351597590812377 kg |
| cog_x | 0.015275288306 m |
| cog_y | 0.001069474264 m |
| cog_z | -0.047551249361 m |
| inertia_xx | 0.065000061483 kg m^2 |
| inertia_xy | -0.000000727899 kg m^2 |
| inertia_xz | 0.000019015080 kg m^2 |
| inertia_yy | 0.064952656340 kg m^2 |
| inertia_yz | 0.000000059167 kg m^2 |
| inertia_zz | 0.128992110664 kg m^2 |

`/grape_param_estim/ground_truth` は評価専用である。estimator の選択 topic 集合にこの topic はなく、初期 particle、proposal、gate に truth file を使わない。解析 bag に truth が残るのは、元 message を全て保存して事後評価するためである。

`evaluate_sanity.py` は推定終了後の別 process として final posterior mean を truth と比較する。既定 pass 条件は、質量相対誤差 5%以下、CoG Euclidean error 2 cm以下、重心慣性の Frobenius 相対誤差 15%以下である。95% marginal interval の truth coverage も報告するが、現時点では pass/fail 条件には含めない。

この試験は実装した運動方程式と filter の自己整合性試験である。generator と estimator が同じ剛体式を使うため、別実装による独立な dynamics validation や実 actuator calibration の検証ではない。

## 9. 実 bag へ適用するときの解釈

収録済み Grape hovering bag には calibrated wrench がないため、通常は command mode になる。`base_thrust` は目標値であり実推力 sensor ではない。共通 thrust scale \(k\) が未知なら、hover の主要式は概ね \(k\sum_i u_i\simeq mg\) となり、\(m\) と \(k\) を同時には識別できない。実際、command をそのまま N とみなした単純釣り合いの見かけ質量はおよそ `2.85--2.95 kg` で、q=0 URDF composite の約 `2.3516 kg` と一致しない。

したがって command mode の出力は `model` field に `command_as_force_effective` を付け、effective parameters として扱う。次も守る必要がある。

- nominal CoG から生成された `/gimbalrotor/uav/cog/odom` は独立観測に使わない。
- hover だけで full inertia が励起されるとは限らない。rank と particle spread を必ず確認する。
- `/tf_static` の `motor_arm*` と現 checkout の `rotor_arm*` には版差がある。現在の再構成は上表の固定 geometry であり、bag 内 TF から自動構成していない。
- gimbal angle は actuator force direction には使うが、機体を q-dependent articulated rigid-body system としては計算しない。gimbal の内部運動が大きい場合、固定 \(c,J_C\) の近似誤差になる。
- command と実 thrust の遅れ、飽和、ESC calibration、空力 drag、床や tether の外力は現在の likelihood にない。
- command mode で flight-state topic が欠落すると HOVER gate を適用できないため、入力 topic を事前に監査する。

## 10. 解析 bag と出力 topic

出力 writer は input bag をバイト列のまま append しない。元 message を record time 順に読み、解析 event と merge して新しい bag を作る。元 message を書き戻す際には connection header を渡すため、型、caller、latching を含む metadata と `/tf_static` の latch を保持する。出力完成前は同じ directory の一時 bag を使い、最後に atomic rename する。入力 bag 自体は変更しない。

主な出力は次のとおりである。

| topic | 内容 |
|---|---|
| `/grape_param_estim/estimate` | parameter 順序、mean、MAP、std、95% interval、10 x 10 covariance、ESS、seed、観測数、provenance |
| `/grape_param_estim/particles` | posterior を systematic weighted decimation した particle snapshot |
| `/grape_param_estim/diagnostics` | ESS、resampling、MCMC acceptance、NIS、residual norm、excitation rank/condition、gate reason |
| `/grape_param_estim/predicted_wrench` | posterior mean から予測した `fc` wrench |
| `/grape_param_estim/wrench_residual` | actuator observation minus prediction |
| `/probtf/grape_param_estim/cog` | inertial-particle posterior から誘導した CoG edge |

`InertialParameterEstimate.covariance` は `parameter_names` と同じ順序の row-major 行列である。`log_likelihood` field には sequential importance update で累積した log evidence が入る。

`ParameterParticleSet` は全 particle の完全 dump ではない。元の normalized weight の累積分布上へ等間隔の点を置く weighted systematic decimation で代表 particle を選び、重複によって元の確率質量を近似する。したがって出力後の各 retained row は等重みである。完全な内部 particle array と一対一に対応する、と解釈してはならない。

### 10.1 ProbTF の CoG 表現

質量や慣性テンソルは SE(3) transform ではないため、ProbTF edge に押し込まない。ProbTF へ出すのは `fc` から推定 CoG への位置だけである。

- parent: `gimbalrotor/fc`
- child: `grape_param_estim/estimated_cog`
- translation mean: posterior の `[cog_x,cog_y,cog_z]`
- translation residual covariance: 10 x 10 parameter covariance の CoG 3 x 3 marginal
- orientation: identity の `DIRAC`
- rotation/translation coupling: zero
- representative: moment representative
- `is_static=false`: 物理的 CoG は静的だが、解析時刻とともに posterior record が更新されるため

identity Dirac の Bingham shape は `wxyz` 順で

\[
\operatorname{diag}(1.5,-0.5,-0.5,-0.5)
\]

であり、inverse concentration は 0 である。CoG marginal は一般には Gaussian ではないため、message は `MOMENT_SUMMARY` の lossy approximation と明示する。

## 11. 実行と確認

ビルドする。

```bash
cd /home/leus/catkin_ws
catkin build grape_param_estim
source devel/setup.bash
```

synthetic の生成、推定、評価を行う。

```bash
rosrun grape_param_estim generate_sanity_bag.py \
  --output-bag /tmp/grape_sanity_input.bag --seed 7

rosrun grape_param_estim estimate_grape_bag.py \
  --input-bag /tmp/grape_sanity_input.bag \
  --output-bag /tmp/grape_sanity_analysis.bag \
  --config "$(rospack find grape_param_estim)/config/estimator.yaml" \
  --seed 7

rosrun grape_param_estim evaluate_sanity.py \
  --analysis-bag /tmp/grape_sanity_analysis.bag
```

実 bag の例は HOVER_STATE を含む episode を使う。

```bash
INPUT=/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_7_2026-06-12-17-41-34.bag

rosrun grape_param_estim estimate_grape_bag.py \
  --input-bag "$INPUT" \
  --output-bag /tmp/grape_hovering_7_analysis.bag \
  --config "$(rospack find grape_param_estim)/config/estimator.yaml" \
  --seed 7 \
  --start-offset 45 \
  --duration 8
```

この smoke test で `--start-offset` と `--duration` が制限するのは推定に使う窓だけである。元 bag の message は窓の外も含めて全て解析 bag へ保存される。

解析 bag は Foxglove Studio の **Open local file** から直接開ける。再生して ROS topic として見る場合は `roscore` の後に次を実行する。

```bash
rosbag play --clock /tmp/grape_hovering_7_analysis.bag
rostopic echo /grape_param_estim/estimate
```

Foxglove では mean と 95% interval、ESS、residual、NIS、rank/condition、gate reason を同じ時刻軸で確認する。平均だけでなく particle snapshot も見て、多峰性、bounds への集中、rank 不足を区別する。

## 12. 検証状況

2026-07-22 時点で、剛体式、URDF composite、Grape geometry、運動学、particle filter に対する 19 件の unit test が成功している。試験には次が含まれる。

- physical inertia rejection と parallel-axis shift
- vectorized wrench prediction
- q-dependent URDF composite inertia
- Grape wrench reconstruction と numerical allocation の round trip
- translational/rotational derivative、quaternion sign flip、1 deg noise 下の angular acceleration
- bounded-uniform particle、resampling、full-excitation recovery、hover rank deficiency

同日の現行版について、seed 7、既定 24 秒 trajectory、1024 particle で documented synthetic pipeline を end-to-end 実行した。実行前後でファイル hash が変わらないことを確認しており、`estimator.yaml` の SHA-256 は `04dbde11015f28f19e448ee027c5b9b9075357a0b5dc1f49c7b8a87e75f75900` である。主な結果は次のとおりである。

| 項目 | 結果 |
|---|---:|
| synthetic pose / wrench | 1201 / 1147 message |
| generator excitation | rank 10、condition 19.5597 |
| estimator | 42 update、210 observation、final ESS 745.9 / 1024 |
| mass relative error | 0.00198316 (0.1983%) |
| CoG Euclidean error | 0.00112740 m (1.127 mm) |
| inertia Frobenius relative error | 0.0426485 (4.2649%) |
| 95% marginal truth coverage | 10 / 10 parameter |
| estimator wall time | 48.71 s |

3 個の既定 threshold check は全て pass した。input bag は start/end `1.0/25.0 s`、2350 message、analysis bag は同じ start/end と 24 秒 duration を保ったまま 2602 message であり、解析 message は 252 個追加された。

さらに `/probtf/grape_param_estim/cog` の 42 message 全てが ProbTF v2 reader で validation と wire round-trip に成功した。全 record は parent `gimbalrotor/fc`、child `grape_param_estim/estimated_cog`、identity Dirac orientation、一 component であり、translation covariance の全 message を通した最小固有値も `3.04e-8` と非負だった。

この結果は上記 config hash と実行時コードに対する回帰結果である。古い解析 bag の値を、変更後コードの保証値として無条件に流用してはならない。

## 13. 現在残る近似と次の改善

1. **mocap covariance の尤度伝播**  
   SG derivative の noise estimate を、particle ごとの wrench covariance へまだ伝播していない。\(s,\omega,\alpha\) に対する wrench Jacobianと actuator noise を組み合わせる必要がある。

2. **時間相関**  
   51-point SG window に対して推定 sample 間隔は 5 point なので、隣接 observation は強く相関する。現在の likelihood は独立として積を取るため、credible interval が過度に狭くなる可能性がある。window-aware thinning、block likelihood、または相関 covariance が必要である。

3. **mocap gap の filter footprint**  
   age gate は grid point 単位である。gap 自体の sample は除外するが、SG window が gap の補間値を含む周辺 sample まで自動的に dilation していない。

4. **計算量**  
   MAP と excitation metrics は履歴を繰り返し評価し、resample-move も全履歴 likelihood を使う。長い episode では update 数に対して概ね二次に増える処理がある。current-density cache、解析的 regressor、incremental information matrix、低頻度の大域 move が改善候補である。

5. **articulated dynamics と actuator calibration**  
   q-dependent composite inertia、gimbal の \(\dot q,\ddot q\)、motor dynamics、thrust scale を joint model に入れる必要がある。calibrated wrench がなければ mass/scale gauge は残る。

6. **episode geometry と provenance**  
   bag 内 TF と motor calibration から geometry を構成し、入力 window、CLI override、コード/geometry version を machine-readable provenance に追加する余地がある。

これらを解決するまでは、synthetic sanity の成功と実 bag で URDF nominal parameter が同定できたことを同一視してはならない。
