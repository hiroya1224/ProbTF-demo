# Grape offline Bayesian analysis

Grape の rosbag を直接読み、trajectory smoother、effective-response
同定、反実仮想の安全 gate、質量・重心・慣性の particle filter を扱う
オフライン解析パッケージです。実機 parameter を書き換える経路はなく、
出力は human review 用の artifact または元 bag と別の解析用 ROS 1 bag
です。`rosbag play` や実機 controller の起動は解析に必要ありません。

数式、座標系、filter、bag merge、ProbTF 表現、既知の限界は [`lectures/grape_parameter_estimation.md`](lectures/grape_parameter_estimation.md) にまとめています。

## ビルド

```bash
cd /home/leus/catkin_ws
catkin build grape_param_estim
source devel/setup.bash
```

既定設定は `config/estimator.yaml` です。ここで指定する prior は URDF nominal 値を中心とする Gaussian ではなく、物理制約を満たす有限範囲の bounded uniform です。乱数 seed、parameter bounds、入力 bag、使用区間、設定内容は再現性と provenance の一部です。

## Bayesian real-bag vertical slice

`config/counterfactual.yaml` は bag 4、7、8 の固定区間、topic、source bag
SHA-256、ENU/FLU/SI 規約、20 Hz の共通 pipeline、seed、27 candidate の
共通 grid を定義します。実行は clean な Git checkout と完全一致する
source commit を必須にし、dirty tree、bag hash 不一致、frame/unit
不一致を出力前に拒否します。

```bash
GRAPE_BAG_ROOT=/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering
GRAPE_SLICE_OUT=/tmp/grape_real_bag_slice

rosrun grape_param_estim analyze_grape_counterfactual.py \
  --config "$(rospack find grape_param_estim)/config/counterfactual.yaml" \
  --bag-root "$GRAPE_BAG_ROOT" \
  --output-root "$GRAPE_SLICE_OUT"
```

各 run directory には次のファイルを保存します。

| file | 内容 |
|---|---|
| `summary.json` | source/config/commit/input/trajectory hash、frame、diagnostic、全 hard gate |
| `trajectory.csv` | desired、diagnostic nominal、actual posterior mean/std、residual の時系列 |
| `trajectory_particles.npz` | coherent RTS actual sample と、同じ sample 初期状態から積分した nominal sample、ID、weight |
| `candidate_grid.csv` | 共通 27 candidate。exact oracle 不在時は確率欄を空にした未評価 grid |
| `REPORT.md` | RMS、effective-response posterior、識別性、gate の人向け要約 |
| `artifact_manifest.json` | 上記 payload の SHA-256 と byte size |

必要な場合だけ `--analysis-bag-root` を追加すると、元 bag を変更せず、
解析 message を record-time 順に merge した派生 bag と
`*.analysis.json` SHA/count sidecar を生成します。派生 bag は元 bag
全体を含むため、repository へ commit せず容量に余裕のある出力先を
指定してください。

```bash
GRAPE_ANALYSIS_BAG_ROOT=/tmp/grape_analysis_bags

rosrun grape_param_estim analyze_grape_counterfactual.py \
  --config "$(rospack find grape_param_estim)/config/counterfactual.yaml" \
  --bag-root "$GRAPE_BAG_ROOT" \
  --output-root "$GRAPE_SLICE_OUT" \
  --analysis-bag-root "$GRAPE_ANALYSIS_BAG_ROOT"
```

派生 bag の application topic は次のとおりです。

| topic | type | 内容 |
|---|---|---|
| `/analysis/grape_param_estim/trajectory/desired` | `TrajectoryParticleSet` | 記録済み controller target |
| `/analysis/grape_param_estim/trajectory/nominal` | `TrajectoryParticleSet` | sample ごとの診断用 local nominal。exact PC/MCU replay ではない |
| `/analysis/grape_param_estim/trajectory/actual_posterior` | `TrajectoryParticleSet` | offline EKF/RTS の coherent trajectory samples |
| `/analysis/grape_param_estim/model_mismatch` | `ModelMismatch` | matched sample の SE(3) log tracking/model residual、区間、covariance |

現在の実 bag 結果は
[`results/real_bag_vertical_slice/2026-07-24/INDEX.md`](results/real_bag_vertical_slice/2026-07-24/INDEX.md)
にあります。exact PC/MCU replay、bag-derived exact fixture、controller
integrator state、joint state/parameter inference、12-fold calibration が
未接続なので、全 run は `EXPERIMENTAL`、
`recommendation_available=false` です。candidate CSV は推奨ではなく
未評価 grid です。また per-time の
`world→{desired, nominal, actual}` ProbTF edge materialization は未接続で、
現時点の trajectory/mismatch 可視化契約は application message 側です。

## Synthetic sanity check

まず、真値を既知にした synthetic bag で生成、推定、評価を一続きに確認します。

```bash
rosrun grape_param_estim generate_sanity_bag.py \
  --output-bag /tmp/grape_sanity_input.bag \
  --seed 7

rosrun grape_param_estim estimate_grape_bag.py \
  --input-bag /tmp/grape_sanity_input.bag \
  --output-bag /tmp/grape_sanity_analysis.bag \
  --config "$(rospack find grape_param_estim)/config/estimator.yaml" \
  --seed 7

rosrun grape_param_estim evaluate_sanity.py \
  --analysis-bag /tmp/grape_sanity_analysis.bag
```

generator が書く ground truth は評価専用です。estimator は ground-truth topic、URDF nominal parameter、未来の観測を読んではいけません。評価は推定終了後に別 process で行います。

この sanity bag では、1 cm / 1 deg（接空間）の mocap noise を加えた pose と、既知の剛体運動から生成した校正済み actuator wrench を別 topic に記録します。後者を使うことで mass と共通 thrust scale の gauge をいったん切り離し、10 次元の剛体慣性 parameter を broad bounded-uniform 初期粒子から回収できるかだけを検証します。推定器が読む topic 一覧に ground truth は含まれません。

## 実 rosbag のオフライン推定

主要 topic がそろう短い bag での smoke test 例です。

```bash
INPUT=/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_7_2026-06-12-17-41-34.bag

rosrun grape_param_estim estimate_grape_bag.py \
  --input-bag "$INPUT" \
  --output-bag /tmp/grape_hovering_7_analysis.bag \
  --config "$(rospack find grape_param_estim)/config/estimator.yaml" \
  --seed 7 \
  --start-offset 45 \
  --duration 8

rosbag info /tmp/grape_hovering_7_analysis.bag
```

同じ処理は launch file からも起動できます。

```bash
roslaunch grape_param_estim offline_estimator.launch \
  input_bag:="$INPUT" \
  output_bag:=/tmp/grape_hovering_7_analysis.bag \
  start_offset:=45 \
  duration:=8
```

入力 bag は変更しません。出力先には別 path を指定してください。`--start-offset` と `--duration` は推定に使う区間だけを選び、元 bag の message は区間外も含めて出力へ保存します。推定器は original message の型・内容・record timestamp と connection metadata（`/tf_static` の latch を含む）を保ったまま、解析 message と record-time 順に merge して新しい bag を作ります。Header がある sensor はその event time、Header がない `/gimbalrotor/four_axes/command` などは bag record time を使います。

## 解析 bag の再生

端末 1 で ROS master を起動します。

```bash
roscore
```

端末 2 で解析 bag を再生します。

```bash
source /home/leus/catkin_ws/devel/setup.bash
rosbag play --clock /tmp/grape_hovering_7_analysis.bag
```

端末 3 では、例えば posterior summary を確認できます。

```bash
source /home/leus/catkin_ws/devel/setup.bash
rostopic echo /grape_param_estim/estimate
```

## 出力 topic

| topic | type | 内容 |
|---|---|---|
| `/grape_param_estim/estimate` | `grape_param_estim/InertialParameterEstimate` | parameter 名、posterior mean/MAP/95% 区間/covariance、ESS、尤度、provenance |
| `/grape_param_estim/particles` | `grape_param_estim/ParameterParticleSet` | weighted systematic decimation した等重み particle。完全な全粒子 dump ではない。`values` は particle-major の row-major 配列 |
| `/grape_param_estim/diagnostics` | `grape_param_estim/EstimatorDiagnostics` | resampling、NIS、force/torque residual、MCMC、励起 rank、gate 理由 |
| `/grape_param_estim/predicted_wrench` | `geometry_msgs/WrenchStamped` | posterior mean による actuator wrench 予測 |
| `/grape_param_estim/wrench_residual` | `geometry_msgs/WrenchStamped` | 観測 wrench - posterior mean 予測 |
| `/probtf/grape_param_estim/cog` | `probtf_msgs/ProbabilisticTransformStamped` | parameter 粒子から誘導した `fc`→推定 CoG の moment summary。mass/inertia 自体は Prob-TF edge に格納しない |
| `/grape_param_estim/ground_truth` | `grape_param_estim/InertialParameterEstimate` | synthetic generator が評価専用に書く真値。実 bag には存在しない |

`covariance` は `parameter_names` と同じ順番の正方行列を row-major で格納します。`ParameterParticleSet.particle_count` は出力に残した particle 数、`stride` は一 particle あたりの値数です。

## Foxglove Studio

Foxglove Studio の **Open local file** から解析 bag を直接開けます。ROS bridge や `rosbag play` は不要です。

確認項目は次のとおりです。

- Raw Messages で `parameter_names` と配列 index の対応を確認する。
- Plot で `mean`、`lower_95`、`upper_95` を同じ parameter index について重ねる。
- `effective_sample_size`、`ess_before`、`ess_after` と `resampled` を並べ、particle collapse の有無を見る。
- force/torque residual、NIS、`excitation_rank`、`gate_reason` を見て、rank 不足を伴う更新と gate された区間を区別する。
- particle snapshot を使い、平均と covariance だけでは見えない多峰性や境界への集中を確認する。

## 解釈上の制約

- 「事前情報を使わない」は、URDF nominal 値を中心にした informative prior を使わないという意味です。無限範囲の一様分布は proper distribution にならないため、bounded uniform の上下限と物理制約は必要です。
- Synthetic sanity check でも ground truth を estimator の初期値、proposal、gate、入力 topic に使いません。真値は生成と事後評価だけに使います。
- `/gimbalrotor/four_axes/command` は nominal controller が出した目標推力であり、実推力 sensor ではありません。質量と共通 thrust scale は同時に観測できない場合があります。
- `/gimbalrotor/uav/cog/odom` は nominal CoG を用いた派生量であり、CoG を推定するときの独立観測として扱いません。
- bag の `/tf_static` にある `motor_arm*` と現 checkout の `rotor_arm*` には版差があります。現段階の command-to-wrench 経路は監査済み Grape geometry を固定値として使い、bag 内 TF は解析 bag に保存します。geometry が一致しない episode へ暗黙適用せず、次段階では episode TF から geometry を構成してください。
- zero covariance は zero noise ではなく、未設定の場合があります。設定した noise floor と使用 policy を解析結果へ残します。
- hovering data だけでは full inertia、CoG、actuator scale が十分に励起される保証はありません。励起 rank が不足する parameter は nominal 値へ強制せず、未観測または prior/bounds 支配として報告します。
- 収録済み hovering bag の command を N とみなす単純釣り合いでは見かけ質量が概ね 2.85--2.95 kg となり、q=0 URDF composite の約 2.3516 kg とは一致しません。実 bag で nominal に強制収束させることはせず、出力を effective parameter として扱います。
- 実 bag 上で全慣性 parameter が URDF 値へ一致することは、この sanity check だけでは保証しません。episode を分けた held-out validation と actuator calibration が必要です。
