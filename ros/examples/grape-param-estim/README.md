# grape_param_estim

Grape の full closed-loop system を forecast operator として使う、
weak-constraint ensemble smoothing の実装です。設計の正本は
`docs/plans/grape_weak_constraint_ienks_estimation_plan_ja.md` です。

現在は Phase 1–6 を実装しています。旧 Streamlit GUI、rosbag open-loop replay、
segment reset、static particle estimator、旧 result schema との後方互換は
意図的に削除しました。

## Phase 1 の内容

- `PoseLinearController::controlCore()` と
  `GimbalrotorController::controlCore()` の対象実験 branch を移植した
  stateful Python controller
  - position mode の x/y/z/roll/pitch/yaw PID
  - yaw shortest-angle error
  - z integral の非負制約
  - roll/pitch integration gate
  - P/I/D/error/output clamp
  - `gimbal_calc_in_fc=false`、`gimbal_dof=1`、fully-actuated allocation
- Grape URDF の四つの vectoring rotor、full inertia tensor、CoG、反トルク
- actual gimbal angle に応じて全リンクから再計算する controller 側の CoG、
  full inertia、thrust-link origin と、固定 arm-coordinate basis
- thrust/gimbal delay、first-order response、force/angle/rate saturation
- position、quaternion、world velocity、body angular velocityを持つ full 6-DoF plant
- nominal plant と、parameter mismatch・drag・actuator mismatch・時間変化外乱を
  持つ truth plant
- x/y/z/roll/pitch/yaw を同時に励起する一つの連続 closed-loop episode
- position/orientation だけの observation generator
- `Z_k = T_nominal,k^-1 T_real,k` の correction-transform path

## Phase 2 の内容

- 一つの全飛行 window を black-box forecast する strong-constraint IEnKS
- joint control `z=(x0, c0, theta)`（36次元）
  - initial position / SO(3) tangent / latent velocity / angular velocity
  - six PID integral states
  - log mass、full SPD inertia chart、CoG、rotorごとの log force/torque
    effectiveness
- pose-only residual
  `[(p-p_obs), Log(R_obs^T R)]` の既知 covariance による whitening
- ensemble cloud の secant regression だけを使う ensemble-space Gauss–Newton
- raw static-parameter ensemble、full latent trajectory ensemble、
  correction-transform path ensemble
- model family が完全に一致する Experiment A と、pose noise だけを加えた
  Experiment B
- mass・full inertia・全 force effectiveness の common-scale exact ridge の保持

解析・有限差分 Jacobian、segment reset、parameter hard bounds、particle weights は
使いません。

## Phase 3 の内容

- 各 integration interval に独立な6次元 innovation block を持つ full-block
  weak-constraint IEnKS-Q
- 不規則時刻にも対応する stationary Gauss–Markov / OU residual body wrench
- static parameter は飛行全体で一つの global variable のまま保持
- interval residual は、その区間の全 RK4 stage へ同じ値を加算
- static parameter、innovation、residual wrench、full trajectory、
  correction-transform path を同じ member ID のまま保存
- drag、actuator lag/delay、時間変化外乱を truth だけへ加えた Experiment C で、
  strong constraint との parameter bias / latent state / path coverage 比較
- truth state 上で nominal controller/actuator を因果的に独立 replay して求めた
  counterfactual residual-wrench oracle（truth command の流用なし）
- 同じ static truth・observation-noise realization・window から model error だけを
  除いた matched strong-control run による excess parameter bias の分離

時間方向を一つの低rank path sampleへ潰してはいません。augmented dimension は
`36 + 6 * (interval count)` で、既定では ensemble size をさらに2大きくし、全Q
block を prior ensemble が張るようにします。Experiment C の Q scale は既知の
synthetic model-error RMS から機械的に校正します。

## Phase 4 の内容

- common-scale exact ridge 上の5本の full closed-loop rollout と pose likelihood
  の不変性
- proper prior metric で分解した raw ridge coordinate / 17次元 quotient law、
  prior-whitened information leak、true correction-path coverage
- IEnKS-Q の perfect-model zero-residual realization における raw ridge law と、
  非ゼロ residual を含む exact augmented symmetry
  `(theta, eta) -> (theta + lambda v, exp(lambda) eta)`
- strong `M=38,46` と短い full-block weak `D=78, M=80,88` の実同化を使う
  ensemble-size convergence
- prior-whitened quotient と pose-whitened full correction path の deterministic
  sliced-Wasserstein-1 比較
- plant actuator channel wiring の nominal / 0--1 swap を、一つの ensemble に
  混ぜず mode ごとに独立同化する Experiment D
- pose-only Laplace mode weight と、独立 wiring inspection による weight のみの
  conditioning（raw posterior member は再同化・再サンプル・混合しない）

reaction-torque 符号だけを mode にする案は、短い pose window では両方の真値を
区別できなかったため採用していません。wiring mode は両方の synthetic truth で
pose weight の argmax が切り替わることを回帰試験しています。
IEnKS-Q でも proper-prior ridge law と augmented likelihood symmetry が保たれたため、
この phase では particle-based correction を追加していません。

## Phase 5 の内容

- rosbag の header stamp ではなく record time を正本にした実データ adapter
- `flight_state=5` の一つの連続 airborne 区間を既定 window とし、別 flight や
  不連続区間を連結しない episode selection
- CoG odometry の位置と baselink odometry の姿勢だけを likelihood に使用
  （twist、IMU、加速度は観測へ追加しない）
- 記録された `PoseControlPid` reference/feedforward、PID integral、
  dynamic-reconfigure/YAML gain snapshot、gimbal/thrust anchor の因果的復元
- preflight 静止区間からロバスト推定する位置・SO(3) 観測 covariance
- body residual wrench を sparse OU knot で表し、piecewise-linear wrench の
  integration interval 平均を全 RK4 stage に保持する連続 weak forecast
- 観測 pose が要求する wrench と、観測 pose 上で因果 replay した nominal
  controller/actuator wrench の差による Q scale・相関時間の事前校正
- raw static parameter、OU innovation/knot/interval wrench、full latent trajectory、
  correction-transform path、ridge/mode/resolution 診断の member-aligned 保存

対象 bag の既定 window は約 21 秒です。takeoff 直後の接地期間は、ground-contact
model を持たない free-flight rigid body へ入力すると床を突き抜けるため、自動では
含めません。`--full-flight` は ground-contact model を追加した診断時だけ明示的に
使うための option です。

## Phase 6 の内容

- Phase 5 NPZ の selected-mode raw member を平均化せず読み戻し、各 controller
  parameter 候補を全 member で full closed-loop simulation
- member ごとの static plant parameter、初期機体/controller/actuator state、Q path を
  同じ row のまま維持
- controller nominal mass scale と roll/pitch/yaw PID scale の明示的な候補集合
- desired pose からの correction-transform path、mean tracking loss、upper CVaR、
  failure probability、log-scale parameter change の decision score
- numerical/physical forecast failure を member ID と理由つきで保持し、failure
  probability へ算入（欠損 path は NaN として明示）。1件でも forecast exception を
  含む候補は推奨対象外とし、全候補が該当すれば偽の推奨値を出さない
- 全 candidate × member の trajectory/path/loss と、source mode law、decision policy、
  provenance を pickle-free NPZ に保存

既定の scenario は、Phase 5 と同じ reference・member 初期状態・posterior residual
wrench path を繰り返す counterfactual です。したがって「同じ実験を controller
parameter だけ変えて再試行する」提案であり、新しい風を予言したものではありません。
`--residual-policy zero` を使う場合も、その仮定を artifact に明記します。

loss scale、failure threshold、CVaR level、各 decision weight は物理同定 parameter
ではなく明示的な意思決定規則です。CLI option と artifact に全値を残すため、推奨値を
採用する前にそれらを変えた sensitivity check を行えます。dynamic_reconfigure への
自動書き込みは行いません。

## 実行

```bash
cd /home/leus/catkin_ws
catkin build grape_param_estim --no-deps
source devel/setup.bash

rosrun grape_param_estim grape_weak_constraint_synthetic.py \
  --output /tmp/grape_phase1_synthetic.npz
```

perfect-model の恒等性を確認する場合は次を使います。

```bash
rosrun grape_param_estim grape_weak_constraint_synthetic.py \
  --perfect-model \
  --duration 3.0 \
  --output /tmp/grape_phase1_perfect.npz
```

Phase 2 の Experiment A（観測 noise realization はゼロ、covariance は正定値）と
Experiment B（位置姿勢 noise のみ）は次で実行します。

```bash
rosrun grape_param_estim grape_strong_constraint_ienks.py \
  --experiment A \
  --output /tmp/grape_phase2_a.npz

rosrun grape_param_estim grape_strong_constraint_ienks.py \
  --experiment B \
  --output /tmp/grape_phase2_b.npz
```

Phase 3 の Experiment C は次で実行します。

```bash
rosrun grape_param_estim grape_weak_constraint_ienks_q.py \
  --output /tmp/grape_phase3_c.npz
```

Phase 4 の三つの検証は、それぞれ独立に実行できます。

```bash
rosrun grape_param_estim grape_phase4_validate.py \
  --section ridge \
  --output /tmp/grape_phase4_ridge.npz

rosrun grape_param_estim grape_phase4_validate.py \
  --section convergence \
  --output /tmp/grape_phase4_convergence.npz

rosrun grape_param_estim grape_phase4_validate.py \
  --section mode \
  --output /tmp/grape_phase4_mode.npz
```

Phase 5 の実 rosbag 同化は次で実行します。既定は `dt=0.04 s`、12 knots、
augmented dimension + 2 members、1 iteration です。

```bash
rosrun grape_param_estim grape_real_rosbag_ienks_q.py \
  --bag /home/leus/catkin_ws/bags/grape-drone/20260613_grape_hovering/20260613_grape_hovering_3_2026-06-13-15-12-51.bag \
  --output /tmp/grape_phase5_real.npz
```

CLI summary の `q_resolution_sufficient` が false の場合、その knot 数で Q の
時間解像度が十分だとは主張できません。`--maximum-knots 0` は校正された OU bridge
criterion を満たす全 knot を使いますが、augmented dimension と必要 member 数が
増えるため計算時間も大きくなります。計算量を理由に knot を減らした事実は artifact
へ保存されます。

Phase 6 は Phase 5 artifact を直接読みます。次は baseline と二つの mass 候補だけを
評価する短い例です。`--candidate` を省略すると mass と姿勢 PID の既定9候補を使います。

```bash
rosrun grape_param_estim grape_posterior_predictive.py \
  --phase5-artifact /tmp/grape_phase5_real.npz \
  --candidate baseline,1,1,1,1 \
  --candidate controller_mass_0p9,0.9,1,1,1 \
  --candidate controller_mass_1p1,1.1,1,1,1 \
  --output /tmp/grape_phase6_proposal.npz
```

異なる policy への感度は、たとえば `--failure-threshold`、`--cvar-level`、
`--mean-weight`、`--cvar-weight`、`--failure-weight`、`--change-weight` を変えて比較します。
member thinning option は設けていないため、CLI は常に source artifact の raw member
全体を評価します。

長い window では full-block control と必要 ensemble size も増えます。明示する
場合、`--ensemble-size` は必ず augmented dimension より大きくしてください。

Phase 2 の NPZ は member 順を保った control/physical parameter/full trajectory/
correction path ensemble、ridge covariance、iteration diagnostics を保存します。
parameter posterior の一点化や Gaussian summary を正本にはしません。

Phase 3 の NPZ は strong/weak の比較に加え、weak posterior の raw innovations、
decoded residual-wrench path、static parameters、full trajectory、correction path を
member-aligned arrays として保存します。

Phase 4 の NPZ は ridge coordinate / quotient / path の raw law、各 ensemble size
の raw law と convergence 指標、または mode 別の full posterior member と
pose/independent-measurement weight を保存します。mode-conditioned posterior は
選択 mode の raw ensemble そのもので、mode 横断の Gaussian summary ではありません。

Phase 5 の NPZ は、bag hash/record-time/window/controller/calibration provenance、
observed/nominal/posterior trajectory、raw parameter/Q/path ensemble、OU knot 解像度、
ridge と selected-mode 診断を同じファイルへ保存します。

Phase 6 の NPZ は source member ID、mode weights/conditioning、scenario assumption、
candidate scale と適用後 mass/PID、member-level forecast success/reason、
trajectory/correction path/loss、
mean/CVaR/failure/change score と最終候補を member-aligned arrays として保存します。

NPZ は pickle を使わず、次を保存します。

- reference position / RPY
- nominal と truth の full latent trajectory
- nominal と truth の rotor/gimbal command
- pose-only observations と covariance
- correction translation / rotation-vector path

`particles` や `weights` は Phase 1 の出力には存在しません。

## C++ controller golden

`tests/grape_param_estim/test_controller.py` は、同一の state/reference/PID state
を Python port と ROS 非依存 C++ exact oracle の両方へ入力します。比較対象は
4 rotor thrust、4 gimbal angle、8次元 vectoring force、PID integral stateです。

oracle の provenance は次です。

- repository: `/home/leus/catkin_ws/src/jsk_aerial_robot`
- source commit: `9ae2159277489ef74892486291655deac2dc38dc`
- protocol: `grape.exact-controller-oracle/v1`
- fidelity: `pc_exact`

C++ executable が build 済みなら unit test は live oracle も実行します。未 build
環境でも、同じ oracle から固定した数値 fixture との比較は必ず実行されます。

actuator は estimator 用の明示的な連続近似です。FC firmware の battery/PWM
変換そのものではありません。対象 synthetic episode では rotor 上限が非活性で、
truth 側だけへ delay・一次遅れを入れて actuator mismatch を生成します。

## テスト

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo
PYTHONPATH=ros/examples/grape-param-estim/src:${PYTHONPATH} \
  /usr/bin/python3 -m unittest discover \
  -s tests/grape_param_estim -p 'test_*.py' -v
```

catkin からも同じ test directory を登録しています。

```bash
cd /home/leus/catkin_ws
catkin run_tests grape_param_estim
catkin_test_results build/grape_param_estim
```
