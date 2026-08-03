# grape_param_estim

Grape の full closed-loop system を forecast operator として使う、
weak-constraint ensemble smoothing の実装です。推定器と GUI 統合の設計は
`docs/plans/grape_weak_constraint_ienks_estimation_plan_ja.md` と
`docs/plans/grape_param_estim_gui_integration_next_experiment_plan_ja.md` に従います。

synthetic experiment、strong/weak-constraint smoothing、検証、実 rosbag 同化、
posterior-predictive controller 評価を実装しています。旧 Streamlit GUI、rosbag
open-loop replay、segment reset、static particle estimator、旧 result schema との
後方互換は意図的に削除しました。

## Synthetic closed-loop flight

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

## Strong-constraint experiment

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

## Weak-constraint experiment

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

## Ridge・ensemble convergence・mode validation

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
この検証では particle-based correction を追加していません。

## Real rosbag assimilation

- rosbag の header stamp ではなく record time を正本にした実データ adapter
- `flight_state=5` の一つの連続 airborne 区間を既定 window とし、別 flight や
  不連続区間を連結しない episode selection
- CoG odometry の位置と baselink odometry の姿勢だけを likelihood に使用
  （twist、IMU、加速度は観測へ追加しない）
- 記録された `PoseControlPid` reference/feedforward、PID integral、
  dynamic-reconfigure update の gain snapshot、gimbal/thrust anchor の因果的復元
- preflight 静止区間からロバスト推定する位置・SO(3) 観測 covariance
- body residual wrench を sparse OU knot で表し、piecewise-linear wrench の
  integration interval 平均を全 RK4 stage に保持する連続 weak forecast
- 観測 pose が要求する wrench と、観測 pose 上で因果 replay した nominal
  controller/actuator wrench の差による Q scale・相関時間の事前校正
- raw static parameter、OU innovation/knot/interval wrench、full latent trajectory、
  correction-transform path、ridge/mode/resolution 診断の member-aligned 保存

既定では完結した episode 内の最長 `flight_state=5` interval を選びます。存在しない
場合は control-active interval を警告付き候補として返し、人間が GUI 上で確認します。
ground-contact model を持たないため、接地区間を free-flight forecast へ自動追加しません。

## PID proposal evaluation

- selected-mode raw physical member を平均化せず、各 member の mass、full inertia、
  CoG、rotor effectiveness から `xy`、`z`、`roll_pitch`、`yaw` の exact P/I/D
  proposal を導出
- controller nominal model と PID limit は固定し、controller nominal mass を候補にしない
- current、明示的に選択した member-derived candidate、exact user candidate を、
  全 raw member・全 selected bag の full closed-loop simulation で評価
- position RMSE、orientation RMSE、各 maximum error、forecast completion、numerical
  failure を単位別に保持し、bag equal-weight mean / upper CVaR を計算
- weighted sum で順位を作らず、独立した物理指標と completion による Pareto dominance を保存
- candidate × member × bag の trajectory、true correction translation / rotation-vector、
  failure reason を pickle-free directory bundle に保存

既定の scenario は、同化時と同じ reference・member 初期状態・posterior residual
wrench path を繰り返す counterfactual です。したがって「同じ実験を controller
parameter だけ変えて再試行する」提案であり、新しい風を予言したものではありません。
request で `residual_policy="zero"` を選ぶ場合も、その仮定を artifact に明記します。

位置・姿勢 threshold は実験要件として設定された場合だけ用い、未設定時は
`Not configured` として Pareto 判定から外します。proposal YAML は exact gain だけを
出力し、dynamic_reconfigure への自動書き込みは行いません。dynamic_reconfigure は
利用者が選んだ gain を controller へ書き込む機構であり、parameter 推定や PID 候補生成は
行いません。

## 実行

### Desktop GUI

GUI は estimator の ROS Python 環境とは別の Python 3.10 以上の環境へ
インストールします。rosbag の inspection と同化 worker は `QProcess` から
catkin 環境の `/usr/bin/python3` で起動されます。検証済み環境は pyenv の
Python 3.10.18 と `gui/.venv` です。

```bash
/home/leus/.pyenv/versions/3.10.18/bin/python -m venv \
  ros/examples/grape-param-estim/gui/.venv
source ros/examples/grape-param-estim/gui/.venv/bin/activate
python -m pip install -e ros/examples/grape-param-estim/gui

source /home/leus/catkin_ws/devel/setup.bash
rosrun grape_param_estim run_gui.py
```

catkin が launcher の shebang を ROS 側 Python に書き換えていても、launcher は PySide6 を
import する前に active `VIRTUAL_ENV/bin/python` へ一度だけ移り直します。明示的に GUI
interpreter を選ぶ場合は `GRAPE_PARAM_ESTIM_GUI_PYTHON` に Python 3.10 以上の実行ファイルを
設定します。worker interpreter を変更する場合は
`GRAPE_PARAM_ESTIM_WORKER_PYTHON` に catkin/ROS package を読み込める
interpreter の絶対パスを設定します。GUI の `Save Project` は raw rosbag、
inspection、run、PID evaluation、GUI state を含む標準 ZIP/ZIP64 を保存し、
`Load Project` は同梱 bag の SHA256 を検査して `projects/` 以下へ展開します。

`Next experiment` では shared selection の raw member を明示し、current、member-derived
exact candidate、任意で入力した exact 4 x 3 user candidate を評価します。user candidate の
初期値は baseline bag の記録済み controller snapshot だけから復元し、別設定値へ fallback
しません。baseline controller snapshot、`posterior_replay` / `zero`、CVaR level、explicit
selection target を画面上で選択でき、threshold は既定で `Not configured` のままです。
実行は同じ progress / ETA / cancel 経路を使い、complete artifact だけを自動ロードします。
結果の3D比較も、画面で選択中の bag・member・candidate に対応する保存済み forecast path
だけを表示します。

実環境では `gui/.venv` に PySide6 6.9.3、pyqtgraph 0.14.0、PyVista 0.46.5、
PyVistaQt 0.11.4、VTK 9.5.2 を導入し、GUI test 49 / 49、skip 0 を確認しました。
さらに `DISPLAY=:1`、Qt `xcb` backend、Mesa software rendering で実 UI と VTK を起動し、
Master、Bag browser の world / correction、PID の translation / rotation / trajectory を
視覚確認しました。14 枚の PNG と機械可読な `summary.json` は
`/tmp/grape-gui-visual-acceptance` にあります。
ウィンドウ画像は X server の `QScreen.grabWindow` で取得しているため、ネイティブ VTK
子画面も含みます。各 VTK framebuffer も別画像として保存しています。再実行には
`gui/tests/visual_acceptance.py` へ strict-load 可能な assimilation bundle、同じ run ID の
PID evaluation bundle、出力 directory を指定します。

ホスト側に不足していた `libxcb-cursor0` は sudo で system install せず、deb を
`gui/.venv/qt-runtime` へ展開し、その `usr/lib/x86_64-linux-gnu` を受入実行時の
`LD_LIBRARY_PATH` に追加しました。これは検証環境だけの補完であり、system library は
変更していません。実 artifact は変更せず、GUI freshness 検査に必要な
`project_request_fingerprint` は `/tmp/grape-visual-assimilation-run` の視覚確認用コピーにだけ
追加しました。

```bash
cd /home/leus/catkin_ws
catkin build grape_param_estim --no-deps
source devel/setup.bash

rosrun grape_param_estim grape_generate_synthetic_flight.py \
  --output /tmp/grape_synthetic_closed_loop.npz
```

perfect-model の恒等性を確認する場合は次を使います。

```bash
rosrun grape_param_estim grape_generate_synthetic_flight.py \
  --perfect-model \
  --duration 3.0 \
  --output /tmp/grape_synthetic_perfect.npz
```

Strong-constraint の Experiment A（観測 noise realization はゼロ、covariance は正定値）と
Experiment B（位置姿勢 noise のみ）は次で実行します。

```bash
rosrun grape_param_estim grape_run_strong_constraint_experiment.py \
  --experiment A \
  --output /tmp/grape_strong_constraint_a.npz

rosrun grape_param_estim grape_run_strong_constraint_experiment.py \
  --experiment B \
  --output /tmp/grape_strong_constraint_b.npz
```

Weak-constraint の Experiment C は次で実行します。

```bash
rosrun grape_param_estim grape_run_weak_constraint_experiment.py \
  --output /tmp/grape_weak_constraint_c.npz
```

三つの検証は、それぞれ独立に実行できます。

```bash
rosrun grape_param_estim grape_validate_assimilation.py \
  --section ridge \
  --output /tmp/grape_ridge_validation.npz

rosrun grape_param_estim grape_validate_assimilation.py \
  --section convergence \
  --output /tmp/grape_ensemble_convergence.npz

rosrun grape_param_estim grape_validate_assimilation.py \
  --section mode \
  --output /tmp/grape_mode_validation.npz
```

実 rosbag の追加、inspection、区間選択、joint smoothing、project 保存・復元は
Desktop GUI から実行します。GUI が worker 用 request JSON を project 内へ保存し、
`grape_inspect_flights.py --request ... --output ...` と
`grape_assimilate_flights.py --request ... --output ...` を起動します。

artifact の `q_resolution_sufficient` が false の場合、その knot 数で Q の
時間解像度が十分だとは主張できません。assimilation request の `maximum_knots=0` は
校正された OU bridge criterion を満たす全 knot を使います。計算量を理由に knot を
減らした事実と診断は artifact へ保存されます。

real joint solver は ensemble size が augmented dimension より小さい場合も、実際の prior
ensemble span 内で更新します。要求 prior forecast が非有限になる場合は、全 member の中心
からの偏差を共通係数で縮めた最初の有限 ensemble を実効 prior とし、要求 / 実効 ensemble、
scale、rank、失敗理由を保存して GUI に警告します。member の除外や有限な罰 residual への
置換は行いません。

Strong-constraint NPZ は member 順を保った control/physical parameter/full trajectory/
correction path ensemble、ridge covariance、iteration diagnostics を保存します。
parameter posterior の一点化や Gaussian summary を正本にはしません。

Weak-constraint NPZ は strong/weak の比較に加え、weak posterior の raw innovations、
decoded residual-wrench path、static parameters、full trajectory、correction path を
member-aligned arrays として保存します。

Validation NPZ は ridge coordinate / quotient / path の raw law、各 ensemble size
の raw law と convergence 指標、または mode 別の full posterior member と
pose/independent-measurement weight を保存します。mode-conditioned posterior は
選択 mode の raw ensemble そのもので、mode 横断の Gaussian summary ではありません。

Assimilation run directory は bag hash/record-time/window/controller/calibration provenance、
shared posterior、bag ごとの observed/nominal/posterior trajectory、raw parameter/Q/path
ensemble、OU knot 解像度、ridge と selected-mode 診断を可変長 bag ごとに保存します。

PID proposal evaluation directory は source member ID、mode law、scenario assumption、
current/proposed exact PID、candidate/member/bag ごとの forecast success/reason、
trajectory/correction path、単位別 mean/CVaR/Pareto 指標、提案 YAML を保存します。

成功飛行を fitting から隔離した検証は専用 request で実行できます。source run から移送する
値は raw physical member と constant delay だけで、held-out 側の residual wrench は zero、
観測は先頭 pose / velocity anchor と最後の scoring 以外には使いません。

```bash
rosrun grape_param_estim grape_validate_held_out_flight.py \
  --request /path/to/held-out-validation-request.json \
  --output /path/to/held-out-validation
```

実装契約は [`lectures/implementation_ja.md`](lectures/implementation_ja.md)、指定された失敗
bag と隔離した成功 bag の実測結果、Q-knot 感度、推奨なしの判定は
[`lectures/real_flight_validation_ja.md`](lectures/real_flight_validation_ja.md) に記録しています。

NPZ は pickle を使わず、次を保存します。

- reference position / RPY
- nominal と truth の full latent trajectory
- nominal と truth の rotor/gimbal command
- pose-only observations と covariance
- correction translation / rotation-vector path

`particles` や `weights` は synthetic output には存在しません。

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

GUI test suite は次で実行します。上記の検証済み venv では Qt widget / 3D test を含む
49 tests がすべて成功し、skip はありません。依存 package が揃わない環境では Qt widget /
3D test だけを skip し、request、project archive、artifact loader、launcher、signal cancel の
pure tests は実行されます。

```bash
PYTHONPATH=ros/examples/grape-param-estim/gui/src:ros/examples/grape-param-estim/src \
  ros/examples/grape-param-estim/gui/.venv/bin/python -m unittest discover \
  -s ros/examples/grape-param-estim/gui/tests -p 'test_*.py' -v
```
