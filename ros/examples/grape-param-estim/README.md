# grape_param_estim

Grape の full closed-loop system を forecast operator として使い、対角 process-noise covariance `Q` の推定と fixed-Q static-parameter assimilation を分けた二段階 ensemble filter / smoother の実装です。
現行の実 rosbag 推定は、Stage 1 の diagonal-Q EM と Stage 2 の augmented EnKF / EnRTS を順番に実行します。

現行の二段階推定に加え、synthetic experiment、旧 strong/weak-constraint 回帰、検証、実 rosbag inspection、posterior-predictive controller 評価を実装しています。
旧 Streamlit GUI、rosbag open-loop replay、segment reset、static particle estimator、旧 result schema との後方互換は意図的に削除しました。

## 収録 rosbag で GUI を起動する最短デモ

初回だけ後述の Desktop GUI セットアップと `catkin build grape_param_estim --no-deps` を済ませてください。
次のコマンドは、指定された2026年6月12日の飛行を新しい一時projectへコピーし、GUIをbare `rosrun`で起動してinspectionを自動開始します。
このbagは完結した `flight_state=5` を含まず、control-activeな `flight_state=3` を警告付きで選ぶ失敗飛行の例です。

```bash
cd /home/leus/catkin_ws
source devel/setup.bash
rosrun grape_param_estim run_gui.py \
  --projects-root /tmp/grape-param-estim-failed-demo \
  --bag "$(rospack find grape_param_estim)/samples/rosbags/20260612_grape_hovering_4_2026-06-12-17-33-59.bag"
```

inspectionが完了するとGUIは `Bag browser` へ自動的に移動し、`Trajectory`、`Flight state`、3D world trajectoryへpreviewを表示します。
`Master`、`Correction transform`、`Residual wrench` はStage 2のfixed-Q parameter estimation結果を表示する領域なので、Stage 2完了前は空です。
このbagにはhardware configuration provenanceが記録されていないため、GUIはconfiguration group確認画面を開きます。
単独bagのデモでは既定の `single-bag-bd3fc7f71797` をそのまま確認すると、欠落warningを保持したままこのbagだけのgroupとして `Use` が有効になり、toolbarの `Run estimation…` を実行できます。
`Run estimation…` は二段階の実行方法を選ぶ画面を開き、推奨の `Run one stage at a time`（`STEP`）または連続実行する `Run all stages`（`ALL`）を選べます。
同じgroup IDを複数bagへ指定する操作は、それらが同じpayload、rotor、geometry、robot model、wiring、hardwareであると利用者が確認できる場合に限ります。
選択したbagのconfirmed configuration fingerprintが混在する場合、現行staged workflowは実行を拒否し、mismatchを許すoverrideはありません。
実行中に `Stop` した場合、不完全なstage artifactは `cancelled` として読み込まず、bagは `ready` に戻るため同じstageを再試行できます。
Stage 1 が完了していれば、そのartifactは次回起動でも検証後に再利用され、Stage 2 から再開できます。
一方、実行中stageの数値計算途中から再開するcheckpointはなく、cancel・failure・アプリ再起動で中断されたstageは先頭から再試行します。

次のコマンドは、2026年6月13日の成功飛行を同じGUI経路で開きます。
このbagは完全hold-out検証にも使った例で、完結した `flight_state=5` 区間を211 samples含みます。
選定された完結episodeはtakeoff、約21.06秒のhover、land、stopまで遷移し、hover位置referenceに対するRMSEは約0.0602 mです。

```bash
cd /home/leus/catkin_ws
source devel/setup.bash
rosrun grape_param_estim run_gui.py \
  --projects-root /tmp/grape-param-estim-success-demo \
  --bag "$(rospack find grape_param_estim)/samples/rosbags/20260613_grape_hovering_3_2026-06-13-15-12-51.bag"
```

両方を一つのprojectへ読み込む場合は、同じコマンドへ2個の `--bag` を指定できます。
収録ファイルは元bagのbasenameと内容を維持しており、GUIはsourceを変更せずproject側へコピーします。
どちらのbagもhardware configuration provenanceの一部を記録していないため、inspection後にGUIがconfiguration groupの確認を求めますが、これはtopic/data欠落を意味しません。

## Synthetic closed-loop flight

- `PoseLinearController::controlCore()` と `GimbalrotorController::controlCore()` の対象実験 branch を移植した stateful Python controller
  - position mode の x/y/z/roll/pitch/yaw PID
  - yaw shortest-angle error
  - z integral の非負制約
  - roll/pitch integration gate
  - P/I/D/error/output clamp
  - `gimbal_calc_in_fc=false`、`gimbal_dof=1`、fully-actuated allocation
- Grape URDF の四つの vectoring rotor、full inertia tensor、CoG、反トルク
- actual gimbal angle に応じて全リンクから再計算する controller 側の CoG、full inertia、thrust-link origin と、固定 arm-coordinate basis
- thrust/gimbal delay、first-order response、force/angle/rate saturation
- position、quaternion、world velocity、body angular velocityを持つ full 6-DoF plant
- nominal plant と、parameter mismatch・drag・actuator mismatch・時間変化外乱を持つ truth plant
- x/y/z/roll/pitch/yaw を同時に励起する一つの連続 closed-loop episode
- position/orientation だけの observation generator
- `Z_k = T_nominal,k^-1 T_real,k` の correction-transform path

## 現行の二段階推定

### Stage 1: diagonal-Q EM

- body-frame residual wrench を6次元の OU process state として EnKF / EnRTS で filtering / smoothing
- `Q = diag(q_Fx, q_Fy, q_Fz, q_tau_x, q_tau_y, q_tau_z)` の6成分を別々に保持
- position / orientation の observation covariance `R`、vehicle parameter、初期 delay、OU correlation time はこのstage中に固定
- E-step の smoothed wrench pathからOU sufficient statisticsを集計し、M-stepで6個のstationary varianceを更新
- `Q` の各成分、単位、frame、EM trace、入力fingerprintをstrict artifactへ保存

`Q` を単位行列の定数倍にはせず、force 3軸とtorque 3軸の異なるスケールを保持します。
residual wrenchの各時刻値はdynamic stochastic stateであり、時刻ごとの最適化変数としてparameter vectorへ追加しません。

### Stage 2: fixed-Q static parameters

Stage 1 の検証済み `Q` artifactを固定入力とし、19個のshared static coordinatesとbag-local dynamic stateをaugmented EnKFで逐次更新し、最後にEnRTSでfixed-interval smoothingします。
shared static coordinatesの内訳は、vehicle parameter 18個とcontinuous constant delay 1個です。

一つのbagに対する推定未知量の内訳は次のとおりです。

| 区分 | 内訳 | 次元 |
|---|---|---:|
| shared static | mass 1、full SPD inertia 6、CoG 3、force effectiveness 4、torque effectiveness 4、delay 1 | 19 |
| bag-local initial | position / orientation tangent / linear velocity / angular velocity 12、PID integral 6、actuator thrust / gimbal 8 | 26 |
| 合計 | `19 + 26` | 45 |

時刻ごとのfilter stateはshared static 19成分とdynamic 32成分からなる51次元です。
dynamic 32成分の内訳はrigid-body state 12、PID integral 6、actuator state 8、現在のresidual wrench 6です。
residual wrench 6成分は `Q` に従うMarkov processとして毎intervalで遷移し、全時刻分を未知parameterとして積み上げません。

51次元analysis anomalyに直交する6本のprocess-noise directionをexact ensembleで確保するには、中心化で失う1自由度も含めて `51 + 6 + 1 = 58` members以上が必要です。
そのため現行staged workflowはensemble sizeを58以上に制限し、既定値は128です。
複数bagでは19個のstatic coordinatesだけを共有し、26個の初期座標とdynamic pathはbagごとに独立に保ちます。

### 識別性と検証

- common-scale exact ridge 上のfull closed-loop rolloutとpose likelihoodの不変性
- raw ridge coordinate、17次元quotient law、prior-whitened information leak、correction-path coverage
- position / orientationを分離したposterior predictive metricと完全hold-out flight
- plant actuator channel wiringをmodeごとに独立同化するsynthetic regression

mass・inertia・effectivenessにはpose-only観測で識別できないridgeが残るため、marginalの狭さだけを推定成功と解釈しません。
解析・有限差分Jacobian、segment reset、parameter hard bounds、particle weightsは使いません。

## Real rosbag assimilation

- rosbag の header stamp ではなく record time を正本にした実データ adapter
- `flight_state=5` の一つの連続 airborne 区間を既定 window とし、別 flight や不連続区間を連結しない episode selection
- CoG odometry の位置と baselink odometry の姿勢だけを likelihood に使用（twist、IMU、加速度は観測へ追加しない）
- 記録された `PoseControlPid` reference/feedforward、PID integral、dynamic-reconfigure update の gain snapshot、gimbal/thrust anchor の因果的復元
- preflight 静止区間からロバスト推定する位置・SO(3) 観測 covariance
- nominal vehicle modelを固定したStage 1で6成分のdiagonal `Q` をEM推定
- Stage 1 artifactの `Q` を固定したStage 2でshared static parameterとbag-local stateをEnKF / EnRTS推定
- raw static parameter、filter / smoother state、residual-wrench path、full latent trajectory、correction-transform path、ridge診断のmember-aligned保存

既定では完結した episode 内の最長 `flight_state=5` interval を選びます。
存在しない場合は control-active interval を警告付き候補として返し、人間が GUI 上で確認します。
ground-contact model を持たないため、接地区間を free-flight forecast へ自動追加しません。

## PID proposal evaluation

- selected-mode raw physical member を平均化せず、各 member の mass、full inertia、CoG、rotor effectiveness から `xy`、`z`、`roll_pitch`、`yaw` の exact P/I/D proposal を導出
- controller nominal model と PID limit は固定し、controller nominal mass を候補にしない
- current、明示的に選択した member-derived candidate、exact user candidate を、全 raw member・全 selected bag の full closed-loop simulation で評価
- position RMSE、orientation RMSE、各 maximum error、forecast completion、numerical failure を単位別に保持し、bag equal-weight mean / upper CVaR を計算
- weighted sum で順位を作らず、独立した物理指標と completion による Pareto dominance を保存
- candidate × member × bag の trajectory、true correction translation / rotation-vector、failure reason を pickle-free directory bundle に保存

既定の scenario は、同化時と同じ reference・member 初期状態・posterior residual wrench path を繰り返す counterfactual です。
したがって「同じ実験を controller parameter だけ変えて再試行する」提案であり、新しい風を予言したものではありません。
request で `residual_policy="zero"` を選ぶ場合も、その仮定を artifact に明記します。

位置・姿勢 threshold は実験要件として設定された場合だけ用い、未設定時は `Not configured` として Pareto 判定から外します。
proposal YAML は exact gain だけを出力し、dynamic_reconfigure への自動書き込みは行いません。
dynamic_reconfigure は利用者が選んだ gain を controller へ書き込む機構であり、parameter 推定や PID 候補生成は行いません。

## 実行

### Desktop GUI

GUI は estimator の ROS Python 環境とは別の Python 3.10 以上の環境へインストールします。
rosbag の inspection と同化 worker は `QProcess` から catkin 環境の `/usr/bin/python3` で起動されます。
検証済み環境は pyenv の Python 3.10.18 と `gui/.venv` です。

```bash
/home/leus/.pyenv/versions/3.10.18/bin/python -m venv \
  ros/examples/grape-param-estim/gui/.venv
ros/examples/grape-param-estim/gui/.venv/bin/python -m pip install -e \
  ros/examples/grape-param-estim/gui

source /home/leus/catkin_ws/devel/setup.bash
rosrun grape_param_estim run_gui.py
```

devel space の `rosrun` は source script の shebang ではなく、catkin が生成した relay の Python で開始します。
そのため host 固有の venv 絶対パスを shebang へ固定せず、launcher が PySide6 を import する前に interpreter を選んで `execve` します。
選択順は `GRAPE_PARAM_ESTIM_GUI_PYTHON`、active `VIRTUAL_ENV`、package 内 `gui/.venv`、現在の Python 3.10 以上です。
したがって上記 workspace では venv の activate や GUI 用環境変数なしで `rosrun grape_param_estim run_gui.py` を実行できます。
検証環境の package-local `qt-runtime` も同じ再実行時にだけ自動追加します。
worker interpreter を変更する場合は `GRAPE_PARAM_ESTIM_WORKER_PYTHON` に catkin/ROS package を読み込める interpreter の絶対パスを設定します。
同化画面で `forecast` と表示される処理は、一本の順方向軌道だけではありません。
Stage 1 は各EM iterationで全memberのEnKF forecastとEnRTS smoothingを行い、Stage 2も各観測intervalで全memberを順方向に伝播してanalysisした後、全区間を後向きにEnRTS smoothingします。
したがって主な計算量は概ねmember数、時刻interval数、Stage 1のEM反復数に比例し、既定128 membersでは単一rolloutより大幅に重くなります。

各memberの順方向伝播は独立なので、2 workers以上では `spawn` process poolをfilter pass中ずっと保持し、intervalごとのprocess生成を避けます。
`forecast_workers="auto"` はCPU affinityの半数、ensemble size、32の最小値を使うため、自動選択の上限は32 workersです。
明示値はproject manifestの `estimator_settings.forecast_workers` で指定でき、`1` はprocess間通信を使わない直列の参照経路です。
GUIはBLASの入れ子並列によるoversubscriptionを避けるため、`OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS` が未設定の場合だけ既定値1をworkerへ渡し、利用者が明示した環境変数は保持します。
短いwindowや小さいensembleではprocess間通信の固定費により直列より遅い場合がありますが、長い実bagではpersistent poolによりmember並列を利用できます。
GUI の `Save Project` は raw rosbag、inspection、run、PID evaluation、GUI state を含む標準 ZIP/ZIP64 を保存し、`Load Project` は同梱 bag の SHA256 を検査して `projects/` 以下へ展開します。

`Next experiment` では shared selection の raw member を明示し、current、member-derived exact candidate、任意で入力した exact 4 x 3 user candidate を評価します。
user candidate の初期値は baseline bag の記録済み controller snapshot だけから復元し、別設定値へ fallback しません。
baseline controller snapshot、`posterior_replay` / `zero`、CVaR level、explicit selection target を画面上で選択でき、threshold は既定で `Not configured` のままです。
実行は同じ progress / ETA / cancel 経路を使い、complete artifact だけを自動ロードします。
結果の3D比較も、画面で選択中の bag・member・candidate に対応する保存済み forecast path だけを表示します。

実環境では `gui/.venv` に PySide6 6.9.3、pyqtgraph 0.14.0、PyVista 0.46.5、PyVistaQt 0.11.4、VTK 9.5.2 を導入し、Qt widget、staged workflow、project、plot / 3DのGUI testを実行できる状態にしました。
さらに `DISPLAY=:1`、Qt `xcb` backend、Mesa software rendering で実 UI と VTK を起動し、Master、Bag browser の world / correction、PID の translation / rotation / trajectory を視覚確認しました。
14 枚の PNG と機械可読な `summary.json` は `/tmp/grape-gui-visual-acceptance` にあります。
ウィンドウ画像は X server の `QScreen.grabWindow` で取得しているため、ネイティブ VTK 子画面も含みます。
各 VTK framebuffer も別画像として保存しています。
再実行には `gui/tests/visual_acceptance.py` へ strict-load 可能な assimilation bundle、同じ run ID の PID evaluation bundle、出力 directory を指定します。

ホスト側に不足していた `libxcb-cursor0` は sudo で system install せず、deb を `gui/.venv/qt-runtime` へ展開し、その `usr/lib/x86_64-linux-gnu` を受入実行時の `LD_LIBRARY_PATH` に追加しました。
これは検証環境だけの補完であり、system library は変更していません。
実 artifact は変更せず、GUI freshness 検査に必要な `project_request_fingerprint` は `/tmp/grape-visual-assimilation-run` の視覚確認用コピーにだけ追加しました。

### Legacy synthetic regression tools

次のstrong/weak-constraintコマンドはsynthetic回帰と旧方式の比較用であり、現行の実rosbag二段階workflowではありません。

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

Strong-constraint の Experiment A（観測 noise realization はゼロ、covariance は正定値）と Experiment B（位置姿勢 noise のみ）は次で実行します。

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

実 rosbag の追加、inspection、区間選択、二段階推定、project 保存・復元は Desktop GUI から実行します。
GUIはworker用request JSONをproject内へ保存し、inspectionには `grape_inspect_flights.py`、Stage 1には `grape_estimate_diagonal_q.py`、Stage 2には `grape_estimate_augmented_parameters.py` を起動します。

Stage 1のdiagonal-Q artifactは、6個のstationary variance、固定 `R`、bag provenance、EM trace、smoothed residual-wrench lawを保存します。
Stage 2のassimilation run directoryは、上流diagonal-Q artifactのcontent fingerprint、bag hash / record-time / window / controller provenance、shared posterior、bagごとのobserved / nominal / forecast / analysis / smoother trajectory、residual-wrench path、ridge診断を保存します。
入力または上流artifactが変わると既存stageは `STALE` になり、自動再利用しません。

Strong-constraint NPZ は member 順を保った control/physical parameter/full trajectory/ correction path ensemble、ridge covariance、iteration diagnostics を保存します。
parameter posterior の一点化や Gaussian summary を正本にはしません。

Weak-constraint NPZ は strong/weak の比較に加え、weak posterior の raw innovations、decoded residual-wrench path、static parameters、full trajectory、correction path を member-aligned arrays として保存します。

Validation NPZ は ridge coordinate / quotient / path の raw law、各 ensemble size の raw law と convergence 指標、または mode 別の full posterior member と pose/independent-measurement weight を保存します。
mode-conditioned posterior は選択 mode の raw ensemble そのもので、mode 横断の Gaussian summary ではありません。

PID proposal evaluation directory は source member ID、mode law、scenario assumption、current/proposed exact PID、candidate/member/bag ごとの forecast success/reason、trajectory/correction path、単位別 mean/CVaR/Pareto 指標、提案 YAML を保存します。

成功飛行を fitting から隔離した検証は専用 request で実行できます。
source run から移送する値は raw physical member と constant delay だけで、held-out 側の residual wrench は zero、観測は先頭 pose / velocity anchor と最後の scoring 以外には使いません。

```bash
rosrun grape_param_estim grape_validate_held_out_flight.py \
  --request /path/to/held-out-validation-request.json \
  --output /path/to/held-out-validation
```

現行の実装契約は [`lectures/implementation_ja.md`](lectures/implementation_ja.md) に記録しています。
指定された失敗bagと隔離した成功bagの [`lectures/real_flight_validation_ja.md`](lectures/real_flight_validation_ja.md) は、旧time-indexed residual-wrench optimizerによる過去結果であり、現行二段階推定の検証結果ではありません。

NPZ は pickle を使わず、次を保存します。

- reference position / RPY
- nominal と truth の full latent trajectory
- nominal と truth の rotor/gimbal command
- pose-only observations と covariance
- correction translation / rotation-vector path

`particles` や `weights` は synthetic output には存在しません。

## C++ controller golden

`tests/grape_param_estim/test_controller.py` は、同一の state/reference/PID state を Python port と ROS 非依存 C++ exact oracle の両方へ入力します。
比較対象は 4 rotor thrust、4 gimbal angle、8次元 vectoring force、PID integral stateです。

oracle の provenance は次です。

- repository: `/home/leus/catkin_ws/src/jsk_aerial_robot`
- source commit: `9ae2159277489ef74892486291655deac2dc38dc`
- protocol: `grape.exact-controller-oracle/v1`
- fidelity: `pc_exact`

C++ executable が build 済みなら unit test は live oracle も実行します。
未 build 環境でも、同じ oracle から固定した数値 fixture との比較は必ず実行されます。

actuator は estimator 用の明示的な連続近似です。
FC firmware の battery/PWM 変換そのものではありません。
対象 synthetic episode では rotor 上限が非活性で、truth 側だけへ delay・一次遅れを入れて actuator mismatch を生成します。

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

GUI test suite は次で実行します。
上記の検証済みvenvではQt widget、二段階workflow、artifact再利用、plot / 3D testを含むsuiteを実行します。
依存 package が揃わない環境では Qt widget / 3D test だけを skip し、request、project archive、artifact loader、launcher、signal cancel の pure tests は実行されます。

```bash
PYTHONPATH=ros/examples/grape-param-estim/gui/src:ros/examples/grape-param-estim/src \
  ros/examples/grape-param-estim/gui/.venv/bin/python -m unittest discover \
  -s ros/examples/grape-param-estim/gui/tests -p 'test_*.py' -v
```
