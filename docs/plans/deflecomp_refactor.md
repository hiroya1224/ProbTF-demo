# plan.md: `online-deflecomp` の ROS1 catkin package 分割計画

## -1. 前提
- git commit は，main ブランチで構わない
- online-deflecomp は残してもらって，新たに用意した空の deflecomp ディレクトリに移植していってください 

## 0. 目的

現在の `online-deflecomp` では，たわみ補償・剛性推定・遅れ推定・ROS ノード・動的シミュレータ・デモが同じリポジトリ内で混在している．
この作業では，ROS1 の catkin workspace で扱いやすいように，以下の構成へ分割する．

```text
catkin_ws/src/
  deflecomp/
    deflecomp_core/
    deflecomp_ros/
    deflecomp_sim/
    deflecomp_description/
    deflecomp_msgs/        # 必要なら
    deflecomp_examples/    # 任意
```

最重要方針は次の通りである．

```text
補償器はシミュレータを知らない．
シミュレータは補償器を試すために存在する．
ROS ノードは補償器の thin wrapper にする．
```

したがって，`deflecomp_core` は ROS 非依存にし，`deflecomp_ros` と `deflecomp_sim` が `deflecomp_core` に依存する構造にする．

---

## 1. 分割後の責務

### 1.1 `deflecomp_core`

`deflecomp_core` は，たわみ補償の本体である．
ここには ROS 依存を入れない．

含めるもの：

- Pinocchio を用いたロボット幾何・力学 wrapper
- 重力トルク計算
- ばねモデル
- 平衡計算
- 平衡感度計算
- feedforward 補償指令生成
- feedback 補償
- Bingham 行列生成
- IMU 観測を使う観測モデル
- 剛性推定器
- 遅れ推定器
- 1 ステップの補償 pipeline
- 角度処理・線形代数などの utility

含めないもの：

- `rospy`
- `sensor_msgs`
- `std_msgs`
- `geometry_msgs`
- ROS topic 名
- ROS parameter 読み込み
- `rosbag`
- RViz 設定
- シミュレータ専用の真値生成

想定構成：

```text
deflecomp_core/
  package.xml
  CMakeLists.txt
  setup.py
  src/deflecomp_core/
    __init__.py

    robot/
      __init__.py
      pinocchio_robot.py
      gravity.py

    model/
      __init__.py
      spring.py
      equilibrium.py
      sensitivity.py

    control/
      __init__.py
      feedforward.py
      feedback.py
      limiter.py

    observation/
      __init__.py
      bingham.py
      imu_observation.py

    estimator/
      __init__.py
      stiffness_wekf.py
      delay_rls.py

    pipeline/
      __init__.py
      compensator.py

    utils/
      __init__.py
      angle.py
      linalg.py
```

### 1.2 `deflecomp_ros`

`deflecomp_ros` は実機・ROS 通信用の package である．
ここでは ROS topic / parameter / message と `deflecomp_core` の変換だけを行う．

含めるもの：

- 実機用 ROS node
- estimator / controller node
- parameter YAML
- launch file
- topic interface
- debug publish

含めないもの：

- 平衡計算の本体
- 剛性推定の本体
- 遅れ推定の本体
- Bingham 行列の数式実装
- 動的シミュレータ

想定構成：

```text
deflecomp_ros/
  package.xml
  CMakeLists.txt
  setup.py
  nodes/
    deflecomp_node.py
    stiffness_estimator_node.py        # 必要なら
    delay_estimator_node.py            # 必要なら

  launch/
    deflecomp.launch
    real_robot.launch

  config/
    simple6r.yaml
    estimator.yaml
    controller.yaml
    imu_frames.yaml
```

基本方針：

```text
ROS input
  -> ROS message を numpy 配列へ変換
  -> deflecomp_core を呼ぶ
  -> 結果を ROS message へ変換
  -> publish
```

### 1.3 `deflecomp_sim`

`deflecomp_sim` は，たわみを実現するロボットシミュレータである．
補償対象の真値を作るための package であり，補償器本体ではない．

含めるもの：

- 動的シミュレータ
- quasi-static / relax-to-equilibrium mode
- synthetic IMU 生成
- simulated `JointState` publish
- simulated `Imu` publish
- simulation launch

含めないもの：

- 剛性推定器の本体
- 補償指令生成の本体
- 実機用 ROS node

想定構成：

```text
deflecomp_sim/
  package.xml
  CMakeLists.txt
  setup.py
  src/deflecomp_sim/
    __init__.py
    dynamic_simulator.py
    sensor_simulator.py
    imu_publisher.py
    joint_state_publisher.py

  nodes/
    sim_node.py

  launch/
    sim.launch
    sim_with_deflecomp.launch

  config/
    sim_params.yaml
```

### 1.4 `deflecomp_description`

`deflecomp_description` はロボット記述 package である．
URDF / xacro / mesh / RViz 設定を置く．

想定構成：

```text
deflecomp_description/
  package.xml
  CMakeLists.txt
  urdf/
    simple6r.urdf
    simple6r.xacro        # 可能なら後で xacro 化

  meshes/
  rviz/
    simple6r.rviz
```

### 1.5 `deflecomp_msgs`

既存の `sensor_msgs/JointState`，`sensor_msgs/Imu`，`std_msgs/Float64MultiArray` で足りるなら作らない．

候補：

```text
deflecomp_msgs/
  msg/
    StiffnessEstimate.msg
    DelayEstimate.msg
    DeflecompDebug.msg
```

### 1.6 `deflecomp_examples`

ROS 非依存の offline demo や検証 script を置く．
研究用 script を `deflecomp_core` に混ぜないための置き場である．

想定構成：

```text
deflecomp_examples/
  package.xml
  CMakeLists.txt
  setup.py
  scripts/
    offline_demo.py
    simulator_flexible.py
    run_estimation_pipeline.py
```

---

## 2. 依存関係

依存方向は次のようにする．

```text
deflecomp_description
        ↑
        |
deflecomp_core
        ↑
        |
  ----------------
  |              |
deflecomp_ros   deflecomp_sim
  |
deflecomp_examples
```

より具体的には：

```text
deflecomp_core:
  depends on numpy, scipy, pinocchio
  must not depend on rospy or ROS messages


deflecomp_ros:
  depends on rospy, sensor_msgs, std_msgs, geometry_msgs, deflecomp_core


deflecomp_sim:
  depends on rospy, sensor_msgs, geometry_msgs, deflecomp_core, deflecomp_description


deflecomp_description:
  depends on urdf, xacro, robot_state_publisher if needed


deflecomp_msgs:
  depends on message_generation, std_msgs, sensor_msgs if added
```

禁止する依存：

```text
deflecomp_core -> deflecomp_ros
deflecomp_core -> deflecomp_sim
deflecomp_core -> rospy
deflecomp_core -> sensor_msgs
deflecomp_core -> std_msgs
deflecomp_core -> simple6r 専用 topic 名
deflecomp_core -> simulation-only true parameter
```

---

## 3. 既存ファイルの移動方針

現在のファイルを次のように移動する．
ファイル名は移動後に多少変更してよいが，まずは最小変更を優先する．

| 現在 | 移動先 | 備考 |
|---|---|---|
| `utils/robot.py` | `deflecomp_core/src/deflecomp_core/robot/pinocchio_robot.py` | Pinocchio wrapper |
| `controller/command.py` | `deflecomp_core/src/deflecomp_core/control/feedforward.py` | 補償指令生成 |
| `controller/equilibrium.py` | `deflecomp_core/src/deflecomp_core/model/equilibrium.py` | 平衡計算 |
| `utils/bingham.py` | `deflecomp_core/src/deflecomp_core/observation/bingham.py` | Bingham 行列 |
| `estimator/ekf.py` | `deflecomp_core/src/deflecomp_core/estimator/stiffness_wekf.py` | 剛性推定 |
| `estimator/cmd_lag_ekf.py` | `deflecomp_core/src/deflecomp_core/estimator/delay_rls.py` | 遅れ推定 |
| `estimator/lag_ekf.py` | `deflecomp_core/src/deflecomp_core/estimator/lag_ekf.py` | 残すなら core 側 |
| `estimator/observations.py` | 分割 | core と sim に分ける |
| `pipeline.py` | `deflecomp_core/src/deflecomp_core/pipeline/compensator.py` | ROS 非依存 pipeline |
| `simulation/dynamic_simulator.py` | `deflecomp_sim/src/deflecomp_sim/dynamic_simulator.py` | シミュレータ |
| `ros/sim_node.py` | `deflecomp_sim/nodes/sim_node.py` | simulation ROS node |
| `ros/estimator_node.py` | `deflecomp_ros/nodes/deflecomp_node.py` | 実機用 wrapper |
| `examples/offline_demo.py` | `deflecomp_examples/scripts/offline_demo.py` | 任意 |
| `examples/simulator_flexible.py` | `deflecomp_examples/scripts/simulator_flexible.py` | 任意 |
| `simple6r.urdf` | `deflecomp_description/urdf/simple6r.urdf` | description package |

---

## 4. 特に注意する分割点

### 4.1 ばねモデルを明示する

現在のコードでは，線形ばねモデルと円周整合な非線形ばねモデルが混在している．
このままだと，どの計算がどの近似に基づいているか分かりにくい．

`deflecomp_core.model.spring` に次のようなクラスを作る．

```python
class SpringModel:
    def torque(self, theta, theta_cmd, kp_vec):
        raise NotImplementedError

    def stiffness_diag(self, theta, theta_cmd, kp_vec):
        raise NotImplementedError


class LinearSpringModel(SpringModel):
    pass


class PeriodicSpringModel(SpringModel):
    pass
```

各クラスの意味：

```text
LinearSpringModel:
  tau_s = K (theta - theta_cmd)

PeriodicSpringModel:
  tau_s = 2 K sin((theta - theta_cmd) / 2)
```

`EquilibriumSolver`，`CommandGenerator`，`SensitivityCalculator` は，暗黙にばねモデルを決め打ちせず，必ず `SpringModel` を受け取る．

```python
solver = EquilibriumSolver(robot, spring_model)
command_generator = CommandGenerator(robot, spring_model)
sensitivity = SensitivityCalculator(robot, spring_model)
```

### 4.2 `ObservationBuilder` を分割する

現在の `ObservationBuilder.build_A_multi` は，真の剛性から真の平衡姿勢を解いて Bingham 行列を作る．
これは実機用の観測処理ではなく，simulation / synthetic observation の意味が強い．

したがって，次のように分ける．

```text
deflecomp_core.observation.imu_observation:
  - 実 IMU の重力方向から A_map を作る
  - frame id と観測値を管理する
  - true kp を知らない
  - true theta_eq を作らない


deflecomp_sim.sensor_simulator:
  - true theta_eq から仮想 IMU を作る
  - true theta_eq から synthetic A_map を作る
  - true kp を使ってよい
```

### 4.3 `DynamicSimulator` は core に置かない

`DynamicSimulator` は補償対象の真値を作るための simulation component である．
実機利用時には不要であるため，`deflecomp_core` には置かない．

置き場所：

```text
deflecomp_sim/src/deflecomp_sim/dynamic_simulator.py
```

### 4.4 ROS node を薄くする

`deflecomp_ros/nodes/deflecomp_node.py` は巨大な制御器本体にしない．
ROS node は以下だけを担当する．

1. parameter を読む
2. subscriber を作る
3. ROS message を numpy 配列に変換する
4. `deflecomp_core.pipeline.compensator.DeflectionCompensator.step(...)` を呼ぶ
5. 結果を publish する
6. debug 情報を publish する

実ロジックは `deflecomp_core` に置く．

---

## 5. `DeflectionCompensator` の導入

`deflecomp_core.pipeline.compensator` に，ROS 非依存の facade を作る．
ROS node は基本的にこれだけを呼べばよいようにする．

候補 API：

```python
class DeflectionCompensator:
    def __init__(
        self,
        robot,
        spring_model,
        stiffness_estimator,
        delay_estimator=None,
        command_generator=None,
        equilibrium_solver=None,
        config=None,
    ):
        pass

    def step(self, theta_ref, imu_observations, dt, stamp=None):
        """Run one compensation step.

        Args:
            theta_ref: desired rigid-joint reference angle.
            imu_observations: frame-wise IMU observations converted to plain numpy objects.
            dt: time step.
            stamp: optional timestamp. Do not require ROS time.

        Returns:
            A plain Python object or dataclass containing:
              - theta_cmd
              - theta_eq_hat
              - kp_hat
              - tau_hat
              - debug values
        """
        pass
```

戻り値は ROS message にしない．
`dataclasses.dataclass` などの plain Python object にする．

---

## 6. catkin package 作成方針

各 package には最低限以下を置く．

```text
package.xml
CMakeLists.txt
setup.py        # Python package を持つ場合
```

Python package を持つ場合，`CMakeLists.txt` には以下を入れる．

```cmake
catkin_python_setup()
```

node script には実行権限を付ける．

```bash
chmod +x deflecomp_ros/nodes/deflecomp_node.py
chmod +x deflecomp_sim/nodes/sim_node.py
```

Python import は catkin install/devel space で通るようにする．
`sys.path` を手動でいじらない．

---

## 7. 移行手順

### Phase 1: package skeleton を作る

1. `catkin_ws/src/deflecomp/` を作る．
2. 以下の package を作る．
   - `deflecomp_core`
   - `deflecomp_ros`
   - `deflecomp_sim`
   - `deflecomp_description`
   - 必要なら `deflecomp_examples`
3. 各 package に `package.xml`，`CMakeLists.txt`，必要なら `setup.py` を追加する．
4. `catkin_make` または `catkin build` が通ることを確認する．

完了条件：

```bash
catkin build
```

が空 package 状態で通る．

### Phase 2: description を移動する

1. `simple6r.urdf` を `deflecomp_description/urdf/simple6r.urdf` へ移動する．
2. URDF path を parameter で渡せるようにする．
3. 既存コード内に hard-coded な `simple6r.urdf` path があれば削除する．

完了条件：

- URDF path を外から指定できる．
- core は特定の相対 path に依存しない．

### Phase 3: core を移動する

1. `utils/robot.py` を `deflecomp_core.robot.pinocchio_robot` へ移動する．
2. `controller/command.py` を `deflecomp_core.control.feedforward` へ移動する．
3. `controller/equilibrium.py` を `deflecomp_core.model.equilibrium` へ移動する．
4. `utils/bingham.py` を `deflecomp_core.observation.bingham` へ移動する．
5. `estimator/ekf.py` を `deflecomp_core.estimator.stiffness_wekf` へ移動する．
6. `estimator/cmd_lag_ekf.py` を `deflecomp_core.estimator.delay_rls` へ移動する．
7. import を修正する．
8. core 内に ROS import が残っていないか確認する．

確認コマンド：

```bash
grep -R "import rospy\|from sensor_msgs\|from std_msgs\|from geometry_msgs" deflecomp_core/src || true
```

完了条件：

- `deflecomp_core` が ROS message なしで import できる．
- 主要 class / function の単体 import が通る．

### Phase 4: spring model を明示化する

1. `deflecomp_core.model.spring` を作る．
2. `LinearSpringModel` と `PeriodicSpringModel` を作る．
3. 平衡計算・指令生成・感度計算から，暗黙のばね式を減らす．
4. 既存挙動を壊さないよう，default は現行コードでオンライン制御に使っている `PeriodicSpringModel` に寄せる．
5. 線形ばねを使っていた箇所は，明示的に `LinearSpringModel` を渡す．

完了条件：

- どの処理が線形ばねか非線形ばねか，コード上で明示される．
- 既存の demo が大きく挙動変化しない．

### Phase 5: observation を core と sim に分割する

1. 実 IMU から Bingham 行列を作る部分を `deflecomp_core.observation.imu_observation` に置く．
2. true state から synthetic observation を作る部分を `deflecomp_sim.sensor_simulator` に置く．
3. `ObservationBuilder` が true kp / true theta_eq に依存するなら，それは sim 側へ移す．
4. core 側の観測処理は true parameter を要求しない形にする．

完了条件：

- 実機用処理が true kp を要求しない．
- synthetic observation は sim package に隔離される．

### Phase 6: simulator を移動する

1. `simulation/dynamic_simulator.py` を `deflecomp_sim` へ移動する．
2. `ros/sim_node.py` を `deflecomp_sim/nodes/sim_node.py` へ移動する．
3. import を修正する．
4. simulation launch を作る．

完了条件：

- `deflecomp_sim` 単体で simulated `JointState` と `Imu` を publish できる．
- `deflecomp_core` は `deflecomp_sim` に依存しない．

### Phase 7: ROS wrapper を作る

1. `ros/estimator_node.py` を `deflecomp_ros/nodes/deflecomp_node.py` へ移動する．
2. ノード内ロジックを段階的に `deflecomp_core.pipeline.compensator` へ移す．
3. node は wrapper に近づける．
4. config YAML を `deflecomp_ros/config/` に移す．
5. launch file を `deflecomp_ros/launch/` に作る．

完了条件：

- ROS node は `DeflectionCompensator.step(...)` を呼ぶ構造になる．
- ROS node 内に数式実装がほとんど残らない．

### Phase 8: examples を整理する

1. offline demo を `deflecomp_examples/scripts/` へ移す．
2. demo が `deflecomp_core` を import して動くようにする．
3. simulation demo は必要に応じて `deflecomp_sim` を使う．

完了条件：

- ROS なしで offline demo が実行できる．
- simulation 付き demo は launch で実行できる．

---

## 8. import 修正ルール

相対 import の多用は避ける．
catkin package として install/devel space で動くよう，絶対 import を使う．

例：

```python
from deflecomp_core.robot.pinocchio_robot import RobotArm
from deflecomp_core.model.equilibrium import EquilibriumSolver
from deflecomp_core.control.feedforward import theta_cmd_from_theta_ref
```

避ける：

```python
from ..utils.robot import RobotArm
import sys
sys.path.append(...)
```

---

## 9. 動作確認

最低限，次を確認する．

### 9.1 build

```bash
cd catkin_ws
catkin_make
# または
catkin build
```

### 9.2 core import

```bash
python3 - <<'PY'
from deflecomp_core.robot.pinocchio_robot import RobotArm
from deflecomp_core.model.equilibrium import EquilibriumSolver
from deflecomp_core.observation.bingham import simple_bingham_unit
print("core import ok")
PY
```

### 9.3 ROS 非依存チェック

```bash
grep -R "import rospy\|from sensor_msgs\|from std_msgs\|from geometry_msgs" deflecomp_core/src || true
```

この grep でヒットしないことを目標にする．

### 9.4 simulation launch

```bash
roslaunch deflecomp_sim sim.launch
```

確認する topic：

```bash
rostopic list
rostopic echo /joint_states
rostopic echo /imu/link1/data
```

### 9.5 compensation launch

```bash
roslaunch deflecomp_ros deflecomp.launch
```

確認する topic：

```bash
rostopic echo /deflecomp/theta_cmd
rostopic echo /deflecomp/kp_hat
rostopic echo /deflecomp/debug
```

---

## 10. 既存挙動の保存

この作業は package 分割が目的であり，アルゴリズム変更を主目的にしない．

原則：

1. まずは移動と import 修正を優先する．
2. 数式・制御則・推定則はできるだけ変えない．
3. 挙動変更が必要な場合は，別 commit に分ける．
4. ファイル移動 commit とロジック変更 commit を混ぜない．

推奨 commit 分割：

```text
1. create catkin package skeletons
2. move description files
3. move core modules without logic changes
4. introduce explicit spring model abstraction
5. split observation builder into core and sim parts
6. move simulator package
7. convert ROS estimator node into thin wrapper
8. move examples and update launch/config files
9. cleanup imports and documentation
```

---

## 11. 命名方針

### 11.1 package 名

```text
deflecomp_core
deflecomp_ros
deflecomp_sim
deflecomp_description
deflecomp_msgs
deflecomp_examples
```

### 11.2 Python 変数名

Python 変数名には ASCII 文字のみを使う．
ギリシャ文字や日本語を変数名に使わない．

よい例：

```python
theta_ref = None
theta_cmd = None
kp_hat = None
```

避ける例：

```python
θ_ref = None
剛性 = None
```

### 11.3 既存のクラス名

既存名が分かりにくい場合は，意味が明確な名前へ変更してよい．
ただし，大きな rename は別 commit に分ける．

候補：

```text
MultiFrameWeirdEKF -> MultiFrameStiffnessWEKF
CmdLagEKF -> CommandDelayRLS
DynamicSimulator -> FlexibleJointSimulator
```

---

## 12. README 更新

最終的に `deflecomp/README.md` を作り，次を書く．

1. package 全体の説明
2. 各 package の責務
3. build 方法
4. simulation の起動方法
5. 補償 node の起動方法
6. topic interface
7. parameter YAML の説明
8. offline demo の実行方法

README の最初には，次のような説明を入れる．

```text
This repository separates the deflection compensation logic from the simulator.
The core compensation, estimation, and observation models are implemented in
`deflecomp_core` without ROS dependencies. ROS nodes and simulation components
are provided as wrappers around the core package.
```

---

## 13. 完了条件

この分割作業は，次を満たせば完了とする．

1. `catkin_make` または `catkin build` が通る．
2. `deflecomp_core` が ROS 非依存で import できる．
3. `deflecomp_core` から `rospy` と ROS message import が消えている．
4. `deflecomp_sim` から simulated `JointState` と `Imu` を publish できる．
5. `deflecomp_ros` から補償指令 `theta_cmd` を publish できる．
6. URDF は `deflecomp_description` に移っている．
7. シミュレータ専用の true parameter が実機用 core logic に入り込んでいない．
8. 線形ばねと非線形ばねのどちらを使っているか，コード上で明示されている．
9. offline demo が `deflecomp_core` を使って実行できる．
10. README に package 分割の説明がある．

---

## 14. 作業時の優先順位

優先順位は次の通り．

1. 依存関係を正しくする．
2. `deflecomp_core` を ROS 非依存にする．
3. シミュレータを core から分離する．
4. ROS node を thin wrapper にする．
5. ばねモデルの混在を明示的に扱う．
6. その後に rename や整理を行う．

この順番を守ること．
最初から大規模 rename や設計変更をしすぎないこと．
