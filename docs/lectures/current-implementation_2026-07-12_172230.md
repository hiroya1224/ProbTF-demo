# ProbTF-demo current implementation notes

- 作成日時: 2026-07-12 17:22:30
- 対象リポジトリ: `ProbTF-demo`
- 対象状態: Phase 1 移行後の `main`
- 主な参照: `src/probtf`, `ros/core/probtf_core`, `ros/core/probtf_msgs`, `src/symaware_grasp/prob_tf`, `src/deflecomp_core/observation`

## 0. 要約

現状の `ProbTF-demo` は、Phase 1 の目的である「分散していた producer と確率表現を一つの文脈に集約する」段階までは達成している。root の `src/probtf` には ROS 非依存の分布表現、Bingham moment、evidence fusion、IMU 相対 pose producer、orientation filter、sensor config、symbolic URDF helper が置かれている。`ros/core/probtf_core` は ROS node と message 変換を担当し、`ros/core/probtf_msgs` は wire format を提供している。

一方で、設計上はまだ整理途中である。特に次の三点が重要である。

1. `ros/core/probtf_core` が ROS package でありながら、root の `src/probtf` と `third_party/BinghamNLL/src/bingham` を catkin へ relay install している。これは「Python package は repository root の `src` のみに置き、ROS 内部は bridge のみに使う」という設計思想に反する。
2. `src/probtf` の中に、ProbTF 基盤部分と個別推定法が混在している。基盤とは、確率的座標変換の表現、frame graph、path lookup、合成、逆変換、時刻付き buffer、依存関係管理、分布族 backend、source-aware fusion などである。個別推定法とは、IMU の遠心力・角速度から相対 pose を推定する方法、gyro prediction、gravity/magnetic evidence などである。
3. `symaware_grasp.prob_tf` には、現在の `probtf` にまだ吸い上げられていない lookup 的実装が残っている。とくに `ProbTfTree`, `PathExpression`, root-to-link moment propagation, attached point moment propagation, tangent surrogate は、将来の ProbTF 基盤に近い内容である。

したがって、次 phase では「ROS bridge の純化」と「ProbTF 基盤 / 推定 producer / example の分離」を優先すべきである。

## 1. 現在のディレクトリ構造の意味

現在の大きな構造は次のようになっている。

```text
ProbTF-demo/
  src/
    probtf/                    # ROS 非依存の共通 Python 実装
    symaware_grasp/            # symmetry-aware grasp example と旧 ProbTF prototype
    deflecomp_core/            # deflection compensation example の Python 実装
    deflecomp_sim/
    deflecomp_examples/
  ros/
    core/
      probtf_msgs/             # ProbTF の ROS message
      probtf_core/             # ROS node と probtf_ros adapter
    examples/
      probtf_imu_demo/
      probtf_orientation_demo/
      deflecomp/
      symaware_grasp/
  third_party/
    BinghamNLL/
  tests/
```

root の `src/probtf` は、ROS なしで import できるように作られている。`tests/test_ros_boundary.py` でも、root package 側に `rospy`, `tf2_ros`, `probtf_msgs` などが混入しないことを検査している。この方針は正しい。

問題は、`ros/core/probtf_core/setup.py` が root package を catkin 側へ再公開している点である。

```python
PROBTF_SOURCE = "../../../src"
BINGHAM_SOURCE = "../../../third_party/BinghamNLL/src"

setup(
    name="probtf_core",
    packages=probtf_packages + bingham_packages + ["probtf_ros"],
    package_dir={
        "probtf": PROBTF_SOURCE + "/probtf",
        "bingham": BINGHAM_SOURCE + "/bingham",
        "probtf_ros": "src/probtf_ros",
    },
)
```

この実装は catkin devel/install 空間で `probtf` を使えるようにするための実用的な暫定策である。しかし、`probtf_core` という ROS package が Python core package の配布責務も持ってしまっている。将来的には撤去対象である。

## 2. root `src/probtf` の現状

`src/probtf` は現在、次の役割をまとめて担っている。

```text
src/probtf/
  __init__.py
  geometry.py
  models.py
  bingham.py
  fusion.py
  imu_preprocessing.py
  imu_relative_pose.py
  orientation_filter.py
  sensor_config.py
  symbolic_urdf.py
```

### 2.1 `models.py`

`models.py` は ROS 非依存の domain model を定義している。

- `GaussianPosition`
- `BinghamRotation`
- `ProbabilisticTransform`
- `ImuKinematics`
- `SensorMount`

`GaussianPosition` は parent frame で表された並進 Gaussian であり、平均 `mean` と 3x3 covariance を持つ。`BinghamRotation` は quaternion Bingham parameter を `[w, x, y, z]` basis で持ち、最大固有値を 0 にする canonical gauge へ正規化する。`ProbabilisticTransform` は directed edge の summary であり、`parent_frame_id`, `child_frame_id`, `position`, `orientation`, `stamp`, `edge_id`, `source_id`, `evidence_source_ids`, `approximation_type`, `closure_approximation` を持つ。

重要な設計判断は、`ProbabilisticTransform` が「child frame の vector を parent frame へ写す rotation」と「child origin を parent frame で表した translation」を保持することである。これは通常の TF の directed transform と同じ向きに揃っている。

現状の制限は、`ProbabilisticTransform` が position と orientation の joint law を表さないことである。Gaussian position と Bingham orientation が別々に入っているため、rotation-translation coupling は保持されない。`closure_approximation` はこの問題を明示するための flag だが、現在の Python object 自体に coupling を表す slot はない。

### 2.2 `geometry.py`

`geometry.py` は quaternion と S2 tangent helper の低レベル関数を提供している。

- `quat_normalize`
- `quat_conj`
- `quat_mul`
- `quat_left_matrix`
- `quat_right_matrix`
- `quat_to_rotmat`
- `axis_angle_to_quat`
- `rpy_to_quat`
- `complete_orthonormal_basis`
- `tangent_projector`
- `tangent_basis`
- `exp_s2`

quaternion は一貫して `[w, x, y, z]` である。ROS の `geometry_msgs/Quaternion` は field 表示が `x, y, z, w` なので、変換責務は `probtf_ros.conversions` へ閉じ込める方針になっている。

この module は基盤に置いてよい。ただし、将来的には `probtf.math.geometry` や `probtf.linalg` のように、ProbTF graph/domain object から少し距離を置いた math helper として分けると見通しがよい。

### 2.3 `bingham.py`

`bingham.py` は現在の ProbTF の確率計算の中心である。

主な機能は次の通り。

- Bingham parameter の validation と canonicalization
- mode quaternion の取得
- Bingham normalizer derivative を使った second moment
- Hessian を使った fourth moment
- second moment から Bingham parameter への moment matching
- 独立 quaternion の Hamilton 積に対する second moment の厳密伝播
- quaternion second/fourth moment から rotation matrix moment への変換
- `RotationMoment` による `E[R]` と `E[R kron R]` の保持

これは明確に ProbTF 基盤である。理由は、frame 合成時に必要な回転の確率伝播、回転が並進へ作用する際の `E[R v]` と `E[R A R^T]` の計算、Bingham summary への closure などがここに含まれるためである。

ただし、現状の `bingham.py` は BinghamNLL の normalizer 実装に依存する数値実装と、ProbTF graph/query で使う moment operator が同居している。将来的には次のように分けるとよい。

```text
src/probtf/
  probability/
    bingham_distribution.py     # parameter, gauge, mode, normalizer, moments
    bingham_matching.py         # moment matching
  transforms/
    rotation_moments.py         # RotationMoment, E[R], E[R kron R]
    quaternion_moments.py       # Hamilton product moment propagation
```

現段階ではファイルを分ける必要はないが、設計上は「Bingham 分布そのもの」と「ProbTF の回転合成 backend」は別概念である。

### 2.4 `fusion.py`

`fusion.py` は source-aware evidence fusion を実装している。

主な object は次の通り。

- `TransformEvidence`
- `EvidenceProvenance`
- `FusedTransformEvidence`
- `fuse_transform_evidence`
- `fuse_evidence`

`TransformEvidence` は一つの source が出した directed transform に対する likelihood contribution である。orientation は Bingham natural parameter、position は Gaussian information matrix/vector で表される。`fuse_transform_evidence` は、同じ directed frame pair に対する独立 evidence を natural parameter の加算で融合する。重複 `source_id` は default で拒否され、二重計上を避ける。

これは ProbTF 基盤の一部である。ただし現状の fusion は「同一 edge に対する likelihood product」に限定されており、TF 的な lookup ではない。つまり、`world -> imu` と `imu -> tool` を合成して `world -> tool` の分布を返す処理はここにはない。

将来必要な発展は次の通り。

- `TransformEvidence` と `ProbabilisticTransform` の関係を明確化する。前者は likelihood、後者は posterior/summary である。
- `evidence_kind="prediction"` の意味を再定義する。現状は orientation demo で gyro prediction も `TransformEvidence` として流しているが、prediction prior と likelihood を同じ加算で扱ってよいかは graph API で明示すべきである。
- `source_id` だけでなく、共通 raw sample、共通 calibration、共通 latent edge を表す dependency metadata を持つ。
- 同じ edge 上の evidence fusion と、path 上の transform composition を別 API に分ける。

### 2.5 `imu_preprocessing.py`

`ImuKinematicsPreprocessor` は raw IMU sample から局所多項式近似で kinematic derivative を作る。

入力は stamp、angular velocity、specific force、それぞれの covariance である。sliding window に対して polynomial fit を行い、次を出す。

- angular velocity
- angular acceleration
- specific force
- angular velocity covariance
- angular acceleration covariance
- specific force covariance

これは ProbTF 基盤ではなく、IMU relative pose producer の前処理である。root `src` に置くこと自体は問題ないが、将来的には `probtf_estimators.imu` か `probtf.producers.imu` のような層へ移すべきである。

### 2.6 `imu_relative_pose.py`

`ImuRelativePoseEstimator` は Phase 1 で移植された最重要 producer である。

回転は、child IMU frame の vector を parent IMU frame の vector に align する Bingham likelihood として蓄積する。使う vector は主に angular velocity と angular acceleration である。

並進は次の剛体運動式を使う。

```text
R f_child - f_parent = ([omega]x^2 + [alpha]x) r
```

ここで `r` は child IMU origin を parent IMU frame で表した位置である。`RecursiveGaussianLeastSquares` が information-form の逐次最小二乗を担当する。

この module には、基盤と producer が混ざっている。

基盤寄りのもの:

- `skew`
- `rigid_point_acceleration_operator`
- `vector_alignment_bingham`
- `RecursiveGaussianLeastSquares`

producer 固有のもの:

- `JointGeometry`
- `ImuRelativePoseEstimator`
- angular velocity / angular acceleration / specific force から relative pose を推定する update logic

`vector_alignment_bingham` は他 module と重複している。`orientation_filter.py` の `vector_alignment_bingham_evidence`、`deflecomp_core.observation.bingham.simple_bingham_unit` と同種の実装であり、規約と scale が揃っていない。これは統合対象である。

### 2.7 `orientation_filter.py`

`orientation_filter.py` は bingham_orientation_filter 由来の内容を ProbTF 表現へ移したものである。

主な機能は次の通り。

- Gaussian gyro sample を quaternion exponential へ cubature 伝播し、`E[delta_q delta_q^T]` を得る。
- 現在姿勢の Bingham second moment と delta quaternion second moment を Hamilton product second moment で合成する。
- 合成した second moment を Bingham に moment matching する。
- gravity vector と magnetic vector を別々の Bingham evidence として作る。
- `OrientationBinghamFilter` が prediction と independent evidence fusion を一 cycle として管理する。

これは producer / estimator であり、ProbTF 基盤そのものではない。ただし、内部で使っている `delta_quaternion_second_moment`, `predict_orientation_bingham`, `vector_alignment_bingham_evidence` は、他の推定器からも再利用される可能性が高い。

分離方針としては、次のように考えるとよい。

```text
probtf.transforms.quaternion_process
  delta_quaternion_second_moment
  predict_bingham_from_gyro

probtf.producers.orientation_imu
  OrientationBinghamFilter
  gravity_bingham_evidence
  magnetic_bingham_evidence
```

ただし、gyro prediction は「座標変換を介した確率伝播」の基礎演算ではなく、IMU orientation estimator の prediction model である。そのため ProbTF core の最深部ではなく、estimator library 側に置くのが自然である。

### 2.8 `sensor_config.py`

`sensor_config.py` は robot ごとの sensor mount YAML を ROS 非依存で読む。

対応している schema は大きく二つある。

- 新しい `sensors` mapping/list schema
- 既存 deflecomp の `imu_frames` / `static_transforms` schema

出力は `SensorMount` であり、`source_id`, `frame_id`, `parent_frame_id`, `position_xyz`, `orientation_wxyz` を持つ。

これは ProbTF 基盤というより、sensor ingress support である。root `src` に置くのはよいが、将来的には `probtf.io.sensor_config` や `probtf.sensors.config` へ切り出すとよい。

なお、`deflecomp_core.observation.imu_frame_config` に似た parser が残っている。こちらは `R_model_imu` や deflecomp robot frame resolution を持つため完全同一ではないが、低レベルの YAML alias、rpy/quaternion conversion、legacy compact syntax の処理は重複している。

### 2.9 `symbolic_urdf.py`

`symbolic_urdf.py` は symbolic URDF の placeholder を扱う。

従来形式の `#|name|#` placeholder を XML comment 外で検出し、substitution を適用して materialized URDF を作る。DOCTYPE/ENTITY を拒否し、well-formed XML を検査する。

これは ROS 非依存であり、root `src` に置くのは正しい。ただし ProbTF 基盤の中心ではない。位置づけとしては「ProbTF producer の結果を deterministic URDF へ落とす terminal consumer の helper」である。

将来的には symbolic URDF は ProbTF core からさらに外し、`probtf_symbolic_urdf` または example support package としてもよい。理由は、materialization は分布を失う不可逆操作であり、ProbTF の lookup/fusion の中心処理ではないからである。

## 3. ROS 側の現状

### 3.1 `ros/core/probtf_msgs`

`probtf_msgs` は reusable message package である。現在の message は次の通り。

- `BinghamDistribution`
- `GaussianPosition`
- `ProbabilisticTF`
- `ProbabilisticTFArray`
- `TransformEvidence`
- `ImuKinematics`
- `GraspCandidate`
- `IKResult`

Phase 1 で追加された重要な field は次である。

- `ProbabilisticTF.edge_id`
- `ProbabilisticTF.source_id`
- `ProbabilisticTF.evidence_source_ids`
- `ProbabilisticTF.has_position`
- `ProbabilisticTF.has_orientation`
- `ProbabilisticTF.closure_approximation`
- `TransformEvidence.evidence_kind`
- `TransformEvidence.source_id`
- `TransformEvidence.has_sequence`
- `TransformEvidence.has_position`
- `TransformEvidence.has_orientation`
- `ImuKinematics.angular_acceleration`

message layer は ROS bridge として妥当である。ただし、将来的には `ProbabilisticTF` が「posterior summary」なのか「edge distribution」なのかを明確にする必要がある。現在は `orientation_filter_node.py` が posterior を `ProbabilisticTF` として publish し、`probtf_fusion_node.py` も fused result を `ProbabilisticTF` として publish する。この使い方自体は自然だが、graph に再投入する場合は `closure_approximation` と dependency をどう扱うかが未定である。

### 3.2 `ros/core/probtf_core`

`probtf_core` は package description では「ROS adapters for the ROS-independent ProbTF Python core」と書かれている。実際に node と `probtf_ros` は bridge として働いている。

含まれる node は次の通り。

- `imu_kinematics_node.py`
- `imu_relative_pose_node.py`
- `orientation_filter_node.py`
- `probtf_fusion_node.py`
- `sensor_mount_tf_node.py`
- `symbolic_urdf_materializer_node.py`

`src/probtf_ros/conversions.py` は次を提供する。

- `imu_kinematics_from_msg`
- `probabilistic_transform_to_msg`
- `transform_evidence_from_msg`
- `transform_evidence_to_msg`

この `probtf_ros` は ROS bridge として妥当である。問題は package 名と setup 責務である。`probtf_core` という名前は「ProbTF の中核」を示すが、実体は ROS node と adapter である。また、`setup.py` が root `src/probtf` と `third_party/bingham` を catkin package として配布しているため、ROS package が Python core の所有者に見える。

次 phase では、少なくとも設計上は次へ寄せるべきである。

```text
ros/core/
  probtf_msgs/       # message only
  probtf_ros/        # ROS bridge only。旧 probtf_core の rename 候補

src/
  probtf/            # Python core と producer libraries
```

catkin devel/install で `probtf` をどう見せるかは別途考える必要がある。候補は次の通り。

1. 開発時は root package を `pip install -e /home/leus/catkin_ws/src/ProbTF-demo` する。
2. catkin 側に environment hook を持たせ、root `src` を `PYTHONPATH` に加える。ただし ROS package が Python package を所有しないよう、責務は明確化する。
3. root に独立した Python package install 手順を置き、ROS bridge package はそれに runtime dependency を持つだけにする。

現状の relay setup は実用上便利だが、長期設計としては撤去すべきである。

### 3.3 `imu_kinematics_node.py`

raw `sensor_msgs/Imu` を受け取り、`ImuKinematicsPreprocessor` を通して `probtf_msgs/ImuKinematics` を publish する。

ROS node 側にある logic は主に次である。

- frame_id の決定
- default covariance fallback
- sample rejection の warning
- ROS message への詰め替え

概ね bridge として妥当である。ただし `_covariance` fallback は `orientation_filter_node.py` にも似た実装があり、`probtf_ros` の helper に寄せてもよい。

### 3.4 `imu_relative_pose_node.py`

親子の `ImuKinematics` topic を `message_filters.ApproximateTimeSynchronizer` で同期し、`ImuRelativePoseEstimator` に渡して `ProbabilisticTF` を publish する。

この node はほぼ bridge である。推定の本体は root `src/probtf/imu_relative_pose.py` にあるため、方向性はよい。

### 3.5 `orientation_filter_node.py`

`sensor_msgs/Imu` と `sensor_msgs/MagneticField` を受け、orientation prediction と gravity/magnetic evidence を作る。

publish する topic は次である。

- `~prediction`: `TransformEvidence`
- `~gravity_evidence`: `TransformEvidence`
- `~magnetic_evidence`: `TransformEvidence`
- `~posterior`: `ProbabilisticTF`

この node は `OrientationBinghamFilter` を使っているが、ROS 側にもそれなりに判断 logic がある。

- child frame の決定
- dt gap handling
- magnetic age gate
- covariance fallback
- posterior message の構築

bridge として許容範囲だが、gating policy や posterior construction は将来的には root producer 側へ寄せられる。ROS node は「parameter を読み、message を domain object へ変換し、domain service を呼ぶ」程度にするのがよい。

### 3.6 `probtf_fusion_node.py`

複数の `TransformEvidence` topic の latest value を保持し、stamp skew を見て `fuse_transform_evidence` へ渡す。出力は `ProbabilisticTF` である。

現状の fusion node は「同じ directed frame pair に対する latest evidence の同期融合」であり、ProbTF graph lookup ではない。名前は `probtf_fusion` で妥当だが、将来の `lookup` と混同しないように API 名を分けるべきである。

現状の制限は次の通り。

- 全 topic が揃うまで出力しない。
- latest value 方式であり、時系列 buffer ではない。
- evidence dependency は `source_id` 重複しか見ない。
- `evidence_kind` が `prediction` でも natural parameter addition で扱う。
- position information が singular な場合は `ProbabilisticTF.has_position=False` に落とす。

### 3.7 `sensor_mount_tf_node.py`

`sensor_config.py` で YAML を読み、static TF を publish する。これは ROS bridge として妥当である。

ただし、ProbTF の設計思想から見ると、sensor mount も本来は deterministic transform の特殊例として ProbTF graph に入れられるべきである。現状は tf2 static transform として publish するのみで、ProbTF graph には登録されない。

### 3.8 `symbolic_urdf_materializer_node.py`

`ProbabilisticTF` の収束判定を行い、symbolic URDF placeholder を deterministic value で置き換え、`/robot_description` と optional file へ書く。

この node には bridge 以上の application logic が入っている。

- position covariance threshold
- orientation eigenvalue gap threshold
- placeholder binding
- materialization の完了判定
- output parameter/file の更新

将来的には、収束判定と materialization policy を root 側の consumer/service として切り出し、ROS node は parameter と topic の bridge のみにすべきである。

## 4. examples の現状

### 4.1 `ros/examples/probtf_imu_demo`

two-IMU relative pose の demo である。launch は次を起動する。

- `sensor_mount_tf_node.py`
- parent 用 `imu_kinematics_node.py`
- child 用 `imu_kinematics_node.py`
- `imu_relative_pose_node.py`
- optional `symbolic_urdf_materializer_node.py`
- optional `robot_state_publisher`

この demo は、raw IMU から `ProbabilisticTF` を生成し、必要なら symbolic URDF を実体化する流れを示している。Phase 1 の主目的に沿っている。

### 4.2 `ros/examples/probtf_orientation_demo`

orientation filter の demo である。`orientation_filter_node.py` が gyro prediction、gravity evidence、magnetic evidence を別 topic に分けて publish し、`probtf_fusion_node.py` が common fusion node として統合する。

この demo は「producer が source-separated evidence を出し、共通 fusion が合成する」という設計を示している。ただし、現在の fusion は同一 edge だけであり、TF lookup 的な chain composition はまだない。

### 4.3 `ros/examples/deflecomp`

deflecomp は ProbTF の応用先として残っている。`deflecomp_core` は root `src` にあり、ROS package は `ros/examples/deflecomp` に移されている。

`deflecomp_core.observation` には ProbTF と重複・近接する実装がある。

- `BinghamUtils.simple_bingham_unit`
- `FrameImuObservation`
- `ImuObservationBuilder`
- `ImuBuffer`
- `ImuFrameConfig` parser

これらは現在 deflecomp の stiffness estimation pipeline に密接に結び付いている。すぐに移す必要はないが、ProbTF core の sensor/evidence 体系が固まった後に、共通化できる部分を吸い上げるべきである。

### 4.4 `ros/examples/symaware_grasp` と `src/symaware_grasp/prob_tf`

`symaware_grasp.prob_tf` は旧 ProbTF prototype として重要である。Bingham moment と geometry の一部は既に `probtf` への compatibility export になっている。

一方で、次の実装はまだ `symaware_grasp` 側に残っている。

- `ProbTfTree`
- `ProbTfEdge`
- `ProbTfResult`
- `PathExpression`
- `EdgeView`
- root-to-link moment propagation
- attached point moment propagation
- tangent surrogate
- ProbTF YAML loader
- visualization helpers

このうち `ProbTfTree` と `PathExpression` は、ユーザが言及している「TF でいう lookup 的なやつ」の最も近い既存実装である。ただし現状は次の制限がある。

- root-to-target の forward path summary が中心である。
- 非 root source からの full moment summary は未実装である。
- inverse path は expression としては扱えるが、moment summary は forward edge 前提である。
- repeated edge を含む path は dependency-aware propagation が必要なため未対応である。
- `ProbabilisticTransform` や `TransformEvidence` とはまだ統合されていない。

したがって、将来の ProbTF graph API はこの実装を参考にしつつ、`src/probtf` 側へ再設計して移すのがよい。

## 5. 現状の ProbTF 基盤として足りないもの

### 5.1 TF 的 lookup API

現状の `src/probtf` には、`lookup(parent, child, time)` に相当する API がない。`fusion.py` は同一 directed edge の evidence を融合するだけで、frame graph 上の path composition は扱わない。

将来必要な最小 API は次のような形である。

```python
graph = ProbTfGraph()
graph.set_edge_distribution(edge)
graph.add_edge_evidence(evidence)

result = graph.lookup("world", "tool0", stamp=t, representation="moment")
point = graph.lookup_point("world", "tool0", [0.0, 0.0, 0.1], stamp=t)
```

このとき `lookup` が保証すべきことは次である。

- path 上の physical edge を同定する。
- inverse view は別 edge として扱わず、同じ latent transform の関数として扱う。
- repeated edge が出る path では二重計上を避ける。
- edge 間が独立でない場合は dependency metadata または joint law を参照する。
- 必要に応じて moment closure を使い、`closure_approximation` を明示する。
- 返す representation が posterior summary なのか exact expression なのかを区別する。

### 5.2 Frame graph と time buffer

現状は `ProbabilisticTransform.stamp` や `TransformEvidence.timestamp` は存在するが、時刻付き buffer はない。`probtf_fusion_node.py` は latest evidence のみを保持する。

TF 的な使い方をするには、次が必要である。

- edge ごとの時系列 buffer
- interpolation / nearest / exact policy
- max age と extrapolation policy
- asynchronous producer の扱い
- latest query と timestamp query の明確な区別
- stale evidence の invalidation

`deflecomp_core.observation.ImuBuffer` は timestamped unit-vector buffer を持っているが、ProbTF graph buffer ではない。発想は近いので、共通 buffer 実装を作る際の参考にはなる。

### 5.3 確率則付き frame composition

現状の Bingham moment 実装は、回転合成に必要な材料を持っている。

- quaternion product second moment
- Bingham second/fourth moment
- `E[R]`
- `E[R kron R]`

しかし、`src/probtf` の main API として「edge chain を合成して transform distribution を返す」実装はまだない。`symaware_grasp.prob_tf.tree` には root-to-target の moment propagation があるが、`ProbabilisticTransform` とは接続されていない。

将来は、少なくとも次を core に入れるべきである。

- deterministic edge の特殊扱い
- Gaussian/Bingham edge の moment propagation
- translation mean/cov propagation
- attached point propagation
- inverse edge view
- summary Bingham への moment matching
- approximation provenance

### 5.4 Rotation-translation coupling

現在の `ProbabilisticTransform` は position と orientation を別々に持つ。これでは「上流回転の不確かさが下流位置の不確かさになる」ことは chain query の中でしか表せない。

将来的には、内部表現として次のいずれかが必要である。

- full joint distribution over `SE(3)`
- moment representation: `E[R]`, `E[R kron R]`, `E[t]`, `E[t t^T]`, `E[vec(R) t^T]`
- tangent Gaussian backend
- mixture backend
- factor/joint law backend

外部 message としては summary を返してもよいが、内部 graph は summary だけを真実にしてはいけない。

### 5.5 Dependency-aware fusion

現状は `source_id` の重複拒否で二重計上を少し防いでいる。しかし、実際には次のような依存関係がある。

- 同じ IMU sample から作られた gyro prediction と gravity evidence
- 同じ calibration parameter を共有する複数 sensor
- 同じ physical edge の prior と posterior summary
- chain query の途中で同じ edge を順逆に通る path
- materialized URDF から再 publish された deterministic TF

`source_id` だけではこれらを表せない。`dependency_group_id`, `latent_edge_id`, `sample_id`, `calibration_id`, `derived_from` のような provenance が必要になる。

## 6. 重複・統合候補

### 6.1 Vector alignment Bingham

同種の Bingham likelihood 実装が複数ある。

- `probtf.imu_relative_pose.vector_alignment_bingham`
- `probtf.orientation_filter.vector_alignment_bingham_evidence`
- `deflecomp_core.observation.bingham.simple_bingham_unit`

これらは、vector before/after の alignment を quaternion Bingham parameter に変換する点で同じである。一方で、目的、scale、式の書き方が少し異なる。

提案:

- `probtf.probability.orientation_likelihoods` のような module を作る。
- API 名を `bingham_vector_alignment_likelihood` に統一する。
- quaternion basis `[w,x,y,z]`、sign convention、residual variance / concentration の解釈を文書化する。
- deflecomp はこの common helper を呼ぶ。

### 6.2 Quaternion / rotation conversion

`probtf.geometry` と `deflecomp_core.observation.imu_frame_config` に重複がある。

- rpy to quaternion / matrix
- quaternion to matrix
- matrix to quaternion
- quaternion order conversion

提案:

- pure math は `probtf.geometry` へ寄せる。
- deflecomp 側は `xyzw` message/YAML convention の変換だけを持つ。
- ROS message conversion は `probtf_ros.conversions` へ閉じ込める。

### 6.3 Sensor mount config parser

`probtf.sensor_config` と `deflecomp_core.observation.imu_frame_config` は目的が近い。

相違点:

- `probtf.sensor_config` は `SensorMount` を返し、source/frame/static transform 管理に寄っている。
- `deflecomp_core.observation.imu_frame_config` は `ImuFrameConfig` を返し、robot model frame resolution と `R_model_imu` を持つ。

提案:

- YAML alias の解釈、rpy/quaternion parsing、legacy compact syntax を `probtf.sensors.config` へ共通化する。
- deflecomp 固有の `robot.has_frame`, `suggest_imu_frames`, `R_model_imu` は deflecomp 側に残す。

### 6.4 Bingham gauge convention

`probtf.bingham` は最大固有値 0 gauge を canonical とする。一方、`symaware_grasp.prob_tf.bingham_moments.ensure_trace_zero` は trace zero gauge を返す。

Bingham は identity shift に対して同じ分布を表すため、どちらも数学的には等価である。しかし、実装内で gauge が混ざると debug が難しくなる。

提案:

- storage/transport は最大固有値 0 gauge に統一する。
- trace zero は旧 compatibility または local numerical helper に限定する。
- message comment と doc に gauge convention を明記する。

### 6.5 Graph / lookup prototype

`symaware_grasp.prob_tf.tree` の `ProbTfTree` は現在の root `src/probtf` にはない機能を持つ。

移すべき要素:

- `PathExpression`
- `EdgeView`
- physical edge ID と inverse view
- path reduction
- repeated edge の検出
- root-to-target moment propagation
- attached point propagation
- tangent surrogate

そのまま移すべきでない要素:

- symaware demo の YAML schema に密着した `ProbTfEdge`
- root-to-target forward path だけの制限
- summary を independent edge と誤解しやすい API

提案:

- `probtf.graph` に `ProbTfGraph`, `ProbTfEdgeRef`, `PathExpression` を新設する。
- `symaware_grasp.prob_tf.tree` は compatibility wrapper にする。
- `ProbabilisticTransform` / `TransformEvidence` と接続する。

### 6.6 ROS node 内 helper

`imu_kinematics_node.py` と `orientation_filter_node.py` に covariance fallback helper が重複している。`probtf_ros.conversions` に vector assignment helper もあるが、node 側で直接 message field を詰める箇所が残っている。

提案:

- `probtf_ros.conversions` に covariance extraction/fallback、Vector3 変換、Header time helper を集約する。
- node は domain object を作って conversion function を呼ぶだけにする。

### 6.7 Symbolic URDF materialization policy

`symbolic_urdf.py` は pure parser/materializer でよい。一方、`symbolic_urdf_materializer_node.py` は収束閾値や output lifecycle を持つ。

提案:

- `probtf.consumers.symbolic_urdf_materializer` のような root module に、threshold policy と binding update を移す。
- ROS node は topic と parameter の bridge にする。
- materialized URDF がどの ProbTF edge summary から作られたかを metadata として残す。
- deterministic TF として再投入する場合は、元の stochastic edge との dependency を明示する。

## 7. `probtf_core` の設計上の問題と整理案

ユーザ指摘の通り、現在の `probtf_core` は設計思想に反している。正確には、二つの意味の `core` が混ざっている。

1. ProbTF の数学的・確率的 core
2. ROS workspace 内の core package

現在の `ros/core/probtf_core` は 2 であるべきだが、`setup.py` により 1 の Python package 配布も兼ねている。

望ましい方向は次である。

```text
ProbTF-demo/
  src/
    probtf/
      core/                 # ProbTF graph, distribution, lookup semantics
      probability/          # Bingham, Gaussian, moment matching
      transforms/           # SE(3), SO(3), composition, inverse, moments
      fusion/               # evidence model, natural parameter fusion
      producers/            # estimator libraries
      io/                   # sensor config, symbolic URDF helper
  ros/
    core/
      probtf_msgs/
      probtf_ros/           # bridge only。旧 probtf_core の rename 候補
```

ただし、`src/probtf/core` という Python subpackage 名を作るかどうかは慎重に決めるべきである。`probtf_core` という top-level Python package を作ると、ROS package 名と再び混ざる。top-level は `probtf` に統一し、その中に `probtf.graph`, `probtf.transforms`, `probtf.probability` などを置く方がよい。

移行手順案:

1. `ros/core/probtf_core` を `ros/core/probtf_ros` へ rename する設計を決める。実作業は別 commit でよい。
2. `ros/core/probtf_core/setup.py` から `probtf` と `bingham` の relay install を撤去する方針を決める。
3. root package の install 手順を README と dev tooling に明記する。
4. catkin build で ROS node が root `src/probtf` を見つける仕組みを、bridge package の所有権に見えない形で用意する。
5. `probtf_ros` は `probtf_msgs` と `probtf` を import するだけにする。

## 8. ProbTF 基盤と推定 producer の分離案

現状の `src/probtf` を、概念的には次の三層に分けるとよい。

### 8.1 ProbTF foundation

ここに置くべきもの:

- frame graph
- physical edge と inverse view
- path lookup
- time buffer
- transform distribution interface
- deterministic transform as degenerate distribution
- Gaussian/Bingham/moment/mixture backend interface
- transform composition
- inverse transform
- attached point / direction action
- source/evidence provenance
- dependency-aware fusion
- approximation metadata

現在の該当候補:

- `models.GaussianPosition`
- `models.BinghamRotation`
- `models.ProbabilisticTransform`
- `fusion.TransformEvidence`
- `fusion.FusedTransformEvidence`
- `bingham.RotationMoment`
- `bingham.quaternion_product_second_moment`
- `bingham.rotation_first_moment`
- `bingham.rotation_kronecker_moment`
- `symaware_grasp.prob_tf.PathExpression`
- `symaware_grasp.prob_tf.ProbTfTree` の一部

### 8.2 Probability / math backend

ここに置くべきもの:

- Bingham distribution helper
- moment calculation
- moment matching
- quaternion algebra
- SO(3)/S2 geometry
- Gaussian information form utilities
- numerical validation utilities

現在の該当候補:

- `geometry.py`
- `bingham.py`
- `fusion.py` の Gaussian information validation 部分
- `symaware_grasp.prob_tf.tangent_surrogate`

### 8.3 Producers / estimators

ここに置くべきもの:

- IMU relative pose estimator
- IMU kinematics preprocessor
- gyro orientation prediction filter
- gravity/magnetic evidence producer
- deflecomp stiffness observation adapters
- future camera/tag/marker producers

現在の該当候補:

- `imu_preprocessing.py`
- `imu_relative_pose.py`
- `orientation_filter.py`
- `deflecomp_core.observation.imu_observation`
- `deflecomp_core.observation.bingham`

この層は ProbTF を生成・更新する方法であり、ProbTF の定義そのものではない。将来的には `probtf.producers.imu_relative_pose` のような名前にして、core lookup と分けるとよい。

### 8.4 IO / consumers

ここに置くべきもの:

- sensor mount YAML loader
- symbolic URDF parser/materializer
- deterministic TF export
- visualization export

現在の該当候補:

- `sensor_config.py`
- `symbolic_urdf.py`
- `symaware_grasp.prob_tf.visualize`
- `symbolic_urdf_materializer_node.py` の root 化候補

この層は ProbTF を使うための周辺機能であり、foundation と混ぜすぎない方がよい。

## 9. 提案する次 phase の優先順位

### Priority A: core graph / lookup の設計を先に決める

最初に決めるべきは、`ProbTfGraph.lookup` の semantics である。ここが決まらないと、message、fusion、producer、URDF materialization の関係が定まらない。

最小実装案:

- `ProbTfGraph`
- `ProbTfEdgeRecord`
- `PathExpression`
- `lookup_path(source, target)`
- `lookup(source, target, stamp=None, representation="moment")`
- `lookup_point(source, target, point, stamp=None)`

初期制限として、independent edge の tree graph だけ対応してもよい。ただし API 上は dependency-aware extension を阻害しないようにする。

### Priority B: `symaware_grasp.prob_tf.tree` を吸い上げる

`symaware_grasp.prob_tf.tree` は既に lookup prototype と moment propagation を持つ。これを捨てるのではなく、`probtf.graph` の設計材料として吸い上げる。

作業案:

1. `PathExpression` と `EdgeView` を root `src/probtf` へ移す。
2. `ProbTfTree.lookup_path` を `ProbTfGraph.lookup_path` として再実装する。
3. root-to-target の moment propagation を `MomentComposer` のような内部 class へ分ける。
4. `symaware_grasp.prob_tf.tree` は wrapper にする。

### Priority C: ROS bridge package の責務を純化する

`probtf_core` rename と setup relay 撤去は、設計上は早めに決めるべきである。ただし作業としては catkin build への影響があるため、graph 設計と並行して段階的に行う。

目標:

- ROS package は `probtf_ros` として bridge のみを持つ。
- `probtf_msgs` は message のみを持つ。
- Python core は root `src/probtf` のみが所有する。
- `probtf_ros` は root package に依存するが、所有しない。

### Priority D: producer modules を分離する

`imu_relative_pose.py`, `imu_preprocessing.py`, `orientation_filter.py` は今後も重要だが、ProbTF foundation とは切り分ける。

候補:

```text
src/probtf/
  producers/
    imu_kinematics.py
    imu_relative_pose.py
    orientation_imu.py
```

この段階で、`vector_alignment_bingham` と covariance fallback などを共通化する。

### Priority E: dependency metadata を導入する

`source_id` だけでは不十分である。少なくとも次の field/API を検討する。

- `source_id`
- `sample_id`
- `dependency_group_id`
- `latent_edge_id`
- `derived_from`
- `calibration_id`
- `evidence_kind`
- `closure_approximation`

ROS message へすぐ追加する前に、Python domain model で semantics を決めるとよい。

## 10. 現状で良い点

現状の実装には、次 phase に活かせる良い判断が既にある。

- root `src/probtf` は ROS import を含まない。
- quaternion basis が `[w,x,y,z]` に統一されつつある。
- Bingham posterior は最大固有値 0 gauge へ寄せられている。
- `TransformEvidence` は source ID と provenance を持ち、重複 source を拒否する。
- `ProbabilisticTF` は partial distribution を `has_position` / `has_orientation` で明示する。
- `closure_approximation` が導入され、summary 再利用の危険を表現できる。
- orientation demo は prediction / gravity / magnetic evidence を分けており、source-separated design の例になっている。
- symaware prototype には lookup と moment propagation の実験成果が残っている。

これらは残すべきである。

## 11. 現状で危険な点

次の点は、放置すると設計が混乱する。

- `probtf_core` という ROS package 名が、ProbTF core Python package と誤解される。
- `probtf_core/setup.py` が root `probtf` を catkin 側から配布している。
- `fusion.py` の同一 edge evidence fusion と、TF 的 path lookup がまだ分かれていない。
- `orientation_filter_node.py` の `prediction` evidence が likelihood と同じ pipeline に流れている。
- `ProbabilisticTransform` が rotation-translation coupling を表さないまま、graph edge として再利用されうる。
- symbolic URDF materialization により stochastic edge が deterministic URDF/TF へ落ちるが、由来 metadata が残らない。
- `symaware_grasp.prob_tf.tree` と `src/probtf` が二つの ProbTF 実装に見える。
- deflecomp の Bingham/vector alignment と sensor config が root `probtf` と重複している。

## 12. 推奨する長期構成

長期的には、次の構成が最も設計思想に合う。

```text
ProbTF-demo/
  src/
    probtf/
      graph/
        buffer.py
        edge.py
        path.py
        lookup.py
      transforms/
        se3.py
        moments.py
        composition.py
      probability/
        bingham.py
        gaussian.py
        information.py
        mixtures.py
      evidence/
        model.py
        fusion.py
        provenance.py
      producers/
        imu_preprocessing.py
        imu_relative_pose.py
        orientation_imu.py
      io/
        sensor_config.py
        symbolic_urdf.py
  ros/
    core/
      probtf_msgs/
      probtf_ros/
    examples/
      probtf_imu_demo/
      probtf_orientation_demo/
      deflecomp/
      symaware_grasp/
```

この構成では、ROS は message と bridge のみに限定される。推定法は root package 内にあるが、foundation とは別 subpackage にする。examples は examples として残し、旧 prototype は compatibility wrapper に落としていく。

## 13. 直近の実装タスク案

次に手を動かすなら、commit を細かく分けて次の順がよい。

1. `docs`: この文書を基に次 phase の設計方針を確定する。
2. `probtf.graph`: `PathExpression` と lookup path だけを root `src/probtf` へ導入する。まだ確率合成はしない。
3. `probtf.graph`: deterministic / Gaussian-Bingham edge の root-to-target moment composition を入れる。
4. `symaware_grasp`: 旧 `ProbTfTree` を新 graph API の wrapper にする。
5. `probtf.producers`: IMU relative pose と orientation filter を producer namespace に移す。
6. `probtf.probability`: vector alignment Bingham likelihood を統一する。
7. `ros`: `probtf_core` を bridge-only にする設計を反映する。rename は影響が大きいため別 stage にする。

重要なのは、`probtf_core` の rename や package relay 撤去より先に、root `src/probtf` に「本当の core」と呼べる graph/lookup API を作ることである。そうすると ROS package 側を `probtf_ros` と呼ぶ理由が自然になる。

## 14. 結論

現状の実装は、Phase 1 としては妥当である。IMU relative pose、source-aware evidence fusion、orientation prediction、symbolic URDF、ROS messages/nodes が一つの repository に集約され、共通の `ProbabilisticTransform` / `TransformEvidence` に載り始めている。

しかし、ProbTF の本体である「確率的 frame graph と lookup」はまだ root `src/probtf` に存在しない。最も近い実装は `symaware_grasp.prob_tf.tree` に残っている。また、ROS package `probtf_core` が Python package relay も兼ねているため、設計上の境界が曖昧である。

次 phase の中心課題は、次の二つである。

1. root `src/probtf` に ProbTF foundation、つまり graph、path、lookup、composition、dependency-aware fusion を作る。
2. ROS 内部を bridge に限定し、推定 producer と Python package 配布責務を root `src` 側へ戻す。

この分離ができれば、遠心力等による推定法、orientation filter、deflecomp、symaware grasp は ProbTF の上に乗る producer / consumer / example として整理できる。
