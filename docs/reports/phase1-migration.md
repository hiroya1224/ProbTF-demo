# Phase 1 移行記録

## 目的と境界

Phase 1 では、`urdf_estimation_with_imus`、`stochastic_joint_estim`、
`bingham_orientation_filter` に分散していた機能を ProbTF の共通表現へ
移し、`ProbTF-demo` 内へ集約した。最終的な graph/query API の確定は次
phase に残し、本 phase では既存デモが要求する producer、分布演算、
sensor ingress、ROS transport を動作する単位として整理した。

再利用可能な計算は `src/probtf` に置き、ROS package は message 変換、
topic 同期、parameter、launch、TF broadcast のみを担当する。root Python
package に ROS import が入らないことは `tests/test_ros_boundary.py` で検査する。

## 段階別コミット

| commit | 内容 |
| --- | --- |
| `151076f` | 3 repository の Phase 0 分析 |
| `b0cfc5a` | ROS package を `core` / `examples` へ再配置 |
| `396caa3` | 再配置後の test asset path 修正 |
| `5203008` | ProbTF 分布、Bingham moment、evidence fusion の共通 core |
| `6a2e096` | two-IMU 相対 pose producer と symbolic URDF consumer |
| `1752f3a` | source-aware sensor ingress と likelihood fusion |
| `2d163f1` | quaternion 確率運動方程式と orientation producer |

## 優先度 1: IMU 相対 pose

### 移植した機能

- `ImuKinematicsPreprocessor` は raw IMU の局所多項式近似から角速度、
  角加速度、比力と各 covariance を生成する。
- `ImuRelativePoseEstimator` は child 系の vector を parent 系へ写す回転を
  quaternion Bingham likelihood として蓄積する。
- 並進は次の剛体運動式を information-form recursive least squares で解く。

  `R f_child - f_parent = ([omega]x^2 + [alpha]x) r`

- 出力は Gaussian position と Bingham orientation を持つ
  `ProbabilisticTransform` であり、quaternion second moment と fourth moment
  も保持する。
- `SymbolicUrdfTemplate` は従来の `#|name|#` comment-compatible placeholder
  を維持し、収束後の実体化を ROS consumer へ分離した。
- `probtf_imu_demo` は raw `sensor_msgs/Imu` から
  `probtf_msgs/ProbabilisticTF` までの経路を提供する。

旧 package の pybind11 covariance helper は取り込まず、既存の純 Python
Bingham normalizer/moment 実装へ統一した。旧実装にあった joint 登録判定の
逆転や quaternion 積の typo も引き継いでいない。

### 現時点の近似

- rotation と translation の cross-covariance は出力しない。
- 未知並進の更新では Bingham mode rotation を plug-in する。orientation が
  1 軸観測だけで未観測自由度を持つ間は position update を開始しない。
- fourth moment は chain propagation に利用できる形で保持するが、相対 pose
  estimator 自身は全 joint law を保持しない。
- symbolic URDF 実体化は分布情報を失う terminal operation である。

## 優先度 2: sensor ingress と fusion

旧 `stochastic_joint_estim` の particle filter は、実装欠落と既存
deflecomp との重複があるため移植していない。抽出したのは次の要素である。

- `TransformEvidence` は source ID、frame pair、時刻、sequence、
  `prediction` / `likelihood` の種別を持つ。
- orientation evidence は Bingham natural parameter、position evidence は
  Gaussian information matrix/vector として入力する。
- 独立 evidence は natural parameter の加算で融合する。source ID の重複は
  default で拒否し、暗黙の二重計上を防ぐ。
- `sensor_config` は robot ごとの YAML から IMU、camera、marker 等の取付
  frame と固定 transform を読む。既存 deflecomp `imu_frames.yaml` と新しい
  `sensors` schema の両方を扱う。
- `probtf_fusion_node.py` と `sensor_mount_tf_node.py` が ROS ingress と static
  TF broadcast を担当する。

現在は source ID の異なる evidence を独立と仮定する。共通 calibration、
同一 raw sample、chain 上の同じ latent edge に由来する相関を表す dependency
metadata と joint law は次 phase の課題である。

## 優先度 3: orientation prediction

- gyro の Gaussian 角速度を quaternion exponential へ cubature 伝播し、
  `E[delta_q delta_q^T]` を計算する。
- 独立な現在姿勢と delta quaternion の second moment を Hamilton 積で厳密に
  合成し、その後に Bingham moment matching を行う。
- gravity と magnetic field は別々の vector-alignment Bingham likelihood と
  して生成する。
- ROS node は `prediction`、`gravity_evidence`、`magnetic_evidence` を別 topic
  へ publish し、共通 fusion node が合成できる。直接利用向け posterior も
  orientation-only `ProbabilisticTF` として publish する。
- 旧 `symaware_grasp.prob_tf` の Bingham moment 実装は `probtf.bingham` の
  compatibility export に置き換え、共通実装を一つにした。

gyro prediction は second-moment closure のたびに Bingham へ戻すため、出力の
`closure_approximation` は true である。長時間 chain での近似誤差評価は次
phase で行う。

## Wire 規約

- core quaternion と Bingham matrix の基底順は `[w, x, y, z]`。
- `geometry_msgs/Quaternion` の field 順との差は `probtf_ros` だけで変換する。
- Bingham posterior は最大固有値を 0 とする gauge を canonical form とする。
- translation は child origin を parent frame で表したもの。
- `header.frame_id` は `parent_frame_id` と一致させる。
- partial distribution は `has_position` / `has_orientation` で明示する。
- summary を graph edge として再投入する場合は `closure_approximation` を明示する。

## 検証

- Python: `120 passed` (`pytest -q`)
- catkin: `probtf_msgs`、`probtf_core`、`probtf_imu_demo`、
  `probtf_orientation_demo` の targeted build 成功
- devel space: `probtf`、`probtf_ros`、生成 message の import 成功
- install-space 相当: `probtf_core/setup.py` から `probtf`、`bingham`、
  `probtf_ros` を `/tmp` へ配置し import 成功
- launch: two-IMU demo と orientation demo の node 解決成功

数値 test は Bingham gauge/antipodal symmetry、second/fourth moment、quaternion
積、zero/nonzero gyro prediction、synthetic centripetal relative-position recovery、
evidence 重複拒否、YAML compatibility、symbolic URDF 実体化、ROS field 変換を含む。

## 次 phase へ残す事項

- graph buffer、時刻付き lookup、inverse view、dependency-aware path composition
- rotation-translation coupling と edge 間相関を含む joint representation
- Bingham/mixture/tangent Gaussian backend の query semantics
- `ProbabilisticTF` schema 変更に対する旧 topic/message bridge
- 実体化 URDF の lifecycle、version、rollback、分布への参照
- IMU 外乱、非剛体運動、磁気異常に対する gating/outlier model
- moment closure の Monte Carlo / 実機 data による誤差評価
