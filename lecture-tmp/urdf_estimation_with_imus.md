# urdf_estimation_with_imus の整理

## 現状の実装サマリ

`urdf_estimation_with_imus` は、複数 IMU の時系列から IMU 間の相対 pose、ひいては URDF 内の未確定な joint/link 配置を推定する ROS + Python パッケージである。README では「Estimation module for URDF with multiple IMUs」と説明されており、ROS message、launch、symbolic URDF、Python 推定ライブラリ、可視化、Docker 環境を一つのリポジトリに含んでいる。

入力側では `scripts/imu_preprocessor.py` が `/imu` を受け、IMU ごとの時刻列を多項式で局所近似し、角速度、角加速度、角加加速度、加速度、加速度微分とそれぞれの共分散を `ImuDataFilteredList` として同期・補間して出す。`scripts/imu_calibrator.py` と `pypkg/src/imu_relpose_estim/preprocess/sensor_calibrator.py` は、加速度の楕円体フィットによるスケール・バイアス補正や共分散推定を担当する。

推定の中心は `pypkg/src/imu_relpose_estim/estimator/extended_leastsq.py` の `EstimateImuRelativePoseExtendedLeastSquare` である。二つの IMU の観測から相対回転を Bingham 分布の 4x4 パラメータ `Amat` として更新し、相対位置を Gaussian の平均・共分散として更新する。回転は gyro ベースの Bingham 更新と force/acceleration ベースの Bingham 更新を持ち、位置は逐次最小二乗で推定される。

ROS node としては `scripts/extleastsq_estim_imu_relpose.py` が `--this` と `--child` で指定された IMU ペアを推定し、`/estimated_relative_pose/<this>__to__<child>` に `PoseWithCovAndBingham` を publish する。この message は `geometry_msgs/Pose`、`float64[9] position_covariance`、`float64[16] rotation_bingham_parameter` を持つため、現状でも Prob-TF の最小表現にかなり近い。

`symbolic_models/*.symburdf` と `scripts/robot_description_setter.py` は、`#|joint_name_xyz|#` や `#|joint_name_rpy|#` のような未定値を含む symbolic URDF を、推定が十分収束した段階で具体値に置き換え、`/robot_description` と `/tmp/robot_realized.urdf` を生成する。これは「不定座標変換」を最終的に決定論的 URDF へ落とす処理である。

`scripts/relpose_estimator_registrator.py` は symbolic URDF から IMU ペアを抽出し、ペアごとの推定ノードを起動する。`scripts/relpose_visualizer.py` は推定位置分布と Bingham 姿勢分布を可視化する。

## Prob-TF 文脈での翻訳

このリポジトリは、Prob-TF の文脈では最も直接的な distribution producer である。出力している `PoseWithCovAndBingham` は、ある parent IMU frame から child IMU frame への相対変換分布であり、回転は quaternion Bingham、並進は Gaussian summary として表されている。

Prob-TF overview の用語で言えば、`this_imu -> child_imu` は physical edge に対応し、`rotation_bingham_parameter` と `position_covariance` はその edge posterior の summary である。特に、URDF 中で未確定だった module 間の取り付け位置・姿勢を観測から推定するため、overview の「不定座標変換」をそのまま扱っている実装と見なせる。

一方、現在の実装ではこの分布は専用 topic で流れ、tf2 tree や Prob-TF graph とは分離している。さらに `robot_description_setter.py` は、分布の不確かさが閾値以下になった段階で単一の `xyz/rpy` に変換し、URDF を固定値として書き換える。Prob-TF に統合する場合、この処理は「推定分布を query 可能な edge として保持する段階」と「必要に応じて代表値で deterministic URDF を materialize する段階」に分けるのがよい。

また、現在の message には `parent_frame_id` と `child_frame_id` が独立フィールドとして存在せず、`header.frame_id` に `<this>__to__<child>` の ID を入れている。Prob-TF の transport layer に載せるには、frame ID、edge ID、source、representation type、近似フラグ、時刻、依存関係 metadata を明示する必要がある。

## 統合時の候補設計

1. `PoseWithCovAndBingham` を `probtf_msgs/ProbabilisticTF` または後継 message に写像する adapter を作る。
2. `this_imu_name` と `child_imu_name` を `parent_frame_id`、`child_frame_id`、`edge_id` として明示する。
3. `rotation_bingham_parameter` は `BinghamDistribution.matrix` へ、`relative_position.position` と `covariance` は `GaussianPosition` 相当へ移す。
4. `approximation_type` には `bingham_rotation_plus_gaussian_translation` や `extended_leastsq` のように、推定法と表現を分けて残す。
5. `robot_description_setter.py` は Prob-TF consumer として再定義し、`lookup_distribution` の収束判定後に deterministic URDF を生成する役割にする。

## 移行上の注意点

- 回転と並進の cross-covariance は現在の message には出ていない。推定内部で回転と位置が結合している箇所があるため、Prob-TF では「独立 Gaussian 並進 + Bingham 回転」として公開してよい条件を metadata で明示する必要がある。
- 逆向き変換は単に別 edge として publish せず、Prob-TF の inverse view として扱うべきである。`T` と `T^{-1}` は同じ latent transform を共有する。
- symbolic URDF の placeholder を置換して `/robot_description` を更新する処理は、分布の情報を失う terminal operation である。Prob-TF graph に戻す場合は `closure_approximation=true` 相当の明示が必要になる。
- `joint_estim_hdbingham.py` には高次元 Bingham 方向の実験的コードが残っているが、未完成に見える部分がある。初期統合では `extended_leastsq.py` の実動経路を優先するのが現実的である。
