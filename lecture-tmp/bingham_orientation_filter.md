# bingham_orientation_filter の整理

## 現状の実装サマリ

`bingham_orientation_filter` は、IMU の角速度・加速度、必要に応じて磁気センサと mocap 由来姿勢を使い、姿勢を quaternion Bingham 分布として逐次更新する小さな ROS パッケージである。ファイル構成は最小限で、`scripts/imu_filter.py` がほぼ全実装を持ち、`CMakeLists.txt` と `package.xml` は catkin の雛形に近い。

`BinghamParameterUtils` は、Bingham パラメータ行列 `A` と quaternion 二次モーメント `CovQ` の相互変換を担当する。固有値から Bingham 正規化定数の導関数を計算し、`Cov_to_Amat`、`Amat_to_Cov`、`estim_Z` などで moment matching 的に分布パラメータを更新する。

`BinghamFilterForIMU` がフィルタ本体である。状態は `self.Amat`、つまり姿勢 quaternion 上の Bingham 分布で表される。prediction では gyro の平均角速度とノイズから微小回転 quaternion の二次モーメントを近似し、現在の `CovQ` と quaternion 積の二次モーメントを合成して次時刻の Bingham へ戻す。update では、観測された加速度ベクトルと基準重力ベクトルの対応から Bingham 型の観測項 `obsA` を作り、prediction の `A` と足し合わせる。

`BinghamFilterForIMUForROS` は ROS wrapper と可視化を兼ねている。`/delta/imu` の `spinal.msg/Imu`、`/delta/mocap/pose` の `PoseStamped` を購読し、初期姿勢や磁気基準を mocap から与える。推定結果は ROS message として publish するのではなく、`bingham.visualize.SO3s.draw_bingham_distribution` で matplotlib の 3D 図として表示する構成である。

現状では package metadata に `rospy`、`sensor_msgs`、`geometry_msgs`、`spinal`、`bingham`、`scipy`、`matplotlib` などの実依存が明示されていない。また、フィルタ結果を他ノードが利用するための message 出力も未実装である。

## Prob-TF 文脈での翻訳

Prob-TF の観点では、このパッケージは「IMU body frame の姿勢 transform 分布を生成する producer」である。例えば `world -> imu` または `base -> imu` の回転成分について、単一 quaternion ではなく Bingham 分布 `A` を時系列で更新している。

この実装は、Prob-TF overview で重要視されている quaternion の antipodal symmetry に自然に対応している。`q` と `-q` を同一姿勢として扱えるため、通常の Gaussian quaternion よりも Prob-TF の Bingham rotation payload に近い。

ただし、現在扱っているのはほぼ姿勢のみであり、並進や回転-並進 coupling は持たない。したがって Prob-TF edge としては、translation が既知または未使用の `Bingham rotation only` producer として整理するのがよい。IMU がロボット上の固定位置にあるなら、URDF 由来の deterministic translation と、このフィルタ由来の stochastic rotation を同じ edge に合成する設計が考えられる。

また、現状の ROS wrapper は推定分布を publish せず、可視化で閉じている。Prob-TF へ統合するには、`Amat`、mode quaternion、時刻、frame ID、観測ソース、ノイズパラメータを含む distribution message を publish する必要がある。

## 統合時の候補設計

1. `BinghamFilterForIMU` を ROS 非依存の Python core として `src/probtf` または IMU producer 用 namespace に移す。
2. ROS wrapper は `sensor_msgs/Imu` と必要な独自 IMU message を adapter として扱い、出力を `probtf_msgs/BinghamDistribution` または `ProbabilisticTF` に寄せる。
3. `parent_frame_id`、`child_frame_id`、`edge_id` を ROS parameter で指定できるようにする。
4. deterministic translation を持つ場合は、URDF/tf2 の固定 transform と組み合わせて `Bingham rotation + fixed translation` edge として publish する。
5. 可視化は consumer に分離し、Prob-TF query 結果を描画する形にする。

## 移行上の注意点

- `A` のスケール、trace shift、正規化定数の計算条件を Prob-TF 側の Bingham convention と一致させる必要がある。
- prediction で `CovQ -> Amat` へ戻す処理は近似であるため、message metadata に `closure_approximation` または `moment_matched` 相当の情報を残すべきである。
- mocap を初期化と基準磁場推定に使っているため、実運用 producer と評価用可視化コードを分けた方がよい。
- 並進を含まない姿勢分布を通常の `TransformStamped` 的な edge に見せる場合、translation の由来が deterministic なのか未推定なのかを明示しないと consumer が誤解する。
