# IMU対から不確かさを残した相対座標変換を推定する

`/imu` の `sensor_msgs/Imu` から2台を選び、位置のGaussian分布と方位の
Bingham分布を逐次推定して、native v2 の `/probtf`
(`probtf_msgs/ProbabilisticTransformStamped`) に出力する。
URDF、symburdf、関節名の一覧、CADの取り付け位置、IKは不要。
収束したところで固定せず、その後の観測でも分布を更新し続ける。

この推定は **同じ剛体に固定された2台のIMU** の未知の取り付け変換を対象にする。
例えば、元のロボット構成ではある関節のchild IMUと次の関節のparent IMUの対である。
動く関節を挟む2台に、固定変換用のこの推定式をそのまま適用しない。

## 起動

このPCではリポジトリ直下の `.venv` にPython 3.8の数値計算環境を用意している。
catkinにもそのPythonを指定する。生成されたrosrun/roslaunch用の実行ファイルが
このPythonを使うため、起動用shell scriptやvenvのactivateは不要。

```bash
cd ~/catkin_ws
catkin build relative_coords_from_paired_imu --cmake-args \
  -DPYTHON_EXECUTABLE="$HOME/catkin_ws/src/ProbTF-demo/.venv/bin/python"
source devel/setup.bash

roslaunch relative_coords_from_paired_imu imu_relative_pose.launch \
  parent_frame_id:=joint1_child_imu \
  child_frame_id:=joint2_parent_imu
```

上記のframe名は例なので、実際に選ぶ2台の `/imu/header/frame_id` に合わせる。
`joint1_child_imu` のような名前は、Fire受信ノードの
`joint1_child_imu__abcdef` にも一致する。MACまで指定すればその1台だけを選べる。
同じ論理名に異なるMACが現れたら、混ぜずに曖昧な入力として通知する。
選ばなかったIMUは無視する。Fireが複数台でも、入力は共通の `/imu` のままでよい。

入力topicは `imu_topic:=/imu`、出力は `probtf_topic:=/probtf` で変更できる。
複数の対を推定する場合は、対ごとに `node_name:=link2_imu_relative_pose` など
異なるノード名と異なる2つのframeを指定する。複数の変換を同じ `/probtf` に出せる。

```bash
rostopic echo /probtf
# ウィンドウ・事後・MACの対応をまとめてリセットする場合
rostopic pub -1 /imu_relative_pose/reset std_msgs/Empty '{}'
```

このlaunchが立ち上げるのは受信・推定ノードだけである。
Fireのserial受信ノードは既存のものを別途起動して `/imu` を流す。

## このexample内の実装構成

ペアIMU専用の実装・型・メッセージ・テストは、このディレクトリにまとめている。
ROSに依存しない数値処理と、ROSの入出力はファイルを分ける。

```text
relative_coords_from_paired_imu/
  src/relative_coords_from_paired_imu/  # ROS非依存の推定・SG・型
  nodes/                              # rospy・ROSメッセージの入出力
  msg/ImuKinematics.msg               # 旧2-topic例の専用メッセージ
  config/  launch/                    # 設定と起動
  tests/                             # 数値処理・非同期入力の回帰テスト
  test/smoke_raw_imu_ros.py            # 独立ROS masterでの実接続テスト
```

`src/relative_coords_from_paired_imu/` の主なモジュールは次のとおり。

- `imu_extended_leastsq.py`: `ExtendedLeastSquaresImuRelativePoseEstimator`。
  同一時刻で評価した2つの `ImuKinematics` を受け取り、native ProbTFオブジェクトを返す。
- `latest_imu_pair.py`: `LatestImuRelativePosePipeline`。生データ保持と推定更新を分離し、workerが最新の有効なSGペアだけを更新する。
- `imu_pair_pipeline.py`: オフライン互換APIの `ImuRelativePosePipeline`。
  frame名、取得時刻、gyro、accel、各3×3共分散を直接受け取り、
  非同期入力の同期と上記の推定をまとめて実行する。
- `imu_kinematics.py`: ROSメッセージに依存しない入力データ型。
- `ros_interface.py`: 旧2-topic例のメッセージフィールド変換。ROSのimportは不要。

`nodes/raw_imu_relative_pose_node.py` がROSメッセージの入出力・受信worker・設定・固定bias補正を担当する。
共通 `probtf_core` は汎用の分布・モーメント等の依存先であり、ペアIMUの実装を所有しない。

```python
from relative_coords_from_paired_imu import ImuRelativePosePipeline

pipeline = ImuRelativePosePipeline("imu_parent", "imu_child")
# 各IMUの取得ごとに呼ぶ。単位はrad/s, m/s^2、stampは秒。
for record in pipeline.update_all(frame_id, stamp, gyro, accel, gyro_cov, accel_cov):
    component = record.distribution.components[0]
    position_mean = component.translation.mean_at_reference
    position_covariance = component.translation.residual_covariance
    bingham_A_wxyz = component.orientation.parameter_matrix()
```

上の `update_all` は全取得時刻を再現するオフライン互換APIで、live ROSノードでは使用しない。
従来の1件返却の `update` も残しており、余分な結果は破棄せずFIFOに保持する。
このAPIで有限個の入力を処理する場合は、最後に `pipeline.drain()` も取り出す。

このexampleのPythonパッケージ、汎用 `probtf`、NumPy/SciPy/PyYAMLだけで使える。
`rospy`、生成済みROSメッセージ、元リポジトリ、numpy-quaternion、外部BinghamNLLを
importする必要はない。Binghamモーメントには既存ProbTFの正規化定数実装を使う。

## 移植した式と分布の意味

元コードは `urdf_estimation_with_imus` developの `291f66f` にある
`pypkg/src/imu_relpose_estim/estimator/extended_leastsq.py` と
`utils/sequential_leastsq.py`。その **剛体対・既知joint位置なし** の経路を移植した。
元リポジトリのファイルは変更していない。

座標変換は `x_parent = R(q) x_child + p`。つまり、出力headerのframeがparent、
`child_frame_id` がchildで、`p` はparent座標で表したchildの原点である。
四元数の数値コアの並びはwxyzで、ROS境界のフィールドはxyzw。

方位はgyroの対応 `omega_parent = R(q) omega_child` を使い、
元の `H` と `Q = N (diag(C_parent, C_child) ⊗ I/4) Nᵀ` から
`A_next = gamma_rot A - Hᵀ Q⁻¹ H / 2` を更新する。
`Q` は各対のSG補間後の共分散から毎回計算する。
元実装の `set_Qinv_once` は、直後の `update_Amat` が `CovInv` を消すため、
実際には観測のたびに再計算する。最初の対で固定してはならない。
ここで使う一様四元数二次モーメント `I/4` によって、Qはgyro共分散のtraceに依存する。
未観測の方位を一点へ固定せず、初期は一様分布であり、有限なBingham分布を保つ。

位置はBinghamから計算した **`E[R]`** を使う。mode回転への置換ではない。

```text
Omega_i = [E omega_i]x^2 + Cov(omega_i) - trace(Cov(omega_i)) I + [E alpha_i]x
Delta_f = E[R] f_child - f_parent
X = block_diag(Omega_parent, E[R] Omega_child E[R]^T)
y = [Delta_f, Delta_f]
```

既定の `use_child_gyro: true` は元と同じ6次元の逐次最小二乗と、
2つの3次元Gaussianのprecisionによる合成を行う。
`false` なら元の平均operatorを使う3次元経路に切り替わる。
元の現在の作業ツリーの位置更新は `Gamma = I / gamma` なので、
`position_observation_weight` は新しい情報への重みであり、過去を忘れる係数ではない。
既定1.0は全観測を蓄積する。`position_residual_variance` の既定1.0もこの式に対応する。

移植時に数値計算を変更した箇所は次のとおり。

- 位置の初期共分散は各fit `1e30 I m^2` で、元と同じである。
  元の繰り返しの擬似逆に代えてinformationを蓄積し、同じGaussian式をsolveで計算する。
  必要なsolveが実際に失敗した更新は通知する。未識別方向を零分散に置き換えたり、
  更新途中にjitterや分散clipを加えたりしない。
  2つのfitを合成する既定経路では、観測前の合成共分散はその1/2となる。
- `E[R]` は既存ProbTFの `E[qqᵀ]` から直接計算する。
  元の四次モーメントを `qᵀq=1` で縮約する式と数学的に同じである。
- 入力は元と同じ11件・7次多項式、各IMUで直近10個のSGフィットを既定とする。
  時間軸を窓の中央の取得時刻で原点移動・scaleし、数値的に安定なfitを行う。
  元と同じくparentの各観測イベントで両raw窓の重なりの中央を評価時刻とし、
  各IMUの履歴から、その時刻を含み中心が最も近いSG窓を選ぶ。
  同じ多項式を `t* - window_center` の符号で評価・微分し、
  同一取得時刻のgyro・角加速度・比力を計算する。
  fitの共分散は各測定の全3×3共分散を同じ線形fitの重みで伝播する。

## 非同期入力とSG時間合わせ

liveの受信callbackは各IMUの実測データを保存し、推定workerは前の計算が終わったら
最新の有効なペアを一つだけ選ぶ。生データの周期と推定更新・描画の周期を分け、
過去のすべての観測に対する事後更新をFIFOで追いかけない。出力queueも1件である。

IMU間の到着順、バッチ分割、片側の先行を許容する。各IMU内部の取得stampは単調増加が必要。
両IMUの最新取得時刻の小さい方を共通frontierとし、そこまでの最新の実測窓の重なりの中点を選ぶ。
各IMUの直近10窓相当から、その時刻を含み中心が最も近いSG窓を使う。これは元の窓選択方針を保持する。
全窓を全入力でfitせず、選ばれた窓だけをfitする。補間と微分には同じ多項式を使い、
整数ナノ秒で差を取ってから秒へ変換し、正しい `t_eval - t_center` を評価する。

窓が重ならない、先行入力の保持容量 `raw_history_capacity=16384` を越えて必要なデータが消えた、
または設定 `max_sample_gap_s=0.05` を越える取得空白が窓内にある場合は待機する。
窓外の外挿、端点・過去の微分値による代用、到着時刻への付け替えは行わない。
`provenance.detail` の `Latest SG selection` に選んだ全取得stampをナノ秒で残す。
処理速度によりliveで選ぶ時刻と更新件数は変わるが、同じ選択窓の値・微分・共分散は変わらない。

ROS非依存APIは `LatestImuRelativePosePipeline.ingest(frame, stamp_ns, payload)` と
`update_latest()`。payloadはgyro・force・両共分散の組で、必要ならdecode関数で校正する。
全履歴の再現が必要なオフライン処理は従来の `ImuRelativePosePipeline.update_all` を使う。

元の状態は `N(p; mean, covariance) × Bingham(q; A)` で、位置と方位の交差依存を
推定していない。そのためnative ProbTFにはGaussian全共分散とBingham全形状を渡し、
`rotation_coupling = 0` として、この因子化近似を `approximation.lossy=true` と説明文で残す。
元の2つの位置fitの共有観測の相関、回転不確かさが位置事後へ与える全依存、
重なる多項式窓の時間相関も含む厳密な同時事後ではない。
加速度ノイズ・角加速度ノイズの全伝播まで導入した新しいBayes推定と説明しない。
シリアライズで分布を落としたり、6×6の接空間Gaussianに置き換えたりはしない。
分布の期待値・modeからURDFを作る処理が必要なら、後段で明示的に行う。

## 校正と時刻

`config/relative_pose.yaml` のbiasとunknown-covariance時の分散は設定可能。
既定biasは零で、分散は仮定値であり、このノードが静止校正を済ませた値ではない。
入力が既に校正されている場合はbiasを二重に引かない。
`sensor_msgs/Imu` の全零共分散はunknownとして設定値を使い、先頭が-1なら
その必須測定を利用不可として通知する。orientationフィールドは使用しない。

出力stampは **2台の観測窓を評価した共通の取得時刻**。
MACを含む元frame名と両窓の取得時刻範囲もprovenanceに入る。
ノードの処理時刻やpublish時刻に付け替えない。
入力stampに残るFire間の時計同期誤差は、このノードだけでは解消できない。
未知位置の識別には複数方向の回転・角加速度による励起が必要。

既存 `two_imu_relative_pose.launch` は別topic・簡略推定・URDF例の互換用として残している。
この新しい `/imu` 経路では `imu_relative_pose.launch` を使用する。

## 検証

```bash
cd ~/catkin_ws/src/ProbTF-demo
PYTHONPATH=ros/core/probtf_core/src:ros/examples/relative_coords_from_paired_imu/src \
  .venv/bin/python -m pytest -q \
  ros/examples/relative_coords_from_paired_imu/tests \
  tests/probtf/test_ros_v2.py \
  tests/probtf/test_bingham.py \
  tests/probtf/test_v2_composition.py \
  tests/probtf/test_materialization.py \
  tests/probtf/test_ros_conversions.py \
  tests/probtf/test_orientation_filter.py \
  tests/probtf/test_fusion.py
```

元のBingham・Gaussian更新式との比較、非自明な回転と位置を持つ剛体データ、
更新継続、非同期取得stamp、MACの選択・曖昧性、外挿防止、
ROS変換前後の分布・共分散・frame・時刻・出自の同一性を検証する。

ROS実ノードも専用の一時ROS masterで検証できる。ユーザーのROS masterや実機には接続しない。

```bash
source ~/catkin_ws/devel/setup.bash
PYTHONPATH="$HOME/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages" \
  .venv/bin/python ros/examples/relative_coords_from_paired_imu/test/smoke_raw_imu_ros.py
```

交互到着、10件ずつ、片側全件先着、到着jitter、異なる取得周期、欠測、同stampの
送信順逆転を検証する。元のSG窓選択を独立に記述した参照計算とも照合する。
解析的なSG係数、正負の補間時間差、微分と共分散の伝播も確認する。
2026-11-05（JST）、配置変更後に関連96テストと対象3パッケージのcatkin buildが成功した。
移動前に保存した5種類の非同期入力それぞれ109更新と比較し、取得時刻・位置の平均と共分散・
Binghamパラメータ等の数値結果がbit単位で一致することも確認した。

ROS実接続では、200件のraw IMUを交互・10件バッチ・parent先着・child先着・
2つの独立した送信プロセスで流す。最新frontierへの追従と、実際に選ばれたSG窓を使った
独立な参照更新との全出力比較を行い、位置平均のwire変換誤差0を確認した。
各条件の更新件数は処理タイミングにより異なる。全入力と同数の事後更新を要求しない。

実200mm bagの独立校正・精度・10倍速ROS入力はdthesis-final-demoの
`estimation/validate_200mm_bag.py` と `estimation/smoke_latest_ros.py` で検証する。
通常蓄積で、実ROSの最終距離197.22mm、距離誤差2.78mm、位置ベクトル誤差6.95mm、
外挿・処理エラー0、最新frontierへの追従を確認した。真値は事後評価にだけ使う。
