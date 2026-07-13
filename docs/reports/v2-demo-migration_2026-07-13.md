# ProbTF demo v2 全面移行 作業報告

- 開始日時: 2026-07-13 02:57:12 JST
- 対象 repository: `/home/leus/catkin_ws/src/ProbTF-demo`
- 開始時 HEAD: `ad8a23922a0bcaffaff060e123973e229dd8806d`
- 開始時状態: clean

## 1. 目的

各 demo を legacy `ProbabilisticTF` 表現と個別の伝播実装から切り離し、native v2
`TransformDistributionStamped`、`ProbTfGraph`、topic listener、lookup kernel を使う構成へ
移行する。移行完了時には v1 message、v1 domain model、v1/v2 adapter、symaware 固有の
旧 `ProbTfTree` を削除する。

`2846fced8` で追加された `SOURCE_DIR = "../../../../src"` による catkin package 外の
Python source relay も廃止し、各 namespace を所有する catkin package の `src/` に置く。

## 2. 移行時の設計判断

1. `/probtf` graph に登録するのは、並進と回転がともに定義された physical SE(3) edge
   だけとする。orientation-only likelihood に zero translation を補わない。
2. orientation-only posterior と singular likelihood evidence は、full transform record と
   分けた v2 message で運ぶ。
3. grasp target や end-effector belief の derived transform は、v2 component lawを閉じたまま保持し、
   派生元edge IDをprovenanceへ残せる場合だけ別edgeとして登録する。application topicは用途metadataと
   完全なv2 payloadを運ぶ。point momentなどのterminal summaryはgraphへ再登録しない。
4. lookup は中央 bridge process への独自 RPC ではなく、tf2 と同様に各 consumer が
   `/probtf` と `/probtf_static` を listen して local `ProbTfGraph` を構築する方式にする。
5. deflecomp の stiffness posterior は SE(3) 分布ではないため `/probtf` に偽装しない。
   frame transport と可視化 query を TF import -> v2 listener/lookup に移す。
6. point cloud は lookup kernel の point moment result だけを入力とし、表示用 Gaussian sample
   は terminal visualization として生成する。生成した summary を graph edgeへ戻さない。

## 3. 基準状態

- Python tests: `180 passed, 1 warning`
- `probtf` は catkin devel space から import できず、root project の事前 install が必要だった。
- v1 ROS publisher/consumer: two-IMU、orientation filter/fusion、symaware grasp 一式。
- symaware link cloud: YAML -> legacy `ProbTfTree` -> local tangent propagation。
- deflecomp: v1 message は使わないが、ProbTF graph/topic と未接続。

## 4. 作業ログ

### 4.1 catkin package 内への Python source 正規配置

- `probtf`、`probtf_estimators` を `ros/core/probtf_core/src/` へ移した。
- `deflecomp_core`、`deflecomp_sim`、`deflecomp_examples` を各 package の `src/` へ移した。
- `symaware_grasp` を ROS package の `src/` へ移した。
- 3個の deflecomp `setup.py` から `SOURCE_DIR = "../../../../src"` を削除した。
- `probtf_core` と `symaware_grasp` の setup を package-local `find_packages()` に統一した。
- catkin dependency を明示し、test 内の `sys.path.insert()` を削除した。
- package boundary test を新しい source ownership に合わせて更新し、parent path relay と
  test-time path injection を検出するようにした。

検証:

- Python tests: `181 passed, 1 warning`
- `catkin build probtf_core`: 成功
- `catkin build probtf_msgs symaware_grasp`: 成功
- `catkin build deflecomp_sim deflecomp_ros`: 成功
- `source devel/setup.bash` 後、`probtf`、`probtf_estimators`、`probtf_ros`、
  `deflecomp_core`、`deflecomp_sim`、`deflecomp_examples`、`symaware_grasp` の import: 成功

### 4.2 Bingham 正規化積分の core 内収容

- `probtf.bingham` が外部 `bingham` Python package を importする構成をやめた。
- 実際に必要な正規化定数と導関数の積分実装だけを `probtf._vendor` に収容した。
- upstream の copyright / Apache-2.0 header を保持し、`THIRD_PARTY_NOTICES.md` に由来を記録した。

検証:

- external `bingham` importを強制拒否した状態で `import probtf`: 成功
- Bingham / v2 distribution / kernel tests: `42 passed`

### 4.3 deterministic right composition の v2 閉包

- stochastic record の各 component に deterministic child offsetを右合成する core 演算を追加した。
- `R_new = R_old R_fixed` に伴う coupling の基底変換と `R_old r` 項を、3x9
  `rotation_coupling` に保持する。
- mixture weight、residual covariance、representative、approximationを保持し、derived edgeを
  provenanceへ記録する。
- symaware grasp target は、この演算を使うことで v1 の mode plug-in / covariance samplingを
  廃止できる。

検証:

- 任意 quaternion 20点で、旧 component + fixed transform の直接評価と合成後 component の
  conditional meanが `1e-12` 以内で一致
- composition tests: `2 passed`

### 4.4 symaware YAML の native v2 loader

- demo arm YAMLから `TransformDistributionStamped` の static edgeを直接構築する loaderを追加した。
- revolute joint は finite `BinghamOrientation`、fixed joint は Dirac orientationとして表す。
- translation、component weight、representative、authority、provenanceを v2 recordに格納する。
- frame一覧とedge topologyの一致を検証し、旧 treeを経由せず `ProbTfGraph` を構築する。

検証:

- 7 edgeのorientation kind / static flag / couplingを検査
- `link_3 -> base_link` の native lookup pathとpoint moment評価を検査
- v2 config tests: `2 passed`

### 4.5 local graph listener と bounded runtime bridge

- `ProbTfGraph` の insert / lookup / record解決を `RLock` で保護した。
- frame / physical edge の immutable snapshot APIを追加した。
- transport-neutral listener に lookup path/kernel、point moments、can/wait lookupを追加した。
- dynamic/static v2 topicを所有して local graphを作る `RosProbTfListener` を追加した。
- dynamic/static channelの `is_static` 不一致を拒否し、runtime historyを既定1000件/edgeに制限した。
- broadcasterに複数recordとstatic setを一度に送るAPIを追加した。
- bridge launchに v2 topic名、history上限、TF import child-prefix filterを公開した。

検証:

- graph/listener/bridge関連 tests: `45 passed`
- `catkin build probtf_msgs probtf_core`: 成功
- 担当差分 `git diff --check`: 成功

### 4.6 estimator domain と two-IMU demo の native v2 化

- `ImuKinematics` と `SensorMount` を legacy transform model から分離した。
- two-IMU estimator の出力を、旧 Gaussian/Bingham summary から単一 component の
  `TransformDistributionStamped` へ変更した。
- 登録済み joint geometry の `p = a - R b` を `rotation_coupling` に保持し、回転 mode を
  代入しただけの位置平均へ潰さないようにした。
- 未登録 joint の位置 RLS は orientation mode を plug-in する loss のある近似であることを、
  `ApproximationInfo` と provenance に明示した。
- symbolic URDF materializer は v2 record を購読し、point moment kernel で coupling を含む位置
  moment を計算するようにした。mixture は暗黙に一成分へ縮約せず拒否する。

### 4.7 orientation-only evidence / posterior の v2 wire contract

- 旧 `TransformEvidence.msg` を `TransformEvidenceStamped.msg` に置き換えた。
- Bingham evidence は trace-zero natural parameter、並進 evidence は singular PSD を許す
  information form として wire 上に保持する。
- orientation-only posterior 専用の `OrientationDistributionStamped.msg` を追加した。
  この message には意図的に translation field がない。
- orientation filter / fusion demo を新 message、structured approximation、provenance に移行した。
- filter の gyro prediction closure と、likelihood product の意味を metadata に保持した。

検証:

- Python tests: `198 passed, 1 warning`
- `catkin build probtf_msgs probtf_core probtf_imu_demo probtf_orientation_demo`: 成功
- devel space の生成 message / estimator import: 成功
- two-IMU / orientation launch の node 解決: 成功

### 4.8 native v2 sampling backend

- Dirac、uniform、finite Bingham orientation を v2 domain model から直接 sample するAPIを追加した。
- mixture の正規化済み正 weight、component ごとの conditional Gaussian translation、
  `rotation_coupling` を保った joint transform sample を生成する。
- sampled transform の forward / inverse point action を一つのAPIで実装した。
- `KernelEvaluator.SAMPLES` を Gaussian point input、mixture、coupled translation、
  inverse、composed path に接続した。
- 反復 latent edge は既存の dependency checkで拒否し、独立な再標本化をしない。
- stochastic sample result は `MONTE_CARLO` approximation として型付けする。

検証:

- sampling / kernel tests: `18 passed`
- `tests/probtf`: `143 passed`
- finite Bingham sampleの二次moment、mixture比率、coupling、forward/inverse統計を検査

### 4.9 deflecomp frame runtime の v2 接続

- `deflecomp_frames.launch` に、`ref`、`cmd`、`equil` child frameだけを対象とする
  import-onlyの共通 TF -> ProbTF v2 bridgeを組み込んだ。
- 実際の robot-state-publisher / static TFを `/deflecomp/probtf` と
  `/deflecomp/probtf_static` の native v2 recordとして配信する。
- 別 processの `RosProbTfListener` から `lookup_path` と `lookup_point_moments` を呼ぶ
  consumerを追加し、point mean / covariance axisをRViz markerとして表示する。
- stiffness `kp_*` posteriorはSE(3) transformではないため、ProbTF graphへ登録していない。
- `viewer:=false` ではGUI/RVizを起動しないheadless構成にし、`viewer:=true` の既存表示を保った。

検証:

- Deflecomp core/sim/v2 runtime tests: `28 passed, 1 warning`
- 関連7 catkin package build: 成功
- 指定されたYamaguchi arm URDF / IMU configでviewer false/trueのlaunch解決: 成功
- devel spaceだけを使ったconsumer node import: 成功
- 実ROS master上のtopic smokeはsandboxのnetwork-interface制限により、この時点では未実施

### 4.10 core / ROS message の v1 廃止

- `probtf.models` と `probtf.compatibility` を削除し、v1 domain modelと双方向adapterを廃止した。
- `probtf_ros.conversions` と `probtf_ros.legacy_conversions` を削除した。
- `ProbabilisticTF.msg`、`ProbabilisticTFArray.msg`、`GaussianPosition.msg`、
  `BinghamDistribution.msg` をmessage生成対象とsourceから削除した。
- `ApproximationKind.LEGACY_ADAPTER` を廃止した。既存v2 wire number 2は別の意味へ
  再利用せずreserved slotとしてdeserialize時に拒否する。
- legacy module/symbol/messageがsourceとCMakeへ再導入されないboundary testを追加した。
- legacy adapter専用testを削除し、残すべきIMU covariance testはv2 estimator testへ移した。

検証:

- core / boundary tests: `143 passed`
- `catkin clean/build probtf_msgs probtf_core`: 成功
- core runtime sourceに対するlegacy symbol検索: 0件

### 4.11 symaware grasp / link cloud の native v2 runtime 化

- object、hand、grasp target、selected target、IK resultをsymaware固有messageへ分離し、
  各application messageが完全な `ProbabilisticTransformStamped` v2 payloadを内包するようにした。
- object / hand / derived grasp target recordを `/probtf`、YAML arm edge 7件を
  `/probtf_static` へpublishするbroadcasterを実装した。
- grasp target node、IK、visualizerを `RosProbTfListener` のexact/latest lookupへ移行した。
- grasp offsetの右合成はmixture、residual covariance、`rotation_coupling`、派生元edge provenanceを保持する。
- link cloudは全linkの軸端点を `lookup_point_moments()` で取得し、Gaussian samplingは
  PointCloud2を作る表示終端だけで行う。
- 一般visualizerはlistenerで得たdistributionをcore v2 samplerへ渡し、全componentを描画する。
- IKは全mixture componentのcoupled point momentsを評価し、finite Bingham以外を必要とする
  Bhattacharyya methodでは暗黙変換せず明示的に拒否する。
- handのsample fitはlossy `MOMENT_SUMMARY`、設定object lawは`PRODUCER_SUPPLIED` として、
  component / record / ROS wireの全てにapproximationを保持する。
- 旧 `ProbTfTree`、`ptf_utils`、外部Bingham runtime、v1 ROS conversion、手計算sample scriptを削除した。
- Bingham Bhattacharyya用のgauge-aware `bingham_log_normalizer()` をcoreへ追加した。

検証:

- 全Python tests: `199 passed, 1 warning`
- devel/message未sourceのsymaware source-only tests: `26 passed`
- `catkin build probtf_msgs probtf_core symaware_grasp`: 成功
- generated application message roundtrip / node import / launch node解決: 成功
- link-cloud 12秒実動: static v2 record 7件とlistener pointcloud publishを確認
- full demo 18秒実動: grasp target、selected grasp、IK node正常終了を確認

## 5. コミット

| commit | 内容 |
| --- | --- |
| `db91ed4` | Python sourceを所有 catkin package 内へ正規配置 |
| `89f3811` | Bingham 正規化積分を core 内へ収容 |
| `61fb198` | deterministic right compositionで v2 couplingを保持 |
| `2f870d2` | symaware YAMLを native v2 static recordsとしてload |
| `09038d8` | bounded v2 topic listener と lookup APIを追加 |
| `89c1e85` | estimator demoとorientation-only wire contractをnative v2化 |
| `42c699e` | native v2 transform kernel samplingを実装 |
| `52a7868` | deflecomp frame runtimeをv2 bridge/listener/lookupへ接続 |
| `a497a15` | ProbTF v1 core model、adapter、ROS wire contractを削除 |
