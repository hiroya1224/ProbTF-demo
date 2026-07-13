# ProbTF-demo 現行アーキテクチャの理論的詳細

- 更新日時: 2026-07-13 JST
- 対象範囲: ProbTF v2 foundation、ROS 1 transport、producer demo、symaware grasp、deflecomp、テスト境界
- DOT source: [`implemented-graph.dot`](./implemented-graph.dot)
- rendered graph: [`implemented-graph.svg`](./implemented-graph.svg)

本書は将来案ではなく、repository に存在する source、message、launch、test に基づく
現行実装の理論・構造説明である。実装変更に合わせて同じdirectoryのgraphと同時に更新する。
移行作業の経緯と検証結果は
[`v2-demo-migration_2026-07-13.md`](../../reports/v2-demo-migration_2026-07-13.md)、
数理 contract は
[`probtf_jmaa_kernel_architecture.md`](../probtf_jmaa_kernel_architecture.md) を参照する。

## 0. 結論

現在の `ProbTF-demo` は ProbTF v2 に統一されている。旧
`ProbabilisticTF` / `ProbabilisticTFArray` wire、旧独立 position/orientation domain model、
旧 v1/v2 adapter、symaware 固有 `ProbTfTree` は runtime と source から削除済みである。

現行 architecture の要点は次の七点である。

1. full SE(3) uncertainty は joint component mixture
   `TransformDistributionStamped` で表す。
2. physical edge は `/probtf` と `/probtf_static` の native v2 message で運ぶ。
3. lookup は中央 RPC ではなく、各 consumer の `RosProbTfListener` が local
   `ProbTfGraph` を構築して行う。
4. orientation-only posterior は translation を補わず、専用 message で運ぶ。
5. two-IMU producer は rotation/translation coupling を保持した full v2 edge を直接 publish する。
6. symaware grasp は global v2 graph と用途固有 application topic の両方を使う。
7. deflecomp は stiffness posterior を SE(3) に偽装せず、実 TF だけを scoped v2 graph に import する。

旧資料にあった「v2 bridge と v1 demo wire が未接続」という runtime gap は存在しない。
残っている課題は exact induced spherical law backend、stochastic inverse moments、temporal model、
closed-mixture reduction などの明示的に unavailable な数値機能であり、v1 relay の不足ではない。

## 1. Repository と package ownership

### 1.1 物理配置

```text
ProbTF-demo/
  ros/
    core/
      probtf_msgs/                 # native v2 ROS messages
      probtf_core/
        src/
          probtf/                  # ROS-free foundation
          probtf_estimators/       # ROS-free producer algorithms
          probtf_ros/              # ROS adapters/listener/bridge
        nodes/probtf_bridge_node.py
    examples/
      probtf_imu_demo/             # two-IMU producer + materializer
      probtf_orientation_demo/     # orientation-only filter/fusion
      symaware_grasp/
        src/symaware_grasp/        # application domain/runtime helpers
        msg/                        # application messages containing v2 payload
      deflecomp/
        deflecomp_core/src/deflecomp_core/
        deflecomp_sim/src/deflecomp_sim/
        deflecomp_examples/src/deflecomp_examples/
        deflecomp_ros/
        deflecomp_debug/
        deflecomp_description/
  third_party/BinghamNLL/          # provenance-preserved upstream source
  tests/
  docs/
```

Python namespace は所有する catkin package の `src/` に置かれる。以前の root `src/` を
catkin package から相対 path で relay install する構成はない。

- `probtf_core/setup.py` は自 package の `src/` だけを `find_packages()` する。
- deflecomp と symaware の各 setup も自 package の `src/` だけを所有する。
- `source devel/setup.bash` 後は parent path relay や test の `sys.path.insert()` なしで import できる。
- repository root の `setup.py` は非 catkin の統合 distribution 用に各 package-owned source root を
  列挙するが、catkin runtime の source ownership を変更しない。

### 1.2 catkin package

現在の catkin package は11個である。

| 分類 | package | 責務 |
| --- | --- | --- |
| core | `probtf_msgs` | ProbTF v2 wire contract |
| core | `probtf_core` | foundation、estimators、ROS adapter、generic bridge |
| producer | `probtf_imu_demo` | two-IMU preprocessing、relative pose、URDF materialization |
| producer | `probtf_orientation_demo` | orientation-only prediction/evidence/fusion |
| application | `symaware_grasp` | native v2 grasp composition、IK、visualization |
| deflecomp | `deflecomp_core` | equilibrium、WEKF、control、robot model |
| deflecomp | `deflecomp_sim` | flexible-joint simulation、synthetic IMU |
| deflecomp | `deflecomp_ros` | ROS closed loop、scoped ProbTF consumer |
| deflecomp | `deflecomp_examples` | offline examples |
| deflecomp | `deflecomp_description` | URDF、RViz configuration |
| deflecomp | `deflecomp_debug` | stiffness plotter |

### 1.3 依存方向

```text
probtf_estimators ---> probtf ---> NumPy / SciPy / vendored Bingham numerics
probtf_ros ---------> probtf + probtf_msgs + ROS 1
producer nodes -----> probtf_estimators + probtf_ros + probtf_msgs
symaware_grasp -----> probtf + probtf_ros + probtf_msgs
deflecomp_core -----> probtf geometry primitives + Pinocchio
deflecomp_ros ------> deflecomp_core + probtf_ros
```

`probtf` foundation は `probtf_estimators`、ROS、example application を import しない。
`probtf_ros` generic bridge は estimator を import しない。この境界は
[`tests/test_ros_boundary.py`](../../../tests/test_ros_boundary.py) の AST test で固定される。

## 2. ProbTF v2 domain model

### 2.1 Transform action と数値規約

physical edge `(parent, child)` は child frame の点を parent frame へ写す action である。

\[
z_{parent}=R(Q)z_{child}+X
\]

| 項目 | 規約 |
| --- | --- |
| core quaternion | `[w, x, y, z]` |
| ROS quaternion | `x, y, z, w` field。adapter だけで変換 |
| `vec(R)` | column-major |
| translation coupling | `C` は `3 x 9` |
| perturbation | reference rotation に対する right perturbation |
| lookup order | `lookup_*(target_frame, source_frame, ...)` |
| forward edge | child から parent への physical action |
| inverse edge | 同じ latent edge の逆向き view。別分布を生成しない |

### 2.2 Joint component law

中心 model は
[`probtf.distributions`](../../../ros/core/probtf_core/src/probtf/distributions) の immutable 型である。
component `l` は概念的に次を保持する。

\[
Q\mid L=l\sim \operatorname{Bing}(A_l)
\]

\[
X\mid Q=q,L=l\sim \mathcal N\left(
m_l+C_l(\operatorname{vec}R(q)-\operatorname{vec}R_{ref,l}),S_l
\right)
\]

| 型 | 内容 |
| --- | --- |
| `BinghamOrientation` | `FINITE_BINGHAM`、`DIRAC`、`UNIFORM` を型で分離 |
| `ConditionalGaussianTranslation` | reference mean、residual covariance、rotation coupling `C` |
| `TransformComponent` | raw weight と一つの joint pose hypothesis |
| `TransformDistribution` | component mixture、weight normalization/status |
| `TransformDistributionStamped` | parent/child、stamp、edge ID、authority、static、metadata |

有限 Bingham は trace-zero JMAA shape と inverse concentration を分けて保存する。Dirac を
極端な有限 concentration に置き換えず、uniform も zero parameter の曖昧な special value ではなく
独立 kind とする。

mixture weight は raw value を保存する。利用時に負 weight を0へ clampして診断を残し、正 mass で
正規化する。全て非正なら `ZERO_MASS`、NaN/Inf を含むなら `INVALID` であり、identity transform へ
黙って置換しない。

deterministic transform が exact に得られるのは、正規化後の一 component が Dirac orientation、
zero residual covariance、zero coupling を持つ場合だけである。stochastic law の representative は
`RepresentativePolicy` と `RepresentativeKind` を明示する。

### 2.3 Composition と provenance

[`composition.py`](../../../ros/core/probtf_core/src/probtf/distributions/composition.py) は stochastic
record の右側へ deterministic offset を合成する。これは grasp offset のような

\[
T_{world,grasp}=T_{world,object}T_{object,grasp}
\]

を mode plug-in へ潰さず、各 component の coupling basis を変換して保持する演算である。

[`probtf.provenance`](../../../ros/core/probtf_core/src/probtf/provenance) は source ID、派生元 edge ID、
method、detail を component/record ごとに持つ。`ApproximationInfo` は次を区別する。

- `EXACT`
- `PRODUCER_SUPPLIED`
- `TANGENT_SURROGATE`
- `NUMERICAL_INTEGRATION`
- `MONTE_CARLO`
- `MOMENT_SUMMARY`
- `MIXTURE_REDUCTION`
- `BINGHAM_CLOSURE`
- `REPRESENTATIVE_PROJECTION`
- `UNAVAILABLE`

`lossy`、source、detail、任意 error bound を同時に保持する。旧 wire number 2 は再利用せず、
deserialize 時に拒否される reserved slot である。

### 2.4 Bingham numerics

`probtf.bingham` が必要とする正規化定数と導関数の積分は
[`probtf._vendor`](../../../ros/core/probtf_core/src/probtf/_vendor) に由来を保持して収容される。
runtime は外部 `bingham` Python namespace の import に依存しない。
upstream notice は [`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md) に記録される。

## 3. Graph と temporal semantics

### 3.1 Local graph

[`ProbTfGraph`](../../../ros/core/probtf_core/src/probtf/graph/query.py) は次を所有する。

1. `ProbTfTopology`: disconnected component を許す TF-style forest
2. edge ID ごとの `EdgeTimeBuffer`: timestamp 順の static/dynamic history

topology は cycle と multiple parent を既定で拒否する。parent change は明示 policy がある場合だけ
診断付きで許可される。graph insert/lookup と frame/edge snapshot は lock で保護される。

buffer は out-of-order insert を時刻順に並べる。同一 timestamp の異 authority conflict は既定で
拒否し、`REPLACE` / `KEEP_FIRST` も選択できる。一 edge に static/dynamic を混在できない。
static edge は time invariant で、同一 payload の再送だけが idempotent である。

`RosProbTfListener` は既定で一 edge あたり1000 recordに historyを制限する。これは長時間 node の
local graph が無制限に増えないための runtime policy である。

### 3.2 Temporal policy

| policy | 現在の実装 |
| --- | --- |
| `EXACT` | 指定 stamp と完全一致する sample |
| `NEAREST_WITHIN_TOLERANCE` | tolerance 内最近傍。tie は古い sample |
| `LATEST` | 指定時刻以前の最新。任意 `max_age` |
| `LATEST_COMMON` | path 全体の最新共通時刻と zero-order hold |
| `INTERPOLATE_WITH_MODEL` | contract のみ。現在 unavailable |
| `PREDICT_WITH_MODEL` | contract のみ。現在 unavailable |

`LATEST_COMMON` は mixture を補間しない。dynamic edge histories の共通 availability interval を求め、
その時刻以前の最新 record を使う。sample stamp が共通時刻と異なる場合は
`LATEST_COMMON_ZERO_ORDER_HOLD` 診断を残す。

### 3.3 Lookup API

```python
listener = RosProbTfListener(
    dynamic_topic="/probtf",
    static_topic="/probtf_static",
)

path = listener.lookup_path(
    target_frame="base_link",
    source_frame="tool0",
    policy=TemporalPolicy.LATEST,
)
kernel = listener.lookup_kernel(
    "base_link",
    "tool0",
    policy=TemporalPolicy.LATEST,
)
moments = listener.lookup_point_moments(
    "base_link",
    "tool0",
    [0.0, 0.0, 0.1],
    policy=TemporalPolicy.LATEST,
)
```

中央 bridge process への query service/action はない。各 process が dynamic/static topic を listen し、
自分の local graph に対して `can_lookup`、`wait_for_lookup`、`lookup_path`、`lookup_kernel`、
`lookup_point_moments` を呼ぶ。これは tf2 buffer と同じ ownership pattern である。

## 4. Lazy kernel、moments、sampling

### 4.1 Lazy expression

graph lookup は分布を lookup 時点で一成分へ縮約せず、次の expression を構築する。

- `IdentityTransformKernel`
- `ForwardEdgeKernel`
- `InverseEdgeKernel`
- `MixtureTransformKernel`
- `ComposedTransformKernel`

同じ latent edge が stochastic path 内で反復された場合、独立 sample を引き直さない。
dependency-aware evaluator がない表現は `DEPENDENCY_UNRESOLVED` になる。

### 4.2 Point moments

forward component の point first/second moments は Bingham rotation moment、input covariance、
residual covariance、`rotation_coupling` を含めて評価される。deterministic path は exact、stochastic
path の first/second moment summary は元の law と同一ではないため `MOMENT_SUMMARY` として
lossy に型付けされる。

moment summary は terminal query result であり、新しい独立 edge として graph へ再登録しない。
symaware link cloud と deflecomp marker はこの規則を守る。

### 4.3 Native sampling

[`probtf.probability.sampling`](../../../ros/core/probtf_core/src/probtf/probability/sampling.py) は
native v2 law から直接 sample する。

- Dirac orientation: reference quaternion を反復
- uniform orientation: unit quaternion の一様 sample
- finite Bingham: rejection sampler
- mixture: 正規化済み正 weight で component を選択
- translation: quaternion 条件付き mean と residual Gaussian
- point action: forward / inverse
- path: independent edge の forward / inverse / composed sample

stochastic sample result は `MONTE_CARLO`、deterministic Dirac point path は exact として返る。
symaware visualizer はこの sampler で全 component と coupling を保持した表示 sample を作る。

### 4.4 現在の evaluator matrix

| representation / operation | status |
| --- | --- |
| lazy expression | 実装済み |
| deterministic forward/inverse/composition | exact |
| stochastic forward point moments | 実装済み、`MOMENT_SUMMARY` |
| native stochastic samples | forward/inverse/composed まで実装済み |
| Dirac/uniform/zero-vector induced law | exact special case |
| finite Bingham tangent induced law | 明示的 `TANGENT_SURROGATE` |
| finite Bingham exact induced density | unavailable |
| coupled numerical ISL integration | unavailable |
| stochastic inverse analytic covariance | unavailable |
| closed-mixture projection | explicit backend未実装 |

## 5. ROS v2 wire contract

### 5.1 `probtf_msgs`

core message は10種である。

| message | 用途 |
| --- | --- |
| `BinghamOrientation` | finite/Dirac/uniform、shape/scale/reference |
| `ConditionalGaussianTranslation` | mean、upper covariance、`3 x 9` coupling |
| `ProbabilisticTransformComponent` | weight と joint component law |
| `ProbabilisticTransformStamped` | 一 physical SE(3) edge |
| `ProbabilisticTransformArray` | static set 等の record 配列 |
| `ApproximationInfo` | typed loss metadata |
| `Provenance` | source/derived edge metadata |
| `ImuKinematics` | two-IMU preprocessing output |
| `TransformEvidenceStamped` | natural-parameter evidence |
| `OrientationDistributionStamped` | translation を持たない orientation posterior |

`ProbabilisticTransformStamped.header.frame_id` が parent/target frame の唯一の wire field である。
child、edge ID、authority、static flag、representative kind、component mixture、approximation、
provenance を同じ record に含む。

### 5.2 Dynamic/static transport

| topic | type | semantics |
| --- | --- | --- |
| `/probtf` | `ProbabilisticTransformStamped` | dynamic recordを個別 publish |
| `/probtf_static` | `ProbabilisticTransformArray` | broadcaster所有の全 static setをlatched publish |
| `/tf` | `tf2_msgs/TFMessage` | deterministic dynamic TF |
| `/tf_static` | `tf2_msgs/TFMessage` | deterministic static TF |

`ProbTfBroadcaster` は dynamic record と static set を正しい channel へ送る。
`RosProbTfListener` は dynamic channel 上の static record、static channel 上の dynamic record を拒否する。

### 5.3 Generic TF bridge

[`probtf_bridge_node.py`](../../../ros/core/probtf_core/nodes/probtf_bridge_node.py) は
`ProbTfTfBridge`、`ProbTfBroadcaster`、in-process graph を持つ。

- TF import: deterministic TF を Dirac orientation、zero residual covariance、`C=0` の exact
  one-component v2 recordへ変換する。
- TF authority: ROS connection caller ID を provenance/authority に保持する。
- TF export: exact edge、または呼び出し側が明示した representative policyだけを出力する。
- loop prevention: 自 node authority と export済み signature を除外する。
- filter: import対象 child frame prefixをlaunch parameterで制限できる。

generic bridge は application topic を自動発見する中央 serverではない。TF と v2 transport の境界だけを
担当し、queryは各consumerのlocal listenerが行う。

## 6. two-IMU relative-pose runtime

[`two_imu_relative_pose.launch`](../../../ros/examples/probtf_imu_demo/launch/two_imu_relative_pose.launch)
の dataflow は次である。

```text
/imu_parent/data ---> parent imu_kinematics_node --+
                                                    +--> imu_relative_pose_node --> /probtf
/imu_child/data  ---> child imu_kinematics_node ---+       v2 full SE(3) edge
                                                                  |
                                                                  v
                                                 symbolic_urdf_materializer
                                                                  |
                                                     /robot_description + RSP
```

二つの preprocessing node は local polynomial fit で angular velocity/acceleration、specific force、
各 covariance を `ImuKinematics` にする。`ApproximateTimeSynchronizer` が二 sensor を揃える。

`ImuRelativePoseEstimator` は単一 component の `TransformDistributionStamped` を返す。
登録済み joint geometry の `p=a-Rb` は `rotation_coupling` に保持される。未登録 geometry の位置 RLS は
orientation mode plug-in を用いるため、その loss を approximation/provenance に明記する。

出力は full SE(3) edge なので `/probtf` に直接 publishできる。materializer は v2 record の
point moments と orientation concentration を閾値で検査し、ready な summaryだけを symbolic URDFへ
terminal materializationする。mixtureを暗黙に一成分へ縮約しない。

sensor mount は設定から deterministic `/tf_static` を publishするが、relative-pose v2 record自体を
旧 message へ変換する経路はない。

## 7. orientation-only runtime

orientation demo は full SE(3) graph と意図的に分離される。

```text
/imu/data + /imu/mag
          |
          v
orientation_filter_node
  |-- prediction ---------- TransformEvidenceStamped
  |-- gravity evidence ---- TransformEvidenceStamped
  |-- magnetic evidence --- TransformEvidenceStamped
  |-- posterior ----------> OrientationDistributionStamped
          |
          v
probtf_fusion_node --------> OrientationDistributionStamped
```

topic は既定で次になる。

- `/orientation_filter/prediction`
- `/orientation_filter/gravity_evidence`
- `/orientation_filter/magnetic_evidence`
- `/orientation_filter/posterior_orientation`
- `/orientation_fusion/fused_orientation`

`TransformEvidenceStamped` は orientation natural parameter を trace-zero symmetric upper 10要素で
保持する。任意 translation evidence は singular PSD を許す information formであり、sequence、
approximation、provenanceを持つ。

posterior用 `OrientationDistributionStamped` には translation fieldが存在しない。gyro convolutionの
moment matchingは `BINGHAM_CLOSURE` / lossy、gravity/magnetic likelihoodはexactとして記録される。
fusionは同一 directed frame pairの独立 sourceを自然 parameter加算し、重複 sourceを既定で拒否する。

orientation-only lawへ zero translationを補って `/probtf` physical edgeにする処理はない。これは
未接続のgapではなく、fake SE(3)を防ぐdomain boundaryである。

## 8. symaware grasp runtime

### 8.1 Global graph と application topic

symaware は共通 `/probtf` と `/probtf_static` を使用する。

- YAML arm modelの7 static edge: `/probtf_static`
- object belief: `/probtf`
- hand/end-effector belief: `/probtf`
- composed grasp target records: `/probtf`

同時に用途 metadata を持つ application topicを使う。

| topic | application message | v2 payload |
| --- | --- | --- |
| `/symaware_grasp/object_belief` | `ObjectBelief` | 完全な stamped v2 transform |
| `/symaware_grasp/hand_belief` | `HandBelief` | 完全な stamped v2 transform |
| `/symaware_grasp/grasp_targets` | `GraspTargetArray` | 各 targetに完全なv2 transform |
| `/symaware_grasp/selected_target` | `SelectedGraspTarget` | 選択targetの完全なv2 transform |
| `/symaware_grasp/symmetry_aware_ik_result` | `IKResult` | solver result metadata |

application messageだけで独自 graphを作らず、message内のframe/stampで
`RosProbTfListener.wait_for_lookup()` と exact direct-edge lookupを行う。

### 8.2 Producer と composition

`probtf_static_broadcaster.py` は native v2 YAML loaderから static recordを直接作る。revolute jointは
finite Bingham、fixed jointはDiracであり、旧treeや外部Bingham runtimeを経由しない。

object nodeは設定 lawを `PRODUCER_SUPPLIED` としてpublishする。hand beliefはjoint sampleから
one-component moment fitを行うため `MOMENT_SUMMARY` / lossyを保持する。

grasp target nodeはobject recordへdeterministic grasp offsetを右合成する。mixture、residual covariance、
rotation coupling、raw weightsを保ち、object edge IDをderived provenanceへ残す。生成targetも
`ProbTfBroadcaster` でglobal graphへpublishされる。

### 8.3 Consumer、IK、visualization

`symmetry_aware_ik_node.py` はdefault launchに含まれ、grasp targetとjoint stateを待ってone-shot solveする。
全mixture componentのcoupled point momentsを評価し、deterministic baselineも別結果としてpublishする。
Bhattacharyya methodが扱えないorientation kindを暗黙なfinite Binghamへ変換せず拒否する。

link cloudは各link軸端点を `lookup_point_moments()` で取得する。表示用Gaussian sampleは
`PointCloud2`を作るterminal stepだけで生成し、graph edgeへ戻さない。

general visualizerはlocal listenerでrecordを解決し、native v2 samplerで全component、conditional
translation、couplingを含むaxis cloudを作る。component mode markerは表示用representativeであり、
graph lawの置換ではない。

## 9. deflecomp scoped runtime

### 9.1 ProbTFへ登録するもの、しないもの

deflecompのWEKFが推定するstiffness `Kp` posteriorはjoint parameter distributionであり、SE(3) edgeではない。
したがって次のtopicはProbTF graphへ登録しない。

- `/deflecomp/kp_hat`
- `/deflecomp/kp_est`
- `/deflecomp/kp_exec`
- `/deflecomp/kp_exec_target`
- `/deflecomp/kp_cov_diag`

これらはdeflecomp estimator/control/debugのdomain topicのままである。

ProbTFへ登録するのは、`robot_state_publisher` と static TF publisher が実際に出す `ref`、`cmd`、
`equil` frame transformだけである。

### 9.2 Scoped TF import

[`deflecomp_frames.launch`](../../../ros/examples/deflecomp/deflecomp_ros/launch/deflecomp_frames.launch)
はgeneric bridgeをnamespace `deflecomp`内にincludeする。

```text
ref/cmd/equil robot_state_publisher + static anchors
                         |
                  /tf + /tf_static
                         |
                         v
       /deflecomp/probtf_bridge (import=true, export=false)
                |                         |
                v                         v
 /deflecomp/probtf            /deflecomp/probtf_static
  dynamic v2 records           latched static v2 set
                \                         /
                 \                       /
                  v                     v
          /deflecomp/probtf_point_moments
                    RosProbTfListener
                 lookup path + point moments
                            |
                            v
       /deflecomp/probtf_point_moments MarkerArray
```

import child prefixは既定で `ref,cmd,equil`、TF exportはfalseである。したがって既存TFを
v2へmirrorするが、同じrecordをTFへ戻すloopは作らない。dynamic/static topicはglobal symaware graphと
分離されたscoped runtimeである。

consumerはURDFからbase/tipを解決し、各prefixのtip originについて `LATEST_COMMON` path lookup後に
point mean/covarianceを評価する。RVizはmean pointとcovariance principal axesのMarkerArrayを表示する。
consumerは `/tf` を直接lookupせず、必ず`RosProbTfListener`のlocal graphを使う。

`viewer:=false`ではnon-GUI joint-state publisherを使いRViz/plotterを起動しない。`viewer:=true`では
既存のthree-robot表示、stiffness plotter、ProbTF point-moment markerを表示する。

実ROS master上のheadless smokeでは、dynamic `/deflecomp/probtf`、latched
`/deflecomp/probtf_static`、および3系統すべてを含む
`/deflecomp/probtf_point_moments` MarkerArrayを確認した。dynamic topicはrobot-state-publisherの
各edgeを個別recordとして継続配信し、consumerは起動直後のgraph構築を待った後、`base_link`から
`ref/module4_link2`、`cmd/module4_link2`、`equil/module4_link2`までをlocal v2 graphだけで解決する。
指定された`viewer:=true` launchのnode graphも解決しており、残るGUIの視認確認にはX displayが必要である。

## 10. ProbTF v1 の完全廃止

次はsourceとmessage generationから削除されている。

- `probtf.models`
- `probtf.compatibility`
- `probtf_ros.conversions`
- `probtf_ros.legacy_conversions`
- `ProbabilisticTF.msg`
- `ProbabilisticTFArray.msg`
- `GaussianPosition.msg`
- `BinghamDistribution.msg`
- `ApproximationKind.LEGACY_ADAPTER`
- symaware `ProbTfTree` と旧manual propagation/sample scripts

v1からv2へrelayするruntime nodeも、v2からv1へprojectionするadapterもない。これは移行漏れではなく、
v1 contract自体を廃止した結果である。boundary testは旧module/symbol/message/CMake entryの再導入を検出する。

source内に残る `legacy` という語はsensor configやsymbolic URDFの入力互換を指す場合があるが、
ProbTF v1 transform domain/wireを意味しない。

## 11. Runtime boundary と運用上の規則

1. full physical SE(3) edgeだけを`ProbabilisticTransformStamped`としてgraphへ入れる。
2. orientation-only lawへfake translationを補わない。
3. stiffness、IK score、grasp IDなどapplication stateをSE(3) edgeへ偽装しない。
4. moment summaryやdisplay sampleをgraphへ再登録しない。
5. representative exportはexactまたは明示policyだけを許す。
6. application messageがedgeを指す場合、listenerのframe/stamp lookupでgraph recordを解決する。
7. static recordはlatched full set、dynamic recordは個別messageで運ぶ。
8. repeated latent dependencyを独立と仮定しない。
9. approximation/provenanceをcomponent、record、wireで保持する。
10. catkin nodeはpackage-local sourceをimportし、parent path relayを追加しない。

## 12. Architecture invariant とテスト境界

テストは数理 unit testだけでなく、architecture regressionを含む。

- foundationがROS/exampleへ逆依存しないこと
- package-local setupとparent relay不在
- ProbTF v1 module/symbol/messageの不在
- graph topology、time policy、bounded history、thread-safe lookup
- v2 message round-trip、TF import/export、local listener
- Bingham moments、joint coupling、right composition
- native finite/uniform/Dirac sampling、mixture比率、forward/inverse action
- two-IMU outputのstamp/edge/authority/provenance
- orientation-only messageにtranslationがないこと
- symaware static graph、app message、IK、visualizer、launch runtime
- deflecomp scoped bridge launchとpoint-moment consumer

具体的なpass数、build結果、runtime smokeの記録はimplementation reportで管理し、本書では
何をarchitecture invariantとして固定するかだけを記載する。

## 13. 現行実装の明示的な制約

### 13.1 数値backendのavailability

- finite Binghamのexact induced spherical/vector density evaluatorはunavailableである。
- rotation couplingを含むjoint numerical point-action integratorはunavailableである。
- stochastic inverseのanalytic moment/covariance evaluatorはunavailableである。
- repeated latent edgeは独立と仮定せず、shared latent evaluatorが必要なqueryを拒否する。
- closed-mixture reductionは明示policy/backendなしでは実行しない。

native Monte Carlo samplingはforward/inverse/composed pathまで実装済みであり、上記の
analytic/numerical law evaluatorとは別のrepresentationである。

### 13.2 Temporal semantics

`INTERPOLATE_WITH_MODEL`と`PREDICT_WITH_MODEL`はcontractだけがあり、具体的なprocess model、
uncertainty growth、diagnostic backendは存在しない。authority/parent changeもlocal graphの
明示policyを超える長時間運用の上位policyを持たない。

現在の`LATEST_COMMON`は意図的にzero-order holdであり、確率分布補間の代替ではない。

orientation-only posteriorをfull SE(3) graphへ入れること、stiffness posteriorをtransform化すること、
v1 adapterを復活させることは制約の解消ではなく、禁止しているdomain violationまたは廃止contractの再導入になる。

## 14. DOT の読み方

DOT graphは次の色分けを使う。

- 青: ProbTF foundation、graph、kernel、v2 transport
- 緑: producer algorithmsとtwo-IMU runtime
- 黄: orientation-only domain
- 桃: symaware grasp
- 紫: deflecomp
- 灰: external/runtime infrastructure
- 赤: 禁止境界。v1 gapではなくfake domain encodingを禁止する規則

太い実線は実行時dataflow、細い実線はcode dependency、破線はquery/terminal evaluation、
赤いT字は「このdataをSE(3) graphへ入れない」という境界を表す。
