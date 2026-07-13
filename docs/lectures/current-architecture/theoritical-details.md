# ProbTF-demo 現行アーキテクチャの理論的詳細

- 更新日時: 2026-07-13 JST
- 対象範囲: ProbTF v2 foundation、ROS 1 transport、producer demo、symaware grasp、deflecomp
- DOT source: [`implemented-graph.dot`](./implemented-graph.dot)
- rendered graph: [`implemented-graph.svg`](./implemented-graph.svg)

本書は将来案ではなく、repository に存在する source、message、launch、test に基づく
現行実装の理論・構造説明である。実装変更に合わせて同じdirectoryのgraphと同時に更新する。
移行作業の経緯と検証結果は
[`v2-demo-migration_2026-07-13.md`](../../reports/v2-demo-migration_2026-07-13.md)、
数理 contract は
[`probtf_jmaa_kernel_architecture.md`](../probtf_jmaa_kernel_architecture.md) を参照する。

## 0. 結論

現在の`ProbTF-demo`は、full SE(3) uncertainty、orientation-only law、application stateを
型とtransportの境界で分離するnative v2 architectureを採用する。

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

Python namespaceは、それを責務として持つcatkin packageの`src/`に属する。

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

`probtf` foundationは`probtf_estimators`、ROS、example applicationへ逆依存しない。
`probtf_ros` generic bridgeもestimator policyを所有しない。

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

listenerが所有するlocal graphのdynamic historyはboundedでなければならない。

### 3.2 Temporal policy

| policy | semantics |
| --- | --- |
| `EXACT` | 指定 stamp と完全一致する sample |
| `NEAREST_WITHIN_TOLERANCE` | tolerance 内最近傍。tie は古い sample |
| `LATEST` | 指定時刻以前の最新。任意 `max_age` |
| `LATEST_COMMON` | path 全体の最新共通時刻と zero-order hold |
| `INTERPOLATE_WITH_MODEL` | 明示されたinterpolation modelを使う |
| `PREDICT_WITH_MODEL` | 明示されたprocess modelを使う |

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

### 4.3 Sampling representation

`SAMPLES` representationはnative joint lawからmixture component、orientation、conditional translationを
同じrealizationとして標本化する。path上で同じlatent edgeが反復される場合は同じrealizationを共有し、
独立な再標本化を行わない。stochastic sample resultは`MONTE_CARLO`として型付けし、表示sampleを
physical edgeへ再登録しない。

## 5. ROS v2 wire contract

### 5.1 `probtf_msgs`

core wire contractは次のmessageで構成される。

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

## 6. two-IMU relative-pose architecture

two-IMU producerのdataflowは次である。

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

二つのsensor observationから得る相対姿勢はfull SE(3) edgeであり、回転と並進のcouplingを
同じjoint lawに保持する。symbolic URDFはそのlawを置き換えるgraph edgeではなく、明示的な
判定を通ったterminal materializationである。

## 7. orientation-only architecture

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

`TransformEvidenceStamped` は orientation natural parameter を trace-zero symmetric upper 10要素で
保持する。任意 translation evidence は singular PSD を許す information formであり、sequence、
approximation、provenanceを持つ。

posterior用 `OrientationDistributionStamped` には translation fieldが存在しない。predictionと
independent likelihoodの近似種別・provenanceを保持し、同一sourceの二重計上を許さない。

orientation-only lawへ zero translationを補って `/probtf` physical edgeにする処理はない。これは
未接続のgapではなく、fake SE(3)を防ぐdomain boundaryである。

## 8. Symaware grasp architecture

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

### 8.2 Composition とterminal query

object lawへdeterministic grasp offsetを右合成するとき、mixture、conditional translation、coupling、
派生元provenanceを保持する。IKとvisualizationはlocal listenerでgraph recordを解決し、point moments、
sample、representativeをterminal resultとしてだけ利用する。これらのsummaryを独立edgeとして
graphへ戻さない。

## 9. Deflecomp scoped architecture

### 9.1 ProbTFへ登録するもの、しないもの

deflecompのWEKFが推定するstiffness `Kp` posteriorはjoint parameter distributionであり、SE(3) edgeではない。
したがってstiffnessのestimate、control target、covarianceはdeflecomp domainに留まり、ProbTF graphへ
登録しない。

ProbTFへ登録するのは、`robot_state_publisher` と static TF publisher が実際に出す `ref`、`cmd`、
`equil` frame transformだけである。

### 9.2 Scoped TF import

Deflecompはgeneric bridgeをscoped namespace内で使用する。

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

consumerは`/tf`を直接queryせず、scoped topicsから構築したlocal `RosProbTfListener` graphへ
`LATEST_COMMON` lookupを行う。point mean/covarianceとその可視化はterminal queryであり、
stiffness posteriorやmoment summaryをSE(3) edgeへ変換しない。

## 10. Native v2 only boundary

ProbTF transform domainとwireはnative v2だけであり、v1 compatibility pathを持たない。
orientation-only law、application state、terminal summaryをv1的なpartial transformへ投影する経路も設けない。

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

## 12. DOT の読み方

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
