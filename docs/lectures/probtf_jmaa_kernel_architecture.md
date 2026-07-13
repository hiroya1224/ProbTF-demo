# ProbTF JMAA kernelの理論・実装contract

## 1. 定義と責務

Prob-TF の一次的な対象は、一般的な `SE(3)` Gaussian ではなく、physical
edge ごとの Bingham orientation と conditional Gaussian translation の joint
component mixture である。component `l` は次で定義される。

\[
Q\mid L=l\sim \operatorname{Bing}(A_l),
\]

\[
X\mid Q=q,L=l\sim \mathcal N\left(
m_l+C_l(\operatorname{vec}R(q)-\operatorname{vec}R_{\mathrm{ref},l}),S_l
\right).
\]

child frame の点 `z` を parent frame へ写す forward action は

\[
Y=R(Q)z+X
\]

である。この積分を `probtf.spherical_law` と `probtf.kernels` の backend
contract とする。moment summary は optional evaluator であり、元の joint law
ではない。

## 2. package 境界

依存方向は次で固定する。

```text
probtf_estimators  --->  probtf
probtf_ros         --->  probtf
examples           --->  probtf, probtf_estimators, probtf_ros

probtf             -X->  probtf_estimators
probtf             -X->  ROS
probtf             -X->  examples
```

主要 directory の責務は次の通りである。

| directory | responsibility |
| --- | --- |
| `ros/core/probtf_core/src/probtf/distributions` | immutable orientation/translation/component/mixture storage |
| `ros/core/probtf_core/src/probtf/graph` | physical forest、path、timestamped buffer、query |
| `ros/core/probtf_core/src/probtf/kernels` | forward/inverse/composed/mixture expression と evaluator |
| `ros/core/probtf_core/src/probtf/spherical_law` | ISL protocol、special case、tangent surrogate adapter |
| `ros/core/probtf_core/src/probtf/probability` | optional Bingham/rotation/point moment summary |
| `ros/core/probtf_core/src/probtf/provenance` | provenance、dependency、approximation metadata |
| `ros/core/probtf_core/src/probtf_estimators` | IMU producer、same-edge evidence fusion、Hessian coupling |
| `ros/core/probtf_msgs` | native v2 wire contract |
| `ros/core/probtf_core/src/probtf_ros` | ROS conversion と bridge のみ |

`tests/test_ros_boundary.py` はこの依存方向、各catkin packageによるPython source ownership、
parent path relayの不在を検査する。

## 3. frame と traversal

physical edge `(parent, child)` は常に

\[
z_{parent}=R(Q)z_{child}+X
\]

を表す。`EdgeDirection.FORWARD` は child から parent への action、`INVERSE`
は同じ latent edge の parent から child への view である。inverse 用の別
distribution は作らない。

public query は tf2 と同じ action order を使う。

```python
graph.lookup_kernel(
    target_frame="world",
    source_frame="tool0",
    stamp=stamp,
)
```

返り値は source から target へ順に適用する `ComposedTransformKernel` である。
lookup 時点で sampling、moment closure、mixture reduction は行わない。

## 4. quaternion、matrix、scale

内部規約は次で固定する。

- quaternion: `w, x, y, z`
- ROS quaternion field: `x, y, z, w`。変換は `probtf_ros` のみ
- `vec(R)`: column-major
- coupling `C`: `3 x 9`、wire 上は row-major flat
- local coupling: right perturbation `R_ref exp([u]x)`
- covariance upper: `xx, xy, xz, yy, yz, zz`
- Bingham upper: `ww, wx, wy, wz, xx, xy, xz, yy, yz, zz`

Bingham storage は trace-zero parameter `A` の固有値を
`lambda1 >= lambda2 >= lambda3 >= lambda4` と並べ、JMAA manuscript Section
6.1 の定義

\[
\kappa_A=\lambda_1+\lambda_2,
\quad \check A=A/\kappa_A,
\quad \text{inverse_concentration}=1/\kappa_A
\]

を使う。Frobenius normalization は使わない。finite backend の直前だけ
max-eigenvalue-zero gauge へ identity shift する。

orientation kind の storage は次の通りである。

| kind | inverse concentration | shape |
| --- | --- | --- |
| `FINITE_BINGHAM` | positive finite | trace zero、JMAA magnitude 1 |
| `DIRAC` | `0` | normalized `q q.T - I/4` |
| `UNIFORM` | `+inf` | zero matrix |

`DIRAC` を finite Bingham normalizer に渡さない。`UNIFORM` の `+inf` は
`1/kappa` の数学的な値であり、この kind に限って明示的に許可する。

## 5. coupling

`ConditionalGaussianTranslation.mean_at_reference` は marginal mean ではない。
`TransformComponent.conditional_translation_mean(q)` が orientation 内の
reference quaternion を binding し、

\[
m+C(\operatorname{vec}R(q)-\operatorname{vec}R_{ref})
\]

を評価する。translation object 単体で coupling を評価する場合は reference
quaternion を明示的に渡す。

producer 用 `probtf_estimators.coupling_from_hessian` は

\[
B=-H_{xx}^{-1}H_{xu},\quad C=BD^+,\quad D^+=D^T/2
\]

を right perturbation convention で実装する。singular `Hxx` に jitter を
追加せず、明示的に失敗する。

## 6. mixture と status

component は orientation と translation を一つの joint hypothesis として持つ。
raw weight `a_l` は保存時に書き換えず、利用時に

\[
a_l^+=\max(a_l,0),\qquad w_l=a_l^+/\sum_j a_j^+
\]

とする。負値は `NEGATIVE_WEIGHT_CLAMPED` diagnostic を残す。全て非正なら
`ZERO_MASS`、NaN/Inf weight は `INVALID` である。どちらも identity、zero
Dirac、uniform へ変換しない。

representative は `RepresentativePolicy` が要求された場合だけ作る。
highest-weight component の mode は `COMPONENT_MODE_APPROXIMATION` であり、
global MAP とは呼ばない。

## 7. graph と time

`ProbTfTopology` は disconnected component を許す forest であり、cycle と
multiple parent を拒否する。parent change は default で拒否し、明示 policy
`REPLACE_WITH_DIAGNOSTIC` の場合だけ replacement diagnostic を残す。

`EdgeTimeBuffer` は out-of-order insert を timestamp 順に保持する。同一 stamp
かつ同一 authority は replacement、異なる authority は default で
`AUTHORITY_CONFLICT` である。

実装済み policy:

- `EXACT`
- `NEAREST_WITHIN_TOLERANCE`。tie は古い sample
- `LATEST`。指定時刻以前の最新 sample
- `LATEST_COMMON`

`LATEST` と `LATEST_COMMON` は optional `max_age` を受け取り、許容 age を
超えた zero-order hold は `TEMPORAL_STALE` で失敗する。

任意 Bingham mixture interpolation は未実装である。現在の
`LATEST_COMMON` は全 dynamic edge の availability interval の最新共通時刻を
選び、各 edge でその時刻以前の最新 sample を使用する。sample が共通時刻と
異なる場合、`LATEST_COMMON_ZERO_ORDER_HOLD` diagnostic が
`PathExpression` に残る。これは exact temporal interpolation ではない。

`INTERPOLATE_WITH_MODEL` と `PREDICT_WITH_MODEL` は interface のみで、呼ぶと
`UNSUPPORTED_TEMPORAL_POLICY` になる。static uncertain edge は任意 query
時刻で同じ record を返す。

## 8. kernel evaluation

`KernelRepresentation` は次を型で区別する。

- `EXPRESSION`: lazy action
- `NUMERICAL_LAW`: backend law
- `SAMPLES`: samples
- `MOMENTS`: first/second moment summary
- `CLOSED_MIXTURE`: explicit closure result

deterministic edge path は ISL backend を呼ばず、通常の rigid transform として
forward、inverse、composition を exact に評価する。

forward component の moment evaluator は coupling を保持したまま

\[
H=A_z+C,
\quad E[Y]=m-Cr_{ref}+H E[\operatorname{vec}R]
\]

と `E[vec(R) vec(R).T]` を使って exact first/second moments を計算する。
`E[R kron R]` と `E[vec(R) vec(R).T]` は異なる index arrangement なので、
後者専用 helper を使用する。返り値は `MOMENT_SUMMARY`、`lossy=True` であり、
元の law へ再登録しない。

同じ latent dependency が path に繰り返される場合は
`DEPENDENCY_UNRESOLVED` である。独立 sample として扱わない。

## 9. ISL backend status

`probtf.isl` (`probtf.spherical_law` の short alias) にある
`IslInducedLawBackend.rotate_direction` と `rotate_vector` が primary backend
contract である。zero vector は orientation に関係なく zero Dirac、nonzero
vector は norm と unit direction に分ける。

現在の status:

| operation | status |
| --- | --- |
| Dirac orientation action | exact |
| uniform orientation direction/vector action | exact special case |
| finite-Bingham exact ISL density | `UNAVAILABLE_EXACT_ISL_BACKEND` |
| numerical joint coupling/Gaussian convolution | unavailable |
| tangent leading-exponent moments | explicit `TANGENT_SURROGATE` approximation |
| forward point moments | exact moments、lossy summary representation |
| stochastic inverse covariance | unavailable |
| native stochastic sampling | forward/inverse/composed pathまで実装済み、`MONTE_CARLO` |
| closed-mixture reduction | unavailable without policy |

JMAA manuscript には exact global density があるが、migration 開始時点で source
evaluator と regression code が存在しなかった。この実装では式を新規転記せず、
`UnavailableExactIslBackend` を返す。代替として Wang 型 `SE(3)` propagation
は実装していない。

## 10. ROS v2 と TF bridge

`probtf_msgs` はnative v2だけを生成する。full physical SE(3) edgeのmessageは次で構成される。

- `BinghamOrientation`
- `ConditionalGaussianTranslation`
- `ProbabilisticTransformComponent`
- `ProbabilisticTransformStamped`
- `ProbabilisticTransformArray`
- `ApproximationInfo`
- `Provenance`

これに加え、two-IMU境界の`ImuKinematics`、orientation-only likelihoodの
`TransformEvidenceStamped`、translationを持たないposteriorの
`OrientationDistributionStamped`を生成する。

`ProbabilisticTransformStamped.header.frame_id`が唯一のparent frameである。componentはraw
weight、joint orientation/translation、provenance、approximationを保持する。

`ProbTfBroadcaster`と`RosProbTfListener`は`/probtf`と`/probtf_static`用のtransport facadeである。
late subscriberが全static edgeを受け取れるよう、`/probtf_static`は全static setの
`ProbabilisticTransformArray`をlatchして再送する。listenerはtopicごとにlocal `ProbTfGraph`を作り、
中央RPCなしでlookupする。

通常TF importはzero covariance/couplingのDirac componentを作る。ProbTFからTFへのexportは
exact edgeまたは明示されたrepresentativeだけで、stochastic edgeではexplicit `TfExportPolicy`を
要求する。bridgeはown authorityとexport signatureを使ってimmediate re-import loopを防ぐ。
IMU、orientation、fusion producer nodeは各demo packageが所有する。

## 11. v1廃止とdomain boundary

v1 domain model、ROS message、v1/v2 adapter、symaware固有`ProbTfTree`はsourceとruntimeから
削除されている。v1へ投影するcompatibility pathは存在しない。

- orientation-only posteriorへzero translationを補い、SE(3) edgeへ偽装しない。
- stiffness posterior、IK score、grasp IDをtransformとしてgraphへ登録しない。
- moment summary、display sample、representativeを独立なphysical edgeとして再登録しない。
- mixture/couplingを暗黙に一成分へ縮約しない。
- 同じlatent dependencyを独立sampleとして扱わない。

これらはbackend不足を埋める暫定compatibilityではなく、確率変数とapplication domainを守る境界である。

## 12. install ownershipと明示的制約

`probtf`、`probtf_estimators`、`probtf_ros`は`probtf_core` catkin packageの`src/`が所有する。
deflecompとsymawareのPython namespaceも各catkin packageの`src/`が所有する。root `setup.py`は
非catkin開発向けにfirst-party source rootだけを集約し、catkin packageからparent pathをrelayしない。
Bingham normalizerは`probtf._vendor`に収容され、runtimeは外部`bingham` packageに依存しない。

次の項目はinterfaceまたはexplicit unavailable statusまでである。

- finite-Bingham exact JMAA density evaluatorとregression values
- arbitrary mixture temporal interpolation/prediction
- stochastic inverse analytic high-order moment evaluator
- coupled numerical lawとdependency-aware shared realization evaluator
- mixture mergeとBingham-Gaussian closure policy
- general edge-correlation/factor-graph backend
