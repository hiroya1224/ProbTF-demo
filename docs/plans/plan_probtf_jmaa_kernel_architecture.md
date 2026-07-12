# Prob-TF の ISL-kernel アーキテクチャ実装計画

## 0. この計画の目的

本計画では，現在の `ProbTF-demo` を，従来の ROS TF / TF2 の broadcaster–listener，frame tree，timestamped buffer，path lookup という役割を保ちながら，**ISL で構築した「Bingham 分布に従う quaternion によってベクトルを回転したときの induced law」**を基本演算とする Prob-TF へ再編する．

最重要方針は次の通りである．

> Prob-TF を「Wang 型の \(SE(3)\) 上の分布伝播」や「一般的な \(SE(3)\) covariance propagation」の実装として定義しない．  
> 各 edge は Bingham 回転と，その回転に条件づけられた Gaussian 並進からなる確率的座標変換として保持し，点・方向への作用は ISL の induced law を用いる確率 kernel として定義する．

moment は利用してよいが，Prob-TF の一次的な定義にはしない．moment は Bingham closure，summary query，mixture reduction，高集中極限の比較，および計算高速化のための補助表現とする．

本計画は，以下を前提資料とする．なお，本 plan は self-contained に書かれているので，以下の情報は必ずしも必要としない．

- `ProbTF-demo/docs/lectures/current-implementation_2026-07-12_172230.md`
- `ProbTF-demo/docs/additional_infos/vec_R_coupling_lecture_note_ja.md`
- `ProbTF-demo/docs/additional_infos/TePRA2013_Foote.pdf`
- 現在の `ProbTF-demo` リポジトリ
- JMAA 投稿原稿(`ProbTF-demo/docs/additional_infos/jmaa_manuscript.pdf`)および同原稿に対応する既存コード
  - ISL とは，induced spherical law のことである．重複した名前になっており，相応しくない場合は，その暫定分布名（Sato--Okada distribution）から略称を作って良い．

なお，文中の Wang とは，以下の文献を指すが，これに基づく実装を禁じるため，参考にする必要はない．
- `ProbTF-demo/docs/additional_infos/Wang06.pdf`

---

## 1. Codex が守るべき最重要制約

### 1.1 採用する立場

実装上の基本対象は，mixture component ごとの

\[
Q \mid L=\ell
\sim
\operatorname{Bing}(A_\ell)
\]

および

\[
X \mid Q=q,\ L=\ell
\sim
\mathcal{N}
\left(
m_\ell+
C_\ell
\left(
\operatorname{vec}R(q)-\operatorname{vec}R_\ell
\right),
S_\ell
\right)
\]

である．

点 \(z\in\mathbb{R}^3\) に対する forward action は

\[
Y=R(Q)z+X
\]

である．したがって，第 \(\ell\) component では

\[
Y\mid Q=q,L=\ell
\sim
\mathcal{N}
\left(
R(q)z+
m_\ell+
C_\ell
\left(
\operatorname{vec}R(q)-\operatorname{vec}R_\ell
\right),
S_\ell
\right)
\]

となる．この Bingham–Gaussian 積分を Prob-TF の基本 kernel とする．

### 1.2 厳格な禁止事項

以下は**決して**行わないこと．

1. Prob-TF の中心クラスを `SE3Mixture`，`SE3Gaussian`，`SE3Posterior` などと命名しない．
2. Wang の \(SE(3)\) propagation を Prob-TF の基礎定義として実装しない．
3. tangent-space の \(6\times6\) Gaussian covariance を wire format や内部表現の唯一の真実にしない．
4. position と orientation を独立と仮定して coupling を捨てない．
5. path lookup のたびに，結果を無言で単一 Bingham + Gaussian に潰さない．
6. inverse transform を forward edge とは独立な確率変数として生成しない．
7. exact ISL density の式がリポジトリ内に見つからない場合，Codex が式を推測して実装しない．
8. ISL backend が未実装だからという理由で，Wang 型 propagation へ置き換えない．
9. mixture component 数の削減を，明示的な policy と approximation metadata なしに行わない．
10. `src/probtf` から ROS package や個別 estimator を import しない．

### 1.3 ISL 実装が見つからない場合

作業開始時に，リポジトリ全体から以下を検索すること．
(補足： `ProbTF-demo/src/symaware_grasp/prob_tf/tangent_surrogate.py` 内部に使えそうな実装がある)

- induced spherical density
- Bingham で回転された unit vector の density
- exact law
- leading-exponent surrogate
- tangent approximation
- point / direction push-forward
- `R(q) v`
- JMAA 原稿に対応する関数・script

利用可能な実装が見つかった場合は，そのコードを `probtf.isl` backend から呼べるように整理する．

利用可能な実装が見つからない場合は，

- backend protocol
- lazy kernel expression
- validation
- deterministic path
- numerical integration backend を挿し込める interface

までを実装し，exact ISL evaluator は明示的に `NotImplementedError` または `UNAVAILABLE_BACKEND` を返すこと．誤った近似式を埋めてはならない．

---

## 2. 今回の実装範囲

### 2.1 必須成果物

今回の作業では，少なくとも以下を完成させる．

1. ROS 非依存の Prob-TF domain model
2. Bingham–conditional-Gaussian mixture component
3. weight normalization と validation
4. trace-zero Bingham shape + inverse concentration 表現
5. \(\operatorname{vec}R\) coupling の保持と評価
6. physical edge / inverse view / path expression
7. tree / forest の topology 管理
8. timestamped edge buffer
9. exact / nearest / latest-common の時刻 query
10. point action kernel の interface
11. forward / inverse / composed kernel の lazy expression
12. deterministic TF を Dirac component として扱う fast path
13. optional moment summary backend
14. ROS message v2 と conversion
15. producer / core / ROS bridge の一方向依存
16. 旧実装からの compatibility adapter
17. unit test と architecture document

各作業ごとに，commit を作成すること．

### 2.2 今回は完了を要求しないもの

以下は interface と TODO を用意してよいが，完全実装を必須としない．

- 任意の Bingham mixture に対する exact temporal interpolation
- 任意の path composition を単一の closed-form Bingham–Gaussian component に戻す処理
- 高精度な mixture merge
- edge 間相関を含む一般 factor-graph backend
- cross-time joint distribution
- 全ての ISL asymptotic surrogate
- AR marker producer 本体
- Wang との数値比較
- ROS 2 への移植
- RViz plugin

ただし，将来追加できない設計にしてはならない．

---

## 3. 数学的 contract

## 3.1 frame と transform の向き

physical edge \(e=(p,c)\) は，child frame \(c\) の座標を parent frame \(p\) に写す変換を表す．

\[
z_p=R_e z_c+X_e.
\]

既存の `ProbabilisticTransform` と通常の TF の規約を維持する．

全ての docstring，message comment，test でこの向きを明記すること．

## 3.2 mixture component

離散変数 \(L\in\{1,\ldots,K\}\) に対し，

\[
\Pr(L=\ell)=w_\ell
\]

とする．各 component は一つの joint pose hypothesis であり，orientation array と translation array を別々に持って後から index 対応させる設計にはしない．

標準 component は以下を持つ．

- `component_id`
- raw `weight`
- Bingham orientation
- conditional Gaussian translation
- provenance
- approximation metadata

## 3.3 weight の利用時正規化

raw weight を \(a_\ell\) とし，

\[
a_\ell^+=\max(a_\ell,0),
\qquad
Z=\sum_\ell a_\ell^+
\]

とする．

- \(Z>0\) のとき \(w_\ell=a_\ell^+/Z\)
- \(Z=0\) のとき `ZERO_MASS`
- 負値は 0 に clamp し diagnostic を残す
- NaN / Inf は component invalid
- publisher は総和 1 を目指すが，consumer は利用時に必ず正規化する

`ZERO_MASS` を identity transform，原点 Dirac，uniform distribution のいずれにも解釈しないこと．

## 3.4 Bingham parameter 規約

storage / transport では

\[
A=\kappa_A\check A
\]

とし，以下を保持する．

- trace-zero の normalized shape \(\check A\)
- `inverse_concentration = 1 / kappa_A`

数値 backend へ渡す直前に限り，最大固有値を 0 にする identity shift を行ってよい．storage 自体を最大固有値 0 gauge にしない．

\(\check A\) の normalization は，JMAA 原稿で採用した定義に厳密に従うこと．Codex が Frobenius norm 等を独自に選んではならない．既存 helper がある場合は再利用し，なければ normalization を一箇所に隔離し，仕様未確定であることを明示する．

### orientation kind

少なくとも次を区別する．

- `FINITE_BINGHAM`
- `DIRAC`
- `UNIFORM`

`inverse_concentration = 0` は `DIRAC` のみで許可する．有限 Bingham normalizer に 0 を渡して極限として処理しない．

Dirac mode \(q\) に対応する shape を必要とする場合は，

\[
q q^\top-\frac{1}{4}I_4
\]

を出発点とし，ISL shape normalization を適用する helper を用いる．単なる \(q q^\top\) を trace-zero matrix として保存しない．

## 3.5 quaternion と vectorization の規約

内部規約を以下に固定する．

- quaternion basis: `[w, x, y, z]`
- \(\operatorname{vec}R\): column-major
- local coupling derivation: right perturbation
- covariance upper triangle: `xx, xy, xz, yy, yz, zz`
- symmetric \(4\times4\) upper triangle:
  `ww, wx, wy, wz, xx, xy, xz, yy, yz, zz`
- \(C\in\mathbb{R}^{3\times9}\): row-major flat array

ROS の quaternion field order `x, y, z, w` との変換は `probtf_ros` のみに置く．

## 3.6 coupling

各 component の translation は

\[
X=
m+
C\left(\rho(Q)-\rho_\mathrm{ref}\right)+\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,S)
\]

とする．ここで

\[
\rho(q)=\operatorname{vec}R(q).
\]

field の意味は以下とする．

- `mean_at_reference`: \(m\)
- `reference_quaternion`: \(q_\mathrm{ref}\)
- `residual_covariance`: \(S\)
- `rotation_coupling`: \(C\)

`mean_at_reference` は周辺平均ではない．

局所 Hessian から生成する場合は，

\[
B=-H_{xx}^{-1}H_{xu},
\qquad
C=BD^+
\]

を別 helper として実装する．ただし，この生成 helper は producer / estimator 側に置いてよい．core が Hessian の意味を知る必要はない．

core validator は次を検査する．

- shape
- finite value
- covariance symmetry
- covariance PSD
- `C.shape == (3, 9)`
- reference quaternion normalization
- orientation kind と scale の整合性

## 3.7 point action kernel

一つの edge component に対し，入力点 \(z\) の出力 law は

\[
Y=R(Q)z+X
\]

である．

forward kernel は，条件付きで

\[
Y\mid Q=q
\sim
\mathcal N
\left(
R(q)z+
m+
C\left(\rho(q)-\rho_\mathrm{ref}\right),
S
\right)
\]

を評価する．

mixture 全体は component kernel の重み付き和とする．

## 3.8 inverse action

forward edge が

\[
z_p=R(Q)z_c+X
\]

なら inverse view は

\[
z_c=R(Q)^\top(z_p-X)
\]

である．

\(Q=q\) に条件づけると，

\[
z_c\mid Q=q
\]

は mean

\[
R(q)^\top
\left[
z_p-
m-
C\left(\rho(q)-\rho_\mathrm{ref}\right)
\right]
\]

と covariance

\[
R(q)^\top S R(q)
\]

を持つ Gaussian になる．

一般に inverse component は元と同じ標準 family の parameter へ単純には戻らない．したがって inverse は `InverseTransformKernel` という lazy expression として保持し，無理に `TransformComponent` へ閉じないこと．

## 3.9 path composition

path

\[
P=(e_1^{\sigma_1},\ldots,e_n^{\sigma_n})
\]

に対して，kernel を

\[
\mathcal K_P
=
\mathcal K_{e_n^{\sigma_n}}
\circ\cdots\circ
\mathcal K_{e_1^{\sigma_1}}
\]

とする．

lookup の基本結果は closed distribution ではなく，

- `PathExpression`
- `ComposedTransformKernel`

である．

必要な consumer のみが，

- exact / numerical distribution
- samples
- moments
- Bingham–Gaussian closure

を要求する．

## 3.10 moment の位置づけ

moment は optional evaluator とする．

許される用途:

- `E[R]`
- `E[R kron R]`
- quaternion product の second moment
- point mean / covariance
- Bingham moment matching
- mixture reduction
- debug / regression test
- Wang や局所 Gaussian 極限との比較

禁止する用途:

- moment summary を original joint law と同一視する
- closure 後の summary を provenance なしに independent edge として再投入する
- moment backend しかないことを理由に Prob-TF の定義を moment propagation へ変更する

---

## 4. 目標ディレクトリ構成

最終的に次へ近づける．一度に全 rename を行う必要はないが，依存方向はこの形にする．

```text
ProbTF-demo/
  src/
    probtf/
      distributions/
        bingham_orientation.py
        conditional_translation.py
        transform_component.py
        transform_distribution.py
        validation.py

      kernels/
        base.py
        forward.py
        inverse.py
        composed.py
        mixture.py
        evaluation.py

      spherical_law/
        protocol.py
        induced_direction.py
        induced_vector.py
        numerical.py
        adapters.py

      graph/
        edge.py
        path.py
        topology.py
        buffer.py
        query.py
        status.py

      temporal/
        policy.py
        model.py

      probability/
        bingham.py
        bingham_matching.py
        moments.py
        gaussian.py

      provenance/
        model.py
        dependency.py

      geometry/
        quaternion.py
        rotation.py
        vectorization.py

    probtf_estimators/
      imu_preprocessing.py
      imu_relative_pose.py
      orientation_imu.py
      coupling_from_hessian.py

    symaware_grasp/
    deflecomp_core/

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

### 4.1 dependency rule

依存方向を以下に固定する．

```text
probtf_estimators  --->  probtf
probtf_ros         --->  probtf
examples           --->  probtf, probtf_estimators, probtf_ros

probtf             -X->  probtf_estimators
probtf             -X->  ROS
probtf             -X->  examples
```

`tests/test_ros_boundary.py` を維持・拡張する．

---

## 5. Python domain model

実装には `dataclass(frozen=True)` または同等の immutable object を優先する．NumPy array は constructor 内で copy し，外部 mutation によって validation が破られないようにする．

## 5.1 enum / status

以下を用意する．

```python
class OrientationKind(Enum):
    FINITE_BINGHAM = "finite_bingham"
    DIRAC = "dirac"
    UNIFORM = "uniform"


class DistributionStatus(Enum):
    OK = "ok"
    ZERO_MASS = "zero_mass"
    INVALID = "invalid"


class KernelRepresentation(Enum):
    EXPRESSION = "expression"
    NUMERICAL_LAW = "numerical_law"
    SAMPLES = "samples"
    MOMENTS = "moments"
    CLOSED_MIXTURE = "closed_mixture"
```

上記に加えて，error class を追加する．

## 5.2 `BinghamOrientation`

最低限の field:

```python
@dataclass(frozen=True)
class BinghamOrientation:
    kind: OrientationKind
    inverse_concentration: float
    shape_matrix: np.ndarray
    reference_quaternion_wxyz: np.ndarray
```

責務:

- trace-zero validation
- symmetry validation
- ISL shape normalization validation
- quaternion normalization
- finite Bingham / Dirac / uniform の整合性
- numerical backend 用 parameter matrix 生成
- mode の取得
- ROS serialization helper から独立

`parameter_matrix` という曖昧な field 名だけにせず，storage が shape であることを class 名と docstring で明示する．

## 5.3 `ConditionalGaussianTranslation`

```python
@dataclass(frozen=True)
class ConditionalGaussianTranslation:
    mean_at_reference: np.ndarray
    residual_covariance: np.ndarray
    rotation_coupling: np.ndarray
```

責務:

```python
def conditional_mean(self, quat_wxyz: np.ndarray) -> np.ndarray:
    ...
```

```python
def conditional_covariance(self, quat_wxyz: np.ndarray) -> np.ndarray:
    ...
```

現行モデルでは conditional covariance は \(S\) で一定なので，後者は copy を返せばよい．将来の拡張を考え interface は残してよい．

## 5.4 `TransformComponent`

```python
@dataclass(frozen=True)
class TransformComponent:
    component_id: str
    raw_weight: float
    orientation: BinghamOrientation
    translation: ConditionalGaussianTranslation
    provenance: ComponentProvenance
    approximation: ApproximationInfo
```

weight の normalized value を object 自体に二重保持しない．`TransformDistribution.normalized_components()` で算出する．

## 5.5 `TransformDistribution`

```python
@dataclass(frozen=True)
class TransformDistribution:
    components: tuple[TransformComponent, ...]
```

提供 API:

```python
def normalize_weights(self) -> NormalizedTransformDistribution:
    ...
```

```python
def status(self) -> DistributionStatus:
    ...
```

```python
def representative(self, policy: RepresentativePolicy):
    ...
```

`representative()` は本体ではなく lossy projection である．

## 5.6 stamped edge record

```python
@dataclass(frozen=True)
class TransformDistributionStamped:
    parent_frame_id: str
    child_frame_id: str
    stamp: float
    edge_id: str
    authority: str
    distribution: TransformDistribution
    representative: Optional[DeterministicTransform]
    provenance: TransformProvenance
```

`edge_id` は physical latent edge を表す stable ID とする．

---

## 6. Kernel API

## 6.1 base protocol

```python
class TransformKernel(Protocol):
    def apply(
        self,
        input_law: PointLaw,
        options: KernelEvaluationOptions,
    ) -> KernelResult:
        ...
```

ただし，初期実装では expression 構築と evaluator を分けてもよい．

推奨:

```python
class TransformKernelExpression(ABC):
    pass


@dataclass(frozen=True)
class ForwardEdgeKernel(TransformKernelExpression):
    edge_record: TransformDistributionStamped


@dataclass(frozen=True)
class InverseEdgeKernel(TransformKernelExpression):
    edge_record: TransformDistributionStamped


@dataclass(frozen=True)
class ComposedTransformKernel(TransformKernelExpression):
    kernels: tuple[TransformKernelExpression, ...]
```

## 6.2 input point law

最初は以下を用意する．

```python
class PointLaw(ABC):
    pass


@dataclass(frozen=True)
class DiracPointLaw(PointLaw):
    point: np.ndarray


@dataclass(frozen=True)
class GaussianPointLaw(PointLaw):
    mean: np.ndarray
    covariance: np.ndarray
```

任意 law は将来の protocol へ拡張する．

## 6.3 output

```python
@dataclass(frozen=True)
class KernelResult:
    status: DistributionStatus
    representation: KernelRepresentation
    value: object
    approximation: ApproximationInfo
    diagnostics: KernelDiagnostics
```

exact law，numerical law，moments，samples を同じ object と誤認しない設計にする．

## 6.4 ISL backend protocol

```python
class IslInducedLawBackend(Protocol):
    def rotate_direction(
        self,
        orientation: BinghamOrientation,
        direction: np.ndarray,
        options: IslEvaluationOptions,
    ) -> InducedDirectionLaw:
        ...

    def rotate_vector(
        self,
        orientation: BinghamOrientation,
        vector: np.ndarray,
        options: IslEvaluationOptions,
    ) -> InducedVectorLaw:
        ...
```

`rotate_vector` は \(v=0\) を Dirac at zero として扱う．非零なら norm と unit direction に分け，ISL の unit-vector induced law を再利用する．

coupling と residual Gaussian convolution は kernel evaluator 側で行う．

## 6.5 lazy composition

path lookup の時点では積分・sampling・moment closure を行わない．

```python
kernel = buffer.lookup_kernel(
    target_frame="world",
    source_frame="tool0",
    stamp=stamp,
)
```

consumer が評価を要求する．

```python
result = evaluator.apply_to_point(
    kernel=kernel,
    point=np.array([0.0, 0.0, 0.1]),
    representation=KernelRepresentation.MOMENTS,
)
```

これにより，graph / time resolution と確率計算を分離する．

---

## 7. graph と path lookup

## 7.1 `PhysicalEdge`

physical edge と traversal direction を分ける．

```python
@dataclass(frozen=True)
class PhysicalEdge:
    edge_id: str
    parent_frame_id: str
    child_frame_id: str


@dataclass(frozen=True)
class EdgeView:
    edge_id: str
    direction: EdgeDirection
    sample_stamp: float
```

`EdgeDirection` は `FORWARD` / `INVERSE` とする．

inverse view 用に新しい distribution を作らない．

## 7.2 topology

初期版は TF と同様に forest とする．

- child は高々一つの parent
- cycle 禁止
- disconnected component 許可
- parent change を timestamped topology として扱うかは，現行 TF の behavior を調べた上で最小実装を決める
- 不明確な場合，初期版では topology change を edge replacement として扱い，診断を出す

## 7.3 `PathExpression`

既存 `symaware_grasp.prob_tf.PathExpression` を調査し，概念を吸い上げる．単純 copy ではなく，現在の domain model に合わせて再実装する．

```python
@dataclass(frozen=True)
class PathExpression:
    source_frame: str
    target_frame: str
    resolved_stamp: float
    edge_views: tuple[EdgeView, ...]
```

## 7.4 repeated latent edge

同じ `edge_id` が path expression 内に複数回現れる場合，独立 sampling してはならない．

初期版では次のいずれかとする．

- dependency-aware evaluator が同じ latent sample を共有する
- 未対応なら `DEPENDENCY_UNRESOLVED` を返す

黙って独立と仮定しないこと．

---

## 8. timestamped buffer

## 8.1 edge history

各 physical edge ごとに，timestamp 順の record を保持する．

```python
class EdgeTimeBuffer:
    def insert(self, record: TransformDistributionStamped) -> None:
        ...

    def resolve(
        self,
        stamp: float,
        policy: TemporalPolicy,
    ) -> ResolvedEdgeRecord:
        ...
```

out-of-order insert を許可する．同一 timestamp / authority conflict の policy を明示する．

## 8.2 初期 temporal policy

以下を実装する．

- `EXACT`
- `NEAREST_WITHIN_TOLERANCE`
- `LATEST`
- `LATEST_COMMON`

`INTERPOLATE_WITH_MODEL` と `PREDICT_WITH_MODEL` は interface のみでよい．

通常の deterministic edge は将来 tf2 相当 interpolation を追加できる．Bingham mixture に deterministic SLERP を無条件適用してはならない．

## 8.3 latest common time

path 上の全 edge が評価可能な最新共通時刻を解決する．TePRA 2013 の tf の意味論に合わせる．

失敗時は invalid transform を返さず，status / exception を返す．

---

## 9. ROS message v2

既存 message を直ちに破壊せず，新 message を追加し，adapter を用意する方針を推奨する．最終的な名前は既存 package と衝突しないよう調整する．

## 9.1 message 候補

```text
BinghamOrientation.msg
ConditionalGaussianTranslation.msg
ProbabilisticTransformComponent.msg
ProbabilisticTransformStamped.msg
ProbabilisticTransformArray.msg
ApproximationInfo.msg
Provenance.msg
```

### `BinghamOrientation.msg`

```text
uint8 FINITE_BINGHAM=0
uint8 DIRAC=1
uint8 UNIFORM=2

uint8 kind
float64 inverse_concentration

# trace-zero, ISL-normalized shape
# order: ww, wx, wy, wz, xx, xy, xz, yy, yz, zz
float64[10] shape_upper_wxyz

# coupling reference; normally component mode
geometry_msgs/Quaternion reference_quaternion
```

### `ConditionalGaussianTranslation.msg`

```text
geometry_msgs/Vector3 mean_at_reference

# order: xx, xy, xz, yy, yz, zz
float64[6] residual_covariance_upper

# 3 x 9, row-major
# vec(R) is column-major
float64[27] rotation_coupling
```

### `ProbabilisticTransformComponent.msg`

```text
string component_id
float64 weight
BinghamOrientation orientation
ConditionalGaussianTranslation translation
```

### `ProbabilisticTransformStamped.msg`

```text
std_msgs/Header header
string parent_frame_id
string child_frame_id
string edge_id
string authority

geometry_msgs/Transform representative
uint8 representative_kind

ProbabilisticTransformComponent[] components
ApproximationInfo approximation
Provenance provenance
```

`header.frame_id` と `parent_frame_id` の二重管理は避けられるなら避ける．既存 ROS convention を確認し，片方に統一する．

## 9.2 representative

representative は compatibility projection であり，本体ではない．kind を持たせる．

候補:

- `NONE`
- `EXACT_MAP`
- `COMPONENT_MODE_APPROXIMATION`
- `PRODUCER_SUPPLIED`
- `MOMENT_REPRESENTATIVE`

mixture の最大 weight component の mode を global MAP と呼ばないこと．

## 9.3 old message adapter

旧 `ProbabilisticTF` から新 model への変換:

- orientation 1 component
- translation 1 component
- coupling \(C=0\)
- weight 1
- 旧 gauge を trace-zero + scale へ変換
- conversion が lossy / ambiguous なら diagnostic

新 model から旧 message への変換:

- single component のみ原則対応
- coupling は失われる
- mixture は policy を要求
- loss を warning / metadata に残す

---

## 10. tf / tf2 bridge

## 10.1 import

通常の `/tf` と `/tf_static` は Dirac component として Prob-TF buffer へ import できるようにする．

- deterministic rotation: `DIRAC`
- deterministic translation: residual covariance zero，coupling zero
- `/tf_static`: distribution が時不変
- loop prevention 用 authority / provenance を付与

## 10.2 export

Prob-TF から `/tf` へ出す場合は representative のみであり，lossy である．

export bridge が生成した `/tf` を再 import して stochastic edge と二重登録しないよう，authority と `derived_from` を検査する．

## 10.3 ROS package の純化

`ros/core/probtf_core` は最終的に `probtf_ros` へ寄せる．今回 rename の影響が大きければ，まず setup responsibility だけ分離してもよい．

最終状態:

- `probtf_msgs`: message only
- `probtf_ros`: ROS bridge only
- `src/probtf`: foundation
- `src/probtf_estimators`: producer

`probtf_ros/setup.py` が root `probtf` や third-party Bingham package を所有・再配布する構造を撤去する計画を立てる．

---

## 11. producer の分離

以下は Prob-TF core に置かない．

- IMU 遠心力・角速度から relative pose を推定する処理
- gyro prediction
- gravity / magnetic evidence
- deflection estimation
- AR marker pose estimation
- sensor-specific gating
- factor graph
- calibration algorithm

これらは Prob-TF distribution を publish する producer である．

現在の

- `imu_preprocessing.py`
- `imu_relative_pose.py`
- `orientation_filter.py`

を `probtf_estimators` 側へ段階的に移す．既存 import を壊さない compatibility re-export を一時的に残してよい．

`fusion.py` については分割する．

- 同一 edge の evidence fusion: estimator / evidence layer
- path transform composition: graph + kernel layer

両者を同じ `fusion` API と呼ばない．

---

## 12. current implementation からの移行順序

## Phase A: 調査と safety net

1. repository tree を記録する．
2. 現行 test を全実行する．
3. `symaware_grasp.prob_tf.tree` の path / inverse / moment 実装を読む．
4. ISL induced law 実装を検索する．
5. Bingham parameter normalization の現行規約を全箇所で調査する．
6. ROS message 利用箇所を grep する．
7. 変更前 test result を保存する．

この phase では behavior を変更しない．

## Phase B: domain model と validation

1. `probtf.distributions` を追加する．
2. orientation kind を追加する．
3. trace-zero shape + inverse concentration を実装する．
4. conditional translation と coupling を実装する．
5. joint component array を実装する．
6. weight normalization / zero mass を実装する．
7. old model adapter を追加する．
8. unit test を追加する．

この時点では graph lookup を変更しない．

## Phase C: graph / path / buffer

1. `PhysicalEdge`
2. `EdgeView`
3. `PathExpression`
4. forest validation
5. edge time buffer
6. exact / nearest / latest / latest-common
7. `lookup_path`
8. `lookup_kernel`

を実装する．

既存 `symaware_grasp` prototype は新 API の wrapper にする．

## Phase D: kernel expression

1. `ForwardEdgeKernel`
2. `InverseEdgeKernel`
3. `ComposedTransformKernel`
4. `MixtureTransformKernel`
5. deterministic fast path
6. expression diagnostics
7. dependency detection

を実装する．

この phase では exact density evaluator がなくてもよい．

## Phase E: ISL backend

1. 既存 ISL code adapter
2. unit direction action
3. non-unit vector action
4. zero vector
5. Bingham mixture
6. conditional Gaussian convolution / numerical integration
7. optional sample evaluator
8. optional moment evaluator

を実装する．

exact implementation が存在しない場合は protocol と unavailable error までに留める．

## Phase F: ROS v2

1. new messages
2. Python conversion
3. publisher / listener
4. `/probtf`, `/probtf_static`
5. `/tf`, `/tf_static` import
6. representative export
7. old message bridge
8. demo launch migration

## Phase G: producer separation

1. estimator package を追加する．
2. IMU producer を移す．
3. orientation filter を移す．
4. ROS node は domain service 呼び出しに限定する．
5. old import を deprecation wrapper にする．
6. `probtf_core` の relay install を撤去する．

---

## 13. test 計画

## 13.1 validation

- Bingham shape symmetry
- trace-zero
- normalization
- invalid NaN / Inf
- finite Bingham の positive inverse concentration
- Dirac の zero inverse concentration
- covariance symmetry / PSD
- coupling shape
- quaternion normalization
- matrix compression round-trip

## 13.2 weight

- 正常な sum 1
- sum 1 でない正 weight
- 負 weight clamp
- 一部 zero
- 全 zero -> `ZERO_MASS`
- NaN -> invalid
- normalized order preservation

## 13.3 coupling

- \(q=q_\mathrm{ref}\) で conditional mean が \(m\)
- quaternion sign flip で \(R(q)\) と coupling 結果が不変
- `vec` column-major の固定 test
- \(C=0\) で uncoupled model
- \(S=0\) で conditional Dirac translation
- known \(B,D\) に対する \(CD=B\)
- minimum-norm helper の test

## 13.4 deterministic reduction

Dirac orientation，zero covariance，zero coupling の component が通常の rigid transform と一致すること．

- forward point
- inverse point
- two-edge composition
- identity edge
- tf import/export round-trip

## 13.5 kernel

- one-component linearity
- mixture linearity
- composed expression order
- inverse expression
- zero mass propagation
- unavailable ISL backend error
- deterministic fast path が ISL backend を呼ばない
- repeated latent edge の dependency error または shared realization

## 13.6 ISL

既存理論・code に対応した regression value を利用する．Codex が期待値を独自に作らない．

最低限:

- unit vector
- non-unit vector scaling
- zero vector
- quaternion antipodal invariance
- deterministic concentration limit
- mixture weighted law
- uncoupled Gaussian convolution

## 13.7 moment

- exact deterministic moment
- Bingham second moment の既存 test 維持
- `E[R]` と Monte Carlo の比較
- point mean / covariance と Monte Carlo の比較
- coupling cross covariance
- closure を行った場合 `approximation` が立つこと

## 13.8 graph / time

- disconnected graph
- cycle rejection
- common parent
- forward / inverse traversal
- out-of-order insert
- exact time
- nearest tolerance
- latest common
- stale / out-of-range error
- static uncertain edge

## 13.9 architecture boundary

- `src/probtf` に `rospy`, `tf2_ros`, `probtf_msgs` import がない
- `src/probtf` が `probtf_estimators` を import しない
- ROS node に数理 algorithm が逆流していない
- examples が core から import されない

---

## 14. acceptance criteria

以下を全て満たした時点で本計画の実装を完了とする．

1. `ProbTfGraph.lookup_kernel(target, source, stamp)` が path に対応する lazy kernel expression を返す．
2. deterministic edge のみの path は通常の TF と同じ結果を返す．
3. stochastic edge は Bingham + conditional Gaussian + coupling + mixture として保存される．
4. Gaussian array と Bingham array が別々に存在せず，一つの joint component array になっている．
5. weight は利用時に正規化され，全 zero は `ZERO_MASS` になる．
6. storage Bingham parameter は trace-zero ISL shape + inverse concentration である．
7. inverse は同じ latent edge の view である．
8. ISL induced law が point / direction action の primary backend として API に現れる．
9. moment evaluator は optional であり，primary definition ではない．
10. Wang / generic \(SE(3)\) propagation を primary backend とするコードがない．
11. approximation / closure / representative projection が metadata で判別できる．
12. `src/probtf` は ROS 非依存である．
13. estimator は core に一方向依存する．
14. old demo は adapter 経由で最低限動作する．
15. test suite が通る．
16. architecture document に数式，規約，依存方向，未実装範囲が記載される．

---

## 15. 実装時の注意

- 既存の unrelated change を巻き戻さないこと．
- 大規模 rename と数理変更を同一 commit に混ぜないこと．
- 各 phase ごとに test を通すこと．
- 既存 public API を消す前に compatibility wrapper を追加すること．
- matrix order と quaternion order を docstring だけでなく test で固定すること．
- exact，numerical，moment，closure を型または enum で区別すること．
- `closure_approximation: bool` 一つだけで全近似を表さず，kind と詳細を持たせること．
- sampling backend を実装する場合，seed / RNG を外から渡せるようにすること．
- NumPy array を default argument にしないこと．
- singular covariance を勝手に正定値化しないこと．修正する場合は policy と補正量を記録すること．
- Bingham の identity shift は law を変えない gauge 操作だが，ISL scale decomposition と混同しないこと．
- component ID は array index ではなく stable string とすること．
- AR marker ambiguity を想定し，component order が時刻間で変わっても ID で追跡できるようにすること．

---

## 16. Codex の最終報告形式

作業完了時には，以下を日本語で報告すること．

1. 変更した directory / file
2. 新しい主要 class と責務
3. 数学的規約
4. ISL backend の実装状況
5. exact / approximate / unavailable の区別
6. old API との compatibility
7. 実行した test と結果
8. 未実装項目
9. 次に実装すべき phase
10. 設計上判断を要した箇所

ISL exact evaluator が未実装の場合は，その事実を明確に述べること．代替として Wang 型 propagation を実装して完了扱いにしてはならない．

---

## 17. 最終的な Prob-TF の定義

本実装における Prob-TF は，次のものとする．

> Prob-TF は，frame tree / forest の各 physical edge に，Bingham 分布に従う quaternion と，その quaternion に条件づけられた Gaussian translation の mixture を時刻付きで保持する．source–target lookup は path 上の forward / inverse edge view を解決し，それらが点・方向の確率 law に作用する kernel の合成を返す．Bingham 回転によるベクトルの作用は ISL の induced law を基本演算とし，moment，Gaussian approximation，mixture closure は必要に応じて明示的に選択される補助表現とする．

この定義を実装，message，doc，test の全てで一貫させること．
