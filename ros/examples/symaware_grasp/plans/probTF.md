# Prob-TF 実装プラン

`urdf/simple_six_dof_arm.urdf` の構造に，Prob-TF (確率的 TF)を導入します．
その実装方針が以下です．これに基づいて，prob-TF のプロトタイプを実装してください．
なお，これについても，rviz 上で分布を確認できるようにしたいが，座標系ごとに pointcloud + pose (frame) が並ぶと rviz の操作が大変なので，点群は一つにまとめてしまってよい．また，pose も，通常の TF で代用してしまうので，点群だけ出してもらえればよい．

# Prob-TF prototype implementation plan for a 6-DoF manipulator

このメモは，Codex にそのまま渡して Prob-TF のプロトタイプを実装させるための仕様書である．PDF に依存しないように，必要な数式・設計判断・URDF 由来の TF 情報をすべてここにまとめる．

目的は，通常の TF tree の各 edge を確率的 TF に置き換え，`base_link` から各 link までの平均位置・共分散・近似姿勢分布を可視化できるようにすることである．

---

## 1. 実装の基本方針

### 1.1 目標

次を実装する．

1. URDF から通常の kinematic tree を読み取る．
2. 各 revolute joint の回転を quaternion Bingham 分布で表現する．
3. 各 link origin の `base_link` 座標での平均位置と共分散を計算する．
4. 各 link の累積姿勢について，必要なら moment-matched Bingham を返す．
5. 各 link の Prob-TF を表・CSV・図として確認できるようにする．

### 1.2 重要な設計判断

Prob-TF は次の hybrid 方式で実装する．

- **内部の主計算**は 2 次・4 次モーメントに基づく moment-based propagation で行う．  
  これにより，`q1 * q2` を Bingham 分布に近似しなくても，位置の平均・共分散を計算できる．
- **外部 API や cache 用**に，累積 quaternion 分布を再び Bingham 型で持ちたい場合だけ moment-matching を使う．
- **各 edge が固定ベクトルを回す影響の局所可視化**には，接空間 Gaussian surrogate を使う．

したがって，moment-matching は「必須の主計算」ではなく，「Bingham 型に閉じた Prob-TF object を返すための closure」として使う．

### 1.3 絶対に守るべきこと：latent layer と summary layer を分ける

Prob-TF tree では，逆向き edge を別の独立な分布として登録してはいけない．ただし，この prototype では具体的な samples を常に持つわけではない．したがって，「同じ変換の逆元」を sample で実現する代わりに，**同じ physical edge ID を参照する latent random variable の view** として管理する．

この実装では，必ず次の 2 層を分ける．

1. **latent layer**
   - 各 physical TF edge に一意の `edge_id` を与える．
   - `parent -> child` は `EdgeView(edge_id, direction=+1)` とする．
   - `child -> parent` は新しい edge ではなく，`EdgeView(edge_id, direction=-1)` とする．
   - lookup はまず `EdgeView` の列，すなわち path expression を返す．
   - この段階では Gaussian summary，Bingham closure，ambient mean/cov への射影をしない．

2. **summary layer**
   - reduced path expression が確定した後でだけ，moment propagation，tangent Gaussian surrogate，Bingham moment-matching を使う．
   - summary の結果として得られる `mean_translation`, `cov_translation`, `bingham_rotation` は，原則として terminal output である．
   - summary result を新しい independent edge として graph に登録して再合成してはいけない．どうしても行う場合は `closure_approximation=True` の近似として明示する．

重要なのは，`R(q) r` の law や接空間 Gaussian 近似に落とした時点で，元の quaternion `q` との結合情報は失われることである．その summary だけから逆向き変換を作ると，forward と inverse が同じ実現値を共有していることを表せない．したがって，逆向き処理は summary 後ではなく，必ず latent path expression の段階で行う．

### 1.4 inverse view の正確な意味

physical edge `e` が parent `p` から child `c` への random transform

\[
T_e = (R_e, a_e)
\]

を表すとする．ここで `a_e` は parent frame で見た child origin の deterministic translation である．

順向き view は

\[
T_e^{+} = (R_e, a_e)
\]

であり，逆向き view は

\[
T_e^{-} = T_e^{-1} = (R_e^\top, -R_e^\top a_e)
\]

である．`T_e^{-}` は別の random transform ではない．同じ `edge_id = e` を参照し，direction だけが `-1` である．

したがって，latent expression の段階では

\[
T_e^{+} \circ T_e^{-} = I,\qquad T_e^{-} \circ T_e^{+} = I
\]

が sample-wise に成り立つものとして扱う．ただし，この cancellation は **summary 計算前** に行う必要がある．

悪い例は，`T_e^{-}` の marginal law だけを別途 Bingham/Gaussian として作り，それを `T_e^{+}` と独立に合成することである．この場合，一般に

\[
\mathbb{E}[R_e] \, \mathbb{E}[R_e^\top] \neq I
\]

なので，`e` を往復しても identity に戻らない．

### 1.5 独立性に関するルール

moment propagation では，基本的に path 上の異なる physical edges は独立であると仮定する．一方，同じ `edge_id` が複数回現れる場合，それらは独立ではない．特に `direction=+1` と `direction=-1` は同じ random variable の関数である．

したがって実装は次のルールに従う．

- tree lookup が返す simple path では，同じ physical edge は高々 1 回しか現れない．
- user が手動で path expression を与えた場合，まず `e+` と `e-` の隣接 cancellation を行う．
- cancellation 後にも同じ `edge_id` が 2 回以上残る場合，初期 prototype では `NotImplementedError` を投げる．これは高次の joint moments または factor graph 的な依存管理が必要になるためである．
- summarized distribution 同士を合成する API は default では禁止する．

### 1.6 今回の prototype の対応範囲

最初の完成目標は，`base_link` から各 link origin への Prob-TF を計算して可視化することである．この root-to-link query はすべて forward edges のみからなるので，2 次・4 次 moment propagation が最も素直に使える．

任意の `source -> target` lookup については，まず latent path expression を返せるようにする．ただし，その path に inverse views が含まれる場合，translation term `-R_e^T a_e` が rotation と相関するため，root-to-link と同じ簡単な式では処理できない．初期版では次の仕様にする．

- `lookup_path(source, target)` は任意の frame pair に対して実装する．
- `lookup(source, target, summarize=True)` の full moment summary は，まず `source == root` の場合を正式対応とする．
- `source != root` の full summary は，後で general path moment propagation を追加するまで `NotImplementedError` にしてよい．
- ただし deterministic fallback と Monte Carlo fallback はオプションとして実装してよい．

この制限は安全側の仕様である．不完全な独立性仮定で誤った covariance を出すより，未対応として止める方がよい．

---

## 2. 数学的仕様

### 2.1 Quaternion Bingham 分布

scalar-first convention を使う．

\[
q = (w, x, y, z)^\top \in S^3.
\]

Bingham 分布は

\[
p(q \mid A) \propto \exp(q^\top A q)
\]

で定義する．ここで `A` は実対称な `4 x 4` trace-zero matrix とする．

モードを `q_mode` に置きたい場合，`q_mode` を第 1 列に持つ直交行列 `U` を作り，

\[
A = U \operatorname{diag}(\lambda) U^\top
\]

とする．この convention では，最大固有値の固有ベクトルが mode になる．例えば isotropic な集中分布として

\[
\lambda = (3 k, -k, -k, -k)
\]

を使うと trace-zero になる．

### 2.2 回転行列の 2 次・4 次モーメント

各 edge の random rotation を

\[
R_i = R(q_i), \qquad q_i \sim \mathcal{B}(A_i)
\]

とする．

次の 2 つを各 edge の基本 moment object として持つ．

\[
G_i := \mathbb{E}[R_i],
\]

\[
\mathcal{T}_i(S) := \mathbb{E}[R_i S R_i^\top].
\]

`G_i` は quaternion の 2 次モーメントから計算できる．`T_i` は quaternion の 4 次モーメントから計算できる．

実装では，`T_i` を `9 x 9` 行列 `K_i` として持つとよい．column-major `vec` を使えば，

\[
\operatorname{vec}(R_i S R_i^\top) = (R_i \otimes R_i) \operatorname{vec}(S)
\]

なので，

\[
\operatorname{vec}(\mathcal{T}_i(S)) = \mathbb{E}[R_i \otimes R_i] \operatorname{vec}(S).
\]

したがって，

\[
K_i := \mathbb{E}[R_i \otimes R_i] \in \mathbb{R}^{9 \times 9}.
\]

この `K_i` は `R(q)` の各成分を quaternion 成分の 2 次式として展開し，4 次モーメントを代入して作る．

### 2.3 累積回転の moment propagation

累積回転を

\[
R_{1:k} := R_1 R_2 \cdots R_k
\]

とする．独立性を仮定すると，

\[
\mathbb{E}[R_{1:k}] = G_1 G_2 \cdots G_k.
\]

また，固定行列 `S` に対して

\[
\mathbb{E}[R_{1:k} S R_{1:k}^\top]
= (\mathcal{T}_1 \circ \mathcal{T}_2 \circ \cdots \circ \mathcal{T}_k)(S).
\]

つまり `K_total = K_1 @ K_2 @ ... @ K_k` でよい．

### 2.4 Link origin の位置平均・共分散

URDF の parent-to-child transform を

\[
T_i = (R_i, a_i)
\]

と書く．ここで `a_i` は parent frame から child frame origin への translation であり，joint motion の前に入る fixed translation とする．この URDF では各 joint origin の rpy はすべてゼロなので，この扱いでよい．

`base_link` から第 `k` link origin までの位置は

\[
p_k = \sum_{j=1}^{k} R_{1:j-1} a_j,
\]

ただし `R_{1:0} = I` である．

各項を

\[
y_j := R_{1:j-1} a_j
\]

とおく．平均は

\[
\mathbb{E}[y_j] = G_{1:j-1} a_j.
\]

2 次モーメントは，`i <= j` に対して

\[
\mathbb{E}[y_i y_j^\top]
= (\mathcal{T}_{1:i-1})\left(a_i (G_{i:j-1} a_j)^\top\right).
\]

ここで `G_{i:j-1} = G_i G_{i+1} ... G_{j-1}` であり，空積は identity とする．

したがって

\[
\mathbb{E}[p_k p_k^\top]
= \sum_{i=1}^{k}\sum_{j=1}^{k} \mathbb{E}[y_i y_j^\top]
\]

を計算し，

\[
\operatorname{Cov}(p_k)
= \mathbb{E}[p_k p_k^\top] - \mathbb{E}[p_k]\mathbb{E}[p_k]^\top
\]

とする．

実装では，`i > j` の場合は

\[
\mathbb{E}[y_i y_j^\top]
= \mathbb{E}[y_j y_i^\top]^\top
\]

を使う．

### 2.5 接空間 Gaussian surrogate

固定ベクトル `r != 0` を Bingham 回転で回した

\[
x = R(q) r
\]

の分布を近似したいときに使う．

まず

\[
v = r / \|r\|.
\]

対角化された shape parameter を

\[
\mu = (\mu_1, \mu_2, \mu_3)^\top, \qquad \mu_1 \geq \mu_2 \geq |\mu_3|
\]

とする．非 polar type では次を使う．

\[
D_\mu = \operatorname{diag}(\mu_1, \mu_2, -\mu_3),
\]

\[
d_\mu(v) = \operatorname{tr}(D_\mu) - v^\top D_\mu v,
\]

\[
B_\mu = \operatorname{diag}\left(
(\mu_1 + \mu_2)(\mu_1 - \mu_3),
(\mu_1 + \mu_2)(\mu_2 - \mu_3),
(\mu_1 - \mu_3)(\mu_2 - \mu_3)
\right),
\]

\[
P_v = I_3 - v v^\top.
\]

接空間上の leading-exponent precision は

\[
\Lambda_{\mathrm{tan}}(v, \mu)
= \frac{1}{2 d_\mu(v)} P_v B_\mu P_v.
\]

有限 concentration で tangent-plane density の Jacobian `sinc` まで含める場合は

\[
\Lambda_{\mathrm{loc}}(v, \mu)
= \Lambda_{\mathrm{tan}}(v, \mu) + \frac{1}{3} P_v
\]

を使う．最初の実装では `use_jacobian_correction=True` を default とし，この `Lambda_loc` を使う．大きな concentration では差は小さい．

`Lambda_loc` は ambient `3 x 3` matrix としては rank 2 であり，nullspace は `span(v)` である．Moore-Penrose pseudo-inverse を使って

\[
\Sigma = \Lambda_{\mathrm{loc}}^+
\]

とする．

接空間 Gaussian を

\[
u \sim N(0, \Sigma), \qquad u \in T_v S^2
\]

とし，球面上へ

\[
x_G = \exp_v(u)
\]

で戻す．

指数写像は

\[
\exp_v(u) = \cos(\|u\|) v + \sin(\|u\|) \frac{u}{\|u\|}
\]

であり，`u = 0` では `exp_v(0) = v` とする．

ambient `R^3` での平均・共分散近似は

\[
\mathbb{E}[x_G] \approx \left(1 - \frac{1}{2}\operatorname{tr}\Sigma\right)v,
\]

\[
\operatorname{Cov}(x_G)
\approx
\frac{1}{2}\operatorname{tr}(\Sigma^2) v v^\top
+ \Sigma
- \frac{1}{3}\left(\operatorname{tr}(\Sigma)\Sigma + 2\Sigma^2\right).
\]

非単位入力 `r` については

\[
M(r; A) = \|r\| M(v; A), \qquad
V(r; A) = \|r\|^2 V(v; A).
\]

#### general parameter case

一般の Bingham parameter `A` では，diagonal-reducing frame pair `(H, R_mode)` を使う．

1. `v0 = H.T @ v` に変換する．
2. 対角 case で `M0`, `V0` を計算する．
3. `Q = R_mode @ H` として，

\[
M(v; A) = Q M_0, \qquad V(v; A) = Q V_0 Q^\top.
\]

最初のプロトタイプでは，`diagonal_reduction.py` に以下を実装する．

- `eigendecompose_bingham(A)` で `A = M diag(lambda) M.T` を得る．
- eigenvalue convention から shape parameter `mu` を計算する．
- 論文通りの `(H, R_mode)` の完全実装が重い場合，初期版では次の簡略化を許す．
  - Bingham parameter を作る段階で，`A = U diag(lambda) U.T` の `U` を保存しておく．
  - `U` から `H`, `R_mode` を復元する関数をあとで差し替え可能にする．

### 2.6 Polar type の扱い

初期版では，接空間 Gaussian surrogate は **non-polar type を主対象**にする．polar type は実装上は以下のどちらかにフォールバックする．

1. sampling による Monte Carlo estimate，または
2. 4 次モーメントに基づく moment propagation のみを使い，tangent surrogate は使わない．

後で polar type の明示式を追加する場合は，`tangent_surrogate.py` に `compute_lambda_tan_polar` を追加する．

---

## 3. BinghamNLL repository の利用

4 次モーメントの実装は以下を使う．

```bash
git clone -b develop https://github.com/hiroya1224/BinghamNLL.git third_party/BinghamNLL
cd third_party/BinghamNLL
git submodule update --init --recursive
pip install -e .
```

Codex は repository 内で以下を探すこと．正確な関数名は未確認なので，adapter 層で吸収する．

```bash
grep -R "moment" -n src test .
grep -R "normal" -n src test .
grep -R "Bingham" -n src test .
grep -R "log" -n src test .
```

`prob_tf/bingham_moments.py` に wrapper を作る．外部からは以下だけを使う．

```python
def bingham_second_moment(param_mat):
    """Return E[q q^T] as a 4x4 numpy array."""


def bingham_fourth_moment(param_mat):
    """Return E[q_i q_j q_k q_l] as a 4x4x4x4 numpy array."""


def bingham_log_normalizer(lambda_vec):
    """Return log C(lambda) under the exp(q^T A q) convention."""
```

PyTorch 実装の場合でも，外側 API は numpy を返すように統一する．

---

## 4. Moment-matching closure

累積 quaternion distribution を再び Bingham として返す場合に使う．

### 4.1 Quaternion covariance から Bingham parameter を復元する

Bingham 分布では antipodal symmetry があるので，orientation distribution の closure には quaternion covariance

\[
C_q = \mathbb{E}[q q^\top]
\]

を使う．

`C_q = U diag(omega) U.T` と固有分解し，`omega` を降順に並べる．次に trace-zero constraint の下で

\[
\lambda^* = \arg\min_{\sum_i \lambda_i = 0}
\left(\log C(\lambda) - \omega^\top \lambda\right)
\]

を解く．

最後に

\[
A = U \operatorname{diag}(\lambda^*) U^\top
\]

とする．

### 4.2 Quaternion product の covariance

独立な `q_a`, `q_b` に対して

\[
q_{ab} = q_a \otimes q_b
\]

である．左積行列を `L(q_a)` とすれば

\[
q_{ab} = L(q_a) q_b.
\]

よって

\[
\mathbb{E}[q_{ab}q_{ab}^\top]
= \mathbb{E}\left[L(q_a) C_b L(q_a)^\top\right],
\]

ここで `C_b = E[q_b q_b.T]` である．これは `q_a` の 2 次モーメントと `q_b` の 2 次モーメントから計算できる．

この covariance から 4.1 の方法で `A_ab` を作る．

### 4.3 使い分け

- `lookup_prob_transform(..., return_bingham=True)` のときだけ moment-matched cumulative Bingham を返す．
- link position の平均・共分散は moment-matching せず，2 次・4 次モーメント propagation で計算する．
- moment-matched cumulative Bingham は summary/closure であり，default では新しい tree edge として登録しない．
- closure を cache する場合は，`source`, `target`, `path_expression_hash`, `closure_approximation=True` を metadata として保存する．この cache は可視化・表示・外部 API 用であり，依存関係を無視した再合成には使わない．

---

## 5. 6-DoF robot arm の TF 情報

以下の URDF chain を実装対象とする．joint origin の rpy はすべてゼロである．

| edge index | joint name | parent | child | type | origin xyz | origin rpy | axis |
|---:|---|---|---|---|---|---|---|
| 1 | joint_1 | base_link | link_1 | revolute | (0, 0, 0.12) | (0, 0, 0) | (0, 0, 1) |
| 2 | joint_2 | link_1 | link_2 | revolute | (0, 0, 0.18) | (0, 0, 0) | (0, 1, 0) |
| 3 | joint_3 | link_2 | link_3 | revolute | (0.36, 0, 0) | (0, 0, 0) | (0, 1, 0) |
| 4 | joint_4 | link_3 | link_4 | revolute | (0.30, 0, 0) | (0, 0, 0) | (1, 0, 0) |
| 5 | joint_5 | link_4 | link_5 | revolute | (0.14, 0, 0) | (0, 0, 0) | (0, 1, 0) |
| 6 | joint_6 | link_5 | link_6 | revolute | (0.12, 0, 0) | (0, 0, 0) | (1, 0, 0) |
| 7 | tool0_joint | link_6 | tool0 | fixed | (0.10, 0, 0) | (0, 0, 0) | none |

### 5.1 位置式

`a_i` を上表の origin xyz とする．この URDF では

\[
p_{\mathrm{link1}} = a_1,
\]

\[
p_{\mathrm{link2}} = a_1 + R_1 a_2,
\]

\[
p_{\mathrm{link3}} = a_1 + R_1 a_2 + R_1R_2 a_3,
\]

\[
p_{\mathrm{link4}} = a_1 + R_1 a_2 + R_1R_2 a_3 + R_1R_2R_3 a_4,
\]

以下同様で，

\[
p_{\mathrm{tool0}} = \sum_{j=1}^{7} R_{1:j-1} a_j.
\]

`tool0_joint` は fixed なので，`R_7 = I` として扱うか，translation だけ追加する．

### 5.2 default probabilistic overwrite

プロトタイプでは，nominal joint angle はすべて 0 とする．各 revolute joint の nominal quaternion は

\[
q_{\mathrm{nom},i} = q_{\mathrm{origin},i} \otimes q_{\mathrm{axis},i}(\theta_i)
\]

である．この URDF では `q_origin_i` は identity で，初期値 `theta_i = 0` なので全 joint の nominal quaternion は identity でよい．

最初に作る default config は次でよい．

```yaml
root: base_link
frames:
  - base_link
  - link_1
  - link_2
  - link_3
  - link_4
  - link_5
  - link_6
  - tool0
edges:
  - joint: joint_1
    parent: base_link
    child: link_1
    type: revolute
    translation: [0.0, 0.0, 0.12]
    axis: [0.0, 0.0, 1.0]
    nominal_angle: 0.0
    bingham_kappa: 500.0
    bingham_eigenvalues: [1500.0, -500.0, -500.0, -500.0]
  - joint: joint_2
    parent: link_1
    child: link_2
    type: revolute
    translation: [0.0, 0.0, 0.18]
    axis: [0.0, 1.0, 0.0]
    nominal_angle: 0.0
    bingham_kappa: 450.0
    bingham_eigenvalues: [1350.0, -450.0, -450.0, -450.0]
  - joint: joint_3
    parent: link_2
    child: link_3
    type: revolute
    translation: [0.36, 0.0, 0.0]
    axis: [0.0, 1.0, 0.0]
    nominal_angle: 0.0
    bingham_kappa: 400.0
    bingham_eigenvalues: [1200.0, -400.0, -400.0, -400.0]
  - joint: joint_4
    parent: link_3
    child: link_4
    type: revolute
    translation: [0.30, 0.0, 0.0]
    axis: [1.0, 0.0, 0.0]
    nominal_angle: 0.0
    bingham_kappa: 350.0
    bingham_eigenvalues: [1050.0, -350.0, -350.0, -350.0]
  - joint: joint_5
    parent: link_4
    child: link_5
    type: revolute
    translation: [0.14, 0.0, 0.0]
    axis: [0.0, 1.0, 0.0]
    nominal_angle: 0.0
    bingham_kappa: 300.0
    bingham_eigenvalues: [900.0, -300.0, -300.0, -300.0]
  - joint: joint_6
    parent: link_5
    child: link_6
    type: revolute
    translation: [0.12, 0.0, 0.0]
    axis: [1.0, 0.0, 0.0]
    nominal_angle: 0.0
    bingham_kappa: 250.0
    bingham_eigenvalues: [750.0, -250.0, -250.0, -250.0]
  - joint: tool0_joint
    parent: link_6
    child: tool0
    type: fixed
    translation: [0.10, 0.0, 0.0]
```

この値は visualization 用の仮値である．後で実測や推定された Bingham parameter に置き換える．

---

## 6. Python package structure

次の構成で実装する．

```text
prob_tf_proto/
  pyproject.toml
  README.md
  configs/
    simple_six_dof_prob_tf.yaml
  prob_tf/
    __init__.py
    geometry.py
    bingham_moments.py
    bingham_match.py
    rotation_moments.py
    tangent_surrogate.py
    path_expression.py
    tree.py
    urdf_override.py
    visualize.py
  scripts/
    write_simple_six_dof_prob_tf.py
    show_link_prob_tf.py
    sample_check_simple_six_dof.py
  tests/
    test_geometry.py
    test_rotation_moments.py
    test_tree_roundtrip.py
    test_simple_six_dof.py
```

### 6.1 `geometry.py`

実装する関数：

```python
def normalize_vec(vec, eps=1e-12):
    pass


def quat_normalize(q, eps=1e-12):
    pass


def quat_conj(q):
    pass


def quat_mul(q1, q2):
    pass


def quat_left_matrix(q):
    pass


def quat_to_rotmat(q):
    pass


def axis_angle_to_quat(axis, angle):
    pass


def rpy_to_quat(roll, pitch, yaw):
    pass


def complete_orthonormal_basis(first_vec):
    """Return a 4x4 orthogonal matrix whose first column is first_vec."""


def exp_s2(v, u, eps=1e-12):
    pass


def tangent_projector(v):
    pass


def tangent_basis(v):
    """Return a 3x2 orthonormal basis of T_v S^2."""
```

### 6.2 `bingham_moments.py`

BinghamNLL への adapter．

```python
def bingham_second_moment(param_mat):
    pass


def bingham_fourth_moment(param_mat):
    pass


def ensure_trace_zero(param_mat):
    return param_mat - np.trace(param_mat) / 4.0 * np.eye(4)
```

### 6.3 `rotation_moments.py`

```python
class RotationMoment:
    def __init__(self, mean_rot, kron_rot):
        self.mean_rot = mean_rot      # 3x3, E[R]
        self.kron_rot = kron_rot      # 9x9, E[R kron R]

    def apply_second(self, mat):
        vec_mat = mat.reshape(9, order="F")
        vec_out = self.kron_rot @ vec_mat
        return vec_out.reshape((3, 3), order="F")


def rotation_moment_from_bingham(param_mat):
    c2 = bingham_second_moment(param_mat)
    c4 = bingham_fourth_moment(param_mat)
    mean_rot = compute_mean_rot_from_c2(c2)
    kron_rot = compute_kron_rot_from_c4(c4)
    return RotationMoment(mean_rot, kron_rot)
```

`compute_kron_rot_from_c4` は，`R(q)` の各成分を quadratic coefficient として持ち，係数同士を掛けて `E[q_a q_b q_c q_d]` を合計する実装にする．

### 6.4 `tangent_surrogate.py`

```python
class TangentSurrogateResult:
    def __init__(self, mean, cov, lambda_mat, sigma_mat, mode):
        self.mean = mean
        self.cov = cov
        self.lambda_mat = lambda_mat
        self.sigma_mat = sigma_mat
        self.mode = mode


def induced_vector_moments_tangent(r, param_mat, use_jacobian_correction=True):
    """Approximate moments of R(q) r using tangent-plane Gaussian surrogate."""
```

初期版では，`param_mat` が diagonal-reduction metadata を持たない場合は，`NotImplementedError` ではなく，わかりやすい message を出す．ただし，default config では `A` を mode quaternion から作るので，metadata を保存しておけばよい．

### 6.5 `path_expression.py`

このファイルは latent layer を担当する．summary 計算の前に，edge ID と direction を保持した path expression を作る．

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeView:
    edge_id: str
    direction: int  # +1 for parent->child, -1 for child->parent

    def inverse(self):
        return EdgeView(self.edge_id, -self.direction)


class PathExpression:
    def __init__(self, views):
        self.views = list(views)

    def reduce_adjacent_inverses(self):
        stack = []
        for view in self.views:
            if stack and stack[-1].edge_id == view.edge_id and stack[-1].direction == -view.direction:
                stack.pop()
            else:
                stack.append(view)
        return PathExpression(stack)

    def assert_no_repeated_edge_ids(self):
        ids = [view.edge_id for view in self.views]
        if len(ids) != len(set(ids)):
            raise NotImplementedError(
                "The same physical edge appears multiple times after path reduction. "
                "This requires dependency-aware higher-order propagation and is not supported in the initial prototype."
            )

    def reversed(self):
        return PathExpression([view.inverse() for view in reversed(self.views)])
```

`PathExpression` は distribution ではない．これは「どの latent random variables をどの向きに使うか」を表す式である．

### 6.6 `tree.py`

```python
class ProbTfEdge:
    def __init__(
        self,
        parent,
        child,
        translation,
        joint_type,
        axis=None,
        nominal_angle=0.0,
        bingham_param=None,
        rotation_moment=None,
    ):
        pass


class ProbTfResult:
    def __init__(
        self,
        source,
        target,
        mean_translation,
        cov_translation,
        mean_rotation=None,
        bingham_rotation=None,
        path=None,
        method=None,
    ):
        pass


class ProbTfTree:
    def __init__(self, root):
        self.root = root
        self.edges = {}

    def add_edge(self, edge):
        pass

    def lookup_path(self, source, target):
        """Return a reduced PathExpression with EdgeView objects.

        This method is the latent-layer lookup. It must not compute a distribution.
        It must not create a new random variable for an inverse edge.
        """

    def lookup(self, source, target, return_bingham=False, summarize=True):
        """Return ProbTfResult from source to target.

        Initial prototype requirement:
        - Always support summarize=False by returning the PathExpression.
        - Support full moment summary for source == root.
        - For source != root, raise NotImplementedError unless a safe fallback is explicitly selected.
        """

    def compute_link_origin_moments(self, target):
        """Specialized safe path for root-to-link origin moments.

        This uses only the root-to-target forward chain and the formulas in Section 2.4.
        """
```

### 6.7 `urdf_override.py`

```python
def load_prob_tf_yaml(path):
    pass


def build_tree_from_prob_tf_yaml(path):
    pass


def make_bingham_param_from_mode(q_mode, eigenvalues):
    q_mode = quat_normalize(q_mode)
    basis = complete_orthonormal_basis(q_mode)
    eigs = np.asarray(eigenvalues, dtype=float)
    eigs = eigs - np.mean(eigs)
    return basis @ np.diag(eigs) @ basis.T
```

### 6.8 `visualize.py`

```python
def covariance_ellipsoid(mean, cov, scale=1.0):
    pass


def plot_link_prob_tf(results, out_png):
    pass


def write_results_csv(results, out_csv):
    pass
```

`scale` は最初 `sqrt(chi2.ppf(0.99, df=3))` を使う．SciPy がない場合は固定値 `3.368214` を使う．

---

## 7. Scripts

### 7.1 `scripts/write_simple_six_dof_prob_tf.py`

上の YAML を `configs/simple_six_dof_prob_tf.yaml` に書き出す．

実行例：

```bash
python scripts/write_simple_six_dof_prob_tf.py
```

### 7.2 `scripts/show_link_prob_tf.py`

YAML を読み，`base_link` から各 link への Prob-TF を計算して表示する．

実行例：

```bash
python scripts/show_link_prob_tf.py --config configs/simple_six_dof_prob_tf.yaml --out-dir outputs/simple_six_dof
```

出力：

```text
outputs/simple_six_dof/link_prob_tf.csv
outputs/simple_six_dof/link_prob_tf.png
outputs/simple_six_dof/link_prob_tf.json
```

CSV columns:

```text
frame, mean_x, mean_y, mean_z, cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz, std_x, std_y, std_z, trace_cov
```

### 7.3 `scripts/sample_check_simple_six_dof.py`

Monte Carlo sampling による sanity check．Bingham sampling が BinghamNLL にない場合は，初期版では実装を skip してよい．sampling が可能な場合，moment propagation の結果と sample mean/cov を比較する．

---

## 8. Tests

最低限，以下をテストする．

### 8.1 quaternion and rotation tests

- `quat_mul(q, quat_conj(q))` が identity になる．
- `quat_to_rotmat(q)` が orthogonal になる．
- `quat_to_rotmat(q1 * q2) == quat_to_rotmat(q1) @ quat_to_rotmat(q2)` が成り立つ．

### 8.2 moment operator tests

- ほぼ Dirac 的な Bingham parameter で，`G` が nominal rotation に近い．
- `T(I) = I` に近い．実際，任意の rotation で `R I R.T = I` なので，これは重要な check．
- `T(S)` が symmetric `S` に対して symmetric に近い．

### 8.3 tree round-trip and dependency tests

Prob-TF では逆向き edge を独立扱いしない．そのため，次を必ずテストする．

- `lookup_path("base_link", "link_3")` は `[joint_1(+), joint_2(+), joint_3(+)]` になる．
- `lookup_path("link_3", "base_link")` は `[joint_3(-), joint_2(-), joint_1(-)]` になり，同じ edge IDs の inverse view である．
- `EdgeView("joint_3", +1).inverse()` は `EdgeView("joint_3", -1)` を返し，新しい edge ID を作らない．
- `PathExpression([joint_3(+), joint_3(-)]).reduce_adjacent_inverses()` は空 path になる．
- `base_link -> link_1 -> link_2 -> link_3 -> link_2` のような manual path は，summary 前に `joint_3(+) joint_3(-)` が消える．
- cancellation 後にも同じ edge ID が残る path は，初期版では `NotImplementedError` になる．
- `ProbTfResult` を `add_edge` で再登録しようとすると，`closure_approximation=True` が明示されていない限り失敗する．

### 8.4 simple six dof tests

不確かさを極端に小さくした場合，平均位置が deterministic FK と一致することを確認する．

---

## 9. Implementation order for Codex

次の順番で実装する．

1. `geometry.py` を実装する．
2. `path_expression.py` を実装し，`EdgeView` と `PathExpression` の cancellation test を通す．
3. `write_simple_six_dof_prob_tf.py` で YAML を生成する．
4. `tree.py` で latent-layer `lookup_path` と root-to-link position formula を実装する．この段階では `G_i = R_nom`, `K_i = R_nom kron R_nom` でよい．
5. `rotation_moments.py` を実装し，Bingham moments による `G_i`, `K_i` に差し替える．
6. `bingham_moments.py` で BinghamNLL adapter を実装する．
7. `show_link_prob_tf.py` で CSV/PNG を出す．
8. `tangent_surrogate.py` を実装し，各 edge translation や tool offset の local Gaussian surrogate を確認できるようにする．
9. `bingham_match.py` を追加し，`return_bingham=True` のときだけ cumulative Bingham closure を返す．
10. tests を増やす．

---

## 10. Acceptance criteria

最低限，次ができればプロトタイプ完成とする．

1. `python scripts/write_simple_six_dof_prob_tf.py` が YAML を生成する．
2. `python scripts/show_link_prob_tf.py --config configs/simple_six_dof_prob_tf.yaml --out-dir outputs/simple_six_dof` が成功する．
3. `outputs/simple_six_dof/link_prob_tf.csv` に `link_1` から `tool0` までの平均・共分散が出る．
4. `outputs/simple_six_dof/link_prob_tf.png` に平均 link chain と covariance ellipsoid が描かれる．
5. 不確かさを小さくすると，平均位置が deterministic FK に近づく．
6. `T(I) = I` の unit test が通る．
7. `lookup_path` が tree の reduced path を使い，逆向き edge を同じ `edge_id` の inverse view として返す．
8. `lookup(..., summarize=True)` は初期版では `source == root` の full summary を正式対応し，それ以外は安全でない独立性仮定を置かずに止める．
9. summarized `ProbTfResult` を新しい independent edge として再登録しない設計になっている．

---

## 11. 注意点

- `E[R]` は一般には SO(3) の元ではない．mean rotation matrix として扱い，姿勢そのものとして可視化する場合は projection to SO(3) が必要である．
- `Cov(p)` は数値誤差でわずかに非対称になることがあるので，最後に `(cov + cov.T) / 2` を取る．
- covariance の固有値が小さな負値になる場合は，visualization 前だけ clip する．計算本体では clip しすぎない．
- Bingham product を Bingham に戻す moment-matching は近似である．位置の平均・共分散の主計算には使わない．
- graph に複数 path がある場合，「分散が小さい path を選ぶ」は最初の実装では行わない．複数 path は pose graph / factor graph の問題として別扱いにする．
- `R(q)r` の Gaussian summary は元の `q` との結合情報を失う．したがって，Gaussian summary を逆向きにしたり，別 summary と独立に合成したりしない．
- 逆向き translation `-R^T a` は rotation と相関する．任意 source-target の covariance propagation ではここを無視しない．初期版で未実装なら `NotImplementedError` にする．

---

## 12. Minimal README text

```markdown
# Prob-TF prototype

This package implements a prototype probabilistic TF tree for a simple 6-DoF manipulator.
Each revolute joint is modeled by a quaternion Bingham distribution. The internal representation keeps a latent path expression with physical edge IDs, so inverse traversal is an inverse view of the same random transform rather than a new independent distribution. Link-origin means and covariances are propagated by rotation moment operators using second and fourth quaternion moments. Moment-matched Bingham closure is used only when an API consumer requests a Bingham-like cumulative orientation object.

Run:

```bash
python scripts/write_simple_six_dof_prob_tf.py
python scripts/show_link_prob_tf.py --config configs/simple_six_dof_prob_tf.yaml --out-dir outputs/simple_six_dof
```
```
