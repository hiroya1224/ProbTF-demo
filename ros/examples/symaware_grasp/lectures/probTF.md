# Prob-TF 実装まとめ

## 1. 概要

現在の `Prob-TF` 実装は，6 自由度アームの各 joint を確率的回転として扱い，
`base_link` から各 link および各 link に付随する fixed point の位置分布を
**sampling-free** に計算するものである．

実装の中心思想は次の 3 点である．

1. latent layer では「同じ physical edge を direction 付きで参照する」だけに留める．
2. 主計算は quaternion Bingham 分布の 2 次・4 次モーメントから行う．
3. 可視化のための point cloud 生成でのみ最後に Gaussian sample を使う．

したがって，

$$
\text{Prob-TF の計算} \neq \text{Monte Carlo}
$$

であり，Monte Carlo は RViz 表示のための描画手段にすぎない．

---

## 2. 確率的 edge のモデル

各 edge を

$$
T_i = (R_i, a_i)
$$

とする．

- $R_i \in \mathrm{SO}(3)$: 回転
- $a_i \in \mathbb{R}^3$: parent frame で見た deterministic translation

revolute joint では quaternion

$$
q_i \in S^3
$$

を用い，

$$
q_i \sim \mathrm{Bingham}(A_i), \qquad
p(q_i \mid A_i) \propto \exp(q_i^\top A_i q_i)
$$

とする．

fixed joint では

$$
q_i = (1,0,0,0)^\top, \qquad R_i = I_3
$$

である．

---

## 3. latent layer と summary layer

### 3.1 latent layer

edge は独立な forward / inverse 2 本としては保持しない．
代わりに physical edge ID を 1 個だけ持ち，

$$
\mathrm{EdgeView}(e,+1), \qquad \mathrm{EdgeView}(e,-1)
$$

として向きを管理する．

path は

$$
\Pi = (e_1^{\sigma_1}, e_2^{\sigma_2}, \dots, e_m^{\sigma_m}),
\qquad \sigma_i \in \{+1,-1\}
$$

として表現される．

隣接する

$$
e^{+1}, e^{-1}
$$

あるいは

$$
e^{-1}, e^{+1}
$$

は summary 前に消去する．

### 3.2 summary layer

moment propagation や tangent surrogate は，
この reduced path が確定した後にだけ使う．

現在の正式対応は

$$
\texttt{source} = \texttt{root}
$$

すなわち `base_link -> target` の root-to-link query である．

---

## 4. 回転モーメント

各 edge の回転に対し，以下を基本量として持つ．

$$
G_i := \mathbb{E}[R_i] \in \mathbb{R}^{3 \times 3},
$$

$$
\mathcal{T}_i(S) := \mathbb{E}[R_i S R_i^\top].
$$

ここで

$$
\mathrm{vec}(R_i S R_i^\top)
=
(R_i \otimes R_i)\,\mathrm{vec}(S)
$$

であるから，

$$
K_i := \mathbb{E}[R_i \otimes R_i] \in \mathbb{R}^{9 \times 9}
$$

を持てば

$$
\mathrm{vec}(\mathcal{T}_i(S)) = K_i \, \mathrm{vec}(S)
$$

である．

実装では

$$
G_i = G(A_i), \qquad K_i = K(A_i)
$$

を quaternion の 2 次・4 次モーメントから構成する．

### 4.1 2 次モーメント

quaternion の 2 次モーメント

$$
C_i^{(2)} := \mathbb{E}[q_i q_i^\top]
$$

から回転行列の各成分

$$
R_{ab}(q)
$$

の期待値を計算し，

$$
G_i = \mathbb{E}[R_i]
$$

を得る．

### 4.2 4 次モーメント

quaternion の 4 次モーメント

$$
C_i^{(4)} := \mathbb{E}[q_a q_b q_c q_d]
$$

から

$$
K_i[(a,b),(c,d)] = \mathbb{E}[R_{ac} R_{bd}]
$$

を作る．

ここで重要なのは，`kron` の添字順が

$$
\mathbb{E}[R_{ac} R_{bd}]
$$

でなければならないことである．
この順序であれば

$$
\mathcal{T}_i(I_3) = \mathbb{E}[R_i I_3 R_i^\top] = I_3
$$

が保証される．

現在の実装ではこの添字順を修正済みであり，
これが point cloud の異常な wide spread を解消した主要因である．

---

## 5. 累積回転の moment propagation

path 上の回転を

$$
R_{1:k} := R_1 R_2 \cdots R_k
$$

とする．

異なる edge が独立であるという仮定のもとで，

$$
\mathbb{E}[R_{1:k}] = G_1 G_2 \cdots G_k
$$

である．

また 2 次作用素については

$$
\mathcal{T}_{1:k}
=
\mathcal{T}_1 \circ \mathcal{T}_2 \circ \cdots \circ \mathcal{T}_k
$$

であり，

$$
K_{1:k} = K_1 K_2 \cdots K_k
$$

と書ける．

実装では prefix ごとに

$$
G_{1:k}, \qquad \mathcal{T}_{1:k}, \qquad
C_{q,1:k} := \mathbb{E}[q_{1:k} q_{1:k}^\top]
$$

をキャッシュしている．

---

## 6. link origin の位置平均・共分散

`base_link` から第 $k$ link origin までの位置は

$$
p_k = \sum_{j=1}^{k} R_{1:j-1} a_j
$$

である．

ここで

$$
y_j := R_{1:j-1} a_j
$$

とおけば

$$
\mathbb{E}[p_k] = \sum_{j=1}^{k} \mathbb{E}[y_j]
= \sum_{j=1}^{k} G_{1:j-1} a_j.
$$

さらに $i \le j$ に対して

$$
\mathbb{E}[y_i y_j^\top]
=
\mathcal{T}_{1:i-1}
\left(
a_i (G_{i:j-1} a_j)^\top
\right)
$$

であるから

$$
\mathbb{E}[p_k p_k^\top]
=
\sum_{i=1}^{k}\sum_{j=1}^{k}
\mathbb{E}[y_i y_j^\top]
$$

となる．

したがって共分散は

$$
\mathrm{Cov}(p_k)
=
\mathbb{E}[p_k p_k^\top]
- \mathbb{E}[p_k]\mathbb{E}[p_k]^\top
$$

である．

この計算が `lookup()` と `compute_link_origin_moments()` の本体である．

---

## 7. 累積 quaternion 分布の closure

位置計算そのものには不要だが，
姿勢分布を外部 API や surrogate 計算に使うために，
累積 quaternion の 2 次モーメント

$$
C_{q,1:k} = \mathbb{E}[q_{1:k} q_{1:k}^\top]
$$

を再び Bingham 分布へ閉じる．

独立な quaternion 積

$$
q_{ab} = q_a \otimes q_b
$$

に対し，

$$
C_{ab} = \mathbb{E}[q_{ab} q_{ab}^\top]
$$

は $C_a, C_b$ から解析的に計算できる．

この $C_{q,1:k}$ に対して

$$
A_{1:k}
=
\operatorname{matchBingham}(C_{q,1:k})
$$

を解き，moment-matched cumulative Bingham parameter を得る．

これは **closure approximation** であり，
主計算そのものではない．

---

## 8. attached point の exact summary

target link に固定ベクトル

$$
r \in \mathbb{R}^3
$$

が付いているとする．
その world 座標は

$$
x_{k,r} = p_k + R_{1:k} r
$$

である．

この量に対する exact moment summary は

$$
\mathbb{E}[x_{k,r}]
=
\mathbb{E}[p_k] + \mathbb{E}[R_{1:k} r]
$$

および

$$
\mathrm{Cov}(x_{k,r})
=
\mathrm{Cov}(p_k)
 + \mathrm{Cov}(R_{1:k}r)
 + C_{\mathrm{cross}} + C_{\mathrm{cross}}^\top
$$

で与えられる．

ここで

$$
C_{\mathrm{cross}}
:=
\mathbb{E}[p_k (R_{1:k}r)^\top]
- \mathbb{E}[p_k]\mathbb{E}[R_{1:k}r]^\top
$$

である．

さらに

$$
\mathbb{E}[p_k (R_{1:k}r)^\top]
=
\sum_{j=1}^{k}
\mathbb{E}\!\left[
R_{1:j-1} a_j (R_{1:k} r)^\top
\right]
$$

であり，各項は

$$
\mathbb{E}\!\left[
R_{1:j-1} a_j (R_{1:k} r)^\top
\right]
=
\mathcal{T}_{1:j-1}
\left(
a_j (G_{j:k} r)^\top
\right)
$$

と書ける．

現在の `lookup_point()` はこの exact summary を返す．

---

## 9. 接空間 Gaussian surrogate

### 9.1 基本問題

fixed vector を cumulative rotation で回した

$$
x = R(q)\,r
$$

の分布を，球面上の局所 Gaussian として近似したい．

まず

$$
v = \frac{r}{\|r\|}
$$

とする．

### 9.2 shape parameter

累積 Bingham を

$$
A = M \,\mathrm{diag}(\lambda_1,\lambda_2,\lambda_3,\lambda_4)\, M^\top,
\qquad
\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge \lambda_4
$$

と対角化する．

論文定義に従い

$$
\mu_1 = \lambda_1 + \lambda_2,\qquad
\mu_2 = \lambda_1 + \lambda_3,\qquad
\mu_3 = \lambda_2 + \lambda_3
$$

を shape parameter とする．

さらに

$$
D_\mu = \mathrm{diag}(\mu_1,\mu_2,-\mu_3)
$$

と置く．

### 9.3 general parameter case

実装では対角 case に落とすため，

$$
Q = R_{\mathrm{mode}} H
$$

に相当する 3 次元回転を構成し，

$$
v_0 = H^\top v
$$

に対応する reduced vector を求める．

現在のコードでは，mode quaternion の接空間固有方向を
3 次元ベクトルへ押し戻して $H$ の近似を構成している．

### 9.4 non-polar type

non-polar では

$$
d_\mu(v_0)
=
\mathrm{tr}(D_\mu) - v_0^\top D_\mu v_0
$$

を用いて

$$
B_\mu
=
\mathrm{diag}
\begin{pmatrix}
(\mu_1+\mu_2)(\mu_1-\mu_3) \\
(\mu_1+\mu_2)(\mu_2-\mu_3) \\
(\mu_1-\mu_3)(\mu_2-\mu_3)
\end{pmatrix}
$$

を定義する．

また

$$
P_{v_0} = I_3 - v_0 v_0^\top
$$

として，

$$
\Lambda_{\tan}
=
\frac{1}{2 d_\mu(v_0)} P_{v_0} B_\mu P_{v_0}
$$

とする．

### 9.5 polar type

polar type，すなわち

$$
d_\mu(v_0)=0
$$

かつ

$$
\mu_2 = \mu_3,\qquad v_0 = \pm e_1
$$

では，上式をそのまま使わず

$$
\Lambda_{\tan}
=
\frac{\mu_1-\mu_2}{2} P_{v_0}
$$

を使う．

### 9.6 Jacobian 補正込み local precision

球面指数写像の Jacobian を含めて

$$
\Lambda_{\mathrm{loc}}
=
\Lambda_{\tan} + \frac{1}{3} P_{v_0}
$$

とする．

その Moore-Penrose 擬逆を

$$
\Sigma = \Lambda_{\mathrm{loc}}^+
$$

とおく．

### 9.7 平均・共分散近似

接空間 Gaussian

$$
u \sim \mathcal{N}(0,\Sigma), \qquad u \in T_{v_0} S^2
$$

を指数写像

$$
\exp_{v_0}(u)
=
\cos(\|u\|) v_0
+
\sin(\|u\|)\frac{u}{\|u\|}
$$

で球面へ戻したとき，
ambient $\mathbb{R}^3$ での平均・共分散は

$$
\mathbb{E}[x_G]
\approx
\left(1-\frac{1}{2}\mathrm{tr}\Sigma\right) v_0
$$

および

$$
\mathrm{Cov}(x_G)
\approx
\frac{1}{2}\mathrm{tr}(\Sigma^2) v_0 v_0^\top
+
\Sigma
- \frac{1}{3}\left(\mathrm{tr}(\Sigma)\Sigma + 2\Sigma^2\right)
$$

で近似する．

一般座標系へ戻すと

$$
M(r;A) = \|r\|\,Q\,M_0,
\qquad
V(r;A) = \|r\|^2\,Q\,V_0\,Q^\top
$$

である．

`induced_vector_moments_tangent()` はこの $M(r;A),V(r;A)$ を返す．

---

## 10. attached point の tangent-surrogate summary

可視化では，各 link の axis endpoint

$$
r_x = \ell e_x,\qquad
r_y = \ell e_y,\qquad
r_z = \ell e_z
$$

を world 座標へ持ち上げた分布が欲しい．

現在の実装では，これを

$$
x_{k,r} = p_k + R_{1:k} r
$$

に対する **hybrid summary** として計算する．

すなわち，

1. $p_k$ の平均・共分散は exact moment propagation
2. $R_{1:k}r$ は cumulative Bingham に対する tangent surrogate
3. cross covariance は exact moment propagation

で与える．

したがって

$$
\mathbb{E}[x_{k,r}]
=
\mathbb{E}[p_k] + M(r;A_{1:k})
$$

および

$$
\mathrm{Cov}(x_{k,r})
=
\mathrm{Cov}(p_k)
 + V(r;A_{1:k})
 + C_{\mathrm{cross}}
 + C_{\mathrm{cross}}^\top
$$

となる．

ここで

$$
C_{\mathrm{cross}}
=
\mathbb{E}[p_k (R_{1:k}r)^\top]
- \mathbb{E}[p_k]M(r;A_{1:k})^\top
$$

である．

この形にした理由は，

$$
\text{translation} \perp \text{rotated vector}
$$

という誤った独立化を避けるためである．
以前の wide spread は，主にこの相関破壊と回転 4 次モーメントのバグにより生じていた．

---

## 11. RViz 可視化ノード

`prob_tf_link_cloud_node.py` は各 frame について

$$
\mu_{k,\alpha}, \Sigma_{k,\alpha},
\qquad \alpha \in \{x,y,z\}
$$

を

$$
\mu_{k,\alpha}, \Sigma_{k,\alpha}
=
\texttt{lookup\_point\_tangent\_surrogate}(r_\alpha)
$$

で一度だけ求める．

その後，表示用にだけ

$$
\xi_{k,\alpha}^{(n)}
\sim
\mathcal{N}(\mu_{k,\alpha}, \Sigma_{k,\alpha})
$$

を生成し，1 個の `PointCloud2` に詰めて publish する．

ここで使っている sampling は

$$
\text{RViz 表示用}
$$

であり，Prob-TF の内部計算ではない．

---

## 12. 現在の実装上の整理

現在の構成をまとめると次の通りである．

### 12.1 sampling-free な部分

- 各 joint の Bingham 2 次・4 次モーメント
- 回転平均 $G_i$
- 2 次作用素 $K_i$
- root-to-link の平均位置と共分散
- cumulative quaternion の 2 次モーメント
- cumulative Bingham closure
- tangent surrogate の平均・共分散
- attached point の cross covariance

### 12.2 sampling を使う部分

- RViz 用点群の描画

---

## 13. 現在の API の意味

### `lookup(root, target)`

返すものは

$$
{}^{\mathrm{root}}p_{\mathrm{target}}
$$

すなわち target link origin の Gaussian summary である．

### `lookup_point(root, target, r)`

返すものは

$$
{}^{\mathrm{root}}(p_{\mathrm{target}} + R_{\mathrm{target}} r)
$$

の **exact moment summary** である．

### `lookup_point_tangent_surrogate(root, target, r)`

返すものは上と同じ量に対する

$$
\text{origin exact} + \text{vector tangent surrogate} + \text{exact cross}
$$

という hybrid summary である．

現在の RViz node はこの API を使っている．

---

## 14. 検証上の重要点

今回の修正後，少なくとも次の性質が成り立つようになっている．

### 回転作用素の整合性

$$
\mathcal{T}_i(I_3) = I_3
$$

### root-to-link 位置分布

analytic covariance が Monte Carlo covariance と同程度のスケールになる．

### tangent-surrogate attached point

`lookup_point_tangent_surrogate()` の covariance trace は，
`lookup_point()` の exact summary と同じオーダーを保つ．

---

## 15. まとめ

最新の Prob-TF 実装は，

$$
\text{Bingham quaternion} \;\longrightarrow\;
\text{rotation moments} \;\longrightarrow\;
\text{Gaussian summary}
$$

という流れを基本にしている．

link origin には exact moment propagation を使い，
axis endpoint には

$$
\text{exact origin} + \text{tangent surrogate} + \text{exact cross covariance}
$$

を使う．

そのため，現在の point cloud 可視化は

1. 内部計算は sampling-free
2. 球面ベクトルの近似は tangent surrogate
3. sampling は RViz 表示の最後だけ

という構成になっている．
