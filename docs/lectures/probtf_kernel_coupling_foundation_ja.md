# Prob-TF における Bingham–Gaussian coupling と kernel composition

## 0. この文書の目的

本書では，Prob-TF の各 edge が点または点の確率 law に作用する仕組みを，Bingham 分布，条件付き Gaussian 分布，および Markov kernel を用いて整理する．

中心となる立場は次の通りである．

- Prob-TF を，一般的な $SE(3)$ covariance propagation として一次的に定義しない．
- 姿勢は Bingham 分布に従う quaternion によって表す．
- 並進は，姿勢 quaternion に条件づけた Gaussian 分布として表す．
- 点への作用 $R(Q)z + X$ を直接扱う．
- JMAA で導出した Bingham-induced vector law を，回転作用の基本演算として使う．
- source–target lookup は，path 上の edge kernel の合成として表す．
- moment は必要に応じて計算する補助表現であり，Prob-TF の一次的な定義ではない．

特に，$m$，$C$，$S$ が何を表すか，それらが局所二次近似からどのように現れるか，期待値がどの段階で登場するかを明確にする．

---

## 1. 最初に押さえるべき点

条件付き Gaussian モデルは

```math
X \mid Q = q
\sim
\mathcal{N}
\left(
m
+
C
\left(
\operatorname{vec} R(q)
-
\rho_{\mathrm{ref}}
\right),
S
\right)
```

である．

ここで重要なのは，この式自体は，まだ $Q$ に関する周辺期待値を計算した式ではないことである．

単に，

> 姿勢 $Q$ の値を $q$ に固定したとき，並進 $X$ がどの Gaussian 分布に従うか

を定めている．

したがって，

- $m$ は，基準姿勢に条件づけた並進平均
- $S$ は，姿勢に条件づけた後にも残る並進共分散
- $C$ は，姿勢の変化に応じて条件付き並進平均を変化させる coupling

である．

$m$ や $S$ を理解するために，最初から $Q$ に関する期待値を取る必要はない．

周辺平均や周辺共分散は，この条件付きモデルを定義した後で，必要なら全期待値・全分散の公式を使って計算する量である．

---

## 2. 基本となる確率変数

一つの確率的座標変換を考える．

姿勢を表す unit quaternion の確率変数を

```math
Q \in \mathbb{S}^{3}
```

とする．

対応する回転行列を

```math
R(Q) \in SO(3)
```

と書く．

$Q$ と $-Q$ は同じ回転を表す．

並進の確率変数を

```math
X \in \mathbb{R}^{3}
```

とする．

入力点を

```math
z \in \mathbb{R}^{3}
```

とすると，変換後の点は

```math
Y
=
R(Q) z
+
X
```

である．

ここでは，

- $z$ は変換前の点
- $Y$ は変換後の点
- $Q$ はランダムな姿勢
- $X$ はランダムな並進

である．

---

## 3. 姿勢分布

姿勢 quaternion は Bingham 分布に従うものとする．

```math
Q
\sim
\mathfrak{B}(A)
```

ここで，$\mathfrak{B}(A)$ は Bingham 確率測度であり，$A$ は対称な Bingham parameter matrix である．

実装上は，JMAA の規約に従って

```math
A
=
\kappa_{A}
\check{A}
```

と分解する．

$\check{A}$ は trace-zero の shape parameter，$\kappa_A$ はその大きさを表す scale である．

message 上では，deterministic limit との接続のために

```math
s_{A}
=
\frac{1}{\kappa_{A}}
```

を inverse concentration として保持する方針を採る．

数値 backend で Bingham normalizer などを評価する際には，必要に応じて最大固有値が 0 となる gauge へ identity shift してよい．ただし，storage と理論上の標準規約は trace-zero とする．

---

## 4. 回転行列の vectorization

回転行列を $\mathbb{R}^{9}$ に埋め込む写像を

```math
\rho(q)
:=
\operatorname{vec} R(q)
\in
\mathbb{R}^{9}
```

とする．

$\operatorname{vec}$ は column-major，すなわち回転行列の各列を順に積み重ねる規約とする．

coupling の基準となる quaternion を

```math
q_{\mathrm{ref}}
\in
\mathbb{S}^{3}
```

とし，

```math
R_{\mathrm{ref}}
:=
R(q_{\mathrm{ref}})
```

および

```math
\rho_{\mathrm{ref}}
:=
\operatorname{vec} R_{\mathrm{ref}}
```

と置く．

通常，$q_{\mathrm{ref}}$ は Bingham component の mode とする．

差分

```math
\rho(q)
-
\rho_{\mathrm{ref}}
```

は $SO(3)$ 上の群差ではない．

これは回転行列を $\mathbb{R}^{9}$ に埋め込んだ後の外在的な差分であり，component の基準姿勢近傍における局所一次情報を表すために使う．

---

## 5. 条件付き Gaussian モデル

並進の条件付き分布を

```math
X \mid Q = q
\sim
\mathcal{N}
\left(
m
+
C
\left(
\rho(q)
-
\rho_{\mathrm{ref}}
\right),
S
\right)
```

と定める．

各パラメータの意味は次の通りである．

### $m$

```math
m
\in
\mathbb{R}^{3}
```

は，基準姿勢 $q_{\mathrm{ref}}$ に条件づけた並進平均である．

実際，

```math
\mathbb{E}
\left[
X
\mid
Q = q_{\mathrm{ref}}
\right]
=
m
```

である．

### $C$

```math
C
\in
\mathbb{R}^{3 \times 9}
```

は rotation–translation coupling である．

姿勢の外在差分

```math
\rho(q)
-
\rho_{\mathrm{ref}}
```

を，並進の条件付き平均の変位へ写す．

### $S$

```math
S
\in
\mathbb{R}^{3 \times 3}
```

は，姿勢 $Q=q$ を条件づけた後にも残る並進の residual covariance である．

したがって，$S$ は並進 $X$ の周辺共分散そのものではない．

### 生成的な表現

条件付きモデルは

```math
X
=
m
+
C
\left(
\rho(Q)
-
\rho_{\mathrm{ref}}
\right)
+
\varepsilon
```

および

```math
\varepsilon
\sim
\mathcal{N}(0,S)
```

と書ける．

ここで $\varepsilon$ は，姿勢の変化では説明されない並進ノイズを表す．

---

## 6. $m$，$C$，$S$ はどのように得られるか

### 6.1 一般の Prob-TF message における立場

Prob-TF の architecture 上は，$m$，$C$，$S$ は一つの conditional model を指定するパラメータである．

したがって producer は，センサモデル，最適化，近似，学習，または既知の力学モデルから，これらを直接与えてよい．

Prob-TF core は，それらがどの estimator から得られたかを仮定しない．

### 6.2 局所二次近似から得る場合

joint density が，位置 $x_{\mathrm{ref}}$ と姿勢 $R_{\mathrm{ref}}$ の近傍に mode を持つとする．

姿勢を右摂動の局所変数 $u \in \mathbb{R}^{3}$ により

```math
R(u)
=
R_{\mathrm{ref}}
\exp
\left(
[u]_{\times}
\right)
```

と表す．

位置について

```math
x
=
x_{\mathrm{ref}}
+
\delta x
```

と書く．

mode 近傍で負の対数密度を二次近似すると，位置と姿勢に関する項は

```math
\frac{1}{2}
\delta x^{\top}
H_{xx}
\delta x
+
\delta x^{\top}
H_{xu}
u
+
\frac{1}{2}
u^{\top}
H_{uu}
u
```

の形になる．

$H_{xx}$ が正定値であるとする．

$\delta x$ に関して平方完成すると，

```math
X \mid U = u
\approx
\mathcal{N}
\left(
x_{\mathrm{ref}}
+
B u,
S
\right)
```

を得る．ただし，

```math
B
=
-
H_{xx}^{-1}
H_{xu}
```

および

```math
S
=
H_{xx}^{-1}
```

である．

この段階でも，姿勢変数 $U$ に関する周辺期待値は取っていない．

単に $U=u$ と条件づけた Gaussian 分布を，平方完成によって読み取っている．

したがって，局所二次近似から構成する場合，

```math
m
=
x_{\mathrm{ref}}
```

であり，$S$ は位置方向の Hessian block の逆行列として得られる．

### 6.3 $C$ の導出

局所変数 $u$ による回転行列の変化を考える．

```math
\rho(u)
:=
\operatorname{vec}
\left(
R_{\mathrm{ref}}
\exp
\left(
[u]_{\times}
\right)
\right)
```

とする．

$u=0$ における微分を

```math
D_{\mathrm{rot}}
:=
\left.
\frac{\partial \rho(u)}
{\partial u}
\right|_{u=0}
\in
\mathbb{R}^{9 \times 3}
```

と置く．

すると，

```math
\rho(u)
-
\rho_{\mathrm{ref}}
=
D_{\mathrm{rot}} u
+
O
\left(
\lVert u \rVert^{2}
\right)
```

である．

$\operatorname{vec} R$ による条件付き平均

```math
m
+
C
\left(
\rho(u)
-
\rho_{\mathrm{ref}}
\right)
```

を，局所二次近似から得た

```math
x_{\mathrm{ref}}
+
B u
```

と一次で一致させるには，

```math
C
D_{\mathrm{rot}}
=
B
```

を要求すればよい．

標準的には最小 Frobenius ノルム解

```math
C
=
B
D_{\mathrm{rot}}^{+}
```

を採用する．

ここで $D_{\mathrm{rot}}^{+}$ は Moore–Penrose 擬似逆である．

以上から，局所二次近似を用いる場合の生成順序は次のようになる．

1. 平方完成により $m=x_{\mathrm{ref}}$，$B=-H_{xx}^{-1}H_{xu}$，$S=H_{xx}^{-1}$ を得る．
2. 回転埋め込みの微分 $D_{\mathrm{rot}}$ を計算する．
3. $C D_{\mathrm{rot}}=B$ を満たす標準解 $C=B D_{\mathrm{rot}}^{+}$ を得る．

したがって，$m$ と $S$ が $C$ から導かれるわけではない．

$m$，$B$，$S$ は同じ条件付き Gaussian の平方完成から現れ，$C$ はその局所 coupling $B$ を $\operatorname{vec} R$ 表現へ移す際に導かれる．

---

## 7. 期待値はどの段階で登場するか

条件付きモデルを定義する段階では，$Q$ に関する周辺期待値は取らない．

まず，

```math
\mathbb{E}
\left[
X
\mid
Q = q
\right]
=
m
+
C
\left(
\rho(q)
-
\rho_{\mathrm{ref}}
\right)
```

である．

特に，

```math
\mathbb{E}
\left[
X
\mid
Q = q_{\mathrm{ref}}
\right]
=
m
```

である．

これが $m$ の直接的な意味である．

その後，並進の周辺平均が必要になった場合に初めて，全期待値の公式を使う．

```math
\mathbb{E}[X]
=
m
+
C
\left(
\mathbb{E}[\rho(Q)]
-
\rho_{\mathrm{ref}}
\right)
```

したがって，一般には $m$ と $\mathbb{E}[X]$ は一致しない．

ただし，

```math
\mathbb{E}[\rho(Q)]
=
\rho_{\mathrm{ref}}
```

である場合や，

```math
C
=
0
```

である場合には一致する．

共分散については，全分散の公式から

```math
\operatorname{Cov}(X)
=
S
+
C
\operatorname{Cov}
\left(
\rho(Q)
\right)
C^{\top}
```

となる．

さらに，

```math
\operatorname{Cov}
\left(
X,
\rho(Q)
\right)
=
C
\operatorname{Cov}
\left(
\rho(Q)
\right)
```

である．

したがって，

- $S$ は conditional residual covariance
- $C \operatorname{Cov}(\rho(Q)) C^{\top}$ は姿勢不確かさから誘導される並進共分散

である．

この期待値・共分散の計算は，条件付きモデルの定義後に得られる帰結であり，$m$，$C$，$S$ の一次的な定義ではない．

---

## 8. 点への条件付き作用

入力点 $z \in \mathbb{R}^{3}$ に対し，

```math
Y
=
R(Q) z
+
X
```

とする．

$Q=q$ に条件づけると，

```math
Y \mid Q = q
\sim
\mathcal{N}
\left(
\mu(z,q),
S
\right)
```

となる．ここで，

```math
\mu(z,q)
:=
R(q) z
+
m
+
C
\left(
\rho(q)
-
\rho_{\mathrm{ref}}
\right)
```

である．

この $\mu(z,q)$ は，

1. 入力点の回転 $R(q)z$
2. 基準並進 $m$
3. coupling による並進平均の変化

の和である．

---

## 9. 条件付き Gaussian 確率 $G$

この節までは mixture index $\ell$ を導入しない．

単一 component に共通する一般式を先に定義する．

パラメータを

```math
\vartheta
:=
\left(
m,
C,
S,
\rho_{\mathrm{ref}}
\right)
```

とまとめる．

出力空間の可測集合を

```math
D
\in
\mathcal{B}
\left(
\mathbb{R}^{3}
\right)
```

とする．

$S$ が正定値である場合，姿勢を $q$ に固定したとき，出力点が $D$ に入る確率を

```math
G
\left(
z,
q,
D;
\vartheta
\right)
:=
\frac{1}
{(2 \pi)^{3/2}
\sqrt{\det S}}
\int_{D}
\exp
\left[
-
\frac{1}{2}
\left(
y
-
\mu(z,q)
\right)^{\top}
S^{-1}
\left(
y
-
\mu(z,q)
\right)
\right]
dy
```

と定義する．

ここで，

- $D$ は変換後の点が属する領域
- $y$ は変換後の点を表す積分変数
- $G(z,q,D;\vartheta)$ は $Q=q$ の下で $Y\in D$ となる条件付き確率

である．

すなわち，

```math
G
\left(
z,
q,
D;
\vartheta
\right)
=
\Pr
\left[
Y \in D
\mid
Q = q
\right]
```

である．

文脈上 $z$ と $\vartheta$ が固定されている場合は，

```math
G(q,D)
```

と略記してよい．

この定義では，積分領域が最初から $D$ なので，

```math
D
-
R(q)z
```

のような集合の平行移動記法を使う必要がない．

### $S$ が特異な場合

deterministic translation や退化 Gaussian を含める場合，$S$ は半正定値であり，逆行列と通常の Lebesgue 密度が存在しないことがある．

その場合の正式な定義は，Gaussian 確率測度を用いて

```math
G
\left(
z,
q,
D;
\vartheta
\right)
:=
\mathcal{N}
\left(
\mu(z,q),
S
\right)
(D)
```

とする．

正定値の場合に限り，先ほどの明示的な積分式と一致する．

---

## 10. 単一 Bingham component の kernel

ここでもまだ mixture index $\ell$ は導入しない．

Bingham parameter $A$ を持つ単一 component の kernel を

```math
\mathcal{K}
\left(
z,
D;
A,
\vartheta
\right)
:=
\int_{\mathbb{S}^{3}}
G
\left(
z,
q,
D;
\vartheta
\right)
\mathfrak{B}(A)(dq)
```

と定義する．

これは，

> 入力点 $z$ を，Bingham 姿勢と条件付き Gaussian 並進からなる単一 component に通したとき，出力点が $D$ に入る確率

を表す．

文脈上 $z$ と $\vartheta$ が固定されている場合は，

```math
\mathcal{K}(D;A)
:=
\int_{\mathbb{S}^{3}}
G(q,D)
\mathfrak{B}(A)(dq)
```

と略記してよい．

この段階では，

- $G$ が $q$ を固定した条件付き Gaussian 確率
- $\mathcal{K}$ が $q$ を Bingham 測度で周辺化した単一 component kernel

である．

---

## 11. edge mixture

ここで初めて mixture index $\ell$ を導入する．

edge $e$ が $L_e$ 個の component を持つとする．

第 $\ell$ component のパラメータを

```math
w_{e,\ell},
\quad
A_{e,\ell},
\quad
\vartheta_{e,\ell}
```

とする．

ここで，

```math
\vartheta_{e,\ell}
=
\left(
m_{e,\ell},
C_{e,\ell},
S_{e,\ell},
\rho_{e,\ell}
\right)
```

である．

edge kernel を

```math
\mathcal{K}_{e}(z,D)
:=
\sum_{\ell=1}^{L_e}
w_{e,\ell}
\mathcal{K}
\left(
z,
D;
A_{e,\ell},
\vartheta_{e,\ell}
\right)
```

と定義する．

文脈上，入力点と各 component の Gaussian 側パラメータが明らかな場合は，

```math
\mathcal{K}_{e}(D)
=
\sum_{\ell=1}^{L_e}
w_{e,\ell}
\mathcal{K}
\left(
D;
A_{e,\ell},
\vartheta_{e,\ell}
\right)
```

と略記してよい．

総和を取った後の左辺には，特定 component の $A_{e,\ell}$ を自由パラメータとして残さない．

したがって，

```math
\mathcal{K}_{e}(D;A_{\ell})
```

よりも

```math
\mathcal{K}_{e}(D)
```

と書く方が自然である．

---

## 12. mixture weight

publisher が与える raw weight を

```math
a_{e,\ell}
```

とする．

利用時には，

```math
a_{e,\ell}^{+}
:=
\max
\left(
a_{e,\ell},
0
\right)
```

とする．

その総和を

```math
Z_{e}
:=
\sum_{\ell=1}^{L_e}
a_{e,\ell}^{+}
```

とする．

$Z_e>0$ の場合，

```math
w_{e,\ell}
=
\frac{a_{e,\ell}^{+}}
{Z_e}
```

と正規化する．

全ての raw weight が 0 以下である場合，これは確率分布ではなく zero measure であるため，`ZERO_MASS` として扱う．

identity transform，原点 Dirac，uniform distribution のいずれにも読み替えない．

---

## 13. 入力が確率 law の場合

ここまでは入力点 $z$ を固定していた．

入力点自体が確率測度

```math
\nu
\in
\mathcal{P}
\left(
\mathbb{R}^{3}
\right)
```

に従う場合，edge kernel の作用を

```math
\left(
\mathcal{K}_{e}
\nu
\right)
(D)
:=
\int_{\mathbb{R}^{3}}
\mathcal{K}_{e}(z,D)
\nu(dz)
```

と定義する．

$\nu$ は変換前の点の law，$\mathcal{K}_{e}\nu$ は変換後の点の law である．

入力が決定論的な点 $z_0$ の場合は，

```math
\nu
=
\delta_{z_0}
```

なので，

```math
\left(
\mathcal{K}_{e}
\delta_{z_0}
\right)
(D)
=
\mathcal{K}_{e}(z_0,D)
```

である．

---

## 14. kernel composition

二つの edge kernel $\mathcal{K}_1$，$\mathcal{K}_2$ を順に通る場合，kernel composition を

```math
\left(
\mathcal{K}_{2}
\circ
\mathcal{K}_{1}
\right)
(z,D)
:=
\int_{\mathbb{R}^{3}}
\mathcal{K}_{2}(y,D)
\mathcal{K}_{1}(z,dy)
```

と定義する．

意味は次の通りである．

1. 入力点 $z$ を第1 edge に通し，中間点 $y$ の law を得る．
2. 中間点 $y$ を第2 edge に通す．
3. 最終出力が $D$ に入る確率を，中間点 $y$ について積分する．

path

```math
P
=
\left(
e_1,
\ldots,
e_n
\right)
```

に対して，

```math
\mathcal{K}_{P}
=
\mathcal{K}_{e_n}
\circ
\cdots
\circ
\mathcal{K}_{e_1}
```

とする．

入力 law $\nu_{\mathrm{in}}$ に対する source–target lookup の出力 law は

```math
\nu_{\mathrm{out}}
=
\mathcal{K}_{P}
\nu_{\mathrm{in}}
```

である．

これが Prob-TF の path lookup を kernel composition として表す中心的な定義である．

---

## 15. JMAA の結果が使われる場所

kernel の内部には

```math
R(Q) z
```

が現れる．

$z \neq 0$ の場合，

```math
R(Q) z
=
\lVert z \rVert
R(Q)
\frac{z}
{\lVert z \rVert}
```

である．

したがって，JMAA で扱った

```math
R(Q) v,
\quad
v
\in
\mathbb{S}^{2}
```

の induced law を，非零ベクトル $z$ に対してそのまま利用できる．

Prob-TF の基本演算は，

> Bingham 分布に従う quaternion によって，点または方向を回転したときの induced law

である．

moment，tangent Gaussian approximation，Wang 型の局所 covariance propagation は，必要に応じて比較・近似・summary のために利用できるが，Prob-TF の一次的な定義ではない．

---

## 16. inverse edge

forward edge が

```math
z_{p}
=
R(Q) z_{c}
+
X
```

である場合，inverse action は

```math
z_{c}
=
R(Q)^{\top}
\left(
z_{p}
-
X
\right)
```

である．

inverse view に用いる $Q$ と $X$ は，forward edge と同じ latent variables である．

inverse 用に独立な Bingham 分布と Gaussian 分布を新たに生成してはならない．

一般には，inverse kernel は元の Bingham–conditional-Gaussian family に単純には閉じない．そのため，inverse は同一 physical edge の lazy view として保持する．

---

## 17. 記号一覧

| 記号 | 意味 |
|---|---|
| $Q$ | ランダムな unit quaternion |
| $q$ | $Q$ の実現値 |
| $R(q)$ | quaternion $q$ に対応する回転行列 |
| $X$ | ランダムな並進 |
| $z$ | 変換前の入力点 |
| $Y$ | 変換後の点 $R(Q)z+X$ |
| $A$ | Bingham parameter matrix |
| $\mathfrak{B}(A)$ | Bingham 確率測度 |
| $q_{\mathrm{ref}}$ | coupling の基準 quaternion |
| $\rho(q)$ | $\operatorname{vec}R(q)$ |
| $\rho_{\mathrm{ref}}$ | $\operatorname{vec}R(q_{\mathrm{ref}})$ |
| $m$ | 基準姿勢に条件づけた並進平均 |
| $B$ | 局所回転変数 $u$ から条件付き並進平均への coupling |
| $C$ | $\operatorname{vec}R$ 差分から条件付き並進平均への coupling |
| $S$ | 姿勢に条件づけた後にも残る residual covariance |
| $D_{\mathrm{rot}}$ | 局所回転変数から $\operatorname{vec}R$ への微分 |
| $D$ | 出力空間の可測集合 |
| $y$ | $D$ 上の積分変数 |
| $\vartheta$ | $(m,C,S,\rho_{\mathrm{ref}})$ の略記 |
| $G$ | $Q=q$ の下で出力点が $D$ に入る条件付き Gaussian 確率 |
| $\mathcal{K}$ | 単一 Bingham component の kernel |
| $w_{e,\ell}$ | edge mixture の第 $\ell$ component weight |
| $\mathcal{K}_{e}$ | edge 全体の mixture kernel |
| $\nu$ | 入力点の確率 law |
| $\mathcal{K}_{e}\nu$ | edge により変換された出力 law |
| $\mathcal{K}_{P}$ | path 上の kernel composition |

集合を表す $D$ と，回転埋め込みの微分を表す行列が衝突しないよう，本書では後者を $D_{\mathrm{rot}}$ と書いた．

---

## 18. 最終的な理解

このモデルを理解する順序は，次の通りである．

1. $Q$ を Bingham 分布で表す．
2. $Q=q$ に条件づけた並進 $X$ を Gaussian として表す．
3. 局所二次近似を使う場合，平方完成から $m$，$B$，$S$ を得る．
4. $B$ を $\operatorname{vec}R$ 表現へ移し，$C$ を得る．
5. $Q=q$ に条件づけた出力点の確率を $G$ として定義する．
6. $q$ を Bingham 測度で積分し，単一 component kernel $\mathcal{K}$ を得る．
7. mixture component を足し合わせ，edge kernel $\mathcal{K}_e$ を得る．
8. 入力 law に $\mathcal{K}_e$ を作用させる．
9. path 上では edge kernel を合成する．
10. 必要な場合に限り，結果の moment や近似分布を計算する．

最も重要な区別は次である．

> $m$，$C$，$S$ は条件付きモデルのパラメータであり，その定義段階では $Q$ に関する周辺期待値を取っていない．

局所二次近似から構成する場合，

```math
m
=
x_{\mathrm{ref}}
```

```math
B
=
-
H_{xx}^{-1}
H_{xu}
```

```math
S
=
H_{xx}^{-1}
```

```math
C
=
B
D_{\mathrm{rot}}^{+}
```

である．

その後で必要なら，

```math
\mathbb{E}[X]
=
m
+
C
\left(
\mathbb{E}[\rho(Q)]
-
\rho_{\mathrm{ref}}
\right)
```

を計算する．

したがって，周辺期待値の式は $m$ の定義ではなく，条件付きモデルから導かれる後段の帰結である．
