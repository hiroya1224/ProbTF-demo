# \(\operatorname{vec} R\) を用いた位置–姿勢カップリングの数学的取り扱い

## 1. 目的

本ノートでは，剛体変換の位置成分と姿勢成分の間に存在する依存関係を，混合分布の各成分ごとに

\[
\operatorname{vec} R(Q)-\operatorname{vec} R_\ell
\]

を用いて表現する方法を整理する．

ここで重要なのは，この差分を \(SO(3)\) 上の群差と解釈しないことである．これは \(SO(3)\) を行列空間へ埋め込んだ後の外在的な差分であり，各 mixture component の mode 近傍における局所一次情報を公開パラメータとして表現するために用いる．

以下では，画像処理や特定のセンサモデルには立ち入らず，確率分布としての数学的構造にのみ焦点を当てる．

---

## 2. 確率変数と混合分布

位置の確率変数を

\[
X \in \mathbb{R}^3
\]

とし，姿勢を表す単位 quaternion の確率変数を

\[
Q \in \mathbb{S}^3
\]

とする．\(Q\) と \(-Q\) は同じ回転を表すものとし，対応する回転行列を

\[
R(Q) \in SO(3)
\]

と書く．

混合成分を表す離散確率変数を

\[
L \in \{1,\ldots,K\}
\]

とし，

\[
\mathbb{P}(L=\ell)=w_\ell,
\qquad
w_\ell \geq 0,
\qquad
\sum_{\ell=1}^{K}w_\ell=1
\]

とする．

各 mixture component \(\ell\) は，位置と姿勢の joint distribution の一つの局所的な mode を表す．

---

## 3. 各 mixture component の分布モデル

第 \(\ell\) 成分の姿勢分布を Bingham 分布

\[
Q \mid L=\ell
\sim
\operatorname{Bing}(A_\ell)
\]

で表す．

この成分の代表 quaternion を \(q_\ell\) とし，対応する代表回転を

\[
R_\ell := R(q_\ell)
\]

とする．通常は \(q_\ell\) を Bingham 成分の mode とする．\(q_\ell\) と \(-q_\ell\) は同じ \(R_\ell\) を与えるため，以下の表現は quaternion の符号選択に依存しない．

行列の vectorization を

\[
\rho(q):=\operatorname{vec}R(q)\in\mathbb{R}^9
\]

と定義し，

\[
\rho_\ell:=\operatorname{vec}R_\ell
\]

と置く．本ノートでは \(\operatorname{vec}\) は列方向に成分を積み重ねるものとする．実装では行方向を用いてもよいが，全ての式で同じ規約を使用しなければならない．

位置の条件付き分布を

\[
X \mid Q=q,\ L=\ell
\sim
\mathcal{N}
\left(
m_\ell+
C_\ell\bigl(\rho(q)-\rho_\ell\bigr),
S_\ell
\right)
\]

と定める．ここで

\[
m_\ell\in\mathbb{R}^3,
\qquad
S_\ell\in\mathbb{R}^{3\times 3},
\qquad
C_\ell\in\mathbb{R}^{3\times 9}
\]

は確率変数ではなく，第 \(\ell\) 成分を指定する固定パラメータである．

このモデルは生成的には

\[
X
=
m_\ell+
C_\ell\bigl(\rho(Q)-\rho_\ell\bigr)
+\varepsilon_\ell,
\qquad
\varepsilon_\ell\sim\mathcal{N}(0,S_\ell)
\]

と書ける．

したがって，

\[
\mathbb{E}[X\mid Q=q,L=\ell]
=
m_\ell+
C_\ell\bigl(\rho(q)-\rho_\ell\bigr)
\]

である．特に \(q=q_\ell\) では

\[
\mathbb{E}[X\mid Q=q_\ell,L=\ell]=m_\ell
\]

となる．

このため，\(m_\ell\) は第 \(\ell\) 成分における位置の周辺平均ではなく，**代表姿勢 \(q_\ell\) に条件づけた位置平均**である．

---

## 4. \(\operatorname{vec}R(q)-\operatorname{vec}R_\ell\) の意味

差分

\[
\rho(q)-\rho_\ell
=
\operatorname{vec}R(q)-\operatorname{vec}R_\ell
\]

は \(SO(3)\) 上の群差ではない．

これは埋め込み

\[
SO(3)\hookrightarrow\mathbb{R}^{3\times 3}\simeq\mathbb{R}^9
\]

の後に取ったユークリッド差であり，外在的な chordal difference である．

したがって，この量を回転ベクトル，角速度，あるいは大域的な回転誤差と解釈してはならない．本モデルでは，各 mixture component の代表回転 \(R_\ell\) の近傍において，姿勢の局所変位を \(\mathbb{R}^9\) に埋め込んで表すために用いる．

---

## 5. 局所二次近似から現れる本来のカップリング

第 \(\ell\) 成分の joint density が \((x_\ell,R_\ell)\) に局所 mode を持つとする．

\(R_\ell\) の近傍の回転を，右摂動の局所変数 \(u\in\mathbb{R}^3\) により

\[
R_\ell(u)
=
R_\ell\exp([u]_\times)
\]

と表す．ここで \([u]_\times\) は \(u\) に対応する \(3\times3\) 歪対称行列である．

位置について

\[
x=x_\ell+\delta x
\]

と書き，負の対数密度を mode 近傍で二次近似すると，

\[
-\log p(x,R_\ell(u))
\approx
\mathrm{const.}
+
\frac{1}{2}
\begin{pmatrix}
\delta x\\
u
\end{pmatrix}^{\top}
\begin{pmatrix}
H_{xx,\ell} & H_{xu,\ell}\\
H_{ux,\ell} & H_{uu,\ell}
\end{pmatrix}
\begin{pmatrix}
\delta x\\
u
\end{pmatrix}
\]

となる．

\(H_{xx,\ell}\) が正定値であるとする．\(\delta x\) に関して平方完成すると，

\[
X\mid U=u,\ L=\ell
\approx
\mathcal{N}
\left(
x_\ell+B_\ell u,
S_\ell
\right)
\]

を得る．ただし，

\[
B_\ell
:=
-H_{xx,\ell}^{-1}H_{xu,\ell},
\qquad
S_\ell
:=
H_{xx,\ell}^{-1}
\]

である．

この

\[
B_\ell\in\mathbb{R}^{3\times3}
\]

が，局所二次近似から直接現れる本来の位置–姿勢カップリングである．

したがって，カップリング自体は任意に追加されたヒューリスティックではない．joint density の交差 Hessian \(H_{xu,\ell}\) を条件付き平均の形に書き直したものである．

---

## 6. 局所カップリングから \(\operatorname{vec}R\) 表現への変換

### 6.1 埋め込みの微分

標準基底を \(e_1,e_2,e_3\) とし，

\[
E_i:=[e_i]_\times
\]

と置く．

写像

\[
u\longmapsto
\operatorname{vec}\left(
R_\ell\exp([u]_\times)
\right)
\]

の \(u=0\) における微分を

\[
D_\ell
:=
\begin{bmatrix}
\operatorname{vec}(R_\ell E_1) &
\operatorname{vec}(R_\ell E_2) &
\operatorname{vec}(R_\ell E_3)
\end{bmatrix}
\in\mathbb{R}^{9\times3}
\]

とする．

すると，

\[
\operatorname{vec}R_\ell(u)-\operatorname{vec}R_\ell
=
D_\ell u+O(\|u\|^2)
\]

である．

\(D_\ell\) は rank \(3\) を持つ．したがって，mode 近傍では \(\operatorname{vec}R\) の差分から局所変数 \(u\) の一次情報は失われない．

### 6.2 \(C_\ell\) が満たすべき条件

\(\operatorname{vec}R\) による条件付き平均

\[
m_\ell+
C_\ell
\left(
\operatorname{vec}R(q)-\operatorname{vec}R_\ell
\right)
\]

を，局所二次近似から得られた

\[
x_\ell+B_\ell u
\]

と一次で一致させるには，

\[
C_\ell D_\ell=B_\ell
\]

を満たせばよい．

この等式が \(C_\ell\) の数学的な定義条件である．

---

## 7. \(C_\ell\) の非一意性と標準的な選択

\(D_\ell\) は \(9\times3\) 行列であるため，

\[
C_\ell D_\ell=B_\ell
\]

を満たす \(3\times9\) 行列 \(C_\ell\) は一般に一意ではない．

この非一意性は，\(\mathbb{R}^9\) のうち実際に \(SO(3)\) の接方向として使われる部分が3次元しかないことに由来する．\(C_\ell\) の法方向への作用は，局所一次モデルには影響しない．

一意な規約を与えるため，最小 Frobenius ノルム解

\[
C_\ell
=
B_\ell D_\ell^{+}
\]

を採用するのが自然である．ここで \(D_\ell^{+}\) は Moore–Penrose 擬似逆である．

列方向 vectorization と右摂動の規約では，

\[
D_\ell^\top D_\ell=2I_3
\]

が成り立つ．実際，

\[
\left\langle
R_\ell E_i,
R_\ell E_j
\right\rangle_{\mathrm{F}}
=
\left\langle
E_i,E_j
\right\rangle_{\mathrm{F}}
=
2\delta_{ij}
\]

である．したがって，

\[
D_\ell^{+}
=
\frac{1}{2}D_\ell^\top
\]

であり，

\[
C_\ell
=
\frac{1}{2}B_\ell D_\ell^\top
\]

と明示的に書ける．

この選択では，

\[
C_\ell
=
C_\ell P_\ell,
\qquad
P_\ell:=D_\ell D_\ell^{+}
\]

が成り立つ．\(P_\ell\) は \(\operatorname{vec}R_\ell\) における \(SO(3)\) の接空間像への直交射影である．

したがって，\(C_\ell\) は \(\mathbb{R}^9\) 全体に任意の作用を持つのではなく，接方向に必要な作用だけを持つ標準形として固定される．

---

## 8. 情報が保持される範囲

### 8.1 局所一次情報

各 mixture component の mode 近傍では，

\[
\operatorname{vec}R(q)-\operatorname{vec}R_\ell
=
D_\ell u+O(\|u\|^2)
\]

であり，\(D_\ell\) は rank \(3\) を持つ．

したがって，

\[
C_\ell D_\ell=B_\ell
\]

を満たす \(C_\ell\) を用いれば，局所二次近似から得られる位置–姿勢 coupling の一次情報は欠落しない．

### 8.2 二次以上の情報

一方，

\[
\operatorname{vec}R(q)-\operatorname{vec}R_\ell
\]

は局所変数 \(u\) に対して非線形であり，

\[
D_\ell u
\]

との差には \(O(\|u\|^2)\) の項がある．

したがって，本モデルは joint density の任意の高次 coupling を完全に保持するものではない．局所 Laplace 近似が保持する二次形式のうち，条件付き平均に現れる一次 coupling を公開形式へ移したものと理解するべきである．

必要であれば，将来的に

\[
u_i u_j
\]

または

\[
\bigl(\rho(q)-\rho_\ell\bigr)
\bigl(\rho(q)-\rho_\ell\bigr)^\top
\]

に依存する二次項を追加できるが，標準モデルでは扱わない．

---

## 9. 密度と正規化

第 \(\ell\) 成分の Bingham 密度を \(b_{A_\ell}(q)\) とし，3次元 Gaussian 密度を \(\varphi_3\) とする．

各成分の joint density は

\[
p_\ell(x,q)
=
b_{A_\ell}(q)
\,
\varphi_3
\left(
x;
m_\ell+
C_\ell\bigl(\rho(q)-\rho_\ell\bigr),
S_\ell
\right)
\]

である．

各 \(q\) に対して Gaussian 密度は \(x\) について積分すると1であり，Bingham 密度も正規化されているため，

\[
\int_{\mathbb{S}^3}
\int_{\mathbb{R}^3}
p_\ell(x,q)
\,dx\,d\sigma(q)
=
1
\]

である．

したがって，全体の混合密度

\[
p(x,q)
=
\sum_{\ell=1}^{K}
w_\ell p_\ell(x,q)
\]

も正規化される．

また，

\[
R(q)=R(-q)
\]

および Bingham 密度の antipodal symmetry により，

\[
p_\ell(x,q)=p_\ell(x,-q)
\]

が成り立つ．したがって，この分布は自然に

\[
\mathbb{R}^3\times SO(3)
\]

上の分布として解釈できる．

---

## 10. 平均，共分散，交差共分散

\[
\rho(Q)=\operatorname{vec}R(Q)
\]

とする．第 \(\ell\) 成分について，

\[
\bar{\rho}_\ell
:=
\mathbb{E}[\rho(Q)\mid L=\ell]
\]

と置くと，

\[
\mathbb{E}[X\mid L=\ell]
=
m_\ell+
C_\ell(\bar{\rho}_\ell-\rho_\ell)
\]

である．

したがって，\(\rho_\ell\) に mode の \(\operatorname{vec}R_\ell\) を用いる場合，\(m_\ell\) は一般には位置の周辺平均ではない．

さらに，

\[
\Sigma_{\rho,\ell}
:=
\operatorname{Cov}
\bigl(
\rho(Q)\mid L=\ell
\bigr)
\]

と置けば，

\[
\operatorname{Cov}(X\mid L=\ell)
=
S_\ell+
C_\ell
\Sigma_{\rho,\ell}
C_\ell^\top
\]

である．

また，

\[
\operatorname{Cov}
\left(
X,\rho(Q)\mid L=\ell
\right)
=
C_\ell\Sigma_{\rho,\ell}
\]

である．

この分解により，

- \(S_\ell\)：姿勢を条件づけた後にも残る位置不確かさ
- \(C_\ell\Sigma_{\rho,\ell}C_\ell^\top\)：姿勢不確かさから誘導される位置不確かさ

を分離して解釈できる．

---

## 11. 局所座標規約への依存

導出で用いた

\[
R_\ell(u)=R_\ell\exp([u]_\times)
\]

は右摂動の規約である．左摂動

\[
R_\ell(u)=\exp([u]_\times)R_\ell
\]

を用いる場合，\(D_\ell\) と局所 coupling \(B_\ell\) の双方が変わる．

したがって，

\[
B_\ell,
\qquad
D_\ell,
\qquad
C_\ell
\]

を計算するときは，局所座標の規約を統一しなければならない．

一方，最終的に公開される条件付き平均

\[
m_\ell+
C_\ell
\left(
\operatorname{vec}R(q)-\operatorname{vec}R_\ell
\right)
\]

は，規約に従って正しく変換された \(C_\ell\) を用いる限り，局所座標 \(u\) 自体を外部へ公開する必要がない．

---

## 12. 各パラメータの意味

第 \(\ell\) mixture component は，少なくとも次のパラメータで指定される．

\[
w_\ell
\]

は mixture weight である．

\[
A_\ell
\]

は姿勢の Bingham 分布を指定する．

\[
q_\ell
\]

は第 \(\ell\) 成分の代表 quaternion であり，通常は mode とする．

\[
R_\ell=R(q_\ell),
\qquad
\rho_\ell=\operatorname{vec}R_\ell
\]

は coupling の基準姿勢を与える．

\[
m_\ell
\]

は \(Q=q_\ell\) に条件づけた位置平均である．

\[
S_\ell
\]

は姿勢を条件づけた後の位置共分散である．

\[
C_\ell
\]

は姿勢の外在的変位

\[
\operatorname{vec}R(q)-\operatorname{vec}R_\ell
\]

を位置の条件付き平均の変位へ写す固定線形写像である．

局所二次近似から構成する場合，

\[
B_\ell=-H_{xx,\ell}^{-1}H_{xu,\ell}
\]

を計算し，

\[
C_\ell=B_\ell D_\ell^{+}
\]

と定める．

---

## 13. 標準仕様としての推奨形

各 mixture component に対して，次の分布を標準形とする．

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
\right).
\]

ここで，

\[
R_\ell=R(q_\ell)
\]

は第 \(\ell\) 成分固有の代表回転である．

さらに，\(C_\ell\) は接方向以外への任意性を除くため，

\[
C_\ell=C_\ell P_\ell
\]

を満たす最小ノルム標準形に固定するのが望ましい．局所二次近似から得た \(B_\ell\) に対しては，

\[
C_\ell=B_\ell D_\ell^{+}
\]

とする．

---

## 14. まとめ

\[
\operatorname{vec}R(q)-\operatorname{vec}R_\ell
\]

は \(SO(3)\) 上の群差ではなく，回転行列を \(\mathbb{R}^9\) に埋め込んだ後の外在差分である．

各 mixture component の mode 近傍では，

\[
\operatorname{vec}R(q)-\operatorname{vec}R_\ell
=
D_\ell u+O(\|u\|^2)
\]

であり，\(D_\ell\) は rank \(3\) を持つ．したがって，局所一次の姿勢情報は保持される．

joint density の局所二次近似からは，

\[
B_\ell=-H_{xx,\ell}^{-1}H_{xu,\ell}
\]

という位置–姿勢 coupling が現れる．これを \(\operatorname{vec}R\) 表現に移すには，

\[
C_\ell D_\ell=B_\ell
\]

を要求する．標準的には，

\[
C_\ell=B_\ell D_\ell^{+}
\]

を採用する．

したがって，\(C_\ell\) は恣意的な補正項ではなく，局所 Laplace 近似の交差 Hessian から得られる coupling を，局所座標そのものに依存しない公開表現へ移した固定パラメータである．
