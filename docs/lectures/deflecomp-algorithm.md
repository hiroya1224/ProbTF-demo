# Deflecomp の数理・実装仕様と妥当性監査

更新日: 2026-07-13  
対象: `deflecomp_frames.launch` から起動される現行 working tree  
基準: `HEAD=8c10793` に本作業中の推定・追従修正を加えた状態

## 0. この文書の目的と結論

本書は、柔軟関節ロボットの重力たわみ補償器 `deflecomp` について、実装されている数式、ROS 上の因果関係、推定器、シミュレータ、可視化の意味、および科学的に許される主張の範囲を一つの仕様として固定するものである。理想化を隠した性能説明にならないよう、アルゴリズムの説明と実装監査を分離せず記す。

監査の結論は次のとおりである。

1. 補償器がシミュレータの `K_true`、`/equil/joint_states`、未来の参照値を直接読む経路はない。したがって、ground-truth の直接漏洩は確認されなかった。
2. 赤色の `cmd` は物理姿勢ではない。補償器が計算したばね基準角 setpoint を、剛体 URDF の関節角だと仮定して描画した反実仮想の姿勢である。赤の時間追従性は、実プラントの追従性能を表さない。
3. 2026-07-11 の変更 `d281b1f` 以降、指令低域通過の後に無制限の逆平衡 Newton 補正を常時適用していた。この処理は `theta_cmd_tau` と不可観測方向の feedforward 制限をほぼ完全に迂回していたため、本監査で修正した。
4. 修正後は、任意の逆平衡補正を raw 指令生成段へ移し、既定で無効にした。低域通過後の送信指令は一切書き換えず、`theta_eq_hat` は実際に送る指令から読み取り専用で予測する。
5. WEKF は `R_base,frame` の quaternion に body tangent と WORLD 角速度 Jacobian を混在させていた。Yamaguchi 実 FK の有限差分に対する相対誤差は姿勢・frame により約 `0.34--1.35` だった。base 表現の相対角速度と spatial tangent の組へ修正し、誤差を $10^{-9}$ 程度まで下げた。この欠陥は WEKF の勾配だけでなく supervisor の探索方向も誤らせ得た。
6. IMU endpoint hold を query time で再刻印し、同じ sensor sample を独立観測として 50 Hz で反復投入できる欠陥があった。元 sample の support timestamp を保持し、frame ごとに未処理の sample だけを WEKF/supervisor へ渡すよう修正した。
7. simulator は無制限速度で $q_{k+1}$ を積分した後、返却する $\dot q_{k+1}$ だけを clip していた。設定上は $4\,\mathrm{rad/s}$ でも 1 ms に `0.2--0.3 rad` 跳ぶ再現例があり、発散や branch jump の直接原因になり得た。速度制限を実際の状態遷移へ適用するよう修正した。
8. controller と simulator は、同じ URDF、同じ Pinocchio、同じ重力、同じばね実装、同じ IMU 外部標定を共有する。これは数式・配線の整合試験には適切だが、推定性能や実機性能を検証するには強い matched-model 条件、いわゆる inverse crime である。
9. 現行の剛性推定器は通常の measurement-residual EKF ではなく、Bingham 方向尤度を `x=log K` の周りで局所二次近似する Laplace/Gauss--Newton 型更新である。
10. `particle supervisor` は particle filter、RBPF、厳密な MAP 推定ではない。事前分布を score に含めない、決定論的・有界バッチ最尤 proposal である。通常 launch では既定で無効である。
11. 追加監査で、`project_unobservable_feedforward` が姿勢依存部分空間への参照増分を逐次積分するため経路依存に漂い、既知の目標姿勢とは別の姿勢で重力を評価していたことを確認した。実測静止例ではこれが重力トルクの符号を反転させ、真値プラントの目標姿勢誤差を `0.00413 rad` から `0.689 rad` へ悪化させていた。これは既定無効へ変更した。
12. Pinocchio の `centerOfMass()` が universe に固定された base inertia を集計しない一方、旧コードはそれを含む全質量を COM に掛けていた。Yamaguchi ではポテンシャル勾配が一般化重力の `1.96556` 倍になっていたため、`computePotentialEnergy()` へ統一した。
13. joint limit 上の平衡を raw torque residual の最小二乗で精密化していた処理を廃止し、同じ物理ポテンシャルの box-constrained KKT 解へ統一した。感度も active joint の $dq_i/dx=0$ を含む active-set 系へ変更した。
14. WEKF の局所 Laplace step は exact likelihood を悪化させても全量適用されていた。さらに、非悪化だけを要求しても `A_param=1000` では一度に $\|\Delta\log K\|_\infty=2.93$、$\|\Delta q_{eq}\|_2=0.562\,\mathrm{rad}$ 動く別 branch 候補を受理できた。Gaussian prior を含む exact posterior objective の最大 16 回 backtracking に加え、$\|\Delta\log K\|_\infty\le0.25$ と $\|\Delta q_{eq}\|_2\le0.10\,\mathrm{rad}$ の trust region を追加した。
15. command は実 publish 時刻付き履歴として保存し、全 IMU の最新共通時刻に有効だった command と対応させる。command と reference が 0.5 s 安定した準静的 record だけを更新へ通す。

したがって、現在のシミュレーションは「内部モデルが自己整合しているか」を調べる baseline には使えるが、marker 動画だけから実機追従、動的オンライン同定、モデル誤差に対する頑健性を主張してはならない。

## 1. 信号、記号、marker の意味

### 1.1 主な記号

| 記号 | 実装名 | 意味 |
|---|---|---|
| $q_r$ | `theta_ref` | ユーザが与える参照リンク関節角 |
| $u$ | `theta_cmd` | 補償器が publish するばね基準角またはモータ側 setpoint |
| $u_{raw}$ | `theta_cmd_raw` | 低域通過前の補償指令 |
| $u_{eff}$ | `q_ref_eff` | simulator 内部の slew/一次遅れ後のばね基準角。現在は publish されない |
| $q$ | simulator の `q` | 柔軟関節プラントのリンク側状態 |
| $\hat q_{eq}$ | `theta_eq_hat` | controller の内部モデルが、実際に送る $u$ から予測した静的平衡 |
| $K_{true}$ | `kp_true` | simulator だけが保持するプラント剛性 |
| $x$ | `x_est` | 推定状態。$x=\log K$ |
| $K_{est}$ | `kp_est` | 推定器の直近値 $\exp x$ |
| $K_{exec}$ | `kp_exec` | 指令生成に実際に使う、平滑化済み剛性 |
| $\tau_g(q)$ | `tau_gravity` | URDF/Pinocchio から得る一般化重力トルク |
| $\tau_s(q,u,K)$ | spring `torque` | 本書で定義するばね残差トルク |

### 1.2 三つの色は三つの物理状態ではない

| 色・prefix | topic | 実体 | 性能評価への使用 |
|---|---|---|---|
| 青 `ref` | `/ref/joint_states` | 参照 $q_r$ | 目標として使用する |
| 赤 `cmd` | `/cmd/joint_states` | 要求 setpoint $u$ を剛体 FK したもの | プラント追従評価には使用不可 |
| 緑 `equil` | `/equil/joint_states` | `eq_mode: dynamic` では動的 simulator state $q$ | 現 simulator 内の追従評価対象 |

`equil` という topic 名は歴史的名称である。現在の `eq_mode: dynamic` では、緑は毎時刻の静的平衡解ではなく、質量・重力・ばね・減衰を積分した動的状態である。

赤はさらに simulator 内部の $u_{eff}$ より前の値である。`robot_state_publisher` は `/cmd/joint_states` の数値をリンク側関節角だと解釈するが、本モデルでの $u$ はばねの自然角であり、一般に $q \ne u$ である。したがって、赤いロボット形状は「剛性が無限大ならこの姿勢になる」という補助表示にすぎない。

### 1.3 実装されたデータフロー

```text
/ref/joint_states ───────────────┐
                                 │
/imu ── quality gate ── buffer ──┼─> stiffness estimator ─> K_est
                                 │                           │
                                 │                           v
                                 │                    log-space smoothing
                                 │                           │
                                 v                           v
                         inverse statics <─────────────── K_exec
                                 │
                          optional raw refine
                                 │
                           command low-pass
                                 │
                  /cmd/joint_states = sent command u
                         │                    │
                         │                    └─> read-only equilibrium prediction q_eq_hat
                         v
                 simulator command shaping u_eff
                         │
                    dynamics q, qdot
                         │
             /equil/joint_states and synthetic /imu
```

controller は `/equil/joint_states` を subscribe しない。緑は評価用の simulator 出力であり、制御器への直接 feedback ではない。実姿勢の情報は、合成または実 IMU の重力方向を通じてのみ間接的に入る。

## 2. ロボットモデル

### 2.1 URDF と reduced model

`RobotArm` は URDF を Pinocchio model へ変換する。mimic joint、fixed joint、velocity limit が 0 の joint など、`urdf_info` が非制御と判定した joint は 0 rad で lock し、reduced model を作る。Yamaguchi 6 軸では $n_q=n_v=6$ で、全 active joint は 1-DoF である。実装は configuration vector の長さにも一貫して `model.nv` を用い、1 Pinocchio joint に 1 ROS joint 名を対応させるため、一般の $n_q\ne n_v$、spherical joint、multi-DoF joint、floating base には対応しない。ROS publish 時だけ full movable-joint list へ展開する。

active でない joint の値は、controller では最新の reference、simulator では最新の command を fallback としてコピーする。このコピーは active 6 軸の物理シミュレーションではないため、非制御 gripper 等を含む marker を定量評価してはならない。

現実装は固定基台を仮定し、Pinocchio world と robot base の重力方向が一致するものとしている。浮遊基台や base の未知傾斜は状態に含まれない。

### 2.2 重力

一般化重力トルクを

```math
\tau_g(q)=\operatorname{computeGeneralizedGravity}(q)
```

とする。重力ベクトルは両 node で

```math
g_W=(0,0,-9.81)^\top\;\mathrm{m/s^2}
```

に固定される。重力トルク微分は

```math
G_q(q)=\frac{\partial\tau_g(q)}{\partial q}
```

であり、平衡感度と command refinement に使う。

重力ポテンシャルは

```math
U_g(q)=\operatorname{computePotentialEnergy}(q)
```

を直接用いる。旧実装は `sum(model.inertias.mass)` と `centerOfMass()` を組み合わせていたが、Pinocchio は固定 base inertia を universe へ集約し、`centerOfMass()` の可動系質量から除く。Yamaguchi では全質量 `1.6245677 kg` と可動系 COM 質量 `0.8265163 kg` の比 `1.9655603` がそのまま勾配誤差になった。`computePotentialEnergy()` と `computeGeneralizedGravity()` を同じ library に委ねることで、有限差分勾配との相対誤差は約 `5.6e-9` になった。

## 3. ばねモデル

### 3.1 線形ばね

要素積を $\odot$ とすると、線形モデルは

```math
\tau_s(q,u,K)=K\odot(q-u)
```

```math
V_s(q,u,K)=\frac12(q-u)^\top\operatorname{diag}(K)(q-u)
```

である。局所剛性と log 剛性感度は

```math
\frac{\partial\tau_s}{\partial q}=\operatorname{diag}(K),\qquad
\frac{\partial\tau_{s,i}}{\partial\log K_i}=K_i(q_i-u_i)=\tau_{s,i}
```

となる。

### 3.2 現在の `periodic` ばね

現行 Yamaguchi 設定は controller と simulator の双方で `spring_model: periodic` を選ぶ。実装式は

```math
\delta=q-u
```

```math
\tau_s(q,u,K)=2K\odot\sin\left(\frac{\delta}{2}\right)
```

```math
V_s(q,u,K)=\sum_i4K_i\left[1-\cos\left(\frac{\delta_i}{2}\right)\right]
```

```math
k_{s,i}=\frac{\partial\tau_{s,i}}{\partial q_i}
=K_i\cos\left(\frac{\delta_i}{2}\right)
```

である。

名称とは異なり、この関数の周期は $2\pi$ ではなく $4\pi$ である。実際、

```math
\tau_s(\delta+2\pi)=-\tau_s(\delta)
```

である。したがって、回転関節の物理的な $2\pi$ 周期ばねを忠実に表すとは限らない。また $|\delta|>\pi$ では局所剛性が負になり得る。これは複数平衡、branch jump、平衡感度の特異化、動力学不安定化の原因になる。

### 3.3 静力学平衡

外力なしの平衡残差を

```math
F(q,u,K)=\tau_g(q)+\tau_s(q,u,K)
```

と定義し、

```math
F(q_{eq},u,K)=0
```

を満たす $q_{eq}$ を求める。局所安定性には少なくとも全ポテンシャル Hessian

```math
J_q=\frac{\partial F}{\partial q}
=G_q(q_{eq})+\operatorname{diag}(k_s)
```

の性質が関係するが、現 solver は Hessian 正定値性を採択条件にしていない。したがって「残差が小さいこと」と「物理的に安定な平衡であること」は同値ではない。

## 4. 平衡ソルバ

`EquilibriumSolver` は全ポテンシャル

```math
V_{total}(q)=U_g(q)+V_s(q,u,K)
```

を joint position bounds の下で最小化する。最初から低剛性の非凸問題を解かず、人工剛性を

```math
K_{eff}(\lambda)=K+100\lambda\mathbf 1
```

として、$\lambda=1\rightarrow0$ の 10 段階で L-BFGS-B を行う。各段の解を次段の初期値にする。

`equilibrium_refine: true` のときも、元の $K$ に対する同じ $V_{total}$ を bounds 付き L-BFGS-B で再最小化する。採用条件は、目的関数が悪化せず、box constraint の projected gradient、すなわち KKT residual が改善することである。既定 tolerance は $10^{-12}$、最大反復は 40 である。

ここでいう `equilibrium_refine` は、与えられた $u$ の平衡を数値的に精密化する機能である。後述する `theta_cmd_equilibrium_refine` は、$u$ 自体を変える別機能であり、両者を混同してはならない。

joint limit 上では反力のため active joint の $F_i$ は一般に 0 にならない。下限では $F_i\ge0$、上限では $F_i\le0$ を許し、free joint のみ $F_i=0$ を要求する。感度計算は現在の active set が変わらない局所で、active joint に対応する $J_q$ の行と列を 0 にして対角を 1、$J_x$ の対応行を 0 とし、$dq_i/dx=0$ を課す。列も除くのは、free block が特異なときに Moore--Penrose 解が active joint を使って残差を下げる漏れを防ぐためである。active set が切り替わる点では尤度は非微分になり得るため、通常の Laplace 近似が滑らかであるという保証はない。

周期ばねは非凸なので、平衡 solver は一般には単値写像でない。厳密には、初期値 $q^{(0)}$ と solver 設定 $c$ に依存する branch-continuation operator

```math
q_{eq}=\mathcal Q(u,K;q^{(0)},c)
```

である。同じ $(u,K)$ でも初期値から異なる局所解へ収束し得る。本書で以後 $Q(u,K)$ と略記する箇所は、特に断らない限り「同じ設定と、記載した warm-start から継続される局所 branch」を意味する。$Q$ の微分は、active joint-limit 集合が変わらず、$J_q$ の rank が一定で、同じ滑らかな内点 branch に留まる場合だけ正当化される。

## 5. 補償指令の生成

### 5.1 閉形式 inverse statics

目標 $q_r$ が平衡になる $u$ を、現在の $K_{exec}$ とモデル重力から求める。線形ばねでは

```math
u_{ff}=q_r+\tau_g(q_{eval})\oslash K_{exec}
```

である。ここで $\oslash$ は要素除算、$q_{eval}$ は後述する重力評価姿勢である。

周期ばねでは

```math
u_{ff}=q_r+2\arcsin\left(
\frac{\tau_g(q_{eval})}{2K_{exec}}
\right)
```

を使う。$q_{eval}=q_r$ かつ clip が起きなければ、同じモデル内では構成的に

```math
F(q_r,u_{ff},K_{exec})=0
```

となる。したがって内部予測平衡が参照へ一致すること自体は、推定精度の独立な証拠ではない。

周期ばねの可解条件は各軸で

```math
|\tau_{g,i}(q_{eval})|\le 2K_{exec,i}
```

である。実装は条件違反を error にせず、arcsin の引数を $[-1+10^{-9},1-10^{-9}]$ へ clip する。clip 近傍では $|q-u|\simeq\pi$、局所ばね剛性 $k_s\simeq0$ となるため、平衡感度と逆解が悪条件化する。論文評価では clip 回数を必ず記録すべきである。

### 5.2 $K_{est}$ と $K_{exec}$ の分離

推定器の急変を command へ直結しないため、

```math
x_{exec,target}=x_{est}
```

```math
\Delta x_{exec}
=\left(1-e^{-\Delta t/\tau_K}\right)
(x_{exec,target}-x_{exec})
```

とし、各 cycle で

```math
\Delta x_{exec,i}\in[-0.05,0.05]
```

へ clip する。現設定は $\tau_K=0.5\,\mathrm{s}$ であり、指令生成には

```math
K_{exec}=\exp(x_{exec})
```

を使う。

初期値は simulator 真値ではなく、bounds の対数中点

```math
K_{0,i}=\exp\left(\frac{\log1+\log500}{2}\right)=\sqrt{500}
\simeq22.3607
```

で全軸同一である。Yamaguchi simulator の真値は `[5,5,5,10,20,20]` なので、初期段階では特に第 1--4 軸を実際より硬いと仮定する。このため赤 $u$ が青 $q_r$ に近いことは「良い補償」ではなく、補償量を過小評価している可能性も示す。

### 5.3 重力評価姿勢と、既定無効の可観測方向 projection

通常設定では

```math
q_{eval}=q_r
```

とする。$q_r$ は外部から与えられた既知の目標であって simulator truth ではない。閉形式 inverse statics が $F(q_r,u,K)=0$ を満たすには、ばねの anchor だけでなく重力も同じ $q_r$ で評価しなければならない。

旧実装は、重力方向 IMU が見ない参照変化によって補償値を不用意に変えない意図で、joint 空間の heuristic projection を既定有効にしていた。frame $f$ の重力方向 Jacobianを $J_{g,f}$ とし、

```math
\mathcal G(q_r)=\sum_fJ_{g,f}(q_r)^\top J_{g,f}(q_r)
```

を固有分解する。固有値が

```math
\lambda_i>\max(10^{-10},10^{-4}\lambda_{max})
```

を満たす basis を $U_g$ とする。内部重力評価姿勢は

```math
q_{eval,k}=q_{eval,k-1}
+U_gU_g^\top(q_{r,k}-q_{r,k-1})
```

で更新していた。しかし $U_g(q_r)$ は姿勢で変化するので、この増分積分は endpoint だけで決まらず参照経路に依存する。結果として $q_{eval}$ が $q_r$ から大きく漂い、anchor は $q_r$、重力は別姿勢 $q_{eval}$ という静力学的に不整合な command を生成した。

Yamaguchi の保存静止例では、projection 有効時の command による真値プラントの目標誤差は `0.689172 rad` で、実測緑姿勢を `5.31e-7 rad` で再現した。同じ $K_{exec}$ のまま $q_{eval}=q_r$ とすると、内部平衡誤差は `9.93e-10 rad`、真値プラント誤差は `0.004132 rad` まで低下した。従ってこの大誤差は剛性の非識別性ではなく、projection が inverse-statics の整合条件を破った結果である。

`project_unobservable_feedforward` は互換・研究比較用に残すが既定は `false` である。これは $K$ の Bayesian observability から導かれたものでもないため、通常の補償性能評価には使用しない。

### 5.4 L1 command regularization

`theta_cmd_l1_regularization: true` の場合は、閉形式解を seed として、静力学残差と補償量の smooth-L1 を組み合わせた目的関数を最小化する。概略は

```math
\min_u\frac12\left\|
\frac{\tau_g(q_{eval})+\tau_s(q_r,u,K_{exec})}{K_{exec}}
\right\|_2^2
+\lambda\sum_i\sqrt{(u_i-q_{r,i})^2+\epsilon^2}
```

である。現行 `controller.yaml` では `false` なので、通常 launch では使用しない。

### 5.5 任意の raw-command equilibrium refinement

閉形式以外のばね則や近似解を試験するため、opt-in の
`theta_cmd_equilibrium_refine` を残している。固定した滑らかな branch 上の平衡写像を

```math
Q(u,K;q^{(0)})=\operatorname{EquilibriumSolve}(u,K;q^{(0)})
```

とする。平衡式を $u$ で微分すると、

```math
\frac{\partial Q}{\partial u}
=J_q^\dagger\operatorname{diag}(k_s)=B
```

なので、局所補正を

```math
\Delta u=B^\dagger(q_r-Q(u,K;q^{(0)}))
```

として最大 2 回適用する。修正後の安全策は次のとおりである。

- 既定値は `false`。
- 有効でも $u_{raw}$ の生成段でのみ実行する。
- 前回送信値が存在する通常 cycle では、その後の command low-pass を必ず保持する。初回 cycle は §5.6 の例外を持つ。
- 1 iteration、1 joint 当たりの補正を既定 $\pm0.25\,\mathrm{rad}$ に制限する。
- `max_delta=0` は「無制限」ではなく「補正禁止」と解釈する。
- `project_unobservable_feedforward: true` との同時使用は拒否する。full model で $Q(u,K)=q_r$ を解けば、projection が保持した成分を再構成してしまうためである。
- model derivative の shape、不有限値、例外を検出した場合は $\Delta u=0$ として fail-closed する。関節誤差をそのまま command 補正に流す identity-Jacobian fallback は使用しない。

将来両者を併用するなら、$U_g$ 上に制約した逆問題として再定式化する必要がある。

### 5.6 送信 command の低域通過

前回送信値があるとき、

```math
\alpha_u=1-\exp\left(-\frac{\Delta t}{\tau_u}\right)
```

```math
u_k=u_{k-1}+\alpha_u(u_{raw,k}-u_{k-1})
```

とする。現設定は $\tau_u=0.2\,\mathrm{s}$、controller 周期は $0.02\,\mathrm{s}$ なので nominal 係数は

```math
\alpha_u=1-e^{-0.1}\simeq0.0951626
```

である。最初の cycle は過去値がないため $u=u_{raw}$ となる。startup の物理的 bumpless transfer は別途未実装である。

修正後、低域通過の後に command を変更する処理はない。`theta_eq_hat` は

```math
\hat q_{eq,k}=Q(u_k,K_{exec,k};\hat q_{eq,k-1})
```

として読み取り専用で計算する。

## 6. 発見した明白な問題と修正記録

### 6.1 修正前の問題

修正前は、上の低域通過を実行した直後に、常時有効な `_refine_theta_cmd_for_equilibrium_ref()` が走っていた。

```text
inverse statics -> low-pass -> unconstrained Newton correction -> publish
```

補正は最大 2 回、tolerance $10^{-5}$、`max_delta=0`、すなわち無制限だった。現在の $q_r$ と内部モデルを使って $Q(u,K_{exec})\simeq q_r$ を強制するため、次を同時に破っていた。

- `theta_cmd_tau` が定める時間応答
- 不可観測方向を保持する $q_{eval}$ projection
- simulator の actuator shaping と区別された command 制約
- 「予測は送信 command を説明するだけ」という因果方向

しかも `equilibrium_refine` parameter では無効化できず、hard-coded で常時実行されていた。unit test も `theta_cmd_tau=10 s` であるにもかかわらず $\hat q_{eq}=q_r$ が即時成立することを要求しており、バグを仕様として固定していた。

### 6.2 修正前の再現値

Yamaguchi URDF、$K_0=\sqrt{500}$、$\tau_u=0.2\,\mathrm{s}$、$\Delta t=0.02\,\mathrm{s}$、IMU 更新なしで、参照を 0 から

```text
[ 0.4, -0.4, 0.3, 0.2, -0.5, 0.4 ] rad
```

へ変えた。監査時の再現では、低域通過直後の変化 norm が `0.0874 rad` であるのに、実際に publish された command は `0.9188 rad` 変化した。内部 $\hat q_{eq}$ と $q_r$ の差は `4.67e-8 rad` だった。

別の同条件計測では、LPF output と raw の距離が `0.8314 rad` であるのに、publish 値と raw の距離は `2.3e-5 rad` だった。すなわち本来約 9.5% だけ進むべき cycle で、ほぼ 100% raw へ飛んでいた。

これは `K_true` の覗き見ではない。しかし、同じ内部モデルで作った逆問題を同じ内部モデルで解き、その結果で時間制約を上書きしていたため、表示上の性能を不自然に良くする実装上のチートと判定した。

### 6.3 修正内容

処理順を次へ変更した。

```text
inverse statics
  -> optional bounded raw refinement (default OFF)
  -> command low-pass
  -> publish exactly this command
  -> read-only equilibrium prediction from the published command
```

追加した設定は次である。

| parameter | default | 意味 |
|---|---:|---|
| `theta_cmd_equilibrium_refine` | `false` | raw inverse refinement の opt-in |
| `theta_cmd_equilibrium_refine_maxiter` | `2` | raw Newton 最大反復 |
| `theta_cmd_equilibrium_refine_tol` | `1e-5` | raw 平衡誤差 tolerance |
| `theta_cmd_equilibrium_refine_max_delta` | `0.25` | 1 joint・1反復当たりの最大補正 rad |

`project_unobservable_feedforward: true` のときは、requested であっても raw refinement を skip する。`CompensationStepResult.debug` には `observable_feedforward_projection_enabled` を理由として記録する。ただしこれら command-refinement 固有 field は、現 ROS node の `/deflecomp/debug` 数値 vector には未収録であり、process 内 test 以外からは直接観測できない。

### 6.4 修正後の再現値

同じ Yamaguchi 一段ステップを修正後に実行した結果は次である。

```text
alpha                         = 0.095162581964
||published - expected_LPF|| = 0.0 rad
||published - raw||          = 0.8391117015 rad
||published - previous||     = 0.0882501480 rad
post_filter_correction       = 0.0 rad
||q_eq_hat - q_ref||         = 0.8381913491 rad at t=20 ms
```

`q_eq_hat-ref` が 20 ms 時点で大きいことは失敗ではない。時定数 0.2 s の command 制約を実際に守った結果である。追従性能はこの過渡を含めて緑 $q$ から評価しなければならない。

修正後は、この量を `theta_cmd_sent_equilibrium_error_norm` として内部 debug dictionary に残す。旧 consumer との型互換性のため `theta_cmd_equilibrium_refine_error_norm` にも同じ値を入れるが、意味は旧版の「refinement 後誤差」から「実送信 command の予測追従誤差」へ変わっており、意味互換ではない。また現 ROS `/deflecomp/debug` には両 scalar と `theta_cmd_post_filter_correction_norm` を publish していない。外部監査には schema と Header を持つ stamped diagnostic message が必要である。

回帰 test は、

- LPF の解析値と publish command が一致すること
- equilibrium solver の最終入力が publish command と一致すること
- post-filter correction が 0 であること
- projection 有効時に raw refinement が拒否されること
- opt-in refinement を使っても、それが LPF より前にしか作用しないこと
- derivative が壊れた opt-in refinement が identity 補正へ fallback せず、補正 0 で fail-closed すること

を固定している。

### 6.5 同じ監査で修正した推定・時刻・動力学の欠陥

command overwrite 以外にも、報告された estimator 悪化と simulator 発散へ直接関係する三件を確認した。

1. **WEKF quaternion tangent の座標系混在**: `stiffness_wekf.py` は `R_base,frame` の quaternion に対し、body/local 角速度用の $L(z)_{[:,1:]}$ と WORLD 角速度 Jacobian を乗じていた。勾配・information・supervisor 探索固有方向が誤る。`pinocchio_robot.py` に base に対する相対角速度 Jacobian を追加し、`bingham.py` の spatial tangent $\mathscr R(z)_{[:,1:]}$ と組み合わせた。詳細と有限差分値は §11.2 に記す。
2. **同じ IMU sample の反復更新**: `ImuBuffer` の endpoint hold は最後の sample を返す一方、ROS node は query time で再刻印していた。このため sensor が停止または controller より低 rate のとき、同じ sample を新規 record として繰り返した。buffer が newest supporting sample stamp を返し、`FrameImuObservation.source_stamp` を通じて frame ごとに未処理 sample だけを更新する構造へ変更した。詳細は §8.4 に記す。
3. **velocity clip 後の state 不整合**: `dynamic_simulator.py` は未制限の $\dot q$ で $q_{k+1}$ を積分した後、返却値 $\dot q_{k+1}$ だけを clip していた。速度 limit 発火時に $(q_{k+1}-q_k)/h\ne\dot q_{k+1}$ となった。非 quasistatic mode では state transition 自体を clip 済み step velocity から再構成するよう変更した。詳細は §7.4 に記す。
4. **最初の WEKF update の branch seed**: 前 cycle の実送信 command に対する `last_theta_eq` が既にある場合でも、estimator 自身の履歴が空なら current desired pose を平衡初期値にしていた。非凸 solver を desired branch へ bias し得るため、estimator 履歴、compensator の前回送信平衡、最後に current reference、の順で warm-start するよう修正した。詳細は §13 に記す。

これらは `K_true` の直接漏洩ではないが、前二件は推定精度を人工的に悪化・過信させ、三件目は発散や別平衡 branch への jump を作る実装欠陥である。

## 7. 動的 simulator

### 7.1 連続時間モデル

現在の通常設定は `eq_mode: dynamic` である。リンク側状態を $(q,\dot q)$、内部整形後のばね基準角を $u_{eff}$ とすると、実装された運動方程式は

```math
M(q)\ddot q+b(q,\dot q)+D\dot q
+\tau_s(q,u_{eff},K_{true})=\tau_{ext}
```

である。`b(q, qdot)` は `rnea(q,qdot,0)` であり、Coriolis/centrifugal と重力を含む。静止、外力なしでは

```math
\tau_g(q)+\tau_s(q,u_{eff},K_{true})=0
```

となり、controller の静力学符号と一致する。

motor rotor の独立状態、gear ratio、motor inertia、電流 loop、torque loop は存在しない。$u_{eff}$ はばね自然角として直接与えられ、リンク側だけを一慣性系として積分する。

### 7.2 command shaping

simulator は `/cmd/joint_states` の $u$ を直接ばねへ入れず、最初に component-wise slew limit を適用する。

```math
u_{slew,k}=u_{slew,k-1}
+\operatorname{clip}\left(
u_k-u_{slew,k-1},
-v_{ref,max}\Delta t,
v_{ref,max}\Delta t
\right)
```

次に一次遅れを適用する。

```math
u_{eff,k}=u_{eff,k-1}
+\left(1-e^{-\Delta t/\tau_{ref}}\right)
(u_{slew,k}-u_{eff,k-1})
```

現設定は

```text
ref_max_vel = 4.0 rad/s
ref_tau     = 0.1 s
```

である。赤 marker は $u$ を示し、$u_{slew}$ も $u_{eff}$ も示さない。従って赤と緑の間には、柔軟関節だけでなく未表示の actuator shaping も存在する。

### 7.3 減衰

粘性係数が直接指定されない場合、初期姿勢 $q_0=0$ における質量行列対角を使い、

```math
D_i=2\zeta\sqrt{K_{true,i}M_{ii}(q_0)}
```

とする。現設定は $\zeta=0.9$ である。これは coupled mode の厳密な modal damping ではなく、対角近似である。また $D$ の計算は nominal $K_{true}$ を使い、周期ばねの局所剛性 $K\cos((q-u)/2)$ は使わない。

### 7.4 半陰的積分

積分周期は $h=0.001\,\mathrm{s}$ である。粘性だけを backward Euler とし、他の力を現在状態で評価する。

```math
(M_k+hD)\dot q^*_{k+1}
=M_k\dot q_k+h\left[
\tau_{ext,k}-\tau_s(q_k,u_{eff,k},K_{true})-b(q_k,\dot q_k)
\right]
```

```math
q^*_{k+1}=q_k+h\dot q^*_{k+1}
```

ここで star は velocity-limit projection 前の trial state であり、position active-set がある場合は自由成分に対する式である。軽い wrist inertia に対して explicit damping が数値発散したため、この変更は数値安定化として導入された。静止時の粘性は 0 なので静的平衡そのものを変更しないが、過渡高周波成分を強く散逸させる。

joint limit を越えようとする速度成分は active-set で固定する。非 quasistatic mode の trial state $q^*_{k+1}$ に対して、修正後は

```math
v_{step}=\frac{q^*_{k+1}-q_k}{h},\qquad
\bar v=\operatorname{clip}(v_{step},-v_{lim},v_{lim})
```

```math
q_{k+1}=q_k+h\bar v,\qquad \dot q_{k+1}=\bar v
```

として、返却速度だけでなく実際の位置遷移へ velocity limit を適用する。その後に position bound と外向き速度 0 を適用する。現在の $v_{lim}$ は $4\,\mathrm{rad/s}$ である。

修正前の Yamaguchi 再現では $h=1\,\mathrm{ms}$、$q=\dot q=0$、$D=0$、強い command step に対し、返却速度は $\pm4\,\mathrm{rad/s}$ でも実位置差分は軸により約 `198.9`, `-303.7`, `219.2 rad/s` 相当となり、1 step で `0.2--0.3 rad` 跳んだ。修正後の回帰 test は、joint-limit contact がないとき

```math
\left|\frac{q_{k+1}-q_k}{h}\right|\le v_{lim},\qquad
\dot q_{k+1}=\frac{q_{k+1}-q_k}{h}
```

を固定する。quasistatic mode は nominal equilibrium へ瞬時に移る定義を保ち、velocity limit は synthetic perturbation の $\dot q$ にだけ適用するため、この動的制約の対象外である。joint-limit 接触は完全非弾性的であり、実機 hard stop の反発・摩擦・衝撃を再現しない。

### 7.5 simulator の時計

内部積分は ROS timer callback 1 回につき固定 (h=0.001) s 進む。出力は 100 Hz へ decimate し、header には publish 時の ROS time を付ける。Python callback が実時間 1 kHz を維持できない場合、内部物理時間と header time は一致しない。論文の時間応答評価では `/clock` を伴う deterministic simulation、または実測 callback rate と simulated time の記録が必要である。

### 7.6 simulator が含まない主な現象

- Coulomb friction、stiction、Stribeck effect
- gear backlash、hysteresis、dead zone
- motor/gear/link の二慣性以上の振動
- current/torque/voltage saturation
- command position、速度、加速度、jerk の実機 limit 一式
- cable force、unknown payload、接触力、温度依存性
- encoder quantization、sensor/transport delay、packet loss
- off-diagonal stiffness、構造リンクの分布弾性

外力 topic は実装されているが、通常は 0 である。controller の feedforward は外力を推定せず、重力だけを補償する。

## 8. 合成 IMU と ROS 前処理

### 8.1 合成 IMU

simulator は真の $(q,\dot q,\ddot q)$ から Pinocchio で frame kinematics を計算する。IMU が link frame 原点から $r$ だけ offset しているとき、point acceleration は概略

```math
a_p=a_o+\dot\omega\times r+\omega\times(\omega\times r)
```

である。IMU frame での specific force は

```math
f_I=R_{WI}^\top(a_W-g_W)
```

として publish する。orientation と angular velocity も publish するが、剛性尤度は orientation field を直接使わない。

現行 dynamic mode では IMU white noise、bias、bias random walk、scale factor、axis nonorthogonality、quantization、saturation、timestamp jitter、dropout、通信遅延を加えない。`qs_noise_std_deg` と `qs_vib_amp_deg` は quasi-static mode の姿勢摂動であり、`eq_mode: dynamic` では無効である。

controller と simulator は同じ IMU YAML を読むため、取付回転 $R_{model,imu}$ と位置 offset は完全に一致する。

### 8.2 重力方向への変換

準静的条件 $a_W\simeq0$ では $f_I\simeq-R_{WI}^\top g_W$ なので、sensor frame の観測重力方向を

```math
\hat g_I=-\frac{f_I}{\|f_I\|}
```

とする。これを既知の外部標定で model frame へ回す。

```math
\hat g_f=R_{model,imu}\hat g_I
```

正規化により加速度の大きさを捨てる。従って剛性推定が利用する主情報は、複数 link における重力方向だけである。gyro は観測 likelihood ではなく、準静的 gate のためだけに使う。

### 8.3 quality gate

次を同時に満たす sample だけを受理する。

```math
\left|\|f_I\|-9.81\right|\le0.75\;\mathrm{m/s^2}
```

```math
\|\omega_I\|\le0.20\;\mathrm{rad/s}
```

棄却時はその frame の buffer を clear し、sample timestamp から 0.25 s の settle time を置く。buffer 内を補間する場合は左右の supporting sample が query time から各 0.10 s 以内でなければ使わない。query が最新 sample より後なら、過去側 endpoint を最大 0.10 s hold できる。最初の sample より前へは外挿しない。

この gate は、動的 specific force を重力と誤認しないために必要である。しかし acceleration norm が偶然 9.81 に近い並進運動や、0.20 rad/s 未満の低速過渡を完全には排除しない。このため現在は sensor gate に加え、command と reference の履歴が 0.5 s の全 window でそれぞれ `1e-3 rad`、`1e-4 rad` 以内に留まることも要求する。隣接 sample 差でなく window endpoint との差を見るので、小さい増分が累積する低速 ramp も静止と誤認しない。

従って現推定器は「動作中の動的推定器」ではなく、「主に動作間の準静的区間で更新する静的 parameter estimator」と記述すべきである。

### 8.4 時刻対応

旧 node は timer callback 開始時刻を command header に入れた後、約 15 ms の推定・指令計算を経て publish していた。一方、次回更新ではその callback 開始時刻の IMU と、callback 末に初めて届いた command を組にした。保存 run の command header age 中央値は `14.906 ms` であり、観測が command 適用より前になる逆因果対応だった。

現在は command を実 publish 直前時刻 $t^u_j$ と値 $u_j$ の piecewise-constant 履歴として保存する。各更新候補では、全 IMU frame が共有できる最新時刻 $t^z_k$ を選び、

```text
u(t^z_k) = latest command whose publish/application time <= t^z_k
```

を履歴から検索する。全 frame がその時刻へ補間可能で、IMU が fresh であり、さらに $[t^z_k-0.5,t^z_k]$ で command と reference が安定している場合だけ $(u(t^z_k),A(t^z_k))$ を WEKF へ渡す。`command_apply_delay` の既定は、publish 直前 stamp 化後は `0.0 s` である。実機で別途計測した transport/application delay がある場合だけ上書きする。

gate 状態は `/deflecomp/estimation_gate_status` に `ready`、`command_not_settled`、`reference_not_settled`、`latest_imu_is_stale` などとして publish する。

同じ status 文字列には、前段 gate だけでなく `est_update_applied`、`est_update_skipped_reason`、`laplace_step_scale`、`laplace_dx_max_abs` も含める。従って `reason=ready` でも WEKF が `no_observable_information` や `exact_posterior_not_improved` で更新しなかった場合を区別できる。ただし `std_msgs/String` で Header を持たないという計測上の制約は残る。

補間値には二種類の時刻を保持する。

- `stamp`: 観測値が表す query/alignment time $t^z_k$。
- `source_stamp`: その値を構成した newest supporting sensor sample の時刻。interior interpolation では右側 $t_1$、endpoint hold では実際の最後の sample 時刻である。

修正前は endpoint hold した同じ値にも毎 cycle 新しい query `stamp` を付けたため、compensator の duplicate-stamp gate を迂回し、同じ測定を最大 `imu_max_age=0.10 s` の間、独立な Bingham evidence として反復投入できた。これは covariance を過度に縮め、supervisor window を重複 record で満たす。

修正後は model frame 名ごとに最後に処理した `source_stamp` を保存し、その frame の support が進んだ観測だけで $A_f$ を構成する。ある frame A が新しく、frame B が endpoint hold のままなら、その cycle の update には A だけを入れ、B を再加算しない。同じ右 support $t_1$ に基づく異なる query-time 補間値も一度だけ使うため保守的である。これは独立性を完全に保証する処理ではなく、隣接 sensor sample の時間相関は posterior model に依然含まれない。

dedupe key は物理 sensor ID でなく `FrameImuObservation.frame_name` である。提示された Yamaguchi YAML は sensor/model frame が一対一なので問題ないが、同一 model frame に複数 IMU を割り当てる一般構成では key が衝突する。その場合は observation に sensor ID を追加して source state を分離しなければならない。

simulator には $\tau_{ref}=0.1\,\mathrm{s}$ と動力学があるが、推定モデルはそれを含まない。0.5 s dwell と IMU gate はこの model mismatch を準静的 record の選別で避ける処理であり、動力学を推定モデルへ組み込んだわけではない。実機の静定時間が長い場合は `estimation_settle_time` を延ばす必要がある。

`CommandDelayRLS` はクラスとして存在するが、現在の ROS node は生成しておらず、この対応誤差を推定していない。

## 9. Bingham 重力方向尤度

観測された model-frame 重力方向を $b_f$、world 重力単位ベクトルを $a$ とする。pure quaternion として

```math
v=(0,b_f^\top)^\top,\qquad y=(0,a^\top)^\top
```

を作り、quaternion left/right multiplication 行列 $L,R$ から

```math
P_f=L(y)-R(v)
```

```math
A_f=-\frac{\alpha}{4}P_f^\top P_f
```

を得る。現在の local YAML 差分は $\alpha=A_{param}=1000$ である（直前の既定値は 100）。$A_f$ は負半定値であり、予測 frame quaternion $z_f$ の log-likelihood 相当項を

```math
\ell_f(x)=z_f(q_{eq}(x))^\top A_fz_f(q_{eq}(x))
```

とする。方向が整合すると 0 に近づき、不整合ほど負になる。quaternion の符号反転 $z\leftrightarrow-z$ に対して不変である。

$\alpha=1000$ も実 sensor covariance から同定した値ではなく heuristic concentration である。旧 WEKF では値を 1000 へ上げると無雑音初回尤度が `-134.7` から `-355.9` へ悪化し、K2 が下限へ飛ぶ再現があった。backtracking 後は悪化候補を棄却できるが、これは concentration の統計的校正を代替しない。posterior update の強さと particle supervisor の gain threshold は $\alpha$ の scale に依存するため、実機では sensor residual 分布から校正すべきである。

一つの重力方向は frame の gravity-axis 周りの回転を観測しない。複数 frame は異なる joint combination を拘束し得るが、6 個の剛性が常に同定可能になるとは限らない。

## 10. 平衡感度

状態を

```math
x=\log K,\qquad K=\exp x
```

とする。平衡条件

```math
F(q_{eq},u,\exp x)=0
```

を $x$ で微分すると、

```math
J_q=\frac{\partial F}{\partial q}
=G_q(q_{eq})+\operatorname{diag}(k_s)
```

```math
J_x=\frac{\partial F}{\partial x}
=\operatorname{diag}\left(
\frac{\partial\tau_s}{\partial\log K}
\right)
```

```math
\frac{\partial q_{eq}}{\partial x}=-J_q^\dagger J_x
```

を得る。線形ばね、現周期ばねの双方で $\partial\tau_{s,i}/\partial\log K_i=\tau_{s,i}$ である。

$J_q$ が特異または悪条件なら、微小な $x$ 変化で平衡 branch が大きく変わる。実装は `rcond=1e-12` の Moore--Penrose pseudoinverse を使うため数値は返るが、感度の物理的信頼性を保証しない。特に、

- 重力トルクが小さく $\tau_s\simeq0$ の joint
- 伸び切り等で kinematic Jacobian が退化する姿勢
- 周期ばねの $k_s\simeq0$ または負剛性領域
- joint limit 上の平衡

では剛性識別が弱いか、局所線形化が破綻しやすい。

またこの陰関数微分は §4 の $\mathcal Q(u,K;q^{(0)},c)$ を一つの滑らかな内点 branch に制限した局所式である。solver が別の local minimum へ jump した場合、joint-limit active set が変わった場合、または $J_q$ の rank が変わった場合に、上式を branch 間の有限差へ外挿してはならない。

## 11. `MultiFrameStiffnessWEKF`

### 11.1 正確な位置づけ

名称に `WEKF` が残っているが、実装は標準的な

```math
y=h(x)+v
```

に対する residual EKF ではない。Gaussian random-walk prior

```math
x_k=x_{k-1}+w_k,\qquad w_k\sim\mathcal N(0,Q)
```

と Bingham log-likelihood を現在の $x$ 周りで局所二次化し、観測可能部分だけ Gaussian/Laplace 更新する静的 parameter estimator である。

現設定は

```math
Q=10^{-8}I
```

である。初期標準偏差は log bounds 幅の 1/4、

```math
\sigma_{x,0}=\frac{\log500-\log1}{4}\simeq1.55365
```

である。

### 11.2 局所 gradient と information

frame quaternion $z_f$ は Hamilton 規約の wxyz quaternion であり、回転

```math
R(z_f)=R_{bf}=R_{Wb}^\top R_{Wf}
```

すなわち frame 座標から base 座標への写像を表す。Pinocchio WORLD 表現の frame/base 角速度 Jacobian をそれぞれ $J_f^W,J_b^W$ とすると、frame の base に対する相対空間角速度を base 表現した Jacobian は

```math
J_{bf}^{b}=R_{Wb}^\top(J_f^W-J_b^W)
```

である。quaternion right-multiplication matrix $\mathscr R(z)$ を

```math
\mathscr R(z)p=p\otimes z
```

で定義し、その vector 部分の列を

```math
Q_{sp}(z)=\mathscr R(z)_{[:,1:4]}
```

とする。このとき base/spatial 角速度 $\omega^b$ に対して

```math
\dot z_f=\frac12Q_{sp}(z_f)\omega^b
```

である。これは body/local 角速度 $\omega^f$ に対する

```math
\dot z_f=\frac12L(z_f)_{[:,1:4]}\omega^f
```

と同値だが、角速度の座標系を交換してはならない。

平衡感度の符号を除いた量を

```math
X=J_q^\dagger J_x
```

```math
M_f=Q_{sp}(z_f)J_{bf}^{b}X
```

と置く。実際の平衡感度には負号があるため、

```math
\frac{\partial z_f}{\partial x}\simeq-\frac12M_f
```

である。局所 likelihood gradient と positive-semidefinite 近似 information は

```math
g_f=-M_f^\top A_fz_f
```

```math
\mathcal I_f\simeq
-\frac12M_f^\top A_fM_f
```

となる。複数 frame について

```math
g=\sum_fg_f,\qquad
\mathcal I=\operatorname{sym}\left(\sum_f\mathcal I_f\right)
```

を使う。

これは $z(x)$ の二階微分を無視した Gauss--Newton/Laplace 近似である。非凸な平衡 branch 全体を表さない。

#### 11.2.1 監査で発見した座標系バグと修正

修正前の実装は

```math
M_f^{old}=L(z_f)_{[:,1:4]}J_f^W X
```

としていた。左 multiplication の vector 列は body/local 角速度用なのに、WORLD 表現かつ base の運動を差し引かない Jacobian を入力しており、座標系が不整合だった。Yamaguchi `module4_link2`、

```text
q = [0.2, -0.4, 0.3, 0.1, -0.2, 0.25] rad
```

で $\partial z/\partial q$ を中央差分したところ、旧式の相対 Frobenius 誤差は `0.34348647`、修正式

```math
\frac12Q_{sp}(z_f)J_{bf}^b
```

は `6.4e-10` だった。別姿勢・複数 Yamaguchi frame では旧誤差は約 `0.34--1.35` であり、無視できる数値誤差ではない。

修正は次の三点である。

1. `RobotArm.frame_angular_jacobian_base()` が $R_{Wb}^\top(J_f^W-J_b^W)$ を返す。
2. `BinghamUtils.spatial_qmat_from_quat_wxyz()` が $\mathscr R(z)_{[:,1:4]}$ を返す。
3. WEKF はこの二つだけを組み合わせる。

回帰 test は、固定 base だけでなく `base_link="link1"` とした可動・回転 base に対し、実 RobotArm FK の quaternion 中央差分を比較する。さらに実 FK で $\ell(x)=z(\theta-Xx)^\top Az(\theta-Xx)$ を直接差分し、WEKF gradient と Gauss--Newton information の符号・係数まで検証する。旧 test は実装と同じ誤った $M$ から人工 $z(x)$ を作っていたため自己参照的で、このバグを検出できなかった。

### 11.3 観測可能部分空間

$P^-=P+Q$ の対称平方根を $S=(P^-)^{1/2}$ とし、無次元の prior-whitened information と gradient を

```math
\widetilde{\mathcal I}=S\mathcal I S,\qquad \widetilde g=Sg
```

とする。$\widetilde{\mathcal I}$ を固有分解して負固有値を 0 へ clip し、

```math
\widetilde{\mathcal I}=U\operatorname{diag}(\lambda)U^\top,qquad
\lambda_i>\max(10^{-10},10^{-4}\lambda_{max})
```

を満たす列を $U_o$ とする。旧実装は raw $\mathcal I$ の各時刻最大固有値を基準にしたため、K2 の強情報に対し K4--K6 の小さい正情報を毎 sample 永久に捨てた。prior whitening 後は、既に学習した強モードの covariance が縮むとその無次元情報も下がり、prior uncertainty が残る弱い独立モードが後続 sample で rank に入る。large negative raw eigenvalue は debug flag に残す。

### 11.4 exact posterior backtracking

局所 Laplace step は非線形平衡写像の大域近似ではない。そこで最大 16 候補について $\alpha=1,1/2,1/4,\ldots,2^{-15}$ を試し、観測部分空間で

```math
v_i(\alpha)=\frac{1}{1+\alpha(\lambda_i+\epsilon)},qquad
\Delta y_o(\alpha)=v(\alpha)\odot\alpha U_o^\top\widetilde g
```

```math
\Delta x(\alpha)=S U_o\Delta y_o(\alpha)
```

を作る。candidate $x_c=\operatorname{clip}(x^-+\Delta x)$ は、まず estimator mean の trust region

```math
\|x_c-x^-\|_\infty\le0.25
```

を満たさなければ、非線形 solve を行わず次の $\alpha$ へ進む。これは各剛性の一回の倍率変化を $\exp(0.25)\simeq1.284$ 以下にする。境界へ step を投影すると方向と covariance update の解釈が変わるため、投影せず同じ tempering 列を縮小する。

通過した candidate は平衡を非線形 solver で解き直し、現在の予測平衡 $q_{eq}^-$ から

```math
\|q_{eq}(x_c)-q_{eq}^-\|_2\le0.10\ \mathrm{rad}
```

の局所 branch に留まることを要求する。この gate は同一 branch の数学的証明ではないが、周期ばねや joint limit による有限 jump を局所微分更新として受理することを防ぐ。その後、exact Bingham likelihood $\ell(x_c)$ と Gaussian prior を含む

```math
\mathcal J(x_c)=\ell(x_c)
-\frac12(x_c-x^-)^\top(P^-)^\dagger(x_c-x^-)
```

がともに更新前の $\ell(x^-)$ 以上となる最初の candidate だけを commit する。trust region 超過、branch jump、または目的関数悪化によって全 16 候補が棄却された場合は mean と covariance を更新せず、`exact_posterior_not_improved` とする。受理した $\alpha$ に対応して covariance も tempered update し、返却する $q_{eq}$ は旧 $K$ の予測でなく受理後 $K$ で再計算した値である。

trust region の必要性を示す Yamaguchi、`A_param=1000`、無雑音 matched-model の再現では、非悪化 gate だけの旧更新が初回に $\|\Delta\log K\|_\infty=2.932$、$\|\Delta q_{eq}\|_2=0.562\,\mathrm{rad}$ を受理した。8 更新後の joint-pose 誤差は `0.2454` から `0.3397 rad` へ悪化し、`K2=1`、`K3=500` の上下限へ到達した。現 trust region では、同じデータの 8 更新後誤差は `0.0716 rad` まで低下し、上下限衝突を回避した。

別の既存回帰では、16 更新すべてで exact likelihood が非悪化で、prior-whitened rank は `[3,3,3,2,2,2,2,2,2,3,3,3,3,3,4,4]`、同一 command に対する予測平衡 joint-pose 誤差は `0.30847` から `0.01463 rad` へ低下した。これらは真の姿勢を初期 seed に使わず、現剛性から予測した初期平衡を用いた結果である。

debug には `laplace_step_scale`、`laplace_dx_max_abs`、平衡 jump の 2-norm/最大成分、前後 likelihood/posterior、各 trial の棄却理由、prior-whitened spectrum と threshold を保存する。

旧講義資料に記載されていた `measurement_info_eig_cap`、`stiffness_update_gain`、旧名 `max_log_kp_step` は現在の estimator update には存在しない。現 estimator mean の trust region は `max_log_kp_update_step`、command に使う $K_{exec}$ 側の別の平滑化 cap は `max_log_kp_exec_step` であり、混同してはならない。

### 11.5 三種類の「観測可能性」

実装には互いに異なる三つの固有空間がある。

1. WEKF: $\log K$ 空間の Bingham 局所 information $\mathcal I_x$。
2. feedforward projection: joint-angle 空間の kinematic matrix $\mathcal G(q)=\sum J_g^\top J_g$。
3. particle supervisor: window 内の WEKF information の和。

これらを同一の observability と説明してはならない。特に feedforward projection は観測された gravity 値そのものではなく、frame の存在と局所 kinematics から basis を作る heuristic である。

### 11.6 識別可能性の限界

重力方向だけでは、一般に $K\in\mathbb R^n$ を一意に識別できない。不可観測性は少なくとも次から生じる。

- 各 IMU frame の gravity-axis 周りの回転 fiber
- 重力トルクが 0 に近い joint の $J_x\simeq0$
- 複数 K combination が同じ link 重力方向を作る parameter coupling
- 伸び切り、特異姿勢、対称姿勢
- 周期ばねと重力による複数平衡 branch
- IMU frame より distal な joint の影響欠如

したがって `K_est` の全成分を真値へ収束させる主張には、複数姿勢の persistent excitation と rank/condition の実証が必要である。より基本的に、gravity-only 尤度が直接保証できるのは各 IMU の**観測重力方向**との整合であり、重力軸回りを含む full frame orientation や full joint pose の一意性ではない。複数 frame と matched model によって局所写像が十分拘束される回帰では joint-pose error も低下したが、これは一般保証ではない。full pose が必要なら encoder、磁気方位、vision など重力方向とは独立な観測を追加しなければならない。制御目的では、full K error だけでなく observable subspace に射影した parameter error、gravity-direction residual、および独立 trajectory 上の tracking error を報告すべきである。

## 12. Deterministic stiffness proposal supervisor

### 12.1 通常起動時の状態

`estimator.yaml` には `particle_scan_enabled: true` が残っているが、`deflecomp.launch` は YAML 読み込み後に launch argument を private parameter へ設定する。argument の default は `false` なので、本書冒頭の通常 command では supervisor は無効である。

```bash
particle_scan_enabled:=true
```

を明示したときだけ有効になる。従って、通常起動で観察された赤 `cmd` の急追従は particle supervisor が原因ではない。

### 12.2 呼称

現実装は particle 集合、importance weight、resampling、state transition を持たないため particle filter/RBPF ではない。また候補 score に Gaussian prior や `P_est` を含めないため MAP でもない。正確な名称は、

> WEKF を主推定器とし、過去の準静的 IMU 尤度を局所情報固有方向上で走査する、非同期 deterministic bounded maximum-likelihood proposal supervisor

である。

### 12.3 record と window

各 record は

```math
\mathcal D_r=
\left(u_r,\{A_{r,f}\}_f,q^{(0)}_{eq,r},t_r,\mathcal I_r\right)
```

を保存する。現在の window は 20 records であり、oldest 15 records を discovery、newest 5 records を chronological holdout validation に使う。

隣接 sample は同じ静止姿勢、同じ bias、同じ model error を共有する。従ってこの 15/5 split は code-level holdout ではあるが、統計的に独立な validation ではない。

### 12.4 探索方向

discovery records の WEKF information を加算する。

```math
\mathcal I_{win}=\sum_{r\in train}\operatorname{sym}(\mathcal I_r)
```

固有値が

```math
\lambda_j>\max(10^{-8},10^{-4}\lambda_{max})
```

を満たす方向を大きい順に最大 2 本選ぶ。固有ベクトルの符号は、最大絶対成分を正にして deterministic にする。

各方向 $v_j$ について、現在値 $x_0$ の周囲に 21 点を置く。

```math
x_{j,l}=\operatorname{clip}(x_0+d_lv_j,\log K_{min},\log K_{max})
```

```math
d_l\in\operatorname{linspace}\left(
-\frac{\log2}{\|v_j\|_\infty},
\frac{\log2}{\|v_j\|_\infty},21
\right)
```

従って各 parameter の候補 jump は最大 $\log2$、すなわち約 2 倍である。二方向の tensor grid は作らず、各直線を別々に走査する。重複を除いた候補数は最大およそ 41 である。

### 12.5 score と採択

候補 $x$ と各 record について静的平衡を解き、

```math
L_{train}(x)=\sum_{r\in train}\sum_f
z_{r,f}(q_{eq,r}(x))^\top A_{r,f}z_{r,f}(q_{eq,r}(x))
```

を最大化する。prior penalty

```math
-\frac12(x-x_{est})^\top P_{est}^{-1}(x-x_{est})
```

は含まれない。

候補平衡が現在値の予測平衡から、いずれかの joint で 0.35 rad を超えて変化すれば branch discontinuity として棄却する。採択には次が必要である。

```math
\frac{L_{train}(x_*)-L_{train}(x_0)}{15}\ge1
```

```math
\frac{L_{val}(x_*)-L_{val}(x_0)}{5}\ge1
```

```math
0.02\le\|x_*-x_0\|_\infty\le\log2
```

採択 gate の debug field `training_gain_per_obs` と `validation_gain_per_obs` の分母は、IMU frame 数や scalar observation 数ではなく、それぞれ record 数 15 と 5 である。従って上の threshold は正確には `1 per training/validation record` である。一方、`StiffnessParticleScanResult` のトップレベル `gain_per_obs` は

```math
\frac{(L_{train}+L_{val})(x_*)-(L_{train}+L_{val})(x_0)}{20}
```

という combined gain per window record であり、15/5 の各 gate 値とは別である。各 record の score はその cycle で新規だった全 frame の尤度和なので、frame 数が増えるほど threshold は相対的に緩くなる。これらは現在の `A_param=1000` と frame 数の heuristic scale に依存し、確率的な significance threshold ではない。

### 12.6 非同期 freshness と適用

scan は原則別 process、起動失敗時は thread で走る。snapshot 結果を適用するとき、

```math
N_{record,now}-N_{record,snapshot}\le5
```

```math
\|x_{est,now}-x_{snapshot}\|_\infty\le0.15
```

を要求する。採択後は 20 records の cooldown、通常 scan 間隔は 5 records である。

候補を直接置換せず、探索固有方向が support を持つ index 集合 $\mathcal A$ について

```math
x_{pursuit,i}=\begin{cases}
x_{*,i} & i\in\mathcal A\\
x_{old,i} & \text{otherwise}
\end{cases}
```

```math
x_{new}=(1-w)x_{old}+wx_{pursuit}
```

と混合する。通常 YAML は $w=0.8$ なので、候補が $\log2$ 離れていれば一回で最大約 $2^{0.8}=1.74$ 倍変化し得る。共分散には mixture の between-mean term を加える。`reset_std=0.1` の floor、すなわち $0.1^2$ は pursuit component $P_{pursuit}$ の active 対角にだけ適用される。最終

```math
P_{mix}=(1-w)(P_{old}+d_zd_z^\top)
+w(P_{pursuit}+d_pd_p^\top)
```

の active 対角が $0.1^2$ 以上になる保証はなく、$w<1$ なら下回り得る。その後、`K_exec` が時定数 0.5 s で追従する。

### 12.7 「肘が曲がる」failure mode

旧版 supervisor は、1 record から全剛性軸を bounds 全域で走査し、holdout、information gate、branch gate、cooldown、freshness gate を持たず、小さな score 改善でも $w=0.8$ で繰り返し適用していた。これは精度を壊し得る構造だった。

現行版は保守化されたが、根本問題は残る。

- score に tracking error は入らない。
- link encoder など、実際の肘角を直接拘束する観測がない。
- gravity direction だけでは elbow configuration と剛性 combination が一意でない場合がある。
- periodic spring と重力の非凸ポテンシャルは複数 equilibrium branch を持ち得る。
- 0.35 rad gate は真の姿勢でなく、現在モデルの予測 branch との差だけを見る。
- 小さな branch shift を複数回採択すれば、別 branch へ漸進的に移れる。
- 最大 2 本の固有方向でも、固有ベクトルが dense なら active index は全関節になり得る。
- prior penalty がないため、弱観測方向で nominal K から離れる罰則がない。
- 同じ姿勢で時刻だけ異なる新規 sensor sample を 20 回測れば evidence を 20 回加算し、姿勢多様性を要求しない。修正後は同一 `source_stamp` の endpoint hold は重複加算しない。
- command と観測の静的対応誤差、URDF 誤差、IMU bias、外力をすべて K に吸収できる。

従って、肘を伸ばした姿勢で曲がった model equilibrium へ移る現象は単なる parameter tuning だけの問題ではない。重力方向だけの非識別性、非凸平衡、score に実 joint pose がないことが主要因である。通常 launch で無効にするのは妥当であり、再有効化には独立 episode validation、prior/MAP 項、平衡安定性・残差・reference error gate、正しい command--observation 対応が必要である。

なお period、active dimension、gain、jump、cooldown 等の高度な gate は `DeflectionCompensator` 内の default として存在するが、現 `deflecomp_node.py` は大半を ROS parameter として読み渡さない。YAML に項目を追加するだけでは調整できない。

## 13. 一制御 cycle の厳密な順序

修正後の node と `DeflectionCompensator.step()` は次の順で動く。

```text
input: q_ref,k, IMU buffers, command/reference histories, dt, t_k

1. 全 IMU の最新共通時刻 t_z を選ぶ
2. command/reference が直前0.5 s安定していれば、履歴から u(t_z) を取得して観測を構成
3. completed asynchronous supervisor result があれば freshness 検査後に適用
4. frame ごとに `source_stamp` が新しい IMU だけを選び、空でなければ $(u(t_z),A(t_z))$ で WEKF update
5. 最大16回の exact posterior backtracking を行い、$\|\Delta\log K\|_\infty\le0.25$、$\|\Delta q_{eq}\|_2\le0.10\,\mathrm{rad}$、exact likelihood/posterior 非悪化を全て満たす candidate だけを K_est へ commit
6. K_est を K_exec_target に設定
7. log K_exec を時定数 0.5 s、最大 0.05/cycle で更新
8. 既定では q_eval=q_ref とし、tau_g(q_ref) と K_exec から inverse-statics seed を生成
9. L1 regularization が有効なら raw seed を最適化
10. raw equilibrium refinement が明示的に有効かつ projection 無効なら bounded refinement
11. u_raw を theta_cmd_tau=0.2 s で low-pass して u_k を確定
12. u_k と K_exec から q_eq_hat を解く。ただし u_k は変更しない
13. `step()` 内で $u_k$ と $q_{eq,hat}$ を次 cycle の state に保存し、result を return
14. node が実 publish 直前時刻を header/history に記録して u_k を publish
```

重要な不変条件は

```math
u_{published,k}=u_k
```

```math
\hat q_{eq,k}=Q(u_{published,k},K_{exec,k};\hat q_{eq,k-1})
```

である。`theta_cmd_post_filter_correction_norm` は内部 result debug で 0 でなければならない。ただし現実装は post-filter correction path 自体を削除した上で、この field に定数 `0.0` を代入する。独立に送信前後の差を測った runtime monitor ではない。command audit scalar は現 `/deflecomp/debug` vector にも含まれないため、ROS 上からこの不変条件を直接検査できるという意味ではない。

平衡 branch の warm-start 規則も因果性に関係する。WEKF update はまず estimator 自身の前回平衡、なければ compensator が前 cycle の実送信 command から予測した平衡、両方なければ current reference を初期値に使う。修正前は最初の estimator update で、すでに前 cycle の送信 command 平衡が存在しても current desired pose を seed にし、非凸問題を desired branch へ bias し得た。送信後の $\hat q_{eq,k}$ は前回 $\hat q_{eq,k-1}$、初回だけ $q_{r,k}$ を seed にする。

また §5.6 の low-pass recurrence は $u_{k-1}$ が存在する cycle に対する不変条件であり、初回 $k=0$ は $u_0=u_{raw,0}$ である。

## 14. ROS rate と可視化 rate

| 処理 | nominal rate |
|---|---:|
| reference GUI / joint state publisher | 50 Hz |
| controller | 50 Hz (`dt=0.02`) |
| simulator integration | nominal 1000 Hz (`dt=0.001`) |
| simulator state/IMU publish | 100 Hz |
| ProbTF TF import | 最大 10 Hz |
| point-moment marker lookup/publish | 10 Hz |

ProbTF import/marker は各 edge の latest sample を使い、中間軌跡を捨てる。これは 30 s 級 FIFO backlog を避けるための低遅延モニタ設計である。従って marker は状態の目視確認には使えるが、lag、overshoot、settling time の計測器には使えない。定量評価は元の timestamp 付き `JointState`/IMU/topic を rosbag 等へ記録して行う。

point marker が示すのは tip frame 原点の 3 次元位置だけである。tip 姿勢、各 joint error、肘の異なる configuration は点だけでは判別できない。

## 15. 真値漏洩と理想仮定の監査

### 15.1 controller が読むもの、読まないもの

| 変数・信号 | controller が使用するか | 経路 |
|---|---:|---|
| reference `q_ref` | 使用 | `/ref/joint_states` |
| IMU specific force / gyro gate | 使用 | `/imu` |
| 前回 publish command | 使用 | controller 内部 state |
| URDF mass/COM/inertia/kinematics | 使用 | launch で渡された URDF |
| IMU extrinsic | 使用 | launch で渡された同じ YAML |
| simulator `K_true` | 不使用 | simulator private parameter のみ |
| simulator state、`/equil` | 不使用 | subscriber が存在しない |
| simulator 内部 `u_eff` | 不使用 | publish されない |
| 未来の reference | 不使用 | callback の最新値のみ |
| external wrench truth | 不使用 | controller model に項がない |

この意味で direct ground-truth leakage はない。

### 15.2 inverse crime

| 項目 | controller | simulator | 独立性 |
|---|---|---|---|
| URDF | 指定された Yamaguchi URDF | 同じ file | なし |
| kinematics/dynamics library | Pinocchio | Pinocchio | なし |
| mass/COM/inertia | URDF nominal | 同じ nominal | なし |
| gravity | `[0,0,-9.81]` | 同じ | なし |
| spring law | `PeriodicSpringModel` | 同じ class | なし |
| spring structure | 完全対角 | 完全対角 | なし |
| IMU extrinsic | 指定 YAML | 同じ YAML | なし |
| IMU forward kinematics | Pinocchio | 同じ robot model | なし |
| K の数値 | 推定、初期 22.36 | `[5,5,5,10,20,20]` | ここだけ非一致 |

未知なのは主に K の数値であり、model class、geometry、inertial parameters、sensor calibration は一致する。従ってこれは「unknown parameter under a perfectly known model」の上限試験である。

実際、初期 `K_exec=22.36` と真値 `K_true=[5,5,5,10,20,20]` を使い、§6.2 と同じ

```text
q_ref = [0.4, -0.4, 0.3, 0.2, -0.5, 0.4] rad
```

で raw inverse command に対する真値側平衡を、`PeriodicSpringModel` と既定 `EquilibriumSolver`、初期値 $q^{(0)}=q_{ref}$ で解くと、joint の L2 error は `0.1634487 rad`、tip frame `module4_link2` の position Euclidean error は `0.0754556 m` だった。これは `K_true` の直接注入がない証拠である一方、model mismatch に対する頑健性の証拠ではない。

### 15.3 「チート」の分類

| 分類 | 判定 | 説明 |
|---|---|---|
| `K_true` / actual q の直接漏洩 | なし | controller 入力経路がない |
| wall-clock の未来観測 | なし | 1-cycle fixed-lag はある |
| 同一生成モデルでの inverse crime | あり | URDF、ばね、重力、IMU model が一致 |
| 赤 marker の semantic leakage | あり | setpoint を物理姿勢のように描画 |
| LPF 後の command overwrite | 修正済み | 旧実装は制約を無条件で迂回 |
| 内部予測の自己目的化 | 修正済み | 予測で送信 command を変更しない構造へ変更 |
| WEKF quaternion/Jacobian 座標混在 | 修正済み | body tangent と WORLD Jacobian の誤結合を relative-base spatial 表現へ統一 |
| held IMU の独立観測としての反復 | 修正済み | sensor support timestamp を frame ごとに追跡して重複を除外 |
| velocity limit と実位置遷移の不一致 | 修正済み | non-quasistatic state transition 自体へ limit を適用 |
| actuator/sensor の過度な理想化 | あり | 上限 baseline としてのみ妥当 |

## 16. 現在の effective parameters

### 16.1 controller

| parameter | value |
|---|---:|
| `dt` | `0.02 s` |
| `theta_cmd_tau` | `0.2 s` |
| `theta_cmd_l1_regularization` | `false` |
| `theta_cmd_equilibrium_refine` | `false` |
| `theta_cmd_equilibrium_refine_maxiter` | `2` |
| `theta_cmd_equilibrium_refine_max_delta` | `0.25 rad/joint/iteration` |
| `spring_model` | `periodic` |
| equilibrium solver `refine` | `true` |
| equilibrium solver tolerance | `1e-12` |

### 16.2 estimator

| parameter | value |
|---|---:|
| `A_param` | `1000`（現在のローカル YAML。未校正） |
| `update_stiffness` | `true` |
| `kp_min`, `kp_max` | `1`, `500` |
| initial K | `sqrt(500) = 22.3607` all axes |
| `log_kp_process_noise_var` | `1e-8` |
| `observability_rcond` | `1e-4` |
| `observability_abs` | `1e-10` |
| Laplace backtracking candidates | `16` (`alpha=1,1/2,...,2^-15`) |
| `max_log_kp_update_step` | `0.25/update` (`K` ratio `<= exp(0.25)`) |
| `max_equilibrium_pose_jump` | `0.10 rad` (`2`-norm of joint-vector change) |
| `project_unobservable_feedforward` | `false` |
| `kp_exec_tau` | `0.5 s` |
| `max_log_kp_exec_step` | `0.05/cycle` |
| supervisor effective launch default | `false` |
| IMU acceleration tolerance | `0.75 m/s^2` |
| IMU max angular speed | `0.20 rad/s` |
| IMU settle time | `0.25 s` |
| IMU max age | `0.10 s` |
| stiffness-update settle time | `0.50 s` |
| command stability tolerance | `1e-3 rad` |
| reference stability tolerance | `1e-4 rad` |
| command apply delay | `0.0 s` |

提示された Yamaguchi YAML は 5 frame、`module1_link1`、`module2_link1`、`module3_link1`、`module4_link2`、`module5_d405_link` を使う。全て既存 URDF frame と identity extrinsic であり、実 sensor mounting error はない。

### 16.3 simulator

| parameter | value |
|---|---:|
| integration `dt` | `0.001 s` |
| publish rate | `100 Hz` |
| `K_true` | `[5, 5, 5, 10, 20, 20]` |
| `zeta` | `0.9` |
| `vel_limit` | `4 rad/s` |
| `ref_tau` | `0.1 s` |
| `ref_max_vel` | `4 rad/s` |
| `eq_mode` | `dynamic` |
| `integrator` | `semi_implicit` |
| `spring_model` | `periodic` |
| external wrench | default zero |
| sensor noise/bias/delay | none |

## 17. 論文相当の評価プロトコル

### 17.1 評価仮説を分離する

少なくとも次を別の仮説として検証する。

1. 既知または fixed nominal K の inverse statics が no compensation より定常たわみを減らすか。
2. WEKF が independent trajectory 上の tracking を fixed nominal K より改善するか。
3. supervisor が WEKF 単独より改善するか、または悪化させるか。
4. command shaping を守った上で過渡性能が許容範囲か。
5. model/sensor mismatch 下でも改善が残るか。

内部予測が reference に一致することは、仮説 1--5 の証拠ではない。同じモデル内の inverse consistency にすぎない。

### 17.2 必須 baseline

| baseline | command/estimation |
|---|---|
| B0 no compensation | `u = q_ref` |
| B1 fixed nominal | fixed `K0` の direct inverse |
| B2 WEKF | supervisor off |
| B3 WEKF + supervisor | supervisor on |
| B4 oracle | `K_true` を controller へ与える。上限であり実方式と明確に区別 |

さらに以下を ablation する。

- `theta_cmd_equilibrium_refine` off/on。ただし常に LPF 前。
- `project_unobservable_feedforward` off/on。
- command low-pass と simulator shaping の各時定数。
- linear spring と current periodic spring。
- equilibrium residual refinement off/on。
- supervisor mixture weight `w = 0, 0.25, 0.5, 0.8, 1.0`。
- supervisor の prior なし ML と、`P_est` を用いる正則化/MAP 版。

旧 command refiner を再現する場合は「既知の不正 baseline」と明記し、主結果に使わない。

### 17.3 training と test の分離

- identification trajectory と tracking test trajectory を分ける。
- test 中に K update を freeze した結果も示す。
- supervisor validation は隣接 sample でなく、異なる姿勢 episode を使う。
- straight elbow だけでなく、information rank を上げる複数姿勢を設計する。
- candidate/WEKF が見た record と、最終評価 record を共有しない。

### 17.4 matched と mismatched test

matched-model baseline に加え、plant 側だけを変える。

- mass、COM、inertia の独立摂動
- spring law の違い、off-diagonal stiffness
- Coulomb friction、backlash、hysteresis
- unknown payload、外力、接触
- IMU bias、noise、scale、extrinsic error
- command/sensor delay、timestamp jitter、dropout
- torque/rate/position/acceleration saturation
- base tilt と gravity magnitude error

可能なら forward data generator を controller と別実装または別 physics engine にし、最終的には実機 data を使う。parameter perturbation の乱数 seed と分布を固定し、複数 seed の confidence interval を報告する。

### 17.5 指標

revolute joint error は wrap を明示して

```math
e_q(t)=\operatorname{wrap}_{[-\pi,\pi)}(q(t)-q_r(t))
```

とする。主指標は次である。

- joint RMSE、max error、steady-state bias
- end-effector position error と SO(3) geodesic orientation error
- settling time、overshoot、cross-correlation lag
- requested command と applied command の速度・加速度・total variation
- command/rate/joint-limit violation 回数
- log K error と observable-subspace projected error
- IMU accept rate、observable rank、information condition number
- equilibrium residual、branch change、negative-stiffness/asin-clip 回数
- divergence/joint-limit-contact rate
- controller cycle time p50/p95/p99、deadline miss

`cmd-ref`、赤 marker と青 marker の距離は tracking 指標に含めない。補償では command が reference と異なるのが正常だからである。

simulator 内部の `u_eff` を新たに publish/log し、requested command、applied spring reference、physical link state を分離する必要がある。

### 17.6 marker ではなく raw topic を記録する

少なくとも次を保存し、message Header がある topic はその event timestamp、ない topic は bag receive time を区別して扱う。

```text
/ref/joint_states
/cmd/joint_states
/equil/joint_states
/imu
/deflecomp/kp_est
/deflecomp/kp_exec
/deflecomp/kp_exec_target
/deflecomp/kp_cov_diag
/deflecomp/theta_eq_hat
/deflecomp/debug
/deflecomp/estimation_gate_status
/deflecomp/particle_scan_status
/deflecomp/particle_scan_debug
```

`/ref/joint_states`、`/cmd/joint_states`、`/equil/joint_states`、`/imu` には Header がある。一方、`/deflecomp/kp_est`、`kp_exec`、`kp_exec_target`、`kp_cov_diag`、`theta_eq_hat`、`debug`、`particle_scan_debug` は `Float64MultiArray`、`estimation_gate_status` と `particle_scan_status` は `std_msgs/String` であり、いずれも Header を持たない。`estimation_gate_status` の本文には alignment stamp を含むが、型としての event timestamp ではない。さらに `/deflecomp/debug` には field 名・version schema もない。従って現状の全 topic を厳密な共通 event-time 軸へ置くことはできず、bag receive time による近似になる。論文計測には sent-command stamp と schema/version を含む stamped custom diagnostic message が必要である。

加えて simulator の `u_slew`、`u_eff`、内部 simulated time、asin clip/limit/debug を publish することが望ましい。

## 18. 許される主張と許されない主張

### 18.1 現時点で許される主張

- controller は `K_true` や `/equil` を直接参照していない。
- 赤 marker は要求 command が低遅延で ROS/TF へ届いたことを示す。
- 修正後の publish command は設定した `theta_cmd_tau` の解析式に一致する。
- `theta_eq_hat` は、送信 command と現在の `K_exec` に対する内部静力学予測である。
- 緑は、記載した理想 simulator の動的 state を示す。
- matched-model/noise-free 条件で数式と ROS 配線の整合性を検査できる。
- estimator は準静的 gravity-direction records で、局所的に観測可能な log K 方向だけを更新する。

### 18.2 現時点で許されない主張

- 赤 marker から実姿勢、モータ角、closed-loop tracking 性能を主張すること。
- `theta_eq_hat` と reference の一致から K 推定精度や実姿勢精度を主張すること。
- 10 Hz marker 動画から 50/100/1000 Hz 系の lag や overshoot を定量化すること。
- 現結果から実機で同等の性能が得られると主張すること。
- motion 中にも剛性を正しく推定していると主張すること。
- gravity direction だけで全 6 軸剛性を常に一意同定できると主張すること。
- supervisor を有効にすれば必ず精度が上がると主張すること。
- 外力、payload、摩擦、backlash 下の補償性能を主張すること。

## 19. 既知の未解決事項と優先順位

### P0: 科学的妥当性と安全性

1. requested command と simulator applied command を別 topic と marker へ分離する。
2. periodic spring の 4-pi 周期を仕様として正当化するか、物理ばねに適した law へ置換する。
3. asin infeasibility、`J_q` condition、active-set transition、equilibrium residual、Hessian stability を明示的に gate する。
4. independent plant model と sensor imperfections を評価系へ入れる。

### P1: estimator

1. gravity-only で識別可能な parameter combination を事前解析する。
2. `A_param` を実 residual covariance から校正する。
3. K1 のような厳密不可観測軸を固定し、full K でなく識別可能な低次元 parameterization も比較する。
4. negative information、branch change、backtracking の収束性を検証する。
5. 隣接準静的 sample の相関を effective sample size または episode-level likelihood で扱う。

### P2: supervisor

1. score に Gaussian prior、stability、residual、reference error を導入する。
2. adjacent record でなく別姿勢 episode を validation に使う。
3. 同一姿勢の重複 evidence を effective sample size で補正する。
4. advanced safety parameters を ROS/YAML から設定可能にする。
5. dense eigenvector support と covariance reset の範囲を分離する。

## 20. 実装対応表

| 概念 | file / symbol |
|---|---|
| URDF wrapper、重力、frame Jacobian | `ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/robot/pinocchio_robot.py` |
| linear / periodic spring | `ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/model/spring.py` |
| staged equilibrium solver | `ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/model/equilibrium.py` |
| `J_q`, `J_x` | `ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/model/sensitivity.py` |
| Bingham matrix | `ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/observation/bingham.py` |
| IMU observation builder | `ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/observation/imu_observation.py` |
| quality gate / timestamp buffer | `ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/observation/imu_buffer.py` |
| local-Laplace stiffness estimator | `ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/estimator/stiffness_wekf.py` |
| deterministic ML supervisor | `ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/estimator/stiffness_particle_supervisor.py` |
| command pipeline と本監査修正 | `ros/examples/deflecomp/deflecomp_core/src/deflecomp_core/pipeline/compensator.py` |
| ROS timing、topic、parameter | `ros/examples/deflecomp/deflecomp_ros/nodes/deflecomp_node.py` |
| controller parameter | `ros/examples/deflecomp/deflecomp_ros/config/controller.yaml` |
| estimator parameter | `ros/examples/deflecomp/deflecomp_ros/config/estimator.yaml` |
| launch override | `ros/examples/deflecomp/deflecomp_ros/launch/deflecomp.launch` |
| ref/cmd/equil RSP wiring | `ros/examples/deflecomp/deflecomp_ros/launch/deflecomp_frames.launch` |
| dynamic plant | `ros/examples/deflecomp/deflecomp_sim/src/deflecomp_sim/dynamic_simulator.py` |
| synthetic IMU と ROS sim node | `ros/examples/deflecomp/deflecomp_sim/nodes/sim_node.py` |
| simulator parameter | `ros/examples/deflecomp/deflecomp_sim/config/sim_params.yaml` |
| command regression tests | `tests/deflecomp_core/test_compensator_k_exec.py` |
| WEKF/FK finite-difference tests | `tests/deflecomp_core/test_stiffness_wekf.py` |
| IMU source-time tests | `tests/deflecomp_core/test_imu_buffer.py` |
| inverse-statics / KKT / active-set tests | `tests/deflecomp_core/test_equilibrium_constraints.py` |
| dynamics limit/integration tests | `tests/deflecomp_sim/test_dynamics.py` |

## 21. 最小擬似コード

```text
state:
    x_est, P_est
    x_exec, x_exec_target
    u_prev, qeq_hat_prev, t_prev
    command_history, reference_history, imu_buffers

on cycle(q_ref, imu_buffer, t):
    t_obs = latest_common_imu_stamp(imu_buffers)
    u_obs = settled_value_at(command_history, t_obs, dwell=0.5)
    ref_ok = settled_value_at(reference_history, t_obs, dwell=0.5)
    obs = []
    if u_obs exists and ref_ok:
        obs = interpolate_all_imu_at(t_obs)
        obs = keep_new_source_stamp_per_frame(obs)

    if obs is fresh and u_obs exists:
        A = build_bingham_matrices(obs)
        x_est, P_est = exact_backtracked_laplace_update(
            prior=(x_est, P_est),
            sent_command=u_obs,
            observation=A,
            max_trials=16,
            max_abs_delta_log_k=0.25,
            max_equilibrium_delta_l2=0.10
        )
        x_exec_target = x_est

    x_exec = bounded_log_lowpass(x_exec, x_exec_target)
    K_exec = exp(x_exec)

    tau_g_eval = gravity(q_ref)

    u_raw = inverse_statics(q_ref, tau_g_eval, K_exec)
    if l1_enabled:
        u_raw = regularize_static_inverse(u_raw)

    if raw_refine_requested and not observable_projection_enabled:
        u_raw = bounded_equilibrium_inverse_refine(u_raw, q_ref, K_exec)

    u = lowpass(u_prev, u_raw, theta_cmd_tau)
    qeq_hat = equilibrium_solve(u, K_exec, seed=qeq_hat_prev)  # read-only prediction

    t_publish = now_after_computation()
    publish(u, stamp=t_publish)
    append(command_history, t_publish, u)
    publish(qeq_hat, exp(x_est), K_exec, P_est)
    save(u_prev=u, qeq_hat_prev=qeq_hat, t_prev=t)
```

この擬似コードで最も重要なのは、最後の `equilibrium_solve` が command を変更しないことである。

## 22. まとめ

本方式の本質は、IMU の準静的重力方向から柔軟関節の有効剛性を局所推定し、その剛性と URDF 重力モデルを用いてばね基準角を逆静力学計算することである。理論上の中核は、平衡式、陰関数感度、Bingham 方向尤度の局所 Laplace 近似である。

一方、性能を正しく解釈するには、reference、requested command、applied command、physical state、internal equilibrium prediction を厳密に分ける必要がある。今回の監査では、この区別を壊して低域通過後の command を内部逆解で上書きする処理を発見し、既定無効・raw 段限定・bounded・projection 併用禁止・derivative failure 時 fail-closed へ修正した。

同時に、WEKF の quaternion tangent と角速度 Jacobian の座標混在、held IMU sample の再刻印による反復 evidence、simulator の velocity clip と位置遷移の不一致も修正した。前者は estimator/supervisor の探索方向、二番目は posterior confidence と record window、三番目は動的発散と平衡 branch jump に影響する。いずれも実 FK・source timestamp・state-transition invariant に対する回帰 test を追加し、旧来の自己参照 test だけに依存しないようにした。

追加監査では、剛性推定とは独立に追従を壊していた経路依存 feedforward projection を既定無効にした。既知の $q_r$ で重力を評価することで analytic inverse と内部平衡の整合性を回復し、保存 Yamaguchi 姿勢では同じ推定 K のまま真値プラント誤差を `0.689 rad` から `0.00413 rad` へ下げた。これは `K_true` や実姿勢を読む変更ではない。

さらに Pinocchio の固定 base inertia/COM 質量仕様を混用した重力ポテンシャル、joint-limit 上の非 KKT refinement、active-set を無視した感度、exact likelihood を悪化させる WEKF full step、非悪化でも大き過ぎる local step、raw information の弱モード hard cutoff、command publish 前 IMU との逆因果対応を修正した。推定更新は最大16回の backtracking と log-K/平衡姿勢 trust region を通し、実 publish command 履歴と最新共通 IMU 時刻を対応させ、0.5 s の静止 window を満たす場合だけ実行する。

直接の truth leakage はない。しかし matched-model/noise-free 条件、gravity-only の非識別性、準静的 estimator、非凸 periodic spring、setpoint marker の誤解可能性は残る。特に K1 は静止重力だけでは厳密に推定不能であり、観測重力方向の整合と full joint pose の一意性は同義でない。従って現段階の結果は内部整合性 baseline として扱い、独立 plant、sensor mismatch、raw topic に基づく定量評価を経てから実機性能へ一般化すべきである。
