# Single-bag geometric SG rigid-body estimation: implementation and ablation plan

## 0. この plan の位置づけ

この文書は、`hiroya1224/ProbTF-demo` の以下の commit を基準として、新しい single-bag 推定系と ablation 実験系を実装するための作業指示書である。

```text
base commit:
7fecffe27aa25865c70669f919cefb16b624c46e
add nondimensionalized method with KKT based optimization
```

想定配置先は `minimal/` の一つ上である。

```text
ros/examples/grape-param-estim/single_bag_savgol_ablation_implementation_plan.md
```

この plan に書かれた仕様について、実装エージェントが独自に科学的・数理的な設計変更を行ってはならない。

例外は、後述する **「agent-proposed improved ablation candidates」** の追加だけである。この例外についても、許可範囲と禁止事項をこの文書で明示する。

---

# 1. 作業開始時の基準確認

作業開始時に必ず repository HEAD を確認する。

```text
git rev-parse HEAD
```

期待値は以下である。

```text
7fecffe27aa25865c70669f919cefb16b624c46e
```

一致しない場合、勝手に別 commit へ port、rebase、cherry-pick、仕様変更して進めないこと。

この plan は上記 commit 相対で書かれている。

---

# 2. 今回の最終目的

今回必要なのは **単一 rosbag に対する rigid-body parameter estimation** である。

複数 bag を一つの objective に入れない。

各 bag から独立に parameter-space の local factor / local uncertainty / ridge information を得る。

複数 bag を使用する場合は、後段で各 bag の分布・factor を合成する。

今回の実装範囲には multi-bag joint optimization を含めない。

---

# 3. `minimal/` の整理

## 3.1 現行実装の退避

現在 `minimal/` 直下に存在する既存実装・既存 document・既存 output は、新しい実装と混在させない。

既存の `minimal/legacies/` はそのまま legacy 領域として保持する。

それ以外の現在の `minimal/` 内容は、原則として以下のような snapshot directory へ移す。

```text
minimal/
    legacies/
        pre_single_bag_rewrite_7fecffe/
            <現在 minimal/ 直下にある既存内容>
```

現在 tracked されている `minimal/output/` も新しい `minimal/outputs/` と混同しないよう legacy snapshot 側へ移す。

履歴を消す目的で削除しない。

## 3.2 新しい `minimal/` から legacy 実装を直接 import しない

新しい実装が旧コード中の機能を必要とする場合、

```text
minimal/legacies/...
```

を runtime dependency として直接 import してはならない。

必要な部分だけを、新しい root 側へ抽出する。

今回再利用してよいものは後述する。

## 3.3 新しい root の想定構成

最終的な `minimal/` は少なくとも以下を持つ。

```text
minimal/
    single_bag_savgol_estimator.py
    single_bag_savgol_ablation.py

    single_bag_savgol_core.py
    single_bag_savgol_covariance.py
    single_bag_savgol_reports.py
    single_bag_wrench_replay.py

    savgol_trajectory.py
    smooth_command.py

    tests/
        ...

    outputs/
        ...

    legacies/
        ...
```

上記は責務分離のための基本構造である。

不用意にさらに大きな framework 化を行わない。

---

# 4. 再利用してよい実装

## 4.1 geometric SG

現在の `minimal/savgol_trajectory.py` にある、

- actual timestamp を直接使う local polynomial fit
- `R^3` の local polynomial fit
- `SO(3)` の rotation-vector local polynomial fit
- SO(3) left Jacobian
- left Jacobian の directional derivative
- `R`, `omega`, `alpha` の geometric SG evaluation
- centered window support
- local design matrix / pseudoinverse

を再利用する。

ただし、新実装では covariance を拡張するため、必要な local fit coefficient、residual、pseudoinverse row 等を取得できるようにする。

## 4.2 smooth ZOH

現在の `minimal/smooth_command.py` にある `QuinticSmoothZoh` の考え方・実装を再利用する。

用途は lag search の smooth continuation のみである。

最終評価は strict ZOH で行う。

## 4.3 rigid-body / actuator primitives

`src/grape_param_estim/` にある以下は canonical implementation として再利用する。

- `actuator_wrench`
- actuator wrench Jacobian
- `advance_actuators`
- actuator transition Jacobian
- `FullSixDofPlant`
- geometry helpers
- quaternion / rotation helpers
- rosbag loader
- sensor extrinsics

## 4.4 parameter chart

現行 `dimensionless_savgol_experiment.py` にある second-moment matrix exponential chart と `expm_frechet` の実装を基礎として再利用する。

ただし **fixed-reference nondimensionalization は再利用しない**。

## 4.5 KKT

現行の exact common-scale gauge に対する KKT step を再利用する。

## 4.6 external-wrench replay

現行 `deterministic_spline_dynamics_estimator.py` にある trajectory-fitted external-wrench reconstruction の考え方と必要実装を抽出する。

新しい split rotor/gimbal lag と、新しい actuator-history implementation に合わせる。

---

# 5. single-bag input

入力は 1 bag のみである。

使用する主入力は、

```text
raw pose
recorded rotor command
recorded gimbal command
measured initial gimbal state
vehicle model
sensor extrinsics
```

である。

parameter objective には IMU measurement を入れない。

IMU は diagnostic にのみ使用する。

bag の start/end が指定される場合、その区間だけを使用する。

multi-bag weight は存在しない。

---

# 6. geometric SG trajectory

raw pose sensor point を `S` とする。

SG により各 valid center time `t_k` で、

```math
R_k
```

```math
\omega_k
```

```math
a_{S,k}
```

```math
\alpha_k
```

を得る。

position / orientation 自体は parameter objective の residual にしない。

pose は dynamics を評価する trajectory state と、その微分量を得るために使う。

---

# 7. specific acceleration

world gravity vector は、

```math
g_W =
\begin{bmatrix}
0\\
0\\
-9.80665
\end{bmatrix}
```

とする。

SG から得た pose sensor point specific acceleration は、

```math
s_{S,k}^{SG}
=
R_k^T
\left(
a_{S,k} - g_W
\right)
```

とする。

`R_k` は raw orientation ではなく、geometric SG から得た `R_k` を使う。

---

# 8. SG local covariance

## 8.1 raw observation noise assumption

raw pose measurement noise は、各時刻間で独立である working assumption を用いる。

SG estimate 同士が overlapping window により相関することは認識した上で、point estimation では各 center time の marginal covariance を用いる。

cross-time correlation は後処理の uncertainty correction で扱う。

## 8.2 local pose residual covariance

各 SG window で translation residual と rotation-vector residual を作る。

```math
e_{p,i}
=
p_i - \hat p_i
```

```math
e_{\phi,i}
=
\phi_i - \hat\phi_i
```

これを、

```math
e_i
=
\begin{bmatrix}
e_{p,i}\\
e_{\phi,i}
\end{bmatrix}
```

として 6-vector にまとめる。

polynomial degree を `d`、window sample count を `n` とすると、

```math
\widehat\Omega
=
\frac{1}{n-(d+1)}
\sum_i e_i e_i^T
```

を local `6 x 6` raw-pose noise covariance estimate とする。

position-rotation cross block を保持する。

## 8.3 rotation local coefficients

rotation-vector polynomial は、

```math
\phi(t_0+\tau)
\simeq
\rho_0
+
\rho_1\tau
+
\frac{1}{2}\rho_2\tau^2
+\cdots
```

とする。

`rho_0`, `rho_1`, `rho_2` はそれぞれ単純な `R`, `omega`, `alpha` ではない。

geometric map を通して、

```math
R_{SG}
=
\operatorname{Exp}(\rho_0^\wedge) R_{ref}
```

```math
\omega_{spatial}
=
J_l(\rho_0)\rho_1
```

```math
\alpha_{spatial}
=
D J_l(\rho_0)[\rho_1]\rho_1
+
J_l(\rho_0)\rho_2
```

を得る。

その後 body coordinates へ変換する。

## 8.4 intermediate 12-D covariance

parameter residual を評価するために必要な SG-derived random quantities は、

```math
\xi_k
=
\begin{bmatrix}
a_{S,k}\\
\rho_{0,k}\\
\rho_{1,k}\\
\rho_{2,k}
\end{bmatrix}
```

の 12-D とする。

SG pseudoinverse の derivative rows を使い、`Cov(xi_k)` の full `12 x 12` covariance を構成する。

translation / rotation の cross covariance も、`Omega` の cross block と同一 window の SG linear operator から構成する。

## 8.5 generalized-acceleration covariance

最終 SG-derived quantity を、

```math
z_k
=
\begin{bmatrix}
s_{S,k}^{SG}\\
\alpha_k^{SG}
\end{bmatrix}
```

とする。

map

```math
\xi_k
\mapsto
z_k
```

の Jacobian を `G_k` として、

```math
\Sigma_k
=
G_k
\operatorname{Cov}(\xi_k)
G_k^T
```

を求める。

これが point estimation で使う local full `6 x 6` acceleration covariance である。

`R_SG` uncertainty が specific acceleration に与える影響を含める。

`a` と rotation fit の cross correlation も full version では含める。

---

# 9. parameter chart

physical calculation は SI unit のまま行う。

deterministic residual の fixed-reference nondimensionalization は行わない。

nominal vehicle model は、

- parameter chart origin
- optimizer initial point

としてだけ使用する。

## 9.1 mass

```math
m
=
m_0 e^{q_m}
```

## 9.2 inertia

nominal inertia `J_0` から、

```math
\Sigma_0
=
\frac{1}{2}\operatorname{tr}(J_0)I
-
J_0
```

```math
B_0
=
\Sigma_0^{1/2}
```

を作る。

candidate second moment は、

```math
\Sigma(q)
=
B_0
\exp(S(q_\Sigma))
B_0
```

とする。

inertia は、

```math
J(q)
=
\operatorname{tr}(\Sigma(q))I
-
\Sigma(q)
```

とする。

この chart により physical inertia admissibility を保つ。

## 9.3 CoG

```math
c(q)
=
c_0 + q_c
```

`q_c` は meter 単位である。

## 9.4 rotor force effectiveness

```math
f_i(q)
=
f_{i,0} e^{q_{f_i}}
```

4 rotor を独立に保持する。

---

# 10. exact common-scale gauge

common-scale symmetry は、

```math
(m,J,f_1,f_2,f_3,f_4)
\mapsto
(\lambda m,\lambda J,\lambda f_1,\lambda f_2,\lambda f_3,\lambda f_4)
```

である。

14-D chart 上の direction は、

```math
v_{scale}
=
(1,1,1,1,0,0,0,0,0,0,1,1,1,1)^T
```

とする。

default estimator では各 optimization step `p` に、

```math
v_{scale}^T p = 0
```

を KKT で hard constraint として課す。

unknown near-ridge に scientific singular-value cutoff を入れない。

最終 ridge analysis は KKT constraint や LM damping を加えていない raw data Jacobian から行う。

---

# 11. actuator history

rotor / gimbal actuator history は parameter objective と diagnostic rollout で同一実装を共有する。

特に gimbal command を actual gimbal angle と直接同一視しない。

measured initial gimbal position から開始し、

```text
delayed command
-> advance_actuators
-> actual actuator state
```

とする。

gimbal rate limit / angle limit が存在する場合、それを `advance_actuators` と同一 semantics で扱う。

rotor / gimbal の actual state history を保存する。

active clamp / rate-limit count も保存する。

---

# 12. robot wrench

actual actuator state と candidate parameter から、

```math
w_{robot,k}
=
\begin{bmatrix}
F_{B,k}\\
\tau_{B,k}
\end{bmatrix}
```

を計算する。

この wrench generator は既存の actuator wrench implementation を使用する。

---

# 13. Newton--Euler prediction

pose sensor point と CoG の body-frame lever arm を、

```math
\ell_S
=
p_{S/B} - c
```

とする。

angular acceleration prediction は、

```math
\hat\alpha_k
=
J^{-1}
\left(
\tau_{B,k}
-
\omega_k
\times
J\omega_k
\right)
```

とする。

pose sensor point specific acceleration prediction は、

```math
\hat s_{S,k}
=
\frac{F_{B,k}}{m}
+
\hat\alpha_k \times \ell_S
+
\omega_k
\times
\left(
\omega_k \times \ell_S
\right)
```

とする。

---

# 14. parameter residual

各 SG center time の residual は、

```math
r_k
=
\begin{bmatrix}
s_{S,k}^{SG} - \hat s_{S,k}\\
\alpha_k^{SG} - \hat\alpha_k
\end{bmatrix}
```

とする。

pose residual は含めない。

IMU residual は含めない。

prior は含めない。

raw residual wrench penalty は含めない。

trajectory reconstruction error は含めない。

---

# 15. objective と `1/N`

default point-estimation objective は、

```math
L(q,\delta_r,\delta_g)
=
\frac{1}{2}
\sum_{k=1}^{N}
r_k^T
\Sigma_k^\dagger
r_k
```

とする。

`1/N` は掛けない。

理由は、bag ごとの factor を後段で合成するとき、sample count に応じた情報量を保持するためである。

ただし single bag の point estimate について、

```math
L_{mean}
=
\frac{1}{N}L
```

は constant scaling である。

そのため `sum` と `mean` の比較は scientific ablation ではなく regression / invariance check とする。

以下を確認する。

```math
q_{sum}
\simeq
q_{mean}
```

また、同一 point で、

```math
H_{sum}
\simeq
N H_{mean}
```

となることを確認する。

solver stopping の scale dependence により point estimate が有意に異なる場合のみ、実装問題として記録する。

---

# 16. lag

lag parameter は、

```math
\delta_r
```

```math
\delta_g
```

の split rotor / gimbal lag を default とする。

smooth ZOH は lag search continuation のためだけに使用する。

最終 objective、最終 parameter refinement、最終 report は strict ZOH を使用する。

既存の、

```text
smooth continuation
-> strict lag screen
-> physical refinement
```

の構造を基礎として再利用する。

lag search hyperparameter は ablation 対象とする。

---

# 17. raw residual wrench

SG trajectory が要求する wrench を、

```math
w_{req}
```

とする。

modeled actuator wrench を、

```math
w_{robot}
```

とする。

raw inverse-dynamics residual wrench を、

```math
w_{raw}
=
w_{req} - w_{robot}
```

とする。

`w_raw` は parameter objective に入れない。

diagnostic quantity として保存する。

---

# 18. trajectory-fitted external wrench

parameter fit 完了後、parameter を固定した別問題として external wrench history を fitting する。

これを、

```math
w_{replay}
```

とする。

目的は SG trajectory に整合する trajectory reconstruction を得ることである。

この replay optimization は parameter estimation へ feedback してはならない。

---

# 19. cross-time covariance correction

point estimate は local marginal `Sigma_k` による working-independence objective で求める。

その後、SG window overlap による cross-time covariance を使い parameter uncertainty を補正する。

default fit point `q_hat` における residual Jacobian を `J_k`、local weight を、

```math
W_k
=
\Sigma_k^\dagger
```

とする。

naive curvature matrix を、

```math
A
=
\sum_k
J_k^T W_k J_k
```

とする。

SG-derived generalized acceleration の cross-time covariance を、

```math
C_{k\ell}
=
\operatorname{Cov}(z_k,z_\ell)
```

とする。

Godambe / sandwich middle matrix を、

```math
B
=
\sum_{k,\ell}
J_k^T
W_k
C_{k\ell}
W_\ell
J_\ell
```

とする。

exact gauge section 上で naive covariance と overlap-corrected covariance を両方求める。

```text
parameter_covariance_naive
parameter_covariance_overlap_corrected
```

を必ず保存する。

この cross-time correction は point estimator を再実行しない post-processing とする。

---

# 20. ridge output

最終 point estimate 周辺の raw whitened Jacobian を保存する。

以下を保存する。

```text
raw whitened Jacobian
J^T J
singular values
right singular vectors
machine-precision numerical rank
exact scale gauge direction
J v_scale diagnostic
```

scientific ridge threshold は設けない。

KKT / LM damping を information matrix に加えない。

---

# 21. default single validation script

entry point:

```text
minimal/single_bag_savgol_estimator.py
```

この script は 1 bag だけを処理する。

algorithmic feature の default は、この plan で定める default configuration を使う。

SG window は scientific hyperparameter なので、暗黙に新しい値を発明しない。

入力 config または CLI で明示する。

SG degree の default は現在の geometric SG と同じ degree 5 とする。

---

# 22. ablation runner

entry point:

```text
minimal/single_bag_savgol_ablation.py
```

ablation runner は case ごとに独立に実行する。

ある case が失敗しても experiment 全体を停止してはならない。

## 22.1 case failure handling

各 case を top-level try/except で保護する。

case が失敗した場合も、必ず case directory と JSON を書く。

最低限以下を保存する。

```json
{
  "status": "failed",
  "case_name": "...",
  "source_commit": "...",
  "failure_stage": "...",
  "exception_type": "...",
  "message": "...",
  "elapsed_seconds": 0.0
}
```

可能であれば traceback text も保存する。

case failure は ablation experiment 自体の failure ではない。

次 case へ進む。

## 22.2 optimization non-success

optimizer が、

- nfev limit
- trust-region collapse
- solver failure
- non-finite evaluation
- no valid strict lag solution

等で終了した場合、その case を隠さない。

`completed` と `failed` を区別する。

失敗 reason を JSON に残す。

## 22.3 ablation experiment root summary

全 case 完了後、root summary に、

```text
case name
status
failure reason
elapsed time
point estimate
lag
common evaluation metrics
ridge summary
uncertainty summary
```

を集約する。

---

# 23. fixed ablation cases

以下は agent の判断で削除・置換しない。

## 23.1 covariance

```text
default_full_covariance

cov_identity
cov_diagonal
cov_block_s_alpha
cov_full_no_R_uncertainty_in_s
cov_full_no_position_rotation_cross
cov_global_full
```

意味は以下である。

### `cov_identity`

local covariance weighting を使わない naive control。

### `cov_diagonal`

full `6 x 6` covariance の diagonal だけ使う。

### `cov_block_s_alpha`

```math
\Sigma
=
\begin{bmatrix}
\Sigma_{ss} & 0\\
0 & \Sigma_{\alpha\alpha}
\end{bmatrix}
```

とする。

### `cov_full_no_R_uncertainty_in_s`

specific acceleration、

```math
s=R^T(a-g)
```

への uncertainty propagation で `R` を確定値扱いする。

### `cov_full_no_position_rotation_cross`

raw local pose covariance の position-rotation cross block を 0 とする。

### `cov_global_full`

local window ごとの covariance を使わず、bag 全体に一つの full covariance を使う比較 case。

## 23.2 cross-time covariance post-processing

default point estimate に対し、

```text
naive parameter covariance
overlap-corrected parameter covariance
```

を両方計算する。

これは optimizer を再実行する case ではない。

## 23.3 KKT

KKT ON/OFF を比較する。

scale-direction initial offset を複数用意する。

少なくとも、

```text
negative scale offset
zero scale offset
positive scale offset
```

を比較する。

KKT ON/OFF の双方で同じ initial-offset set を使う。

KKT OFF case が失敗しても experiment を停止しない。

## 23.4 lag

以下を比較する。

```text
lag_zero
lag_fixed
lag_common_estimated
lag_split_estimated
lag_split_strict_only
```

`lag_fixed` の fixed value は input/config に明示された値、または既に定められた data-derived initial value を使う。

agent が新しい固定 lag 値を発明しない。

## 23.5 actuator propagation

```text
actuator_stateful
actuator_direct_command
```

を比較する。

`actuator_stateful` は measured initial actuator state と `advance_actuators` を使う。

`actuator_direct_command` は delayed command を actual actuator state とみなす naive case。

active-set count を両 case で保存する。

## 23.6 SO(3) geometric correction

default geometric map と、

```text
omega approximately rho1
alpha approximately rho2
```

とする naive local-rotation-vector derivative を比較する。

## 23.7 solver

default custom KKT-LM と、gauge section を 13-D に明示 parameterize した standard least-squares solver を比較する。

比較の目的は custom LM/trust implementation 自体が必要かを判断することである。

standard solver case にも exact gauge ambiguity を残さない。

## 23.8 Jacobian

analytic Jacobian と finite-difference Jacobian を比較する。

これは scientific model ablation ではなく implementation complexity / runtime comparison とする。

## 23.9 external-wrench replay

raw residual wrench diagnostic と trajectory-fitted external-wrench reconstruction の価値を比較する。

parameter point estimate は変えない。

## 23.10 `naive_all`

主要な追加処理を可能な範囲で外した展示用 case を一つ用意する。

これは個々の処理の causal contribution 判定には使わない。

---

# 24. hyperparameter / robustness ablation

以下も既に議論した検証対象として実装する。

## 24.1 SG window

複数 window width を sweep できるようにする。

window values は CLI / config から与える。

agent が実験結果を見て勝手に追加値を hand-pick しない。

## 24.2 SG degree

degree を複数指定して比較可能にする。

default は degree 5。

## 24.3 lag initialization

lag initial value を複数 seed で比較可能にする。

## 24.4 lag bounds

lag search bound を複数設定で比較可能にする。

## 24.5 smooth continuation schedule

smooth width schedule を複数設定で比較可能にする。

## 24.6 LM settings

以下の numerical solver settings を sweep 可能にする。

```text
initial damping
initial trust radius
maximum trust radius
minimum trust radius
acceptance ratio
gtol
ftol
xtol
```

## 24.7 termination limits

```text
smooth max nfev
strict max nfev
strict alternation limit
```

を比較可能にする。

## 24.8 initial physical chart point

nominal chart origin 周辺の複数 initial point を比較できるようにする。

## 24.9 actuator limits / rate

actuator limit / rate が実データ中で active になっているかをまず記録する。

比較値が input/config で明示されていない限り、agent が arbitrary な alternate limit を発明しない。

## 24.10 actuator time constant

0 と、input/config で明示された candidate value を比較できるようにする。

agent が新しい time constant value を発明しない。

## 24.11 bag time segment

同一 bag 内の start/end を変更した区間比較を可能にする。

区間は config で明示する。

---

# 25. common evaluation

ablation case ごとに objective definition が異なるため、各 case 自身の objective 値だけを横比較しない。

全 case の final point に、同じ reference evaluator を適用する。

reference evaluator は default full SG covariance を使う。

```math
L_{reference}
=
\frac{1}{2}
\sum_k
r_k^T
\Sigma_{k,full}^\dagger
r_k
```

を common score とする。

さらに単位を保った、

```text
specific-acceleration RMSE [m/s^2]
angular-acceleration RMSE [rad/s^2]
```

を別々に保存する。

以下も共通で保存する。

```text
physical parameters
nominal -> estimated delta
rotor lag
gimbal lag
raw residual wrench
trajectory-fitted residual wrench
free / reconstruction diagnostic arrays
measured sensor comparison
raw whitened Jacobian spectrum
runtime
nfev
solver termination
actuator active-set counts
naive parameter covariance
overlap-corrected parameter covariance
```

固定の「何 % 以上なら採用」の threshold を設けない。

値をそのまま比較可能にする。

---

# 26. outputs

新しい output root は、

```text
minimal/outputs/
```

とする。

その下に **実装実行時の source commit hash** を必ず入れる。

例:

```text
minimal/outputs/
    7fecffe27aa25865c70669f919cefb16b624c46e/
        default/
            ...
        ablation/
            ...
```

実際に source code が新しい commit に進んだ後は、その実行時 commit hash を使う。

output JSON には必ず、

```text
source_commit
base_plan_commit
```

を保存する。

`base_plan_commit` は、

```text
7fecffe27aa25865c70669f919cefb16b624c46e
```

とする。

---

# 27. ablation output structure

ablation run ごとに一つの run directory を作る。

```text
outputs/<source-commit>/ablation/<run-id>/
    manifest.json
    summary.json
    cases/
        <case-name>/
            status.json
            result.json
            arguments.json
            timing.json
            arrays.npz
            report.pdf
            ...
```

失敗 case でも `status.json` / `result.json` を残す。

---

# 28. ablation ごとの必須 PDF

各 ablation case について **一つの `report.pdf`** に、最低限以下の 4 項目をまとめる。

case が parameter optimization に失敗し、物理 parameter point が得られない場合は、得られたところまでの情報と failure reason を PDF に出す。

## 28.1 trajectory

trajectory page では比較を増やしすぎない。

表示する trajectory は、

```text
observed trajectory
estimated parameters + trajectory-fitted residual wrench による reconstructed trajectory
```

の 2 本を基本とする。

free rollout や raw-residual-wrench replay trajectory を、この trajectory comparison page に追加しない。

trajectory-fitted external wrench が case algorithm の一部に存在しない場合でも、**standardized evaluation-only reconstruction** として parameter fit 後に計算してよい。

この standardized reconstruction は case の parameter optimization へ feedback してはならない。

## 28.2 residual wrench

同じ plot に最低限、

```text
raw SG inverse-dynamics residual wrench
trajectory-fitted external wrench
```

を表示する。

case 内に他の wrench 相当 history が存在する場合、それも比較可能なように表示する。

色だけで区別しない。

各 wrench history に、

```text
solid
dashed
dotted
dash-dot
```

等の異なる line style を割り当て、白黒印刷でも識別できるようにする。

6 wrench components の単位を明示する。

## 28.3 measured sensor comparison

実測 sensor と、trajectory reconstruction から得られる predicted sensor quantities を比較する。

最低限、

```text
gyro
specific force
```

を表示する。

SG-implied sensor series を既に計算している場合は併記してよいが、実測との比較が分からなくなるように曲線を増やさない。

## 28.4 nominal -> estimated parameters

nominal physical values と estimated physical values を比較する。

最低限、

```text
mass
inertia
CoG
rotor force effectiveness
lag
```

を読み取れるようにする。

absolute value と nominal からの変化量を記載する。

---

# 29. case が report quantity を直接持たない場合

ablation case の algorithm が、

- fitted external wrench
- trajectory reconstruction
- sensor reconstruction

等を内部処理として持たない場合もある。

その場合、case の final parameter point を固定した **共通 evaluation pipeline** で必要 quantity を計算してよい。

ただし共通 evaluation quantity を case algorithm に feedback してはならない。

つまり、

```text
fit
-> freeze result
-> common evaluation / report generation
```

の順序を厳守する。

---

# 30. test policy

`minimal/tests/` に新しい test を置く。

test の目的は **implementation bug を除去すること** である。

real-data performance を保証する test にしてはならない。

## 30.1 禁止する performance-dependent assertion

以下を unit/integration test の pass/fail 条件にしない。

```text
real bag で optimizer が必ず success する
objective が一定値以下
trajectory RMSE が一定値以下
parameter が nominal に近い
lag が特定値に近い
residual wrench が小さい
singular value が特定 threshold を超える
ablation A が ablation B より良い
Godambe correction が必ず大きい / 小さい
```

これらは experiment result として JSON/PDF に出す。

test assertion にしない。

## 30.2 ablation-specific semantics

KKT OFF case に、

```math
v_{scale}^T p = 0
```

を要求する test を掛けない。

lag zero case に lag improvement を要求しない。

naive covariance case に full covariance の性質を要求しない。

feature を意図的に外した case に、その feature の成果を要求しない。

---

# 31. 必須 tests

以下は implementation correctness test として実装する。

## 31.1 import / syntax

新しい modules と entry scripts が import / compile できる。

## 31.2 parameter chart physicality

random finite chart point について、

- mass > 0
- inertia symmetric
- inertia positive definite
- second-moment chart が finite

を確認する。

## 31.3 parameter chart Jacobian

analytic physical Jacobian と central finite difference を比較する。

## 31.4 exact scale transformation

chart の scale direction に移動したとき、

```text
mass
inertia
force effectiveness
```

が同じ factor で scale し、CoG が不変であることを確認する。

## 31.5 KKT step

synthetic gauge-null least-squares problemで KKT ON のとき、

```math
v_{scale}^T p
```

が numerical zero になることを確認する。

KKT OFF にはこの assertion を適用しない。

## 31.6 Newton--Euler Jacobian

parameter residual Jacobian と central finite difference を synthetic data で比較する。

## 31.7 hover gravity sanity

水平静止状態で、

```math
a=0
```

なら、

```math
s=R^T(-g)
```

が 1 g を与えることを確認する。

## 31.8 free-fall sanity

```math
a=g
```

なら、

```math
s=0
```

になることを確認する。

## 31.9 SG polynomial exactness

noise-free polynomial translation trajectory について、対応 degree 以下の local polynomial value / derivative が numerical precision で復元されることを確認する。

## 31.10 SG rotation sanity

noise-free simple SO(3) trajectory について、`R`, `omega`, `alpha` の coordinate/frame convention が実装と一致することを確認する。

## 31.11 covariance shape / symmetry

各 local covariance が期待 shape を持ち、finite な場合は symmetric であることを確認する。

full covariance の block extraction が diagonal / block / cross-zero ablation で意図どおり変わることを確認する。

## 31.12 covariance propagation Jacobian

`xi -> (s, alpha)` の propagation Jacobian を central finite difference と比較する。

## 31.13 sum / mean invariance algebra

同一 residual/Jacobian array に対して、

```math
L_{sum}
=
N L_{mean}
```

```math
H_{sum}
=
N H_{mean}
```

を確認する。

real optimizer の最終解一致は hard test にしない。

## 31.14 actuator history consistency

同一 initial state / commands / lag / times を与えたとき、parameter objective と standardized rollout が同じ actuator-history generator を通ることを確認する。

同一 query grid では actual thrust / gimbal history が一致することを確認する。

## 31.15 active-set reporting

clamp / rate-limit を意図的に発生させた synthetic command で active-set count が正しく記録されることを確認する。

## 31.16 strict ZOH final evaluation

smooth continuation を使用した場合も、final result が strict ZOH evaluator から生成されていることを確認する。

## 31.17 raw residual wrench identity

同一 SG state / parameter / actuator state に対し、

```math
w_{raw}
=
w_{req} - w_{robot}
```

が component-wise に成立することを確認する。

## 31.18 report pipeline isolation

standardized evaluation / replay を ON/OFF しても、既に得られた parameter objective result が書き換えられないことを確認する。

## 31.19 ablation failure continuation

test-only の deliberate failure case を一つ挿入し、

```text
case A completed
case B deliberate failure
case C completed
```

のように、B の failure 後も C が実行されることを確認する。

B の failure JSON に reason が残ることを確認する。

## 31.20 output commit path

output directory と result JSON に source commit / base plan commit が記録されることを確認する。

## 31.21 report smoke test

synthetic result から `report.pdf` が生成できることを確認する。

plot style mapping が wrench histories に異なる line styles を割り当てることを確認する。

PDF の numerical quality を pass/fail 判定しない。

---

# 32. hyperparameter recording

すべての run で、使用した hyperparameter を JSON に完全保存する。

最低限以下を含む。

```text
SG window
SG degree

covariance mode
R uncertainty propagation ON/OFF
position-rotation cross covariance ON/OFF
local/global covariance mode
cross-time correction ON/OFF

KKT ON/OFF

lag mode
lag initial values
lag bounds
smooth continuation widths

actuator time constants
actuator limits
gimbal rate limit

solver type
nfev limits
LM damping settings
trust-region settings
ftol
xtol
gtol

initial physical chart coordinate

bag path
bag start
bag end
```

deprecated no-op options は新実装へ持ち込まない。

---

# 33. agent-proposed improved ablation candidates

ここだけは agent に新しい候補を追加する判断を許可する。

目的は、default / fixed ablation を実装・実行した上で、より良い algorithmic variant が明確に考えられる場合、その候補を experiment として残すことである。

## 33.1 expected properties を事前に書く

agent-proposed case を実行する前に、その case directory に proposal metadata を書く。

最低限、

```json
{
  "case_name": "agent_candidate__...",
  "changed_from_default": "...",
  "expected_property": "...",
  "reason_for_expectation": "...",
  "cheat_guard_acknowledged": true
}
```

を残す。

`expected_property` は事前に明示する。

許容される expected property の例は、

```text
same scientific objective で convergence failure を減らす
initialization dependence を減らす
solver hyperparameter dependence を減らす
same final point をより短時間で得る
same reference evaluator でより良い stationary point に到達する
numerical singularity / non-finite trial を減らす
lag search の strict final solution をより安定に見つける
uncertainty calculation を同じ point estimate のままより整合的にする
```

等である。

## 33.2 cheat 禁止

agent-proposed improved case では以下を禁止する。

### desired parameter value の注入禁止

nominal / 期待 parameter 値へ引き戻す penalty を追加しない。

prior を deterministic parameter objective に入れない。

### replay wrench feedback 禁止

`w_raw` や `w_replay` を parameter objective に入れない。

trajectory-fitted external wrench を parameter update に feedback しない。

### trajectory reconstruction score による parameter selection 禁止

parameter estimator の scientific objective に含まれない trajectory reconstruction RMSE を、hidden model-selection criterion として使わない。

### IMU を hidden objective に追加しない

IMU は diagnostic という fixed rule を破らない。

### observed pose residual を hidden objective に追加しない

pose itself は objective に含めないという fixed rule を破らない。

### 他 ablation の答えを oracle initial value に使わない

他 case の最終 parameter estimate を、agent candidate の hidden warm start として使わない。

fixed design として明示された smooth->strict continuation は除く。

### bag-specific manual tuning 禁止

case output を見た後に、その bag だけに都合のよい hard-coded constant を追加しない。

### report quantity の feedback 禁止

common evaluation / PDF generation で得た値を optimization 内へ戻さない。

## 33.3 agent candidate は固定 cases を置換しない

agent candidate が良く見えても、その場で default implementation を置換しない。

まず ablation result として並べる。

default adoption の判断は experiment output を確認した後に行う。

---

# 34. standardized report semantics

case 間比較を壊さないため、report quantity の定義を共通化する。

## 34.1 trajectory report

trajectory は、

```text
observed
estimated model + trajectory-fitted external wrench
```

のみを主要比較として表示する。

## 34.2 wrench report

最低限、

```text
raw SG inverse-dynamics residual wrench
trajectory-fitted external wrench
```

を同じ axes で比較する。

wrench history ごとに色 + line style の両方を変える。

## 34.3 sensor report

同一 reconstruction に対する predicted sensor と measured sensor を比較する。

## 34.4 parameter report

nominal と estimated の absolute values / differences を表示する。

---

# 35. outputs に必ず残す raw data

PDF だけに情報を閉じ込めない。

少なくとも NPZ / JSON として、

```text
SG time
SG R
SG omega
SG a_S
SG alpha
SG s
local Omega
local Sigma_xi
local Sigma_z

actual thrust history
actual gimbal history

modeled wrench
required wrench
raw residual wrench
fitted external wrench

residual acceleration
whitened residual
raw Jacobian
whitened Jacobian

singular values
right singular vectors

naive parameter covariance
overlap-corrected parameter covariance

reconstructed trajectory
predicted sensor series
measured sensor series
```

を保存可能にする。

---

# 36. failure case の PDF

case が途中で失敗し、通常の 4 項目 PDF を完全には生成できない場合、PDF generation 自体を理由に ablation runner を止めない。

利用可能な結果まで表示し、先頭 page に、

```text
status
failure stage
exception type
message
```

を記載する。

parameter point が存在しない場合は、存在しない quantity を捏造しない。

---

# 37. implementation order

エージェントは以下の順序で作業する。

## Phase 1: legacy cleanup

1. HEAD の確認。
2. 現 `minimal/` の legacy snapshot 化。
3. 新しい clean `minimal/` skeleton 作成。
4. 新実装から legacy を直接 import しない状態にする。

## Phase 2: reusable core extraction

1. geometric SG extraction。
2. smooth ZOH extraction。
3. SI-unit parameter chart。
4. actuator-history generator。
5. Newton--Euler residual。
6. analytic Jacobian。
7. KKT solver。

## Phase 3: SG covariance

1. local translation / rotation residual。
2. local full `Omega`。
3. `Sigma_xi`。
4. `xi -> (s, alpha)` propagation。
5. covariance modes。

## Phase 4: default single-bag estimator

1. single-bag loader。
2. strict objective。
3. lag search。
4. KKT fit。
5. ridge output。
6. raw residual wrench。
7. naive + overlap-corrected uncertainty。
8. replay reconstruction。
9. sensor diagnostic。
10. report / JSON / NPZ。

## Phase 5: tests

performance-independent tests を実装し、通す。

## Phase 6: fixed ablation runner

この plan で列挙した fixed cases を実装する。

case failure continuation を実装する。

## Phase 7: hyperparameter sweeps

SG / lag / solver / initial-point / segment 等の sweep infrastructure を追加する。

## Phase 8: agent-proposed improved cases

固定 cases が実行可能になった後だけ agent candidate を追加する。

proposal metadata と cheat guard を必須とする。

---

# 38. 完了条件

以下を満たした時点で implementation phase を完了とする。

- `minimal/` が新実装中心の clean tree になっている。
- 旧実装が legacy snapshot に保存されている。
- 新実装は legacy code を runtime import していない。
- single-bag default estimator が存在する。
- ablation runner が存在する。
- ablation case failure で全 experiment が停止しない。
- failure reason が JSON に残る。
- performance-independent tests が通る。
- fixed ablation cases が runner から選択可能である。
- `1/N` sum/mean invariance test が存在する。
- naive / overlap-corrected covariance の両方が出る。
- source commit hash ごとの output namespace が存在する。
- 各 completed ablation case に `report.pdf` が存在する。
- report PDF に trajectory / wrench / measured sensor / nominal-vs-estimated の 4 項目がある。
- trajectory page は observed と fitted-wrench reconstruction を主要 2 本として表示する。
- wrench page は raw residual と fitted external wrench を同時表示する。
- wrench curve は line style でも区別できる。
- raw JSON / NPZ が保存される。
- agent candidate は fixed cases と別 namespace で保存される。
- agent candidate proposal に expected property と変更点が事前記録される。
- agent candidate が cheat 禁止事項を破っていない。

---

# 39. 実装エージェントへの最終禁止事項

この plan に書かれていない scientific objective の項を勝手に追加しない。

以下を勝手に追加しない。

```text
prior penalty
nominal regularization
residual-wrench penalty
trajectory-error penalty
pose-error penalty
IMU-error penalty
scientific SVD ridge cutoff
multi-bag joint objective
fixed-reference nondimensionalized residual
arbitrary parameter bounds
arbitrary numerical parameter scaling
```

default scientific design を「改善した方が良さそう」という理由だけで書き換えない。

新しい改善案を試したい場合は、必ず `agent_candidate__...` ablation として分離する。

その結果を見ずに default へ昇格させない。

---

# 40. 最終的な default estimation problem

default single-bag estimator が解く問題は、

```math
\min_{q,\delta_r,\delta_g}
\frac{1}{2}
\sum_{k=1}^{N}
r_k(q,\delta_r,\delta_g)^T
\Sigma_k^\dagger
r_k(q,\delta_r,\delta_g)
```

subject to the exact common-scale gauge section,

```math
v_{scale}^T q = 0
```

である。

各 residual は、

```math
r_k
=
\begin{bmatrix}
R_k^T(a_{S,k}-g_W)-\hat s_{S,k}\\
\alpha_k-\hat\alpha_k
\end{bmatrix}
```

である。

parameter point estimate を求めた後、

- raw Jacobian ridge
- naive local curvature uncertainty
- SG overlap-corrected uncertainty
- raw residual wrench
- trajectory-fitted external wrench
- reconstructed trajectory
- measured sensor diagnostics

を後処理として得る。

これが新しい single-bag implementation の固定仕様である。
