# Single-bag geometric SG rigid-body estimator
# 現行実装修正・診断追加 plan

## 0. 基準 commit とこの文書の目的

この文書は、`hiroya1224/ProbTF-demo` の以下の commit を基準とする。

```text
fb45718f1f9a4d3d4b94c35d4061fa17c07bd8d8
Run production three-bag ablations
```

この commit より後の実装では、新しい estimator / ablation entry script を追加しない。

現在すでに存在する以下の実装を修正する。

```text
ros/examples/grape-param-estim/minimal/
    single_bag_savgol_estimator.py
    single_bag_savgol_ablation.py
    single_bag_savgol_core.py
    single_bag_savgol_covariance.py
    single_bag_savgol_reports.py
    single_bag_wrench_replay.py
    savgol_trajectory.py
    smooth_command.py
    run_full_ablation.sh
    tests/
```

旧実装を置いた

```text
minimal/legacies/
```

は今回の修正対象にしない。

既存 production output は過去の実行結果として保持し、上書き・再解釈・移動しない。

---

# 1. 今回の方針

今回の修正では、以下を原則とする。

1. **有限差分による数値微分を production estimator から完全に除去する。**
2. parameter optimization は解析 Jacobian のみを用いる。
3. covariance propagation も解析微分のみで行う。
4. scientific ridge は、最終的に選択された rotor/gimbal lag に条件づけた **14-D physical parameter Jacobian** から求める。
5. stateful actuator propagation、geometric SO(3) derivative、full SG covariance、smooth lag continuation、trajectory-fitted external wrench 等、物理・幾何・統計モデルとして自然な処理は、数値的効果が小さいという理由だけでは削除しない。
6. 既存 ablation の結果を踏まえ、新しい主目的は **covariance weighting が parameter ridge と point estimate にどう作用しているかを診断可能にすること** とする。
7. overlap-aware parameter covariance correction は残し、未補正の covariance と必ず並べて出力する。
8. 既存の実 rosbag 3本を検証対象として明示的に使用する。
9. unit / integration test は implementation correctness のみを検査し、real-bag performance を pass/fail 条件にしない。
10. この文書に書かれていない新しい scientific objective、prior、regularization、parameter bound、residual term を追加しない。

---

# 2. 今回使用する実 rosbag

現在 `minimal/bag_jsons/` に入っている以下の3本をそのまま用いる。

## 2.1 `single_rosbag_1.json`

```text
bag:
/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_4_2026-06-12-17-33-59.bag

start:
19.0 s

end:
25.0 s
```

## 2.2 `single_rosbag_2.json`

```text
bag:
/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_6_2026-06-12-17-40-34.bag

start:
25.5 s

end:
31.0 s
```

## 2.3 `single_rosbag_succeeded.json`

```text
bag:
/home/leus/catkin_ws/bags/grape-drone/20260613_grape_hovering/20260613_grape_hovering_1_2026-06-13-13-44-01.bag

start:
65.0 s

end:
75.0 s
```

これらの path / interval を agent が変更しない。

別 interval を試す場合は、既存の `sweep_config` の `segments` から明示指定する。

---

# 3. 現行コードに対応する用語定義

以降では、曖昧な「raw」「weighted」「ridge」等の語を単独では使わず、現在の code field と対応させる。

## 3.1 physical chart coordinate

`single_bag_savgol_core.py` の `SiParameterChart` が扱う 14-D coordinate を、

```text
physical_coordinate
```

と呼ぶ。

成分は現在の実装順に、

```text
0       mass log coordinate
1:7     symmetric second-moment matrix-log coordinate
7:10    CoG displacement [m]
10:14   rotor force-effectiveness log coordinates
```

である。

`result.json` では、

```text
parameters.chart_coordinate
```

に対応する。

---

## 3.2 physical parameters

`SiParameterChart.decode_with_jacobian()` が生成する `VehicleParameters` のうち今回推定するものは、

```text
mass
inertia
cog_offset
force_effectiveness
```

である。

report / JSON では、

```text
parameters.estimated.mass_kg
parameters.estimated.inertia_kg_m2
parameters.estimated.cog_position_body_m
parameters.estimated.force_effectiveness
```

に対応する。

---

## 3.3 SG output

`PoseSgEvaluation` の以下を用いる。

```text
sensor_acceleration_world
body_rotation
body_angular_velocity
body_angular_acceleration
local_windows
```

記号では、

```math
a_{S,k}
R_k
\omega_k
\alpha_k
```

と書く。

---

## 3.4 generalized acceleration data `z_k`

`single_bag_savgol_covariance.py` の `SgCovarianceEvaluation.z` に対応する 6-vector を、

```math
z_k
=
\begin{bmatrix}
s_{S,k}^{SG}\\
\alpha_k^{SG}
\end{bmatrix}
```

とする。

ここで、

```math
s_{S,k}^{SG}
=
R_k^T(a_{S,k}-g_W)
```

かつ、

```math
g_W
=
(0,0,-9.80665)^T
```

である。

code / NPZ では、

```text
sg_s
sg_alpha
```

に対応する。

---

## 3.5 acceleration residual

`DynamicsEvaluation.acceleration_residual` を、

```text
acceleration residual
```

と呼ぶ。

記号では、

```math
r_k
=
\begin{bmatrix}
s_{S,k}^{SG} - \hat s_{S,k}\\
\alpha_k^{SG} - \hat\alpha_k
\end{bmatrix}
```

である。

NPZ では、

```text
residual_acceleration
```

に対応する。

この 6-vector は covariance whitening 前の値である。

---

## 3.6 local SG acceleration covariance

`SgCovarianceEvaluation.local_sigma_z[k]` を、

```text
local generalized-acceleration covariance
```

または code field 名そのまま、

```text
local_Sigma_z
```

と呼ぶ。

記号では、

```math
\Sigma_{z,k}
=
\operatorname{Cov}
\begin{bmatrix}
s_{S,k}^{SG}\\
\alpha_k^{SG}
\end{bmatrix}
```

である。

---

## 3.7 whitening matrix

`SgCovarianceEvaluation.whitening[k]` を、

```math
W_k
```

と書く。

現在の実装では、

```math
W_k^T W_k
=
\Sigma_{z,k}^{\dagger}
```

となる symmetric whitening matrix である。

ここで `dagger` は Moore--Penrose pseudoinverse であり、machine-precision rank 判定のみを用いる。

---

## 3.8 whitened residual

`DynamicsEvaluation.whitened_residual` を、

```text
whitened residual
```

と呼ぶ。

```math
\tilde r_k
=
W_k r_k
```

である。

NPZ では、

```text
whitened_residual
```

に対応する。

---

## 3.9 physical acceleration Jacobian

`DynamicsEvaluation.acceleration_jacobian` を、

```text
raw physical acceleration Jacobian
```

と呼ぶ。

shape は、

```text
N x 6 x 14
```

で、

```math
J^{raw}_k
=
\frac{\partial r_k}{\partial q}
```

である。

NPZ では現在、

```text
raw_physical_jacobian
```

に対応する。

---

## 3.10 whitened physical Jacobian

`DynamicsEvaluation.whitened_jacobian` を、

```text
whitened physical Jacobian
```

と呼ぶ。

```math
J^{white}_k
=
W_k J^{raw}_k
```

である。

shape は、

```text
N x 6 x 14
```

である。

NPZ では、

```text
whitened_physical_jacobian
```

に対応する。

---

## 3.11 scientific ridge

この修正後、`EstimationResult.ridge` に入れる scientific ridge は、

```text
whitened physical Jacobian
```

を `6N x 14` に flatten した行列、

```math
J_{\mathrm{ridge}}
```

の SVD とする。

```math
J_{\mathrm{ridge}}
=
U S V^T
```

である。

lag coordinate を含めない。

したがって default case では、既知の common-scale gauge による exact null direction が1本存在することを期待する。

---

## 3.12 common-scale gauge

現在の、

```text
COMMON_SCALE_DIRECTION
```

をそのまま使う。

```math
v_{\mathrm{scale}}
=
(1,1,1,1,0,0,0,0,0,0,1,1,1,1)^T
```

である。

---

## 3.13 raw residual wrench

`DynamicsEvaluation.raw_residual_wrench` を指す。

```math
w_{\mathrm{raw},k}
=
w_{\mathrm{req},k}
-
w_{\mathrm{robot},k}
```

である。

これは parameter objective に入れない。

---

## 3.14 trajectory-fitted external wrench

`WrenchReplayResult.fitted_external_wrench` を指す。

parameter fit 完了後に parameter を固定して求める diagnostic quantity である。

parameter optimization へ feedback しない。

---

## 3.15 naive parameter covariance

`ParameterCovarianceResult.naive` を指す。

selected lag と final physical point における local curvature から、common-scale gauge を除いた subspace 上で求める covariance である。

---

## 3.16 overlap-corrected parameter covariance

`ParameterCovarianceResult.overlap_corrected` を指す。

overlapping SG windows が同じ raw pose sample を共有することで発生する cross-time covariance を、post-processing で反映した covariance である。

point estimate は変更しない。

---

# 4. default scientific model は変更しない

以下は current default のまま残す。

## 4.1 SI-unit dynamics

fixed-reference nondimensionalization は復活させない。

## 4.2 matrix-exponential second-moment inertia chart

`SiParameterChart` をそのまま用いる。

## 4.3 KKT common-scale gauge

default は KKT ON。

## 4.4 full SG covariance

default の、

```text
covariance_mode = "full"
```

を維持する。

今回の production result で極端な parameter が出たことを理由に、default を `identity` へ変更しない。

代わりに診断を追加して作用を明示する。

## 4.5 geometric SO(3) SG

`left_jacobian_with_directional_derivative()` を使う geometric derivative を default として残す。

`so3_naive_rotation_vector_derivatives` は ablation として残す。

## 4.6 stateful actuator propagation

default は、

```text
actuator_propagation = "stateful"
```

を維持する。

`actuator_direct_command` は ablation として残す。

## 4.7 split rotor/gimbal lag

default は、

```text
lag_mode = "split_estimated"
```

を維持する。

## 4.8 smooth continuation -> strict ZOH

現在の smooth continuation と strict lag screen / strict physical refinement を維持する。

final evaluation は必ず strict ZOH。

## 4.9 external-wrench replay

`fit_external_wrench_replay()` による standardized replay を維持する。

parameter objective には入れない。

---

# 5. 有限差分の完全除去

この項は今回の主要な algorithmic change である。

## 5.1 `single_bag_savgol_core.py`: parameter Jacobian mode を解析微分だけにする

現在の、

```text
jacobian_mode = "analytic" | "finite_difference"
```

という分岐を廃止する。

`SingleBagDynamicsProblem` は parameter residual / Jacobian を常に analytic path で評価する。

削除対象:

```text
finite_difference
finite-difference parameter Jacobian branch
finite-difference helper specific to parameter optimization
```

CLI / config に `jacobian_mode` を残さない。

JSON arguments にも `jacobian_mode` を出さない。

---

## 5.2 `single_bag_savgol_estimator.py`: CLI から finite-difference option を削除する

`--jacobian-mode` が存在する場合、削除する。

default / parser / `EstimatorConfig` に finite-difference switch を残さない。

---

## 5.3 `single_bag_savgol_ablation.py`: finite-difference fixed case を削除する

`FIXED_CASE_NAMES` から、

```text
jacobian_finite_difference
```

を削除する。

`jacobian_analytic` は、比較相手がなくなるため fixed ablation としては不要である。

ただし production result の archival directory は削除しない。

`naive_all` に現在入っている、

```text
jacobian_mode = "finite_difference"
```

も削除する。

`naive_all` も parameter Jacobian は analytic を使う。

したがって `naive_all` は、

```text
covariance identity
KKT off
lag zero
direct-command actuator
naive SO(3) derivatives
analytic parameter Jacobian
```

となる。

---

## 5.4 strict-ZOH lag の finite-difference ridge columns を削除する

現在 `estimate_single_bag()` は lag-estimated case で 15-D / 16-D global coordinate の lag columns を finite difference で作り、ridge に入れている。

この処理を全削除する。

lag の scientific diagnosis は `strict_lag_screen()` が返す strict-ZOH candidate costs によって行う。

---

# 6. covariance propagation の有限差分を解析式へ置換する

`single_bag_savgol_covariance.py` の `propagation_jacobian()` は現在、

```text
d alpha_B / d rho0
```

だけ centered finite difference を使っている。

ここを解析式へ置換する。

新しい script は作らない。

SO(3) Jacobian helper は `savgol_trajectory.py` 内で拡張する。

---

# 7. SO(3) left Jacobian の second directional derivative

現在、

```python
left_jacobian_with_directional_derivative(phi, direction)
```

は、

```math
J_l(\phi)
```

と、

```math
D J_l(\phi)[\eta]
```

を返す。

この helper に second directional derivative を計算する機能を追加する。

別 module は作らない。

---

## 7.1 notation

```math
A = \phi^\wedge
```

```math
E = \eta^\wedge
```

```math
F = \zeta^\wedge
```

```math
i = \phi^T \eta
```

```math
j = \phi^T \zeta
```

```math
h = \eta^T \zeta
```

とする。

現在の実装と同じく、

```math
J_l(\phi)
=
I + a A + b A^2
```

とする。

現在の first directional derivative は、

```math
D J_l[\eta]
=
aE
+
b(EA+AE)
+
c\,i\,A
+
d\,i\,A^2
```

である。

---

## 7.2 second directional derivative

新たに、

```math
D^2 J_l[\eta,\zeta]
```

を以下で実装する。

```math
D^2 J_l[\eta,\zeta]
=
c\,j\,E
+
d\,j\,(EA+AE)
+
b(EF+FE)
+
e\,ij\,A
+
c\,h\,A
+
c\,i\,F
+
f\,ij\,A^2
+
d\,h\,A^2
+
d\,i\,(FA+AF).
```

ここで `e`, `f` は scalar coefficient である。

---

## 7.3 non-small-angle coefficient

```math
r=\|\phi\|
```

として、

```math
e
=
\frac{
r^2\cos r
-
5r\sin r
+
8(1-\cos r)
}{
r^6
}
```

```math
f
=
\frac{
r^2\sin r
+
7r\cos r
+
8r
-
15\sin r
}{
r^7
}.
```

現在の `radius_squared < 1e-8` branch と同じ branch criterion を使用する。

---

## 7.4 small-angle series

small-angle branch では、

```math
e
=
\frac{1}{90}
-
\frac{r^2}{1680}
+
\frac{r^4}{75600}
-
\frac{r^6}{5987520}
```

```math
f
=
\frac{1}{630}
-
\frac{r^2}{15120}
+
\frac{r^4}{831600}
-
\frac{r^6}{77837760}.
```

現在の `a,b,c,d` series と同じ考え方で使用する。

agent が別の finite-difference approximation を入れない。

---

# 8. `propagation_jacobian()` の analytic `rho0` block

`single_bag_savgol_covariance.py` の、

```math
\xi
=
[a_S,\rho_0,\rho_1,\rho_2]
```

から、

```math
z
=
[s_S,\alpha_B]
```

への Jacobian をすべて解析式で作る。

---

## 8.1 specific-force block

現在の式を維持する。

```math
s_S
=
R^T(a_S-g_W)
```

```math
\frac{\partial s_S}{\partial a_S}
=
R^T
```

`R` uncertainty を含める default では、

```math
\frac{\partial s_S}{\partial\rho_0}
=
R^T
(a_S-g_W)^\wedge
J_l(\rho_0).
```

`cov_full_no_R_uncertainty_in_s` ablation ではこの block を 0 にする。

---

## 8.2 angular acceleration

spatial angular acceleration を、

```math
\alpha_s
=
D J_l(\rho_0)[\rho_1]\rho_1
+
J_l(\rho_0)\rho_2
```

とする。

body angular acceleration は、

```math
\alpha_B
=
R^T \alpha_s.
```

---

## 8.3 `rho1` derivative

direction `eta` に対し、

```math
D_{\rho_1}\alpha_s[\eta]
=
D J_l[\eta]\rho_1
+
D J_l[\rho_1]\eta.
```

body coordinate では、

```math
D_{\rho_1}\alpha_B[\eta]
=
R^T
D_{\rho_1}\alpha_s[\eta].
```

現在の analytic implementation を維持する。

---

## 8.4 `rho2` derivative

```math
D_{\rho_2}\alpha_B[\eta]
=
R^T J_l(\rho_0)\eta.
```

現在の analytic implementation を維持する。

---

## 8.5 `rho0` derivative

direction `zeta` に対し、

```math
D_{\rho_0}\alpha_s[\zeta]
=
D^2J_l[\rho_1,\zeta]\rho_1
+
D J_l[\zeta]\rho_2.
```

また、

```math
D_{\rho_0}(R^T\alpha_s)[\zeta]
=
R^T
\left[
\alpha_s^\wedge J_l(\rho_0)\zeta
+
D_{\rho_0}\alpha_s[\zeta]
\right].
```

この式で `propagation_jacobian()[3:, 3:6]` を構成する。

finite difference loop は削除する。

---

# 9. scientific ridge の修正

## 9.1 final lag に条件づける

最終的な、

```text
result.ridge
```

は、

```text
result.evaluation.whitened_jacobian
```

だけから作る。

これは `N x 6 x 14` である。

flatten して、

```text
6N x 14
```

にする。

---

## 9.2 ridge に lag coordinate を入れない

以下を scientific ridge から削除する。

```text
rotor lag column
gimbal lag column
common lag column
```

`GLOBAL_SPLIT_DIMENSION=16` は lag optimization 内部で必要なら残してよいが、ridge reporting dimension として使わない。

---

## 9.3 ridge JSON

`result.json["ridge"]` には最低限、

```text
dimension = 14
machine_numerical_rank
nullity
machine_rank_tolerance
singular_values
right_singular_vectors
exact_scale_gauge_direction
j_v_scale_norm
```

を出す。

default KKT-enabled solution について、

```text
nullity == 1
```

を real-bag performance test の hard assertion にはしない。

ただし result と report に明示する。

---

# 10. lag diagnosis の修正

lag は strict ZOH では連続微分による ridge analysis を行わない。

`strict_lag_screen()` の candidate evaluation を lag diagnosis の正本とする。

---

## 10.1 full candidate table

現在 trace に selected lag / candidate count / expansion は残るが、各 strict candidate pair の cost をすべて保存する。

split lag の場合、各 candidate について、

```text
rotor_lag_seconds
gimbal_lag_seconds
strict_cost
selected
```

を保存する。

common lag の場合、

```text
lag_seconds
strict_cost
selected
```

を保存する。

---

## 10.2 lag boundary flags

最終 result に、

```text
lag_diagnostics.rotor_at_lower_bound
lag_diagnostics.rotor_at_upper_bound
lag_diagnostics.gimbal_at_lower_bound
lag_diagnostics.gimbal_at_upper_bound
```

を追加する。

floating-point equality ではなく、strict candidate grid 上で selected candidate が bound candidate かどうかで判定する。

agent が新しい scientific tolerance を発明しない。

---

## 10.3 production result で確認すること

特に、

```text
single_rosbag_1
single_rosbag_succeeded
```

では production result で rotor lag が 0.2 s 上限付近にあったため、lag-bound sweep を行う。

sweep 値は `sweep_config` に事前に記述する。

実行後に結果を見て都合のよい bound を追加しない。

---

# 11. covariance weighting diagnosis の追加

default full covariance は維持する。

今回追加するものは objective の変更ではなく diagnosis である。

---

## 11.1 per-time covariance spectrum

各時刻 `k` の、

```math
\Sigma_{z,k}
```

について、

```text
eigenvalues
machine rank
condition number over retained eigenvalues
```

を求める。

NPZ へ、

```text
sigma_z_eigenvalues
sigma_z_machine_rank
sigma_z_retained_condition_number
```

を保存する。

---

## 11.2 whitening gain

各 retained eigenvalue `lambda` に対する、

```math
1 / sqrt(lambda)
```

を、

```text
whitening_gain
```

として保存する。

これは covariance の小さい方向が objective 内でどれほど増幅されるかを示す diagnostic である。

---

## 11.3 per-time Mahalanobis contribution

各時刻について、

```math
m_k
=
r_k^T
\Sigma_{z,k}^{\dagger}
r_k
=
\|\tilde r_k\|^2
```

を保存する。

NPZ 名:

```text
mahalanobis_contribution_per_time
```

JSON には、

```text
minimum
median
maximum
selected quantiles
```

を summary として保存する。

quantile set は report helper 内で一箇所に固定し、run ごとに変更しない。

---

## 11.4 covariance eigenmode contribution

```math
\Sigma_{z,k}
=
U_k
\Lambda_k
U_k^T
```

としたとき、

```math
c_{k,j}
=
\frac{
(u_{k,j}^T r_k)^2
}{
\lambda_{k,j}
}
```

を保存する。

これは「どの covariance eigenmode が weighted objective を支配したか」を見るための diagnostic である。

NPZ:

```text
covariance_eigenmode_residual
covariance_eigenmode_mahalanobis_contribution
```

---

# 12. full-covariance metric と identity metric の cross-evaluation

各 completed case の final point について、同じ `acceleration_residual` から2種類の scalar を出す。

## 12.1 full covariance metric

```math
L_{\mathrm{full}}
=
\frac12
\sum_k
r_k^T
\Sigma_{z,k,\mathrm{full}}^\dagger
r_k.
```

これは現在の `reference_objective_sum` と同じ意味を保つ。

名称を曖昧にしないため、

```text
full_covariance_objective_sum
```

を新たに併記する。

既存 `reference_objective_sum` は backward compatibility のため残してよい。

---

## 12.2 identity metric

```math
L_{\mathrm{identity}}
=
\frac12
\sum_k
r_k^T r_k.
```

を、

```text
identity_objective_sum
```

として保存する。

---

## 12.3 nominal point も同じ metric で評価する

final point だけでなく physical chart origin、

```text
q = 0
```

を同じ selected lag で評価する。

JSON:

```text
metric_cross_evaluation:
    nominal:
        full_covariance_objective_sum
        identity_objective_sum
        specific_acceleration_rmse_m_per_s2
        angular_acceleration_rmse_rad_per_s2
    estimated:
        full_covariance_objective_sum
        identity_objective_sum
        specific_acceleration_rmse_m_per_s2
        angular_acceleration_rmse_rad_per_s2
```

parameter selection にこの diagnostic を使わない。

---

# 13. covariance weighting と ridge の対応 diagnosis

## 13.1 unwhitened physical Jacobian

現在の、

```text
raw_physical_jacobian
```

を保存し続ける。

これは、

```math
J^{raw}
=
\partial r/\partial q
```

である。

その singular values は単位の異なる residual components をそのまま並べた数値量なので、scientific ridge の正本とはしない。

diagnostic としてのみ扱う。

---

## 13.2 whitened physical Jacobian

現在の、

```text
whitened_physical_jacobian
```

を scientific ridge の正本とする。

---

## 13.3 diagnostic SVD comparison

同一 final point について、

```text
unwhitened physical Jacobian SVD
whitened physical Jacobian SVD
```

を両方出す。

JSON 名を明確に分ける。

```text
ridge.unwhitened_diagnostic_singular_values
ridge.whitened_singular_values
```

既存 `singular_values` は whitened を意味するものとして backward compatibility 上残してよい。

---

## 13.4 parameter displacement in whitened ridge basis

physical chart origin から final point への displacement は、

```math
\Delta q
=
\hat q.
```

whitened ridge の right singular vector を `v_i` として、

```math
d_i
=
v_i^T \Delta q
```

を計算する。

さらに、

```math
\|\Delta q\|^2
```

に対する各 `d_i^2` の割合を保存する。

JSON / NPZ:

```text
parameter_displacement_ridge_coordinates
parameter_displacement_ridge_energy_fraction
```

これにより極端な parameter change が weak singular directions に沿っているかを直接確認する。

---

# 14. raw residual wrench と acceleration residual の closure diagnosis

現在の raw residual wrench は parameter objective へ入れない。

ただし acceleration residual との関係を correctness / interpretation diagnostic として出す。

---

## 14.1 rotational closure

現在の符号 convention に合わせて、

```math
\tau_{\mathrm{raw},k}
```

と、

```math
J
\left(
\alpha_k^{SG}
-
\hat\alpha_k
\right)
```

の component-wise difference を計算する。

NPZ:

```text
torque_acceleration_closure_error
```

summary JSON:

```text
torque_acceleration_closure_error_max_abs
torque_acceleration_closure_error_rms
```

---

## 14.2 translational closure

sensor lever arm を、

```math
\ell_S
=
p_{S/B}
-
c
```

とする。

current residual sign convention に合わせて、`F_raw` と `r_s`, `r_alpha`, `m`, `ell_S` から再構成した force residual の差を出す。

実装時に current `required_wrench` / `modeled_wrench` の sign convention と照合し、test で identity を固定する。

agent が report 上だけ符号を変更しない。

NPZ:

```text
force_acceleration_closure_error
```

JSON:

```text
force_acceleration_closure_error_max_abs
force_acceleration_closure_error_rms
```

---

# 15. principal inertia diagnosis

extreme inertia solution の読み取りを容易にするため、report / JSON に principal inertia を明示する。

nominal と estimated について、

```text
principal_moments_kg_m2
principal_axes_body
```

を保存する。

さらに、

```text
estimated / nominal principal-moment ratio
```

を保存する。

parameter objective は変更しない。

---

# 16. overlap covariance correction は残す

`parameter_covariances()` による、

```text
parameter_covariance_naive
parameter_covariance_overlap_corrected
```

を残す。

「効果が小さかった場合に削除する」ことを今回の実装内では行わない。

---

# 17. 現在の cross-time covariance model を明示する

現在の `cross_time_generalized_covariance()` は、window `k,l` が共有する raw sample に対して、

```math
\frac12
(
\Omega_k+\Omega_l
)
```

を使っている。

この revision では、agent が新しい raw-noise field model を独自に発明しない。

したがって formula 自体は変更しない。

代わりに result JSON に、

```text
cross_time_covariance_model:
    pairwise_mean_local_raw_pose_covariance
```

を必ず記録する。

この近似を用いた補正であることを明示する。

---

# 18. overlap correction の効果 diagnosis

whitened ridge right singular vectors `v_i` に沿って、

```math
\sigma^2_{i,\mathrm{naive}}
=
v_i^T
\Sigma_{\mathrm{naive}}
v_i
```

```math
\sigma^2_{i,\mathrm{overlap}}
=
v_i^T
\Sigma_{\mathrm{overlap}}
v_i
```

を計算する。

さらに、

```math
\text{inflation}_i
=
\frac{
\sigma^2_{i,\mathrm{overlap}}
}{
\sigma^2_{i,\mathrm{naive}}
}
```

を出す。

JSON / NPZ:

```text
uncertainty_variance_naive_in_ridge_basis
uncertainty_variance_overlap_in_ridge_basis
uncertainty_variance_inflation_in_ridge_basis
```

これにより overlap correction が特にどの weak direction へ効いたかを確認する。

---

# 19. `single_bag_savgol_reports.py` の修正

新しい report file は作らない。

既存の各 case の、

```text
report.pdf
```

へ diagnostic pages を追加する。

現在の重要4 section は維持する。

```text
1. trajectory
2. residual wrench
3. measured sensor comparison
4. nominal -> estimated parameters
```

これらの定義・表示順は壊さない。

---

## 19.1 追加 page: covariance weighting

表示する。

```text
Sigma_z eigenvalues vs time
whitening gain vs time
Mahalanobis contribution vs time
```

極端な時刻が分かるようにする。

---

## 19.2 追加 page: parameter ridge

表示する。

```text
whitened singular values
unwhitened diagnostic singular values
parameter displacement in whitened ridge basis
overlap uncertainty inflation in whitened ridge basis
```

exact common-scale direction は別途 annotation する。

---

## 19.3 追加 page: lag

表示する。

split lag の場合、

```text
strict candidate costs over rotor/gimbal lag candidates
selected pair
bounds
boundary hit
```

を表示する。

候補数が少なければ scatter / grid でよい。

新たな interpolation による smooth cost surface を作らない。

---

## 19.4 追加 page: acceleration/wrench closure

表示する。

```text
angular-acceleration residual
raw torque residual
principal inertia
closure error summary
```

raw torque が巨大でも angular residual が小さい case を読み取れるようにする。

---

# 20. `result.json` に追加する fields

既存 field は backward compatibility のため可能な限り維持する。

追加する。

```text
diagnostics:
    covariance:
        sigma_z_eigenvalue_summary
        sigma_z_rank_summary
        whitening_gain_summary
        mahalanobis_contribution_summary

    metric_cross_evaluation:
        nominal:
            full_covariance_objective_sum
            identity_objective_sum
            specific_acceleration_rmse_m_per_s2
            angular_acceleration_rmse_rad_per_s2
        estimated:
            full_covariance_objective_sum
            identity_objective_sum
            specific_acceleration_rmse_m_per_s2
            angular_acceleration_rmse_rad_per_s2

    lag:
        candidate_table
        rotor_at_lower_bound
        rotor_at_upper_bound
        gimbal_at_lower_bound
        gimbal_at_upper_bound

    closure:
        force_acceleration_closure_error_max_abs
        force_acceleration_closure_error_rms
        torque_acceleration_closure_error_max_abs
        torque_acceleration_closure_error_rms

    inertia:
        nominal_principal_moments_kg_m2
        estimated_principal_moments_kg_m2
        estimated_to_nominal_ratio

    overlap_correction:
        cross_time_covariance_model
        variance_inflation_in_ridge_basis
```

---

# 21. `arrays.npz` に追加する arrays

現在の arrays を削除しない。

追加する。

```text
sigma_z_eigenvalues
sigma_z_machine_rank
sigma_z_retained_condition_number
whitening_gain

mahalanobis_contribution_per_time
covariance_eigenmode_residual
covariance_eigenmode_mahalanobis_contribution

unwhitened_physical_jacobian_singular_values
unwhitened_physical_jacobian_right_singular_vectors
whitened_physical_jacobian_singular_values
whitened_physical_jacobian_right_singular_vectors

parameter_displacement_ridge_coordinates
parameter_displacement_ridge_energy_fraction

uncertainty_variance_naive_in_ridge_basis
uncertainty_variance_overlap_in_ridge_basis
uncertainty_variance_inflation_in_ridge_basis

force_acceleration_closure_error
torque_acceleration_closure_error

lag_candidate_rotor_seconds
lag_candidate_gimbal_seconds
lag_candidate_strict_cost
lag_candidate_selected
```

common lag / fixed lag 等で存在しない dimension は、意味の分からない dummy value を作らない。

case に対応した明示的 shape / empty array を使う。

---

# 22. failure bookkeeping の修正

現在、一度 failed stage が発生した後に後続 stage が `ftol` success すると、

```text
status = failed
message = cost tolerance satisfied
```

のような矛盾が起こりうる。

これを修正する。

`estimate_single_bag()` で、

```text
first_failure_stage
first_failure_status
first_failure_message
```

を一度だけ記録する。

最初の failure 後に後続 message で上書きしない。

`EstimationResult` が unsuccessful の場合、

```text
status
message
```

はこの first failure を返す。

`single_bag_savgol_estimator.py` の failure JSON もこの値を使う。

---

# 23. ablation cases の扱い

## 23.1 削除する fixed case

finite difference を完全削除するため、

```text
jacobian_finite_difference
```

を削除する。

`jacobian_analytic` も比較対象がなくなるので fixed-case list から外してよい。

ただし過去 output は残す。

---

## 23.2 残す fixed case

以下は原則残す。

```text
default_full_covariance

cov_identity
cov_diagonal
cov_block_s_alpha
cov_full_no_R_uncertainty_in_s
cov_full_no_position_rotation_cross
cov_global_full

kkt_on_scale_negative
kkt_on_scale_zero
kkt_on_scale_positive
kkt_off_scale_negative
kkt_off_scale_zero
kkt_off_scale_positive

lag_zero
lag_fixed
lag_common_estimated
lag_split_estimated
lag_split_strict_only

actuator_stateful
actuator_direct_command

so3_geometric_correction
so3_naive_rotation_vector_derivatives

solver_custom_kkt_lm
solver_standard_gauge_least_squares

external_wrench_raw_only
external_wrench_trajectory_fitted

naive_all
```

物理的に自然な default 処理を、前回の数値効果が小さいという理由だけで消さない。

---

# 24. focused validation で新たに確認する内容

full 29-case production ablation を毎回必須とはしない。

既存 `single_bag_savgol_ablation.py` の sweep mechanism を用いる。

新しい script は作らない。

---

## 24.1 SG-window vs covariance mode

既に研究中に使用してきた、

```text
0.5 s
1.0 s
1.5 s
2.0 s
```

について、

```text
full covariance
identity covariance
```

を最低限比較する。

目的は、

```text
extreme parameter movement
Sigma_z spectrum
whitening gain
whitened ridge
parameter displacement along weak ridge
```

が SG window に依存するかを見ることである。

---

## 24.2 lag-bound sweep

`lag_bounds` を `sweep_config` から明示指定して比較する。

目的は、rotor lag が current upper bound `0.2 s` に張り付いた case で、

```text
selected lag is a genuine local minimum
```

なのか、

```text
current admissible intervalの端を選んだだけ
```

なのかを区別することである。

結果を見た後に bound 値を追加しない。

---

## 24.3 overlap correction

各 default / focused case について、

```text
parameter_covariance_naive
parameter_covariance_overlap_corrected
```

を必ず両方出す。

point estimate は一度だけ求める。

correction 有無で optimizer を再実行しない。

---

## 24.4 physical-natural ablations

以下は残してよい。

```text
actuator_stateful vs actuator_direct_command
so3_geometric_correction vs naive rotation-vector derivative
smooth split lag vs strict-only split lag
```

結果が近くても、default から自動的に削除しない。

---

# 25. tests: finite difference を oracle にしない

`minimal/tests/` の tests を修正する。

有限差分との一致を test oracle として使わない。

---

## 25.1 parameter chart derivative

closed-form property を使う。

mass:

```math
\frac{\partial m}{\partial q_m}=m
```

force effectiveness:

```math
\frac{\partial f_i}{\partial q_{f_i}}=f_i
```

chart origin では、

```math
D\exp(0)[E]=E
```

を使って second-moment / inertia derivative を直接検査する。

---

## 25.2 exact gauge derivative

synthetic state で、

```math
J v_{\mathrm{scale}} = 0
```

を検査する。

---

## 25.3 KKT

既存 synthetic KKT test を維持する。

```math
v_{\mathrm{scale}}^T p = 0
```

を検査する。

---

## 25.4 SO(3) first/second directional derivative

新しい analytic second directional derivative について、finite difference ではなく以下を検査する。

### zero-angle closed-form

```math
\phi=0
```

で Taylor series から得られる closed-form value と一致すること。

### bilinearity

```math
D^2J[\eta_1+\eta_2,\zeta]
=
D^2J[\eta_1,\zeta]
+
D^2J[\eta_2,\zeta]
```

および second direction に対する bilinearity。

### symmetry

smooth map の Hessian として、

```math
D^2J[\eta,\zeta]
=
D^2J[\zeta,\eta]
```

を numerical roundoff 内で検査する。

これは finite difference ではない。

---

## 25.5 covariance propagation special cases

以下の解析的 special case を使う。

```text
rho0 = 0
rho1 = 0
```

等で `propagation_jacobian()` の expected block を直接検査する。

---

## 25.6 ridge dimension

lag-estimated synthetic problem でも scientific ridge が、

```text
14 columns
```

だけを持つことを検査する。

lag column が入らないことを hard assertion にする。

---

## 25.7 lag candidate reporting

synthetic command history で、

```text
all evaluated strict candidates are recorded
selected candidate is marked exactly once
boundary flag matches selected grid candidate
```

を検査する。

どの lag が良いかは performance assertion にしない。

---

## 25.8 covariance diagnosis consistency

各 `k` について、

```math
\sum_j c_{k,j}
=
r_k^T\Sigma_{z,k}^{\dagger}r_k
```

が machine precision 内で成立することを検査する。

---

## 25.9 ridge-basis variance

各 ridge direction `v_i` について、

```math
v_i^T \Sigma v_i
```

から保存された variance が作られていることを検査する。

variance が増えること自体は assertion にしない。

---

## 25.10 closure identities

synthetic data で force / torque acceleration closure error が numerical roundoff 程度になることを検査する。

real bag で residual wrench が小さいことは要求しない。

---

## 25.11 failure bookkeeping

deliberately failed stage を作り、

後続処理に success message が存在しても、

```text
first_failure_stage
first_failure_message
```

が保存されることを検査する。

---

# 26. performance-dependent test の禁止

以下を test pass/fail 条件にしない。

```text
real bag optimizer success
real bag objective threshold
mass が nominal に近い
inertia が nominal に近い
CoG が小さい
lag が特定値になる
full covariance が identity より良い
identity が full covariance より良い
overlap correction が必ず variance を増やす
stateful actuator が direct より良い
geometric SO(3) が naive より良い
smooth continuation が strict-only より良い
ridge singular value が特定値以上
```

これらは JSON / PDF で比較する。

---

# 27. `run_full_ablation.sh`

新しい shell script を作らない。

既存 `run_full_ablation.sh` を修正する。

finite-difference case が fixed list から消えることに追従する。

3 bag の並列実行、case failure isolation は維持する。

既存 environment variables,

```text
GRAPE_ABLATION_CASE_WORKERS
GRAPE_ABLATION_NUMERIC_THREADS
```

も維持する。

---

# 28. outputs

output root は現在の、

```text
minimal/outputs/<source-commit>/
```

を維持する。

今回の実装後の run は、新しい source commit hash の namespace に入る。

現在の、

```text
minimal/outputs/a6599d9b52f98b28f84d05da4b375aff11fd8990/
```

を変更しない。

---

# 29. report.pdf の重要4 section は維持する

各 completed case の `report.pdf` に現在ある、

```text
trajectory
residual wrench
measured sensor comparison
nominal -> estimated parameters
```

は削除しない。

trajectory は引き続き、

```text
observed
estimated model + trajectory-fitted external wrench reconstruction
```

を主要比較とする。

residual wrench は、

```text
raw SG inverse-dynamics residual
trajectory-fitted external wrench
```

を同時表示する。

line style による白黒識別も維持する。

---

# 30. agent の判断範囲

この revision では agent が新しい scientific mechanism を追加しない。

特に、以下を禁止する。

```text
new prior
nominal regularization
physical box bounds
new parameter scaling
residual-wrench penalty
trajectory-error penalty
pose-error penalty
IMU-error penalty
scientific singular-value cutoff
new covariance floor
new robust loss
new lag regularizer
new bag weighting
multi-bag joint optimization
```

existing `agent_candidates` mechanism 自体は残してよいが、この implementation revision の途中で agent が勝手に candidate を追加して default を変更しない。

---

# 31. 実装順序

## Phase 1: finite difference removal

1. `single_bag_savgol_core.py` から parameter finite-difference path を削除。
2. `single_bag_savgol_estimator.py` から Jacobian mode CLI/config を削除。
3. `single_bag_savgol_ablation.py` から finite-difference case を削除。
4. `naive_all` を analytic Jacobian にする。
5. strict lag ridge finite-difference columns を削除。

## Phase 2: analytic SO(3) covariance propagation

1. `savgol_trajectory.py` に analytic second directional derivative を追加。
2. `single_bag_savgol_covariance.py` の `rho0` finite difference loop を解析式へ置換。
3. finite-difference oracle を使わない tests を追加。

## Phase 3: 14-D scientific ridge

1. final selected lag で `final.whitened_jacobian` を flatten。
2. 14-D SVD。
3. exact scale gauge diagnostic。
4. lag information を ridge JSON から分離。

## Phase 4: diagnostics

1. covariance spectrum / whitening gain。
2. Mahalanobis contribution。
3. full/identity cross-evaluation。
4. unwhitened vs whitened Jacobian diagnosis。
5. parameter displacement in ridge basis。
6. naive vs overlap covariance in ridge basis。
7. lag candidate table / boundary flag。
8. acceleration-wrench closure。
9. principal inertia。

## Phase 5: reporting

1. result JSON fields。
2. arrays NPZ fields。
3. report.pdf diagnostic pages。
4. failure bookkeeping fix。

## Phase 6: tests

performance-independent tests を更新し、finite-difference oracle をなくす。

## Phase 7: focused real-bag validation

3つの fixed bag JSON を使う。

SG-window / covariance focused sweep と lag-bound sweep を existing ablation runner から実行する。

---

# 32. 完了条件

以下をすべて満たす。

- current commit `fb45718...` 相対の既存 scripts を修正している。
- 新しい estimator / ablation script を作っていない。
- production estimator に finite-difference Jacobian path が存在しない。
- covariance propagation に finite difference が存在しない。
- strict-ZOH lag ridge finite difference が存在しない。
- `jacobian_finite_difference` ablation が存在しない。
- `naive_all` も analytic parameter Jacobian を使う。
- scientific ridge は 14-D physical coordinate のみ。
- lag candidate costs が全て保存される。
- lag boundary hit が JSON に保存される。
- covariance spectrum / whitening gain / Mahalanobis contributions が保存される。
- full / identity metric が同じ final point で両方計算される。
- parameter displacement が whitened ridge basis へ射影される。
- naive / overlap covariance が ridge basis 上で比較できる。
- acceleration residual と raw residual wrench の closure diagnostic が出る。
- principal inertia と nominal ratio が出る。
- failure reason が後続 success message で上書きされない。
- existing 4-section `report.pdf` が維持される。
- diagnostic pages が同じ `report.pdf` に追加される。
- tests は finite difference を oracle に使わない。
- tests は real-bag performance を合否条件にしない。
- existing production outputs は変更されない。
- 新しい run は新しい source commit namespace へ保存される。

---

# 33. この revision 後に判断すること

この implementation revision の目的は、その場で algorithm を削ることではない。

実 rosbag の新しい outputs を見て、少なくとも以下を判断可能な状態にすることである。

1. full SG covariance による weighting が、どの observation direction / time に大きな weight を与えているか。
2. extreme parameter displacement が whitened Jacobian の weak ridge directions に沿っているか。
3. SG window を変えたとき、covariance spectrum と weak ridge の関係がどう変わるか。
4. rotor lag の 0.2 s 付近の解が current bound artifact かどうか。
5. overlap correction が、どの parameter ridge directions の uncertainty をどれだけ変えるか。
6. raw residual torque が巨大になる case で、小さい angular-acceleration residual と巨大 inertia の関係が closure identity 上で説明できるか。
7. stateful actuator / geometric SO(3) / smooth continuation 等の物理的に自然な処理について、数値効果が小さくても、その事実を保持した上で default に残すかどうか。

この revision 中に 7 の処理を自動削除しない。
