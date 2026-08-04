# 疎な全軌道 batch estimator の定式化

## 1. 解く問題

この estimator は、選択した各 rosbag 区間を一つの smoothing window とし、全時刻の状態と全 bag で共有する静的物理パラメータを同時に推定する。
観測時刻ごとの filter reset、独立な短区間 fit の連結、時刻ごとの residual-wrench 未知状態は行わない。

bag を `b`、knot を `k` とし、共有する 18 次元 chart coordinate を `c` と書く。

```math
c = (\log m, c_J, r_{\mathrm{CoG}}, \log e_F, \log e_\tau) \in \mathbb R^{18}.
```

`c_J` は nominal inertia に対する full SPD inertia の相対 chart 6 次元であり、mass、inertia、effectiveness の物理制約は chart の decode で満たす。
continuous constant command delay `tau` は全 selected bag で共有するが、ZOH command に対する目的関数が区分的に滑らかなため inner Gauss--Newton coordinate には含めない。

## 2. knot state と総次元

各 knot の state は次の 26 local coordinate を持つ。

| state | local dimension | 表現 |
|---|---:|---|
| position | 3 | world Cartesian |
| orientation | 3 | current `SO(3)` point の right tangent |
| linear velocity | 3 | world frame |
| angular velocity | 3 | body frame |
| controller integral | 6 | position 3 と attitude 3 |
| actual rotor thrust | 4 | N |
| actual gimbal angle | 4 | rad |
| 合計 | 26 | |

absolute orientation は 3 成分の global rotation vector や自由な quaternion ではなく、proper rotation matrix として保持し、iteration ごとに `R <- R Exp(delta)` で更新する。
各 bag は gyro bias 3 次元を持ち、calibrated accelerometer factor が有効な bag だけ accelerometer bias 3 次元も持つ。

```math
n_{\mathrm{inner}}
=18+\sum_b\left(26N_b+3+3I_{\mathrm{acc},b}\right).
```

科学的な共有未知量は `c` と `tau` の 19 個だが、inner sparse linear system の shared block は `c` の 18 次元だけである。
対角 Q の 6 成分は Laplace-EM hyperparameter であり、inner state vector へ追加しない。

## 3. 全軌道 factor graph

objective は whitened residual の二乗和と正規化項から構成する。

```math
\Phi(X,c;\tau,Q)
=\frac12\sum_i\|r_i(X,c;\tau,Q)\|^2.
```

実装する factor family は次である。

- static physical parameter と bag-local initial state/bias の Gaussian prior。
- position と orientation の pose observation。
- world linear velocity observation。
- sensor frame の gyro observation。
- frame、lever arm、bias contract を満たす場合だけの specific-force observation。
- recorded issued rotor command、issued gimbal command、actual gimbal position、controller integral の observation。
- controller integral transition。
- thrust と gimbal の first-order actuator transition と saturation active set。
- position と orientation の kinematic consistency。
- body force 3 軸と body torque 3 軸の dynamics consistency。

観測は非同期 timestamp のまま各 factor へ結び、trajectory state を観測 sample ごとに reset しない。
補間 policy は Euclidean quantity が linear、orientation が SO(3) geodesic、issued command が record issue time に対する causal ZOH である。

## 4. controller と actuator

controller snapshot は bag startup の recorded dynamic-reconfigure update を正本とし、selected interval 内で一定であることを request policy が要求する。
controller integral state は latent trajectory に含め、recorded debug が使用可能なら observation factor でも拘束する。
issued command は controller と plant の間の入力であり、静的 delay `tau` を適用してから actuator transition へ渡す。

actuator model は request の必須 contract であり、thrust/gimbal time constant、thrust bounds、gimbal angle/rate bounds、provenance を manifest へ保存する。
hidden な instantaneous-actuator default は使用しない。

## 5. dynamics residual

interval midpoint の state から required wrench と modeled wrench を body frame で計算する。

```math
w_{\mathrm{req},k}
=
\begin{bmatrix}
mR_m^T((v_{k+1}-v_k)/\Delta t-g) \\
J(\omega_{k+1}-\omega_k)/\Delta t
+\omega_m\times J\omega_m
\end{bmatrix}.
```

```math
w_{\mathrm{model},k}
=w_{\mathrm{actuator},k}
-
\begin{bmatrix}
D_vR_m^Tv_m \\
D_\omega\omega_m
\end{bmatrix}.
```

```math
\xi_k=w_{\mathrm{req},k}-w_{\mathrm{model},k}\in\mathbb R^6.
```

midpoint attitude は endpoint を結ぶ principal SO(3) geodesic の中点、velocity、omega、thrust、gimbal は endpoint の算術平均である。
`xi_k` は state と static parameter から決まる factor residual であり、独立な latent force/torque path ではない。

Q の統計座標は body frame の `body_wrench` 6 成分だけであり、force は N、torque は N*m で表す。

## 6. 対角 Q と可変 interval

Q は全 selected interval と全 bag で共有する正の 6 対角成分である。

```math
Q=\operatorname{diag}(q_x,q_y,q_z,q_{roll},q_{pitch},q_{yaw}).
```

interval-average residual covariance は `Q/dt_k` と定義するため、whitening は `diag(sqrt(dt_k/q))` になる。
`body_wrench/continuous_spectral_density`、component name、unit は一組の科学的 contract であり、loader や GUI が推測しない。

## 7. sparse MAP solve

各 factor は residual と関係する variable key ごとの小さな解析 Jacobian block を返す。
linearization はこれらを canonical column order の CSC Jacobian/Hessian へ assembly し、異なる bag の local variable 間に cross block を作らない。

scaled coordinate `z` と物理 coordinate `delta = D z` に対し、LM step は次を解く。

```math
(D H D+\lambda I)z=-Dg.
```

各 bag-local blockを sparse LU で消去し、全 bag から得た寄与を 18 次元 shared Schur complement へ集約する。
18 次元 system を解いた後で bag-local step を back-substitute するため、全 state Hessian を dense 化しない。
LM は actual/predicted reduction ratio、scaled gradient、scaled step、relative objective、factorization failure、model evaluation failure、active-set oscillationを記録する。

## 8. delay profile

ZOH delay の breakpoint では ordinary smooth derivative を仮定できないため、delay を固定した inner sparse MAP を外側から評価する。
delay の選択に使う profile は、近似周辺尤度ではなく次の MAP profile である。

```math
\Phi^\star(\tau;Q)=\min_{X,c}\Phi(X,c;\tau,Q).
```

Laplace log-determinant を含む approximate marginal objective は Q candidate の受理判定にだけ使い、delay profile の最小点を変えない。
最初に bounded coarse grid を評価し、収束した最良点の隣接点から bracket を作り、golden-section refinement を行う。
各候補は最も近い収束済み state を warm start に使うが、candidate objective 自体はそれぞれ完全な sparse solve である。
各収束点について delay、MAP objective、approximate marginal objective、18 次元 static MAP coordinate を別々に保存する。
これが実 bag run で単純な一回の forward simulation より時間がかかる主因である。

### 8.1 static parameter--delay local geometry

最終 Q の profile から、最終 MAP 点に最も近い収束済み lower/center/upper の三点だけを選ぶ。
非等間隔三点公式を用いて、static MAP path の感度 `s=dc*/dtau`、MAP profile gradient、正の curvature `kappa=Phi*''` を計算する。
固定 delay での 18 次元 reduced posterior Hessian を `A` とすると、有効な 19 次元 joint information は次である。

```math
I_{c,\tau}
=
\begin{bmatrix}
A & -As\\
-s^TA & \kappa+s^TAs
\end{bmatrix}.
```

その逆行列は次であり、`A^-1` は固定 delay 条件付き covariance であって joint static marginal ではない。

```math
C_{c,\tau}
=
\begin{bmatrix}
A^{-1}+ss^T/\kappa & s/\kappa\\
s^T/\kappa & 1/\kappa
\end{bmatrix}.
```

最良点が delay bound 上にある場合、両側 support がない場合、curvature が非正または非有限の場合、profile stationarity が refinement tolerance と support scale に対して不安定な場合は local geometry を `valid=false` とする。
invalid geometry は reason と method を明示し、parameter--delay cross covariance と joint information/covariance を空配列で保存するため、ゼロ相関の有効な covariance と取り違えない。
この場合の MCMC quadratic surrogate だけは、固定 delay の `A` と request の `delay_scale_seconds` から作る監査済み proposal-only block-diagonal fallback を使う。
MCMC の Laplace-dispersion chain 初期値も同じ契約に従い、有効時は 19 次元 joint covariance から static coordinate と delay を相関させて同時に生成し、invalid 時だけ block-diagonal fallback covariance から生成する。

## 9. multi-bag の意味

複数 bag を同時に選ぶ場合、18 次元 static chart、delay、Q は共有し、trajectory と bias は bag ごとに独立に持つ。
bag ごとに別々に fit して最後に平均する方式ではない。
同じ configuration group を共有できない bag を混ぜると model assumption が崩れるため、configuration fingerprint と operator confirmation を入力 contract に含める。

## 10. code 対応

| 責務 | module |
|---|---|
| variable key と column layout | `batch/variables.py`, `batch/layout.py` |
| nonlinear state と SO(3) retraction | `batch/state.py` |
| factor contract と sparse assembly | `batch/factor.py`, `batch/linearize.py`, `batch/problem.py` |
| observation/model/dynamics factor | `batch/factors/` |
| Schur solve と LM | `batch/sparse_solver.py`, `batch/lm.py` |
| continuous delay profile | `batch/lag_profile.py` |
| joint static--delay local geometry | `batch/evidence.py` |
| rosbag preparation と graph build | `batch/preparation.py`, `batch/graph_builder.py`, `real_estimation.py` |
| strict request と artifact | `batch_request.py`, `batch_artifact.py`, `batch_artifact_export.py` |

## 11. 結果の読み方

MAP trajectory が observed pose に重なることは必要な診断の一つだが、それだけでは物理 parameter が識別されたことを意味しない。
sensor residual と dynamics residual の normalized scale、Q 更新、prior と likelihood を分離した reduced Hessian、delay profile、MCMC の chain mixing を合わせて評価する。
特に prior によって proper な MAP と posterior covariance が得られても、likelihood 側の ridge はデータから識別できない方向として残して報告する。
