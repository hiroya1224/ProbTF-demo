# 対角 Q の Laplace-EM 推定

## 1. Q が表すもの

Q は観測 noise ではなく、隣接 latent state と rigid-body/actuator model の間に残る 6 次元 dynamics residual のモデル化誤差を表す。
全 selected bag と全 interval で一つの対角 Q を共有する。

```math
Q=\operatorname{diag}(q_x,q_y,q_z,q_{roll},q_{pitch},q_{yaw}).
```

request は residual quantity を `body_wrench` または `specific_acceleration` として明示し、各成分の unit と interval model も同時に固定する。
Q の 6 成分は等しいと仮定せず、force/acceleration 3 軸と torque/angular-acceleration 3 軸を個別に保持する。
時刻ごとの Q、bag ごとの Q、off-diagonal covariance は現在の model には含めない。

## 2. interval model

`continuous_spectral_density` の場合、interval-average residual `xi_k` の covariance を次で定義する。

```math
\operatorname{Cov}(\xi_k)=Q/\Delta t_k.
```

このとき whitened dynamics residual は `diag(sqrt(dt_k/q)) xi_k` になる。
`fixed_interval_covariance` の場合は `Cov(xi_k)=Q` であり、sampling interval による rescale を行わない。
二つの定義は数値値と unit が異なるため、artifact は `quantity/interval_model` を一つの Q definition として保存する。

## 3. E-step

固定した Q と delay に対して全軌道 MAP を sparse LM で求め、LM damping を除いた final Gauss--Newton Hessian を factorize する。
各 dynamics residual について必要なのは、その residual が依存する state/static block の covariance だけである。
full dense covariance は構築せず、bag-local sparse factorization と 18 次元 Schur complement から selected covariance action を計算する。

residual の局所線形化を `xi ~= xi_hat + G delta`、Laplace covariance を `Sigma` とすると期待二乗は次になる。

```math
E[\xi_{k,j}^2\mid Y]
\simeq
\widehat\xi_{k,j}^2
+(G_k\Sigma G_k^T)_{jj}.
```

第一項は MAP residual second moment、第二項は covariance correction である。
第二項を省いて MAP residual の二乗だけで更新する方法は hard EM であり、この実装の Q 更新ではない。

## 4. M-step

continuous spectral density の interval weight を `w_k=dt_k`、fixed interval covariance の weight を `w_k=1` と書くと、各成分の closed-form target は概念的に次である。

```math
q_j^*=\frac{1}{K}\sum_k w_k E[\xi_{k,j}^2\mid Y].
```

実装は request の成分別 floor を適用し、floor activation を iteration ごとに保存する。
target は six-vector のため、軸ごとの異なる model discrepancy scale を保持する。

## 5. damping と受理判定

nonlinear model では closed-form target をそのまま採用すると Laplace marginal objective が悪化する場合がある。
input Q と target Q の間を log 空間で補間する。

```math
\log Q(\alpha)=(1-\alpha)\log Q_{in}+\alpha\log Q^*,
\qquad \alpha\in(0,1].
```

`alpha=1` から二分し、candidate Q で固定-delay sparse E-step を再実行する。
candidate の solve が成功し、approximate marginal objective が tolerance 内で非悪化のときだけ受理する。
全 alpha が失敗した場合は input Q を保持し、rejection reason と全 trial を診断へ残す。

## 6. marginal objective

Q を比較するとき、whitened least-squares graph objective だけでは Gaussian normalization が抜ける。
undamped Laplace volume termも Q に依存するため、受理判定は次の和を使う。

```math
\widetilde\Phi_{marg}
=\Phi_{MAP}
+\frac12\sum_k\log\det\Sigma_{\xi,k}(Q)
+\frac12\log\det H_{post}.
```

ここで `H_post` は LM damping を含まない final posterior Gauss--Newton Hessian である。
LM damping を posterior precision や evidence に混ぜない。

## 7. delay との交互更新

最初の E-step は full bounded delay profile を評価する。
delay profile の目的関数は `Phi*(tau;Q)=min_{z,c} J` という MAP objective であり、Laplace volume を含む approximate marginal objective ではない。
近似 marginal objective は Q candidate acceptance に使う一方、同じ profile point の別診断量として保存する。
Q candidate の受理評価中は delay を固定し、Q を受理した後で local delay profile を再評価する。
Q と delay の両方を一度に smooth variable として更新しないため、ZOH breakpoint と Q scale の相互作用を audit できる。
各 iteration は input/output lag、lag change、profile failure、MAP/marginal objective change を保存する。

## 8. 停止条件

minimum iteration 後に log-Q change、delay change、MAP objective change、marginal objective change の全 tolerance を満たした場合だけ `convergence_tolerances` とする。
ほかに `maximum_iterations`、`repeated_q_rejection`、`repeated_lag_profile_failure` を明示的な termination reason として持つ。
Q が finite で artifact が書けたことと、Laplace-EM が収束したことは別である。

## 9. artifact の読み方

`q_em.npz` は各 iteration について次を保存する。

- `input_q`、`target_q`、`accepted_q` と `alpha`。
- `expected_residual_second_moment`、`map_residual_second_moment`、`covariance_correction`。
- `map_objective` と `approximate_marginal_objective`。
- selected `lag`、accepted flag、reason、floor activation。

`map_static.npz` の `q_diagonal` は選択 mode の最終 accepted Q である。
GUI の Q panel は最終値だけでなく iteration history と `sqrt(Q/dt)` reference band を表示する。

`laplace.npz` の 18 次元 `covariance` は `static_covariance_conditioning=fixed_delay_conditional` と明示され、delay を積分した static marginal ではない。
有効な最終 Q profile geometry が得られた場合は、非等間隔 lower/center/upper 三点、MAP objective、18 次元 static coordinate、profile gradient、curvature、`dc*/dtau`、19 次元 joint information/covariance、parameter--delay cross covariance を保存する。
boundary optimum、両側 support 不足、非正 curvature、または不安定な stationarity では `delay_local_geometry_valid=false` と reason を保存し、cross と joint 配列は空にする。
invalid geometry の MCMC surrogate は `proposal_only_block_diagonal_fallback_v1` として明示され、空の joint covariance をゼロ埋め covariance と解釈しない。
invalid 時の `delay_local_uncertainty` は表示と診断のための一様 prior 標準偏差を保持する一方、MCMC surrogate と chain 初期化の delay block は request の proposal scale `delay_scale_seconds` を使い、二つを同じ推定分散として扱わない。
MCMC chain の Laplace dispersion も valid 時は同じ 19 次元 joint covariance を使うため、parameter--delay cross covariance を初期値生成から捨てない。

## 10. 18--24 秒 run での確認

失敗 sample の `18.0--24.0 s` を用いた短縮検証では、初期 Q `[25, 25, 25, 1, 1, 1]` から target `[14.4779, 14.5221, 13.4510, 0.0527430, 0.0568025, 0.184592]` へ `alpha=1` で更新した。
MAP residual second moment は `[0.02925, 0.00250, 0.07976, 0.000201, 0.007242, 0.000111]` であった。
covariance correction は `[14.4486, 14.5196, 13.3712, 0.052542, 0.049561, 0.184480]` であり、target の大部分を占めた。
この差は実装が hard EM ではなく Laplace covariance correction を使用したことを示す。

ただしこの run は `maximum_iterations=1` なので termination は `maximum_iterations`、`converged=false` である。
さらに observation/fixed-factor covariance と 18 次元 prior は暫定設定であるため、得られた Q を校正済み実機 model discrepancy と解釈しない。
まず sensor covariance、actuator dynamics、interval、prior を校正し、複数 EM iteration と sensitivity run で安定性を確認する必要がある。

## 11. code 対応

| 責務 | module |
|---|---|
| Q definition と target/update | `batch/laplace_em.py` |
| selected residual moment | `batch/dynamics_moments.py` |
| sparse selected covariance | `batch/covariance.py` |
| evidence と normalization | `batch/evidence.py` |
| outer EM/lag alternation | `batch/em_loop.py` |
| strict export | `batch_artifact_export.py` |

Q は model mismatch を吸収できる自由度なので、Q が大きくなって MAP が収束したことを物理 parameter の妥当性と取り違えない。
observation covariance、controller reconstruction error、actuator time constant、frame errorを Q へ押し込めないよう、それぞれの contract を先に監査する。
