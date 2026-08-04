# Ridge、Laplace、MCMC の診断

## 1. MAP と識別性を分ける

proper Gaussian prior があれば posterior MAP は一つに決まり、posterior Hessian も正則になりうる。
しかし prior によって選ばれた一点を、likelihood が識別した物理値と解釈してはならない。
この実装は bag-local trajectory を Schur 消去した 18 次元 information を likelihood と posterior に分けて保存する。

```math
H_{post}^{red}=H_{like}^{red}+H_{prior}.
```

LM damping はこのどの項にも含めない。

## 2. reduced information

final sparse factorization では bag-local blockを消去し、共有 static block の Schur complement を得る。
posterior covariance は proper な `H_post^red` の inverse から計算する。
likelihood information は posterior reduced Hessian から明示的な static prior precision を差し引いて作る。
数値的な微小非対称は symmetrize するが、material な非対称や負の curvature は artifact を作らず error とする。

eigenvalue は小さい順に並べ、最大 eigenvalue に対する相対値で effective rank を決める。
rank cutoff より小さい方向は named parameter loading とともに ridge として保存する。
prior によって posterior rank が 18 でも、likelihood rank と condition number を別に読む必要がある。

## 3. exact common-scale direction

mass、inertia、force effectiveness、torque effectiveness の共通 scale には、観測と model の組合せによって exact または near ridge が生じうる。
physical chart はこの解析方向を 18 次元 unit vector として返す。
数値 likelihood Hessian の最小 eigenvector との absolute inner product を `ridge_alignment` として保存する。
alignment が低い場合は「別の near-ridge が優勢」「model/factor が scale invariance を破る」「数値/contract の問題」のいずれかであり、期待 ridge を確認済みとは報告しない。

## 4. Laplace approximation

`laplace.npz` は次を持つ。

- reduced likelihood Hessian と reduced posterior Hessian。
- static posterior covariance。
- likelihood eigensystem、effective rank、condition number。
- exact ridge direction と numerical minimum direction との alignment。
- delay profile grid/objective、local curvature、delay uncertainty source。

Laplace marginal は MAP 近傍の局所 Gaussian 近似であり、曲がった ridge、多峰性、delay boundary を一つの covariance へ押し込む限界がある。
Laplace draw を MCMC draw と表示しない。

## 5. MCMC target

MCMC の chain state は 18 次元 static chart と continuous delay の 19 次元だけである。
全 knot の latent trajectory を chain state に含めない。
各 proposal 点では static coordinate と delay を固定し、bag-local trajectory の conditional sparse MAP を共通の selected-mode MAP trajectory から解いて Laplace-marginal target を評価する。
現在の chain state で得た trajectory を次 proposal の warm start に引き継がず、同じ point の exact target が chain history や resume の有無に依存しないようにする。

概念的な log target は次である。

```math
\log\widetilde p(c,\tau\mid Y,\widehat Q)
=-\Phi_{cond}(c,\tau)
-\frac12\log\det H_{local}(c,\tau)
+\log p(\tau)
+\mathrm{const}.
```

fixed static coordinate は large penalty で近似せず、conditional solver が shared 18 次元 increment を厳密に zero にする。
inner solve failure は低い density へ黙って置換せず、failure flag と reason を MCMC 診断へ残す。

## 6. delayed acceptance と proposal mixture

第一段は MAP/Laplace 近傍の安価な surrogate で候補をふるい、通過した候補だけ exact conditional target を評価する delayed acceptance を使う。
proposal mixture は local、exact-ridge、near-ridge、identified、delay direction を分けて持つ。
各 kernel の attempted、stage-one accepted、stage-two attempted、accepted、inner failure、cache hit を記録する。
delay proposal は configured bounds 内へ reflection し、境界へ probability mass を積まない。

## 7. chain initialization

`chain-000` は MAP から開始する。
ほかの chain は static Laplace covariance による dispersion と exact-ridge direction への明示的 dispersion を交互に使い、delay も local uncertainty で分散させる。
すべての chain を同一点から始めて見かけの一致を作らない。

## 8. equal-weight retained draw

warmup 後に thinning を適用して retained draw を保存する。
retained draw は equal-weight sample であり、importance weight や架空の追加 weight を付けない。
同じ chain state が rejection によって繰り返されても正しい draw だが、artifact 上の `sample_id` は各 retained row で一意にする。
`sample_id`、`chain_id`、`draw_index`、static/physical parameter、delay、target breakdown、accepted kernel、source mode を alignment を保って保存する。

## 9. convergence diagnostics

最低限、各 static coordinate、delay、exact-ridge coordinate に対して split-R-hat と effective sample size を計算する。
chain ごとの acceptance、kernel acceptance、inner solve failure、trace も保存する。
configured `rhat_threshold` と `minimum_effective_sample_size` の全条件を満たした場合だけ MCMC diagnostics を converged とする。
draw 数が多いこと、acceptance が高いこと、MAP 周辺に cloud が見えることだけでは convergence としない。

## 10. checkpoint の境界

`posterior/checkpoint.py` は一 chain の current target、retained draw、kernel counter、completed transition、NumPy RandomState を pickle-free NPZ に保存する。
`run_mcmc_chains` は完了 chain と proposal-boundary checkpoint の両方を受け取って再開できる。
同じ chain を completed と in-progress の両方で渡すことは拒否する。
`grape_estimate_flights.py` は同一 batch request と run directory の `resume=true` をこの checkpoint へ接続する。
`grape_sample_parameter_posterior.py` は complete な estimate-only run と元 request を全 fingerprint で照合し、MAP/EM/Laplace を再実行せず同じ directory へ MCMC を原子的に追加する。
cancel 時は complete estimate-only artifact を変更せず proposal checkpoint を保持し、同一 sampling request fingerprint の `resume=true` だけを受け付ける。
linear factorization の途中は保存せず、保存済み MAP state から undamped factorization を再生成する。

## 11. 実 bag run の現在地

### 11.1 `18.0--24.0 s` estimate-only run

2026 年 8 月 4 日の実 bag validation は `estimate_only` で実行したため、MCMC sample は生成していない。
Laplace artifact は作成され、likelihood eigenvalue は広い dynamic range を持ち、condition number は約 `1.32e8` だった。
この一 run は暫定 covariance、暫定 prior、EM 一回、delay boundary `0.0 s` という条件なので、ridge や parameter posterior を科学的に確定する材料には不足している。

### 11.2 clean `18.0--18.3 s` posterior sampling smoke

clean revision `5b08e5c290925d7585024f3c5350a7f88a7f1fe9` の `run b` では、5-knot estimate-only artifact に後段 MCMC を追加した。
estimate-only は wall `9.04 s`、posterior sampling は wall `11.71 s`、progress elapsed `11.316 s` で complete になった。
sampling は 2 chains × 4 retained draws で、8 retained draws すべてについて fresh conditional sparse MAP trajectory を保存した。
保存した conditional objective と対応する MCMC target breakdown は最大絶対誤差 `8.88e-15` で一致した。

この clean run の前に、現在の chain trajectory を次 proposal の warm start にすると、nonlinear stopping point を介して target が history-dependent になる不具合を実 E2E で検出した。
revision `5b08e5c` は全 exact evaluation を共通の selected-mode MAP warm start から開始するよう修正し、同じ point の target と checkpoint resume を history independent にした。
上記 `run b` は修正後に最初から作り直した artifact である。

各 chain 4 draws では configured R-hat/ESS threshold を満たさず、artifact も `MCMC completed without satisfying convergence thresholds` を warning として保存した。
8 selected conditional trajectories は sample-local state の診断と可視化を監査できることを示すが、MCMC 収束や posterior の科学的妥当性は示さない。
科学的な実 bag MCMC validation は covariance/actuator contract の校正、複数 EM iteration、delay profile の安定化後に十分な warmup と retained draws で行う。

## 12. GUI での確認順序

1. MAP と prior/likelihood objective の内訳を確認する。
2. likelihood と posterior の eigenvalue、effective rank、ridge loading を分けて確認する。
3. exact-ridge alignment と delay profile/boundary を確認する。
4. chain trace、R-hat、ESS、acceptance、inner failure を確認する。
5. converged retained sample だけを `Next experiment` の plant population へ渡す。

posterior marginal が狭くても likelihood ridge が prior で埋められている場合がある。
PID candidate を作る前に、data が識別した方向と prior が選んだ方向を明示的に分ける。

## 13. code 対応

| 責務 | module |
|---|---|
| reduced eigensystem/ridge | `batch/ridge.py` |
| prior-separated Laplace geometry | `batch/evidence.py` |
| conditional Laplace target | `posterior/laplace_target.py` |
| delayed acceptance | `posterior/delayed_acceptance.py` |
| proposal/chain runner | `posterior/mcmc.py`, `posterior/run.py` |
| R-hat/ESS | `posterior/diagnostics.py` |
| pickle-free checkpoint | `posterior/checkpoint.py` |
| estimate-only への後段 sampling | `posterior_sampling_request.py`, `posterior_sampling_cli.py` |
