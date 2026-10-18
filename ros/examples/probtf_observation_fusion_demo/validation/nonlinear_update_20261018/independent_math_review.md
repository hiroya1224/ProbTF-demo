# 非線形観測更新の最終独立レビュー

重大な実装上の取り違えは見つからなかった。これはコード・数式・近似の説明のread-onlyレビューであり、新しい実走の精度や跳ね解消を保証するものではない。

確認した範囲: 原共同prior一回、native画像固定chart、観測固定S2角度残差、正しいCholesky白色化、全cross GN共分散、evidence定数と全mixture正規化、有限非収束継続、原sourceとreadonly、K/freshgoal/表示への同一law接続、旧affineへの暗黙fallbackなし。同期一回内の観測順canonical化は時系列イベント順と区別した。

保護214ファイル: 全バイト一致。全回帰ログ末尾: `1124 passed, 1 warning in 183.04s (0:03:03)`。

## 解釈上の限界

- The finite modal embedding is the Bingham-derived local Gaussian proxy pushed through t=tref+delta_t,q=qref*rvq(delta_r); it is not exact global Bingham nonlinear conditioning.
- Image fixed-chart normalized Gaussian is a declared scalar likelihood proxy, not the raw-pixel likelihood or a globally normalized Haar pose density. IMU fixed S2 Log Gaussian is likewise a scalar angular proxy, not an asserted globally normalized S2 density.
- Posterior covariance is Gauss-Newton, not the exact nonlinear Hessian inverse or exact posterior moment. One finite proxy per explicit prior/native branch does not enumerate all nonlinear posterior modes.
- At a finite nonstationary iterate the reported GN-Laplace value is not the exact integral of the tangent quadratic: no .5 g.T M^-1 g completed-square correction is included. Logs explicitly disclose finite_nonstationary_iterate_used/exact_nonlinear_evidence_claimed=False; do not call it exact evidence or a verified global MAP.
- evaluator_calls counts successfully completed evaluations only. Failed trial calls are represented separately in line_search.actual_trial_error. completed_iterations counts attempted outer iterations, including a final unsuccessful line search.
- No new runtime, throughput, reaching, estimation accuracy or jump-resolution claim is established by this read-only review. New 90-second runs must be assessed on their own source-frozen records.

各式とコード行、source hashは同名JSONに記録した。production・testsを編集せず、物理runは起動していない。
