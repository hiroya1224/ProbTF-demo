# Grape parameter estimator 実装ノート

このファイルは講義資料の入口である。
現行 backend は sparse full-trajectory MAP、continuous delay profile、diagonal-Q Laplace-EM、prior-separated Laplace/ridge analysis、static-parameter MCMC、posterior PID cross-evaluation で構成する。

詳細は次の資料を順に参照する。

1. [batch_estimator_formulation_ja.md](batch_estimator_formulation_ja.md) は推定変数、factor graph、sparse Schur/LM、delay profile を説明する。
2. [analytic_jacobian_implementation_ja.md](analytic_jacobian_implementation_ja.md) は SO(3)、physical chart、controller/actuator/dynamics factor の解析 Jacobian を説明する。
3. [laplace_em_q_estimation_ja.md](laplace_em_q_estimation_ja.md) は covariance correction を含む対角 Q の Laplace-EM を説明する。
4. [ridge_and_mcmc_diagnostics_ja.md](ridge_and_mcmc_diagnostics_ja.md) は likelihood/posterior ridge、Laplace geometry、multiple-chain MCMC を説明する。
5. [real_flight_validation_ja.md](real_flight_validation_ja.md) は失敗 rosbag `18--24 s` の batch 結果と、短区間の clean estimate→MCMC→PID→tuning E2E evidence を記録する。
6. [pid_particle_evaluation_ja.md](pid_particle_evaluation_ja.md) は MCMC sample 由来 PID 候補の posterior cross-evaluation を説明する。
7. [synthetic_recovery_validation_ja.md](synthetic_recovery_validation_ja.md) は perfect-model、known-Q、lag、sensor、MCMC の truth recovery 結果を記録する。

操作方法、収録 rosbag、CLI、artifact schema、GUI virtual environment は [README.md](../README.md) を参照する。
