# Sparse batch estimator の残作業

## 現在の到達点

Sparse batch MAP、解析 Jacobian、diagonal-Q Laplace-EM、static parameter と delay の MCMC、PID 粒子評価、strict artifact、GUI、production synthetic generator の実装経路は通常テストで動作する形まで作成した。
Production synthetic では perfect-model dynamics、exact common-scale ridge、sub-sample delay、multi-bag sparse solve の回帰試験を用意した。
一方、実 rosbag で物理パラメータ推定が成功したとはまだ判断できない。

既存の失敗飛行 `18.0--24.0 s` run は solver と artifact の配管確認には成功したが、position と velocity を含む複数の observation RMS が observation-anchored initialization より悪化した。
同 run の delay は探索範囲の下限 `0.0 s` に張り付き、Laplace-EM は一 iteration で停止し、expected Q moment の大部分を covariance correction が占めた。
この結果の詳細は [実飛行 validation](real_flight_validation_ja.md) に記録している。
したがって、現在の MAP parameter、Q、delay、MCMC sample、PID proposal を実機の校正値として使用してはならない。

## 優先度 A: 実推定を再実行する前に必要な作業

### Observation covariance の校正

Position、orientation tangent、direct velocity、gyro、gimbal observation、controller debug の covariance を quantity ごとに分離して校正する。
各 covariance は message covariance、preflight static interval、sensor specification のいずれを根拠にしたかを明記する。
Pose と twist が同一 estimator 由来で相関を持つ場合は、独立 factor とみなす近似を manifest と報告に明記する。
現在準備されていた最終 verification request は全 observation、fixed numerical tolerance、initial-state prior を一律の対角1としており、科学的な根拠を監査できていないため、そのまま再利用しない。

### Actuator と controller contract の校正

暫定値である gimbal time constant `0.02 s` を step response または別の system-identification data で確認する。
Rotor command の単位、saturation、record issue time、thrust conversion、gimbal command と joint observation の時刻対応を再監査する。
Controller integral、P/I/D output、reference、feedforward の再構成誤差を dynamics Q に吸収させないことを residual ごとに確認する。

### Initial Q の作成

現在の固定 initial Q `[25, 25, 25, 1, 1, 1]` は科学的根拠の監査が完了していない。
Observation から作った初期 latent trajectory と nominal static parameter から body-frame wrench-balance residual を計算し、sensor-invalid interval を除外した可変 `dt` 対応の robust per-axis scale を initial Q とする。
Raw pose の二階差分を initial Q の正本にしない。
Initial Q の作成区間と provenance を request または artifact に保存する。

## 優先度 B: Clean real-flight inference

全校正値を固定した clean Git revision で、第一 target bag の `18.0--24.0 s` を最初から再推定する。
Laplace-EM は最低 iteration 数に達しただけで収束扱いにせず、log-Q、delay、MAP objective、approximate marginal objective の停止条件を確認する。
MAP の position、orientation、velocity、gyro、gimbal、controller residual と raw body-wrench residual を initialization と比較する。
Q で正規化した residual が小さいことだけを成功根拠にせず、raw residual と Q magnitude を併記する。
Delay が再び境界へ張り付く場合は symmetric local Gaussian を表示せず、command/response contract、探索範囲、励起不足を再調査する。
単一 bag で識別できない方向は、同じ configuration group の複数 bag を joint problem として追加して再評価する。

## 優先度 C: Posterior と PID の科学的受入

十分な warmup と retained draw を持つ複数 chain を実行し、全監視座標で split-R-hat と ESS の閾値を満たすまで MCMC を converged と呼ばない。
Exact-ridge coordinate、delay marginal、mode 間移動、delayed-acceptance の各 kernel acceptance、inner solve failure を確認する。
Role 付き selected conditional trajectory と retained sample ID の対応を strict loader で再監査する。
PID candidate は全 retained posterior plant sample と全 selected bag で cross-evaluateする。
Parameter-only と sampled-Q counterfactual を区別し、current PID baseline より Pareto 改善しない場合は推薦なしとする。
Repository 内の成功 bag は既に GUI demo と tuning に使用したため hold-out と呼ばず、最終 validation には未閲覧かつ configuration compatibility を確認した別 bag を使用する。

## 優先度 D: 性能と最終画面受入

旧 `18.0--24.0 s` run の wall time は約 `252 s` であり、主な時間は単純な一回の forward forecast ではなく、delay profile と Q candidate ごとの反復 sparse MAP に費やされた。
高速化は profile evaluation 数、warm-start、factor kernel、symbolic sparse structure reuse を測定してから行う。
独立な PID forecast は process 並列経路を維持し、推定の非線形 solve と混同しない。
最新 revision で `rosrun grape_param_estim run_gui.py` を実行し、role 付き sample、joint/conditional Laplace、delay marginal、direct observation、body-wrench Q 表示を実画面で再確認する。
スクリーンショットは対象 GUI の window ID を明示して取得し、別アプリケーションの背後に隠れた画像を受入証拠にしない。

## 完了判定

Backend の通常テストが通ることは実装の動作条件であり、物理推定成功の十分条件ではない。
実推定の完了は observation と dynamics の同時再現、Laplace-EM の有限収束、delay 境界の説明、prior と likelihood の ridge 分離、MCMC convergence、PID cross-evaluation を同じ校正済み data split で確認した時点とする。
非収束、識別不能、configuration incompatibility、current PID を改善しない結果も失敗として隠さず記録する。
