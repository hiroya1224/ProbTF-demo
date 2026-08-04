# 最小構成の実機パラメータ推定

このディレクトリの `estimate_recorded_control.py` は deterministic と probabilistic の二つを切り替える共通エントリポイントです。

既定では `deterministic_estimator.py` のベースライン推定法を呼び出します。

ベースライン推定法は、GUI、Q、潜在状態、residual wrench、EM、MCMC を使わず、SciPy の `least_squares` だけで質量、慣性行列、CoG offset、相対 rotor effectiveness を推定します。

固定 Q と 1 iteration の Laplace-EM を実 bag で比較した結果、解釈、残作業は [現時点の比較結果](RESULTS_ja.md) に記録しています。

既定では同梱 rosbag の 19–24 秒を読み、記録された rotor thrust command と gimbal command を固定アクチュエータモデルへ入れます。

最初に観測 gyro の時間微分と IMU specific force を使う局所運動方程式で初期化し、最後に最初の観測状態から一度も状態をリセットしない 5 秒間の open-loop 軌道誤差を直接最小化します。

## 実行

### Deterministic baseline

```bash
source /home/leus/catkin_ws/devel/setup.bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py"
```

推定結果は `minimal/output/result.json` に、軌道と時系列の比較図は `minimal/output/trajectory.pdf` に生成されます。

PDF は rosbag の observed、推定前の nominal-parameter rollout、推定後の estimated-parameter rollout を同じ図へ重ねます。

Observed は青の実線、nominal は橙の破線、estimated は緑の点線で表し、位置、姿勢、速度、角速度、specific force を表示します。

試行時間を短くする場合は `--max-nfev` を小さくし、軌道 refinement を長く続ける場合は大きくします。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" --max-nfev 50
```

### GUI backend の根幹アルゴリズムとの比較

`--method probabilistic` は GUI、project、worker process、lag profile、MCMC を通さず、GUI 版と同じ full-trajectory factor graph、解析 Jacobian、sparse MAP、Laplace covariance を直接呼び出します。

最初は変更点を latent full-trajectory MAP だけに限定するため、command lag と Q を固定して実行してください。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method probabilistic \
  --q-policy fixed
```

この段階でも static parameter 18 次元に加えて、各 knot の pose、velocity、angular velocity、controller integral、actuator thrust、gimbal angle と bag-local IMU bias を同時に推定します。

時刻ごとの residual wrench は推定変数にせず、隣接 latent state と物理パラメータから得る 6 次元 body-wrench dynamics residual だけを使います。

次の段階では同じ sparse MAP と Laplace covariance を使い、Q の六つの対角成分を generalized Laplace-EM で更新します。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method probabilistic \
  --q-policy laplace_em \
  --q-em-iterations 2
```

固定 Q の結果は `minimal/output/probabilistic/fixed_q/` に生成されます。

Laplace-EM の結果は `minimal/output/probabilistic/laplace_em/` に生成されます。

各 directory の `method_comparison.pdf` は deterministic、probabilistic、nominal、observed の四者を同じ図で比較します。

現在の bridge では deterministic の推定結果を static parameter の初期値かつ prior mean とし、lag は既定の `0.01 s` に固定しています。

Lag profile と MCMC は、固定 Q と Laplace-EM Q のどちらが open-loop 軌道再現性を維持できるか確認した後に追加する TODO です。
