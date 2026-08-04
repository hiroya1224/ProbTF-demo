# 最小構成の実機パラメータ推定

このディレクトリの `estimate_recorded_control.py` は、GUI、Q、潜在状態、residual wrench、EM、MCMC を使わず、SciPy の `least_squares` だけで質量、慣性行列、CoG offset、相対 rotor effectiveness を推定します。

既定では同梱 rosbag の 19–24 秒を読み、記録された rotor thrust command と gimbal command を固定アクチュエータモデルへ入れます。

最初に観測 gyro の時間微分と IMU specific force を使う局所運動方程式で初期化し、最後に最初の観測状態から一度も状態をリセットしない 5 秒間の open-loop 軌道誤差を直接最小化します。

## 実行

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
