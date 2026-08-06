# 最小構成の実機パラメータ推定

このディレクトリの `estimate_recorded_control.py` は deterministic-multiple-shooting、smooth-lag multiple-shooting、deterministic、deterministic-Sobol、deterministic-tempered、deterministic-continuation、deterministic-Q、probabilistic を切り替える共通エントリポイントです。

既定では `deterministic_multiple_shooting_estimator.py` を呼び出します。

各 method は同梱 rosbag の 19–24 秒を既定区間とし、記録された rotor thrust command と gimbal command を既知入力として使います。GUI には依存しません。

旧 deterministic baseline、固定 Q、Laplace-EM の比較結果は [現時点の比較結果](RESULTS_ja.md) に記録しています。

## 実行

### SE(3)-only deterministic multiple shooting（既定）

`deterministic_multiple_shooting_estimator.py` は、記録された rotor / gimbal command を既知入力として用い、全区間共通の質量、慣性、CoG offset、相対 rotor force effectiveness を推定します。慣性は二次モーメントを `Σ = L L^T`、`J = tr(Σ)I - Σ` と表す6次元 Cholesky 座標を用いるため、正定値性と主慣性モーメントの三角不等式を探索中も構造的に満たします。この既定methodではcommand lagを `0.16 s` に固定します。

Thrust と gimbal の一次遅れ時定数はモデルへ入れません。thrust command は即時反映し、gimbal command は一次遅れなしで、記録済みの角速度・角度制限だけを適用します。13個の物理座標にはbox上下限を置かず、mass、二次モーメント、force effectiveness の正値性と慣性の三角不等式だけを座標変換で保証します。

無制限座標に存在する識別不能なridgeを固定するため、nominal座標を平均とする独立Gaussian soft priorを既定で用います。標準偏差は、massの対数座標 `1.5`、二次モーメントCholesky対角の対数座標 `1.5`、同非対角座標 `2.0`、CoG `0.25 m`、force-effectiveness contrast `1.5` と広めです。hard clippingではなく二次ペナルティであり、`--prior-weight 0` で無効化できます。

観測 Loss は各時刻の

```text
Log_SE(3)(T_observed^-1 T_simulated)
```

だけです。並進成分は相対並進を観測座標系へ移した後、`SO(3)` 左 Jacobian の逆を用いて `se(3)` の並進座標へ写します。velocity、angular velocity、specific force、acceleration は観測 Loss に入りません。velocity と gyro は shooting node の初期値を作るためだけに使います。

全時系列を既定 0.5 秒の区間へ分け、内部境界の CoG pose、velocity、angular velocity、actuator thrust、gimbal angle を補助変数にします。区間終端と次区間始端の一致は augmented Lagrangian の連続性制約として反復的に強制します。最終選択は、推定物理パラメータを初期時刻から一本で再積分した full-rollout の SE(3) Loss で行います。

内側の最小二乗は既定 `max_nfev=120`、外側の augmented-Lagrangian は既定10反復です。短時間の動作確認では、`--max-nfev` と `--augmented-lagrangian-iterations` の両方を小さくできます。

```bash
source /home/leus/catkin_ws/devel/setup.bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py"
```

Command delayは既定で `0.16 s` に固定し、必要な場合だけ `--command-delay` で変更します。SE(3) log residual `[rho, phi]` の軌道lossは、従来の固定translation/rotation scaleではなく、nominal質量 `m0` とnominal慣性 `J0`による `||rho||^2 + phi^T (J0 / m0) phi` を各サンプルで評価して平均します。すなわち方位誤差をnominal慣性半径で位置誤差と同じ長さの次元へ正規化します。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --command-delay 0.16 \
  --segment-duration 0.5 \
  --max-nfev 120 \
  --augmented-lagrangian-iterations 10
```

結果は `minimal/output/deterministic_multiple_shooting/` に保存されます。

### Smooth-lag search + strict-ZOH multiple shooting

`--method deterministic_smooth_lag_multiple_shooting` は、既定methodを変更せずに、13物理座標とcommand lagを同時に探索する別推定器です。rotor thrust commandとgimbal commandの各ZOH切替をquintic smoothstepで局所的に滑らかにし、command値からactuator、body wrench、剛体軌道、SE(3) pose residualまでのlag感度を14列目の解析forward sensitivityとして伝播します。重複command timestampは同一時刻の最後のmessageへまとめ、各切替半幅は隣接周期の49%以下としてtransitionの重複を防ぎます。

Smooth searchはcommand周期に対する半幅比 `0.50 → 0.20 → 0.05` の3段階で行い、各段階で前段階の物理座標、lag、shooting nodeをwarm startします。入力モデルが変わる段階ごとにaugmented-Lagrangian multiplierはゼロから開始します。

Smooth段階は、continuityが許容値へ入った後の小さなdata改善を長く追わないよう、augmented-Lagrangian 1反復あたり既定60評価（`--smooth-max-nfev`）で区切ります。最終strict-ZOH refinementには従来どおり `--max-nfev`（既定120）を使います。

Smoothstep解は正式結果には使いません。推定lagの既定±4 msを1 ms刻みでstrict causal ZOH full rollout評価し、上位3候補だけをlag固定のmultiple shootingで再最適化します。continuity toleranceを満たす候補があれば、その中でstrict-ZOH full-rollout lossが最小のものを選びます。このため広い範囲のZOH profileは実行しません。

```bash
source /home/leus/catkin_ws/devel/setup.bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_smooth_lag_multiple_shooting \
  --delay-bounds 0.0 0.20 \
  --initial-delay 0.01 \
  --smoothstep-width-fractions 0.50 0.20 0.05 \
  --zoh-polish-radius 0.004 \
  --zoh-polish-step 0.001 \
  --zoh-polish-top-k 3
```

Pose lossは既定methodと同じ固定nominal計量 `||rho||² + phiᵀ(J0/m0)phi` です。`--body-displacement-scale` はこの6次元残差全体へ適用する単一の長さscaleで、既定は `1.0 m` です。Thrust/gimbalの一次遅れ時定数は入りません。

結果は `minimal/output/deterministic_smooth_lag_multiple_shooting/` に保存されます。`result.json` は各smoothstep段階のlag、continuity、full-rollout loss、lag gradientと、strict-ZOHのscreening/refinement結果を分離して記録します。`trajectory.pdf` と `parameters.txt` は最終strict-ZOH解だけを表示し、`delay_profile.pdf` はsmooth推定点、strict-ZOH局所profile、最終選択点を比較します。

### Multiple-bag shared-parameter multiple shooting

`deterministic_multi_bag_multiple_shooting_estimator.py` は、複数bagで13物理座標とcommand lagを共有し、shooting nodeだけをbagごとに独立に持つjoint問題を解きます。同一機体構成のbagだけを一つの設定へ含めてください。各bagの観測区間、初期状態、command history、サンプル数は異なって構いません。

設定はJSONで、`bags`の各要素に一意な`id`、bagの`path`、record-localの`start` / `end`、正の`weight`を指定します。相対pathは設定JSONの親directoryを基準に解決します。weightは内部で総和1へ正規化するため、すべて`1.0`なら各bagの係数は`1 / B`です。`initial_delay_seconds`を省略した場合は`0.01 s`です。

```json
{
  "bags": [
    {
      "id": "failure_1",
      "path": "/absolute/path/to/example_1.bag",
      "start": 19.0,
      "end": 24.0,
      "weight": 1.0
    },
    {
      "id": "failure_2",
      "path": "/absolute/path/to/example_2.bag",
      "start": 12.5,
      "end": 18.0,
      "weight": 1.0
    }
  ],
  "initial_delay_seconds": 0.01
}
```

提示済みの既存method名に`--config`を加える呼び方と、明示的なmulti-bag method名の両方を受け付けます。`--config`がなければ従来のsingle-bag既定methodのままです。

```bash
source /home/leus/catkin_ws/devel/setup.bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_multiple_shooting \
  --config /path/to/multi_bag_config.json
```

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_multiple_shooting_multi \
  --config /path/to/multi_bag_config.json
```

各bagのpose residualは従来どおりサンプル数の平方根で正規化した後、正規化weightの平方根を掛けて連結します。共有soft priorはbagごとのblockから除外し、joint residualの末尾へ一度だけ追加します。Continuity residualはweightを掛けず、各bagについて独立の等式制約として連結します。smoothstep continuationと最終strict-ZOH polishも、全bagのweighted full-rollout lossを使って共通lagを選びます。

Multi-bag版では、長い軌道の発散を人工的なshooting-node boxで遮らないよう、剛体node correctionの既定上下限を設けません。Thrustとgimbalのnodeはactuatorの物理範囲を維持します。Augmented-Lagrangian終了時にcontinuity toleranceへ達していない場合は、共有パラメータを固定したまま各bagのsegmentを先頭から順に伝播し、次nodeを直前segmentの終端へ置く構造的restorationを行います。これにより正式出力ではcontinuityを満たし、stitched軌道とfull rolloutを一致させます。明示的に有限node boundを指定してこの逐次解が範囲外になる場合だけ、bounded least-squares restorationへフォールバックします。

出力は`minimal/output/deterministic_multiple_shooting_multi/`です。共有結果は`result.json`と`parameters.txt`、lag診断は`delay_profile.pdf`へ保存し、bag別の`result.json`と`trajectory.pdf`は`bags/<id>/`へ保存します。共有結果にはjoint loss、prior cost、bag別loss/RMSE/continuity、weighted contribution、stitched/full-rollout差を記録します。

### Deterministic baseline

```bash
source /home/leus/catkin_ws/devel/setup.bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic
```

推定結果は `minimal/output/result.json` に、軌道と時系列の比較図は `minimal/output/trajectory.pdf` に生成されます。

PDF は rosbag の observed、推定前の nominal-parameter rollout、推定後の estimated-parameter rollout を同じ図へ重ねます。

Observed は青の実線、nominal は橙の破線、estimated は緑の点線で表し、位置、姿勢、速度、角速度、specific force を表示します。

試行時間を短くする場合は `--max-nfev` を小さくし、軌道 refinement を長く続ける場合は大きくします。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic --max-nfev 50
```

### Sobol multi-start deterministic baseline

`--method deterministic_sobol` は、現行物理パラメータをscreeningだけでなく局所最適化seedにも必ず含めた上で、既定では16次元の bounded chart に `2^9=512` 個の scrambled Sobol 点を生成します。

Mass、相対 rotor effectiveness、thrust/gimbal time constant は対数座標、inertia は正の対角を持つ Cholesky 座標、CoG と delay は通常の有界座標です。
Force effectiveness は geometric mean を1に固定する三つの contrast で表し、baseline と同じ common effectiveness/mass scale ambiguity を追加しません。

各点では剛体 inertia の triangle inequality を確認します。Production PID proposal と同じ acceleration-response compensation からPID gainも計算して結果へ記録しますが、既定ではPID範囲による棄却は行いません。
PID範囲を再び有効にする場合は、下限と上限を両方指定します。
通過した点だけを全軌道 recorded-control open-loop simulation で評価し、損失順に並べてから、正規化パラメータ距離が近すぎる点を除外して最大16点を局所最適化します。

これとは別に、nominal 値から局所運動方程式初期化と軌道 refinement を行う従来 deterministic baseline の経路も、同じrun内で必ず実行して incumbent として保持します。最終出力は、制約適合Sobol解がこの incumbent 以下の軌道損失を達成した場合だけSobol解へ置き換えるため、従来baselineより悪化しません。Baseline incumbent がSobol探索boxまたはPID許容帯を外れる場合は、最終出力の `selected_constraint_eligible` を `false` とし、制約内の最良解も別にJSONへ残します。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_sobol \
  --sobol-power 9 \
  --local-start-count 16 \
  --minimum-seed-distance 0.20
```

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_sobol \
  --pid-gain-min-scale 0.8 \
  --pid-gain-max-scale 1.2
```

結果は `minimal/output/deterministic_sobol/` に保存されます。

十分な全軌道損失を事前に定義できる場合は、収束かつ有効で、境界近傍の変数数も許容内の候補が得られた時点で残りを打ち切れます。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_sobol \
  --early-stop-loss 400 \
  --early-stop-max-boundary-hits 0
```

Delay は causal ZOH command に対して滑らかな変数ではないため、Sobol 点では探索しますが各局所 least-squares 中は固定します。
最終選択では、最小 trajectory loss の既定1%以内にある収束候補から、box bound、inertia triangle inequality、PID許容帯の境界近傍成分が最も少ないものを採用します。

各局所 least-squares の15変数 Jacobian は数値差分ではなく、Cholesky/log物理座標、active branch の actuator model、RK4 rigid-body rollout、SO(3)を含む全観測残差を連鎖した解析 forward sensitivity で計算します。Delay は局所solve中に固定されるため、このJacobianには含まれません。

### Local parallel-tempering deterministic initializer

`--method deterministic_tempered` は、nominal-start deterministic baseline の最良点を中心に、局所 Gaussian 内だけを replica exchange で探索します。全物理範囲へ点を撒かず、baseline 点での解析 trajectory Jacobian の SVD から、細い ridge に沿った proposal covariance を構成します。固有値には上下限を置くため、弱同定方向へ無制限には飛びません。

最初の局所 proposal 48点で実際の正の trajectory-loss 増分を測り、その中央値から8 replica の温度を自動較正します。既定では各 replica を96 sweep 動かし、4 sweep ごとに隣接温度間で状態交換します。各 proposal は forward simulation 1回だけで、全履歴の最良点から解析 Jacobian の least-squares を最後に1回だけ実行します。baseline incumbent も最終比較に残すため、出力 trajectory loss は baseline より悪化しません。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_tempered \
  --replica-count 8 \
  --sweeps 96 \
  --workers 4
```

PID gain は常に計算してJSONへ記録しますが、棄却は既定でオフです。有効にする場合だけ上下限を両方指定します。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_tempered \
  --pid-gain-min-scale 0.8 \
  --pid-gain-max-scale 1.2
```

温度を固定したい場合は `--temperature-min` と `--temperature-max`、探索幅を変える場合は `--proposal-scale` と15成分の `--local-prior-scales` を指定できます。結果は `minimal/output/deterministic_tempered/` に保存されます。Delay はこの局所探索でも固定です。

### Trajectory-length continuation with delay profile

`--method deterministic_continuation` は、Sobol点、tempering、時定数推定を使いません。推定対象は、mass 1、inertia Cholesky座標 6、CoG 3、相対force effectiveness 3と、外側でprofileするcommand delay 1の合計14次元です。Thrustとgimbalの時定数はそれぞれ `0.01 s`、`0.02 s`に固定します。

最初にnominal delayで13物理座標の厳密なnominal値から開始し、同じ開始時刻の軌道を `0.5 s → 1 s → 2 s → 全区間` と伸ばします。各段階は前段階の解をwarm startにし、13列の解析Jacobianでleast-squaresを解きます。
既定の評価上限は部分区間ごとに35回、最終の全区間だけ80回です。

Delayはzero-order hold commandに対して微分せず、既定では `0–0.08 s`を`0.02 s`刻みで粗くprofileします。Nominal delayの解から正負それぞれの隣接delayへwarm startし、最良の粗delayの周囲だけを`0.0025 s`刻みで再探索します。最後に全delay候補を初期時刻から全区間replayし、trajectory lossが最小のものを採用します。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_continuation \
  --continuation-horizons 0.5 1.0 2.0 \
  --coarse-delay-step 0.02 \
  --fine-delay-step 0.0025
```

PID gainは各候補について計算してJSONへ記録しますが、既定では棄却しません。有効にする場合だけ `--pid-gain-min-scale` と `--pid-gain-max-scale` を両方指定します。

結果は `minimal/output/deterministic_continuation/` の `result.json`、`trajectory.pdf`、`delay_profile.pdf` に保存されます。

### Deterministic baseline と対角 Q の交互推定

`--method deterministic_q` は deterministic の single-shooting 軌道誤差を維持し、観測 IMU 軌道上の required-minus-modeled body wrench に対する対角 Q 尤度を追加します。

固定 Q のパラメータ最適化と、Mahalanobis 項および `log det(Q/dt)` を含む Gaussian 尤度の閉形式 Q 更新を交互に実行します。

```bash
python3 "$(rospack find grape_param_estim)/minimal/estimate_recorded_control.py" \
  --method deterministic_q \
  --q-iterations 2 \
  --q-max-nfev 8
```

結果は `minimal/output/deterministic_q/` に保存されます。

この手法は latent trajectory を持たず、Q の residual は順方向シミュレーション軌道ではなく観測軌道から計算する inverse-dynamics residual です。

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
