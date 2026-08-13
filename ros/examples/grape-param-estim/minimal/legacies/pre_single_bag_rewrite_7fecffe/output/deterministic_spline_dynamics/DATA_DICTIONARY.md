# Deterministic spline dynamics estimator: data dictionary

この文書は、`minimal/output/deterministic_spline_dynamics/` に保存される数値が何を表すかを列挙するためのものです。プロットの採否はこの一覧を基準に決められます。

## 記号と共通規約

| 記号 | 意味 |
|---|---|
| `N_o` | `output_time` 上の出力サンプル数。現行本番データではfailure_1が120、failure_2が110 |
| `N_c` | `collocation_time` 上のspline/dynamics評価点数。spline fit全区間から両端の境界影響区間を除いた点数 |
| `W` | world frame |
| `B` | main body frame |
| `S_pose` | mocap pose sensor frame |
| `S_vel` | world velocityを出すsensorの位置 |
| `S_imu` | gyro/specific-force sensor frame |
| `R_WB` | body座標のvectorをworld座標へ写す回転行列 |

- 時刻の単位は秒です。値はrosbag record開始時刻を0としたbag-local timeであり、各切出し区間の先頭を0とした時間ではありません。経過時間として描く場合は、各time arrayからその先頭要素を引きます。
- 3-vectorのcomponent順は常に`[x, y, z]`です。
- quaternionは`[x, y, z, w]`順です。pose orientationは`R_WS_pose`、すなわちsensor frameからworld frameへの回転です。
- body wrenchの6-vector順は`[F_x, F_y, F_z, M_x, M_y, M_z]`です。力はN、torqueはN mです。
- `observed`はbagの実測値、`spline`はobserved poseだけから作った5次spline、`estimated_forward`は推定parameterだけの補正なし自由積分です。
- `external_wrench_forward`は推定parameterに同じbagから逆算した外力を加えた再積分です。これは独立な予測ではありません。
- `nominal_forward`はnominal parameterとconfigのinitial lagによる補正なし自由積分です。
- spline fitはobserved poseの全区間を使います。parameter loss、required/model/residual wrenchは、既定では5次basisの半supportに当たる3 knot spansを両端から除いた内側区間だけを使います。

## `bags/<id>/spline_dynamics.npz`

### 時刻軸

| key | shape | 内容 |
|---|---:|---|
| `output_time` | `(N_o,)` | observed channelと全forward rolloutの共通時刻。既定間隔は0.05 s |
| `collocation_time` | `(N_c,)` | pose spline、解析微分、dynamics wrenchの評価時刻。設定上の間隔は0.01 s |
| `inferred_external_body_wrench_time` | `(N_c,)` | 外力系列の時刻。現在は`collocation_time`と完全に同一の互換alias |
| `spline_fit_time_bounds` | `(2,)` | splineをfitした全supportの`[start,end]` |
| `parameter_estimation_time_bounds` | `(2,)` | parameter lossとwrench計算に使った内側supportの`[start,end]` |
| `parameter_estimation_output_mask` | `(N_o,)` | `output_time`がparameter推定support内なら`True`となるboolean mask |
| `spline_boundary_exclusion_seconds_start_end` | `(2,)` | spline fit supportから実際に除いた秒数の`[start side,end side]` |
| `spline_boundary_exclusion_knot_spans_each_side` | scalar | 両端から除外するknot span数。既定は3.0 |

`output_time`系列と`collocation_time`系列を同じplotへ載せる場合、明示的にinterpolationする必要があります。array indexをそのまま対応させてはいけません。

### Observed channels

| key | shape | 単位／表現 | frame | 内容・推定への使用 |
|---|---:|---|---|---|
| `observed_sensor_position` | `(N_o,3)` | m | `W` | 実測pose sensor位置。5次位置splineの入力。parameter lossへはsplineを介して入る |
| `observed_sensor_orientation_xyzw` | `(N_o,4)` | unit quaternion | `R_WS_pose` | 実測pose sensor姿勢。5次quaternion splineの入力。parameter lossへはsplineを介して入る |
| `observed_sensor_velocity_world` | `(N_o,3)` | m/s | vectorを`W`で表現、位置は`S_vel` | 実測world velocity。parameter lossには未使用、forward rolloutの独立検証用 |
| `observed_angular_velocity_sensor` | `(N_o,3)` | rad/s | `S_imu` | 実測gyro。preflight biasを含む観測値。parameter lossには未使用 |
| `observed_specific_force_sensor` | `(N_o,3)` | m/s² | `S_imu` | 実測specific force。重力そのものは含めず、preflight accelerometer biasを含む。parameter lossには未使用 |

### Pose spline and analytic derivatives

| key | shape | 単位／表現 | frame | 内容 |
|---|---:|---|---|---|
| `spline_sensor_position` | `(N_c,3)` | m | `W` | observed pose sensor位置へfitした5次B-splineの値 |
| `spline_sensor_velocity_world` | `(N_c,3)` | m/s | `W` | `spline_sensor_position`の解析1階微分 |
| `spline_sensor_acceleration_world` | `(N_c,3)` | m/s² | `W` | `spline_sensor_position`の解析2階微分 |
| `spline_body_rotation` | `(N_c,3,3)` | rotation matrix | `R_WB` | 5次正規化quaternion splineから得たbody姿勢 |
| `spline_body_angular_velocity` | `(N_c,3)` | rad/s | `B` | 同じquaternion splineの解析微分から得たbody角速度 |
| `spline_body_angular_acceleration` | `(N_c,3)` | rad/s² | `B` | 同じquaternion splineの解析2階微分から得たbody角加速度 |

positionとquaternionのspline degreeは5です。内部knotが単純knotなので、curveはC4、2階微分は少なくともC2連続です。ここに保存されるのはspline値と微分値であり、spline coefficientやknot vectorそのものは現在保存していません。

### Required, modeled, and residual body wrench

| key | shape | 単位 | frame | 内容 |
|---|---:|---|---|---|
| `required_body_wrench` | `(N_c,6)` | `[N,N,N,N m,N m,N m]` | `B` | pose splineの運動を実現するために必要なwrench |
| `modeled_body_wrench` | `(N_c,6)` | 同上 | `B` | 推定parameter、推定lag、記録command、strict-ZOH actuator modelが生成するwrench |
| `residual_body_wrench` | `(N_c,6)` | 同上 | `B` | `required_body_wrench - modeled_body_wrench` |
| `inferred_external_body_wrench` | `(N_c,6)` | 同上 | `B` | 現在は`residual_body_wrench`と完全に同一の互換alias |

推定されたCoG offsetを`c`、body原点からpose sensorまでの既知vectorを`r_BS`とすると、`r_GS = r_BS - c`です。splineからCoG加速度`a_G`を計算し、次を保存しています。

```text
required force  = m R_WB^T (a_G - g_W)
required torque = J alpha_B + omega_B x (J omega_B)
residual wrench = required wrench - modeled wrench
```

重要事項：`inferred_external_body_wrench`は独立にsmooth splineとして推定した外力ではありません。0.01 sごとの生の逆動力学残差です。推定parameterの`m`, `J`, `c`にも依存します。

### Estimated free forward rollout

共通時刻は`output_time`です。推定した共有parameterとstrict-ZOH lagを使い、先頭状態だけをpose splineから作って、その後を外力補正なしで自由積分します。

| key | shape | 単位／表現 | frame／位置 |
|---|---:|---|---|
| `estimated_forward_sensor_position` | `(N_o,3)` | m | pose sensor位置、`W` |
| `estimated_forward_sensor_orientation_xyzw` | `(N_o,4)` | quaternion | `R_WS_pose` |
| `estimated_forward_sensor_velocity_world` | `(N_o,3)` | m/s | `S_vel`の速度を`W`で表現 |
| `estimated_forward_angular_velocity_sensor` | `(N_o,3)` | rad/s | `S_imu`。model body角速度をsensor frameへ変換し、記録済みpreflight gyro biasを加えた値 |
| `estimated_forward_specific_force_sensor` | `(N_o,3)` | m/s² | `S_imu`。IMU lever-arm項と記録済みpreflight accelerometer biasを含む |

### Estimated rollout with inferred external wrench

推定parameterのforward modelへ`inferred_external_body_wrench`をbody force/torqueとして加えた系列です。wrenchはparameter推定support内の`collocation_time`間だけを時間方向に線形補間します。有効support外では0とし、端点値のhold外挿はしません。共通時刻は`output_time`です。

| key | shape | 単位／表現 | frame／位置 |
|---|---:|---|---|
| `external_wrench_forward_sensor_position` | `(N_o,3)` | m | pose sensor位置、`W` |
| `external_wrench_forward_sensor_orientation_xyzw` | `(N_o,4)` | quaternion | `R_WS_pose` |
| `external_wrench_forward_sensor_velocity_world` | `(N_o,3)` | m/s | `S_vel`の速度を`W`で表現 |
| `external_wrench_forward_angular_velocity_sensor` | `(N_o,3)` | rad/s | `S_imu`、gyro bias込み |
| `external_wrench_forward_specific_force_sensor` | `(N_o,3)` | m/s² | `S_imu`、accelerometer bias込み |

### Nominal forward rollout

| key | shape | 単位／表現 | frame／位置 | 内容 |
|---|---:|---|---|---|
| `nominal_forward_sensor_position` | `(N_o,3)` | m | pose sensor位置、`W` | nominal物理parameter、config initial lag、外力なし |
| `nominal_forward_sensor_orientation_xyzw` | `(N_o,4)` | quaternion | `R_WS_pose` | nominal物理parameter、config initial lag、外力なし |

nominal rolloutのvelocity、gyro、specific forceは現在NPZへ保存していません。

## 現在計算しているがNPZへ保存していない系列

必要なら保存対象へ追加できます。現時点でplot用NPZから直接は読めません。

| 内部変数 | 主なshape | 内容 |
|---|---:|---|
| `BagDynamicsEvaluation.cog_position` | `(N_c,3)` | splineと推定CoG offsetから計算したCoG位置、`W`、m |
| `BagDynamicsEvaluation.cog_velocity_world` | `(N_c,3)` | spline由来CoG速度、`W`、m/s |
| `BagDynamicsEvaluation.cog_acceleration_world` | `(N_c,3)` | spline由来CoG加速度、`W`、m/s² |
| `BagDynamicsEvaluation.actuator_thrust` | `(N_c,4)` | lag適用後の4 rotor thrust state、N |
| `BagDynamicsEvaluation.actuator_gimbal` | `(N_c,4)` | lag・rate/angle limit適用後の4 gimbal angle、rad |
| forward simulationの`cog_position` | `(N_o,3)` | estimated／external-wrench／nominal rolloutのCoG位置 |
| forward simulationの`cog_velocity_world` | `(N_o,3)` | 各rolloutのCoG速度 |
| forward simulationの`actuator_thrust` | `(N_o,4)` | 各rolloutのrotor thrust state |
| forward simulationの`actuator_gimbal` | `(N_o,4)` | 各rolloutのgimbal angle |
| spline coefficient／knot vector | bag依存 | 5次position/quaternion splineの生parameter。現在は非保存 |

## `result.json`: shared result

共有結果は`minimal/output/deterministic_spline_dynamics/result.json`です。

| JSON path | 内容 |
|---|---|
| `initial_estimate.*` | optimizer開始値。既定実行では`source_kind="nominal"`、13次元`physical_coordinate`は全要素0 |
| `settings.*` | sample/integration/collocation step、degree、knot候補、prior、lag探索設定 |
| `smoothstep_stages[]` | smooth command lag continuation各段のparameter、loss、optimizer状態 |
| `strict_zoh_polish[]` | strict-ZOH lag候補のscreening cost、再最適化結果、選択flag |
| `selection.physical_coordinate` | 選ばれた13次元のsmooth物理座標 |
| `selection.delay_seconds` | 選ばれたrecorded-command lag、s |
| `selection.parameters` | 13次元座標を物理単位へdecodeした共有parameter |
| `selection.joint_dynamics_loss` | 全bagのweighted dynamics data loss。soft priorを含まない |
| `selection.soft_prior_cost` | 共有物理座標へ一度だけ加えたsoft prior cost |
| `selection.joint_objective_cost` | data lossとprior costの合計 |
| `bag_diagnostics[]` | 各bagのspline、wrench、forward、sensor診断 |
| `outputs.*` | 生成物への相対path |

### `selection.parameters`

| key | shape | 単位 | 内容 |
|---|---:|---|---|
| `mass_kg` | scalar | kg | 推定質量 |
| `inertia_kg_m2` | `(3,3)` | kg m² | CoGまわり、body frame表現の推定慣性行列 |
| `cog_offset_m` | `(3,)` | m | body frame原点からCoGへのvector、body frame表現 |
| `force_effectiveness` | `(4,)` | 1 | nominal rotor forceへ掛けるrotor別倍率 |
| `torque_effectiveness` | `(4,)` | 1 | rotor torque倍率。現推定器では固定nominal |
| `linear_drag` | `(3,)` | model固有 | 現推定器では固定nominal |
| `angular_drag` | `(3,)` | model固有 | 現推定器では固定nominal |

### 13次元`physical_coordinate`

index順は`selection.physical_parameter_names`にも保存されています。

| index | name | decode先 |
|---:|---|---|
| 0 | `log_mass_scale` | `m = m0 exp(q0)` |
| 1 | `log_second_moment_cholesky_xx_scale` | second-moment Cholesky `L_xx = L0_xx exp(q1)` |
| 2 | `log_second_moment_cholesky_yy_scale` | `L_yy = L0_yy exp(q2)` |
| 3 | `log_second_moment_cholesky_zz_scale` | `L_zz = L0_zz exp(q3)` |
| 4 | `normalized_second_moment_cholesky_yx_offset` | `L_yx`への正規化加算 |
| 5 | `normalized_second_moment_cholesky_zx_offset` | `L_zx`への正規化加算 |
| 6 | `normalized_second_moment_cholesky_zy_offset` | `L_zy`への正規化加算 |
| 7 | `cog_offset_x_m` | nominal CoG offsetのxへ加算、m |
| 8 | `cog_offset_y_m` | nominal CoG offsetのyへ加算、m |
| 9 | `cog_offset_z_m` | nominal CoG offsetのzへ加算、m |
| 10 | `force_effectiveness_contrast_1` | rotor別log effectiveness contrastの第1座標 |
| 11 | `force_effectiveness_contrast_2` | rotor別log effectiveness contrastの第2座標 |
| 12 | `force_effectiveness_contrast_3` | rotor別log effectiveness contrastの第3座標 |

慣性は`Sigma = L L^T`, `J = trace(Sigma) I - Sigma`でdecodeします。これにより正定値性と主慣性momentのtriangle inequalityを保ちます。force effectivenessは4 rotorの共通scaleを質量と重複させないzero-sum contrast basisです。

## `bags/<id>/result.json`: bag-local diagnostics

| JSON path | 内容 |
|---|---|
| `source.*` | bag id/path/hash、要求区間、raw/normalized weight |
| `shared_parameters` | shared resultの物理parameterの複製 |
| `shared_physical_coordinate` | shared 13次元座標の複製 |
| `shared_delay_seconds` | shared lagの複製 |
| `diagnostics.spline.degree` | 5 |
| `diagnostics.spline.selected_knot_spacing_seconds` | 時間幅固定block CVで選ばれたbag別knot spacing |
| `diagnostics.spline.fit_interval_seconds` | observed poseをspline fitした全区間 |
| `diagnostics.spline.parameter_estimation_interval_seconds` | parameter lossとwrenchに使用した内側区間 |
| `diagnostics.spline.boundary_exclusion_knot_spans_each_side` | 各端から除いたknot span数 |
| `diagnostics.spline.actual_boundary_exclusion_seconds_start_end` | 実際の秒単位の開始側／終了側除外幅 |
| `diagnostics.spline.blocked_cross_validation[]` | `settings.spline_cross_validation_block_seconds`幅の連続blockをfoldへ巡回配置したCVについて、各knot候補のpose validation error、CV成否・失敗理由、微分sanity |
| `diagnostics.spline.fit_metrics` | 全pose sample上のspline fit errorと最大加速度 |
| `diagnostics.dynamics_loss` | bag単独の平均spline dynamics loss |
| `diagnostics.residual_wrench_statistics` | 6軸residualのmean、RMSE、一次trend、時間積分 |
| `diagnostics.estimated_forward_metrics` | observed対estimated free rolloutのpose error |
| `diagnostics.estimated_with_external_wrench_forward_metrics` | observed対外力込みrolloutのpose error |
| `diagnostics.nominal_forward_metrics` | observed対nominal rolloutのpose error |
| `diagnostics.sensor_validation` | observed対estimated free rolloutのvelocity/gyro/specific-force軸別統計 |
| `diagnostics.sensor_validation_with_external_wrench` | observed対外力込みrolloutの同統計 |
| `outputs.*` | bag内生成物名 |

### Metricsの定義

| key | 定義 |
|---|---|
| `position_rmse_m` | `sqrt(mean(||p_pred-p_obs||²))`。3軸vector normのRMSE |
| `position_component_rmse_m` | 各軸別`sqrt(mean(error_axis²))` |
| `orientation_angle_rmse_rad/deg` | `Log(R_obs^T R_pred)`のnormを各時刻の角度誤差として取ったRMSE |
| `terminal_position_error_m` | 最終時刻のposition error norm |
| `terminal_orientation_error_deg` | 最終時刻のorientation log error norm |
| `*_forward_pose_score_m2` | `0.5 mean(||position error||² + phi^T(J0/m0)phi)` |
| sensor `rmse` | そのchannel・軸の`prediction-observation` RMSE |
| sensor `mean_bias` | そのchannel・軸の`prediction-observation`平均 |
| sensor `pearson_correlation` | 平均除去後のPearson correlation |
| sensor `maximum_cross_correlation_time_shift_seconds` | 非正規化cross-correlation最大点のsample lag |
| wrench `cumulative_impulse` | 全区間のresidual componentの台形積分。forceはN s、torqueはN m s |

## 現在のPDFと重複関係

| file | 現在の内容 |
|---|---|
| `spline_fit.pdf` | 6 pages: position fit、RPY orientation fit、pose residual、並進微分、回転微分、knot-spacing CV |
| `trajectory_3d.pdf` | 9 pages: 3D trajectory、position、orientation、velocity、gyro、specific force、position error、orientation error、補助nominal比較 |
| `trajectory.pdf` | `trajectory_3d.pdf`のbyte-for-byte copy。別データではない |
| `sensor_validation.pdf` | 3 pages: velocity、gyro、specific forceのobserved/free/external-wrench rollout比較 |
| `residual_wrench.pdf` | 7 pages: 6軸residual、required/model force、force residual、force積分、required/model torque、torque residual、torque積分 |
| `external_wrench.pdf` | `residual_wrench.pdf`のbyte-for-byte copy。別データではない |
| shared `parameters.pdf` | `parameters.txt`と同内容のmonospace PDF |
| shared `delay_profile.pdf` | smooth lag、strict-ZOH候補、再最適化候補、選択lag |

## Plot系列を指定するときの指定形式

曖昧さを避けるため、次の4項目を指定できます。

```text
bag: failure_1 / failure_2 / both
x: output_time または collocation_time（必要なら先頭を引く）
y: spline_dynamics.npzのkeyとcomponent
comparison: 同じaxisへ載せる別key
```

例：

```text
bag: both
x: output_time - output_time[0]
y: observed_specific_force_sensor[:, 2]
comparison:
  estimated_forward_specific_force_sensor[:, 2]
  external_wrench_forward_specific_force_sensor[:, 2]
```
