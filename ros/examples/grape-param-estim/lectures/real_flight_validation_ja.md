# Grape 実飛行データによる同化・完全 hold-out 検証報告

## 1. 結論

実装した worker と artifact 契約は、指定された失敗飛行を読み、position / orientation
だけを尤度に用いる full closed-loop weak-constraint 同化を最後まで実行できた。また、
同化完了後まで隔離した成功飛行に対し、失敗飛行から得た raw physical posterior member
と一定遅れだけを移送する完全 hold-out 検証も完走した。

ただし、今回の推定値は次回実機飛行の品質保証にはならない。Q の全 89 knots を使った
正本 run は、成功飛行の位置・姿勢の全指標で nominal baseline より悪かった。8 knots へ
制限した感度 run では位置だけが改善し、姿勢は悪化した。開発用失敗飛行でも posterior の
姿勢 RMSE は nominal より悪く、pose component coverage は 0 だった。初期 prior ensemble
の縮約が必要で、終了理由も `maximum_iterations` である。したがって本報告の PID 結論は
**推奨なし** である。

## 2. データ分割と provenance

### 開発・同化データ

- bag:
  `/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/20260612_grape_hovering_4_2026-06-12-17-33-59.bag`
- SHA256:
  `bd3fc7f71797c0f5cb665acc50832da93c590e540fa170f9977182ecedf93bf8`
- 選択 local record-time interval: `7.386647939682007--24.901896476745605 s`
- inspector の判定: 完結 episode 内に `flight_state=5` はなく、control-active な
  `flight_state=3` 区間を警告付き fallback として選択
- 記録 PID snapshot (`xy`, `z`, `roll_pitch`, `yaw`):
  `[[3, 0.1, 1], [5, 1, 2.5], [20, 1, 8], [4, 1, 2]]`

### 完全 hold-out 検証データ

- bag:
  `/home/leus/catkin_ws/bags/grape-drone/20260613_grape_hovering/20260613_grape_hovering_3_2026-06-13-15-12-51.bag`
- SHA256:
  `a1569a48bf9a1d4d3f10a40bfc0e2c3c0cba192660b32204eeb37d1416425071`
- 要求 interval: `41.84692621231079--62.90664887428284 s`
- 0.1 s resample 後の実 interval: `41.84692621231079--62.84692621231079 s`
- inspector の判定: 完結 episode 内の `flight_state=5`、211 samples
- 記録 PID snapshot (`xy`, `z`, `roll_pitch`, `yaw`):
  `[[4, 0.1, 2], [5, 1, 2.5], [13, 1, 20], [6, 1, 2]]`

両 bag の configuration fingerprint は同じ値だったが、値は
`incomplete:38a4d489f6217d168795eb30416335110746c5f3270495f5e8edc721fdd57aa2`
である。payload、rotor / propeller、geometry、robot model revision、actuator wiring、
hardware revision が bag から復元できていないため、同一 hardware であることを fingerprint
だけで証明できてはいない。この不足は inspection と run の warning に残している。

## 3. hold-out における情報分離

成功飛行は、失敗飛行の同化 run が完成するまで同化・Q calibration・候補選択に使って
いない。成功飛行へ移送した source-run の値は、同じ `member_id` に属する次の値だけである。

- mass
- full inertia
- CoG
- rotor force effectiveness
- rotor torque effectiveness
- continuous constant delay

source bag の初期状態、controller state、actuator state、reference、residual-wrench path、
observations は移送していない。成功 bag 側では、記録済み reference、controller snapshot、
controller / actuator anchor を既知条件として使い、先頭の pose samples は初期 pose と
velocity の anchor を作るためだけに使った。その後は観測による state reset を一度も行わず、
全 interval の residual wrench を明示的に zero として full closed-loop forecast した。
全 pose target は forecast 入力と別に保持し、forecast 完了後の scoring で参照した。

比較する nominal baseline も同じ成功-flight anchor、reference、controller を使う。
物理値は固定 `VehicleParameters.nominal()`、一定遅れは 0 s であり、source run の nominal
推定値や posterior member ではない。したがって以下は「完全に未知の初期状態から将来を
予言する」検証ではなく、飛行開始時の pose / velocity で条件付けた後の observation-reset-free
closed-loop trajectory 検証である。

## 4. 失敗飛行に対する M=32、全 Q-knot 同化

正本とした run は `/tmp/grape-failed-assimilation-run-m32-full-backoff` である。設定は
sample period 0.1 s、ensemble size 32、maximum iterations 3、OU knot 89、delay prior
`0.020 +/- 0.015 s`、seed 20260612 である。joint control dimension は 579 であり、solver
は rank 31 の ensemble span 内で更新した。

要求 prior の scale 1.0 では member 25 の forecast が overflow した。member の除外や
有限な罰 residual への置換はせず、全 member の prior center からの偏差を同じ dyadic
係数で縮め、scale 0.5 で全 32 forecasts が有限になった。中心の最大差は
`5.90e-17`、rank は `31 -> 31` で、member ID と方向は保存された。失敗した scale、
例外型・理由、要求 / 実効 prior ensemble は artifact に保存している。この縮約は観測への
fit 値を使わない numerical-feasibility 操作だが、実効 prior を狭めるため、推定結果の重要な
条件である。

iteration objective は次のように低下した。

| iteration | objective | accepted objective | accepted fraction |
|---:|---:|---:|---:|
| 0 | `9.7522e11` | `1.1084e10` | 1.0 |
| 1 | `1.1084e10` | `2.7713e9` | 1.0 |
| 2 | `2.7713e9` | `1.9813e9` | 1.0 |

しかし termination は `maximum_iterations`、`converged=false` である。観測位置に対する
norm RMSE は nominal `2.7241 m`、posterior raw-member median `0.12071 m` まで低下したが、
観測姿勢に対する norm RMSE は nominal `0.20526 rad` から posterior median
`0.22072 rad` へ悪化した。pose component 95% coverage は 0 である。

shared posterior の主要 marginal は次の通りである。

| quantity | median | 95% interval |
|---|---:|---:|
| mass [kg] | 2.611299 | [2.611268, 2.611320] |
| constant delay [s] | 0.0009976 | [0.0009958, 0.0009992] |
| CoG x [m] | 0.0092578 | [0.0092563, 0.0092595] |
| CoG y [m] | -0.0108008 | [-0.0108022, -0.0107995] |
| CoG z [m] | 0.0043822 | [0.0043799, 0.0043836] |

この極端に狭い interval を精密な同定と解釈してはならない。ensemble rank は 31、run は
非収束、coverage は 0 であり、初期 prior 縮約と短い一飛行への conditional ensemble collapse
を含む。

Q calibration の stationary standard deviation は force
`[0.2796, 0.1643, 7.3114] N`、torque `[0.04188, 0.03579, 0.01413] N m`、
correlation time `0.4175 s` だった。校正された全 89 knots を使い、
`q_resolution_sufficient=true` である。

### 4.1 8-knot 感度 run

計算量を制限した `/tmp/grape-failed-assimilation-run-m32-k8-backoff` は control dimension 93、
`q_resolution_sufficient=false` である。同じ M=32 と scale 0.5 でも mass median は
`2.42415 kg`、delay median は `38.456 ms` となり、全-knot run の `2.61130 kg`、
`0.998 ms` と大きく異なった。knot 数の違いに対して static physical parameter と delay が
安定せず、Q path と delay の分離が実飛行で確認できていない。

### 4.2 既知遅れ synthetic 回帰

実飛行の不整合と、continuous-delay 実装そのものの誤りを区別するため、0.04 s の simulation
格子に対して真の sub-sample delay を 0.017 s とした pose-only joint assimilation を固定 seed
で回帰試験した。delay prior standard deviation は 0.012 s である。

| Q law | delay 95% interval [s] | posterior SD [s] |
|---|---:|---:|
| near-zero Q | [0.01458, 0.03439] | 0.00586 |
| flexible Q | [0.01266, 0.03297] | 0.00571 |

両方が真値 0.017 s を被覆し、flexible Q でも 95% 下限が zero-delay boundary より大きく、
posterior variance は prior より小さかった。mass / effectiveness の exact ridge、全 physical
chart、delay、Q を同時に扱っている。この回帰は既知 family の synthetic 識別性を確認するが、
Q-knot 数で 1 ms と 38 ms に分かれた実飛行 posterior の妥当性を救済するものではない。

## 5. 成功飛行での M=32 完全 hold-out 結果

artifact は `/tmp/grape-heldout-success-validation-m32-full-q` で、32 / 32 member と nominal
baseline が数値破綻なく完走した。表は nominal と posterior raw-member median を比較する。

| scoring target / metric | nominal | posterior median | relative change |
|---|---:|---:|---:|
| observed position RMSE [m] | 0.17520 | 0.21024 | +20.0% |
| observed orientation RMSE [rad] | 0.30822 | 0.39936 | +29.6% |
| observed maximum position error [m] | 0.45947 | 0.56440 | +22.8% |
| observed maximum orientation error [rad] | 0.44917 | 0.54627 | +21.6% |
| reference position RMSE [m] | 0.14179 | 0.17818 | +25.7% |
| reference orientation RMSE [rad] | 0.08687 | 0.13060 | +50.3% |
| reference maximum position error [m] | 0.41909 | 0.52420 | +25.1% |
| reference maximum orientation error [rad] | 0.21465 | 0.22405 | +4.4% |

全-knot posterior は八指標すべてで nominal より悪かった。position と orientation を
一つの weighted score へ足していないので、個別の悪化を隠していない。

8-knot M=32 run は observed / reference position RMSE を 18.2% / 24.1% 改善した一方、
observed / reference orientation RMSE を 23.3% / 154.4% 悪化させた。比較用の早期
M=16 / 2-iteration run でも、observed position RMSE は 13.7% 改善した一方、
observed orientation RMSE は 5.5% 悪化した。reference orientation は M=16 では改善し、
8-knot M=32 では大幅に悪化した。全-knot run は全指標が悪化しており、Q resolution、
ensemble size、iteration、prior-feasibility 処理に対して推定が安定しているとは言えない。

## 6. PID proposal の実データ評価

M=16 run について、recorded current と全 16 member-derived exact candidates を、
16 physical members x 1 failed-flight scenario で評価した。`posterior_replay` と `zero` の
二つの residual policy は別 artifact とし、いずれも 272 forecasts が完走した。threshold
は未設定であり、墜落確率や physical failure probability は計算していない。

`posterior_replay` では current を含む 6 candidates、`zero` では current を含む
9 candidates が Pareto non-dominated になり、最良 metric を与える member は物理指標ごとに
異なった。policy を変えると Pareto 集合も変わった。明示 selected candidate は null のため、
artifact の `recommendation_available` は false であり、YAML diff は空である。

最新の全-knot M=32 posterior から導いた raw proposal scale median は `xy=0.83567`、
`z=0.90145`、`roll_pitch=0.97288`、`yaw=0.67725` だったが、95% range は非常に狭い。
これは前節の posterior collapse をそのまま push-forward した結果であり、確信度とは
解釈しない。また M=32 physical member law は hold-out の全指標を悪化させたため、この
exact gain を実機へ適用する根拠にはならない。

## 7. 判定と再現性

今回確認できたことは、指定失敗 bag を使う end-to-end data path、raw-member alignment、
continuous delay、full closed-loop forecast、strict artifact、完全 hold-out scoring が機械的に
動作すること、および Q を制限した感度 run では失敗飛行 posterior が成功飛行の位置挙動に
一部整合したことである。全 Q-knot を用いる正本 run の外的整合は確認できなかった。

確認できなかったことは、姿勢を含む joint physical law の整合、posterior coverage、Q と
constant delay の安定した分離、complete hardware fingerprint、反復・ensemble size に対する
安定性、次回実機飛行の安全性である。したがって current PID を自動変更せず、提案値を
flight recommendation として出力しない。

再実行には、同梱 worker の request-file interface を使う。

```bash
rosrun grape_param_estim grape_assimilate_flights.py \
  --request /path/to/assimilation-request.json \
  --output /path/to/assimilation-run

rosrun grape_param_estim grape_validate_held_out_flight.py \
  --request /path/to/held-out-validation-request.json \
  --output /path/to/held-out-validation
```

source bag、request SHA、選択 interval、controller snapshot、要求 / 実効 prior、raw forecast、
metric は各 directory bundle に pickle-free で保存され、strict loader が再計算して検査する。

## 8. 実装・runtime 検証

最終 source tree で次を実行した。

| check | result |
|---|---:|
| backend `unittest` 全 discovery | 192 tests、0 failures、0 errors |
| `catkin build grape_param_estim --no-deps` | success、warning なし |
| `catkin run_tests grape_param_estim --no-deps` | 192 tests、0 failures、0 errors、0 skips |
| GUI test discovery (`gui/.venv`) | 49 tests、49 success、0 skips |
| Python `compileall` | success |

GUI は pyenv Python 3.10.18 の `gui/.venv` で検証した。依存 version は PySide6 6.9.3、
pyqtgraph 0.14.0、PyVista 0.46.5、PyVistaQt 0.11.4、VTK 9.5.2 である。request、strict artifact
adapter、project ZIP / ZIP64、freshness、launcher、SIGINT / terminate / kill 後の cancellation、
Qt widget、実 manifest / NPZ を含む project load E2E、3D test を含む 49 tests がすべて成功した。

自動 test に加え、`DISPLAY=:1`、Qt `xcb` backend、Mesa software rendering で production UI と
VTK plotter を起動した。Master、Bag browser の world / correction、Next experiment の PID
translation / rotation / trajectory を確認し、14 枚の PNG と各画像の寸法・byte 数を含む
`summary.json` を `/tmp/grape-gui-visual-acceptance` に保存した。summary では bag world / correction
と PID plotter の availability も true である。window image は X server の
`QScreen.grabWindow` で取得し、ネイティブ VTK 子画面を含む。VTK framebuffer の直接画像も
別に保存して確認した。

ホストに不足していた `libxcb-cursor0` は sudo による system install を行わず、deb を
`gui/.venv/qt-runtime` へ展開し、その library directory を `LD_LIBRARY_PATH` に追加した。
視覚受入のために元の assimilation / PID artifact は変更していない。GUI freshness 検査に必要な
`project_request_fingerprint` は `/tmp/grape-visual-assimilation-run` の視覚確認用コピーへだけ
追加した。

本報告の実 rosbag run は、configuration provenance が不足した事実を warning に保持したうえで、
worker / CLI を直接使って数値経路を検証したものである。production GUI は同じ incomplete
fingerprint を `needs_configuration_confirmation` とし、payload、rotor / propeller、geometry、
robot model revision、actuator wiring、hardware revision を確認して再 inspection するまで joint
run を開始しない。この安全側の GUI 契約を、報告用の直接実行によって緩めてはいない。
