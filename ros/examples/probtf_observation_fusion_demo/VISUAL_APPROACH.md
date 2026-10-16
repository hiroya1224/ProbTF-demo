# 初期のマーカー中央維持と近距離のIK

## 画像中央維持を初期点提案へ限定する訂正（2026年10月17日最新）

最新の通常接近は [REACHING_CONTROL.md](REACHING_CONTROL.md) に従う。初期から毎周期、同じ推定Kと完全な正規化Gaussian/Bingham規準Jの勾配で指令を選ぶ。以下に記録したvisual局所二乗損失は初期点の提案にだけ使い、その速度を直接実行しない。共通boundsへ収めた提案がJを改善し、全目標成分の相対位置二乗誤差期待値sが最大減速基準以下の場合だけ初期点に採用し、続いて完全Jの勾配更新を一回行う。提案がなくても、拒否されても、近距離でも完全IKを使う。

`near_distance_m=0.08` とlatchはvisual提案を停止する条件としてだけ残り、完全IKを始める条件ではない。通常接近の `reaching_phase` は最初から `indet_ik`、Phase表示は `IK` である。探索Actionとの独立、実行区間sourceと次計画sourceの区別は維持する。以下の `visual_approach` phase／`APPROACH | CENTER MARKER` 表示は旧sourceの履歴である。

位置条件は同じ全lawの `s=sum_k w_k (||m_rel,k-d_MG||²+trace(S_rel,k))` であり、真の到達誤差や全マーカー輪郭の視野保証ではない。連成QPと同lawのline searchで扱い、完全J・共分散・正規化定数・全mixtureは変更しない。画像中央維持の初期点だけで視野や実検出の継続を保証しない。board・停止観測探索・カメラ露光0.09秒・画像サイズ・粒子・軸・RViz設定を維持する。

以下の遠距離visual選定と近距離完全IKへの切替は撤回済みの実装記録であり、本節に優先しない。旧凍結sourceの実画像・試験結果を新しい指令選定の性能証拠へ流用しない。

## 近距離閾値・探索step・ブラーの追加訂正（2026年10月17日）

近距離切替の既定値を0.18 mから0.08 mへ下げ、画像中央維持をより近くまで続ける。一度のnear latchは維持する。探索は関節5先行、初期振幅0.06 rad・増分0.03 rad・上限0.18 rad・速度上限0.048 rad/sに変更し、外側と中心の各waypointで完全停止して0.35秒観測する。詳細な停止後観測の融合・採用条件は [REQUIREMENTS_SEQUENTIAL_UPDATE.md](REQUIREMENTS_SEQUENTIAL_UPDATE.md) 冒頭の最新訂正に従う。

モーションブラーの露光だけを0.18秒から0.09秒へ半減する。`config/switch_press_scene.yaml` の `camera.exposure_time` 以外のカメラ設定は維持し、Gaussianぼかし2 px、時間補間7点、noise_std=8、画像960×720と較正を変えない。真値を使うのは従来どおり模擬画像生成と検証だけであり、推定・制御へ真の姿勢を渡す変更ではない。annotationは緑輪郭と姿勢だけを保ち、画像表示サイズ・粒子・軸・RViz設定を変えない。旧sourceでの画像・動作監査は新設定の性能検証と分ける。

露光変更時は保存した変更前sceneファイルとの比較で、この1行以外の全バイトが一致することを確認した。`test_moving_scene.py` の3試験も通過した。これは設定差分・既存物理入力境界の検証であり、新しい探索の視野維持・到達成功を示す実配信結果ではない。

2026年10月17日のユーザー指示により、初期はマーカーを画像中央へ保ちながら接近し、近距離で既存のIKへ切り替える。この追加は既定の非同期 `async` に適用する。従来の同期 `live` と `prepared_playback` は診断用の既存規準を維持し、新しいphaseを適用した実行とは説明しない。

本書と `AGENTS.md` 冒頭の追加訂正は、「通常の接近指令はすべて完全なGaussian/Bingham規準Jだけで選定する」という旧方針に、遠距離phaseの範囲で優先する。近距離の完全J、同じ候補分布をIK・表示へ渡す条件、全mixture、逐次保持、剛性補正、停止・探索、表示設定の保護を緩めない。

## 変更しない候補分布と推定

`MovingController.reaching_step` は、その時点の同じcanonical身体、根元固定目標の全mixture、全交差共分散を参照する。目標構成は [FRESH_OBSERVATION_GOALS.md](FRESH_OBSERVATION_GOALS.md)、欠測時の保持と指令輸送は [REQUIREMENTS_SEQUENTIAL_UPDATE.md](REQUIREMENTS_SEQUENTIAL_UPDATE.md) に従う。

新しい初期事前や表示専用分布を作らず、目標の最大重み一枝や平均だけへ置き換えない。遠距離と近距離のどちらでも、全成分のsource ID・指令・時刻・版、weight・log weightを保つ。局所制御損失を候補分布そのものとして使わず、剛性モデルから身体を再生成しない。新しい画像の融合、BF更新、剛性密度比因子を制御のために重複適用しない。

同じ周期に凍結した推定Kの指令作用と、その指令Jacobianを使う。Kが得られていない場合は宣言済みの指令右作用を使い、真のKや物理姿勢を渡さない。IKの試行に使う作用と、実行後の次事前へ運ぶ作用の対応は維持する。

## 遠距離phaseの規準

観測済みの固定目標がある場合、`visual_approach` では各目標成分 `k` の同じ共同lawから、次をcoreで照会する。

- カメラ相対のマーカー位置のmean・covariance・全身体cross。
- マーカー相対の手先位置のmean・covariance・全身体cross。
- 同じ凍結指令作用に対する解析的modal Jacobian。

カメラ相対位置の平均を `(x_k,y_k,z_k)` とし、較正済みpinholeの正規化画像座標を `b_k=(x_k/z_k,y_k/z_k)` とする。目標のマーカー相対位置を `d_MG`、手先平均を `m_H,k` とすれば、位置誤差は `e_k=m_H,k-d_MG` である。画像座標のcovarianceはpinholeの位置Jacobianで明示的に押し出す。

制御では、全成分の重み付き局所画像・位置二乗損失から、減衰付きの連立一次方程式で要求関節速度を求める。画像中央へ戻す速度と位置接近速度のgainを別に持ち、中央から大きく外れたときは位置接近の寄与を弱める。

これは完全なGaussian/Bingham重なり規準Jの勾配ではない。既存coreのmeanと全covarianceを使うが、短い局所モデル内ではモーメント補正とcovarianceを固定し、微分には解析的modal指令作用Jacobianを使う。真のmean・covarianceの完全な指令微分を計算したとは説明しない。位置・画像二乗損失の期待値にはcovarianceのtraceを記録するが、固定したcovariance項はこの局所速度の勾配を変えない。完全Jは遠距離phaseでは診断として評価し、採否規準へすり替えない。

既定の中央維持設定は `control.visual_approach` に置く。`center_gain=2.0`、`approach_gain=0.8`、`center_weight=25.0`、`center_tolerance=0.10`、`center_stop=0.15` である。後二者は画素数ではなく、焦点距離で正規化した画像座標のRMSである。各成分の平均bearingから `sqrt(sum_k w_k ||b_k||²)` を求め、0.10以下では位置接近のgateを1、0.15以上では0、その間では線形とする。マーカーの幅や輪郭の画面内保証を意味する閾値ではない。

遠距離phaseで一成分でもカメラ前方の深度が利用できない場合は、別の成分だけを選ぶ代わりに接近速度を要求せず、共通の加速度制約で減速する。全成分の照会・質量は保持する。推定画像中心を制御する方式であり、実画像が必ず中央へ収束することや、検出が途切れないことは未検証のまま主張しない。既存coreの方向別S²近似と根元lawへの再表現による逆照会の非閉包も残り、同文書のclosure記録と実画素を別に確認する。

## 近距離の判定と完全IK

各目標成分の同じマーカー相対手先lawから、位置誤差の二乗期待値を使う。

\[
D^2=\sum_k w_k\left(
\|m_{H,k}-d_{MG}\|^2+\operatorname{tr}S_{H,k}
\right).
\]

既定の `near_distance_m=0.08` に対して `D<=0.08 m` になったら、`near_latched=True` とし、既存の `MovingController.step` による完全なGaussian/Bingham規準Jの短い勾配更新へ切り替える。方位、共分散、正規化項、全目標mixtureのlog-sum-expを保持する。真の手先距離、平均位置だけの距離、最大重み一枝で切り替えない。

この判定は推定lawのRMS距離であり、物理的に0.08 m以内へ到達した証拠ではない。一度近距離へ切り替えたら、後の観測や距離変動で遠距離へ戻さない。同じcontrollerのlatchを保持する。近距離のIKは画像中央維持の局所損失を同時に追加する方式ではない。最終姿勢とマーカー視野の関係は、実際のカメラ取付け・マーカーの画素位置で別に検証する。

観測済みの目標記憶がまだない場合、またはvisual phaseを無効にした場合は、既存の完全IKを使う。no sensor／IMU単独を、画像中央維持を実行したケースや近距離判定済みのケースとして表示しない。

## 共通の動作制約と喪失時の優先順位

要求速度を共通倍率で通常速度上限へ収めた後、従来の速度・加速度・関節限界・停止距離制約へ渡す。既定の通常上限0.048 rad/s、加速度0.30 rad/s²を維持する。後段の制約で関節ごとの変位が変わり得るため、制約後も局所画像軌道が完全一致するとは保証しない。

マーカー喪失時は、`braking`→`settling`→有界な `searching` を優先する。記憶目標があることや、visual／IKどちらのphaseであることも、喪失中に接近し続ける理由にしない。新しいstep探索は外側・中心の各waypointで完全停止し0.35秒保持する。探索中も最後に採用した接近phaseの記録を保持するが、物理指令のActionは探索である。移動中に再検出しても減速停止し、停止後に取得した新しい全観測成分が融合され、同じ固定目標mixtureを持つ制御snapshotとして採用されてから、静止条件を満たして接近へ復帰する。停止前の観測やraw previewだけで復帰しない。

## scene表示と実行区間の対応

画像annotationは従来どおりマーカーの緑輪郭と姿勢だけを表示する。phase・数値診断を画像領域へ追加せず、画像サイズ、既存粒子・軸の描画設定、視点・レイアウト、RViz設定ファイルを変更しない。目標modal TFや候補分布の生成にも変更を加えない。

scene statusに、探索のAction行とは別のPhase行を表示する。

| `reaching_phase` | Phase表示 | 意味 |
|---|---|---|
| `visual_approach` | `APPROACH \| CENTER MARKER` | 遠距離の局所画像・位置規準 |
| `indet_ik` | `IK` | 既存の完全J。未観測・visual無効も含む |
| `waiting_for_control` | `WAITING FOR CONTROL` | 採用可能な制御planがまだない |

`indet_ik` は近距離判定なしでも使うため、表示に一律の `NEAR` は付けない。探索中は、例えば `Phase: APPROACH | CENTER MARKER` と `Action: MARKER SEARCH` を別に示す。Phaseが物理探索を上書きしたように見せない。

各物理trajectory rowには、そのframeへ至る実行区間で採用した制御diagnosticの `reaching_phase` と、次の取得元を `reaching_phase_source` として残す。

- `source_stamp`、`source_command`、`source_version`。
- `body_source_state_ids`、全 `goal_source_state_ids`。

最初の待機rowは `waiting_for_control`、sourceは `None` とする。通常のPhaseでは上の取得元を必須とする。公開するexecution metadataも同じrowの値を使い、MotionDisplayはPhaseとsourceの完全一致を送信前に検査する。候補snapshotと制御planの取得時刻は遅延により異なり得るため、取得元を現在frameの時刻へ付け替えない。

現在の物理frameの生成後に新しい非同期planが完成しても、そのframeのPhaseを書き換えない。次区間向けの `planned_reaching_phase`／`planned_reaching_phase_source` を別に公開する。過去frameの再生も、そのframeのrowから表示する。実探索軌跡のオレンジ色は従来の `command_interval_mode` だけで決め、接近Phaseを使って過去区間を塗り替えない。

## 追加前sourceの視野診断

初期画像中央維持を追加する根拠として、追加前の実画像と物理幾何を別に調べた。元記録は `/tmp/probtf_fov_analysis_20261017.json`、元sourceは `/tmp/probtf_fresh_goal_final_20261017`、凍結source一覧のSHA-256は `8a67a847513040a0942ad8fb26f3557cbac63a8fbdb212d8d8f25a205aada0c1` である。この測定は新しいvisual phaseの性能証拠ではない。

| 追加前ケース | 取得画像 | 検出なし | 瞬間物理幾何で輪郭が画面外 | 喪失直前の検出輪郭余白の中央値 | 有効選択画像での最小推定goal RMS |
|---|---:|---:|---:|---:|---:|
| marker | 898 | 488 | 486 | 8.0 px | 0.257 m |
| imu_marker | 901 | 644 | 648 | 5.17 px | 0.315 m |

この追加前sourceでは、当時の近距離閾値0.18 mへ達する前の範囲でマーカーが下側の画面端へ寄り、輪郭欠けが多数あった。検出なしでも物理幾何上の四隅がすべて画面内だった画像がmarkerで9枚、imu_markerで3枚あるため、全欠測を画面外だけで説明しない。瞬間幾何、ブラーを含む実画像、検出結果は異なる診断である。物理幾何は検証だけに使い、制御・推定へ渡さない。

## 検証

新sourceでは、同じ候補law・全mixture・凍結K作用を使うこと、遠距離の解析的局所Jacobianと近距離の完全Jの区別、近距離latch、共通動作制約、喪失時の探索優先を検査する。実画像の全保存取得、選択・未選択、画素中心、輪郭bounds、検出喪失・再検出とphaseを照合し、推定画像中心だけで実検出継続を証明しない。

表示試験は実MarkerArrayをserialize/deserializeし、Phaseと採用diagnosticの取得元、非同期で到着した次planとの区別、過去frameの再生、Actionの独立、既存geometry・描画サイズの不変性を確認する。RViz設定は起動前後にバイト比較する。

本書は実装の規準と検証条件を記録する。新しい実画像・実配信の結果は、対応する凍結sourceとともに別途記録する。中央維持、検出継続、近距離切替、到達成功を本書の記録だけで達成済みとはしない。
