# 観測で更新した座標変換候補から動くスイッチ到達デモ

旧スイッチ例と同じ6軸ロボット、325 mmの棒、カメラ、パネル配置と重力ばねモデルを使う。観測なし・IMU・カメラ・併用の4条件を、同じ初期指令・候補分布・指令選定処理で実行する。観測から更新した共同候補が、次の実送信指令へ反映される。

全条件で物理側のたわみを計算する。観測なしは同じ推定処理へ空の観測集合を渡すだけであり、物理モデルや事前分布を変更しない。不透明モデルは物理身体、半透明モデルは同じ現在指令の剛体FK、RGB点群は推定共同状態への原点・XYZ軸端点照会である。

**IMU単独の一部時刻では剛性更新が未解決。** 補助平衡の安定枝の特異点に近づき、滑らかな停留点と正定値Hessianを持つLaplace近似を得られない。該当時刻は剛性事後を `null`、情報ランクを未評価として記録し、保持した事前と失敗理由をRVizへ表示する。全条件・全時刻で剛性の事後更新が成立したとは報告しない。

**旧 `switch_scene_20261016/` の4動画は到達性能の比較として無効。** 過去の成功指令CSVを全条件で再生し、推定結果を指令へ反映していなかった。観測なしでも成功した原因はこの指令流用である。現行の実行経路はCSVを読み込まない。旧16秒静止動画も動作比較の成果物ではない。

元の推定仕様は [REQUIREMENTS.md](REQUIREMENTS.md)、動作要件は [MOTION_REQUIREMENTS.md](MOTION_REQUIREMENTS.md)、剛性推定の優先訂正は [REQUIREMENTS_STIFFNESS_CORRECTION.md](REQUIREMENTS_STIFFNESS_CORRECTION.md) に記録している。剛性推定を禁止した旧指示は撤回された。

剛性更新と更新不能の表示を含む実RViz動画：

- [観測なし](../../../recordings/probtf_observation_fusion_demo/stiffness_correction_20261016/no_sensors.mp4)
- [IMUのみ](../../../recordings/probtf_observation_fusion_demo/stiffness_correction_20261016/imu.mp4)
- [カメラのみ](../../../recordings/probtf_observation_fusion_demo/stiffness_correction_20261016/marker.mp4)
- [IMU＋カメラ](../../../recordings/probtf_observation_fusion_demo/stiffness_correction_20261016/imu_marker.mp4)

## 起動

    cd ~/catkin_ws
    catkin build probtf_observation_fusion_demo --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
    source devel/setup.bash
    roslaunch probtf_observation_fusion_demo no_sensors.launch
    roslaunch probtf_observation_fusion_demo imu.launch
    roslaunch probtf_observation_fusion_demo marker.launch
    roslaunch probtf_observation_fusion_demo imu_marker.launch

4つは独立した起動例。同じROS masterでは1つずつ起動する。既定でRVizも開く。30秒の観測駆動動作を準備計算してから20 fpsで同期再生し、終端を保持する。推定計算自体が実時間20 fpsという意味ではない。準備時間・計算fps・再生Hzを `motion_manifest.json` へ記録する。

`rviz:=false` でノードだけ起動する。`autostart:=false` は初期表示で待機する。

    rosservice call /observation_fusion/replay '{}'

replayは準備済みの同じ推定・指令・物理記録を再生し、画像を重複して融合しない。記録先は `/tmp/probtf_observation_fusion_demo/` 以下で、`output_dir:=/absolute/new/path` でも指定できる。既存先は上書きしない。旧静止版は `moving:=false` で起動できる。

## 共通入力と情報境界

| 対象 | 入力・責務 |
| --- | --- |
| `config/switch_press_scene.yaml` | 全条件共通の物理場面・関節ばね・質量・初期指令・障害物・カメラ条件。実姿勢は毎回重力平衡を解く |
| `config/moving_estimation.yaml` | 直接指定の局所候補族、未知マーカー事前、既知校正、マーカー→目標関係、速度・加速度上限 |
| 旧例の `independent_transform_laws.yaml` | 変更せず使う6リンクの並進Gaussian・Binghamと明示した指令右作用 |
| `moving_scene.SensorInputs` | 送信指令・時刻・実画像・IMU方向のみ。物理姿勢・剛性・実目標位置を含まない |
| `model.run` | 同じ事前からのcore照会、センサ局所推定、持ち上げ、分布比融合、同時再表現 |
| `moving_control.MovingController` | 現在の融合共同状態を受け、既知の指令右作用で試行候補を伝播し、同じ密度規準で選定 |
| 各入口の差 | 空の観測リスト、IMU重力方向、画像マーカー、両者のリスト |

数値は直接指定した例示入力で、較正済みの身体分布ではない。指令は実現角度ではない。初期事前だけが積分布で、観測後のリンク間・回転並進間・マーカーとの交差共分散を保持する。未知マーカー配置を真の根元姿勢へ置換しない。身体分布の構成にはURDF取付形状と関節限界を用いる。剛性更新の補助モデルだけは、既知入力と明示した同じURDFの質量・慣性・重力も用いる。シミュレータの真の剛性・実姿勢は渡さない。

各0.05秒時刻で、指令依存候補族から事前を照会し、その時刻の観測だけで固定時刻更新を行う。身体について未指定の時系列予測・過去事後の累積は追加しない。補助的な剛性分布は、下記の密度比で逐次更新する。未検出時はその画像観測の恒等更新とし、真の角点で補完しない。この設定では検出を失うと当該時刻の身体更新結果が事前へ戻る。各リンクの分布伝播は毎時刻実行されている。

## 補助的な剛性分布の更新

`stiffness_identification.StiffnessIdentifier` は、`model.run` が再表現前に保持した同じ凍結座標の身体事前・融合事後を受け取る。未知マーカーをGaussianの周辺化で積分し、6リンクの36次元全共分散と全混合重みを使う。

    ell(K) = log p_X^+(chart(Phi_u(K))) - log p_X^-(chart(Phi_u(K)))
    pi_K^+ ∝ pi_K^- exp(ell(K))

`Phi_u` は既存deflecompの `RobotArm`、`LinearSpringModel`、`EquilibriumSolver` と感度計算による補助予測。平衡の残差を同じポテンシャルのNewton法で精密化する。モデルが説明する有効剛性の推定であり、物性値の保証ではない。身体の主分布は置き換えず、現在の制御にもKを戻さない。将来の初期提案への利用は未実装。

複数の安定平衡があるため、各更新でK事前の最大重みモードを補助枝の基準点とし、既存ソルバの剛性continuationで基準平衡を求める。各K照会は必ずその同じ基準点からlog Kの直線をNewton法で追跡する。評価順によるwarm startは使わず、枝が不安定化すれば拒否する。この規約・基準点・能動関節制限を記録する。枝の基準点の選択でK事前や身体事後の混合をつぶすことはなく、目的関数は全成分の密度比を使う。最適化は直前K分布の全共分散で座標を白色化し、密度規準を変えずに数値条件を改善する。

事前は既存 `deflecomp_ros/config/estimator.yaml` のkp範囲と `initial_log_kp_state/std` からそのまま構成する。6関節共通でlog K平均3.1073040492110957、分散2.4138346135609177、K中央値22.360679774997894 N m/rad。範囲はこのGaussianの構成に用い、分布を切断しない。新たな荷重・モデル誤差分散・過程雑音・分散上限は加えない。

log Kで決定論的MAP・全Hessianの局所Laplace近似を行う。既存K成分と全身体成分を起点に探索し、見つかった異なる有効モードとLaplace重みを保持する。全域での単峰性やモード探索の完全性は保証しない。支持不一致・平衡不成立・非正定Hessian・最適化不成立は記録上の失敗とし、jitterや標本で置換しない。無観測ではKの全配列をビット単位で維持する。

失敗時は `posterior: null`、`retained_prior: <直前の事前分布>`、`prior_retained: true`、`update_applied: false`、`information_rank: null` と完全な理由を保存する。内部のK記憶は保持し、次時刻に同じ推定経路を再試行する。保持値を新しい事後分布と表示せず、HUDに `K UPDATE UNAVAILABLE; prior retained` と明示する。これは必要な剛性事後を計算できなかった時刻の報告であり、その時刻の更新成功ではない。

同じ剛性記録を `estimation.jsonl`、各 `estimate_NNN.json`、`/observation_fusion/stiffness`、snapshot、RVizの数値HUDへ渡す。全身体成分のsource ID・重み・版・観測ID・指令・時刻を照合し、不一致なら公開前に拒否する。HUDには有効K中央値、log K標準偏差、情報ランクと更新状態を表示する。カメラ画像には検出マーカーの緑枠と全姿勢候補のRGB軸だけを描く。

IMUはlink_3の重力方向だけでyawを観測しない。画像はAprilTag検出器と `PoseMixtureEstimator` へ渡す。`translation_prior_variance=None` とし、センサ既定の別の並進事前は加えない。画像のGauss–Newton Hessianから局所Laplace Gaussianを回収し、凍結した同じ事前の姿勢座標へ移す。これはセンサ尤度の局所近似である。画像検出器の全成分・重みを保持して同じ局所事前と積を取り、センサ事後を作る。

## 解析的伝播と融合

| 計算 | core実装 |
| --- | --- |
| 第4章のS²代理の一次平均・共分散 | `tangent.induced_vector_moments_tangent(..., order=1)`。既定のorder=2は変更しない |
| Bingham由来の共通回転共分散 | `first_order.bingham_rotation_coordinates`: `C_A=2(λI−NᵀAN)⁻¹` |
| 共同状態と経路照会 | `JointTransformState`, `point_query`, `direction_query`, `pose_query`, `linear_query` |
| 全共分散の二項 | `conditional_total_covariance`: 条件付きwithinとbetweenを実際に計算し、その和を返す |
| センサ局所分布 | `fusion.freeze_query`, `condition_gaussian` |
| 身体への持ち上げ | `lift_sensor_posterior`。残る条件付き共分散と、対象事後が作る共分散を保持 |
| 共通事前を補正した融合 | `fuse_sensor_posteriors`, `gaussian_reference_product`。混合ラベル・積分定数logC・全重みを保持 |
| 支持上の計算 | `support.covariance_support`。法線jitter・分散floorなし |
| 同時事後再表現 | `reexpress_state`。全交差ブロックを輸送し、各Binghamを一次共分散から再表現 |

点の平均はS²作用の収縮と混合微分による平均補正を残す。全体で共分散Pの一次とし、P²や再帰によるP₁P₂を含めない。方向ごとの平均から回転行列を作らない。局所変動座標は入力Binghamから導いたもので、独立に仮定したSO(3)/SE(3) Gaussianへ差し替えない。

一次APIの適用域は `λ_max(C_A) ≤ 0.1 rad²`。超過は例外にし、自動で分布を狭めない。これは近似の適用範囲であり誤差保証ではない。

`pose_query` は合成回転の平均補正に必要な `Σ_ab P_ab ∂²R/∂z_a∂z_b` を、共分散との縮約後の3×3行列として伝播する。旧実装の66×66×3×3テンソルを最後に縮約する計算と代数的に同一で、混合二階微分や交差共分散を省略していない。これは平均のP一次補正で、共分散のP²項を加えるものではない。

## 観測から次の指令まで

現在指令cで得た融合後の全成分を選定元とする。試行指令qには既知の局所右作用を適用する。各辺の回転座標は `R_delta^T` で変わるため、全共同共分散を `P(q)=G P(c) G^T` で輸送する。マーカー変量自体は固定し、辺ごとの独立周辺を再構成しない。

身体とマーカーは観測後に相関する。そのため、このデモは同じ共同候補から `marker^-1 * hand` をcoreで相対照会し、既知のマーカー→目標関係 `(d,r)` における密度を規準とする。最終の比較近似は、相対位置のGaussianと、Bingham由来の一次回転共分散から再表現した方位Binghamの積。姿勢照会の一次平均補正も含める。

相対照会までは位置・方位・リンク・マーカー間の全交差共分散を保持する。規準段階だけ位置と方位の積へ近似し、相対6×6共分散の位置・方位間ブロックを使わない。厳密な共同相対姿勢密度や物理的な成功確率ではない。表示用の直接端点照会は、この積へ置き換えず同じ元共同状態を参照する。

    log K(q) = LSE_k [ log w_k + log N(d; m_k(q), S_k(q))
                        + r^T A_k(q) r - log Z(A_k(q)) ]

位置のlog determinant、方位のlog normalizer、全混合重みを含める。独立でない身体・目標の共分散を単純加算しない。既知MGはDiracのままで、演出用の目標幅は加えない。旧スイッチ例の独立H/G積分布の重なりとは異なる、共有依存を保持した相対候補密度の規準である。

各周期で、この負の対数規準全体の勾配から1回の短い更新を行う。勾配は基準指令と6方向の微小変化で目的関数全体を評価する有限差分。各評価で共分散・Binghamパラメータ・正規化項も再計算する。位置・方位のモード感度と比較共分散から作る6×6行列は、勾配方向の前処理だけに使う。モードや平均だけの感度を目的関数の勾配とは呼ばない。必要な短いline searchで刻み幅を調整する。これは指令空間から乱数で候補を抽出する方法ではない。

速度・加速度・関節範囲を制約とし、ブラーを抑える追加指示により上限は全条件共通で0.12 rad/s、0.30 rad/s²。勾配・評価数・刻み幅・実行点の出自を記録する。ユーザーの軽量化指示により毎周期に反復最適化を収束させ切る構成ではなく、次の観測で再び更新する。センサ名で制御則を切り替えない。

勾配の基準点は、増分0に最も近い許容点である。勾配が小さい場合やline searchで更新が採用されない場合は、加速度制限内で減速・停止する。前周期の速度を永続保持して旋回する旧処理を修正した。加速度制約により増分0が許されない場合もあるため、現在の指令cから常に規準が改善するとは主張しない。line searchは記録された許容基準点からの改善を検査する。

現在表示のstate_idは現在指令cのもの。未来候補qは別のstate_idを持ち、親のstate_id・観測時刻・元指令・試行指令・照会パラメータを記録する。未来の予測分布を現在の物理姿勢へ重ねない。

## 表示・判定・記録

元のRViz設定ファイル全体と画像サイズを保持する。ユーザーの追加指示により描画だけ軸長・線幅・点サイズを0.75倍にし、軸長0.07500000111758709、点サイズ0.0074999998323619365、軸線幅0.00225とした。リンク180点・マーカー200点、seed29は保持。rootはbase_linkで、既知world→base `[0.22,0,0]` を一度だけ適用する。

全リンク・手先・カメラ・マーカーの原点とXYZ端点は、推定と指令選定元の同じ共同状態へのcore照会。物理真値への再中心化はしない。主伝播・融合・選定にはモンテカルロを使わず、点は照会後のGaussianから描画だけのために生成する。混合成分ごとのalphaは0.85×元の重み。各フレームの物理・観測・候補・版・指令を一致させる。

緑ランプは実際の棒先端が物理目標から6 mm以内に入り、それまで禁止干渉がない場合に点灯する。接触力やボタンの押込み変形はモデル化していない。真の到達距離・成功判定は指令選定や停止に戻さない。全ケースを同じ時間まで実行する。

- `estimation.jsonl`: 共同事前・センサ別局所事後・持ち上げ二項・融合重み・全共同共分散と、指令選定の出自・規準・次指令。
- `estimate_NNN.json` / `.markers.bin` / `.png`: 同じsnapshotのcore照会パラメータ、実際のROS wireメッセージ、対応する観測画像。点を再生成してwire全体を完全一致で検査。
- `sent_commands.json`: 実際に生成した指令列。次フレームの送信指令と選定結果を照合。
- `physical_trajectory.json`: 物理実姿勢、実先端・剛体先端、画像hash、IMU入力、離隔、実到達判定。
- `motion_manifest.json`: 共通設定hash、指令・物理記録hash、実結果、準備時間、計算fps、再生Hz。

4動画の録画・実ROS受信照合は、新しい保存先を指定して実行する。

    rosrun probtf_observation_fusion_demo record_four_cases.py \
      --output /absolute/new/video-directory \
      --runtime-output /tmp/new-motion-validation --port 11766

元・静止版・動作版のRVizファイルを事前保存し、終了後もバイト一致を検査する。

## 検証

現行コードの156テストと、依存を含む12パッケージのビルドが成功。身体の全交差共分散がK更新を変える解析解、全混合重み、未知マーカーの周辺化、退化支持、センサ順序、無観測のビット恒等性、身体分布の無変更、固定平衡枝の評価順不変性を検査した。公開する剛性値・HUD・身体照会・画像の同一性と、更新不能時に架空の事後や情報ランクを作らないことも検査する。大きい有限のlog K共分散を持つ正当な事後も保持する。テスト成功は、全時刻の剛性事後を計算できたことを意味しない。

現行の動画と全時刻の剛性分布・未成立時刻の記録は [録画manifest](../../../recordings/probtf_observation_fusion_demo/stiffness_correction_20261016/manifest.json) と [独立検証](../../../recordings/probtf_observation_fusion_demo/stiffness_correction_20261016/validation.json) に保存する。実時間の新規計算性能は [計測記録](validation/realtime_stiffness_20261016.json) に記録する。20 fpsの同期再生と区別する。

各条件は30秒・601時刻。共通の初期K中央値は全6関節22.36067977499789 N m/rad、log K標準偏差は1.5536520246055479。全混合成分のlog K平均・全共分散・重み、比較した身体の36次元事前・事後、観測と指令の出自を、実際に受信した同じ記録として圧縮保存した。

| 条件 | K更新成立 | 恒等更新 | 更新不能 | 実先端の終端誤差 | 全時刻の分布 |
| --- | ---: | ---: | ---: | ---: | --- |
| 観測なし | 0 | 601 | 0 | 539.462 mm | [記録](../../../recordings/probtf_observation_fusion_demo/stiffness_correction_20261016/no_sensors_stiffness.jsonl.gz) |
| IMU | 473 | 0 | 128 | 74.205 mm | [記録](../../../recordings/probtf_observation_fusion_demo/stiffness_correction_20261016/imu_stiffness.jsonl.gz) |
| マーカー | 40 | 561 | 0 | 541.371 mm | [記録](../../../recordings/probtf_observation_fusion_demo/stiffness_correction_20261016/marker_stiffness.jsonl.gz) |
| 併用 | 601 | 0 | 0 | 104.794 mm | [記録](../../../recordings/probtf_observation_fusion_demo/stiffness_correction_20261016/imu_marker_stiffness.jsonl.gz) |

終端の有効K中央値を関節1〜6の順に示す。分布の全共分散・混合重みは上記の記録を参照する。

| 条件 | 終端のK中央値 [N m/rad] |
| --- | --- |
| 観測なし | [22.36, 22.36, 22.36, 22.36, 22.36, 22.36] |
| IMU | [22.36, 104.9, 150.9, 6.718e6, 5.120, 23.96] |
| マーカー | [22.36, 4003, 42.03, 7.630, 2007, 5.035] |
| 併用 | [22.36, 1798, 47.52, 8.516, 68.18, 3.468] |

これらは既知補助モデルと局所Gaussian代理の下での有効剛性であり、真の物性値の同定成功ではない。IMU終端の局所情報ランクは4/6で、関節4のlog K標準偏差は2.689。IMUの更新不能は127回の非停留・1回の平衡残差未達で、失敗時刻と理由を全件記録した。安定枝の特異点付近をまたぐ非Gaussianの剛性事後の扱いは未解決である。全条件とも物理的には未到達・禁止干渉なしだった。併用が単独条件より常に良いという結果ではない。

実時間計測は各40回の新規計算・実ROS公開で、観測なし6.57 Hz、IMU4.76 Hz、マーカー2.43 Hz、併用2.27 Hz。剛性更新の平均は順に0.823、41.497、89.928、106.246 ms。短い計測区間の更新失敗は0だが、30秒全体の適用条件を保証しない。同じCPUへの固定・BLAS/OMP各1スレッドで測り、他の準備計算が並走した条件を記録している。

### 剛性追加前の検証記録

core解析例、融合、情報境界、依存保持、指令への反映、推定と実表示の同一性、既存kernel・画像検出器の回帰を検査する。現在の実行結果と録画の検証値は、検証後に動画manifestとともに記録する。

15テストで旧dense微分計算を独立に保持し、相関した66次元共同状態・逆向き経路・同じ変量の反復を含む全照会結果を比較した。計算式・近似次数・分布を変更せず、姿勢照会1回は約29 msから約1.8 msへ短縮した。最終の勾配更新・動画を含む検証値は、実行後のmanifestへ記録する。

剛性追加前の勾配更新を含む119テストが成功。独立に組み立てたGaussian密度・Bingham正規化定数・全混合の式でも、記録した目的関数とその勾配を検算した。初回は全ケース8評価・勾配1回で、観測なし／IMUが約42 ms、カメラ／併用が約86–87 ms。画像推定・描画・保存を含む周期全体の時間ではない。

剛性追加前の4動画は24.5秒、1840×1200、20 fps。全1,444フレームの推定元・実受信MarkerArray・snapshot・画像が一致し、欠落なし。画像の実受信は19.978〜20.020 fps。全1,440指令更新で完全規準の勾配を1回計算し、規準評価は平均7.98〜9.76回だった。元・静止版・動作版のRViz設定は終了後もバイト一致した。4本の通しデコードと、各初期・途中・終端の計12画面も確認した。

| 条件 | 実物の終端誤差 | 有効マーカー観測 | 結果 |
| --- | ---: | ---: | --- |
| 観測なし | 539.462 mm | 使用しない | 未到達・干渉なし |
| IMU | 79.152 mm | 使用しない | 未到達・干渉なし |
| カメラ | 539.464 mm | 16 / 361 | 未到達・干渉なし |
| 併用 | 92.236 mm | 84 / 361 | 未到達・干渉なし |

観測なしでは実先端が目標より525 mm下へ外れた。カメラ単独では、旧設定の0.24秒露光によるモーションブラーで検出を失い、その後は事前からの恒等更新になる。観測を蓄積しない現設定の制約であり、成功や融合の優越を示す結果ではない。全ケースを通した準備時間は各51〜71秒、計算スループットは5.1〜7.1 fpsで、20 fpsの再生とは区別する。

動画の属性と実ROS照合は [manifest.json](../../../recordings/probtf_observation_fusion_demo/closed_loop_gradient_20261016/manifest.json)、独立監査・完全規準の勾配検算・実行統計は [validation.json](../../../recordings/probtf_observation_fusion_demo/closed_loop_gradient_20261016/validation.json)。全実受信メッセージと画像は `/tmp/probtf_closed_loop_gradient_recording_20261016/` に保存している。

旧静止版の81テスト・実ROSの全12段階の照合記録は `/tmp/probtf_observation_fusion_validation/final/`。旧固定成功軌道版の90テストと4動画は、今回の閉ループ実行の検証結果として扱わない。
