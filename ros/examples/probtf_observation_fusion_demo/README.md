# 同じスイッチ到達動作で比較するセンサ別の姿勢推定

旧スイッチ例と同じ6軸ロボット、325 mmの棒、カメラ、パネル配置と重力ばねモデルを使う。
35.609秒の共通指令軌道を再生し、無観測・IMU・カメラ・併用の4条件だけを変えて姿勢候補を推定する。
実ロボットは不透明、同じ指令の剛体FKは半透明、推定した候補はcoreと同じ分布の原点・軸端点群として表示する。
物理動作と指令は4条件で同一で、センサは姿勢推定に使う。この比較ではセンサごとに制御器を変更しない。
要件は [REQUIREMENTS.md](REQUIREMENTS.md)、今回の動作・4動画への変更は [MOTION_REQUIREMENTS.md](MOTION_REQUIREMENTS.md)。

同じ画角・速度・42.5秒で録画した実RVizの動画：

- [観測なし](../../../recordings/probtf_observation_fusion_demo/switch_scene_20261016/no_sensors.mp4)
- [IMUのみ](../../../recordings/probtf_observation_fusion_demo/switch_scene_20261016/imu.mp4)
- [カメラのみ](../../../recordings/probtf_observation_fusion_demo/switch_scene_20261016/marker.mp4)
- [IMU＋カメラ](../../../recordings/probtf_observation_fusion_demo/switch_scene_20261016/imu_marker.mp4)

不透明の棒先端がボタンへ接近し、剛体前提での棒先端と離れる様子、途中リンク・手先の候補が観測によって変わる様子を比較できる。
以前の[16秒の静止分布切替動画](../../../recordings/probtf_observation_fusion_demo/imu_marker_rviz_20261016.mp4)は動作比較の動画ではない。

## 起動

    cd ~/catkin_ws
    catkin build probtf_observation_fusion_demo --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
    source devel/setup.bash
    roslaunch probtf_observation_fusion_demo no_sensors.launch
    roslaunch probtf_observation_fusion_demo imu.launch
    roslaunch probtf_observation_fusion_demo marker.launch
    roslaunch probtf_observation_fusion_demo imu_marker.launch

4つのコマンドはそれぞれ独立した起動例。同じROS masterでは1つずつ起動する。
既定でRVizも起動する。共通の物理フレームと推定を準備してから35.609秒の動作を再生し、最終状態を保持する。
検証機の準備時間は併用例で約37秒。準備中の処理時間によって物理時刻・画像の乱数列を変えない。
`rviz:=false` でノードだけ起動できる。`autostart:=false` は事前表示で待機する。

    rosservice call /observation_fusion/replay '{}'

再実行は同じ物理軌道と同じ観測入力を初期時刻から再生する。既に使った画像を事後へ重複して融合しない。
記録は `/tmp/probtf_observation_fusion_demo/` 以下に作り、`output_dir:=/absolute/new/path` でも指定できる。既存の記録先は上書きしない。

旧静止版は各launchに `moving:=false` を指定して実行できる。静止版のheadless検証：

    rosrun probtf_observation_fusion_demo observation_fusion_demo.py --headless \
      --observations /absolute/path/config/imu_marker.yaml --output /tmp/new-output

## 入力と情報境界

| 共有するもの | 内容 |
| --- | --- |
| `config/switch_press_scene.yaml` / `switch_press_commands.csv` | 旧例と同じ物理場面・剛性・質量・障害物・カメラ条件と、記録から取り出した指令列。実姿勢は毎回重力平衡を解いて再計算 |
| `config/moving_estimation.yaml` | 指令依存の局所候補族への参照、未知マーカー事前、既知撮像条件、同じ描画値 |
| 旧例の `independent_transform_laws.yaml` | そのまま使用する6リンクの並進Gaussian・Binghamと、明示した指令の右作用 |
| `model.run` | 事前生成、coreへの照会、アダプター呼出し、分布の持ち上げ、共通融合、事後全体の再表現 |
| `moving_scene` / `moving_inference` | 物理からは画素・IMU方向・指令・時刻だけを推定へ渡し、各時刻に同じcoreパイプラインを実行 |
| `motion_node` / `motion_display` / `display` | 同じ物理再生、参照ロボットの描画と、不変snapshotのMarkerArrayへの変換 |
| 各入口の差 | `observations` のみ。無情報は空、IMUは重力方向、markerは画像、併用は同じ2観測のリスト |

数値は例示用の明示入力であり、較正済みの身体分布を主張しない。
指令は実現角度ではない。各リンクは3軸のBingham候補を持つ。
初期事前だけが積分布。マーカー配置も有限の候補分布を持つ共有変量で、既知の根元姿勢へ置換しない。
工具・カメラ取付けとカメラ内部パラメータ・印刷寸法は明示した既知入力。

各0.5秒の観測時刻では、直接与えた指令依存候補族から共通事前を照会し、その時刻の観測で更新する。
更新した同時分布と交差共分散をそのまま全リンク・手先の照会へ渡す。未指定の時系列遷移モデルや剛性推定による予測は追加しない。
この動画は各固定時刻の推定比較であり、過去の全観測を蓄積する時系列フィルタや、センサによる閉ループ制御の比較ではない。

`MovingSwitchScene.sensor_inputs` は独立した物理場面から生成した画素・重力方向と、送信指令・時刻だけを返す。
撮像後は明示した推定用入力だけを取り出し、物理場面や描画設定をセンサアダプターの引数からも除外する。
正解角点・物理姿勢・剛性を推定器へ渡さない。IMUは設定に明示した重力方向観測で、磁気方位を持たない。
画像は既存AprilTag検出器と `PoseMixtureEstimator` に渡す。
後者の既定の広い並進事前を `translation_prior_variance=None` で無効にして、画像だけの局所尤度混合として扱う。
画像推定器がGauss–Newton Hessianから作る局所Laplace Gaussianを、公開されるSchurブロックから回収し、凍結した事前の姿勢座標へ移す。
これはセンサ側の画像尤度の局所近似であり、nativeな条件付きBingham変換法則の厳密モーメント計算ではない。
検出器の全成分と重みを保持し、同じカメラ相対マーカー事前との積からセンサ局所事後を作る。
未検出ならその観測の情報更新は恒等操作であり、検出したことにしない。

## 数式とcoreの実装

| 計算 | 実装 |
| --- | --- |
| 第4章のS²代理の一次平均・共分散 | `spherical_law.tangent.induced_vector_moments_tangent(..., order=1)`。既定 `order=2` は従来どおり |
| Binghamから共通右回転座標の共分散 | `first_order.bingham_rotation_coordinates`: `C_A=2(λI−NᵀAN)⁻¹` |
| 共同状態と経路・逆向き・方向・姿勢照会 | `JointTransformState`, `point_query`, `direction_query`, `pose_query`, `linear_query` |
| 全共分散の主計算 | `conditional_total_covariance`: 条件付き共分散からwithinとbetweenを計算し、その和を返す |
| センサ局所分布 | `fusion.freeze_query`, `condition_gaussian`。IMUは事前モードに固定した2次元接座標 |
| 身体への持ち上げ | `lift_sensor_posterior`: 残る条件付き共分散＋事後対象分布が作る共分散 |
| 共通事前を補正する融合 | `fuse_sensor_posteriors`, `gaussian_reference_product`: 分布比、混合ラベル、logCを含む重みの正規化 |
| 支持上の行列演算 | `support.covariance_support`。零分散へのjitterや分散floorなし |
| 同時事後の再表現 | `reexpress_state`: 全共分散の座標移送後、各Binghamを `A=−2NΩ⁻¹Nᵀ` として一括公開 |

`point_query` の平均は方向作用の収縮を残し、独立部分と相関による混合微分の寄与を分けて加える。
全体で共分散の一次に揃え、Σ²・再帰によるP₁P₂を含めない。
軸ごとの代理平均から回転行列を作らない。方位・並進・リンク間の交差共分散は同じ共同状態に残す。
出力は指定された高集中Gaussian代理で、元の一般共同変換法則の厳密解ではない。

単一回転の作用では、`L_v=−R⋆[v]×`, `Σ=L_v C_A L_vᵀ/||v||²` として、
`m=(1−tr(Σ)/2)R⋆v`, `S=L_v C_A L_vᵀ` を使う。
複数リンクでは、この独立部分の一次補正に、実際の合成・逆作用の混合微分と共同Pの縮約を加える。
共分散の主計算は、`B=P₂₁P₁₁⁺`, `P₂|₁=P₂₂−BP₁₁Bᵀ` から
`within=J₂P₂|₁J₂ᵀ`, `between=(J₁+J₂B)P₁₁(J₁+J₂B)ᵀ` の二項を求める。

各センサは同じ凍結代理 `z_s=c_s+H_s z` の `Q_s−` と、自身の観測だけから得た `Q_s+` を返す。
持ち上げは `K_s=PH_sᵀ(S_s−)⁺` を用い、
`μ_s=μ+K_s(m_s+−m_s−)`, `P_s=(P−K_sS_s−K_sᵀ)+K_sS_s+K_sᵀ`。
後段は `p+ ∝ ∏p_s/(p−)^(m−1)` を同じ支持で計算する。
混合成分ごとに共通事前ラベルを保持し、Gaussian積の積分定数Cを含む重みをlog-sum-expで正規化する。
無観測は同じ演算の単位元で、初期分布をそのまま返す。

局所座標は入力Binghamのモードと集中度から導く。別にSO(3)/SE(3) Gaussianを仮定して候補を差し替えない。
単純最大固有値と、一次APIが明示する集中適用域 `λ_max(C_A) ≤ 0.1 rad²` を検査する。
この上限は今回の近似を使用する明示的な適用範囲で、誤差保証ではない。超過は診断付き例外とし、分布を自動で狭めない。
同じ支持・版・時刻・指令の事前で全センサの代理を凍結する。
IMUは観測感度の階数が2で、yawを観測したと扱わない。下流の局所相対候補は独立初期事前ならIMU単独で変わらない。
カメラだけでは全身体自由度は特定できず、未知マーカー配置・身体間の相関と不定性が残る。

## 表示と記録

既存スイッチ押下デモのRVizから新しい `observation_fusion_motion.rviz` を作り、実／剛体RobotModelとパネル表示を保持した。
候補の根元は入力の `base_link`。RVizでは元の場面で明示されたworld→base配置 `[0.22,0,0]` を使い、元のworld視点をそのまま保つ。
元設定の視点、画像ドックとウィンドウ寸法を保持し、軸長0.10000000149、点サイズ0.009999999776、リンク180点・マーカー200点、seed29を引き継ぐ。

全リンク・手先・カメラ・マーカーの原点とXYZ軸端点は、推定で保持する同じ状態・同じ経路へのcore照会。
候補の主伝播・融合に乱数はなく、最終の解析Gaussianから描画点を生成する。物理側の画像・IMU雑音は共通の観測生成として別に管理する。
混合成分は個別に描き、広がりは変更しない。点数180/200はGaussian成分ごとの値で、各成分のalphaは0.85×その成分の重み。
紫のIMU方向は、観測したセンサ座標のベクトルに同じリンク候補を作用させたroot座標での点照会。
画像パネルには実際のカメラ相対局所事後 `Q_CM+` の成分重み・並進と方位の平均・共分散対角を注記する。
同じ各成分への線形周辺照会 `L=[I₂ 0]` で並進XYの2σ楕円も表示する。
図の軸はカメラ座標のm単位で、全成分の平均±2σから決めた範囲を明記する。身体の座標系や分布の広がりを変更する図ではない。
画像上の3D軸は画像検出器の尤度モードと明示している。
1本の動画には、その条件の事後分布だけを描く。事前・センサ別事後・融合後の全分布は記録へ保存する。
物理描画は20 Hz、推定は2 Hzで、画面に物理時刻と推定の観測時刻を両方表示する。
候補を最新の真値へ移し替えず、観測時点の候補のまま表示する。

動作版の `estimation.jsonl` に各観測時点のセンサ局所事前・事後、混合重み、全共同共分散、リンク間ブロック、持ち上げの二項、Bingham再表現、手先の二項を記録する。
`estimate_NNN.json` に実際に表示アダプターが読んだcore照会の平均・共分散・出自を保存し、
同じsnapshotとseedから再生成した点列と実際にserialize/deserializeしたMarkerArray全体をバイト比較する。
送信したメッセージそのものを `estimate_NNN.markers.bin` に残す。
`/observation_fusion/snapshot` は公開した段階の照会パラメータ・時刻・指令・版を送る。

`physical_trajectory.json` は参照物理モデルの姿勢・実先端・剛体先端・画像hash・IMU入力・離隔を記録する。
`motion_manifest.json` と4動画の `manifest.json` で、指令と物理軌道（画像・IMU入力を含む）のhashを照合する。

## 動作版の検証（2026-10-16）

依存を含む8パッケージのcatkin buildが成功。静止版・動作版・coreと検出器の回帰を含む90テストが成功した。
同じ35.609秒を20 Hzで714フレーム計算し、推定は73時刻。
4条件の各時刻で共通事前が一致し、実画像／IMUを使った292更新を確認した。
候補モードの幾何と、同じ指令による剛体FKを5時刻・9経路・原点とXYZ端点で照合し、最大差は `7.77e-16 m`。
区間の干渉検査は旧例の `motion_samples` の適応分割を再利用する。

共通の実先端は最終で目標まで5.683 mm、同じ指令の剛体FK先端は634.925 mm。
緑ランプは旧例と同じ「実先端が6 mm以内で、それまで禁止干渉なし」の到達判定で点灯する。
ボタンの押込み変形・接触力はモデル化していない。
各条件の手先位置共分散traceは終端で、無観測0.012115、IMU0.006899、カメラ0.001738、併用0.001377 m²。
これらは推定分布の広がりで、物理的な到達成功確率ではない。
直接指定した元の事前分布は物理たわみから較正したものではなく、センサを加えても全リンクの推定が真値へ一致するとは主張しない。

4本とも42.5秒、1840×1200、20 fps、850フレームで録画し、通しデコードと初期・途中・最終の画面を確認した。
各73更新、合計292更新の実受信MarkerArrayとsnapshotを、推定から保存したデータへ完全一致で照合した。
指令・全物理フレーム・画像hash・IMU入力は4条件で一致し、元と新しいRViz設定は録画終了後もバイト一致した。
4本の撮影とROS受信値の照合結果は、動画と同じディレクトリの `manifest.json` に保存している。
詳細な実受信メッセージと検証画面は `/tmp/probtf_motion_four_case_recording_20261016_run2/`。
録画を再現するにはROS環境をsourceして、未使用のmasterポートと新しい出力先を指定する：

    rosrun probtf_observation_fusion_demo record_four_cases.py \
      --output /absolute/new/video-directory \
      --runtime-output /tmp/new-motion-validation --port 11766

## 静止版の検証記録（動作版の結果とは別）

`catkin build probtf_observation_fusion_demo` は依存を含む5パッケージで成功。
指定次数の解析例、融合、センサ情報境界、表示同一性と既存kernel・検出器の回帰を合わせて **81テスト成功**。
対象は `ros/core/probtf_core/test/test_first_order_bingham_queries.py`、
`tests/probtf/{test_first_order_fusion,test_kernel_evaluation}.py`、
`tests/prob_artag_detector/{test_estimator,test_camera_and_detector,test_corner_uncertainty}.py`、本パッケージの `tests/`。
ROS環境をsourceした上で `/usr/bin/python3 -m pytest -q` にこれらのパスを渡す。
検証機ではpytestのみ `/tmp/probtf_switch_press_pytest` をPYTHONPATHへ追加して実行した。

| 入口 | 有効観測 | 事後成分数 | 事後重み | 共同Pの非対角リンクブロックのFrobenius norm |
| --- | ---: | ---: | --- | ---: |
| 無情報 | 0 | 1 | 1 | 0 |
| IMU | 1 | 1 | 1 | 0.0020509144 |
| marker | 1 | 2 | 0.4998305760 / 0.5001694240 | 0.0032532581 |
| IMU＋marker | 2 | 2 | 0.4992254734 / 0.5007745266 | 0.0033693934 |

normは先頭成分の並進・回転を含む混合単位の共同ブロックで、姿勢誤差や成功確率ではない。
初期の同normは0。観測後の相関を除くと、次回の手先照会の平均・共分散も変わることをテストした。

4入口を既定のRViz起動で実行し、全12段階について別subscriberで受けた実際のMarkerArrayと保存したwireデータをバイト照合した。
公開メタデータの平均・共分散・重み・state_id・版・指令・時刻も一致。
実際のImageの画素・encoding・frame・時刻を、推定に対応する保存画像と照合した。
各段階の画面で原点・軸端点群、センサ別→融合の進行、IMU方向とカメラ相対事後の図を確認した。
最後に入力境界を明示した版でも、4入口すべての記録・MarkerArray・画像が実ROS検証時とバイト一致することを再確認した。

証拠は `/tmp/probtf_observation_fusion_validation/final/` 以下の `live_validation.json`、
`boundary_validation.json` と `live_{no_sensors,imu,marker,imu_marker}/`。
各ディレクトリに共同事前・事後、センサ別局所推定、全covariance、実wireデータ、全段階の画面と画像を保存している。
既存スイッチ押下RVizと新しいRVizの両方が作業前の保存ファイルとバイト一致した。
元ファイルのSHA256は `3ca46dbb7e5eb1a7502bc584f933eab20c08a5f561ea77e03b0857f4aff92c58`。

確定TFツリーを配信しない構成なので、RVizのGlobal StatusにはTFデータなしの警告が出る。
各MarkerArrayは実際のroot `base_link` に直接表示し、表示のStatusはOK。

この節の81テストと全12段階の記録は、旧静止版についての検証である。
