# Bingham変換候補の一次伝播とセンサ別分布の融合

3リンクの局所変換候補と未知マーカー配置の候補を直接与え、IMU・手先カメラのそれぞれの推定分布を共通事前で補正して融合する。
全リンクの事後分布とリンク間の依存を保持し、同じcore snapshotへの原点・軸端点照会をRVizで表示する。
実装の契約は [REQUIREMENTS.md](REQUIREMENTS.md) に保存した指示書全文。

[実行動画（MP4・16秒）](../../../recordings/probtf_observation_fusion_demo/imu_marker_rviz_20261016.mp4)では、
実際のRVizで事前→IMU単独推定→カメラ単独推定→融合後の分布更新を確認できる。

## 起動

    cd ~/catkin_ws
    catkin build probtf_observation_fusion_demo --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
    source devel/setup.bash
    roslaunch probtf_observation_fusion_demo no_sensors.launch
    roslaunch probtf_observation_fusion_demo imu.launch
    roslaunch probtf_observation_fusion_demo marker.launch
    roslaunch probtf_observation_fusion_demo imu_marker.launch

4つのコマンドはそれぞれ独立した起動例。同じROS masterでは1つずつ起動する。
既定でRVizも起動し、3秒ずつ事前→センサ別単独推定→融合後の順に表示して最終状態を保持する。
`rviz:=false` でノードだけ起動できる。`autostart:=false` は事前表示で待機する。

    rosservice call /observation_fusion/replay '{}'

再実行は同じ固定時刻の更新を共通事前から再表示する。既に使った画像を事後へ重複して融合しない。
記録は `/tmp/probtf_observation_fusion_demo/` 以下に作り、`output_dir:=/absolute/new/path` でも指定できる。既存の記録先は上書きしない。

ROS masterなしで同じ計算とROSメッセージのserialize/deserialize検証を実行する場合：

    rosrun probtf_observation_fusion_demo observation_fusion_demo.py --headless \
      --observations /absolute/path/config/imu_marker.yaml --output /tmp/new-output

## 入力と情報境界

| 共有するもの | 内容 |
| --- | --- |
| `config/common.yaml` | 直接指定した並進GaussianとBingham、初期独立性、指令、観測時刻、未知マーカー事前、既知取付け、撮像条件、物理場面、描画値 |
| `model.run` | 事前生成、coreへの照会、アダプター呼出し、分布の持ち上げ、共通融合、事後全体の再表現 |
| `node` / `display` | 同じ進行と不変snapshotのMarkerArrayへの変換 |
| 各入口の差 | `observations` のみ。無情報は空、IMUは重力方向、markerは画像、併用は同じ2観測のリスト |

数値は例示用の明示入力であり、較正済みの身体分布を主張しない。
指令は固定された更新単位のメタデータで、実現角度ではない。各リンクは3軸のBingham候補を持つ。
初期事前だけが積分布。マーカー配置も有限の候補分布を持つ共有変量で、既知の根元姿勢へ置換しない。
工具・カメラ取付けとカメラ内部パラメータ・印刷寸法は明示した既知入力。

`scene.render_image` は独立した物理場面から画素だけを返す。
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

既存スイッチ押下デモのRVizをコピーし、分布表示だけを計算済みMarkerArrayへ変更した。
根元は入力の `base_link`。旧worldへ合わせる変換は追加しない。
元設定の視点、画像ドックとウィンドウ寸法を保持し、軸長0.10000000149、点サイズ0.009999999776、リンク180点・マーカー200点、seed29を引き継ぐ。

全リンク・手先・カメラ・マーカーの原点とXYZ軸端点は、推定で保持する同じ状態・同じ経路へのcore照会。
乱数は最終の解析Gaussianから描画点を生成する箇所だけにある。
混合成分は個別に描き、広がりは変更しない。点数180/200はGaussian成分ごとの値で、各成分のalphaは0.85×その成分の重み。
紫のIMU方向は、観測したセンサ座標のベクトルに同じリンク候補を作用させたroot座標での点照会。
画像パネルには実際のカメラ相対局所事後 `Q_CM+` の成分重み・並進と方位の平均・共分散対角を注記する。
同じ各成分への線形周辺照会 `L=[I₂ 0]` で並進XYの2σ楕円も表示する。
図の軸はカメラ座標のm単位で、全成分の平均±2σから決めた範囲を明記する。身体の座標系や分布の広がりを変更する図ではない。
画像上の3D軸は画像検出器の尤度モードと明示している。
センサ単独の段階と融合後を同じ表示経路で切り替え、画面に版・時刻・指令・有効観測数とwithin/betweenのtraceを示す。

`estimation.json` にセンサ局所事前・事後、混合重み、全共同共分散、リンク間ブロック、持ち上げの二項、Bingham再表現、手先の二項を記録する。
`stage_XX.json` に実際に表示アダプターが読んだcore照会の平均・共分散・出自を保存し、
同じsnapshotとseedから再生成した点列と実際にserialize/deserializeしたMarkerArray全体をバイト比較する。
送信したメッセージそのものを `stage_XX.markers.bin` に残す。
`/observation_fusion/snapshot` は公開した段階の照会パラメータ・時刻・指令・版を送る。

## 検証結果（2026-10-16）

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

検証で保持した併用例のGUIは専用master `http://localhost:11765/` に接続している。
この画面で再実行する場合：

    ROS_MASTER_URI=http://localhost:11765/ rosservice call /observation_fusion/replay '{}'
