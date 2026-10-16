# 観測で身体候補を保持・更新するスイッチ到達デモ

直接与えたリンク間座標変換候補と未知マーカー配置の同時分布を主状態にする。初期化は実験開始時だけ行い、その後の全事後成分・重み・交差共分散を次の指令へ伝播する。途中リンクのIMUと手先カメラは、同じ予測事前からそれぞれ分布を推定し、共通事前を補正した同じcore演算で後段融合する。

**旧版の「毎周期、独立な初期事前からやり直す」処理は撤回した。** マーカーを見失っても身体・マーカーを捨てない。補助的な剛性記憶から制御用Kを選び、同じ身体事後に対する指令作用を平衡角差で補正する。その同じ補正分布の完全な規準で、短い指令を一回更新する。

要件の優先順は [逐次更新の訂正](REQUIREMENTS_SEQUENTIAL_UPDATE.md)、[剛性推定の訂正](REQUIREMENTS_STIFFNESS_CORRECTION.md)、[動作要件](MOTION_REQUIREMENTS.md)、[元の推定仕様](REQUIREMENTS.md)。YAGNI等を理由に要求済みの保持・更新・実行を省略しない。

## 起動

    cd ~/catkin_ws
    catkin build probtf_observation_fusion_demo --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
    source devel/setup.bash
    roslaunch probtf_observation_fusion_demo no_sensors.launch
    roslaunch probtf_observation_fusion_demo imu.launch
    roslaunch probtf_observation_fusion_demo marker.launch
    roslaunch probtf_observation_fusion_demo imu_marker.launch

4つは独立した実験で、同じmasterでは一つずつ起動する。違うのは観測リストだけ。身体・ばね重力物理世界・初期指令・初期候補・指令作用・制御・表示条件を共通にする。

手先フランジ `tool0` にIMUを置く追加実験は、次で起動する。

    roslaunch probtf_observation_fusion_demo imu_hand.launch

IMUの取付先だけをlink_3からtool0へ変え、同じ物理モデル・観測精度・初期候補・指令選定を使う。重力方向の観測経路が6関節すべてを含む。表示と推定は同じ共同事後を参照し、replay時も手先取付を保持する。元の4条件は変更しない。

既定の `execution_mode:=live` は、その場で物理フレームから観測を生成し、推定・公開してから次指令を選ぶ。全期間を準備計算してから再生する経路ではない。シミュレーション刻み0.05秒は実計算20 Hzの保証ではなく、遅い計算では壁時計上の動作も遅くなる。別時刻の候補と物理姿勢を混ぜない。

`rviz:=false` はノードのみ、`autostart:=false` は初期フレームで待機する指定。

`max_updates:=6` は、元の刻み・終了時刻を変えず、新観測を6回処理した時点で明示的に停止する短期試行である。既定値0は回数無制限。上限で止まった場合はRVizと公開metadataに `PAUSED` / `experiment incomplete` を表示し、元の30秒実験を完走したとは記録しない。画像混合を全組合せ保持するケースの短期検証にも使う。

    rosservice call /observation_fusion/replay '{}'

liveのreplayは同じ初期状態からの新実験として再観測・再計算する。既に取り込んだ画像を現在の事後へ重複融合しない。明示的な `execution_mode:=prepared_playback` だけが、同じ推定経路で準備した記録の再生である。旧3軸の静止した一回更新は `moving:=false` で区別する。保存先は `output_dir:=/absolute/new/directory` で指定し、既存先を上書きしない。

## 身体の逐次推論

初期分布は `independent_transform_laws.yaml` の直接指定Gaussian・Binghamと未知マーカーの分布から作る。例示用の入力であり、真のたわみから作った較正値ではない。既知の棒・カメラ取付はDirac変換である。

現在指令cから次指令qへ、指定済みの `right_joint_action` を同じ候補に作用させる。2026年10月17日の追加指示により、観測でKを得た周期は、回転量を `q-c` から `theta_eq(q,K)-theta_eq(c,K)` へ変更する。Kと平衡枝は一周期固定する。右回転座標の輸送をGと書けば、全共同共分散も `P(q)=G P(c) G^T` とする。マーカーは同じ時間共有変量として保持し、身体との相関も残す。新たな過程雑音・分散回復・剛性モデルによる身体再生成は加えない。

    初期の直接指定候補族（一度だけ）
        ↓
    前回の全事後を次指令へ伝播
        ↓
    同じ事前から各センサが個別に推定
        ↓
    共通事前を補正して後段融合
        ├─ 同じ全事後をcore照会し、そのまま公開・表示
        ├─ 新観測の身体密度比を剛性記憶へ追加
        └─ 次の短い指令を選定 → 実行 → 次の観測

IMUはlink_3の重力方向に対する2座標の局所事後を返す。yawや並進を観測したことにはしない。カメラは実画像のAprilTag検出・姿勢Hessianから全画像候補を返す。未知マーカーの真の根元姿勢を代入しない。画像だけの検出・Hessian計算は観測ごとに一度行い、全事前成分で共有する。

各親成分hについて、観測evidenceと画像枝kの条件付き質量から、全体のセンサ事後質量 `alpha[s,h,k]` を計算する。既存coreの `fuse_labelled_sensor_posteriors` が、同じhを基準に事前重複を除いて融合する。異なる親chartを同じ座標とみなしてGaussianを足さない。

全交差共分散・成分対応・log weightを保存する。通常のfloat weightがunderflowしても、有限のlog weightを持つ正の成分を削除しない。空観測や画像未検出は保持した同時分布への恒等更新である。

**非圧縮経路は過去の画像候補の全組合せを保持する。** 各画像が2候補なら成分数は2のべきで増える。小さい枝の削除、無断のGaussian縮約、検出器での単一解選定は行わない。解析的な成分縮約を導入する場合も、追加近似への明示的な許可が必要である。

## 第4・7章の近似とcore

BinghamをS²へ押し出したベクトル作用の高集中一次代理を使う。共分散は全体O(P)までとし、平均の収縮と相関由来の平均補正を残す。モードFKへ最後に分散を付ける処理ではない。

- `point_query / direction_query / pose_query`: 同じ共同候補の経路作用と全交差ブロックを保持する解析照会。
- `conditional_total_covariance / lift_sensor_posterior`: 条件付き共分散の平均と、条件付き平均の共分散を実際に計算する。
- `gaussian_reference_product / fuse_labelled_sensor_posteriors`: 共通事前を補正した成分対応付き融合。
- `reexpress_state`: 各事後成分の同時再表現。全交差ブロックを保持する。
- `covariance_support`: 支持上で計算する。exact-zeroの既知固定座標は固有分解から分離し、Dirac取付へ数値丸めの偽分散を入れない。jitter・固有値floorを使わない。

一次APIの適用域は `lambda_max(C_A) <= 0.1 rad²`。適用域外を分布の縮小や別バックエンドで隠さない。これは近似の適用域であり全条件での誤差保証ではない。

## 剛性の補助記憶と指令提案

同じ新観測の身体事前・事後を各親の凍結chartと全混合質量付きで受け取る。未知マーカーを周辺化し、6リンクの36次元全共分散を使う。

    ell_t(K) = log p_X,t+(Phi_u,t(K)) - log p_X,t-(Phi_u,t(K))
    pi_K,t+(x) = initial_logK_density(x) * exp(sum_r ell_r(exp(x))) / Z_t

`Phi_u` は既存deflecompのばね・重力平衡とFK。異なる親chart間では共通測度への体積係数を含める。これも指定された高集中局所Gaussian代理の計算であり、厳密な大域SO(3)密度への置換とは主張しない。

主K記憶は、既存の明示的初期log K Gaussianと各新観測の密度比因子の積である。局所Laplaceの不成立を理由に因子を捨てない。全体の正規化定数Zと全体モーメントは未計算として明示する。有限のZを持つ条件と上下界、chartの近似位置、密度照会不能点を記録し、評価不能を零尤度へ置換しない。

提案用の点は、既存事前から定めた決定論的起点を累積密度で比較し、密度scoreに基づく短い数値stepも使う。これらの有限点を剛性分布・その積分・中央値やMAPの代用にしない。得たKによる平衡たわみ補償方向を初期指令候補にする。

制御側は同じdensity ID・身体source ID・時刻・指令を照合する。`StiffnessCommandMap` が、現在の身体事後に平衡角差を右作用する一周期の指令依存分布族を定める。現在指令では厳密な恒等操作であり、KモデルのFKへ再中心化しない。前処理の感度もKKT式から `dtheta/dq=H^{-1}diag(K)` とする。補正された完全な規準に対して、速度・加速度・関節限界内で同じ勾配更新を一回行う。選定に使った補正を次周期の事前にも同じsource IDのまま適用する。無観測では観測更新が身体・剛性とも厳密な恒等操作となる。まだ新観測のK因子がない場合は、未取得のKによる補正を実施したとは記録しない。

## 指令の規準と表示

同じ共同候補から `marker^-1 * hand` の相対分布を伝播し、既知のマーカー→目標関係における密度を規準とする。比較段階だけ位置Gaussianと方位Binghamの積へ近似する。位置log determinant、方位log normalizer、全成分log-sum-expを含める。相対照会までは全交差ブロックを保持する。

毎周期、完全な負の対数規準の勾配から一回の短い更新とline searchを行う。平均・モードの感度だけの代用や少数点による分布積分を行わない。勾配の有限差分は指令に対する数値微分で、身体候補のサンプリングではない。

不透明ロボットは物理側の重力たわみ姿勢、半透明モデルは現在指令の剛体FK。RGB点は、同じ推定共同状態の根元→各リンク原点・軸端点へのGaussian照会である。局所変換周辺を配置したものや、物理姿勢へ再中心化したものではない。

描画点もGaussian分位点への決定論的照会で生成する。推定・伝播・融合・剛性記憶・描画にMonte Carloを使わない。乱数は物理側の模擬観測生成だけにある。点径・点数・軸長・画像サイズ・RViz設定を保持する。

実際のsnapshot・MarkerArray・画像・剛性密度と提案・source ID・版・時刻・指令を照合する。失敗時は失敗tickと最後に公開したtickを分けて停止理由を表示し、旧表示のまま動作中を装わない。

## 検証と記録

### 2026年10月17日のK補正による1ステップ更新

推定Kによる平衡角差を、IKの全試行・感度・次周期の事前伝播に適用した。同じ固定K・身体source・全共同共分散を使うことを、実受信データを含めて独立に照合した。199テストが通過した。動画は追加していない。

| 条件 | 実行範囲 | 実更新 | 実先端の誤差 | 物理結果 |
| --- | --- | ---: | ---: | --- |
| 手先IMU | 30秒・601周期 | 4.055 Hz | 終端27.586 mm | 未到達・途中に禁止干渉あり、最小距離−0.475 mm |
| マーカー | 0.25秒・6周期 | 0.252 Hz | 停止時365.304 mm | 短時間試行で停止・未到達 |
| IMU＋マーカー | 0.25秒・6周期 | 0.223 Hz | 停止時364.099 mm | 短時間試行で停止・未到達 |

マーカー2条件は全混合成分を2→4→8→16→32→64と保持した。長時間実行の完了結果ではない。過去の混合を減らす追加近似への許可が得られるまでは、勝手な枝削除やモーメント縮約を導入しない。停止時はRVizに未完了と表示し、6周期分の物理・指令ログも保存する。手先IMUのセンサ図形が旧link_3に残る表示不具合も修正し、実受信したtool0のTFとの一致を確認した。RViz設定26ファイルはバイト一致している。

詳細は [K補正と次事前の検証](validation/K_corrected_control_20261017.json) と [実受信表示の検証](validation/K_corrected_display_20261017.json) に記録した。K補正や分布の同一性の検証通過は、物理到達成功の主張ではない。

単発更新の一致だけでは逐次推論の検証にならない。独立観測を繰り返す解析的batch解、全成分のevidence、次周期に残る全交差共分散、欠測時の保持、正の微小log質量、固定Dirac制約、同じ公開データへの対応を検査する。

    rosrun probtf_observation_fusion_demo benchmark_realtime.py --publish --frames 40

専用ROS masterで実計算Hzと実受信Hzを測定する。全混合保持の長時間計算量を、短いsmokeのHzから達成扱いしない。

    rosrun probtf_observation_fusion_demo record_four_cases.py \
      --output /absolute/new/video-directory \
      --runtime-output /tmp/new-sequential-validation --port 11766

live録画は同じシミュレーション時間の完走まで待つ。動画fpsと新観測の実受信Hzを別記録する。成分増加により完走できていない実行を、完走動画として公開しない。

`recordings/probtf_observation_fusion_demo/stiffness_correction_20261016/` と旧版の性能値は、身体を毎周期初期化していた履歴であり、この逐次保持版の受入証拠ではない。

緑ランプは実物理先端が目標から6 mm以内で、それまで禁止干渉がない場合だけ点灯する。実到達判定・真の誤差・真の剛性は推定や指令選定へ入力しない。

### 2026年10月16日の逐次保持版の実測

186テストと、依存を含む12パッケージのビルドが成功した。繰り返す観測の平均・共分散・全成分重みは、独立に計算したGaussianのbatch情報形式と一致した。画像候補を含む4ケースの短い実処理では、カメラの全枝が2→4→8と保持され、過去snapshotとK記録は後続処理でも変化しなかった。

同じ設定の実live録画は以下の2条件で30シミュレーション秒・601周期を完走した。カメラを含む長時間実行は全組合せの指数的増加が残っており、4条件すべての長時間比較を完了したとは報告しない。

| 条件 | 実画像更新 | 動画の壁時計長 | 実先端の終端誤差 | 物理結果 |
| --- | ---: | ---: | ---: | --- |
| 観測なし | 6.198 Hz | 99.50 s | 539.328 mm | 未到達・禁止干渉なし |
| IMU | 4.130 Hz | 147.95 s | 415.232 mm | 未到達・禁止干渉なし |

[観測なし動画](../../../recordings/probtf_observation_fusion_demo/sequential_update_20261016/no_sensors.mp4)、[IMU動画](../../../recordings/probtf_observation_fusion_demo/sequential_update_20261016/imu.mp4)、[実ROS照合・録画manifest](../../../recordings/probtf_observation_fusion_demo/sequential_update_20261016/manifest.json)。動画の符号化は20 fpsだが、新規観測15 fpsの目標は未達である。到達性能が旧版より向上したという結果でもない。

IMUでは全601件の剛性密度比因子を保持し、600回の指令選定のうち270回でK記憶からの提案を完全な身体規準により初期点へ採用した。無観測では全601回が恒等更新で、K因子は0のまま。IMUのlink_1/2/3局所回転共分散traceは、終端で初回事後の約0.252/0.145/0.129倍となった。毎周期初期化する旧版のように、同じ一回分の更新を繰り返す結果ではない。

実際の身体推定の全パラメータ、公開snapshot、公開K記録、画像・wireのhash、指令・物理軌道を[圧縮記録とhash一覧](../../../recordings/probtf_observation_fusion_demo/sequential_update_20261016/archives.json)へ保存した。Kの全体正規化定数・モーメントは、この実測でも未計算である。
