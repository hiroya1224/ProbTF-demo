# 3×3マーカーボードからの中央観測

2026年10月17日の追加要件を記録する。各マーカーの画像単独mixtureを保持し、既知の取付関係でボード中央へ写して融合する。今回の中央観測だけは適用域内で単一Gaussian×単一Binghamへ近似でき、広い場合は狭い中央枝の全mixtureを保持する。その同じlawを身体推定・目標構成・IK・表示へ渡す。

ボード導入後、K代表追従・探索条件の訂正前sourceの実配信検証は [full_k_ik_20261017.json](validation/full_k_ik_20261017.json) に記録した。当該sourceの457テストと、両ケース合計422周期の推定・制御・実配信の分布同一性を確認した。各ケースの60秒試行は物理到達に成功していない。ボード融合・配信の検証と、到達動作の成否を区別する。後続の訂正・source別結果は [K_ACTION_AUDIT.md](K_ACTION_AUDIT.md) と [MARKER_VISIBILITY.md](MARKER_VISIBILITY.md) を参照する。

両訂正を含むsource `7403ad...` では464試験と、全422周期の実配信・18,568点照会・画像内ボード融合124回を再照合した。両60秒試行は物理到達した。これは後続のK代表追従と探索条件分離を含む結果であり、ボード変更単独の効果や絶対姿勢推定の正確さを示す比較ではない。[新sourceの検証記録](validation/visible_marker_hold_20261017.json) に詳細を保存した。

## 最新例外の範囲

従来は画像観測のnative全成分を、そのまま比較用目標の全mixtureへ残していた。最新のボード指示では、次を区別する。

1. 各markerのraw画像観測は、全native成分・weight・log weightを保持する。
2. 既知の剛体配置で中央へ写し、複数の画像観測を融合する。
3. この派生した中央観測に限り、既存coreの高集中域内で単一Gaussian×単一Binghamの近似を許可する。広い中間近似は延期し、最終でも広い場合は全ての狭い派生枝のmixtureを保持する。
4. 単峰化できた中央観測と既知の中央→スイッチ関係から構成する今回の `p_goal` は1成分でよい。中央がmixtureなら `p_goal` も全成分・質量・log weightを保持する。

これは全raw観測の最大重み一枝だけを選ぶことでも、一般の観測・目標mixtureを単峰化することでもない。ボードを使わない従来の単一markerのnative mixtureには、従来の全成分保持を適用する。

本書と `AGENTS.md` 冒頭の追加訂正は、この中央観測と中央由来目標の単峰近似の範囲に限り、`REQUIREMENTS.md`、`REQUIREMENTS_SEQUENTIAL_UPDATE.md`、`FRESH_OBSERVATION_GOALS.md`、`VISUAL_APPROACH.md` の一般的なmixture保持規定に優先する。身体・未知地図の全共同内部事後、剛性履歴、根元固定目標、実指令ledger、分布同一性、coreとRViz設定の保護は維持する。

## 既知の配置と物理描画

ボード中央を `M_0`、各markerを `M_i` とする。中央IDは21、3×3の配置は上から次のとおりである。

| 行 | 左 | 中央 | 右 |
|---|---:|---:|---:|
| 上 | 17 | 18 | 19 |
| 中 | 20 | 21 | 22 |
| 下 | 23 | 24 | 25 |

各markerの一辺は0.072 m、中心間隔は0.09 mとする。行・列を0始まりで `(row,col)` とすれば、既知の中央→marker取付 `D_i=T_{M_0M_i}` の並進は

\[
d_i=((col-1)0.09,\ (1-row)0.09,\ 0),
\]

回転はすべて恒等である。中央の `x` が右、`y` が上で、同じ平面に載る。これらは既知の剛体取付であり、推定した各markerの独立な根元位置を事前に与える構成ではない。

黒いmarker領域の全体幅は0.252 m、外側の1 module quiet zoneを含む白いボード土台は0.27 m角である。旧 `marker.id=21`・`marker.size=0.18` は単一markerを使う経路の既存参照として残るが、board有効時の外形は `camera.board_size`、各native markerの姿勢推定に使う実寸は0.072 mである。ID21を0.18 mの単一markerとして推定しない。画像中の白土台とRVizのwhite CUBEも同じ0.27 m角へ拡大し、黒いmarkerだけ拡大して土台を旧寸法に残さない。

`board_camera.py` は9個のnative AprilTag bitmapを480×480の平面textureへ配置し、既存の `PlanarCamera` でボード全体を一度描画する。時刻補間によるモーションブラー、Gaussianぼかし、画像noiseは合成済み画像へ一度適用する。各markerごとに画像全体のnoiseやblurを重複適用しない。カメラ画像960×720、Gaussian sigma=2、noise_std=8、露光0.09秒、時間補間7点を維持する。

板の根元姿勢、スイッチ、ロボット、panel障害物と衝突条件は変更しない。物理scene表示にも同じ9 bitmapと既知offsetを使う。真の板・カメラ姿勢を使うのは従来どおり模擬画像生成・物理参照描画・検証だけであり、検出器や制御へその姿勢を渡さない。

## 一枚の画像から中央へ写す

画像時刻を `t`、有効に検出・推定できた既知ID集合を `V_t` とする。各 `i` の画像単独カメラ相対候補を

\[
L_{Ci,t}=\sum_k w_{i,k,t}L_{Ci,k,t}
\]

とする。検出器のtranslation priorは従来どおり無効にし、各native成分のHessian由来位置・方位結合と、weight・有限log weightを保存する。通常weightのunderflowを成分削除の根拠にしない。

既知取付だけを使って、各候補を

\[
T_{CM_0}=T_{CM_i}D_i^{-1}
\]

へ押し出す。取付の向きを逆にしない。既存coreのBingham由来S²高集中近似・姿勢モーメント照会を使い、markerから中央への腕長による位置・方位の結合と全6次元covarianceを計算する。この段階で位置・方位crossを捨ててから中央へ写してはならない。

中央画像融合に身体、共通カメラ候補、保持地図の事前は入れない。複数markerから別々に根元候補を作って共通カメラの不確かさを独立扱いし、その積で分散を縮める構成は禁止する。中央への画像単独作用を済ませてから、共通身体／カメラの処理へ一度渡す。

## 中央の逐次近似と適用域による縮約延期

一枚の画像内で有効marker IDの昇順を固定し、中央変量だけのassumed density filtering（ADF）を行う。最初の有効markerの中央候補全mixtureから始め、保持地図や前画像の中央観測を初期priorとして使わない。画像間の情報蓄積は既存の内部身体・未知地図事後で行い、画像単独中央観測と混同しない。

各stepでは、現在の中央proxyまたは縮約を延期した全Gaussian枝と、次markerの中央候補の全native成分を、共通のBingham由来参照座標へ明示的に輸送する。局所Gaussian代理の全枝の密度積を、重なり、座標の体積係数、元の画像log weightと前段各枝のlog質量を含めて正規化する。身体・カメラの事前を各stepへ追加せず、各markerの観測を一度だけ使う。

正規化後の派生枝を `a_k`、平均を `m_k`、全covarianceを `P_k` とすれば、中央の中間モーメントは

\[
\bar m=\sum_k a_km_k,
\qquad
\bar P=\sum_k a_k\left[P_k+(m_k-\bar m)(m_k-\bar m)^\top\right]
\]

で計算する。withinとbetweenの両方を含め、共通座標で位置・方位crossまで求める。既存APIでBinghamの四元数二次モーメントを整合し、coreの高集中域を検査する。集中時だけ、今回許可された中央のGaussian×Bingham積近似へ射影し、この位置で中央の位置・方位crossを切断する。捨てるブロックと前後のモーメントを記録する。身体の全共同covarianceや、後段で生じる身体・根元目標crossまで削除する許可ではない。

単峰近似の方位covariance最大固有値がcoreの既存上限0.1 rad²を超えたら、精密なモーメントfitが得られていてもその単峰化を適用しない。中間では全狭い枝と全covariance・cross・正規化log質量を同じ参照座標で保持し、次markerの全native成分と積を取る。広い単峰lawを無理にcoreへ通したり、上限変更、分散clip、枝pruneで代用したりしない。

最終でも単峰化が広い場合は、全ての狭い派生枝をそれぞれGaussian×Binghamへ表現し、その全中央mixtureを出力する。この最終の各成分の位置・方位cross切断も記録する。各出力枝は既存高集中域内であることを確認し、混合の広さと各枝の広さを区別する。中央singleは許可であり必須ではない。

通常は単峰中央proxyを次markerへ渡すが、延期中は積枝数が増える。常に2枝や指数的列挙なしとは主張しない。単峰化を行った段階を含む固定ID順ADF近似であり、全組合せの厳密な共同事後、センサ順序不変、厳密なSE(3)密度積だとは説明しない。順序、各stepの全入力・積枝、参照座標、log overlap・log weight、within/between、保持／切断cross、適用／延期理由と最大生存枝数を公開する。

Bingham整合は既存coreの二次モーメント・fitを使う。高集中時の絶対残差停止や初期値によりfit残差が残る場合は、同じ二次モーメント方程式を既存モーメント関数で決定論的に精密化し、core／最終残差と関数評価数を記録する。これは数値解の精密化であり、coreの分布法則や適用域を変更するものではない。

画像内のmarker残差を、共通の中央姿勢と既知較正・取付の下で条件付き独立な画像尤度として扱う仮定を明示する。同じ画像に共通するblurや未モデル化の較正誤差まで、現実に独立であると主張しない。native Hessian代理、座標輸送、中央単峰化による近似を残し、厳密な画像Bayes事後とは呼ばない。

## 身体更新と根元目標への接続

画像処理は一度だけ行い、その時刻の中央画像観測をkeep/reset双方で共有する。各markerを別々の身体posteriorへ更新してから、同じ身体事前を補正せず掛け合わせてはならない。中央proxyを一つの画像観測として既存のセンサ別更新・共通事前補正融合へ渡し、IMU併用でも一つの新有効観測packetにつきBF変化判定を一度だけ行う。剛性密度比因子もマーカー数だけ重複生成しない。

観測融合・単一化を終えた取得時canonical身体／カメラ候補と、同じ画像単独中央proxyから、根元→カメラ→中央→既知のスイッチ関係の順にtask目標を構成する。共通カメラのroot liftingは一度であり、marker数だけ複製しない。身体に画像を既に使った後のtask専用独立cutと、coreの逆照会非閉包については `FRESH_OBSERVATION_GOALS.md` の規定を維持する。中央融合を追加しただけでこの近似が厳密化したとは主張しない。

集中時の中央proxyは単一Gaussian×単一Binghamであり、その目標はweight=1・log weight=0の1成分でよい。単峰化できない最終中央mixtureでは、全狭い枝・weight・log weightをそのまま目標へ伝播する。いずれもraw native画像から一枝だけを選んだ結果ではなく、全有効markerの全native枝を上の手順で処理した派生近似である。recordに保存するraw marker成分数と、中央成分数・task目標成分数を区別する。既存の `native` というfield名が中央派生入力を指す場合も、その意味とraw観測への参照を明記する。

`central_distribution` は単峰・mixture共通で `components`・`component_count`・`representation` と親子frame・stampを持つ。単峰時だけ旧単一成分fieldも併記する。各 `steps.projection` の `applied`・`deferred`・`single_proxy_domain_reason`、`deferred_projection_marker_ids`・`maximum_live_branch_count`・`final_mixture_fallback_used` と、全出力成分のlineage・cross切断前のcovarianceを保存する。

同じ中央law、取得画像ID・時刻・指令・版、中央成分ID、全visible/valid marker ID、既知取付、ADF順と各入力sourceを、身体推定・fresh goal・固定目標記憶・IK・snapshot・実wireへ通す。制御と表示で別の中央平均・covarianceを作らず、シミュレータ真値へ再中心化しない。太い目標軸と成分別modal TFも同じ目標照会を使う。

## 旧小型sourceでの停止と新sourceの区別

最初の小型board（marker辺長0.048 m・pitch 0.06 m・白土台0.18 m）のsourceでは、中間の単峰Binghamの方位covariance最大固有値が約0.1304 rad²となり、既存coreの0.1 rad²域を超えて両backendが停止した。そのfit自体の二次モーメント誤差は約1.2e-14だったため、数値fitの精密化だけで高集中近似の適用域を広げてよい問題ではない。

旧記録は `/tmp/probtf_marker_board_20261017` と、SHA-256 `8558e0581415ffb0d3fc71dca2c8dce783379e383910a12b3613b6827fecedff` の39ファイルsource一覧に対応する。失敗を保存し、このsourceの不変性・単体試験を完走の証拠にはしない。

大型化、白土台拡大、縮約延期と最終mixture保持を含む新記録は `/tmp/probtf_marker_board_large_20261017` と、SHA-256 `ccb633159ae05dff91f5b9c7672ce863c131c3eb33d8940411033f5237848d01` の別source一覧に分ける。旧記録を上書きしない。新しい実画像・実配信・物理到達の結果は、同じ新sourceで別途検証する。

## 部分検出と全欠落

中央ID21が見えなくても、他の有効なmarkerから既知取付を使って中央候補を構成できる。全9個の同時検出を要求せず、その画像の有効ID集合を使う。設定外ID、重複ID、検出したが姿勢推定に使えなかったIDは理由とともに区別し、使っていない画像情報を融合済みとしない。

有効な中央観測が一つも得られないときは、既存のマーカー欠測処理を使う。固定根元目標の全周辺・質量・元画像情報を保持し、身体・目標crossを既定のSchmidt／BF仮説混合・再表現・実指令作用で輸送する。身体・未知地図・剛性履歴を初期化せず、既存の減速停止、waypointごとの静止、局所探索、停止後の新有効観測を融合・採用してからの復帰を維持する。

raw previewの可視性は、中央IDだけでなく設定済みボードIDのいずれかが検出されたこととする。ただしraw previewだけを、新しい融合済み中央観測や接近復帰の根拠にはしない。方向優先探索、hold時計の補正など、中断した追加処理を本変更へ組み込まない。

## 表示と検証

画像annotationはgreen輪郭と姿勢を維持し、余分な数値やID・weight説明を画像へ追加しない。画像表示サイズ、粒子径、既存の軸長・太さ・標本数、視点・レイアウトを変更しない。RViz全35設定ファイルを新しい作業前baselineへ保存し、実行・終了後に全バイトを比較する。画像は物理側でも推定側でも同じboard配置・marker実寸・較正を使う。

新sourceで次を検証する。

1. 9 bitmapとID・取付・marker実寸の一致、および画像生成時にblur/noiseを一度だけ作用させること。
2. 各raw native mixtureの成分・質量・log weight・画像元情報が保持され、全有効ID・全枝が中央融合へ入ること。
3. 既知取付の逆作用、全6次元covariance、固定ID順ADFのoverlap・座標輸送・within/between・中央積近似の位置が記録と一致すること。
4. 身体・カメラ・保持地図priorの不使用、後段の共有事前補正、keep/reset・IMU併用時のBF・剛性更新の一回性。
5. 中央不検出の部分集合でも中央を構成し、全欠測では旧目標・内部履歴を保持すること。raw可視性と融合・制御snapshot採用を区別すること。
6. 推定・IK・snapshot・serialize/deserialize後の候補MarkerArray・modal TFが同じ中央由来lawと全取得元を参照すること。
7. coreの既定法則変更・モンテカルロ・真値入力・表示専用分散がなく、全動作制約とRViz35ファイルのバイト一致を維持すること。

統合試験では、大型boardの実描画画像に中央ID21だけを白く隠す明示的な部分遮蔽を加え、他の全検出IDを保持した入力で検証する。別の解析的native-mode fixtureで、中間の広い単峰化を延期して次tagとの全積後に単峰化できる場合と、最終も広く2枝をfresh goal・実MarkerArrayへ全保持する場合を分けて検査する。後者は適用域・データ受渡しの試験であり、実画像にそのモードが生じる頻度や推定精度の実測ではない。

画像検出数、中心の画素位置・輪郭bounds、欠落・再検出、推定・制御・描画Hz、物理到達は新sourceで個別に測定する。過去の単一markerや旧探索の実測を流用せず、新しいraw画像、融合選択、指令、candidate/physical clock、実wireと凍結sourceを一組で記録する。
