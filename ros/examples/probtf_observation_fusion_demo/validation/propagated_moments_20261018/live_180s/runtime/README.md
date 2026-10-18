# 伝播式による逐次更新：同sourceでの180秒再走

2026-10-18。observation_update.method=propagated_moments、1138テスト通過版を固定し、Marker単独とIMU＋Markerを新しい初期状態から各180物理秒実行した。直前の90秒と全359 source/configファイルおよび4 native binaryのSHAは同じであり、変更は/tmp orchestrationのduration・出力先・専用port・画像ラベルだけだった。壁時計の進みと非同期選択結果は各試行で違うので、両試行の観測列まで同一という比較ではない。

現在の方式はcore高集中Bingham由来のmean/cov・全crossの伝播と、native観測Gaussianによる一回の解析joint条件付けである。旧point MAP/GNの計算を現在の観測更新と説明しない。core/RViz保護対象214ファイルを保ち、実走開始前とshutdown時のsource/config/native/RViz検査はtrueだった。終了後の説明文書追記とは区別する。

## 物理側の結果

| ケース | 180秒終端距離 | 保存tick端点最小距離 | 最小の時刻 | 到達flag | 実物理Hz | 画像Hz | control結果受信Hz |
|---|---:|---:|---:|---|---:|---:|---:|
| marker | 0.03510 m | 0.03371 m | 176.84 s | 未達 | 39.93 | 10.38 | 4.89033 |
| imu_marker | 0.09696 m | 0.08058 m | 133.06 s | 未達 | 35.18 | 10.05 | 4.41567 |

最小値は保存された外側物理tick端点についての値であり、未保存の連続時間全域の厳密最小値とは呼ばない。両試行とも保存tickの到達flagはfalseだった。物理干渉判定OFFのため、無干渉は未検証。到達成功やリンク推定jump解消は主張しない。リンクの同一q比較・branch closureは別の診断資料で評価する。

control結果受信Hzはcutoff manifestのactual_control_hzそのものを記載した。これは開始後の即時control結果受信時刻列について(count−1)/(last−first)を求めた値で、待機・不採用結果も含む。物理servo tickやaccepted gradientだけの周波数ではない。実physical elapsed<=180へ絞る独立計算はsummary.jsonへ分けて記録した。終了後のrateやsource age0で動作中の値を置き換えていない。

## raw、身体、Kの会計

| ケース | 保存raw/身体journal | 有効融合 | time-thinning | K完了/保存archive | 未完K |
|---|---:|---:|---:|---:|---:|
| marker | 10520 / 10520 | 57 | 1810 | 58 / 58 | 0 |
| imu_marker | 9856 / 9856 | 442 | 5136 | 103 / 443 | 340 |

全rawを保存したことと全raw観測を融合したことは別である。通知・無効センサ・欠測・明示time-thinningを別会計にした。K件数には各caseの初期job1件を含む。180秒後の最大60秒drainを分け、IMU側の未完340件を元ratio・原取得t/q・実ledgerとともにarchiveへ保持した。

全20376rawについてpayload SHA・原取得t/q/sequence/frame、spool、実command endpointを照合し、BLOB計2,543,802,864bytesもhashして一致した。全499有効観測にK archiveが対応し、初期を含む501 archiveのcanonical共同body/goalの全mean/cov・Bingham・mass/weight/source数値identityを再hashして一致した。欠落・時刻付け替え・sequence逆転・source/K整合エラーは検出されなかった。archive ledgerは直前推定stateからの輸送区間であり、前の有効観測からの全区間と同一とは限らない。

raw画像全pixelはDBとPNGへ保持した。一方、全画像の検出corner座標配列を実行時に独立保存したとは言えない。preview journalはindex/stamp/visible/IDsであり、描画したcornersは記録されていない。selected画像の全native観測分布/Schur/cov/sourceは保持する。rawから後解析でcornerを再抽出する場合は、実行時に保存済みのkeypointとは区別する。

## 元観測の古さと計算の古さ

次のageは、実際に積分したcommand intervalのtick記録ごとの非加重統計である。各欄は中央値/最大秒。実使用packetの原source stampを使い、最新推定値で過去を再計算していない。

| ケース | gradient自身 | body観測 | goal観測 | 推定Kの原source |
|---|---:|---:|---:|---:|
| marker | 0.253 / 3.266 | 7.881 / 50.421 | 7.881 / 50.421 | 9.899 / 53.856 |
| imu_marker | 0.272 / 4.259 | 3.164 / 35.909 | 3.741 / 35.909 | 43.167 / 166.796 |

gradient計算が新しくても、入力body観測やKまで新しいとは限らない。manifestのestimate_stampは欠測時の輸送でも進む処理frontierなので、bodyの有効観測ageと同じとは扱わない。5秒sampleの処理frontier age、body/K待ち、実使用source ageをsummary.jsonで分離した。

runtime stageのtotalはinclusiveである。raw_batch_completion_and_checkpointsはbody更新等を含むため、そのtotalをDB/checkpoint固有overheadとして解釈しない。stage_exclusive.jsonでは同processの直接子interval unionを差し引いた残りも示す。その残りも未計時処理/記録境界の費用を含み、DBやJSONだけに帰属させない。入れ子stageのinclusive totalを足して全実時間としない。GPUの実描画完了時刻は測っていない。

## 表示と初期prior

両初期snapshotはobservation IDs空、K未選定、gradient無しでwire ackされ、同じcandidate identityを持つ。初期link/hand/camera modal姿勢は、真のK/関節角を使わない通常の送信指令rigid FKと約1e-15以内で一致した。初期左PNGはreplay前のpause中に同期撮影した。

両RVizは同一marker.rvizの全byte同一copyで起動し、こちらはwindow x/yだけを動かした。撮影時のIMU 3D viewはMarkerよりzoomしており、変化の時点/原因は確認できない。初期IMU画像は選定撮影していない。**同一視点での画角比較とは説明しない。** こちらから視点・粒子・画像・軸サイズを変更せず、その時の実viewerを撮影した。loaded/sourceのfile SHA保護とin-memory viewの一致は別である。

180秒画像は両ケースの180秒動作終了を検知後に撮影し、後段drainが進んでいるためcutoff manifestそのものと同一snapshotとは限らない。capture前後のphysical/source時刻はimage_capture_metadata.jsonへ残した。3枚はいずれもRViz内容領域から直接X11取得し、desktop/browser/terminalは含まない。post-capture crop/resizeはしていない。新2RViz/各masterは可視で残し、owned demo workersだけを正常終了した。

このfolderは小容量選定資料である。full raw/DB/spool/pickle/巨大JSONL/動画/source_freeze全文は含めない。必要範囲のsource/RViz/native SHAだけをsource_integrity_scoped.jsonへ記録し、元の完全manifestは/tmp参照とSHAを残す。原入力・全job・全raw・後解析toolは元/tmpへ保持する。

全1705保存snapshot・360,868,317 bytesは、保存core queryから再生成したmarker wireと全byte一致した。即時ready・後着artifact・表示採用・gradient採否journalのpacket/source/K identityも一致した。physical cutoffまで1676件、後drain29件を分離した。全候補が実動作採用されたという意味ではない。表示採用済みでpublication ackだけ欠けるものが4件あり、追加の配信完了/未配信断定はしない。subscriber captureやGPU paint完了は未測定である。
