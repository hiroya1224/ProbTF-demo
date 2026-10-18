# カメラ表示の遅延と取得処理の分離

2026年10月19日、手先姿勢によるK更新をRVizで確認中、画像に数十秒の遅れが発生した。

## 原因

再起動後のMarker試行では、物理時刻97.377秒に対して最新previewは78.085秒だった。IMU併用も93.458秒に対して77.278秒だった。その後は取得155秒に対してpreview約119秒まで遅れが拡大した。画像の取得時刻を現在時刻へ付け替えたものではなく、取得workerのFIFO内で未処理画像が積み上がっていた。

同じ960×720画像・7時刻の露光・ぼかし・ノイズ設定で、現在の物理側requestを使って計測した中央値は次の通り。

| 処理 | 中央値 |
|---|---:|
| 物理画像の生成 | 42.41 ms |
| マーカー検出 | 25.66 ms |
| PNG保存 | 11.79 ms |
| 画像のSQLite FULL同期保存 | 6.52 ms |
| 関節通知1件のSQLite FULL同期保存 | 4.13 ms |

旧実装は一つの取得workerで、全関節・IMU通知のDB処理、画像生成、PNG保存、検出、画像DB保存、推定用DB選択を直列に処理した。画像だけでも約86 msで、他の全センサ保存が加わるため、同じ処理列では設定された画像12 Hzに追い付かない。計測rawは `/tmp/probtf_camera_latency_profile_20261019`、独立した別計測は `/tmp/probtf_acquisition_profile_20261019/result.json` に保持する。

RVizの画像受信キューは既に1件だった。取得後の配信にも、scene描画・raw配信・JSON保存を待つ処理順があり、そこも遅延に寄与し得た。

## 修正

画像生成・検出と、PNG／DB保存を分ける。カメラは一つのforeground処理で原取得順に動かし、既存の露光・ぼかし・ノイズと乱数列を変更しない。その同じ画素からpreviewを構成し、`preview_ready` を先に渡す。

単一のarchiveスレッドが専用のSensorEventStore接続を所有し、全raw・画像・DB操作を受け渡された元のFIFO順に処理する。元のID・時刻・指令・received_wall_stampを保持する。PNGは従来どおりlosslessで保存し、DBは既存のFULL同期を維持する。古い未保存画像を飛び越して推定対象を選ばない。表示を新しくするために推定入力の画像を削除したり、画像を別に生成したりしない。

`preview_ready` は表示・raw検出の準備ができたことだけを表す。raw保存済み件数は、従来の `image_ready` による完了通知でのみ加算する。画像のraw配信も保存完了後に一度だけ行う。先に新しいpreviewが届いた場合、古い保存完了通知で表示や検出状態を巻き戻さない。

配信側は同じ元時刻のpreviewを、重いraw配信・scene描画・JSON保存より前にpublishする。画像publisherの待ちキューを1件とし、全rawと軌跡の保存・配信は維持する。RViz設定、画像寸法・表示サイズ、粒子・軸、候補分布、推定・制御規準は変更しない。

motion manifestには `camera_preview_source_stamp`、`camera_preview_age_s`、`camera_preview_capture_index` と、保存待ちにしないことを記録する。これは身体推定・K・候補表示のageとは別の量である。有限の遅延・未達を停止条件にはしない。

## 検証

保存処理を意図的に止めたテストで、同じ画像・元時刻のpreviewが先に届き、raw完了件数は増えないことを確認する。保存後のPNGを復号し、元画像・DB画像の画素が一致すること、全rawの時刻・指令・順序、古い完了通知で新しいpreviewが戻らないことを検証する。配信テストでは、遅いscene処理の前に元画素・元時刻のpreviewがpublishされることを確認する。

修正版の最初の実行はディスク容量不足で約25秒時点に中断した。今回作成した中断済みの旧RViz一時データ約20 GBだけを削除し、約21 GBの空きを確保してから初期状態からやり直した。容量不足前の測定を180秒完走として扱わない。

復旧後のRViz実行の最初の31～35秒を集計した。取得側からcoordinatorへpreviewが届いた時点のageは以下だった。

| 指標 | Marker | IMU＋Marker |
|---|---:|---:|
| 集計物理時刻 | 34.65秒 | 31.04秒 |
| preview件数 | 366 | 327 |
| age中央値 | 69.45 ms | 68.87 ms |
| age 95 percentile | 106.89 ms | 109.73 ms |
| age最大値 | 661.82 ms | 552.67 ms |
| 集計時のraw保存待ち件数 | 6 | 11 |

数十秒単位の増加はこの区間では見られず、保存済み件数とは独立して画像を表示できた。MarkerのROS画像トピックを別のsubscriberで2秒間受信すると25枚が届き、非同期に受け取った最新motion metadataと画像stampの差は最大0.949秒だった。取得側の中央値だけをRVizまでの最大遅延と説明せず、scene配信や物理tickの遅れも区別する。

これは新sourceの短い実測区間であり、全180秒の完走、全推定drain、常時一定の最大遅延を保証するものではない。全rawを捨てずに保存しながら両RVizで180秒の動作を続ける。取得・保存の詳細は [live_prefix.json](validation/camera_preview_20261019/live_prefix.json)。修正関連38件と非同期統合・手先更新12件を含む追加53件のテストが通過した。

## 180秒全体に残った待ちと、取得側からの直接配信

上記の短い区間に続いて180秒全体を集計すると、preview準備完了をcoordinatorが受け取った時点のage中央値はMarker90.7 ms／IMU併用93.7 msだった。一方で最大値は5.06秒／3.23秒まで伸びた。全rawは9,502件／9,147件を保存できたが、身体・K推定はdrain未完了だった。到達は両方未達で、終端誤差は94.8 mm／59.1 mmだった。これは画像保存の分離だけを適用した版の [180秒記録](validation/camera_preview_20261019/live_180s.json) である。

最大ageのMarker画像では、foreground待ちは1.7 ms、生成37.4 ms、検出22.0 msだった。画像生成を待って5秒になったわけではない。同じ終盤にcoordinatorのartifact受信後から次の推定完了markまで約3.06秒のCPU処理区間があり、配信FIFOにも最大約4.06秒の待ちがあった。大きいartifactのunpickleやPythonオブジェクト処理が候補であるが、GCそのものを同定した計測ではない。IMU併用でも配信FIFOの最大待ちは約5.38秒だった。

そのため、通常のROS Liveでは取得workerが検出直後の同じ注釈画素を直接 `/observation_fusion/detection_image` へ配信する。既存ROS初期化と別の匿名ノード、queue size 1、元の画像stamp・BGR8・寸法を使う。新たな画像生成や推定は行わない。coordinatorとscene描画のFIFOを通らず、旧publication workerの同topicへの送信は無効化し、古い画像が後から戻る二重配信を防ぐ。headless実行は従来の戻り値経路を既定として保つ。

この直接経路は画像UIだけである。推定へ渡す全raw、取得順のDB保存、raw検出と有効な融合目標の区別、身体law・K・制御・候補表示を変更しない。カメラ取得側が保持する物理姿勢を推定へ渡すこともない。旧coordinator経由の `camera_preview_age_s` はその受信側のageであり、直接画像UIの配信ageと混同しない。manifestの `camera_preview_publication_route` で経路を明示する。

`camera_live_publication.jsonl` に、元の取得wall時刻、原source stamp、検出完了wall時刻、ROS publish返却時刻を記録する。ROS送信はRVizの描画完了ではないので、別subscriberで原取得journalと画像headerを照合した。

| 実ROS subscriberの3秒測定 | Marker | IMU＋Marker |
|---|---:|---:|
| 受信画像数 | 14 | 31 |
| 取得wall時刻との照合不一致 | 0 | 0 |
| 取得→受信の中央値 | 64.1 ms | 65.6 ms |
| 95 percentile | 84.6 ms | 81.5 ms |
| 最大値 | 302.3 ms | 1,343.6 ms |

接続時にはlatchで直前の画像も受信する。coordinatorが物理tickと取得要求そのものを止めた区間について、この直接経路で未取得画像を作り出すことはできない。取得後の表示待ち、取得間隔の欠落、RViz描画の停止を分ける。カメラだけを独立配信したことを、物理・制御ループ全体の停止解消と説明しない。

通常の画像生成・検出・保存の数値法則を変えず、実行中の制御・候補読み込みの停止を画像転送から分離した変更である。保存を止めるテストと、遅いsceneのテストに加え、直接配信の画素・原stamp・単一publisher・headless無効時の無副作用・所有ROSノードの終了を検証する。

### 直接配信版の180秒結果

直接配信版を両ケース180秒動かした。完走後に集計した値は以下の通り。

| 指標 | Marker | IMU＋Marker |
|---|---:|---:|
| 配信画像数 | 1,786 | 1,752 |
| 取得→ROS送信の中央値 | 63.33 ms | 62.38 ms |
| 同95 percentile | 83.72 ms | 82.40 ms |
| 準備完了→ROS送信の中央値 | 0.442 ms | 0.412 ms |
| 準備完了→ROS送信の最大値 | 2.934 ms | 2.493 ms |
| 元取得の最大間隔 | 3.102秒 | 3.073秒 |
| 配信の最大間隔 | 3.105秒 | 3.068秒 |
| raw保存済み件数 | 9,439 | 9,144 |

全画像のframe index・取得stamp・元wall取得時刻との対応は不一致0件、時系列の逆転も0件だった。生成済み画像をcoordinatorやscene描画の後で待たせる経路は外せたが、約3秒の元取得の停止は残る。両ケースともrawは全件保存し、集計時の身体・K推定全体はdrain未完了だった。到達は未達で、最終距離は186.9 mm／58.1 mm。表示経路の変更だけで到達性能が改善したとは説明しない。

出自・処理待ち・取得間隔と配信間隔の分離、および全件検査は [direct_live_180s.json](validation/camera_preview_20261019/direct_live_180s.json)。再集計は `validation/tools/summarize_camera_preview_runs.py`。rawは `/tmp/probtf_direct_camera_rviz_20261019` に保存した。

最終の直接配信sourceで、観測融合デモ全テストとvalidation/toolsの回帰をまとめて再実行し、1,350件全て通過した。RViz43ファイルと保護したcoreファイルのバイト一致も確認し、画像・粒子・軸のサイズは変更していない。

容量確保のため、停止済み4試行のcontrolログだけをgzipで可逆圧縮した。元16.93 GB→3.21 GBで、各ファイルは復号後のSHA-256一致を確認してから元を置き換えた。raw画像と観測DBは変更していない。対応表は `/tmp/probtf_compressed_control_archives_20261019.json`。古い対照試行を再集計する場合は該当controlログを展開してから行う。
