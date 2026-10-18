# 固定priorのGaussian全mean/cov更新・180秒実行記録

Marker単独とIMU併用を、それぞれ初期状態から180物理秒実行した。両方とも180秒へ到達したが、到達条件は満たしていない。終端距離はMarker 213.789 mm、IMU併用38.740 mmで、保存した外側物理tickの最小距離も同じであった。干渉検査はOFFであり、無干渉の検証ではない。

180秒終了後の推定drainでIMU併用の実エラー `Bingham modal gap must be symmetric` が発生し、約13秒でdrainを中断した。両demoノードは正常な終了要求で閉じたが、全処理済みではない。独立した2つのROS masterとRVizは残した。この実行を「失敗なし」「K推定まで完了」とは記述しない。

## 原記録・時刻・未処理

`validation_summary.json` は物理終端時点の固定値とshutdown時点の会計を分ける。K件数には最初のevidenceなし初期化jobを1件含む。shutdown時のK archiveはMarker109/IMU457件、完了採用78/77件、未完archive31/380件。Markerのcoordinator未完カウンタ29件との差2件は、保存済みだが終了までに通知採用されなかった末尾archiveであり、未保存や完了と扱わない。

IMU併用は9,721件の取得要求の全payloadがDBに保存されている。Markerは10,275要求中9,978payloadがDBへ保存済みで、残る297件はjoint114、IMU113、画像70件。joint/IMUの原測定値と全画像取得要求・露光履歴は元spoolに残る。画像70件は停止時点でpixels/PNGがまだ生成されていなかった。別のrecovery archiveでは元numeric値と、凍結renderer/seed/取得履歴から後解析で再生成したpixelsを出自付きで保存した。後者を元の保存済み画像と偽らない。最初2枚・最後2枚の保存済み画像との再生成SHAは完全一致。元DB・manifest・rawは変更していない。

取得のt/q/frame/immutable sequence、実行済みledger、各有効イベントの全cross・native枝・K用前後lawを照合した。時刻は取得時刻であり、推定完了や到着壁時計へ付け替えていない。画像は0.09秒までの露光履歴を7姿勢で描画し、その単一画像を終端取得時刻/qの尤度へ対応させる既存近似である。取得時に完成したbody snapshotを保存していたという意味ではなく、後で直前事後を実ledgerで取得時刻まで輸送して処理する。

## 同じlawと残る問題

初期表示には観測IDも選定Kもなく、剛体指令FKとの位置差は最大7.22e−16 m。全1,806個の保存済みcontrol artifact、382,236,602 bytesを原snapshotから再生成し、numeric candidate identity・gradient source・ROS wireの完全一致を確認した。表示FIFOの遅延と未採用/未ackは別計数であり、現在active controlと現在displayが常に同じIDという主張ではない。

推定リンクの更新量は同じ取得t/qでKeep事前とcanonical事後を比較した。初期補正を別にすると最大はMarker151.834 mm、IMU併用107.355 mmで、跳ねが消えたとは言えない。代表2例ではほぼReset枝へ切り替わり、raw最適化Gaussian→core再照会の数値closureは小さい。一方24反復で未収束の枝が残る。新方式を完全Bayes事後・厳密なmoment・収束済みと説明しない。raw画像の外れや物理姿勢精度の原因はこの同一性検証だけでは確定しない。

最終の実積分区間ではK取得元のageがMarker147.068秒、IMU168.499秒、身体観測ageが33.327秒、26.256秒だった。勾配自体のageは約0.24/0.30秒でも、古いK/観測を用いて現在指令へ予測した結果である。view/task比、Kの一回score更新・未収束、実使用packetの出自は `control_metrics_selected.json` へ分けた。これらだけで手先振動の単独原因とは断定しない。

## 計時・source・画像

実測物理更新は約36.95/33.74 Hz。設定は120 Hzの目標、joint40/IMU30/画像12 Hzで、このrunが240 Hzを達成したという証拠ではない。control約5.13/4.79 Hzは即時reply受信のhost monotonic時計から求め、待機/不採用replyも含む。servo tickやaccepted gradientだけの周波数とは区別する。source ageは実際に積分したcommands区間のみから集計する。stage時間は入れ子なので和を取らず、`runtime_stage_exclusive.csv` は直接子区間のunionを引いた残りであり、DB/JSONだけの時間とは同定していない。GPU描画完了時間は測定していない。

source/config/core/RViz/native5本はshutdown時まで凍結一致。保護対象184件はhistorical214のうち当該core142件/RViz43件（重複1）で、無関係例・log30件を除く。ユーザー承認の高集中閾値撤去は2つのcoreファイルに限定し、式/分散clipは変更していない。今回runのbody nativeはd91ded4…、終了後の算術修正replayはd747c13…であり同じ実測sourceと扱わない。

初期2枚・終端2枚はRViz内容領域の直接capture。両方同じmarker.rviz全bytesを読み、window位置以外は操作していない。終端画像は両物理終了検知後のdrain開始中に撮影し、画面上も180.0/180.0、FINISHEDと確認した。撮影中に表示candidateは進み得るため、固定物理終端manifestと同一candidateの瞬間とは主張しない。元の画像/粒子/軸/視点設定は保護した。

postrun修正の単一イベントreplayは元prior/native/最適化m/P/Fをbitwise保持したまま成功し、223,133-byte wireも一致した。ただし全残件drain・K全履歴や新180秒の実走を完了したという証拠ではない。修正前失敗と修正後replayは別の記録を保持する。

元の大容量DB/raw/pickle/全jsonlは `/tmp/probtf_variational_commonchart_180s_live_20261018` に保持する。この選定資料へ複製しない。再現scriptと入力SHAは各JSONに記録した。
