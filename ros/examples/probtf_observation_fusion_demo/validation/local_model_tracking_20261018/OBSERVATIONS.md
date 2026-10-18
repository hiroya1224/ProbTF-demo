# 凍結sourceによる90秒の実走と候補表示の異常

収録時HEAD: `7790a0674bd7f31b39f516514b177544dc899503`。実走した作業ツリーの内容はcapture_metadata.jsonの55ファイルのSHAで識別する。HEADだけと一致するコードだったと主張しない。marker単独とIMU＋markerを別ROS masterで90秒ずつ同時実行した。ソース55ファイルのSHAは開始前後ですべて一致。両RViz設定は基準marker.rvizの全バイトと一致した。rootの回帰試験と同時実行のため、単独負荷での性能測定ではない。

**リンク候補の異常な跳ねは解消していない。** 制御指令の連続性を、推定姿勢の滑らかさや正しさの証拠として扱わない。

|項目|marker|IMU＋marker|
|---|---:|---:|
|物理更新 Hz|72.6318|71.5607|
|完成gradient Hz|3.9841|4.4565|
|画像取得 Hz|10.8541|11.0056|
|終端の物理距離 m|0.4623|0.3926|
|終端bodyの取得時刻 s|27.4743|19.9354|
|表示予測age s|18.9699|3.4248|

両者とも未到達。外部干渉判定はOFFなので、無干渉の成功を検証した実験ではない。終端manifestは壁時刻cutoffのmetadataであり、終端後の新gradient採用も含み得る。実際の動作中gradient age・指令差・K区間はcommandsの全使用区間から別集計した。

marker: 使用gradientのage中央値 0.340s、p95 0.663s、最大 2.668s。物理終端時のK完了 6/7、pending [6]。表示artifact待ち 47。実行中candidateと表示candidateは異なっていた。

停止時点: raw 6094/6094件を保存、body DB状態 {'completed': 4727, 'processing': 1367}、K pending [6, 7, 8]。全処理済みではない。処理済み／予約中の区別、元raw、凍結Kjob、指令ledgerは/tmpの元実験へ保持した。今回のdemo workerだけを停止しRViz/masterは残した。

IMU＋marker: 使用gradientのage中央値 0.321s、p95 0.510s、最大 1.707s。物理終端時のK完了 4/6、pending [4, 5]。表示artifact待ち 14。実行中candidateと表示candidateは異なっていた。

停止時点: raw 6188/6188件を保存、body DB状態 {'completed': 1470, 'pending': 1428, 'processing': 3290}、K pending [5, 6]。全処理済みではない。処理済み／予約中の区別、元raw、凍結Kjob、指令ledgerは/tmpの元実験へ保持した。今回のdemo workerだけを停止しRViz/masterは残した。

markerの最大表示跳び: request 142→143、hand origin mode 1.346m、mean 1.345m、modal軸方向 136.1°。凍結canonical身体を元の取得時刻でcoreへ直接照会しても hand mean 1.254m変わる。FIFO表示遅滞やmodal軸だけの問題では説明できない。BFSのReset事後への切替との対応を確認したが、これだけで原因を確定・修正したとは主張しない。

IMU＋markerの最大表示跳び: request 45→46、hand origin mode 1.102m、mean 1.101m、modal軸方向 122.8°。凍結canonical身体を元の取得時刻でcoreへ直接照会しても hand mean 1.094m変わる。FIFO表示遅滞やmodal軸だけの問題では説明できない。BFSのReset事後への切替との対応を確認したが、これだけで原因を確定・修正したとは主張しない。

全770候補のsaved point-queryからMarkerArray全162,947,814 bytesを再生成し一致した。即時readyと後着artifactのnumeric-law candidate checksum、採用g/H/K/sourceも一致した。物理cutoff内708候補、後着62候補を分けてwire_identity.jsonに記録した。ROS配信journalとの照合までで、subscriber受信やGPU描画完了は測っていない。点照会記録は完全な共同変換密度ではなく、この照合はraw推定／core伝播全体の独立再計算を意味しない。

制御実装の別検証として、実使用全12,868区間を独立した13×13行列指数のODE解と比較した。位置誤差≤4.44e-16rad、速度誤差≤2.78e-17rad/s。採用前の区間へ新packetを遡及使用せず、採用時のu/v・区間境界のu/vのjumpは零だった。これは候補姿勢の異常を否定する結果ではない。

物理90秒終了後の別撮り `post90_both_windows.png` はローカル保存だけとし、デスクトップ全体の画像をGitへ含めない。開始時の窓移動helperがQt補助窓を選び、元の20/60/90秒写真では実窓が重なったため、終了後にメイン窓だけ位置移動した。カメラ・パーティクル・軸・レイアウト・窓サイズ・RVizファイルは変えていない。

データ: summary.json、runtime_samples.csv、runtime_stage_totals.csv、wire_identity.json、link_jumps.json、canonical_jumps.json。巨大raw/DB/JSONL/中間pickle/録画は含めていない。
