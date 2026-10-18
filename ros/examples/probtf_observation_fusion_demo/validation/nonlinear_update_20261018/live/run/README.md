# 新しい非線形逐次更新の90秒実走と独立監査

2026-10-18。マーカー単独とIMU＋マーカーを、それぞれ新しい初期priorから独立したROS master・RVizで90物理秒実行した。両RVizはユーザーのmarker.rvizと全byte同一のcopyで、位置だけ移動した。画像・粒子・軸・レイアウトは変更していない。旧owned11875/11876は記録済PIDとコマンドが一致した4processだけ終了し、新11877/11878の2窓は実走後も残した。

実走sourceは1124テスト通過版で固定した。実行開始から終了・worker shutdownまで、source/config/native/core/RVizの全記録SHAが一致した。後から行ったexact-antipode境界修正はこの実走sourceと区別する。修正後1126テスト通過を、この実走のsource同一性とは扱わない。

## 観察された結果

| ケース | 90秒時点の実棒先端距離 | 到達条件 | 物理実測Hz | 画像実測Hz | control実測Hz |
|---|---:|---|---:|---:|---:|
| marker | 0.06032 m | 未達 | 53.36 | 10.52 | 5.67 |
| imu_marker | 0.06462 m | 未達 | 47.15 | 10.05 | 5.17 |

目標周期は実測と別である。実時間の全目標周波数を達成したとは説明しない。物理干渉判定は今回OFFであり、無干渉は未検証。最終距離はシミュレーション側の実棒先端を後から評価したもので、推定・IKへ真値を入れていない。残るリンク推定jumpは別の同一指令比較資料で調べており、位置・速度が連続であることやこの距離だけから解消済みとは判断しない。

## 取得と処理を区別した会計

| ケース | 保存raw | 身体処理journal | 有効融合 | 明示time-thinning | 終了時K完了/保存job |
|---|---:|---:|---:|---:|---:|
| marker | 5823 | 5823 | 137 | 797 | 66/138 |
| imu_marker | 5418 | 5418 | 501 | 2450 | 76/502 |

全rawは残したが、全raw観測を融合したとは説明しない。初期job1件もK保存/完了数に含む。marker側のIMUは比較条件で無効、指令通知は観測尤度でなく通知であり、その処理理由もjournalに残っている。body処理完了はK完了を意味しない。90秒終了後の最大60秒drainを分離し、未完K jobはmarker72件・IMU併用426件をraw/全ratio/実ledger付きarchiveとして保持した。未完を成功済みへ変更していない。

全11241rawについて取得ID・stamp・command・frame・sequence、spool、payload SHA、実行済み全command区間を照合し、1,281,586,488BLOB bytesもhashした。全640 K archive（初期2件込み）から共同body/goalの全mean/cov・Bingham・mass/weight/sourceを再hashし、保存numeric identityと一致した。全638有効融合観測がK archiveへ対応した。原時刻を現在へ付け替えた例、過去ledgerを最新Kで書き換えた例、同時刻sequenceの逆転・重複は検出していない。

取得時にオンライン推定lawを保存済みという構造ではない。遅れて処理するとき、直前canonical posteriorを実ledgerで原取得t/qへ運び、その観測の尤度を一回適用する。DB受信/ready host時刻、取得のsynthetic物理時刻、計算完了時刻は別である。カメラ描画は0.09秒・7点の姿勢履歴を使うが、native画像尤度はendpoint取得t/qへ割り当てる近似のままで、露光中候補軌道を積分する推定ではない。

## 推定・制御・表示の同一性

両初期snapshotは観測ID空、K未選定、gradient無しでwire ackされた。同一candidate IDを持ち、初期各link/hand/camera modal姿勢は通常の送信指令による剛体FKと約1e-15以内で一致した。この検査へ真のKや実関節角は使っていない。

全1003保存snapshot、212,278,197bytesについて、保存core queryからmarkerを独立再生成し、保存ROS marker wireと全byte一致した。即時ready packet、後着artifact、表示採用、gradient受領/採否journalにあるpacketのcandidate/source/K identityも一致した。1003候補が全て実動作に採用されたという意味ではなく、動作中の実採用と終了後の採用はsummary.jsonで分離した。物理cutoffまで977候補、後drain26候補を分けて集計した。marker1候補は表示採用記録があるがpublication journal ackが欠けるため、未配信とも配信完了とも追加断定しない。subscriberでのpacket captureやGPU描画完了は測定していない。

identity一致は、遅着artifactが元の採用候補と同じという意味である。現在activeな制御candidateと、画面上の古いcandidateが同時に同一という意味ではない。5秒ごとの18サンプルは全てactive/display IDが異なり、表示の遅れを隠していない。

## 時間がかかる段階

実際に積分されたcommand区間でのgradient source age中央値はmarker0.232秒・IMU併用0.247秒、最大1.73秒・1.97秒だった。終了後のsource age0や新packet採用を、動作中の性能へ含めていない。controlのtask/view gradient中央値は約62msだった。

gradient即時readyからartifact完成までの遅れは中央値0.260/0.275秒、artifact serialize自体の中央値は約80/83msだった。表示保存は別processで制御gradientの後に進んだ。publication queue受信遅れの中央値は0.028/0.030秒であり、GPU描画時間とは別である。

身体観測ageは5秒サンプルで中央値2.34/2.12秒、最大9.57/6.39秒だった。K evidenceの待ちはさらに長い。raw batch completion/checkpoints stageの42.3/56.6 wall秒は、body_event_update等を包むinclusive値であり、DB/checkpoint固有のoverheadではない。同processのstage nestから直接の子区間のunionを差し引くと、子を除く残りは14.48/16.56秒だった。この残りにも未計時の複数処理・記録境界のoverheadを含み、DBやJSONだけへの帰属ではない。動作window内に終了したstageを集計した推定なので、境界をまたぐ未完stageの費用を含む全90秒の厳密profilerではない。stageは入れ子なので、各inclusive totalを足して実時間の内訳として扱わない。runtime_stages.csv、runtime_stage_exclusive.csv、stage_exclusive.json、runtime_timeline.csvへ再検討可能な数値を残した。

## exact-antipode境界修正の影響確認

元の有効IMU400イベント・Keep/Reset800親について、旧runに凍結したevaluatorでGN計算を再生し、その2452評価点を修正後evaluatorでも対で計算した。residual/Jacobianはdtype・shape・符号付き零を含むraw array bytesが全一致し、新cutは0件だった。元保存の最終mean差は最大7.8e-16、Φ/log evidence差は最大7.2e-14、evaluator call数は全枝一致した。

これは保存された全trial座標そのものを読み出した検査ではなく、同じ元prior/観測から旧GN軌道を数値再現した検査である。今回の再現軌道でboundary修正の影響は検出されなかったが、新release sourceを再び90秒ライブ実行した結果ではない。全枝の詳細reportは/tmpに保持し、小資料にはsummary/source SHAのみ収めた。

## 保存物

このfolderはsmall選定資料だけで、raw画像、SQLite、spool全件、pickle、巨大JSONL、動画を含まない。rawデータと全factor/job、後解析scriptは `/tmp/probtf_nonlinear_fixed_prior_live_20261018` と `/tmp/probtf_nonlinear_*_20261018.py` に保持した。

- initial_prior_requested_marker.png：左窓はcapture前後とも物理t0。右initialはcapture中にt0→0.81へ進んだため、この資料の純prior画像には選ばなかった。
- physical_90s_marker.png / physical_90s_imu_marker.png：両90物理秒到達を検知してからの表示領域。撮影中も身体/Kの後段処理は進むため、終了cutoff manifestそのものと同一snapshotとは限らない。取得/表示source時刻とcapture前後の時刻はimage_capture_metadata.jsonに保持。
- 画像はRViz窓内のtoolbar下48pxから直接取得した。post-capture crop/resizeやviewer設定の変更はしておらず、browser/terminal等のdesktopは含めていない。
- 現在の新2窓は残した。worker停止後にK未完archiveが自動消去されることはない。

Git選定copyの3 CSVだけ改行をCRLFからLFへ揃えた。CSVとしての全field文字列は同一、原/tmp出力は変更していない。元SHAと選定copyのSHAはfiles.jsonに併記した。
