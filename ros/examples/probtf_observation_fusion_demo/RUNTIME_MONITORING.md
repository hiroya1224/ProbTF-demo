# 実行段階と遅滞の監視

`runtime_monitor.py` は計算内容や採否条件を変えず、各processの `runtime/<role>_progress.jsonl` と `<role>_current.json` を保存する。後者はatomic replaceされ、長い計算が戻る前にも開始段階を外部から読める。各行はPID、Unix壁時計、monotonic時計、process CPU時計、stage、原batch/request IDとsource時刻を持つ。終了行はwall/CPU所要時間と実例外を保持し、例外を握り潰さない。入れ子の時間は包含関係にあり、親子を単純加算してはならない。

推定側はraw読込、共同身体更新、K因子/evidence、K数値代表選定、記録とhandoff、全rawの処理済み確定を分ける。制御側は実区間ledgerによる輸送、K捕捉とreanchor、task/view勾配、snapshot/marker作成、sidecar保存、control JSON保存を分ける。coordinatorは結果受信、canonical law/Kの採用、勾配採用を実物理時刻付きで記録する。Queueの `put` は背景threadへ移すため、enqueue完了を受信・unpickle完了と同一視しない。

配信側はqueue受信、raw publish、候補wire、候補JSON、剛性JSON、scene、metadata、manifest、preview、記録を分ける。ROS publishの返却はRViz GPU描画完了を意味しない。旧 `queue_delay_s` はraw配信後までの時間を含むため従来の位置で保持し、純粋な受信時点までの `queue_received_delay_s` を別記する。

外部observerは小さいcurrentファイルと `/proc/<pid>/stat` を読み、計算中のstage経過時間とCPU増分を監視できる。CPU時間はそのprocessの全threadの合計であり、wall時間との差を単独でI/O待ちと断定できない。複数caseの同時実行、録画、OSスケジューリングを実測条件へ記録する。

動作開始前の `INITIAL_PRIOR`、物理90秒終了、推定処理中、全raw処理済みを区別する。未完了stageの実測下限と完了所要時間を区別する。有限な長時間計算を監査閾値で打ち切ったり、未処理観測を捨てて速度を装う機構は追加していない。
