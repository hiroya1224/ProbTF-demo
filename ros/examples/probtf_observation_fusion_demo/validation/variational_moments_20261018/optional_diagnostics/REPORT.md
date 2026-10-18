# 未使用の条件付き目標診断による停止

閾値撤去後の試行は、Marker1.8149秒／IMU併用2.3459秒で終了した。原画像 `image:00000127`、sequence89、t=1792080001.0845249、元指令と実76区間ledgerから、同じSchur差の固有値−1.1975682986156904e−12まで再現した。元取得を処理時刻へ付け替えていない。

身体事後のcanonical共同平均・全共分散は既に計算できていた。その後、各raw枝の $P_{EE}-P_{EB}P_{BB}^{+}P_{BE}$ を使う条件付き目標の構築で例外となった。この画像では実目標を原native観測とcanonical身体からfreshに構成するため、その条件付き目標はtask/K/制御/表示で使わない。全推定の停止を実際の身体事後やfresh goalの計算不能と説明しない。

`project_update` の既定では従来どおり条件付き目標を構築する。呼出し側で新しい有効画像または保持目標がある場合だけ `construct_conditional_goals=False` を渡し、未使用のSchur計算・条件付きlawの再表現を省く。canonical共同平均、全P・全cross、within/between、全raw枝・log質量・sourceはそのまま保持する。新画像も保持目標もなく、このlawを実際に使う場合は既定の構築を維持する。PSD閾値の変更・分散clip・jitterを行ったものではない。

省略は `conditional_goals_constructed=False` と `conditional_goal_omission_reason` に記録する。空の診断goal一覧を、零情報観測や実目標の削除として解釈しない。

同じ失敗画像の [再処理](failed_event_replay.json) では、両親の元prior・最適化m/P・Fの各加算項・native観測が旧失敗時とbitwise一致した。旧失敗直前に完成していたcanonical raw mean・全covariance/crossも新値とbitwise一致した。未使用の条件付き計算の呼出しは0回で、fresh goalと同じlawのwire照合まで完了した。旧実走には成功したgoal/wireが存在しないため、それらが旧失敗実走と一致したとは主張しない。

新しい5回帰では、条件付き計算が成功する別のfixtureで省略前後を比較し、全native枝・underflowしたlog質量・canonical身体・実目標・制御指令・RViz配信bytesの厳密一致を検証した。条件付き計算を強制例外化しても未使用経路は呼び出さず、必要な既定経路はその実エラーを維持することも検証する。

変更後の関連15ファイル130件と新規5件が通過した。それ以前のcore・デモ全1263件の通過とは [別に記録](regression_coverage.json) し、重複件数を合計しない。今回変えた既存数値sourceは接続する2ファイルだけで、伝播・変分規準・K・core・native libraryは変更していない。最終sourceは親ディレクトリの198ファイルmanifestに保存した。大きいpickle・raw画像・取得DBは元の `/tmp` 出力に保持し、Gitへは複製しない。
