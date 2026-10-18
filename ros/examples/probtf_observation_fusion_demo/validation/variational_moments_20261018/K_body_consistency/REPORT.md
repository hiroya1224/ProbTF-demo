# 捕捉Kと現時点の身体分布の整合診断

180秒実走の最後の指令区間で実際に使用されたgradient packetからrequestを特定し、そのrequestと同じ身体law・指令・Kを再構成した。最後に届いただけのpacketや、終了後のK推定drain結果は使用していない。物理の再実行・新観測更新・平衡再solve・真値入力は行っていない。

|case|request|request時刻[s]|K取得age[s]|手先modal差[mm]|手先補正平均差[mm]|最大global方位差[rad]|body joint Mahalanobis²|
|---|---:|---:|---:|---:|---:|---:|---:|
|marker|925|179.758538|146.826737|384.230|384.154|0.525782|351083.006|
|imu_marker|863|179.700738|168.199640|220.382|220.413|0.581412|3804654.004|

Markerは原body job106から実647区間、IMU併用は原job442から実387区間を輸送し、全candidate identity（state ID・全数値law hash・質量・source・ledger・K参照）が元制御候補と完全一致した。ledger全prefix-chain SHAも検証した。比較するFKは元action_jsonに捕捉されたauxiliary_equilibriumをURDFで一度評価したもの。

Mahalanobis²は身体と取付の全60座標・rank36の周辺Gaussianを使い、body間の全crossを保持している。環境は周辺化する。両例のaffine支持残差は約1e−15であり、支持外を有限密度として扱った値ではない。各local/globalの全共分散・補正平均・残差は各caseの診断JSONに保存した。方位補正平均はcoreのmodal姿勢まわりの局所座標であり、単一のglobal平均回転とは呼ばない。

この差は「現在の推定身体と、捕捉したKの絶対平衡FKが一致していない」ことの証拠である。Kの遅延だけが原因、またはこの差だけが手先振動の原因とは断定できない。Kモデルは固定局所並進を持つ6関節平衡系だが、身体候補は各局所変換の全自由度を持つため、Kの変更だけでは表せない身体補正もあり得る。実際IMU併用のlink6局所並進差は15.52mmである。

現在のK推定は既に解析的に平衡式を微分し、全linkの共同posterior/prior密度比のscoreを逆伝播している。現在の身体posterior自体の尤度だけを最大化するものではない。全履歴・BFSを含む連続K密度と、その代表値の既定1回score更新を区別する必要がある。現在posteriorを改めて独立因子として追加すれば、過去情報を二重使用する可能性があり、診断追加の範囲では規準を変更していない。

新診断はtrajectory requestのgradient-ready後、artifact processで実行する。失敗や支持外はunavailable/partialと記録し、制御拒否・身体再生成・共分散変更・表示補正には使用しない。新10単体回帰と既存3解析感度/score回帰は13PASS（0.74秒）。これは診断の検証であり、新実走の到達や振動解消を証明しない。

絶対姿勢の差と、指令増分に対する平衡感度の精度は別の問題である。本後解析は実走の増分予測精度や、現在の指令変更で振動が解消することを検証していない。補助モデルの既存解析感度FD回帰は別の小さい固定fixtureに対する検証である。

更新規準と既存解析scoreの数式は [STIFFNESS_BODY_CONSISTENCY.md](../../../STIFFNESS_BODY_CONSISTENCY.md) に記録している。再現はROS環境を読み込んだPythonで `reproduce_last_used_request.py --input-run /path/to/original/run --output /tmp/reconstructed_K_body` を実行する。元runの `commands.jsonl`・`robot.urdf`・選定job pickleが必要であり、これらの大きいraw履歴はGitへ複製していない。保存したsource extractは元制御のimmutable identity・捕捉K・prediction path・実使用packetの小さい抜粋であり、観測画像や物理真値を含まない。
