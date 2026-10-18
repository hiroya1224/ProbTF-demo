# 1244f3fのRViz再実行と接近効率の解析

実走sourceは `1244f3fcd838165b040fadc701f6b00f47ada7f6`。MarkerとIMU併用を初期状態から各180秒で動かした。全source/native/RViz設定ファイルは凍結一致し、物理180秒と、その後の60秒推定drainを分けて記録した。両方とも未到達である。

- [主報告](../../MOTION_EFFICIENCY_AUDIT_20261018.md)：速度制限、接近効率、完全位置尤度の共分散微分による離反例。
- [実行要約](rerun_summary.json)：終端・最小距離、実Hz、全raw保存と身体処理、未完K、新診断の非停止動作。
- [実軌跡](motion/REPORT.md)：全物理区間の接近／接線変位、捕捉K作用、物理応答遅れと検証範囲。
- [規準の分解](objective/REPORT.md)：同じ6候補の完全勾配再現、位置mean/covariance・正規化・方位・viewの寄与。
- [局所モデルのスペクトル](control_damping_spectrum.json)：採用済みpacketの固定モデルに対する減衰比。全閉ループの安定性保証ではない。

終端の2画像は直接RViz内容領域のcapture。読込設定ファイルのバイト一致と、実行中のユーザーによる未保存のビュー操作は別であり、画像を元の視点へ修正していない。RViz本体と専用masterは終了後も残した。

数値ソース、指令・時計・候補IDとSHAを各JSONへ保存した。後解析に必要な大容量raw/DB/全JSONL/pickle/NPZ/native binaryは元の `/tmp/probtf_1244f3f_rviz_rerun_20261018` と各解析ディレクトリに保持する。再現scriptはこれら原アーカイブを必要とする。Gitに保存したのは選定済みの小さい報告・数値・図だけである。今回の作業は実行と解析・文書のみで、制御・候補分布・推定・物理・ゲイン・速度制約を変更していない。
