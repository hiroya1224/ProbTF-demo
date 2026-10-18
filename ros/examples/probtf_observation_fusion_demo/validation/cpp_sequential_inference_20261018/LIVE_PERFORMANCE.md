# C++逐次実走の性能帰属監査

対象は2026-10-18の両90秒実走。physical_end_manifestを基準に動作区間を切り、後着のdrainを動作中の成果に含めない。元ファイルはread-only、production変更なし。

## marker

|区間 [s]|実物理区間/s|coordinator CPU [%]|
|---|---:|---:|
|0–10|84.50|82.1|
|10–30|72.80|89.2|
|30–60|34.17|101.7|
|60–90|25.93|101.7|

90秒全体の既存明示scopeは、指令計算 5.840秒、凍結K作用 2.718秒、C++物理＋端点FK 6.562秒。残り 74.879秒はこの3scopeの外で、特定の関数へ割り当てる計時がない。

- task_and_view_gradient 中央値 [0–30,30–60,60–90秒]：0.0608, 0.0615, 0.0617 秒。
- capture_K_and_reanchor 中央値 [0–30,30–60,60–90秒]：0.0064, 0.0098, 0.0125 秒。
- stiffness.evidence.keep 中央値 [0–30,30–60,60–90秒]：0.4420, 1.5103, 1.6784 秒。
- K_representative_selection 中央値 [0–30,30–60,60–90秒]：0.0407, 0.0950, 0.1127 秒。
- snapshot_artifact_serialization 中央値 [0–30,30–60,60–90秒]：0.0607, 0.0802, 0.0889 秒。
- control_record_json_write 中央値 [0–30,30–60,60–90秒]：0.0188, 0.0312, 0.0377 秒。

## imu_marker

|区間 [s]|実物理区間/s|coordinator CPU [%]|
|---|---:|---:|
|0–10|65.80|90.3|
|10–30|58.45|92.7|
|30–60|31.60|101.2|
|60–90|24.77|101.8|

90秒全体の既存明示scopeは、指令計算 4.790秒、凍結K作用 2.339秒、C++物理＋端点FK 8.040秒。残り 74.830秒はこの3scopeの外で、特定の関数へ割り当てる計時がない。

- task_and_view_gradient 中央値 [0–30,30–60,60–90秒]：0.0601, 0.0606, 0.0613 秒。
- capture_K_and_reanchor 中央値 [0–30,30–60,60–90秒]：0.0073, 0.0103, 0.0124 秒。
- stiffness.evidence.keep 中央値 [0–30,30–60,60–90秒]：0.4557, 1.4824, 2.5322 秒。
- K_representative_selection 中央値 [0–30,30–60,60–90秒]：0.0414, 0.0859, 0.1228 秒。
- snapshot_artifact_serialization 中央値 [0–30,30–60,60–90秒]：0.0661, 0.0844, 0.0947 秒。
- control_record_json_write 中央値 [0–30,30–60,60–90秒]：0.0224, 0.0337, 0.0404 秒。

coordinator後半のprocess CPUは約1コアを消費しており、単なる待機だけでは説明できない。一方、完全task/view勾配とK capture/reanchorの処理時間は概ね一定で、これらが時間とともに数倍へ膨らんだ証拠はない。

保持履歴の大きいsnapshot・JSONの出力には成長がある。既存_freeze_arraysを保存JSONへ適用した別processの補助測定では、lateのstiffness subtreeは20/22ms、全保存JSONは26/32msだった。これはruntime ndarray構造そのものを測った値ではなく、また実coordinatorの1.3–1.4秒のCPUを伴う隙間を単独で説明しない。

Queue受信とunpickle、候補の再freeze/adoption、manifestとDB統計、取得spool、lock、GCの直接の内訳計時は今回のsourceにない。そこを測らず、どれか一つを根因と断定しない。KのKeep積分は履歴とともに重くなり、後半中央値1.68/2.53秒で別途backlogを作るが、新しい身体候補はK完了を待たず採用されている。

stageには入れ子があるため、異なるstageのtotalを足して全時間としない。stageの壁時計区間境界はmin(sample_wall−captured_frame_elapsed)から復元した上界であり、元ログにないreader/frame lagの誤差を含む。raw・snapshot・動画の全件をGitへ追加する作業は行っていない。
