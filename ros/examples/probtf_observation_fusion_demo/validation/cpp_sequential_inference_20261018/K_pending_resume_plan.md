# 保存済みKジョブの継続計画

物理90秒を再実行せず、保存済みの選択観測に対応するKジョブを処理できる。
この検討では本番コードを変更せず、追加のK計算も開始していない。

## 既存保存形式から分かる範囲

`stiffness_worker` は `StiffnessIdentifier` を初期化し、job sequence 0からの
連番だけを受け付ける。終了時のidentifierそのものや、再開可能な数値checkpointを
保存する機能はない。`FrozenKPacket` は記録と指令用代表値のJSONであり、
連続密度を評価する全因子のオブジェクト・取得chart・BFSの束縛関数を復元するAPIではない。

`stiffness_jobs/*.pickle` は観測前後の全身体law、取得時刻・指令・元ID、実ledgerを
保持する。対応するJSONに連番、source candidate ID、payload SHA256、バイト数がある。
`load_stiffness_job` は読み込むたびにpayload hashとsource/FIFO identityを検査し、
workerはさらに実際の候補からcandidate identityを再計算する。

今回の保存入力はmarker 206件（0〜205、432,777,985 bytes）、IMU併用692件
（0〜691、1,363,532,075 bytes）。連番、対応ファイルの存在、保存バイト数は一致し、
書き込み途中のtmpファイルはない。全payloadのhash再読込は計画段階では行っていない。
実際の再生時には全件を既存loaderで検証する。

coordinatorが受け取った完了数と、shutdown直前にworkerが保存し終えた数は異なる。
post-drain manifestはmarker 72件、IMU併用81件だが、runtimeにはそれぞれjob72、job81までの
計算完了がある。次のjob73/job82の途中で終了している。原動作に採用されたKを
後から完了した推定で書き換えてはならない。

## 推奨する既存経路

1. 元runのfrozen Python source、frozen native library、同じURDF・設定を使う。
   source/backend SHAを元manifestと照合する。物理更新器やカメラ取得器は起動しない。
2. 元archive参照をsequence 0から末尾まで並べ、同じ`StiffnessIdentifier.update`と
   `command_proposal`を、原イベントごとに一度ずつ実行する。
3. 各入力は元loaderのhash検査と、workerと同じ候補law/source照合を通す。
   初回joint job0は初期化だけであり、架空のK観測を追加しない。
4. 既計算prefixのfactor ID、BFS density ID/log mass、代表値を元結果と比較する。
   これは別実行の検証であり、同じIDだけを根拠に数値一致を仮定しない。
5. 結果は別のoffline-drain出力へ保存する。元DB、指令、物理、live adoption記録には
   変更を加えない。全選択ジョブの完了と、全raw融合を区別する。
6. 任意の監査toleranceや時間制限で有限な更新を不採用にしない。
   必要演算の実失敗時は、失敗ジョブ・入力・未処理suffixを残す。

既存workerへそのまま全参照を渡す方法は数値経路を最も変えない。ただし毎回の
全履歴JSONの重複保存は増える。保存の圧縮や軽量checkpointを作る場合も、
入力・連続密度の内容を捨てず、保存形式の変更として別に検証する必要がある。

## 正確なcheckpointを新設する場合

代表値だけをcheckpointと呼べない。少なくとも全frozen factor（全Gaussian/crossと
chart・branch anchorを含む）、BFSの全updateとlog mass・event signature、消費済みID、
`last_stamp/key/record`、`_bfs_evidence_modes`、`_proposal_representative`、
途中pending factorの有無を保持する必要がある。
新identifierの機械モデルを同じURDFから構築し、BFSのfactor関数を新ownerへ束縛し直す。
可変Pinocchio/native workspaceを共有・そのままpickleしない。
数値cacheの省略はlawの省略ではないが、復元後のlog密度・score・次の更新が
連続実行と一致するテストが必要である。今回はこの機能を未実装として扱う。

## 所要時間の見込みと限界

最後の10件のevidenceと代表選定の合計平均はmarker 4.23255秒、IMU併用3.11020秒。
この速さが一定という仮定なら、未完了133/610件は約9.4/31.6分、job0から全件は
約14.5/35.9分となる。これは上界でも完了保証でもない。
全連続因子履歴が増え、起点・mode・各数値照会数も変わるため、後半は長くなる可能性がある。
再生の開始・進捗と実測を記録して更新する。既動作のK推定がリアルタイムで完了したとの
主張に転用しない。

数値一覧: `/tmp/probtf_cpp_sequential_K_resume_inventory_20261018.json`。
