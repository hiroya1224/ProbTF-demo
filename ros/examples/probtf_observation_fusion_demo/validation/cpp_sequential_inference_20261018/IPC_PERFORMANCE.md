# 推定結果の受信処理に残った遅滞

この資料はC++逐次版の90秒試行後に行った追加調査である。身体候補や観測モデルを変更せず、不変な記録のプロセス間転送を調べた。C++の数値処理の速度、全推定の速度、物理更新の配信頻度を区別する。

## 変更前の実coordinator診断

独立したROS masterでIMU＋markerを物理35秒動かし、元の`AsyncMotionLive.advance/publish`のtimer threadに外部からcProfileを適用した。推定・制御・数値停止条件は変更していない。終了後の推定待ちは別に記録した。この実走にはprofilingの負荷があるため、通常動作の周波数の測定として扱わない。

`advance`の累積34.35秒のうち、Queue受信のunpickleが10.97秒を占めた。その内部で不変JSONを再構築する`freeze_json_record`が1,032万回、9.46秒呼ばれていた。受信後の`_freeze_arrays`は別に1,215万回、10.40秒だった。呼出元では身体候補の受理6.39秒、表示artifactの受理3.56秒が大きく、勾配packetの受理は0.445秒だった。

動作中のgeneration 2 GCは35回、累積4.98秒、最大362 msだった。これらのpauseは上の呼出時間に含まれ得るので加算しない。`publish`の31.18秒のうち28.23秒は関数self時間で、shared lockの待ちを含む。保存された候補の受理が物理更新と配信を待たせる構造と整合する。

詳細は[変更前profile](coordinator_ipc_before_profile.json)。大容量のraw、DB、pickle、pstats、JSONLは/tmpの診断runに残し、本フォルダへ追加しない。

## 同じ候補を保つ修正

不変JSONの各子はpickleの復元時点で既に復元済みである。親を復元する際に、子の全数値まで再帰検査する処理をやめた。初回のJSON構築時には、全子の型を従来どおり検証する。普通のdict/listと等しいJSON表現、元の数値、共有する子への参照、不変性を保持する。旧reducerを使う保存pickleも読める。

配列のreadonly flagを復元するときは、型としてnumpy配列を含まないことを保証した不変JSONの部分木を再走査しない。普通のdictや候補state/queryにある配列は引き続き走査してreadonlyに戻す。候補、全交差共分散、出自ID、数値hash、表示wireを省略しない。GCの無効化、履歴の削除、新しい数値停止条件は加えていない。

## 同じ実archiveによる転送比較

保存済みの同じ身体候補3件を使い、各実装で符号化し直して7回ずつ復元した。元ファイルの読み込み、試行間のGC、候補同一性の検査は計時外である。復元結果のcandidate identityと全記録JSONのSHAは全試行で一致した。

| 対象 | 復元の中央値・変更前→後 | 配列走査の中央値・変更前→後 |
|---|---:|---:|
| archive 1 | 15.90→7.31 ms | 65.27→54.69 ms |
| archive 2 | 15.14→6.79 ms | 64.06→54.18 ms |
| archive 3 | 14.93→6.53 ms | 62.81→52.51 ms |

復元は約2.2倍、走査は約16%短くなった。この比較は転送処理だけの測定である。普通の可変記録には引き続き走査が必要であり、全coordinatorが同じ倍率で速くなるとはいえない。全身体推定、K evidence、到達性能の改善の証拠でもない。

入力archiveのhash、候補とJSONのhash、対象sourceのhash、各計時は[変更前](immutable_transfer_before.json)と[変更後](immutable_transfer_after.json)に保存する。再計測には`validation/tools/benchmark_immutable_transfer.py`を使う。

## 同じ外部profileによる変更後の確認

同じwrapperを使う35秒の別実走を行い、両試行で物理終了・exit 0・実走中のsource不変を確認した。全回帰テストは1,056件通過した。productionの差はIPCの2ファイルと、前述の時刻被覆を直したselector v2である。

| 計時scope | 変更前 | 変更後 |
|---|---:|---:|
| Queue unpickle | 10.970 s／3,299回 | 5.907 s／5,310回 |
| 受信後の配列走査 | 10.396 s／1,215万calls | 4.752 s／548万calls |
| 身体候補受理からの配列走査 | 6.387 s／325件 | 1.066 s／322件 |
| 表示artifact受理からの配列走査 | 3.563 s／123件 | 2.898 s／193件 |
| generation 2 GC | 4.980 s／35回 | 6.279 s／35回 |
| generation 2 GCの最大pause | 362 ms | 500 ms |

旧`freeze_json_record`による再帰復元はなくなり、受理数が増えてもその部分と身体候補の再走査は減った。一方、通常記録の走査と保持グラフに対するGCの費用は残る。GCを無効化したり、履歴を捨てる修正は行っていない。

profile付きの物理更新率は約22.0→49.1 Hzだった。観測選択・スケジューリングも異なるため、これをIPC変更単独の速度倍率や通常動作の性能と扱わない。変更後も物理終端でKは323 job中39件完了にとどまり、全Kのdrain・実時間性能・到達は未達である。入れ子の計時とGCを重複加算しない。詳細は[比較記録](coordinator_ipc_before_after.json)。
