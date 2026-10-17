# 推定剛性と実行剛性の分離

## 最新訂正：一回の完全IK中の実行Kを固定する

ユーザーの振動指摘により、3回の仮想IKと1.5秒軌道を撤回した。一requestで保持身体・固定目標の全候補と実行Kを捕捉し、完全Jの勾配を一回評価する。選定した次位置q_nextと0.5秒の所要時間を出力し、source時計で出発q_sourceから線形補間した位置を20 Hz目標の物理tickで参照する。0.5秒後も有限q_nextを保持して残りの位置誤差を追い、次planへつなぐ。所要時間は停止期限ではなく、経過だけで零速度を要求しない。遅着採用で補間を数え直さない。追加端点停止速度・通過軸clamp・旧一定速度延長は使わない。

filter sample・推定packet・身体／目標source・全cross・安定平衡branchを一回の全試行中に固定する。coordinatorのK補間・raw取得・非同期推定は独立に進み、次requestが新しいsampleを捕捉する。law・凍結K・位置plan・snapshotをatomicに採用し、実区間と次事前へ同じ作用を使う。元の密度・代表・`command_proposal` は保持する。

既存速度・加速度・関節・停止距離条件、8回backtrackと許容集合内部への修復、完全J／s・PDF・core・RViz設定は変更しない。詳細は [SINGLE_STEP_IK.md](SINGLE_STEP_IK.md) に従う。3更新の仕様と実測は [SHORT_HORIZON_IK.md](SHORT_HORIZON_IK.md) に歴史として保存し、その571テストや一時到達を新方式の滑らかさの証拠にしない。

K dampingの仕様は、deflecompと同じ実行側のlog K補間である。補間は次の推定・IKを待たずcoordinator tickごとに進め、一回のIK中だけ捕捉したsampleを固定する。IK候補ごとにfilterを進めたり、推定代表・身体事後を補間値へ置換したりしない。

## 推定に残すもの

連続密度の全履歴factor、初期分布、正規化未評価という規約、数値代表 `K_est`、その密度とscoreの評価記録・source ID・取得時刻・指令・版を変更しない。`FrozenKPacket` は実際の推定記録をimmutableなJSONとして保持する。元の `command_proposal` も原文のまま残す。実行filterを新しい観測、密度の更新、MAP推定、剛性事後の平均とは呼ばない。

身体・未知地図の保持事後、全交差共分散、目標mixtureと固定根元周辺は既存のまま保持する。補間したKから身体を再生成せず、現在の身体を補助平衡へ再中心化しない。K未選定のケースは既存の宣言済み指令作用を使い、架空のKや別の失敗分岐を加えない。

## 実行filterの式と時刻

`x_est=log(K_est)`、直前の実行値を `x_exec^-` とする。各coordinator物理tickで、直前filter時刻からの実際のdtを用いて、

```text
alpha = 1 - exp(-dt / tau)
raw_delta = alpha * (x_est - x_exec^-)
applied_delta_i = clip(raw_delta_i, -max_step, +max_step)
x_exec = x_exec^- + applied_delta
K_exec = exp(x_exec)
```

を計算する。既定は `tau=0.5 s`、`max_step=0.05`、coordinatorの目標周期は20 Hzである。上限は一tickあたりの無次元log増分であり、毎秒の変化率ではない。遅いtickではdtに応じた一次遅れを計算した上で同じ上限を適用する。新しい推定がなくても、保持した `K_est` を目標にして補間を続ける。推定・IK workerがbusyでもcoordinatorはfilterを進める。

初回の有効推定で `K_exec=K_est` とし、初期化時点から異なる値へ人工的に遅らせない。dt=0では実行値を厳密に保持する。時刻は非減少、Kは正の有限一次元ベクトルとし、次元不一致・不正値を明示的なエラーにする。設定のtau=0はdt>0で遅れを除き、max_step=0は増分capを除く。既定値の式は [deflecompの実装](../deflecomp/deflecomp_core/src/deflecomp_core/pipeline/compensator.py#L608) と [設定](../deflecomp/deflecomp_ros/config/estimator.yaml#L33) に対応する。deflecompのK限界clipを推定支持の制約として持ち込まない。

`LogStiffnessDamper` はnumpyだけを用いる純filterであり、過去のfilter記録から復元・replayできる。`previous` はIK採用済み記録に限らない。filter時刻と採用済みplanの時刻を混同しない。

## 一回のIKで固定するもの

各requestは、その予測開始時刻にcoordinatorの補間sampleをcaptureする。sample時刻と `result.prior.stamp`、sampleが目標にした推定Kと元packetの `selected_K` を一致させる。sampleには推定packet・密度のIDと推定元時刻を結び付けてhashを持たせる。

`FrozenExecutionKPacket` はその元 `FrozenKPacket` と、捕捉したfilter sample・安定平衡branch・現在指令のanchorを保持する。元packetのrecord/proposalは改変しない。前の採用済み実行packetは、安定branchの継続と出自の参照に使う。filterそのものを前の採用時刻へ戻すためには使わない。

現在指令cで捕捉した `K_exec` の平衡を解き、全試行qの作用を

```text
delta_theta(c,q) = theta_eq(q,K_exec) - theta_eq(c,K_exec)
```

とする。補間によりKが変わっても、元のK_estの平衡を流用せず、K_execで安定性とKKT条件を検証する。試行順序によるbranch変更を避け、同じ捕捉anchorから照会する。q=cでは身体lawは厳密に同じオブジェクトを返す。

一回の完全J勾配更新、visual初期候補、全試行、line searchと位置s制約は、同じK_exec・branch・身体／目標lawを使う。計算の途中で最新の推定や補間sampleへ差し替えない。計算中にもcoordinatorの補間信号は進むが、その進行によって計算済みの目的関数や方向を変更しない。次のIK requestは次の開始時刻のsampleを捕捉する。

完成したplanは、law・凍結K・snapshotとatomicに採用する。物理tickが実行する指令区間と次事前への輸送には、採用したplanの同じ作用を使う。実指令ledgerは区間ごとのK_exec・推定密度ID・捕捉branch・source情報を保持する。遅着事後を現在へ運ぶ際にも全実行区間を順に適用し、過去の区間を最新Kや現在の補間信号で置き換えない。

## 履歴：一端点停止方式とその物理未達

実行K分離直後のsource `3dae353d…` には、一回の勾配が選んだ短端点を越えて旧速度を延長する不整合が残った。markerは51.310秒に4.439 mmへ到達したが、その後離れ、最終は8.463 mmだった。IMU＋markerの最終は2.000 mmだった。K段差だけで全ての過走を説明できない。

続くsource `3fc6d95b…` は、短端点までの離散停止速度とsourceから0.5秒の期限で一回の指令を追う方式だった。端点通過後の外向き要求は旧158／260区間から0／0へ減り、531テストが通過した。しかし両60秒試行とも物理到達は未達で、最終は227.890／214.919 mmだった。続く3更新方式も振動指摘により撤回された。現在は一回の位置選定を時間補間し、3fc6方式の追加端点停止速度・通過軸clampは使わない。

元の実行・数理・表示・物理結果は [K補間の歴史検証](validation/damped_execution_K_20261017.json) と [短端点実行の歴史検証](validation/damped_K_short_targets_20261017.json) に保存する。いずれの到達やテスト通過も、現行一回位置参照方式の証拠に読み替えない。

## 数値公開と表示

元の `command_proposal` は推定代表を記述する。実行用のK、平衡と補償指令、現在のbody sourceは別の `execution_command_proposal` に置く。実行Kにおける密度やscoreを再評価したかのような項目は追加せず、実行再anchorを新しい推定結果としてcaptureすることを拒否する。

filter記録は推定K・実行K・両log値、直前実行log値、実dt・alpha・生増分・適用増分、tau・cap、初期化状態を保存する。coordinatorは `execution_stiffness.jsonl` へ各sampleを保存する。IK packetは、そのsampleと推定元のID・時刻・指令、現在anchorのID・時刻・指令・版を別々に保存する。

HUDは `Estimated K [Nm/rad]` と `Execution IK K [Nm/rad]` を分ける。後者は表示しているIK snapshotの凍結作用と一致させる。現在の連続補間値と、最後に採用したIKの捕捉値が計算遅延の間に異なる場合は、両者を同じ現在値と説明しない。実行metadataの `interpolated_execution_K`・その時刻と `active_execution_K` を区別する。

身体・手先・目標の分布表示には従来と同じcore照会を使い、K dampingのための別共分散、別粒子、再中心化を作らない。候補lawの実MarkerArrayをserialize/deserializeしてsource・query・パラメータの同一性を検査し、別公開のHUDでも両Kと同じ実行packetの対応を検査する。core、画像annotation・画像サイズ、粒子・軸・標本数、RViz全設定ファイルのバイトは変更しない。

## 検証の範囲

純filterの43試験とpacket統合の16試験を実施した。前者は式、両方向cap、dt=0、初期化、継続追従、復元と不正入力を確認した。後者は50 msごとの補間、IK capture後のpacket不変、元推定記録・密度の不変、別実行proposal、安定平衡、q=c恒等、同じKによる完全IKと実ledgerの全covariance作用、HUD・実wire、hashと意味の改ざん拒否、実行viewの推定への逆流拒否、モンテカルロ・密度再更新の禁止を確認した。

短目標実行の追加7試験では、実際の完全IK一回更新replyを0.2秒遅れて採用し、source基準の期限、blocked worker中の有限端点・減速・停止、期限後到着での静止、到達／通過軸の非反転、端点までの離散停止条件と加速度優先の不可避な惰行を確認した。同じKのledger、元推定・候補表示wire・gradient diagnosticの不変、実行記録との対応も検査した。これらは旧3fc6一端点停止方式の検証であり、現行の時間付き位置参照の到達・滑らかさの証拠にはしない。

現行source `7a3a0ecb…` は549テストと両60秒の実配信検証を完了し、全386の完全IK、2,271のKfilter sample、2,265の実区間、13,585のcatchup ledger作用で同じ捕捉Kを照合した。時間だけを理由とした未到達目標への零要求は旧283／361区間から0／0となった。最終の物理距離はmarker6.007989 mm／併用2.955592 mmであり、markerは6 mm条件の外にある。詳細と残存振動・絶対姿勢誤差は [SINGLE_STEP_IK.md](SINGLE_STEP_IK.md) と [最終数値記録](validation/position_hold_ik_20261017.json) に記録する。

旧runで報告されたKの約1.61倍の変化は、実行Kを分離する動機にはなるが、それだけから指令の跳ねの全原因を断定できない。Kの連続化だけで非線形IKの連続性や到達が保証されるとは説明しない。旧sourceの到達や未達を本変更の性能結果に読み替えない。
