# 完全規準が選んだ移動方向の分解（1244f3f実走）

MarkerとIMU併用の100〜180秒付近から、実際に指令へ使われた各3requestを選んだ。原canonical身体archiveと原全ledgerから各snapshotを復元し、全candidate identityが元制御候補と完全一致することを検証した。元task勾配6成分と、実行器のg_source+H(q−q_source)も全例で差0だった。新しい観測更新、K密度計算、指令実行、規準変更は行っていない。解析に用いた10個のproduction/configファイルは、実走のfrozen sourceとSHAが一致した。

## 位置の共分散変化によって、距離を増やしても完全Jが下がる例

Marker request848の実指令interval（158.443〜158.474秒付近）では、同じ保持lawでの補正平均距離が3.745 mm/s、modal距離も3.660 mm/s増える方向へ動いていた。別の後解析による原物理距離も7.156 mm/s増えていた。一方、同じlawでcurrent qを評価した完全Jの変化率は−1592.967/sである。viewは零だった。

位置部分で d=mu−y とすると、

`J_pos = 0.5 d^T S^{-1} d + 0.5 log det S + const`

`dJ_pos = (S^{-1}d)^T dmu - 0.5 (S^{-1}d)^T dS (S^{-1}d) + 0.5 tr(S^{-1}dS)`

である。二番目はMahalanobis二次項の中にある共分散微分で、log determinantとは別である。

|request848・実qdot方向|Jの変化率[/s]|
|---|---:|
|位置quadraticのmean変化|+273.169|
|位置quadraticのcovariance変化|−1713.656|
|位置logdet|+0.392|
|方位Bingham quadratic|−153.196|
|方位Bingham normalizer|+0.324|
|view|0|
|合計|−1592.967|

距離を悪化させるmean変化よりも、残差方向の共分散変化による尤度改善が大きい。sourceでの共分散由来位置勾配normは140166、mean由来は43923、方位quadraticは19304である。これは「回転やnormalizerだけを優先した」「内向き指令がK作用によって横へ曲がった」という説明とは異なる。正規化項も含めた、現在指定されている完全規準自体の具体的な方向配分である。

このrequestの実行local gradientと、current qで再評価した完全selected gradientのcosは0.996905だった。保持した局所二次式の方向誤差だけで、この外向き区間を説明する根拠は弱い。ridgeの復元力normは770、source selected gradient normは123467であり、この例の元の選定方向をridgeが作ったわけでもない。ridgeはJに加わる損失ではなく、local Hに加わる項である。

## 他の選定例と推定方向・物理方向の区別

|case/request|sample時刻[s]|推定平均距離[mm]|推定距離変化率[mm/s]|実物理距離変化率[mm/s]|current完全g/local gのcos|
|---|---:|---:|---:|---:|---:|
|marker/634|112.582|254.161|-30.515|-56.372|0.997497|
|marker/848|158.474|195.157|+3.745|+7.156|0.996905|
|marker/927|178.976|222.960|-1.936|-2.919|0.999102|
|imu_marker/637|123.575|93.398|-18.315|-18.141|0.861287|
|imu_marker/703|140.010|65.819|-0.761|-0.152|0.998996|
|imu_marker/854|179.013|46.762|-0.578|-0.200|0.999943|

位置quadraticの勾配は全6例で最大級である。logdetとBingham normalizerの勾配normは約2〜35で、位置quadraticの約3.3万〜79.6万より小さい。方位quadraticは1.0万〜3.5万で無視できず、特に終盤IMUでは位置項と同程度である。viewは4例で零、Marker634ではnorm599、Marker927では9111だった。従ってviewが全区間の接線運動を説明するとは言えない。

sourceの位置mean補正normは0.050〜3.404 mm、modal/補正平均の距離微分は選んだ6例で同符号だった。この小範囲では、補正平均だけが接近と解釈してmodalが外へ動くという逆転は見つかっていない。これは全時系列のclosure保証ではない。

## Fisherを方向のmetricに使う案も、診断だけでは一般改善しない

同じsourceで、−g、−H^{-1}g、実速度方向をL2 norm=1に揃えて比較した。H/g/law/Kは元snapshotのものを使った。両候補とも完全selected Jの一次slopeは全例で負だったが、平均距離のslopeは次の通りだった。正が外向きで、単位はm/radである。これは微小方向の診断であり、実行可能な位置・速度・有限軌道を提案したものではない。

|case/request|−g|−H^{-1}g|
|---|---:|---:|
|marker/634|-0.45754|-0.11279|
|marker/848|+0.19562|+0.05785|
|marker/927|-0.29709|-0.03984|
|imu_marker/637|-0.69112|-0.03810|
|imu_marker/703|-0.14998|-0.15813|
|imu_marker/854|-0.28142|-0.13231|

Marker848は前処理しても外向きのままだった。他の多くでは接近slopeの絶対値が減った。従って、この比較からFisher前処理の一般的な接近改善は主張できない。modal/補正meanそれぞれのradial/tangent、Jのslope、精度行列・固有軸・残差の固有軸成分はsummary.jsonに保存している。

## 範囲と再現

本分析は指定済みの完全Jを分解したものであり、項の削除・weight変更・共分散微分の停止・位置非増大制約の復活・clip・新しい制御器は実装していない。全native中央mixtureを使ったが、この6件の実中央lawは各1成分だった。

再現には元runのcontrol.jsonl/commands.jsonl/gradient_adoptions.jsonl/robot.urdfと、指定したimmutable body archiveが必要である。巨大rawや履歴は本smallreportへ複製せず、source_input_referencesに元path・request・candidate ID・archive SHA・ledger hashを記録した。extract_sources.pyで元requestを選び、decompose_objective.pyで同候補/K/anchorの決定論的差分を再計算できる。検証用差分はモンテカルロではない。物理NPZはdynamics担当が原同indexの保存物理ログから別途計算した後解析で、controllerやJには入力していない。

再現用CLI例: `python extract_sources.py --input-run /path/to/run --output /tmp/objective_inputs`、続いてROS/PYTHONPATH環境で `python decompose_objective.py --input-run /path/to/run --source-dir /tmp/objective_inputs --output /tmp/objective_replay`。取得sourceと現コードが異なる場合は、原1244f3fのfrozen source環境を使う必要がある。
