# 伝播モーメントを逆伝播するGaussian変分更新

2026年10月18日の「ベイズで沿わせて、逆伝播で修正する方針」に対応する。観測前の分布を保持し、一件の観測から得る事後をfull Gaussianで近似する。各試行Gaussianを既存の高集中Bingham由来の式で伝播し、その補正平均・全共分散を通して中心と全共分散を微分する。固定共同Gaussianの単回条件付け、候補一点のMAP/GN、身体Laplace近似とは区別する。

続く明示指示「高集中の基準は『勝手な』基準です。取っ払ってください」に従い、0.1 rad²による適用域の停止を撤去した。Python coreとデモC++の両方を変更したが、伝播式・Bingham二次形式・平均・全共分散・全crossの式は変えていない。相対modal-gap閾値も撤去し、正のmodal gapと回転共分散の必須逆計算には数値rankによる切捨てを使わず、正則化なしのCholesky・solveを用いる。大きくても有限な幅はそのまま計算する。実際の非一意mode、実非正定値、非有限値など必須演算が成立しない場合とは区別する。一般の共同Gaussian/Dirac支持の既存規約は維持する。大きい幅で近似精度が保証されるという主張ではなく、近似精度を終了基準にしない変更である。

## 固定するものと最適化するもの

原取得時刻・実指令でのKeep/Reset各親の共同事前を $p_h^-(z)=\mathcal N(z;m_h,P_h)$ とする。身体・カメラ・未知マーカー・既知取付の元chartと全crossを保持する。原観測の各native成分 $k$ の尤度は、既存の固定観測chartで $L_k(y_k\mid F_k(z))=\mathcal N(y_k;F_k(z),R_k)$ とする。原観測の成分・重み・log weight・位置方位crossを変えない。

変数は候補一点 $z$ ではなく、共同Gaussian $q_{hk}(z)=\mathcal N(z;m,P)$ の平均 $m$ と全共分散 $P$ である。各試行で同じ元chartからリンク候補を再表現し、その同じlawに対して既存の伝播式を再評価する。根元からEEへの経路ならEE分布、実観測がIMU/画像ならその重力方向/カメラ相対マーカー分布を評価する。task目標を身体の観測として使わない。

Bayes事後 $p(z\mid y)\propto p_h^-(z)L_k(y_k\mid F_k(z))$ への逆向きKL最小化は、定数を除いて $\mathcal F_{hk}(q)=\operatorname{KL}(q\|p_h^-)-\mathbb E_q[\log L_k(y_k\mid F_k(Z))]$ の最小化である。Gaussianの族を選び、この規準のパラメータ微分で求めるGaussian変分近似である。[Blei, Kucukelbir and McAuliffe, *Variational Inference: A Review for Statisticians*, JASA 2017](https://arxiv.org/abs/1601.00670) のKL/ELBOによる定式化に対応する。mean-field独立化や同論文の確率的勾配を採用するものではない。

## 伝播式を使う期待尤度

試行 $q$ の伝播平均・共分散を $\mu_k(q),S_k(q)$ とすると、Gaussian尤度の期待負対数は $\frac12[(\mu_k-y_k)^\top R_k^{-1}(\mu_k-y_k)+\operatorname{tr}(R_k^{-1}S_k)+\log\det R_k+d_k\log(2\pi)]$ である。複数の独立センサを同時に診断する場合は各尤度の対数を足すが、試行qは共通のfull共同分布である。身体をセンサごとに独立コピーして掛けない。

既存coreの高集中近似で得る $O(P)$ 補正平均と全crossを含む共分散を使う。平均だけをmodal姿勢へ代用せず、trace項・観測の正規化定数・KLのlog determinantを落とさない。候補の位置/方位平均、Jacobian、平均補正、全P、そのchart輸送、固定native観測chartへの輸送を毎試行評価し、すべてを逆微分する。モンテカルロは使わない。

厳密な伝播モーメントが分かる場合、この二次式はGaussian尤度の期待値そのものである。実装は高集中モーメント近似を用いるため、非線形の場合の規準を厳密なELBO下界と保証しない。これを診断に明示し、最適化の有限反復・単峰近似・最終canonical化の誤差と区別する。

## 全共分散とDirac支持

固定事前の実支持平方根を $B$、その階数を $r$ とし、$m=m_h+Ba$、$P=BLL^\top B^\top$ と白色化する。$L$ は正の対角を持つfull下三角行列で、対角はlog値で表す。KLは $\frac12[a^\top a+\operatorname{tr}(LL^\top)-2\sum_i\log L_{ii}-r]$ となる。Dirac座標を広げず、全下三角要素を更新してリンク間crossを残す。共分散へのfloor・jitter・軸ごとのclipを加えない。

KLは元の事前chartで評価する。伝播のための再表現で使う $G(m)PG(m)^\top$ を、元priorのPと直接比較してKLを作ってはならない。逆微分したm/Pの勾配を白色化とCholeskyの連鎖律へ渡す。一回の最適化中は元priorを固定し、次の新観測が来たときだけ今回の事後を次priorへ渡す。

過去の観測情報は、この引継ぎ事前の平均・全共分散・crossを通して次回へ作用する。単峰近似を繰り返すため、過去の全尤度を厳密な形で再現できるという意味ではない。既存BFSのResetは宣言した変化仮説であり、新観測だけで常に再初期化する方式とも区別する。raw取得記録とKの全履歴因子は既存どおり別に保存する。観測との不一致が大きい場合、Bayesに沿う近似でも事後平均が大きく動き得るため、この変更だけで跳ねの消失を保証しない。

初期分布候補として、同じ固定priorの凍結Gaussian観測代理による解析条件付けも計算する。元priorとこの候補を同じ非線形変分規準で比較し、良い方から全mean/covの最適化を開始する。これは以前の単回条件付けを最終更新として実行するものではなく、別priorへの差し替えでもない。線形Gaussianではこの候補が厳密な変分最適解となり、不要な反復を避けられる。非線形では元priorより悪い候補を初期点に採用せず、以後の各試行で伝播と勾配を再評価する。明示した0反復設定では元priorを保持する。

## 全観測枝とevidence近似

明示的な変化親hとnative成分kを残し、各枝で上のGaussianを最適化する。枝質量は $\alpha_{hk}\propto w_h w_k\exp[-\mathcal F_{hk}(q_{hk})]$ とする。これは離散枝を残す構造化変分近似であり、最大枝選択ではない。小さい通常weightがunderflowしても有限log weightと元枝を捨てない。

各枝の $-\mathcal F_{hk}$ を近似log evidenceとして同じ規約で比較する。線形Gaussian観測では最適Gaussianが真事後を含むため、通常の解析事後・正規化evidenceとの一致を検証する。非線形では高集中モーメントと有限最適化を含む近似であり、身体BFSの質量やK因子を厳密なBayes値と呼ばない。

最適化した原chart Gaussianを既存のFrozenBodyBranchへ保存し、元のKeep/Reset混合事前をK比の分母に使う。観測後は既存の全within/betweenとcrossを含むcanonical単峰化を一回行い、同じ結果を次prior・制御・K比の分子・表示が使う。最適枝の観測整合と最終canonical後の整合を別々に評価する。

## 固定目標の周辺分布と身体cross

画像欠落時に固定目標の根元周辺を保持する処理は、身体への新しいBayes観測ではなく既存のcutである。旧Schmidt式 $C_{BE}^{+}=P_B^{+}(P_B^-)^+C_{BE}^{-}$ は $P_B^+\preceq P_B^-$ という収縮を前提とする。Gaussian変分事後ではKeep内部でも膨張や相関方向の変化が起こり、この式のままでは共同共分散が非PSDになる場合を回帰テストで確認した。事後の分散切断やcrossの零化で修復しない。

新methodでは固定事前支持内の対称Gaussian周辺輸送 $A=(P_B^-)^{-1/2}[(P_B^-)^{1/2}P_B^+(P_B^-)^{1/2}]^{1/2}(P_B^-)^{-1/2}$ を使い、$C_{BE}^+=A C_{BE}^-$ とする。逆行列と平方根は実支持上で扱う。$A P_B^- A^\top=P_B^+$ なので、元joint Gaussianの身体側だけをAで写す合同変換から共同PSDが従う。身体の新周辺、目標の既存周辺・質量は変えず、追加の物理過程雑音も導入しない。

BFSではKeep枝のGaussian周辺へこの輸送を適用し、Reset枝は既存の独立redrawによる零crossを使う。両枝の固定目標平均・周辺は同じなのでbetween-branchの身体/目標crossは零であり、Keep質量 $\kappa$ と既存のcanonical body chart $G_B$ によって最終crossは $\kappa G_B A C_{BE}^-$ となる。身体の全within/between共分散は既存projectionのままである。

この結合は周辺だけからBayes的に一意に決まるものではなく、固定目標cutの明示した規約である。新VIでは収縮時も同じ滑らかな輸送を使い、収縮/膨張の品質閾値で旧式へ切り替えない。旧比較methodとhelper既定のSchmidt式は歴史比較用に残す。支持外の情報を創作せず、輸送のroundtrip誤差を診断として記録する。

## 検証

線形厳密解、事前支持、全mean/cov微分、逐次二観測の情報蓄積、原sourceの保持、全native/変化枝、欠測恒等性、推定/制御/表示の同一性を検証する。IMU併用ケースの保存画像job22の再合成不一致を同じ原prior/native入力で比較し、物理truthを入力へ使わない。C++の式と既存coreのmean/covを比較する。検証後、同sourceの両RVizを各180秒で実行し、到達/推定精度/遅延を別に報告する。

閾値撤去前の全回帰は1219件通過した（220.79秒、既存hppfclの非推奨警告1件）。当時の原ログと194ファイルの数値実装・設定・テストのSHAは [検証資料](validation/variational_moments_20261018/) の `before_threshold_removal` 名の記録に保存した。nativeのABIは1で、ビルドは `catkin build probtf_observation_fusion_demo --no-deps --jobs 2 --make-args -j2` で行った。この1219件は閾値撤去後の検証へ流用しない。RVizには変更を加えていない。

手先IMUの旧回帰は、二観測目でも全リンクの回転分散が必ず減ることを要求していた。実際には全6リンクのtraceが前回比0.13〜0.47%増え、Keep枝の規準は−5.894944から−6.260462へ改善し、勾配最大値3.05e−7となった。この単調収縮は新Gaussian変分更新の性質ではないため、回帰を元priorの厳密な引継ぎ・全6リンクへの観測感度とcross・実更新・同じ規準の改善へ改めた。推定と表示の厳密な同一性の検証は保持した。productionの分散を狭めて旧検査へ合わせたものではない。

旧180秒試行の画像job22を、同じ元priorとnative観測だけで比較した。既定24反復では、旧再合成位置残差55.139 mmのReset枝が1.081 mm、旧15.318 mmのKeep枝が0.841 mmとなった。100反復では各0.111 mm、0.587 mmである。全ケースで指定勾配toleranceには未達で、非線形の収束・大域最適性の証拠にはしない。最終原chart Gaussianのoptimizer内モーメントと、既存coreで再表現・再照会したモーメントは平均差1.5e−15以下、共分散差3.4e−18以下だった。元priorは旧演算子の履歴から得たものなので、この単一更新の結果と新方式を初期状態から実行した結果を区別する。詳細は [固定観測の比較](validation/variational_moments_20261018/fixed_observation/REPORT.md) に記録した。

最初の新方式の実走は、Marker33.843秒／IMU併用33.464秒で終了した。IMUの原イベント `imu:00002386`（取得時刻約29.935秒）でKeep事後の原chart回転分散がlink_2で0.123715394 rad²となり、最適化後の再表現で旧0.1上限が例外を発生させた。有限なGaussianに対する近似適用域の停止であり、NaN/Infや必要な逆行列が計算不能だった事象ではない。元prior・観測・全cross・実8区間ledgerと最適化結果を保存し、同じ事象を再現した。ユーザーの最新指示による閾値撤去後は、この同じm/Pを変更せず再表現・伝播できることを検証する。約34秒の失敗試行を180秒の完走や到達成功として扱わない。

撤去後のcore・デモ全回帰は1263件通過した（212.04秒、既存hppfcl警告1件）。原ログは [full_tests.log](validation/variational_moments_20261018/logs/full_tests.log)、当時の197ファイルの凍結SHAは [regression_sources_before_optional_diagnostics.json](validation/variational_moments_20261018/regression_sources_before_optional_diagnostics.json) に保存した。coreとnativeの有限幅0.099/0.101/0.2/2/20/1e15、微小な正modal gap、全crossの保持、実非正定値・実非一意modeの区別を検証した。中央ボードも有限幅を理由に全枝fallbackへ移さず、元native全枝の記録・質量と同じgoal/表示を検証する。任意のlog-gap refinementが零gapを表現できない場合は、元の有限Binghamを保持して診断付きで継続する。

## 実走で見つかった接続不具合

閾値撤去後の第2試行は、原画像 `image:00000127` で未使用の条件付き目標診断のSchur差が−1.1975683e−12となり、約2秒で停止した。この経路ではfresh goalを別に構成するので、その診断用lawを下流は使わない。実目標を使う場合の既定処理は残し、未使用時の構築だけを省略した。元prior、最適化結果、canonical全mean/P/crossはbitwiseで維持した。[修正と検証](validation/variational_moments_20261018/optional_diagnostics/REPORT.md) を参照する。

第3試行は原画像 `image:00002388`、約30秒で、Reset分布をKeepの幾何chartへ表記するときKeep事前の確率支持を強制し、停止した。共通の座標で表すこととKeep事前へ条件付けることを混同していた。デモ内の `materialize_state_in_chart` は既存の変換・全GPGᵀ・Bingham再表現と同じ式を使い、参照する別lawの支持を制約に加えない。各元priorに対するBayes更新・元raw source照合、K密度比に必要な実身体支持条件は維持する。原Gaussian・全枝・微小log質量・時刻・指令・実ledgerを変えず、canonical、fresh goal、K分子、表示まで接続した。[原事象の再現](validation/variational_moments_20261018/common_chart/REPORT.md) を参照する。

この二つの接続修正後、200ファイルを凍結し、関連280件が42.92秒で通過した。元の全1263件とはsourceが異なり、重複する件数を加算しない。coverageとsourceの対応は [common_chart/regression_coverage.json](validation/variational_moments_20261018/common_chart/regression_coverage.json) に記録した。テスト通過だけでリンク推定や到達の改善を主張しない。

## 180秒の実動作と残る不整合

第4試行は両ケースとも180物理秒を完了した。実棒先端の終端距離はMarker213.789 mm、IMU併用38.740 mmで、到達判定は両方falseだった。設定は物理120 Hz・画像12 Hzのままで、実測は物理36.954/33.742 Hz、画像10.353/10.041 Hzである。180秒終了後の推定drainで別の実装エラーが発生し、全観測・K処理が完了した試行ではない。選択済みの全raw・実ledger・未処理の出自と、保存が間に合わなかったrawの復元範囲を [実走報告](validation/variational_moments_20261018/live/REPORT.md) に明記する。

drainの原IMU `imu:00006013` では、数学的には対称なBingham modal gapの内部行列積に約5.63e−17相対の丸めskewが生じ、C++だけが対称化前の許容判定で停止した。対称部の固有値は全て正で、必要な逆計算は可能だった。内部の二次形式をPython coreと同じ順序で対称に組み立て、元の全Pと外部入力の検査を保持した。修正後、同じ元prior/native/最適化m/P/Fをbitwiseで維持したまま、この原イベントからcanonical・保持目標・223,133-byte wireまで再処理できた。旧180秒実走と修正後の一件replayを区別し、全残件drainを完了したとは記述しない。

同じ時刻・指令で比較したリンク分布の更新量は、初期化を除いて最大Marker151.834 mm、IMU107.355 mmだった。選定した12更新の [独立後解析](validation/variational_moments_20261018/link_updates/REPORT.md) では、raw VIの伝播平均とcore再照会の差は最大1.58e−13だった。一方、画像の相対位置残差が2.74 mmでも、根元基準の実棒先端誤差が205.97 mmという例が残る。相対観測とのfit、絶対身体精度、同じlawの表示、有限反復のstationarityを同一視しない。動作終了後の採用事象を、その前の振動原因として扱わない。

ユーザーの追加指摘に対応した [Kと身体の整合度](STIFFNESS_BODY_CONSISTENCY.md) は、軌道生成で捕捉した同じ候補を使い、勾配ready後の別processで記録する。K更新には既に平衡暗黙微分と全身体密度比scoreがあるが、数値代表の遅れと現在身体への適合は別に検証する。今回の追加診断はK規準・履歴・BFSを変更しない。

内部対称積の算術修正とK診断追加後、202ファイルを凍結した最終core・デモ全回帰は1304件通過した（207.59秒、既存hppfcl警告1件）。[最終coverage](validation/variational_moments_20261018/final_regression_coverage.json) にsourceとnative5本のSHA、実行コマンド、ログを保存した。承認済みcore2ファイル以外の保護対象とRViz43ファイルは一致した。旧1263件・中間280件・180秒実走とsourceを分け、件数を足さない。
