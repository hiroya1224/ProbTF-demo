# 局所規準モデルの追従と非同期更新

7790a06の調査後にユーザーが指定した改修を記録する。候補測度を保持することと、評価位置から離れても同じ勾配を使うことを区別する。規準の完全勾配は $g_s=\nabla_c[J_s(c)+\lambda V_s(c)]_{c=u_s}$ のままとし、実行層では $g_{\mathrm{local}}(u)=g_s+H_s(u-u_s)$ を用いる。$H_s$ は同じ凍結law/Kの既存照会から作るSPD近似であり、候補測度や元の完全Jを置き換えるものではない。

二階系は $\dot u=v,\;\dot v=-\gamma v-\eta g_{\mathrm{local}}(u)$。既存gain $\eta=10^{-6},\gamma=8$ を保持する。一定SPDモデルでは有限の平衡 $u_s-H_s^{-1}g_s$ を持つ。packetを切り替えた全非線形閉ループの安定性、関節内への平衡の存在、候補の絶対推定精度は別問題として測る。旧定勾配保持を比較用に残し、凍結した二次規準・身体/目標/Kで順に比較する。

即時経路から表示保存を外す。勾配readyには同じ不変候補の識別情報と数値hashを付け、表示workerへはその候補をそのまま渡す。後着wireが一致する候補を描くことを確認し、表示時刻を新しい観測時刻へ付け替えない。

身体のバッチ更新とK evidenceを別processへ分ける。身体は更新完了後に利用し、K専用の全prefix比・原観測ID/時刻/指令・実ledgerをFIFOへ保持する。身体版とK版は一回の制御計算に捕捉し、K未完了の間は最後の完了Kを使う。未選定を架空のKで埋めない。全rawの保存、身体融合完了、K融合完了を別々に数える。

新勾配の受信時刻と有効化時刻を区別する。受信してから次の物理tick境界まで旧packetを使い、その境界でu/vを変えずに新しい局所モデルへ切り替える。結果が届く前の区間へ適用しない。

視野因子の新しい既定は $\phi(b)=\frac12[\max(0,\cos\theta_0-b_z)]^2$。各native成分を同じ重みで集約し、独立因子 $\exp(-\lambda V)$ として合成する。$\theta_0=10$ 度は較正画像の最小半FOV約26.565度の内側に置く中央許容域で、board全四隅が見えることを保証しない。$\lambda=2\times10^6$ は維持する。旧常時中央化因子も比較条件として残す。

旧d36bae9の90秒実測は [RUNAWAY_INVESTIGATION_20261018.md](RUNAWAY_INVESTIGATION_20261018.md) に残す。本変更の実測と混同しない。最適化の連続時間モデルと、遅延・物理系を含む実行系を区別する背景として、Hauswirthほかの [Optimization Algorithms as Robust Feedback Controllers](https://arxiv.org/abs/2103.11329) を参照した。本実装のpacket切替系について同論文の安定性定理を適用できたという主張ではない。

## 非同期受け渡しと保存

`event_motion.py` のcoordinatorは、取得・身体融合・K・制御・表示保存の5 workerを持つ。推定と制御のURDFモデル、平衡cacheはprocess間で共有しない。取得workerは従来どおり原画像・IMU・実送信指令を全て保存し、身体workerは固定frontierまでの全readyイベントを時刻順の原ID・指令で共同処理する。

身体workerはバッチ末尾のcanonical身体と固定目標ができたら `inference_body_ready` を送る。この時点でcoordinatorの身体版だけを更新する。K因子・evidence・代表選定の終了は待たない。K向けには近似前の全prefix候補、全cross、log質量、各観測の原時刻・指令、実区間ledgerを `stiffness_jobs/*.pickle` に不変保存し、SHA付き参照を別K workerへFIFOで送る。初期joint-only事前は観測前画面を作る明示的な初期化であり、K観測を捏造しない。

K workerは既存の全密度比・BFS・連続密度・evidence計算をその順序で実施する。結果 `stiffness_ready` が届けば最新の完了Kだけを更新し、身体をその過去のK jobの身体へ戻さない。次の制御requestが、その時点の身体版と独立に最新の完了K版を捕捉する。未完了中は旧K、初回未選定なら従来の宣言済み指令作用を使い、未選定を零や真のKで埋めない。Kの実演算失敗時には失敗jobと後続のblocked jobを保存し、旧Kによる実行と身体更新を続ける。失敗因子を飛ばして後続Kを成功扱いしない。

DBの各イベントの `body_processed` / `K_processed` は「身体完了時」の状態である。後者を同時に完了したと偽らず、Kの現在状態は `stiffness_job_sequence` とK完了journalから結合する。manifestではraw保存、body drain、K drainを別々に公開する。物理90秒終了も推定drain完了を意味しない。

制御workerは実ledgerで身体を評価指令へ輸送し、同じK作用を捕捉して完全J・view・Hを計算する。その後 `control_gradient_ready` を先に返す。readyにはg/H/u_sと候補の不変識別情報を含め、snapshot/marker wire/全K履歴JSONを含めない。表示jobには同じ不変lawを渡す。別artifact workerがsnapshot、粒子、wire、JSONを作り `control_artifacts_ready` を返す。表示保存の終了は次の勾配の採用条件ではない。

候補識別情報はbody/goal全成分のmean・全covariance・各変換のGaussian/Bingham値のSHA、全weight/log weight、ID・source時刻・指令・版、実ledger、K密度IDとaction packet IDを含む。表示workerで再計算した数値hashがreadyと一致する候補だけを結び付ける。別pipeの到着順が逆なら表示を一時保持し、元のreadyが届いてから照合する。`active_control_candidate_id` と `displayed_candidate_id`、表示の予測時刻・ageを分け、表示遅延を隠して新しい候補のラベルへ付け替えない。

同じK履歴を毎request再転送せず、初回登録後は不変packet IDで参照する。`PhysicalActionEngine` 内でも同じpacketの全履歴を一度だけSHA検証・decodeし、内部の読取り専用recordを再利用する。公開 `FrozenKPacket.record` とreanchorの記録は従来どおり独立copyであり、外部変更を内部cacheへ漏らさない。同IDに別JSONが届けば出自不一致として記録する。密度・代表値・作用はこのcache化で変更しない。

## 適用時計と積分

ready受信時にはmonotonic時計、最後の物理frame時刻、動作開始からの受信経過秒を保存する。`advance()` 冒頭で新勾配を採用する旧処理を撤回した。先に旧packetで次の物理境界まで進め、実送信指令と区間ledgerを保存してから、受信済みpacketをその境界で採用する。そこでu/vを変更しない。境界後に初めて受信した結果は、さらに次の境界まで待つ。起動前の静止には未積分区間がないので即時採用できる。

`control_ready_receipts.jsonl`、`gradient_adoptions.jsonl`、`commands.jsonl` で受信・採用・実使用区間を照合できる。動作終了後に届いた結果の採用は `activation_after_motion_end` で区別し、過去の90秒へ作用したとは数えない。遅着packetをageや距離の監査閾値で捨てず、有限な局所モデルを次の周期にも利用する。

Hが非零なら、状態を `[u-u_s, v, 1]` とした13次元のaffine行列の指数関数で一回の実dtを進める。右上の定数入力は `-eta*g_s`、位置から加速度への作用は `-eta*H`、減衰は `-gamma*I` である。tick内で候補の再伝播・最適化はしない。H=None/厳密零なら従来の定勾配積分をそのまま使う。関節範囲の実逸脱・非有限値は既存の明示停止として残す。保守的な区間boundだけが曖昧な場合は診断に留め、数値監査閾値を追加停止条件にはしない。

## 回帰検証

単純な固定6次元二次規準、実画像から構成した固定body/goal/K、実spawnの5 worker、実ROS/RVizを順に区別して確認する。テストではH=0の旧式とのbitwise一致、異なるdtへの分割一致、u/v連続、実関節逸脱・非有限入力、有限な監査未達での継続、全候補hash、表示処理が待っていてもreadyが先に届くこと、身体完了がK完了より先に届くこと、FIFO内の全原指令/時刻/prefix、公開copyとcacheの隔離を扱う。

時計の回帰では旧packetと逆向きの新packetを用い、物理0秒で壁時計0.5秒の結果を受信し、0→1秒は旧packetだけで積分、1秒境界でu/vを保持して採用、次区間から新packetを使うことを検査する。単に保存式を再計算して一致したという検査だけではない。


## 視野因子

各保持goal成分kについて、同じ身体・根元marker候補を既存coreでcamera optical frameへ照会し、その平均位置をp_k、単位方向をb_k=p_k/||p_k||とする。全成分のnative log weightから安定に正規化した重みをbeta_kとする。新しい既定は `penalty_mode=tolerance_cone`、`tolerance_angle_deg=10.0` で、`e_k=max(0,cos(theta0)-b_k,z)`、`phi_k=e_k^2/2`、`V=sum_k beta_k phi_k`、独立選定因子は `exp(-lambda V)` とする。lambda=2e6は維持する。円錐内では位置を光軸そのものへ寄せる勾配は零になる。

10度は既知の960×720画像、fx=fy=720、cx=480、cy=360の最小半視野角26.565度より小さい中央の許容角として選んだ暫定設定である。一般の較正では主点から各画像縁へのatanで最小半視野角を診断表示する。これはmarker中心方向だけの条件であり、boardの全角点が入ること、検出継続、遮蔽回避を保証しない。背面成分を落とさない。p=0のような実の演算不能は因子使用不能として記録するが、微小でも有限なpを独自のnorm閾値で捨てない。

旧 `penalty_mode=always_center` は比較用として残し、従来通り `phi=1-b_z` を使う。disabled/weight=0ではview queryを行わず、未計算のVを零と偽らない。task J/gは元の値を保つ。局所モデルenabled/disabledは視野因子enabled/disabledと別の設定である。

## 同じ照会結果の再利用

完全Jの勾配g_taskは従来の `gradient_at_command` のままであり、正規化項・位置共分散・Bingham正規化・全mixture/log weightを含む。sourceと各関節の前後の最大13個の決定論的指令照会を一度だけ行う。関節位置範囲や補助branchの実評価可能性に応じて同じ片側差分も使う。確率候補を少数点へ選定するモンテカルロではない。

新しいoptional details observerは、既に計算したtrial states、queries、branchesを選定層へ渡す。従来の3引数observer APIは維持する。H構成のための追加の指令push/core queryは0回である。source/K/body/goal/時計を変えて別のlawからHを作らない。

前進幅h_plus、後退幅h_minusが両方評価可能なら、任意の記録済み量fの微分を `D_i f=[h_minus*(f_plus-f0)/h_plus-h_plus*(f_minus-f0)/h_minus]/(h_plus+h_minus)` とする。一方しかなければ同じ前方/後方差分にする。元のJのstencilをそのまま再使用し、curvatureだけの追加候補を生成しない。

## taskの正の曲率近似

source指令をu_sとする。各全比較枝のsource密度L_kとnative weight w_kから、`alpha_k=w_k L_k/sum_l(w_l L_l)` を得る。これは局所metricの責任率であり、p_goalのmixtureを変更する操作ではない。大きい共通log PDFを相殺してからlog weightを加え、全branchを保持する。線形weightがunderflowしても有限のlog weightを先に零へ落とさない。

位置側には既存queryの平均m_kと共分散S_kを使う。support inverseをS_k^+とし、`F(S,m)_{ij}=(D_i m)^T S^+ (D_j m)+(1/2) tr(S^+ (D_i S) S^+ (D_j S))` とする。実装ではmean/covarianceを支持上でwhiteningし、そのJacobianのGramとして組む。分散floorやjitterを候補に入れない。

方位側には、完全Jが既に計算した補正後のBingham modal quaternion r_kと、既存coreのBingham/S2高集中近似由来の方位共分散Omega_kを使う。新しいSO(3) exp/logや別の回転ノイズlawは導入しない。`N(r)` を既存のquaternion左積行列の後ろ3列とする。sourceの基底をN_sとし、各trial quaternionはsourceと内積が非負になる符号へ合わせる。これはrと−rが同じBingham方位を表すことに対応する。

sourceへ線形射影したmeanは `a_k(u)=2 N_s^T(r_k(u)-r_k(u_s))`、共分散は `Omega_tilde_k(u)=P_k(u) Omega_k(u) P_k(u)^T`、`P_k(u)=N_s^T N(r_k(u))` とする。この既存quaternion tangentの線形射影に対し、同じfrozen FDからD a、D Omega_tildeを求め、位置と同じFisher式Fを使う。これは単一Binghamの高集中tangent Gaussianを使うmetric近似であり、native Binghamの正規化定数の正確な二階微分と主張しない。

`H_task=sum_k alpha_k [F(S_k,m_k)+F(Omega_tilde_k,a_k)]` とする。coreの全身体/環境crossはrelative queryを作る段階まで元のまま保持する。位置×方位の積による比較proxyは既存taskの定義であり、Hのために新しく身体stateのcrossを切らない。mixtureのobserved Hessianに含まれる負のcomponent-score covariance等はこの正のcomplete-component Fisherには入れない。完全なg_taskは引き続きそれら全てを含む元のJから計算する。

## viewのGauss–Newton項

許容円錐の外側では `D_i e_k=(b_k,z b_k-e_z)^T D_i p_k/||p_k||`、内側と境界では零を使う。`H_view=lambda sum_k beta_k (D e_k)^T(D e_k)` とする。各pのFDも既に取得したview probeから求める。円錐内のH_viewは零で、候補分布のcovarianceをそのために縮めたり広げたりしない。

旧always_center比較では `phi=||b-e_z||^2/2` に対応する `H_view=lambda sum_k beta_k (D b_k)^T(D b_k)` を使う。視野因子とH_viewはセンサ観測ではなく、body/K/BFSへ融合しない。

## 宣言するridgeとpacket

最終Hは `H=H_task+H_view+delta I`、`delta=1e5+1e-6*trace(H_task+H_view)/n` とする。これは局所モデルだけの正則化であり、元の候補covariance、K、lambda、eta/gammaやJ/gを変える値ではない。absolute/relative/addされた実数をpacketへ記録する。数学的にはGramと正のridgeで対称正定値になる。固有値、Cholesky、定常点solveは診断に限り、それら任意監査の失敗で有限なH/gの使用可能な更新を止めない。H構成に実際に必要な支持上の逆計算や、非有限なH等が成立しなければ理由付きの失敗packetとする。

当初のabsolute=relative=1e−6では、同一凍結例のFisher定常点がmarker約18.75rad、IMU併用約20.35rad離れ、joint box外へ出た。共通候補absolute=1e3/1e4/1e5/5e5/1e6を同じlaw/gで比較し、1e5ではcone/旧center×marker/IMUの4case全ての定常点がbox内になった。初期cone例の最大固有値に対して約5–6%のridgeである。これはその例での数値根拠であり、別lawやpacket切替での範囲・到達保証ではない。ライブごとにこの探索を行ったり、q*をboxへclipする実装ではない。

immutable packetは `curvature`、`local_model_method`、`local_model_diagnostics` を持つ。g/H/u_s/K/body/goal/density/時刻は一体で採用する。実行則は `du/dt=v`、`dv/dt=-gamma v-eta[g_s+H(u-u_s)]` とし、同じ二階系の厳密解で進める。eta=1e−6/gamma=8を保ち、採用時u/vをresetせず、速度clipも追加しない。`q*=u_s-H^-1 g_s` は診断だけで、実行に必要な逆計算や指令選定ではない。H=None/厳密zeroは旧一定勾配方式へbitwise一致する明示比較経路として残す。

## 凍結比較の観測と限界

同じbody/goal/K/source、cone既定、同じ初期u/vで旧定勾配と最終Fisherを30秒比較した。実物理を30秒動かす実験ではなく、初期の実観測から構成した候補lawを固定して指令モデルだけ進めた診断である。新しい観測や勾配更新は加えていない。true physical姿勢/距離を制御入力にもこの比較の到達判定にも使っていない。

旧定勾配はmarker28.18秒、IMU併用20.54秒にjoint2下限を越す次指令が生じ、そのtrialを明示停止した。Fisherは両30秒で範囲内だったが、固定局所二次モデルの定常点までmarker0.456rad、IMU0.240rad残った。最弱modeの時定数は79.37/79.40秒であり、30秒で定常へ到達したという結果ではない。

真の選定規準J+lambda Vは単調には減らなかった。Fisherの5→30秒はmarker約12022→24136、IMU併用15618→67354。元の初期値は約93053/93141である。初期に円錐内だったのでH_view=0を30秒固定したが、その後に別途照会した真のVは増えた。候補からの相対位置RMSもmarker0.3679→0.3250mに対しIMU併用0.3680→0.4193mで悪化した。固定SPD二次モデルの安定性と、非線形の完全規準・推定距離・ライブ到達改善を同一視しない。


設計係数と30秒固定lawの比較は [validation/local_model_tracking_20261018](validation/local_model_tracking_20261018) に小容量資料として保存する。これは新しい観測を含む非同期ライブの性能の証拠ではない。

## ROS/RVizの90秒実測

新しいPython/configのhashを起動時に固定し、MarkerとIMU＋Markerを別masterで並行実行した。RVizにはユーザーが調整した同じ `marker.rviz` のbyte copyを使い、粒子・画像・軸・視点・レイアウトは編集していない。窓の位置だけを変えた。物理・joint・IMU・画像の目標周期は120/40/30/12 Hz、eta/gammaは既定のままである。

| 90秒動作終了付近の指標 | Marker | IMU＋Marker |
|---|---:|---:|
| 実物理更新Hz | 72.63 | 71.56 |
| 実画像取得Hz | 10.85 | 11.01 |
| control ready Hz | 3.98 | 4.46 |
| 実手先の終端誤差 | 462.3 mm | 392.6 mm |
| 物理到達判定 | 未到達 | 未到達 |
| 採用済み身体の観測frontier | 27.474秒 | 19.935秒 |
| K job完了 / 提出 | 6 / 7 | 4 / 6 |
| 表示artifact待ち | 47 | 14 |
| 表示候補の予測時刻のage | 18.970秒 | 3.425秒 |
| 保存済みraw件数 | 6,094 | 6,188 |
| 身体処理完了raw件数 | 1,990 | 1,470 |

この表の物理終端は90秒に固定している。その他はobserverが両終端を確認したwall時点のmanifestであり、終端直後に採用した制御packetも含み得る。その時点の `gradient_source_age_s=0` を「動作中の遅延が零」とは扱わない。実際の勾配age・評価指令からの距離、採用/実使用時刻、積分式との一致は、90秒までの `commands.jsonl` に使われたpacketだけで別集計する。終了後のbody/K処理は物理動作を延長していない。

旧d36bae9でのcontrol readyは約1.11/1.76 Hzだった。今回の約3.98/4.46 Hzは別の非同期観測履歴での実測であり、同じrawを使った計算時間倍率ではない。今回も120 Hzの物理目標、12 Hzの画像目標には届いていない。古い身体候補を現指令へ輸送して勾配を再評価することと、新しい観測の身体事後が届くことを区別する。身体融合自体とK evidenceは依然遅く、全rawを処理済みとも、推定が実時間に追従したとも主張しない。

表示保存を制御の採用条件から外した結果、表示側の処理能力を超えるreadyが到着すると表示jobのFIFOが積み上がる。Markerの約19秒の表示ageは残存する問題である。画面のlawは、その表示candidate IDの過去の制御計算に使った同一lawであり、現在実行中のpacketと常に同時刻であるという意味ではない。manifestの `display_follows_same_frozen_control_candidate` はこの出自照合方式を表す。現時点の一致は `active_control_candidate_id` と `displayed_candidate_id` の値を照合して判断する。

両実行は90秒を通して関節範囲逸脱や非有限値による停止を起こさなかったが、両方とも到達せず、初期の実誤差約369.6 mmより終端誤差が大きかった。旧試行の586/885 mmより小さいことだけを改善保証としない。固定局所モデルでの規準非単調性に加え、観測更新の遅延とpacket切替を含む系の課題が残る。本変更により暴走・到達問題が完全に解決したとは結論しない。

物理終了後は約2分半の処理経過も記録し、その後に検証用demoノードを明示停止した。未完了の全raw・進行中身体batch・K FIFO入力archiveは `/tmp/probtf_local_model_live_20261018` に保持した。未完了jobを完了扱いにせず、停止を演算失敗や収束判定へ付け替えていない。RVizは終了時の画面を残した。90秒中の窓配置に重なりがあったため、両窓の比較図は終了後に位置だけを直して撮影し、物理終端画像とは別ファイルにした。後撮りを90秒時点の画像として使用しない。

## 実行・再現

通常のデモはworkspaceをsourceして `roslaunch probtf_observation_fusion_demo marker.launch` または `roslaunch probtf_observation_fusion_demo imu_marker.launch` で起動する。本検証では両masterを分け、`async_duration_s=90.0` と `autostart=false` を指定した。初期の剛体事前分布の表示を確認してから `/observation_fusion/replay` を呼んだ。

固定候補の比較は、パッケージディレクトリから `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 validation/tools/compare_frozen_candidate_tracking.py --output-dir /tmp/probtf_frozen_candidate_comparison` で再現できる。初期の実描画画像とIMUを入力する互換fixtureを使い、この診断fixtureだけK-BFSを明示無効にする。既定のライブ設定を変更する処理ではない。単純な二次規準の比較には `validation/tools/compare_frozen_quadratic_tracking.py` を使う。

対象デモの全テストは982件通過、既存hppfclの非推奨警告1件であった。コマンド・source hashは [test_summary.json](validation/local_model_tracking_20261018/test_summary.json)、固定比較は [frozen_summary.json](validation/local_model_tracking_20261018/frozen_summary.json) と [比較図](validation/local_model_tracking_20261018/frozen_comparison.png) に記録した。214保護対象は全byte一致した。テスト通過を到達成功の証拠にはしない。
