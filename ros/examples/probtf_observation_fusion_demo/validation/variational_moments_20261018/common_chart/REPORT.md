# 共通座標chartをKeepの確率支持と混同した不具合

第3試行はMarker29.67695秒／IMU併用29.95001秒で終了した。原画像 `image:00002388`、t=1792080027.2517104のReset事後を、Keepを参照する共通chartへ表す処理が停止した。元の各仮説の変分更新と、その元chartでの再表現は完了していた。

失敗入力のKeep共同共分散は数値rank41、支持判定のtoleranceは4.2540e−14だった。Reset平均のその支持からの残差は0.00630964、平均normは0.319918である。既知の固定座標に架空の値を入れたものではなく、Keepのbody/environment共同依存とは異なるReset法則を同じ幾何chartで表す場面だった。

各元事前 $p_h^-$ の下で観測更新する場合、事後Gaussianはその元支持上で最適化する。一方、既存の座標写像 $f_{h\to K}$ によって別候補をKeepのchartで表記することは、Keepの確率法則への条件付けではない。既存の一次輸送 $m'=f_{h\to K}(m)$、$P'=Df\,P\,Df^\top$ に、$m'\in\operatorname{range}(P_K^-)$ という条件を追加する根拠はない。Resetが独立redrawで共同依存を解く場合、確率支持が広がり得ることは宣言したkernelの仕様である。

デモ内に `materialize_state_in_chart` を追加した。元事前の支持を検査する既存 `reexpress_state` と演算の役割を分け、同じ共有kernelで変換合成・全GPGᵀ・Bingham二次形式を計算する。与えた平均・P・cross・質量は変更しない。新APIも実finite/PSD・固定回転との整合・必要な正定値逆を確認し、分散floor・jitter・架空のpriorを作らない。cacheのkeyには操作種別を含め、chartで受理した入力を後のstrictなBayes用呼出しへ流用しない。

共通chartへ表すReset各枝、`project_update` のraw照合・混合canonical化、Kへ渡すcanonical分子のsource照合に、このAPIを使う。各元priorでのVI・元raw childのsource照合はstrictのままである。coreの一般支持定義、尤度・質量、共通chart輸送式、K密度比の基準測度条件は変更していない。

原画像の [再処理](failed_event_replay.json) は、両親の元prior、最適化m/P、F各加算項、native観測が旧失敗時とbitwise一致した。新APIに渡す参照ID・輸送後の全m/P/cross・kwargsも旧throw直前入力とbitwise一致した。Resetのweight6.4282e−227、log mass−520.82611797を保持し、枝を落とすことなくcanonical・fresh goal・同law wireの照合まで完了した。旧失敗実走には成功したcanonical/wireのbaselineがないため、その存在を仮定しない。

Kの [下流照合](stiffness_ratio_validation.json) では、実際のKeep/Reset身体周辺が両方36/36支持であり、分母は元の二仮説、分子は同じ `projected_body_numerator` であることを確認した。環境を周辺化した身体密度比の支持条件は通過する。新APIで照合したcanonical state IDは実推定・表示と厳密一致した。Kの全履歴evidenceの再計算やK同定成功を示す検証ではない。

関連native/cache60件、共通chart統合5件を含む当時の関連280件が42.92秒で通過した。元の全1263件と、その後の接続変更を [coverage](regression_coverage.json) で区別し、重複件数を足さない。元prior支持のstrict検査、全native/log質量・underflow枝、時刻/指令、全joint mean/P/cross、実goal/control/wire、cache操作分離を検証している。新しいbody libraryは従来ABI1を維持し、chart拡張ABI1を追加した。sourceは親の `regression_sources_before_modal_gap_and_K_diagnostic.json` の200ファイル、binary/source hashesは [backend](backend.json) に保存した。

原raw・元ledger・pickleは元の `/tmp` アーカイブに保持し、Gitには小さい数値要約だけ選定した。180秒の新実走は、この約30秒の失敗試行と別に記録する。
