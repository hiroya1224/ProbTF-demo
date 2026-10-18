# Gaussian変分更新と高集中閾値撤去の検証

前回の共同Gaussianを次の観測の事前に引き継ぎ、各原取得時刻・実指令で $F(q)=\operatorname{KL}(q\|p_0)+\mathbb E_q[-\log L]$ を最小化する。既存の伝播補正平均・全共分散をデモ内C++で逆微分する。反復中は原事前・観測を固定し、同じ観測を重複融合しない。仕様は [VARIATIONAL_MOMENT_UPDATE.md](../../VARIATIONAL_MOMENT_UPDATE.md) を参照する。

## 検証の区別

- [数値検証](numerical/README.md) は線形Gaussian極限、全mean/covarianceの微分、逐次蓄積、固定目標cutの共同PSDと、小演算の計時を扱う。全体の実時間性能や物理到達を意味しない。
- [固定観測job22の比較](fixed_observation/REPORT.md) は旧履歴の同じ元prior・native観測への新演算子の適用である。新方式を初期状態から実行した結果と区別する。
- [手先IMUの二観測](hand_imu/summary.json) は全6リンクの実更新・前回分布の引継ぎ・同じ規準の改善を記録する。非線形更新では各周辺分散の単調収縮を要求しない。
- [停止した原観測の再現と修正](concentration_removal/saved_q_materialization.json) は、旧最適化結果のm/Pを一切変えずに再表現・伝播が可能となったことを検証する。
- [未使用の条件付き目標診断](optional_diagnostics/REPORT.md) と [共通chartと確率支持の分離](common_chart/REPORT.md) は、実走で見つかった接続不具合の再現と修正である。
- [両180秒の実走](live/REPORT.md) は終端距離・実Hz・全wire・時系列と未処理を記録する。物理動作は完了したが、到達は未達、終了後のdrainは実エラーで中断した。
- [12更新の後解析](link_updates/REPORT.md) は同じ取得t/qで分布変化と実身体精度を分ける。観測に合うことを絶対姿勢の正確さと同一視しない。
- [Kと身体の整合度](K_body_consistency/REPORT.md) は、実際に最後の区間へ使った同じ候補を復元した後解析である。新診断は軌道生成時の非同期artifact処理へ追加したが、上の180秒runには含まれない。

## 高集中閾値による実停止

最初の試行はMarker33.843秒、IMU併用33.464秒で終了した。IMUの `imu:00002386` は原取得時刻1792080029.9349635、sequence1982であり、その時刻までの実8区間ledgerと前回canonical事後から同じ失敗を再現した。元観測を処理到着時刻へ付け替えていない。

Keep枝の原chart回転分散はlink_2で0.123715394 rad²、link_6で0.11033110 rad²となり、旧0.1上限により最適化後のmaterializationが例外停止した。非有限値や必須の逆計算が不可能だった事象ではない。この約34秒を180秒の完走として扱わない。全raw・原事前・元観測・全cross・実区間ledgerは元の `/tmp/probtf_variational_moments_180s_live_20261018/` に保持し、Gitへは小さい診断のみ選定した。

最新明示指示により、Python coreとデモC++の0.1上限、相対modal-gap判定、回転の必須逆計算での数値rankによる切捨てを撤去した。式は $C=2(\lambda I-N^\top A N)^{-1}$ と $A=-2NC^{-1}N^\top$ のままであり、正定値行列の実逆作用を正則化なしのCholesky・solveで求める。分散clip・jitter・枝選択を加えない。一般の共同Gaussian/Dirac支持の規約は変更していない。

保存済み旧m/Pを直接渡す比較では、両枝の原prior・最適化結果をbitwiseで維持したまま、native/Python両方で再表現・原IMU照会が完了した。平均差は最大1.75e−14、共分散差は5.49e−17である。新sourceで最適化からやり直した比較では、Keepのm/Pはbitwise一致した。Resetは逆行列因子分解の丸め差が伝わり、最終m差1.30e−7、P差4.47e−8となった。両比較を [別の記録](concentration_removal/fixed_event_replay.json) に残し、変化がないと誤記しない。

これらは停止判定の撤去と同じ候補分布の継続を検証するもの。大きい幅での幾何近似の正確性や、全過去データの厳密Bayes事後を保証するものではない。

## ソースと保護範囲

最終のcore・デモ全回帰は1304件通過、207.59秒、既存hppfcl警告1件である。[最終coverage](final_regression_coverage.json) と [原ログ](logs/final_full_tests.log) は、内部対称積の算術修正とK診断追加後の202ファイルに対応する。下の1263件・中間280件は別sourceの履歴であり、合算しない。[保護範囲の照合](final_integrity.json) ではcore・RViz計184ファイル、RViz43ファイルの設定を確認した。

撤去後のcore・デモ全回帰は1263件通過、212.04秒、既存hppfclの非推奨警告1件である。[最終ログ](logs/full_tests.log) を保存した。数値資料内の小演算計時と固定観測job22は撤去前に測ったものなので、撤去後の全体性能へ読み替えない。

閾値撤去前の1219テストと194ファイルのsource SHAは `before_threshold_removal` 名で保存した。撤去後の全1263件は `regression_sources_before_optional_diagnostics.json` の197ファイルに対応する。接続修正後の実180秒runは200ファイルで、`regression_sources_before_modal_gap_and_K_diagnostic.json` およびlive内のsource記録に対応する。現在の数値実装・設定・テスト202ファイルは [regression_sources.json](regression_sources.json) に記録する。coreについて明示許可された変更は `first_order.py` とその回帰テストだけであり、[元baselineと許可差分](concentration_removal/authorized_protected_baseline.json) を保持する。RViz43ファイル・画像・粒子・軸サイズは変更しない。

各証拠のsource/input/library SHAと原出力先を保存した。原pickle・raw画像・取得DB・全wire・大容量中間生成物はGitへ追加していない。再現scriptの固定入力パスを使う場合は、その原アーカイブとhashを照合する。
