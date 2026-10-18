# 元画像job22・同一事前による初期候補付き変分更新

この比較は高集中閾値の撤去前のsourceで実行した。撤去後の全回帰・180秒実走とは別の固定観測検証であり、source SHAはJSONに保存した。

原取得画像 `image:00000149`、t=1792080001.3467875、sequence109、frame149、元Keep/Reset共同Gaussian（全cross）と元native中央画像分布だけを用いた。同stampの先行IMUが事前へ反映済みなので、この画像jobの実区間ledgerは空である。真値・真剛性・raw pixelsを推定へ渡していない。

各枝の元事前p0を固定し、旧Gaussian条件付けを最適化の初期分布候補としてのみ評価した。p0と初期候補を同じFで比較し、低い側から全mean/covarianceを最適化する。seedを新priorに置換せず、観測の二重融合でもない。

|予算|元親|seed採用|旧再合成画像位置残差mm|新残差mm|F|最終gradient∞|全体での枝質量|
|---:|---|---|---:|---:|---:|---:|---:|
|24|predicted:149:0|False|15.318|0.841|-29.197769|1.76|1|
|24|reset:448|True|55.139|1.081|35.141090|195|3.81e-29|
|100|predicted:149:0|False|15.318|0.587|-29.935936|0.0692|1|
|100|reset:448|True|55.139|0.111|1.100751|9.61|1.11e-14|

全4結果は設定toleranceに未達である。24と100反復を比較しても収束や最適性を保証しない。canonical後の同じnative画像座標での位置残差は24反復0.841mm、100反復0.587mmだった。最適化中μ/Sと、同じ最終原chart Gaussianを既存再表現しcoreで再照会したμ/Sの差は平均1.5e-15以下、共分散3.4e-18以下。表示側専用の補正や平滑化はない。

規準はKLと観測二次残差期待値・正規化定数の和であり、位置残差だけで枝を比較しない。F、KL、全共分散trace、各辺の原chart回転shift、native観測の全分散/log weight、原prior SHA、全枝質量、optimizer履歴をJSONへ保存した。`-F` は高集中モーメント近似を含むevidence proxyで、非線形での厳密evidenceや保証された下界ではない。

実行対象のsource SHAは前後で一致。元archiveは旧`propagated_moments`履歴の事前なので、この単一観測の比較を、新方式を初期状態から実行した性能・全跳ね解消の根拠にしない。旧prior-start結果は別の `/tmp/probtf_variational_job22_audit_20261018/` に保存し上書きしていない。

再現script `/tmp/probtf_variational_job22_audit_warmstart_20261018.py`、ROS workspace source・BLAS/OMP各1thread。元input/native/library/sourceSHAは `report.json`。この短い単一観測の計時はlive取得/制御周波数と区別する。
