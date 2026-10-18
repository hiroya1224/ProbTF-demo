# 高集中モーメント更新: 独立レビューと固定旧事前の検算

新しい3moduleに重大な実装取り違えは見つからなかった。coreのO(P)補正平均、実際の伝播共分散、全共同crossを一回のGaussian条件付けへ入力している。元priorは固定で、MAP・Gauss–Newton・body evidenceのLaplace近似・反復更新は使っていない。

観測native固定chartへの移し方は、原modal相対角phiにおける一回の線形座標輸送である。平均にも同じDとoffsetを作用させ、coreのO(P)補正を落としていない。非線形chartのHessianによる追加の平均補正まで計算したとは説明していない。IMUは観測固定S2接平面で、反対向きの射影の曖昧性はこの局所モデルでは解消しない。

原M93/M94/M95/I321の共同priorと既存中央native観測を固定し、新production operatorを呼んだ。準備済みの元観測を読み戻すoffline callbackだけを使用し、検出のやり直しや真値の代入をしていない。各Keep/Resetについてcore condition_gaussianを独立oracleとして比較すると、平均最大差8.92e-13、共分散1.60e-15、evidence1.37e-9（大きな負log evidence上）で一致した。供給S/Cを直接使った公式、SciPy Gaussian logpdfとも照合した。親・nativeのlogweightによる全枝正規化は厳密に一致。

これは旧MAP/GN方式が過去に作ったpriorに一度だけ新operatorを適用する局所比較であり、新方式を初期状態から通して走らせた結果ではない。新sourceの到達性能や跳ね解消の証拠へ流用しない。

## 再合成closureは零ではない

凍結したGaussian代理での観測事後平均と、更新したjoint stateを再合成してcoreで照会した平均との差を、同じnative固定chartで計算した。位置差はKeep/Reset順で:

- M93: 0.603 / 67.47mm（後者の事後logweightは−30.82で小さい）。
- M94: 19.39 / 92.31mm（Resetがほぼ全質量）。
- M95: 61.24 / 62.01mm（Resetがほぼ全質量）。
- I321: 7.48 / 105.99mm（Resetがほぼ全質量）。

最大回転座標差はM95 Keepの0.04778rad。これは単回の凍結Gaussian代理を有限再表現した際の非閉包であり、解析的条件付けの行列計算ミスとは区別する。反復で合わせる、再中心化する、観測を弱める、別方式へfallbackする処理は追加していない。

同期に複数センサを一度渡す診断経路は、共有P由来の観測間crossを保持し、Rだけをblock diagonalにしている。全既存中央成分・Keep/Reset親を残す。個別raw marker成分がbody段階で直接全Cartesian productとして列挙される、という意味ではなく、従来のboard観測構成は別段階として保持される。

元source ID・t/q・readonly・Dirac支持は保持され、保護214ファイルも全バイト一致。source編集・テスト起動・新物理runは行っていない。式・入力archive hash・数値誤差・各枝のclosure値はsummary.jsonに保存した。
