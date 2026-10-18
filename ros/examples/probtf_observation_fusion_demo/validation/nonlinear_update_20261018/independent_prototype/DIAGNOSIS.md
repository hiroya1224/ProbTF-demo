# 固定事前の単一観測非線形更新：独立オフライン検算

2026-10-18。production/srcは変更せず、元90秒runの画像イベント marker147、marker148、IMU併用487を読み直した。最後の例も**画像更新**であり、IMU単独の非線形更新の実験ではない。元の各Keep/Reset親、全リンク交差、原取得時刻・指令、全native画像成分とlog weightを保持した。3画像とも中央native観測は1成分である。多成分画像の実データ検証まで済んだとは説明しない。

結論は限定的である。同一事前を固定した非線形MAPでは、旧一次更新後に再合成した観測残差を大幅に減らせた。一方、真値を使わない推定の改善と、実リンク姿勢の改善は一致しない。**この固定priorでmarker147の大きな誤補正は残る。ジャンプ・到達問題を解決した結果ではない。** 元prior・地図は旧一次更新の履歴で形成されたものであり、その一観測operatorだけを置き換えた検算である。新方式を初期priorから通した結果でも同じ跳ねが必ず起きる、またはResetモデルが根本原因である、とはこの検算から断定できない。

## 候補、尤度、近似の範囲

元のGaussian座標代理を $z\sim\mathcal N(0,P)$ とする。今回の6親では元meanはすべて零であり、支持域固有分解で $P=LL^\top$、$z=L\xi$、$\xi\sim\mathcal N(0,I_r)$ とした。元66次元、支持rank42を保持し、零分散方向へjitterを足さない。各辺の候補を

$t_j(z)=\bar t_j+z_{t,j},\qquad R_j(z)=\bar R_j\operatorname{Exp}(z_{r,j})$

として有限変換を合成・反転し、カメラ相対マーカー $(t_{CM}(\xi),R_{CM}(\xi))$ を得る。これは既存Bingham由来の局所Gaussian座標代理の有限写像への拡張であり、全 $\mathbb R^r$ で元の厳密なBingham共同lawと同じとは主張しない。coreは変更していない。

native画像成分 $k$ の観測mode $R_k$ を固定し、

$h_k(\xi)=\begin{bmatrix}t_{CM}(\xi)\\\operatorname{Log}(R_k^\top R_{CM}(\xi))\end{bmatrix},\quad L_k(\xi)=w_k\,\mathcal N_6(h_k(\xi);y_k,\Sigma_k)$

をスカラー尤度として用いた。$y_k,\Sigma_k,w_k$ は元画像Hessianから復元したnative中央成分である。同じ固定native chartの関数をKeep/Resetの両親に使う。候補ごとの体積係数は掛けない。今回native参照点でのchart log-volumeは零である。このGaussianスカラー尤度はnormalized Bingham conditional-lawの点密度とは別であり、その両者を同一視しない。旧更新は親ごとのchartを使うので、旧規準と新規準が大域的に厳密同値という比較でもない。

最適化する関数は

$\Phi_k(\xi)=\tfrac12\|\xi\|^2+\tfrac12(h_k(\xi)-y_k)^\top\Sigma_k^{-1}(h_k(\xi)-y_k).$

各親×native成分を独立に解く間、元事前を固定する。反復ごとに事前を置き換えたり、同じ画像を再観測として重複融合したりしない。candidateの点写像にcoreの $O(P)$ 平均補正を加えない。事後を再表現した後のcore query平均・共分散には、既存の $O(P)$ 計算を別に使う。

## actual HessianとGNの区別

TRFによるMAP参照解を原prior mode $\xi=0$ と旧一次posterior modeの2起点から求めた。2起点の最終 $\Phi$ 差は最大 $2.3\times10^{-10}$ だった。この3画像で局所解の差を見なかっただけで、一般の大域最適性保証ではない。

actualHの独立oracleは解析的scoreの中央差分幅 $5\times10^{-5}$ から $H=\nabla^2\Phi$ を求め、実Choleskyで

$C_\xi=H^{-1},\quad C_z=LC_\xi L^\top,\quad \log Z_k=\log w_k-3\log(2\pi)-\tfrac12\log\det\Sigma_k-\Phi_k(\xi^*)-\tfrac12\log\det H$

を評価した。whitened priorの正規化とLaplaceの支持次元係数が相殺するので、$\det L$ を追加しない。実Hessianは全6枝でSPDだった。clip、jitter、actualH失敗時のGNへの無言fallbackはない。有限iterateのscoreは $1.7\times10^{-9}$〜$9.7\times10^{-6}$ であり、厳密stationaryな点や厳密非線形積分の値とは呼ばない。このactualH式にはlinear-score補正項を足していない。

同じ点 $a$ で $A=Dh_k(a)$ として $G=I+A^\top\Sigma_k^{-1}A$ を構成した。GNでは

$g=\nabla\Phi_k(a),\quad \mu_\xi=a-G^{-1}g,\quad C_\xi=G^{-1},$

$\log Z_{\mathrm{tangent},k}=\log w_k-3\log(2\pi)-\tfrac12\log\det\Sigma_k-\Phi_k(a) +\tfrac12g^\top G^{-1}g-\tfrac12\log\det G$

を別に記録した。これはその点で凍結した接線尤度のGaussian積分であり、非線形尤度の厳密な積分ではない。stationaryに十分近いと補正項と平均移動は小さくなる。有限反復の接線平均 $\mu_\xi$ での実際の $\Phi$ とscoreも照会し、接線上の改善だけで成功としなかった。

次表の残差は全て同じnative chartでcore事後query平均を観測共分散でwhitenした二乗値であり、MAP点の残差とは区別する。共分散差は全66×66行列のFrobenius相対差である。

| 例・親 | 旧一次更新後 | 新actualH | 新GN | GN全cov相対差 | GN−actualH logE |
|---|---:|---:|---:|---:|---:|
| marker:147 Keep | 5794 | 396.85 | 351.27 | 0.6547 | -1.189 |
| marker:147 Reset | 3896.3 | 22.676 | 22.454 | 0.067393 | 0.1246 |
| marker:148 Keep | 3489.2 | 206.69 | 196.85 | 0.5041 | 0.1916 |
| marker:148 Reset | 1652.8 | 1.0042 | 1.3936 | 0.013744 | -0.01866 |
| imu_marker:487 Keep | 1380 | 392.1 | 376.69 | 0.75888 | -1.425 |
| imu_marker:487 Reset | 16757 | 6.3911 | 5.9758 | 0.089335 | -0.2297 |

GN−actualHのReset/Keep log BF差はmarker147が+1.31395、marker148が−0.210223、IMU併用487が+1.19509であった。今回は両方ともReset確率が浮動小数点で1となるが、Keepの有限log weightを削除していない。弱い方向の共分散差はKeepで大きく、GNとactualHが数値的に同じとは言えない。

actualHの差分＋分解は約0.17秒/枝、GNの分解＋事後再表現は約0.0053秒/枝だった。後者は最適化全体を含まず、C++本番の実時間性能や90秒動作の性能値ではない。

## 原prior modeからのGN反復数

別の純GN＋Armijoを零起点から最大30回評価した。Reset147は10回でscore約 $8.9\times10^{-9}$、Reset148は8回で約 $1.9\times10^{-9}$、Reset487は10回で約 $2.8\times10^{-6}$ となり、16回付近で約 $1.8\times10^{-8}$ に下がった。以後のArmijo不採用は有限精度の停止として記録し、架空のstationarityや観測無効を作らない。

Keep147は10/20/30回で $\Phi=557.1754/447.3331/441.5695$、十分最適化した参照解は441.2721で、30回scoreも12.43である。Keep148の10回scoreは0.619、Keep487の10回scoreは0.215であった。**全枝が10回で収束するとは言えない。** Resetが優勢でも、残るKeep枝の非収束を隠したり削除したりしない。詳細はcompact.jsonのfinite_GN_from_original_prior_modeに残した。

## 真値は最適化完了後に別プログラムでだけ比較

原画像と同じ物理frame・stamp・指令の保存実関節角からFKした。真のK、平衡計算、物理姿勢を尤度・MAP・枝選定へ渡していない。以下はメートル単位のmodal位置誤差であり、表の新Resetが最終の単峰projectionと完全に同じという主張ではない。

| 例 | 部位 | 元Keep prior | 旧canonical | 新Reset |
|---|---|---:|---:|---:|
| marker147 | link6 | 0.19260 | 0.29329 | 0.29148 |
| marker147 | hand | 0.32414 | 0.33806 | 0.34218 |
| marker148 | link6 | 0.28119 | 0.17906 | 0.17394 |
| marker148 | hand | 0.31292 | 0.31375 | 0.30892 |
| IMU併用487 | link6 | 0.07845 | 0.12401 | 0.10881 |
| IMU併用487 | hand | 0.09742 | 0.11920 | 0.08331 |

marker147は観測fitが改善しても、link6の約29cmの誤差と誤った大補正が残り、handは旧canonicalより少し悪い。IMU併用487のhandは良くなるがlink6は元Keepより悪い。未知地図・身体の識別性、Reset候補、センサモデル等の残る影響を、この3画像だけで特定したとは言わない。単一観測内の凍結一次外挿を直すだけで全体のジャンプ解消まで保証できない。一方、これは旧履歴を持つ固定priorでの局所比較なので、新方式を初期から逐次実行した場合の改善限界を示すものでもない。その評価には新source・初期priorからの独立した通し実行が必要である。

## 検算・出自

- 全3画像のnative中央lawを原rawから再構成し、保存lawの全成分・数値とbitwise一致した。各元K-job payloadの実SHAと保存SHAも一致した。
- 旧一次モデルの独立Gaussian条件付けoracleは、保存posterior meanが最大 $2.91\times10^{-13}$、covarianceが $8.92\times10^{-15}$、log evidenceが $2.89\times10^{-10}$ 以内で一致した。新有限mapの解析Jacobianと中央差分の最大絶対差は約 $7.1\times10^{-11}$ だった。
- 最終prototype実行前後83ファイルのSHAが一致。元runで保護した214 core/RViz等ファイルも変更なしだった。production変更、MC、観測棄却、分散clip、jitter、表示平滑化はしていない。
- 新finite C ABI追加後のライブラリで最終core queryを実行した。bodyライブラリは旧90秒の `3879a1845062ab2e3ebfd998eecf941b0b0f1cc292586f5397e90ff95dae6cef` から今回 `26baa1ed012f1de7849f3e09373e7a8b18860cfacded7fa2ea10a5ebe382c0c8` に変わっている。CPPと一部Pythonの変更もsource_provenanceへ列記した。**旧90秒runの全source/binaryと同一とは主張しない。** 有限candidate mapはこの/tmpスクリプトの独立NumPy/SciPy実装で、新C ABIを呼んでいない。
- 原入力：`/tmp/probtf_cpp_sequential_live_20261018/{marker,imu_marker}/stiffness_jobs/` の147・148・487、対応SQLite raw、URDF。全ID・取得指令・stamp・frame・payload SHAはcompact.jsonに保存。

再現コマンド（単一BLAS/OMP thread）：

```bash
source /home/leus/catkin_ws/devel/setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 /tmp/probtf_fixed_prior_nonlinear_prototype_20261018.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 /tmp/probtf_fixed_prior_nonlinear_physical_20261018.py
```

スクリプトSHA：

- prototype：`645b862194d0af7c8f105f566f882db56eb9d1fe5cde2141a1d49d09c8901a5d`
- 別物理後解析：`713d54bc67c42a645abb1c7d9456df6d1025f077817e7360f65d0b4a20644979`

全行列付き `report.json` と各リンク後解析 `physical_comparison.json` は/tmpに保持した。レビュー用compact.jsonは行列を反復保存せず、全枝の主要指標・実source SHA・全出自を保持する。これはオフライン診断であり、新本番sourceの実ROS到達結果ではない。
