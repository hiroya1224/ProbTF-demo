# 同じ観測前事前を固定した非線形観測更新

2026年10月18日、ユーザーは「保留を解除し、非線形更新を設計・実装する」と承認した。本書はC++逐次化の後に追加する観測更新を記述する。実装・実走の検証結果は末尾に区別して記録する。既存のIK、剛性BFS、物理シミュレーション、RViz設定はこの変更の対象ではない。

## 修正する問題

旧方式は各取得ごとに前の事後を使っていたが、単一観測の処理中は観測写像を一次式に固定していた。そこで求めた有限な姿勢補正を全リンクへ合成すると、その一次式が見込んだ観測との一致を失う例があった。[元の取得に対する診断](validation/cpp_sequential_live_20261018/noninitial_jump_diagnosis/DIAGNOSIS.md)に数式と反例を記録している。

新方式は、一件の取得時刻・実指令と、その直前の共同事前を固定したまま、有限なリンク変換を合成して同じ尤度を評価する。最適化中の反復は同じ関数を解く計算であり、観測を反復回数だけ融合する処理ではない。ここで変えるのはリンク姿勢候補であり、IKで選ぶ関節指令ではない。

## 事前と候補の幾何

Keep/Resetの各親 $h$ の、Binghamに由来する既存の共同Gaussian代理を

$z\sim\mathcal N(m_h,P_h),\qquad P_h=U_hD_hU_h^\top,\qquad L_h=U_h\sqrt{D_h}$

とする。$U_h,D_h$ は既存coreの支持計算が返すものを使い、固有値floorやjitterを加えない。構造上零のDirac座標は零のまま残す。支持上で

$z=m_h+L_h\xi,\qquad \xi\sim\mathcal N(0,I_r)$

と表す。一件の処理中は $m_h,P_h,L_h$ とそのsource IDを変更しない。

各辺の有限候補は、既存の再表現と同じ写像

$t_e(z)=t_{e,0}+z_{e,t},\qquad q_e(z)=q_{e,0}\,\rho(z_{e,r})$

で構成する。$\rho$ は既存のrotation-vectorから単位四元数への写像である。経路では同じ辺の候補を共有し、逆変換も含めて有限の変換を合成する。これをデモ内のC++ kernelで評価し、元の共同座標に関する解析Jacobianも返す。

これは既存Bingham由来の局所Gaussian代理を有限候補へ写す近似モデルであり、元のBinghamの全域の密度を厳密に積分したものではない。coreの候補法則を別のLie代数Gaussianへ書き換える変更ではない。

候補一点の幾何写像には、coreの分布照会が計算する $O(P)$ の平均補正を足さない。その補正は候補を分布として押し出した際の平均に由来する。最終的な事後lawの表示・制御照会では、従来どおりcoreが全crossを含む平均・共分散を計算する。最適化した一点と、伝播した分布の平均を区別する。

## 観測側の座標を固定する

画像では各native成分 $k$ の元のjoint Laplace Gaussianを、検出側のmode $t_k,R_k$ と全共分散 $\Sigma_k$ から復元する。位置・方位の交差ブロックを残し、native log weight $\ell_k$ も保持する。

$r_k(z)= \begin{bmatrix} t_{CM}(z)-t_k\\ \operatorname{Log}(R_k^\top R_{CM}(z)) \end{bmatrix},\qquad \mathcal L_k(z)=e^{\ell_k}\mathcal N(r_k(z);0,\Sigma_k).$

Logは検出器のnative回転座標を比較するための演算である。全Keep/Resetが同じ $R_k,\Sigma_k,\ell_k$ を使い、親のmodeごとに別のGaussianを定義しない。scalar尤度の値を評価するため、候補点に応じた座標体積をさらに掛けない。これは画像Hessian由来のGaussian尤度代理であり、生画素尤度やnative conditional Bingham lawの全域密度そのものとは呼ばない。

IMUでは観測された単位重力方向 $b$ とその接平面の基底 $B$ を固定する。候補から予測した単位方向を $d(z)$ とし、

$v=B^\top d,\quad s=\|v\|,\quad c=b^\top d,\quad \theta=\operatorname{atan2}(s,c),\qquad r_{\rm imu}(z)=\frac{\theta}{s}v$

というS²の角度残差を使う。零角度の極限は解析的に評価する。尤度代理は既存の角度標準偏差 $\sigma$ に対する

$\mathcal L_{\rm imu}(z)= \mathcal N(r_{\rm imu}(z);0,\sigma^2I_2)$

であり、headingの観測を加えない。旧接平面への単純な射影は反対向きの $d=-b$ でも残差が零になるため、有限な非線形写像にはそのまま使わない。近傍では元の接平面残差に一次で一致する。

この2次元Gaussianは、固定された観測座標でのscalar尤度代理として使う。S²全域で正規化された方向密度と主張せず、追加の点依存体積を掛けない。正反対の点ではscalarの値は有限でもLogの方向・微分が一意でない。これを観測のNaNや欠測と取り違えず、真のchart特異点として記録する。近いが有限な候補を任意の角度閾値で捨てない。

## 同じ事前を一度だけ使う最適化

各親・native成分の組について、固定した残差 $r$ とproperな観測共分散 $R$ を使い、

$\Phi(\xi)=\frac12\|\xi\|^2+ \frac12r(m_h+L_h\xi)^\top R^{-1}r(m_h+L_h\xi)$

を最小化する。同時に複数の有効観測を渡す明示的な診断経路では、条件付き独立な各観測尤度の積を同じ事前へ一度掛ける。通常の逐次経路では原sequence順の一件ずつである。

$A=R^{-1/2}\frac{\partial r}{\partial z}L_h,\quad g=\xi+A^\top R^{-1/2}r,\quad H_{\rm GN}=I_r+A^\top A,\quad \Delta\xi=-H_{\rm GN}^{-1}g.$

更新方向はGauss–Newton法で求め、同じ非線形 $\Phi$ の値でbacktrackする。$I_r$ は白色化した元の事前のprecisionであり、跳ねを隠すために足す正則化や過程雑音ではない。途中の事後を新しい事前へ入れ直さない。

有限の反復予算内で得た点 $\widehat\xi$ を使う。stationarity、反復数、line search、元事前の数値hashを各枝に記録する。反復上限・有限な非収束・精度診断の未達で、その後の取得処理や動作を止めない。NaNや実際に必要な分解不能は演算・入力・出自を記録し、架空のcovarianceやevidenceで隠さない。

## 事後とevidenceの近似

各枝のGaussian代理は

$\widehat m=m_h+L_h\widehat\xi,\qquad \widehat P=L_hH_{\rm GN}^{-1}L_h^\top$

とする。全共同ブロックを一度に計算し、辺ごとの独立な周辺分布に分解しない。これは有限反復点を中心とするGauss–Newton/Laplace近似であり、厳密な非線形事後の平均・共分散、厳密Hessianを使うLaplace近似、または厳密なmoment matchingとは呼ばない。未収束点では接二次式を平方完成した積分とも異なるため、その境界を診断へ残す。

観測次元を $d$ とすると、native log weightを除く各枝のevidence近似は

$\log\widehat Z_{h,k} =-\Phi(\widehat\xi)-\frac d2\log(2\pi) -\frac12\log\det R-\frac12\log\det H_{\rm GN}.$

支持白色化の体積は元のprior densityの逆係数と相殺する。支持のdeterminantや $(2\pi)^{r/2}$ をさらに加えない。位置・方位の全共分散のdeterminantを使い、native log weight、Keep/Resetのlog prior weightを加えて全枝でlog-sum-expする。小さいlog質量を大きい共通項へ先に足して消さない。

残差がaffineなら一回の完全なGauss–Newton更新で通常のGaussian条件付けに戻り、平均・全共分散・evidenceの式も一致する。支持rank 0では候補はDiracのまま、evidenceだけがその一点の尤度になる。無観測・未検出・未選択では新しい観測更新をせず、従来の恒等保持を守る。

## 次の更新、剛性、表示との接続

新方式は元の共同prior chartで求めた事後枝を直接、既存の全within/between単一化・Bingham再表現へ渡す。旧SensorPosteriorのaffine liftingをもう一度適用しない。観測側のlocal posteriorを記録する場合は、この共同事後をcoreで押し出したviewと明示する。

その同じcanonical lawを次事前・K密度比の分子・制御・表示へ渡す。K比の分母は実際に使ったKeep/Reset遷移事前のまま保持する。原t/q/ID、各実行区間の凍結K、全cross、native画像成分、固定目標の構成と保持は変えない。

非線形な観測との整合が改善しても、全リンクの絶対姿勢の識別やResetに伴う全ての大更新が解決するとは限らない。マーカー相対観測の識別性、保持地図、Reset kernelの仮定を分けて評価する。真値は別の後解析にのみ使い、推定や制御へ入力しない。実走の滑らかさ・到達と、固定入力での数値的一致を別々に検証する。


## 固定入力での検証（2026年10月18日）

厳密な反平行点の境界修正後、全回帰は1,126件通過した。実行コマンド・元commit・sourceのSHA・出力logのSHAを [regression_manifest.json](validation/nonlinear_update_20261018/regression_manifest.json)、実出力を [regression_test_output.txt](validation/nonlinear_update_20261018/regression_test_output.txt) に保存した。これは数値実装の検証であり、実動作の跳ね解消や到達の証拠とは区別する。

元90秒試行から取り出した異常更新3件について、Keep/Resetの6枝を原取得画像から再構成した。C++を呼ぶ実装と、独立したNumPy有限写像・Gauss–Newton実装を照合し、平均の最大絶対差は $1.62\times10^{-14}$、全共分散は $1.18\times10^{-14}$、log evidenceは $1.23\times10^{-11}$ 以下だった。元画像のnative中央lawは保存lawとbitwise一致した。この固定画像検算と後述の90秒実走は、反平行境界修正前の1,124テスト通過sourceで行った。詳細は [production_fixture_audit.json](validation/nonlinear_update_20261018/production_fixture_audit.json) に記録した。元の履歴で形成された事前を固定した一観測の比較であり、初期状態からの新しい逐次実行とは異なる。

独立したactual Hessianとの比較では、Gauss–Newtonの共分散との差がKeep枝で50〜76%となる例もあった。両者を同じ近似とは説明しない。観測との一致が改善しても、Marker単独の実リンク絶対姿勢の誤補正が残る例もある。比較する点・共分散・evidence・物理後解析の区別と数値は [独立prototypeの報告](validation/nonlinear_update_20261018/independent_prototype/DIAGNOSIS.md) に記録した。

有限候補の経路合成API単体では、11辺・66座標の8条件でC++が72〜78マイクロ秒、独立Python参照が937〜2,216マイクロ秒だった。値・解析Jacobianの最大絶対差は $8.69\times10^{-15}$ 以下だった。これは単一APIの12.9〜28.8倍という比較であり、観測更新全体やK計算、ROS実動作がその倍率で速くなったとは主張しない。[単体計測](validation/nonlinear_update_20261018/finite_modal_benchmark.json) を保存した。

core・RViz等の保護対象214ファイルはbaselineと全バイト一致した。旧数値kernelは追加前の全sourceをprefixとして保持し、新しい有限候補APIだけを末尾に追加している。既存core、画像・粒子・軸のサイズ、視点・レイアウトの変更は含まない。


## 真の特異点と近傍の有限候補

独立レビューで、IMUの正反対の方向 $d=-b$ において、非軸方向の $b$ ではbasisの丸めにより $B^\top d$ が厳密な零にならない場合を見つけた。その場合、旧境界判定は不定なLog軸を検出できなかった。新境界は既存の零接成分判定に加え、machine計算で外積が全成分零かつ内積が負の反平行だけを真のcutとして扱う。近傍を落とす角度閾値は追加していない。非軸の2例と $10^{-10}$ radの近傍を追加回帰した。これは観測尤度・GN式・共分散・BFS規準の変更ではない。

90秒実走の取得元sourceと、この最終境界修正後sourceはhashで区別する。実走sourceの回帰は [live_source_regression_manifest.json](validation/nonlinear_update_20261018/live_source_regression_manifest.json)、境界差分は [exact_antipode_fix_manifest.json](validation/nonlinear_update_20261018/exact_antipode_fix_manifest.json) に保存した。


## 初期状態からの90秒実走

MarkerとIMU＋Markerを各90秒動かした。最終実手先誤差は60.32 mm、64.62 mmで、どちらも到達未達だった。リンクの非初期補正は最大175.49 mm、56.55 mmで、大きな跳ねが残る。旧一次外挿の不一致は全638有効観測・1,276枝で検出されなかったが、画像の全native候補自体が外れたとき、現行BFSがそれを身体変化として説明して大きく補正する例を特定した。

最大例の直前画像は有効IDが一つだけで、両native候補が外れていた。中間ボード縮約を延期してもその外れは変わらなかった。4起点の同一関数再最適化でも大きく異なる解は見つからず、Kの同時採用でも説明できなかった。観測失敗の別仮説は現行モデルに定義されていない。無断の観測棄却・分散調整・表示平滑化は追加していない。

推定の跳ね・取得時刻と実指令の対応・wire同一性・未完K・実Hz・近似の限界を [実走・原因調査報告](validation/nonlinear_update_20261018/live/REPORT.md) にまとめた。全raw、未完因子、巨大ログは/tmpへ保持し、Gitへは小さい選定資料だけを含める。非線形更新を実装したことと、残る跳ねの問題を解決したことを区別する。
