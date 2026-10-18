# Gaussian変分更新の数値検証資料

2026年10月18日。既存の小さいログ・計測JSON・再現scriptを集約したもので、この集約時にはproduction、テスト、raw取得を変更していない。新しい実走も行っていない。正確な入力・source・library SHAと検証範囲は `report.json` に保存した。

## 保存した証拠

|対象|件数|原ログ|
|---|---:|---|
|generic solver|15 PASS|`logs/generic_15.log`|
|native逆微分|43 PASS|`logs/native_43_transport_16.log` 内の59件|
|固定目標の周辺輸送|16 PASS|同上|
|model integration|7 PASSと担当者報告|専用ログは保存されていない。実ログ `logs/integrated_55.log` は関連テストを合わせた55 PASS。重複するため合計件数に足さない。|

全回帰と新180秒実走の最終結果はrootが別に管理する。この資料のfocused testsを、実到達・跳ね解消・全体実時間性能の証拠にはしない。

## 固定事前と全Gaussianの微分

原事前の実支持上で $P_0=BB^\top$、$B=U\sqrt D$ とする。変数は候補一点ではなく、$m=m_0+Ba$、$P=BLL^\top B^\top$ というGaussianの平均と全共分散である。$L$の全下三角要素を最適化し、対角はlog値で表す。

$F(q)=\operatorname{KL}(q\Vert p_0)+\sum_s\mathbb E_q[-\log L_s],\qquad \operatorname{KL}=\tfrac12(a^\top a+\operatorname{tr}LL^\top-2\log\det L-r).$

各trialの伝播補正平均 $\mu(q)$、全共分散 $S(q)$ から、Gaussian観測の期待負対数尤度を

$\tfrac12\{(\mu-y)^\top R^{-1}(\mu-y)+\operatorname{tr}(R^{-1}S)+\log\det R+d\log(2\pi)\}$

で評価する。native正規化項は一回だけ含める。backendの $g_m,G_P$ をfull symmetric Frobenius規約で受け、

$\nabla_aF=a+B^\top g_m,\qquad \nabla_LF=L-L^{-\top}+2B^\top\operatorname{sym}(G_P)BL$

を使う。log対角の微分には対応する $L_{ii}$ を掛ける。有限差分は検証だけに使い、production逆微分や候補生成でサンプリングはしない。

一回の最適化で $p_0$ と観測を固定する。旧Gaussian条件付けで作るseedも同じ $p_0$ に対するKLを払い、同じFで元事前候補と比較する。seedを新priorに差し替えず、観測を二度掛けない。rank0と0反復設定では元priorを保持する。

## affine極限と逐次蓄積

厳密なaffine Gaussian尤度では

$F(q)=\operatorname{KL}(q\Vert p(z\mid y))-\log p(y).$

真事後が対象Gaussian族に含まれるため、変分最適点では解析事後とevidenceに一致する。異方・非対角事前、相関R、Dirac支持でoracleと比較した。analytic affine seedも同じ原priorの目的でstationaryとなり、追加反復なしの一致を確認している。

次の新観測では、前観測から保持した事後をその観測の固定priorにする。二つの異なる観測の逐次結果・各evidenceを解析Bayes再帰と比較した。有限最適化誤差は次のprior/evidenceにも伝わるので、有限予算だけで厳密一致を保証するとは説明しない。

`benchmarks/rank54_affine_prior_start_and_exact_seed.json` はwarmstart API追加前の設計診断を原バイトで保存したもの。66次元・支持rank54・1539パラメータでは、prior開始24反復でgradient∞=8.103、evidence gap=3.424が残った。一方、同じ元priorのaffine解析seedはgradient∞=5.12e-12、evidence誤差4.26e-14だった。これはseed採用の根拠であり、非線形の既定24反復が収束する証拠ではない。元JSONの「production optionではない」という文は当時の記録であり、その後APIへ追加された経緯と区別する。

## 実66次元の逆微分検算と計時の範囲

`benchmarks/actual66d_vjp_and_directional_fd.json` は旧M93アーカイブの同じ原共同prior・native観測を使う。11辺・66座標、実支持rank42で、試行meanも非零、Pは全crossを含む。

- 独立core再表現・照会との最大差: mean 7.77e-16、covariance 1.27e-18、尤度cost 3.16e-11。
- 四つの決定論的なmean/P混合方向で、解析VJPとcore中心差分を比較。最大絶対差9.28e-6、最大相対差7.81e-8。中心差分幅2e-5。原fixtureの全パラメータ差分を取った計時ではない。
- nativeは**momentsと解析gradientの両方**でmedian 1.831 ms（50回）。Python core比較は**momentsだけ**でmedian 6.544 ms（25回）。

仕事量が異なる二つのbaselineである。これをVJP対「全勾配有限差分」の速度倍率、全推定器の更新周波数、ライブ制御性能と読み替えない。全勾配有限差分の計時は行っていない。計測時native source/binding/libraryのSHAは集約時にも一致した。

## 残る近似と未検証範囲

非線形ではcore高集中モーメント自体が近似なので、$-F$ はmoment-ELBO proxyであり、厳密evidenceや保証された下界ではない。局所最適化・有限未収束・最終canonical単峰化を別に診断する。原枝の改善をcanonical後の改善と同一視しない。finite best Gaussianで後続を続け、stationarity未達を停止条件にしていない。

固定目標OTは明示したcutの結合規約である。$APA^\top=Q$ による合同変換で共同PSDを保つが、元支持上でAが可逆の場合に限って条件付きSchur残差の等値も言える。支持が縮小する一般の場合はその等値を主張しない。

原job22のwarmstart比較はrootが `/tmp/probtf_variational_job22_audit_warmstart_20261018/` から別に選定する。本資料へ巨大packet・raw pixels・物理truthを複製していない。既存sourceを初期状態から走らせた結果でもない。

## 再現

workspaceのsetupを読み、BLAS/OMPを各1threadにする。専用テストはrepoの元パスから実行する。

```bash
source /home/leus/catkin_ws/devel/setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=/tmp/probtf_switch_press_pytest:$PYTHONPATH /usr/bin/python3 -m pytest -q ros/examples/probtf_observation_fusion_demo/tests/test_variational_conditioning.py ros/examples/probtf_observation_fusion_demo/tests/test_moment_reverse.py ros/examples/probtf_observation_fusion_demo/tests/test_gaussian_marginal_transport.py ros/examples/probtf_observation_fusion_demo/tests/test_variational_model.py
```

`reproduction/` の二つのprofile scriptは既存scriptの原バイトコピー。actual66d側は元fixture index `/tmp/probtf_propagation_comparison_fixtures_20261018/index.json` と、そこから参照するアーカイブpayloadを要する。大容量入力は同梱していない。出力先も元scriptの固定 `/tmp` パスのままであり、再実行はその既存診断JSONを上書きする。`report.json` の入力hashを照合してから、必要なときだけ明示実行する。
