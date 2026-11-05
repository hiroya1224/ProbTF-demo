# 姿勢と並進加速度の共同推定：人工IMUとMadgwickの比較

この実験は、2026年11月6日に指定された方法を、独立したIMU姿勢推定デモに追加したものである。ドローンの機体・一次遅れ同定からは独立しており、既存の実測比較と等方ノイズの人工動画を上書きしない。ここでは最新指定に従い、一般の非等方ノイズを保持する。

## 推定するものと既知条件

未知状態は、IMUから世界への姿勢qと、センサ位置の並進加速度をbodyで表したaである。較正、重力g、世界磁場bは既知固定で、バイアス・スケール・取付けずれを追加同定しない。

\[
y_\omega=\omega+n_\omega,\qquad
y_a=a-R(q)^Tg+n_a,\qquad
y_m=R(q)^Tb+n_m.
\]

センサ共分散は全3×3行列で、生成側と推定側を意図的に異ならせる。センサ内の非等方性と相関を保持し、センサ間・時刻間の生成白色ノイズは独立とする。未モデル化バイアスは生成しない。

同じ初期姿勢を全方式へ与え、その後の真値は評価だけに用いる。推定器は生成側の共分散、将来の姿勢、並進加速度・位置の真値を受け取らない。入力境界は `translational_synthetic_data.estimator_inputs` で分離する。

## 保持する共同法則

\[
p(q,a)=p_\theta(q)\,\mathcal N(a;m+Fh_g(q),S),\quad h_v(q)=R(q)^Tv,
\]
\[
p_\theta(q)\propto\exp\{q^TAq-\tfrac12h_g(q)^TW_gh_g(q)
                                  -\tfrac12h_b(q)^TW_bh_b(q)\}.
\]

Aは4×4、WgとWbは3×3対称行列。スカラー単位行列の密度ゲージを除くと19自由係数である。通常のBinghamへ四次項を落とす実装ではない。加速度と姿勢のcouplingはFで保持する。時間予測の射影後にWbが非零になることは、磁気観測を捏造したことを意味しない。Wg/Wbは十分統計量の基底に対する係数であり、観測履歴をセンサ別に分解した帳簿ではない。今回の観測因子は別途構成する。

## 解析的な観測更新

V=S+Σa、G=F−I、d=ya−mとして、

\[
L_a(q)=\mathcal N(y_a;m+Gh_g(q),V),\qquad
A_a=H(g,G^TV^{-1}d),\quad W_a=G^TV^{-1}G,
\]

\[
q^TH(x,z)q=x^TR(q)z.
\]

並進加速度をGaussian積分で周辺化した式であり、観測を重力単体へ置換したり、接空間の線形観測へ置換したりしない。非等方性による四次項を残す。

地磁気は、新規取得された時刻だけ

\[
A_m=H(b,\Sigma_m^{-1}y_m),\qquad W_m=\Sigma_m^{-1}
\]

を構成する。保持値を独立な観測として再融合しない。事前は一つだけ使い、

\[
A^+=A^-+A_a+A_m,\quad W_g^+=W_g^-+W_a,\quad W_b^+=W_b^-+W_m
\]

と合成する。地磁気がないtickではAm/Wmを加えない。今回の指定どおり、この統合結果を次時刻へ返す。

K=SV⁻¹として、条件付き加速度は

\[
m^+=m+Kd,\quad F^+=F-KG,\quad S^+=S-SV^{-1}S
\]

で更新する。実装では最後の式と同値なJoseph形を使う。これは丸めを抑える計算順の選択で、分散の追加ではない。

## 同じ不定回転による時間予測

公称増分はジャイロ読み値×実dtから作る。増分ノイズは、body小回転共分散C=Σωdt²から、高集中の対応としてBinghamのvector blockを−2C⁻¹に置いた**Bingham過程モデル**を採用する。Gaussian角速度の指数写像そのものと厳密同一とは主張しない。

既存coreの `bingham_second_moment`、`bingham_fourth_moment`、quaternion積モーメントを用いる。正規化は既存の決定論的200点計算で、coreを変更しない。姿勢qと加速度aには同じ増分δを作用させる。

\[
q'=q\odot\delta,\qquad a'=\beta D^Ta+w,\quad
\beta=e^{-dt/\tau_a},\quad w\sim\mathcal N(0,Q_a).
\]

\[
E[a'h_g'^T]=\beta E_\delta[D^TE[ah_g^T]D].
\]

回転の二次・四次モーメントから必要な平均・全共分散・crossを伝播する。aとqへ独立な二つの回転を当てる置換はしない。小さい回転ノイズの中心項はRnoise−Iの二次形式で計算し、ほぼ1の二つの差によってノイズを消さない。

## 有限表現へ戻す位置と近似

不定増分後の共同法則は、一般にこの族へ厳密には閉じない。予測したモーメントからforward-KL射影を行う。

- 姿勢側はMq=E[qqᵀ]、Ngg=E[hg hgᵀ]、Nbb=E[hb hbᵀ]を合わせる19係数の双対問題を、既定3反復で近似する。
- 条件付き加速度側は、予測した共同モーメントからF=Cov(a,hg)Cov(hg)⁻¹、m=E[a]−F E[hg]、S=Cov(a)−F Cov(hg)Fᵀを求める。

観測後の正規化とモーメントには、射影座標

\[
q(z)=\frac{q_{ref}+Ez}{\sqrt{1+z^Tz}},\qquad
Z=2\int_{\mathbb R^3}e^{\ell(q(z))}(1+z^Tz)^{-2}\,dz
\]

における決定論的Gauss–Hermite求積を用いる。既定は基準chartに加え、S³上に空間的に分散させた24個の固定初期値と、Haar格子の密度上位3点から、batch化したNewton探索で得た局所modeのchartを使う。各chartのGHは7³=343点で、球面全体を覆うHopf座標のHaar積分576点を組み合わせる。chart数Cに対して343C+576点となる。24個の初期値は座標軸4個、符号付き二軸方向12個、符号を同一視した四次元cube方向8個で、最適化の開始点に限る。これらや積分点を候補の粒子として保持・更新しない。点は積分の評価点であり、粒子・モンテカルロ標本ではない。積分のJacobianと四次の対数密度はそのまま評価する。局所曲率は積分点を配置するproposalにだけ使い、姿勢分布をそのGaussianで置き換えない。同じ峰へ収束した数値proposalは、符号同一視したquaternion間距離5×10⁻⁷以内でまとめる。これは積分proposalの重複整理であり、候補lawの枝・確率質量を捨てる操作ではない。元の基準proposalは別に保持し、予測で新しく生じたgyro分散を覆う。

局所proposalだけでは遠方の四次項の山を見落としうるため、正規化定数とモーメント、射影の目的関数に同じ全S³積分を使う。積分proposalはC個のlocal proposalとHaar一様密度uの混合rmix=(α/C)Σj rj+(1−α)uとし、α=0.999999は積分の重なりを調整する数値設定に限る。密度pθをこの割合のmixtureへ置き換えるものではない。

\[
\int f\,d\sigma
=\frac{\alpha}{C}\sum_{j=1}^{C}E_{r_j}[f/r_{mix}]+(1-\alpha)E_u[f/r_{mix}].
\]

遠方ではrmix≈(1−α)uとなるため、係数は相殺され、遠方の確率質量を10⁻⁶倍に弱める処理ではない。各期待値をそれぞれの決定論的な積分規則で評価する。数値探索の初期値も、元の輸送lawと曲率seedの二つについて、同じ連続S³上の双対目的関数を比べて選ぶ。最適化の各試行lawに対してchart・積分点・正規化定数・勾配・Hessianを計算し直す。最初の有限点集合に対する指数重みの変更だけでは、連続分布の射影を代用しない。

射影の積分点は、古い狭い分布の曲率だけではなく、予測したMqからCproposal=EᵀMqE/(qrefᵀMq qref)というモーメント比を使って配置する。これは射影座標の共分散の厳密式ではなく、積分proposalの選択である。二次項の曲率もこの幅に合わせて数値探索の初期値を選び、四次項Wg/Wbは保持する。その後の最適化の目標は、変更していない元の全モーメントである。これにより、極端に集中した事前からでも、今回のジャイロ分散を積分点の外へ落とさない。

高集中時の微弱な四次方向を白色化して無理に逆算すると、大きな係数と離れた偽modeが生じたため、自然係数上のdamped Newtonで有限射影する。dampingは最適化ステップの局所方程式だけに使い、状態共分散、目標モーメント、観測因子にjitterを加えない。全19方向を保持する。有限反復の残差を保存し、未収束の有限値を理由に以後の観測更新を止めない。

代表modeの数値探索は、各反復で四元数を基準に取り直す有限Newton探索とする。探索座標を無限大へ飛ばさず、exactな密度・勾配・Hessianを用いた局所ステップとbacktrackingを使う。この数値探索の制御も、姿勢の法則や状態共分散へ制限を加えるものではない。

したがって、Gaussian周辺化・観測因子の係数加算は解析的だが、全体を「完全な解析フィルタ」とは呼ばない。数値積分・有限射影、加速度の時間モデルという近似・仮定がある。Haar側を含めても有限解像度の積分である。非常に狭い未解像の峰を全て捕捉する保証はなく、GH次数とHaar解像度を上げ、さらに別の固定回転を施した24個の初期値も使う固定状態の感度確認を別に記録する。

## 比較条件

モデル・センサの[基本設定](config/translational_comparison.json)を固定し、数値積分の計算負荷を抑えた[オフライン比較設定](config/translational_comparison_reference.json)で結果を作る。後者は30秒・50Hzの別取得条件であり、60秒・200Hzの列を間引いた結果ではない。雑音の設定値は連続時間のパワースペクトル密度ではなく、1観測当たりの共分散である。レート変更後も同じ宣言値を使うため、両設定を同じ物理雑音過程と扱わない。

| 項目 | 設定 |
|---|---|
| 期間 | 各30秒 |
| gyro / accel | 50 Hz |
| 地磁気 | 3 tickごと、約16.7 Hz |
| 真の姿勢 | roll振幅20°、pitch振幅15°、yawも変化 |
| 並進 | なし／滑らかな位置軌道／持続加速度パルス |
| 加速度の時間モデル | τa=0.7秒、定常std=1.5 m/s²、Qa=(1−β²)×定常共分散 |
| 初期加速度 | 平均0、std=0.3 m/s²の事前。各時刻の真値は使わない |
| 初期姿勢 | 全方式で同じ初期quaternion、提案法の初期std尺度3° |
| Madgwick既存gain | β=0.1、元のEuler積分と正規化勾配 |

3条件は姿勢・実現したセンサ雑音を共有し、並進だけを変える。人工データの比力はRᵀ(a_world−g)から生成する。空力・ドローンの機体シミュレーションとは説明しない。

比較方式は、共同推定6軸／9軸、並進を厳密に零とした従来観測モデル9軸、Madgwick IMU／MARG、gyro積分。全native時刻の出力からピッチRMSE等を計算する。動的区間だけの評価は補助値として区別する。gainや過程ノイズを真値で最適化しない。

Madgwickの参照は[2010年原報告](https://x-io.co.uk/downloads/madgwick_internal_report.pdf)と[著者の公開コード案内](https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/)。標準MARGは観測から磁気dipを求め、full共分散を使わない。共同推定側は既知磁場と共分散、加速度の時間モデルを使うため、同じセンサ入力でも同じ構造・情報の使い方ではない。

さらに[事前固定したgain比較](config/translational_madgwick_gain_comparison.json)として、推定側ジャイロ共分散だけからβrms=0.5√trace(Σω)を計算したMadgwickも追加する。これは原報告§3.6の誤差尺度に対応するquaternion微分RMSであり、最適gainの保証ではない。生成側共分散や真値を使わず、β=0.1の結果も残す。

## 図・保存値・再現

```bash
run_src=ros/examples/probtf_orientation_demo/scripts
run_out=/tmp/probtf-translation-comparison
publish_out=ros/examples/probtf_orientation_demo/outputs/synthetic_translation_coupled_20261106
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 "$run_src/compare_translational_acceleration.py" \
  --config ros/examples/probtf_orientation_demo/config/translational_comparison_reference.json --output "$run_out"
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 "$run_src/validate_translational_quadrature.py" \
  --run-root "$run_out" --output "$run_out/fixed_state_quadrature_check.json" --stride 50 --include-indices 641 642
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 "$run_src/validate_translational_global_modes.py" \
  --run-root "$run_out" --output "$run_out/global_modes_check.json" --stride 50 --extra-indices 641 642 --maxseeds 6 --jobs 3
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 "$run_src/summarize_translational_comparison.py" \
  --run-root "$run_out" --publication "$publish_out"
```

カメラ・RViz・rosbagは不要。新実験の生入力・全native推定状態・中間図は/tmp側へ保持し、Gitには比較図、全時刻で計算した統計、全native時刻の図用数値列（必要な項目のみ）、再現コードと設定を保存する。表示用間引きから統計を計算し直さない。既存図の上書きや、推定結果の表示用平滑化は行わない。

ピッチ二乗誤差に対する点出力は、同じ事後分布から計算した**事後ピッチ平均**とする。Madgwick・gyro積分はそれぞれの点姿勢のピッチを使う。局所密度modeの誤差も別に保存し、結果を見て良い方を選ばない。次の積分chartの参照はMqの主固有vectorで選ぶ確率質量の中心とし、鋭いが質量の小さいmodeへ参照が引かれることを防ぐ。これは分布を単一姿勢へ置換する処理ではない。APIの`mean_quaternion_wxyz`はこの主固有vectorであり、反対称なquaternionの算術平均ではない。

帯は事後ピッチ平均±2×同じlawのピッチ標準偏差の診断表示であり、中央95%区間や被覆保証とは呼ばない。

今回の正確な初期姿勢・既知較正・小さいgyro白色ノイズという条件では、gyroだけでも高精度になりうる。一般の未知バイアス、初期誤差、大きな外乱、多峰分布への優越をこの比較だけで主張しない。実装は精度を確認するオフライン参照であり、センサ取得周期でのリアルタイム動作の達成とは区別する。

## 検証コマンド

```bash
PYTHONPATH=ros/core/probtf_core/src:ros/examples/probtf_orientation_demo/scripts \
  OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -m unittest discover \
  -s tests/probtf -p 'test_translational_*.py' -q
```

観測因子の密度式、Gaussian周辺化・条件付き更新、共有回転のcross、極小のgyro分散、遠方に峰を持つ既知Binghamの正規化、各射影候補での正規化再計算、真値の入力境界、地磁気の取得maskを検査する。保存状態の積分精度は上の二つのvalidatorで別に確認する。有限解像度どうしの一致を、全時刻・全分布に対する厳密性の証明とはしない。

比較結果と解釈は[今回の結果](outputs/synthetic_translation_coupled_20261106/README.md)にまとめた。
