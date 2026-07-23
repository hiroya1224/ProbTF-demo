# 確率的 AprilTag 姿勢推定の原理

## 0. この資料の目的

この資料は、`prob_artag_detector` が1枚の単眼画像から AprilTag の
**多峰な確率的姿勢**を構成し、ProbTF v2 の変換分布として公開するまでを、
実装を直接読まなくても再構成できる粒度で説明する。

入力は次の4要素である。

1. AprilTag のID
2. 順序付き4隅の画素座標
3. 4隅をまとめた \(8\times 8\) 画素共分散
4. カメラ内部パラメータ、歪み係数、実タグ寸法

出力は、平面姿勢の複数解を残した次の混合分布である。

$$
p(X,Q\mid y)
\approx
\sum_{\ell=1}^{K}
w_\ell\,
p_\ell(Q)\,
p_\ell(X\mid Q).
$$

ここで、

- \(Q\in S^3\) はマーカー姿勢を表す単位四元数
- \(X\in\mathbb{R}^3\) はカメラ座標系でのマーカー原点位置
- \(K\) は保持された局所姿勢モード数
- \(w_\ell\) は各モードの混合重み
- \(p_\ell(Q)\) は Bingham 姿勢分布
- \(p_\ell(X\mid Q)\) は姿勢に条件づけられた並進 Gaussian

である。

重要なのは、最終的な並進と姿勢を独立な Gaussian として扱わないことである。
画像上では姿勢の傾きと奥行き・横位置が強く結合するため、各成分は
**回転–並進相関を持つ同時分布**として表現される。

---

## 1. 全体像

1フレームの処理は、概念的には次の順番で進む。

```text
Image + CameraInfo
        |
        v
AprilTag ID と順序付き4隅を検出
        |
        +--> 同一フレームの微小摂動・再検出
        |        |
        |        v
        |    spatial 8x8 corner covariance
        |
        +--> タグIDごとの過去2フレーム
                 |
                 v
             temporal excess covariance
        |
        v
現在の4隅 y と full covariance Σ_img
        |
        v
IPPE_SQUARE で平面PnP候補をすべて取得
        |
        v
各候補を同じ Mahalanobis 目的関数で局所最適化
        |
        v
枝ごとの Gauss–Newton Hessian
        |
        +--> conditional translation (m, S, C)
        +--> Bingham orientation A
        +--> Laplace mass
        |
        v
重複・不正な枝を除外して重みを正規化
        |
        v
mixture ProbTF edge T_C_M をpublish
```

時系列処理は、現在の4隅を平均化するためには使わない。過去フレームは
**現在観測の共分散を広げる目的だけ**に使われる。そのため、出力位置が履歴に
引っ張られる暗黙のトラッカーにはならない。

---

## 2. 座標系、変換方向、単位

カメラの ROS optical frame を \(C\)、タグ座標系を \(M\) とする。
出力する変換は

$$
{}^{C}T_M=(R(Q),X)
$$

であり、タグ座標の点 \(z_M\) をカメラ座標へ写す。

$$
z_C = R(Q)z_M + X.
$$

ProbTFメッセージ上では、

- `header.frame_id`: カメラ optical frame
- `child_frame_id`: `apriltag_<id>`

となる。したがって、通常のTF記法では `camera -> apriltag_<id>` のエッジに
見えるが、数値的な作用は「childの座標をparentへ写す」
\({}^{C}T_M\) である。

OpenCV/ROS optical frame は次の右手系である。

- \(x_C\): 画像右
- \(y_C\): 画像下
- \(z_C\): カメラ前方

位置とタグ寸法の単位は m、画素観測とその標準偏差の単位は px、
回転の局所摂動は rad である。

### 2.1 タグ四隅

タグの黒い外枠を含む正方形の一辺を \(L\) とし、タグ中心を原点とする。
OpenCV `SOLVEPNP_IPPE_SQUARE` が要求する物体点順序は

$$
p_1=
\begin{bmatrix}-L/2\\+L/2\\0\end{bmatrix},\quad
p_2=
\begin{bmatrix}+L/2\\+L/2\\0\end{bmatrix},\quad
p_3=
\begin{bmatrix}+L/2\\-L/2\\0\end{bmatrix},\quad
p_4=
\begin{bmatrix}-L/2\\-L/2\\0\end{bmatrix}.
$$

これを detector が返す

1. top-left
2. top-right
3. bottom-right
4. bottom-left

の順と対応させる。順序が1点でもずれると、共分散以前に別の姿勢を解くことに
なる。

`tag_size_m` は印刷したタグのこの正方形の実測値に一致させる必要がある。
値を \(a\) 倍にすると、推定並進もほぼ \(a\) 倍になる。

---

## 3. カメラモデル

歪みなし pinhole の場合、カメラ座標
\(P=(P_x,P_y,P_z)^\top\) の投影は

$$
\pi(P)=
\begin{bmatrix}
f_x P_x/P_z+c_x\\
f_y P_y/P_z+c_y
\end{bmatrix}.
$$

実際の投影は OpenCV の `projectPoints` と同じモデルを使い、
`plumb_bob` または `rational_polynomial` の歪み係数も扱う。

第 \(j\) 四隅の予測画素は

$$
h_j(X,Q)=\pi\!\left(X+R(Q)p_j\right)
$$

であり、4点を積んだ

$$
h(X,Q)=
\begin{bmatrix}
h_1^\top & h_2^\top & h_3^\top & h_4^\top
\end{bmatrix}^{\!\top}
\in\mathbb{R}^8
$$

を姿勢尤度に使う。

### 3.1 CameraInfo の受理条件

定量的な推定では、入力画像と一致する `sensor_msgs/CameraInfo` を使う。
現行系が受理するのは次を満たすモデルである。

- 正の \(f_x,f_y\)
- 正の画像幅・高さ
- `plumb_bob`、`rational_polynomial`、または空の distortion model
- 既定設定では画像と同じ解像度
- binningなし
- ROI cropなし
- 0、4、5、8、12、14個の有限な歪み係数

raw imageに対応する `K` と `D` を使い、rectified image用の `R`、`P` は
使わない。CameraInfoはノード内で最新値を保持する方式で、Imageと厳密な
timestamp同期は行わない。

異なる解像度、binning、ROIの内部パラメータをそのまま流用すると、尤度の
座標系自体が不一致になる。そのため黙って補正せず、設定に応じて拒否または
fallbackへ切り替える。`require_calibration_resolution_match=false` なら
解像度検査だけは無効化できるが、内部行列を別解像度へ自動scaleする機能では
ない。

また、後述する無歪み解析Jacobianは通常のzero-skew内部行列を仮定する。
CameraModelは \(K_{01}=0\) を明示検査しないため、skewを持つ特殊なモデルでは
解析式をそのまま定量利用しないこと。

### 3.2 未校正 fallback

実演用に、水平画角 \(\theta_x\) と画像幅 \(W\) から

$$
f_x=\frac{W/2}{\tan(\theta_x/2)},\qquad f_y=f_x,
$$

$$
c_x=\frac{W-1}{2},\qquad c_y=\frac{H-1}{2}
$$

を作る zero-distortion fallback がある。既定の水平画角は
\(60^\circ\) である。

これはパイプラインを起動するための近似であり、焦点距離や歪みの不確かさは
後述の Hessian に入らない。fallback使用時は provenance と approximation
detail にその事実が残る。定量的な奥行きと共分散には実キャリブレーションが
必要である。

---

## 4. 画像観測ベクトル

既定の検出器は OpenCV ArUco APIの `DICT_APRILTAG_36h11` dictionaryを使う。
BGR/BGRA入力はgrayscale `uint8` へ変換し、既定では
`CORNER_REFINE_SUBPIX` により四隅をsubpixel refinementする。dictionaryの
内部符号からIDと面内回転を同定した後、姿勢推定器へ渡すのはID、順序付き4隅、
画素共分散だけである。PnPは検出段から分離されている。

検出された4隅を

$$
y=
\begin{bmatrix}
u_1&v_1&u_2&v_2&u_3&v_3&u_4&v_4
\end{bmatrix}^{\!\top}
\in\mathbb{R}^8
$$

と置く。観測モデルは

$$
y=h(X,Q)+\varepsilon,\qquad
\varepsilon\sim\mathcal{N}(0,\Sigma_{\mathrm{img}})
$$

である。

\(\Sigma_{\mathrm{img}}\) は単なる最終出力の飾りではない。
その逆行列が最適化の重みとなるため、

- 局所モードの位置
- 並進・回転の局所精度
- 回転–並進結合
- Bingham の集中度
- 各モードの Laplace 重み

のすべてへ伝播する。

現行系では、画素共分散を

$$
\Sigma_{\mathrm{img}}
=
\Sigma_{\mathrm{spatial}}
+
\Sigma_{\mathrm{temporal,excess}}
$$

として構成する。

---

## 5. 同一フレームの spatial corner covariance

### 5.1 最小の正定値 floor

既定の基礎共分散は

$$
\Sigma_{\mathrm{floor}}
=
\sigma_0^2 I_8,\qquad
\sigma_0=0.5\ \mathrm{px}
$$

である。この値は全誤差を表す固定モデルではなく、adaptive推定が過小になら
ないための正定値 floor として扱われる。

明示的な \(8\times 8\) 共分散をAPIへ渡した場合は、それが厳密な固定
overrideとなり、同一フレームbootstrapは無効になる。

### 5.2 micro-bootstrap の考え方

元画像で得た四隅を \(y^{(0)}\) とする。同じタグ周辺の画像へ、再現可能な
微小摂動を \(N\) 通り加え、同じ detector で再検出する。

各摂動には次が含まれる。

- subpixel平行移動
- Gaussian blur
- 画素強度 Gaussian noise

平行移動とnoiseは正負の対を作り、random seedを固定する。これにより、
同じ画像と設定からは同じ共分散が得られる。

タグごとに、四隅のmedian辺長を \(s_{\mathrm{img}}\) として

$$
m_{\mathrm{ROI}}=\max(12\ \mathrm{px},\,0.5s_{\mathrm{img}})
$$

の余白を持つ局所ROIを切り出す。同じIDの再検出結果から元の四隅に最も近い
候補を選び、そのcorner RMS距離が

$$
\max(5\ \mathrm{px},\,0.25\times\text{mean edge length})
$$

を超える候補は対応失敗とする。付加した画像平行移動は検出座標から差し引く
ため、測っているのは既知warpそのものではなく、そのwarpに対する detector
の局在感度である。

成功した再検出を \(y^{(b)}\)、その差を

$$
\Delta_b=y^{(b)}-y^{(0)},\qquad b=1,\ldots,n_s
$$

とする。平均を引いた標本共分散ではなく、報告値 \(y^{(0)}\) のまわりの
raw second moment

$$
\widehat M
=
\frac{1}{n_s}
\sum_{b=1}^{n_s}
\Delta_b\Delta_b^\top
$$

を使う。

これは

$$
\widehat M
\approx
\operatorname{Cov}(\Delta)
+
\operatorname{E}[\Delta]\operatorname{E}[\Delta]^\top
$$

であり、分散だけでなく再現性のある局在biasもMSEとして数える。したがって、
エッジのギザギザ、aliasing、blurに伴う系統的なcorner移動も、可能な範囲で
共分散へ反映される。

### 5.3 full \(8\times 8\) 相関

\(\widehat M\) は対角だけを使わない。たとえば上辺の2隅が同じ方向へ動けば、
\((u_1,u_2)\) 間の共分散が残る。この相関は、四角形全体の平行移動・回転・
せん断のような detector の揺らぎを表すために重要である。

少数サンプルで過度な相関を作らないよう、対角への shrinkage を行う。

$$
\widehat M_\lambda
=
(1-\lambda)\widehat M
+
\lambda\,\operatorname{diag}(\widehat M).
$$

既定値は \(\lambda=0.25\) である。

### 5.4 再検出失敗の扱い

成功率を

$$
\rho=\frac{n_s}{N}
$$

とする。成功率が既定の下限
\(\rho_{\min}=0.5\) より小さい場合、観測できたMSEを

$$
g(\rho)=
\left(
\frac{\rho_{\min}}{\max(\rho,1/N)}
\right)^2
$$

倍する。さらに、見失った割合へ

$$
(1-\rho)\sigma_{\mathrm{drop}}^2 I_8
$$

を加える。既定の \(\sigma_{\mathrm{drop}}\) は \(2.0\) px である。
全再検出が失敗した場合でも、この項により「安定している」と誤解しない。

最終的な同一フレーム共分散は概ね

$$
\Sigma_{\mathrm{spatial}}
=
\Sigma_{\mathrm{floor}}
+
g\,\widehat M_\lambda
+
(1-\rho)\sigma_{\mathrm{drop}}^2 I_8
$$

であり、最後に対称化し、machine epsilon未満の固有値を正の微小値へfloorする。

この処理は厳密な独立標本bootstrapではなく、
**既知の小摂動に対する detector 感度から局在MSEを測る局所実験**である。
摂動族に含まれないrolling shutterや強いmotion blurまで保証するものではない。

---

## 6. 時系列の temporal excess covariance

同一フレームbootstrapだけでは、フレームごとに変わるthreshold、subpixel fit、
圧縮ノイズなどによる揺れを取りこぼすことがある。そのため、タグfamilyとIDの
組ごとに短い履歴を持ち、現在の spatial model を超えるジッタだけを推定する。

### 6.1 不規則時刻に対応した定速度innovation

3時刻の画素ベクトルを

$$
(t_0,z_0),\quad(t_1,z_1),\quad(t_2,z_2)
$$

とし、

$$
\Delta t_0=t_1-t_0,\qquad
\Delta t_1=t_2-t_1,\qquad
r=\frac{\Delta t_1}{\Delta t_0}
$$

と置く。定速度予測は

$$
\widehat z_2=(1+r)z_1-rz_0
$$

である。raw residual と正規化innovationは

$$
e_2=z_2-\widehat z_2,
$$

$$
\nu_2=
\frac{e_2}
{\sqrt{1+(1+r)^2+r^2}}.
$$

画素軌道が時間に対して完全に一次なら、不規則なフレーム間隔でも
\(\nu_2=0\) になる。

各観測の spatial covariance を \(R_0,R_1,R_2\) とすると、独立観測に対する
innovation自身の既知共分散は

$$
R_{\nu,2}
=
\frac{
R_2+(1+r)^2R_1+r^2R_0
}{
1+(1+r)^2+r^2
}.
$$

3フレームが同じ \(R\) を持つなら \(R_{\nu,2}=R\) となるよう正規化されて
いる。

### 6.2 spatial成分を二重計上しない

innovation列から得た共分散を \(\widehat\Sigma_\nu\)、同じ期間の
\(R_\nu\) の平均を \(\overline R_\nu\) とする。時系列で追加すべきなのは

$$
\widehat\Sigma_\nu-\overline R_\nu
$$

の正の部分だけである。spatial covarianceを先に差し引くため、bootstrapが
すでに表した相関ノイズをtemporal段でもう一度足さない。

差分を対角へshrinkした後、固有値分解でPSD部分を取る。

$$
E_{\mathrm{raw}}
=
\widehat\Sigma_\nu-\overline R_\nu,
$$

$$
E_{\lambda}
=
(1-\lambda_t)E_{\mathrm{raw}}
+
\lambda_t\operatorname{diag}(E_{\mathrm{raw}}),
$$

$$
E_+
=
V\operatorname{diag}\!\left(
\min(\max(\eta_i,0),\sigma_{\max}^2)
\right)V^\top.
$$

したがって、出力の各固有値は

$$
0\leq \eta_i\leq \sigma_{\max}^2
$$

へ制限される。既定の
\(\sigma_{\max}=5.0\) px は**追加temporal成分だけ**の上限であり、
大きな spatial covariance を縮めることはない。

出力は

$$
\Sigma_{\mathrm{img},2}=R_2+E_+
$$

となる。

### 6.3 warmup と忘却

既定では8個の有効innovationが集まるまで \(E_+=0\) とする。これは8フレーム
ではなく、統計更新へ受理されたinnovationが8個という意味である。hard
outlierはtrackをresetし、optional freeze有効時の `motion_frozen` はこの数へ
加算されない。

warmup中は通常の標本平均・標本共分散を使い、その後は時定数ではなく
half-life \(T_{1/2}\) で指定した指数忘却へ切り替える。

$$
\alpha
=
1-\exp\!\left(
-\ln 2\,\frac{\Delta t}{T_{1/2}}
\right).
$$

既定の \(T_{1/2}\) は \(0.5\) s である。

連続する3フレーム窓は互いに重なるため、innovation列は厳密には独立標本では
ない。このrunning covarianceは実用的なnoise inflation推定であり、完全な
時系列生成モデルの最尤推定ではない。また、前フレームの処理中は新しい画像を
dropするため、ここで使う \(\Delta t\) はカメラの公称周期ではなく、実際に
処理された観測時刻の差である。

### 6.4 robust clipping

現在の予測共分散を

$$
\Sigma_{\mathrm{ref}}=R_{\nu,2}+E_+
$$

とし、innovation平均 \(\bar\nu\) に対するMahalanobis距離

$$
d^2
=
(\nu_2-\bar\nu)^\top
\Sigma_{\mathrm{ref}}^{-1}
(\nu_2-\bar\nu)
$$

を求める。warmup完了後に \(d^2>\chi^2_{\max}\) なら、

$$
\nu_2
\leftarrow
\bar\nu
+
\sqrt{\frac{\chi^2_{\max}}{d^2}}
(\nu_2-\bar\nu)
$$

として更新量をHuber型に制限する。既定の閾値は \(20.09\) である。
warmup前は、実際に過小評価されているジッタが育たなくなるのを避けるため、
このclippingを行わない。

### 6.5 motion、outlier、gap

raw residualを4本の2Dベクトルへ戻し、そのRMSを

$$
e_{\mathrm{rms}}
=
\sqrt{
\frac{1}{4}
\sum_{j=1}^{4}\|e_j\|^2
}
$$

とする。gateは固定px値とタグの画像上のmedian辺長に対する割合の大きい方を
使うため、タグの見かけサイズにある程度追従する。

非常に大きな不整合がhard outlier閾値を超えた場合は、

$$
\Sigma_{\mathrm{img}}
=
\Sigma_{\mathrm{spatial}}
+
\sigma_{\max}^2 I_8
$$

をそのフレームへ返し、履歴をリセットする。誤ったcorner setを次の速度予測へ
残さないためである。

共有平行移動、回転、scale、shearなど、1つのaffine warpで説明できる
residualの割合も最小二乗で測る。`temporal_freeze_affine_motion=true` の場合
だけ、十分大きくかつaffine説明率が高いinnovationを実運動として学習停止する。

既定値は `false` である。単眼の4隅だけでは、カメラ・タグの加速度運動と、
4隅が同じ方向へ跳ねる detector jitter を完全には分離できないため、既定では
後者を見逃さない保守的な側へ倒している。

次の場合もtrackをリセットする。

- timestampが逆行または同一
- フレーム間隔が `temporal_max_gap_sec` を超える
- 直前と現在の \(\Delta t\) 比が許容範囲外
- camera frame、内部パラメータ、歪み、解像度、calibration sourceが変わる

長時間見えない別タグのtrackはTTLで破棄する。

---

## 7. 画像空間の尤度と prior

residualを

$$
r(X,Q)=h(X,Q)-y
$$

とし、画素precisionを

$$
W=\Sigma_{\mathrm{img}}^{-1}
$$

とする。現行実装は \(\Sigma_{\mathrm{img}}\) に有限・対称・正定値を要求し、
非正定値行列を姿勢推定段で黙って正則化しない。

並進には全IPPE枝で共通の等方Gaussian priorを置く。

$$
X\sim\mathcal{N}(\mu_0,\sigma_X^2 I_3).
$$

姿勢priorは一様である。定数を除いた負の対数posteriorは

$$
\Phi(X,Q)
=
\frac{1}{2}r(X,Q)^\top W r(X,Q)
+
\frac{1}{2\sigma_X^2}
\|X-\mu_0\|^2.
$$

既定値は

$$
\mu_0=0,\qquad \sigma_X^2=10^6\ \mathrm{m}^2
$$

であり、非常に広いがproperなpriorである。これは各IPPE seedの周りへ別々に
置くpriorではなく、カメラ座標系に固定された1つのglobal priorである。

`translation_prior_variance: null` とすると、explicitなproper並進priorを
無効にする。結果の局所並進lawがproperになるかは、画像尤度だけから得た
Hessianの正定値性に依存する。

---

## 8. 右摂動座標と投影 Jacobian

モード候補 \((X_\ell,R_\ell)\) の周りで

$$
X=X_\ell+\delta x,
$$

$$
R(u)=R_\ell\exp([u]_\times)
$$

という右摂動を使う。局所変数は

$$
\xi=
\begin{bmatrix}
\delta x\\u
\end{bmatrix}
\in\mathbb{R}^6.
$$

歪みなしpinholeについて、第 \(j\) 物体点のカメラ座標を

$$
P_j=R_\ell p_j+X_\ell
=
\begin{bmatrix}a_j&b_j&c_j\end{bmatrix}^{\!\top}
$$

とすると、投影の3D点Jacobianは

$$
G_j=
\begin{bmatrix}
f_x/c_j & 0 & -f_xa_j/c_j^2\\
0 & f_y/c_j & -f_yb_j/c_j^2
\end{bmatrix}.
$$

右摂動では

$$
\left.
\frac{\partial (R_\ell\exp([u]_\times)p_j)}
{\partial u}
\right|_{u=0}
=
-R_\ell[p_j]_\times
$$

なので、画素Jacobianは

$$
J_j=
\begin{bmatrix}
G_j & -G_jR_\ell[p_j]_\times
\end{bmatrix}.
$$

4点分を縦に積み

$$
J=
\begin{bmatrix}
J_1^\top&J_2^\top&J_3^\top&J_4^\top
\end{bmatrix}^{\!\top}
\in\mathbb{R}^{8\times6}
$$

を得る。

歪み係数が非ゼロなら、歪み込みの `projectPoints` を中央差分し、同じ
\([\delta x,u]\) chartのJacobianを求める。既定の差分幅は \(10^{-6}\) である。
歪みなしの場合は解析Jacobianを使い、診断設定では中央差分との一致も検査できる。

---

## 9. IPPEによる多峰性の初期化

単一平面正方形では、ほぼ同じ射影を生む複数の3D姿勢が存在しうる。特に
正面に近い観測では、姿勢posteriorを最初から単峰Gaussianへ潰すべきではない。

現行系は

```text
solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)
```

が返す候補をすべてseedとして使う。通常は2候補だが、後段は固定で2とは仮定
しない。

OpenCVが返すreprojection errorは初期診断にだけ使う。最終重みは、全候補へ
同じ \(\Sigma_{\mathrm{img}}\)、同じprior、同じ目的関数を適用した結果から
計算する。

### 9.1 cheirality

候補は全4隅について

$$
\left(Rp_j+X\right)_z > z_{\min}
$$

を満たす必要がある。既定の \(z_{\min}\) は \(10^{-6}\) m である。
初期候補と最適化後の両方で検査する。

---

## 10. 枝を保った Gauss–Newton refinement

各seedごとに、Gauss–Newton近似

$$
H
=
J^\top WJ
+
\begin{bmatrix}
\sigma_X^{-2}I_3&0\\
0&0
\end{bmatrix},
$$

$$
g
=
J^\top Wr
+
\begin{bmatrix}
\sigma_X^{-2}(X-\mu_0)\\
0
\end{bmatrix}
$$

を作り、

$$
H\Delta\xi=-g
$$

を解く。

候補更新は

$$
X' = X+s\Delta x,\qquad
R' = R\exp([s\Delta u]_\times)
$$

で行い、\(s=1,1/2,1/4,\ldots\) のbacktrackingを最大24回試す。
目的関数が有限かつ減少し、cheiralityを満たす更新だけを受理する。

### 10.1 seed Voronoi guard

通常の局所最適化だけでは、2つのIPPE seedが同じ良い解へ収束し、平面の
もう一方の離散仮説が消えることがある。これを避けるため、seed \(i\) からの
枝は

$$
d_i(X,R)
=
\frac{\|X-X_i^{(0)}\|}{L}
+
d_R(R,R_i^{(0)})
$$

という無次元距離で、そのseedのVoronoi cell内に留める。

回転距離は

$$
d_R(R_1,R_2)
=
\cos^{-1}
\left[
\frac{\operatorname{tr}(R_1^\top R_2)-1}{2}
\right].
$$

更新が別seed側へ越える場合はstepを半分にする。それでも越える場合、最後の
有効な枝内姿勢を `voronoi_guard` として残す。

複数seedの第2仮説は、4点reprojection目的の無制約な停留点にならないことも
ある。その場合も、減少する枝内stepがなければ
`seed_branch_fallback` として最後の有効姿勢を明示的に保持する。

これは「必ず2成分を捏造する」という意味ではない。cheirality違反、
非有限値、特異Hessianなど、確率分布として成立しない枝は除外される。

### 10.2 重複除去

有効モードを目的関数の小さい順に並べ、並進距離と回転距離の両方が設定閾値
以下なら重複とみなす。既定閾値は非常に小さく、

$$
\epsilon_X=10^{-7}\ \mathrm{m},\qquad
\epsilon_R=10^{-7}\ \mathrm{rad}
$$

である。

---

## 11. 局所 Hessian から同時分布を作る

第 \(\ell\) モードでのGauss–Newton Hessianを

$$
H_\ell=
\begin{bmatrix}
H_{xx}&H_{xu}\\
H_{ux}&H_{uu}
\end{bmatrix}
$$

と分割する。

局所Laplace近似は

$$
p(\delta x,u\mid y,\ell)
\propto
\exp\!\left[
-\frac12
\begin{bmatrix}\delta x\\u\end{bmatrix}^{\!\top}
H_\ell
\begin{bmatrix}\delta x\\u\end{bmatrix}
\right]
$$

である。

### 11.1 条件付き並進

次を定義する。

$$
S=H_{xx}^{-1},
$$

$$
B=-H_{xx}^{-1}H_{xu},
$$

$$
\Lambda
=
H_{uu}-H_{ux}H_{xx}^{-1}H_{xu}.
$$

平方完成すると

$$
\frac12
\begin{bmatrix}\delta x\\u\end{bmatrix}^{\!\top}
H
\begin{bmatrix}\delta x\\u\end{bmatrix}
=
\frac12(\delta x-Bu)^\top H_{xx}(\delta x-Bu)
+
\frac12u^\top\Lambda u.
$$

したがって、

$$
u\sim\mathcal{N}(0,\Lambda^{-1}),
$$

$$
X\mid u
\sim
\mathcal{N}(X_\ell+Bu,S).
$$

\(B\) が、姿勢が少し変わったときに最も尤もらしい並進がどう動くかを表す。
ここで \(S\) は姿勢を固定したときの**条件付き**並進共分散である。局所
Gaussianの並進marginalは

$$
\operatorname{Cov}(\delta x)
=
S+B\Lambda^{-1}B^\top
$$

であり、一般に \(S\) より広い。

現行系は \(H_{xx}\) と \(\Lambda\) の正定値性を明示的に検査する。成立しない
場合、非公開のjitterを足して通さず、その枝をrejectする。

### 11.2 \(3\times9\) rotation coupling

ProbTFは局所回転ベクトル \(u\) をwireへ直接保存せず、回転行列の
column-major vectorization

$$
\operatorname{vec}(R)
=
\begin{bmatrix}
R_{11}&R_{21}&R_{31}&R_{12}&\cdots&R_{33}
\end{bmatrix}^{\!\top}
\in\mathbb{R}^9
$$

を使う。

基準回転 \(R_\ell\) で

$$
D
=
\left.
\frac{\partial\operatorname{vec}
\left(R_\ell\exp([u]_\times)\right)}
{\partial u}
\right|_{u=0}
\in\mathbb{R}^{9\times3}
$$

とすると、

$$
D^\top D=2I_3,\qquad
D^+=\frac12D^\top.
$$

最小ノルムのglobal couplingを

$$
C=BD^+=\frac12BD^\top
\in\mathbb{R}^{3\times9}
$$

とする。ProbTF成分の条件付き並進は

$$
X\mid Q=q
\sim
\mathcal{N}
\left(
m_\ell(q),
S_\ell
\right),
$$

$$
m_\ell(q)
=
X_\ell
+
C_\ell
\left[
\operatorname{vec}(R(q))
-
\operatorname{vec}(R_\ell)
\right].
$$

基準姿勢近傍では

$$
C\left[
\operatorname{vec}(R(u))
-
\operatorname{vec}(R_\ell)
\right]
\approx Bu
$$

となり、Hessianの回転–並進相関を保存する。

---

## 12. 回転精度を Bingham 分布へ移す

単位四元数は \(q\) と \(-q\) が同じ回転を表すため、通常の
\(\mathbb{R}^4\) Gaussianより Bingham 分布が適している。

$$
p_\ell(q)
=
\frac{1}{Z(A_\ell)}
\exp(q^\top A_\ell q),
\qquad q\in S^3.
$$

これは

$$
p_\ell(q)=p_\ell(-q)
$$

を自動的に満たす。

基準四元数を \(q_\ell\) とし、その左積行列の虚部3列を

$$
E_\ell=L(q_\ell)[:,1:4]\in\mathbb{R}^{4\times3}
$$

とする。右回転摂動に対し

$$
q(u)\approx q_\ell+\frac12E_\ell u.
$$

Schur precision \(\Lambda_\ell\) から

$$
A_\ell^{(0)}
=
-2E_\ell\Lambda_\ell E_\ell^\top
$$

を作ると、

$$
q(u)^\top A_\ell^{(0)}q(u)
-
q_\ell^\top A_\ell^{(0)}q_\ell
\approx
-\frac12u^\top\Lambda_\ell u.
$$

したがって、Bingham exponentは局所Gaussianの回転曲率と二次まで一致する。

単位球上では \(A+cI\) が同じ分布を表すため、保存前に

$$
A_\ell
\leftarrow
A_\ell^{(0)}
-
\frac{\operatorname{tr}(A_\ell^{(0)})}{4}I_4
$$

としてtrace-zero gaugeへ移す。wire上ではさらに、JMAA正規化したshapeと
inverse concentrationへ分けて保存する。trace-zero \(A\) の固有値を昇順に
\(\lambda_1,\ldots,\lambda_4\) とすると、

$$
\kappa=\lambda_4+\lambda_3,\qquad
\mathrm{shape}=\frac{A}{\kappa},\qquad
\mathrm{inverse\ concentration}=\frac{1}{\kappa}.
$$

これは情報を失う分解ではなく、有限Bingham parameter \(A_\ell\) を一意な
規約で表したものである。

ただし、元のposteriorをBinghamへ置き換える操作自体は、モード近傍の
tangent surrogateでありlossyである。

---

## 13. Laplace質量と mixture weight

各枝の重みは、IPPEが返す順位や単純なreprojection RMSだけでは決めない。
局所posteriorの高さと体積を両方使う。

6次元Laplace近似では

$$
M_\ell
\propto
\exp(-\Phi_\ell)
\det(H_\ell)^{-1/2}.
$$

block determinantより

$$
\det(H_\ell)
=
\det(H_{xx,\ell})\det(\Lambda_\ell)
$$

なので、枝ごとのlog massは共通定数を除いて

$$
\log M_\ell
=
-\Phi_\ell
-
\frac12\log\det H_{xx,\ell}
-
\frac12\log\det\Lambda_\ell.
$$

目的関数が低いだけでなく、局所的に広い枝はより大きな積分質量を持つ。
逆に、非常に鋭いが体積の小さい枝は、ピーク高さだけで決めた場合と異なる重み
になる。

数値overflow/underflowを避けるため、最大log massを引いて

$$
w_\ell
=
\frac{
\exp(\log M_\ell-\max_k\log M_k)
}{
\sum_j\exp(\log M_j-\max_k\log M_k)
}
$$

とする。

この重みは局所Gaussian/Laplace体積に基づく近似であり、各枝領域を厳密に
数値積分したposterior massではない。具体的には、seed Voronoi境界で局所
Gaussian積分を切り詰める補正や、Bingham surrogateへ置換した後の再正規化は
重みへ戻していない。また、Gauss–Newton Hessianはresidualの二階微分を省略
しており、branch fallback点は厳密な停留点でない場合がある。

---

## 14. 最終的な1成分と混合分布

各枝 \(\ell\) は次の同時分布になる。

$$
p_\ell(X,Q)
=
\operatorname{Bingham}(Q;A_\ell)
\,
\mathcal{N}
\left(
X;
X_\ell+
C_\ell[
\operatorname{vec}(R(Q))-\operatorname{vec}(R_\ell)
],
S_\ell
\right).
$$

最終分布は

$$
p(X,Q\mid y)
\approx
\sum_{\ell=1}^{K}w_\ell p_\ell(X,Q).
$$

この形が表せるものは次の3種類である。

1. **離散的な平面姿勢ambiguity**: 複数の \(\ell\)
2. **各枝内の姿勢不確かさ**: Bingham \(A_\ell\)
3. **枝内の回転–並進相関**: \(C_\ell\)

単一の \(6\times6\) Gaussianへまとめると、1と四元数の幾何を失う。姿勢と
並進を独立にすると、3を失う。

---

## 15. ProbTF v2 への格納

1タグにつき1個の
`probtf_msgs/ProbabilisticTransformStamped` をpublishする。

### 15.1 record

| フィールド | 内容 |
|---|---|
| `header.frame_id` | camera optical frame \(C\) |
| `child_frame_id` | `apriltag_<id>` \(M\) |
| `edge_id` | 通常 `camera__to__apriltag_<id>` |
| `authority` | detector node名 |
| `is_static` | `false` |
| `components` | 保持されたIPPE枝 |
| `provenance` | tag family/ID、camera model source、推定法 |
| `approximation` | tangent surrogateであることと制限 |

この推定器はrecordへ決定論的representativeを保存しないため、
`representative_kind` は `NONE` である。必要な表示・bridge側が明示的な
policyで代表値を導出する。

### 15.2 component

各 `ProbabilisticTransformComponent` は

- `weight`: \(w_\ell\)
- `orientation`: Bingham shape、inverse concentration、基準四元数
- `translation.mean_at_reference`: \(X_\ell\)
- `translation.residual_covariance_upper`: \(S_\ell\)
- `translation.rotation_coupling`: \(C_\ell\)
- IPPE seed indexを含むprovenance
- `converged`、`voronoi_guard`、`seed_branch_fallback` の近似説明

を持つ。

対称行列は上三角だけ、\(C_\ell\) は \(3\times9\) row-majorでwireへ格納する。
一方、\(C\) が作用する \(\operatorname{vec}(R)\) はcolumn-majorである。

元の \(8\times8\) 画素共分散、bootstrap sample、temporal track、\(6\times6\)
Hessian、log mass、候補別diagnosticsはwireへ直接保存されない。これらは最終
分布へ畳み込まれ、推定法と近似性だけがprovenance/approximationに残る。

### 15.3 provenance と fallback

source IDには少なくともtag familyとIDが入り、利用したcamera modelの由来も
追加される。未校正fallbackの場合は、

- intrinsic/distortion uncertaintyを分布に含めていない
- metric depthと共分散が近似である

ことをrecordと各componentのapproximation detailへ明記する。

---

## 16. representative TF は確率分布そのものではない

ProbTF recordは全成分を保持する。一方、通常の `/tf` へ出すには確率分布を
1本の決定論的変換へ落とす必要がある。

現行デモのbridgeは、正の重みが最大のcomponentを選び、

1. そのBingham mode \(q_{\ell^\star}\)
2. そのmodeに条件づけた並進平均
   \(m_{\ell^\star}(q_{\ell^\star})\)

を通常TFとして出す。

これは

- mixture全体の平均
- mixture全体の厳密なglobal MAP
- もう一方のIPPE枝

ではない。`/tf` は互換表示用のlossy projectionであり、確率的な後段処理は
ProbTF topicを購読する必要がある。

2枝の重みが交差すると、通常TFの代表値は枝間で不連続に切り替わりうる。
これはmixture自体の不連続ではなく、1本だけを選ぶprojectionの性質である。

---

## 17. RViz点群の数学的な意味

ProbTF native displayは、各sampleについて原点だけを描くのではない。
まず

$$
\ell\sim\operatorname{Categorical}(w_1,\ldots,w_K),
$$

$$
Q\sim\operatorname{Bingham}(A_\ell),
$$

$$
\epsilon\sim\mathcal{N}(0,S_\ell),
$$

$$
X
=
X_\ell
+
C_\ell[
\operatorname{vec}(R(Q))-\operatorname{vec}(R_\ell)
]
+
\epsilon
$$

を同時にsampleする。その同じ \((X,R)\) から、長さ \(L_a\) の正の3軸端点

$$
p_x=X+R(L_a e_x),
$$

$$
p_y=X+R(L_a e_y),
$$

$$
p_z=X+R(L_a e_z)
$$

を赤・緑・青で描く。

したがって点群には、

- 並進分散
- 回転分散
- 回転–並進相関
- mixtureの枝分かれ

が同時に現れる。

`Axis Length` \(L_a\) を大きくすると、姿勢揺らぎのlever armも長くなるため、
点群の球・弧・リングが大きく見える。これは確率分布や共分散自体を拡大した
わけではない。中央のProbTF representative軸も同じ \(L_a\) を使う。

別の `rviz/TF` displayやMarkerArrayが描く軸は独立であり、この値には追従
しない。詳しい操作は
[probtf_rviz README](../../../core/probtf_rviz/README.md)を参照する。

AprilTagデモのMarkerArrayが表示する2-sigma楕円体は、各orientation modeに
条件づけた \(S_\ell\) の可視化である。上式の並進marginal
\(S_\ell+B_\ell\Lambda_\ell^{-1}B_\ell^\top\) や、mixture全体のmarginalでは
ない。

---

## 18. 1フレームを再実装するための擬似コード

```text
input:
    image
    camera model K, distortion
    tag side length L

observations = detect_apriltag(image)

for each observation (id, y):
    Σ_floor = corner_sigma_px^2 * I8

    if adaptive spatial covariance:
        samples = []
        for deterministic small image perturbation:
            y_b = redetect_same_id_near_original()
            if success:
                remove_known_image_shift(y_b)
                samples.append(y_b)
        M = mean((y_b - y)(y_b - y)^T)
        M = shrink_toward_diagonal(M)
        inflate_for_low_redetection_success(M)
        Σ_spatial = Σ_floor + M + dropout_penalty
    else:
        Σ_spatial = Σ_floor

    if adaptive temporal covariance:
        ν = normalized_constant_velocity_innovation(last_two_y, y)
        R_ν = propagated_spatial_covariance(last_two_Σ, Σ_spatial)
        reject_hard_outlier_or_reset_bad_timing()
        optionally_freeze_affine_motion()
        robustly_update_running_covariance(ν, R_ν)
        E = PSD_part(shrink(cov(ν) - mean(R_ν)))
        cap_only_E()
        Σ_img = Σ_spatial + E
    else:
        Σ_img = Σ_spatial

    W = inverse(Σ_img)
    seeds = all_IPPE_SQUARE_candidates(y, K, distortion, L)

    modes = []
    for each seed i:
        reject_if_any_corner_is_behind_camera()
        refine Φ with right-perturbation Gauss-Newton
        keep_update_inside_seed_i_Voronoi_cell()
        compute H = J^T W J + explicit_translation_prior

        S = inverse(H_xx)
        B = -inverse(H_xx) H_xu
        Λ = H_uu - H_ux inverse(H_xx) H_xu
        reject_if_H_xx_or_Λ_is_not_SPD()

        D = d vec(R exp([u]x)) / du at u=0
        C = 0.5 B D^T
        A = trace_zero(-2 E Λ E^T)
        log_mass = -Φ - 0.5 logdet(H_xx) - 0.5 logdet(Λ)
        modes.append(X, R, S, C, A, log_mass)

    deduplicate_nearly_identical_modes()
    reject_observation_if_no_mode_remains()
    weights = softmax(log_masses)
    publish_one_ProbTF_component_per_mode()
```

---

## 19. 主要設定と確率的な効果

現行の既定値は
[`config/default.yaml`](../config/default.yaml)にある。

### 19.1 検出・同一フレーム共分散

| パラメータ | 既定値 | 大きくしたとき |
|---|---:|---|
| `corner_sigma_px` | 0.5 px | 全方向の最低分散が増える |
| `bootstrap_samples` | 8 | 共分散が安定するが処理時間が増える |
| `bootstrap_noise_std` | 6.0 intensity | intensity noiseへの感度を強く測る |
| `bootstrap_dither_px` | 0.35 px | alias/subpixel位相への感度を強く測る |
| `bootstrap_blur_sigma_px` | 0.35 px | blurへの感度を強く測る |
| `bootstrap_covariance_shrinkage` | 0.25 | corner間相関を抑え、対角寄りになる |
| `bootstrap_min_success_ratio` | 0.5 | 低成功率へのpenalty開始が厳しくなる |
| `bootstrap_dropout_sigma_px` | 2.0 px | 再検出消失時の等方penaltyが増える |

### 19.2 時系列超過共分散

| パラメータ | 既定値 | 意味 |
|---|---:|---|
| `temporal_warmup_samples` | 8 | excessを有効にするinnovation数 |
| `temporal_half_life_sec` | 0.5 s | 過去ジッタを忘れる半減期 |
| `temporal_shrinkage` | 0.25 | excess相関の対角shrinkage |
| `temporal_huber_chi2` | 20.09 | warmup後のrobust clipping閾値 |
| `temporal_max_excess_sigma_px` | 5.0 px | temporal固有値だけの上限 |
| `temporal_freeze_affine_motion` | false | affine residualを実運動として学習停止するか |
| `temporal_max_gap_sec` | 0.5 s | これより長い欠測でtrack reset |
| `temporal_max_dt_ratio` | 2.5 | 不規則すぎる時刻間隔でreset |
| `temporal_track_ttl_sec` | 2.0 s | 見えない別タグのtrack保持時間 |

### 19.3 姿勢推定

| パラメータ | 既定値 | 意味 |
|---|---:|---|
| `tag_size_m` | 0.12 m | 実タグの黒枠正方形の一辺 |
| `max_iterations` | 30 | 枝ごとのGauss–Newton上限 |
| `convergence_tolerance` | \(10^{-9}\) | 6D step normの停止閾値 |
| `min_depth` | \(10^{-6}\) m | cheiralityの最小corner depth |
| `finite_difference_step` | \(10^{-6}\) | 歪みありcameraのJacobian差分幅 |
| `spd_tolerance` | \(10^{-12}\) | local precision受理判定 |
| `translation_prior_mean` | \([0,0,0]\) m | 全枝共通のglobal prior mean |
| `translation_prior_variance` | \(10^6\) m² | 全枝共通の広いproper prior |

`convergence_tolerance` はmとradを同じ6D step normへ入れる実装上の閾値で
ある。また `finite_difference_step` は並進列ではm、回転列ではradへ同じ
数値を使う。

---

## 20. 診断状態の読み方

debug imageの

$$
\sigma_{\mathrm{eq}}
=
\sqrt{\frac{\operatorname{tr}(\Sigma_{\mathrm{img}})}{8}}
$$

は、full covarianceを1つのpx標準偏差へ要約した表示値である。相関や方向差は
この数値からは分からず、推定器内部とProbTF構成にはfull matrixが使われる。

temporal statusは次を表す。

| status | 意味 |
|---|---|
| `warmup` | 履歴または有効innovationが不足 |
| `accepted` | innovationを統計へ反映 |
| `motion_frozen` | optional affine motion判定で更新停止 |
| `outlier` | hard outlierとして最大excessを返しtrack reset |
| `gap_reset` | 時刻gapまたはdt比によりtrack reset |

姿勢候補側では、

- seed数
- accepted mode数
- duplicate mode数
- cheirality reject数
- refinement reject数
- non-SPD local law reject数
- seedごとの初期error、最終objective、iteration、reason

を区別する。画像からタグが見つかったことと、確率分布として有効な姿勢枝が
得られたことは別である。

---

## 21. この分布が含む不確かさ、含まない不確かさ

### 21.1 含むもの

- detectorの順序付き4隅に対する固定px floor
- 小さなshift、noise、blurに対する再検出感度
- 同一フレームで観測されたcorner間相関
- 定速度モデルで説明できないframe-to-frame超過ジッタ
- 画素共分散から投影Jacobianを通した局所姿勢不確かさ
- 平面PnPの複数枝
- 各枝の回転–並進相関
- local posterior高さと体積による枝重み

### 21.2 現在は含まないもの

- \(K,D\) 自体の校正共分散
- タグ実寸 `tag_size_m` の誤差
- タグの非平面性、印刷伸縮
- rolling shutterの明示モデル
- exposure中の3D運動モデル
- 摂動族を超える強いmotion blur、occlusion、照明変化
- 誤IDやcorner order誤りを別hypothesisとして持つモデル
- 複数フレームの3D状態・速度を推定するBayesian tracker
- 同じfamily/IDを持つ複数の物理タグを区別するtemporal track
- Laplace近似領域外の各枝の非Gaussianな形
- IPPEがseedとして返さなかった未知のposterior mode

特にfallback cameraでは内部パラメータ誤差が含まれないため、点群が狭く見えても
metricに正しいとは限らない。

---

## 22. 実運用での調整順

確率分布が実際のガタつきより狭い場合、無条件に
`corner_sigma_px` だけを上げる前に、次の順で原因を分けるとよい。

1. 実カメラを校正し、解像度とCameraInfoが一致することを確認する。
2. `tag_size_m` が黒枠正方形の実測値か確認する。
3. debug imageでcornerがどの方向へ跳ねるか確認する。
4. センサの強度noiseに合わせて `bootstrap_noise_std` を調整する。
5. subpixel位相・aliasへの感度が足りなければ `bootstrap_dither_px` を調整する。
6. blur時だけ過信するなら `bootstrap_blur_sigma_px` を調整する。
7. 再検出が消えるのに狭いなら `bootstrap_dropout_sigma_px` を調整する。
8. フレーム間だけ跳ねるならtemporal stageとhalf-lifeを確認する。
9. 最後に、常時存在する未モデル化誤差として正当化できる分だけ
   `corner_sigma_px` を上げる。

実運動をtemporal jitterへ含めたくない場合だけ
`temporal_freeze_affine_motion` を有効にする。ただし、共有方向へ動くdetector
jitterも同時に除外されうる。

---

## 23. 数式とwire fieldの対応表

| 数式 | 次元 | 単位 | ProbTF/componentでの表現 |
|---|---:|---|---|
| \(w_\ell\) | 1 | 無次元 | `weight` |
| \(q_\ell\) | 4 | 無次元 | `orientation.reference_quaternion` |
| \(A_\ell\) | \(4\times4\) | 無次元（局所回転ではrad\(^{-2}\)相当） | normalized shape + inverse concentration |
| \(X_\ell\) | 3 | m | `translation.mean_at_reference` |
| \(S_\ell\) | \(3\times3\) | m² | `residual_covariance_upper` |
| \(C_\ell\) | \(3\times9\) | m | `rotation_coupling` |
| \(\Sigma_{\mathrm{img}}\) | \(8\times8\) | px² | detector内部でHessianへ使用 |
| \(H_\ell\) | \(6\times6\) | 混合単位 | wireには直接保存しない |
| \(\Lambda_\ell\) | \(3\times3\) | rad\(^{-2}\) | Bingham \(A_\ell\) へ変換 |

---

## 24. まとめ

この推定器の中心は、次の3段階を切らずにつなぐことである。

1. detectorの画素レベルの不安定さをfull \(8\times8\) 共分散にする
2. その共分散を共通の画像尤度と局所Hessianへ入れる
3. 平面の離散枝と枝内相関を残したProbTF mixtureへ変換する

最終的なモデルは

$$
\boxed{
p(X,Q\mid y)
\approx
\sum_\ell
w_\ell\,
\operatorname{Bingham}(Q;A_\ell)\,
\mathcal{N}
\left(
X;
X_\ell+C_\ell[\operatorname{vec}R(Q)-\operatorname{vec}R_\ell],
S_\ell
\right)
}
$$

である。

これにより、「最も良いPnP解1個とその周りの小さなGaussian」ではなく、
平面タグ固有の複数解、エッジ検出のガタつき、姿勢–奥行き相関を同時に
下流へ渡せる。
