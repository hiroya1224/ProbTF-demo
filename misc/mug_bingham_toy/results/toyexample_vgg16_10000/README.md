# 学習済みVGG16によるtoyマグの5方向図

[PDF](toyexample.pdf)・[PNG](toyexample.png)。参照した`~/dthesis/figs/symaware-bingham/toyexample.pdf`と同じ7.35×4.06 inch、同じ5方向・行配置・配色で描いた。参照図自体は変更していない。

元メッシュはユーザー指定の`/home/leus/Downloads/mug.STL`。保存した元STLとbytesが一致し、その軸を維持して中心移動・一様スケールを行った`../../assets/mug_centered_colored.ply`を読む。色は単色`#BC1543`。画像・照明・カメラ距離・入力前処理は学習時と同じである。旧PoseCNNのマグメッシュ、画像、推定値は使っていない。

VGG16全体を10,000枚で50 epoch学習した保存重みのうち、検証NLLで選んだepoch 48を使った。この5画像の結果で重みを選び直していない。ネットワークには画像全体640×360を入力し、図の共通切り出し`x=[157,483), y=[74,330)`は表示だけに使う。

## 座標と分布

Hamilton四元数の順序はwxyz。ネットワークの出力は物体からOpenCVカメラへの方位分布である。9成分は

\[
\theta=(A_{00},A_{01},A_{02},A_{03},A_{11},A_{12},A_{13},A_{22},A_{23}),
\qquad A_{33}=-A_{00}-A_{11}-A_{22}.
\]

下三角を対称に補い、trace zeroの行列を作る。表の出力はこの生のカメラ座標系の\(\theta_1,\theta_2,\theta_9\)を小数1桁へ丸めたもの。旧図の10成分出力に合わせて架空の\(\theta_{10}\)を作ってはいない。

図示ではマグの元のZ軸を上向きとする世界座標へ既知の固定回転で移す。カメラ仰角は10度、表示角\(\alpha\)に対して物体の世界yawを\(180^\circ-\alpha\)とする。カメラの物体方位\(q_C\)と世界方位\(q_W\)の関係は

\[
q_W=L(q_{WC})q_C,
\qquad A_W=L(q_{WC})A_CL(q_{WC})^\top.
\]

したがって\(q_W^\top A_Wq_W=q_C^\top A_Cq_C\)であり、同じ密度を別の座標系で表している。集中度の調節や分布の再フィッティングは行わない。X／Y／Z軸を赤／緑／青で描き、線はモードの軸である。

点はBingham密度がピークの半分になる四元数の等密度面

\[
q^\top A_Wq=\lambda_{\max}(A_W)-\log 2
\]

を決定論的な960点で表し、各四元数の3軸先端を固定の斜投影へ写したもの。参照図と同じ構成であり、確率サンプリングでも、S²へ押し出した軸の周辺密度の等密度線でもない。点数や見栄えのために分布を広げていない。

半値角はモードから**固定された世界Z軸の周り**へ回したとき、元のBingham密度が半分になる片側の物理角度である。モードを\(q_*\)、\(v=(0,0,0,1)\odot q_*\)とすると、

\[
\Delta=q_*^\top A_Wq_*-v^\top A_Wv,
\qquad \theta_{1/2}=2\arcsin\sqrt{\frac{\log2}{\Delta}}.
\]

これは標準偏差や確率質量95%の区間ではない。

## 今回の値

| 表示角 [deg] | 半値角 [deg] | モードと教師の方位角誤差 [deg] |
|---:|---:|---:|
| 0 | 3.0282 | 1.0304 |
| 60 | 4.0086 | 1.2078 |
| 90 | 4.2891 | 0.7696 |
| 120 | 4.0692 | 0.5947 |
| 180 | 2.8879 | 0.9770 |

取っ手が隠れる3例でZ軸周りの半値角が大きくなった。ただし単一メッシュ・固定距離・固定光源で学んだtoy結果であり、実画像での不確かさの校正を示す実験ではない。原点がSTLのbounds中心であることも含め、描画条件を変えれば他の手掛かりも変わり得る。

## 保存物と再現

- `figure_data.npz`：表示に使った5枚の全RGB、教師四元数、9成分生出力、カメラ・世界の同じBingham行列、モード、座標変換。
- `geometry.npz`：実際に描いた半高等密度面の四元数・投影された軸先端・モード軸。
- `render.json`、`inference.json`、`figure.json`：メッシュ、重み、ソース、入力、数値のhashと設定。巨大な学習重みと画像生成の中間ファイルはGitに含めない。
- `../../verification/training_completed.json`：全50 epochの使用枚数と独立テスト結果。

リポジトリrootから、保存数値だけで再描画できる。ネットワークの再実行や参照先のdthesisリポジトリは不要。

```bash
/usr/bin/python3 misc/mug_bingham_toy/toyexample.py plot
```

画像生成・推論から再実行する場合は、ローカルに保存した重みを使い、次の順序で実行する。表示のみの変更では上の`plot`だけを実行する。

```bash
/usr/bin/python3 misc/mug_bingham_toy/toyexample.py render
/home/leus/posecnn_ws/.venv/bin/python misc/mug_bingham_toy/toyexample.py infer
/usr/bin/python3 misc/mug_bingham_toy/toyexample.py plot
```

幾何の検査は次で行う。参照図と集中した場合の点の完全一致、独立した数値求根による半値角、座標変換での密度不変性、幅広い分布や一様分布での表示も検査する。

```bash
/usr/bin/python3 -m unittest discover -s misc/mug_bingham_toy/tests -p test_figure_geometry.py -v
```
