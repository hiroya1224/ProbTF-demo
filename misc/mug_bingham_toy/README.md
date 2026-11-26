# 単色マグの画像からBingham分布を学ぶtoy example

指定`mug.STL`を単色`#BC1543`にし、白い空間で回転させた画像からVGG16全体を学習する。出力はtrace zeroの対称4×4 Bingham行列を表す9成分、損失はBingham NLL。

## 円筒中心への修正

回転中心は**マグの円筒の中心**とする。元STLでは円筒軸が`x=40, y=40`、高さの中点が`z=45`なので、`(40,40,45)`を原点へ移す。取っ手込みの外接箱中心は使わない。

旧実装は`(56.0566101074,40,45)`を回転中心にし、円筒が回転に伴って移動する誤りがあった。誤った学習画像・重み・図・数値・例示・検証記録を削除した。修正版では画像から全生成し、旧重みを使わず乱数初期値から再学習する。旧結果は本実験の結果として扱わない。

- 元STL：[assets/mug_original.stl](assets/mug_original.stl)。ユーザー指定ファイルと同一bytes。
- 修正した着色メッシュ：[assets/mug_cylinder_centered_colored.ply](assets/mug_cylinder_centered_colored.ply)。すべての頂点はRGBA `[188,21,67,255]`。
- 中心と変換の記録：[assets/mug_cylinder_center_metadata.json](assets/mug_cylinder_center_metadata.json)。元の軸を維持し、一様scaleは`1/112.11322021484375`のまま。
- カメラ距離2.5、画像全体640×360、白背景、ambientとカメラ後方上側の点光源。roughnessは0.4。ブラーは適用せず、レンダラのRGBをそのまま使う。[描画規約](RENDERING.md)。

最新の[レンダリング一覧](examples/cylinder_pointlight_preview/render_preview.png)は、次で元画像・設定と一緒に上書きする。

```bash
/usr/bin/python3 misc/mug_bingham_toy/toyexample.py preview
```

## 学習と起動

標準VGG16 configuration D、13畳み込み層、adaptive average pooling 7×7、4096→4096→9の全結合部を全体学習する。事前学習重みは使わない。全134,297,417パラメータを乱数初期化し、50 epoch、batch 16、AdamW、初期学習率1e-4、weight decay 1e-4、cosine scheduleで学習する。各epochは10,000枚をすべて一度ずつ使用する。検証NLLで重みを選び、全epoch完了後に独立テストを実行する。画像のcrop・resize・augmentationはしない。

データは`data/cylinder_center_no_blur_20261127/`、重みとログは`runs/vgg16_10000_cylinder_center_no_blur_20261127/`へ保存する。学習・検証・テストの枚数は10,000／1,000／1,000、四元数生成のPCG64 seedは1234／1235／1236。4次元標準Gaussianを単位長へ正規化してHaar一様回転を生成する。教師は物体からOpenCVカメラへのHamilton四元数wxyzで、画像の実際の回転と一致する。

ブラーありの学習はユーザー指定で停止し、その生成物を削除した。ブラーなしで全画像を再生成し、乱数初期値から新しく学習する。学習中は各epochの検証後、固定5画像に対する同じ予測Aから物体X/Z方向の接空間共分散を計算する。既存の`metrics.jsonl`に生Aと評価値を一緒に記録し、別の報告書は作らない。

長時間処理用の専用セッションに接続する。

```bash
tmux a -t mug-bingham-toy
```

`Ctrl-b`、続けて`d`で離れられる。現在の生成・学習状況は次で確認できる。

```bash
/usr/bin/python3 misc/mug_bingham_toy/status.py
```

専用セッションを新規に立てるコマンドは`bash misc/mug_bingham_toy/launch_tmux.sh`。画像生成、画像の完了を待つ新規学習、進行表示、ライブ画面の4ウィンドウを作る。全学習と独立テストが完了したら、同じ新しい重みで5方向図も自動生成する。ライブ画面は5入力画像と同じAからのBingham標本、X/Z方向の接空間の主標準偏差[deg]を表示する。下段の値は保存した主分散の平方根を180/π倍した1σで、学習ログの共分散は変更しない。

リポジトリrootからの再生成・新規学習は次の順序。`run.sh`はデータが揃い、円筒中心のメッシュで生成されたことを確認してから、空の新規runで開始する。

```bash
/usr/bin/python3 misc/mug_bingham_toy/render.py --output misc/mug_bingham_toy/data/cylinder_center_no_blur_20261127
bash misc/mug_bingham_toy/run.sh
```

## 分布と損失

出力は

\[
\theta=(A_{00},A_{01},A_{02},A_{03},A_{11},A_{12},A_{13},A_{22},A_{23}),
\qquad A_{33}=-A_{00}-A_{11}-A_{22}.
\]

下三角を対称に補い、

\[
p(q\mid I)=\frac{\exp(q^TA_\theta(I)q)}{Z(A_\theta(I))},
\qquad \mathcal L(I,q)=\log Z(A_\theta(I))-q^TA_\theta(I)q.
\]

S³の表面積測度を用い、一様分布の正規化定数は`2*pi**2`。Chen–Tanaka式の決定論的数値積分と解析的一次微分を使う。有限の求積誤差があり、厳密な正規化定数とは区別する。既存のProbTF coreは変更しない。画像ネットワークはGPU bfloat16、最終出力float32、NLL float64で計算する。

## 検査と図

円筒軸上の中点が原点へ移され、その投影が回転しても固定されることを検査する。円筒断面の外周・内周の円からも軸が`(40,40)`であることを確認する。

```bash
/usr/bin/python3 -m unittest discover -s misc/mug_bingham_toy/tests -p test_render.py -v
/usr/bin/python3 -m unittest discover -s misc/mug_bingham_toy/tests -p test_figure_geometry.py -v
/home/leus/posecnn_ws/.venv/bin/python -m unittest discover -s misc/mug_bingham_toy/tests -p test_bingham.py -v
/home/leus/posecnn_ws/.venv/bin/python -m unittest discover -s misc/mug_bingham_toy/tests -p test_model_training.py -v
```

修正版の学習完了後、`~/dthesis/figs/symaware-bingham/toyexample.pdf`形式で図を作れる。画像生成・推論・描画を分離し、描画は保存された同じBingham行列を参照する。

方位分布は、保存された同じ`A_camera`から四元数を1,000個抽出して表示する。既存のProbTF coreのC++サンプラを呼び、dthesis第3章のAlgorithm `math:alg:bingham-bacg`（Kent BACG棄却法）をそのまま使う。第4章の標本化と同じく、同一の1,000回転を物体X/Y/Z軸へ作用させるため、赤・緑・青の先端が各1,000点となる。接空間Gaussianからの疑似標本や、軸ごとに独立した回転ではない。

入力と同じカメラ座標で、Z軸を極とする緯度経度wireframeを維持する。点の不透明度は手前0.7、奥（カメラZ>0）0.35で、奥側のワイヤも手前の1/2とする。モード座標軸は全て不透明・線幅1.8ptで最前面へ描く。表示専用の固定seedをパネルごとに使い、同じAの再描画で点を振り直さず、学習の乱数状態へ触れない。`--samples`と`--sample-seed`はライブ表示と`toyexample.py plot`で変更できる。静的図の`geometry.npz`には実際に描いた四元数・三軸の先端・seedを、`figure.json`には標本数とアルゴリズムを保存する。標本は表示専用で、NLL・半値角・接空間共分散には使わない。半値角は従来どおりマグの上向き軸周りの物理回転角で、カメラ座標へ同じ軸を移して評価する。

```bash
/usr/bin/python3 misc/mug_bingham_toy/toyexample.py render
/home/leus/posecnn_ws/.venv/bin/python misc/mug_bingham_toy/toyexample.py infer
/usr/bin/python3 misc/mug_bingham_toy/toyexample.py plot
```

生成画像・巨大checkpoint・optimizer・中間cacheはGitへ入れない。元STL、修正メッシュ、コード、少数の入力例、検証記録を通常commit/pushする。
