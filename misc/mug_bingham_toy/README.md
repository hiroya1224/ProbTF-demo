# 単色マグの画像からBingham分布を学ぶtoy example

`mug.STL`を一色の`#BC1543`にして白い空間で回転させ、画像と四元数の組からVGG16全体を学習する。方位の出力はtrace zeroの対称4×4行列を表す9成分で、損失はBinghamの負の対数尤度である。

## 起動中のセッション

```bash
tmux a -t mug-bingham-toy
```

`train`ウィンドウは画像生成の完了を待って学習を開始する。`status`ウィンドウには生成枚数と学習の状況を表示する。`Ctrl-b`を押してから`d`で離れられ、学習は継続する。

## メッシュと画像

- 元のSTL：[assets/mug_original.stl](assets/mug_original.stl)。指定された`/home/leus/Downloads/mug.STL`と同一のbytesを保存した。
- 単色の入力メッシュ：[assets/mug_centered_colored.ply](assets/mug_centered_colored.ply)。すべての頂点がRGBA `[188,21,67,255]` で、レンダラはこのファイルを読む。
- 元の軸を保ち、boundsの中心を原点へ移して最長辺を1にした。STLの物理単位は仮定しない。変換とhashは[mesh_metadata.json](assets/mesh_metadata.json)に記録する。
- 画像はRGB、640×360。カメラ距離は正規化後の座標で2.5、白背景、固定のambientとカメラ後方上側の光源群を使う。[レンダリングの詳細](RENDERING.md)。
- 学習10,000枚、検証1,000枚、独立テスト1,000枚。PCG64のseedを1234／1235／1236に固定し、それぞれ独立に生成する。
- 4次元標準Gaussianを単位長へ正規化して一様なS³四元数を作る。各四元数は物体からOpenCVカメラへの能動回転を表し、順序はwxyz。画像の向きと同じ座標規約の四元数を教師にする。

生成物は`data/{train,val,test}/images/000000.png`等と、それに対応する`poses.npz`・`metadata.json`へ保存する。`q`と`-q`は同じ画像・同じ密度を表す。

[入力12例](examples/contact_sheet.png)は学習集合の先頭12件と同じ四元数で描画した。対応する四元数とカメラ設定も`examples/`へ保存している。

## ネットワークと学習

標準VGG16の13畳み込み層、7×7のadaptive average pooling、4096→4096→9の全結合部を使う。実装はPyTorchだけで自己完結する。初期値は乱数で、畳み込みを含む134,297,417個のパラメータすべてを学習する。画像全体640×360を入力し、学習時の切り出しや拡大縮小は行わない。

設定は50 epoch、batch 16、AdamW、初期学習率1e-4、weight decay 1e-4、cosine learning-rate schedule。GPUの画像ネットワークにはbfloat16 AMPを使い、出力をfloat32、NLLをfloat64で計算する。各epochは学習10,000枚をすべて一度ずつ使う。検証NLLが最良の重みを別保存し、全epoch後にその重みで独立テストを評価する。途中の検証値による早期終了は行わない。

出力順は

\[
\theta=(A_{00},A_{01},A_{02},A_{03},A_{11},A_{12},A_{13},A_{22},A_{23}),
\qquad A_{33}=-A_{00}-A_{11}-A_{22}.
\]

下三角には上三角をコピーする。分布と損失は

\[
p(q\mid I)=\frac{\exp(q^TA_\theta(I)q)}{Z(A_\theta(I))},
\qquad
\mathcal L(I,q)=\log Z(A_\theta(I))-q^TA_\theta(I)q.
\]

積分測度はS³の表面積で、一様分布の正規化定数は`2*pi**2`。正規化定数はChen–Tanaka式の決定論的数値積分で評価し、解析的な一次微分を逆伝播する。重複固有値を持つ一様分布付近でも、固有ベクトルの微分へ依存しない形で計算する。元の実装・hashは[bingham.py](bingham.py)に記録した。

有限の積分誤差を持つ実装であり、すべての集中度で厳密解であるとは扱わない。モードの角度誤差は補助指標で、広い分布や最大固有値の重複では代表姿勢が一意でない点に注意する。

## 再現コマンドと保存先

リポジトリrootから画像を生成する。

```bash
/usr/bin/python3 misc/mug_bingham_toy/render.py --output misc/mug_bingham_toy/data
```

画像生成が終わってから、次のコマンドで学習を実行できる。使用するPythonは`TOY_MUG_PYTHON`で指定でき、既定はPyTorchが入っている既存環境である。

```bash
bash misc/mug_bingham_toy/run.sh
```

学習の重み・optimizer・ログは`runs/vgg16_10000_20261126/`へ保存する。`status.py`で参照できる。

```bash
/usr/bin/python3 misc/mug_bingham_toy/status.py
```

保存済みepoch末尾から再開する場合は、同じコード・データ・設定で次を実行する。

```bash
bash misc/mug_bingham_toy/run.sh --resume /home/leus/catkin_ws/src/ProbTF-demo/misc/mug_bingham_toy/runs/vgg16_10000_20261126/last.pt
```

レンダラと学習器は既存の別Python環境を使うため、検査も分けて実行する。

```bash
/usr/bin/python3 -m unittest discover -s misc/mug_bingham_toy/tests -p test_render.py -v
/home/leus/posecnn_ws/.venv/bin/python -m unittest discover -s misc/mug_bingham_toy/tests -p test_bingham.py -v
/home/leus/posecnn_ws/.venv/bin/python -m unittest discover -s misc/mug_bingham_toy/tests -p test_model_training.py -v
```

生成画像、特徴cache、全VGG16の大きなcheckpointはGitの対象外とする。コード、元メッシュ・単色メッシュ、入力例、検証数値だけを通常commit/pushする。以前のマグ・ボトル・IMUの実験やProbTF coreは変更しない。

## 今回の実行確認

[生成記録](verification/dataset_generation.json)、[19件の検査](verification/unit_tests.json)、
[GPUのメモリ・速度確認](verification/gpu_benchmark_batch16.json)、
[学習開始時の記録](verification/training_launch.json)を保存した。開始記録は途中のスナップショットで、学習完了を意味しない。
最新の進行状況と最終テスト結果はローカルrunディレクトリの`metrics.jsonl`／`test_metrics.json`を参照する。
