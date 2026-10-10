# PIDゲインの三次元領域と断面図

failure 1・2をそれぞれの推定モデル分布で評価した、編集・再描画用の保存セットです。各ケースのRoll/Pitch、Yaw、XY、Zについて、三次元俯瞰図とP–I・P–D・I–D断面図を保存しています。

- [図の一覧をブラウザで開く](index.html)
- [半透明ボクセルのオフラインビューア](interactive/index.html)：33点・65点の格子を比較し、視点、透明度、I断面を変更できます。
- [failure 1の俯瞰図](figures/failure1/all_groups_3d_oblique.png) / [PDF](figures/failure1/all_groups_3d_oblique.pdf)
- [failure 2の俯瞰図](figures/failure2/all_groups_3d_oblique.png) / [PDF](figures/failure2/all_groups_3d_oblique.pdf)
- [failure 1のP–D断面一覧](figures/failure1/all_groups_PD_at_success_I.png)
- [failure 2のP–D断面一覧](figures/failure2/all_groups_PD_at_success_I.png)
- [成功・記録ゲインと、その点での生存率](reference_gains.csv)
- [方法と論文向けの説明](METHODS.md)
- [データの形式](DATA_FORMAT.md)
- [全ファイルのサイズとSHA-256](files.tsv) / [保存サイズの集計](inventory.json)

GitHub上のHTMLはダウンロード後に開いてください。HTMLには必要なJavaScriptと数値が入っているので、ネット接続やROS、元の計算ディレクトリは不要です。failure 1・2はケース選択で切り替え、それぞれの分布を混ぜずに表示します。

## 図の構成

80組の図をPDF・SVG・400 dpi PNGで保存しています。主な図は65点の格子を使っています。記録ゲインを厳密な内点として加えているため、実際の軸の要素数は65より多い場合があります。

| 種類 | 組数 | 内容 |
| --- | ---: | --- |
| 各PIDグループの三次元図 | 16 | 2ケース × 4グループ × 斜め・真上 |
| 全グループの三次元一覧 | 4 | 2ケース × 斜め・真上 |
| 記録ゲインを通る断面 | 34 | P–I、P–D、I–D。成功・記録ゲインの固定座標が同じ場合は1枚に統合 |
| 複数位置の断面一覧 | 24 | 各軸の10・30・50・70・90%に最も近い計算点と記録ゲインの断面 |
| 成功設定のIで切ったP–D一覧 | 2 | ケースごとに4グループ |

断面図は左に95%で分けた赤・緑、右に0〜100%の連続色を描いています。黒線は実際のセルの辺に沿った95%境界です。断面上にある記録点だけを描き、異なる固定座標の点は投影していません。真上の三次元図はD方向の全層を重ねた投影で、特定Dの断面とは異なります。S・Rには実際のD値を併記しています。

PDF・SVGの文字、軸、記録点はベクトル形式です。三次元図の密な半透明面だけを220 dpiでラスタライズし、ファイルを小さくしています。二次元断面のセルはベクトル形式で、値が完全に等しい隣接セルだけを矩形にまとめて容量を抑えています。各セルの表示値と境界位置はそのままです。図ごとの説明は[figure_index.csv](figures/figure_index.csv)に保存しています。

## 再描画・編集

このディレクトリで実行します。動作確認環境はPython 3.8.10、NumPy 1.24.3、Matplotlib 3.7.1、Plotly 5.24.1です。依存ライブラリは[requirements.txt](requirements.txt)にあります。

```bash
python3 -m pip install -r requirements.txt
python3 scripts/plot_paper_figures.py --output custom_figures
python3 scripts/build_interactive.py --output custom_figures/interactive.html
```

[figure_settings.json](figure_settings.json)で95%の閾値、赤・緑、透明度、視点、フォント、解像度、出力形式、断面一覧の位置を変更できます。データは再計算せずに使えます。配布済みの図を残して編集できるよう、上記では別の出力先を指定しています。

failure 1のRoll/Pitchだけを再描画する例：

```bash
python3 scripts/plot_paper_figures.py \
  --case failure1 --group roll_pitch --output custom_figures/roll_pitch
```

P–D断面を任意のI付近で描く例：

```bash
python3 scripts/plot_paper_figures.py \
  --case failure1 --group roll_pitch --fixed-axis I --fixed-value 1.5 \
  --output custom_figures/at_I_1p5
```

指定値に最も近い**実際の計算点**を選び、採用した固定値を標準出力・図・図の索引に明記します。補間は行いません。`--grid 33`で粗い格子も再描画できます。図の索引は、その出力先に保存済みの図と今回の図をまとめて書き出します。

## 保存したデータと容量

`data/<case>/<group>/grid_33.npz`と`grid_65.npz`の計16ファイルに、全解像度のゲイン軸・生存率・判定数・最大スペクトル半径を保存しています。各ノードの128サンプルの安定判定・有限値判定は、ビット列にして可逆圧縮しています。色、境界値、断面、視点を後から変更し、サンプルごとの安定判定を解析できます。

**全ノード×全サンプルのfloat64スペクトル半径は、このセットには含めていません。** 元のローカル計算結果に残っています。ここには全ノードの最大半径と、成功・記録ゲインでの128サンプルの半径をfloat64で保存しています。ρ<1という安定性判定自体を別の数値許容差に変えて全ノードを評価し直す場合は、元データが必要です。

単一ファイルの上限は50,000,000 bytes未満です。書き出し時とcommit前に検査しています。大きな元データの分割アップロードやGit LFSを使わず、図と解析に必要な情報を圧縮しました。`files.tsv`には全ファイルのサイズ・ハッシュが入っています。自身のハッシュが循環しないよう、`files.tsv`と`inventory.json`だけはその一覧から除外しています。

## 検証

```bash
python3 -m unittest discover -s tests -v
node tests/test_pid_voxel_viewer.cjs interactive/index.html
```

全16格子について、元のfloat64スペクトル半径から求めた安定・有限値フラグと保存後のビット列を照合しました。生存率、判定数、ゲイン軸、各診断値も元の配列と一致しています。結果は[validation/data_export.json](validation/data_export.json)に保存しています。ブラウザ描画、格子間の照合、図・ファイル上限の検査結果も`validation/`に保存しています。

入力モデル、計算時の設定、ソースのハッシュ、完了時刻は`data/<case>/manifest.json`と`export.json`にあります。`provenance/computation/`には当時の計算用コードも記録しています。再描画用スクリプトはこれらをimportせず、保存したNPZだけを読みます。
