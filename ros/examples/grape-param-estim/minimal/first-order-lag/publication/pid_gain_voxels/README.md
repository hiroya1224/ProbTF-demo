# SuccessとFailureのPIDゲインの比較

示す結論は、**failure 1・2のRoll/Pitchの図で、失敗時のゲインは赤、成功時のゲインは緑に入る**ことです。

[比較図を開く](index.html) · [主図PDF](figures/roll_pitch_success_failure_PD.pdf) · [主図SVG](figures/roll_pitch_success_failure_PD.svg)

主図は、SuccessとFailureの両点を通るI=1のP–D断面です。failure 1と2を同じ表示範囲・同じ95%の判定で左右に並べ、点を直接Success／Failureと表示しています。

| 推定に使ったケース | Successのゲイン (P,I,D) | Failureのゲイン (P,I,D) | Successの生存率 | Failureの生存率 |
| --- | --- | --- | ---: | ---: |
| failure 1 | (13,1,20) | (20,1,8) | 100%：緑 | 55.91%：赤 |
| failure 2 | (13,1,20) | (10,1,8) | 100%：緑 | 14.06%：赤 |

ページの3Dボタンでも同じ2ケース・同じ2点を比較できます。I=1で手前を切り、両点がある赤・緑の面を露出させています。[3D PDF](figures/roll_pitch_success_failure_3D.pdf)も保存しています。

## 再描画

このディレクトリで実行します。

```bash
python3 -m pip install -r requirements.txt
python3 scripts/plot_paper_figures.py --output custom_figures
python3 scripts/build_interactive.py
```

標準実行は比較図とその3D版だけを生成します。[figure_settings.json](figure_settings.json)で色、閾値、透明度、視点、出力形式を変更できます。PDF・SVG・400 dpi PNGを保存しています。

再解析用の数値は[データ形式](DATA_FORMAT.md)、判定と図の読み方は[方法](METHODS.md)を参照してください。`data/`には33・65点格子の全グループの配列とサンプル別安定判定を保存しています。全点×全サンプルのfloat64半径の生配列は元のローカル計算結果に保持しています。

全ファイルは50,000,000 bytes未満です。[保存一覧](files.tsv)にサイズとSHA-256、`validation/`に検証結果を保存しています。元データの数値や従来の色の判定は変更していません。
