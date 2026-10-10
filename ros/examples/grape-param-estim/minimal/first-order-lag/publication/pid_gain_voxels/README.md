# 全PID軸のSuccessとFailure

**failure 1・2の全4軸で、SuccessのPIDゲインは緑の領域に入ります。**

[比較図を開く](index.html) · [主図PDF](figures/all_pid_success_failure_3D.pdf) · [主図SVG](figures/all_pid_success_failure_3D.svg)

主図は上段がfailure 1、下段がfailure 2、列がRoll/Pitch・Yaw・XY・Zの三次元ボクセルです。P・I・Dをすべて表示し、全8図にSuccessの判定を明記し、Success／Failureの全点に`rate : xx.x%`を表示しています。全ての面を同じ判定ごとの透明度で描き、淡い緑と赤で領域を示し、紺の菱形がSuccess、紫の大きな印と矢印がFailureです。

Zのゲインは成功・失敗飛行で同じため、紫の円でSuccessの点を囲み、同一座標であることを表示しています。点をずらしてはいません。

| 軸 | Successのゲイン (P,I,D) | Failureのゲイン (P,I,D) | Success：failure 1 / 2 | Failure：failure 1 / 2 |
| --- | --- | --- | --- | --- |
| Roll/Pitch | (13,1,20) | failure 1: (20,1,8)、failure 2: (10,1,8) | 緑100% / 緑100% | 赤55.9% / 赤14.1% |
| Yaw | (6,1,2) | (4,1,2) | 緑100% / 緑100% | 緑96.9% / 緑98.4% |
| XY | (4,0.1,2) | (3,0.1,1) | 緑100% / 緑100% | 緑100% / 緑100% |
| Z | (5,1,2.5) | (5,1,2.5) | 緑100% / 緑100% | 緑100% / 緑100% |

Failureは失敗飛行で使った各軸のゲインの位置を表します。各図では他の9ゲインを成功設定に固定しているため、Failureの各点の判定は上表のとおりです。Roll/Pitchだけが赤になります。判定は従来どおり、成功設定で安定なサンプル集合に対する生存率95%以上が緑です。

俯瞰図は両点を通るIで手前を切り、点のある面を露出させています。ページの「両点を通る断面」で全軸のP–D断面に切り替わります。「点の位置を操作」では各軸を拡大して回転でき、切り出しを解除すると元の探索範囲全体を表示できます。

## 再描画

このディレクトリで実行します。

```bash
python3 -m pip install -r requirements.txt
python3 scripts/plot_paper_figures.py --output custom_figures
python3 scripts/build_interactive.py
```

標準実行で全軸の俯瞰図と断面図の2種類を生成します。[figure_settings.json](figure_settings.json)で色、閾値、透明度、視点、出力形式を変更できます。PDF・SVG・400 dpi PNGを保存しています。

[データ形式](DATA_FORMAT.md)と[方法](METHODS.md)に再解析の手順と判定を記載しています。`data/`には全軸の33・65点格子とサンプル別安定判定を保存しています。全点×全サンプルのfloat64半径の生配列は元のローカル計算結果に保持しています。

全ファイルは50,000,000 bytes未満です。[保存一覧](files.tsv)にサイズとSHA-256、`validation/`に検証結果を保存しています。
