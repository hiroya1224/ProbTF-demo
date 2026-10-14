# 9軸の全情報の融合とMadgwickの9軸版の比較

比較する2系列を「9軸の全情報の融合」と「Madgwick（9軸）」に絞った。新しい図5.5・5.6は、図5.3・5.4と同じ縦向きの4段で、上からロール、ピッチ、ヨー、3軸の合成誤差を示す。各段には2系列だけを重ねる。線形図と片対数図は同じ構成・色・線種・時刻範囲で、縦軸の種類を変える。

同じ2025年3月10日のrosbagに対し、85.446秒、17,088時刻を比較する。Madgwickの固定係数はbeta=0.1で、角速度・加速度・地磁気の入力と初期方位をそろえる。推定を再計算せず、保存済みの2系列を取り出した。モード、各軸と3軸の合成誤差は以前の値を保持する。

| 条件 | ロールRMS | ピッチRMS | ヨーRMS | 3軸の合成誤差RMS |
| --- | ---: | ---: | ---: | ---: |
| 9軸の全情報の融合 | 6.4791 | 3.0014 | 18.5729 | 20.3905 |
| Madgwick（9軸） | 12.3425 | 4.8189 | 29.6862 | 28.1676 |

数値は度である。各軸は参照とのEuler角の差を±180度へ折り返した絶対値、3軸は相対回転角で評価する。論文の表もこの2行に整理した。6軸との比較、7系列の重ね描き、参照の傾きを使う地磁気由来のヨー、固定係数を変えた補助比較は今回の本文から外した。以前の補助図と計算資料は履歴として`references/imu-orientation-filter/20250310-delta-linear-rolling-madgwick/`へまとめた。

[線形比較図](../../ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_nine_axis_madgwick/method_comparison.pdf)と[片対数比較図](../../ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_nine_axis_madgwick/method_comparison_semilog.pdf)をPDF/SVG/PNGで保存した。色は「9軸の全情報の融合」が朱色の実線、Madgwick（9軸）が濃紫の破線である。線形図は0から表示し、片対数図は誤差が0の標本を表示しない。図5.5・5.6は縦向きで配置する。

ProbTF-demoでは`compare_madgwick_rolling.py`の標準出力もこの2系列の図に変更した。保存済みの比較結果からの再生成は、リポジトリのルートで次を実行する。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 ros/examples/probtf_orientation_demo/scripts/compare_madgwick_rolling.py \
  --render-only \
  --input ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_madgwick \
  --output ros/examples/probtf_orientation_demo/outputs/20250310_delta_linear_rolling_nine_axis_madgwick
```

新しい出力先には2枚の図、集計、2系列の時系列を保存する。`--render-only`を省く場合は既存の5条件の入力を指定し、Madgwickの9軸版を計算して比較する。本文側の数値と使用コードは`references/imu-orientation-filter/20250310-delta-linear-rolling-nine-axis-madgwick/`へ保存した。計算版と今回の評価・描画版のハッシュを分けて記録する。

## 最終精査

2系列のモード、RPY、各軸と3軸の合成誤差は、保存済みの9軸の値と全時刻で完全一致した。入力・参照の7配列も一致した。両図のSVGで4段・各段2本と系列名を照合し、計算・描画の検査11件が通過した。新規計算の入口も20標本で検査し、保存した9軸の結果と一致した。

本文の全体検査、用語の再生成、配置の検査、PDFのビルドが通過した。862ラベル、75引用、75図、28件の未検証事項を照合し、全362ページが縦向き・回転属性0であることを確認した。図5.5・5.6は171・172ページにあり、図5.3・5.4と同じ4段の縦向きの図である。追加部分のyomiyasu検査は100/100だった。

以前の補助図21ファイルはハッシュを保持して再生成資料へ移し、今回の本文から参照を外した。旧比較の集計とコードも履歴として保持した。[最終検証記録](orientation-rolling-nine-axis-madgwick-final-validation_2026-10-15.json)、dthesisの本文・PDF検査、配置、文章へ照合結果を保存した。
