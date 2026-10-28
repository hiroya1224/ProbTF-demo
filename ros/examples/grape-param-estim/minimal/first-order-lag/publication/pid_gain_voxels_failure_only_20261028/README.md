# 失敗記録だけから計算したPID安定割合

2026年10月28日の監査修正後に、成功情報を入力せず再計算した結果です。[オフラインビューアー](index.html) で2ケースと4群を切り替えられます。

元の各128機体標本を固定分母とし、有限かつ有効なスペクトル半径が1未満の標本の割合を表示します。他9ゲインは当該失敗記録の値へ固定しています。成功点の100%を保証する条件付けはありません。

実行した格子は3→5、公開したのは最終5格子です。既知の失敗値を正確に挿入した実配列は各軸6点、1群216点です。33・65格子の新結果ではありません。

| 群 | 失敗1の図 | 失敗2の図 |
|---|---|---|
| Roll/Pitch | [PNG](data/failure1/roll_pitch/grid_5/pid_3d.png) | [PNG](data/failure2/roll_pitch/grid_5/pid_3d.png) |
| XY | [PNG](data/failure1/xy/grid_5/pid_3d.png) | [PNG](data/failure2/xy/grid_5/pid_3d.png) |
| Z | [PNG](data/failure1/z/grid_5/pid_3d.png) | [PNG](data/failure2/z/grid_5/pid_3d.png) |
| Yaw | [PNG](data/failure1/yaw/grid_5/pid_3d.png) | [PNG](data/failure2/yaw/grid_5/pid_3d.png) |

`data/<case>/` に入力estimate、同じ順序の全機体標本、元の計算manifest、全12失敗ゲインの参照結果を保存しています。各群の `grid_5/volume.npz` には全格子点×全128標本のスペクトル半径、`stability_fraction.npz` には安定数・割合・有効数・未評価数が入っています。中間memmap・低解像度途中結果・重複HTML・大容量raw bagは同梱しません。

- [修正・数式・限界・実測の報告](../../AUDIT_CORRECTION_20261028.md)
- [再計算の数値要約](recalculation_summary.json)
- [元標本と原推定の同一性検証](validation/sample_identity.json)
- [回帰テスト](validation/tests.json)
- [実ブラウザーでの8通りの切替検証](validation/browser.json)

この割合は有限モデル標本の局所線形安定性です。実飛行成功確率、格子間全点の保証、異なる群の任意の良い点を組み合わせた制御器の保証ではありません。全群を変更する候補は全12ゲインを同時評価してください。
