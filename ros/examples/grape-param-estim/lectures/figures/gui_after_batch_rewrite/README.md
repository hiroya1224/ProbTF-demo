# Sparse batch rewrite 後の GUI visual acceptance

この directory は 2026-08-04 に実行した rewrite 後 GUI の視覚的受入結果を保存する。
すべて 1680 x 1040 pixel の production window を、対象 X11 window ID を特定して `xwd` で取得し、PNG へ変換した画像である。
最初に試した active-window screenshot は VS Code を取得していたため不採用とし、この directory には含めていない。

`bag_browser_rosrun.png` は `rosrun grape_param_estim run_gui.py` から失敗 sample を開き、inspection を完了して 3D observed trajectory と 2D observation plot が描画された実画面である。
`bag_browser_map.png`、`bag_browser_correction.png`、`bag_browser_dynamics_residual.png`、`master_map_laplace.png` は失敗 sample の `18.0--24.0 s` を解いた `/tmp/grape-sparse-real-18-24-run-20260804-c` を一時的な GUI fixture へ直接ロードし、production widget で描画した実 artifact の画面である。
後者の fixture は artifact 表示の視覚確認用であり、通常 project archive への自動 import 経路を証明するものではない。

| file | acceptance target | SHA256 |
|---|---|---|
| `bag_browser_rosrun.png` | rosrun 起動、inspection、PyVista 3D、pyqtgraph observation | `fcbe08d29a24137acc7a00e44974fca7a2e94aab375e381938281d6eb8311619` |
| `bag_browser_map.png` | observed、nominal、MAP world trajectory | `ed727960cb49c7748020dfd8316177888472554e46efec4270df6c65bba42dd0` |
| `bag_browser_correction.png` | actual correction-transform path | `b9d2353cea4e8f6047984e95e43faecd9f5b72358ef275608c23f028fde48861` |
| `bag_browser_dynamics_residual.png` | force/torque dynamics residual と Q reference band | `2dbc531f2f6198707d89ba452372c5c80fbe2a1ded19211f755d57002f89eeb3` |
| `master_map_laplace.png` | MAP、Q、EM、Laplace ridge、非収束 warning | `71a3a8cb33791f2e51c6237610f81ccdce48ed1a513182788834a9b2d27f4555` |

この実 run では observed と nominal/MAP trajectory が一致しておらず、推定成功と評価していない。
視覚的受入の合格対象は、新 schema の実データを fake curve なしで描画し、非収束と乖離を隠さず表示できることである。
