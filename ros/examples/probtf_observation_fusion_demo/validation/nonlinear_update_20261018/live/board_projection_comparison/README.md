# マーカーボードの中間縮約を延期した独立比較

問題の画像94では、各ID間の中間ADF縮約が外れの原因ではありません。有効な元観測はID17の2native枝だけであり、中間縮約が存在しません。同じ2枝を最後まで保持して最終一回だけ縮約すると、記録された中央観測と位置は完全一致し、姿勢差は1.32e−15 rad、共分散差は最大1.34e−17です。真のカメラ相対姿勢に対する後解析誤差69.133 mm／0.560832 radも残ります。

## 比較の入力と方法

`/tmp/probtf_nonlinear_fixed_prior_live_20261018/marker/stiffness_jobs` のjob93・94・95を読み、各payloadのSHA256をsidecarに照合しました。各jobに保存された `sensor_estimates[].marker_board_fusion.raw_observations` の全ID・全native成分、元の全共分散・位置方位cross・log質量・既知取付・体積係数を使いました。raw画像の再検出や新しい姿勢候補の生成はしていません。

各元nativeから中央へ写した既存のGaussian枝は変更せず、最初のIDの最初のnative方位を共通chartに固定しました。このchart内で全Gaussian積枝を列挙し、途中のBingham再表現、単峰化、位置方位cross切断を省き、全IDを掛け終わった後に既存の `_project_central` を一回だけ適用しました。全3件で既存coreの単一高集中域に入りました。分散clip・jitter・枝prune・モンテカルロはありません。

この計算は、同じ局所Gaussian代理を固定chartに写した積の比較です。raw画素からの厳密な非線形共同姿勢推定や、SE(3)上の厳密な全native密度積とは主張しません。省いた中間近似には、単峰化、再chart化、Bingham再fit、cross切断が含まれ、差をcross切断だけの効果と解釈しません。

| job／原画像ID | 有効ID数 | 保持した全組合せ | 既存ADFとの位置差 | 姿勢差 |
|---|---:|---:|---:|---:|
| 93／image:00002488 | 9 | 512 | 0.06197 mm | 0.002607 rad |
| 94／image:00002549 | 1 | 2 | 0 mm | 1.32e−15 rad |
| 95／image:00002602 | 4 | 16 | 0.58183 mm | 0.017129 rad |

各組合せの6次元平均、全6×6共分散、元label列、正規化前log質量、正規化log weight、通常weightを `job093_full.json` 等に保存しました。全labelは一意、全log質量は有限です。job93では492枝、job95では12枝の通常weightが浮動小数点で零へunderflowしますが、枝も有限log weightも削除していません。

## 物理真値との後解析比較

真値は別agentが、各原取得時刻の保存された実角度とカメラFKから算出した値です。融合・weight・chart・候補選定には使わず、以下の評価だけに使いました。

| job | 位置誤差：既存→最終一回 | 方位誤差：既存→最終一回 |
|---|---:|---:|
| 93 | 1.452→1.489 mm | 0.007121→0.004530 rad |
| 94 | 69.133→69.133 mm | 0.560832→0.560832 rad |
| 95 | 1.534→0.993 mm | 0.019972→0.003047 rad |

job94では元ID17の両native枝が既に外れていることが、別のraw mode後解析で確認されています。dominant枝の元log weightは約−2.08e−5、もう一枝は約−10.78です。全枝保持は、この二枝に含まれていない正しい候補を作り出しません。

比較用に同じ保存native入力から既存ADFも再実行し、保存済み結果を再現しました。位置差は全件零、姿勢差は最大6.80e−15 rad、共分散差は最大1.34e−17でした。保存時と今回のnative board演算libraryのSHAは一致し、関連sourceの前後hashも不変です。

## 保存資料

- 要約：`summary.json`
- 全枝と共分散：`job093_full.json`、`job094_full.json`、`job095_full.json`
- 再現script：`/tmp/probtf_board_deferred_projection_audit_20261018.py`
- 元物理後解析oracle：`/tmp/probtf_nonlinear_live_jump_audit_20261018/neighborhood_physical_oracle.json`

production、core、RViz、元job・raw DBは変更していません。この3画像の差を、全画像での改善や新しい実走・到達性能へ一般化しません。
