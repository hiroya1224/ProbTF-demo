# Probability AprilTag Phase 3 評価報告

実施日: 2026-07-22

## 結論

Phase 1 の deterministic renderer が生成した clean synthetic image を、Phase 2 の
`prob_artag_detector` と mixture estimator に直接入力した。遮蔽なしの5条件では
42/42 tag を正しい ID で検出し、false positive は0だった。3タグをカメラ移動下で
10フレーム追跡する `multi_tag` 条件も30/30 tag、ID accuracy 100%、precision 100%
だった。

強い遮蔽条件は0/3検出だったが、全フレームを例外なく処理し、3件を `missed` として
記録した。これは遮蔽下の recall 達成ではなく、検出不能入力を安全に扱う確認である。

## 固定した評価 fixture

- dictionary: `DICT_APRILTAG_36h11`
- tag edge: 0.12 m
- renderer seed: 2
- camera: 640×480、fx=fy=600 px、歪みなし
- corner covariance: \(\Sigma_u=0.5^2 I_8\) px²
- GT近傍IPPE解の判定: translation ≤0.02 m かつ rotation ≤5°
- `frontal`, `moderate`, `oblique`, `small`, `occluded`: 各3フレーム、1タグ
- `multi_tag`: 10フレーム、各3タグ
- OpenCV 4.7.0、NumPy 1.24.3、SciPy 1.10.1、pyrender 0.1.45

集計値の機械可読版は
[`prob_artag_phase3_metrics.json`](prob_artag_phase3_metrics.json) に保存した。

## 結果

`nearest mode` は、各GT tagに対応するrefined modeを、近傍判定閾値で正規化した
translation/rotation誤差の二乗和で選ぶoracle指標である。別に、refinement前のIPPE
seedだけを用いて「GT近傍解が二候補中に存在するか」、二候補の再投影RMSE差、回転・
並進それぞれの独立minimumも集計した。accepted modeとrejected seedを同じ集計へ混ぜず、
二解平均を単一姿勢精度のように扱っていない。near rateの分母は、IPPE候補が厳密に
2件生成されたtagだけである。候補未生成は「近傍解なし」ではなくcoverage不足として分ける。

| 条件 | frames / GT | recall | ID accuracy | 2-IPPE coverage | near / 2-IPPE | IPPE reproj. gap mean [px] | corner RMSE mean / max [px] | nearest translation mean / max [m] | nearest rotation mean / max [deg] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frontal | 3 / 3 | 100% | 100% | 100% | 3 / 3 | 0.202 | 0.715 / 0.720 | 0.0026 / 0.0026 | 0.706 / 0.861 |
| moderate | 3 / 3 | 100% | 100% | 100% | 0 / 3 | 0.356 | 1.568 / 1.591 | 0.1091 / 0.1144 | 1.507 / 1.684 |
| oblique | 3 / 3 | 100% | 100% | 100% | 3 / 3 | 1.268 | 0.643 / 0.702 | 0.0014 / 0.0018 | 0.943 / 1.047 |
| small | 3 / 3 | 100% | 100% | 100% | 0 / 3 | 0.247 | 1.924 / 1.949 | 0.1048 / 0.1070 | 1.206 / 1.362 |
| multi_tag | 10 / 30 | 100% | 100% | 100% | 10 / 30 | 0.261 | 1.190 / 1.523 | 0.0722 / 0.1478 | 1.049 / 4.225 |
| occluded | 3 / 3 | 0% | — | 0% | — | — | — | — | — |

投影 edge が50 px以上の16 tagだけを見ると、corner RMSE は平均0.708 px、nearest modeの
並進誤差は平均1.95 mm・最大2.64 mm、回転誤差は平均0.472°・最大1.047°だった。
一方、約29–40 pxの tagでは1 px前後のcorner誤差が平面PnPの奥行きへ強く増幅され、
並進誤差が約0.1 mに達した。これは clean imageでも小さい平面マーカのdepthが弱観測
になることを示しており、実画像での利用時は projected size gate、複数フレーム統合、
または追加sensor constraintが必要になる。

## 記録内容

評価器 `misc/prob_artag_benchmark` はrendererをimportせず、Phase 1の
`rgb.png` と `metadata.json` をwire boundaryとして読む。各実行で次を出力する。

- `metrics.json`: frame/tag/detection/candidateの階層レポート
- `frames.csv`: detection、ID、miss、false positive、frame error
- `tags.csv`: ordered corner error、GT近傍IPPE解の有無、二解の再投影差、pose error
- `candidates.csv`: 全IPPE seed/modeの姿勢、誤差、再投影、objective、weight、status
- `overlays/*.png`: GT/detected corner orderと全IPPE seed axes

返却modeと元のIPPE候補は `PoseMixtureResult.seed_indices` で対応し、同じIPPE seed tupleを
estimatorへ渡すため候補順を再計算しない。旧buildに限りcomponent provenance、最後に
pose proximityを使う互換fallbackを持つ。

代表画像:

- [frontal](prob_artag_phase3_overlays/frontal.png)
- [moderate](prob_artag_phase3_overlays/moderate.png)
- [oblique](prob_artag_phase3_overlays/oblique.png)
- [small](prob_artag_phase3_overlays/small.png)
- [occluded](prob_artag_phase3_overlays/occluded.png)
- [multi-tag](prob_artag_phase3_overlays/multi_tag.png)

## 再現手順

workspaceをbuild/sourceし、rendererとbenchmarkのsourceを `PYTHONPATH` に加える。
headless renderではこの環境でMesa EGL vendorを明示した。

```bash
cd /home/leus/catkin_ws
source devel/setup.bash
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export PYTHONPATH="$PWD/src/ProbTF-demo/misc/prob_artag_renderer/src:$PWD/src/ProbTF-demo/misc/prob_artag_benchmark/src:$PWD/src/ProbTF-demo/ros/examples/prob_artag_detector/src:$PYTHONPATH"
```

editable install済みのconsole scriptで、単一タグ条件を次のように作る。

```bash
prob-artag-generate-dataset --output /tmp/prob-artag-phase3/frontal/dataset \
  --seed 2 --scenario frontal --frames 3 --overwrite
prob-artag-benchmark /tmp/prob-artag-phase3/frontal/dataset \
  /tmp/prob-artag-phase3/frontal/report
```

同様に `moderate`, `oblique`, `small`, `occluded` を各3フレーム作る。
multi-tag条件は次の通りである。

```bash
prob-artag-generate-dataset --output /tmp/prob-artag-phase3/multi_tag/dataset \
  --seed 2 --scenario multi_tag --frames 10 --count 3 --overwrite
prob-artag-benchmark /tmp/prob-artag-phase3/multi_tag/dataset \
  /tmp/prob-artag-phase3/multi_tag/report
```

multi-tag datasetを異なる2 rootへseed=2で再生成した。各rootのRGB、depth、instance、
metadata計40 fileと、それぞれを評価したJSON、全CSV、overlay計14 fileを `diff -rq` で
比較し、すべてbyte-identicalだった。54 fileの相対path・size・SHA-256から作ったtree hashは
`084e0be01df462f93901303d9d8c0ef876e562a8bbfcfc29f925f1c6d8724f12` である。
report内のimage pathもdataset rootに依存しない。aggregateは比較treeのschema、config、
summary、candidate数、seed、scenarioが本評価のmulti-tag入力と一致することも検証する。

6条件のaggregate JSONは次で再生成できる。

```bash
prob-artag-aggregate /tmp/prob-artag-phase3 \
  src/ProbTF-demo/docs/reports/prob_artag_phase3_metrics.json \
  --projected-size-threshold-px 50 \
  --compare-tree /tmp/prob-artag-regeneration-a /tmp/prob-artag-regeneration-b
```

## 適用範囲と残課題

この結果はclean synthetic fixtureに対するgeometry/API integration testであり、
確率分布のcalibration保証ではない。現時点で未評価なのは、実camera calibration誤差、
printer/optics、複合degradation下のrecall、mixture weightのcoverage/NLL、連続時系列の
Prob-TF fusionである。Phase 1は各degradationを独立に切り替えられるので、以後は同じ
benchmarkで一因子ずつsweepできる。
