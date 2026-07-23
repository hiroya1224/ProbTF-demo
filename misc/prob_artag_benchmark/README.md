# prob_artag_benchmark

Phase 1 の合成データセットを Phase 2 の detector / mixture estimator に
入力する、ROS master 不要のオフライン評価器です。renderer package は import
せず、各 frame の `metadata.json` を wire format として読みます。このため
OpenGL を利用できない評価ホストでも既に生成済みのデータセットを評価できます。

## 入出力

入力は次の Phase-1 layout、または `frames/` 自体です。

```text
dataset/
  frames/
    000000/
      rgb.png
      metadata.json
```

`metadata.json` の camera は `camera_matrix` と `distortion`、tag は `id`,
`size_m`, `T_C_M`, ordered `corners_px` を持つものとして扱います。
`corners_px` は IPPE_SQUARE の厳密な `TL, TR, BR, BL` 順です。

出力は入力順や wall-clock time に依存しません。

```text
output/
  metrics.json
  frames.csv
  tags.csv
  candidates.csv
  overlays/000000.png
```

- `metrics.json`: frame/tag/detection/candidate の完全な階層レポート
- `frames.csv`: recall、ID match、miss、false positive
- `tags.csv`: corner error、近傍IPPE解、再投影差、nearest/minimum pose error
- `candidates.csv`: 全 IPPE candidate の translation/rotation/reprojection
  error、objective、mixture weight、accept/reject reason
- overlay: GT/detected ordered corners と全 IPPE seed の x/y/z axes

検出なし、画像欠落、IPPE 退化、refinement/Hessian 失敗は行を失わず、明示的な
status/reason として記録します。

corner RMSE は4 cornerそれぞれのEuclidean errorの二乗平均平方根です。
GT近傍IPPE解は既定でtranslation ≤0.02 mかつrotation ≤5°と定義します。
`nearest_*` はこの2閾値で正規化したSE(3)誤差の二乗和が最小の同一候補、
`minimum_ippe_*` は回転・並進を独立に最小化した値です。
GT-nearと二候補再投影差はIPPE seedが厳密に2件ある場合だけ定義し、候補生成coverage
とconditional near rateを分けます。ID対応はfamily一致と25 pxのgeometry gateを使い、
同一ID候補群とwrong-ID候補群の各段階でglobal minimum-cost assignmentを行います。

## 実行

catkin workspace を build/source 済みなら、benchmark sourceだけを加えます。

```bash
cd /home/leus/catkin_ws
source devel/setup.bash
PYTHONPATH="$PWD/src/ProbTF-demo/misc/prob_artag_benchmark/src:$PYTHONPATH" \
python3 -m prob_artag_benchmark.cli \
  /path/to/dataset /tmp/prob-artag-metrics
```

実行にはOpenCV contrib版、SciPy、およびbuild/source済みの
`prob_artag_detector` / ProbTF Python packageが必要です。pip環境では
`pip install -e '.[test]'` でbenchmark側の依存を追加できます。ROSのapt版OpenCVを
使う環境では、workspaceをsourceしたうえで `--no-deps` を選べます。

未インストールの各 source tree を直接使う場合は次の形です。

```bash
PYTHONPATH="src/ProbTF-demo/misc/prob_artag_benchmark/src:\
src/ProbTF-demo/ros/examples/prob_artag_detector/src:\
src/ProbTF-demo/ros/core/probtf_core/src:$PYTHONPATH" \
python3 -m prob_artag_benchmark.cli DATASET OUTPUT
```

`recall` は正しい ID で対応した tag 数 / 評価対象 GT 数です。ID が違うものの
corner geometry が近い検出は `geometric_wrong_id` として別集計します。既定では
back-facing tag を除外し、`visible_fraction >= 0` を評価対象にします。

## Phase-2 API boundary

既定 adapter が利用する公開 API は次だけです。

- `CameraModel(camera_matrix, distortion, width, height)`
- `ArucoCornerDetector(...).detect(image)`
- `solve_ippe_square_candidates(corners, camera, tag_size_m)`
- `PoseMixtureEstimator(...).estimate(...)`
- result の `record`, `diagnostics`, `rotations`, `translations`,
  `seed_indices`, `log_masses`, `weights`

API が欠ける場合は detector package を変更せず `ApiMismatchError` にします。
`PoseMixtureResult.seed_indices` が返却 mode と元の IPPE `seed_index` を直接対応
させ、現行APIでは同じseed tupleをestimatorへ渡して二重solveを避けます。古いdetector
buildに対しては独立solveを許し、component provenance の
`Mode initialized by IPPE candidate N.` を狭く読み、それも存在しない場合だけ
pose proximity で決定的に補完します。この互換方針は `metrics.json` の
`api_notes` にも記録します。

Phase 3 で固定した seed、評価条件、定量結果、代表 overlay は
[`docs/reports/prob_artag_phase3_evaluation_ja.md`](../../docs/reports/prob_artag_phase3_evaluation_ja.md)
にあります。
