# Portable data format（旧方式・撤回済み）

> **2026年10月28日：この図系列を、失敗データだけから安定ゲインを予測した証拠として使うことを撤回します。**
> 以下のデータ・図・計算snapshotは旧方式の監査用履歴です。成功ゲインで安定だった標本だけを分母にし、他の9ゲインも成功値に固定していたため、成功点の100%は独立な予測結果ではありません。
> 修正版は元の全標本を固定分母とし、断面の他9ゲインを同じ失敗記録の値に固定します。旧NPZの母数だけを変えて、修正版の図として再利用することはしません。
> この撤回告知を加えた文書・入口HTML以外の旧数値・図・snapshotは保持しています。`files.tsv` と既存validationは告知追加前の履歴資料です。

修正後の独立した再計算は [失敗記録だけを使った新結果](../pid_gain_voxels_failure_only_20261028/README.md) を参照してください。

以下は旧方式の記録です。

Read any archive with `numpy.load(path, allow_pickle=False)`. Gain-grid dimensions are always `(nP,nI,nD)`; the sample dimension is last where present. No pickled Python objects or external filesystem references are needed to load the arrays.

| Key | Type / shape | Meaning |
| --- | --- | --- |
| `p`, `i`, `d` | float64 vectors | Exact nonuniform evaluated gains, including reference anchors |
| `survival_fraction` | float64 grid | Q(g) on the fixed success-stable cohort |
| `caused_break_count` | int32 grid | Unstable or unresolved candidates inside the conditioning set |
| `conditional_unresolved_count` | int32 grid | Nonfinite candidates inside the conditioning set |
| `classification` | uint8 grid | Historical 2/3 classification at 0.95; preserved from the original run, unused by plotting |
| `alpha` | scalar float64 | Historical threshold, 0.95; unused by plotting |
| `baseline_stable_mask` | bool, 128 | The fixed conditioning-set membership |
| `baseline_spectral_radius` | float64, 128 | Radii for the full successful-flight controller |
| `sample_count` | scalar int32 | 128 |
| `stable_bits` | uint8, grid + (16,) | Packed finite ρ<1 decisions for all 128 plants |
| `valid_bits` | uint8, grid + (16,) | Packed finite-radius decisions for all plants |
| `delta` | float64 grid | Absolute diagnostic: 1 minus the worst radius |
| `worst_spectral_radius` | float64 grid | Exact worst sample radius, as stored in the original run |
| `worst_sample_index` | int32 grid | Index of the worst plant sample |
| `pole_valid_count` | int32 grid | Number of finite pole results |
| `near_kink_count` | int32 grid | Original near-kink diagnostic count |
| `success_spectral_radius` | float64, 128 | Exact per-plant radii at this group's successful reference node |
| `recorded_case_spectral_radius` | float64, 128 | Exact radii at the hybrid controller using this group's recorded triple |

The figures use `survival_fraction` directly with a common continuous 0–100% color scale. The archived `classification` and `alpha` fields are retained for reproducibility and do not control colors.

Unpack and recompute the original survival fraction:

```python
import numpy as np

with np.load('data/failure1/roll_pitch/grid_65.npz') as a:
    stable = np.unpackbits(a['stable_bits'], axis=-1,
                           count=int(a['sample_count']), bitorder='little').astype(bool)
    finite = np.unpackbits(a['valid_bits'], axis=-1,
                           count=int(a['sample_count']), bitorder='little').astype(bool)
    baseline = a['baseline_stable_mask']
    breaks = np.count_nonzero(~stable[..., baseline], axis=-1)
    survival = 1.0 - breaks / np.count_nonzero(baseline)
    assert np.array_equal(survival, a['survival_fraction'])
    rate_percent = 100 * survival
```

Sample-weighted or subset analyses can use the unpacked decisions with the same sample indices and a clearly stated conditioning rule. Changes to sample selection change the meaning of the resulting fraction; keep them distinct from the published default cohort.

`data/<case>/plant_samples.npz` stores the original `quotient_coordinates`. `estimate.json` contains the posterior estimate and covariance metadata. `references.json` stores the complete recorded and successful gain matrices in XY, Z, Roll/Pitch, Yaw order and their full-controller radii. These full-controller reference radii differ conceptually from the per-group hybrid-reference radii in each grid archive.

`export.json` maps every portable archive to its original run and hashes the original radius tensor, original conditional archive and portable result. Original manifests are preserved without rewriting their historical absolute paths. These paths provide provenance only; plotting uses relative paths inside this bundle.
