# Methods and interpretation

## Conditional-survival criterion

Each case uses its own estimated plant distribution (`conservative_fusion`). A fixed cohort of 128 scrambled-Sobol, inverse-normal-transformed samples (seed 0) is used throughout that case. Sampling coordinates, reference equilibrium trims, estimates and runtime settings are included in `data/`.

Let ρₛ(g) be the spectral radius of the exact, sampled, 26-state closed-loop linearization for plant sample s and controller gains g. A sample is stable when ρ is finite and strictly smaller than 1. Define the conditioning set once using the complete successful-flight controller:

```text
S0 = {s : isfinite(rho_s(g_success)) and rho_s(g_success) < 1}
Q(g) = 1 - count{s in S0 : not stable_s(g)} / |S0|
```

An unresolved candidate radius within S0 counts as a break. The denominator never changes with the candidate gain. The fixed denominator is 127 for failure 1 and 128 for failure 2. Accordingly, Q(g_success) = 1 by construction; that fact is not an independent validation of the successful-flight controller against every posterior sample.

The categorical plots are green for Q >= 0.95 and red for Q < 0.95. Continuous plots use Matplotlib's `RdYlGn` on a fixed [0,1] scale. Absolute worst-sample margin (`1 - max rho`) and raw absolute-margin classifications do not determine plot colors. The 95% threshold denotes an empirical conditional-survival fraction, not a confidence interval or a certified probability of safe flight. This is a local closed-loop stability analysis around the reference equilibrium, not a replay of a complete flight trajectory.

## Gain interventions and reference points

Only one control group's P, I and D gains change in each volume; the other nine gains remain at the successful-flight settings. Groups are XY, Z, Roll/Pitch and Yaw. The underlying full controllers and their per-sample radii are in `references.json`. The following triples are in P, I, D order:

| Group | Successful flight | Recorded failure 1 | Recorded failure 2 |
| --- | --- | --- | --- |
| Roll/Pitch | (13, 1, 20) | (20, 1, 8) | (10, 1, 8) |
| Yaw | (6, 1, 2) | (4, 1, 2) | (4, 1, 2) |
| XY | (4, 0.1, 2) | (3, 0.1, 1) | (3, 0.1, 1) |
| Z | (5, 1, 2.5) | (5, 1, 2.5) | (5, 1, 2.5) |

The blue diamond S denotes the successful-flight triple. The purple cross R denotes the recorded failure flight's triple for the selected group. R corresponds to a hybrid controller with the other nine successful gains; it does not represent simultaneous substitution of every failed-flight gain. Some R points consequently lie in green regions, and Z has coincident S/R points. The Roll/Pitch R points have conditional survival 71/127 (55.91%) for failure 1 and 18/128 (14.06%) for failure 2; both lie in red regions.

All reference gains are exactly evaluated grid nodes, strictly inside the computed domain, with at least one sixth of the axis span as clearance to either end. Domains are unchanged from the completed extended-domain calculations:

| Group | P bounds | I bounds | D bounds | Finest array shape (P,I,D) |
| --- | --- | --- | --- | --- |
| Roll/Pitch | [5,25] | [0.25,3] | [4,28] | 66 × 66 × 67 |
| Yaw | [2,10] | [0.25,3] | [0.5,5] | 65 × 66 × 66 |
| XY | [1.5,6] | [0.025,0.3] | [0.25,3] | 67 × 66 × 67 |
| Z | [2,10] | [0.25,3] | [0.75,5] | 65 × 66 × 66 |

The nominal 33/65 grids have nonuniform coordinate arrays because exact reference anchors were inserted. Each finest case contains 1,154,406 distinct gain nodes across the four groups. The cases share gain domains and reference successful settings but use different posterior plant cohorts. They must not be pooled.

## Voxel and slice geometry

For a strictly increasing axis a, the display edges are `[a[0], midpoint(a[0],a[1]), ..., midpoint(a[-2],a[-1]), a[-1]]`. Each node's computed value is assigned to that cell. No regression plane, convex hull, trilinear interpolation or smoothed contour is used. Same-class internal faces are culled for rendering; interfaces retain their adjacent class faces. The cells visualize node evaluations and do not certify stability for every unsampled gain within a cell.

Three-dimensional figures use an orthographic camera and normalized box aspect. Each physical gain axis has 10% display padding; only calculated cells are colored. A top view looks along the D direction and superposes layers; it is not a single-D classification map. Reference D values are labeled in top views. Transparency and the visible order of layers affect the apparent mixed color in a top projection; use the slices for unambiguous classifications at a fixed coordinate.

Slices are exact coordinate planes with piecewise-constant, nonuniform cells. P–I fixes D, P–D fixes I, and I–D fixes P. Black threshold boundaries follow cell edges. Reference markers are shown only when their fixed coordinate equals the displayed node exactly. Supplemental contact sheets select the closest evaluated coordinate to 10%, 30%, 50%, 70% and 90% of each physical axis span and additionally include the S/R coordinates.

For compact vector exports, axis-aligned adjacent rectangles are merged only when their scalar display values are exactly equal. This does not change any cell value, threshold boundary or gain-coordinate extent. The original individual-node arrays remain in the NPZ archives.

## Archive and reproducibility

The portable NPZ archives preserve all plotted float64 survival/coordinate values, integer counts, classification and diagnostic arrays, and the stable/finite decision for every plant at every node. Packing uses NumPy `packbits`, little bit order, along the final sample dimension. Sample index order is identical to `plant_samples.npz`, reference radii and baseline masks. Both reference nodes also retain their exact 128 float64 radii.

The archive intentionally omits the full node-by-sample float64 radius tensors; these remain at the original local paths in `export.json`. The compact archive supports changing the conditional-survival threshold, reference-plane selection, colors, cameras, typography and per-sample stability analyses. Reclassifying all nodes with a different spectral-radius tolerance requires the original tensors. The exact preserved maximum radius supports worst-sample-margin plots without those tensors.

`validation/data_export.json` records lossless roundtrip checks against all original grids. `validation/resolution_comparison.csv` compares coincident 33/65 nodes. Coincident-node agreement checks data consistency, and does not bound the approximation between unsampled points. `validation/bundle_validation.json` records figure coverage, editable text, sizes and portable regeneration. `provenance/computation/` captures the local computation scripts named in the original run manifests; the remaining model/solver implementation is in this repository at the recorded source revision and input hashes.

Suggested caption for a three-dimensional overview:

> Success-conditioned local stability over the PID gain domain for each control group, evaluated on a fixed posterior cohort. Green cells have conditional survival at least 0.95 and red cells have lower survival. The blue diamond marks the successful-flight gain triple and the purple cross the recorded failed-flight triple for that group; all other controller gains remain at the successful-flight settings. Cells display exact gain-node evaluations without interpolation. The two cases use separate posterior distributions.

Suggested caption for a reference slice:

> Exact PID gain slices through a recorded reference coordinate. The left panel shows the 0.95 survival threshold and the right panel the continuous conditional-survival fraction. Black boundaries follow cell edges, and markers appear only when their fixed coordinate lies on the displayed plane. Other control groups retain successful-flight gains.
