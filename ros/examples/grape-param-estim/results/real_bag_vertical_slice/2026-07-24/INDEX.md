# Grape real-bag vertical-slice index — 2026-07-24

All three runs are diagnostic-only: `workflow_status=EXPERIMENTAL`,
`recommendation_available=false`. The exact PC/MCU replay oracle,
bag-derived exact fixture, controller integrator state, joint
state/parameter inference, and complete held-out probability calibration are
not available.

- Source commit: `9b83e4de3eca38e59a65957c2387a5d2c6750bdc`
- Config SHA-256: `aebedd9b52b0702a050b7725e8197c5aa89908b3d8c3cabef2d5a1dd3d890faa`
- Config: `config/counterfactual.yaml`
- Frame/unit gate: ENU world, FLU body, SI pose/kinematics; passed for all runs
- Candidate grid: 27 common but unevaluated candidates per run

| episode | run ID | source bag SHA-256 | input slice SHA-256 | trajectory evidence SHA-256 |
|---|---|---|---|---|
| 20260612-04 | `b1b32ee30d43c05fe357` | `bd3fc7f71797c0f5cb665acc50832da93c590e540fa170f9977182ecedf93bf8` | `95063dbaf4391b6dd15b957a64d3548962454744db5c29e2a25c30425940feec` | `e387c1bb838c556dc1400a78fdac7f41c74c95d077184afe7f81b3c82a88a41e` |
| 20260612-07 | `c63e4ebe570b6943f7dd` | `75292b0c79dd1a3be2869eb3a0c3766df9561336efe2948386ebcb86e67297b8` | `fd3c6810721049765a244173d363ca104ac4116bda0e6840d873645214a81f01` | `0474c1d0005513198a7451bb401d16cf05b359644e263d2046888a73f27b7f35` |
| 20260612-08 | `bcb0786294cad8f7f565` | `dc141d4f6d9d3289d279771eb8d85b0765d1e76cd960f5c79211bf57869863c0` | `d53c84a02950b30d6912a3748f91658bdfacfce7b4e73592c2e2037fb31f2cc7` | `847be3904467b0498b1973fe40514009ceb3afaac9cf64d2fe131c5d8a802825` |

## Output file SHA-256

### 20260612-04 / b1b32ee30d43c05fe357

| file | SHA-256 |
|---|---|
| `REPORT.md` | `a8bf236fefc8d3b06cacce8c931ddc9b6534d6bdb5e0bc8e9d9c0179cecca98e` |
| `artifact_manifest.json` | `d19a87190b79f8a71996525b7b0dbb75110ed4d55f94763cbc3d44d2070d92db` |
| `candidate_grid.csv` | `f2397d0abaa58d06b07a2547d8993a4da8693451236640583e99d674c0deef5b` |
| `summary.json` | `7f362c9ac755828763f2a86e5a711e4bb7ae766ce68419099f43913794046105` |
| `trajectory.csv` | `432c8b34a044e4e4d84d73f3474a63a033ea7bc5f108abd4f2c16636b5f86ff7` |
| `trajectory_particles.npz` | `a3002578001966a914fa2e3ad8ec45a1be67b7a0898678993674171dde14142a` |

### 20260612-07 / c63e4ebe570b6943f7dd

| file | SHA-256 |
|---|---|
| `REPORT.md` | `afa52564fbc48ab467352140810ea01d4596b637e0786f854979044df68c6d69` |
| `artifact_manifest.json` | `27ad92c9fabbde51b2f112da52e4cc9a3d93a43be136b6d730d8182ccc9dd2c6` |
| `candidate_grid.csv` | `f2397d0abaa58d06b07a2547d8993a4da8693451236640583e99d674c0deef5b` |
| `summary.json` | `a03fa93ed0982713dae15992e0864f2e751cd543e4564bcf05b52dd50c3002aa` |
| `trajectory.csv` | `f0971ecadc7e1fbda11af5e6784e05ba336c31c20b400acbab49c065072c1abc` |
| `trajectory_particles.npz` | `20f3c80198a49c08dbb3875d42bf5e51192f6f8c1775c54603d6f661d32e1c52` |

### 20260612-08 / bcb0786294cad8f7f565

| file | SHA-256 |
|---|---|
| `REPORT.md` | `9a3aabd7360af43c3741ce6759234b654b56ec05ec1dfc7d06a6d9432aec8ab7` |
| `artifact_manifest.json` | `919e316e5b7e020b3a49633c14c83090e8e30b005fdec8c1140557e9f327969d` |
| `candidate_grid.csv` | `f2397d0abaa58d06b07a2547d8993a4da8693451236640583e99d674c0deef5b` |
| `summary.json` | `628bf97e4bc302f82878a5e5e6e42972a83ec7410de2967df1150161265aa418` |
| `trajectory.csv` | `d60bf54d45be661459db587248a1e10602ebc8b7c3bcbc542c6d19526c0c4f58` |
| `trajectory_particles.npz` | `e7a062702887572df4b645e2024f952fc1d0af6017d8dfa1024f311e2e86e14b` |

Each run-local `artifact_manifest.json` independently binds the five payload
files beside it. The NPZ contains the coherent RTS actual samples, matched
same-initial-state nominal samples, sample IDs/weights, and the exact
trajectory-evidence hash preimage.
