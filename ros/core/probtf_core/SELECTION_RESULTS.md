# Temporal model selection results

評価日: 2026-07-24

## 結論

**production `DEFAULT` は未選定である。**

conformance、有限値、quaternion normalization、PSD などの共通 gate は通過した
一方、moment/sample のどちらも全 distribution stratum の nominal 95% coverage
gate を通過しなかった。sample backend はさらに factor-level joint-sample
contract を満たしていない。したがって backend の hard gate conjunction は
両方 `false` であり、これに依存する motion model も production default に
昇格させない。

## 再現 provenance

| 項目 | 記録値 |
|---|---|
| run timestamp (UTC) | `2026-07-24T12:15:43.542116+00:00` |
| repository HEAD at run | `ccef72f10dd1762cfb340d7f458d62f1d9ddda3f` |
| comparison/core implementation commit | `dee44b6` |
| whole worktree dirty | `false` |
| core/tests worktree dirty | `false` |
| core HEAD tree | `37f1f25b8cee439b9c23e0a2079afa85f99f8659` |
| `tests/probtf` HEAD tree | `21066323f2e2349f0711e47a68f1c67521c19f3e` |
| evaluated core source SHA-256 | `0a4db5b3f5b6cf443ed230178ffe552d2648343c665bbd13247c397dfa431f7e` |
| config SHA-256 | `fe775ac61e6ffc686f2dee218e8019f9ea2030025a452f8733e79ab96cd4781d` |
| runner SHA-256 | `dde2723d3ae19c95db041e49a34b77e797091d7a9e597f3da3c200ee7c05f405` |
| corpus SHA-256 | `cc0486c72fba0a91bbc64bc153f2414ec608f05c2448eb8e2a1ac343bb4b12fc` |
| result artifact SHA-256 | `72e61c62e2f65dffc9d9ddacd678476796b61e21202be319b6894613bae97c22` |
| conformance | `192 passed in 9.53s`, exit `0` |

`comparison/core implementation commit` は凍結 config が比較基準として指定した
commit である。実際に評価した checkout は `repository HEAD at run` と
tree/source hash で一意に追跡する。

raw artifact:
[test/temporal_selection_results_2026-07-24.json](test/temporal_selection_results_2026-07-24.json)

## 凍結 protocol

- config は結果を見る前の `2026-07-24T00:00:00+09:00` に凍結した。
- 一つの synthetic corpus を 4 training episode と 2 held-out episode に固定した。
- random seed は `1729`, `2718`, `31415`, `65537`。
- uncertainty metric の bootstrap 単位は `distribution_case_seed` (`n=16`)。
- motion model の選択単位は episode、held-out は `n=2`。
- coverage は全体平均ではなく distribution stratum ごとの Wilson 95% CI と
  nominal `0.95` の包含を conjunction した。
- correctness gate を性能より先に適用し、不通過候補から default を選ばない。

## Hard correctness gate

| backend | common gates | covariance PSD | 全 stratum coverage | shared-factor contract/safe policy | 全 hard gates |
|---|---:|---:|---:|---:|---:|
| tangent-space moment | pass | pass | **fail** | pass（近似を明示） | **fail** |
| sample-wise | pass | pass | **fail** | **fail** | **fail** |

最大 quaternion norm error は `2.220446049250313e-16`。legacy `Qd` adapter の
0.05 s/0.1 s 二レート等価性は、pose error
`2.8379169458017997e-16`、covariance Frobenius error
`2.043951178627398e-18` で、絶対 tolerance `1e-9` を通過した。

### Distribution stratum ごとの empirical 95% coverage

括弧内は Wilson 95% CI。gate は CI が nominal `0.95` を含む場合だけ pass。

| distribution | moment | gate | sample | gate |
|---|---:|---:|---:|---:|
| bimodal orientation | 0.960327 (0.955880–0.964343) | fail | 0.932478 (0.926434–0.938058) | fail |
| Dirac | 0.947998 (0.942977–0.952600) | pass | 0.928711 (0.922522–0.934441) | fail |
| local Gaussian | 0.948608 (0.943613–0.953183) | pass | 0.944057 (0.938495–0.949143) | fail |
| strong translation-rotation coupling | 0.940308 (0.934968–0.945235) | fail | 0.942383 (0.936748–0.947544) | fail |

## Uncertainty backend metric

平均と distribution case/seed 構造を保持した bootstrap 95% CI を示す。

| backend | pose error | covariance relative error | energy distance |
|---|---:|---:|---:|
| moment | 0.114587 (0.031410–0.223002) | 0.500399 (0.160267–0.944308) | 0.001209 (0.000937–0.001498) |
| sample | 0.010750 (0.007653–0.014001) | 0.117808 (0.097845–0.136260) | 0.000775 (0.000564–0.000993) |

sample の energy-distance 改善率は 0.337342
(95% CI 0.145404–0.490399) で、事前設定した 10% rule は通過した。しかし
coverage と shared-factor contract が hard gate なので、この改善だけで
`OPTIONAL` や `DEFAULT` には昇格させない。

## Performance

各値は 20 repetitions。memory は Python peak bytes。

| candidate | p50 (s) | p95 (s) | peak bytes |
|---|---:|---:|---:|
| moment constant twist | 0.015979 | 0.016359 | 51,074 |
| sample constant twist (256 samples) | 0.351664 | 0.360837 | 824,638 |
| moment constant acceleration | 0.427165 | 0.434205 | 50,213 |

sample constant twist の p50 は moment の `22.008x`。sample scaling の p50 は
32/64/128/256 samples でそれぞれ
0.052141/0.095862/0.182273/0.351284 s だった。

## Motion model の一標準誤差則

held-out episode は `constant_angular_acceleration` と
`timestamp_jitter_dropout` の二つ。

| model | mean episode pose RMSE | standard error | best + 1 SE 内 |
|---|---:|---:|---:|
| constant body acceleration | `1.5020e-8` | `3.3944e-9` | yes |
| constant body twist | `0.0127701` | `0.0028670` | no |

constant acceleration は synthetic corpus 内の
`constant_linear_acceleration`、`constant_angular_acceleration`、
`timestamp_jitter_dropout` で 10% 改善 rule を通過した。しかし全 episode は
同じ一つの synthetic corpus に属する。`OPTIONAL` に必要な二つの独立 corpus
を満たさないため `EXPERIMENTAL` とする。

## 最終 disposition

| candidate | status | 理由 / 次に必要な evidence |
|---|---|---|
| tangent-space moment backend | `EXPERIMENTAL` | 最速だが bimodal/coupled coverage gate が fail。application corpus で再同定・再評価が必要 |
| sample backend | `EXPERIMENTAL` | nonlinear metric は改善したが coverage fail。factor-level joint-sample contract が必要 |
| constant body twist | `EXPERIMENTAL` | 最小の causal baseline だが hard-gate-passing backend がなく、held-out 1-SE rule の外 |
| constant body acceleration | `EXPERIMENTAL` | 改善は一つの synthetic corpus だけ。独立した二つ目の application corpus が必要 |
| endpoint-conditioned sample interpolation | `EXPERIMENTAL` | offline non-Gaussian analysis に有用だが sample backend と同じ joint-factor 制約 |
| discrete `Qd` compatibility adapter | `OPTIONAL` | migration 専用。sample period と adaptation diagnostic を必須にし、canonical config は `Qc` のまま |
| automatic model selector | `PRUNE` | selector 誤りの uncertainty を表現できないため未実装。public registry に置かない |

package config と API docs は production `DEFAULT` を指定しない。実装は比較、
追加 corpus 収集、明示 opt-in のため保持する。

## 適用範囲と未実証事項

- corpus は synthetic/anonymous fixture の一つだけで、実 rosbag を含まない。
- moment backend の cross-time dependence は、表現不能な場合に
  `DEPENDENCE_APPROXIMATED` として安全側に診断する近似であり、厳密な joint
  posterior ではない。
- sample backend は marginal record 間の共有 calibration factor と独立 residual
  を分離できない。
- Grape adapter/plugin、controller と MCU firmware を結合した exact replay、
  Grape の独立 application corpus における calibration は評価していない。
- 標準 TF2 へ代表 pose を射影した経路では、この選定が必要とする distribution
  と model provenance を保持できない。

再実行手順と API safety contract は
[docs/temporal_evaluation.md](docs/temporal_evaluation.md) を参照する。
