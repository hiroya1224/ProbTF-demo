# MCMC sample による PID particle evaluation

## 1. 目的

PID 選択でいう plant particle は retained MCMC physical sample、PID particle は sample から導いた gain candidate である。
particle filter や時系列 residual state の再生ではない。
一つの source sample にだけ適合する gain を選ばず、candidate と posterior plant population の Cartesian product を full closed-loop simulation で評価する。

```text
candidate × plant sample × selected bag × discrepancy replicate
```

## 2. 入力 posterior

source は complete な `grape-param-estim/batch-estimation-run/v1` で、MCMC retained draw を含む必要がある。
各 plant sample は `sample_id`、mass、full inertia、CoG、force/torque effectiveness、delay、source mode ID を持つ。
retained draw は equal weight であり、component-wise mean plant や平均 gain へ潰さない。
linear/angular drag は static 18-D chart に含まれないため、PID request の `fixed_plant_parameters` で明示する。

## 3. exact sample-derived proposal

controller nominal allocation と各 physical sample の acceleration allocation を比較し、closed-loop generalized acceleration responseを得る。
`xy`、`z`、`roll_pitch`、`yaw` の各 coupled groupについて identity response に最も近い正の least-squares scale を一つ求める。
recorded current PID の各 P/I/D 成分に同じ group scale を掛け、sample-aligned exact proposal を作る。
gain を成分ごとに posterior 平均した candidate は自動生成しない。

current PID の正本は request の `baseline_bag_id` が指す rosbag controller snapshot である。
repository 内 YAML や別 bag の値へ暗黙に fallback しない。

## 4. candidate source

strict request は次の candidate source と population policy を受け付ける。

- `current` は baseline snapshot の 4 group × P/I/D で、必ず比較基準に含める。
- `sample-derived` は backend が全 retained MCMC sample から一対一で生成する exact raw proposal である。
- `user` は 4 × 3 の nonnegative exact gain を request に直接保存する。

request は raw sample-derived population を全件評価するか、log group-scale と delay 空間の deterministic k-medoids で指定上限へ絞るかを明示する。
GUI で選択中の source sample は `required_source_sample_ids` に入り、k-medoids による制限後も必ず exact candidate として残る。
backend の particle type は mutation generation と parent candidate provenance も表現できるが、現行 one-command request の直接指定 source は current と user であり、sample-derived candidate は backend が監査可能に生成する。
自動 refinement を追加する場合も bounded mutation と generation/parent ID を artifact に残し、最終候補を full posterior で再評価する。

## 5. counterfactual scenario

各 bag は estimation artifact と同じ selected interval、initial latent state、recorded reference/controller modeを使う。
plant static parameter と delay は各 MCMC sample の値へ置き換え、candidate PID で full closed-loop trajectory を最初から計算する。
観測 pose による途中 reset は行わない。
estimation 時に得た dynamics residual path を replay しない。

actuator model は estimation manifest の明示値をそのまま使い、PID worker 独自の hidden default を持たない。
bag path と SHA256、controller snapshot fingerprint が estimation artifact と一致しない場合は実行を拒否する。

## 6. model discrepancy policy

request は `zero_model_discrepancy` または `sample_model_discrepancy` を選ぶ。
前者は parameter-only counterfactual、後者は estimation run の最終 diagonal Q から future interval discrepancy を生成する posterior predictive counterfactual である。
過去の推定 residual path を未来へ繰り返す policy は存在しない。

sampled policy は estimation manifest の Q residual quantity と interval model を継承する。
seed は `base_seed`、sample ID、bag ID、replicate index から安定に導き、candidate ID を seed に含めない。
したがって同じ plant/bag/replicate では全 candidate が common random numbers を共有し、候補差と noise realization 差を混同しにくい。

## 7. plant subset と計算量

plant 側の `all_equal_weight_mcmc_samples` は全 retained sample、`explicit_equal_weight_mcmc_subset` は明示 sample ID の部分集合を使う。
candidate 側は全 retained sample の raw proposal を必ず先に生成し、`all_raw_mcmc_samples` または `deterministic_k_medoids` を別に選ぶ。
探索初期に subset を使った場合、最終 candidate は全 posterior sample で再評価してから推薦を判断する。
forecast 数は `candidate_count * sample_count * bag_count * replicate_count` であり、progress unit もこの Cartesian product を正本にする。
各 forecast は独立なので、この段階は process parallelization と cache の対象にしやすい。

## 8. metric

各 forecast は次を単位を保ったまま保存する。

| metric | unit/meaning |
|---|---|
| position RMSE | m |
| orientation RMSE | rad |
| maximum position error | m |
| maximum orientation error | rad |
| forecast completion | `[0,1]` |
| numerical failure count | count |
| actuator saturation duration | s |
| actuator saturation rate | `[0,1]` |

candidate ごとに equal-weight mean、configured quantile、upper CVaR を cost metric 別に計算する。
completion は lower quantile と lower CVaR を使い、低い tail を悪い結果として扱う。
gain change magnitude は current からの symmetric relative RMS とし、性能とは別の Pareto objective にする。
meter と radian を arbitrary weight で一つの scalar score に足さない。

## 9. recommendation policy

まず completion、failure、physical error、saturation tail、gain change の全 objective で Pareto non-dominated set を求める。
そのうち current PID に対して全 performance component を weakly improve し、少なくとも一つを strictly improve する current 以外の candidate だけを recommended とする。
条件を満たす candidate がなければ `recommendation_available=false` とし、`recommendation unavailable: no Pareto candidate improves current` を保存する。
operator が `selected_candidate_id` を指定しても、selection policy を満たさない candidate を安全な推薦へ昇格させない。

## 10. strict request と CLI

request schema は `grape-param-estim/pid-proposal-evaluation-request/v2` である。
主な必須 field は evaluation/output path、estimation run、resume、`forecast_workers`、baseline bag、bag path/SHA256、selected mode、fixed drag、model discrepancy policy/seed/replicates、plant subset、derived candidate population、current/user candidate、quantile/CVaR、selected candidate、maximum reference age である。

```bash
cd /home/leus/catkin_ws
source devel/setup.bash
rosrun grape_param_estim grape_evaluate_pid_proposals.py \
  --request /absolute/path/to/pid-proposal-evaluation-request.json
```

標準出力は strict JSONL progress、標準エラーは診断と完了 path である。
`forecast_workers` は `auto` または `1--32` の明示値で、BLAS thread を一つに制限した deterministic process pool が独立 forecast を実行する。
各完了 record は candidate、sample、bag、replicate の content key とともに atomic、pickle-free checkpoint へ保存する。
同一 request fingerprint の `resume=true` は checkpoint を検証し、未完了 record だけを再計算して canonical order に戻す。
request identity が異なる checkpoint、重複 record、異なる common-random seed は拒否する。

## 11. artifact

artifact schema は `grape-param-estim/pid-proposal-evaluation/v2` である。

```text
pid_proposal_evaluation/
  manifest.json
  source_samples.npz
  candidate_particles.npz
  summary.npz
  proposed_GimbalrotorControl.yaml
  proposed_GimbalrotorControl.diff.yaml
  bags/<bag_id>.npz
```

manifest は estimation/request fingerprint、Q policy/quantity/interval model/seed/replicates、plant sample subset、raw/evaluated derived candidate 件数、candidate population method/上限/必須 source sample、bag/candidate ID、selection policy、Pareto/recommended ID、recommendation availability、rejection reason、selected candidateを保存する。
`source_samples.npz` は MCMC physical sample に加えて、全 sample の exact group scale、gain、6 × 6 acceleration response を sample ID と同じ order で保存する。
`candidate_particles.npz` は実際に評価した source/generation/parent/gain、`summary.npz` は robust metric、各 bag NPZ は candidate/sample/replicate/seed ごとの metric record を保存する。
loader は sample-derived candidate の gain が raw proposal と bit-exact に一致し、k-medoids 上限と必須 sample が manifest と一致することを検証する。
YAML と diff は exact selected gain の提案物であり、`jsk_aerial_robot`、controller YAML、dynamic_reconfigure を自動変更しない。

## 12. 現在の validation 状態

2026 年 8 月 4 日の `18.0--24.0 s` 実 bag run は `estimate_only` で MCMC を実行していないため、この run を source にした実データ PID recommendation は存在しない。
backend の candidate × sample × bag × replicate、sampled-Q common-random-number、strict request/artifact 経路は synthetic/test fixture で検証している。
実機向け評価へ進むには、まず covariance/actuator calibration、複数回 EM、delay/ridge stability、multiple-chain R-hat/ESS を満たす MCMC run が必要である。
成功 bag は現在 data split 候補として既に閲覧しているため、これを使った candidate tuning を外部 hold-out validation と報告しない。

## 13. code 対応

| 責務 | module |
|---|---|
| posterior/sample proposal | `pid/proposal.py` |
| batch artifact adapter | `pid/input.py` |
| predictive closed-loop rollout | `pid/predictive.py` |
| Cartesian evaluation/Q sampling | `pid/particle_search.py` |
| robust metric/Pareto/recommendation | `pid/metrics.py` |
| strict request/worker | `pid/request.py`, `pid/cli.py` |
| strict artifact/YAML | `pid/artifact.py` |

PID particle evaluation は posterior uncertainty を制御判断へ伝える道具であり、flight safety certification ではない。
completion、tail error、saturation、numerical failure、data split を人間が確認し、実機への適用は別の承認手順で行う。
