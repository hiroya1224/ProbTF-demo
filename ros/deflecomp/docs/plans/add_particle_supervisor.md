# add_particle_supervisor.md

## 目的

現在の `MultiFrameStiffnessWEKF` を主推定器として維持しつつ，剛性パラメータ `K` の局所解への固着を検出・補正するために，低頻度で動く deterministic particle scan supervisor を追加する。

今回の目的は，まずは，既存の WEKF 更新後に，`log(K)` 空間の候補点を決定論的に評価し，現在の `K_est` より明確に良い候補が見つかった場合だけ `K_est` を補正する。

既存設計では，推定値 `K_est` と実行値 `K_exec` が分離されており，`K_exec` は `K_est` に一次遅れで追従する。そのため，particle scan が `K_est` をある程度急に補正しても，実行側への反映は既存の一次遅れ機構で緩和される。この設計は維持する。

---

## 基本方針

### 追加するもの

- `log(K)` 空間における deterministic particle scan
- 観測履歴 window の保存
- 候補 `log(K)` に対する exact Bingham log-likelihood 評価
- WEKF の共分散が十分小さい成分だけを対象とする active dimension selection
- 明確な score 改善があった場合だけ `K_est` を補正
- 補正後に `P_est` の該当成分を少し膨らませる処理
- debug 情報の追加
- unit test の追加

### 今回は追加しないもの

- 各 particle に EKF を持たせる RBPF
- resampling
- stochastic particle sampling
- background thread / async node
- 全次元 tensor grid
- particle weighted mean による `K_est` 更新
- 候補 `K` ごとの `theta_cmd` 再生成

今回の supervisor は，あくまで

```text
WEKF + deterministic particle MAP correction
```

として実装する。

---

## 既存コード上の接続点

現在，剛性推定は主に以下で行われている。

- `deflecomp_core/src/deflecomp_core/estimator/stiffness_wekf.py`
  - `MultiFrameStiffnessWEKF`
  - `update_with_multi(...)`
  - 内部状態: `x_est`, `P_est`, `last_theta_eq`

- `deflecomp_core/src/deflecomp_core/pipeline/compensator.py`
  - `DeflectionCompensator.step(...)`
  - `theta_cmd_sent = self.last_theta_cmd.copy()`
  - `a_map = self.observation_builder.build_A_map(imu_observations)`
  - `self.stiffness_estimator.update_with_multi(...)`
  - `self._update_exec_stiffness_target(...)`
  - `self._smooth_exec_stiffness(...)`

particle scan は，`update_with_multi(...)` の直後，`_update_exec_stiffness_target(...)` の前に差し込む。

重要な順序は次の通り。

```text
1. WEKF update_with_multi を実行する
2. 今回の観測 record を particle supervisor に追加する
3. 必要なら particle scan を実行する
4. scan が accepted なら stiffness_estimator.x_est / P_est / last_theta_eq を補正する
5. 補正後の x_est を使って _update_exec_stiffness_target を呼ぶ
6. 既存通り K_exec は一次遅れで追従する
```

---

## 数学的な意味

履歴 window を `W` とする。各 record は，実際に送信された指令 `theta_cmd_sent` と，IMU 観測から構成された `A_map` を持つ。

候補 `x = log(K)` に対して，

```text
K = exp(x)
theta_eq = solve(theta_cmd_sent, K)
ell_s(x) = sum_f z_f(theta_eq)^T A_f z_f(theta_eq)
L_W(x) = sum_{s in W} ell_s(x)
```

を評価する。

WEKF はこの likelihood を局所二次近似して更新している。particle supervisor は，その局所近似に頼らず，候補点を直接評価する役割を持つ。

採用する候補は MAP とする。

```text
x_best = argmax_x L_W(x)
```

ただし，採用条件を満たす場合だけ `x_est` を `x_best` に置き換える。

---

## 実装手順

## 1. `MultiFrameStiffnessWEKF` に exact likelihood 評価を追加

対象ファイル:

```text
deflecomp_core/src/deflecomp_core/estimator/stiffness_wekf.py
```

### 1.1 追加する dataclass

必要なら次を追加する。

```python
@dataclass
class StiffnessLikelihoodEvaluation:
    log_likelihood: float
    theta_eq: np.ndarray
    kp_vec: np.ndarray
    x_eval: np.ndarray
    valid: bool
    error: Optional[str]
```

既存の `StiffnessUpdateResult` の近くに置く。

### 1.2 追加するメソッド

`MultiFrameStiffnessWEKF` に次を追加する。

```python
def evaluate_log_likelihood_at_x(
    self,
    x_eval: np.ndarray,
    theta_cmd_sent: np.ndarray,
    A_map: Dict[int, np.ndarray],
    theta_init_eq_pred: Optional[np.ndarray],
    kp_lim: Optional[Tuple[float, float]] = None,
) -> StiffnessLikelihoodEvaluation:
    ...
```

要件:

- `self.x_est` と `self.P_est` を変更しない。
- `kp_lim` が与えられたら `x_eval` を `log(kp_min)` と `log(kp_max)` で clip する。
- `kp_vec = np.exp(x_eval)` を使う。
- `self.solver.solve(theta_cmd=theta_cmd_sent, kp_vec=kp_vec, theta_init=theta_init_eq_pred)` を呼ぶ。
- 各 frame について，既存実装と同じく

```python
A_sym = 0.5 * (A_f + A_f.T)
z_f = self.robot.frame_quaternion_wxyz_base(theta_eq, fid)
ell_f = float(z_f.T @ (A_sym @ z_f))
```

を足し合わせる。

- 例外や非有限値が出た場合は，`valid=False`, `log_likelihood=-np.inf`, `error=<reason>` として返す。
- 成功時は `valid=True`, `error=None` とする。

### 1.3 推定値補正用メソッドを追加

外部から直接 `x_est` / `P_est` を書き換えるより，専用メソッドにまとめる。

```python
def apply_particle_correction(
    self,
    x_new: np.ndarray,
    active_indices: np.ndarray,
    reset_std: float,
    theta_eq: Optional[np.ndarray] = None,
    kp_lim: Optional[Tuple[float, float]] = None,
) -> None:
    ...
```

要件:

- `x_new` を `kp_lim` で clip して `self.x_est` に入れる。
- `active_indices` に含まれる成分について，

```python
P_est[j, j] = max(P_est[j, j], reset_std ** 2)
```

とする。

- `P_est` は最後に対称化する。
- `theta_eq` が `None` でなければ `self.last_theta_eq = theta_eq.copy()` とする。
- `last_update_step`, `last_update_norm`, `last_update_applied`, `last_debug` は，最低限破綻しない値に更新する。
  - 例: `last_update_step = x_new - x_old`
  - 例: `last_update_applied = True`

---

## 2. particle supervisor を追加

新規ファイル:

```text
deflecomp_core/src/deflecomp_core/estimator/stiffness_particle_supervisor.py
```

### 2.1 追加する dataclass

```python
@dataclass
class StiffnessParticleRecord:
    theta_cmd_sent: np.ndarray
    A_map: Dict[int, np.ndarray]
    theta_init_eq_pred: Optional[np.ndarray]
    stamp: Optional[float]
```

`A_map` は後で変更されないよう，record 作成時に各行列を copy する。

```python
@dataclass
class StiffnessParticleScanConfig:
    enabled: bool = False
    window_size: int = 20
    period: int = 5
    grid_size: int = 21
    max_active_dims: int = 2
    std_trigger: float = 0.15
    info_abs: float = 1.0e-8
    min_gain_per_obs: float = 1.0
    min_log_jump: float = 0.05
    reset_std: float = 0.10
    cooldown: int = 20
    mode: str = "axis"
```

```python
@dataclass
class StiffnessParticleScanResult:
    attempted: bool
    accepted: bool
    reason: str
    x_current: Optional[np.ndarray]
    x_best: Optional[np.ndarray]
    score_current: float
    score_best: float
    gain_per_obs: float
    active_indices: np.ndarray
    candidate_count: int
    theta_eq_best: Optional[np.ndarray]
    debug: Dict[str, Any]
```

### 2.2 追加するクラス

```python
class StiffnessParticleScanSupervisor:
    def __init__(self, config: StiffnessParticleScanConfig) -> None:
        ...
```

内部状態:

```python
self.config
self.records
self.step_count
self.cooldown_count
self.last_result
```

`records` は `collections.deque(maxlen=config.window_size)` を使う。

### 2.3 record 追加メソッド

```python
def add_record(
    self,
    theta_cmd_sent: np.ndarray,
    A_map: Dict[int, np.ndarray],
    theta_init_eq_pred: Optional[np.ndarray],
    stamp: Optional[float],
) -> None:
    ...
```

要件:

- `theta_cmd_sent` を copy する。
- `theta_init_eq_pred` は `None` なら `None`，そうでなければ copy する。
- `A_map` の key はそのまま，value は `np.asarray(...).copy()` する。

### 2.4 active dimension selection

```python
def _active_indices(
    self,
    P_est: np.ndarray,
    information: Optional[np.ndarray],
) -> np.ndarray:
    ...
```

基本条件:

```text
sqrt(P_est[j, j]) <= std_trigger
```

かつ，`information` があれば

```text
abs(information[j, j]) >= info_abs
```

を満たす成分のみ active とする。

候補が `max_active_dims` より多い場合は，`abs(information[j, j])` が大きい順に絞る。`information` がない場合は，`P_est[j, j]` が小さい順に絞る。

ここでの狙いは，「まだ分散が広い成分」ではなく，「分散が狭く，観測に効いている成分」を particle scan の対象にすることである。

### 2.5 candidate 生成

最初は `mode == "axis"` のみ実装する。

```python
def _make_axis_candidates(
    self,
    x_current: np.ndarray,
    active_indices: np.ndarray,
    kp_lim: Tuple[float, float],
) -> List[np.ndarray]:
    ...
```

要件:

- `x_current` 自身を候補に含める。
- `grid = np.linspace(log(kp_min), log(kp_max), grid_size)` を使う。
- 各 active dimension `j` について，`x_current` の `j` 成分だけを grid 値に置き換えた候補を作る。
- `kp_lim` は必須扱いでよい。`kp_lim` がない場合は scan を skip する。
- 重複候補は取り除く。

候補数の目安は以下。

```text
1 + len(active_indices) * grid_size
```

全次元 tensor grid は実装しない。

### 2.6 window score 評価

```python
def _score_candidate(
    self,
    estimator: MultiFrameStiffnessWEKF,
    x_candidate: np.ndarray,
    kp_lim: Tuple[float, float],
) -> Tuple[float, Optional[np.ndarray], Dict[str, Any]]:
    ...
```

要件:

- 保存された `records` 全体に対して `evaluate_log_likelihood_at_x(...)` を呼ぶ。
- score は各 record の `log_likelihood` の和とする。
- 途中で invalid が出た場合は，その candidate の score を `-np.inf` とする。
- `theta_eq_best` として使えるよう，最後の record に対する `theta_eq` を返す。

### 2.7 correction 判定

```python
def maybe_scan(
    self,
    estimator: MultiFrameStiffnessWEKF,
    latest_information: Optional[np.ndarray],
    kp_lim: Optional[Tuple[float, float]],
) -> StiffnessParticleScanResult:
    ...
```

skip 条件:

- `enabled == False`
- `len(records) < window_size`
- `cooldown_count > 0`
- `step_count % period != 0`
- `kp_lim is None`
- active dimension が空
- candidate が作れない
- current score が非有限
- best score が非有限

採用条件:

```text
gain_per_obs = (score_best - score_current) / len(records)
max_jump = max(abs(x_best - x_current))
```

以下をすべて満たす場合だけ accepted とする。

```text
gain_per_obs >= min_gain_per_obs
max_jump >= min_log_jump
```

accepted のとき:

- `cooldown_count = cooldown`
- `x_best`, `active_indices`, `theta_eq_best` を result に入れる。

not accepted のとき:

- estimator は変更しない。

注意:

- `maybe_scan(...)` 自体は estimator を変更しない方針にする。
- estimator の補正は `DeflectionCompensator` 側で `apply_particle_correction(...)` を呼んで行う。
- これにより，debug や target 更新の順序を `DeflectionCompensator.step(...)` で明示できる。

---

## 3. `DeflectionCompensator` に組み込む

対象ファイル:

```text
deflecomp_core/src/deflecomp_core/pipeline/compensator.py
```

### 3.1 import 追加

```python
from deflecomp_core.estimator.stiffness_particle_supervisor import (
    StiffnessParticleScanConfig,
    StiffnessParticleScanSupervisor,
)
```

### 3.2 `__init__` で supervisor を作る

`self.config` 設定後に追加する。

```python
self.stiffness_particle_supervisor = None
if _as_bool(self.config.get("particle_scan_enabled", False)):
    particle_config = StiffnessParticleScanConfig(
        enabled=True,
        window_size=int(self.config.get("particle_scan_window_size", 20)),
        period=int(self.config.get("particle_scan_period", 5)),
        grid_size=int(self.config.get("particle_scan_grid_size", 21)),
        max_active_dims=int(self.config.get("particle_scan_max_active_dims", 2)),
        std_trigger=float(self.config.get("particle_scan_std_trigger", 0.15)),
        info_abs=float(self.config.get("particle_scan_info_abs", 1.0e-8)),
        min_gain_per_obs=float(self.config.get("particle_scan_min_gain_per_obs", 1.0)),
        min_log_jump=float(self.config.get("particle_scan_min_log_jump", 0.05)),
        reset_std=float(self.config.get("particle_scan_reset_std", 0.10)),
        cooldown=int(self.config.get("particle_scan_cooldown", 20)),
        mode=str(self.config.get("particle_scan_mode", "axis")),
    )
    self.stiffness_particle_supervisor = StiffnessParticleScanSupervisor(particle_config)
```

### 3.3 `step(...)` の WEKF 更新後に scan を入れる

現在の流れでは，以下の直後に入れる。

```python
update_result = self.stiffness_estimator.update_with_multi(...)
```

現在はすぐに

```python
self._update_exec_stiffness_target(update_result.x_est)
```

しているが，ここを次のように変更する。

```python
if self.stiffness_particle_supervisor is not None:
    self.stiffness_particle_supervisor.add_record(
        theta_cmd_sent=theta_cmd_sent,
        A_map=a_map,
        theta_init_eq_pred=theta_init,
        stamp=observation_stamp,
    )
    scan_result = self.stiffness_particle_supervisor.maybe_scan(
        estimator=self.stiffness_estimator,
        latest_information=update_result.information,
        kp_lim=kp_lim,
    )
    debug["particle_scan"] = scan_result.debug
    debug["particle_scan_attempted"] = scan_result.attempted
    debug["particle_scan_accepted"] = scan_result.accepted
    debug["particle_scan_reason"] = scan_result.reason
    debug["particle_scan_gain_per_obs"] = scan_result.gain_per_obs
    debug["particle_scan_candidate_count"] = scan_result.candidate_count
    debug["particle_scan_active_indices"] = scan_result.active_indices.copy()

    if scan_result.accepted:
        self.stiffness_estimator.apply_particle_correction(
            x_new=scan_result.x_best,
            active_indices=scan_result.active_indices,
            reset_std=self.stiffness_particle_supervisor.config.reset_std,
            theta_eq=scan_result.theta_eq_best,
            kp_lim=kp_lim,
        )

self._update_exec_stiffness_target(self.stiffness_estimator.x_est)
```

要点:

- `_update_exec_stiffness_target(...)` は scan 補正後に一回だけ呼ぶ。
- scan は `K_est` を補正するだけで，`K_exec` には直接触れない。
- `K_exec` は既存の `_smooth_exec_stiffness(...)` に任せる。

---

## 4. ROS config に parameter を追加

対象ファイル:

```text
deflecomp_ros/config/estimator.yaml
```

以下を末尾に追加する。

```yaml
# Deterministic particle scan supervisor for stiffness estimation.
# This is a low-frequency global correction for K_est, not a full RBPF.
particle_scan_enabled: false
particle_scan_window_size: 20
particle_scan_period: 5
particle_scan_grid_size: 21
particle_scan_mode: axis
particle_scan_max_active_dims: 2
particle_scan_std_trigger: 0.15
particle_scan_info_abs: 1.0e-8
particle_scan_min_gain_per_obs: 1.0
particle_scan_min_log_jump: 0.05
particle_scan_reset_std: 0.10
particle_scan_cooldown: 20
```

デフォルトは `false` にする。既存挙動を壊さないためである。

---

## 5. ROS node に parameter を通す

対象ファイル:

```text
deflecomp_ros/nodes/deflecomp_node.py
```

現在 `DeflectionCompensator(config={...})` に渡している辞書に，particle scan 関連 parameter を追加する。

`main()` では，次の param を読む。

```python
particle_scan_enabled = parse_bool(rospy.get_param("~particle_scan_enabled", False))
particle_scan_window_size = int(rospy.get_param("~particle_scan_window_size", 20))
particle_scan_period = int(rospy.get_param("~particle_scan_period", 5))
particle_scan_grid_size = int(rospy.get_param("~particle_scan_grid_size", 21))
particle_scan_mode = str(rospy.get_param("~particle_scan_mode", "axis"))
particle_scan_max_active_dims = int(rospy.get_param("~particle_scan_max_active_dims", 2))
particle_scan_std_trigger = float(rospy.get_param("~particle_scan_std_trigger", 0.15))
particle_scan_info_abs = float(rospy.get_param("~particle_scan_info_abs", 1.0e-8))
particle_scan_min_gain_per_obs = float(rospy.get_param("~particle_scan_min_gain_per_obs", 1.0))
particle_scan_min_log_jump = float(rospy.get_param("~particle_scan_min_log_jump", 0.05))
particle_scan_reset_std = float(rospy.get_param("~particle_scan_reset_std", 0.10))
particle_scan_cooldown = int(rospy.get_param("~particle_scan_cooldown", 20))
```

`DeflecompNode.__init__(...)` の引数に追加して，`DeflectionCompensator(config={...})` に渡す。

ただし，実装変更を小さくしたい場合は，この段階では `DeflectionCompensator` の `config` に直接書くところまででよい。ROS parameter の全追加は second step に回してもよい。

---

## 6. debug 出力

`DeflectionCompensator.step(...)` の `debug` に最低限以下を入れる。

```text
particle_scan_attempted
particle_scan_accepted
particle_scan_reason
particle_scan_gain_per_obs
particle_scan_candidate_count
particle_scan_active_indices
particle_scan_score_current
particle_scan_score_best
particle_scan_max_jump
```

`scan_result.debug` 内には，必要に応じて以下を入れる。

```text
window_size
step_count
cooldown_count
active_indices
candidate_count
score_current
score_best
gain_per_obs
max_jump
x_current
x_best
kp_current
kp_best
```

数値は ndarray のままでもよいが，ROS debug publisher が配列前提なら，既存の `/deflecomp/debug` へ全部を載せる必要はない。まずは Python debug dict に残ればよい。

---

## 7. unit test を追加

対象ファイル候補:

```text
deflecomp_core/test/test_stiffness_particle_supervisor.py
```

### 7.1 likelihood 評価が estimator 状態を変えないこと

テスト内容:

- fake solver / fake robot / fake sensitivity を使う。
- `evaluate_log_likelihood_at_x(...)` を呼ぶ。
- 呼び出し前後で `x_est`, `P_est`, `last_theta_eq` が変わらないことを確認する。

### 7.2 active dimension selection

テスト内容:

- `P_est = diag([0.01, 1.0, 0.0025])` のような値を使う。
- `std_trigger` を設定する。
- `information` の diag で active 成分が絞られることを確認する。
- `max_active_dims` が効くことを確認する。

### 7.3 axis candidate 生成

テスト内容:

- `x_current = log([10, 20, 30])`
- active index を `[0, 2]`
- `kp_lim = (1.0, 100.0)`
- `grid_size = 5`
- 候補数が `1 + len(active_indices) * grid_size` 以下であることを確認する。
- すべての候補が `log(kp_min)` と `log(kp_max)` の範囲に入ることを確認する。
- current が含まれることを確認する。

### 7.4 scan が改善を検出して accepted になること

テスト内容:

- fake estimator の `evaluate_log_likelihood_at_x(...)` を，ある target `x_true` に近いほど score が高くなる関数に差し替える。
- record を `window_size` 個追加する。
- current と離れた grid 点に `x_true` がある状況を作る。
- `maybe_scan(...)` が `accepted=True` を返すことを確認する。

### 7.5 gain が小さい場合は accepted にならないこと

テスト内容:

- best score と current score の差が `min_gain_per_obs` 未満になるようにする。
- `accepted=False` を確認する。

### 7.6 `DeflectionCompensator` との結合テスト

既存の

```text
deflecomp_core/test/test_compensator_k_exec.py
```

に追加，または新規 test file を作る。

確認すること:

- particle scan が accepted になったとき，`stiffness_estimator.x_est` は変わる。
- `x_exec` はその step で直接 `x_est` に一致しない。
- `x_exec_target` は補正後の `x_est` に一致する。
- つまり，`K_est` の correction と `K_exec` の一次遅れ分離が保たれている。

---

## 8. 実装上の注意

### 8.1 `theta_cmd_sent` は絶対に固定入力として扱う

particle 候補ごとに `theta_cmd` を再生成してはいけない。

過去の観測は，実際に送った `theta_cmd_sent` に対する結果である。したがって，score 評価では，候補 `K` に対して

```text
theta_eq = solve(theta_cmd_sent, K)
```

を計算する。

### 8.2 `K_exec` には直接触れない

particle supervisor が変更するのは `K_est` のみ。

`K_exec` は既存の

```python
_update_exec_stiffness_target(...)
_smooth_exec_stiffness(...)
```

の流れに任せる。

### 8.3 `P_est` は補正後に少し膨らませる

補正後の `P_est` をそのままにすると，補正先でも過信したままになる。active 成分については `reset_std` を下限として入れる。

### 8.4 score は window 長で正規化して採用判定する

window size を変えても threshold の意味が壊れないよう，採用判定には

```text
gain_per_obs = (score_best - score_current) / window_size
```

を使う。

### 8.5 最初は MAP のみ

weighted mean は使わない。多峰的な場合，平均がどの候補にも対応しない不自然な `K` になる可能性がある。

### 8.6 既存挙動を壊さない

`particle_scan_enabled` の default は `false` とし，無効時には現在の挙動と一致させる。

---

## 9. 完了条件

以下を満たせば完了とする。

- `particle_scan_enabled: false` のとき，既存 test が通る。
- `MultiFrameStiffnessWEKF.evaluate_log_likelihood_at_x(...)` が追加され，状態を変更しない。
- `StiffnessParticleScanSupervisor` が追加されている。
- `DeflectionCompensator.step(...)` で，WEKF update 後かつ `_update_exec_stiffness_target(...)` 前に scan が入っている。
- scan accepted 時に `x_est` が補正される。
- scan accepted 時にも `x_exec` は直接ジャンプせず，既存の一次遅れで追従する。
- particle scan 関連 debug key が `debug` に入る。
- unit test が追加されている。

---

## 10. 将来拡張のメモ

今回の実装は完全な RBPF ではない。ただし，将来的には以下へ拡張できる。

```text
p(K, xi | D) ~= sum_i w_i delta_{K_i}(K) N(xi; m_i, P_i)
```

その場合は，各 particle が局所状態 `xi` の EKF を持つ。ただし，今回はそこまで実装しない。

今回の位置づけは，

```text
RBPF に発展可能な low-frequency particle supervisor
```

である。
