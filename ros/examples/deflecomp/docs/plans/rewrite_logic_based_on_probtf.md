# plan.md: WEKF / 局所 Laplace 近似への整理

## 0. 目的

現在の `deflecomp` の剛性推定部を，粒子法・UKF・EnKF には広げず，既存の `MultiFrameStiffnessWEKF` を **WEKF / 局所 Laplace 近似** として明確に整理する．

今回の作業で達成したいことは次の 3 点である．

1. `x = log K` に対する prior Gaussian と，IMU Bingham likelihood の局所 2 次近似から posterior Gaussian を作る，という形に実装を整理する．
2. 既存の平衡ソルバ呼び出し回数を増やさない．基本的に 1 update あたり 1 回の equilibrium solve に留める．
3. 既存の観測可能部分空間への射影，情報量 cap，更新 gain，step cap，`kp_min` / `kp_max` は維持し，挙動を大きく変えずに理論上の意味づけを明確にする．

## 1. 今回やらないこと

以下は今回の変更対象に含めない．

- particle filter の導入．
- UKF / EnKF の導入．
- sigma point や ensemble member ごとの equilibrium solve．
- IMU Bingham の fiber 方向正則化の本格実装．
- ProbTF 全体のクラス設計．
- ROS topic 名や message 型の変更．
- simulator の高速化．
- feedforward projection の設計変更．

今回の目的は，既存 WEKF を「IMU Bingham likelihood の局所 Laplace 近似」として書き直すことだけである．

## 2. 数式上の整理

状態は現在と同じく

```math
x = \log K
```

とする．時刻更新後の prior を

```math
x \sim \mathcal{N}(m^-, P^-)
```

と書く．実装では，現在の `self.x` が `m^-` に対応し，`self.P + Q` が `P^-` に対応する．

現在の平均 `m^-` に対して

```math
K^- = \exp(m^-)
```

を作り，現在の `theta_cmd` と `K^-` から平衡姿勢を 1 回だけ解く．

```math
\theta_{\rm eq}^- = \Theta_{\rm eq}(\theta_{\rm cmd}; K^-)
```

各 IMU frame `f` について，予測 quaternion を

```math
z_f^- = z_f(\theta_{\rm eq}^-)
```

とする．IMU 観測から得られた Bingham 行列を `A_f` とすると，観測 likelihood の対数は

```math
\ell(x)
=
\sum_f
z_f(x)^\top A_f z_f(x)
```

である．これを `m^-` のまわりで

```math
\ell(x)
\simeq
\ell(m^-)
+
g^\top (x - m^-)
-
\frac{1}{2}
(x - m^-)^\top
\mathcal{I}
(x - m^-)
```

と 2 次近似する．ここで

- `g` は `x = log K` に関する log-likelihood gradient．
- `I` または `info` は `- Hessian ell` に対応する局所情報行列．
- `info` は理想的には半正定値であり，観測されない方向では固有値が 0 または非常に小さい．

posterior は

```math
P^+ = \left((P^-)^{-1} + s\mathcal{I}\right)^{-1}
```

```math
m^+ = m^- + P^+ (s g)
```

で近似する．ここで `s` は既存の `stiffness_update_gain` と `measurement_info_eig_cap` による測定情報量のスケールである．

重要：`A_f` は負半定値に近い Bingham 行列なので，`z^T A_f z` は通常 concave な likelihood になる．したがって，実装内では必ず符号を確認し，`info = - Hessian ell` が観測方向で非負になるようにする．

## 3. 既存実装との対応

現在の `MultiFrameStiffnessWEKF` には，すでに以下の構成要素がある．

- `x = log K` を状態にする．
- `K = exp(x)` から equilibrium solve を行う．
- `J_q` と `J_x` から平衡感度を作る．
- frame ごとに `A_f`，`z_f`，`Q_z`，`J_omega_f` を使う．
- gradient と information を集約する．
- information の固有分解により観測可能部分空間だけを更新する．
- `max_log_kp_step`，`kp_min`，`kp_max` で更新を制限する．

今回の変更では，これらを壊さず，次のように意味づけを変える．

- `A_f` は「観測から K の分布を作るもの」ではなく，「予測 quaternion `z_f(x)` を評価する Bingham log-likelihood」として扱う．
- `g` は local Laplace 近似における `grad ell(m^-)` として扱う．
- `mathcal{I}` は `- Hessian ell(m^-)` として扱う．
- WEKF update は Gaussian prior と局所 2 次 likelihood から得られる Gaussian posterior として扱う．

## 4. 変更対象ファイル

主な変更対象は以下に限定する．

- `deflecomp_core/src/deflecomp_core/estimator/stiffness_wekf.py`
- 必要なら `deflecomp_core/src/deflecomp_core/observation/bingham.py`
- 必要なら `deflecomp_core/src/deflecomp_core/observation/imu_observation.py`
- テストが存在する場合は estimator / observation 周辺の test file
- 可能なら概要ドキュメントまたは README の該当箇所

以下は原則として変更しない．

- `deflecomp_core/src/deflecomp_core/model/equilibrium.py`
- `deflecomp_core/src/deflecomp_core/model/sensitivity.py`
- `deflecomp_core/src/deflecomp_core/pipeline/compensator.py`
- ROS node
- simulator
- URDF / launch files

ただし，既存 API との整合のために小さな rename や debug dict の追加が必要な場合は許容する．

## 5. 実装方針

### 5.1 `stiffness_wekf.py` に局所 Laplace 計算を分離する

`MultiFrameStiffnessWEKF` の中で，frame ごとの Bingham 項から `g` と `info` を作る処理を，明示的な helper method に分離する．

候補名：

```python
_compute_local_laplace_terms(...)
```

戻り値の例：

```python
return {
    "log_likelihood": ell,
    "gradient": grad,
    "information": info,
    "frame_terms": frame_terms,
}
```

この helper は次を満たすこと．

- 入力は，現在の `theta_eq`，`K`，`J_q`，`J_x`，IMU frame ごとの `A_f` など，既存 `update_with_multi` で計算済みの量を使う．
- 追加の equilibrium solve を呼ばない．
- `gradient` は `x = log K` に関する `ell(x)` の一次項である．
- `information` は `- Hessian ell(x)` の近似である．
- `information` は最後に必ず対称化する．

対称化は次の形で行う．

```python
info = 0.5 * (info + info.T)
```

ただし，既存実装の内部変数が `H0` を使っている場合は，`info = -0.5 * (H0 + H0.T)` の形を維持してもよい．その場合も，コメントで `info = - Hessian ell` であることを明記する．

### 5.2 符号と係数を finite difference で検証する

今回の最重要点は，`gradient` と `information` の符号・係数を間違えないことである．

小さなテスト用関数を追加する．

対象の scalar 関数は

```python
ell(x) = sum(z_f(theta_eq(x)).T @ A_f @ z_f(theta_eq(x)))
```

である．ただし unit test では equilibrium solve を毎回呼ぶと重いので，まずは以下の 2 段階で検証する．

1. `theta_eq` と `J_theta_x` を固定した局所線形モデルで，analytic gradient / information と finite difference を比較する．
2. 余裕があれば，小さいモデル・少数 frame・少数方向だけで，実際に equilibrium solve を含む finite difference を行う．これは重いので slow test 扱いでよい．

finite difference では，ランダム方向 `d` に対して

```math
\frac{\ell(m + \epsilon d) - \ell(m - \epsilon d)}{2\epsilon}
```

が

```math
g^\top d
```

に近いことを確認する．

また，2 次差分

```math
\frac{\ell(m + \epsilon d) - 2\ell(m) + \ell(m - \epsilon d)}{\epsilon^2}
```

が

```math
- d^\top \mathcal{I} d
```

に近いことを確認する．

### 5.3 posterior update を明示的に Laplace 形式にする

`update_with_multi` の中で，以下の構造が見えるように整理する．

1. predict covariance

```python
P_pred = self.P + self.Q
```

2. current mean

```python
x_pred = self.x.copy()
```

3. current stiffness

```python
K_pred = np.exp(x_pred)
```

4. equilibrium solve

```python
theta_eq = solve_equilibrium(theta_cmd, K_pred)
```

5. local Laplace terms

```python
terms = self._compute_local_laplace_terms(...)
grad = terms["gradient"]
info = terms["information"]
```

6. observability projection

```python
info = 0.5 * (info + info.T)
eigvals, eigvecs = np.linalg.eigh(info)
```

7. observed eigen-directions only

```python
obs_mask = eigvals > max(observability_abs, observability_rcond * eigvals.max())
```

8. posterior covariance and update in observed subspace

```python
P_obs = U_obs.T @ P_pred @ U_obs
g_obs = U_obs.T @ grad
P_post_obs = inv(inv(P_obs) + scaled_info_obs + jitter * I)
dx_obs = P_post_obs @ scaled_g_obs
dx = U_obs @ dx_obs
```

9. damping / step cap / bounds

```python
dx = clip_by_max_log_kp_step(dx)
self.x = clip_log_kp(x_pred + dx)
```

10. covariance update

観測可能部分空間の covariance を posterior に置き換え，不可観測部分空間は prior を維持する．既存実装にこの処理がすでにある場合は，基本的に維持する．

### 5.4 negative eigenvalue の扱い

数値誤差や線形化誤差により，`info` に小さな負固有値が出る可能性がある．

方針：

- 小さな負固有値は 0 に clip する．
- 大きな負固有値が出た場合は，その update を危険とみなし，debug warning を出す．
- 大きな負固有値の閾値は，まず `-1e-9` から `-1e-8` 程度を仮置きでよい．

例：

```python
if eigvals.min() < -negative_info_tol:
    debug["laplace_info_has_large_negative_eig"] = True

eigvals = np.maximum(eigvals, 0.0)
```

### 5.5 debug dict を追加する

レビューしやすいように，少なくとも次を debug dict に入れる．

- `laplace_log_likelihood`
- `laplace_grad_norm`
- `laplace_info_eigs`
- `laplace_obs_rank`
- `laplace_dx_norm`
- `laplace_dx_max_abs`
- `laplace_info_negative_min_eig`
- `laplace_update_skipped_reason` if skipped

既存の debug key と衝突しないように，prefix は `laplace_` に統一する．

## 6. 観測可能性射影の扱い

今回も，重力方向 IMU では見えない方向を無理に更新しない．

したがって，`info` の固有分解によって観測可能方向を決める現在の処理は残す．

```math
\mathcal{I} = U\Lambda U^\top
```

```math
\lambda_i > \max(\lambda_{\rm abs}, r_{\rm cond}\lambda_{\max})
```

を満たす方向だけを更新する．

注意：`info` がすべて 0 に近い場合は update を skip する．このとき covariance には process noise だけを反映する．

## 7. Bingham 行列の意味づけの修正

`observation/bingham.py` またはその docstring で，`A_f` の意味を以下のように書き換える．

- `A_f` は `K` の分布を直接作るものではない．
- `A_f` は，予測 quaternion `z_f(theta_eq; K)` に対する log-likelihood

```math
z_f^\top A_f z_f
```

を与える．

- `K` の posterior は，prior `p(x)` と likelihood `exp(z_f^T A_f z_f)` の積を局所 Laplace 近似することで得る．

このコメント修正は重要である．今後 ProbTF に接続するとき，`A_f` は「観測 ProbTF」または「観測 likelihood」であり，`K` 分布そのものではない，という整理になるためである．

## 8. 既存パラメータの扱い

既存パラメータの意味はできるだけ維持する．

- `q_proc`: prior covariance に加える process noise．
- `observability_rcond`: `info` 固有値による相対閾値．
- `observability_abs`: `info` 固有値による絶対閾値．
- `measurement_info_eig_cap`: 1 update で測定情報量が強くなりすぎないようにする cap．
- `stiffness_update_gain`: likelihood 情報量と gradient に掛ける gain．
- `max_log_kp_step`: 1 cycle の `log K` 更新量 cap．
- `kp_min`, `kp_max`: `K` の上下限．

今回，新しいパラメータは原則として増やさない．どうしても必要なら，以下程度に留める．

- `laplace_negative_info_tol`
- `laplace_jitter`

ただし，既存の `epsilon` や jitter があるならそれを流用する．

## 9. 期待される実行フロー

`update_with_multi` の最終的な概念フローは次の形にする．

```text
input: theta_cmd, imu_bingham_map, dt

1. x_pred = self.x
2. P_pred = self.P + Q
3. K_pred = exp(x_pred)
4. theta_eq = equilibrium_solve(theta_cmd, K_pred)
5. compute J_q, J_x
6. compute local Laplace terms from Bingham observations
   - ell
   - grad = d ell / d x
   - info = - d^2 ell / d x^2
7. eigen-decompose info
8. keep observable directions only
9. compute Gaussian posterior in observable subspace
10. apply update gain, info cap, step cap, and K bounds
11. update self.x and self.P
12. return debug info
```

この流れが code 上でも読み取れるようにする．

## 10. テスト項目

### 10.1 単体テスト

可能なら以下を追加する．

1. `info` が対称であること．
2. `info` の小さな負固有値が clip されること．
3. 観測情報がゼロに近い場合，`x` が更新されないこと．
4. `max_log_kp_step` が守られること．
5. `kp_min`, `kp_max` が守られること．
6. `grad = 0` かつ `info` 非ゼロの場合，平均は動かず covariance だけ縮むこと．
7. `info = 0` かつ `grad` 非ゼロに近い異常ケースでは update を安全に skip または抑制すること．

### 10.2 finite difference テスト

少なくとも局所線形モデルに対して，以下を確認する．

- analytic gradient と central difference が一致する．
- analytic information と second directional difference が一致する．
- `info = - Hessian ell` の符号になっている．

### 10.3 regression test

既存 baseline について，以下を確認する．

- no-noise / no-delay baseline で `theta_equil` が `theta_ref` に近づく関係が壊れない．
- 合成 IMU が完全一致する場合，剛性が不要に drift しない．
- yaw 的に観測できない方向で，既存の観測可能性 projection が効く．
- update cycle あたりの equilibrium solve 回数が増えていない．

## 11. レビュー時に確認したい点

実装後，以下を重点的に見る．

1. `A_f` の Bingham likelihood と `grad` / `info` の符号が合っているか．
2. `info` の固有値が観測方向で非負になっているか．
3. `stiffness_update_gain` と `measurement_info_eig_cap` の意味が以前から大きく変わっていないか．
4. `P` の不可観測方向が勝手に縮んでいないか．
5. update skip 時に `P_pred = P + Q` が反映されているか．
6. `max_log_kp_step` と `kp_min` / `kp_max` が最後に必ず適用されているか．
7. equilibrium solve が余計に増えていないか．
8. debug dict で Laplace update の状態を追えるか．

## 12. 完了条件

この変更は，以下を満たしたら完了とする．

- `MultiFrameStiffnessWEKF` が，局所 Laplace 近似として読める構造になっている．
- `A_f` が `K` 分布ではなく Bingham likelihood であることが docstring / comment で明示されている．
- `grad` と `info` の符号・係数が finite difference で検証されている．
- 既存 baseline の挙動が大きく壊れていない．
- 1 update あたりの equilibrium solve 回数が増えていない．
- 観測不能方向の drift 抑制が維持されている．
