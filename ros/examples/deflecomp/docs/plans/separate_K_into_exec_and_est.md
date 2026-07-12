# plan.md: `K_est` と `K_exec` を分離した逐次 WEKF 実装計画

## 0. 目的

現在の ProbTF / 局所 Laplace 近似ベースの剛性推定実装を前提に，次の構成へ変更する．

- 推定用の剛性 `K_est` と，コマンド生成用の剛性 `K_exec` を分離する．
- 推定側は，更新時刻で得られた 1 回分の IMU 観測を使って，通常の逐次 Kalman filter 的に `K_est` を更新する．
- コマンド生成側は，常に現在の `K_exec` を使って `theta_cmd` を生成する．
- `K_est` の更新結果は `K_exec` に即時代入せず，時間補間によって滑らかに反映する．
- 推定時の入力は `theta_ref` ではなく，実際に送信済みの `theta_cmd_sent` とする．

今回の実装では moving window，batch 推定，UKF，EnKF，particle filter は導入しない．

---

## 1. 重要な設計原則

### 1.1 送信済み command を既知入力として扱う

推定側では，候補剛性 `K_est` ごとに `theta_cmd` を作り直してはいけない．

誤った構成は次である．

\[
K \to \theta_{\rm cmd}(K) \to \theta_{\rm eq}(\theta_{\rm cmd}(K), K) \to z_f.
\]

これは推定器の中にコマンド生成側の閉ループを入れてしまうので，不安定になりやすい．

正しい構成は次である．

\[
\theta_{{\rm cmd}, k}^{\rm sent} \text{ を固定し，}
\quad
\theta_{{\rm eq}, k}(x)
=
\operatorname{EquilibriumSolve}
\left(
\theta_{{\rm cmd}, k}^{\rm sent},
\exp x
\right)
\]

とする．ここで `x = log K_est` である．

### 1.2 `K_est` と `K_exec` を分離する

内部状態を次の 2 系統に分ける．

\[
x_{\rm est} = \log K_{\rm est},
\quad
x_{\rm exec} = \log K_{\rm exec}.
\]

- `x_est`: 推定器が保持する剛性推定値．観測更新により素早く動いてよい．
- `x_exec`: 実際のコマンド生成に使う剛性値．時間補間により滑らかにしか動かさない．

`x_est` の変化をそのまま `theta_cmd` へ反映しないことが，今回の主な安定化である．

### 1.3 同じ観測を二重に使わない

今回は moving window を使わない．したがって，通常の逐次 Kalman filter と同じく，各 IMU 観測は一度だけ `x_est` の更新に使う．

実装では，観測 timestamp または内部 sequence id を使って，処理済み観測を再利用しないようにする．

---

## 2. 全体の処理順序

1 control cycle の理想的な処理順序は次である．

1. 前 cycle で送信した `theta_cmd_sent_prev` と，それに対応する IMU 観測があれば，推定器を 1 回更新する．
2. 推定器が新しい `x_est` を返したら，コマンド生成側の目標値 `x_exec_target` を更新する．
3. `x_exec` を `x_exec_target` に向かって時間補間する．
4. 現在の `x_exec` から `K_exec` を作る．
5. `K_exec` を使って今回の `theta_cmd` を生成する．
6. 今回送信した `theta_cmd` を，次回推定用の `theta_cmd_sent_prev` として保存する．

数式では次の形である．

\[
K_{{\rm exec}, k} = \exp x_{{\rm exec}, k}.
\]

\[
\theta_{{\rm cmd}, k}^{\rm sent}
=
\operatorname{InverseStatics}
\left(
\theta_{{\rm ref}, k},
\tau_g(\theta_{g,k}),
K_{{\rm exec}, k}
\right).
\]

次の観測更新では，この `theta_cmd_sent` を固定入力として使う．

---

## 3. 推定側の定式化

### 3.1 状態

推定器の状態は従来通り

\[
x = \log K
\]

とする．ただし，これは `x_est` であり，コマンド生成に直接使う `x_exec` ではない．

prior は

\[
x^- \sim \mathcal{N}(m^-, P^-)
\]

とする．

process noise は従来通り

\[
P^- = P + Q
\]

でよい．

### 3.2 観測モデル

更新時刻で得られた IMU 観測から，各 frame の Bingham 行列

\[
A_{f,k}^{\rm obs}
\]

が得られているとする．

推定時には，前 cycle で実際に送信された

\[
\theta_{{\rm cmd},k}^{\rm sent}
\]

を固定入力として，候補 `x` に対して

\[
\theta_{{\rm eq},k}(x)
=
\operatorname{EquilibriumSolve}
\left(
\theta_{{\rm cmd},k}^{\rm sent},
\exp x
\right)
\]

を計算する．

各 IMU frame の予測 quaternion を

\[
z_{f,k}(x)=z_f(\theta_{{\rm eq},k}(x))
\]

とする．

Bingham 観測 likelihood は

\[
\ell_k(x)
=
\sum_f
z_{f,k}(x)^\top A_{f,k}^{\rm obs} z_{f,k}(x)
\]

である．

### 3.3 局所 Laplace / WEKF 更新

現在の推定平均 `m^-` の周りで

\[
\ell_k(x)
\approx
\ell_k(m^-)
+
g_k^\top (x - m^-)
-
\frac{1}{2}
(x - m^-)^\top
\mathcal{I}_k
(x - m^-)
\]

と 2 次近似する．

このとき posterior は

\[
P^+
=
\left((P^-)^{-1}+\mathcal{I}_k\right)^{-1},
\]

\[
m^+ = m^- + P^+ g_k
\]

で近似する．

既存実装の `gradient` と `information` の作り方を流用し，その意味をこの式に対応させる．

### 3.4 観測可能性 projection

既存の観測可能部分空間への射影は維持する．

すなわち，情報行列を固有分解し，十分に観測されている方向だけで更新する．不可観測方向は，IMU residual や数値誤差で勝手に動かさない．

ただし，推定値の反映を緩やかにする役割は `K_exec` 側の時間補間へ移す．

---

## 4. 推定側から外す制限

今回，`K_est` はコマンド生成に直接使わないので，推定側の人工的な更新制限は原則として外す．

外す，またはデフォルトで無効化する候補は次である．

- `stiffness_update_gain`
- `max_log_kp_step`
- `measurement_info_eig_cap`

ただし，次の制限は安全・物理範囲・数値安定性のために残す．

- `kp_min`
- `kp_max`
- covariance の対称化
- covariance の正定値性を保つための jitter
- 観測可能性 projection
- NaN / inf の検出
- equilibrium solve failure 時の update skip

実装上は，既存 parameter をすぐ削除せず，互換性のために残してもよい．ただし，今回の新しい default では推定側の step cap や gain は無効にする．

---

## 5. コマンド生成側の定式化

### 5.1 実行用剛性

コマンド生成側は

\[
x_{\rm exec}
\]

を保持する．初期値は

\[
x_{{\rm exec},0}=x_{{\rm est},0}=\log K_0
\]

でよい．

### 5.2 推定結果の受け取り

推定器が新しい

\[
m_{\rm est}^+
\]

を返したら，コマンド生成側は

\[
x_{\rm exec,target} \leftarrow m_{\rm est}^+
\]

とする．

ここでも `kp_min`, `kp_max` に対応する log 範囲へ clip してよい．

### 5.3 時間補間

各 control cycle で

\[
x_{\rm exec}
\]

を

\[
x_{\rm exec,target}
\]

へ一次遅れで近づける．

\[
\alpha_K = 1 - \exp\left(-\frac{\Delta t}{\tau_K}\right).
\]

\[
x_{\rm exec}^{\rm raw,new}
=
x_{\rm exec}
+
\alpha_K
\left(x_{\rm exec,target}-x_{\rm exec}\right).
\]

必要なら，実行側の 1 step 変化量だけを制限する．

\[
x_{\rm exec}^{\rm new}
=
x_{\rm exec}
+
\operatorname{clip}
\left(
\alpha_K(x_{\rm exec,target}-x_{\rm exec}),
-\Delta x_{\rm exec,max},
\Delta x_{\rm exec,max}
\right).
\]

ここでの制限は推定器の制限ではなく，実際にロボットへ送る command の変化を滑らかにするための制限である．

### 5.4 `theta_cmd` 生成

現在の

\[
K_{\rm exec}=\exp x_{\rm exec}
\]

を使って

\[
\theta_{\rm cmd}
=
\operatorname{InverseStatics}
\left(
\theta_{\rm ref},
\tau_g(\theta_g),
K_{\rm exec}
\right)
\]

を生成する．

推定器の `K_est` は，この式に直接入れない．

---

## 6. 実装変更対象

### 6.1 `deflecomp_core/estimator/stiffness_wekf.py`

変更方針:

1. estimator 内の状態を明確に `x_est`, `P_est` として扱う．
2. `update_with_multi(...)` の入力に渡される姿勢は，必ず送信済みの `theta_cmd_sent` とする．
3. `theta_ref` という名前で渡されている箇所があれば，推定用途では `theta_cmd_sent` にリネームする．
4. update 内で `theta_cmd` を再生成しないことを確認する．
5. 人工的な更新緩和を外す．具体的には，推定更新に `stiffness_update_gain` や `max_log_kp_step` を掛けない．
6. ただし，`kp_min`, `kp_max`，観測可能性 projection，NaN check は維持する．
7. update result として，少なくとも次を返す．
   - `x_est`
   - `kp_est`
   - `P_est`
   - `gradient`
   - `information`
   - `obs_rank`
   - `update_applied`
   - `update_skipped_reason`

注意:

- 既存の `gradient` / `information` の符号が，
  `m_plus = m_minus + P_plus g` に対応していることを確認する．
- finite difference test で，`gradient` が likelihood の増加方向を向いていることを確認する．

### 6.2 `deflecomp_core/pipeline/compensator.py`

変更方針:

1. `DeflectionCompensator` に `x_exec` と `x_exec_target` を追加する．
2. 既存の `x` または `kp_hat` を command 生成に直接使っている箇所を，`x_exec` / `K_exec` に置き換える．
3. estimator が返した `x_est` は，直接 `x_exec` に代入せず，`x_exec_target` に保存する．
4. 各 step で `x_exec` を `x_exec_target` へ補間する関数を追加する．
5. `theta_cmd_raw` は `K_exec` から作る．
6. `theta_eq_hat` も，原則として command 側の整合性を優先し，`theta_cmd` と `K_exec` から計算する．
7. 推定側へ渡す姿勢は，前回送信済みの `theta_cmd_sent_prev` とする．
8. 現在の `theta_cmd` を送信したら，次回更新用に保存する．

追加する helper の候補:

- `_update_exec_stiffness_target(x_est)`
- `_smooth_exec_stiffness(dt)`
- `_get_kp_exec()`
- `_get_kp_est()`

### 6.3 ROS node / publish 周り

既存の `/deflecomp/kp_hat` は，互換性のために `K_est` として残す．

追加で次を publish することを検討する．

- `/deflecomp/kp_exec`: コマンド生成に実際に使っている剛性
- `/deflecomp/kp_est`: 推定器の最新剛性
- `/deflecomp/kp_exec_target`: 補間先の剛性

既存 topic を増やすのが重い場合は，まず debug vector / debug dict に `kp_exec`, `kp_est`, `kp_exec_target` を含めるだけでもよい．

### 6.4 config

新しい parameter を追加する．

- `kp_exec_tau`: `K_exec` が `K_est` に追従する時定数．初期値は安全寄りにする．
- `max_log_kp_exec_step`: 1 control cycle あたりの `log K_exec` の最大変化量．必要なら使用する．
- `publish_kp_exec`: `K_exec` topic を publish するか．

既存 parameter の扱い:

- `kp_min`, `kp_max`: 維持．
- `q_proc`: 維持．
- `observability_rcond`, `observability_abs`: 維持．
- `stiffness_update_gain`: 推定側では使わない．互換性のため残すなら default を `1.0` にする．
- `max_log_kp_step`: 推定側では使わない．新しい `max_log_kp_exec_step` と混同しないようにする．
- `measurement_info_eig_cap`: 推定側では原則無効化．残す場合は optional safety として扱う．

---

## 7. update の時刻対応

現在の構成では，IMU 観測は前 cycle の command に対する応答として入ってくる．この対応は維持する．

実装上は次を保存する．

- `prev_theta_cmd_sent`
- `prev_cmd_timestamp`
- `last_processed_imu_timestamp` または `last_processed_observation_seq`

更新条件:

1. `update_stiffness == true`
2. `prev_theta_cmd_sent` が存在する
3. 対応する IMU 観測が存在する
4. その IMU 観測が未処理である

この条件を満たすときだけ estimator update を実行する．

---

## 8. debug 出力

debug には最低限，次を入れる．

- `kp_est`
- `kp_exec`
- `kp_exec_target`
- `log_kp_est`
- `log_kp_exec`
- `log_kp_exec_target`
- `log_kp_exec_delta`
- `est_update_applied`
- `est_update_skipped_reason`
- `est_obs_rank`
- `est_information_eigs`
- `est_gradient_norm`
- `used_theta_cmd_sent_for_update`

これにより，推定値は飛んでいるが実行値は滑らかに動いているかを確認できる．

---

## 9. テスト方針

### 9.1 既知入力固定のテスト

目的: estimator が `theta_cmd` を再生成していないことを確認する．

方法:

- `update_with_multi(...)` に `theta_cmd_sent` を渡す．
- estimator 内で `InverseStatics` または command generator が呼ばれないことを mock で確認する．
- `theta_ref` を変えても，推定 update の入力として使われないことを確認する．

### 9.2 `K_est` と `K_exec` の分離テスト

目的: 推定値が急変しても，コマンド生成用剛性は滑らかにしか変わらないことを確認する．

方法:

- estimator update 後に `x_est` を大きく変化させる．
- 直後の `x_exec` が `x_est` に一致しないことを確認する．
- `kp_exec_tau` に従って `x_exec` が指数的に追従することを確認する．

### 9.3 推定側 step cap 無効化テスト

目的: `K_est` は速く動けることを確認する．

方法:

- 旧 `max_log_kp_step` より大きな更新が必要な synthetic case を作る．
- `x_est` はその差分を許容する．
- `x_exec` は `max_log_kp_exec_step` または `kp_exec_tau` により制限される．

### 9.4 観測の二重利用防止テスト

目的: 同じ IMU 観測を複数回 update に使わないことを確認する．

方法:

- 同じ timestamp の IMU 観測を複数回 step に渡す．
- estimator update が 1 回だけ行われることを確認する．

### 9.5 no-observation 時の挙動

目的: 推定結果が来ない間も command generation が継続することを確認する．

方法:

- IMU 観測なしで複数 step 進める．
- `x_exec` は既存 target に向かって補間を続ける．
- `theta_cmd` は常に `K_exec` に基づいて生成される．

### 9.6 既存 baseline の確認

目的: no-noise / true K 初期値の baseline が壊れていないことを確認する．

確認項目:

\[
\|\theta_{\rm equil} - \theta_{\rm ref}\|
<
\|\theta_{\rm cmd} - \theta_{\rm ref}\|
\]

が従来通り成立すること．

---

## 10. 実装順序

### Step 1: 状態名の整理

- estimator の状態を `x_est`, `P_est` として明示する．
- command generation 側の状態として `x_exec`, `x_exec_target` を追加する．
- 初期値では三者を同じ値にする．

### Step 2: estimator 入力の整理

- 推定 update の入力名を `theta_cmd_sent` に統一する．
- `theta_ref` が estimator likelihood に入っていないことを確認する．
- `theta_cmd_sent` を固定して equilibrium solve する．

### Step 3: 推定側の人工制限を外す

- `stiffness_update_gain` を update 式から外す，または `1.0` 固定にする．
- `max_log_kp_step` を estimator update から外す．
- `measurement_info_eig_cap` を無効化する，または optional にする．
- `kp_min`, `kp_max` は残す．

### Step 4: `K_exec` の補間を実装する

- estimator result を `x_exec_target` に保存する．
- 各 step で `x_exec` を `x_exec_target` へ一次遅れで近づける．
- 必要なら `max_log_kp_exec_step` を適用する．

### Step 5: command generation を `K_exec` に切り替える

- `theta_cmd_raw` の生成に `K_exec` を使う．
- `theta_eq_hat` の計算も，原則として `theta_cmd` と `K_exec` を使う．
- ROS publish / debug で `K_est` と `K_exec` を区別できるようにする．

### Step 6: 時刻対応と二重利用防止

- `prev_theta_cmd_sent` と timestamp を保存する．
- IMU 観測が未処理かどうかを判定する．
- 同じ観測で estimator update が複数回走らないようにする．

### Step 7: テストとログ確認

- 上記の unit test / integration test を追加する．
- RViz / debug topic で `kp_est` と `kp_exec` の差を確認する．
- `K_est` が大きく変化しても，`theta_cmd` が急変しないことを確認する．

---

## 11. 完了条件

この変更は，次を満たせば完了とする．

1. 推定側は `theta_cmd_sent` を固定入力として使っている．
2. 推定側で `theta_cmd` を候補 `K_est` から再生成していない．
3. `K_est` と `K_exec` が内部状態として分離されている．
4. `theta_cmd` 生成は常に `K_exec` を使っている．
5. `K_est` の更新結果は `K_exec_target` に入り，`K_exec` は時間補間で追従する．
6. 同じ IMU 観測を二重に update に使わない．
7. 推定側の人工的な step cap は外れている．
8. 物理範囲 `kp_min`, `kp_max` は維持されている．
9. 観測可能性 projection は維持されている．
10. debug で `kp_est`, `kp_exec`, `kp_exec_target` を区別して確認できる．

---

## 12. 今回やらないこと

- moving-window 推定
- batch MAP 推定
- UKF
- EnKF
- particle filter
- background thread 化
- 複数観測をまとめた非同期 optimizer
- `theta_cmd(K_est)` を likelihood の内部で再生成する設計

background thread 化や UKF は，今回の逐次 WEKF と `K_est` / `K_exec` 分離が安定してから検討する．
