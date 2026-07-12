# Symmetry-aware Prob-TF IK 実装

以下に示す実装プランに従って，ロボットモデルの手先IKを解くコードを書いてください．

`launch/probabilistic_tf_demo.launch` を起動して生成される，ロボットモデルと，probabilistic-TF で出てくるノイズ付きの axis の存在を前提とします．

別途ターミナルで，IKの solve を実行するノードを叩いたら，IKを解いて，そこに向かって手先をリーチングすることを想定します．
その際，ロボットモデルの手先TFも確率的である必要があると思いますが，それは `launch/probabilistic_tf_demo.launch` を修正して，そのようにできるようにしてください．
また，手先TFについても同じく，点群の表示ができるようにしておくと助かります．


# 実装プラン

## 0. 目的

本実装の目的は，対称性を持つ物体を把持するときに，物体姿勢を一点推定に潰さず，Bingham 分布で表された object Prob-TF をそのまま IK の目的関数に入れることで，対称性を考慮した IK を解けるようにすることである．

従来の IK では，対象物の姿勢を一点の目標姿勢 $T_{\mathrm{target}}$ として与えるため，対称物体に対しても，何らかの代表姿勢を人為的に選ぶ必要がある．しかし，軸対称物体などでは，物体軸まわりの回転は把持に本質的でないことが多い．この場合，代表姿勢に合わせることは不要な制約になり，手先経路が長くなったり，関節限界に近づいたり，解が不安定になったりする．

そこで，物体姿勢の不確かさと対称性を Bingham 分布として持つ object Prob-TF を用い，IK を「目標 pose への距離最小化」ではなく，「target Prob-TF に対する尤度最大化」として定式化する．

## 1. 基本方針

本実装では，object pose estimator が次の Prob-TF を出すものとする．

- object position:
  - 平均 $\mu_O \in \mathbb{R}^3$
  - 共分散 $\Sigma_O \in \mathbb{R}^{3 \times 3}$

- object orientation:
  - quaternion Bingham parameter $A_O \in \mathbb{R}^{4 \times 4}$

つまり，world frame から object frame への Prob-TF を

$${}^W T_O \sim \mathrm{ProbTF}(\mu_O, \Sigma_O, A_O)$$

として扱う．

把持候補は，object frame から grasp frame への deterministic TF として与える．

$${}^O T_{G,k}, \quad k = 1, \dots, K$$

これを object Prob-TF に合成することで，world frame における grasp target Prob-TF を得る．

$${}^W T_{G,k} = {}^W T_O \, {}^O T_{G,k}$$

最後に，各 grasp target Prob-TF に対して IK を解き，最も良い候補を選ぶ．

## 2. 全体アーキテクチャ

実装上は，次の node / module に分ける．

```text
object_pose_node
  |
  | publishes object Prob-TF
  v
prob_tf_grasp_target_node
  |
  | composes object Prob-TF and grasp library
  v
symmetry_aware_ik_node
  |
  | solves Bingham-likelihood IK
  v
robot_controller
````

各 node の役割は次の通りである．

### 2.1 object_pose_node

物体認識器である．

入力:

* point cloud
* RGB-D image
* object id
* camera calibration
* camera-to-world TF

出力:

* object position mean $\mu_O$
* object position covariance $\Sigma_O$
* object orientation Bingham parameter $A_O$

ここで，ICRA の Bingham pose estimation は，主に $A_O$ を出す node として機能する．

### 2.2 prob_tf_grasp_target_node

object Prob-TF と grasp library を合成し，grasp target Prob-TF を生成する node である．

入力:

* object Prob-TF
* grasp candidates ${}^O T_{G,k}$

出力:

* grasp target position mean $\mu_{G,k}$
* grasp target position covariance $\Sigma_{G,k}$
* grasp target orientation Bingham parameter $A_{G,k}$

### 2.3 symmetry_aware_ik_node

grasp target Prob-TF に対して IK を解く node である．

入力:

* current joint angle $\theta_{\mathrm{now}}$
* robot model
* grasp target Prob-TF
* joint limits
* collision constraints if available

出力:

* selected grasp index $k^\star$
* target joint angle $\theta^\star$
* score
* optional diagnostics

### 2.4 robot_controller

通常の関節制御器である．

入力:

* target joint angle $\theta^\star$
* trajectory

出力:

* motor command

## 3. データ構造

### 3.1 ProbTF message

まずは厳密な SE(3) 上の確率分布を作ろうとせず，実装用の近似 Prob-TF として次を持つ．

```text
ProbTF
  header
  parent_frame_id
  child_frame_id

  position_mean: Vector3
  position_covariance: Matrix3x3

  orientation_bingham_A: Matrix4x4
  orientation_mode: Quaternion  # optional

  approximation_type: string
```

`orientation_mode` は必須ではないが，可視化や初期値生成に便利なので持たせてもよい．

### 3.2 GraspCandidate

object frame 上の把持候補である．

```text
GraspCandidate
  grasp_id
  object_to_grasp_position: Vector3
  object_to_grasp_orientation: Quaternion
  approach_axis: Vector3
  finger_axis: Vector3
  weight: double
```

`approach_axis` や `finger_axis` は，後で接触条件やアプローチ方向制約を足す場合に使う．最初の実装では使わなくてもよい．

### 3.3 IKResult

```text
IKResult
  grasp_id
  theta_solution: double[]
  total_cost: double
  position_cost: double
  orientation_cost: double
  motion_cost: double
  joint_limit_cost: double
  success: bool
```

## 4. 数学的定式化

### 4.1 Hand pose

関節角 $\theta$ に対する hand frame の FK を

$${}^W T_H(\theta) = (p_H(\theta), q_H(\theta))$$

と書く．

ここで，

* $p_H(\theta) \in \mathbb{R}^3$ は手先位置
* $q_H(\theta) \in S^3$ は手先姿勢 quaternion

である．

### 4.2 Target Prob-TF

object Prob-TF と grasp candidate ${}^O T_{G,k}$ から，grasp target Prob-TF を作る．

$${}^W T_{G,k} \sim \mathrm{ProbTF}(\mu_{G,k}, \Sigma_{G,k}, A_{G,k})$$

ここで，

* $\mu_{G,k}$ は grasp target position の平均
* $\Sigma_{G,k}$ は grasp target position の共分散
* $A_{G,k}$ は grasp target orientation の Bingham parameter

である．

### 4.3 Position likelihood

位置側は Gaussian likelihood として評価する．

# $$J_{\mathrm{pos},k}(\theta)

\frac{1}{2}
(p_H(\theta) - \mu_{G,k})^\top
\Sigma_{G,k}^{-1}
(p_H(\theta) - \mu_{G,k})$$

実装上は，数値安定化のために

$$\Sigma_{G,k}^{\mathrm{reg}} = \Sigma_{G,k} + \epsilon I_3$$

を使う．

### 4.4 Orientation likelihood

姿勢側は Bingham likelihood として評価する．Bingham density は正規化定数を除いて

$$p(q) \propto \exp(q^\top A q)$$

である．

したがって，negative log-likelihood は定数を除いて

# $$J_{\mathrm{ori},k}(\theta)

* q_H(\theta)^\top A_{G,k} q_H(\theta)$$

と書ける．

この項が symmetry-aware IK の中核である．

対象物に軸対称性がある場合，$A_{G,k}$ はその対称性に沿った広がりを持つ．そのため，IK は対称性により等価な姿勢に対して低いコストを与える．つまり，物体姿勢を一点に潰さなくても，Bingham likelihood によって自然に symmetry-aware な姿勢評価が得られる．

### 4.5 Motion prior

現在の関節角から大きく離れない解を選びたい場合，関節空間の prior を入れる．

# $$J_{\mathrm{motion}}(\theta)

\frac{1}{2}
(\theta - \theta_{\mathrm{now}})^\top
W_\theta
(\theta - \theta_{\mathrm{now}})$$

これは「対称性の中で最短の手先経路を選ぶ」効果を持つ．

### 4.6 Joint limit cost

関節限界を避けるための cost を入れる．最初は hard constraint でもよい．

soft cost として入れるなら，例えば各関節について

# $$J_{\mathrm{limit}}(\theta)

\sum_i
\phi_i(\theta_i)$$

とする．

ここで $\phi_i$ は関節限界に近づくと増える barrier 型の関数である．最初の実装では，既存 IK solver の joint limit constraint を使えばよい．

### 4.7 Total cost

各 grasp candidate $k$ に対して，次を最小化する．

# $$J_k(\theta)

w_p J_{\mathrm{pos},k}(\theta)
+
w_R J_{\mathrm{ori},k}(\theta)
+
w_m J_{\mathrm{motion}}(\theta)
+
w_l J_{\mathrm{limit}}(\theta)
+
J_{\mathrm{collision}}(\theta)$$

最終的に，

# $$k^\star, \theta^\star

\operatorname*{argmin}_{k,\theta}
J_k(\theta)$$

を選ぶ．

## 5. Bingham parameter の push-forward

object orientation が

$$q_O \sim \mathrm{Bingham}(A_O)$$

であり，object frame から grasp frame への相対姿勢が quaternion $q_{OG,k}$ で与えられるとする．

grasp orientation は

$$q_{G,k} = q_O \odot q_{OG,k}$$

である．

quaternion の右乗算を行列 $R(q_{OG,k})$ で表すと，

$$q_{G,k} = R(q_{OG,k}) q_O$$

と書ける．

このとき，Bingham parameter は

# $$A_{G,k}

R(q_{OG,k})^{-\top}
A_O
R(q_{OG,k})^{-1}$$

で push-forward される．

ただし，$R(q_{OG,k})$ は直交行列なので，実装上は

# $$A_{G,k}

R(q_{OG,k})
A_O
R(q_{OG,k})^\top$$

または

# $$A_{G,k}

R(q_{OG,k})^\top
A_O
R(q_{OG,k})$$

のどちらになるかを，使用している quaternion convention に合わせて確認する．

重要なのは，実装中で

```text
q_G = q_O * q_OG
```

としているなら，それと一致する線形写像を使って $A_O$ を変換することである．

## 6. Position distribution の合成

最初の実装では，位置側は次の一次近似で十分とする．

object position が

$$p_O \sim \mathcal{N}(\mu_O,\Sigma_O)$$

であり，grasp offset が object frame で $r_{OG,k}$ とする．

object orientation の mode を $R_O^{\mathrm{mode}}$ とすると，

# $$\mu_{G,k}

\mu_O
+
R_O^{\mathrm{mode}} r_{OG,k}$$

とする．

共分散は，まずは

# $$\Sigma_{G,k}

\Sigma_O
+
\Sigma_{\mathrm{rot},k}
+
\Sigma_{\mathrm{floor}}$$

と置く．

ここで $\Sigma_{\mathrm{rot},k}$ は object orientation uncertainty が grasp offset に与える位置不確かさである．

最初の簡易実装では，次のどちらかでよい．

1. orientation uncertainty を無視して $\Sigma_{G,k} = \Sigma_O + \Sigma_{\mathrm{floor}}$ とする．
2. Bingham からサンプルし，$R(q) r_{OG,k}$ の標本共分散を $\Sigma_{\mathrm{rot},k}$ として使う．

P-TF の理論実装が進んだら，Bingham-induced vector distribution の Gaussian surrogate を使って $\Sigma_{\mathrm{rot},k}$ を高速に計算する．

## 7. MAP-IK としての解釈

この IK は，確率論的には MAP 推定として解釈できる．

関節角 $\theta$ を未知変数とし，grasp target Prob-TF を観測 likelihood とみなす．

$$p(\theta \mid \mathrm{target})
\propto
p(\mathrm{target} \mid \theta) p(\theta)$$

ここで，

* $p(\mathrm{target} \mid \theta)$ は hand pose が target Prob-TF に乗っている尤度
* $p(\theta)$ は現在姿勢から近い，関節限界を避ける，などの prior

である．

負の対数を取ると，

# $$\theta^\star

\operatorname*{argmin}_{\theta}
\left[
-\log p(\mathrm{target} \mid \theta)
------------------------------------

\log p(\theta)
\right]$$

となる．

このうち，

* position likelihood が Gaussian cost
* orientation likelihood が Bingham cost
* joint-space prior が motion cost

に対応する．

## 8. 実装ステップ

### Step 1: object Prob-TF を受け取る

まず，object pose estimator から次を受け取る．

```text
mu_O
Sigma_O
A_O
q_O_mode
```

可視化用に，Bingham mode と principal axes を RViz などに出す．

### Step 2: grasp library を読む

object id に応じて，事前に定義された grasp candidates を読む．

```text
grasp_candidates = load_grasp_library(object_id)
```

各候補は

```text
p_OG_k
q_OG_k
```

を持つ．

### Step 3: grasp target Prob-TF を作る

各 candidate について，

```text
mu_G_k = mu_O + R(q_O_mode) @ p_OG_k
Sigma_G_k = compose_position_covariance(...)
A_G_k = pushforward_bingham(A_O, q_OG_k)
```

を計算する．

### Step 4: 各 candidate に対して IK を解く

各 candidate について，既存の数値最適化 solver で

```text
minimize J_k(theta)
```

を解く．

最初は次の solver のいずれかでよい．

* scipy.optimize.minimize
* NLopt
* trajopt
* MoveIt の IK solver に cost wrapper を足す
* 自前 Gauss-Newton / Levenberg-Marquardt

最初の検証では，2D or 6DoF arm の Python 実装で十分である．

### Step 5: 最良候補を選ぶ

すべての candidate の解を比較し，

```text
best = min(results, key = total_cost)
```

で選ぶ．

### Step 6: 実行する

得られた $\theta^\star$ に対して，通常の trajectory generator で実行する．

実機では，衝突回避と手先アプローチ方向を追加で見る．

## 9. 疑似コード

```python
def solve_symmetry_aware_prob_tf_ik(object_prob_tf, grasp_candidates, theta_now, robot_model):
    results = []

    mu_O = object_prob_tf.position_mean
    Sigma_O = object_prob_tf.position_covariance
    A_O = object_prob_tf.orientation_bingham_A
    q_O_mode = object_prob_tf.orientation_mode

    for grasp in grasp_candidates:
        p_OG = grasp.position
        q_OG = grasp.orientation

        mu_G = mu_O + quat_to_rot(q_O_mode) @ p_OG
        Sigma_G = compose_position_covariance(Sigma_O, A_O, q_O_mode, p_OG)
        A_G = pushforward_bingham_right(A_O, q_OG)

        def cost(theta):
            p_H, q_H = robot_model.forward_kinematics(theta)

            pos_err = p_H - mu_G
            J_pos = 0.5 * pos_err.T @ inv_reg(Sigma_G) @ pos_err

            J_ori = - q_H.T @ A_G @ q_H

            dtheta = theta - theta_now
            J_motion = 0.5 * dtheta.T @ W_theta @ dtheta

            J_limit = joint_limit_cost(theta, robot_model)
            J_collision = collision_cost(theta, robot_model)

            return (
                w_p * J_pos
                + w_R * J_ori
                + w_m * J_motion
                + w_l * J_limit
                + J_collision
            )

        theta_init = make_initial_guess(theta_now, grasp, robot_model)
        theta_sol, success = optimize(cost, theta_init, robot_model.joint_limits)

        results.append({
            "grasp_id": grasp.grasp_id,
            "theta": theta_sol,
            "cost": cost(theta_sol),
            "success": success,
        })

    feasible_results = [r for r in results if r["success"]]

    if len(feasible_results) == 0:
        return None

    best = min(feasible_results, key = lambda r: r["cost"])
    return best
```

## 10. 最初に作るべき最小実装

最初から full Prob-TF にしない．まずは以下の minimal version を作る．

### Minimal version

* object position は一点または小さい Gaussian
* object orientation は Bingham
* grasp candidate は複数持つ
* position cost は通常の二乗誤差
* orientation cost だけ Bingham likelihood にする
* motion cost で現在姿勢に近い解を選ぶ

つまり，最初は

# $$J_k(\theta)

## w_p |p_H(\theta) - \mu_{G,k}|^2

w_R q_H(\theta)^\top A_{G,k} q_H(\theta)
+
w_m |\theta - \theta_{\mathrm{now}}|^2$$

で十分である．

これだけでも，対称物体では「任意の代表姿勢へ合わせに行く」のではなく，「Bingham 分布の高尤度領域の中で，現在姿勢から近い解を選ぶ」という挙動が出る．

## 11. 次に拡張する部分

### 11.1 position uncertainty

位置側の Gaussian covariance を入れ，Mahalanobis distance にする．

### 11.2 robot body Prob-TF

hand pose 側も deterministic FK ではなく，robot body belief から誘導される Prob-TF にする．

この場合，IK の目的は

$$\Pr[{}^W T_H(\theta) \approx {}^W T_G]$$

を最大化する問題になる．

### 11.3 grasp success probability

単なる pose likelihood ではなく，grasp success probability を評価する．

例えば，

```text
success_score = pose_likelihood * reachability_score * collision_free_score
```

のようにする．

### 11.4 active perception / active reaching

target Prob-TF の不確かさが大きい場合，すぐに把持せず，次の観測姿勢を選ぶ．

これは next-best-view や active sensing に接続できる．

## 12. 評価実験

### 12.1 比較対象

以下を比較する．

1. Deterministic IK

   * object pose を Bingham mode の一点に潰す．
   * その姿勢に対して通常 IK を解く．

2. Symmetry-enumerated IK

   * 対称性を手で列挙し，複数の代表姿勢に対して IK を解く．
   * 最良のものを選ぶ．

3. Symmetry-aware Prob-TF IK

   * Bingham likelihood を直接使う．
   * 代表姿勢を選ばず，分布の高尤度領域として解く．

### 12.2 評価指標

* IK success rate
* grasp success rate
* joint motion length $|\theta^\star - \theta_{\mathrm{now}}|$
* end-effector path length
* joint limit margin
* computation time
* sensitivity to object pose ambiguity
* robustness to perception noise

### 12.3 期待される結果

対称物体では，deterministic IK は物体姿勢の代表値に過剰に合わせようとする．そのため，関節移動量が大きくなったり，関節限界に近づいたりする．

Symmetry-aware Prob-TF IK では，Bingham likelihood が対称性方向の余裕を持つため，現在姿勢から近く，かつ把持に必要な制約だけを満たす解を選びやすい．

## 13. デモ案

### Demo 1: 軸対称物体の把持

対象:

* 円筒
* ペットボトル
* コップ
* 筒状部品

内容:

1. object pose estimator が Bingham orientation を出す．
2. object 軸まわりの回転が曖昧な分布になる．
3. deterministic IK は Bingham mode に対応する姿勢へ合わせに行く．
4. symmetry-aware Prob-TF IK は軸まわりの自由度を使い，より近い把持姿勢を選ぶ．
5. 関節移動量と成功率を比較する．

### Demo 2: 現在姿勢によって把持姿勢が変わる

同じ object Prob-TF に対して，ロボットの初期姿勢を変える．

期待される挙動:

* deterministic IK は常に同じ代表姿勢を目指す．
* symmetry-aware Prob-TF IK は，対称性の範囲内で，現在姿勢から近い把持姿勢を選ぶ．

### Demo 3: robot body Prob-TF との統合

object 側だけでなく，robot hand 側にも Prob-TF を入れる．

期待される挙動:

* object uncertainty と robot body uncertainty を同じ Prob-TF framework で扱える．
* robot の身体 belief が不確かな場合，より安全な把持候補を選ぶ．

## 14. 博論での位置づけ

この実装は，ICRA の Bingham pose estimation を Prob-TF framework に接続する応用として位置づける．

ICRA 論文の内容は，単に「対称物体の姿勢を推定した」ではなく，

```text
object frame の orientation uncertainty を Bingham 分布として publish する perception node
```

として解釈する．

その object Prob-TF を IK に渡すことで，後段のロボット動作が symmetry-aware になる．

したがって，本章の主張は次のようになる．

```text
Bingham 分布で表された object Prob-TF を IK の姿勢尤度として用いることで，
対称性を持つ物体に対して，任意に選んだ代表姿勢へ手先を合わせる必要がなくなる．
その結果，対称性により等価な把持姿勢の中から，
現在のロボット姿勢や関節制約に対して最も実現しやすい IK 解を選ぶことができる．
```

## 15. 実装上の注意

### 15.1 quaternion convention

最も危険なのは quaternion の左右乗算と座標系 convention である．

特に，

```text
q_G = q_O * q_OG
```

なのか，

```text
q_G = q_OG * q_O
```

なのかを実装全体で統一する．

Bingham parameter の push-forward も，この convention と一致させる必要がある．

### 15.2 quaternion sign

Bingham 分布は antipodal symmetry を持つので，$q$ と $-q$ を同一視できる．

ただし，数値最適化で quaternion を FK から得る場合，符号が不連続に反転すると cost が不安定になる可能性がある．Bingham cost 自体は $q^\top A q$ なので符号反転に不変だが，mode との距離などを別途使う場合は注意する．

### 15.3 Bingham parameter scaling

$A_G$ のスケールが大きすぎると，姿勢 cost が強くなりすぎる．

実装では，$w_R$ と $A_G$ のスケールを分けて調整できるようにする．

### 15.4 singular covariance

位置共分散 $\Sigma_G$ が特異または悪条件になる可能性がある．

必ず

$$\Sigma_G^{\mathrm{reg}} = \Sigma_G + \epsilon I_3$$

を使う．

### 15.5 まずは cost の可視化を行う

実機に入れる前に，姿勢 cost

$$J_{\mathrm{ori}}(q) = - q^\top A_G q$$

をサンプリングして可視化し，対称性方向にコストが平らになっていることを確認する．

## 16. 最初のマイルストーン

### Milestone 1

Python で toy model を作る．

* 6DoF arm or simple 3DoF arm
* cylindrical object
* Bingham orientation target
* Bingham-likelihood IK

出力:

* deterministic IK との比較
* joint motion length
* cost landscape visualization

### Milestone 2

ROS node 化する．

* object Prob-TF message
* grasp target Prob-TF generator
* symmetry-aware IK solver
* RViz visualization

### Milestone 3

実ロボットで把持デモを行う．

* 対象は円筒物体
* 物体姿勢の代表値を変えても，symmetry-aware IK の解が不自然に変わらないことを示す
* deterministic IK と比較する

### Milestone 4

robot body Prob-TF と統合する．

* object Prob-TF
* robot body Prob-TF
* hand target likelihood
* reaching / grasping success probability

## 17. まとめ

この実装の核は，IK の姿勢誤差を通常の回転距離ではなく，Bingham likelihood として書くことである．

普通の IK:

```text
hand orientation を target orientation に一致させる
```

Symmetry-aware Prob-TF IK:

```text
hand orientation が target orientation distribution の高尤度領域に入るようにする
```

この違いにより，対称物体に対して，物体姿勢を一点に潰す必要がなくなる．

結果として，対称性により等価な把持姿勢の中から，現在姿勢に近く，関節制約を満たし，実行しやすい解を選ぶことができる．

```
