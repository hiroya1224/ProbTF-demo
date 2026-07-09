# たわみ補償実装の詳細説明

この文書は 2026 年 7 月 9 日時点の `deflecomp` 実装を，数式とコード上の責務分担を対応させながら説明する．対象の最新コミットは `900f739 add refinement of result` であり，直前の変更では観測可能性に基づく剛性更新の安定化と，平衡解の静力学残差 refinement が追加されている．

## 1. 全体構成

実装は ROS 非依存の中核と，ROS wrapper，シミュレータ，URDF，例題に分かれている．設計上の重要点は，補償器がシミュレータを直接知らないことである．補償器は参照関節角と IMU 観測から補償指令を計算し，シミュレータはその指令を受けて柔軟関節の応答と合成 IMU を生成する．

| package | 主な責務 |
| :-- | :-- |
| `deflecomp_core` | ばねモデル，重力トルク，平衡計算，感度計算，Bingham 観測，剛性 WEKF，補償 pipeline |
| `deflecomp_ros` | ROS topic と parameter の読み書き，`JointState` と `Imu` の変換，補償結果の publish |
| `deflecomp_sim` | 柔軟関節シミュレーション，quasi-static 平衡，動的応答，合成 IMU publish |
| `deflecomp_description` | `simple6r.urdf` と RViz 設定 |
| `deflecomp_examples` | ROS 非依存の offline demo と可視化補助 |

中核のデータフローは次の形である．

```text
theta_ref, IMU observations
  → stiffness update if enabled
  → gravity feedforward
  → theta_cmd filtering
  → equilibrium prediction
  → ROS publish
```

ここで，`theta_ref` は目標姿勢，`theta_cmd` は柔軟関節へ与える補償済み指令，`theta` または `theta_eq` はばねと重力が釣り合った実姿勢，`K` は関節ごとの剛性である．

## 2. ロボットモデルと reduced model

`deflecomp_core.robot.RobotArm` は Pinocchio の URDF model を読み込み，重力トルク，重力トルク微分，frame 姿勢，frame 角速度ヤコビアンなどを提供する．

URDF に mimic joint，fixed joint，velocity limit が 0 の joint が含まれる場合，`RobotArm` はそれらを非制御 joint とみなし，Pinocchio の reduced model を作る．制御対象は `model_joint_names` に残った active joint だけであり，ROS に publish するときは `expand_joint_positions` で URDF 上の movable joint list へ戻す．これにより，たとえば gripper や camera link を持つ URDF でも，補償器は 6 軸 arm の低次元モデルで計算し，RViz/TF には欠けのない `JointState` を渡せる．

重力トルクは Pinocchio の一般化重力である．

```math
\tau_g(\theta) = \operatorname{computeGeneralizedGravity}(q)
```

重力トルク微分は次の行列として使われる．

```math
G_\theta(\theta) = \frac{\partial \tau_g(\theta)}{\partial \theta}
```

また，frame `f` で見た重力方向は，world 重力方向 `g_w` と frame 回転 `R_{wf}` から次のように計算される．

```math
g_f(\theta) = \frac{R_{wf}(\theta)^\top g_w}{\|g_w\|}
```

この重力方向の関節ヤコビアンは `gravity_dir_jacobian_in_frame` で計算される．コードでは frame 表現の角速度ヤコビアン `J_{\omega,f}` と skew 行列を用いている．

```math
J_{g,f}(\theta) =
[g_f(\theta)]_\times J_{\omega,f}(\theta)
```

これは後述する feedforward の観測可能性 projection に使われる．

## 3. ばねモデル

ばねモデルは `deflecomp_core.model.spring` に集約されている．全モデルは次の interface を持つ．

| method | 意味 |
| :-- | :-- |
| `torque(theta, theta_cmd, kp_vec)` | ばね由来の一般化トルク `tau_s` |
| `potential(theta, theta_cmd, kp_vec)` | ばねポテンシャル `V_s` |
| `stiffness_diag(theta, theta_cmd, kp_vec)` | `partial tau_s / partial theta` の対角成分 |
| `log_stiffness_jacobian_diag(...)` | `partial tau_s / partial log K` の対角成分 |
| `theta_cmd_from_theta_ref(...)` | 目標姿勢を静力学的に実現する feedforward 指令 |

### 3.1 線形ばね

線形ばねでは，関節ごとに次を使う．

```math
\tau_s(\theta,\theta_{\rm cmd},K) = K \odot (\theta - \theta_{\rm cmd})
```

```math
V_s(\theta,\theta_{\rm cmd},K) = \frac{1}{2}(\theta - \theta_{\rm cmd})^\top
\operatorname{diag}(K)(\theta - \theta_{\rm cmd})
```

ここで `\odot` は要素ごとの積である．このモデルの local stiffness は単純に `K` である．

```math
\frac{\partial \tau_s}{\partial \theta} = \operatorname{diag}(K)
```

推定状態は `x = log K` なので，log 剛性方向の微分は次になる．

```math
\frac{\partial \tau_s}{\partial x_i} = K_i(\theta_i - \theta_{{\rm cmd},i})
```

### 3.2 周期ばね

回転関節では角度の周期性を扱うため，現在の default 設定では `spring_model: periodic` を使う．実装上のトルクは次である．

```math
\delta = \theta - \theta_{\rm cmd}
```

```math
\tau_s(\theta,\theta_{\rm cmd},K) = 2K \odot \sin\left(\frac{\delta}{2}\right)
```

ばねポテンシャルは次である．

```math
V_s(\theta,\theta_{\rm cmd},K) = \sum_i 4K_i\left(1 - \cos\left(\frac{\delta_i}{2}\right)\right)
```

local stiffness は次になる．

```math
\frac{\partial \tau_{s,i}}{\partial \theta_i} = K_i \cos\left(\frac{\delta_i}{2}\right)
```

log 剛性方向の微分はトルクそのものと同じ形になる．

```math
\frac{\partial \tau_{s,i}}{\partial x_i} = 2K_i \sin\left(\frac{\delta_i}{2}\right)
```

`JointTypeAwareSpringModel` は URDF の joint type を見て，`revolute` と `continuous` を周期ばね，それ以外を線形ばねとして混在させる．ROS node と sim node では `spring_model: auto` のときにこの model を選ぶが，現在の設定ファイルでは controller と simulator の両方を `periodic` に固定している．

## 4. 静力学 feedforward

補償指令 `theta_cmd` は，目標姿勢 `theta_ref` が静力学平衡になるように作る．平衡条件は次である．

```math
F(\theta,\theta_{\rm cmd},K)
= \tau_g(\theta) + \tau_s(\theta,\theta_{\rm cmd},K)
```

```math
F(\theta_{\rm ref},\theta_{\rm cmd},K) = 0
```

線形ばねでは次の閉形式になる．

```math
\theta_{\rm cmd} = \theta_{\rm ref} + K^{-1}\odot \tau_g(\theta_{\rm ref})
```

周期ばねでは次を使う．

```math
\theta_{\rm cmd} = \theta_{\rm ref} +
2\arcsin\left(\frac{\tau_g(\theta_{\rm ref})}{2K}\right)
```

実装では arcsin の引数を `[-1, 1]` の少し内側へ clip し，過大な重力トルクや小さい剛性でも NaN を避けている．`CommandGenerator` は `robot.tau_gravity(theta_ref)` を呼び，選択されたばねモデルの `theta_cmd_from_theta_ref` に委譲するだけの薄い class である．

## 5. 平衡ソルバ

`EquilibriumSolver` は，ある `theta_cmd` と `K` に対する静的な実姿勢 `theta_eq` を求める．目的関数は重力ポテンシャルとばねポテンシャルの和である．

```math
V_{\rm total}(\theta) = U_g(\theta) + V_s(\theta,\theta_{\rm cmd},K)
```

その勾配は静力学残差そのものになる．

```math
\nabla_\theta V_{\rm total}(\theta) = \tau_g(\theta) + \tau_s(\theta,\theta_{\rm cmd},K)
```

最新実装では二段階で解く．第一段階は staged L-BFGS-B で，`lambda` を 1 から 0 へ動かしながら人工的な追加剛性を徐々に抜く．

```math
K_{\rm eff}(\lambda) = K + \lambda k_{\rm stage}\mathbf{1}
```

各 stage では joint position limit を bounds として，次を最小化する．

```math
\theta_\lambda = \arg\min_\theta
\left\{U_g(\theta) + V_s(\theta,\theta_{\rm cmd},K_{\rm eff}(\lambda))\right\}
```

第二段階が `900f739` で入った refinement である．L-BFGS-B の解を初期値として，元の剛性 `K` に対する静力学残差を直接 `least_squares` で潰す．

```math
r(\theta) = \tau_g(\theta) + \tau_s(\theta,\theta_{\rm cmd},K)
```

```math
J_r(\theta) = \frac{\partial r}{\partial \theta} = G_\theta(\theta) +
\operatorname{diag}\left(k_s(\theta,\theta_{\rm cmd},K)\right)
```

ここで `k_s` はばねモデルの `stiffness_diag` である．refinement は残差ノルムが改善した場合だけ採用される．この変更により，staged 最適化が近い basin を見つけ，その後に力の釣り合い式そのものを高精度に満たす構成になった．既存ログでは Yamaguchi arm の `module1_joint1 = 1.51` ケースで，`eq-ref` ノルムが `2.217e-3` 程度から `2.230e-16` 程度へ改善している．

## 6. IMU 観測と Bingham 行列

`deflecomp_ros` は `sensor_msgs/Imu.linear_acceleration` から重力方向を取り出す．静的または低加速度の前提では，IMU の加速度計測 `a_{\rm meas}` に対して frame 内重力方向を次のように置く．

```math
\hat{g}_f = \frac{-a_{\rm meas}}{\|a_{\rm meas}\|}
```

`ImuObservationBuilder` は各 frame の観測重力方向と world 重力方向から Bingham 行列 `A_f` を作る．実装は `simple_bingham_unit` であり，観測方向 `b` と world 方向 `a` を pure quaternion として扱う．

```math
v_q = (0,b_x,b_y,b_z)^\top,\quad
x_q = (0,a_x,a_y,a_z)^\top
```

left/right quaternion multiplication 行列を `L(\cdot)`，`R(\cdot)` とすると，次を作る．

```math
P = L(x_q) - R(v_q)
```

```math
A_f = -\frac{\alpha}{4} P^\top P
```

ここで `alpha` は `A_param` であり，現在の default は `100.0` である．この `A_f` は，予測 quaternion `z_f(theta_eq)` が観測された重力方向と整合するほど Bingham 評価が良くなるように使われる．複数 IMU frame がある場合は，frame id ごとの `A_f` を map として WEKF へ渡す．

## 7. 平衡感度

剛性推定では `K` を直接推定せず，正値制約を自然に満たすため `x = log K` を状態にする．平衡条件を次で置く．

```math
F(\theta_{\rm eq},x) = \tau_g(\theta_{\rm eq}) +
\tau_s(\theta_{\rm eq},\theta_{\rm cmd},\exp(x)) = 0
```

この陰関数の local linearization に必要な行列は `SensitivityCalculator` が返す．

```math
J_q = \frac{\partial F}{\partial \theta} = G_\theta(\theta_{\rm eq}) +
\operatorname{diag}\left(k_s(\theta_{\rm eq},\theta_{\rm cmd},K)\right)
```

```math
J_x = \frac{\partial F}{\partial x} = \operatorname{diag}\left(\frac{\partial \tau_s}{\partial x}\right)
```

理論的には，平衡姿勢の log 剛性に対する感度は次で表される．

```math
\frac{\partial \theta_{\rm eq}}{\partial x} = -J_q^{-1}J_x
```

実装では特異または悪条件の姿勢にも耐えるため，`pinv` を使っている．この感度は，IMU quaternion の変化を剛性状態へ戻すための橋渡しになる．

## 8. MultiFrameStiffnessWEKF

`MultiFrameStiffnessWEKF` は複数 IMU frame の Bingham 観測を使い，log 剛性 `x` を更新する．処理の入口は `update_with_multi` であり，1 cycle の大まかな順序は次である．

1. `P` に process noise `Q` を加える．
2. 現在の `theta_cmd` と `K = exp(x)` から `theta_eq` を解く．
3. `J_q` と `J_x` を計算する．
4. 各 IMU frame の Bingham 行列から gradient と information を集める．
5. 観測可能部分空間だけに更新を射影する．
6. 情報量 cap，更新 gain，log K step cap，K 上下限を適用する．

各 frame では，予測 quaternion `z_f` と tangent map `Q_z`，frame 角速度ヤコビアン `J_{\omega,f}` を使う．実装上の中間量は次である．

```math
v_f = Q_z^\top A_f z_f
```

```math
u_f = J_{\omega,f}^\top v_f
```

剛性方向の二次近似に使う行列は次の形で集める．

```math
X = J_q^\dagger J_x
```

```math
M_f = Q_z J_{\omega,f} X
```

```math
H_{0,f} = \frac{1}{2}M_f^\top A_f M_f
```

全 frame の和から gradient と information を作る．

```math
u = \sum_f u_f,\quad
H_0 = \sum_f H_{0,f}
```

```math
y = (J_q^\top)^\dagger u
```

```math
g = -J_x^\top y
```

```math
\mathcal{I} = -\frac{1}{2}(H_0 + H_0^\top)
```

`mathcal{I}` は観測が log 剛性のどの方向を見ているかを表す局所情報行列として扱われる．現在の実装では，この情報行列を固有分解し，十分に観測されている方向だけを残す．

```math
\mathcal{I} = U\Lambda U^\top
```

```math
\lambda_i
> \max(\lambda_{\rm abs}, r_{\rm cond}\lambda_{\max})
```

この条件を満たす固有ベクトルだけを `U_obs` とし，それ以外を不可観測成分とする．観測可能部分空間では次のように covariance と gradient を射影する．

```math
P_{\rm obs} = U_{\rm obs}^\top P U_{\rm obs}
```

```math
g_{\rm obs} = U_{\rm obs}^\top g
```

情報量 scale `s` は `stiffness_update_gain` と `measurement_info_eig_cap` から決まる．cap は 1 回の IMU 更新が強すぎて剛性を急に動かすことを避けるためのものである．観測空間での posterior は次で近似される．

```math
P_{{\rm post},{\rm obs}} = \left(P_{\rm obs}^{-1} +
s\operatorname{diag}(\lambda_{\rm obs}) +
\epsilon I\right)^{-1}
```

更新量は次である．

```math
\Delta x = U_{\rm obs}P_{{\rm post},{\rm obs}}(s g_{\rm obs})
```

その後，`max_log_kp_step` で各 cycle の `log K` 最大変化量を制限し，`kp_min` と `kp_max` で `K` の範囲を制限する．不可観測成分の covariance は `U_unobs` に射影して保持する．これにより，重力方向 IMU から見えない方向の剛性が residual や noise によって積分的に流れることを抑えている．

## 9. 観測可能性に基づく feedforward projection

`ca09381` の重要な変更は，剛性更新だけでなく feedforward に使う重力評価姿勢も局所観測可能性で制限した点である．問題になっていたのは，yaw 方向に参照姿勢を振ったあと，IMU residual の微小な bias/noise が WEKF を動かし，時間が経つと `theta_cmd` が `theta_ref` 側へ寄ってしまう現象である．

重力方向 IMU は，姿勢によっては yaw 方向をほとんど観測できない．ただし実装は joint 名や yaw という言葉に依存しない．現在の姿勢と IMU frame から次の情報行列を作る．

```math
\mathcal{G}(\theta) = \sum_f J_{g,f}(\theta)^\top J_{g,f}(\theta)
```

固有分解で観測可能 basis `U_g` を取り，参照姿勢の増分だけを観測可能部分空間に射影する．

```math
\Delta\theta_{\rm ref} = \theta_{\rm ref,k} - \theta_{\rm ref,k-1}
```

```math
\Delta\theta_g = U_g U_g^\top \Delta\theta_{\rm ref}
```

補償器は内部状態 `theta_ref_for_feedforward` を持ち，これを次のように更新する．

```math
\theta_{g,k} = \theta_{g,k-1} + \Delta\theta_g
```

実際の指令生成では，目標値としては full の `theta_ref` を使い，重力トルクの評価点だけを `theta_g` にする．

```math
\theta_{\rm cmd,raw} = \operatorname{FeedForward}
\left(\theta_{\rm ref}, \tau_g(\theta_g), K\right)
```

ここが設計上のポイントである．目標姿勢そのものは射影しないため，ユーザが与えた参照値は維持される．一方で，IMU が局所的に見ていない参照増分を使って重力 feedforward を再評価しないので，保持中の不自然な drift を抑えられる．IMU 観測が全くない cycle では `theta_g` を reset せず，直前値を保持する．

## 10. 補償 pipeline

`DeflectionCompensator.step` は 1 cycle の補償計算をまとめている．入力は `theta_ref`，IMU 観測列，`dt`，timestamp である．出力は `theta_cmd`，`theta_cmd_raw`，`theta_eq_hat`，`kp_hat`，`tau_hat`，debug dict である．

処理順序を数式に近い形で書くと次になる．

```math
K_k = \exp(x_k)
```

```math
\theta_{g,k} = \operatorname{ProjectObservable}
(\theta_{{\rm ref},k}, \{\hat{g}_{f,k}\})
```

```math
\theta_{{\rm cmd,raw},k} = \operatorname{InverseStatics}
(\theta_{{\rm ref},k}, \tau_g(\theta_{g,k}), K_k)
```

`theta_cmd_tau` が正の場合は一次遅れを入れる．現在の default は `0.0` なので no-delay baseline では raw 指令がそのまま使われる．

```math
\alpha_k = 1 - \exp\left(-\frac{\Delta t_k}{\tau_{\rm cmd}}\right)
```

```math
\theta_{{\rm cmd},k} = \theta_{{\rm cmd},k-1} +
\alpha_k(\theta_{{\rm cmd,raw},k} - \theta_{{\rm cmd},k-1})
```

最後に，現在の `theta_cmd` と `K` から予測平衡を解く．

```math
\theta_{{\rm eq},k} = \operatorname{EquilibriumSolve}
(\theta_{{\rm cmd},k},K_k)
```

```math
\tau_{{\rm hat},k} = \tau_g(\theta_{{\rm eq},k})
```

剛性更新は `update_stiffness: true` かつ前回の `theta_cmd` と IMU 観測がある場合だけ実行される．これは，観測された IMU が前 cycle の指令に対する結果として入ってくるためである．

## 11. ROS node

`deflecomp_ros/nodes/deflecomp_node.py` は ROS の thin wrapper である．入力 topic は `simple6r.yaml` で次に設定されている．

| 項目 | 現在値 |
| :-- | :-- |
| reference `JointState` | `/ref/joint_states` |
| IMU | `/imu` |
| command output | `/cmd/joint_states` |
| IMU frames | `[link6, link3, link2]` |

IMU は frame ごとに `ImuBuffer` へ時刻付きで保存される．timer callback では，補償器の前回 timestamp に揃えて IMU 重力方向を線形補間し，`FrameImuObservation` を作る．この時刻合わせにより，補償器は「前回指令に対して得られた観測」を使って剛性更新できる．

publish される主な topic は次である．

| topic | 内容 |
| :-- | :-- |
| `/cmd/joint_states` | active joint の `theta_cmd` を full movable joint list へ展開した指令 |
| `/deflecomp/kp_hat` | 推定剛性 `K = exp(x)` |
| `/deflecomp/kp_cov_diag` | log 剛性 covariance の対角 |
| `/deflecomp/theta_eq_hat` | 補償器が予測した平衡姿勢 |
| `/deflecomp/tau_hat` | `theta_eq_hat` における重力トルク |
| `/deflecomp/debug` | raw command，平衡予測，重力トルク，covariance を連結した debug vector |

`deflecomp_frames.launch` は，`ref`，`cmd`，`equil` の 3 系統の `robot_state_publisher` を立ち上げ，それぞれに TF prefix を付ける．RViz 上では参照姿勢，補償指令姿勢，シミュレータ平衡姿勢を重ねて比較できる．

## 12. シミュレータ

`deflecomp_sim` の中心は `FlexibleJointSimulator` である．入力は補償器が publish する `/cmd/joint_states`，出力は `/equil/joint_states` と `/imu` である．

動的 mode では，Pinocchio の質量行列と非線形項を使い，次の形の運動方程式を解く．

```math
M(q)\ddot{q} = \tau_{\rm ext} -
\tau_s(q,q_{\rm ref,eff},K) -
D\dot{q} -
b(q,\dot{q})
```

ここで `b(q,dot q)` は `rnea(q, dot q, 0)` で計算される非線形項であり，重力項を含む．damping `D` が明示されない場合は，初期姿勢の慣性対角と剛性から次を作る．

```math
D_i = 2\zeta\sqrt{K_i M_{ii}}
```

積分器は `rk4` または簡易 Euler を選べる．また，入力指令には slew rate limit と一次遅れを入れられる．

```math
q_{{\rm ref,slew},k} = q_{{\rm ref,slew},k-1} +
\operatorname{clip}\left(
q_{{\rm ref},k}-q_{{\rm ref,slew},k-1},
\, -v_{\max}\Delta t,
\, v_{\max}\Delta t
\right)
```

```math
q_{{\rm ref,eff},k} = q_{{\rm ref,eff},k-1} +
\left(1-\exp\left(-\frac{\Delta t}{\tau_{\rm ref}}\right)\right)
(q_{{\rm ref,slew},k}-q_{{\rm ref,eff},k-1})
```

現在の `sim_params.yaml` は `eq_mode: quasistatic` なので，動的積分ではなく平衡ソルバで `q_eq` を求め，そこへ任意の noise/vibration を足して publish する．現在は noise も vibration も 0 である．

```math
q_k = q_{\rm eq} + \eta_k + v_k
```

合成 IMU では，frame origin の加速度，frame 内 offset，角速度，角加速度，重力を組み合わせ，IMU 座標の specific force を publish する．

```math
a_{p,l} = a_{o,l} +
\alpha_l \times r_{li} +
\omega_l \times (\omega_l \times r_{li})
```

```math
a_{\rm meas,i} = R_{li}^\top a_{p,l} -
R_{wi}^\top g_w
```

補償 node 側ではこの `a_meas` から `-a_meas / ||a_meas||` を取り，重力方向観測として使う．

## 13. 現在の baseline 設定

現在の設定は，まず静的な関係を確認するための staged baseline になっている．controller 側は次である．

| parameter | value |
| :-- | :-- |
| `dt` | `0.02` |
| `theta_cmd_tau` | `0.0` |
| `spring_model` | `periodic` |
| `equilibrium_refine` | `true` |
| `equilibrium_refine_maxiter` | `40` |
| `equilibrium_refine_tol` | `1.0e-12` |

estimator 側は剛性更新を再有効化しつつ，更新をかなり穏やかに制限している．

| parameter | value |
| :-- | :-- |
| `update_stiffness` | `true` |
| `kp0` | `[5.0, 5.0, 5.0, 10.0, 20.0, 20.0]` |
| `kp_min` | `1.0` |
| `kp_max` | `500.0` |
| `q_proc` | `1.0e-8` |
| `observability_rcond` | `0.0001` |
| `observability_abs` | `1.0e-10` |
| `measurement_info_eig_cap` | `1.0` |
| `stiffness_update_gain` | `0.2` |
| `max_log_kp_step` | `0.002` |
| `project_unobservable_feedforward` | `true` |

simulator 側は真の剛性を estimator 初期値に合わせ，quasi-static，no-delay，no-noise としている．

| parameter | value |
| :-- | :-- |
| `dt` | `0.001` |
| `kp_true` | `[5.0, 5.0, 5.0, 10.0, 20.0, 20.0]` |
| `vel_limit` | `4.0` |
| `ref_tau` | `0.0` |
| `ref_max_vel` | `0.0` |
| `eq_mode` | `quasistatic` |
| `qs_noise_std_deg` | `0.0` |
| `qs_vib_amp_deg` | `0.0` |
| `spring_model` | `periodic` |
| `equilibrium_refine` | `true` |

この状態でまず見るべき関係は次である．

```math
\|\theta_{\rm equil} - \theta_{\rm ref}\|
< \|\theta_{\rm cmd} - \theta_{\rm ref}\|
```

`theta_cmd` は重力たわみを打ち消すために `theta_ref` から意図的にずれる．したがって，正しい挙動では `cmd` が `ref` と一致するのではなく，シミュレータが返す `equil` が `ref` に近づく．

## 14. 直近の問題と解決

直近のデバッグでは，yaw 方向に大きく回したあと，保持中に `theta_cmd` が `theta_ref` 側へ寄る問題があった．完全な合成 IMU 観測では drift しない一方，微小な IMU residual や model mismatch を入れると WEKF が不可観測方向の剛性まで動かし，その結果 feedforward が変化していた．

対策は二つに分かれる．第一に，剛性更新を情報行列の観測可能部分空間へ制限した．これにより，現在の IMU frame と姿勢から見えない log 剛性成分は更新されない．第二に，feedforward の重力評価姿勢 `theta_g` も重力方向ヤコビアンの観測可能部分空間でだけ更新するようにした．これにより，観測できない参照増分によって feedforward が再評価され続けることを避けている．

もう一つの問題は，剛性が真値に合っていても `equil` が `ref` と完全に一致しないことだった．これは推定器ではなく平衡ソルバの収束精度の問題であり，L-BFGS-B の後に静力学残差を直接解く refinement を追加して解決した．

## 15. 実装を読むときの対応表

| 概念 | 実装ファイル |
| :-- | :-- |
| Pinocchio wrapper と reduced model | `deflecomp_core/src/deflecomp_core/robot/pinocchio_robot.py` |
| URDF joint 判定 | `deflecomp_core/src/deflecomp_core/robot/urdf_info.py` |
| 線形ばね，周期ばね，joint type aware ばね | `deflecomp_core/src/deflecomp_core/model/spring.py` |
| 平衡ソルバと refinement | `deflecomp_core/src/deflecomp_core/model/equilibrium.py` |
| 平衡感度 | `deflecomp_core/src/deflecomp_core/model/sensitivity.py` |
| Bingham 行列 | `deflecomp_core/src/deflecomp_core/observation/bingham.py` |
| IMU 観測 builder | `deflecomp_core/src/deflecomp_core/observation/imu_observation.py` |
| 剛性 WEKF | `deflecomp_core/src/deflecomp_core/estimator/stiffness_wekf.py` |
| 補償 pipeline | `deflecomp_core/src/deflecomp_core/pipeline/compensator.py` |
| ROS 補償 node | `deflecomp_ros/nodes/deflecomp_node.py` |
| 柔軟関節 simulator | `deflecomp_sim/src/deflecomp_sim/dynamic_simulator.py` |
| ROS simulator node と合成 IMU | `deflecomp_sim/nodes/sim_node.py` |

## 16. 未解決の論点

現在の 3 frame IMU 構成では，6 個の剛性成分すべてを常に強く同定できるとは限らない．重力方向だけを見る IMU 観測では，姿勢や励起の与え方によって不可観測または弱観測な方向が残る．今回の実装は，その成分を無理に更新しないことで安定性を優先している．

剛性そのものをより正確に真値へ近づけたい場合は，追加 IMU frame，より広い姿勢励起，window/batch 推定，既知不可観測成分の固定，または prior の強化が必要になる．一方で，現在の no-delay/no-noise baseline の目的は，まず `theta_cmd` が補償指令として意味を持ち，`equil` が `ref` に近づく静力学関係を確認することである．
