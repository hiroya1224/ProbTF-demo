# Probabilistic IK Demo 実装詳解

## 1. 概要

このデモの目的は，対象物姿勢を一点推定に潰さず，

$$
{}^W T_O \sim \mathrm{ProbTF}(\mu_O, \Sigma_O, A_O)
$$

という形で，位置を Gaussian，姿勢を quaternion 上の Bingham 分布として保持し，
そのまま逆運動学 (IK) の目的関数へ入れることで，
対称物体に対する symmetry-aware IK を実現することである．

実装は ROS1 ノード群として構成されており，概念的には

```text
object_pose_node
  -> object_prob_tf
prob_tf_grasp_target_node
  -> grasp_target_ptfs
symmetry_aware_ik_node
  -> target_joint_states
robot_controller
  -> joint_states
hand_prob_tf_publisher
  -> hand_prob_tf
ptf_visualizer
  -> point cloud / mode pose / mode axes
```

という流れで動く．

---

## 2. ProbabilisticTF の近似モデル

本デモでは厳密な SE(3) 分布は扱わず，次の分離近似を採用する．

$$
p(T) \approx p(p) \, p(q)
$$

ここで

- $p \in \mathbb{R}^3$ は並進
- $q \in S^3$ は unit quaternion

である．

位置側は

$$
p \sim \mathcal{N}(\mu, \Sigma)
$$

姿勢側は

$$
q \sim \mathrm{Bingham}(A)
$$

とする．

このため，メッセージ `ProbabilisticTF` は

$$
\left(
\mu \in \mathbb{R}^3,\;
\Sigma \in \mathbb{R}^{3 \times 3},\;
A \in \mathbb{R}^{4 \times 4},\;
q_{\mathrm{mode}} \in S^3
\right)
$$

を持つ．

実際のメッセージ構造は

```text
header
parent_frame_id
child_frame_id
position_mean
position_covariance
orientation_bingham
orientation_mode
approximation_type
```

であり，`msg/ProbabilisticTF.msg` に対応する．

---

## 3. Bingham 分布の定義

quaternion $q \in S^3$ に対する Bingham 分布は，正規化定数 $F(A)$ を用いて

$$
p(q; A) = \frac{1}{F(A)} \exp(q^\top A q)
$$

で定義される．

ここで

$$
A = A^\top \in \mathbb{R}^{4 \times 4}
$$

は対称行列である．

negative log-likelihood は定数を除けば

$$
\mathcal{L}_{\mathrm{Bingham}}(q)
= - q^\top A q
$$

である．

この量が大きいほど尤度は低く，$q^\top A q$ が大きいほど尤度は高い．

### 3.1 極限としての通常 IK

もし $A$ の集中度が非常に大きく，

$$
\lambda_1, \lambda_2, \lambda_3 \ll 0,\quad \lambda_4 = 0
$$

かつその mode が一点に強く集中していれば，
高尤度領域は mode 近傍の非常に小さい領域になる．

したがって symmetry-aware IK の姿勢項

$$
J_{\mathrm{ori}}(q_H) = - q_H^\top A q_H
$$

は，極限的には「目標姿勢に一致させる通常 IK」に近づく．

---

## 4. Quaternion の約束

本デモでは quaternion を

$$
q = [w, x, y, z]^\top
$$

の順で保持する．

積は Hamilton 積であり，`ptf_utils.py` の実装に対応して

$$
q_{12} = q_1 \otimes q_2
$$

と書く．

座標変換の合成は

$$
q_{WG} = q_{WO} \otimes q_{OG}
$$

である．

この約束に一致するように，Bingham の push-forward は

$$
q_{WG} = R(q_{OG}) q_{WO}
$$

となる右乗算行列 $R(q_{OG})$ を用いて

$$
A_G = R(q_{OG}) A_O R(q_{OG})^\top
$$

と実装している．

---

## 5. object_pose_node

`object_pose_node.py` は，対象物の Prob-TF を publish する．

出力は

$$
{}^W T_O \sim \mathrm{ProbTF}(\mu_O, \Sigma_O, A_O)
$$

である．

### 5.1 位置

位置平均と共分散は launch から与えられる．

$$
\mu_O =
\begin{bmatrix}
0.78 \\ -0.12 \\ 0.46
\end{bmatrix},
\qquad
\Sigma_O =
\begin{bmatrix}
0.0015 & 0.0002 & 0 \\
0.0002 & 0.0010 & 0 \\
0 & 0 & 0.0008
\end{bmatrix}
$$

### 5.2 姿勢 mode

mode quaternion は RPY から生成する．

$$
q_{O,\mathrm{mode}}
= q(\phi, \theta, \psi)
$$

現在のデフォルトでは

$$
(\phi, \theta, \psi) = (0, 0, 55^\circ)
$$

である．

### 5.3 軸対称 Bingham

対象物が軸対称であるとき，対象物軸周りの回転には不確かさがあってよい．
そのため `axially_symmetric_bingham_matrix` では，
対象軸 $\hat{a}$ について微小回転

$$
q_\pm = q_{O,\mathrm{mode}} \otimes q(\hat{a}, \pm \varepsilon)
$$

を作り，その差

$$
t_{\mathrm{sym}} = q_+ - q_-
$$

を quaternion 空間の接ベクトル近似として用いる．

そのうえで，eigenvector 行列

$$
M = [m_1\; m_2\; m_3\; m_4]
$$

を作り，

$$
m_4 = q_{O,\mathrm{mode}}
$$

とし，対称軸方向の接ベクトルが「最も緩い」方向になるように並べ替える．

最終的に

$$
A_O = M \, \mathrm{diag}(z_1, z_2, z_3, 0) \, M^\top
$$

とし，

$$
z_1 = -\kappa_1,\quad
z_2 = -\kappa_2,\quad
z_3 = -\kappa_3
$$

で集中度を与える．

---

## 6. 把持候補ライブラリ

把持候補は `config/grasp_library.yaml` に記述する．
各候補は

$$
{}^O T_{G,k}
= (r_{OG,k}, q_{OG,k})
$$

を持つ．

本デモでは orientation を Euler 角で直接書く代わりに，

- `approach_axis`
- `finger_axis`

で記述する．

実装では手先フレームを

$$
\hat{x}_G = \text{approach axis}, \qquad
\hat{z}_G = \text{finger axis}
$$

と定め，残りを右手系で

$$
\hat{y}_G = \hat{z}_G \times \hat{x}_G
$$

から構成する．

さらに正規直交化を行い，

$$
R_{OG} =
\begin{bmatrix}
\hat{x}_G & \hat{y}_G & \hat{z}_G
\end{bmatrix}
$$

から quaternion

$$
q_{OG} = \mathrm{quat}(R_{OG})
$$

を得る．

これにより，「EE の $x$ 軸を接近方向へ，EE の $z$ 軸を指の向きへ」
という意味論が実装で明示的に保証される．

---

## 7. object Prob-TF から grasp target Prob-TF への合成

`prob_tf_grasp_target_node.py` は，対象物の Prob-TF と grasp library を合成して，

$$
{}^W T_{G,k} \sim \mathrm{ProbTF}(\mu_{G,k}, \Sigma_{G,k}, A_{G,k})
$$

を publish する．

### 7.1 平均位置

mode 姿勢による一次近似を使い，

$$
\mu_{G,k}
=
\mu_O + R(q_{O,\mathrm{mode}})\, r_{OG,k}
$$

とする．

### 7.2 位置共分散

まず floor を入れて

$$
\Sigma_{G,k}^{(0)} = \Sigma_O + \Sigma_{\mathrm{floor}}
$$

とし，

$$
\Sigma_{\mathrm{floor}} = \epsilon I_3
$$

を加える．

次に，もしオフセット $r_{OG,k}$ が 0 でなければ，Bingham サンプル

$$
q_O^{(n)} \sim \mathrm{Bingham}(A_O),\quad n=1,\dots,N
$$

を用いて

$$
u_k^{(n)} = R(q_O^{(n)}) r_{OG,k}
$$

を作り，標本共分散

$$
\Sigma_{\mathrm{rot},k}
= \mathrm{Cov}\left[u_k^{(1)},\dots,u_k^{(N)}\right]
$$

を計算する．

最終的に

$$
\Sigma_{G,k}
= \Sigma_O + \Sigma_{\mathrm{floor}} + \Sigma_{\mathrm{rot},k}
$$

となる．

### 7.3 姿勢 mode

姿勢 mode は quaternion 合成により

$$
q_{G,k,\mathrm{mode}}
=
q_{O,\mathrm{mode}} \otimes q_{OG,k}
$$

とする．

### 7.4 Bingham parameter の push-forward

前節の quaternion 約束と整合するように，

$$
A_{G,k}
=
R(q_{OG,k}) A_O R(q_{OG,k})^\top
$$

を使う．

---

## 8. 手先の deterministic FK

`ToyArm6DOF` は 6 軸アームの簡易モデルである．

各関節は

$$
\theta =
[\theta_1,\theta_2,\theta_3,\theta_4,\theta_5,\theta_6]^\top
$$

を持ち，順に

- joint 1: $z$ 軸回転
- joint 2: $y$ 軸回転
- joint 3: $y$ 軸回転
- joint 4: $x$ 軸回転
- joint 5: $y$ 軸回転
- joint 6: $x$ 軸回転

である．

各関節の homogeneous transform を

$$
T_i(\theta_i)
=
\mathrm{Trans}(o_i)\,
\mathrm{Rot}(a_i,\theta_i)
$$

とすると，手先変換は

$$
{}^W T_H(\theta)
=
\prod_{i=1}^6 T_i(\theta_i)\,
\mathrm{Trans}(o_{\mathrm{tool}})
$$

である．

したがって

$$
{}^W T_H(\theta)
=
\begin{bmatrix}
R_H(\theta) & p_H(\theta) \\
0 & 1
\end{bmatrix}
$$

から

$$
p_H(\theta) \in \mathbb{R}^3,\qquad
q_H(\theta) = \mathrm{quat}(R_H(\theta))
$$

を得る．

---

## 9. 手先側の Prob-TF

`hand_prob_tf_publisher.py` は，現在の関節角 $\theta_{\mathrm{now}}$ に対して，
手先側にも簡易的な Prob-TF を作る．

### 9.1 位置平均

現在姿勢の FK をそのまま使う：

$$
\mu_H = p_H(\theta_{\mathrm{now}})
$$

### 9.2 位置共分散

関節ノイズ

$$
\tilde{\theta}^{(n)} \sim
\mathcal{N}(\theta_{\mathrm{now}}, \Sigma_\theta)
$$

をサンプルし，

$$
\tilde{p}^{(n)} = p_H(\tilde{\theta}^{(n)})
$$

から標本共分散を計算する：

$$
\Sigma_H
=
\mathrm{Cov}\left[\tilde{p}^{(1)},\dots,\tilde{p}^{(N)}\right]
+
\epsilon I_3
$$

### 9.3 姿勢分布

姿勢側は現状，FK mode の周りに人工的な Bingham を置いている：

$$
q_{H,\mathrm{mode}} = q_H(\theta_{\mathrm{now}})
$$

$$
A_H = \mathrm{DemoBingham}(q_{H,\mathrm{mode}}, \kappa_H)
$$

これは「手先姿勢の信念」を厳密に伝播したものではなく，
将来の robot body Prob-TF 統合のためのプレースホルダである．

---

## 10. 可視化

`ptf_visualizer.py` は `ProbabilisticTF` を受け取り，
位置を Gaussian，姿勢を Bingham からサンプルして point cloud を作る．

サンプル数を $N$ とすると，

$$
p^{(n)} \sim \mathcal{N}(\mu,\Sigma),\qquad
q^{(n)} \sim \mathrm{Bingham}(A)
$$

を生成し，回転行列

$$
R^{(n)} = R(q^{(n)})
$$

から各軸の終点

$$
x^{(n)}_{\mathrm{red}} = p^{(n)} + \ell R^{(n)} e_x
$$

$$
x^{(n)}_{\mathrm{green}} = p^{(n)} + \ell R^{(n)} e_y
$$

$$
x^{(n)}_{\mathrm{blue}} = p^{(n)} + \ell R^{(n)} e_z
$$

を出す．

mode pose は

$$
(\mu, q_{\mathrm{mode}})
$$

であり，別途 marker として 3 軸を出している．

---

## 11. IK 目的関数

`symmetry_aware_ik.py` の中核は

$$
J_k(\theta)
=
w_p J_{\mathrm{pos},k}(\theta)
+
w_R J_{\mathrm{ori},k}(\theta)
+
w_m J_{\mathrm{motion}}(\theta)
+
w_l J_{\mathrm{limit}}(\theta)
$$

である．

### 11.1 位置項

把持候補 $k$ の目標位置平均を $\mu_{G,k}$，共分散を $\Sigma_{G,k}$ とすると，

$$
e_{p,k}(\theta) = p_H(\theta) - \mu_{G,k}
$$

$$
J_{\mathrm{pos},k}(\theta)
=
\frac{1}{2}
e_{p,k}(\theta)^\top
\left(\Sigma_{G,k} + \epsilon I_3\right)^{-1}
e_{p,k}(\theta)
$$

である．

### 11.2 姿勢項

symmetry-aware モードでは

$$
J_{\mathrm{ori},k}^{\mathrm{prob}}(\theta)
= - q_H(\theta)^\top A_{G,k} q_H(\theta)
$$

を使う．

これは Bingham の negative log-likelihood の定数項を落としたものである．

### 11.3 baseline の deterministic 姿勢項

比較用 baseline では，target mode quaternion を $q_{G,k,\mathrm{mode}}$ として

$$
J_{\mathrm{ori},k}^{\mathrm{det}}(\theta)
= 1 - |\langle q_{G,k,\mathrm{mode}}, q_H(\theta)\rangle|^2
$$

を用いる．

これは antipodal symmetry を保った quaternion 距離であり，
Bingham を使わない通常 IK に近い比較対象になる．

### 11.4 motion prior

現在関節角 $\theta_{\mathrm{now}}$ からのずれを抑えるため

$$
J_{\mathrm{motion}}(\theta)
=
\frac{1}{2}
\|\theta - \theta_{\mathrm{now}}\|_2^2
$$

を入れる．

この項により，対称性で等価な姿勢が複数ある場合に，
現在姿勢から近い解が選ばれやすくなる．

### 11.5 joint limit barrier

関節中心を

$$
c_i = \frac{\ell_i + u_i}{2}
$$

半レンジを

$$
h_i = \frac{u_i - \ell_i}{2}
$$

正規化量を

$$
\eta_i = \frac{\theta_i - c_i}{h_i}
$$

とすると，soft barrier は

$$
J_{\mathrm{limit}}(\theta)
=
\alpha \sum_i
\left(
\frac{1}{\max(1-\eta_i^2,\delta)} - 1
\right)
$$

として実装されている．

限界を超えるとその解は無効とする．

---

## 12. 最適化

各 grasp candidate ごとに

$$
\theta_k^\star = \arg\min_\theta J_k(\theta)
$$

を解く．

### 12.1 初期値

初期値は

1. 現在姿勢 $\theta_{\mathrm{now}}$
2. 位置だけに基づく heuristic seed
3. ランダム摂動を加えた複数 seed

である．

### 12.2 SciPy がある場合

`scipy.optimize.minimize` を用い，

$$
\min_{\ell \le \theta \le u} J_k(\theta)
$$

を L-BFGS-B で解く．

### 12.3 SciPy がない場合

有限差分勾配

$$
\frac{\partial J}{\partial \theta_i}
\approx
\frac{J(\theta + \varepsilon e_i) - J(\theta - \varepsilon e_i)}
{(\theta_i + \varepsilon) - (\theta_i - \varepsilon)}
$$

を用いて gradient descent を行う．

### 12.4 grasp 候補選択

全候補の実行可能解集合を

$$
\mathcal{R} = \{(\theta_k^\star, J_k(\theta_k^\star))\}
$$

とすると，最終選択は

$$
k^\star = \arg\min_k J_k(\theta_k^\star)
$$

である．

---

## 13. symmetry_aware_ik_node の実行フロー

`symmetry_aware_ik_node.py` は単発 solve ノードである．

1. `/grasp_target_ptfs` を受信
2. `/joint_states` を受信
3. symmetry-aware solve を実行
4. deterministic baseline も実行
5. 最良解を `/target_joint_states` に publish
6. 選ばれた grasp target を `/selected_grasp_target_prob_tf` に publish
7. 結果を `IKResult` に詰めて publish

これにより，実験時に

$$
J^{\mathrm{prob}}(\theta^\star)
\quad\text{vs}\quad
J^{\mathrm{det}}(\theta^\star)
$$

や

$$
\|\theta^\star - \theta_{\mathrm{now}}\|_2
$$

を直接比較できる．

---

## 14. launch 時のデモ構成

`launch/probabilistic_tf_demo.launch` では以下が起動する．

- `robot_controller`
- `robot_state_publisher`
- `object_pose_node`
- `object_prob_tf_visualizer`
- `prob_tf_grasp_target_node`
- `hand_prob_tf_publisher`
- `hand_prob_tf_visualizer`
- `selected_grasp_target_visualizer`
- `rviz`

このとき，対象物側は高集中な Bingham

$$
(\kappa_1,\kappa_2,\kappa_3) = (5000, 5000, 200)
$$

となっており，「対象物軸周りは比較的自由だが，それ以外はかなり固い」
という設定になっている．

---

## 15. 実装上の注意点

### 15.1 quaternion convention

最重要なのは

$$
q_{WG} = q_{WO} \otimes q_{OG}
$$

の統一である．

もしここが

$$
q_{WG} = q_{OG} \otimes q_{WO}
$$

に変わると，Bingham push-forward もすべて書き換える必要がある．

### 15.2 semantic grasp axes

Euler 角ベースの grasp 記述は，意味論が壊れやすい．
そのため，本実装では

$$
\hat{x}_{EE} = \text{approach axis},\qquad
\hat{z}_{EE} = \text{finger axis}
$$

という semantic 定義を優先している．

### 15.3 軸対称 Bingham の tangent ordering

軸対称の緩い方向が誤った接空間軸に割り当てられると，
集中度を大きくしても通常 IK に近づかない．
そのため，対称軸に対応する tangent を
「最も penalty が小さい非 mode 軸」に置く実装が重要になる．

### 15.4 手先側 Prob-TF は暫定

現状の手先側 Bingham は

$$
A_H = \mathrm{DemoBingham}(q_{H,\mathrm{mode}})
$$

であり，真の確率伝播ではない．
将来的には Jacobian による近似や quaternion uncertainty propagation が必要である．

---

## 16. このデモが示すもの

通常 IK では

$$
q_H(\theta) \approx q_{\mathrm{target}}
$$

という一点一致を課す．

一方，本デモでは

$$
q_H(\theta) \in \text{high-likelihood region of } \mathrm{Bingham}(A_G)
$$

を課している．

その結果，対象物が対称であれば，

$$
\arg\min_\theta J(\theta)
$$

は「任意に決めた代表姿勢」ではなく，
「高尤度領域の中で現在姿勢から実現しやすい解」を選ぶようになる．

これは確率的には

$$
\theta^\star
=
\arg\max_\theta
p(\mathrm{target}\mid \theta)\,p(\theta)
$$

すなわち MAP 推定に対応している．

---

## 17. 今後の拡張

本デモは minimal implementation としては十分だが，
研究実装としては次が自然な拡張になる．

1. 位置側のより厳密な uncertainty propagation

$$
\Sigma_{G,k}
\approx
\Sigma_O + J_r \Sigma_q J_r^\top + \Sigma_{\mathrm{floor}}
$$

2. 手先側の quaternion uncertainty 伝播

$$
{}^W T_H(\theta) \sim \mathrm{ProbTF}(\mu_H, \Sigma_H, A_H)
$$

3. grasp 成功確率の統合

$$
\mathrm{score}
=
p_{\mathrm{pose}}
\cdot
p_{\mathrm{reach}}
\cdot
p_{\mathrm{collision\_free}}
$$

4. trajectory level の最適化

$$
\theta_{0:T}^\star
=
\arg\min_{\theta_{0:T}}
\sum_{t=0}^{T}
J(\theta_t)
+
J_{\mathrm{smooth}}(\theta_{0:T})
$$

---

## 18. まとめ

本デモの核は次の 3 点である．

1. 物体姿勢を

$$
\mathrm{Bingham}(A_O)
$$

で保持すること．

2. grasp 候補へ push-forward して

$$
A_{G,k} = R(q_{OG,k}) A_O R(q_{OG,k})^\top
$$

を作ること．

3. IK 目的関数で

$$
J_{\mathrm{ori}}(\theta) = -q_H(\theta)^\top A_{G,k} q_H(\theta)
$$

を直接最小化すること．

この構成により，
対称物体に対して「mode に無理やり合わせる IK」ではなく，
「分布の高尤度領域へ入る IK」を実装できる．
