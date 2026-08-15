はい。ここまでの議論を反映すると、次の ablation study はかなり整理された形になります。**新しい基準線を一本決め、そこから一要素ずつ外す**構成にします。

現行実装は `split_estimated` の rotor/gimbal 2 lag、手動 `lag_bounds`、smooth schedule `(4,2,1,0.5)`、stateful gimbal replay を前提にしています。また fixed ablation は27ケースあります。
今回、この時間処理と gimbal 処理をかなり単純化します。

# 1. 新しい default estimator

一回の推定では引き続き **1 bag のみ**を扱います。

基準設定は

[
\boxed{
\begin{aligned}
&\text{Pose SG window}=1.0~{\rm s},\qquad d=5,\
&\text{Gimbal SG window}=1.0~{\rm s},\qquad d=5,\
&\text{geometric }SO(3),\
&\text{identity acceleration weighting},\
&\text{measured gimbal trajectory},\
&\text{one common rotor-command lag }\delta,\
&\epsilon_k=2^{-k},\quad k=0,\ldots,9,\
&\text{strict-ZOH cell refinement},\
&\text{analytic Jacobian},\
&\text{matrix-exponential inertia chart},\
&\text{exact common-scale KKT gauge}.
\end{aligned}}
]

とします。

推定変数は smooth phase では

[
x=(q,\delta)\in\mathbb R^{15},
]

ここで

[
q\in\mathbb R^{14}
]

は従来通り

[
m,\quad \Sigma\to J,\quad c,\quad f_1,\ldots,f_4
]

を表します。

最終 strict phase では lag cell を固定するので、各 cell 内で解くのは14次元 (q) だけです。

---

# 2. 入力データ

bag の指定区間から、

[
p(t),R(t)
]

という mocap pose、

[
u_1(t),\ldots,u_4(t)
]

という rotor thrust command、

[
\theta_1^{\rm meas}(t),\ldots,\theta_4^{\rm meas}(t)
]

という actual gimbal joint angle を読みます。

IMU

[
\omega_{\rm imu}(t),\qquad a_{\rm imu}(t)
]

も読みますが、引き続き objective には入りません。

現在は `gimbal_position` を最初の一点だけ actual actuator state として使用し、その後は command から stateful propagation しています。
ここを変更し、**gimbal_position の全時系列を使います。**

---

# 3. Pose と gimbal の SG

Pose は現在通り geometric SG です。

[
p(t),R(t)
\longrightarrow
s^{\rm obs}(t),\omega(t),\alpha^{\rm obs}(t).
]

gimbal は4本の scalar time series なので、普通の irregular-time SG を使います。

各関節について必要なら unwrap してから、

[
\theta_i^{\rm raw}(t)
\longrightarrow
\theta_i^{\rm SG}(t)
]

とします。

そして pose SG と同じ

[
W=1.0~{\rm s},\qquad d=5
]

を使用します。

最終的な dynamics evaluation time

[
t_1,\ldots,t_N
]

は

* pose SG が centered evaluation 可能
* gimbal SG が centered evaluation 可能
* rotor command の causal support が存在

する共通部分に取ります。

SG-window ablation をするときは、

[
W_{\rm pose}=W_{\rm gimbal}
]

を常に保ちます。つまり「pose だけ1.5 s、gimbal は1.0 s」のような余計な自由度は作りません。

---

# 4. rotor command だけに lag を持たせる

gimbal angle は観測済みなので、rigid-body model の gimbal lag はなくします。

推定する timing parameter は

[
\boxed{\delta=\delta_{\rm rotor}}
]

だけです。

初期値は今の方針を維持して、

[
\boxed{
\delta_0=T_r
}
]

とします。

ここで

[
T_r
===

\operatorname{median}
(t^{\rm cmd}_{j+1}-t^{\rm cmd}_j).
]

つまり command 一周期分です。

(\delta=0) は causal domain の端なので、通常初期値にはしません。`lag_zero` ablation のときだけ使います。

---

# 5. 手動 `lag_bounds` は廃止

現在は estimated lag に明示的 `lag_bounds` が必須です。

新方式では CLI の

```text
--lag-bounds
```

を削除します。

代わりに recorded data が許す causal support を自動計算します。

[
0\leq\delta\leq\delta_{\max}^{\rm data}.
]

(\delta_{\max}^{\rm data}) は、

> 全 evaluation time について (t_k-\delta) に実際に記録された過去 rotor command が存在する

という条件から決めます。

可能なら I/O 側では fitting 区間より前の **bag 内に存在する rotor-command prehistory をすべて保持**します。これは fitting observation を区間外へ広げるという意味ではなく、causal input history を保持するだけです。

もし estimator が data-support 上限に到達した場合は、

```text
lag_reached_data_support_boundary: true
```

と報告します。

これは「物理 lag の upper bound に張り付いた」ではなく、

> この bag からそれ以上古い command を参照できない

という意味になります。

---

# 6. rotor actuator model

default では現在の

[
\tau_{\rm thrust}=0
]

をそのまま使います。

したがって objective 内の actual thrust は単純に

[
T_i(t_k;\delta)
===============

\operatorname{clip}
\left(
u_i(t_k-\delta),
T_{\min},T_{\max}
\right).
]

現在の stateful actuator code でも time constant (=0) のとき target thrust に即時一致する実装です。

したがって **objective 用 rotor thrust のために actuator state propagation を行う必要はありません。**

これが smooth (\to) strict 極限を非常に明瞭にします。

---

# 7. wrench と acceleration objective

各 evaluation time で、

[
T_i(t_k;\delta)
]

と

[
\theta_i^{\rm SG}(t_k)
]

から rotor force direction、作用点、reaction torque を計算し、

[
w_k(q,\delta)
=============

\begin{bmatrix}
F_k\
\tau_k
\end{bmatrix}.
]

現在の `actuator_wrench()` は thrust と actual gimbal angle を直接受け取り、gimbal angle から推力方向と作用点を計算しています。

Newton–Euler により

[
\hat\alpha_k
============

J^{-1}
\left[
\tau_k-\omega_k\times(J\omega_k)
\right],
]

pose sensor の CoG からの offset

[
\ell=p_{\rm pose/B}-c
]

を使って

[
\hat s_k
========

\frac{F_k}{m}
+
\hat\alpha_k\times\ell
+
\omega_k\times(\omega_k\times\ell)
]

を出します。

residual は

[
r_k=
\begin{bmatrix}
s_k^{\rm obs}-\hat s_k\
\alpha_k^{\rm obs}-\hat\alpha_k
\end{bmatrix}.
]

default objective は

[
\boxed{
L(q,\delta)
===========

\frac12\sum_{k=1}^N|r_k|^2
}
]

です。

full SG covariance 自体は引き続き全 case で計算し、reference diagnostic として残します。

---

# 8. (2^{-k}) smooth continuation

ここは API の意味も整理します。

dimensionless smoothing parameter を

[
\boxed{
\epsilon_k=2^{-k}
}
]

と定義します。

command transition の**全幅**を

[
w_k=\epsilon_k T_r
]

とします。

したがって half-width は

[
h_k=\frac{\epsilon_k T_r}{2}.
]

現在の `QuinticSmoothZoh` は `width_fraction × median_period` を **half-width** と定義しています。
ここは混乱しやすいので、新 API では直接 `epsilon` を渡し、

```python
half_width = 0.5 * epsilon * median_period
```

とします。

default schedule は

[
\epsilon=
1,\frac12,\frac14,\frac18,\ldots,\frac1{512}
]

です。

つまり

[
k=0,\ldots,9.
]

例えば (T_r=5) ms なら最後の transition 全幅は

[
\frac{5\ {\rm ms}}{512}
\simeq 9.77~\mu{\rm s}.
]

各段階で

[
(q_k,\delta_k)
==============

\arg\min_{q,\delta}
L_{\epsilon_k}(q,\delta)
]

を解析 Jacobian + KKT-LM で解き、

[
(q_k,\delta_k)
]

を次段の初期値にします。

---

# 9. smooth と strict の極限検証を毎段階保存

これは今回かなり重要な diagnostic にします。

各 (k) について、最適点

[
(q_k,\delta_k)
]

を得たら、同じ点を

* smooth command
* exact strict ZOH

の両方で評価します。

保存するのは例えば

[
L_{\epsilon_k}(q_k,\delta_k),
\qquad
L_0(q_k,\delta_k),
]

[
|L_{\epsilon_k}-L_0|,
]

[
\max_{i,n}
|T^{\epsilon_k}*{i,n}-T^0*{i,n}|,
]

[
|\delta_k-\delta_{k-1}|,
]

[
|q_k-q_{k-1}|,
]

[
\left|
\frac{\partial r}{\partial\delta}
\right|
]

です。

したがってレポート上で、

[
\epsilon\downarrow0
]

とともに smooth estimator が strict estimator にどう近づいているかを直接確認できます。

この convergence plot は今回必須にします。

---

# 10. 最後は exact strict-ZOH cell refinement

(k=9) の

[
(q_9,\delta_9)
]

から strict refinement へ移ります。

ここでは黄金分割も Powell も使いません。

strict ZOH で evaluation time (t_k) が command sample (j) を参照している条件は

[
t_j^{\rm cmd}
\le t_k-\delta
<
t_{j+1}^{\rm cmd}.
]

したがってその sample selection を保つ lag interval は

[
t_k-t_{j+1}^{\rm cmd}
<
\delta
\le
t_k-t_j^{\rm cmd}.
]

全時刻について共通部分を取れば、現在の strict-ZOH equivalence cell

[
\boxed{
C=(\delta_L,\delta_R]
}
]

が厳密に求まります。

この cell 内では全 evaluation time で同じ rotor command が使われるので、

[
T_i(t_k;\delta)
]

は完全に同じです。

よって

[
L(q,\delta)
]

も cell 内で (\delta) に依存しません。

---

## strict profile refinement

(\delta_9) を含む cell を (C_0) とします。

その左右の

[
C_{-1},C_{+1}
]

も求めます。

各 cell について代表 midpoint (\bar\delta_C) を選び、

[
\boxed{
F(C)
====

\min_q L_0(q,\bar\delta_C)
}
]

を解きます。

重要なのは、**すべての cell の physical optimization 初期値を同じ (q_9) にすること**です。

これで探索履歴依存を避けます。

中央 cell が最良なら終了。

右が良ければ一 cell 右へ移動して、そのさらに右を評価。

左なら左へ移動。

したがって local discrete descent になります。

最終出力は scalar lag より、

[
\boxed{
\delta_{\rm rotor}\in(\delta_L,\delta_R]
}
]

を主結果にします。

併せて

* cell midpoint
* final smooth (\delta_9)
* cell width
* neighboring cell costs

を保存します。

---

# 11. KKT と exact scale gauge

これは変更しません。

smooth phase では lag を含む15-Dですが、scale gauge direction は

[
\tilde v_{\rm scale}
====================

\begin{bmatrix}
v_{\rm scale}\0
\end{bmatrix}.
]

strict cell 内では従来の14-D

[
v_{\rm scale}
]

です。

KKT は

[
\begin{bmatrix}
J_r^\top J_r+\lambda I & v\
v^\top&0
\end{bmatrix}
\begin{bmatrix}
\Delta q\
\eta
\end{bmatrix}
=============

\begin{bmatrix}
-J_r^\top r\
0
\end{bmatrix}.
]

(\eta) は Lagrange multiplier です。

matrix-exponential inertia chart もそのまま維持します。現在は

[
\Sigma
======

\Sigma_0^{1/2}
\exp S(q)
\Sigma_0^{1/2},
\qquad
J=\operatorname{tr}(\Sigma)I-\Sigma
]

としており、`expm_frechet` で解析 Jacobian も計算しています。

---

# 12. gimbal replay は diagnostic として残す

objective では

[
\theta^{\rm SG}_{\rm measured}(t)
]

を使います。

一方で従来の stateful gimbal simulation は削除せず、**post-fit diagnostic** として残します。

区間先頭の actual angle

[
\theta^{\rm meas}(t_0)
]

から、

[
\text{recorded gimbal command}
\to
\text{existing rate-limit actuator model}
\to
\theta^{\rm replay}(t)
]

を通しで計算します。

そして4関節について

[
\theta_i^{\rm raw}(t),
\qquad
\theta_i^{\rm SG}(t),
\qquad
\theta_i^{\rm replay}(t)
]

を一枚に重ねます。

保存する diagnostic は

[
{\rm RMSE}_i
============

\sqrt{
\frac1N\sum_k
(\theta_i^{\rm replay}-\theta_i^{\rm SG})^2
},
]

max error、rate-limit active counts などです。

これは parameter objective へは一切戻しません。

IMU comparison と同じ位置づけです。

---

# 13. 新 fixed ablation

現行27ケースのうち、split lag や actuator propagation のように今回の設計で意味がなくなったケースを置き換えます。現行の fixed case 群はコード上も `lag_common_estimated`, `lag_split_estimated`, `lag_split_strict_only`, `actuator_stateful`, `actuator_direct_command` などを含んでいます。

新しい fixed set は次の **21 cases** を提案します。

|  # | case                                    | default からの変更                                           | 目的                      |
| -: | --------------------------------------- | ------------------------------------------------------- | ----------------------- |
|  1 | `default`                               | なし                                                      | 新基準線                    |
|  2 | `cov_full`                              | full covariance                                         | identity との比較           |
|  3 | `cov_diagonal`                          | diagonal covariance                                     | covariance cross 項      |
|  4 | `cov_block_s_alpha`                     | (s/\alpha) block                                        | 並進–回転 cross             |
|  5 | `cov_full_no_R_uncertainty_in_s`        | rotation uncertainty 除外                                 | covariance 構成診断         |
|  6 | `cov_full_no_position_rotation_cross`   | position–rotation cross 除外                              | 同上                      |
|  7 | `cov_global_full`                       | global covariance                                       | local covariance の影響    |
|  8 | `gimbal_measured_linear`                | SGせず実測値を補間                                              | gimbal smoothing の影響    |
|  9 | `gimbal_command_replay`                 | measured の代わりに command replay                           | actual gimbal 使用の寄与     |
| 10 | `lag_zero`                              | (\delta=0)                                              | lag 推定の寄与               |
| 11 | `lag_fixed_one_period`                  | (\delta=T_r) 固定                                         | lag optimization の寄与    |
| 12 | `lag_strict_only`                       | smooth continuation 無し                                  | smooth continuation の価値 |
| 13 | `lag_pow2_depth_6`                      | (k_{\max}=6)                                            | continuation depth      |
| 14 | `lag_pow2_depth_12`                     | (k_{\max}=12)                                           | convergence 確認          |
| 15 | `lag_legacy_smooth_schedule`            | 旧相当の広い schedule                                         | 新 (2^{-k}) との比較         |
| 16 | `so3_naive_rotation_vector_derivatives` | geometric correction 無し                                 | geometric SO(3) の寄与     |
| 17 | `kkt_scale_negative`                    | gauge 初期 offset < 0                                     | exact gauge invariance  |
| 18 | `kkt_scale_positive`                    | gauge 初期 offset > 0                                     | exact gauge invariance  |
| 19 | `kkt_disabled`                          | KKT 無し                                                  | gauge handling の寄与      |
| 20 | `solver_standard_gauge_least_squares`   | standard solver                                         | custom KKT-LM 比較        |
| 21 | `naive_all`                             | lag zero + command-gimbal replay + naive SO(3) + no KKT | composite baseline      |

`external_wrench_raw_only` と `external_wrench_trajectory_fitted` は fixed ablation から外してよいと思います。

理由は、どちらも parameter optimization **後**の処理であり、現在ですら standardized replay は全 completed case で生成されています。

今後は全 case に

* raw wrench diagnostic
* trajectory-fitted external wrench diagnostic

の両方を常時出します。

つまり情報は消しません。重複 run だけ消します。

---

# 14. focused sweeps

fixed 21 cases とは別に、従来通り明示的 sweep config を使います。

### SG window

[
W=
0.5,\ 1.0,\ 1.5,\ 2.0~{\rm s}
]

について、

[
\text{identity},\qquad\text{full}
]

の2通り。

計8ケースです。

このとき必ず

[
W_{\rm pose}=W_{\rm gimbal}=W.
]

これで前回の window 結果を、新 measured-gimbal estimator でも再検証できます。

### lag initial seed

default は

[
T_r
]

ですが、

[
0.5T_r,\quad T_r,\quad2T_r
]

から始める3ケースも focused validation として用意します。

これは global optimum を探すためではなく、

> 今回の local continuation が自然な近傍で同じ strict cell に収束するか

を見るためです。

`lag_bounds` sweep は完全に廃止します。

現行 ablation runner には `lag_initials`, `lag_bounds`, `smooth_schedules` sweep が存在します。
これを

```text
lag_initial_multipliers
lag_continuation_depths
```

へ整理します。

---

# 15. ridge は14-Dのまま保存し、比較は13-D quotient で行う

各 bag の final strict solution について、

[
J_b= \frac{\partial r_b}{\partial q}
]

を保存し、

[
J_b=U_bS_bV_b^\top
]

を従来通り計算します。

machine rank だけを使い、exact common-scale direction が唯一の null かを確認します。

さらに今回は、bag 間比較を行うために exact scale direction (v) に直交する basis

[
Q\in\mathbb R^{14\times13},
\qquad
Q^\top Q=I,\quad Q^\top v=0
]

を一つ固定します。

各 bag を

[
z_b=Q^\top q_b
]

へ写し、

[
H_b^{(13)}
==========

Q^\top J_b^\top J_bQ
]

を保存します。

これが「scale を quotient した parameter space」です。

---

# 16. 3 bag の相互整合性を新しい主 diagnostic にする

ここは今回追加したいです。

3 bag を独立推定したあと、

```text
cross_bag_consensus.py
```

を一回走らせます。

estimator 自体は single-bag のままです。

まず各 bag の gauge-invariant physical quantities

[
\boxed{
J/m,\qquad f_i/m,\qquad c
}
]

を並べます。

次に pairwise distribution distance を計算します。

overlap-corrected covariance を13-D quotient に写したものを

[
C_b
]

とすれば、

[
\boxed{
d_{ij}^2
========

(z_i-z_j)^\top
(C_i+C_j)^\dagger
(z_i-z_j)
}
]

を出せます。

これなら nominal は一切関係ありません。

見るべきなのは

[
d_{12},d_{13},d_{23}
]

です。

---

## cross-evaluation matrix

さらにかなり重要なのがこれです。

bag (j) から得た physical parameter

[
q_j
]

を bag (i) に持っていき、その bag の rotor lag だけ strict-cell で局所 profile して、

[
L_i(q_j)
]

を計算します。

つまり

[
\begin{pmatrix}
L_1(q_1)&L_1(q_2)&L_1(q_3)\
L_2(q_1)&L_2(q_2)&L_2(q_3)\
L_3(q_1)&L_3(q_2)&L_3(q_3)
\end{pmatrix}
]

を作ります。

実際には各行の optimum を引いて、

[
\boxed{
\Delta L_{ij}
=============

L_i(q_j)-L_i(q_i)
}
]

を表示します。

これが小さければ、

> 数値上の point estimate は違うが、各 bag の ridge が互いの estimate を許容している

ということが直接分かります。

これは単純な「parameter 値が似ているか」よりかなり有益です。

---

# 17. gimbal / IMU / continuation の新レポート

各 completed case の PDF は、概ね次の構成に更新します。

**Page 1 — SG trajectory**

pose / orientation / derived motion の概要。

**Page 2 — acceleration residual**

[
s^{obs};\text{vs};\hat s,
\qquad
\alpha^{obs};\text{vs};\hat\alpha.
]

**Page 3 — gimbal measurement audit**

4関節について、

[
\theta^{raw},
\quad
\theta^{SG},
\quad
\theta^{replay}.
]

RMSE と max error 併記。

**Page 4 — IMU comparison**

現在同様、measured gyro / specific force と post-fit prediction。

**Page 5 — (2^{-k}) continuation**

横軸

[
k\quad\text{または}\quad\epsilon=2^{-k}
]

で、

[
\delta_k,
\quad
L_{\epsilon_k},
\quad
L_0,
\quad
|L_{\epsilon_k}-L_0|,
\quad
\max|T^\epsilon-T^0|.
]

**Page 6 — strict lag cells**

最終 cell と左右近傍の cost。

[
(\delta_L,\delta_R]
]

を図示。

**Page 7 — parameter estimate**

absolute values とともに

[
J/m,\quad f/m,\quad c
]

を必ず出す。

**Page 8 — ridge spectrum**

singular values、right singular vectors、parameter displacement。

**Page 9 — covariance / uncertainty**

optimization covariance と reference full covariance を明確に分ける。

**Page 10 — wrench / closure**

raw residual wrench、fitted external wrench、Newton–Euler closure。

---

# 18. NPZ / JSON に追加するもの

新たに最低限、

```text
gimbal_raw_time
gimbal_raw_angle
gimbal_sg_angle
gimbal_command_replay_angle

lag_continuation_epsilon
lag_continuation_rotor_lag
lag_continuation_smooth_cost
lag_continuation_strict_cost
lag_continuation_command_max_error
lag_continuation_physical_coordinate

strict_lag_cell_lower
strict_lag_cell_upper
strict_lag_cell_representative
strict_lag_neighbor_cells
strict_lag_neighbor_costs

quotient_basis
quotient_coordinate
quotient_jtj
```

を保存します。

各 stage の (q_k) まで残すので、後から continuation path 自体も解析できます。

---

# 19. コード構成

実装上は次の分離がよいと思います。

`single_bag_savgol_core.py` は physical model と residual evaluation を担当します。`SingleBagDataset` に `measured_gimbal_raw` / `measured_gimbal_sg` を追加し、objective path から `gimbal_lag` と stateful gimbal propagation を除きます。

`gimbal_savgol.py` を新設し、irregular timestamp の4-channel scalar SG を担当させます。

`smooth_command.py` は `width_fraction` ではなく `epsilon` semantics に整理します。

`rotor_lag.py` を新設し、

[
2^{-k}\text{ continuation}
]

と strict-ZOH cell construction/refinement をまとめます。

`single_bag_savgol_estimator.py` では旧

```text
--initial-gimbal-lag
--fixed-gimbal-lag
--lag-bounds
--strict-alternations
--actuator-propagation
```

を default scientific path から削除します。

代わりに概ね

```text
--lag-mode estimated|zero|fixed
--initial-rotor-lag
--fixed-rotor-lag
--lag-continuation-depth 9
--gimbal-source measured_sg|measured_linear|command_replay
```

とします。

`single_bag_savgol_ablation.py` は先ほどの21ケースへ更新します。

最後に `single_bag_cross_bag_consensus.py` を追加します。

---

# 20. 実装テスト

ここは性能テストではなく correctness test に限定します。

特に重要なのは、

1. irregular-time scalar SG が既知 polynomial gimbal trajectory を正確に復元する。
2. default objective が全時刻で measured gimbal SG を使用している。
3. default global Jacobian が14 physical + 1 rotor lag の15列である。
4. (\epsilon\to0) で switch 以外の点について smooth command が exact ZOH に収束する。
5. 同じ strict cell 内の任意の2 lag で rotor command history が完全一致する。
6. 同じ cell 内なら同じ (q) に対する residual / cost が一致する。
7. cell boundary を跨ぐと期待した command index だけが変わる。
8. manual lag bound が scientific path に存在しない。
9. scale regauging
   [
   q\mapsto q+c v_{\rm scale}
   ]
   で residual が machine precision まで不変。
10. (k=6,9,12) continuation 後の strict cell result が synthetic problem で一致する。
11. no finite differences。
12. 一つの ablation case が失敗しても他 case が継続する。

という内容です。

---

# 21. この revision で何を判定するか

この study で特に見るべきものは、単一の「最小 objective」ではありません。

主要な判定軸は、

[
\boxed{
\begin{aligned}
&\text{SG-derived acceleration fit}\
&\text{raw residual wrench}\
&\text{gimbal measured/replay consistency}\
&\text{smooth}\to\text{strict convergence}\
&\text{strict lag cell stability}\
&\text{ridge spectrum}\
&\text{scale-free parameter consistency}\
&\text{3 bag pairwise information distance}\
&\text{3 bag cross-evaluation }\Delta L_{ij}
\end{aligned}}
]

です。

特に最後の

[
\Delta L_{ij}
]

が重要です。

仮に3 bag の point estimate がかなり違っていても、

[
\Delta L_{ij}\ll L_i
]

なら、

> **各 bag は同じ parameter region を実質的に許しているが、異なる ridge 上の代表点へ optimizer が落ちただけ**

というかなり強い結果になります。

逆に cross-evaluation が大きければ、本当に bag 間で説明できない差があり、そのとき初めて model discrepancy や flight-condition dependence を考えるべきです。

この形であれば、今回決めた **measured gimbal 全使用、1-D rotor lag、(2^{-k}) continuation、strict-ZOH exact cell、scale quotient、bag 間分布比較**が全部一つの study にまとまります。
