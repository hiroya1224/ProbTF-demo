# `grape-param-estim` Q&A

このファイルは、このスレッドにおけるユーザーの質問とアシスタントの応答を、時系列に沿って記録する。
resume: `codex resume 019f92a5-1cc7-7fd1-980d-e8696563a7c2`

## 2026-07-24: mocap pose の微分と慣性パラメータの観測可能性

### ユーザー

> こんにちは。/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim の実装に目を通しましたが、mocap の微分をしている理由がわかりませんでした。位置姿勢だけの情報では推定は厳しいものでしょうか？  
> なお、このスレッドでの私の質問・あなたの応答は、完全に追えるように、/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/lectures/QandA.md 内にすべて記録していってください。

### アシスタント（作業中の応答 1）

> 承知しました。まず `grape-param-estim` の実装とリポジトリ内の指示を確認し、mocap 微分がどの観測量・推定式に使われているかをコード上で追います。そのうえで、今回のご質問と最終回答を `lectures/QandA.md` に追記します。以後のやり取りも同じ形式で記録します。

### アシスタント（作業中の応答 2）

> 設計文書の記述だけでなく、実際の観測モデルを確認中です。現時点では、微分は単なる補助処理ではなく、pose から並進加速度・角速度・角加速度を作り、入力 wrench とニュートン–オイラー式を照合するために必要になっています。ただし、「pose だけでは不可能」という意味ではないので、識別可能性とノイズの問題を分けて整理します。

### アシスタント（作業中の応答 3）

> コード経路まで確認できました。結論は、「mocap が pose しか返さないから微分している」のではなく、この推定器が pose の再現誤差を見る forward simulation 型ではなく、運動から得た \(s,\omega,\alpha\) と actuator wrench の整合性を見る inverse dynamics 型だからです。静止 hover で見える量、6-DoF 運動で初めて見える量、そもそも wrench がなければ見えない量を分けて回答にまとめます。

### アシスタント（最終回答）

結論からいうと、**mocap の位置・姿勢の時系列と、既知の actuator wrench があれば、数値微分を明示せずに推定する方法もあります**。ただし現在の実装は inverse dynamics 型なので、位置・姿勢を \(s,\omega,\alpha\) に変換するために微分が必要です。瞬間的な pose だけ、または入力 wrench のない pose 時系列だけから、10 個の慣性パラメータをすべて求めるのは原理的に困難です。

#### 現実装が mocap を微分する理由

この particle filter は機体の状態を時間発展させる filter ではなく、静的なパラメータ

\[
\theta=[m,c_x,c_y,c_z,I_{xx},I_{xy},I_{xz},I_{yy},I_{yz},I_{zz}]
\]

だけを保持する filter です。mocap pose の時系列から

\[
s=R^\top(\ddot p-g),\qquad \omega,\qquad \alpha=\dot\omega
\]

を作り、各 particle が予測する wrench

\[
F=ms+\alpha\times h+\omega\times(\omega\times h),
\]

\[
\tau=J_O\alpha+\omega\times(J_O\omega)+h\times s
\]

を、観測または command から再構成した actuator wrench と比較しています。ここで \(h=mc\) です。

実際に、位置の二階微分と姿勢からの \(\omega,\alpha\) は [kinematics.py](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/src/grape_param_estim/kinematics.py:423) で生成され、上の剛体式は [dynamics.py](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/src/grape_param_estim/dynamics.py:233) に実装されています。それらが [estimate_grape_bag.py](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/scripts/estimate_grape_bag.py:550) で観測 batch に入り、[particle_filter.py](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/src/grape_param_estim/particle_filter.py:230) で wrench residual の尤度になります。

つまり、絶対位置 \(p\) 自体は慣性パラメータの尤度に直接入りません。姿勢 \(R\) は重力を body frame に移すために使われますが、慣性を見分ける主要な情報は「既知の力・トルクに対して、どのように加速・回転したか」です。

#### 位置・姿勢からどこまで推定できるか

| 利用できる情報 | 推定可能性 |
|---|---|
| ある時刻の位置・姿勢だけ | 慣性パラメータはほぼ推定不能 |
| 静止 pose と校正済み wrench | 質量と、重力方向に直交する CoG 成分は推定可能。水平姿勢なら通常 \(c_x,c_y\)。\(c_z\) と慣性テンソルは見えない |
| 十分に励起された 6-DoF pose 時系列と校正済み wrench | 全パラメータを推定できる可能性がある。ただし観測 Jacobian の rank 10 が必要 |
| pose 時系列だけで入力 wrench が不明 | 一般には絶対的な質量・慣性スケールを識別できない |
| pose 時系列と未校正の thrust command | 質量と thrust scale などが混同される |

静止 hover では \(\omega=\alpha=\ddot p=0\) なので、

\[
F\simeq -mR^\top g,\qquad \tau\simeq h\times(-R^\top g)
\]

だけが残ります。慣性テンソル \(J\) は式から消えるため、長時間 hover を観測しても full inertia は決まりません。また \(h\times s\) は \(s\) と平行な CoG 成分を消してしまうため、水平 hover だけでは \(c_z\) も観測できません。

一方、並進・回転を十分に励起すると、\(\alpha\times h\)、\(\omega\times(\omega\times h)\)、\(J\alpha\)、\(\omega\times J\omega\) が現れ、CoG と慣性の情報が増えます。現在の実装が excitation rank を計算しているのはこのためです。

#### 微分せず、pose を直接使う方法

方法はあります。各パラメータ particle について forward dynamics を解き、

\[
(p_k,R_k,v_k,\omega_k)\longrightarrow(p_{k+1},R_{k+1},v_{k+1},\omega_{k+1})
\]

と積分して、予測 pose と mocap pose の残差を尤度にできます。あるいは、速度・加速度を潜在状態にした batch smoother、factor graph、Gaussian-process trajectory などで pose・入力・パラメータを同時推定できます。

この場合、観測としては pose だけを使用できますが、運動の曲率、すなわち速度・加速度の情報が時系列内で暗黙に使われます。物理的に必要な微分情報がなくなるわけではありません。

利点は、noisy pose を単純に二階微分することを避け、mocap uncertainty を状態推定へ自然に入れやすいことです。一方で、初期速度・初期角速度も推定する必要があり、particle ごとの積分コストが増えます。また actuator の遅れ、drag、外力、時刻ずれなどの小さなモデル誤差が積分中に蓄積し、pose residual を悪化させやすいという難しさがあります。現在の inverse-dynamics 方式は局所的で計算が軽く、長時間の積分 drift を避けられる代わりに、微分ノイズを受けます。

#### この実装についての注意点

現在は 50 Hz の pose に対して 51 点、すなわち約 1.02 秒幅の Savitzky–Golay 局所多項式を使い、平滑化と微分を同時に行っています。この設定と `1 cm / 1 deg` の独立 sample 近似からコードが算出する標準偏差は、およそ次の値です。

- 並進加速度: `0.036 m/s²`
- 角速度: `0.0208 rad/s`
- 角加速度: `0.0631 rad/s²`

ただし、その derivative covariance は現在の particle likelihood に伝播されていません。尤度は固定の `0.80 N`、`0.06 N m` を使っています。このため、現実データでは微分値を確定入力として扱うことによる bias や、posterior の過信が残り得ます。

私なら、この synthetic sanity check には現在の inverse-dynamics 方式を残します。一方、実 bag から物理パラメータを求める目的なら、優先順位は次のようにします。

1. 実 thrust/wrench の校正、scale と時間遅れの同定
2. hover だけでなく、並進・回転を独立に励起する trajectory の収録
3. mocap と IMU を使った smoothing、または pose を直接観測する batch state/parameter estimation
4. 少なくとも現方式の derivative uncertainty を particle ごとの wrench covariance へ伝播

したがって、問題は単に「位置姿勢だけでは精度が足りない」ことではありません。**入力 wrench、時間方向の運動情報、十分な励起がそろえば pose センサだけでも推定可能ですが、現在の filter 構造ではその運動情報を明示的な微分として取り出している**、というのが正確です。

今回の質問、作業中の応答、最終回答は [QandA.md](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/lectures/QandA.md) に記録しました。以後もこのスレッドの各質問と応答を追記します。

## 2026-07-24: ProbTF による時間伝播と rosbag からの actuator wrench 復元可能性

### ユーザー

> なるほど。/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/probtf_orientation_demo　でやっていた微分の方法がつかえるかなと思っていて、そしたら確率も加味した時間方向の誤差伝播も ProbTF の文脈に入るかなあと、素人ながら考えています。あと、calibrated wrench がないって話ですが、rosbag 記載のデータを使えば現在の controller の実装をもとに復元できるのか、それすらもできない（rosbag から本質的に情報が欠落している）のか、そこを明らかにしておきたいです。

### アシスタント（作業中の応答 1）

> はい、その二点はかなり本質的です。まず `probtf_orientation_demo` の「微分」が、単なる有限差分なのか、時間相関を含む確率変数として導関数を作っているのかを実装から確認します。次に、Grape の controller が command から実機へ何を送っており、bag に command・角度・回転数・電流・推力推定値などのどこまでが残っているかを突き合わせ、「一意に復元可能」「仮定付きで復元可能」「情報欠落で不可能」に分けます。今回の質問と途中経過、最終回答も `QandA.md` に追記します。

### アシスタント（作業中の応答 2）

> 最初の点には一つ重要な区別がありました。`probtf_orientation_demo` は mocap pose を微分する実装ではなく、IMU の角速度を積分して姿勢分布を時間伝播する実装です。ただし、角速度 covariance を quaternion increment の分布へ写し、Bingham の二次モーメントを時系列で伝播しているので、「微分量を確率変数として扱い、時間方向の不確かさを伝播する」という発想は Grape 側にも十分応用できます。次は controller と実 bag の記録内容を照合します。

### アシスタント（作業中の応答 3）

> 実 bag を一つ実測確認したところ、「wrench が完全に失われている」という状態ではありません。bag には高位の各 rotor 目標推力だけでなく、実際に送られた PWM、battery/ESC voltage、ESC の measured RPM・current、gimbal 実角度、さらに PWM–thrust 校正係数まで記録されています。したがって command wrench と、静的校正モデル上の applied wrench はかなり再構成できます。一方、空中での瞬時の実推力そのものを直接測った値はなく、そこには propeller の動特性・流入・機体間干渉などの未観測要素が残ります。今、その境界と現推定器が bag を正しく読めているかを数値で監査しています。

### アシスタント（最終回答）

結論は二つです。

1. ProbTF の時間伝播を利用する発想は有効です。ただし `probtf_orientation_demo` が行っているのは pose の微分ではなく、uncertain な角速度の積分です。mocap から速度・加速度分布を作るには、時間相関を保持する smoother を一段追加する必要があります。
2. rosbag から controller が意図した nominal wrench はほぼ復元できます。PWM 校正上の wrench も復元可能です。しかし、空中で機体へ実際に加わった瞬時 wrench を一意に復元する情報は欠けています。したがって「完全に不明」でも「calibrated wrench が記録済み」でもなく、**不確かさ付き actuator-wrench evidence を構成できる**という位置づけです。

#### `probtf_orientation_demo` との関係

このデモは、IMU の角速度平均 \(\omega\) と covariance \(\Sigma_\omega\) を受け取り、

\[
\delta q=\operatorname{Exp}_q(\omega\Delta t)
\]

の二次モーメントを cubature で求めています。その後、prior orientation の二次モーメントと quaternion product を取り、一つの Bingham 分布へ moment-match しています。実装は [orientation_imu.py](/home/leus/catkin_ws/src/ProbTF-demo/ros/core/probtf_core/src/probtf_estimators/orientation_imu.py:112) と [orientation_imu.py](/home/leus/catkin_ws/src/ProbTF-demo/ros/core/probtf_core/src/probtf_estimators/orientation_imu.py:147)、呼び出し側は [orientation_filter_node.py](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/probtf_orientation_demo/nodes/orientation_filter_node.py:163) です。

したがって、これはまさに「時間方向への確率伝播」ですが、向きは

\[
\text{angular velocity distribution}
\longrightarrow
\text{orientation distribution}
\]

です。Grape でいま必要としている

\[
\text{pose time series}
\longrightarrow
(v,a,\omega,\alpha)\text{ distribution}
\]

とは逆問題です。

ここで各時刻の ProbTF marginal を独立に差分してはいけません。例えば速度差分なら、

\[
\operatorname{Cov}\!\left(\frac{p_{k+1}-p_k}{\Delta t}\right)
=
\frac{
\Sigma_{k+1}+\Sigma_k
-C_{k+1,k}-C_{k+1,k}^{\mathsf T}
}{\Delta t^2}
\]

となり、時刻間 cross-covariance \(C_{k+1,k}\) が必要です。二階微分では三時刻以上の cross-covariance が必要になります。姿勢も同様で、独立な Bingham marginal 二つだけでは \(q_k^{-1}q_{k+1}\) の正しい分布は決まりません。

そのため適切な構成は、次のようになります。

- latent state として \(p,v,a,R,\omega,\alpha\)、IMU bias を持つ fixed-lag/batch smoother を作る。
- mocap pose、gyro、accelerometer を evidence として融合する。
- smoother 内では時刻間相関を保持する。
- 各時刻の pose marginal は ProbTF として公開する。
- \(s,\omega,\alpha\) の joint marginal は dynamics estimator に渡し、wrench residual へ uncertainty を伝播する。

ProbTF の transform edge 自体へ速度や加速度を押し込むのではなく、temporal estimator が joint trajectory distribution を管理し、その pose marginal を ProbTF に materialize するのが自然です。なお orientation demo の accelerometer は gravity evidence として使われますが、飛行中は並進加速度も含むため、Grape へそのまま適用せず、dynamic acceleration のモデル化または gate が必要です。

#### rosbag に残っている actuator 情報

代表として
`20260612_grape_hovering_7_2026-06-12-17-41-34.bag`
を詳細に監査し、さらに収録済み 12 bag の topic 有無を確認しました。

| 情報 | bag の topic | 意味 |
|---|---|---|
| rotor 目標推力 | `/gimbalrotor/four_axes/command` | controller の nominal thrust。11/12 bag にあり、各 sample は4要素 |
| PWM–thrust calibration | `/gimbalrotor/motor_info` | 5電圧点の多項式、上下限、最小推力。全12 bag にあり |
| FC が出した PWM | `/gimbalrotor/motor_pwms` | firmware が計算し送信した motor command。全12 bag にあり |
| battery voltage | `/gimbalrotor/battery_voltage_status` | PWM変換時の電圧補正に使用可能 |
| ESC telemetry | `/gimbalrotor/esc_telem` | measured RPM、voltage、current、CRC。全12 bag にあり |
| gimbal 実角度 | `/gimbalrotor/joint_states`、`/servo/states` | thrust direction の復元に使用可能 |
| geometry | `/tf`、`/tf_static` | rotor位置、gimbal回転、thrust offset、`fc` の位置 |

controller は `gimbal_calc_in_fc: false` なので、[gimbalrotor_controller.cpp](/home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/src/control/gimbalrotor_controller.cpp:295) が `target_full_thrust_` を4要素の `base_thrust` として送っています。firmware は [attitude_control.cpp](/home/leus/catkin_ws/src/jsk_aerial_robot/aerial_robot_nerve/spinal/mcu_project/lib/Jsk_Lib/flight_control/attitude/attitude_control.cpp:878) の電圧依存多項式で thrust を PWM に変換し、同ファイルの [PWM記録処理](/home/leus/catkin_ws/src/jsk_aerial_robot/aerial_robot_nerve/spinal/mcu_project/lib/Jsk_Lib/flight_control/attitude/attitude_control.cpp:1125) で `/motor_pwms` を出しています。

実 bag の HOVER 区間で、`base_thrust`、bag 内の `motor_info`、battery voltage から PWM を再計算したところ、約 `-6 ms` の伝送ずれを合わせれば、記録 PWM との RMSE は `1.06 µs` でした。motor ごとの RMSE も `1.05–1.08 µs` です。これは量子化程度の差であり、少なくとも

\[
\text{FourAxisCommand}
\rightarrow
\text{voltage compensation}
\rightarrow
\text{PWM}
\]

の controller/firmware 経路は bag からほぼ完全に再現できています。この HOVER 区間では `/gimbalrotor/rpy/pid` も全要素 zero で、隠れた low-level roll/pitch/yaw thrust correction は確認されませんでした。

また bag の TF にある rotor 位置、thrust offset、`fc` 位置は、現在の [grape_geometry.py](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/src/grape_param_estim/grape_geometry.py:16) の数値と一致しています。そのため、nominal rotor thrust と実測 gimbal angle から、force と lever-arm moment は復元できます。

#### それでも一意に復元できないもの

`/motor_pwms` は実際に送った command ですが、force sensor ではありません。`/esc_telem` の RPM と current は実測ですが、bag 内には次がありません。

- 空中状態での \(T=f(\mathrm{RPM},V,\text{inflow},\rho)\) の校正モデル
- rotor ごとの thrust scale と時定数
- propeller 間・機体との空力干渉
- induced flow、機体速度、地面効果による補正
- motor/prop の reaction torque を RPM などから決める動的校正
- 外力、ケーブル力、空力 drag の直接観測

`motor_info` の多項式は「指定 thrust からどの PWM を送るか」という静的 bench calibration です。これを逆に解けば static-calibration-equivalent thrust は得られますが、飛行中の actual thrust と同一とは限りません。ESC RPM から actual thrust に近づけることはできますが、そのための RPM–thrust calibration は bag 外から与えるか、新たに同定する必要があります。

さらに reaction moment の `m_f_rate=-0.0181` と rotor spin direction は現在の config/URDF にはありますが、`motor_info` message 自体には格納されていません。bag だけを単独の完全な provenance とみなすことはできません。記録時の firmware commit/hash も bag にないため、現在の source と flashed binary が同一だったことを形式的には証明できません。ただし、上述の PWM 一致は、対象経路の実際の挙動が現在の実装と一致する強い実証になっています。

#### 現在の estimator の位置づけ

現在の [estimate_grape_bag.py](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/scripts/estimate_grape_bag.py:179) は `base_thrust` と実測 gimbal angle だけを使います。`motor_info`、PWM、battery voltage、RPM、current はまだ使っていません。したがって `command_as_force_effective` という provenance 名は正しく、`calibrated_wrench` と呼ぶべきではありません。

監査した HOVER 区間では4 rotor の command 合計中央値が `28.77 N`、見かけ質量が `2.934 kg` でした。一方、現在の URDF composite mass `2.3516 kg` の重量は `23.06 N` です。水平・定常 hover と仮定すると比は約 `0.802` で、nominal command と actual lift の間に無視できない scale discrepancy があることを示します。これは PWM までの復元誤差ではなく、PWM 以後の actuator/airframe 系または物理モデル側に残る差です。

したがって、復元可能性は次の三段階に整理できます。

| 対象 | 判定 |
|---|---|
| controller が意図した nominal wrench | 復元可能 |
| 記録 PWM と静的校正に対応する equivalent wrench | 仮定付きで復元可能 |
| 飛行中に剛体へ実際に作用した瞬時 wrench | bag だけからの一意復元は不可能 |

ただし最後の段階も、単一の確定値にできないだけです。command、PWM、RPM、voltage、current、gimbal angle をまとめ、thrust scale・time lag・RPM係数を latent variable にした確率的 actuator model は作れます。これは先ほどの temporal smoother とよく整合し、

\[
p(s,\omega,\alpha,w_a,\theta\mid
\text{mocap, IMU, command, PWM, RPM})
\]

を扱う方向へ発展できます。ただし既知 mass、load-cell calibration、または既知 RPM–thrust curve のどれも入れなければ、共通 thrust scale と mass の gauge は残ります。

要するに、rosbag の情報は捨てるには惜しいほど豊富で、現在の estimator はまだその一部しか使っていません。一方で actual wrench の絶対 scale を確定する最後の校正情報は本質的に欠けています。ProbTF の時間伝播と組み合わせるなら、「wrench を復元した確定値」とするのではなく、各 source と近似 provenance を保った **temporal wrench evidence distribution** として dynamics likelihood へ渡すのが最も筋のよい設計です。

今回の質問、作業中の応答、最終回答も [QandA.md](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/lectures/QandA.md) に追記しました。

## 2026-07-24: PID が吸収するモデル誤差を effective parameter として推定する案

### ユーザー

> ああ、そうそう、なんでこの話が始まったのかというと、URDF から予想される質量で飛ばそうとすると、内部のモデル化がいい加減だったりとかで、全然飛ばないんですよね。それで、PID でだましだましやっているみたいな話を、実際に実験をやっている学生から聞いていました。  
> それと、運動方程式から予想できるように、質量単体で合わせるのって多分無理で、何かの比になりますよね？端的な例だと、ma = f は a = f/m と等価で、fとmを同時に推定しようとすると f/m という比が一定であるようなやつ全部が正しい、という状態になる、とか。  
> それで、ここからが案なのですが、もしも PID 制御のパラメータをいじって解決する（PID制御のパラメータから、質量らしきものを取り出すことができる）のであれば、そのPIDパラメータを推定対象として、その考察というか、補正質量情報（や、スラスタの推力との変換情報：これも結構問題があるみたいで、プロペラの経年劣化で全然特性が変わっちゃって、交換したらあっけなく飛んだ、というような話も聞きます）をあとで抽出する、というような運用のほうが正しい気がしています。  
> どうでしょう？この案で回りそうですかね

### アシスタント（作業中の応答 1）

> その案の核心、つまり「真の質量を無理に一個へ押し込まず、閉ループで実際に効いている \(k_{\mathrm{thrust}}/m\) や遅れを同定し、あとから物理解釈する」はかなり筋がよいです。ただし、手で調整した PID gain 自体を物理パラメータとして推定するのは危険で、同定対象は controller を含む閉ループの effective parameter に分けた方がよさそうです。Grape の PID が acceleration、force、PWM のどの単位で動いているかと、積分項・出力 command が bag にどこまで残っているかを確認して、回る案と識別不能になる案の境界を整理します。今回も全応答を `QandA.md` に追記します。

### アシスタント（作業中の応答 2）

> 実装を見ると、学生さんの説明とかなり整合しています。Grape の z 制御には明示的な \(+g\) feedforward が見当たらず、位置誤差の積分項が hover に必要な加速度指令を蓄えていきます。その指令を URDF mass で nominal thrust に変換するので、定常時の I 項は「質量」ではなく、実質的に \(m_{\text{true}}/(k_T m_{\text{model}})\) を吸収します。つまり I 項は有用な診断量ですが、設定した \(K_i\) 自体から質量が出るわけではありません。bag の PID 内訳でこの挙動を数値確認します。

### アシスタント（作業中の応答 3）

> 数値でも確認できました。対象 bag の HOVER 中、z-PID の中央値は total `11.998 m/s²`、I 項 `11.991 m/s²`で、P/D はほぼゼロ、feedforward も実質ゼロでした。URDF mass `2.3516 kg` を allocation に使っているため、I 項から得る補正係数は \(11.991/g=1.223\)、すなわち effective mass は約 `2.875 kg` です。これは先ほど command から得た `2.85–2.95 kg` と一致します。したがって「PID が隠れた質量・推力誤差を吸収している」という見立ては正しいですが、抽出すべきなのは \(K_i\) ではなく、収束した I 出力と閉ループ応答です。

### アシスタント（最終回答）

はい、**案の中心部分は回ると思います**。ただし一か所だけ言い換えるのが重要です。

> PID gain を物理パラメータとして推定する  
> ではなく、  
> **PID が現在暗黙に補償している effective plant parameter を明示的に推定する**

とすべきです。

#### ご指摘どおり、単独の mass は観測できません

実推力を nominal command \(u\) の定数倍

\[
F_{\mathrm{actual}}=k_Tu
\]

とすると、鉛直方向は概略

\[
a_z=\frac{k_T}{m_{\mathrm{true}}}u-g
\]

です。飛行応答から直接見えるのは

\[
\gamma=\frac{k_T}{m_{\mathrm{true}}}
\]

であり、\(k_T\) と \(m_{\mathrm{true}}\) を別々には決められません。ご提示の \(f/m\) の議論そのものです。

別の表し方として

\[
m_{\mathrm{eff}}=\frac{m_{\mathrm{true}}}{k_T}
\]

を導入すると、

\[
a_z=\frac{u}{m_{\mathrm{eff}}}-g
\]

となります。この \(m_{\mathrm{eff}}\) は kg の単位を持ち、controller が必要とする「補正質量」にちょうど対応します。ただし、これは真の質量ではなく、mass と thrust-scale error をまとめた effective mass です。

#### 現在の PID は実際にそれを暗黙推定しています

Grape controller の PID は [pid.h](/home/leus/catkin_ws/src/jsk_aerial_robot/aerial_robot_control/include/aerial_robot_control/control/utils/pid.h:63) で

\[
a_{\mathrm{cmd}}
=K_pe+K_i\int e\,dt+K_d\dot e+a_{\mathrm{ff}}
\]

を作ります。z 軸では [pose_linear_controller.cpp](/home/leus/catkin_ws/src/jsk_aerial_robot/aerial_robot_control/src/control/base/pose_linear_controller.cpp:259) からこの出力が呼ばれ、Gimbalrotor の allocation は URDF の model mass \(m_{\mathrm{model}}\) を使って nominal thrust へ変換します。

\[
u=m_{\mathrm{model}}a_{\mathrm{cmd}}
\]

と単純化すると、定常 hover では

\[
k_Tm_{\mathrm{model}}a_{\mathrm{cmd,hover}}
=m_{\mathrm{true}}g
\]

なので、

\[
\rho
=\frac{m_{\mathrm{true}}}{k_Tm_{\mathrm{model}}}
=\frac{a_{\mathrm{cmd,hover}}}{g},
\qquad
m_{\mathrm{eff}}
=m_{\mathrm{model}}\rho
\]

を得ます。

監査した bag の HOVER 区間では次のようになっていました。

| z-PID の量 | 中央値 |
|---|---:|
| total | `11.998 m/s²` |
| P term | `-0.0022 m/s²` |
| I term | `11.991 m/s²` |
| D term | `-0.0032 m/s²` |
| feedforward | ほぼ `0` |

つまり hover command のほぼ全てを I 項が作っています。ここから

\[
\rho=\frac{11.991}{9.80665}=1.223,
\]

\[
m_{\mathrm{eff}}
=2.3516\times1.223
\simeq2.875\ {\rm kg}
\]

となります。これは command の釣り合いから得た `2.85–2.95 kg` と整合します。逆に真の質量を URDF 値と仮定すれば、共通 thrust scale は

\[
k_T\simeq\frac{1}{1.223}=0.818
\]

です。

したがって、「URDF mass では飛ばず、I 項でだましている」という話は、bag 上でもかなり明瞭に確認できます。

#### ただし \(K_p,K_i,K_d\) 自体から mass を読むことはできません

単純な plant と PID を

\[
\ddot y=\gamma u,
\qquad
u=K_pe+K_d\dot e+K_i\int e\,dt
\]

とすると、閉ループ特性は概略

\[
s^3+\gamma K_ds^2+\gamma K_ps+\gamma K_i=0
\]

になります。応答から見えるのは \(\gamma K_p,\gamma K_i,\gamma K_d\) です。そのため、\(\gamma\) と gain を同時に未知にすると、

\[
\gamma' = c\gamma,\qquad
K'_{p,i,d}=\frac{1}{c}K_{p,i,d}
\]

が同じ閉ループ応答を作る gauge が生じます。

さらに PID gain は、mass だけでなく希望 bandwidth、damping、sensor noise、filter delay、actuator delay、saturation marginにも依存します。学生が手で合わせた gain から一意な質量を逆算することはできません。

一方、\(K_i\) が非zeroで飽和していなければ、定常時の **I term の出力値**は、gain 値によらず必要な定常入力へ収束します。\(K_i\) を変えると主に収束速度と内部の積分状態が変わり、必要な hover 出力そのものは変わりません。

したがって、役割は次のように分けるべきです。

- \(K_p,K_i,K_d\): 既知の controller 設定・実験 provenance
- PID の P/I/D 出力時系列: plant mismatch の診断 evidence
- \(\gamma=k_T/m\)、\(m_{\mathrm{eff}}=m/k_T\): 推定対象
- PID gain の再設計: 推定結果を使う後段処理

#### プロペラ劣化も含めた推定対象

実用上は、各 rotor に対して

\[
T_i(t)
=k_i(t)\,u_i(t-\tau_i)
\]

または一次遅れを含めて

\[
\tau_{m,i}\dot T_i+T_i
=k_i(t)\,u_i(t-\tau_i)
\]

と置くのがよいと思います。

- \(k_i\): rotor ごとの effectiveness
- \(\tau_i\): command/transport delay
- \(\tau_{m,i}\): motor・propeller の応答時定数
- \(k_i(t)\): 経年劣化を表す slow random walk または episode ごとの定数

ESC RPM を使うなら、

\[
T_i=c_{T,i}(t)\,\mathrm{RPM}_i^2
\]

の \(c_{T,i}\) も推定対象にできます。propeller 交換時には同じ parameter を連続させず、新しい episode または change point として posterior を切り替える方が自然です。

ただし全 rotor の \(k_i\) と mass を同時に一様スケール変更する gauge は残ります。飛行データだけで得やすいのは、

- 共通の effective mass \(m_{\mathrm{eff}}\)
- rotor 間の相対 effectiveness \(k_i/\bar k\)
- delay、時定数
- 劣化や故障の change point

です。true mass と絶対 thrust scale を分離するには、少なくとも一つ外部 anchor が必要です。機体を秤で量る、load cell で一つの rotor を校正する、信頼できる RPM–thrust curve を得る、のどれか一つでも大きく改善します。

#### 回る運用案

私は次の二時間スケール構成にします。

1. fast loop は既存 PID のまま残し、安全に機体を安定化する。
2. slow estimator が mocap、IMU、command、PWM、RPM、gimbal angle と PID 内訳を使い、\(m_{\mathrm{eff}},k_i/\bar k,\tau_i,\tau_{m,i}\) を推定する。
3. 信頼度が十分なときだけ、rate limit と bounds を付けて hover feedforward と control allocation を更新する。
4. PID の I 項は、主な重力補償ではなく、残った外乱とモデル誤差だけを補償する状態へ戻す。
5. 推定値と propeller 個体・使用時間・交換イベントを episode provenance として保存する。

特に

\[
u_{\mathrm{ff}}=\hat m_{\mathrm{eff}}g
\]

を明示的な hover feedforward にすれば、離陸のたびに I 項が約 `12 m/s²` まで貯まるのを待つ必要がなくなります。これは「PID でだます」処理を、観測可能な adaptive feedforward へ昇格させることに相当します。

推定器を更新した瞬間に fast loop の値を大きく変えるのは危険なので、最初は完全 offline で bag 間比較を行い、次に shadow mode、最後に bounded adaptation と進めるべきです。また hover だけでは定常比しか得られないため、安全な小振幅 multisine や step を加えないと delay、時定数、慣性は識別できません。

#### ProbTF との役割分担

この構成では、ProbTF/temporal smoother は pose とその時間相関から \(s,\omega,\alpha\) の分布を作ります。actuator 側は command/PWM/RPM から \(w_a\) の分布を作ります。その二つを dynamics likelihood で結び、effective parameter posterior を更新します。

mass、PID gain、thrust coefficient 自体は transform ではないので ProbTF edge には入れません。ProbTF には pose marginal と推定 CoG を出し、effective parameter と actuator health は専用 message に、source、episode、propeller ID、近似方法とともに保存するのがよいです。

要するに、**「PID parameter を推定して後から質量を読む」ままでは gauge が二重になり回りにくいですが、「PID は既知の安定化器として使い、I 出力と運動応答から effective mass・actuator effectiveness を推定する」形なら十分回る案です**。しかも現在の bag だけですでに、z-I 項から `2.875 kg` という補正質量が再現できており、方向性を支持する具体的な証拠があります。

今回の質問、途中応答、最終回答も [QandA.md](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/lectures/QandA.md) に追記しました。

## 2026-07-24: ProbTF の多センサ evidence fusion と Grape 推定問題の再設計

### ユーザー

> そうだなあ、なんか普通のパラメータ推定の話になってきちゃったんで、ProbTF の話しに戻したいのですが、ProbTFの強みは、センサを入れれば（個数の規定なしに）どんどん推定に利用できる点にあります。取れるデータがあればどんどん入れられる構造になっているはずで、それを活かした推定をすべきです。今回の例だと、観測値は mocap とか IMU とかになるんですかね？ああ、そうだ、一個確認しておきたいのは、I ゲインとか言っていますが、これって mocap の結果を使って位置合わせしている理解でいいですかね？制御に使っている量から推定ってなると変な感じですが、、、まあ mocap の情報が一番信頼できるので使わない手はないですが。ちょっと今回の問題を ProbTF の利点を積極的に使ったものにするにはどうしたものかなと思っています。一応このドローンの例は、/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/deflecomp　と同じ系譜に入れる予定です。

### アシスタント（作業中の応答 1）

> ご指摘の軸で整理し直します。まず「I 項が raw mocap を直接見ているのか、mocap/IMU の融合状態を見ているのか」を controller と state estimator の接続から確定し、そのうえで `deflecomp` と同じ系譜として、何を ProbTF の観測・時間発展・潜在パラメータに置くべきかをコードに即して組み立てます。この更新を含め、今回のやり取りも `QandA.md` に記録します。

### アシスタント（作業中の応答 2）

> 接続はかなり明確でした。実機の既定 `estimate_mode=1` では、位置・速度は mocap の位置補正と IMU 加速度による予測を組み合わせた Kalman filter の値で、姿勢も IMU と mocap の役割分担になっています。したがって z の I 項は raw mocap を直接積分しているのではなく、mocap に拘束された融合状態の位置誤差を積分しています。次に、この閉ループ内の値を ProbTF へ二重投入しない構成を `deflecomp` の evidence/provenance 設計と突き合わせます。

### アシスタント（作業中の応答 3）

> `deflecomp` との対応も見えました。Grape 側で ProbTF の特色を出す中心は、「PID 値を別センサとして足す」ことではなく、mocap・IMU・gimbal encoder・ESC telemetry をそれぞれ出所付き likelihood として扱い、command/PWM は因果的な既知入力として、同じ軌道・actuator force・有効パラメータの posterior に接続することです。これなら `deflecomp` の「複数 IMU evidence→有効剛性」と同じ思想を、動的・異種センサへ発展させられます。

### アシスタント（最終回答）

はい。その系譜にするなら、この例の主題は「mass の particle filter」や「PID tuning」ではなく、**任意個数の異種 sensor evidence を、時間・座標系・出所・不確かさ付きで剛体運動モデルへ接続し、有効な機体・actuator parameter を推定すること**に置くのがよいと思います。

観測の中心は mocap と IMU です。ただし、bag にある gimbal encoder と ESC telemetry も重要な観測です。一方、command、PWM、PID 出力は sensor observation ではなく、原則として既知の制御入力または派生値に分類します。

#### I ゲインではなく、mocap/IMU 融合状態から作られた I 項です

まず用語を訂正すると、前の回答で `11.991 m/s²` として使ったのは **I gain \(K_i\)** ではなく、

\[
u_I(t)=K_i\int e_z(t)\,dt
\]

という **I term の出力**です。

実機 launch の既定値は `estimate_mode=1`、すなわち `EXPERIMENT_ESTIMATE` です。[bringup.launch](/home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/launch/bringup.launch:9) と [StateEstimation.yaml](/home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/config/StateEstimation.yaml:40) を見ると、この mode には IMU と mocap の両方が参加します。

- mocap 位置が Kalman filter の position correction に入る。
- IMU 加速度が同じ filter の position/velocity prediction に入る。
- 姿勢は IMU が重力方向と角速度を与え、mocap が主として yaw を拘束する。

controller は raw mocap topic を直接読むのではなく、[pose_linear_controller.cpp](/home/leus/catkin_ws/src/jsk_aerial_robot/aerial_robot_control/src/control/base/pose_linear_controller.cpp:209) でこの融合済み `pos_`、`vel_`、orientation、angular velocity を取得し、

\[
e_z=z_{\rm ref}-\hat z_{\rm fusion}
\]

を I 項へ積分しています。したがって「mocap の結果を使って位置合わせしているか」には **はい**ですが、より正確には「mocap に絶対位置を拘束され、IMU で時間発展した融合状態を使っている」です。

#### PID を observation としてもう一度融合してはいけません

違和感は正しいです。mocap と IMU を新しい ProbTF estimator に直接入れたうえで、同じ情報から controller が作った PID error や I term を独立 likelihood として掛けると、同じ情報を二重に数えます。`/uav/cog/odom`、既存 KF 出力、raw mocap を全部独立センサとして入れるのも同じ問題です。

ただし、閉ループ中の command を parameter estimation に使うこと自体は不自然ではありません。役割を分ければよいです。

\[
u_t=\pi(r_{\le t},\hat X_{\le t}),\qquad
X_{t+1}=f(X_t,u_t,\phi)+w_t
\]

ここで、記録された \(u_t\) は plant を駆動した**既知の因果的入力**です。推定器は \(u_t\) を条件として次の状態を予測します。PID 内訳は command の再構成、時刻ずれの検証、model mismatch の診断には使えますが、独立 sensor evidence にはしません。

#### Grape で使う evidence の分類

| bag の情報 | ProbTF 系での役割 |
|---|---|
| raw mocap pose | 絶対位置の Gaussian evidence、姿勢の Bingham evidence |
| IMU gyro | 時刻間の orientation/twist prediction |
| IMU accelerometer | specific-force/dynamics evidence と bias 推定。飛行中は単純な gravity evidence にしない |
| magnetometer、barometer | 利用可能なら heading、低周波 altitude の追加 evidence |
| gimbal joint/servo state | 各 rotor の thrust direction と lever arm の観測 |
| ESC RPM、current、voltage | rotor force、actuator effectiveness、故障・劣化の evidence |
| thrust command、PWM | actuator model に入れる既知入力 |
| existing fused odom、PID terms | 比較・初期化・診断用。raw evidence と同時に独立融合しない |

ProbTF の evidence fusion も、単に入力数を増やすのではなく、**独立 likelihood の自然パラメータを加える**設計です。[evidence_fusion.py](/home/leus/catkin_ws/src/ProbTF-demo/ros/core/probtf_core/src/probtf_estimators/evidence_fusion.py:302) は同じ `source_id` の重複を既定で拒否しています。センサ数に上限を設けないことと、相関した派生値を無制限に掛け合わせることは別です。

#### ProbTF-native な全体構成

私は次の三層にします。

```text
mocap + IMU
    → 時間相関を持つ trajectory posterior (X, v, omega, bias)

command/PWM + gimbal angle + RPM/current/voltage
    → actuator-force posterior (T1 ... T4, delay, effectiveness)

trajectory posterior + actuator-force posterior
    → rigid-body dynamics likelihood
    → effective plant-parameter posterior
```

時間ごとの潜在状態は

\[
X_t,\ v_t,\ \omega_t,\ b_{a,t},\ b_{\omega,t},\ T_{1:4,t}
\]

とし、遅く変化する parameter は、まず

\[
\phi=
\left[
m_{\rm eff},
\frac{k_1}{\bar k},\ldots,\frac{k_4}{\bar k},
c,\tau_{\rm actuator}
\right]
\]

程度から始めます。十分な 6-DoF excitation と絶対 scale の anchor が得られた段階で、\(m\)、絶対 \(k_i\)、慣性 \(J\) を分離します。

この構成なら、mocap の二階差分を確定値として作る必要はありません。mocap が pose を拘束し、IMU が時刻間を結ぶため、trajectory posterior の cross-time covariance を保ったまま \(a,\omega,\alpha\) の分布を dynamics likelihood へ渡せます。まさに先ほど話していた ProbTF の時間方向への誤差伝播です。

ProbTF edge として公開するのは各時刻の pose marginal と推定 CoG transform です。mass、thrust coefficient、PID gain は transform ではないので、専用の parameter posterior message に source IDs、episode、covariance、観測可能 rank とともに出します。

#### `deflecomp` と同じ系譜になる点

対応はかなりきれいです。

- `deflecomp`: 複数 link IMU の重力方向 likelihoodから有効剛性 \(K_{\rm est}\) を推定する。
- Grape: mocap、IMU、ESC、gimbal の likelihood から有効機体・actuator parameter \(\phi_{\rm est}\) を推定する。
- `deflecomp` の `K_est` / `K_exec` 分離に対応して、Grape でも \(\phi_{\rm est}\) と controller が実際に使う bounded・smoothed な \(\phi_{\rm exec}\) を分ける。
- どちらも command history は sensor truth ではなく、観測時刻より前に実際に送られた因果的入力として使う。
- どちらも観測不能方向を nominal/URDF 値で埋めず、posterior uncertainty と rank deficiency を残す。

つまり、PID は既存の fast stabilizer として残し、ProbTF estimator は raw sensor evidence から plant mismatch を推定します。推定結果を安全に反映する後段だけを `deflecomp` の `K_exec` と同様に分離します。

#### 実装順として妥当な最小構成

1. raw mocap と IMU だけで probabilistic trajectory smoother を作り、現在の Savitzky–Golay 微分と比較する。
2. command、PWM、gimbal angle、RPM を追加し、\(m_{\rm eff}\) と rotor 間の相対 effectiveness を推定する。
3. mocap-only、IMU-only、mocap+IMU、さらに ESC 追加という ablation で、posterior covariance と held-out pose prediction が改善することを示す。
4. その後に 6-DoF excitation と外部 anchor を追加し、CoG・慣性・絶対 thrust scale の分離へ進む。

一点だけ実装上の注意があります。現在の ProbTF core は、一つの transform に対する任意個の evidence fusion や専用 orientation filter は持っていますが、一般的な temporal factor graph が設定だけで完成する状態ではありません。実際、graph buffer の model-based temporal evaluation はまだ未実装です。[buffer.py](/home/leus/catkin_ws/src/ProbTF-demo/ros/core/probtf_core/src/probtf/graph/buffer.py:169) したがって Grape 用の fixed-lag/batch estimator は、`deflecomp` と同じく application-level estimator として実装し、その pose marginal と provenance を ProbTF に公開するのが現実的です。

また、センサを増やしても \(m\) と共通 thrust scale の gauge が自動的に消えるわけではありません。独立な load-cell、秤量値、信頼できる RPM–thrust calibration のどれかがなければ、絶対値ではなく \(m_{\rm eff}\) の posterior として報告します。ProbTF の利点はこの不可観測性を魔法のように消すことではなく、**どの evidence が何を拘束し、どの不確かさと相関が残ったかを失わずに、多数のセンサを同じ推論へ参加させられること**です。

したがって、この例の看板は「ProbTF を出力形式として使った mass estimator」ではなく、**multi-source temporal ProbTF による effective dynamics / actuator calibration と adaptive compensation**にするのがよいと思います。それなら `deflecomp` の思想を自然に拡張した例になります。

今回の質問、作業中の応答、最終回答も [QandA.md](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/lectures/QandA.md) に記録しました。

## 2026-07-24: TF/TF2 の時間評価、全 Grape bag の差分、次回設定への posterior predictive 提案

### ユーザー

> ProbTF 上の model-based temporal evaluation というのは、それに相当するものは通常の TF / TF2 にはありますか？あるならコア機能として追加する必要があるかなと感じています。また、rosbag には飛行に失敗したデータもありますが、それぞれの rosbag の差分とか見れますか？パラメータの情報が残っていればいいんですが、、、。また、ProbTF based のメソッドで得られるものは、何がわかるでしょうか？今回のrosbag 群から取り出しうる情報という意味で、です。たとえば、次の実験でこのように設定するとうまく行く可能性があるとか、そういう情報は得られそうですか？

### アシスタント（作業中の応答 1）

> 三点を分けて確認します。まずローカルの TF/TF2 実装から、補間・外挿がどこまで標準機能で、ProbTF の `PREDICT_WITH_MODEL` に相当するものが本当にないかを確定します。次に全 rosbag を topic・parameter snapshot・飛行結果の特徴量で比較し、失敗/成功を分類できる手掛かりが残っているか調べます。その結果を基に、「この bag 群だけで次回設定へ何を提案でき、何は追加実験が必要か」を切り分けます。今回の経過と最終回答も `QandA.md` に追記します。

### アシスタント（作業中の応答 2）

> TF/TF2 については結論が出ています。標準 TF2 は、保存済み transform の両側サンプル間を並進線形補間・姿勢 SLERP するところまでで、最新値より未来や最古値より過去は `ExtrapolationException` にします。異なる二時刻を fixed frame 経由で照会する API はありますが、運動モデルで未来分布を予測する機能ではありません。旧 TF の extrapolation limit も Noetic では deprecated で、ProbTF の「確率付き model prediction」に相当する標準機能はありません。コア追加候補ではありますが、モデルの登録・依存変数・process noise・予測 horizon を API として明示する必要があります。

### アシスタント（作業中の応答 3）

> bag 差分には想像以上に有用な情報が残っています。特に controller の dynamic-reconfigure 更新が全 bag に記録されており、6月12日の試行で roll/pitch が `P=15,D=8 → P=20,D=8 → P=10,D=8 → P=10,D=12` と変わり、長時間飛行中の bag 8 ではさらに連続 sweep、最終的に `P=13,D=20` へ到達した履歴を時刻付きで復元できました。bag 8 では XY も `P=3,D=1 → P=4,D=2`、yaw は `P=4 → 6` に変わり、その設定が bag 9 と翌日の3本に引き継がれています。PWM–thrust table は全12本で同一でした。一方、URDF全体、servo設定、propeller交換履歴、firmware commit の完全な parameter snapshot は bag にありません。

### アシスタント（作業中の応答 4）

> 失敗データにも明確な価値があります。6月12日 #3 は takeoff 中の `22.220–22.770 s` に mocap が約 `550 ms` 欠落し、`22.731 s` の `Sensor Unhealth Level3 → force landing` と一致しました。#4 は rosout に `roll pitch angles are too large` が残っています。一方、#7 以降は持続 hover です。さらに同一飛行 #8 の sweep では、roll/pitch `P=13` で `D=12,15,18,20` と上げるにつれ姿勢誤差 RMS が概ね `4.61°, 3.99°, 3.15°, 3.10°`、角速度誤差 RMS が `0.164, 0.139, 0.088, 0.086 rad/s` へ下がっています。したがって次回候補はデータから絞れますが、未記録の propeller 交換などと因果を混同しない posterior predictive な提案にすべきです。

### アシスタント（最終回答）

はい、三点ともかなり前向きな答えになります。

- TF2 にあるのは、既知サンプル間の決定論的な補間までです。確率付きの model prediction はないので、ProbTF core に時間モデルの契約を追加する価値があります。
- 12本の bag は比較可能で、controller gain の変更履歴、失敗理由の一部、持続飛行時間、sensor dropout、hover に必要な effective thrust などが残っています。
- 次回設定の候補も絞れます。ただし「この gain なら必ず飛ぶ」ではなく、parameter posterior に対する成功確率・誤差・saturation risk を比較する posterior predictive recommendation として出すべきです。

#### 通常の TF/TF2 にある時間機能

標準 TF2 の `TimeCache` は transform 履歴を保持し、要求時刻を挟む二サンプルがあれば、並進を線形補間し、姿勢を SLERP します。[time_cache.h](/opt/ros/noetic/include/tf2/time_cache.h:92)

一方、履歴の外側を要求すると `ExtrapolationException` です。[buffer_core.h](/opt/ros/noetic/include/tf2/buffer_core.h:116) の「target と source を異なる時刻で fixed frame 経由により問い合わせる API」も、各時刻に保存された transform を合成するもので、運動方程式による予測ではありません。

旧 TF には有限区間の transform 差分から平均 twist を返す `lookupTwist` がありますが、実装は区間両端の位置・姿勢差を時間で割るものです。[listener.py](/opt/ros/noetic/lib/python3/dist-packages/tf/listener.py:115) covariance や process noise を伝播するものではなく、未来 transform の予測にも使われません。旧 `setExtrapolationLimit` も Noetic では deprecated です。[tf.h](/opt/ros/noetic/include/tf/tf.h:102)

したがって、現在 ProbTF に列挙されている

- `INTERPOLATE_WITH_MODEL`
- `PREDICT_WITH_MODEL`

は TF2 の単なる写像ではなく、ProbTF 独自に定義すべき上位機能です。現状は明示的に未実装になっています。[buffer.py](/home/leus/catkin_ws/src/ProbTF-demo/ros/core/probtf_core/src/probtf/graph/buffer.py:169)

#### core へ入れるべき範囲

私は、**時間モデルを呼び出す契約と安全規則は core、具体的な機体モデルは estimator/plugin 側**に置きます。

core 側には少なくとも次が必要です。

1. physical edge と authority に対する temporal model の登録。
2. `interpolate(before, after, requested_stamp)` と、因果的な `predict(history, requested_stamp)` の区別。
3. 必要な履歴長、最大 prediction horizon、process noise、依存入力の宣言。
4. 出力への support sample stamps、prediction horizon、model ID/version、approximation、source IDs の記録。
5. horizon 超過、stale input、authority conflict、未来情報の混入を拒否する status。

ProbTF では各時刻の marginal を二つ補間するだけでは不十分です。同じ trajectory 上のサンプルには時間相関があるため、model が cross-time covariance または条件付き遷移を提供する必要があります。

最初の concrete model としては、constant-twist + process noise、IMU-driven orientation、application callback の三つが妥当です。Grape の剛体運動式そのものを core に埋め込む必要はありません。

今回見つかった `550 ms` の mocap dropout は、この機能のよい実例です。短い欠落なら IMU-driven prediction で pose 分布を継続し、時間とともに covariance を広げられます。ただし `550 ms` を無条件に隠すのは危険なので、controller 用には短い prediction horizon と uncertainty threshold を設け、超過時は従来どおり安全側へ遷移すべきです。

#### rosbag に残っている parameter

完全な ROS parameter server snapshot はありませんが、次は残っています。

- controller の `xy`、`z`、`roll_pitch`、`yaw` の dynamic-reconfigure 初期値と全変更履歴
- FC へ送った `/rpy/gain`
- voltage-dependent PWM–thrust table `/motor_info`
- command、PWM、ESC RPM/current/voltage
- gimbal target/actual angle
- mocap、IMU、融合 odometry、PID 内訳
- TF/TF-static geometry
- flight-state transition、takeoff/land command、rosout

`motor_info` の serialized 内容は全12 bag で同一でした。

一方、次は復元できません。

- recording 開始前を含む完全な `rosparam dump`
- 記録時 URDF、YAML、launch argument の hash
- flashed firmware と source commit の対応
- propeller の個体 ID、交換時刻、使用時間
- payload、秤量値、load-cell calibration
- servo calibration の完全な設定

したがって、bag 間で parameter topic が同じでも、未記録の propeller 交換がなかったとは言えません。

#### 12 bag の実測差分

`airborne` は raw mocap の高さが離陸前中央値より `0.25 m` 以上だった時間、`HOVER` は記録された flight state 5 の時間です。

| bag | roll/pitch gain | airborne | HOVER | 主な結果 |
|---|---:|---:|---:|---|
| 6/12 #1 | P15 I1 D8 | 3.0 s | 0 s | 短時間浮上後に land |
| 6/12 #2 | P20 I1 D8 | 0 s | 0 s | takeoff 後すぐ stop、後に force-land command |
| 6/12 #3 | P20 I1 D8 | 0 s | 0 s | mocap 550 ms 欠落と同時に sensor-unhealthy force landing |
| 6/12 #4 | P20 I1 D8 | 1.1 s | 0 s | `roll pitch angles are too large` failsafe |
| 6/12 #5 | P10 I1 D8 | 0 s | 0 s | takeoff に入らず、`four_axes/command` もなし。調整用記録 |
| 6/12 #6 | P10 I1 D8 | 1.1 s | 0 s | 短時間浮上後に land |
| 6/12 #7 | P10 I1 D12 | 127.7 s | 114.5 s | 初の持続 hover |
| 6/12 #8 | P12 D12 から連続 sweep、最終 P13 D20 | 319.3 s | 309.4 s | 飛行中に gain tuning |
| 6/12 #9 | P13 I1 D20 | 79.1 s | 64.4 s | 持続 hover |
| 6/13 #1 | P13 I1 D20 | 161.5 s | 151.0 s | 持続 hover |
| 6/13 #2 | P13 I1 D20 | 60.6 s | 47.7 s | 持続 hover |
| 6/13 #3 | P13 I1 D20 | 31.9 s | 21.1 s | 持続 hover |

z gain は全 bag で `P5 I1 D2.5` のままです。bag 8 で XY は `P3 I0.1 D1` から `P4 I0.1 D2`、yaw は `P4 I1 D2` から `P6 I1 D2` に変わり、その値が後続 bag に残っています。

特に価値が高いのは、同じ持続飛行中に変更した bag 8 です。roll/pitch を `P=13, I=1` にそろえた区間では次の結果でした。

| D | 姿勢誤差 RMS | 角速度誤差 RMS |
|---:|---:|---:|
| 12 | 4.61 deg | 0.164 rad/s |
| 15 | 3.99 deg | 0.139 rad/s |
| 18 | 3.15 deg | 0.088 rad/s |
| 20 | 3.10 deg | 0.086 rad/s |

これは `D=18–20` が今回の機体・条件で有望という強い evidence です。さらに `D=20` は後続4 bag でも持続飛行しています。ただし epoch ごとの機体位置、tether、battery、propeller 温度は完全には同一でなく、gain だけの無作為比較ではありません。

#### ProbTF-based method から取り出せるもの

今回の bag 群から期待できる posterior は次のとおりです。

| 得られる量 | 確からしさ | 次回実験での用途 |
|---|---|---|
| pose/twist/acceleration と IMU bias の trajectory posterior | 高い | mocap dropout 中の予測、微分 uncertainty の伝播 |
| sensor freshness・outlier・dropout 原因 | 高い | #3 のような sensor failure と control failure の分離 |
| controller gain ごとの閉ループ減衰・誤差分布 | 中～高 | observed gain 候補の順位付け |
| \(m_{\rm eff}=m/k_T\) | 中～高 | hover feedforward、allocation の補正 |
| rotor 間の相対 effectiveness、delay、時定数 | 中程度 | rotor 別 allocation、propeller 劣化検出 |
| CoG offset | excitation 次第 | 定常 torque bias と allocation の補正 |
| full inertia | 低い | hover 中心なので、次回の 6-DoF excitation が必要 |
| true mass と絶対 thrust scale の分離 | 不可能 | 秤量値または load-cell/RPM–thrust anchor が必要 |
| propeller 交換という原因名 | 不可能 | parameter change point は検出できても、交換記録が必要 |

成功した各 bag の定常 z-I term から得る \(m_{\rm eff}\) は約 `2.85–2.96 kg` で、中心は約 `2.90 kg` です。これは単一 bag の偶然ではなく、複数日の持続飛行に再現しています。

ProbTF を使う本質的な利点は、これらを別々の point estimate として出すのではなく、

\[
p(X_{0:T},b_{\rm imu},T_{1:4,0:T},\phi
\mid
y_{\rm mocap},y_{\rm imu},y_{\rm ESC},u_{\rm command})
\]

という一つの posterior として持ち、「この設定で誤差・tilt・saturation が許容範囲に収まる確率」を次回条件について計算できる点です。

#### 現時点で提案できる次回設定

既存データから最も根拠のある baseline は次です。

- roll/pitch: `P=13, I=1, D=18–20`
- XY: `P=4, I=0.1, D=2`
- yaw: `P=6, I=1, D=2`
- z: `P=5, I=1, D=2.5`
- effective thrust mass prior: 中心 `2.90 kg`、少なくとも episode 差 `2.85–2.96 kg` を uncertainty として持たせる

実績を優先するなら `D=20` から開始できます。一方、bag 8 では `D=18` と `20` の差が小さいため、derivative noise と actuator activity を含めた比較では `18` が選ばれる可能性もあります。この二点を tethered A/B test の候補にするのが妥当です。

feedforward は二通りの等価な入れ方があります。

1. 現在の allocation mass `2.3516 kg` を維持し、hover acceleration command を約 `12.0 m/s²` とする。
2. URDF の物理質量とは別に `effective_thrust_mass ≈ 2.90 kg` を設け、標準の \(g\) feedforward を使う。

後者の方が `deflecomp` の effective parameter と同じ意味で明瞭です。既存 I term がすでに約 `12 m/s²` を保持しているため、feedforward を追加するときは integrator を同時に reset/ramp しないと二重加算になります。最初は offline posterior predictive、次に shadow mode、最後に bounded adaptation と進めるべきです。

なお、#3 の失敗は gain 変更では解決しません。次回は gain と同時に、mocap freshness、ProbTF prediction covariance、最大 prediction horizon による safety policy も試験対象にする必要があります。

#### 次回から必ず残したい provenance

次回は bag と同名の manifest を作り、可能なら latched topic として bag 内にも保存するのがよいです。

- 全 `rosparam dump`
- URDF、YAML、launch file の SHA-256
- 関連 repository と firmware の git commit
- physical mass、payload
- motor/propeller ID、装着日、交換イベント、累積時間
- thrust calibration ID
- battery ID
- 実験者が付けた success/failure と介入時刻

要するに、**今回の bag 群だけでも「P13/I1/D18–20 が有望」「effective thrust mass は約2.90 kg」「少なくとも一件は gain ではなく mocap dropout が失敗原因」という次回に直結する情報が得られます**。ProbTF 化すると、これを単なる後知恵の比較ではなく、sensor・actuator・model uncertainty を含む成功確率として提示できるようになります。

今回の質問、作業中の応答、最終回答も [QandA.md](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/lectures/QandA.md) に記録しました。

## 2026-07-24: ProbTF posterior から次回実験の制御パラメータ修正値を提案する

### ユーザー

> OK.結構 rosbag に記録されていて、優秀ですね。実用上必要な（欲しい）情報は何になるかというと、求めている軌道を実現するために必要な各種パラメータの修正値ということになりそうですが、「求める軌道を実現するためのパラメータ値（PIDゲインの修正量など）」を prob tf 的に出すことはできそうですか？今の議論は、主に「過去に得られた実験データからもともとの様子を復元する」方向の議論だと思っていて、そうではなく、次の実験で使える、欲をいえば自動チューニングでの次回更新値の提案（でも話聞く感じもうこの機能はあるんですよね？だったら余計に手出ししなくてもいいかな）をする感じですが。

### アシスタント（作業中の応答 1）

> ここは「推定」と「次の操作を選ぶ意思決定」を分けて確認します。まず現 controller に自動 PID tuning が本当にあるのか、それとも bag 8 のような runtime gain 変更を受け付けるだけなのかをコードで確定します。そのうえで、ProbTF の trajectory distribution から、次回の feedforward・allocation・PID 修正量を確率付きで選ぶ設計に落とします。今回の応答も `QandA.md` に記録します。

### アシスタント（作業中の応答 2）

> 確認できました。現状にあるのは二つです。`dynamic_reconfigure` で操作者が gain を変更する入口と、trajectory の `target_acc` を PID の feedforward 項へ入れる入口です。bag 8 の gain sweep は自動チューナではなく、人が変更した値を callback がそのまま `setPGain/setDGain` したものです。過去データから「次の gain・effective mass・allocation correction」を計算して提案または適用する optimizer は、現 controller にはありません。したがって機能が重複するわけではありませんが、最初から全 PID gain を自動化するより、ProbTF posterior から inverse-dynamics/feedforward 修正を出す方が今回の目的に直接的です。

### アシスタント（作業中の応答 3）

> 結論としては、ProbTF から直接 PID ゲインを「推定」するというより、ProbTF が作る将来軌道の確率分布を評価関数に渡し、次回パラメータを選ぶ構成が自然です。しかも今回、最初に提案すべきなのは PID より `effective thrust mass` や加速度 feedforward／推力配分の補正です。PID はモデル誤差を吸収する最後の層に置きます。この切り分けなら、deflecomp と同じく「推定値」と「実際に使う値」を分離できます。

### アシスタント（作業中の応答 4）

> 提案を記録する際、章の挿入位置が前回質問の直前になっていたため、発言順どおりになるよう記録順を整えています。内容自体の欠落はありません。

### アシスタント（最終回答）

はい、できます。ただし、**ProbTF 自体が PID オートチューナになるのではなく、ProbTF が将来軌道の確率分布を作り、その上の optimizer が次回パラメータを選ぶ**、という役割分担が自然です。

\[
\text{bag の多センサ evidence}
\rightarrow p(\text{状態・機体特性}\mid D)
\rightarrow \text{候補設定ごとの将来軌道分布}
\rightarrow \text{risk-aware 評価}
\rightarrow \text{次回修正値}
\]

#### 現 controller に自動チューニングはありません

確認できた既存機能は次の二つです。

- `dynamic_reconfigure` から渡された P/I/D を setter へ入れる実行時変更口。[pose_linear_controller.cpp](/home/leus/catkin_ws/src/jsk_aerial_robot/aerial_robot_control/src/control/base/pose_linear_controller.cpp:359)
- trajectory の `target_acc` を PID 出力の feedforward 項へ加える経路。[pose_linear_controller.cpp](/home/leus/catkin_ws/src/jsk_aerial_robot/aerial_robot_control/src/control/base/pose_linear_controller.cpp:238) [pid.h](/home/leus/catkin_ws/src/jsk_aerial_robot/aerial_robot_control/include/aerial_robot_control/control/utils/pid.h:63)

bag 8 の gain sweep は前者を使った人手の調整です。過去データを評価して次の gain を計算する optimizer、安定性を確認する仕組み、自動適用機能はありません。したがって、今回の提案機能と重複はしません。

#### ProbTF-based な次回値の選び方

求める軌道を時刻付きの desired transform \(T_d(t)\) として与えます。厳密な一本の軌道だけでなく、位置・姿勢ごとの許容幅を covariance または tolerance として与えるのが ProbTF らしい表現です。

過去データ \(D\) から、effective thrust mass、rotor effectiveness、delay、gimbal offset などの機体パラメータ \(\phi\) の posterior \(p(\phi\mid D)\) を得ます。次に PID・feedforward・allocation をまとめた候補設定 \(\kappa\) ごとに、

\[
p(X_{0:H}\mid T_d,\kappa,D)
=
\int p(X_{0:H}\mid T_d,\kappa,\phi)\,p(\phi\mid D)\,d\phi
\]

を model-based temporal evaluation で予測します。予測 transform と desired transform の誤差

\[
E_t(\kappa)=T_d(t)^{-1}T_{\rm pred}(t;\kappa)
\]

を ProbTF の合成・逆変換・uncertainty propagation で評価し、tracking error、PWM saturation、tilt、制御量、mocap dropout 時の risk を含む目的関数から候補を選びます。例えば、

\[
\kappa^\star
=
\arg\min_\kappa
\left(
\mathbb{E}[J(E,u)]+\lambda\,\mathrm{CVaR}[J(E,u)]
\right)
\]

とし、安全制約を破る確率が閾値以下の候補だけを許可します。これなら単なる過去の復元ではなく、**「この desired trajectory に対し、次の設定で許容誤差内に入る確率は何 % か」**を比較できます。

なお、PID gain、質量、推力係数そのものは transform ではないので、無理に ProbTF edge として表すべきではありません。ProbTF graph が trajectory/error distribution を持ち、別の `ParameterUpdateProposal` が次を持つ構成が明瞭です。

- parameter 名、現在値、提案値、修正量
- credible interval／提案の confidence
- 変更前後の tracking 成功確率
- saturation・tilt・stability risk
- 適用対象の軌道と有効範囲
- 使用した bag、sensor evidence、model version
- `NO_CHANGE`、`SHADOW_ONLY`、`MANUAL_APPROVAL`、`AUTO_APPLY` などの状態

#### 修正する優先順位

求める軌道を実現するという目的なら、順番は次がよいです。

1. inverse dynamics／acceleration・torque feedforward
2. `effective_thrust_mass` と rotor 別 allocation correction
3. gimbal offset、actuator delay・時定数
4. 残った閉ループ誤差に対する PID gain

desired trajectory から直接決まるのは主として feedforward です。PID gain は軌道そのものを生成する値というより、モデル誤差・外乱に対する帯域や減衰を決める値です。最初から PID に物理モデルの誤差を再び吸収させると、現在の「I 項でだましだまし飛ばす」状態へ戻ってしまいます。

#### 現在の bag から出せる最初の提案例

既存データだけを使うなら、概ね次の proposal になります。

| parameter | 現在の nominal | 提案 | 修正量 |
|---|---:|---:|---:|
| `effective_thrust_mass` | 2.3516 kg | 約 2.90 kg | 約 +0.55 kg（+23%） |
| roll/pitch P | 13 | 13 | 0 |
| roll/pitch I | 1 | 1 | 0 |
| roll/pitch D | 20 | 18–20 | 0、または D18 との A/B test |

`effective_thrust_mass` を設けない場合は、同じ補正を hover feedforward acceleration 約 `12.0 m/s²` として入れられます。既存 I 項がこの不足分を保持しているため、適用時は integrator の reset/ramp が必要です。

PID について `修正量 0` も重要な推定結果です。bag 8 では D18 から D20 の改善が小さく、後続 bag は D20 で持続飛行しているので、現データだけなら無理に更新する根拠はありません。D18 を試すなら actuator activity と derivative noise を含めた controlled A/B test として提案するのが適切です。

一方、現在の bag は hover と小運動が中心です。高速・大傾斜など既存データの支持領域外にある desired trajectory について、安全な更新値までは出せません。その場合の正しい proposal は大きな gain 変更ではなく、低振幅の excitation experiment と追加 evidence の収集です。

#### 実装・運用の着地点

`deflecomp` と同じく、推定値と実行値を分離します。

- `phi_est`：ProbTF evidence から得た機体・actuator posterior
- `kappa_proposed`：optimizer が出す次回修正案
- `kappa_exec`：人が承認し、rate limit・上下限を通した実際の設定

最初は offline replay で proposal だけを出し、次に shadow mode で実飛行中に「適用していたらどうなったか」を記録し、その後に bounded な feedforward 補正だけを自動適用するのがよいです。PID の自動適用は最後で十分です。

したがって、**一般的な PID オートチューナを新たに作ること自体は今回の主題にしなくてよい**と思います。一方、ProbTF posterior と desired trajectory から、uncertainty・sensor dropout・actuator saturation を含めて `feedforward / effective parameter / gain` の次回修正案を出す機能は現在なく、ProbTF の利点を示す独自性も実用性もあります。Grape 例の中心は、こちらに置くのがよいです。

今回の質問、作業中の応答、最終回答も [QandA.md](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/lectures/QandA.md) に記録しました。

## 2026-07-24: 失敗 bag の nominal/actual 軌道差から反実仮想の成功パラメータ領域を推定する

### ユーザー

> なんか求めた回答と違うな。べつに PID でだましだまし飛ばすのはそれでいいんですよ、別に真の値を知りたいわけじゃないんで。失敗するケースの rosbag がありますよね？その rosbag のデータから、「この範囲だと目標値に入る可能性がありましたよ」といった情報を引き出せるものなのか？という疑問です。思うに、「実際に飛んでみたときの軌道」と「制御器に与えられた機体の情報から思っていた軌道」がずれて墜落したり不安定な状態に陥っていると思うので、その差分、「このくらいずれていますよ」というのを ProbTF 的な文脈で表現できないか？という疑問です。まあ ProbTF 的、といいますが、データ同化的というか、ベイズ的というか、そんなものを想定していました。先行研究はいくらでもありそうなので、使えそうな理論はどんどん取り込んで、実際に実rosbagから推定するとしたら、どんな感じの実装になるのか、実装案をまとめてほしいです。もちろん、QandA.md に記載しておいてください。

### アシスタント（作業中の応答 1）

> 意図を取り違えていました。欲しいのは真の質量や最適な PID ではなく、失敗 bag に対する反実仮想です。つまり「当時 controller が想定した応答」と「実軌道」の確率的なずれを同定し、同じ初期状態・指令に対して、どの controller 内部パラメータ範囲なら目標軌道の許容領域へ入った可能性が高いかを事後分布として出す、という話ですね。実 bag の topic と controller 実装をもう一度対応付け、利用できる Bayesian system identification／posterior predictive／counterfactual evaluation の理論を調べた上で、再現可能な実装案にします。今回の全応答も `QandA.md` に記録します。

### アシスタント（作業中の応答 2）

> bag 4 を確認すると、この評価に必要な骨格は実際に残っています。`/debug/pose/pid` には各軸の `target_p`, `target_d`, 誤差、P/I/D 各項があり、`/four_axes/command`、PWM、gimbal、ESC、mocap、IMU、融合 odometry も同時記録されています。したがって、目標軌道・当時の制御出力・実軌道を時刻合わせできます。ただし controller 自身は「思っていた将来軌道」を明示的には出していないので、それは記録済み設定と制御則・機体モデルを同じ初期状態から閉ループ再生して構成する必要があります。さらに、実状態を毎時刻与える一段予測誤差と、予測状態を feedback する自由走行の軌道誤差を分けて扱うのが重要です。

### アシスタント（作業中の応答 3）

> この解釈なら、すでに簡単な実データ上の手掛かりも出ています。既存の Savitzky–Golay 微分を診断用に使い、controller が `/debug/pose/pid` に出した期待加速度と mocap の実加速度を比較した暫定値では、失敗 bag 4 の pitch 角加速度残差 RMS は約 `12.36 rad/s²`、持続 hover した bag 7 は約 `2.58 rad/s²` でした。bag 4 の最大 pitch 偏差も約 `0.702 rad`（40°）です。これはまだ時系列相関を正しく含む Bayesian posterior ではありませんが、「失敗前に nominal response からどの軸がどれだけ外れたか」は bag から抽出可能だという実証にはなっています。最終実装では、この微分値を独立観測としてそのまま尤度へ入れず、mocap・IMU の trajectory smoother と状態遷移尤度で扱います。

### アシスタント（作業中の応答 4）

> 先行研究との対応も整理できました。中心に置くべきなのは、(1) nonlinear state-space model の Bayesian smoothing／parameter inference、(2) posterior predictive check による nominal model のずれの可視化、(3) posterior から閉ループを再走査する counterfactual simulation です。exact likelihood が書きにくければ BayesSim 型の likelihood-free simulator calibration、残差に構造が残れば GP state-space／GP discrepancy を追加できます。VRFT は一回の入出力データから model-reference controller を直接求める比較対象になりますが、今回の nonlinear MIMO・飽和・gimbal・sensor dropout を扱う本体としては、状態空間モデル＋反実仮想 rollout の方が適しています。Safe Bayesian optimization はその後に実機で追加探索する段階の手法で、既存 bag 解析そのものとは別です。

### アシスタント（最終回答）

おっしゃるとおりです。前回は「何を直すべきか」に寄りすぎていました。今回欲しいものは、真の物理パラメータではなく、

> **失敗 bag で観測された model–reality gap を同化し、同じ目標に対して、どの設定範囲なら目標領域へ入った可能性が高かったかを反実仮想確率として返す**

ものです。これは実現可能で、むしろ Grape の ProbTF 例としてこちらの方が本筋です。

#### 推定したい二種類の「ずれ」

まず、似ていますが異なる二つを分けます。

\[
\Delta T_{\mathrm{track}}(t)
=T_d(t)^{-1}T_{\mathrm{actual}}(t)
\]

は目標軌道に対する実軌道の tracking error です。一方、

\[
\Delta T_{\mathrm{model}}(t)
=T_{\mathrm{nominal}}(t)^{-1}T_{\mathrm{actual}}(t)
\]

は、当時 controller が使ったモデルと設定を閉ループ再生して得る nominal trajectory と、実軌道との差です。今回の中心は後者です。

```text
目標 r(t)・当時の設定 κ0 ── controller/model replay ── Tnominal(t)
                                                        │
mocap・IMU・ESC ── Bayesian trajectory smoother ── Tactual(t)
                                                        │
                          ΔTmodel(t) と response posterior
                                                        │
候補設定 κ ── closed-loop counterfactual rollout ── P(目標領域に入る | κ, bag)
```

controller は将来軌道そのものを記録していませんが、`/debug/pose/pid` には目標位置・速度と、各軸について controller が要求した並進・角加速度が残っています。[pose_linear_controller.cpp](/home/leus/catkin_ws/src/jsk_aerial_robot/aerial_robot_control/src/control/base/pose_linear_controller.cpp:318) Gimbalrotor controller はその要求加速度を、設定された mass・inertia を使って vectoring force へ変換しています。[gimbalrotor_controller.cpp](/home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/src/control/gimbalrotor_controller.cpp:75) [gimbalrotor_controller.cpp](/home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/src/control/gimbalrotor_controller.cpp:114)

したがって、瞬時の「期待した加速度」と「実際の加速度」は直接比較でき、将来の「思っていた軌道」は controller と nominal dynamics を再生して作れます。

#### 欲しい最終出力

当時の設定を \(\kappa_0\)、候補設定を \(\kappa\)、bag から同化した有効応答パラメータを \(\eta\) とします。\(\eta\) は真の質量である必要はなく、推力・遅れ・軸間干渉をまとめて再現する behavioral parameter で構いません。

\[
p(x_{0:T},\eta\mid D,\kappa_0)
\]

を実 bag \(D\) から求め、候補設定ごとに

\[
p(x^{\mathrm{cf}}_{0:H}\mid r,\kappa,D)
=
\int p(x^{\mathrm{cf}}_{0:H}\mid r,\kappa,\eta)
       p(\eta\mid D)\,d\eta
\]

を閉ループ simulation で作ります。目標 pose の許容幅、最大 tilt、PWM・gimbal 飽和などをまとめた成功事象を \(A\) とすれば、

\[
q(\kappa)=P(A\mid D,r,\operatorname{do}(\kappa))
\]

が「この設定なら目標領域へ入った可能性」です。最終的に返す範囲は、

\[
\mathcal K_\gamma
=\{\kappa:q(\kappa)\ge\gamma\}
\]

です。例えば \(\gamma=0.8\) なら「posterior 上、80%以上の確率で許容領域に入る設定の集合」です。

重要なのは、これは真値の credible interval ではなく、**目標達成確率の高い controller 設定領域**だという点です。また mass-like 値と PID、推力 scale には強い相関があり得るため、個別の一次元区間だけでなく、`P–D`、`controller_mass–thrust_scale` などの joint region として出します。

#### 実 bag で既に確認できたこと

失敗 bag 4 と成功 bag 7について、現在の [kinematics.py](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/src/grape_param_estim/kinematics.py:391) の SG 微分を診断用途に使い、airborne 区間で `/debug/pose/pid` の期待角加速度との差を見ました。

| bag | 結果 | pitch 角加速度残差 RMS | 最大 pitch 偏差 |
|---|---|---:|---:|
| 6/12 #4 | `roll pitch angles are too large` で失敗 | 約 `12.36 rad/s²` | 約 `0.702 rad`（40°） |
| 6/12 #7 | 持続 hover | 約 `2.58 rad/s²` | 約 `0.258 rad`（15°） |

これはまだ filter covariance、時間相関、actuator delay を含まない暫定診断です。ただし「bag 4 は特に pitch 軸で、controller が要求した応答と実応答が大きく外れた」という信号は既に取り出せています。したがって、問いへの短い答えは **はい、失敗 bag からその差分を抽出できます**、です。

#### 推奨する確率モデル

状態は例えば

\[
x_t=(p_t,R_t,v_t,\omega_t,b^a_t,b^\omega_t,z^{\rm act}_t)
\]

とし、pose、velocity、IMU bias、actuator 内部状態を含めます。モデルは

\[
x_{t+1}
=F_\Delta\!\left(x_t,C(r_t,x_t;\kappa),\eta\right)+w_t
\]

\[
y_t^{\rm mocap}=h_m(x_t)+v_t^m,\qquad
y_t^{\rm imu}=h_i(x_t)+v_t^i
\]

という nonlinear state-space modelです。

最初から真の mass・inertia を狙う必要はありません。第一版の \(\eta\) は、次のような「指令加速度から実応答への有効モデル」が適しています。

- 並進・回転各軸の effectiveness
- roll–pitch などの cross coupling
- motor／gimbal の delay と一次遅れ
- velocity・angular-velocity damping
- 定常 bias
- battery、RPM、episode ごとの random effect

例えば、

\[
\tau_a\dot a_{\rm eff}
=G_a a_{\rm cmd}-a_{\rm eff}+b_a,
\qquad
\tau_\alpha\dot\alpha_{\rm eff}
=G_\alpha\alpha_{\rm cmd}-\alpha_{\rm eff}+b_\alpha
\]

を基本にし、必要な軸間項だけを追加します。これなら `mass/thrust` の分離不能性に悩まず、「controller の加速度指令に機体が何割・何 ms 遅れで応じたか」を直接推定できます。プロペラ劣化も、原因名は分からなくても effectiveness の低下として表現できます。

#### 実装案

既存の `grape_param_estim` の bag reader、時刻処理、解析 bag merge、tempered particle filter は再利用し、推定モデルを trajectory-response 用に追加するのがよいです。

1. **episode extraction**

   `mocap/pose`、raw IMU、`debug/pose/pid`、`four_axes/command`、`rpy/gain`、`rpy/pid`、gimbal target/actual、PWM、ESC、battery、flight state、dynamic-reconfigure 更新を event time で一つの episode にします。ground contact、airborne、saturation、mocap dropout、failsafe を別 mode に分けます。bag 3 の mocap dropout を機体 dynamics の悪化として学習させてはいけません。

2. **actual trajectory posterior**

   mocap pose と IMU を使う error-state EKF + RTS smoother、または IMU preintegration を含む factor-graph smoother を offline で走らせます。出力は点推定だけでなく \(p(x_{0:T}\mid y)\) の trajectory sample とし、mocap dropout 中も covariance を拡大しながら補間します。融合 odometry は mocap を既に利用した派生量なので、独立 sensor likelihood には再投入せず検算にだけ使います。

3. **controller replay**

   二つの replay mode を作ります。

   - `teacher_forced`：各時刻の実状態を controller に戻し、一段先予測 residual を得る。
   - `free_run`：予測状態を controller に戻し続け、当時の nominal closed-loop trajectory を得る。

   元の \(\kappa_0\) で replay した P/I/D 項、`four_axes/command`、vectoring force、PWM が bag と一致することを unit test にします。Grape の制御は PC 側だけでは完結せず、spinal 側でも roll/pitch PID 項を motor thrust に加えています。[attitude_control.cpp](/home/leus/catkin_ws/src/jsk_aerial_robot/aerial_robot_nerve/spinal/mcu_project/lib/Jsk_Lib/flight_control/attitude/attitude_control.cpp:379) したがって PID の反実仮想では、PC の `GimbalrotorController` と MCU の `AttitudeController` の両方を、可能なら既存 C++ コードをライブラリ化して再利用します。

4. **Bayesian response identification**

   第一版は smoother の trajectory sample ごとに状態遷移 likelihood を評価し、現在の resample-move SMC を \(\eta\) 用に一般化します。観測 pose の二階微分を独立 Gaussian として掛けるのではなく、\(\eta\) で一段積分した pose・IMU prediction と実観測との差を Student-t likelihood にします。

   episode 間は

   \[
   \eta_e\sim\mathcal N(\bar\eta,\Sigma_{\rm episode})
   \]

   の階層モデルにし、共通傾向と battery・propeller・日ごとの差を分けます。完全な joint state/parameter inference が必要になったら、particle filter と MCMC を結合する [PMCMC](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf) や、state particle を各 parameter particle の内側に持つ [SMC²](https://arxiv.org/abs/1101.1528) が理論的な拡張先です。

5. **model discrepancy**

   parametric な effectiveness・delay を入れても残差が state/action に対して系統的なら、sparse GP residual

   \[
   \delta(x_t,u_t)\sim\mathcal{GP}
   \]

   を \(F_\Delta\) に加えます。GP state-space model は未知 dynamics と将来軌道 uncertainty を同時に扱う既存の枠組みです。[Svensson et al., 2016](https://proceedings.mlr.press/v51/svensson16.html) ただし GP を最初から自由にすると、effectiveness と discrepancy が互いを説明してしまうので、まず低次元の有効モデルを fit し、held-out residual に構造がある場合だけ追加します。

6. **counterfactual sweep**

   候補 \(\kappa\) に PID、controller 内 mass/inertia、allocation scale、delay compensation などを入れます。各候補について posterior の \(\eta\)、初期状態、process noise を Monte Carlo sample し、controller を含む閉ループ全体を再走行します。

   gain を変えると制御入力そのものが変わるため、**別 gain の評価に記録済み command を固定して流用してはいけません**。記録 command の replay は \(\eta\) 同定用、candidate gain の評価は controller を再計算する closed-loop rollout 用です。

   offline simulation は比較的安価なので、最初は Sobol sequence または grid で候補空間を覆い、\(q(\kappa)\) の等高線と \(\mathcal K_\gamma\) を出せば十分です。

7. **validation**

   - nominal model、deterministic least squares、Bayesian response model を一段予測で比較
   - 50/80/95% predictive interval の coverage と innovation whiteness を確認
   - leave-one-bag-out で、未使用 bag の成功・失敗と軌道を予測
   - bag 3 を sensor-dropout negative control、bag 4 を attitude-failure case、bag 7以降を success case とする
   - bag 8 の gain sweep で local counterfactual response を検証
   - 早期警告を評価するときは、故障時刻より未来のデータを使わず、bag の prefix だけで逐次 posterior を作る

失敗 bag 全体を同化して行うのは retrospective diagnosis、失敗前 prefix だけを使うのは online prediction です。この二つは結果を分けて報告します。

#### ProbTF での表現

時刻ごとに次の transform distribution を出します。

- `world → desired/cog`
- `world → nominal/cog`
- `world → actual_posterior/cog`
- `world → counterfactual/<candidate>/cog`
- `nominal/cog → actual_posterior/cog`、すなわち \(\Delta T_{\rm model}\)

相対 transform の

\[
\xi_t=\operatorname{Log}\!\left(
T_{\rm nominal}(t)^{-1}T_{\rm actual}(t)
\right)\in\mathbb R^6
\]

を使えば、「位置で何 cm、姿勢で何 rad ずれ、その uncertainty がいくらか」を一つの SE(3) residual として表示できます。nominal と actual は共通の初期状態などを持つため、covariance を単純加算せず、同じ Monte Carlo sample 内で相対 transform を作って cross-correlation を保ちます。

ただし、現在の `ProbabilisticTransformStamped` の時刻ごとの marginal だけでは、「全 horizon で許容領域内」という事象の確率は計算できません。時間相関を保持するため、application 側に次を含む `TrajectoryParticleSet` 相当を追加します。

- `trajectory_id`, `sample_id`, weight
- 全 timestamp の transform
- candidate parameter vector
- target-tube 判定
- source bag、topic、config hash、model version

ProbTF topic は可視化と時刻問い合わせ、trajectory particle は反実仮想確率の計算に使う、という分担がよいです。

#### 取り込む先行理論の位置づけ

- posterior predictive check：観測軌道が nominal posterior predictive のどこから外れたかを見る基本形。[Gelman, Meng & Stern](https://stat.columbia.edu/~gelman/research/published/A6n41.pdf)
- PMCMC / SMC²：latent trajectory と dynamics parameter の joint Bayesian inference。
- BayesSim：既存 Gazebo 等を black-box simulator として使い、尤度が書けない場合に simulator parameter posterior を求める選択肢。[BayesSim](https://m.roboticsproceedings.org/rss15/p29.pdf)
- GP state-space / discrepancy：単純な有効モデルで残る非線形 residual を uncertainty 付きで学習。
- VRFT：一回の入出力データから reference model に合う controller を直接求める比較 baseline。[Campi et al.](https://doi.org/10.1109/CDC.2000.912835)
- Safe Bayesian optimization：解析後、実機で安全に追加 candidate を試す段階。複数安全制約を分け、quadrotor 実験にも適用されています。[Berkenkamp et al.](https://arxiv.org/abs/1602.04450)

motor delay を含む thrust curve・inertia の flight-data identification も quadrotor で実例があり、今回 ESC/RPM が残っていることと相性がよいです。[Data-Driven System Identification of Quadrotors Subject to Motor Delays](https://arxiv.org/abs/2404.07837)

#### 追加するファイルの具体案

```text
config/counterfactual.yaml
scripts/analyze_grape_counterfactual.py
src/grape_param_estim/episode.py
src/grape_param_estim/state_smoother.py
src/grape_param_estim/controller_replay.py
src/grape_param_estim/effective_response.py
src/grape_param_estim/counterfactual.py
msg/TrajectoryParticleSet.msg
msg/ModelMismatch.msg
msg/CounterfactualCandidate.msg
```

最初の milestone は bag 4 に絞り、

1. desired / nominal / actual posterior と \(\Delta T_{\rm model}(t)\) を出す
2. pitch effectiveness・delay の posterior を出す
3. roll/pitch `P–D` と controller 内 allocation parameter の候補を閉ループ sweep する
4. 「目標 tube 内確率」の等高線と joint parameter region を出す
5. bag 7、8、および後続成功 bag で calibration を検証する

ところまでです。

単一の失敗 bag だけでも、観測された入力近傍について model discrepancy と条件付き反実仮想は出せます。ただし未試験設定、とくに観測範囲から遠い設定について「実際に成功した」と証明することはできません。そこは posterior uncertainty を広げて `UNSUPPORTED` と表示します。今回のように成功・失敗・gain sweep の bag が併存しているため、leave-one-bag-out で反実仮想確率を校正できるのが大きな利点です。

要するに、この例の目的は「真の機体値を当てる」ではなく、**実 bag で controller の期待応答と実応答の probabilistic gap を同定し、その gap を通して「この設定領域なら目標軌道へ入った確率が高い」を返す Bayesian counterfactual trajectory evaluator**に置き直すのがよいです。これはご提示の意図に合っています。

今回の質問、作業中の応答、最終回答も [QandA.md](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/lectures/QandA.md) に記録しました。

## 2026-07-24: ProbTF core と Grape rosbag 解析の二段階 TODO

### ユーザー

> OK.では最後に、今回の話を総括して（必要であれば QandA.md を参照してもらって構いません）、probTF の core 側の TODO と、今回の rosbag 解析の TODO をまとめてほしいです。多数の選択肢がある場合は、その中で最も望ましいもの、両者選び難く、実際にやってみないとわからない場合は、オプションで切り分けられるようにし、実際試して良さそうなほうを残す方針にしましょう。その「実際試して良さそう」という判定方法も記載しておいてください。片方が全くうまく行かなさそうだったらそちらは prune しますし、どっちもそれなりの良さがあったら、両者ともに optional に残す（より良い方を default にする）、という方針で行きます。要は、この TODO は二層構造になっていて、まず第一には実装そのものの TODO と、もう一方は実装後の選定の TODO です。この両者の phase についての TODO を残しておいてください。

### アシスタント（作業中の応答 1）

> 了解しました。今回は議論の要約だけでなく、実際に着手できる TODO としてリポジトリへ残します。ProbTF core と Grape rosbag 解析を分離し、それぞれを「Phase 1: 候補実装」と「Phase 2: 比較・採否判定」の二層にします。各分岐には default 候補、optional 候補、共通の評価データ、定量的な採択・prune 条件を付け、判断が実装者の印象だけに依存しない形にします。今回の質問・途中経過・最終回答も `QandA.md` に記録します。

### アシスタント（作業中の応答 2）

> 設計上の境界も固定します。core には「時間モデルの契約・因果的な時刻評価・不確実性伝播・診断」を置き、Grape 固有の制御器、機体応答、成功条件、反実仮想探索は rosbag 解析側に置きます。軌道全体の粒子表現は、まず Grape 側で実証し、独立した二つ目の利用例が現れた時点で core へ昇格させる方針が最も安全です。

### アシスタント（作業中の応答 3）

> 二つの TODO を作成しました。比較方法は、まず因果性・再現性・calibration などの hard gate を通し、その後に held-out bag と bootstrap 信頼区間で比較し、性能同等なら一標準誤差則で単純な方を default にする構成です。用途別の有意な利点が残る候補だけ optional、全領域で劣る候補は prune、データ不足なら experimental として保留します。これからリンク、用語、既存実装との整合性と、Q&A の完全記録を監査します。

### アシスタント（最終回答）

まとめました。二つの TODO は、どちらも「Phase 1: 候補を実装する」と「Phase 2: 同一条件で比較し、default / optional / prune を決める」の二層構造です。

- [ProbTF core TODO](/home/leus/catkin_ws/src/ProbTF-demo/ros/core/probtf_core/TODO.md): 時間モデルの契約、因果的な interpolation/prediction、不確実性・依存関係・provenance、診断を実装対象にしました。高速な tangent-space moment 伝播と sample-wise 伝播、constant-twist と constant-acceleration を切替候補にしています。
- [Grape rosbag 解析 TODO](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/TODO.md): event-time データ化、mocap+IMU smoother、PC/MCU controller replay、有効応答の Bayesian 同定、closed-loop counterfactual、ProbTF/trajectory 出力を具体的な作業単位にしました。

採否規則も両方に明記しました。

1. bag hash、episode split、seed、metric、閾値を結果を見る前に固定する。
2. 因果性、replay 一致、数値健全性、calibration、再現性の hard gate を先に通す。
3. held-out episode / leave-one-bag-out と bag 単位 bootstrap 95% CI で比較する。
4. 性能が最良候補の一標準誤差以内なら、単純・高速な方を default にする。
5. 特定の利用領域で再現可能な改善がある候補は optional、全領域で Pareto dominated な候補は prune する。
6. 判別できないものは default を変えず `EXPERIMENTAL` とし、必要な追加データを指定する。

当面の default 候補は、core が moment propagation + constant-body-twist、rosbag 側が EKF/RTS + exact C++ controller replay + low-dimensional effective response + tempered SMC です。sample propagation、factor graph、Python surrogate、structured mechanics、PMCMC は比較 option、GP と BayesSim は既定モデルが診断 gate に失敗した場合だけ実装する条件付き option にしました。

また、軌道粒子はまず Grape application 側に置き、独立した二つ目の用途が同じ契約を要求した時だけ core へ昇格します。未試行で support から遠い parameter は、良い予測値でも `UNSUPPORTED` のままです。

[README](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/README.md) から解析 TODO へもリンクしました。`git diff --check` は通過しています。今回は計画文書のみの変更なので、runtime test は実行していません。この質問、三件の作業中の応答、この最終回答も [QandA.md](/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/lectures/QandA.md) に記録済みです。
