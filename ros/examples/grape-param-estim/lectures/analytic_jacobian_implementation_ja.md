# 解析 Jacobian の実装

## 1. 方針

production solver の全 factor は residual と解析 Jacobian block を同じ評価で返す。
finite difference は test の derivative oracle にだけ使用し、production fallback、lag derivative、sample-based regression には使用しない。
この境界は `batch/factor.py` の `FactorEvaluation` と `JacobianBlock` が強制し、各 block の row 数、variable dimension、finite 値、key 重複を検査する。

## 2. SO(3) の局所座標

orientation state は rotation matrix `R` と right increment `delta` で表す。

```math
R(\delta)=R\operatorname{Exp}(\delta),\qquad
\operatorname{Log}(R_1^TR_2)\in\mathbb R^3.
```

実装は Exp、Log、right Jacobian、inverse right Jacobian、geodesic midpoint の endpoint Jacobian を解析的に持つ。
小角度では series expansion を使い、`pi` 近傍の branch は diagnostics として明示する。
orientation observation residual、orientation kinematic residual、dynamics midpoint attitude は同じ right-tangent convention を使う。

## 3. parameter chart derivative

mass と effectiveness は log coordinate から decode するため、物理量の derivative は値自身を係数に持つ。
CoG は Cartesian coordinate なので derivative は identity である。
inertia は nominal inertia に対する symmetric matrix exponential を使い、Fréchet derivativeによって full SPD 6 coordinate の解析 derivative を計算する。
chart decode は physical parameter と mass、inertia、CoG、force effectiveness、torque effectiveness の全 derivative を一緒に返す。

## 4. actuator wrench derivative

各 vectoring rotor の force は actual thrust、actual gimbal angle、arm yaw、force effectiveness から計算する。
torque は CoG から thrust origin への lever arm cross force と rotor reaction torque の和である。
解析 block は actual thrust 4、gimbal angle 4、CoG 3、force effectiveness 4、torque effectiveness 4 への derivative を返す。
gimbal angle で thrust origin 自体も変わるため、force direction だけを微分して lever-arm derivative を落とさない。

## 5. rigid-body dynamics derivative

raw dynamics residual は `required - modeled` の body wrench 6 成分である。
endpoint attitude、linear velocity、angular velocity、thrust、gimbal、static chart に対する block を直接組み立てる。

force row では次の寄与を含める。

- midpoint rotation による world acceleration と body velocity の写像。
- endpoint velocity difference による acceleration。
- mass derivative。
- linear drag。
- actuator force と CoG/efficiency derivative。

torque row では次の寄与を含める。

- endpoint omega difference による angular acceleration。
- `omega cross J omega` の gyroscopic derivative。
- full inertia chart derivative。
- angular drag。
- actuator torque と lever arm/reaction torque derivative。

`specific_acceleration` quantity を選ぶ場合は wrench Jacobianへ左から `diag(I/m, J^{-1})` を掛けるだけでは不十分である。
mass と inertia 自体が static coordinate に依存するため、`1/m` と `J^{-1}` の product-rule term も static block に加える。

## 6. kinematic、controller、actuator transition

position kinematic factor は midpoint velocity を積分する Euclidean residual を使い、endpoint position/velocity の block を持つ。
orientation kinematic factor は endpoint relative rotation と midpoint omega の整合を SO(3) Log で測り、right Jacobian を用いる。
controller integral transition は recorded reference、mode schedule、integration gate、clamp の active set を再現し、非飽和区間の slope と飽和区間の zero slope を解析 block に反映する。
actuator thrust/gimbal transition は first-order response と magnitude/rate saturation を再現し、各 branch の active mask を保存する。

## 7. observation factor

position、velocity、gyro、specific force の frame transform と lever arm は sensor contract から明示的に与える。
pose orientation は quaternion 成分差ではなく SO(3) tangent residual を使う。
gyro bias と accelerometer bias は bag-local constant blockとして Jacobian に現れる。
accelerometer factor は sensor origin、body-to-sensor rotation、lever arm、bias の contract が揃わない限り有効にしない。

## 8. whitening と sparse assembly

各 factor は unwhitened physical residual/Jacobian を定義した後、request の covariance から得た square-root information を左から掛ける。
Q dynamics factor は selected residual quantity と interval model に従い、可変 `dt` を含む diagonal whitening を行う。
assembly は factor-local dense block を COO entry へ展開してから canonical CSC へ変換し、row/column order と nnz を診断へ保存する。

## 9. active set と nonsmooth point

clamp、thrust saturation、gimbal angle/rate saturation は globally smooth ではない。
実装は現在 branch の piecewise derivative と boolean active-set mask を返し、LM trial で branch が変わった事実を記録する。
active set が二状態を繰り返す場合は、収束したように見せず `active_set_oscillation` として停止する。
delay は ZOH event の breakpoint を動かすため、この piecewise derivative の枠にも入れず外側 profile optimization で扱う。

## 10. derivative test

各解析 block は高精度 central difference と比較するが、SO(3) variable は同じ right retraction を通して摂動する。
test は residual value だけでなく block shape、column identity、frame、unit、branch stabilityを固定する。
branch boundary そのものでは derivative が一意でないため、test point は branch 内部と boundary diagnostics を分ける。
whole-objective directional derivative test では複数 factor の assembly と state scaling も含めて解析方向微分を照合する。

## 11. 実装対応

| primitive/factor | module |
|---|---|
| SO(3) primitive | `geometry.py` |
| physical chart | `parameterization.py` |
| actuator wrench | `dynamics.py` |
| factor contract | `batch/factor.py` |
| pose/velocity/IMU | `batch/factors/pose.py`, `velocity.py`, `imu.py` |
| controller/actuator transition | `batch/factors/controller.py`, `actuator.py` |
| kinematics | `batch/factors/kinematics.py` |
| raw/statistical dynamics | `batch/factors/dynamics.py`, `dynamics_factor.py` |
| sparse assembly | `batch/linearize.py` |

解析 Jacobian が finite-difference より速いことだけを完了条件にしない。
frame、unit、SO(3) perturbation side、actuator branch、Q quantity が residual と一致していることが最優先である。
