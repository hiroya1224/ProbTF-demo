# Sparse batch backend の synthetic recovery validation

## 1. 目的と範囲

この資料は perfect-model trajectory、known diagonal Q、continuous lag、asynchronous sensor、MCMC posterior の truth recovery 試験を記録する。
単に finite な値を返すことではなく、既知 truth、ridge rank、posterior moment、低励起時の failure semantics を数値 tolerance で検証する。

統計 case ごとに大規模な `26N` full MAP を反復すると CI 時間が過大になるため、検証を二層に分けた。
全軌道層では production の kinematic/dynamics factor と解析 Jacobian を使って latent trajectory residual を検証し、static recovery 層では同じ production factor の 18 次元 analytic reduced problem を解く。
これは full real-bag E2E の代替ではなく、各数理部品が truth を回復することを高速に切り分ける試験である。

## 2. perfect-model trajectory と 18 次元 static chart

`synthetic_batch.py` は 30 interval の非一様 time step、全 rotor/gimbal を励起する trajectory を生成する。
次時刻の velocity と angular velocity は production rigid-body dynamics residual と解析 Jacobianから Newton correction で求め、別の簡略 simulator や finite difference は使わない。

position/orientation kinematic residual の最大絶対値はそれぞれ `2e-14` と `3e-13` 未満、6 軸 dynamics residual は `2e-10` 未満だった。
18 次元 static coordinate を truth から ridge に直交する方向へ摂動して least squares を解くと、common-scale exact ridge を除く coordinate error は `2e-7` 未満まで回復した。
reduced likelihood Hessian の effective rank は 17、最小 numerical direction と analytic common-scale direction の absolute alignment は `1 - 1e-10` より大きかった。
ridge 上の複数点で likelihood residual norm が `2e-18` tolerance 内で不変であり、ridge を numerical damping が消していないことも確認した。

## 3. known diagonal Q の Laplace-EM

二つの bag、異なる可変 `dt`、6 軸 anisotropy を持つ normal-normal pseudo-observation から、MAP residual moment と Laplace covariance correction を別々に生成した。
case は isotropic-small continuous spectral density、anisotropic continuous spectral density、large fixed-interval covariance の三つである。

各 case で 6 成分の E-step expected second moment と closed-form Q target は truth の相対 `9%` 以内だった。
MAP residual moment だけの relative error norm は `0.75` より大きい一方、covariance correction 後は `0.23` 未満となり、hard-EM では truth を回復しない構成を明示的に検証した。
bag ごとに分けた target も truth の相対 `14%` 以内で、damped M-step の marginal-objective acceptance と accepted Q の一致も確認した。

## 4. continuous ZOH delay

exact delayed ZOH first-order response を用い、複数 command switch と不等 publish interval を含む profile を作った。
truth delay `0.0873 s` は `4 us` 以内で回復し、異なる warm-start marker から同じ optimum を得た。
zero-delay case は下限 `0.0 s` を選び、sub-sample delay を通常の smooth inner derivative に置き換えていない。

command が全 event で一定の low-excitation case では全 profile objective が同じになり、curvature を捏造せず `uniform delay prior because local profile curvature is unavailable` を返した。
このとき uncertainty は区間幅 `0.14 s` の一様分布標準偏差 `0.14/sqrt(12)` と一致した。

## 5. asynchronous sensor、bias、lever arm、frame

knot 間の interpolation fraction `[0.19, 0.37, 0.58, 0.83]` に pose、world velocity、gyro、specific force observation を配置した。
nonzero sensor lever arm、body-to-sensor rotation、CoG、gyro bias、accelerometer bias を truth として与え、production factor の解析 Jacobianで縮約 linear solve を行った。

pose-only、pose+velocity、pose+gyro、pose+velocity+gyro+accelerometer の四構成で active parameter は `3e-12` 以内、最終 residual は `3e-11` 未満まで回復した。
lever arm または gyro frame を誤った値へ置き換えると residual norm は `0.1` より大きくなり、frame/extrinsic mismatch を黙って bias へ吸収しないことを確認した。

## 6. MCMC statistical recovery

production delayed-acceptance sampler を四 chain で実行し、proper prior が likelihood exact ridge を正則化する解析 Gaussian posterior の mean と covariance を検証した。
retained draw の mean は analytic standard error に基づく tolerance 内、covariance は相対 `14%` と絶対 `0.008` の tolerance 内だった。
active coordinate の split R-hat は `1.06` 未満、ESS は 80 より大きく、likelihood null direction の sample variance は proper-prior analytic varianceの `12%` 以内だった。
stage-2 correction が実際に reject を持つことも確認し、quadratic surrogate の draw をそのまま MCMC posterior と表示する経路ではない。

banana-shaped curved ridge では四 chain が ridge の正負両側を横断し、`x` の 3%/97% quantile は `-1.3` 未満と `1.3` より大きかった。
`y` と `x^2` の correlation は `0.82` より大きく、posterior mean が局所 Laplace center から有意に離れることを検証した。
split R-hat は `1.10` 未満、ESS は 45 より大きく、ridge proposal kernel も 1500 回より多く試行された。

## 7. 実行方法と限界

対象 test は次で実行する。

```bash
cd /home/leus/catkin_ws/src/ProbTF-demo
PYTHONPATH=ros/examples/grape-param-estim/src \
  nosetests3 \
  tests/grape_param_estim/test_synthetic_batch_recovery.py \
  tests/grape_param_estim/test_mcmc_statistical_recovery.py
```

新規 8 test の実測時間は約 `8.8 s`、関連 82 test は約 `9.7 s` で全件成功した。
これらは covariance provenance、actuator time constant、controller reconstruction が正しいことを実 rosbag について保証しない。
実 bag では [real_flight_validation_ja.md](real_flight_validation_ja.md) の非収束と軌道乖離を別に評価し、synthetic recovery の成功を実機同定成功へ読み替えない。
