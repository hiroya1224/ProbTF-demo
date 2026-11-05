# 共通Binghamサンプラ

姿勢の乱数生成は、ROS/Python非依存の
[`bingham_sampler.cpp`](../src/bingham_sampler.cpp) 一箇所に置く。
C++の `probtf_core::BinghamSampler` とPythonの
`probtf.probability.sample_bingham_orientation` は同じ実装を呼ぶ。
Pythonには提案分布・採択判定の実装や代替サンプラを置かない。

## 根拠と符号

John T. Kent, Asaad M. Ganeiber, Kanti V. Mardia,
*A new method to simulate the Bingham and related distributions in directional
data analysis with applications*, Section 3, equations (3.4)–(3.6),
[arXiv:1310.8110](https://arxiv.org/abs/1310.8110).

論文のBACG法（Bingham / angular central Gaussian rejection sampling）を使う。
本リポジトリの密度 `exp(xᵀ A x)` は、
`P = λmax(A) I − A` により論文の `exp(−xᵀ P x)` と対応する。
スカラー行列を加えても密度は変わらない。`q` は周囲空間の次元で、姿勢は `q=4`。

1. `P` の固有値 `λi ≥ 0`、最小値0を求める。
2. `Σi 1/(b + 2λi) = 1` を `1 ≤ b ≤ q` で解く。
3. `Ω = I + 2P/b`、`y ~ N(0, Ω⁻¹)`、`x = y/‖y‖` とする。
4. `u = xᵀ P x`、`log M = −(q−b)/2 + (q/2) log(q/b)` として、
   `log U < −u + (q/2) log(1 + 2u/b) − log M` の場合に採択する。

計算は固有基底と対数で行い、密な逆行列・Cholesky分解を作らない。
濃度のクリップ、局所Gaussianへの置換、modeへの置換、試行回数による打ち切りは行わない。
一様分布も `P=0, b=q` の同じBACG経路で扱い、全提案を採択する。
Dirac分布は既知の四元数を返す。

提案の正規乱数は元の座標系で生成し、`Ω⁻¹` の対称平方根を作用させる。
固有ベクトルの符号や重複固有値内の基底の選び方で、固定seedの標本が飛ばない。
PythonはNumPy Generatorから64 bitのseedを一つ渡し、C++内で連続生成する。
同じseedで標本数だけを増やした場合、先頭の標本列は一致する。
以前のPython実装との標本値の一致は保証しない。

## 呼び出し経路

| 利用箇所 | 共通実装への経路 |
|---|---|
| Pythonの姿勢、条件付き並進、mixtureの標本 | `probtf.probability.sampling` → `probtf._sampling` → C++ |
| graph/kernelの標本照会 | 上記Python API |
| temporalの共通乱数による姿勢標本 | 上記Python API（mixture/残差の依存seedを維持） |
| orientation/global-fusion等のPythonデモ | 上記Python API |
| RVizのProbabilisticTF / ProbabilisticPoseのnative姿勢標本 | `sampleTransformDistribution` → 同じC++ |
| dthesis-final-demoのIMUリンク表示 | 上記RVizプラグイン。独自サンプラなし |

Gaussian並進残差、時間過程のGaussian雑音、明示的な点モーメントGaussian表示、
模擬センサ雑音、PID推定のMCMCはBingham分布ではない。
これらの分布をBACGに置き換えない。推定・解析的モーメント・SG補間も変更しない。

## ビルドと使用

ROS側は通常の `catkin build probtf_core probtf_rviz` で共有ライブラリと
使用中のPythonのABIに対応する拡張をビルド・インストールする。
RViz runtimeとPython拡張は同じ `libprobtf_bingham_sampler.so` にリンクする。

ROSを使わない場合は、Eigen3とC++14コンパイラを導入し、次のどちらかでインストールする。
リポジトリ全体のPythonパッケージでも、この同じC++ソースをビルドする。

```bash
python -m pip install ./ros/core/probtf_core
# または、リポジトリ全体:
python -m pip install .
```

```python
import numpy as np
from probtf.distributions import BinghamOrientation
from probtf.probability import sample_bingham_orientation

law = BinghamOrientation.from_parameter_matrix(np.diag([-3., -2., -1., 0.]))
quaternions_wxyz = sample_bingham_orientation(law, 1000, rng=42)
```

C++では `<probtf_core/bingham_sampler.hpp>` をincludeし、
`BinghamSampler::fromParameterMatrix(A)` を準備して `draw` / `drawBatch` を呼ぶ。
ROSのwire形式は従来の `sampleTransformDistribution` をそのまま使える。

検証は `test/test_bingham_sampler.cpp`、`tests/probtf/test_native_bingham_sampling.py`、
既存のPython/C++ transform sampling・temporal/kernel testsで行う。
論文Table 2のS²採択率、S³一様分布、極端な集中、帯状分布、
モーメント、符号・基底不変性、固定seedの先頭一致、条件付き並進を対象とする。

2026-11-06、このPCのReleaseビルドではC++の18テスト、Pythonのsampling・kernel・
temporal・ROS変換・rolling比較・非同期ペアIMUの172テストが通過した。
ROS非依存のコア単体wheel、リポジトリ全体wheelのビルドと、ROSを読み込まない
コア単体wheelからの標本生成も確認した。

S²の10万標本での採択率は、precision `(0,0,10)` で83.45%、`(0,10,10)` で58.05%、
`(0,0,100)` で79.89%、`(0,100,100)` で52.59%だった（論文Table 2は84/58/80/53%）。
S³の `precision=(0,L,L,L)` は `L=1e6, 1e12, 1e300` で約44.77%、一様分布は100%。
同条件の80標本生成をseedを変えて1000回計測した平均は、Python拡張呼び出し込みで
約0.06 ms、計測範囲はサンプラの準備と標本生成である。
