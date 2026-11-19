# 10 Hz観測による非線形／ジャイロ項なしの比較

全200 Hzの追加計算は、途中のユーザー指定により停止した。ここでは同じ保存入力から10 Hzで取得した比較だけを別保存する。[元の1 Hz結果](frequent_publication/INDEX.md)は保持し、[10 Hz結果](rate_10hz_publication/INDEX.md)と比較する。

共通する201個の保存時刻で、観測直後の姿勢中心を比較した結果は次のとおり。

| 取得頻度 | 非線形Eulerの姿勢RMSE [deg] | ジャイロ項なしの姿勢RMSE [deg] |
|---|---:|---:|
| 1 Hz | 31.3212 | 33.0524 |
| 10 Hz | 18.8948 | 19.4696 |

この測定実現値の10 Hz条件では非線形側の姿勢RMSEが約3%小さい。両方とも大きな誤差は残り、全ての軸・条件での優位を示した結果ではない。7点から11点求積への最大RPY中心差は約3.22×10⁻¹⁰ degで、この差を求積誤差で説明することはできない。推定の規準はgyroの尤度であり、姿勢の真値誤差を最小化したものではない。

## 同じ測定列から0.1秒ごとに取得

参照入力は`e4676ed`の`frequent_publication/source_input.npz`と`sensor_input.npz`。実飛行の規模へ校正した同じ20秒の運動と、同じ生成白色σ=2 deg/s・頻発スパイク・固定biasを再生成せずに使う。200 Hzの原列から0、0.1、…、20秒の201件を選ぶ。原native indexは0、20、…、4000で、測定値の平均化・補間・ノイズの再抽選は行わない。選択index、原header/record ns、source indexと原値を`acquisitions.npz`へ保存する。

両モデルのJ・保存された1 kHz既知トルク・初期姿勢・全選択gyro・観測σ=5 deg/sは同じ。違いは角速度式の`omega cross(J omega)`だけで、四元数運動学は共通である。途中の真値・スパイクフラグ・生成ノイズ・biasは推定へ渡さない。

初回gyroを初期法則に一度使い、モデル$m$、初期body角速度$w$に対して、取得済みの201件までのprefixを

\[
\pi_{m,k}(w)\propto
\mathcal N(w;y_0,R)\prod_{j=1}^{k}
\mathcal N(y_j;\omega_m(t_j;w),R),
\qquad R=(5\pi/180)^2I_3
\]

とする。観測ごとに独立Gaussianへ置換せず、同じ初期候補からの流れで姿勢・角速度と全交差を伝播する。各観測の直前／直後を同じ原時刻へ保存し、401行になる。終端の尤度因子は200個で、初期化に使った最初の観測を重複融合しない。

## 計算と表示

内部では決定論的Gauss–Hermite重要度求積を使う。Gaussian proposalは積分点の配置用で、事後状態への置換ではない。元の初期事前×全取得尤度をproposal密度で割って重みを計算する。C++では既存のEuler／四元数RK4の物理式を使い、履歴尤度を伝播中に集約する。全履歴×全点の巨大テンソルを保持しない。次数7・11を予定どおり計算して結果差を記録する。有限の残差や積分差で観測を棄却・停止せず、共分散floorや分散調整は行わない。

主図は上段RPY、中段body角速度平均±2σ、下段に同じ角速度共分散の2σ半幅。生gyroは200 Hzの全4,001件を描き、推定が使った10 Hzの201件を区別する。原測定を全部描くことと、全件を融合することを混同しない。追加の最初1秒の拡大図にも、同じ保存共分散を使う。

1 Hzとの誤差比較は、両結果に保存された共通時刻を使い、同時刻では最後の観測直後を選ぶ。時刻を整数nsへ合わせるのは表示・比較だけで、推定入力を丸め直さない。別途、旧レポートと同じ501個の評価時計でもRMSEを記録する。

RPY中心と帯は同じ回転モーメントからの局所表示近似で、保証付き95%領域ではない。時間相関を持つスパイクを、独立Gaussian誤差とする観測モデルは変更していない。狭くなる幅が実際の誤差を保証するとは主張しない。ロバスト化・外れ値除去・BFS・過程雑音は追加しない。

## 再現

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/synthetic-10hz-mpl \
python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/rate_comparison_run.py \
  --output /tmp/synthetic-10hz-comparison-20261120

python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/compact_rate_publication.py \
  /tmp/synthetic-10hz-comparison-20261120

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 \
  ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/rate_comparison_verify.py \
  /tmp/synthetic-10hz-comparison-20261120

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover \
  -s ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/tests -v
```

設定は`rate_10hz_config.json`。旧1 Hzの入力・設定・図・数値は書き換えない。保存するのは再現コードと小容量の完成図・数値で、生bag・バイナリ・中間キャッシュを含めない。途中次数の計算は`cache/`へ置き、Gitへ追加しない。

`compact_rate_publication.py`は全観測・全401行のモーメント・全proposalパラメータ・最後の求積重みを残し、再構成できる途中の重み行列だけを成果物から外す。残した全配列のbyte hash一致を記録する。有限なproposal最適化残差・反復状況・再配置差は`proposal_refresh_diagnostics.json`に残す。
