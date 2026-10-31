# IMUを使う一次遅れ推定

既存のmocap版とは別に、同じ3記録のgyro・accelから機体パラメータと一次遅れを推定した。姿勢は既存ProbTFのBingham推定を使う。gyroで時間発展し、accelで更新した状態を次時刻へ渡す。地磁気は新規の観測がある時刻の出力にだけ融合し、次時刻の事前へ戻さない。

再現軌道は**同定モデル＋IMUから得た余剰レンチ**で計算する。余剰レンチを零平均の不確かさにだけ入れた旧モデル単体再生は、ユーザー指定に反していたため撤回し、同じ出力先の図を更新した。

結果は [比較PDF（60ページ）](publication/imu_native_20261031/imu_first_order_lag_comparison.pdf) にまとめた。46〜60ページに、3ケースそれぞれの実推定値17個、単位、全17×17共分散を収録している。同じ数値を各ケースの `estimate.json`、`parameter_values_si.csv`、`parameter_covariance_si.csv` にも保存した。

## 今回の結果

| 記録 | 元bag内の解析窓 [s] | IMU件数 | 推力τ [s] | gyro τ [s] | accel τ [s] | GA姿勢RMS [deg] | 地磁気融合出力RMS [deg] | モデル＋余剰レンチ姿勢RMS [deg] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| failure1 | 19–25 | 1200 | 0.219004 | 0.037190 | 0.066206 | 5.641 | 5.753 | 2.454 |
| failure2 | 25.5–31 | 1100 | 0.355210 | 0.061599 | 0.062091 | 1.774 | 16.910 | 2.017 |
| success | 65–75 | 2000 | 0.197712 | 0.030938 | 0.063190 | 0.801 | 4.733 | 0.499 |

GAはgyro時間発展＋accel更新を表す。姿勢誤差は再現列を元mocap取得時刻へSLERPし、SO(3)測地角で比較した値である。mocap範囲外への外挿はしていない。`success` は元の飛行記録の名前であり、今回の同定モデルで飛行を再現できたという意味ではない。

GA姿勢はmocapに比較的近いが、地磁気の出力融合はfailure2・successで悪化した。地磁気を融合しない時刻の出力はGAへ戻るため、その列には切替による振れも現れる。結果を見て地磁気を棄却したり、mocapを使ってゲイン・ノイズを調整したりしていない。

余剰レンチは、遅れ補正したIMUの比力・角速度・角加速度から得る重心まわり必要レンチと、推力モデルのレンチとの差である。各元取得時刻の値を補間し、RK4各段の力・トルクへ必ず加算する。余剰レンチも機体パラメータに依存するため、その全16変数の感度を軌道の共分散伝播へ含めた。実使用の余剰レンチ平均と同じ不確かさを別図でも示す。残差の非中心二次モーメントに由来する追加誤差は、既存の経験的な近似として分けて記録し、余剰レンチ自体の厳密なBayes事後とは説明しない。

初期姿勢・位置は最初のIMU時刻のmocapへ一度だけ合わせ、初速度はそれより前1秒の位置列から与える。以後のmocapを推定・再生へ戻さず、旧mocap再生にある区間ごとの外乱平均適合も行わない。時定数は今回のモデルと前処理に対する有効推定値であり、ハードウェア単体の遅れを独立に校正した結果ではない。

## 図と確率的な広がり

各ケースには以下の図をPNGとPDFで保存した。元のmocap版と同種の残差・スペクトル・軌道図に、IMU姿勢比較と不確かさを追加している。

- 推力指令と一次遅れ後の推力、比力・角加速度の適合。
- 残差wrenchの時系列と元の不等間隔時計によるLomb–Scargleスペクトル。
- 実際に加えた余剰レンチの時系列と、その同じlawの±2σ帯。
- gyro・accelの生値、遅れ補正値、モデル＋余剰レンチのセンサ応答。
- mocap、GA、地磁気融合出力、モデル＋余剰レンチの姿勢列とSO(3)誤差。
- 姿勢の±2σ帯、余剰レンチ付き再生とIMU積分による位置列、その全幅の±2σ帯・楕円体。

帯・楕円体は同じ推定Binghamまたは伝播した共分散因子から計算している。ランダム軌道の描画やモンテカルロ推定は使っていない。平均列を見る図と全分散を見る図を分け、広い分布を表示範囲で隠していない。

加速度と姿勢に同じ情報を使う近似では、未知の交差項に対する宣言した共分散上界を使った。ただし、姿勢再帰内ではSG窓の時間相関・共通の推定τが作る時間相関を保持しない近似が残る。局所Gaussianと代数的な上界は、系統的な磁場誤差・動的加速度を含む全誤差の被覆保証ではない。式と近似の位置は [METHODS.md](METHODS.md) を参照。

## 保存データ

公開先は `publication/imu_native_20261031/`。生bag、最適化の中間Jacobian、重複SVG・PDF分割中間物を含めず、容量は `manifest.json` に記録した。

- `inputs/<case>.npz` とJSON：元のIMU・地磁気新規mask・指令・mocap・初期条件、元時刻とsource index、静止ノイズ校正、bag hash。
- `<case>/fit_results.npz`：適合曲線・残差・全パラメータ共分散と因子。
- `<case>/arrays.npz`：再帰／出力のBingham A、姿勢・共分散、補正信号、再現列と全状態共分散因子。
- `manifest.json`：公開ファイルの容量・SHA256と、省略した重複表現の説明。`run_provenance.json` は実行時の時計・実装hashを保存する。

軌道の全状態共分散は `L = <branch>_state_covariance_factor[k]` に対して `C = L @ L.T` で復元できる。状態順序はFC位置、body右回転、FC速度、body角速度（各3次元）。位置・方位間を含む交差項を因子から削除していない。個別の位置・方位周辺共分散も保存した。

付加した余剰レンチ平均は `model_applied_residual_wrench_body`、同じlawの因子は `model_applied_residual_wrench_covariance_factor`。モデル状態因子31列はパラメータ16列・初期値9列・追加持続誤差6列、レンチ因子22列は同じパラメータ16列・同じ追加持続誤差6列に対応する。レンチ因子の16列目の後へ零列9本を挿入して列を揃えれば、状態・レンチ間の交差も両因子の積から復元できる。

## 再実行

リポジトリルートで、保存済みの小容量入力から再実行できる。元bagやROSの再生は不要で、既存のProbTF・grape物理モジュールはリポジトリ内から読む。NumPy・SciPy・MatplotlibとPopplerの `pdfinfo`・`pdfseparate`・`pdfunite` が必要。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /usr/bin/python3 ros/examples/grape-param-estim/minimal/imu-first-order-lag/run.py \
  --inputs-root ros/examples/grape-param-estim/minimal/imu-first-order-lag/publication/imu_native_20261031/inputs \
  --output /tmp/imu-first-order-lag-rerun
```

公開用の選別は `publish_results.py <実行出力> <新しい公開先>` で再現できる。既存mocap版配下を出力先にすると拒否する。最適化キャッシュを明示再利用する場合も、時計だけでなく全同定入力配列・物理モデルのhashを照合する。

```bash
python3 -m pytest -q ros/examples/grape-param-estim/minimal/imu-first-order-lag/tests
```

33件のテストが成功した。解析微分、全SG時間重なり、全時定数cross、地磁気の再帰非混入、mocap非混入、元時計での比較、四元数符号境界の積分、表示と保存共分散の一致、mocap出力保護を検査する。さらに、非零の余剰レンチが平均軌道を実際に動かすことと、余剰レンチを同じパラメータ候補から再計算した全16変数の微分を検査した。

余剰レンチ付き再生の積分刻みを5 msから2.5 msへ半減した最大位置差は、failure1で1.673 mm、failure2で0.0406 mm、successで1.472 mmだった。これは数値積分の確認であり、mocapとの再現誤差ではない。
