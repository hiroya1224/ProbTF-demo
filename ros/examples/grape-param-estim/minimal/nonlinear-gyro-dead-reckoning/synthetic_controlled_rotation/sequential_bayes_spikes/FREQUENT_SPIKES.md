# 同じスパイク入りgyroによる非線形・ジャイロ項なしの比較

[上書きした図・比較動画・結果](frequent_publication/INDEX.md)。比較するのは、同じスパイク入り観測を受け取る **非線形Euler** と **ジャイロ項なし** の二モデルだけである。白色ノイズのみの対照と旧単発比較の出力は削除した。

## 今回の入力と時計

20秒の、実飛行の規模に合わせた3周波数の人工姿勢運動を使う。profile名は`flight_matched_three_frequency`。ジャイロの生データは200 Hz、Bayes更新に取得する測定は1 Hzの0、1、…、19秒。更新の間は、各モデルの角速度式と共通の四元数式で共同候補を時間発展させる。

白色ノイズの生成標準偏差はユーザー指定の **2 deg/s**。各軸独立・等方で、旧σ=0.2の標準Gaussian実現値を同じseedから10倍して生成する。旧保存測定を上書きしない。固定bias `[0.03,-0.02,0.04] deg/s` は従来と同じで、推定には教えず差し引かない。生成する分散と実現した標本標準偏差は別々に保存する。推定の観測モデルは引き続き等方σ=5 deg/sで、生成側のσ=2とは区別する。

スパイクは0〜15秒に、強度10回/秒のPoisson過程で開始する。連続時刻で生成した開始時刻を200 Hzの原時計へ写し、幅0.03〜0.08秒の矩形パルスを加える。単独パルスのベクトル長は20〜60 deg/sの一様乱数、方向は等方。パルスが重なるとベクトル和になり、合計値が60 deg/sを超える場合もある。

開始時刻、幅、大きさ、方向の乱数ストリームを分離する。開始時刻は1 Hzの取得時計から独立に決め、取得点へ強制配置しない。20秒の固定seed実現値では、15秒間に147回のスパイクが発生し、1 Hzの取得と重なったのは5、8、10、12、13秒の5件。その間にも正常な取得があり、14〜19秒も正常な更新が続く。この個数を得るためのseedの再選択はしていない。

15〜20秒は、あらかじめ定めた回復観察区間としてスパイクの加算を止める。境界をまたぐパルスも15秒で終了させる。白色ノイズσ=2は止めない。これは生成入力の明示的な切替であり、フィルタがスパイクを検出・除去したという意味ではない。推定器には、この切替時刻もスパイクフラグも渡さない。

## 実飛行から定めた運動の規模

校正元は、一日目#8のbagの、既存の独立診断で固定していた40〜60秒区間。mocapのZが初期より約0.60 m高い区間であることを確認したが、飛行全体の漸近安定性まで証明した区間とは扱わない。実gyroは約200 Hz、mocapは約120 Hzで、元のheader・recordのns時刻とsource indexを保持する。

周波数と角速度の強さは、mocap四元数に対する既存の中心0.32秒・3次SG微分から得たbody角速度を参照して定める。四元数の正規化の微分を含み、`omega=2 vec(q^-1 qdot)`を使う。これは人工軌道を設計するためのオフライン校正であり、人工実験のフィルタへ継続的なmocapやSG微分を入力する処理ではない。生gyroは別の参照列として比較図に残す。

全20秒のHann periodogramを、線形detrendと周波数分解能0.05 Hzで計算する。0.05〜3 Hzの範囲から、0.10 Hz以上離れた強い局所極大を各軸3つ選ぶ。スペクトル計算のための等間隔化は校正診断に限り、人工gyroの取得時計や推定入力を変更しない。

固定した周波数のsin/cos 3組と定数項を、測定RPYへ最小二乗で合わせて振幅・位相・offsetを得る。その後、各RPY軸の振幅に正の倍率を掛け、厳密なRPY→body角速度変換で得る各軸標準偏差が、参照body角速度の1.5倍になるよう連成して校正する。

| body軸 | 採用周波数 [Hz] | 実mocap-SGのω標準偏差 [deg/s] | 人工ω標準偏差 [deg/s] |
|---|---|---:|---:|
| x | 0.25, 0.10, 1.30 | 2.91985 | 4.37977 |
| y | 0.75, 0.90, 0.60 | 8.30555 | 12.45833 |
| z | 0.25, 0.45, 0.75 | 3.51241 | 5.26861 |

これは各軸の角速度標準偏差を1.5倍にする指定であり、各RPYの振幅や角速度の最大値を一律1.5倍にする処理ではない。旧人工運動の標準偏差は約`[75.00,62.19,80.99] deg/s`で、現在のprofileでは約`[4.38,12.46,5.27] deg/s`となる。表示軸だけを縮小せず、新しい姿勢軌道から角速度・角加速度・既知トルクをすべて計算し直す。

3本の全域調波では、実飛行の非定常な時間波形やスペクトルの全成分は再現できない。ここで合わせたのは主要周波数と運動の統計的な強さであり、実飛行の再生とは呼ばない。また、profileの選定へ推定結果、推定共分散、白色ノイズやスパイクの実現値を使わない。センサ白色σ=2、観測モデルσ=5、スパイク過程は運動の校正と独立に維持する。

数値と出自は[校正記録](flight_motion/calibration.json)、生成に使う係数は[profile JSON](flight_motion/flight_motion_profile.json)、実飛行・旧人工運動・現在の人工運動の同じ単位での比較は[PDF](flight_motion/comparison.pdf)にある。元のbagを追加コピーせず、既存の小容量抽出入力とそのhashを記録している。

## 保持するBayes法則

状態は両モデルとも \(\omega,q\) で、\(q\in S^3\) はbodyから世界へのHamilton四元数（wxyz順）。回転行列を状態として時間積分しない。未知量は同じ初期body角速度 \(W_0\)。初期姿勢 \(q_0\)、慣性 \(J\)、body重心まわり入力トルク \(\tau(t)\) は既知固定。

\[
\dot\omega=J^{-1}\{\tau-\omega\times(J\omega)\},\qquad
\dot q=\tfrac12 q\odot(0,\omega).
\]

ジャイロ項なしのモデルでは角速度式だけを \(\dot\omega=J^{-1}\tau\) に替え、四元数式は変えない。J・トルク・時刻・初期姿勢・初回gyro・観測共分散・全取得値を共通にする。

モデル \(m\) の流れを \(\Phi_t^m(w)=(q_t^m(w),\omega_t^m(w))\) とすると、初回gyroを一度だけ使った初期法則から、

\[
\pi_{m,k}(w)\propto
\mathcal N(w;y_0,\Sigma_g)
\prod_{j=1}^{k}\mathcal N(y_j;\omega_{t_j}^m(w),\Sigma_g),
\qquad \Sigma_g=(5\pi/180)^2I_3
\]

を保持する。姿勢・角速度は \((\Phi_t^m)_*\pi_{m,k}\) から同時に得る。初回の \(y_0\) を再び尤度として数えず、観測のたびに新しい独立Gaussianへ角速度を置換しない。

推定は両モデルで同じスパイク入り測定列と通常のGaussian尤度を使う。外れ値除去、ロバスト尤度、BFS、過程雑音の追加は行わない。過去のスパイクも過去の尤度として残るため、正常観測が続けば必ず速やかに真値へ戻ると保証するモデルではない。

## 適応するのは積分点の配置

旧試験の固定した事前中心の求積では、多数の観測で細くなった事後を解像しにくい。今回は積分用のGaussian proposal \(g_k(w)\) を置き、tensor Gauss–Hermite求積をその位置・尺度で行う。

\[
\int f(w)\,\pi_k(w)dw
\approx
\frac{\sum_s b_s f(w_s)\,\widetilde\pi_k(w_s)/g_k(w_s)}
     {\sum_s b_s\,\widetilde\pi_k(w_s)/g_k(w_s)},
\]

ここで \(\widetilde\pi_k\) は上記の初期Gaussianと全取得尤度の非正規化積、\(b_s,w_s\) はproposalに対する決定論的求積重み・点である。\(g_k\) を重みから除くので、proposalをBayes事後そのものへ代入する処理ではない。

proposalの中心は、初期Gaussianとその時点までの全観測残差の有限回の数値最小二乗で配置する。共分散はそのGauss–Newton行列の逆で、**積分点の配置用**に限る。実際に表示・保存する姿勢と角速度の平均・共分散は、全target/proposal重みから再計算する。proposalの有限反復状態と、全候補の事後モーメントを混同しない。

新しい求積点で過去の尤度を再評価することは、その尤度を法則へもう一度掛けることではない。毎回、元の初期事前×取得済みの各尤度一回ずつ、という同じtargetを評価する。現在より後の測定は使わない。前回の候補による観測直前を保存し、新しい候補による観測直後を同じ原時刻へ別記録する。既に出力した過去のオンライン列を修正する平滑化はしない。

proposalの局所探索と有限求積は、大域的な積分の厳密解を保証しない。7・11・15点/軸の精細化と、同じ事前を新旧の求積で評価した差を記録する。有限な未収束や積分差を理由に観測を棄却せず、以降の更新を続ける。モンテカルロ推定、枝棄却、共分散floor、jitterは使わない。

## C++の伝播と保存する量

積分点ごとの角速度・四元数の同時RK4評価だけをC++で行い、Python基準実装と同じ式・原トルクの補間・最大刻み0.002秒を使う。コンパイル済みライブラリは`/tmp`のキャッシュへ置く。確率法則や既存ProbTF coreの変更ではない。

各時刻の同じ物理的な候補と重みから、四元数二次モーメント、回転平均、全9×9回転共分散、全3×3角速度共分散、全9×3回転–角速度交差共分散を保存する。RPYは従来と同じ \(f(E[R])\) とJacobianによる局所delta-methodの±2σ表示。保証付き95%領域と呼ばず、表示近似を推定へ戻さない。

生データ4001件、スパイク147件のログ、原nsとindex、1 Hz取得20件、観測直前・直後、proposalパラメータと求積点・対数重みを保存する。`event_proposal_covariance_rad2_s2`は積分配置用、`omega_covariance_rad2_s2`は実際の角速度候補の共分散である。

図には、滑らかな推定平均に加えて、200 Hzの生測定を間引き・平滑化せずに描く。生gyroの全体・固定2〜4秒拡大を示す。角速度は実際に保存した全3×3共分散から±2σを描き、推定誤差と±2σ、および半幅だけの図も併記する。真値を引く推定誤差図は表示・評価専用で、フィルタへ戻さない。

`attitude_and_gyro`本体は三段×三軸とし、上段をRPY、中段をωの平均±2σ、下段を同じω共分散から得た2σ半幅とする。帯・境界線を生測定より上に描き、下段で観測直前を白抜き、直後を塗りつぶしの点で示す。半幅の独立した縦軸はdeg/sであり、元の共分散や帯の数値を拡大する処理ではない。

生成側と推定側に一致する既知J・入力トルクを与えた合成回転の試験である。未知の機体誤差を同定した結果、実機の飛行全体を再現した結果、スパイクを除去した結果とは区別する。

## 再現

リポジトリのルートから実行する。NumPy、SciPy、Matplotlib、Pillow、FFmpeg、g++を使用する。初回に同梱のC++ソースを`/tmp`へビルドする。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/synthetic-bayes-mpl \
python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/flight_motion/calibrate.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/synthetic-bayes-mpl \
python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/frequent_run.py \
  --output /tmp/synthetic-flight-matched-comparison-20261120

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/verify_flight_publication.py \
  /tmp/synthetic-flight-matched-comparison-20261120

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover \
  -s ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation/sequential_bayes_spikes/tests -v
```

最初のコマンドは、既存の実測抽出列から`flight_motion/flight_motion_profile.json`と校正図・数値を再生成する。人工実験の設定は`frequent_config.json`で、現在のprofileはこのJSONから読む。seedや入力の強さを推定結果によって選び直さず、異なる設定は別出力へ保存する。Gitにはソース・小容量入力／数値・完成図・完成MP4だけを追加し、共有ライブラリ、フレーム、途中の動画は追加しない。

`verify_flight_publication.py`は保存済み出力に対する独立の事後診断であり、観測の受理判定や更新の早期終了には使わない。同じスパイク測定列、変更していないノイズ成分、全交差共分散、線形側の解析的な事後との一致、生成軌道と動力学の整合、完成動画の全フレームのdecodeを記録する。今回の検証では、この実験の29テストと既存軌道の8テストも通過した。
