# mocap姿勢のSG微分とジャイロ照合・再積分

2026-11-13。直前の非線形予測で用いた一日目#8のbag時刻50.000988619〜55.000988619秒について、二つの検査を独立に行った。元のデッドレコニング予測・力学モデル・トルク平均・時刻・測定値は変更していない。

- [ジャイロ比較PDF](publication/bag8_50_55s/gyro_comparison.pdf)：原時計の波形、時間ずれだけを動かした比較、lag走査。
- [SG微分→再積分PDF](publication/bag8_50_55s/sg_roundtrip.pdf)：全レート・0.5 Hz保持の姿勢比較と誤差曲線。
- [数値結果](publication/bag8_50_55s/RESULTS.md)、[lagのCSV](publication/bag8_50_55s/lag_table.csv)、[再積分CSV](publication/bag8_50_55s/roundtrip_table.csv)。

## mocap姿勢から角速度を求める方法

mocapは `/gimbalrotor/mocap/pose` のworld姿勢、ジャイロは `/gimbalrotor/imu` のbody角速度を使用。両者のheader/ns時刻とbag record/ns時刻、source indexをそれぞれ保存した。bodyとFCの軸一致は既存の静的TFを照合し、nativeとconverted gyroは同一stampの4,200件で値が完全一致した。

RPYを個別に微分した値はbodyジャイロと同じではないため、四元数を使った。原mocap時刻のまま、四元数の符号を連続に揃え、各成分に中心化した3次局所多項式SGを当てる。窓幅は0.08、0.16、0.32秒。主条件0.16秒は結果を見る前に定めた。補助入力を39.5〜60.5秒から読み、評価の前後にも完全な中心窓を確保した。

多項式の値を \(u\)、その微分を \(\dot u\) として、正規化とその微分を

\[
q=\frac{u}{\|u\|},\qquad
\dot q=\frac{\dot u-q(q^\top\dot u)}{\|u\|}
\]

で計算し、bodyからworldへ写すHamilton四元数について

\[
\omega_{\rm SG}(t)=2\operatorname{vec}\bigl(q(t)^{-1}\odot\dot q(t)\bigr)
\]

を使った。SGで原時計を等間隔へ置換していない。中心窓は未来のmocapも使うオフラインの処理であり、因果的なリアルタイム推定器ではない。

## ジャイロとの一致と時間ずれ

生ジャイロは平滑化・bias除去・scale変更・軸回転の再推定をしていない。比較するのは

\[
\sum_i\|y_g(t_i)-\omega_{\rm SG}(t_i-\ell)\|^2
\]

である。**正のlag \(\ell\) はgyro側の特徴が後に現れる**規約。−250〜+250 msを1 ms刻みで走査し、最小点近傍だけ数値的に精細化した。全シフトで同じ原gyro行を使い、都合のよい時刻だけを残していない。SG微分値の時刻ずらしには原mocap時刻上の微分を線形補間する近似がある。

主区間50〜55秒のほか、40.5〜59.5秒の広域と、別の過去40.5〜49.5秒でもlagを求めた。過去だけで求めたlagを主区間へ適用した値も保存し、主区間への当てはめと区別した。元の測定時刻を書き換えたものではない。

主条件0.16秒では、主区間の最良lagは **+4.163 ms**、軸別RMSEは次のとおり。

| 軸 | 原時計のRMSE [deg/s] | 最良lag適用後 [deg/s] | 原時計の相関係数 |
|---|---:|---:|---:|
| x | 1.9613 | 1.9569 | 0.8638 |
| y | 3.7644 | 3.7624 | 0.9415 |
| z | 1.4087 | 1.4056 | 0.8640 |

vector RMSEは4.4724→4.4678 deg/sで、時間ずれだけを合わせてもほとんど改善しない。SG窓0.08秒では最良lag14.250 ms、0.32秒では17.132 msだった。窓を広げると微分のノイズは減る一方で帯域も変わるため、数値の良さだけで一つを真の遅れと採用していない。0.16秒の過去区間から求めたlag9.815 msを主区間へ適用すると、vector RMSEは4.4831 deg/sだった。

bag record時計で同じ主条件を比較した最良lagも+4.877 ms。record−headerの平均はgyro0.802 ms、mocap0.496 msだった。これらは時計・転送経路の診断であり、センサの物理遅れの直接測定ではない。ここで得た相対時間ずれを、一次遅れモデルの時定数とも同一視しない。

## SG微分の結果を再積分すると戻るか

同じ5秒区間で、最初のmocap姿勢を一度だけSLERPで与え、

\[
\dot Q=\tfrac12 Q\odot(0,\omega_{\rm SG}(t))
\]

を積分した。その後のmocap姿勢によるリセットや補正はない。力学式・慣性・制御トルク・余剰トルクはこの往復検査には入れない。

全レート条件では、原mocap約120 Hzから得る局所多項式を2.5 ms間隔で数値評価し、その曲線を線形補間してRK4各段へ渡した。この400 Hz格子は新しい観測を追加したものではない。疎な対照では、同じSG曲線を開始時と2秒後・4秒後の3回だけ取り、次の取得まで一定として厳密な固定回転増分を合成した。疎な対照の時刻は正確な2秒間隔であり、前回の原gyroの取得時刻にあった約27 µsのずれとは区別して記録した。

SG窓0.16秒の、原mocap姿勢に対するSO(3)誤差は次のとおり。

| 再積分入力 | RMSE [deg] | 最大 [deg] | 終端 [deg] |
|---|---:|---:|---:|
| 全レートSG角速度曲線 | **0.1743** | 0.8592 | 0.0914 |
| 同曲線を0.5 Hz取得・保持 | **11.6475** | 20.1174 | 10.8614 |

全レート再積分とSGで平滑化した姿勢とのRMSEは0.0480度だった。原mocapの細かな変動を完全に戻すわけではないが、元のRPY軌道をよく復元できている。最大積分刻みを2 ms→1 msに半減した姿勢差は約 \(2.33\times10^{-12}\) 度で、この検査では数値積分の刻みよりSG近似・観測の細かな変動の方が大きい。

これによって、body角速度の取り出し方と右四元数積による戻し方が整合することを確認した。0.5 Hzへ間引いて保持するだけで10〜15度程度の誤差が出ることも確認できた。一方、同じmocapから角速度を作った往復なので、独立したIMU推定が0.17度で成功したという結果ではない。

この検査では、先ほどの約109度の非線形モデル予測誤差を、大きな一定時間ずれや四元数の微分・積分の規約だけで説明する証拠は得られていない。元の力学モデルやトルクを、この結果に合わせて変更していない。

## 保存・再実行

公開先は `publication/bag8_50_55s/`。入力の原stamp・record・index、SG微分、lag走査、比較波形、各再積分姿勢・RPY・誤差を保存した。生bag・動画は追加していない。10件の既知解・符号・不等間隔・遅れ符号・再積分の検証が通過した。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/gyro-mocap-sg-mpl /usr/bin/python3 \
  ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/gyro_mocap_sg_check/run.py \
  --input ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/gyro_mocap_sg_check/publication/bag8_50_55s/input.npz \
  --output /tmp/gyro-mocap-sg-rerun
```

元bagからの抽出は `extract.py --output <出力先>`。選別保存は `publish.py <抽出・計算の親出力先> <新しい公開先>`。全図と保存値のSHAは公開先のmanifestに記録する。
