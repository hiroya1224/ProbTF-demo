# ProbTF：リンク変換の候補法則を使う確率尤度IK

各リンクの初期局所変換の共同法則を入力し、関節指令による変換を伝播して、棒先端と画像由来の目標密度から指令を選ぶ。
IKの入力は直接設定した変換法則であり、剛性から候補を生成したり、関節角のGaussian誤差へ置き換えたりしない。
重力と関節ばねは、実行対象を模擬するシミュレータだけで使う。

実装の基準は `~/dthesis/chapter/4-transform-law-propagation.tex` と
`7-transform-law-sharing-infrastructure.tex`。元の法則、演算式、有限和・標本による照会結果を区別する。
旧版の到達記録は現行版の検証証拠に流用しない。

## 起動

ROS Noetic、Pinocchio、NumPy、SciPy、PyYAML、AprilTag辞書を含むOpenCVと、
既存の `deflecomp_core`、`symaware_grasp`、`prob_artag_detector` を使う。

```bash
cd ~/catkin_ws
catkin build probtf_switch_press_demo --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
roslaunch probtf_switch_press_demo switch_press_demo.launch
```

既定では初期姿勢を5秒表示してから開始する。RVizが既に開いている場合は `rviz:=false` を付け、
[`rviz/switch_press_demo.rviz`](rviz/switch_press_demo.rviz) をそのRVizで読み込む。
手動開始には `autostart:=false` を付け、次のサービスを呼ぶ。終了後の再実行にも同じサービスを使う。

```bash
rosservice call /switch_press/replay "{}"
```

実行中の重複要求は受け付けず、再実行の記録先は分ける。
launchの `config:=/absolute/path/config.yaml` で設定を変更できる。
記録先の既定値は `/tmp/probtf_switch_press_demo/<日時>/`、指定する場合は `output_dir:=/absolute/path`。

ROS masterと画面を使わず、同じ画像・実時間制御経路を実行する場合：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  rosrun probtf_switch_press_demo switch_press_demo.py --headless \
  --output /tmp/switch_press_headless
```

## 初期変換法則の設定

[`config/default.yaml`](config/default.yaml) の `control.initial_transform_law` が、
[`config/initial_transform_law.yaml`](config/initial_transform_law.yaml) を指定する。
相対パスはこのROSパッケージを基準に解決する。`law_id`、`revision`、`reference_command` と、
`base_link -> link_1 -> ... -> link_6 -> tool0 -> rod_tip` の全局所変換を与える。

| 入力 | 設定と依存関係 |
| --- | --- |
| `kind: joint_finite` | `frame_names` と `scenarios` を指定。各シナリオに `id`、`weight`、全リンクの `local_transforms` を与える。任意の有限のリンク間相関を共同の組として表す。 |
| `kind: independent` | `edges` の各局所辺にnativeな混合成分を指定し、`independence: true` を明示する。辺間の積法則を仮定する入力。 |

有限共同候補の各変換は4×4行列、または `translation` と `quaternion_wxyz` の組で指定できる。
独立な辺の成分は `id`、`weight`、`orientation`、`translation` を持つ。
方位は `dirac` の `quaternion_wxyz`、`uniform`、または `bingham` の4×4 `parameter_matrix`。
並進は `mean` と、任意指定の3×3 `covariance`、3×9 `coupling` を持つ。省略した共分散と結合はゼロ。
`coupling` は列優先の `vec(R)` と並進の条件付き平均の結合であり、辺間の独立性とは別の情報である。
必要な共同法則が指定されていない共有連続依存には、独立性を補って計算せず拒否する。

既定ファイルは各可動辺に2個のDirac変換、手先と棒先端には各1個の固定変換を明記している。
既知の剛体基準姿勢に対する軸回り±0.08 radの変換を例示した、未較正の有限事前法則である。
実剛性や実姿勢から生成した分布ではない。6辺の独立性を明示しているため、全64組を正確に列挙する。
この64組は元から有限の入力法則であり、仮定した連続分布の標本ではない。

nativeなBingham回転や条件付き並進が連続の場合も、元の成分・結合・重みをそのまま保持する。
照会には `sampling: {count: 64, seed: 21}` などの方針で一度生成した共同標本を使う。
同じ全身の標本行を全指令・動作前後で再利用し、標本表を元の法則として登録し直さない。
この場合は `joint_transform_samples`、`monte_carlo`、標本数、seed、元法則の版を結果へ記録する。
有限入力の `joint_finite_pushforward`・`finite_sum` とは型と近似情報を分ける。

## 伝播と指令の選定

全初期局所変換の共同法則を `Lambda_ref` とする。
既知の関節軸による右回転 `D_i` を使い、`T_i(u) = T_i(ref) D_i(u_i-u_ref_i)` として各候補を写す。
固定辺には関節回転を加えない。これがこのデモで明示した指令の作用モデルであり、
未知の重力たわみが指令に伴ってどう変化するかまでIKで予測するものではない。

指令後の共同法則 `Lambda^u` と、根元から各リンクへの `nu_j^u` は、この写像と変換合成による押し出し。
同じ候補内の局所変換を合成し、各リンクの周辺分布を独立に組み合わせて別の身体を作らない。
動作前後の相対変換も同じ共同実現値を使う。変換とその逆は周辺化する前に相殺する。
先頭の初期変換も指令前後で代数的に相殺し、同一点の密度評価を共有する。共同候補の行は削減しない。
棒の局所取付が確定なら、ゼロ増分の先端は観測時手先から見て固定点になる。
その取付自体を不定に設定した場合は、取付の不確かさを保持する。

各画像から得た目標密度を `p_goal`、伝播した手先法則を `mu_hand^u` とすると、
規準は `K(u) = integral p_goal(x) mu_hand^u(dx)`。
有限入力では全候補の `logsumexp(log w_s + log p_goal(h_s(u)))`、連続入力では同じ元法則からの共同標本で評価する。
平均姿勢だけの密度評価や、成分別対数尤度の平均には置き換えない。Kは期待密度であり、到達確率ではない。

各候補の指令に対するヤコビアンと目標密度の空間勾配を合成し、L-BFGS-Bで `-log K` を最小化する。
微分IKは初期指令の提案に使う。既定は1回2反復、観測時指令から各関節±0.06 rad以内。
候補を期待密度で順位付けし、経路条件を満たす一つの指令を選ぶ。大域最適解は保証しない。
Monte Carlo評価では期待密度の相対標準誤差を推定し、設定した許容値を超えれば選定を拒否する。
これは経験的な指標で、見落としたモードや稀な寄与まで含めた誤差上界ではない。

最初のIKの計算中から、安全な途中指令を適用して移動と画像からのIK計算を別スレッドで進める。
途中指令も期待密度の精度と全候補の経路条件を検査し、実行する候補と同じ問い合わせ結果を表示する。
移動側は20 Hzを目標に、速度0.08 rad/s、加速度0.30 rad/s²以内で絶対指令先を追う。
遅れて返った増分は撮像時の指令へ加え、指令先を更新しても速度をリセットしない。
観測が更新できない間は直前の指令先を追い、有効期間5秒を超えると減速する。

## カメラと目標密度

180 mmのAprilTag 36h11、ID 21を960×960の仮想カメラで撮る。
焦点距離は720 pxのまま縦の画角を広げ、初期姿勢と接近終盤のどちらでもマーカーの余白を残す。
240 msの露光を7時点の投影で近似し、Gaussianぼかし2 pxと輝度ノイズ8/255を加える。
この画像を既存の角点検出器と姿勢推定器へ渡し、適応角点共分散からnativeな
Bingham回転・条件付きGaussian並進の混合法則を得る。単画像の推定であり、履歴を融合した事後分布ではない。

既知のカメラ取付とマーカーから目標への固定変換で、混合重み・残差共分散・回転並進結合を保って伝播する。
身体の初期法則と画像由来の目標法則には独立性を仮定する。
IKには実関節角、実剛性、マーカーの実世界姿勢、正解角点、表示用TFを渡さない。

`ExactIslBackend` は第4章 `jmaa:eq:globaldensity` の有限Bingham誘導方向密度を球面面積測度に対して評価する。
ファイバー積分の閉形式と数値Bingham正規化定数を使い、Dirac・一様分布も扱う。
目標の3次元点密度は別の `ConditionalPointDensity` 問い合わせであり、正定値の残差共分散を必要とする。
元のBinghamと3×9結合を保って、条件付きGaussian密度を全 `S^3` 上で数値積分する。
集中時のgnomonic・Gauss–Hermite座標と、広い分布での有界 `RP^3`・Gauss–Legendre座標は積分のために使う。
元の法則をGaussian代理へ置き換えない。停留点探索の網羅性や、密度の大域的な誤差上界は保証しない。
数値評価に失敗した場合はエラーとして扱う。

## RVizとシミュレータ

不透明なモデルは、実剛性で重力と関節ばねの平衡を解いたシミュレータの姿勢。
Alpha 0.25のモデルは同じ指令を与えた剛体姿勢。棒は長さ325 mm、質量275 gで、取付部とカメラの荷重も含む。
`plant.stiffness` はこのシミュレータだけの設定で、IKの初期変換法則には使わない。

| ProbTF表示 | 意味 |
| --- | --- |
| `root next links` | 選んだ次の指令の `nu_j^u`。根元基準の全8フレーム。 |
| `relative next tip` | 観測時手先基準の、対応を保った動作後の先端法則。IK入力。 |
| `relative observed goal` | 同じ観測時手先基準へ伝播した目標法則。 |
| `observed AR marker` | 検出器のnative法則を観測時カメラの配置へ固定して表示。 |

既存の `probtf_rviz/ProbabilisticTF` と `ProbabilisticPose` を使い、身体の共同候補または共同標本の各行を一度ずつ描く。
点のAlphaは最大重みに対する相対重み。描画時に候補を引き直したり、点群の履歴を積み重ねたりしない。
ARマーカーと目標のnative法則の点描画は表示用であり、IKの目標密度積分には使わない。
赤・緑・青の点は変換が写した各軸端点であり、原点の位置標本ではない。
点サイズは以前の2倍で、リンク6 mm、比較用先端8 mm、目標4 mm、ARマーカー6 mm。

代表軸は問い合わせ結果の位置の重み付き平均と四元数2次モーメントから得る。
実姿勢に中心を付け替えず、支持点と重みを保持する。各リンクの平均軸が一つの身体を構成するとも限らない。
最後に採用した安全な途中指令または最終指令の予測を表示する。現在の実姿勢の推定表示ではない。
元の法則レコードのstampを保ち、配列のheartbeatと受信時刻で表示の鮮度を管理する。
再送を新しい観測や時間予測として扱わない。

## 検査・記録と限界

全候補について、URDF形状を包含する直方体とパネル・ボタン・ランプを分離軸検査する。
関節増分0.008 rad、形状対応点の移動2 mmを基準に経路を分割し、隙間下限4 mmを要求する。
指令先変更時も接続区間を検査する。実軌道は別に監視し、実先端が目標から6 mm以内かつ非干渉の場合だけランプを緑にする。
有限候補・標本と観測成分の代表姿勢による検査であり、連続法則全体への安全証明ではない。
Gazebo、接触力学、ボタンのストローク、自己干渉、床、画像内のロボットによる遮蔽は扱わない。

`initial_transform_law.json` に元のnative法則と指令写像の参照を保存する。
`settings.yaml`、`robot.urdf`、`applied_ik_updates.json`、`cycles.json`、`servo.json`、`trajectory.csv`、`phases.json`、`summary.json` と画像に、
設定、観測、共同候補・標本のIDと重み、選定、実軌道、結果とソースの版を記録する。
近似方法・標本数・seedは問い合わせ結果に付す。保存しただけで妥当性が検証済みになるわけではない。

```bash
cd ~/catkin_ws/src/ProbTF-demo
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 -m pytest -q \
  tests/probtf/test_transform_candidates.py tests/probtf/test_joint_law.py \
  tests/probtf/test_exact_induced_direction.py tests/probtf/test_point_density.py \
  tests/probtf/test_kernel_evaluation.py tests/probtf_switch_press_demo
```

テストはnative入力の保持、有限共同重み・独立な積の列挙、同じ共同実現値の再利用、逆変換の相殺、
期待密度と勾配、既知の関節作用、表示での支持点・時刻の保存を対象とする。
現行版のGUI実行では、実先端誤差5.65 mmで到達し、公開されたリンク・手先・目標の法則が実行記録に一致した。
設定・照合結果・圧縮した実行記録は [`outputs/native_transform_law_20261016/`](outputs/native_transform_law_20261016/) に保存した。
任意の共有連続Bingham依存や確率的逆変換の閉形式を一般に実装済みとはしない。

[`outputs/`](outputs/) にある剛性候補版、表示専用誤差版などの旧出力は、不採用となった旧版の記録として残す。
当時の成功数値や動画を、現行版の検証証拠へ流用しない。
