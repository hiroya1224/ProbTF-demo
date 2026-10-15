# 不定座標変換によるスイッチへの連続接近

カメラと325 mmの棒を持つ6軸アームが、AprilTagから推定した目標へ接近する。
IKには指令依存の局所変換分布を直接与え、全経路を伝播した手先と目標のGaussian–Bingham分布の重なりで指令を選ぶ。
重力たわみは物理シミュレータで計算し、その実カメラから生成した画像を実際の検出器へ渡す。

定式化と変更範囲は [REQUIREMENTS.md](REQUIREMENTS.md)、推定・表示の同一性とRViz保護は [AGENTS.md](../../../AGENTS.md) を参照。
現行の物理実行の検証結果は末尾に記録する。旧版の結果を現行版の検証として扱わない。

## 起動と再実行

ROS Noeticと、既存のdeflecomp_core・symaware_grasp・prob_artag_detectorを使用する。
Python側にはPinocchio、NumPy、SciPy、PyYAML、AprilTag辞書を含むOpenCVが必要。

    cd ~/catkin_ws
    catkin build probtf_switch_press_demo --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
    source devel/setup.bash
    roslaunch probtf_switch_press_demo switch_press_demo.launch

すでにRVizを開いている場合は、最後のコマンドに rviz:=false を付ける。
既定では初期姿勢を5秒表示してから開始する。
手動開始は autostart:=false を指定し、次のサービスを呼ぶ。

    rosservice call /switch_press/replay "{}"

終了後も同じサービスで初期姿勢から再実行できる。実行中の重複要求は受け付けない。
試行ごとに新しい記録先を作る。

ROS masterや画面を使わず、同じ撮像・IK・物理実行経路を動かす場合：

    source ~/catkin_ws/devel/setup.bash
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      rosrun probtf_switch_press_demo switch_press_demo.py --headless

記録先は既定で /tmp/probtf_switch_press_demo/日時/。
headlessでは --output、roslaunchでは output_dir:= を指定できる。既存の記録先は上書きしない。
別の設定は --config または config:=/absolute/path/config.yaml で指定する。

## 入力する変換法則と物理モデル

[config/default.yaml](config/default.yaml) が場面・撮像・制御設定、
[config/independent_transform_laws.yaml](config/independent_transform_laws.yaml) がIKへ直接与える核族である。
後者には根元座標系、独立性、法則ID・版、基準指令、各局所変換の並進Gaussianと方位Bingham、
指令依存性を定める right_joint_action を記述する。
この右作用は入力した核族の定義であり、IKが後から付ける角度雑音ではない。
並進の rotation_coupling を指定すると、nativeな回転・条件付き並進の結合も扱う。
局所観測のない辺のパラメータは維持する。既定の分布は未較正の例示用入力である。

主な設定は次のとおり。

| 設定 | 内容 |
| --- | --- |
| control.transform_law_config | 直接入力する局所変換核族のYAML。default.yamlからの相対パスを解決する。 |
| control.integration_steps | 既存Binghamモーメント・正規化定数の数値評価分解能。既定480。 |
| control.max_joint_step | 短い変位の設定上限。実際の各関節上限は目標までの平均距離に応じて縮める。 |
| control.jacobian_step / ik_max_iterations | 数値微分幅とSLSQP反復上限。既定の上限は6反復。 |
| marker.goal_offset / goal_orientation_wxyz | マーカーから目標への位置・方位取付け。工具軸を接近方向へ合わせる。 |
| motion | サーボ周期、速度・加速度、代表形状の経路検査条件。 |
| plant.stiffness | 物理シミュレータだけの関節剛性。IK入力には使用しない。 |

物理側は棒・カメラの質量と慣性を含むURDFで、重力と関節ばねの安定平衡を解く。
指令角と実現角は異なる。剛性や実現角からIKの候補を生成しない。
撮像には実際のたわんだカメラを使い、露光中の運動、Gaussianぼかし、輝度ノイズを画像へ反映する。
制御器へ真の関節角、マーカー世界姿勢、正解角点、表示TFを渡す経路はない。
撮像はマーカー平面の射影画像であり、腕による遮蔽や照明変化は再現しない。
強いぼけで検出できない場合は、有効な観測を作らず未検出として扱う。

指定の工具目標姿勢でもマーカーを撮像できるよう、既知のカメラ取付けrpyを
[-1.2523505788398919, 0, -1.2924966677897853] [rad] とし、光軸を手先座標の [0.35, 0.10, 0.12] 方向へ向ける。
同じ既知取付けを物理撮像と候補カメラ経路で共有する。
初期指令は [-0.15, -1.85, 2.35, 0, -0.95, 0] [rad]。
画像は960×720画素・焦点距離720画素を維持し、既存のRVizレイアウトも保持する。

## 伝播、比較規準、連続実行

Modelは独立な局所核族を工具先端まで合成し、位置・回転の結合を保持してモーメントを求める。
工具の長さに由来する位置・回転の交差共分散も含める。
全伝播の後にだけ、手先を位置Gaussianと単一Binghamの積へ近似する。
方位は伝播した四元数2次モーメントに整合させる。

目標は撮像指令 \(c_t\) のカメラ候補、画像由来のnativeマーカー法則、位置と方位の目標取付けを、
\(T_{BG}=T_{BC}T_{CM}T_{MG}\) の全経路で根元座標系へ伝播する。
最終比較では、全方位成分に共通の1つの位置Gaussianと、重みを保持したBingham混合にする。
1回のIKではこの \(p_G\) を固定し、将来指令 \(q=c_t+\delta\) の手先分布 \(p_H\) を変える。

\(d=m_H(q)-m_G\)、\(S=S_H(q)+S_G\) として、次の全目的関数を最小化する。

\[
\begin{aligned}
J(q)=-\log K(q)
={}&\frac12 d^{\mathsf T}S^{-1}d+\frac12\log\det S
+\frac32\log(2\pi)+\log Z(A_H(q))\\
&-\operatorname{LSE}_k\!\left[
\log w_k+\log Z(A_H(q)+A_{G,k})-\log Z(A_{G,k})
\right].
\end{aligned}
\]

指令依存の共分散の行列式と全Bingham正規化項を残し、混合の重なりを足してから対数を取る。
IKは既存の決定論的モーメント・正規化定数を使い、Monte Carloや点群密度へ置き換えない。
平均の微分IKは初期値にだけ使用する。

今回の許容集合には関節範囲・短い変位の上限に加え、以下の条件を含める。

- 固定した目標平均に対する手先平均の距離を、撮像指令時から5%以上減らす。
- 同じモデルの代表カメラ姿勢で予測した各マーカー成分の四隅に、画像幅・高さの15%の余白を確保する。
  新しい観測ですでに15%未満の方向は、観測余白に2パーセントポイントを加えた値（上限15%）まで回復させる。
- 同じ候補法則の代表形状を使い、途中点を含む非干渉条件を検査する。

撮像時の手先平均と目標平均の距離を \(D_t\)、設定された変位上限を \(L_{\max}\) として、
各関節の増分上限は \(L_t=\min(L_{\max},\max(0.1L_{\max},0.5D_t))\) とする。
距離ゲイン0.5 [rad/m]で目標付近の指令を短くし、新しい画像を得てから接近を続ける。
これらは許容集合の条件であり、その中で上記の全JをSLSQPで最小化する。
条件を満たす指令が得られない場合は、指令を選定しない。
予測画像は候補の代表姿勢による制約で、Gaussianの全支持の可視性・非干渉を保証するものではない。

IKは専用スレッドで進め、規準を改善して実行条件を満たした途中解も指令先へ反映する。
サーボは速度をリセットせず追従する。遅れて届いた増分は必ず元の撮像指令へ加え、最新指令へ加え直さない。
移動中に新しい画像を撮って目標を再構成する。
観測の有効期間や未検出回数・実行時間の上限はdefault.yamlで指定する。

物理的な成功は、実先端が到達許容距離以内で、そこまでの実行経路に禁止された食い込みがない場合だけ。
実形状の隙間は0以上を要求し、一度でも食い込んだ試行を後で成功に変更しない。
候補側の規準Kや平均位置だけでランプを点灯しない。
検査する干渉対は、各URDF衝突形状とパネル・ボタン・ランプの組である。自己干渉と床はこの設定の検査対に含めない。
検査は保守的なURDF直方体と適応的な途中点によるもので、一般的な連続衝突保証ではない。

## 同じ分布を表示する

不透明な物理モデルと半透明の剛体モデルは場面形状の表示である。
ProbTFの手先・目標は、IKが評価した同じ不変のPoseSummary・GoalSummaryからnativeメッセージへ変換する。
中間リンクには同じnative局所候補の列をそのまま送り、既存coreの点モーメント照会で経路全体へ軸端点を作用させる。
中間リンクの位置と方位を表示専用の積分布へ置き換えない。
AR表示は、同じ比較目標pGへマーカー・目標関係の逆を右合成したマーカー座標照会である。
これは近似前のnativeマーカー経路の全分布そのものの表示ではない。
元のカメラ相対の画像観測はmarker_belief、比較目標はgoal_beliefで公開する。
別の表示用誤差モデルを作らず、真の関節姿勢へ分布を再中心化しない。
ProbTFの軸は、選定候補の指令qに条件づけた分布の代表である。
半透明モデルは現在の指令c、不透明モデルは物理平衡角を表す。
既定の局所候補は未較正の入力であり、その中心が実姿勢に一致するとは仮定しない。
描画用の標本は、この同じ分布に対する照会である。
XYZ端点ごとに、既存のLatestSnapshot.lookupPointMomentsが返した平均・共分散へ一致するGaussianから点を描く。
条件付き並進・回転の結合を保持するBinghamの2次・4次モーメント評価を使い、Lie代数Gaussianを仮定しない。
Gaussianは描画用の点要約にとどめ、変換法則へ戻さない。coreのソース・APIは変更しない。

再送しても元の観測時刻・指令・版を保持し、観測時刻を現在時刻へ書き換えない。
有効期間は既存pluginのFrame Timeoutに従う。ロボットsnapshotはsource時刻の更新停止で薄くなる。
AR Poseは既存の受信継続判定を使い、観測時刻はメッセージ中にそのまま残す。

ユーザーが調節した [RViz設定](rviz/switch_press_demo.rviz) はファイル全体を保持する。
Point Size約0.01、カメラ画像ドックの大きさ、レイアウト、視点、軸長、標本数を再設定しない。

## 記録と検証

- settings.yaml・robot.urdf：場面・制御設定と実際に生成した質量付き身体モデル。
- summary.json：終了理由、実到達・干渉、最終誤差、最小実隙間、実たわみ、ソース版とSHA-256。
- cycles.json：観測、撮像指令、手先・目標の比較分布、全Jの内訳、反復履歴、採用した絶対指令先。
- servo.json・trajectory.csv：実行した指令・実平衡角・実先端・隙間と、サーボの速度・指令先。
- camera_XX.png・camera_final_raw.png・camera_final_detection.png：検出器が扱った画像と検出結果。

テストは次のコマンドで実行する。

    cd ~/catkin_ws
    source devel/setup.bash
    cd src/ProbTF-demo
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      /usr/bin/python3 -m pytest -q tests/probtf_switch_press_demo

検証対象は全経路モーメントと交差項、Bingham整合、全正規化項を含む重なり、
撮像指令の固定、物理・制御の情報境界、および公開された分布とIK入力の同一性である。
過去の実装・到達動画の履歴はGit履歴で参照できる。

## 2026年10月16日の検証結果

[検証記録](outputs/indet_ik_20261016/validation.json)と
[native配信例](outputs/indet_ik_20261016/native_snapshot.json)を保存した。
Python44件、C++31件、デモと依存7パッケージのビルドが通過した。

| 実行 | 実到達誤差 | 最小実隙間 | 到達時刻 |
| --- | ---: | ---: | ---: |
| headless | 5.60 mm | 14.91 mm | 36.16秒 |
| RViz・実ROS配信 | 5.68 mm | 14.26 mm | 35.61秒 |

実配信802レコードを捕捉し、選定41回の410レコードについて、分布種別・A・平均・共分散・結合C・重みとIK記録が完全に一致した。
保存した実ROSバイト列の四元数も一致した。
同じバイト列20件を実RVizの点照会関数へ渡し、106フレーム・318軸の平均と共分散を独立Python計算と照合した。
最大差は平均6.67e-16 m、共分散8.53e-16 m²だった。
coreのソース差分はなく、RViz設定は作業前のユーザー設定と全バイト一致する。

独立にBinghamを再フィットした際の小さなA差は、公開値とIK入力の同一性から区別して記録した。
今回の結果はこの設定・1場面の準静的実行であり、一般の到達・連続衝突・較正済み成功確率を保証しない。
