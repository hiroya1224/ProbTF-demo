# 四成分混合Bingham分布と単一成分による箱への接近

`run_box_grasp_demo.py compare` で取得した二条件の記録である。ROS masterは既存の実行から分離し，物体の方位分布だけを切り替えた。寸法100 × 80 × 180 mm，位置 (0.78, −0.12, 0.46) m，ヨー67.5度の箱は両条件で同じ形状・配置を保つ。

`mixture/` はD2対称性の四成分を各重み1/4で持つ一つの目標，`single_pose/` は同じ基準モードに集中する一成分の目標である。集中度は全成分5000。候補数は両条件とも一つである。方位費用は正規化した混合密度の負の対数を log-sum-exp で計算する。位置・移動量・関節制約・再始動・最適化・制御器・接近経路は共通である。

記録では両条件が同じモード付近へ到達した。混合条件の位置誤差は0.000909 mm，最寄りモードとの角度は0.052425度，単一条件は0.000897 mmと0.052339度だった。指の開口は106 mmである。全観測姿勢での箱との隙間下限の最小値はそれぞれ5.657 mmと5.667 mmで，どちらも要求した0.5 mmを上回る。この比較から混合条件の運動量低減を主張しない。

条件ごとの `manifest.json`，`parameters.yaml`，URDF，RViz設定，`grasp_library.yaml` に設定を保存した。`experiment.bag` には完全なnative物体・把持目標分布，関節状態，指令，IK結果と段階イベントが含まれる。`samples.csv` と `summary.json` は記録を独立に解析した評価値である。`events.jsonl` は実行中，`events.json` は記録から抽出した段階イベントである。

動画は一つのX11画面を分岐して保存した斜め・真上・横並びの3種類で，条件内の全視点は同じフレーム数と20 fpsを持つ。`grasp_oblique.png`，`grasp_top.png`，`grasp_side_by_side.png` は保持区間の画面である。映像内に説明注記は追加していない。`grasp_preview.png` は横並び画面の同一コピーである。

再解析はcatkin環境のsystem Pythonで行う。

```sh
rosrun symaware_grasp run_box_grasp_demo.py analyze /absolute/path/to/mixture --output /tmp/box-analysis
```

新しい比較を記録する場合は既存出力を上書きせず，別の保存先を指定する。

```sh
rosrun symaware_grasp run_box_grasp_demo.py compare --output /absolute/path/to/new-comparison --port 11981
```

今回評価したのは運動学的な接近と指閉鎖である。接触力，摩擦，保持力，持上げは模擬していない。隙間の確認は観測された姿勢と計画経路の離散標本に対するものであり，連続時間の証明ではない。
