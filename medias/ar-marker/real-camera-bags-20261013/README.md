# 実画像ARマーカーのrosbag再生録画

2026年10月13日に実カメラで取得した次のrosbagを、保存済みの画像・ProbTF・TFからRVizへ再生して録画した動画です。

| 動画 | 入力run | bagの長さ | 保存debug画像 | 保存ProbTF |
|---|---|---:|---:|---:|
| [083122_rviz.mp4](083122_rviz.mp4) | `20261013_083122_631446` | 29.979777秒 | 575 | 453 |
| [083323_rviz.mp4](083323_rviz.mp4) | `20261013_083323_181005` | 29.977981秒 | 425 | 657 |

元のbagは、それぞれ次の場所にあります。

```text
/home/leus/catkin_ws/src/ProbTF-demo/recordings/ar_marker/20261013_083122_631446/all_topics_30s.bag
/home/leus/catkin_ws/src/ProbTF-demo/recordings/ar_marker/20261013_083323_181005/all_topics_30s.bag
```

## 再生するデータ

`rosbag play --clock` と `/use_sim_time=true` を使い、元bagの全topicを元のbag記録時刻の間隔で再生します。表示と受信検査に使うのは次の4 topicです。メッセージ本文・取得stampを書き換えず、検出器や推定器は再実行しません。

- `/prob_artag_detector/debug_image`：実カメラの640×480画像と、当時保存した検出表示。
- `/prob_artag_demo/probtf`：当時保存した全候補。1110メッセージすべてが2 mixture成分を持ちます。
- `/tf`：当時保存した `usb_cam → apriltag_12/78` の表示用TF。
- `/tf_static`：当時保存した `camera_link → usb_cam` の固定TF。

各bagに添付されていた `demo.rviz` をそのまま使います。両設定は同一で、粒子600点・サイズ0.003 m、軸0.1 m、画像・視点・レイアウトを保持します。録画用に起動するXサーバーとROS masterは分離します。

1本目はbag開始から約9.2秒、2本目は約2.8秒までProbTFが記録されていません。その区間は候補分布を補完せず、そのまま表示します。画像と候補が保存された際の遅延も再生に含まれます。

録画は再生を開始する前から回し、bagの末尾まで再生した後に止めています。前後の待機を含むMP4の長さはそれぞれ31.867秒、31.667秒です。どちらも2200×1400・30 fpsです。

## 出自と検査

[provenance.json](provenance.json) に入力bag・RViz設定・出力動画のSHA-256、再生・録画条件、録画開始・pause解除・再生終了の時刻を記録しています。4 topicそれぞれについて、元bagのserializedメッセージ列と再生時の受信列の件数・SHA-256がすべて一致しました。両MP4は全編のデコードを確認しています。

Gitへ追加するのは動画、出自情報、再生に必要な小さな設定・スクリプトです。合計約3.4 GBのbag原本は上記の場所に保持します。

添付の [source_settings/](source_settings/) は両bagで共通の保存設定のコピーです。元ファイルとのバイト一致を確認しています。較正・検出設定は収録当時の情報として添付しており、今回の再生時に推定をやり直すためには使っていません。

## 再録画

ROS Noetic、ビルド済みのProbTF RViz plugin、FFmpeg、Xvfbを使います。次のスクリプトは専用の画面・ROS masterを起動し、自分で起動したプロセスだけを終了します。`--scratch` は未作成のディレクトリを指定します。

```bash
source /home/leus/catkin_ws/devel/setup.bash
/usr/bin/python3 ros/examples/prob_artag_detector/tools/record_saved_bag.py \
  --run-folder recordings/ar_marker/20261013_083122_631446 \
  --output /tmp/083122_rviz.mp4 \
  --scratch /tmp/ar-bag-replay-new \
  --display :99 --master-port 11993 --xvfb /path/to/Xvfb
```

2本目は `--run-folder` を `recordings/ar_marker/20261013_083323_181005` に変更し、出力とscratchも別にします。録画時の監査情報と設定コピーはscratchへ保存されます。
