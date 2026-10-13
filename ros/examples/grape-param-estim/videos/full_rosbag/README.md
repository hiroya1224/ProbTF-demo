# rosbag全区間の実時間3D動画

外部カメラ映像との同期用に、failure1・failure2・successのrosbagを開始から最後のメッセージまで動画にしました。推定に使った区間の前後もすべて含めています。

| 動画 | 長さ | フレーム数 | 容量 |
| --- | ---: | ---: | ---: |
| [failure1.mp4](failure1.mp4) | 38.499504825秒 | 4,620 | 6.0 MiB |
| [failure2.mp4](failure2.mp4) | 39.953209917秒 | 4,795 | 5.2 MiB |
| [success.mp4](success.mp4) | 208.999333161秒 | 25,080 | 33.2 MiB |

3本とも960×540、120fps、等速です。FFmpegのH.264・CRF 18で圧縮しています。音声はありません。

## 外部映像との同期

動画の0秒はrosbag内の最初のメッセージの記録時刻です。モーションキャプチャの最初の計測時刻や、離陸時刻には合わせていません。冒頭・末尾への追加映像や、途中の省略もありません。

左上にbag開始からの経過時間、下部にROS時刻をJSTで表示しています。離陸、傾きの変化、着地など、両方の映像で見える動きを動画編集ソフト上で合わせてください。位置・姿勢にはモーションキャプチャの`header.stamp`を使い、bag開始時刻との差をそのまま保っています。記録処理の遅延によって動きがずれないようにするためです。

時刻の関係は次のとおりです。

```text
ROS時刻 [ns] = *_timing.jsonのbag_start_ns + 動画の再生時刻 [ns]
```

通常のフレーム間隔は1/120秒です。最後のフレームの表示時間だけ端数に合わせ、MP4の映像トラックとコンテナの終了時刻をbagの記録長に合わせています。MP4の時間単位は1nsです。最後のフレームにはbag終了時点で利用できる最後の姿勢を表示するため、その画像だけは最大1フレーム分、約8.33ms早く表示されます。同期には通常のフレームを使ってください。

`*_frame_times.csv.gz`には各フレームの再生時刻、表示時間、対応するbag時刻、表示に使った時刻を整数nsで保存しています。`*_flight_events.csv`には記録された飛行状態の切り替わりを保存しています。飛行状態と外部映像の動きには制御上の遅れがあり得るため、最終的な同期は見える動きで確認してください。

## 3D表示

3本で同じ固定カメラ・同じ縮尺を使っています。床の格子は0.25m間隔です。world、main_body、fc、cog、thrust1〜4の座標軸を表示し、Xを赤、Yを緑、Zを青で示しています。rootとmain_bodyは、記録されたTFでは同じ位置・姿勢です。

機体の位置・姿勢は実測モーションキャプチャ、main_bodyからfcへの位置関係とローター配置は`/tf_static`、重心とジンバルの姿勢は`/tf`に基づきます。機体・ローターの形は見やすさのための略図です。プロペラの回転は描いていません。軌跡はfcの実測位置で、直近2秒を明るく表示しています。

位置は線形補間、姿勢はクォータニオンのSLERPで表示しています。100msを超える計測の空白では直前の姿勢を保持し、画面に表示します。冒頭のモーションキャプチャ未記録区間にはbag内のFCオドメトリを表示します。最初・最後の計測範囲の外では位置や姿勢を外挿しません。モデル推定や軌道平滑化は使っていません。

## 再生成

リポジトリのルートで実行します。元のbagは既存の`minimal/bag_jsons/single_rosbag_*.json`から参照します。保存場所が違う場合は抽出時に`--bag-root /path/to/bags`を指定してください。推定区間の設定は動画の切り出しには使いません。

```bash
/usr/bin/python3 ros/examples/grape-param-estim/videos/full_rosbag/render.py \
  extract --cache-dir /tmp/grape-full-rosbag

python3 -m venv /tmp/grape-video-env
/tmp/grape-video-env/bin/pip install \
  -r ros/examples/grape-param-estim/videos/full_rosbag/requirements.txt

/tmp/grape-video-env/bin/python \
  ros/examples/grape-param-estim/videos/full_rosbag/render.py \
  render --cache-dir /tmp/grape-full-rosbag
```

抽出にはROS1の`rosbag`とNumPy、描画にはPython 3.10以上、requirements.txtのライブラリ、FFmpegが必要です。VTKはEGLで画面外に描画します。実行した環境ではMesaによる描画を使っています。

`--case failure1`などで1本ずつ生成できます。`render --preview --output /tmp/grape-preview`で開始・飛行中・最高位置・終了時点の静止画を確認できます。

```bash
PYTHONPATH=ros/examples/grape-param-estim/videos/full_rosbag \
  /tmp/grape-video-env/bin/python -m unittest discover \
  -s ros/examples/grape-param-estim/videos/full_rosbag -p 'test_*.py'
```

生成時には全フレームのPTS・表示時間・総フレーム数・終了時刻を検証し、FFmpegで動画全体をデコードします。`*_timing.json`に入力bagと動画のSHA-256、時刻、描画設定、検証結果を保存しています。MP4の時間設定は[FFmpegの公式仕様](https://ffmpeg.org/ffmpeg-formats.html)に基づき、PyAVで最後のフレームの表示時間を設定しています。
