# 実カメラのARマーカー録画

実際に手で動かしたAprilTagをカメラで撮影し、画像と推定姿勢をRVizで表示した既存の画面録画です。シミュレーターの映像ではありません。

| 動画 | 原録画の長さ | 内容 |
|---|---:|---|
| [real_camera_203426.mp4](real_camera_203426.mp4) | 約55秒 | 実画像とマーカー姿勢の表示 |
| [real_camera_214422.mp4](real_camera_214422.mp4) | 約47秒 | ID 12・78を含む実画像と姿勢の表示 |

元ファイルはそれぞれ次の場所に残しています。

- `/home/leus/Videos/prob_armarker-2026-07-23_20.34.26.mkv`
- `/home/leus/Videos/prob_armarker-2026-07-23_21.44.22.mkv`

日付は元ファイル名に基づきます。動画から実行コミットや対応するrosbagは確定できません。録画中には `calibration=fallback_hfov_60.000deg` の表示があり、後述の較正済みrosbagと同一の収録として扱わないでください。

2026年10月23日に、未登録だった既存録画をMP4へ変換してGitに追加しました。全編・元の解像度を保持し、映像をH.264、音声をAACへ再圧縮しています。切り抜き、描画の作り直し、推定の再実行は行っていません。元ファイルとMP4のSHA-256、容量、フレーム数、時刻の検査結果は [provenance.json](provenance.json) に記録しています。

変換コマンドは次のとおりです。`INPUT` と `OUTPUT` は上記の各録画・MP4のパスへ置き換えます。

```bash
ffmpeg -hide_banner -loglevel error -i INPUT \
  -map 0:v:0 -map '0:a:0?' \
  -c:v libx264 -crf 22 -preset medium -threads 2 -vsync 0 \
  -c:a aac -b:a 128k -movflags +faststart OUTPUT
```

## 別に保存されている実カメラのrosbag

約30秒の実カメラ記録が次の2か所にあります。

```text
/home/leus/catkin_ws/src/ProbTF-demo/recordings/ar_marker/20261013_083122_631446/all_topics_30s.bag
/home/leus/catkin_ws/src/ProbTF-demo/recordings/ar_marker/20261013_083323_181005/all_topics_30s.bag
```

容量はそれぞれ1,758,270,898 bytes、1,629,152,330 bytesです。画像、CameraInfo、検出器のdebug画像、ProbTF候補、TFが保存されています。同じディレクトリに較正・検出設定、`bag_info.yaml`、`validation.json` があります。

これらの原本は既存の `.gitignore` に従ってローカルに保持しています。今回pushするのは上のMP4と小さな出自情報です。この2本のrosbagから作られたと確認できる既存動画は見つかっていません。
