# 33・65格子の完了後の図生成とpush

2026年10月29日のユーザー指示により、融合図は中止し、failure 1・2を独立に表示した図とデータを公開する。両計算が完了したら、通常の `git commit`・`git push origin main` を現在の作業ブランチで実行する監視処理を起動した。

## 稼働中の監視

```bash
tmux attach -t pid-publish-33-65
```

監視は60秒ごとに確認する。数値計算の `pid-failure1-33-65`・`pid-failure2-33-65` には変更を加えていない。両ケースの全33・65格子の完了と、writer lockの解放を確認してから図を生成する。

状態とログは次に残る。

```text
outputs/failure_only_33_65_20261028/publication_watch/
  status.json
  run.log
  configuration.json
  launch.json
  frozen_code/
  run.sh
```

`status.json` の `waiting_for_calculations` は計算待ち、`pushed` はpush成功を表す。成功時にはpublication commitも記録する。再接続は上記tmuxコマンド、停止した監視の再実行は保存済みの `run.sh` を使う。監視と描画コード・旧図の設定も凍結コピーを使っている。

## 作成する図

旧版と同じ2ケース×4群の配置、淡色RdYlGnの共通0〜100%スケール、視点、透明度、フォントを使う。33・65それぞれに3D俯瞰図とP–D断面図を作り、各PDF・SVG・PNGで保存する。成功条件付きの率・成功100%の条件は再導入しない。参照点は実際の全12失敗ゲインであり、他9ゲインもその失敗記録の値である。

公開先は次の新しいディレクトリとする。

```text
publication/pid_gain_voxels_failure_only_33_65_20261029/
  figures/grid_33/
  figures/grid_65/
  data/failure1/
  data/failure2/
  local_data_catalog.json
  publication.json
  README.md
```

## 保存・pushするデータ

実軸座標、全N分母の安定数・有効数・未評価数、標本座標、入力・固定ゲイン・出自を保存する。各格子の `sample_bits.npz` は全128標本のstable/validをlittle-endianのbit列へ可逆圧縮し、評価済みmaskも持つ。これだけで個々の標本と安定数を再検算でき、公開データだけから図も再描画できる。

数GBの生スペクトル半径配列・checkpointはローカルに保持し、Gitには追加しない。その場所・容量・SHA-256を `local_data_catalog.json` に記録する。選定した最終成果物だけを列挙してcommitし、無関係なdirty/staged変更を巻き込まない。履歴書換え・force push・別worktreeは使わない。ネットワークや通常pushの一時失敗では同じcommitの送信を再試行する。

## 検証

- 新しい描画・監視の20回帰テストを含め、first-order-lagとautomationの110テストが通過。
- 実33格子の両ケース8群から、旧形式の3D・PD図をPDF/SVG/PNGへ生成し、全Nの独立再集計と表示を確認。
- 実33格子の全標本bit archiveは合計217,413 bytes、図6ファイルは合計5,219,852 bytes。
- 監視は実入力のfingerprintを固定し、起動時に `waiting_for_calculations` を確認した。この記録は図の最終push完了を意味しない。

実装とテンプレート設定は `automation/watch_failure_publication.py`・`render_failure_publication.py`・`failure_publication_20261029.json`。本番の凍結設定は上記controlディレクトリの `configuration.json` に保存した。
