# failure 1・2の33→65格子計算

10月29日に、両計算完了後の図生成と通常commit/pushの [監視処理](PUBLICATION_WATCH_20261029.md) を追加起動した。融合図はユーザーの訂正で中止し、両ケースを独立に公開する。

2026年10月28日に、監査修正後の方式で両ケースを独立したtmuxセッションに起動した。元の128標本、seed 0、`conservative_fusion` を保持し、成功入力は指定していない。4群とも33格子を計算した後、33の評価点を再利用して65格子へ進む。起動記録は [JSON](HIGH_RESOLUTION_RUNS_20261028.json) に保存した。

| ケース | tmuxセッション | worker |
|---|---|---:|
| failure1 | `pid-failure1-33-65` | 16 |
| failure2 | `pid-failure2-33-65` | 16 |

```bash
tmux attach -t pid-failure1-33-65
tmux attach -t pid-failure2-33-65
```

detachは `Ctrl-b`、続けて `d`。各セッションは終了後もpaneを保持する。計算の完了・失敗は各出力の `status.json`、終了コードは `exit_code`、経過は `run.log` で確認する。

## 保存先

```text
ros/examples/grape-param-estim/minimal/first-order-lag/outputs/
  failure_only_33_65_20261028/
    launch.json
    frozen_source/
    failure1/
    failure2/
```

各ケースに `run.sh` を保存しており、停止後は同じスクリプトでcheckpointから再開できる。稼働中の同じ出力へ二重起動すると既存のlockが拒否する。終了セッションに再接続するだけでは再実行されない。

計算コードはコミット `1c8246047508ebde2fd07fb60a3955a2baaf0407` の内容を凍結した。機体モデルも同じ内容をコピーしている。`inputs/original_estimate.json` は元ファイルと全バイト同一で、実行用の `runtime_estimate.json` は機体モデルの参照先だけを凍結コピーへ変更した。推定値・共分散・ゲイン等の数値は変更していない。

## 後で融合・粗格子化するために残すもの

各ケースの `retention.json` と `manifest.json` に、ケースID、入力・コードhash、全12ゲイン、共分散選択、標本数・順序を記録する。`plant_samples.npz` の全128×13座標は、直前の修正後の低解像度計算とbitwise一致することを起動前に確認した。ケース間の同じ標本番号を同一の身体候補とは扱わず、ケースIDと標本集合hashを組にして識別する。

各群の `grid_33/` と `grid_65/` は別々に残す。65へ進んでも33のデータを上書き・削除しない。

- `metadata.json`：実際のP/I/D軸座標、軸順、shape、他9ゲインを含む全12ゲインの固定条件。
- `spectral_radius.npy`・完了後の `volume.npz`：全格子点×全128標本のスペクトル半径。
- `evaluated.npy`・`volume.npz` 内の `evaluated`：評価済み／未計算を区別するmask。有効性は有限の半径と併せて復元できる。
- `stability_fraction.npz/json`：安定数、分母128、有効数、未評価数、安定割合。
- 入力estimate、凍結した機体モデル・コード、設定・進捗・ログ。

計算開始前の全軸座標も `planned_grid_coordinates.npz` に保存した。現在の3D版は境界だけを局所細分化するadaptive法ではなく、33・65の軸へ当該failureの記録ゲインを挿入した非等間隔の直交格子である。今回の実配列はそれぞれ34³点、66³点となる。failureごとの探索範囲を独立に保持し、共通格子には変更していない。

保存するQは格子点上の安定標本割合であり、ゲインの確率密度やセル質量ではない。今回は積分測度・補間規約・融合を追加していない。将来粗い格子へ移す際に、保存座標と全標本値から必要な重みや集約方法を選べる。Roll/Pitch以外の断面では、failure 1と2で固定されるRoll/PitchのP値も異なるため、融合時には座標だけでなく全12ゲインの条件も照合する。

本記録は起動確認であり、33・65の完了報告ではない。大容量の格子データ・checkpointはローカル出力に保持し、今回のGit変更には含めていない。
