# 真値・候補幅の介入診断：選別した記録

解釈と全条件の比較は [診断報告](../../UNCERTAINTY_ABLATION_20261018.md) を参照する。productionの数値処理は `1244f3f`、試行開始のmainは `efeb1d2`。通常経路・core・RVizを変更した実験ではない。真値入力、有限な候補幅の縮小、依存の変更は診断条件として明記している。

- `frozen/` は、以前の実走から復元した12の不変候補に対する対比較。実際の次の運動や到達率ではない。元の数値資料とスクリプトをそのまま保存した。
- `live/` は初期状態から動かした各条件のsetup、要約、raw回収記録、および保存がある場合の凍結診断helperの索引。全軌跡・DB・画像は含めない。最初のMarker6条件はhelperの開始SHAだけを記録し、両known_landmark条件以降はhelperの実行バイトも凍結している。
- `diagnostic_sources/` は、その凍結helperをSHA別に一度だけ保存したもの。同じファイルを条件ごとに複製せず、各caseの `diagnostic_helpers.json` で元module名・SHA・保存先を対応させた。実行時に凍結した元のmodule配置は `/tmp` に保持する。
- `selected/closed_loop_summary.json` は物理終了時点の結果と取得／処理件数、全物理frameから求めた初到達・到達前後の弧長をまとめたもの。図の線だけを間引き、数値指標には全frameを使う。
- `analysis/` は初回の離反・共通位置ずれ・推定年齢・実関節範囲逸脱の切り分けに使った小容量の数値資料。元の試行と後解析での真値使用を区別する。
- `inventory.json` は保存した各ファイルのバイト数とSHA256。大容量rawの代替や、全rawをGitへ保存したという宣言ではない。

元データは `/tmp/probtf_uncertainty_ablation_20261018/live/`、凍結比較の元実走は `/tmp/probtf_1244f3f_rviz_rerun_20261018/` に保持する。停止後のlossless gzipは各caseのmanifestで追跡でき、解凍後の元SHAを検証している。終了後に回収したrawは、終了済みの動作・推定には使われていない。

初期の後解析JSONにある「truthはscoringだけ」という記述は、その後解析処理の範囲である。oracle試行自体の制御・推定への真値入力を否定する記述として読まない。試行の介入はsetup・専用truth ledger・本報告の条件表で宣言し、現行監査は `audit_truth_used_for_posthoc_scoring_only` と `trial_truth_assisted` を分けている。元の数値資料はそのまま保存し、この区別を補足した。

各条件は一回であり、非同期の観測選択・計算時間・採用時刻は一致しない。同じカメラseedでも、実際のcapture順にPCG64を進めるため、同じ物理時刻で同じ画素noiseを与えた対比較ではない。IMUは元のsource-time依存seedを使う。初期のsetupにある一般的なseed説明は実装より粗く、後半setupではこの区別を記載した。元setupのバイトは書き換えていない。診断wrapperは条件・保存機能をbatch間で追加しており、全条件で同じバイトだったとは主張しない。数値処理・規準・物理・core・RVizが不変であることとは区別する。

再実行例はリポジトリ根元で次のとおり。`--intervention` は報告内の条件名を指定する。推定・指令のモンテカルロは使わない。

```bash
source /home/leus/catkin_ws/devel/setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 \
  ros/examples/probtf_observation_fusion_demo/validation/tools/uncertainty_ablation.py \
  --case marker --intervention baseline --duration 180 --drain 15 \
  --output /tmp/probtf_uncertainty_ablation_repeat/marker_baseline
```

出力先は新しいディレクトリとし、元試行へ追記しない。原取得／実ledgerを新しい時刻へ付け替えない。ここに保存したhelperは実行時の凍結バイトであり、現在のharnessに追加した記録項目等と異なる場合がある。setupのhelper SHAと対応させる。凍結比較のスクリプトは元の `/tmp` データと既存の目的関数分解helperにも依存するため、この小容量の選別資料だけでraw全体を復元できるとは主張しない。
