# 合成回転の日本語解説動画

[完成動画・字幕・章の時刻](publication/INDEX.md)に、動画の長さと実ファイル容量を記録する。出力は960×540 px、15 fps、日本語音声は通常速度の2倍。元の1920×1080案から縦横をそれぞれ半分にした。全6章・48場面で、対象の設定から非線形積分、モーメント伝播、観測前後の分布、比較結果まで説明する。

対象は、角速度1/75の指定を撤回した後の**元の3周波数軌道、0.5 Hz取得、仮定標準偏差5 deg/s**。[元の図と保存値](../publication/three_frequency_0p5hz_sigma5/INDEX.md)を読み、推定を変更せずに描画した。以前の2周波数・18条件の比較結果を、この動画の数値へ混ぜていない。

## 内容と計算上の意味

| 章 | 内容 |
|---|---|
| 1 | 既知入力の比較としての範囲、座標系、3周波数軌道、解析的body角速度・角加速度 |
| 2 | 測定生成と疎な取得、生成ノイズと仮定分布の区別、6点の決定論的求積 |
| 3 | 非線形Euler式、線形側との差、角速度と四元数の同時RK4積分 |
| 4 | 保存した6候補、四元数二次モーメント、回転行列の平均・全9×9共分散 |
| 5 | 観測直前・直後、角速度分布の置換、RPY表示と±2σの意味 |
| 6 | 同じ時刻列での結果、無ノイズ対照、有限近似と再現資料 |

bodyから世界への回転を `R=Rz(yaw) Ry(pitch) Rx(roll)` とする。表示の角速度はdeg/s、積分と共分散の内部計算はrad/s。RPYの時間微分をbody角速度と同一視しない。積分の四元数はHamilton積・`wxyz`順で、保存された参照の`xyzw`順とは明示的に変換する。

三主慣性は既知の `J=diag(0.045,0.070,0.100) kg m²`。生成側で処方軌道から `τ=J ωdot+ω×(Jω)` を作り、1 kHzの既知入力として両方式へ渡す。したがって、これは既知入力下でジャイロ項の影響を分離する実験である。未知のトルクをgyroだけから復元した結果、実機パラメータを同定した結果、並進を含む6自由度の飛行シミュレーションとは扱わない。

測定は200 Hz、生成白色ノイズの標準偏差0.2 deg/s、固定biasは `(0.03,-0.02,0.04) deg/s`、seedは20261113。乱数を使うのはこの測定生成だけで、推定はモンテカルロではない。原index `0,400,800`、時刻 `0,2,4 s` の測定だけを取得する。仮定する入力Gaussianの標準偏差5 deg/sは生成ノイズとは別であり、biasは推定器へ教えず差し引かない。初期姿勢だけを一度既知とし、途中の姿勢真値・未取得gyroは予測へ渡さない。

各取得の平均と共分散から、Cholesky列方向の `±√3`、重み各`1/6`の6点を作る。同じ6個の初期角速度から、角速度と回転増分を同時に積分する。区間途中で独立化・再求積せず、既知入力を各RK4段で補間し、最大刻み0.002秒を使う。線形側で落とすのは `ω×(Jω)` だけで、四元数の運動学は両方式に共通である。

取得区間の境界では、角速度を次の測定を中心とした**新しい独立Gaussianへ置換する**。事前と観測尤度を掛けるBayes条件付けではなく、全軸の幅が必ず狭まるとも限らない。姿勢の候補はその時点で捨てず、次の区間の角速度との独立性を仮定して伝播する。

既存coreの四元数積二次モーメントと、回転行列の平均・全9×9共分散を使用する。区間内の角速度と回転増分は同じ候補から得る。各区間の独立な増分因子をモーメントで合成し、全枝の列挙を避ける。途中で単一Binghamへ状態を置換しない。coreは変更していない。既存coreには、単位四元数二次モーメントの対称化、丸め由来の微小負固有値の処理、trace正規化があり、動画中でもこの数値処理と新たな分散floorを区別する。

RPYの中心は `f(E[R])`、共分散帯は同じ全回転共分散からのJacobianによるdelta-method表示近似である。`E[f(R)]`や保証付き95%領域とは主張しない。表示近似を推定状態へ戻さず、幅の平滑化・切り詰め・見栄えのための分散変更も行わない。

主結果は同じ501時刻で、姿勢中心のSO(3)角度誤差RMSEが非線形約1.143565度、ジャイロ項を省いた側約58.524482度。ノイズ・biasを零にした別検査は別場面として明示し、主結果の測定列と置き換えない。詳しい値と解釈は[全文台本](narration.md)にある。

## アニメーションの元データ

曲線、取得点、6候補の角速度と回転増分、モーメント、共分散、観測前後は、元の`input.npz`、`acquisitions.npz`、`nonlinear.npz`、`linear_forced.npz`および同じlawの描画用NPZから読む。時間カーソルで保存時刻を順に開示し、候補lawを再推定しない。3D座標軸は保存された回転を一定の射影で描く。台本の説明順に合わせるアニメーションの経過と、データ上の時刻は区別して表示する。

`prepare_assets.py`は同じ保存係数から3周波数の成分を分け、同じ6初期候補で最初のRK4一刻み `0～0.002 s` を再計算して実際の4段評価を図示する。この説明用トレースは元の予測を上書きしない。元の最初の保存点まで積分して一致を検査する。全共分散のwithin項・between項も、保存された親候補と増分から分解し、両者の和を元の全共分散へ照合する。これらの差分と入力hashは`derived_data.json`に残す。

場面ごとに使用ファイル・配列・hashを`scene_sources/`へ、章・場面の実時間を`chapter_timing.json`と`scene_timing.json`へ保存する。動画作成の前後で元NPZのhashが変わらないことを`verification.json`に記録する。

## 再現手順

以下はリポジトリのルートから対象フォルダへ移動して実行する。Python側はNumPy、SciPy、Matplotlib、Pillowと、このリポジトリで使用している既存ProbTF coreが必要。描画にはFFmpeg／FFprobeとNoto Sans CJKフォントを使う。音声生成だけは`edge_tts`を使用し、初回生成時にネットワークへ接続する。今回の音声ライブラリは`edge_tts==7.3.1`、音声名は`ja-JP-NanamiNeural`。

```bash
cd ros/examples/grape-param-estim/minimal/nonlinear-gyro-dead-reckoning/synthetic_controlled_rotation
```

既存の`/tmp/evian-narration-venv/bin/python`を利用できる環境では、次で全体を構築する。最終公開物を`explainer/publication`へ、音声・一時クリップ・キャッシュを`/tmp`へ分ける。

```bash
/usr/bin/python3 explainer/build_video.py \
  --work-dir /tmp/synthetic-explainer-20261113 \
  --tts-python /tmp/evian-narration-venv/bin/python \
  --output explainer/publication \
  --workers 4
```

その環境がない場合、音声専用の仮想環境を用意し、そのPythonを`--tts-python`へ指定する。

```bash
/usr/bin/python3 -m venv /tmp/synthetic-explainer-tts
/tmp/synthetic-explainer-tts/bin/pip install edge_tts==7.3.1
/usr/bin/python3 explainer/build_video.py \
  --work-dir /tmp/synthetic-explainer-rerun \
  --tts-python /tmp/synthetic-explainer-tts/bin/python \
  --output /tmp/synthetic-explainer-rerun/final \
  --workers 4
```

音声を先に生成する場合は、次を実行し、できたmanifestを`build_video.py --narration-manifest /tmp/synthetic-explainer-rerun/narration_manifest.json`へ渡せる。キャッシュされた音声が全場面分そろっていれば、再生成せず描画できる。外部の音声サービスは将来同じ出力を返す保証がないため、公開物には実際の音声hashとフレーム数を記録する。

```bash
/tmp/synthetic-explainer-tts/bin/python explainer/synthesize_voice.py \
  --story explainer/story.json \
  --work-dir /tmp/synthetic-explainer-rerun
```

台本の正本は`make_story.py`で、`/usr/bin/python3 explainer/make_story.py`により`story.json`と`narration.md`を生成する。今回の公開動画はその生成済み台本を使用する。変更した台本で作り直す場合は別の実験出力として扱う。

## 音声・字幕・容量と検証

通常速度`+0%`で日本語音声を作り、FFmpegの`atempo=2.0`で音程を保って2倍速にする。語尾を保護する無音を付けたPCMのフレーム数が正確に半分になるよう末尾だけを調整し、切り落とした末尾の実振幅も記録する。映像には音声終了後0.5秒以上の独立した表示時間を加えるため、動画全長を単純に通常音声の半分とはしない。実際の通常／2倍速フレーム数は`narration_manifest.json`にある。

字幕は音声サービスの文境界時刻を2で割り、実際の場面長に合わせて並べる。MP4には日本語の選択式字幕と章情報を含め、別途SRTも出力する。台本全文はMarkdownでも読める。

まず全編を一つのMP4へ連結し、実際のエンコード容量が50,000,000 bytes以上になった場合だけ、場面の境界で自動分割する。単独場面まで上限を超えた場合には、その場面を容量に応じたbitrateで再エンコードする。場面の削除で容量を減らさない。最終ファイル数・実容量・長さ・hashは[公開一覧](publication/INDEX.md)と`verification.json`に記録する。

エンコード後は次で寸法、音声、容量、48場面の実フレームを確認する。抽出画像とコンタクトシートは`/tmp`に置く。

```bash
/usr/bin/python3 explainer/verify_video.py explainer/publication \
  --work-dir /tmp/synthetic-explainer-encoded-review
```

`encoded_checks.json`には寸法960×540、全MP4が50 MB未満、音声の実速度比、非零音声、抽出場面数、エンコード後音声の振幅を保存する。台本の数理と画面を目視確認する作業を併用し、自動音声認識による内容検証を済ませたとは記載しない。中間のPNG、非圧縮音声、通常速度音声、連結前クリップはGitへ追加しない。
