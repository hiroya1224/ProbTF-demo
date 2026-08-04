# Sparse batch rewrite 前の GUI visual baseline

このディレクトリは、sparse batch / Laplace-EM / MCMC backend へ接続し直す直前の production GUI 配置を保存した比較基準である。
画像は 2026-08-04 に `gui/tests/visual_acceptance.py` で 1800 x 1120 の実ウィンドウを `QScreen.grabWindow()` により取得し、PyVista の描画面は plotter 自身の screenshot API でも取得した。
取得前には各対象ウィンドウを明示的に raise / activate しており、別アプリケーションの背後に隠れた画面は採用していない。

この baseline が固定するのは、`Master`、`Bag browser`、`Next experiment` の三画面構成、主要 panel の配置、3D view の配置、および既存の操作密度である。
画像内の `member`、`Residual wrench`、旧 stage、旧 artifact の意味は改装対象であり、維持すべき仕様ではない。
取得に使った旧 artifact は新 reader の互換 fixture ではなく、新しい `grape-param-estim/batch-estimation-run/v2` loader はそれらを拒否しなければならない。

各画像の寸法と byte size、取得対象 ID は [summary.json](summary.json) に記録している。
