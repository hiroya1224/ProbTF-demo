# 伝播式による観測更新と180秒検証

今回Codexが追加した単回Gaussian条件付けは、凍結代理内では数式と一致する。しかし、更新後のリンク候補を有限再合成した観測整合は未達で、180秒でも両ケースは到達していない。数式検算・配信同一性の確認を、推定の正しさや跳ね解消と扱わない。

仕様とユーザーへの説明は [PROPAGATED_MOMENT_UPDATE.md](../../PROPAGATED_MOMENT_UPDATE.md)、今回明らかになった不一致は [POSTERIOR_CLOSURE.md](POSTERIOR_CLOSURE.md) に記録した。

- [180秒の実走結果・時刻/指令/ledger・表示同一性・計算遅延](live_180s/runtime/README.md)
- [同じ時刻/指令でのリンク推定補正・再合成後の不一致](live_180s/link_update_audit/REPORT.md)
- [実手先距離の全保存tickプロット](live_180s/physical_distance.svg)。推定や制御へ入れていない物理側の事後診断である。
- [1,138件の回帰結果とsource hash](regression_manifest.json)、[接続レビュー](integration_review.json)
- [旧prior4例の固定入力検算](independent_operator_review.md)。新方式を初期から動かした実走とは区別する。

実走sourceは親commit `7beee54a0a45bfb001ab34d9b1a0bdeea4b798f2` に本変更を加えた未commit状態だった。実行中はsource/config/nativeを凍結し、保存SHAで識別した。90秒試行の動作を180秒へ継ぎ足したものではなく、同sourceの初期状態から別に開始した試行である。時間延長以外の動作設定を変えていない。最終commit番号はこの資料を収録するcommitの履歴で確認する。

`live_180s/runtime/files.json` は選定runtime資料のchecksumである。READMEの「到達検知」を「180秒動作終了の検知」へ訂正し、CSVは値を変えずCRLFからLFへ正規化した。元/tmpのchecksumとはその分が異なる。物理距離CSVは0.5秒targetに最も近い実保存tickと最初/最後/最小だけを選び、観測や状態を補間していない。図は全保存tickを使う。

full raw/DB/spool/pickle/巨大JSONL/動画・無関係sourceはGitへ追加しない。元入力は `/tmp/probtf_propagated_moments_180s_live_20261018` に、リンク後解析は `/tmp/probtf_moment_180s_jump_audit_20261018` に保持する。物理終了後の推定drainは別会計で、IMU側の未完K340件もarchiveに保持した。
