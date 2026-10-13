# 箱把持試行のbag解析

`box_trial_analysis.py` は、記録した `experiment.bag` と `manifest.json` を読み、運動学的な到達と実際の箱との隙間を計算する。IKの費用や代表成分の表示姿勢から、把持の成功を判断しない。衝突評価にはmanifestの `physical_box` に保存した箱の寸法、位置、回転を用い、`BoxCollisionModel` と `evaluate_box_grasp` を呼び出す。URDFのハッシュがmanifestにあれば照合する。

ObjectBeliefと `/symaware_grasp/grasp_targets` は、すべてのメッセージを既存のnative変換関数で復号する。候補数が1であり、ObjectBeliefと候補の成分数がmixture条件で4、single_pose条件で1であることを確かめる。各成分は正の正規化重みと有限Bingham方位を持つ必要がある。候補のモード列については、四元数の符号を同一視したうえで、manifestの `target_quaternions` と一対一に一致することを確かめる。確認したメッセージ数、成分識別子、正規化重み、モード列を `summary.json` の `native_belief_checks` に保存する。

CSVの観測範囲は `start_stamp <= stamp <= end_stamp` とし、`finished` 段階の関節状態を除く。位置と隙間は、対象のbeliefが更新されない間も、固定した実際の箱に対して計算する。記録した観測姿勢を飛ばさず確認するため、観測範囲内で必要な関節が欠ける場合や左右の指の位置が異なる場合は解析を中止する。

到達判定には `hold_grasp` の最後の1秒の観測を用い、TCP位置誤差2mm以下、最も近い許容モードとの全3次元方位誤差2度以下、閉開口の誤差0.2mm以下を求める。holdの観測期間が1秒以上あること、manifestの完了状態、成功したIK結果も確認する。箱との隙間の下限は、動作開始から終了までのすべての観測で0.5mm以上必要であり、manifestがより厳しい値を指定すればその値を用いる。`summary.final` はこのhold区間の最終行であり、生成したCSVの最終行と一致する。

位置、方位、関節角、開口、手首曲げ、隙間下限、最も近い衝突部位などを `samples.csv` に保存する。`summary.json` にはIK結果、完了と到達の判定、最小隙間、native成分の検査結果を保存し、段階イベントを `events.json` に保存する。解析は接触、摩擦、把持力を評価せず、記録のない時刻の連続的な無衝突も保証しない。動作計画側の経路検証は別に保持する。

実際のROSbagを生成して読む回帰テスト8件が通過した。既存の腕モデルからアニメーション用URDFを作り、native ObjectBeliefとGraspTargetArrayをbagへ記録したうえで、両条件の成分数、最後のhold区間の選択、終了後の観測の除外、成分を潰した記録や候補数の変更の拒否、位置に到達しても方位誤差がある場合の不達、隙間不足、URDFハッシュの不一致を確認した。新規テストは `tests/symaware_grasp/test_box_trial_analysis.py` に保存する。
