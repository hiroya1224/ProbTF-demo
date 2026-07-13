# deflecomp 開発規約（Codex 向け）

このファイルは `ros/examples/deflecomp/` 以下を変更する Codex/agent が最初に読む設計契約である。詳細な数式と監査履歴は `../../../docs/lectures/deflecomp-algorithm.md`、起動・設定方法は `README.md` を参照する。

## 最重要の不変条件: 推定と実行を分離する

`K_est` と `K_exec` は意図的に別の状態である。

- `K_est` は IMU 観測に最もよく整合する**有効剛性の推定値**である。未知 payload、定常外力、摩擦、URDF/ばねモデル誤差など、明示的にモデル化されない準静的効果を吸収してよい。したがって物性値としての真の関節剛性とは限らない。
- 文脈上の `K_eff` は、この外乱込みの有効パラメータという概念を指す。現実装で外部公開される対応信号は主に `K_est`（`/deflecomp/kp_est`、旧名互換の `/deflecomp/kp_hat`）である。
- `K_exec_target` は推定結果を実行系へ渡す目標値、`K_exec` はそこへ一次遅れ・一更新量制限付きで漸近する値である。
- inverse statics、送信 command、送信 command に対する平衡予測には **`K_exec` だけ**を使う。`K_est` の急変を直接 command に入れない。
- 逆に、actuation を穏やかにする目的で `K_est` の収束を遅くしてはならない。`K_est` は外力・payload のステップに対して必要なら大きく更新してよい。実行安全性と滑らかさは `kp_exec_tau`、`max_log_kp_exec_step`、および command 側の制限で担保する。

`max_log_kp_update_step` や `max_equilibrium_pose_jump` は推定器の数値的 trust/平衡 branch 判定であり、actuator の rate limiter ではない。有限性、bounds、尤度・posterior 非悪化、可観測部分空間、同一安定 branch といった健全性判定は残す一方、「指令が急変しそうだから」という理由で推定 step を小さくしない。妥当な payload 変化まで branch guard が拒む場合は、許容値をただ保守化するのではなく、continuation、multi-start、branch identity の評価方法を改善する。

大きな process variance を不可観測方向へ毎 batch 加えて covariance を無限に増やしてはならない。`max_log_kp_covariance_var` は uncertainty の固有値上限であり、`K_est` mean の rate limit ではない。payload step に必要な一 batch の prior variance は確保したまま、長時間の不可観測 mode だけを有限に保つ。

## 信号の意味

- `theta_ref` / `/ref/joint_states`: 実現したい物理的な平衡姿勢。
- `theta_cmd` / `/cmd/joint_states`: 柔軟関節ばねの基準角・setpoint。物理姿勢ではない。赤 marker の追従性をプラント追従性能と解釈しない。
- `/equil/joint_states`: simulator/plant の物理状態。controller の入力にしてはならない。
- `theta_eq_hat`: 実際に送る `theta_cmd` と `K_exec` から計算した読み取り専用の内部平衡予測。これを目的関数として filter 後の command を上書きしない。
- `K_true`、外力の真値、simulator 内部状態: 評価専用。controller/estimator から参照しない。

## 外力・payload の扱い

`apply_frame_load.py` のような未知荷重は現在の controller model に直接入力されない。準静的かつ観測と整合する範囲では、`K_est` がその効果を素早く吸収することを意図する。荷重変化後は timing gate が静止を確認してから推定を進め、`K_exec` が遅れて追従する。

対角剛性だけでは任意の外力や複数姿勢を同時に表現できない。この場合も `K_est` は観測姿勢に対する best-fit の有効値であり、「材料剛性を同定した」と主張しない。残差が構造的に残るなら、外力/payload 状態やより豊かなばねモデルを別状態として追加する。

## 観測可能性と禁止事項

- gravity-only IMU が直接拘束するのは各 frame の重力方向である。重力軸回り、全関節姿勢、全剛性成分の一意性は保証しない。
- 不可観測方向を真値、joint 名、既知 simulator 設定で埋めない。`project_unobservable_feedforward` は経路依存 drift を生むため通常は `false` とする。
- controller/estimator に `K_true`、`/equil/joint_states`、適用外力値、未来の reference を入力する truth leak を作らない。テストで真値を使えるのは、生成と事後評価だけである。
- matched-model/noise-free の成功だけを実機性能の根拠にしない。

## 時刻と因果性

- IMU batch は全 frame の最新共通観測時刻にそろえ、その時刻までに実際に publish 済みの command 履歴と対応させる。
- command/reference が設定された静止 window を満たした record だけを準静的推定へ使う。同じ `source_stamp` の held sample を独立 evidence として再利用しない。
- 一つの synchronized IMU batch に対する非線形再線形化は、同じ prior と同じ evidence を使う一回の MAP 最適化として完了させてから `K_exec_target` を更新する。各内部反復の間に `K_exec`/command の一次遅れや再静止を待ってはならない。また、同じ batch を複数の独立観測として数えて covariance を反復収縮させてはならない。
- 計算後に publish した command を、それ以前の観測の原因として使わない。未来値や最新値への安易な置換を禁止する。
- 外力変更直後に gate が閉じることは正常だが、静止後も更新されない場合は `/deflecomp/estimation_gate_status` の理由を調べる。

## Particle supervisor

Particle supervisor は厳密な particle filter/RBPF/MAP ではなく、決定論的な bounded maximum-likelihood proposal である。通常運用の既定は **opt-in（無効）** とし、明示した比較実験以外で有効化しない。WEKF より精度が上がると仮定せず、独立 holdout、freshness、branch、姿勢多様性を含む試験なしに採択規則を緩めない。

## 変更時の確認事項

1. 変更前に `git status` と対象設定の diff を確認する。未コミット設定はユーザーの実験条件であり、依頼なしに reset、整形、既定値への復元をしない。テスト条件は可能な限りテスト内で注入し、YAML を一時変更しない。
2. `K_est` の大きな step が `K_exec` や `theta_cmd` へ同一 cycle で直結しない回帰テストを保つ。`K_exec` の一次遅れと `max_log_kp_exec_step` は独立に検証する。
3. no-load だけでなく、未知 payload/定常 frame load の step を入れ、静止後の観測尤度・重力方向誤差、`K_est` の収束時間、`K_exec`/command の滑らかさを別々に評価する。
4. exact likelihood/posterior 非悪化、KKT/branch、joint bounds、有限値、因果的 timestamp、重複 sample 排除の回帰テストを維持する。
5. `deflecomp_core` の unit test、該当 ROS test、`catkin build deflecomp_core deflecomp_ros deflecomp_sim` を変更リスクに応じて実行する。アルゴリズムや信号意味を変えた場合は `docs/lectures/deflecomp-algorithm.md` も同期する。
