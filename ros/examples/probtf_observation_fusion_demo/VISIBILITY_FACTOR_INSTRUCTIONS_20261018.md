# ProbTF：Running Gradient Trackingへの視野維持因子の追加 — 数理整理・Codex実装指示書

作成日：2026年10月18日  
対象：`hiroya1224/ProbTF-demo` の移動・観測融合デモ  
位置付け：**既に依頼済みの非同期推定・剛性候補更新・Running Gradient Trackingへの差分追加**。既存の変更を上書きする全面改修ではない。

## 0. 目的・今回の合意

実機（現在はシミュレーション）上のマーカーボードは空間に固定されている。しかし、ロボットの関節のたわみと、身体候補の更新により、**カメラから見たマーカーの相対姿勢は時間とともに変わる**。従来はマーカーが画面から消えやすく、視覚観測が途絶えることがあった。

以前の画像中央維持から作る初期値候補や、別の位置誤差制約 `s` を復活させることは望まない。すでに採用した、**時変の目標規準を逐次勾配で追うシンプルな制御**を保ちたい。そこで、既存の目標達成規準を一切書き換えず、別に定義した**視野維持因子**を制御値選定の直前で乗算する。

設計原則：

1. 既存のタスク評価因子 `task` と、新規の視野維持因子 `view` を**独立の関数・出自・診断値**として実装し、選定時にのみ合成する。新しい観測尤度として身体や剛性の候補測度へ融合してはいけない。
2. 画像が一時的に見えなくても、**最後に観測したマーカーの根元座標系での候補を保持**し、実行指令履歴と、後から到着するIMU等による身体候補の更新を反映して、現在のカメラ相対マーカーを予測する。
3. 画像が見えなくなっただけで、保持中の有効な目標を捨てたり、追跡を直ちに停止して旧探索制御に飛んだりしない。**初めから一度も有効なマーカーを観測していない場合**は別であり、既存の安全な探索・待機ポリシーを利用できる。
4. 制御入力は引き続き Running Gradient Tracking。目標端点IK・旧画像初期値・旧 `s` 制約・SG時間補間・`K_exec` を戻さない。
5. 実行した結果、**マーカーを視野に保持／見失っても復帰し、スイッチ操作が実現するか**を主要な規準とする。物理真値を制御側へ混入させない。計算速度・数値安定性を重視する。

**別作業への配慮：** 前のCodexへ渡した `ProbTF_unified_theory_running_gradient_codex_20261018.md` と `ProbTF_theoretical_framework_20261018.md` の合意事項を優先する。今回の文書はそれらを再実装する依頼ではない。現ブランチ・最新HEAD・未コミット差分を先に確認し、既に実装済みの機能を使い回すこと。旧コードへの `reset`、未コミット変更の破棄、他ワーカーと競合する重複改修はしない。

## 1. 維持する問題の前提

リンク数を $\ell$、姿勢群を $G=SE(3)$、実機への実行指令を $u_t\in\mathbb R^\ell$、候補指令を $c_t\in\mathbb R^\ell$ とする。両者を混同しない。

実機リンク姿勢の対象は $X_{{\rm real},t}\in G^\ell$、それに対応する**候補測度**は $\Upsilon_{{\rm real},t}\in\mathcal M(G^\ell)$ とする。剛性対象 $K\in\mathcal K$ には別に $\Upsilon_{K,t}\in\mathcal M(\mathcal K)$ を対応させる。$\Upsilon$ に正規化を仮定せず、「確率測度」と決めつけない。リンク間の交差を捨てて個別周辺の積へ置き換えない。

たわみモデルは $FK(c_t;K)\in G^\ell$。剛性候補の数値代表 $K_{{\rm est},t}$ を使って現在の共同候補に候補指令の変化を伝播する。**別の $K_{\rm exec}$ 平滑化段は導入しない。** 剛性の候補測度と身体候補の更新には、既存依頼に従ってBayes Factor Surprise（K側のReset事前重み $\rho_{0,K}=0.10$）を用いる。

基本のタスク規準は、目標相対姿勢における予測密度の負の対数 $J_t(c)$ である。相対姿勢分布を得るときは、身体・マーカーの共有情報、全交差、目標成分のmixtureとlog weightを維持する。今後は $J_t$ 自体を変更しない。

物理シミュレーション120 Hz、joint pub 40 Hz、IMU pub 30 Hz、画像pub 12 Hz。各データは取得時刻つきで保存し、推定は非同期に、IMU単独・画像単独のイベントで更新する。途中取得した観測を落とさず、時間合わせのためのSG補間を使用しない。これらは今回の実装の既定前提であり、視野因子のためにセンサ更新を同期化しない。

## 2. 視野維持因子の対象はスイッチではなく**マーカー**

既存のタスクでは、マーカー $M$ に対するスイッチ・手先の目標関係を使う。一方、今回「画面中央に維持したい」対象は、**カメラ $C$ から見たマーカーボード中心（あるいは定義されたマーカー中心）**である。スイッチの到達位置を画像中央へ移す話ではない。

根元座標系を $B$ とすると、相対姿勢は $T_{CM}(c)=T_{BC}(c)^{-1}T_{BM}$。$T_{BC}$ は身体の共同候補を $K_{{\rm est},t}$ に基づき候補指令 $c$ へ伝播したもの、$T_{BM}$ は過去の有効な画像観測から根元座標系で固定目標として保持したマーカーの候補である。数式は説明のための一点の変換表記であり、**評価時には同じ身体・マーカー共同候補に相対変換を作用させる**こと。

画像が見えない間でも、$T_{BM}$ の保持候補を「無効」「初期配置へリセット」「現在の物理真値へ置換」しない。カメラ側は指令に応じて動き、IMU等の観測で身体の候補が更新されるから、$T_{CM}$ の予測は変化する。新しい有効な画像が来たときだけ、画像由来の情報を通常の推定経路で改めて融合する。**単に見えないことを新しいマーカー姿勢の観測とみなさない。**

既存実装（2026-10-17参照版）には `fixed_goal_memory`、`goal_components`、`camera_path`、`marker_path` がある。最新ブランチでも同等の実体を探し、**新しい独立したマーカー位置コピーを作らず**利用する。目的関数 $J_t$ が使う候補と、視野因子が使う候補のバージョン／出自／剛性代表は同一にする。

## 3. 視野維持因子の数理定義

### 3.1 初期実装：光軸方向へ向ける滑らかな因子

既存の目標mixtureの成分を $k$ とし、同一の候補スナップショットから、候補指令 $c$ のカメラ相対マーカー位置の**予測平均**を $m_{CM,k,t}(c)\in\mathbb R^3$ とする。これは現在の画像の画素中心ではなく、共同候補からの相対座標変換照会によって得る。

光軸の単位ベクトルを $e_z=(0,0,1)^\top$ とする（この座標規約が使用中のcamera optical frameと一致することを確認）。$p\ne0$ に対して、$\phi(p)=1-e_z^\top p/\|p\|$ と定義する。

- $\phi(p)=0$：マーカー中心が光軸正面にある。
- $0<\phi(p)<2$：光軸から離れるほど大きい。
- $\phi(p)=2$：光軸正反対の後方にある。

成分の非負重みを $w_{k,t}$ とし、視野規準を $V_t(c)=\big(\sum_k w_{k,t}\phi(m_{CM,k,t}(c))\big)/\big(\sum_k w_{k,t}\big)$ とする。分母は**この視野規準内での成分の相対重み**を定めるもので、元の候補測度 $\Upsilon$ に正規化条件を課す意味ではない。実装がlog weightを持つなら、underflowを避けて評価する。複数成分を最初から一つの平均点にまとめると、多峰性の打ち消しが起きるため、まず**成分ごとに $\phi$ を評価して重み付け**する。

$\lambda_{\rm view}\ge0$ を設定可能な強さとして、視野維持因子を $\Lambda_{{\rm view},t}(c)=\exp[-\lambda_{\rm view}V_t(c)]$ と定義する。この因子は正で、目標中心が光軸へ近づくほど大きい。**真の検出確率や、マーカーボード全角の可視性を表すものではない**。まずは安価な方向規準として使う。

$\|p\|$ が極端に小さいときは角度が未定義・数値不安定になるため、有効な最小ノルムの判定と明示的な `unavailable` / `degenerate` 診断を実装する。絶対値ゼロ除算やNaNを許さない。分布モードが背面にあるときも、$z\le0$ だけを理由に観測の候補を消したり、因子をゼロにしたりしない（この $\phi$ は後方でも定義できる）。

### 3.2 追加する規準の合成

既存のタスク因子を $\Lambda_{{\rm task},t}(c)=\exp[-J_t(c)]$ とする。制御値の選定時に、$\Lambda_{{\rm select},t}(c)=\Lambda_{{\rm task},t}(c)\Lambda_{{\rm view},t}(c)$ と**掛け合わせる**。

数値的には対数で評価し、$\log\Lambda_{{\rm select},t}(c)=-J_t(c)-\lambda_{\rm view}V_t(c)$ とする。元の $J_t$ と、新規の $V_t$ は**別々に計算・ログ出力**し、因子の結合だけを選定レイヤに実装する。数学的には $J_t(c)+\lambda_{\rm view}V_t(c)$ の最小化と等価だが、コードの上で $J_t$ 本体に視野項を埋め込まない。旧制約 $s$ を復活させない。

$\lambda_{\rm view}=0$ または視野因子が明示的に無効な場合は、**Running Gradient Trackingのタスク勾配と完全に一致**することをテストする。別因子は観測更新（$\Upsilon_{\rm real}$ または $\Upsilon_K$）には掛けず、制御値の選定にのみ使う。

### 3.3 勾配の形

制御側へ渡す総合勾配は、$g_{{\rm select},t}=\nabla_c J_t(c)+\lambda_{\rm view}\nabla_c V_t(c)$ を $c=u_{\rm src}$ で評価したものである。これは $-\nabla_c\log\Lambda_{{\rm select},t}$ に等しい。**符号を間違えないこと。**

視野因子単体の微分を確認する場合は、$r=\|p\|$、$z=e_z^\top p$ として、$\nabla_p\phi(p)=zp/r^3-e_z/r$ を利用できる。姿勢候補の指令Jacobianが得られるなら、$\nabla_c\phi=(\partial m_{CM}/\partial c)^\top\nabla_p\phi$。ただし、既存のmodal感度が平均の補正項を固定している場合、それを厳密な平均微分と呼ばない。

**最初の実装では、現在の勾配workerが行っている有限差分評価を再利用してよい。** 同一の試行指令・共同候補・$K_{\rm est}$ の伝播から $J_t(c)$ と $V_t(c)$ を合わせて照会し、`task_gradient`、`view_gradient`、`selected_gradient` を区別して返す。既存の完全な $J$ の差分評価を安易に近似した視覚Jacobianで置き換えない。可能なら既存相対姿勢照会を再利用して、二重に高価な分布計算を行わない。

### 3.4 Running Gradient Trackingは変更しない

従来合意した実行器は $\dot u=v$、$\dot v=-\gamma v-\eta g_{{\rm select},t}$ である。**新しい目標位置の端点を求める処理へ戻さない。**

観測またはK候補の更新で勾配が変わっても、実行器内部の $u,v$ はそのまま引き継ぐ。非同期で計算された新しい有限の $g_{{\rm select},t}$ だけを採用し、物理120 Hz側は最後に受理した勾配をsample-and-holdして軽量な二階系を積分する。既存の数値積分器、出自ID、source time / adopt time、遅延診断を保つ。新因子の追加によって物理tickを重い分布計算でブロックしない。

### 3.5 初期実装で保証しないこと

$\phi$ は光軸への方向合わせの規準であり、**全マーカーコーナーが画像内に入ること、マーカーが正面向きであること、遮蔽がないこと**を保証しない。目標中心がちょうど光軸の真後ろにある場合など、幾何学的に $\nabla_c V$ が消える退化も起こり得る。検出確率がゼロになった状態から必ず回復するという数学的保証は主張しない。

もしこの単純な因子で不十分なら、将来的にフラスタム境界やマーカー四隅の投影、視野内候補質量などの**別の因子**を追加する余地を残す。ただし、初回実装でそれらや独立した視覚サーボ・旧画像初期値を一度に追加しない。

## 4. マーカー喪失／再観測の状態遷移

次の3状態を区別する。

1. **未獲得**：いまだ有効なマーカー観測が一度も得られていない。視野維持の予測対象が確立していないので、架空の固定マーカー位置を制御規準へ入れない。既存の探索・待機ポリシーを継続してよい。
2. **獲得済み・現在も見えている**：観測で保持マーカー候補を更新し、タスク因子と視野因子を同一スナップショットから計算する。
3. **獲得済み・現在は見えていない**：**保持した根元固定マーカー候補を捨てない。** 身体候補の伝播とIMU更新からカメラ相対マーカーを予測し、引き続きタスク因子と視野因子を計算する。画像不在を理由に、視野因子・タスク因子を自動的にゼロにしたり、旧探索へ即座に切り替えたりしない。

画像が再取得できた場合は、新しい画像観測に基づく通常の候補更新を一回行う。以前の保持情報を重複して新観測として数えない。

**重要：** 2026-10-17の旧参照実装には、`marker_search.py` の `MarkerSearchPolicy.plan()` が「reaching中にマーカーが見えなくなったらbrakingへ遷移する」処理があり、`async_motion.py` もpreviewのvisibleと新しい観測時刻によってreachingをゲートしている。最新版ではコード構成が変わっている可能性があるが、**この旧分岐が残っているなら、獲得済みのマーカーについて単なる画像喪失だけでRunning Gradient Trackingを中断しないよう更新する**。未知目標時の探索・危険時の停止を一律削除する指示ではない。

観測長期喪失や予測分布が著しく不安定になった場合に専用の探索・回復ポリシーを適用するかは、別途設計する。初回実装では、**どの条件で因子が無効／有効となり、制御が継続／停止したかをログで判別可能**にすることを優先する。無根拠な固定秒数・共分散閾値を作って自動移行させない。

## 5. Codexへの具体的な変更指示

### 5.1 作業前の確認

- 現ブランチ・HEAD・差分・直近の実装を確認。前のCodexのrunning-gradient改修と非同期推定が既にあるなら、それを保持して**最小差分**で追加する。
- 既存の相対姿勢照会とmarker根元固定候補の継続保持、goal mixtureとlog weight、K代表のスナップショット、非同期勾配workerの構成を確認する。
- 過去の参照用コード：`ros/examples/probtf_observation_fusion_demo/src/probtf_observation_fusion_demo/visual_approach.py`（カメラ相対マーカー照会だけ参考）、`moving_inference.py`（`fixed_goal_memory`）、`moving_control.py`（`control_components` / 既存の $J$）、`marker_search.py`（画像喪失時モード）、`async_motion.py`（勾配・実行workerの結合）。古い呼び出し構造をそのまま新しいコードに移植しない。
- 参照する旧動作コミット：`e7a3c2a83025f9fc34d2d73e5b7f9c3d5f742e5d`。これは理解用のsource freezeであり、**開発先を旧コミットへ戻さない**。

### 5.2 因子インターフェース

視野因子を独立したモジュール／関数に切り出す。例えば `evaluate_visibility(snapshot, candidate_command, selected_K, config)` 相当のインターフェースを設ける（実名は最新版の設計に合わせる）。出力は少なくとも、`log_factor`、`penalty`、`weight`、`per_mode_record`、`source_ids`、`status` を含む。

タスク規準は既存のまま別に評価する。統合するところでは例えば以下を返す。

```python
# Conceptual interface; adapt to current APIs, do not duplicate transport.
def evaluate_selection(snapshot, command, stiffness, config):
    task_score, task_record = evaluate_task_log_factor(snapshot, command, stiffness)
    view_score, view_record = evaluate_view_log_factor(snapshot, command, stiffness, config)
    total_score = task_score + view_score
    return total_score, dict(task=task_record, view=view_record)
```

`task_score` は $-J_t(c)$ に対応する。`view_score` は $\log\Lambda_{{\rm view},t}(c)$ に対応する。いずれも**log factor**であり、指数関数の積を直接数値評価してunderflowさせない。

### 5.3 勾配・非同期packet

- 一つの不変スナップショットで $J$ と視野因子の両方を評価する。見えない画像フレームに存在しない新しい測定値を捏造しない。
- `task_gradient`、`view_gradient`（$\nabla V$ と符号・係数を明記）、`selected_gradient` を別々に記録する。合成は $g_{\rm select}=g_{\rm task}+\lambda_{\rm view}g_{\rm view}$。`log_task_factor`、`log_view_factor`、`log_select_factor`、mode別 $T_{CM}$ 平均・分散・前後判定、出自IDと観測時刻を記録する。
- 以前の勾配workerの有限差分・ソース時刻・採用時刻・K密度IDの管理を維持。旧い勾配packetを最新の観測として扱わない。新因子の追加による処理時間と失敗を測る。
- `view.enabled=false` または `weight=0` の場合は旧 `running_gradient` と同じ計算結果になる。因子計算が使用不能な場合は黙って不正な値を用いず、理由をログ化する。安全なfallback（最後に受理した勾配を保持する、またはタスク単独勾配への明示的な切替）は決定的・再現可能にする。

### 5.4 設定・実行

- `visibility_factor.enabled`、`visibility_factor.weight`、数値的な最小norm、診断項目をconfigに設ける。視野の中心を画像の画素座標で表すなら、カメラ行列からの正規化と光軸座標系の整合を確保する。
- 新方式の初期設定は**視野因子を有効**にする。重み $\lambda_{\rm view}$ は、タスク勾配と視野勾配の数値スケールをログで確認して設定・明記し、意味不明な固定値を隠して入れない。`weight=0` のablationを必ず行う。
- 目標の画像がなくなっても、既存の根元固定マーカー候補と、そこから伝播したカメラ相対マーカー分布が利用可能なら、視野因子を計算し続ける。
- 物理tick・joint/IMU/image publish、観測データベースの仕組みは変更しない。旧 `near_latched`・旧 $s$・旧端点IKのQP・`K_exec`・旧visual-approach初期値を新しい制御経路に持ち込まない。

### 5.5 実装成果物

Codexは変更ファイル、テスト結果、実行コマンド、選んだ $\lambda_{\rm view}$ とその判断材料、マーカー欠落／再取得ログ、従来との比較、commit IDを簡潔に報告する。既存の作業ブランチにおいて小さな機能単位でcommit/pushし、他の並行改修を消去しない。

## 6. 必須テスト

### 6.1 数理・数値テスト

1. **因子の極値と符号**：光軸正面で $\phi=0$、斜め方向で $0<\phi<2$、真後ろで $\phi=2$、正面へ近づけば $\Lambda_{\rm view}$ が増えること。
2. **ゼロ重み**：$\lambda_{\rm view}=0$ ではタスクの目的値・勾配・物理指令が、同じ入力・同じ実行設定の既存running-gradientモードと一致すること。
3. **導関数**：解析的な $\nabla_p\phi$ と有限差分を比較し、指令空間の $\nabla_c V$ も適切な共有ソースで差分チェックすること。
4. **多峰性**：少なくとも2個の目標成分が別々の方向を向いている場合、一点平均へ潰した計算と異なり、各成分の重み・相関を保って寄与が算出されること。ログ重みのunderflow対策を検証する。
5. **退化値**：$\|p\|$ が小さい場合、背面、数値不正、候補なしの各場合に非有限の勾配を送らないこと。ゼロベクトルを黙って返して成功としないこと。
6. **独立性**：視野因子の重みを変更しても、同じ観測入力に対する $\Upsilon_{{\rm real},t}$ と $\Upsilon_{K,t}$ の更新値が変わらないこと（制御指令が異なれば、将来の観測データが変わり得る点とは区別）。

### 6.2 時系列・非同期テスト

1. **最初から未獲得**：固定マーカー候補を捏造せず、旧探索／待機が使える。
2. **一度獲得→画像ロスト→IMUのみ**：保持した根元固定マーカー候補と身体候補から $T_{CM}$ を更新できる。画像がないだけで視野因子が消えず、追跡が継続する。
3. **再取得**：画像から新しい候補を通常の推定ルートで融合し、重複した観測IDを数えない。
4. **勾配ジャンプ**：画像ロスト／再取得・身体/K更新によって勾配が変化しても、実行器の $u,v$ は連続のまま引き継がれる。
5. **出自**：タスク因子と視野因子は同じcandidate snapshot、同じ $K_{\rm est}$、同じ評価点、同じsource stampを使用し、結果のadopt stampをsource stampに偽装しない。
6. **負荷**：物理120 Hzの更新を、視野因子の分布照会や数値微分でブロックしない。既定の40/30/12 Hz配信と観測DB蓄積・全観測の最終的処理を維持する。

### 6.3 動作評価とablation

マーカー単独、およびIMU＋マーカーの両ケースで、できれば同じ初期条件／乱数seed／実行時間を揃え、少なくとも以下を比較する。

- 視野因子 `weight=0`（Running Gradient Tracking単独）と、`weight>0`（視野因子あり）。
- マーカーの視野内率、連続喪失時間、再取得までの時間、予測マーカー中心の光軸角度・画像位置。
- マーカーボードの四隅が実際の画像に入っているか、検出できたかの診断（物理真値を使うなら**評価側に限定**）。
- 手先とスイッチの実物理距離の時系列、最小・終端距離、近傍維持率、スイッチ動作の達否。
- $J$ と $V$、両者の勾配ノルムと方向の競合、出自と処理遅延、勾配評価時間、関節指令の位置・速度接続。

**合否の観点**：視野内維持が改善しても、スイッチへの到達や動作成功が失われるなら無条件で成功としない。反対に、数学的に綺麗なスコアでも、現実のマーカーロストが改善しなければ有効とは評価しない。改善幅を推測で記入せず、再現ログを残す。

## 7. 今回実装しないこと／未決定事項

- 画像が現在見えているかどうかを直接用いるVisual Servoingの復活、旧画像中央初期候補、旧 $s$ 制約、端点IK、MPC、DMP、SG時間補間、`K_exec`、新しいKの一点平滑化。
- モンテカルロを既存の共同候補伝播の主経路とすること。
- 視野内質量の厳密積分、マーカー四隅の確率的フラスタム判定、遮蔽の確率モデル、画像喪失後の任意な時間閾値による強制リセット。
- 速度・加速度の本格的な上限制約（前回のrunning-gradient計画では初回実験の対象外）。

必要になれば、$\Lambda_{{\rm view}}$ に独立の $\Lambda_{{\rm frustum}}$、$\Lambda_{{\rm occlusion}}$ を後から掛けられる設計にしておく。追加可能性を残すだけで、今回は実装しない。

## 8. 参照する実装（2026年10月17日の旧コード・**最新HEADではない**）

- [moving_inference.py：カメラ/マーカーpath、目標の根元固定候補の保持](https://github.com/hiroya1224/ProbTF-demo/blob/e7a3c2a83025f9fc34d2d73e5b7f9c3d5f742e5d/ros/examples/probtf_observation_fusion_demo/src/probtf_observation_fusion_demo/moving_inference.py)
- [moving_control.py：`control_components`、タスク密度 $J$ の評価](https://github.com/hiroya1224/ProbTF-demo/blob/e7a3c2a83025f9fc34d2d73e5b7f9c3d5f742e5d/ros/examples/probtf_observation_fusion_demo/src/probtf_observation_fusion_demo/moving_control.py)
- [visual_approach.py：`pose_query` による `camera_relative_marker`、各成分の平均・共分散・指令感度](https://github.com/hiroya1224/ProbTF-demo/blob/e7a3c2a83025f9fc34d2d73e5b7f9c3d5f742e5d/ros/examples/probtf_observation_fusion_demo/src/probtf_observation_fusion_demo/visual_approach.py)
- [marker_search.py：マーカー喪失時の旧モード遷移](https://github.com/hiroya1224/ProbTF-demo/blob/e7a3c2a83025f9fc34d2d73e5b7f9c3d5f742e5d/ros/examples/probtf_observation_fusion_demo/src/probtf_observation_fusion_demo/marker_search.py)
- [async_motion.py：旧実行分岐、previewと観測のゲート、非同期ジョブ](https://github.com/hiroya1224/ProbTF-demo/blob/e7a3c2a83025f9fc34d2d73e5b7f9c3d5f742e5d/ros/examples/probtf_observation_fusion_demo/src/probtf_observation_fusion_demo/async_motion.py)
- `ProbTF_unified_theory_running_gradient_codex_20261018.md`：直前の統合数理・追跡制御計画。

**要約**：保持されたマーカー位置候補を、現在の身体候補・剛性代表によりカメラ相対へ伝播する。元の目標密度 $J_t$ に手を入れず、光軸方向へ寄せる視野維持因子 $\Lambda_{{\rm view},t}$ を独立に設け、$\Lambda_{{\rm select},t}=\Lambda_{{\rm task},t}\Lambda_{{\rm view},t}$ と掛け合わせる。既存の二階Running Gradient Trackingには両因子の対数勾配を入力し、画像喪失中も有効な固定マーカー候補があれば追跡を続ける。モデル・推定・視野因子・選定を混同せず、実際のマーカー保持率とスイッチ動作の成否で評価する。
