# 高集中Binghamの伝播モーメントによる観測更新

2026年10月18日、ユーザーの「伝播式の方式を使いなさい」に従う現行仕様である。候補一点のMAP探索とGN/Laplaceによる身体更新は撤回する。一件の観測につき、既存coreから予測平均・共分散・身体との交差共分散を得て、一回の解析的Gaussian条件付けを行う。反復最適化を追加しない。

単回Gaussian条件付けは今回Codex側で選んだ更新近似である。「伝播式を使う」という指示そのものが、この更新方式を必須にするわけではない。180秒実走では再合成後の観測整合が未達であり、本実装を問題解決済みとは扱わない。

## 使う伝播式と元の共同事前

観測の原取得時刻・実指令に対応するKeep/Reset各親の共同事前を $z\sim\mathcal N(m,P)$ とする。これは既存coreのBingham由来座標で保持した代理であり、別の方位雑音モデルを新設しない。辺を独立な周辺分布へ分解せず、身体・カメラ・未知マーカーの全交差を保持する。

既存のC++版 `pose_query` / `direction_query` は、保護したPython coreと同じ式で $m_y=\operatorname{mode}(y)+\operatorname{correction}_{O(P)}$、$S=HPH^\top$、$C=PH^\top$ を返す。補正平均をmodal姿勢へ置き換えない。coreが返すwithin/betweenは同じ $S$ を全共分散で分けたもので、追加の残余分散や観測雑音ではない。元のcoreもC++版の既定数値式も変更しない。

根拠は `~/dthesis/chapter/4-transform-law-propagation.tex` の方向Gaussian代理・共有局所感度・全共分散の節と、第7章の `sharing:sec:joint-constraint-update` にある一回の共同Gaussian更新である。第4章の局所Gaussian代理と厳密Bingham誘導法則を同一とは説明しない。

## 観測との一回の条件付け

同じ観測座標で、観測値を $y$、独立な観測共分散を $R$ とする。$W=S+R$、$e=y-m_y$、$G=CW^+$ と置き、$m^+=m+Ge$、$P^+=P-CW^+C^\top$ と更新する。実際の共分散計算は、同値なJoseph形 $P^+=(I-GH)P(I-GH)^\top+GRG^\top$ を用いる。固有値floor・jitter・clipを足さない。退化した身体事前の支持は既存coreの支持計算を使い、Dirac座標を広げない。

同じ予測Gaussianによるevidenceは $Z=\mathcal N(y;m_y,W)$、すなわち $\log Z=-\tfrac12e^\top W^+e-\tfrac12\log\operatorname{pdet}W-\tfrac12\operatorname{rank}(W)\log(2\pi)$ である。特異な $W$ では $e\in\operatorname{Im}(W)$ を前提とする支持上の密度であり、実装もその支持条件を検査する。点の残差をMAP探索した値や、GN曲率の行列式を使う式ではない。小さいnative log weightが大きな共通log値への加算で消えないよう、既存の加算残差を保持する正規化を使う。

観測前の $m,P$、予測の $m_y,S,C,H$、原観測は一回の処理中固定する。更新後に再表現した分布は次の新しい取得へ渡す。同じ観測を繰り返し掛けて事後を狭めたり、新しい事前として同じ観測をやり直したりしない。

ここで「条件付ける条件」は、取得時刻・実指令を固定した上で補助観測変数 $Y$ が $y$ を取ったという観測である。代理モデルは $Y=m_y+H(Z-m)+\varepsilon$、$\varepsilon\sim\mathcal N(0,R)$ であり、$Z\mid Y=y$ を計算する。マーカーでは画像由来の各native成分の平均・共分散を $y_k,R_k$ とし、全成分・質量を残す。生画像そのものに対する厳密な条件付けではない。これは実際の有限合成 $F(Z)$ の代わりに凍結Gaussian代理を使う近似であり、$F(Z)=y$ という等式拘束でもない。

更新対象 $Z$ は、各リンク間の座標変換と未知マーカーを含む共同候補の座標である。全平均・共分散・リンク間crossを一緒に更新し、その同じ結果を既存のBingham/Gaussian表現へ書き直す。伝播式そのものや指令値をこの観測条件付けで更新するのではない。Kの連続密度更新は、この観測前後の共同候補を使う別処理である。

## IMUとマーカーの観測座標

IMUは既存の方向伝播で予測される重力方向の補正平均・共分散・全交差を得る。観測方向で固定した2次元の接平面へ、その同じ平均・共分散・全交差を線形照会で写す。Keep/Resetが同じ接平面・同じ観測分散を使い、headingの観測を加えない。これは高集中での局所的な方向Gaussian代理であり、反対方向まで含む全域の識別を保証するものではない。

マーカーは画像だけから構成した各native成分 $k$ のmode $R_k$、元のjoint Gaussian共分散 $R_{{\rm obs},k}$、位置・方位cross、log weightを保持する。元の観測を親ごとに作り変えない。身体からカメラ相対マーカーへcore伝播したqueryのmodal回転を $R_0$ とし、$a_k=\operatorname{Log}(R_k^\top R_0)$、$D_k=\operatorname{diag}(I_3,J_{\rm Log}(a_k))$ でそのqueryをnative固定chartへ輸送する。

使う予測平均は $m_{y,k}=[m_t;\,a_k+J_{\rm Log}(a_k)m_r]$、予測共分散と交差は $S_k=D_kSD_k^\top$、$C_k=CD_k^\top$、感度は $H_k=D_kH$ である。coreの元の $O(P)$ 平均補正はこの輸送で保持する。これはmodal参照点における既存Gaussian queryの一次座標輸送であり、新しい非線形Log座標の二次平均補正や厳密な再積分を追加したものではない。その近似境界を明記する。

各親は同じnative chartで $y_k=[t_k;0]$ と $R_{{\rm obs},k}$ に対して条件付けられる。evidenceは同じ座標のGaussian重なりで比較し、親依存の別の体積係数を追加しない。入力のnative Gaussianは従来の画像Hessian由来の局所代理であり、画素尤度全体やconditional Bingham法則の厳密な全域密度とは呼ばない。

通常は原取得順に一観測ずつ処理する。同時刻の複数センサを一度に渡す診断では、予測感度を縦に並べ、$S_{ij}=H_iPH_j^\top$ と $C=[PH_1^\top,\ldots]$ により共有身体のcrossを残す。独立なセンサnoiseだけをblock diagonalにする。センサごとに独立な身体事前を複製して掛けない。

## 全成分、単一内部状態、配信との接続

各Keep/Reset親とnative観測成分の全組合せを残す。枝のlog質量は親log weight・native log weight・上記log evidenceの和であり、全枝で正規化する。通常weightがunderflowしても有限log weightと枝を消さない。その後の身体単一化は既存の全within/between・全crossを含む処理で行い、同じBingham再表現を使う。

最終canonical lawを次事前、K比の分子、制御、表示へ渡す。K比の分母は実際のKeep/Reset遷移事前である。観測時刻・原指令・sequence・実行区間ごとの凍結K ledger、目標の全成分、身体/Kの独立非同期処理を保持する。表示用に別の更新や平滑化をしない。無観測・欠測・未選択では恒等保持する。

設定は `observation_update.method: propagated_moments` とする。旧 `nonlinear_fixed_prior` を既定や失敗fallbackとして呼ばない。既存の旧式比較は明示した `affine_reference` だけに限る。旧MAP/GNの実装・文書・実測は撤回済み履歴であり、現在の方式の証拠に転用しない。

## 検証する範囲

core伝播の補正平均・全共分散・crossが更新へ入ること、一回Gaussian条件付け・Joseph形・evidenceの独立oracleとの一致、共通身体を使う複数センサcross、Dirac支持、全native質量、欠測恒等性、同じcanonical lawの配信を検証する。保存した元の異常観測・元事前を使う局所比較と、初期状態からの実走を区別する。前回のMAP/GN版の実走を新方式の性能と説明しない。

画像候補自体の外れ、BFSによる仮説切替、絶対姿勢の識別性は別の問題として残り得る。到達や跳ね解消は新方式の実走で確認し、Gaussian更新の数式が正しいことだけで成功とはしない。

## 撤回した非線形版との違いとpush-forwardの意味

ユーザーの追加質問に対応し、内部状態の分布の型と、観測更新の近似を区別する。撤回した非線形版も各枝の内部状態をGaussianで保持していた。ただし、非線形観測写像 $F$ に対する事後 $p(z\mid y)\propto\mathcal N(z;m,P)\mathcal N(y;F(z),R)$ は一般にはGaussianにならない。旧版はこの事後の有限反復MAP点を平均に代用し、その点のGN曲率から共分散を作っていた。厳密な事後平均・共分散ではなく、非Gaussianの積を一つのGaussianへ近似する方法だった。

現在の候補測度を $\mu$ とすれば、候補ごとの有限変換合成 $F$ が定義する予測は $\nu=F_*\mu$ である。観測尤度 $L_y$ による元候補の更新は $\mu^+(dz)=Z^{-1}L_y(F(z))\,\mu(dz)$ と書け、観測座標での更新は $F_*\mu^+(dx)=Z^{-1}L_y(x)\nu(dx)$ となる。この確率モデルの構成と、その積分・事後をどう近似計算するかは別である。

旧版が計算したのは各試行点の $F(z)$ とその感度であり、$F_*\mu$ の分布全体や伝播モーメントではなかった。今回の方式は、先に既存coreから $F_*\mu$ のモーメントと元候補とのcrossを近似して共同Gaussian代理を作り、一回の条件付けを行う。両方式とも最終的な内部状態はGaussian代理であり、Gaussianの型の有無だけでは性能差を説明できない。

特に、凍結した共同Gaussianを条件付けた観測周辺と、更新後のリンク候補を有限合成して再びcoreで照会した周辺は、近似の下では一致するとは限らない。非線形版は探索中に有限合成を評価し直し、今回の方式は元事前の伝播モーメントを条件付ける。この違いを記録し、実走間の指令・取得・遅延の違いを統制しない比較だけで、一方が常に優れるとは断定しない。


## 固定入力での初期検証

新規12テストと、実推定から表示までを含む関連49テストが通過した。保存された旧MAP/GN試行の4観測・全8 Keep/Reset枝についても、新方式の平均は既存coreのGaussian条件付けと最大 $8.92\times10^{-13}$、全共分散は $1.60\times10^{-15}$ の差以内で一致した。native・親の全log weightは一致し、coreの $O(P)$ 補正平均が実際に更新へ入っていることを検査した。[独立検算](validation/propagated_moments_20261018/independent_operator_oracle.json) と [レビュー](validation/propagated_moments_20261018/independent_operator_review.md) に記録する。

同じ凍結Gaussian query内の条件付けが正しいことと、更新後の有限リンク変換を再合成して照会した平均との一致は別である。`local_marginal_mean_closure_error` と `local_marginal_covariance_closure_max_abs` は前者の固定affine代理内の整合を記録し、後者の再合成誤差を零と示すdiagnosticではない。

旧priorに一回適用した検算での、固定条件付け後の相対観測平均と更新後の再合成core平均の位置差は次のとおりである。

| 原画像・旧job | Keep | Reset |
|---|---:|---:|
| Marker 93 | 0.60 mm | 67.47 mm |
| Marker 94 | 19.39 mm | 92.31 mm |
| Marker 95 | 61.24 mm | 62.01 mm |
| IMU併用 321（画像イベント） | 7.48 mm | 105.99 mm |

これは一回の高集中局所Gaussian代理の非閉包による差であり、伝播式を使うだけで有限更新後の逆一致が零になると主張しない。表示の再中心化や点MAP、観測の反復融合で埋めていない。表の事前は旧方式の履歴で形成されており、新方式を初期から実行した精度・滑らかさの評価でもない。


全回帰は1,138件通過した。実コマンド・source hashとtest logを [regression_manifest.json](validation/propagated_moments_20261018/regression_manifest.json) と [full_test_output.txt](validation/propagated_moments_20261018/full_test_output.txt) に保存した。テスト通過を実到達・跳ね解消の証拠にしない。


## 同sourceでの180秒実走

ユーザー指示により、同じsource・動作設定の両ケースを初期から各180物理秒で再試行した。90秒試行と180秒試行を別folderに保存し、物理動作時間と終了後の最大60秒の推定drainを分けた。実行時間だけをlaunchの一時overrideで変え、RVizファイルや候補分布・動作速度を変更していない。359 source/configと4 native binaryが90秒試行と同じだった。

| ケース | 180秒終端の実距離 | 保存物理tick端点での最小距離 | 到達 |
|---|---:|---:|---|
| Marker | 35.10 mm | 33.71 mm（176.84秒） | 未達 |
| IMU＋Marker | 96.96 mm | 80.58 mm（133.06秒） | 未達 |

到達閾値は6 mmであり、今回の物理干渉判定は既存設定どおりOFFである。時間延長による到達成功は確認できなかった。全20,376rawの原取得時刻・指令・sequence・payloadと実ledgerが一致し、全1,705保存候補のwireも保存core照会からの独立再生成と全バイト一致した。これは推定姿勢が正しい証拠ではない。IMU側は終了後drainを含めてもKの340件が未完であり、全処理完了とは記録しない。

全998枝で伝播モーメントとGaussian条件付け式の実装は一致したが、更新後の候補の有限再合成との大きなclosure差が残った。画像候補が真値に近いIMU併用job22でも、canonical相対modal位置のnative観測との差が約4.69 mmから55.62 mmへ増える例がある。凍結条件付けではReset枝の観測残差が約0.046 mmだったのに、同raw枝の再合成平均とのclosureは約55.18 mmだった。全て別の指標であり、初期観測や弱質量枝を含む全枝最大を、通常時のcanonicalリンク補正と混同しない。

実測・保存範囲・遅延は [実走記録](validation/propagated_moments_20261018/live_180s/runtime/README.md)、リンク補正・独立検算は [推定監査](validation/propagated_moments_20261018/live_180s/link_update_audit/REPORT.md)、条件付けと再合成の違いは [数式説明](validation/propagated_moments_20261018/POSTERIOR_CLOSURE.md) に記録した。伝播式やBingham一般を失敗原因と断定せず、今回の観測更新近似に残る不整合として扱う。
