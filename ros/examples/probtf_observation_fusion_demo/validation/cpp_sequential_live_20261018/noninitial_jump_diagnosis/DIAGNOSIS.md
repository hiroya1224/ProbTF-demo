# 非初期のリンク姿勢ジャンプの分類

対象はC++逐次版のマーカー単独job147・148とIMU併用job487・488・528。同じ原取得時刻・同じ指令で、Keep事前、Reset事前、各親のraw affine事後、canonical事後を比較した。元画像から作り直した中央観測は、保存された中央分布と全要素一致した。全中央成分・weight・log weightを評価し、観測棄却、clip、平滑化、core変更、新しい再線形化は行っていない。

調べた範囲では、時刻・親chart・平均の差し引き・Jacobian・C++再表現の取り違えは見つからなかった。一方、凍結した一次観測モデルでのfitが、事後を座標変換として再合成した後のfitと大きく異なることを数値で確認した。全実装に誤りがないという証明ではなく、この5観測で確認した境界である。

## 一観測で実装している計算

親ごとに原priorを固定し、その共同局所座標を $z\sim\mathcal N(0,P_0)$ とする。観測空間の近似は

$$m_0=h(0)+b(P_0),\qquad h(z)\simeq m_0+J_0z$$

で固定する。$b(P_0)$ は既存coreが計算する、全crossを含む共分散一次の平均補正である。画像成分の局所Gaussianを $y_k,R_k$ とすると、各成分について

$$S_k=J_0P_0J_0^\top+R_k,$$
$$\mu_k=P_0J_0^\top S_k^{-1}(y_k-m_0),\qquad
P_k=P_0-P_0J_0^\top S_k^{-1}J_0P_0$$

を得る。画像の全成分とKeep/Resetの重みは、元のlog evidenceとlog weightで保持している。今回の確認では、原保存queryの平均・共分散・Jacobianは再計算とbitwise一致し、この式による直接条件付けと保存raw joint事後の差は、平均最大 $5.46\times10^{-14}$、全crossを含む共分散最大 $2.30\times10^{-17}$ だった。

そのraw事後を変更していないcoreのPython実装でも独立に再表現した。C++から得たraw stateとの並進差は0、四元数外積差は最大 $7.22\times10^{-16}$、pose平均差は最大約 $10^{-15}$ だった。したがって、この確認ではC++への移植時に再表現式を取り違えた証拠はない。

## 線形fitと、再合成した後のfitは別である

凍結モデルが見込む平均は

$$m_{\rm affine}=h(0)+b(P_0)+J_0\mu.$$

一方、事後の各リンクを有限の $\mu$ だけ動かし、全交差共分散を同じcoreのchartで輸送して再照会すると

$$m_{\rm compose}=h(\mu)+b_\mu(\widetilde P)$$

となる。位置について、その差を実際に

$$m_{\rm compose}-m_{\rm affine}
=\underbrace{h(\mu)-h(0)-J_0\mu}_{\text{有限移動のmodal Taylor残差}}
+\underbrace{b_\mu(\widetilde P)-b(P_0)}_{\text{共分散が変わることによる平均補正の差}}$$

へ分解した。回転は同じ原観測chartでLogとそのJacobianを使って比較した。異なるchartへ同じ数値をそのまま代入した比較ではない。

次の表は選定優勢親の同じ元chartでの値。残差二乗は観測共分散で白色化した値。期待尤度logは、元の画像Gaussian近似と予測Gaussianの重なりであり、厳密な非線形積分や成功確率ではない。

| ケース/job | 優勢親 | affine残差² | 再合成残差² | prior→affine事後→再合成事後の期待尤度log |
|---|---|---:|---:|---:|
| marker147 | Reset | 13.866 | 3895.754 | −53.623 → 28.587 → −18.563 |
| marker148 | Reset | 0.08594 | 1652.801 | 10.891 → 29.782 → 21.338 |
| IMU併用487・画像 | Reset | 19.726 | 16757.213 | −27.391 → 19.013 → −435.099 |
| IMU併用488・IMU | Keep | 7.693 | 6.572 | −18.903 → 4.281 → 4.591 |
| IMU併用528・画像 | Reset | 5.314 | 23475.837 | −20.582 → 28.940 → −52.478 |

marker147・148は再合成後もpriorからは尤度が改善している。ただし、線形モデルが見込んだfitには達していない。IMU487・528の画像更新では、再合成後の期待尤度はそのReset priorからも悪化した。IMU488は直前の画像更新後に生じた重力方向誤差を約11.85度から1.73度へ減らす、Keep優勢の更新だった。

marker147のReset枝では、位置平均の食い違い2.44 cmに対し、modal Taylor残差1.63 cmと共分散補正の差0.871 cmが寄与した。marker148でも1.17 cmの食い違いがあり、対応する二項のnormは1.75 cmと0.885 cmだった。これらはvectorの和なのでnormを単純に足してはいけない。IMU487のReset枝では位置の食い違い15.22 cmに対してmodal Taylor残差が16.03 cm、共分散補正差が0.898 cmであり、有限移動に対する一次写像の外挿差が大きい。

## 17 cmの往復は、単に「Reset=1」で終わらない

marker147のlink6ではKeep→Reset事前が5.61 cm、Reset事前→優勢raw事後が17.08 cm、Keep→canonicalが17.49 cmだった。次の148ではKeep→Reset事前が16.95 cm、Reset事前→raw事後が5.63 cm、Keep→canonicalが15.99 cmだった。いずれもraw優勢枝→canonicalのmodal位置差は $10^{-15}$ m程度であり、最後の単峰化だけが17 cmを作ったのではない。各辺の回転・地図・全crossの記録はJSONに残した。

Resetは保持した地図の周辺を引き継ぐが、身体を直接設定した分布族へ戻し、身体と地図のcrossを0にする。今回、Keep→Resetで地図のmodal位置は変わらず、観測更新後の地図移動はmarker147で2.52 mm、148で0.715 mmだった。大きく変わったのは主に身体側である。Keepのcrossが無言で破棄されたのではなく、宣言されたReset側の独立化と、その親からの有限な観測補正が重なっている。

優勢raw枝の局所回転平均移動の最大値は、marker147が0.317 rad、148が0.0424 rad、IMU487が0.152 radだった。その座標の事後標準偏差はそれぞれ0.0391、0.0391、0.0298 rad。再表現後の最大局所回転分散は約0.00152、0.00153、0.00166 rad²で、既存coreの高集中分散上限0.1 rad²を下回る。高集中の分散条件は、平均移動の大きさや複数リンクを通る有限移動に対する凍結Jacobianの精度を保証する条件ではない。新しい角度閾値や停止条件は追加していない。

## 実姿勢との比較は後解析だけで行った

元のphysical trajectoryの同じframe・stamp・指令に保存された実関節角から、同じURDFでFKした。Kも平衡計算も使わず、FK棒先端と保存された実棒先端が一致することも確認した。この真値は推定・制御へ渡していない。

| 更新 | link6のKeep→canonical位置誤差 | handのKeep→canonical位置誤差 |
|---|---:|---:|
| marker147 | 19.26 → 29.33 cm | 32.41 → 33.81 cm |
| marker148 | 28.12 → 17.91 cm | 31.29 → 31.38 cm |
| IMU487 | 7.85 → 12.40 cm | 9.74 → 11.92 cm |
| IMU488 | 12.01 → 9.20 cm | 11.31 → 9.31 cm |
| IMU528 | 8.63 → 6.71 cm | 8.64 → 5.82 cm |

marker147は実姿勢から遠ざかり、148はlink6について戻るため、この往復をすべて適切な補正とは説明できない。一方、IMU528は観測fitの外挿差が大きくても実位置へは近づいている。大きい更新を一律に異常、または正常と判定する根拠にはしない。

元中央観測は各1成分、全て通常の中央縮約で、mixture fallbackはなかった。使用markerはmarker147が[17,18,20]、148が[17,18]、IMU487が[18]、528が[20,24]。前画像との回転差は約5〜16度で180度flipはない。ただし取得間隔・指令変化もあるため、中心の位置差だけで観測を外れ値と判定していない。

raw全mixtureとcanonical単峰の点密度比較には別の注意がある。IMU487では極小Keep質量 $\log w=-302.732$ の枝が、native点密度 $-660.849$ を持つため、全raw mixtureの点密度logは約 $-963.582$ になる。Reset rawとcanonicalの点密度logは共に約 $-8441.062$。全raw mixtureの点の高さが違うことを、Reset raw→canonicalの再合成不一致とは扱わない。Gaussian期待尤度ではKeepの寄与は小さく、優勢Resetとcanonicalは一致している。

## 未実装の境界

今回の逐次化は、取得ごとに前のcanonical事後を使って新しい観測モデルを構成する。しかし、**一つの観測の中では** $h(z)\simeq m_0+J_0z$ を固定して一度だけBayes条件付けする。この単一観測内の有限更新への外挿精度は、取得順の逐次化だけでは修正されない。

同一の原priorを保持した非線形更新や反復の具体設計は、今回は未実装・保留である。同じ観測を新しい独立データとして繰り返し融合してよいという意味ではない。本調査は数学モデルの変更を実装せず、どの凍結近似がどこで外れたかを記録したものに限る。

詳細はreport.json、physical_pose_comparison.jsonと原K jobのSHAを参照する。raw pixels・DB・物理真値を含む入力を推定側へ追加していない。

## 再現元と入力の対応

- `/tmp/probtf_cpp_noninitial_jump_nonlinear_20261018.py` — SHA256 `d3f88e9d745d579160eae102f739cd245b422b1415513aa34fbd8e1a0370bf38`
- `/tmp/probtf_cpp_noninitial_jump_physical_20261018.py` — SHA256 `b9766d9c19bdab8810418a49971e8943dec34770adb5cbac533087199f4ce0a3`

入力runは `/tmp/probtf_cpp_sequential_live_20261018`。各caseの `observations.sqlite3` から本文・JSONの原event IDを読み、`stiffness_jobs` の原job147/148または487/488/528の参照JSON/pickleを照合した。画像再生成はraw imageだけを使い、前の選択有効画像も同じDBから復元した。物理比較だけは `physical_trajectory.jsonl` の同じ原frameと `robot.urdf` を別プロセスで使う。入力payloadとarchiveのSHAはreport.jsonに記録した。

元runの `source_freeze.json` と照合し、本診断が使うcore・body/board・sensor・projection等151ファイルが同一だった。元runの終了時には全source/config/native/core/RViz不変を確認済みである。後に別目的で改修中の非同期schedulerを、この旧runの実測sourceとして扱っていない。
