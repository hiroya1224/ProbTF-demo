# 条件付け後のGaussian代理と、再表現したBingham候補の観測整合

現在は、観測前の共同Gaussian代理では観測へ条件付けしている。ただし、その結果をBinghamへ再表現し、有限なリンク変換を再合成した後の観測整合までは満たしていない。

`moment_observation.py::propagate` はcoreの補正平均・全共分散・全crossを受け取る。`moment_conditioning.py::condition_moments` は、その同じ凍結した $\mu_0=h_0+b_0(P_0)$、$S_0=H_0P_0H_0^\top$、$C_0=P_0H_0^\top$ を使い、$\delta=m^+-m=C_0(S_0+R)^+(y-\mu_0)$ と一回更新する。このGaussian代理の観測後平均は $\mu_f^+=\mu_0+H_0\delta$ である。これは元priorと観測尤度の積に対応する代理内の条件付けであり、観測値への残差を常に零にする制約ではない。

実際の `JointTransformState` は再表現済みの零中心chartなので、上の $m=0$ であり、得た共同平均が $\delta$ になる。その後、`moment_model.py` は得た共同平均・共分散を `reexpress_state` へ渡す。既存core／同値C++実装は各辺で $t_e^+=t_e+\delta_{t,e}$、$q_e^+=q_e\rho(\delta_{r,e})$ と有限に移し、全共分散を $P_c^+=G(\delta)P_{\rm raw}^+G(\delta)^\top$ へ輸送する。回転のBinghamは、そのmodal quaternionと局所回転共分散から $A_e^+=-2N(q_e^+)\Omega_e^{+,-1}N(q_e^+)^\top$ と再表現する。ここでは観測に対する再合成後の条件を再び解いていない。

同じ固定native観測chartで再照会したcore平均を $\mu_r^+=h_r+b_r(P_c^+)$ と書けば、差は $\mu_r^+-\mu_f^+=[h_r-h_0-H_0\delta]+[b_r(P_c^+)-b_0(P_0)]$ と分けられる。

- 第一項は有限な全リンク合成と、元の感度による一次予測との差である。位置では有限な回転・並進の合成、方位では固定native chartへの有限なLog座標の変化も含む。
- 第二項はcoreの $O(P)$ 平均補正の変化である。観測で共分散が変わることに加え、新しいmodal幾何、全cross、共分散のchart輸送によって補正自体が変わる。旧modal点での補正を線形関数 $b_0(P)$ と書けば、$b_r(P_c^+)-b_0(P_0)=[b_0(P_{\rm raw}^+)-b_0(P_0)]+[b_r(GP_{\rm raw}^+G^\top)-b_0(P_{\rm raw}^+)]$ と、共分散更新と再表現・幾何変更をさらに区別できる。
- Bingham再表現はmodal姿勢と局所共分散を記述する操作であり、厳密なBingham事後への条件付けや、観測空間の平均を保存する厳密な変換ではない。観測方位へのchart輸送もmodal点での一次近似であり、新chartの二階平均補正は追加していない。これらの影響は上の二項にも現れるため、独立な第三の誤差として重複加算しない。

最終canonical化で複数枝を単一化する場合、そのwithin/between混合の効果もraw枝の再表現とは別に扱う。

## 今回、数値で確認できていること

対象は新180秒試行のIMU併用case、job22、`image:00000149`、取得時刻1.3467875秒である。独立後解析では、Reset親の凍結共同Gaussian条件付け後の位置平均とnative中央観測との差は0.0463137 mmだった。同じReset raw枝を再表現してcoreで再合成すると、凍結条件付け平均とのclosure差は55.182454 mmだった。

一方、最終canonical事後のcore平均とnative観測との差は55.138747 mm、canonicalの観測前modal差は4.690464 mm、観測後modal差は55.618783 mmだった。Resetの事後log weightは0で実質的に同枝が支配するが、raw枝とcanonicalの測定を同一の項目として混ぜない。

測定済みなのは、この総closure差と各観測残差である。上記の有限合成項、共分散更新による補正変化、chart／Bingham再表現による変化のベクトル別寄与は、ここでは独立に測定していない。55.618783−55.138747の距離norm差を、そのまま $O(P)$ 補正量とみなすこともできない。

元測定は [選定事象](live_180s/link_update_audit/selected_physical_oracle.json) に保存した。この説明のためにsource・テスト・候補・アルゴリズムを変更していない。
