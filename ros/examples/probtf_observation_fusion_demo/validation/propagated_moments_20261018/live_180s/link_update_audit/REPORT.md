# 180秒・伝播モーメント方式の独立監査

両180秒実走は未到達。物理側の終端距離はmarker35.10mm（最小33.71mm）、IMU併用96.96mm（最小80.58mm）で、両reached=falseである。

## 同じ時刻・指令での推定更新

全保存job58/443件、うち有効な観測更新57/442件を確認した。初期t=0補正を除く、全リンクの最大modal位置補正はmarker25.88mm（hand、source5.3458秒）、IMU併用35.37mm（link6、1.3468秒）。最大方位補正は0.12494/0.13185rad。初期の手先補正238.34/235.53mmは別扱いにし、図でも隠していない。

全998親・native枝で元共同priorとcore照会を再計算した。記録された補正平均・S・C・Hとの差は最大1.12e-16。Gaussian条件付けoracleとの差は平均2.80e-12、共分散1.53e-15、evidence2.91e-9。global logweightは再計算と厳密に一致。全source q/t/ID、原prior一回、fullcrossを確認し、point/MAP/GN/body-Laplace呼出しは0だった。保護214ファイルは全バイト一致。

これは凍結Gaussian代理の行列式が実装どおりであることの確認であり、推定姿勢が正しいことの確認ではない。

## 有限再合成のclosureは大きく残る

凍結Gaussianの観測事後平均と、更新した共同lawを有限再表現してcoreで再照会した平均は一致しない。全枝の最大位置closureは114.41/230.98mm。微小質量枝も含むので、最大事後質量枝を分けると、物理終端前の受理分でmarker60.61mm、IMU169.78mmだった。IMUのsource180秒・physical受理180秒の更新は182.36mmだが、終端時計はdrain中も180へclampされるため、そのログだけで動作中の影響と断定しない。

これは物理誤差とも、同qリンク補正量とも異なる指標である。図fixed_proxy_recomposition_closure.pngでは全枝最大と最大質量枝を分けた。

## 最大事象の誤差を区別する

IMU job22（source1.3468秒、body受理2.6972秒）の画像native平均は、別後解析の真T_CMから0.731mm/0.00645rad。ところがmodal canonical相対姿勢と真T_CMの位置誤差は4.01→55.60mmへ増える。

同じjob22をnative観測平均との差で書くと、prior modal4.690mm、post modal55.619mm、post core補正mean55.139mm。一方、Reset枝の凍結Gaussian条件付けが得た観測事後meanのnative位置残差は0.04631mmである。このfrozenmeanと再合成coremeanとの差が55.182mmのclosureである。数式検算の一致と、有限再合成後の不一致を混同しない。

IMU job422（source144.0912秒、body受理163.0080秒）ではnative画像の真位置誤差0.237mmに対し、canonical modalの真相対位置誤差は163.96→170.27mm、coremeanのnative位置残差は169.75mm。この選定事象を大きな画像位置外れだけで説明することはできない。

marker最大位置補正job13では、nativeの真相対位置誤差2.73mm、更新前後の真相対位置誤差24.06→3.37mm。最大方位補正job31では、native真位置誤差1.32mmに対し更新後相対位置誤差12.71mmであり、観測との一致は一律に改善するわけではない。

真値は保存actual_jointsと保存URDFのカメラFK、元sceneの固定markerによる後解析専用である。推定や制御へ入力していない。実際のK受理packet、原取得時計、body受理時計、各実行区間のK出自はsummaryと最大事象資料に保存した。K切替だけや表示遅延だけへ原因を断定していない。

新sourceを初期状態から180秒実行した本結果と、以前の固定旧prior4例の局所比較は別資料である。原dataの破壊、観測の棄却、分散clip、反復修復、新手法、production変更は行っていない。
