from pathlib import Path
import json,numpy as np
O=Path(__file__).parent;r=json.load((O/'failure_audit.json').open());p=r['last_packet'];prev=r['selected_packets'][-2];f=r['failure'];v=r['violations'][0];a=r['analytic_command_only_reproduction'];truth=r['truth_for_selected_requests'][-1]
s='''# IMU＋Marker・all_oracleの到達後の実指令領域逸脱

この条件は、制御capture時の6リンク中心と根元目標中心に真値、身体SD0.01倍、目標位置SD0.1 mm・回転SD0.001 radを与え、K作用にも真のKを使う。画像が有効検出された場合の中央観測も正確な相対姿勢と狭い有限幅にする。画像の欠測自体は補わない。元の確率法則全体を真値へ置き換えた条件ではない。

180秒要求に対し、実行済み98.925481秒で終了した部分試行だった。初到達は37.413863秒、最小距離は62.505840秒で1.081215 mm。その後大きく離反し、最後の物理距離は1101.634 mmだった。総弧長は41.223 mで、初到達後の保持・安定性の成功と記録してはいけない。

## 最後の停止は実範囲外提案

関節5のURDF下限は−2.3562 rad、最後に実行した位置は−2.342888529 rad、速度は−2.410043953 rad/sだった。次tickの提案位置−2.363306528 radが下限を0.007106528 rad越えた。この位置は送信されていない。次tick終端は98.933907秒だった。

保存済みg/H・最後の位置・速度・実dtだけで、次提案を最大差0で再現した。新しい推定・候補照会・FK・平衡・物理動力学の再実行はしていない。NaN/Infや独自の精度閾値で停止したのではなく、有限な実指令候補の領域違反だった。速度・加速度clipは現行productionどおり無効で、診断で追加していない。

## K利用可能後は真Kで一定

K利用可能後に使われた値は `[350, 125, 95, 60, 40, 30]` Nm/radで、その後は同じ真値だった。最初のK使用区間は約0.536144秒である。それより前は利用不能期間を元仕様どおり保持しており、59区間でK未選定の宣言済み剛体作用による微小動作があった（最大一段指令変位norm0.007192 rad）。全初期動作まで真Kだったとは説明しない。その初期経路の影響も除外していない。

最後の90〜99秒の失敗過程には数値Kの変更はなく、直前のK代表の誤差・跳ねを必要としない。

ただし、真のKを入れた準静的モデルの指令作用と、速度を持つ実動力学の次の姿勢は同じではない。身体・目標の中心をsourceで正確に与えることも、その先の保持局所モデルの妥当性・安定性を保証しない。

## 最終packetと直前の入力

|項目|直前request533|最後request534|
|---|---:|---:|
'''
for label,key in [('source時刻 [s]','source_s'),('採用時刻 [s]','adoption_s'),('source age [s]','gradient_age_at_adoption_s'),('source勾配norm','source_gradient_norm'),('H最大固有値',None),('固定モデル最小減衰比','minimum_fixed_model_damping_ratio'),('最速固有振動数 [Hz]','fastest_natural_frequency_Hz'),('採用時速度norm [rad/s]','velocity_norm_at_adoption')]:
 values=[t['H_eigenvalues'][-1]if key is None else t[key]for t in(prev,p)];s+=f'|{label}|{values[0]:.9g}|{values[1]:.9g}|\n'
s+=f'''
最後のsource勾配normは約1.078e8、視野勾配normは約1.434e6だった。HはSPDだが、無拘束局所平衡点は範囲外で、関節1が−5.07147 rad、関節2が+6.25245 rad、関節4が−8.63602 radだった。関節5の平衡成分だけは−0.67312 radで範囲内である。結合した6自由度の過渡応答なので、各関節の平衡成分だけで範囲保持を判定できない。

最後の切替では、同じuでの旧／新局所勾配cosが{p['switch']['gradient_switch_cosine']:.6f}、加速度入力の段差normが{p['switch']['acceleration_jump_norm_rad_s2']:.6f} rad/s²だった。最後の切替だけが方向反転した例ではない。前のrequest530はsourceから採用まで約0.743秒あり、切替時の旧／新勾配cosは約−0.319だった。複数の外側局所平衡点と入力切替が続いていた。

|採用秒|request|source age秒|source勾配norm|局所平衡点が全関節範囲内|加速度入力段差norm rad/s²|
|---|---:|---:|---:|---|---:|
'''
for q in r['selected_packets'][-8:]:s+=f"|{q['adoption_s']:.6f}|{q['request_id']}|{q['gradient_age_at_adoption_s']:.6f}|{q['source_gradient_norm']:.6g}|{q['local_equilibrium_within_joint_bounds']}|{q['switch']['acceleration_jump_norm_rad_s2']:.6f}|\n"
s+='''
最後のpacketを採用時から約0.081895秒だけ保持した線形実行式でも、範囲外提案までの運動が生じる。採用時から一括計算した位置の差は最大9.55e−8 rad、最後の一tickを保存dtで再計算した差は0だった。前者の小差は丸めた絶対stampの差と、実際のmonotonic dt積算との違いである。数値積分の取り違えを必要とせず、保存された局所実行式の入力で再現できる。

これは固定SPDモデルの無拘束安定性と、有限関節範囲・packet切替・動的物理を含む実行の保証が異なることを示す。全時間の発散機構を一つの項へ同定した結果ではない。

## 最終captureと確認できた範囲

専用truth ledgerのrequest534はsource98.681136秒、同じ送信指令に一致し、6局所リンクと目標の真値を保持している。真値援助のroutingはそのcaptureにも有効だった。source時点の指令と実身体角度の差はL2で約1.143934 radだった。例えば関節2の指令−0.185476 radに対し、実身体は+0.688878 rad。現在中心が正しいことと、準静的な増分作用がこの高速過渡を再現することは区別する。

完成した最後の表示／候補artifactはsource0.287590秒・request5で、失敗直前のrequest534の完全pickleは保存されていない。したがって、最後の全候補から完全Jの位置／方位／共分散項を再伝播してはいない。保存g/H・ID・実指令・真値ledgerの実行器検算と、完全候補PDFの再構成を混同しない。

元の179個のproduction/config/RViz sourceと、6個の凍結診断helperはsetupのSHAと一致した。入力ファイルSHAと、90秒以後のpacket・終盤指令・真値capture・K・fail-safeの対応は`failure_audit.json`に残した。推定・制御・coreの実装や候補分布を修正した調査ではない。

この試行により、中心とKの誤りだけをなくせば安定する、とは言えない。一方、候補の有限な幅・異方性・近似・遅延は残っており、推定と無関係、規準だけが唯一原因、という結論にもしていない。凍結候補で得た共分散項の反例は別の証拠であり、この最後のIMU packetの項別分解へ付け替えない。
'''
(O/'REPORT.md').write_text(s)
