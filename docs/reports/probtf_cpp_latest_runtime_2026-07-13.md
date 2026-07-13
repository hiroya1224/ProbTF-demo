# Prob-TF C++ latest-only runtime 実装報告

## 目的

deflecomp viewerで、50 Hz設定にもかかわらずProb-TF point markerが約40 Hz、markerの
source stamp ageが約99--111 msになっていた。原因は物理simulationではなく、Python bridgeが
dynamic edgeを個別messageとして約1,350 record/s流し、Python listenerのgraph更新と3 pathの
point-moment queryが直列化されることにあった。

## 実装

- `probtf_core`へC++ `LatestSnapshot` libraryを追加した。
  - TF-style forestのforward/inverse path解決
  - Dirac、一様、有限Bingham orientationの一次・二次rotation moment
  - conditional Gaussian translation、mixture、forward point-moment composition
  - exact deterministic inverse
  - stochastic inverseとrepeated latent dependencyの明示的拒否
- C++ `probtf_bridge_node`を追加し、launchの既定実装にした。
  - edgeごとに未処理の最新TFだけを保持
  - 全dynamic edgeを`ProbabilisticTransformArray`のcomplete snapshotとして
    `/probtf_batch`へ一括publish
  - worker変換中に新TFが届いた場合は古いbatchをpublishしない
  - generic用途では従来の個別`/probtf` streamを任意に併送
  - 複数latched `/tf_static` publisherを失わないqueueを使用
- deflecomp point-moment nodeをC++化した。
  - ROS callback 1 threadと計算worker 1 threadの固定構成
  - dynamic batch subscriber queueは1
  - 計算中に新batchが届いた場合、古いmarker resultを破棄して再計算
  - URDFからbase/tipを推定し、従来と同じmean sphereとcovariance axesをpublish
- RViz frame rateを30から60へ変更した。
- Python bridge/consumerは互換・比較用として残した。

## 数値互換性

C++ unit testでは次を確認した。

- deterministic chainのforward composition
- uniform orientationによる点の平均0、covariance `I/3`
- mixture component間covariance
- finite Bingham caseのPython referenceとの一致（平均・covarianceとも誤差2e-10以下）
- stochastic inverseの拒否
- repeated stochastic latent dependencyの拒否

実際のC++ bridge message 39 static edge、27 dynamic edgeを既存Python v2 converterへ入力し、
全recordがlosslessにdecodeされdeterministic transformとして復元できることも確認した。

## Runtime結果

対象はyamaguchi 6-axis URDF、0.35 Hz正弦波reference、ref/cmd 50 Hz、equil 100 Hzである。

| 指標 | Python旧実装 | C++実装 |
| --- | ---: | ---: |
| marker実効rate（動作中） | 約40 Hz | 50.001 Hz |
| marker source age中央値 | 約99--111 ms | cmd 11.3 ms / equil 11.3 ms / ref 17.9 ms |
| marker node CPU | 約70--90% of one core | 約10--13% of one core |
| bridge CPU | 約50--60% of one core | 約7--11% of one core |
| 各C++ node RSS | -- | 約25 MB |

marker位置はmarker自身のstampにおけるTFと`10^-15 m`以下で一致した。最新TFとの差も、中央値と
p95ではほぼfloating-point誤差であり、旧実装で見えていた数mmの時間差は解消した。

## Semantics

`/probtf_batch`は履歴記録用transportではなく、低遅延consumer向けのcomplete latest snapshotである。
中間sampleを意図的に捨てるため、全履歴が必要なestimatorやoffline解析は従来の個別topicまたはrosbagを
使用する。C++ evaluatorは未解決の相関や確率的逆変換を近似して黙って返さず、Python coreと同様に
unavailableとして扱う。
