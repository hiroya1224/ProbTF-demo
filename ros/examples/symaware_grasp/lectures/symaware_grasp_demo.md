# Native ProbTF v2 symmetry-aware grasp demo

## 1. 目的

このデモは、対象物、grasp target、手先の姿勢不確かさを代表姿勢へ早期に潰さず、
ProbTF v2 の joint transform distribution のまま IK と可視化へ渡す。

各 transform component は次を保持する。

- quaternion Bingham orientation
- orientation に条件付けられた Gaussian translation
- mixture weight
- approximation と provenance

特に translation の `rotation_coupling` は、対象物から grasp offset を合成したときの
`R(q) r` を表す。grasp target の生成、ROS 転送、IK の point moment 計算でこの項を保持する。

## 2. runtime 構成

計算用の正本は `/probtf` と `/probtf_static` である。

```text
simple_six_dof_prob_tf.yaml
  -> probtf_static_broadcaster
  -> /probtf_static

object_pose_node
  -> /probtf
  -> /symaware_grasp/object_belief

prob_tf_grasp_target_node
  <- RosProbTfListener lookup(object)
  -> /probtf
  -> /symaware_grasp/grasp_targets

hand_prob_tf_publisher
  -> /probtf
  -> /symaware_grasp/hand_belief

symmetry_aware_ik_node
  <- RosProbTfListener lookup(grasp targets)
  -> target_joint_states
  -> /symaware_grasp/selected_target

ptf_visualizer
  <- application message + RosProbTfListener lookup
  -> PointCloud2 / PoseStamped / Marker
```

producer は最初に native v2 record を `/probtf` へ publish し、その後に application
message を publish する。consumer は application message の frame、stamp、用途 metadata を
トリガーとして使い、実際の transform は `RosProbTfListener.lookup_kernel()` から解決する。

## 3. application message

用途固有の topic では、用途 metadata と完全な v2 payload を同時に運ぶ。

| Message | 主な metadata | v2 payload |
|---|---|---|
| `ObjectBelief` | `object_id` | `ProbabilisticTransformStamped transform` |
| `HandBelief` | `hand_id` | `ProbabilisticTransformStamped transform` |
| `GraspTarget` | object/grasp ID、weight、semantic axes | `ProbabilisticTransformStamped transform` |
| `GraspTargetArray` | object ID | `GraspTarget[]` |
| `SelectedGraspTarget` | object/grasp ID | `ProbabilisticTransformStamped transform` |

これらは v1 の位置 Gaussian と姿勢 Bingham の分離 message へ変換しない。component 数、raw
weight、conditional translation、approximation、provenance は v2 message conversion で往復する。

## 4. object belief と grasp target 合成

object belief は `TransformDistributionStamped` として生成される。各 grasp candidate の固定変換
`T_OG` を右から合成し、`T_WG = T_WO T_OG` を得る。

`compose_with_deterministic_right()` は component ごとに以下を行う。

1. orientation Bingham を quaternion 右作用で変換する。
2. mixture weight と residual covariance を保持する。
3. grasp offset による translation/orientation coupling を `rotation_coupling` に追加する。
4. provenance に元の object edge ID を残す。

したがって、対象物の orientation が広いとき、grasp point の位置共分散も対応して広がる。

## 5. hand belief

`EndEffectorBeliefModel` は joint uncertainty から得た FK sample の quaternion second moment を
core `probtf.bingham.match_bingham_to_second_moment()` で fit する。外部の legacy Bingham runtime
や v1 adapter は使わない。出力は一 component の native v2 record である。

この sample は手先 belief producer の近似生成に用いるものであり、link pointcloud の伝播計算
とは別である。生成した近似であることは source/provenance で識別する。

## 6. symmetry-aware IK

IK は `TransformDistributionStamped` を入力とし、usable component を正規化してすべて評価する。

- position cost は component ごとの coupled point moments を使う。
- pointwise orientation cost は finite Bingham の `-q^T A q` を使う。
- deterministic baseline は各 component mode との符号不変距離を重み付きで使う。
- Bhattacharyya method は target と hand の全 component 対を mixture weight で積分する。

Bhattacharyya method が有限 Bingham 以外を受け取った場合は明示的に拒否する。uniform や Dirac
を有限 parameter とみなす暗黙変換は行わない。

## 7. 可視化

一般 belief visualizer は listener lookup で得た v2 distribution を
`sample_transform_distribution()` へ渡す。この sampler は mixture weight、finite/Dirac/uniform
orientation、conditional Gaussian translation を同時に扱う。sampling は PointCloud2 を作る
表示終端でのみ行う。

mode pose は最大正規化 weight の component を表示する。Marker は全 component の mode axes を
描き、透明度に component weight を反映する。これは表示用 representative であり、計算用 law
の置換ではない。

## 8. 実行

```bash
roslaunch symaware_grasp probabilistic_tf_demo.launch
```

RViz を起動しない場合:

```bash
roslaunch symaware_grasp probabilistic_tf_demo.launch rviz:=false
```

topic を分離する場合は `probtf_topic`、`probtf_static_topic`、各 application topic の launch arg
を変更する。producer と consumer には同じ値が渡される。

## 9. 実装上の境界

- TF の deterministic representative と ProbTF distribution を混同しない。
- app message callback の payload を直接計算へ渡さず、listener で同じ stamp の graph record を解決する。
- mixture reduction や independent translation への射影を暗黙に行わない。
- unsupported orientation kind や temporal lookup failure はログまたは例外で明示する。
- visualization sampling を推論結果として graph へ再登録しない。
