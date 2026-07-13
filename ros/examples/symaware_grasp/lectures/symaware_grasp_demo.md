# Native ProbTF v2 symmetry-aware grasp demo

## 1. 目的

このデモは、軸対称物体の目標姿勢を代表姿勢へ潰さず、ProbTF v2 の Bingham orientationを
deterministic FKで得た手先姿勢のIK costへ直接入れる。手先側は一点であり、joint noiseから
手先分布を作る確率伝播、target/hand分布間距離、deterministic baselineの切替は持たない。

各 transform component は次を保持する。

- quaternion Bingham orientation
- orientation に条件付けられた Gaussian translation
- mixture weight
- approximation と provenance

現行デモ設定の grasp offset は位置0・回転identityであり、object lawをそのままIK targetとして
使う。framework自体は一般のdeterministic right compositionを保持するが、このlaunchの目的は
確率伝播の評価ではなく、確率的targetをpointwise costとして利用する最小例である。

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

symmetry_aware_ik_node
  <- RosProbTfListener lookup(grasp targets)
  -> target_joint_states
  -> /symaware_grasp/symmetry_aware_ik_result

ptf_visualizer
  <- application message + RosProbTfListener lookup
  -> PointCloud2 / PoseStamped / Marker
```

producer は最初に native v2 record を `/probtf` へ publish し、その後に application
message を publish する。consumer は application message の frame、stamp、用途 metadata を
トリガーとして使い、実際の transform は `RosProbTfListener.lookup_kernel()` から解決する。
`probabilistic_tf_demo.launch` はIK nodeを起動しない。sceneとtopicが立ち上がった後、利用者が
`rosrun symaware_grasp symmetry_aware_ik_node.py` を実行した時だけ一回solveする。
後発listenerがlatched application messageだけを受け、対応する過去のdynamic `/probtf` recordを
取り逃すことを避けるため、node起動時刻以後のfresh `GraspTargetArray`を待ってからexact lookupする。

## 3. application message

用途固有の topic では、用途 metadata と完全な v2 payload を同時に運ぶ。

| Message | 主な metadata | v2 payload |
|---|---|---|
| `ObjectBelief` | `object_id` | `ProbabilisticTransformStamped transform` |
| `GraspTarget` | object/grasp ID、weight、semantic axes | `ProbabilisticTransformStamped transform` |
| `GraspTargetArray` | object ID | `GraspTarget[]` |

これらは v1 の位置 Gaussian と姿勢 Bingham の分離 message へ変換しない。component 数、raw
weight、conditional translation、approximation、provenance は v2 message conversion で往復する。

## 4. object belief と grasp target 合成

object belief は `TransformDistributionStamped` として生成される。一般実装では各 grasp candidateの
固定変換 `T_OG` を右から合成し、`T_WG = T_WO T_OG` を得られる。

`compose_with_deterministic_right()` は component ごとに以下を行う。

1. orientation Bingham を quaternion 右作用で変換する。
2. mixture weight と residual covariance を保持する。
3. grasp offset による translation/orientation coupling を `rotation_coupling` に追加する。
4. provenance に元の object edge ID を残す。

ただし現行launchの`cylinder_side_grasp`は`r_OG=0`かつ`R_OG=I`である。従って実行される
デモではobject Binghamを別のrobot-side uncertaintyへ伝播せず、frame/用途名を付けた同じtarget lawを
pointwise IKへ渡す。

## 5. deterministic hand boundary

手先位置・姿勢は `ToyArm6DOF.forward_kinematics(theta)` のdeterministic値だけを使う。
`HandBelief` message、joint noise sample、Bingham fit、hand belief publisherはruntime/sourceから除去した。
IK nodeは`target_joint_states`と数値metadataの`IKResult`だけをpublishし、手先または選択targetの
ProbTF message、sample cloud、mode axesを生成しない。
このため本デモはrobot uncertainty propagationや分布同士のregistration性能を示さない。

## 6. symmetry-aware IK

IK は `TransformDistributionStamped` を入力とし、component weightを正規化してすべて評価する。

- position cost は component ごとの coupled point moments を使う。
- orientation cost はdeterministic FK quaternion $q_H(\theta)$ に対するfinite Binghamの
  $-q_H(\theta)^T A q_H(\theta)$ だけを使う。
- motion priorとjoint-limit costを加え、軸対称な高尤度集合の中から現在姿勢に近い解を選ぶ。

solver method selectorはなく、finite Bingham以外のtarget componentは明示的に拒否する。

## 7. 可視化

一般 belief visualizer は listener lookup で得た v2 distribution を
`sample_transform_distribution()` へ渡す。この sampler は mixture weight、finite/Dirac/uniform
orientation、conditional Gaussian translation を同時に扱う。sampling は PointCloud2 を作る
表示終端でのみ行う。

mode pose は最大正規化 weight の component を表示する。Marker は全 component の mode axes を
描き、透明度に component weight を反映する。これは表示用 representative であり、計算用 law
の置換ではない。

object modeには直径`0.10 m`、高さ`0.18 m`の半透明cylinder markerを重ねる。URDFの`tool0`は
palm、左右finger、内側padからなるparallel gripperとして描画する。tool0原点はgrasp centerなので、
IK後はfinger間の`0.105 m` gapへ直径`0.10 m`のcylinderが収まる。
IK後も追加のbelief cloudは生成せず、手先は通常`/tf`で動くRobotModelだけを表示する。

## 8. 実行

```bash
roslaunch symaware_grasp probabilistic_tf_demo.launch
```

別端末でIKを明示的に一回実行する。

```bash
rosrun symaware_grasp symmetry_aware_ik_node.py
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
- hand uncertainty、Bhattacharyya距離、method切替、launch時の自動IKを再導入しない。
- finite Bingham以外のorientation kindやtemporal lookup failureはログまたは例外で明示する。
- visualization sampling を推論結果として graph へ再登録しない。
