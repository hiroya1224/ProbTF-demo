# Native ProbTF v2 link point moments

## 1. 概要

6 自由度アームの不確かな joint transform は YAML から直接 native ProbTF v2 records として
読み込まれる。link-cloud demo ではスライダの関節角で revolute record の mode を更新する。
旧 `ProbTfTree` や demo 固有 path algebra は使わない。

```text
configs/simple_six_dof_prob_tf.yaml
  + joint_state_publisher_gui -> /joint_states
  -> config_from_mapping(joint_positions=..., dynamic_joints=True)
  -> revolute: TransformDistributionStamped(is_static=False)[] -> /probtf
  -> fixed:    TransformDistributionStamped(is_static=True)[]  -> /probtf_static
  -> ProbTfBroadcaster.send_transforms()
  -> RosProbTfListener
  -> lookup_point_moments()
```

## 2. YAML と関節角から v2 records への変換

各 edge は次を持つ。

- physical `edge_id` と parent/child frame
- finite Bingham または Dirac orientation
- conditional Gaussian translation
- representative transform と代表値の種別
- authority、source、provenance

fixed joint は Dirac orientation、revolute joint は有限 Bingham orientation になる。YAML の frame
一覧と edge topology が一致しない場合、load 時に失敗する。

`follow_joint_states:=true` の `probtf_static_broadcaster.py` は fixed joint を static set として latch
publish し、各 revolute joint は受信した角度を mode に合成して同一時刻の dynamic record として
publish する。Bingham のばらつきは維持され、mode だけがスライダ角に追従する。

## 3. graph と lookup の向き

ProbTF edge は child 座標を parent 座標へ写す。

$$
z_{parent} = R(Q) z_{child} + X
$$

link に固定された点 `r` を `base_link` で求める query は次である。

```python
listener.lookup_point_moments(
    target_frame="base_link",
    source_frame=link_frame,
    point=r,
    policy=TemporalPolicy.LATEST,
)
```

fixed edge の sample stamp は 0 だが、static resolution は要求時刻に依存しない。revolute edge は
`JointState` の stamp を持つ。runtime node は `wait_for_lookup()` で topology の受信完了を確認して
から query する。

## 4. point moments

`lookup_point_moments()` は path を lazy kernel として構成し、各 forward component の first/second
moments を順に適用する。component の conditional translation を

$$
X(Q) = x_{ref} + C(\operatorname{vec}(R(Q)) - \operatorname{vec}(R_{ref})) + \epsilon
$$

とすると、`C` が orientation/translation coupling を表す。point moment evaluator はこの項を
含めて平均と共分散を計算する。

mixture edge では component を collapse せず、正規化 weight による total expectation と total
covariance を計算する。結果は `KernelResult` で返り、status、approximation、diagnostics を伴う。

## 5. link pointcloud

`prob_tf_link_cloud_node.py` は各 link の x/y/z 軸端点について個別に
`lookup_point_moments()` を呼ぶ。得られた平均と共分散から、RViz 用 PointCloud2 の点を最後にだけ
Gaussian sampling する。

この sampling は描画処理であり、次を行わない。

- edge orientation を手計算で sample して path を再構成する。
- sample から ProbTF edge を再推定する。
- tangent surrogate や legacy tree result を graph に戻す。

したがって計算経路は native lookup moments、sampling は terminal rendering に限定される。

## 6. 実行

```bash
roslaunch symaware_grasp prob_tf_link_cloud.launch
```

RViz と `joint_state_publisher_gui` が起動し、6 本のスライダで姿勢と pointcloud を操作できる。

RViz なし:

```bash
roslaunch symaware_grasp prob_tf_link_cloud.launch rviz:=false
```

別の設定を使う場合:

```bash
roslaunch symaware_grasp prob_tf_link_cloud.launch \
  config_path:=/absolute/path/to/config.yaml
```

## 7. 検証点

- revolute joint は `/probtf` の dynamic v2 record、fixed joint は `/probtf_static` の static v2 record である。
- listener から全 configured link の point moments を取得できる。
- covariance は対称 positive semidefinite である。
- `/probtf_static` の channel に dynamic record を入れない。
- link cloud node に legacy tree import や per-edge manual Bingham sampler がない。
