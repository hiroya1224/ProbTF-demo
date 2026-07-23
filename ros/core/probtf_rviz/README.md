# probtf_rviz

`probtf_rviz` は、ProbTF v2 の変換分布をRViz内で直接sampleして描画する
RViz pluginである。表示専用の `PointCloud2`、`PoseStamped`、
`MarkerArray` へ確率分布を変換する必要はない。

2種類のDisplayを提供する。

| Add一覧名 | class ID | 用途 |
|---|---|---|
| **ProbabilisticPose** | `probtf_rviz/ProbabilisticPose` | 1本の `ProbabilisticTransformStamped` を表示 |
| **ProbabilisticTF** | `probtf_rviz/ProbabilisticTF` | dynamic/static ProbTF edge集合をtreeとして表示 |

確率計算とsamplingは `probtf_core`、RViz/Qt/Ogre依存の描画だけをこのpackageが
担当する。

## 1. ビルドとDisplayの追加

```bash
cd /home/leus/catkin_ws
catkin build probtf_rviz
source /home/leus/catkin_ws/devel/setup.bash
rviz
```

RVizで次を操作する。

1. 左下の **Add** を押す。
2. **By display type** を選ぶ。
3. `probtf_rviz` groupを開く。
4. **ProbabilisticPose** または **ProbabilisticTF** を追加する。
5. 後述のTopicとframeを設定する。

tree表示の同梱設定は次でも開ける。

```bash
rviz -d "$(rospack find probtf_rviz)/rviz/probtf_tree.rviz"
```

pluginをbuildし直した場合、起動中のRVizは古い共有libraryを保持している。
`source` し直したterminalからRVizを完全に再起動すること。

## 2. どちらのDisplayを使うか

### Probabilistic Pose

1 topicに流れる1本の確率的姿勢を追う場合に使う。たとえば、物体検出器が
専用topicへ出す `ProbabilisticTransformStamped` の確認に向く。

- 最新messageを1つだけ描画する。
- `header.frame_id` とRViz `Fixed Frame` が異なる場合は通常TFが必要。
- `MessageFilterDisplay` 由来の `Topic`、`Queue Size`、`Unreliable` を持つ。
- Topicの既定値は空である。

### Probabilistic TF

複数edgeの最新集合をProbTF treeとして見たい場合に使う。

- incremental dynamic edgeを1件ずつ受け取れる。
- complete dynamic snapshotとcomplete static setも併用できる。
- 各incoming edgeのchild frameについて1つの確率点群を描く。
- 最上位rootはanchorなので、root自身にincoming edgeがなければ点群を描かない。
- childから選択rootまでのedgeをsampleごとに合成する。

AprilTagデモのように、1 topicへタグごとのdynamic edgeだけが流れる場合は
**Probabilistic TF** を使う。

## 3. 点群が表しているもの

点群はsampleされたframe原点ではなく、1個の同時変換sample
\((R,t)\) ごとに、正のX/Y/Z軸端点

$$
p_x=t+R(Le_x),\qquad
p_y=t+R(Le_y),\qquad
p_z=t+R(Le_z)
$$

を描く。

| 色 | 点 |
|---|---|
| 赤 | 正のX軸端点 |
| 緑 | 正のY軸端点 |
| 青 | 正のZ軸端点 |

ここで \(L\) が `Axis Length` である。1 transform sampleは3 pointsになる。

componentは重みに従って選ばれ、そのcomponentのorientation law
（finite Bingham、Dirac、uniform）とorientation-conditioned translationが
**同じsampleとして**生成される。
したがって、

- 分離したmixture mode
- component内の姿勢分散
- 並進分散
- 回転–並進相関
- 同一sample内の3軸の相関

がmoment Gaussianへ潰されずに見える。

### Axis Lengthは不確かさ倍率ではない

`Axis Length` を大きくすると姿勢誤差のlever armが長くなるため、点群の球・
弧・リングの半径も大きく見える。これは入力分布やcovarianceを拡大している
のではない。

plugin内部の中央representative軸も同じ `Axis Length` を使う。中央軸を表示
するには、

- Pose: `Show Representative`
- TF: `Show Representatives`

を有効にする。

`rviz/TF` や `rviz/MarkerArray` が描く座標軸は別オブジェクトであり、
ProbTFの `Axis Length` には追従しない。

## 4. Probabilistic Pose の設定

| Property | 既定値 | 意味 |
|---|---:|---|
| `Topic` | 空 | 1本の `ProbabilisticTransformStamped` topic |
| `Unreliable` | false | UDP topic transportを優先するか |
| `Queue Size` | 10 | message filterのqueue長 |
| `Frame Timeout` | 15.0 s | dynamic message受信が止まってから消えるまで |
| `Sample Count` | 300 | 描画する同時変換sample数 |
| `Axis Length` | 0.18 m | sample端点とrepresentative軸の共通長 |
| `Point Size` | 0.01 m | endpoint primitiveの直径 |
| `Point Style` | `Spheres` | `Points`、`Squares`、`Spheres` |
| `Alpha` | 0.75 | 点群とrepresentativeの基礎透明度 |
| `Show Representative` | true | 単一代表軸を表示 |
| `Representative Radius` | 0.01 m | 代表軸の太さ |
| `Random Seed` | 7 | 再現可能な見た目のseed |

設定例:

```text
Global Options / Fixed Frame: world
Probabilistic Pose / Topic: /object_pose/probtf
```

messageの `header.frame_id` から `world` への通常TFがmessage時刻に存在すれば
表示される。

## 5. Probabilistic TF のTopic設定

| Property | 既定値 | 入力 |
|---|---|---|
| `Dynamic Topic` | `/probtf` | incremental `ProbabilisticTransformStamped` |
| `Dynamic Snapshot Topic` | `/probtf_batch` | complete latest dynamic `ProbabilisticTransformArray` |
| `Static Topic` | `/probtf_static` | complete latched static `ProbabilisticTransformArray` |
| `Queue Size` | 10 | dynamicとsnapshot subscriberのqueue長 |

空文字を設定したTopicは購読しない。

### incremental dynamic

`Dynamic Topic` はedgeを1件ずつ追加・更新するstreamである。同じ `edge_id` の
最新recordが置き換わる。incremental streamだけでは、publisherから消えた
edgeを集合から削除できないため、期限切れで非表示になっても内部の最新record
自体は残る。

### complete dynamic snapshot

`Dynamic Snapshot Topic` の1 messageは、現在有効なdynamic edgeの完全な集合で
ある。受信するとそれ以前のdynamic集合を全置換し、その後に到着した
incremental updateだけを上書きする。edgeの削除を明示したいpublisherは
complete snapshotを使う。

### complete static set

`Static Topic` も完全な集合である。`is_static=true` のrecordだけを入れ、
通常はlatched publishする。static setは期限切れしない。

## 6. Probabilistic TF の表示設定

| Property | 既定値 | 意味 |
|---|---:|---|
| `Frame Timeout` | 15.0 s | source stampが変化しなくなってから消えるまで |
| `Root Frame` | 空 | ProbTF graph内で合成を開始するancestor |
| `Sample Count` | 80 | child frameごとの同時変換sample数 |
| `Axis Length` | 0.08 m | sample端点とrepresentative軸の共通長 |
| `Point Size` | 0.006 m | endpoint primitiveの直径 |
| `Point Style` | `Spheres` | `Points`、`Squares`、`Spheres` |
| `Alpha` | 0.75 | 点群とrepresentativeの基礎透明度 |
| `Show Representatives` | true | childごとの単一代表軸を表示 |
| `Representative Radius` | 0.004 m | 代表軸の太さ |
| `Random Seed` | 29 | edge samplingの安定seed |

Property上の主な範囲は次の通り。

- `Queue Size`: 1–10,000
- `Frame Timeout`: 1 s以上
- `Sample Count`: 1–100,000
- `Axis Length`、`Point Size`、`Representative Radius`: 0.0001 m以上
- `Alpha`: 0–1
- `Random Seed`: 0以上

`Random Seed` は見た目を再現するための値であり、入力分布を変更しない。

## 7. Root Frame と RViz Fixed Frame

この2つは役割が違う。

| 設定 | 属する系 | 役割 |
|---|---|---|
| `Root Frame` | ProbTF display | ProbTF edgeをどのancestorまで合成するか |
| `Fixed Frame` | RViz Global Options | 最終描画座標系 |

`Root Frame` が空なら、connected componentごとの最上位rootを自動選択する。
非空の場合、そのframeは表示対象childのancestorでなければならない。

ProbTF内でchildからrootまでを確率的に合成した後、rootからRViz
`Fixed Frame` への配置には、両frameが異なる場合だけ通常TFを使う。
したがって一般形は、

```text
ProbTF child --probabilistic edges--> Root Frame
Root Frame   --ordinary TF---------> RViz Fixed Frame
```

前半は常に必要であり、後半は2つのframeが異なる場合だけ必要になる。

`Root Frame` と `Fixed Frame` が同じなら後半はidentityであり、外部TF
publisherは不要である。

例:

```text
Global Options / Fixed Frame: camera_link
Probabilistic TF / Root Frame: camera_optical_frame
```

この場合、`camera_optical_frame -> camera_link` を解決できる通常TFが必要で
ある。解決できない間もsample geometryは保持される。static frame、または
`Frame Timeout` 内のdynamic frameなら、TFが利用可能になった時点で自動復帰
する。dynamic frameが先に期限切れした場合、TFの復旧だけでは再表示されない。
設定を変えない通常運用では、pathのresolved source stampの変化も必要になる。

## 8. representativeの意味

中央座標軸は分布の全モードを示すものではなく、1つの決定論的要約である。

1. messageにrepresentativeが保存されていればそれを使う。
2. なければ、正の重みが最大のcomponentを選ぶ。
3. そのcomponentのorientation modeと、その姿勢でのconditional translationを
   使う。

treeでは各edgeのrepresentativeをpath上で合成する。

この軸は一般に、

- mixture mean
- 全componentのmode軸
- path distribution全体のjoint MAP

ではない。点群の幾何学的中心と一致しないことも正常である。

全modeの個別axisやweight labelが必要なら、producer側の診断MarkerArrayを
別Displayで表示する。

## 9. mixtureの片方が見えない場合

native ProbTF displayは正の重みを持つ全componentをsampling対象にするが、
各componentを最低1点ずつ描く保証はしない。以下の \(w\) はraw field値では
なく、正のweight総和で内部正規化したcomponent選択確率である。

重み \(w\) のcomponentが `Sample Count = N` で一度も選ばれない確率は

$$
(1-w)^N
$$

である。低重みmodeが見えない場合は次を確認する。

1. 通常 `/tf` ではなくnative ProbTF displayを見ているか。
2. messageの `components[].weight` が正か。
3. `Sample Count` を増やす。
4. `Random Seed` を変えて有限sampleの偏りを確認する。
5. 2つのmodeが画面上で重なっていないか確認する。

AprilTagデモが
`tf_export_policy=highest_weight_component_mode` で起動する `/tf` bridgeは、
最高重みcomponentのmodeだけを出すため、もう一方の枝を失う。bridgeの別policy
まで常に同じ挙動という意味ではない。

## 10. Axis Lengthと重複軸

`Axis Length` で点群だけが伸び、中央軸が伸びないように見える場合は、見えて
いる中央軸が別Displayの可能性が高い。

確認手順:

1. ProbTF側の `Show Representative(s)` をtrueにする。
2. RViz内の**すべての** `rviz/TF` Displayで `Show Axes` をfalseにする。
3. MarkerArrayで、名前の末尾が `/axes` の各namespaceを無効にする。
4. producerが固定長mode axesをpublishしていないことを確認する。
5. plugin更新後ならRVizを再起動する。

AprilTagデモでProbTF軸だけを使う推奨構成は次である。

```yaml
# prob_artag_detector config
publish_mode_axes: false
```

```text
Probabilistic AprilTags / Show Representatives: true
all rviz/TF displays / Show Axes: false
```

MarkerArrayは、mode weight、tag plane、orientation mode条件付き並進楕円体だけを
見る用途なら有効にしてよい。`publish_mode_axes: true` で復活するmode axisは
`marker_axis_length_m` の固定長であり、ProbTFの `Axis Length` には追従しない。

## 11. Frame Timeout

dynamic dataだけに `Frame Timeout` を適用する。

- 最初の \(2/3\): 指定 `Alpha` のまま表示
- 最後の \(1/3\): 線形fade
- timeout超過: 点群とrepresentativeを同時に非表示

時間基準はROS timeである。

TF displayでは、path内の最古dynamic source stampが変化したときだけfreshnessを
更新する。新しく受信した `/probtf_batch` に同じ古いrecordが入っていても
寿命は延びない。

Pose displayでは、dynamic messageを受信するたびにfreshnessを更新する。
同じsource stampの再送でも寿命が延びる。`is_static=true` のPose messageと
static tree recordは期限切れしない。

## 12. 描画負荷

tree displayは1 redrawあたり1,500,000 endpointsを通常範囲のtarget capとする。
child frame数を \(F\)、要求sample数を \(N\) とすると、概ね

$$
N_{\mathrm{effective}}
=
\min\!\left(
N,
\max\!\left(
1,
\left\lfloor
\frac{1{,}500{,}000}{3F}
\right\rfloor
\right)
\right)
$$

となる。clampされた場合は `Sampling` statusに実効値を表示する。
極端に \(F>500{,}000\) なら、各childへ最低1 sampleを残すためtarget capを
超えうる。

Pose displayは `Sample Count` の上限100,000、すなわち最大300,000 endpoints
であり、別の総点数capはない。

重い場合は次の順で軽くする。

1. `Sample Count` を下げる。
2. `Point Style` を `Points` にする。
3. 不要なchild frame、Display、topic購読を減らす。
4. `Point Size` を必要以上に大きくしない。

`Spheres` が最も描画負荷が高い。`Queue Size` を増やしてもsample密度は
増えない。大きいqueueはburstやTF解決待ちを吸収できる一方、memory使用量と
最悪backlog latencyを増やしうる。

## 13. Statusとトラブルシュート

### Add一覧に `probtf_rviz` がない

```bash
cd /home/leus/catkin_ws
catkin build probtf_rviz
source /home/leus/catkin_ws/devel/setup.bash
rospack find probtf_rviz
```

その後RVizを再起動する。別workspaceから起動していないかも確認する。

### `Waiting for ProbTF records`

- Topic名が正しいか。
- Topic typeが一致するか。
- publisherが実際にpublishしているか。

```bash
rostopic list | rg probtf
rostopic info /probtf
rostopic echo -n 1 /probtf
```

AprilTagデモの既定topicはglobal `/probtf` ではなく
`/prob_artag_demo/probtf` である。

### `Message` error

`ProbabilisticPose` 固有のstatusである。`header.frame_id` が空でないかを確認する。

### `Transform` error

`ProbabilisticPose` のparent、つまり `header.frame_id` からRViz
`Fixed Frame` への通常TFをmessage stamp時刻に解決できるか確認する。

### `ProbTF` error

tree snapshot全体の検証に失敗している。

- parent、child、`edge_id` が空でないか。
- dynamic topicに `is_static=true`、またはstatic topicに
  `is_static=false` のrecordが混ざっていないか。
- 1つのchildに複数の物理parent edgeがないか。
- snapshot待ちまたはinvalid snapshotになっていないか。

### `Frame/<child>` error

特定childのpath構築、sampling、配置に失敗している。

- 指定 `Root Frame` がchildのancestorか。
- ProbTF graphにcycleがないか。
- path上のdistributionとrepresentativeが妥当か。
- resolved source stamp時刻に、rootからRViz `Fixed Frame` への通常TFを
  解決できるか。
- source stampとROS clockが妥当か。

### 数秒後に消える

- `Frame Timeout` を確認する。
- publisherが止まっていないか確認する。
- treeではmessage到着回数ではなくsource stampが変化しているか確認する。
- `/use_sim_time` と `/clock` を確認する。

AprilTagの同梱RViz設定は通常より短い `Frame Timeout: 2 s` を使う場合がある。

### `Sampling` warning

treeの通常target capによるclampである。warning内の実効sample数を確認し、
`Sample Count` または表示frame数を下げる。

### `Distribution` または `Representative` error

- usableな正のcomponent weightが1つ以上あるか。
- weightが有限か。
- quaternionが単位長か。
- Bingham shape/inverse concentrationの規約が正しいか。
- translation residual covarianceがPSDか。
- representativeを保存した場合、そのtransformが有限か。

invalid/zero-mass lawを、決定論的representativeへ黙って置き換えて表示することは
ない。

## 14. 表示上の注意

- このpluginは可視化器であり、表示結果をProbTF推定へfeedbackしない。
- 点の密度はMonte Carlo sample密度であり、正規化済みpdf値そのものではない。
- `Random Seed` や `Sample Count` で見た目は変わるが、入力分布は変わらない。
- representativeは単一要約であり、点群の平均とは限らない。
- ordinary TFとMarkerArrayは確率情報を欠落させる場合がある。
