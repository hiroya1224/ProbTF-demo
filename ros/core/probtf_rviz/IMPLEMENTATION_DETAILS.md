# probtf_rviz 実装詳細

## 1. この文書の目的

この文書は、`probtf_rviz` の保守、拡張、reviewを行う開発者向けに、
現在の実装が

- ProbTF messageをどこで受信するか
- mixtureからどのように同時変換sampleを生成するか
- ProbTF treeをどのように構成するか
- sampleをどの座標系で描画するか
- RVizのproperty変更をどこまで再計算へ反映するか
- subscriber callback経路とRViz GUI/Ogre経路の境界をどう守るか
- timeout、error、計算量をどう扱うか

を、source codeを順に読まなくても追える粒度で説明するものである。

利用者向けの設定手順とトラブルシュートは [README.md](README.md) を参照する。
ProbTF wire formatそのものの完全な仕様ではなく、RViz pluginと、それが直接使う
`probtf_core` C++ sampling APIの実装解説である。

## 2. 実装を読むときの最重要不変条件

最初に、plugin全体で守っている不変条件を列挙する。

1. 1本のProbTF recordは、child座標をparent座標へ写す変換分布である。
2. 1つの描画sampleではorientationを先にsampleし、そのorientationに条件づけて
   translationを生成する。表示側で別々のsample列を作って組み替えない。
3. mixture component、orientation、orientation-conditioned translationを
   まとめて1つの \((R,t)\) としてsampleする。
4. X/Y/Zの3 endpointは、同じ \((R,t)\) から作る。
5. tree上では、同じsample index同士をpathに沿って合成する。
6. 同じedgeを共有する複数childは、そのedgeの同じsample列を再利用する。
7. stochastic edgeを逆向きにsampleして表示しない。
8. 確率geometryはProbTF rootまたはPose parentの局所座標で保持し、最後に
   通常TFでRViz `Fixed Frame` へ配置する。
9. ROS subscriber callbackではOgre objectを変更しない。
10. invalidな確率分布をrepresentativeへ黙って置き換えない。

これらのうち2–6が、mixtureや回転–並進相関を点群へ残すための中心である。

## 3. 座標変換の規約

`ProbabilisticTransformStamped` の `header.frame_id` をparent \(P\)、
`child_frame_id` をchild \(C\) とする。1 sample \((R_{PC},t_{PC})\) は

$$
p_P = R_{PC}p_C+t_{PC}
$$

としてchild上の点をparentへ写す。

この向きを、以下では

$$
T_{PC}=(R_{PC},t_{PC})
$$

と書く。通常TFと同じく、messageは「child pose expressed in parent」を表す。

### 3.1 描画する点

frame原点そのものではなく、child frameの正の3軸端点を写した

$$
\begin{aligned}
p_x &= t+R(Le_x),\\
p_y &= t+R(Le_y),\\
p_z &= t+R(Le_z)
\end{aligned}
$$

を描く。\(L\) はRViz property `Axis Length` である。

この選択には次の意味がある。

- 並進の不確かさは3色すべての移動として現れる。
- 回転の不確かさは、原点から距離 \(L\) のlever armによって弧や球として現れる。
- 同一sampleの3軸は、同じ回転と並進を共有する。
- 1 transform sampleあたりOgre pointは3個になる。

色は `axisColor()` で次のRGBを与える。point生成時のalphaは1であり、
後からDisplay全体のalphaとfreshness fadeを適用する。

| 軸 | 色 | RGB |
|---|---|---|
| X | 赤 | `(1.0, 0.18, 0.18)` |
| Y | 緑 | `(0.18, 1.0, 0.18)` |
| Z | 青 | `(0.18, 0.32, 1.0)` |

## 4. packageとplugin登録

### 4.1 生成物

[CMakeLists.txt](CMakeLists.txt) は共有library `probtf_rviz` を生成する。
install後のplugin XMLから見えるpathは

```text
lib/libprobtf_rviz
```

である。

主なbuild依存は次である。

| 依存 | 用途 |
|---|---|
| `pluginlib` | RViz Display classの発見とload |
| `probtf_core` | native transform sampling、representative、snapshot metadata |
| `probtf_msgs` | ProbTF v2 ROS message |
| `roscpp` | subscriberとROS time |
| `rviz` | Display、property、FrameManager、PointCloud、Axes |
| `Eigen3` | quaternion、rotation、translation、行列 |
| `Qt5::Widgets` | RViz propertyとsignal/slot |
| Ogre | RViz経由のscene nodeと描画object |

`CMAKE_AUTOMOC` を有効にし、`Q_OBJECT` を持つ2つのDisplay headerを
library sourceへ明示的に含めている。`QT_NO_KEYWORDS` も有効なので、
headerでは `slots` ではなく `Q_SLOTS` を使う。

### 4.2 pluginlibへの公開

[plugin_description.xml](plugin_description.xml) が公開するclass IDは次の2つである。

| class ID | C++ class | base |
|---|---|---|
| `probtf_rviz/ProbabilisticPose` | `probtf_rviz::ProbabilisticPoseDisplay` | `rviz::Display` |
| `probtf_rviz/ProbabilisticTF` | `probtf_rviz::ProbabilisticTfDisplay` | `rviz::Display` |

実際のexport macroは
[plugin_exports.cpp](src/plugin_exports.cpp) にある。
さらに [package.xml](package.xml) の

```xml
<rviz plugin="${prefix}/plugin_description.xml"/>
```

によってRVizがXMLを発見する。

### 4.3 2つのDisplayでbase classが異なる理由

`ProbabilisticPoseDisplay` は

```text
rviz::MessageFilterDisplay<
    probtf_msgs::ProbabilisticTransformStamped>
```

を継承する。1 messageのparent frameを通常TFでfilterするという、
RViz標準の単一message displayの流れを利用できるためである。

`ProbabilisticTfDisplay` は `rviz::Display` を直接継承する。理由は、

- message型が単一recordとarrayの2種類ある
- dynamic incremental、dynamic snapshot、static setの3入力を統合する
- 1 messageではなくlatest tree全体を描く
- childごとに異なるresolved stampとrootを持つ

ためである。

## 5. source構成と責務

| file | 主な責務 |
|---|---|
| [probabilistic_pose_display.hpp](include/probtf_rviz/probabilistic_pose_display.hpp) | 単一Pose Displayの状態とproperty |
| [probabilistic_pose_display.cpp](src/probabilistic_pose_display.cpp) | 単一messageのsampling、配置、status |
| [probabilistic_tf_display.hpp](include/probtf_rviz/probabilistic_tf_display.hpp) | tree Displayの公開classとproperty |
| [probabilistic_tf_display.cpp](src/probabilistic_tf_display.cpp) | 3 topicの統合、tree/path、sampling cache、visual管理 |
| [transform_visual.hpp](include/probtf_rviz/transform_visual.hpp) | 描画dataとstyleの共通interface |
| [transform_visual.cpp](src/transform_visual.cpp) | Ogre scene node、PointCloud、Axesの所有 |
| [frame_freshness.hpp](include/probtf_rviz/frame_freshness.hpp) | ROS timeに基づく表示寿命とfade |
| [sampling.hpp](../probtf_core/include/probtf_core/sampling.hpp) | C++ sampling API |
| [sampling.cpp](../probtf_core/src/sampling.cpp) | mixture、Bingham、conditional translation、path合成 |
| [latest_snapshot.hpp](../probtf_core/include/probtf_core/latest_snapshot.hpp) | latest tree query API |
| [latest_snapshot.cpp](../probtf_core/src/latest_snapshot.cpp) | topology、dependency、resolved stamp検証 |

## 6. 全体data flow

Pose Displayは次の短い流れを持つ。

```text
ProbabilisticTransformStamped
        |
        v
MessageFilterDisplay / processMessage()
        |
        +--> parent -> RViz Fixed Frame の通常TF
        |
        +--> native mixture sampling
        |
        +--> RGB axis endpoints + representative
        |
        v
TransformVisual
```

TF Displayは、callbackと描画の実行経路を分けた次の流れになる。

```text
dynamic record ----+
dynamic snapshot --+--> callback側 pending buffer
static set --------+          |
                              | mutex下でswap
                              v
                     RViz GUI/update()側
                              |
                       active latest sets
                              |
                       LatestSnapshot検証
                              |
                    child -> selected root path
                              |
                  edge sampling cache + path合成
                              |
                 root-local cloud/representative
                              |
                 root -> RViz Fixed Frame 通常TF
                              |
                              v
                        TransformVisual
```

重要なのは、subscriber callbackがpending bufferへmessage pointerを置くだけで、
snapshot構築、sampling、Ogre更新はRVizの `update()` 側で行う点である。

## 7. 共通描画object `TransformVisual`

### 7.1 所有関係

1つの `TransformVisual` は次を所有する。

```text
Display scene_node_
    |
    +-- frame_node_
          |
          +-- rviz::PointCloud
          |
          +-- rviz::Axes
```

`frame_node_` はDisplayの `scene_node_` のchildとして作る。
PointCloud内の座標とrepresentative transformは、Poseではmessage parent座標、
TFでは選択したProbTF root座標で保存する。

その局所座標からRViz `Fixed Frame` への通常TFは、
`frame_node_` のpositionとorientationへ設定する。この分離により、
Fixed Frame変換を各pointへ焼き込む必要がない。TF Displayは毎update、同じ
resolved stampでroot-to-Fixedを再照会し、ProbTF geometryをresampleせずに
scene nodeだけを更新または復帰できる。Pose Displayがframe poseを解決するのは
messageまたはgeometry propertyによるrender時であり、freshnessだけを処理する
通常updateでは再解決しない。

### 7.2 PointCloud

`setPoints()` は `ColoredPoint` のEigen座標を
`rviz::PointCloud::Point` へ変換する。

更新時は

1. 既存cloudを `clear()`
2. 空でなければ `addPoints()`

する。render modeは `Points`、`Squares`、`Spheres` のいずれかである。
`Point Size` は3軸とも同じprimitive直径へ設定する。

### 7.3 representative axes

`setRepresentative()` は `Eigen::Isometry3d` のtranslationとquaternionを
`rviz::Axes` のpositionとorientationへ設定する。

representativeの軸長と半径は、Axes constructorの初期値へ固定されていない。
`setStyle()` のたびに

```text
axes_->set(axis_length, axis_radius, alpha)
```

を呼ぶ。このため `Axis Length` は

- sample endpointの \(L\)
- 中央representative axesの長さ

の両方へ反映される。

別の `rviz/TF` DisplayやMarkerArrayが持つAxesには、このstyleは届かない。

### 7.4 alpha、fade、visibility

最終alphaは

$$
\alpha_{\mathrm{draw}}
=
\alpha_{\mathrm{property}}\,
\alpha_{\mathrm{freshness}}
$$

である。

`setStyle()` はPointCloudのalphaとAxes全体のgeometry/styleを更新する。
`setFade()` はfreshness係数だけを更新し、cloudのsample座標は作り直さない。
`setVisible()` はcloudとAxesを同時に制御するが、Axesについてはさらに
`show_representative` がtrueであることを要求する。

## 8. native transform lawのsampling

RViz pluginは分布を独自にGaussian近似しない。
[sampling.cpp](../probtf_core/src/sampling.cpp) の
`sampleTransformDistribution()` を呼ぶ。

wire fieldは次を参照する。

- [ProbabilisticTransformStamped.msg](../probtf_msgs/msg/ProbabilisticTransformStamped.msg)
- [ProbabilisticTransformComponent.msg](../probtf_msgs/msg/ProbabilisticTransformComponent.msg)
- [BinghamOrientation.msg](../probtf_msgs/msg/BinghamOrientation.msg)
- [ConditionalGaussianTranslation.msg](../probtf_msgs/msg/ConditionalGaussianTranslation.msg)

### 8.1 描画時に参照するwire field

現在の表示経路がfieldをどう使うかをまとめる。

| field | Pose Display | TF Display |
|---|---|---|
| `header.frame_id` | geometryのparent | treeのparent |
| `header.stamp` | parent-to-Fixedの解決時刻 | dynamic pathのresolved stamp |
| `child_frame_id` | sampling自体には不使用 | visual keyとtree topology |
| `edge_id` | sampling自体には不使用 | record identity、seed、path provenance |
| `is_static` | timeout無効化 | topic整合性とpath freshness |
| `representative_kind`, `representative` | 中央軸 | edge代表をpath合成 |
| `components` | native sampling | edgeごとのnative sampling |
| record/component `derived_from_edge_ids` | 不使用 | path上のlatent重複検出 |

`authority`、`component_id`、`ApproximationInfo`、provenanceの
`source_ids`、`method`、`detail`、array自身のheaderは、現在の描画geometry、
色、freshnessには使わない。producerの診断・監査情報としてwireには残るが、
RViz pluginがlabelとして表示するわけではない。

### 8.2 mixture component選択

component weightを \(w_k\) とする。

1. 非有限weightが1つでもあればrecord全体をrejectする。
2. \(w_k\le 0\) のcomponentはmixture massを持たない。
3. 正のweightが1つもなければrejectする。
4. overflowを避けるため、まず
   \(s=\max_k\max(0,w_k)\) で割る。
5. `std::discrete_distribution` が正規化してcomponentを選ぶ。

したがって実際の選択確率は

$$
\pi_k
=
\frac{\max(0,w_k)}
{\sum_j\max(0,w_j)}
$$

であり、weightを共通の正定数倍しても変わらない。

zero/negative-weight componentは選択されないが、message全体の数値健全性を
保つため、そのorientationとtranslation field自体は検証される。

### 8.3 quaternionとBingham表現

quaternionは内部で

$$
q=(w,x,y,z)^\mathsf{T}\in S^3
$$

として扱う。ROS message fieldの並びはxyzwだが、
`Eigen::Quaterniond` とshape packingの意味はwxyzである。

reference quaternionには有限かつ

$$
\left|\lVert q\rVert-1\right|\le 10^{-8}
$$

を要求し、検証後に再正規化する。

Bingham shape \(A\in\mathbb{R}^{4\times4}\) は対称、trace zeroであり、
upper triangle 10要素から復元する。

orientation lawは3種類ある。

#### Dirac

常にreference quaternionを返す。`inverse_concentration == 0` と、
shapeがreference quaternionに対応する

$$
A=2qq^\mathsf{T}-\frac{1}{2}I_4
$$

に一致することを要求する。

#### Uniform

4次元標準正規

$$
g\sim\mathcal{N}(0,I_4)
$$

を生成し、

$$
q=\frac{g}{\lVert g\rVert}
$$

とする。これは \(S^3\) 上の一様sampleになる。

wire上では、正の無限大の `inverse_concentration` とzero shapeを要求する。
reference quaternionはorientationの優先方向ではないが、
representative fallbackには利用される。

#### Finite Bingham

shapeの固有値を昇順に

$$
\lambda_0\le\lambda_1\le\lambda_2\le\lambda_3
$$

とする。JMAA正規化条件として

$$
\lambda_3+\lambda_2=1
$$

を数値tolerance内で要求する。

`inverse_concentration` を \(\tau>0\) とすると、proposal構築用precisionは

$$
d_i=\frac{\lambda_3-\lambda_i}{\tau}
$$

である。小さい負値はzeroへclampするが、toleranceを超える負値や非有限値は
rejectする。

proposal scale \(b\) は

$$
-1+\sum_{i=0}^{3}\frac{1}{b+2d_i}=0
$$

を100回の二分法で解く。固有basis上で標準偏差

$$
\sigma_i=\frac{1}{\sqrt{1+2d_i/b}}
$$

のGaussianを生成し、単位球面へ正規化したcandidateをrejection samplingする。

1 orientation sampleにつきproposalは最大4096回である。上限までacceptできない
場合、分布全体のsamplingをerrorにする。

### 8.4 orientation-conditioned translation

rotation matrixをcolumn-majorで

$$
r(R)=\operatorname{vec}(R)\in\mathbb{R}^{9}
$$

と並べる。componentが持つ

- reference quaternion \(q_{\mathrm{ref}}\)
- reference時の平均 \(\mu_{\mathrm{ref}}\in\mathbb{R}^3\)
- coupling \(C\in\mathbb{R}^{3\times9}\)
- residual covariance \(\Sigma\in\mathbb{R}^{3\times3}\)

に対し、sampled rotation \(R\) の条件付き並進は

$$
t
=
\mu_{\mathrm{ref}}
+C\left(r(R)-r(R_{\mathrm{ref}})\right)
+L_\Sigma\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I_3)
$$

である。

\(L_\Sigma\) はCholesky factorに限定せず、対称固有分解によるPSD square rootで

$$
L_\Sigma L_\Sigma^\mathsf{T}=\Sigma
$$

を満たす。小さい負固有値はzeroへclampし、`-1e-10` 未満ならrejectする。

orientationをsampleした後に、その同じorientationから条件付き平均を計算する
ため、回転–並進couplingは保持される。

### 8.5 outputを途中まで書き換えない

sampling中はlocal vectorへ結果を蓄積する。全sampleが成功した後にだけ
callerのoutputへmoveする。途中でinvalid componentやBingham proposal failureが
起きても、callerが持っていた既存outputは変更しない。

## 9. representative transform

`representativeTransform()` は点群とは別に、中央の単一座標軸を得るAPIである。

### 9.1 保存済みrepresentative

`representative_kind` が `REPRESENTATIVE_NONE` 以外なら、messageの
`representative` を検証し、quaternionを再正規化して使う。

- enum値が既知範囲内
- quaternionが有限かつ単位長
- translationが有限

であることを検証する。

### 9.2 fallback

保存済みrepresentativeがない場合は、

1. 正のraw weightが最大のcomponentを選ぶ
2. そのcomponentのorientation modeを求める
3. そのrotationでconditional translationの平均を評価する

というfallbackを行う。

finite Binghamのmodeは最大固有値に対応する固有vectorである。
quaternionの符号を決定的にするため、絶対値最大の要素が負なら全体を反転する。
Diracとuniformではreference quaternionを返す。

fallback translationは

$$
t_{\mathrm{rep}}
=
\mu_{\mathrm{ref}}
+C\left(r(R_{\mathrm{mode}})-r(R_{\mathrm{ref}})\right)
$$

であり、residual noiseはsampleしない。

これはmixture meanや全分布のjoint MAPであるとは限らない。

## 10. Probabilistic Pose Display

### 10.1 保持する状態

主な状態は次の4つである。

| 状態 | 意味 |
|---|---|
| `visual_` | 1つのPointCloudとrepresentative Axes |
| `latest_message_` | geometry property変更時に再描画する最新message |
| `freshness_` | dynamic messageの受信進捗 |
| `message_renderable_` | 最後のrenderが成功したか |

### 10.2 message処理

`processMessage()` の流れは次である。

1. `latest_message_` を更新する。
2. staticならfreshnessをresetし、dynamicなら現在ROS timeで
   `markProgress()` する。
3. `renderMessage()` を呼ぶ。

`renderMessage()` は次を順に行う。

1. `header.frame_id` の非空検証
2. message headerのframe/stampからRViz `Fixed Frame` への通常TF解決
3. `Random Seed` だけで `std::mt19937` を毎回新規構築
4. native lawから `Sample Count` 個をsample
5. `Axis Length` を使って3 endpoint/sampleを作成
6. representativeを取得
7. `TransformVisual` へframe pose、points、representativeを反映
8. freshnessに応じてfadeとvisibilityを反映

generatorをrenderごとに同じseedから作り直すため、messageとpropertyが同じなら
sample列は安定する。`Axis Length` だけを変更したときにランダムなちらつきが
加わらない。

### 10.3 frameの二段階配置

sample geometryはmessage parent frame内にある。
`FrameManager::getTransform(message->header, ...)` で得た
parent-to-Fixed transformを `frame_node_` へ設定する。

概念的には、

$$
p_{\mathrm{fixed}}
=
R_{\mathrm{fixed},P}\,p_P+t_{\mathrm{fixed},P}
$$

をOgre scene graphが全pointsへ適用する。

### 10.4 property callback

propertyは再samplingの有無で分かれる。

| property | slot | resample |
|---|---|---|
| `Frame Timeout` | `updateAppearance()` | しない |
| `Point Size` | `updateAppearance()` | しない |
| `Point Style` | `updateAppearance()` | しない |
| `Alpha` | `updateAppearance()` | しない |
| `Show Representative` | `updateAppearance()` | しない |
| `Representative Radius` | `updateAppearance()` | しない |
| `Sample Count` | `updateGeometry()` | する |
| `Axis Length` | `updateGeometry()` | する |
| `Random Seed` | `updateGeometry()` | する |

`updateGeometry()` はまずstyleを更新し、その後 `latest_message_` があれば
再renderする。したがって `Axis Length` はAxesの長さだけでなく、endpoint式の
\(L\) にも即座に入る。

Topic、Queue Size、Unreliable transportは `MessageFilterDisplay` が提供する。

### 10.5 reset

`reset()` はbase classをresetした後、

- latest messageを破棄
- freshnessを未初期化へ戻す
- renderable flagをfalse
- pointsを空にする
- visualを非表示
- `Freshness` statusを削除

する。

## 11. Probabilistic TF Displayの入力統合

### 11.1 3 subscriber

`ProbabilisticTfDisplay::Impl` が次を購読する。

| 入力 | 型 | queue | 意味 |
|---|---|---:|---|
| Dynamic Topic | `ProbabilisticTransformStamped` | `Queue Size` | incremental edge |
| Dynamic Snapshot Topic | `ProbabilisticTransformArray` | `Queue Size` | complete dynamic set |
| Static Topic | `ProbabilisticTransformArray` | 1固定 | complete static set |

空topicは購読しない。topic propertyまたはqueue変更時は、

1. 既存subscriberをshutdown
2. pending、active、visual、`Frame/*` と `Sampling` statusをclear
3. Displayがenableなら再subscribe

する。

このresetにはsubscription generation IDはない。`shutdown()` とmutexで通常の
data raceは防ぐが、すでにin-flightの旧subscriber callbackをepochで識別して
破棄する仕組みではない。旧topicとの厳密なgeneration isolationが必要になった
場合は、pending itemへsubscriber generationを持たせる必要がある。

### 11.2 subscriber callback側で行うこと

callbackは確率計算も描画も行わない。mutexを取り、

- global callback sequenceを1増やす
- message pointerとsequenceをpending領域へ保存する

だけである。

pending領域は次の形を持つ。

```text
pending_dynamic_[record_key] = latest PendingRecord
pending_dynamic_batch_       = latest PendingArray
pending_static_              = latest PendingArray
```

同じedgeのincremental updateがRViz frame間に複数回来た場合は、最後の1件だけが
残る。snapshot/static arrayも最後の1件だけを残す。これは表示が
latest-only semanticsであり、全履歴を再生するものではないためである。

### 11.3 record key

pending/active mapのkeyは、`edge_id` が非空ならそれを使う。
空なら一時的に

```text
clean(parent) + "->" + clean(child)
```

を使う。

ただしこれはcallback coalescing用の防御的fallbackにすぎない。
後段の `LatestSnapshot::valid()` は空の `edge_id` をrejectするため、
空ID recordが正常表示されるわけではない。

同じdynamic map内または同じstatic map内で `edge_id` が重複すると、
map登録時に後のrecordが前のrecordを置き換える。static mapとdynamic mapの間では
別々に保持される。物理edge identityとseedを曖昧にしないため、producer側では
全体で一意な `edge_id` を与える必要がある。

`cleanFrame()` はframe文字列の先頭と末尾にある `/` と空白文字を除去する。
文字列内部の文字は変更しない。

### 11.4 RViz GUI/update側への受け渡し

`update()` は最初に `consumePending()` を呼ぶ。
ここでmutex下のpending containerをlocal変数へ `swap()` し、その後はlock外で
active setを更新する。

dynamic snapshotがある場合、

1. active dynamic setを全消去
2. snapshot全recordをkey別に登録
3. snapshot callback sequenceを `batch_sequence_` として保存

する。

同じupdate cycleにincremental recordもある場合、sequenceが

$$
s_{\mathrm{incremental}}\ge s_{\mathrm{batch}}
$$

のrecordだけをsnapshot後に上書きする。つまり、

- snapshotより前に届いたincrementalはsnapshotに含まれたものとして捨てる
- snapshotより後に届いたincrementalは最新差分として残す

という順序になる。

static arrayはcomplete setなので、受理した最新arrayでactive static setを
全置換する。

active setが変われば `dirty_` を立てる。geometryはdirtyなupdateだけで
再構築し、root-to-Fixed poseとfreshnessは毎update確認する。

### 11.5 callback経路とGUI経路の境界

通常のcallback共有stateとしてmutexで保護するのは、

- pending message
- callback sequence
- `dirty_` flag

である。`clear()` だけは、in-flight callbackとのreset競合を直列化するため、
同じmutexを保持したままactive records、visuals、frame bindings、
geometry errors、refresh flagも消去する。

一方、

- active record map
- `LatestSnapshot`
- sample vector
- `TransformVisual`
- Ogre scene node
- RViz status

はsubscriber callbackから変更しない。通常はRVizの `update()` 経路から扱い、
styleだけはQt property slotの `updateAppearance()` から既存visualへ反映する。
この境界を崩してsubscriber callback内から `visuals_` を触る拡張は
行ってはならない。実際のOS thread構成はRViz側のspinnerとQt dispatchにも
依存するため、ここで保証しているのは実装経路の分離である。

## 12. latest snapshotの検証

active dynamic/static mapをそれぞれarrayへ戻し、
`probtf_core::LatestSnapshot` を構築する。

### 12.1 record単位の検証

static setを先、dynamic setを後に登録する。各recordについて、

1. clean後のparentが非空
2. clean後のchildが非空
3. `edge_id` が非空
4. static setでは `is_static == true`
5. dynamic setでは `is_static == false`
6. 1 childに物理parent edgeが1本だけ

を要求する。

同じchildがstaticとdynamicの両方にある場合も「複数parent edge」として
snapshot全体をrejectする。dynamicがstaticを暗黙に上書きする仕様ではない。

### 12.2 cycle検出の位置

`LatestSnapshot::valid()` はrecord局所の整合性を検証するが、snapshot全体を
巡回してcycleを先に探すわけではない。

cycleは、

- Displayの `buildRootPath()`
- `LatestSnapshot::buildPath()`

がchildからancestorをたどるとき、visited setへの再訪として検出する。
そのためcycle errorは通常、snapshot全体の `ProbTF` validationではなく、
対象childの `Frame/<child>` geometry errorになる。

## 13. childからrootへのpath

### 13.1 表示対象

incoming ProbTF edgeを持つ各childが表示対象である。
最上位rootはincoming edgeがなければ表示対象に含まれない。

static recordとdynamic recordから

```text
by_child[child] = incoming edge
```

を作り、childごとにpathを構築する。

### 13.2 rootの選択

`Root Frame` が空なら、childからparentをたどり、incoming edgeがなくなった
frameをそのconnected componentのrootにする。

`Root Frame` が非空なら、たどる途中でそのframeへ到達する必要がある。
到達前にedgeがなくなれば、そのrootは対象childのancestorではないためerrorに
する。

Displayはこの事前処理によって「rootはchildのancestor」という条件を保証して
から

```text
lookupPathMetadata(root, child)
```

を呼ぶ。したがって表示用pathは常にforward edgeだけであり、
stochastic inverseを必要としない。

明示rootが表示対象child自身と同じならpathは空になる。この場合は
`Sample Count` 個のidentity transformを作り、そのchild/root原点のAxesを
通常TFで配置する。path外側のincoming edgeがdynamicでも
`has_dynamic == false` となるため、このidentity表示にはFrame Timeoutを
適用しない。

### 13.3 path metadata

`lookupPathMetadata()` は分布momentを評価せず、

- target/root
- source/child
- pathのedge ID列
- resolved stamp
- provenance dependencyの安全性

だけを検証する。

resolved stampはpath内のdynamic edge stampの最小値、すなわち最古時刻である。

$$
t_{\mathrm{resolved}}
=
\min_{e\in\mathcal{P},\,e\ \mathrm{dynamic}} t_e
$$

all-static pathならzero timeを返す。
このzero timeを通常TFの `FrameManager` へ渡すため、root-to-Fixed配置は
通常TFのlatest transformとして解決される。

### 13.4 latent dependency検証

各edgeについて、

- 自身の `edge_id`
- record provenanceの `derived_from_edge_ids`
- 各component provenanceの `derived_from_edge_ids`

をdependency集合へ入れる。

同じdependency IDがpath上で重複し、かつpathの全edgeがdeterministicではない
場合、独立edge samplingではlatent variableを二重計上する可能性があるため
rejectする。

現在のRViz evaluatorはdependency-aware joint samplerを持たない。
このcheckは相関を無視してもっともらしい点群を出すことを防ぐための
fail-closed動作である。

## 14. tree edge sampling

### 14.1 edgeごとの決定的seed

tree redrawごとに、各edgeを高々1回sampleする。seedは

- user property `Random Seed`
- record keyの64-bit FNV-1a-style stable hash下位32 bit
- 同hash上位32 bit

を `std::seed_seq` へ与えて作る。

概念的には

$$
\mathrm{seed}_e
=
f(\mathrm{user\ seed},\operatorname{stableHash}(\mathrm{recordKey}_e))
$$

である。

mapのiteration順や、別edgeの追加・削除で既存edgeの乱数列がずれにくい。
同じ実装環境、record key、分布、sample count、user seedなら、redraw間で
同じsample列を得る。C++標準libraryやEigen実装をまたぐbitwise再現性までを
wire contractとして保証するものではない。

### 14.2 edge cache

1 redraw内では

```text
sampled_edges[Record*] = TransformSampleVector
```

としてcacheする。

複数child pathが同じupstream edge pointerを通る場合、全childがその同じ
sample vectorを参照する。これにより共有edgeの揺れ方も共有される。

別edgeは別seedのgeneratorでsampleするため、provenance checkで禁止されない
範囲では互いに独立なdrawとして扱う。

### 14.3 sample indexを揃えた合成

childからrootへのpathを

$$
T_1,T_2,\ldots,T_m
$$

とする。\(T_1\) はchildに最も近いedge、\(T_m\) はrootに最も近いedgeである。

各edgeから同数 \(N\) のsampleを得て、index \(i\) ごとに

$$
T_{\mathrm{root},\mathrm{child}}^{(i)}
=
T_m^{(i)}\cdots T_2^{(i)}T_1^{(i)}
$$

を作る。

実装はidentityからchild-to-root順にedgeを左から掛ける。
現在の合成値 \((R,t)\) と次edge \((R_e,t_e)\) に対して、

$$
\begin{aligned}
t &\leftarrow R_e t+t_e,\\
R &\leftarrow R_e R
\end{aligned}
$$

である。

全edge vectorの長さは同一でなければならない。各入力sampleのrotationは
有限・単位長であることを確認し、積の後にはquaternionを再正規化する。

sample indexを無関係にshuffleしたり、軸ごとに別sampleを使ったりしないため、
1 path sampleとしての整合性が残る。

## 15. tree representativeの合成

各edgeで `representativeTransform()` を呼び、root側のedgeから順に右へ掛ける。
child-to-root pathが \([T_1,\ldots,T_m]\) なら

$$
T_{\mathrm{rep}}
=
T_{m,\mathrm{rep}}\cdots T_{1,\mathrm{rep}}
$$

である。

これは「各edgeの単一representativeを合成した値」である。
path distribution全体をsampleしてから平均したものでも、
path全体のjoint modeを最適化したものでもない。

`Show Representative(s)` はAxesのvisibilityだけを変える。falseであっても
representativeの検証と合成はgeometry構築時に実行するため、invalidな
representativeは点群だけの表示にもfailureを伝播する。

## 16. tree geometryとRViz Fixed Frame

### 16.1 root-local geometry

合成済みsampleとrepresentativeは、選択root座標にある。
endpointを作って `TransformVisual` へ保存した後、childごとに
`FrameBinding` を作る。

`FrameBinding` は次を持つ。

| field | 意味 |
|---|---|
| `root` | geometryの局所座標frame |
| `path_signature` | pathを構成するrecord key列 |
| `stamp` | pathのresolved stamp |
| `has_dynamic` | pathにdynamic edgeが1本以上あるか |
| `freshness` | source stamp進捗の状態 |

### 16.2 通常TFの更新

geometryがdirtyでないupdateでも、全visualについて

```text
FrameManager::getTransform(root, resolved_stamp, ...)
```

を呼ぶ。成功したposition/orientationを `frame_node_` へ設定する。

したがって、

- ProbTF distributionが変わらなくても通常TFに追従する
- `Fixed Frame` を変えても確率分布のresampleは不要
- rootとFixedが同じならidentity配置になる

という動作になる。

root-to-Fixed transformを解決できない場合、visualを削除せず非表示にする。
static frame、またはfreshnessがvisibleなdynamic frameなら、後続updateでTFを
解決できた時点で保存済みgeometryを再表示する。TF欠落中にdynamic frameが
期限切れした場合、TFの復旧だけでは再表示されない。設定を変えない通常運用では、
freshnessが観測するresolved stampの変化が必要になる。

### 16.3 geometry failureとの違い

path構築、metadata検証、edge sampling、sample-path composition、
representativeのいずれかに失敗したchildは、そのredrawの
`successful_children` に入らない。
redraw末尾で、そのchildの古いvisualとbindingを消去する。

つまり、

| failure | old geometry |
|---|---|
| root-to-Fixed通常TFが一時的にない | 保持して非表示 |
| distribution/path/composition/representativeがinvalid | 消去 |
| snapshot全体がinvalid | 全visualを消去 |
| dynamic/static setがともに空 | 全visualを消去し、record待ちへ移行 |

となる。

## 17. freshnessとFrame Timeout

[frame_freshness.hpp](include/probtf_rviz/frame_freshness.hpp) の
`StampFreshness` が、表示寿命をROS timeで管理する。

### 17.1 状態遷移

最後に進捗があったROS timeを \(t_p\)、現在時刻を \(t\)、timeoutを \(T\) とし、

$$
a=\max(0,t-t_p)
$$

をageとする。

visibilityは

$$
\mathrm{visible}=(a\le T)
$$

である。fade係数は

$$
\alpha_f(a)
=
\begin{cases}
1,
& 0\le a\le \frac{2T}{3},\\[4pt]
\max\left(0,\dfrac{T-a}{T/3}\right),
& a>\frac{2T}{3}.
\end{cases}
$$

従って \(a=T\) では「visibleだがalpha zero」、\(a>T\) で非表示になる。

未初期化、非有限timeout、または \(T\le0\) はvisible/alpha 1を返す。
RViz property自体には1秒以上の下限がある。

### 17.2 source stamp観測

`observe(source_stamp, now)` は、初回またはsource stampが前回値と異なる場合だけ
`last_progress = now` とする。

大小比較ではなく不一致比較なので、stampが前後どちらへ変化しても進捗として
扱う。同じstampを持つsnapshotを何度受信しても寿命は延びない。

### 17.3 ROS clock rewind

`/use_sim_time` のbag loopなどで現在ROS timeが前回観測より小さくなった場合、
初期化済みfreshnessの `last_progress` を新しい現在時刻へrebaseする。
負ageによって無期限に残ることを防ぎ、rewind直後から新しいtimeoutを数える。

### 17.4 Pose Display

dynamic Poseはmessageを受信するたびに `markProgress(now)` する。
したがって同じheader stampの再送でも寿命が延びる。

static Poseはfreshnessをresetして未初期化状態にするため、期限切れしない。

### 17.5 TF Display

tree pathにdynamic edgeがあれば、resolved stampを `observe()` する。
geometry再構築時に

- rootが同じ
- path signatureが同じ
- 前回もdynamic path

なら、前回のfreshness objectを引き継いでから新stampをobserveする。

同じsnapshotの再送や、同じresolved stampのincremental更新では寿命が延びない。
resolved stampはpath内の最古dynamic stampなので、別edgeが更新されても
path最小stampが変わらない場合はfreshness進捗にならない。dynamic source
stampの最小値を保守的な進捗指標として使っているだけであり、各edgeを
厳密な共通時刻へ補間したことを意味しない。

rootまたはpath signatureが変わると新しいbindingとしてfreshnessを開始する。
all-static pathはfreshnessを観測せず、期限切れしない。

## 18. TF Displayのproperty更新分類

### 18.1 topic stateを全resetするproperty

次は `updateTopics()` を呼ぶ。

- `Dynamic Topic`
- `Dynamic Snapshot Topic`
- `Static Topic`
- `Queue Size`

subscriberだけでなくactive setとvisualもclearする。異なるtopicのrecordを
1つのtreeへ誤って混ぜないためである。

### 18.2 appearanceだけを変えるproperty

次は既存visualへ `VisualStyle` を適用し、resampleしない。

- `Frame Timeout`
- `Point Size`
- `Point Style`
- `Alpha`
- `Show Representatives`
- `Representative Radius`

freshnessによるvisibilityは次のDisplay updateで再評価する。

### 18.3 geometryを再構築するproperty

次は `updateGeometry()` からstyle更新後にdirty flagを立てる。

- `Root Frame`
- `Sample Count`
- `Axis Length`
- `Random Seed`

次のRViz updateで全対象childのpathとgeometryを再計算する。

`Axis Length` はまず既存representative Axesのstyleへ反映され、その後の
geometry redrawでsample endpointsへも反映される。この二段階が、
中央軸と点群のlever armを同じ値へ揃える。

### 18.4 Fixed Frame変更

`fixedFrameChanged()` はgeometryをdirtyにせずrenderをqueueする。
毎update実行する `refreshFramePoses()` が、新しいFixed Frameへのscene node
poseを解決するため、分布の再samplingは不要である。

## 19. sample数と計算量制限

### 19.1 Pose Display

`Sample Count` propertyの最大値は100,000である。
1 sampleが3 endpointsなので、最大300,000 pointsになる。

### 19.2 TF Display

tree全体では通常のtargetとして1,500,000 pointsを置く。
有効snapshotでincoming edgeを持つchild候補数
\(F=\lvert\mathtt{by\_child}\rvert\)、要求sample数を \(N\) とすると、

$$
N_{\mathrm{eff}}
=
\min\left(
N,\,
\max\left(
1,\,
\left\lfloor
\frac{1{,}500{,}000}{3F}
\right\rfloor
\right)
\right)
$$

をchildあたりのsample数に使う。

\(F\) はroot不一致やchild別geometry errorを除外する前の候補数であり、
最終的な描画成功child数ではない。このためfailureがある場合もcap計算は
保守的になる。

\(N_{\mathrm{eff}}<N\) なら `Sampling` warningへ実効値を出す。
各childに最低1 sampleを残すため、理論上 \(F>500{,}000\) ならpoint数は
targetを超える。従ってこれは絶対的なhard capではなく、通常範囲のtarget cap
である。

### 19.3 おおよそのcost

reachable unique edge数を \(E\) とすると、native edge samplingは概ね

$$
O(EN_{\mathrm{eff}})
$$

である。各child pathの合成はpath長を \(d_f\) として

$$
O\left(
N_{\mathrm{eff}}\sum_f d_f
\right)
$$

である。実際に描画成功したchild数を \(S\le F\) とすると、
Ogre endpoint storageは

$$
3SN_{\mathrm{eff}}
\le
3FN_{\mathrm{eff}}
$$

pointsとなる。

edge sample cacheは共有upstream edgeの再samplingを防ぐが、childごとの
合成後cloudは別々に保持する。

finite Bingham rejection samplingは1 accepted orientationにつき最大4096 proposals
を試す。point target capはaccepted sample数だけを制限し、proposal回数を直接
制限しない。極端にaccept率の低い分布ではCPU costが先に支配的になりうる。

## 20. statusとfailure propagation

### 20.1 Pose Display

次の表はこのclassが独自に設定するstatusである。

| status key | level | 条件 |
|---|---|---|
| `Message` | Error | `header.frame_id` が空 |
| `Transform` | Error | parentからRViz Fixed Frameへ変換できない |
| `Distribution` | Error/Ok | native sampling失敗、またはsample数 |
| `Representative` | Error | representative検証/導出失敗 |
| `Freshness` | Warn | dynamic poseが期限切れ |

これとは別に、baseの `MessageFilterDisplay` が `Topic` statusなど、
subscription、message受信、TF message filterに関するstatusを提供する。

render error時はvisualを非表示にし、`message_renderable_ = false` とする。
正常render後は `Message`、`Transform`、`Representative` errorを削除し、
`Distribution` をOkへ更新する。

### 20.2 TF Display

| status key | level | 条件 |
|---|---|---|
| `Topics` | Ok/Error | subscribe成功、またはROS exception |
| `Sampling` | Warn | tree target capでsample数をclamp |
| `ProbTF` | Warn | record待ち、failureなしで全frame期限切れ、または1 frame以上を描画できた状態で一部expired/failure |
| `ProbTF` | Error | snapshot invalid、またはfresh frameを1つも描けずfailureあり |
| `ProbTF` | Ok | 全対象frameを描画 |
| `Frame/<child>` | Error | path、dependency、sampling、composition、representative、binding、通常TFのchild別failure |

valid treeの各updateでは、rendered、expired、failedの件数を集計して
top-level `ProbTF` statusを更新する。

期限切れchildは非表示にし、そのchild固有error statusは削除する。
通常TF failureは `Frame/<child>` errorを残し、geometry failureは
`geometry_errors_` から同じstatusへ出す。

## 21. lifecycle

### 21.1 Pose Display

- constructor: propertyを作り、min/maxとenum optionを設定
- `onInitialize()`: baseを初期化し、1つの `TransformVisual` を作る
- `processMessage()`: 最新messageをrender
- `update()`: base update後にfreshnessを反映
- `reset()`: message、freshness、points、`Freshness` statusをclear

### 21.2 TF Display

- constructor: propertyとPIMPLを作る
- `onInitialize()`: base初期化後にstyleを反映
- `onEnable()`: topicをsubscribe
- `onDisable()`: unsubscribeしてreset
- `update()`: pendingをconsumeし、必要ならrender、その後frame pose/freshness更新
- `fixedFrameChanged()`: renderをqueue
- `reset()`: pending、active set、visual、binding、`Frame/*` と
  `Sampling` statusをclear

PIMPLを使うことで、公開headerからROS message、mutex、Ogre visual collectionなど
大きな内部型を隠している。

## 22. 現在のtest coverage

### 22.1 probtf_rviz package

[test_frame_freshness.cpp](test/test_frame_freshness.cpp) は次を検証する。

- timeout最後の1/3で線形fade
- 同一snapshot stampが寿命を延ばさない
- 新source stampが期限切れframeを復活させる
- zero stampの再送も期限切れする
- receipt-based `markProgress()` は毎回寿命を延ばす
- ROS clock rewind時にrebaseする

[test_plugin_loading.cpp](test/test_plugin_loading.cpp) は、

- 2つのclass IDがpluginlibへ宣言される
- class IDから期待するC++型へ解決される

ことを検証する。Displayを実際にinstantiateしてOgre描画を比較するtestではない。

### 22.2 probtf_core package

[test_sampling.cpp](../probtf_core/test/test_sampling.cpp) は主に、

- Dirac sampleのexact性
- uniform quaternionのisotropy
- finite Bingham second momentとPython実装の一致
- rotated/concentrated Binghamの数値安定性
- scale-safe mixture weights
- mixture modeを保ったpath composition
- child-to-rootの合成順
- invalid weightおよびsample-count不一致pathの代表例で、既存outputを
  変更しないこと
- column-major rotation coupling
- stored/fallback representative

を検証する。

[test_latest_snapshot.cpp](../probtf_core/test/test_latest_snapshot.cpp) は主に、

- exact forward path
- moment評価なしのpath metadata
- uniform/finite Bingham point moments
- mixture間分散
- stochastic inverse拒否
- repeated stochastic dependency拒否

を検証する。

さらに
[prob_artag_detectorのdemo-level test](../../../tests/prob_artag_detector/test_real_camera_demo.py)
は、

- producerの固定長mode axesを生成しない
- demo RViz内の通常TF axesを無効にする
- ProbTF representativeを有効にする
- MarkerArrayのmode axesを無効にする

という「中央軸の所有者をProbTF Displayだけにする設定」を検証する。

### 22.3 現時点でintegration testが必要な領域

次はsource上の責務が明確だが、専用の自動RViz integration testはまだない。

- snapshotとincremental callbackのsequence競合
- topic変更時のstate全clear
- RViz propertyを実際に変更したとき、PointCloud endpointと
  `rviz::Axes` のOgre geometryが同時に伸びること
- root-to-Fixed TF喪失と復帰
- child別geometry error時の古いvisual削除
- tree point target capの境界
- enable/disable/resetを繰り返したときのOgre resource寿命
- 実際のRViz画面でmixtureの両modeが見えること

これらを変更する場合は、少なくとも該当するpure logicを小さく分離してunit testし、
可能ならheadless RVizまたは画像比較のintegration testを追加する。

## 23. 拡張時の設計指針

### 23.1 新しいappearance property

sample座標を変えないpropertyは `VisualStyle` へ追加し、

1. Pose/TF両Displayでpropertyを作る
2. `updateAppearance()` でstyleへ詰める
3. `TransformVisual::setStyle()` で既存objectへ適用

する。不要なresampleを避ける。

### 23.2 新しいgeometry property

分布sample、path、endpoint位置を変えるpropertyは `updateGeometry()` へ接続する。
Poseは最新messageを再renderし、TFはdirty flagを立てて次updateで全treeを
再構築する。

Axesとendpointの両方へ意味があるpropertyは、両方の計算経路へ同じ値を渡す。
`Axis Length` がその代表例である。

### 23.3 新しいtopic semantics

subscriber callbackではmessageをpending stateへ置くところまでに留める。
Ogre、RViz status、active visual mapはcallbackから触らない。

complete snapshotとincremental streamを追加統合する場合は、
「どちらが後に到着したか」をsequenceで明示し、古いincrementalが新snapshotを
上書きしないことをtestする。

### 23.4 新しい確率law

RViz内で独自sampleを実装せず、まず `probtf_core` のwire validation、
sampling、representative APIへ追加する。そのうえでPose/TF両Displayが同じAPIを
使い続ける構造を保つ。

invalid lawをrepresentativeやzero covarianceへ自動変換すると、表示だけが
もっともらしく成功してproducerの不具合を隠す。明示的な近似fieldまたはpolicyが
ない限りfail-closedを維持する。

### 23.5 path上の相関

現在は、

- 同一物理edgeのsample cache共有
- provenance重複があるstochastic pathの拒否

までを行う。

異なるedge間のlatent相関を正しく表示するには、単にseedを同じにするのではなく、
dependency IDに対応する共同sample状態を `probtf_core` 側へ導入する必要がある。
その場合はpath単位のjoint sampler API、snapshot metadata、cache key、testを
一緒に設計する。

## 24. 実装上の制約の要約

現在のpluginが意図的に行わないことを最後にまとめる。

- 点群密度を正規化pdf値として表示しない。
- mixtureを単一Gaussianへmoment matchしてから描画しない。
- 各mixture componentへ最低1 sampleを保証しない。
- stochastic transformを逆向きにsampleしない。
- 異なるedge間の未知のlatent相関を推測しない。
- invalid distributionを代表姿勢へ黙ってfallbackしない。
- ProbTF root-to-Fixed配置をProbTF edgeとして勝手に補完しない。
- timeoutでactive record自体を削除しない。
- RViz表示結果を推定器へfeedbackしない。

これらは単なる未実装ではなく、確率的意味を曖昧にしないための境界でもある。
