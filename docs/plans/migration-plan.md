# Prob-TF への統合

## 対象リポジトリ
以下の３つのリポジトリを ProbTF-demo に統合することを考えます。
- `/home/leus/catkin_ws/src/stochastic_joint_estim`
- `/home/leus/catkin_ws/src/urdf_estimation_with_imus`
- `/home/leus/catkin_ws/src/bingham_orientation_filter`

## phase 0
まず、上記３つのリポジトリに関して、それぞれの内容を `ProbTF-demo/lecture-tmp` 内に、各 repository 名の md ファイルに日本語で書き出していってもらいたい。
たとえば、 `ProbTF-demo/lecture-tmp/stochastic_joint_estim.md` など。

その際、
`/home/leus/catkin_ws/src/ProbTF-demo/prob_tf_crosscut_overview_ja.md` に「Prob-TF」というものの思想と現状について書かれているので、
それで説明されている「Prob-Tf」の文脈に、各リポジトリを翻訳するような書き方にしてほしいです。
まず、現状の実装についてサマライズしたあとに、ProbTF

３つそれぞれのリポジトリに関して md ファイルを書いたら、**コミットはせずに**おいておいてください。

## phase 1
（ros ディレクトリ以下の構造を，core, examples を配置して変更した．core 以下には ProbTF の管理系のコアパッケージが来るべき）
phase 0 で分析した，`stochastic_joint_estim`, `urdf_estimation_with_imus`, `bingham_orientation_filter` のそれぞれについて，ProbTF 表現としてこの ProbTF-demo に移植する．
その際，`/home/leus/catkin_ws/src/ProbTF-demo/ros/core` 以下に，`probtf_core` 等で，中核となるパッケージを配置すべきである．これも python package (probtf) として整理し，ROS 内部では，その probtf を import して関数を使い，ros はその通信にだけ使う，という構成にすべきである．

以下の「優先度」に従って，作業を進めてもらう．本 phase の目的は，**分散したパッケージ群をひとつの文脈に落とし込み，ProbTF-demo 内に集約する**ことにある．したがって，ProbTF の core の詳細な設計は，後ほど行ない，今は現状のコード群が求める機能を集めて，その設計に資する情報を収集する状況である．次の phase で，さらに整理していく．

### 優先度1: urdf_estimation_with_imus
最も優先度が高いのは，`urdf_estimation_with_imus` である．これは ProbTF のデモの中で最も重要度が高いものである．
これの，遠心力・角速度から相対位置姿勢の確率分布を導く箇所が，まさに ProbTF の生成箇所である．これを，共通の ProbTF を使用する形態として分離したい．
symbolic urdf の取り扱いは要検討箇所が多いが，一応このままのコメント方式でよい（現状の，urdf にコメントを用意して，それを置き換える独自の記法は，urdf の記法に準拠しており，従来のシステムとの接続は比較的しやすい．一応この方針で行くが，「実体化 urdf」の取り扱いなどは今後の課題になりそう）
実装方針としては，遠心力と角速度から probTF を出すところがこのパッケージの担うコアとなって，本パッケージでいろいろ行なっていた残りの箇所は，ProbTF 自体の core パッケージに入るべきものと思われる．特に，Bingham パラメータの取り扱いは考えるべきことが多い．4th moment までの計算も行なっており，これはチェインTFの連鎖則で有用である．quaternion の合成を moment matching する方式を，symaware_grasp 内部ではやっていたような気がするが，これを厳密に解けるメリットはある．

### 優先度2: stochastic_joint_estim
次点は stochastic_joint_estim であろうが，これは現状の examples に含まれる deflecomp に大部分が担われている．パーティクルフィルタ関連の実装は，もはや不必要である．むしろ重要なのは，**尤度によるセンサの統合法**などの，その受け口の整理である．
ProbTF リポジトリに導入される際，stochastic_joint_estim の実装を間に受けるべきではない，実装がカスすぎて，混乱を来す可能性がある．そのエッセンスを抽出し，それを導入するほうがよい．
センサ情報の統合に関して言えば，このパッケージはそれを正面から取り組んでおり良い．
ロボットモデル上でのセンサ取り付け位置・マーカー取り付け位置の管理は，
`/home/leus/catkin_ws/src/nejineji-urdfs/yamaguchi_arm_nejineji/config/deflecomp_imu_frames.yaml`
のように，ロボットごとに yaml を持たせる約束にして良いと思う．
このパッケージに関しては，センサの受け口の実装がメインとなる．probTF のコアに入るべき機能と思われるが，センサフュージョンのためのセンサ情報の取り込み口である．その異意味では，msgs ディレクトリにも影響が及ぼされると思われる．

### 優先度3: bingham_orientation_filter
これは単一のサンプルである．内部の確率表現を，各データによって分割し（IMUから求めた姿勢，磁力から求めた姿勢，など），それを ProbTF として publish し，別途合成用のパッケージ（ProbTF のコアに含まれると思われる）によって合成される構成になる．
本パッケージで重要なものは，**quaternion の運動方程式の確率化実装**である．これによって，カルマンフィルタ的な予測 update に組み込める．これは core に実装されるべき機能である．

