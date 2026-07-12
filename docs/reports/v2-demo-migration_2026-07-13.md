# ProbTF demo v2 全面移行 作業報告

- 開始日時: 2026-07-13 02:57:12 JST
- 対象 repository: `/home/leus/catkin_ws/src/ProbTF-demo`
- 開始時 HEAD: `ad8a23922a0bcaffaff060e123973e229dd8806d`
- 開始時状態: clean

## 1. 目的

各 demo を legacy `ProbabilisticTF` 表現と個別の伝播実装から切り離し、native v2
`TransformDistributionStamped`、`ProbTfGraph`、topic listener、lookup kernel を使う構成へ
移行する。移行完了時には v1 message、v1 domain model、v1/v2 adapter、symaware 固有の
旧 `ProbTfTree` を削除する。

`2846fced8` で追加された `SOURCE_DIR = "../../../../src"` による catkin package 外の
Python source relay も廃止し、各 namespace を所有する catkin package の `src/` に置く。

## 2. 移行時の設計判断

1. `/probtf` graph に登録するのは、並進と回転がともに定義された physical SE(3) edge
   だけとする。orientation-only likelihood に zero translation を補わない。
2. orientation-only posterior と singular likelihood evidence は、full transform record と
   分けた v2 message で運ぶ。
3. grasp target や end-effector belief のような derived result は physical forest に再登録せず、
   v2 component modelを含む application topic で運ぶ。
4. lookup は中央 bridge process への独自 RPC ではなく、tf2 と同様に各 consumer が
   `/probtf` と `/probtf_static` を listen して local `ProbTfGraph` を構築する方式にする。
5. deflecomp の stiffness posterior は SE(3) 分布ではないため `/probtf` に偽装しない。
   frame transport と可視化 query を TF import -> v2 listener/lookup に移す。
6. point cloud は lookup kernel の point moment result だけを入力とし、表示用 Gaussian sample
   は terminal visualization として生成する。生成した summary を graph edgeへ戻さない。

## 3. 基準状態

- Python tests: `180 passed, 1 warning`
- `probtf` は catkin devel space から import できず、root project の事前 install が必要だった。
- v1 ROS publisher/consumer: two-IMU、orientation filter/fusion、symaware grasp 一式。
- symaware link cloud: YAML -> legacy `ProbTfTree` -> local tangent propagation。
- deflecomp: v1 message は使わないが、ProbTF graph/topic と未接続。

## 4. 作業ログ

### 4.1 catkin package 内への Python source 正規配置

- `probtf`、`probtf_estimators` を `ros/core/probtf_core/src/` へ移した。
- `deflecomp_core`、`deflecomp_sim`、`deflecomp_examples` を各 package の `src/` へ移した。
- `symaware_grasp` を ROS package の `src/` へ移した。
- 3個の deflecomp `setup.py` から `SOURCE_DIR = "../../../../src"` を削除した。
- `probtf_core` と `symaware_grasp` の setup を package-local `find_packages()` に統一した。
- catkin dependency を明示し、test 内の `sys.path.insert()` を削除した。
- package boundary test を新しい source ownership に合わせて更新し、parent path relay と
  test-time path injection を検出するようにした。

検証:

- Python tests: `181 passed, 1 warning`
- `catkin build probtf_core`: 成功
- `catkin build probtf_msgs symaware_grasp`: 成功
- `catkin build deflecomp_sim deflecomp_ros`: 成功
- `source devel/setup.bash` 後、`probtf`、`probtf_estimators`、`probtf_ros`、
  `deflecomp_core`、`deflecomp_sim`、`deflecomp_examples`、`symaware_grasp` の import: 成功

### 4.2 Bingham 正規化積分の core 内収容

- `probtf.bingham` が外部 `bingham` Python package を importする構成をやめた。
- 実際に必要な正規化定数と導関数の積分実装だけを `probtf._vendor` に収容した。
- upstream の copyright / Apache-2.0 header を保持し、`THIRD_PARTY_NOTICES.md` に由来を記録した。

検証:

- external `bingham` importを強制拒否した状態で `import probtf`: 成功
- Bingham / v2 distribution / kernel tests: `42 passed`

### 4.3 deterministic right composition の v2 閉包

- stochastic record の各 component に deterministic child offsetを右合成する core 演算を追加した。
- `R_new = R_old R_fixed` に伴う coupling の基底変換と `R_old r` 項を、3x9
  `rotation_coupling` に保持する。
- mixture weight、residual covariance、representative、approximationを保持し、derived edgeを
  provenanceへ記録する。
- symaware grasp target は、この演算を使うことで v1 の mode plug-in / covariance samplingを
  廃止できる。

検証:

- 任意 quaternion 20点で、旧 component + fixed transform の直接評価と合成後 component の
  conditional meanが `1e-12` 以内で一致
- composition tests: `2 passed`

## 5. コミット

| commit | 内容 |
| --- | --- |
| `db91ed4` | Python sourceを所有 catkin package 内へ正規配置 |
| `89f3811` | Bingham 正規化積分を core 内へ収容 |
