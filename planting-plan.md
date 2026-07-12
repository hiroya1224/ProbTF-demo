# ProbIK 関連パッケージの統合

以降の作業を行う。
なお、作業ごとに commit しておくこと。以下の phase にこだわらず、作業の最小単位で管理すること。
ブランチは `feat/integrate_packages` を使う。

## phase1
- `/home/leus/catkin_ws/src/probik_demo` と `/home/leus/catkin_ws/src/deflecomp` をこのリポジトリに統合します。
  - 作業中は、もとのファイルは残しておくこと
- git の commit log も含めて、完全に統合します。
- `probik_demo` は、`symaware_grasp` パッケージに名称変更します。
- `deflecomp` は、`deflecomp` のままでよいです。
- ros package 群は、ros/ ディレクトリ以下に配置してください

## phase 2
phase 1 の作業が完了したと想定します。
- `symaware_grasp` と `deflecomp` の両者で、重複して実装されているものがあれば、統合します。
- メッセージ型などは、再利用可能なものは再利用すること。その場合、`probik_msgs` などのパッケージを追加で用意しておくとよい。

## phase 3
- ros はブリッジのみ、本体は pypkg 内部で管理する。
- `/home/leus/catkin_ws/src/ProbIK-demo` 直下で pip install すれば、基本的な関数等は使えるような状態にしておく。
- `bingham` パッケージが必要となるはずだが、これは`https://github.com/hiroya1224/BinghamNLL/tree/develop` (ブランチが重要)を使う。third-party 用のディレクトリを用意し、サブモジュールとして登録しておくと良い。
- `pypkg` という名称にしたが、デファクトスタンダード等、慣習があればそちらに倣って名称を変えて良い。

## phase 4
- python package を import して、それを呼び出すくらいの形で、ros 側のパッケージを実装する。実態は python package 側にあり、ros 関連のブリッジに必要なものと関数の呼び出しくらいが ROS 側にあるような構成にする。

## phase 5
動かす前の 旧`probik_demo` や、`deflecomp` の情報をきちんと保てているか、チェックする。動作に不整合が起きた段階でアウトなので、厳重に確かめる。