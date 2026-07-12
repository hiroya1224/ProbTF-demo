# IKの実装デモへの準備
今の環境は，ROS1 です．カスなんですが，以下をよろしくお願いします

- 簡単な六軸アームを作ってください．urdf ファイルを作って，慣習に基づいてディレクトリを配置し，それに従って urdf ファイルを配置してください．リンクは円筒や直方体などでよいです．軸の構成は，標準的なものを採用してください．
- Bingham 分布で表現された確率的な姿勢と，Gaussian 分布で表現された確率的な位置をまとめたペアで，**ProbabilisticTF** を定義します
  - メッセージ型を定義してください．Bingham 分布は quaternion が従う分布で，4x4 の対称行列で表現されます．
  - Bingham 分布に関して，python パッケージとして，`import bingham` などで使えます．
  - `third_party/BinghamNLL` の `develop` ブランチを submodule として利用します．root で `pip install .` すると同時にインストールされます．
- ProbabilisticTF (ptf) を受け取ると，ランダムなノイズが乗った各点が得られますが，それを pointcloud の形で表現してください．
  - ptf を受け取ったら，pointcloud を返すような node があると良いです．
  - Bingham 分布では，各座標軸 (1,0,0), (0,1,0), (0,0,1) の行き先を，それぞれ，赤，緑，青で表示させるようにしてください．それに，ptf の中にある Gauss ノイズが乗って，平行移動もしているような状況になります．
  - mode-pose （ガウス分布は mode は平均と一致するが，Bingham は球面上の分布なので mean は無意味で，mode になるため，基本的に mode と表現します）も一緒に出すようにしてください．
- 適当なパラメータで作ったら，それを rviz 上で表示するような launch ファイルを書いてください．
  - 全部の node をいっぺんに立てて，pointcloud と mode-pose を表示させるようにしてください．rviz ファイルもあると嬉しいです．
  - launch を立てたら，pointcloud と mode-pose が確認できるようにしてください．

# build について
`~/catkin_ws/` で作業しています．ここで build してください．
