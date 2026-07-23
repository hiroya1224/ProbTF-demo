# probtf_rviz

`probtf_rviz` provides RViz-native visualization for the ProbTF v2 wire
format. It draws the representative coordinate axes and the uncertainty cloud
inside one display, so no display-only `PointCloud2`, `PoseStamped`, or marker
topics are required.

Two displays are exported:

- **Probabilistic Pose** subscribes to one
  `probtf_msgs/ProbabilisticTransformStamped` topic. For each correlated
  transform sample it transforms the positive X, Y, and Z axis endpoints and
  colors them red, green, and blue.
- **Probabilistic TF** subscribes to the incremental dynamic `/probtf` stream,
  the optional complete `/probtf_batch` dynamic snapshot, and the complete
  latched `/probtf_static` set. It assembles the latest tree, evaluates the
  three axis endpoints of every child having an incoming ProbTF edge, and
  draws them in one display. A top-level root is the anchor and is not itself
  drawn unless it also appears as a child edge.

The package lives beside `probtf_core` under `ros/core`: the visualization is
reusable core tooling, while the RViz, Qt, and Ogre dependencies remain out of
the headless runtime package.

## Use

Build and source the workspace, start RViz, select **Add**, and choose either
entry in the `probtf_rviz` group.

For the tree display the defaults are:

- Dynamic Topic: `/probtf`
- Dynamic Snapshot Topic: `/probtf_batch`
- Static Topic: `/probtf_static`
- Frame Timeout: `15 s`

The pose display topic is intentionally empty by default because applications
normally publish dedicated pose distributions. A
`ProbabilisticTransformStamped` uses `header.frame_id` as the parent/target
frame and maps child coordinates into it.

The tree display uses the ProbTF point-moment evaluator and samples only at the
final rendering boundary. The pose display samples the full mixture jointly,
so its three colored endpoints share each sampled transform. Tree geometry is
retained while its root-to-RViz transform is unavailable and becomes visible
automatically once TF can resolve it. To keep accidental settings bounded, the
tree display limits one redraw to 1,500,000 rendered sample points and reports
the effective per-frame sample count in its status when clamping is needed.

Both displays apply `Frame Timeout` to dynamic data. As in RViz's TF display,
the timeout uses ROS time: a frame stays fully opaque for the first two thirds
of its lifetime, fades during the final third, and is then hidden together
with its representative axes. Progress is detected from each ProbTF source
stamp in the tree display rather than a snapshot's arrival time, so a fresh
`/probtf_batch` containing an unchanged old edge does not keep that edge
alive. The pose display instead refreshes on each received dynamic message.
Static tree records and pose messages marked `is_static` do not expire.
