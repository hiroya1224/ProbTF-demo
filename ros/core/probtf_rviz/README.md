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

Both displays sample the full native transform mixture jointly, so distinct
mixture modes remain distinct and each sample's three colored endpoints share
one rotation and translation draw. The tree display samples every edge once
per redraw and composes equal sample indices along each child-to-root path;
shared upstream edges therefore reuse the same draws across child frames.
Tree geometry is retained while its root-to-RViz transform is unavailable and
becomes visible automatically once TF can resolve it. To keep accidental
settings bounded, the tree display limits one redraw to 1,500,000 rendered
sample points and reports the effective per-frame sample count in its status
when clamping is needed.

`Axis Length` is shared by the sampled RGB endpoints and the central
representative coordinate axes. `Show Representative(s)` must be enabled for
those central axes to be visible. Axes drawn by a separate RViz TF or
MarkerArray display are independent objects and do not follow this property.

Both displays apply `Frame Timeout` to dynamic data. As in RViz's TF display,
the timeout uses ROS time: a frame stays fully opaque for the first two thirds
of its lifetime, fades during the final third, and is then hidden together
with its representative axes. Progress is detected from each ProbTF source
stamp in the tree display rather than a snapshot's arrival time, so a fresh
`/probtf_batch` containing an unchanged old edge does not keep that edge
alive. The pose display instead refreshes on each received dynamic message.
Static tree records and pose messages marked `is_static` do not expire.
