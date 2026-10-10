# PID voxel data file catalog

Complete: True. Host: `leus-TRX50-AERO-D`.

This commit contains a catalog. The calculation data remain at the local paths listed below.

`files.tsv` lists every artifact with its size and SHA-256. `catalog.json` also retains calculation conditions and completed grid counts.

| Case | Saved dataset directory | Files | Bytes | Completed UTC |
| --- | --- | ---: | ---: | --- |
| failure1 | `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/first-order-lag/outputs/failure1/pid_gain_voxels_success_context` | 267 | 408306991 | 2026-10-08T14:28:39.776682+00:00 |
| failure2 | `/home/leus/catkin_ws/src/ProbTF-demo/ros/examples/grape-param-estim/minimal/first-order-lag/outputs/failure2/pid_gain_voxels_success_context` | 267 | 410180807 | 2026-10-10T23:22:05.958650+00:00 |

Open `<dataset>/index.html` for the 3D plots. Each `<group>/grid_<size>/` contains `pid_3d.html`, `pid_3d.png`, `volume.vtr`, `volume.npz`, checkpoint arrays, and metadata.

P/I/D vary together for one group; the other nine gains equal the same successful controller. Each case uses its respective bag posterior with a fixed sample set. Values are associated with grid nodes.
