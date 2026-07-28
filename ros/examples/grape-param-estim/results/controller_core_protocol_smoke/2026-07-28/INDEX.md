# PC controller-core protocol smoke — verified 2026-07-29

`controller_core_protocol_smoke.json` records the deterministic two-tick
handshake/replay check against the catkin-built
`gimbalrotor_controller_replay` executable in `jsk_aerial_robot`.
The frozen output includes a two-row `command_timestamp` channel and a
lossless controller-state round trip through the persistent server.
The measured executable and live `/proc/<pid>/exe` SHA-256 are both
`3ec240b29da10a8f7be2cd26b1498225ea7ef8155f1d86d8ffc36d6799e03ef5`;
the controller cores are statically linked and the ELF has no controller-core
DSO or RPATH/RUNPATH dependency.

This is build and protocol evidence only. It is not a substitute for the
bag-derived factual replay conformance gate. The current bags remain
fail-closed because their controller snapshot, geometry/allocation evidence,
and controller-state inputs are incomplete.
