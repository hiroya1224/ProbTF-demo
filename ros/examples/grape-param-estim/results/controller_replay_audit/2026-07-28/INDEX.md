# Controller replay sufficiency audit — 2026-07-28

The immutable Grape bags 4, 7, and 8 were scanned with
`audit_grape_controller_replay.py`. The canonical result is
[`controller_replay_audit.json`](controller_replay_audit.json), bundle SHA-256
`2042e8281cf1848c7c636107bac7f563181253912644b2e9e50d26977dfda65b`.

All three episodes fail closed for `pc_exact` replay:

| episode | available | derivable | missing | blocking missing inputs |
|---|---:|---:|---:|---|
| `20260612-04` | 8 | 3 | 4 | navigator target, control mode, nominal model/geometry snapshot, torque allocation matrix |
| `20260612-07` | 9 | 4 | 2 | nominal model/geometry snapshot, torque allocation matrix |
| `20260612-08` | 9 | 4 | 2 | nominal model/geometry snapshot, torque allocation matrix |

`DERIVABLE` does not mean replay-ready: those channels still require a
materialized, validated fixture. The open-loop plant path may use recorded
commands without a controller claim; exact closed-loop inference remains
disabled until a fixture and external C++ conformance gate pass.
