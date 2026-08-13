#!/usr/bin/env python3
"""Repair companion files after patch_8be7473_rewrite_independent_savgol.py.

Targets the exact untouched 8be7473 companion files left behind by the first
independent-SG patch. It removes stale estimator.base assumptions, updates SG
confidence/ablation for separate rotor and gimbal lags, and keeps the spline
estimator itself untouched. No .bak files are created.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import py_compile
import re
import subprocess
import tempfile

EXPECTED = {
    "savgol_dynamics_confidence.py": "7e9e2200ec7372a09fa75dbfd701bf40e053cb9f",
    "savgol_window_ablation.py": "35d2a87b7beec3ca18cfcfa01e714876c8ab17a6",
    "deterministic_savgol_dynamics_data_dictionary.md": "1eedcc6e6064de6ba192a5d8691c5e42a9ef113d",
    "spline_dynamics_confidence.py": "c3b08eb0249854108571caef3954449621a81cd9",
}

def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()

def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}")
    return text.replace(old, new, 1)

def regex_once(text: str, pattern: str, replacement: str, label: str, flags: int = 0) -> str:
    out, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return out

def patch_legacy_confidence(text: str) -> str:
    text = replace_once(
        text,
        "    reference_rollout = deterministic.forward_rollout(\n"
        "        bag,\n"
        "        np.zeros(\n"
        "            deterministic.PHYSICAL_DIMENSION,\n"
        "            dtype=float,\n"
        "        ),\n"
        "        initial_delay,\n"
        "        reference_parameters,\n"
        "    )\n",
        "    selected_gimbal_delay = (\n"
        "        selected.delay_seconds if getattr(selected, \"gimbal_delay_seconds\", None) is None\n"
        "        else float(selected.gimbal_delay_seconds)\n"
        "    )\n"
        "    initial_gimbal_delay = float(getattr(arguments, \"initial_gimbal_delay\", initial_delay))\n"
        "    reference_rollout = deterministic.forward_rollout(\n"
        "        bag,\n"
        "        np.zeros(deterministic.PHYSICAL_DIMENSION, dtype=float),\n"
        "        initial_delay,\n"
        "        reference_parameters,\n"
        "        gimbal_delay=initial_gimbal_delay,\n"
        "    )\n",
        "confidence reference pair",
    )
    text = replace_once(
        text,
        "    parameter_rollout = deterministic.forward_rollout(\n"
        "        bag,\n"
        "        selected.physical_coordinate,\n"
        "        selected.delay_seconds,\n"
        "        reference_parameters,\n"
        "    )\n",
        "    parameter_rollout = deterministic.forward_rollout(\n"
        "        bag, selected.physical_coordinate, selected.delay_seconds, reference_parameters,\n"
        "        gimbal_delay=selected_gimbal_delay,\n"
        "    )\n",
        "confidence parameter pair",
    )
    text = replace_once(
        text,
        "        arguments,\n"
        "        reference_parameters,\n"
        "    )\n"
        "    replay_rollout = wrench_evaluation.simulation\n",
        "        arguments,\n"
        "        reference_parameters,\n"
        "        gimbal_delay=selected_gimbal_delay,\n"
        "    )\n"
        "    replay_rollout = wrench_evaluation.simulation\n",
        "confidence replay pair",
    )
    return text

def patch_confidence(text: str) -> str:
    text = replace_once(text, 'SCHEMA = "grape-param-estim/savgol-dynamics-confidence/v3"', 'SCHEMA = "grape-param-estim/savgol-dynamics-confidence/v4"', "confidence schema")
    s = text.index("\ndef _residual_parameter_diagnostics(")
    e = text.index("\ndef create_argument_parser()", s)
    text = text[:s] + "\n" + text[e:]
    old = '''    # SG estimator semantics: zero command-lag initialization unless explicitly
    # overridden.  Do this before config handling so the config's historical
    # initial_delay_seconds cannot silently re-enter.
    if arguments.initial_delay is None:
        arguments.initial_delay = 0.0
    deterministic._ACTIVE_WINDOW_SECONDS = float(arguments.window_seconds)
'''
    text = replace_once(text, old, '    deterministic._ACTIVE_WINDOW_SECONDS = float(arguments.window_seconds)\n    deterministic._resolve_lag_defaults(arguments)\n', "confidence lag defaults")
    text = replace_once(text, "    initial_delay = float(arguments.initial_delay)\n", "    initial_delay = float(arguments.initial_delay)\n    initial_gimbal_delay = float(arguments.initial_gimbal_delay)\n", "confidence gimbal init")
    text = replace_once(
        text,
        "    if arguments.deterministic_result is None:\n"
        "        selected, optimizer_history = legacy._estimate_solution(\n"
        "            bag,\n"
        "            arguments,\n"
        "            initial_delay,\n"
        "            vehicle_model.parameters,\n"
        "            parameter_prior,\n"
        "        )\n",
        "    if arguments.deterministic_result is None:\n"
        "        initial_physical = np.zeros(deterministic.PHYSICAL_DIMENSION, dtype=float)\n"
        "        physical_lower, physical_upper = deterministic._physical_bounds(initial_physical)\n"
        "        lag_search = deterministic._split_command_lag_search(\n"
        "            deterministic.SplineDynamicsProblem((bag,), vehicle_model.parameters, parameter_prior),\n"
        "            initial_physical, physical_lower, physical_upper, arguments,\n"
        "        )\n"
        "        selected = lag_search[\"selected_solution\"]\n"
        "        optimizer_history = {\"source\": \"split_command_lag_search\", \"command_lag_search\": {key: value for key, value in lag_search.items() if key != \"selected_solution\"}}\n",
        "confidence standalone split solver",
    )
    text = replace_once(text, '            delay_seconds = float(selection["delay_seconds"])\n', '            rotor_delay_seconds = float(selection.get("rotor_delay_seconds", selection["delay_seconds"]))\n            gimbal_delay_seconds = float(selection.get("gimbal_delay_seconds", rotor_delay_seconds))\n', "confidence result pair")
    text = replace_once(text, "        selected_evaluation = problem.evaluate_strict(\n            physical_coordinate,\n            delay_seconds,\n        )\n", "        selected_evaluation = problem.evaluate_strict(\n            physical_coordinate, rotor_delay_seconds, gimbal_delay_seconds\n        )\n", "confidence strict pair")
    text = replace_once(text, "            delay_seconds=delay_seconds,\n            evaluation=selected_evaluation,\n", "            delay_seconds=rotor_delay_seconds,\n            gimbal_delay_seconds=gimbal_delay_seconds,\n            evaluation=selected_evaluation,\n", "confidence solution pair")
    text = replace_once(text, '        "selected delay {:.6f}s; mass {:.6g} kg".format(\n            selected.delay_seconds,\n            selected.evaluation.decoded.parameters.mass,\n        ),\n', '        "selected lags rotor={:.6f}s gimbal={:.6f}s; mass {:.6g} kg".format(\n            selected.delay_seconds, selected.gimbal_delay_seconds,\n            selected.evaluation.decoded.parameters.mass,\n        ),\n', "confidence pair print")
    # remove residual diagnostic computation
    text = regex_once(text, r'    residual_parameter_diagnostics = _residual_parameter_diagnostics\(.*?\n    \)\n\n', '', "remove residual parameter computation", flags=re.DOTALL)
    text = replace_once(text, '            "second_moment_dimensionless": (\n                residual_parameter_diagnostics["residual_wrench_second_moment"][\n                    "dimensionless"\n                ]\n            ),\n', '            "second_moment_dimensionless": (wrench_dimensionless.T @ wrench_dimensionless) / wrench_dimensionless.shape[0],\n', "direct wrench second moment")
    # delete absorbability block and posterior residual error
    text = regex_once(text, r'        "residual_parameter_absorbability": \{.*?        \},\n', '', "remove likelihood absorbability", flags=re.DOTALL)
    text = regex_once(text, r'        "residual_implied_parameter_error": \(.*?        \),\n', '', "remove posterior implied error", flags=re.DOTALL)
    text = replace_once(text, '            "delay_seconds": float(selected.delay_seconds),\n', '            "delay_seconds": float(selected.delay_seconds),\n            "rotor_delay_seconds": float(selected.delay_seconds),\n            "gimbal_delay_seconds": float(selected.gimbal_delay_seconds),\n', "likelihood pair")
    text = replace_once(text, '            "delay_seconds": selected.delay_seconds,\n            "objective_cost": float(deterministic._solution_cost(selected)),\n', '            "delay_seconds": selected.delay_seconds,\n            "rotor_delay_seconds": selected.delay_seconds,\n            "gimbal_delay_seconds": selected.gimbal_delay_seconds,\n            "objective_cost": float(deterministic._solution_cost(selected)),\n', "confidence deterministic pair")
    text = text.replace('        "residual_parameter_diagnostics": residual_parameter_diagnostics,\n', '')
    # remove files generated solely by removed diagnostic
    s = text.find('    diagnostic_lines = _residual_parameter_diagnostic_lines(')
    if s != -1:
        e = text.index('    _write_json(output_directory / "confidence.json", payload)', s)
        text = text[:s] + text[e:]
    text = regex_once(text, r'\n    print\(\n        "residual absorbability:.*?\n    \)\n', '\n', "remove absorbability print", flags=re.DOTALL)
    return text

def patch_ablation(text: str) -> str:
    text = replace_once(text, 'SCHEMA = "grape-param-estim/savgol-window-ablation/v2"', 'SCHEMA = "grape-param-estim/savgol-window-ablation/v3"', "ablation schema")
    text = replace_once(text, '        "selected_delay_seconds": float(selection["delay_seconds"]),\n', '        "selected_delay_seconds": float(selection["delay_seconds"]),\n        "selected_rotor_delay_seconds": float(selection.get("rotor_delay_seconds", selection["delay_seconds"])),\n        "selected_gimbal_delay_seconds": float(selection.get("gimbal_delay_seconds", selection["delay_seconds"])),\n', "ablation pair")
    text = regex_once(text, r'        "joint_residual_wrench_accumulated_squared_loss_dimensionless": float\(.*?        \),\n', '', "remove ablation wrench objective", flags=re.DOTALL)
    text = text.replace('        axes[0, 0].set_ylabel("residual-wrench accumulated squared loss")\n', '        axes[0, 0].set_ylabel("acceleration-residual dynamics loss")\n', 1)
    text = replace_once(text, '        axes[1, 1].plot(windows, 1000.0 * vector("selected_delay_seconds"), marker="o", label="selected lag")\n', '        axes[1, 1].plot(windows, 1000.0 * vector("selected_rotor_delay_seconds"), marker="o", label="rotor lag")\n        axes[1, 1].plot(windows, 1000.0 * vector("selected_gimbal_delay_seconds"), marker="o", label="gimbal lag")\n', "ablation lag plot")
    text = replace_once(text, '                "  residual-wrench accumulated squared loss={}".format(\n                    item.get("joint_residual_wrench_accumulated_squared_loss_dimensionless")\n                ),\n                "  selected delay={} s".format(item.get("selected_delay_seconds")),\n', '                "  acceleration-residual dynamics loss={}".format(item.get("joint_dynamics_loss")),\n                "  selected rotor delay={} s".format(item.get("selected_rotor_delay_seconds")),\n                "  selected gimbal delay={} s".format(item.get("selected_gimbal_delay_seconds")),\n', "ablation text")
    text = regex_once(text, r'                "  residual absorbable fraction=.*?                "  confidence residual sample count=', '                "  confidence residual sample count=', "remove ablation absorb text", flags=re.DOTALL)
    text = regex_once(
        text,
        r'                            "residual_absorbable_fraction": float\(.*?                            "residual_implied_parameter_std_raw_coordinate": np\.asarray\(.*?                            \),\n',
        '',
        "remove ablation absorb/implied-error fields",
        flags=re.DOTALL,
    )
    text = replace_once(text, '    try:\n        return run(arguments, passthrough)\n    except ValueError as error:\n        raise SystemExit(str(error)) from error\n', '    return run(arguments, passthrough)\n', "ablation propagate errors")
    return text


def patch_ablation_independent(text: str) -> str:
    text = patch_ablation(text)
    replacements = (
        ("estimator.base.multi.load_multi_bag_config", "estimator.multi.load_multi_bag_config"),
        ("estimator.base.plt", "estimator.plt"),
        ("estimator.base.PdfPages", "estimator.PdfPages"),
        ("estimator.base.strict._write_text", "estimator.strict._write_text"),
        ("estimator.base._write_parameters_pdf", "estimator._write_parameters_pdf"),
    )
    for old, new in replacements:
        if old not in text:
            raise RuntimeError("ablation missing expected stale reference: {}".format(old))
        text = text.replace(old, new)
    if "estimator.base" in text:
        raise RuntimeError("ablation still contains estimator.base after repair")
    return text

def patch_dictionary(text: str) -> str:
    start = text.index("## 7. Command timestamp diagnostics")
    replacement = '''## 7. Command lags

The recorded control input contains two separately timestamped command channels:

```text
rotor_command   : four rotor thrust commands
gimbal_command  : four gimbal angle commands
```

The estimator therefore uses two lag coordinates, `rotor_delay_seconds` and
`gimbal_delay_seconds`.  A single common lag is not imposed.

For each channel, the median positive recorded timestamp interval is the
channel's data-derived publish period.  Unless explicitly overridden, the
initial lag is one measured publish period for that channel.

### Smooth continuation

The smooth command is the ZOH initial value plus a sum of command jumps, with
each Heaviside jump replaced by a quintic smoothstep.  Transition supports are
allowed to overlap.  The default transition half-widths are `4, 2, 1, 0.5`
times each channel's measured publish period.

Both lag columns are included in the analytic Jacobian.  Optimizer diagnostics
record rotor/gimbal lag gradients and finite-difference checks along both lag
axes.

### Strict-ZOH refinement

The previous fixed `±4 ms`, `1 ms`, `top-k=3` polish is removed.  Strict ZOH is
screened on a 2-D lag grid whose axis steps are the measured rotor and gimbal
publish periods.  The initial grid spans one period around the smooth result.
If the best point lies on an edge, that axis is extended by one publish period
in the improving direction.  Physical parameters are optimized at the selected
lag pair, the lag grid is screened again, and the alternation stops when the
same pair remains selected.

Detailed history is written to `delay_profile.json`, `delay_profile.txt`, the
text-only `delay_profile.pdf`, and `optimizer_diagnostics.json`.

## 8. Deterministic parameter objective

The deterministic SG objective is again the original acceleration-domain
gradient-matching objective.  Translation uses body-frame acceleration error;
rotation uses angular-acceleration error with the reference inertia/mass metric.
The bag data term is the mean squared residual over valid centered SG times.
The Gaussian physical prior is a separate residual block.

Residual body-wrench mean, covariance, second moment, standard deviation and RMS
remain diagnostics; they are not the deterministic parameter objective.

## 9. Residual wrench and confidence

A raw residual-wrench sample is retained at every valid centered SG evaluation
time.  No confidence-specific temporal thinning is applied.

The temporary residual-parameter absorbability and residual-implied parameter
bias/covariance/second-moment diagnostics are removed.  The data-only SVD,
information matrix, residual-wrench Gaussian model, and Gaussian-prior fusion
remain.

Moore--Penrose pseudoinverse is used only where rank-deficient information or
precision matrices are intentionally part of the model.

## 10. Numerical failure policy

Invalid optimizer trials are not replaced by an artificial large residual or a
zero Jacobian.  Numerical exceptions propagate at the point where the actual
calculation becomes invalid and stop the run.

Physical inertia dynamics use solve-based linear algebra and therefore fail on
a genuinely singular physical inertia.  Pseudoinverse is reserved for intended
rank-deficient information/precision calculations.
'''
    return text[:start] + replacement


def _validate_independent_estimator(path: Path) -> None:
    if not path.is_file():
        raise SystemExit("missing independent estimator: {}".format(path))
    text = path.read_text(encoding="utf-8")
    required = (
        "Independent geometric Savitzky--Golay rigid-body dynamics estimator",
        "ROTOR_DELAY_INDEX = PHYSICAL_DIMENSION",
        "GIMBAL_DELAY_INDEX = PHYSICAL_DIMENSION + 1",
        "def _split_command_lag_search(",
        "def _write_split_delay_report(",
        "def _write_parameters_pdf(",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise SystemExit(
            "deterministic_savgol_dynamics_estimator.py is not the independent-SG state; missing: {}"
            .format(", ".join(missing))
        )
    forbidden = ("import deterministic_spline_dynamics_estimator", "base.")
    remaining = [token for token in forbidden if token in text]
    if remaining:
        raise SystemExit(
            "independent estimator still contains forbidden dependency: {}"
            .format(", ".join(remaining))
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    _validate_independent_estimator(root / "deterministic_savgol_dynamics_estimator.py")

    paths = {name: (root / name).resolve() for name in EXPECTED}
    originals = {}
    for name, expected in EXPECTED.items():
        path = paths[name]
        if not path.is_file():
            raise SystemExit("missing target: {}".format(path))
        data = path.read_bytes()
        actual = blob_sha(data)
        if actual != expected:
            raise SystemExit(
                "refusing to repair {}: expected untouched 8be7473 blob {}, got {}"
                .format(name, expected, actual)
            )
        originals[name] = data

    replacements = {
        "savgol_dynamics_confidence.py": patch_confidence(
            originals["savgol_dynamics_confidence.py"].decode("utf-8")
        ),
        "savgol_window_ablation.py": patch_ablation_independent(
            originals["savgol_window_ablation.py"].decode("utf-8")
        ),
        "deterministic_savgol_dynamics_data_dictionary.md": patch_dictionary(
            originals["deterministic_savgol_dynamics_data_dictionary.md"].decode("utf-8")
        ),
        "spline_dynamics_confidence.py": patch_legacy_confidence(
            originals["spline_dynamics_confidence.py"].decode("utf-8")
        ),
    }
    replacement_bytes = {name: value.encode("utf-8") for name, value in replacements.items()}

    with tempfile.TemporaryDirectory(prefix="repair-independent-sg-") as td_value:
        td = Path(td_value)
        for name, data in replacement_bytes.items():
            if not name.endswith(".py"):
                continue
            target = td / Path(name).name
            target.write_bytes(data)
            py_compile.compile(str(target), doraise=True)

    if "estimator.base" in replacements["savgol_window_ablation.py"]:
        raise RuntimeError("stale estimator.base survived ablation repair")
    if "deterministic.base" in replacements["savgol_dynamics_confidence.py"]:
        raise RuntimeError("stale deterministic.base survived confidence repair")

    written = []
    try:
        for name, data in replacement_bytes.items():
            path = paths[name]
            temporary = path.with_name(path.name + ".repair-tmp")
            temporary.write_bytes(data)
            os.replace(temporary, path)
            written.append(name)
        subprocess.run(
            ["git", "diff", "--check", "--", *replacements.keys()],
            cwd=root,
            check=True,
        )
    except Exception:
        for name in written:
            paths[name].write_bytes(originals[name])
        raise

    print("repaired independent SG companion files")
    print("  savgol_window_ablation.py: estimator.base removed")
    print("  savgol_dynamics_confidence.py: split rotor/gimbal lag aware")
    print("  spline_dynamics_confidence.py: helper calls pass gimbal lag explicitly")
    print("  deterministic_spline_dynamics_estimator.py: untouched")
    print("  backups: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
