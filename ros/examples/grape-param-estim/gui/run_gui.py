#!/usr/bin/env python3
"""rosrun-compatible launcher for the separately installed PySide6 GUI."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


_GUI_REEXEC_GUARD = "GRAPE_PARAM_ESTIM_GUI_REEXECUTED"
_MINIMUM_GUI_PYTHON = (3, 10)


def _selected_gui_python(
    environment=None,
    *,
    current_executable=None,
    version_info=None,
    platform_name=None,
):
    """Resolve a requested GUI interpreter before importing GUI dependencies."""

    values = os.environ if environment is None else environment
    current = Path(current_executable or sys.executable).expanduser().resolve()
    version = tuple(sys.version_info[:2] if version_info is None else version_info[:2])
    platform = os.name if platform_name is None else str(platform_name)
    configured = str(values.get("GRAPE_PARAM_ESTIM_GUI_PYTHON", "")).strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
    else:
        virtual_environment = str(values.get("VIRTUAL_ENV", "")).strip()
        if not virtual_environment:
            if version < _MINIMUM_GUI_PYTHON:
                raise RuntimeError(
                    "the desktop GUI requires Python 3.10 or newer; activate its "
                    "virtual environment or set GRAPE_PARAM_ESTIM_GUI_PYTHON"
                )
            return None
        executable_name = "python.exe" if platform == "nt" else "python"
        executable_directory = "Scripts" if platform == "nt" else "bin"
        candidate = (
            Path(virtual_environment).expanduser().resolve()
            / executable_directory
            / executable_name
        )
    if not candidate.is_file() or not os.access(str(candidate), os.X_OK):
        raise RuntimeError(
            "GUI Python interpreter is not an executable file: {}".format(
                candidate
            )
        )
    try:
        same_interpreter = os.path.samefile(str(candidate), str(current))
    except OSError:
        same_interpreter = candidate == current
    if same_interpreter:
        if version < _MINIMUM_GUI_PYTHON:
            raise RuntimeError(
                "the selected GUI interpreter is Python {}.{}; Python 3.10 or "
                "newer is required".format(*version)
            )
        return None
    if values.get(_GUI_REEXEC_GUARD) == "1":
        raise RuntimeError(
            "GUI Python re-execution did not select the requested interpreter"
        )
    return candidate


def _reexec_gui_python(
    arguments,
    environment=None,
    *,
    current_executable=None,
    version_info=None,
    platform_name=None,
    execve=None,
):
    """Re-execute this launcher once under the GUI interpreter when needed."""

    values = os.environ if environment is None else environment
    target = _selected_gui_python(
        values,
        current_executable=current_executable,
        version_info=version_info,
        platform_name=platform_name,
    )
    if target is None:
        return False
    next_environment = dict(values)
    next_environment[_GUI_REEXEC_GUARD] = "1"
    execute = os.execve if execve is None else execve
    script = Path(__file__).resolve()
    execute(
        str(target),
        [str(target), str(script), *[str(value) for value in arguments]],
        next_environment,
    )
    return True


def _locate_package_root() -> Path:
    candidates = []
    configured = os.environ.get("GRAPE_PARAM_ESTIM_PACKAGE_ROOT")
    if configured:
        candidates.append(Path(configured))
    try:
        result = subprocess.run(
            ("rospack", "find", "grape_param_estim"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    else:
        candidates.append(Path(result.stdout.strip()))
    script = Path(__file__).resolve()
    candidates.extend((script.parent.parent, script.parent))
    for parent in script.parents:
        candidates.append(parent)
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "gui" / "src" / "grape_param_estim_gui").is_dir():
            return root
        if (root / "src" / "grape_param_estim_gui").is_dir() and root.name == "gui":
            return root.parent
    raise RuntimeError(
        "cannot locate grape_param_estim package; set GRAPE_PARAM_ESTIM_PACKAGE_ROOT"
    )


def main() -> int:
    _reexec_gui_python(sys.argv[1:])
    package_root = _locate_package_root()
    gui_src = package_root / "gui" / "src"
    estimator_src = package_root / "src"
    for source in (gui_src, estimator_src):
        if not source.is_dir():
            continue
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    entries = [str(estimator_src)] if estimator_src.is_dir() else []
    if existing_pythonpath:
        entries.append(existing_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(entries)
    from grape_param_estim_gui.app import main as application_main

    return application_main(["--package-root", str(package_root), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
