#!/usr/bin/env python3
"""rosrun-compatible launcher for the separately installed PySide6 GUI."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


_GUI_REEXEC_GUARD = "GRAPE_PARAM_ESTIM_GUI_REEXECUTED"
_MINIMUM_GUI_PYTHON = (3, 10)


def _absolute_path_preserving_symlinks(value) -> Path:
    """Return an absolute path without resolving a virtualenv symlink."""

    return Path(os.path.abspath(str(Path(value).expanduser())))


def _environment_python(environment_root, platform_name) -> Path:
    executable_name = "python.exe" if platform_name == "nt" else "python"
    executable_directory = "Scripts" if platform_name == "nt" else "bin"
    return _absolute_path_preserving_symlinks(
        Path(environment_root).expanduser().resolve()
        / executable_directory
        / executable_name
    )


def _selected_gui_python(
    environment=None,
    *,
    package_root=None,
    current_executable=None,
    version_info=None,
    platform_name=None,
):
    """Resolve a requested GUI interpreter before importing GUI dependencies."""

    values = os.environ if environment is None else environment
    current = _absolute_path_preserving_symlinks(
        current_executable or sys.executable
    )
    version = tuple(sys.version_info[:2] if version_info is None else version_info[:2])
    platform = os.name if platform_name is None else str(platform_name)
    configured = str(values.get("GRAPE_PARAM_ESTIM_GUI_PYTHON", "")).strip()
    if configured:
        candidate = _absolute_path_preserving_symlinks(configured)
    else:
        virtual_environment = str(values.get("VIRTUAL_ENV", "")).strip()
        if virtual_environment:
            candidate = _environment_python(virtual_environment, platform)
        elif package_root is not None:
            local_environment = (
                Path(package_root).expanduser().resolve() / "gui" / ".venv"
            )
            local_candidate = _environment_python(
                local_environment, platform
            )
            if local_candidate.exists():
                candidate = local_candidate
            elif version < _MINIMUM_GUI_PYTHON:
                raise RuntimeError(
                    "the desktop GUI requires Python 3.10 or newer; create "
                    "gui/.venv, activate a compatible virtual environment, or "
                    "set GRAPE_PARAM_ESTIM_GUI_PYTHON"
                )
            else:
                return None
        else:
            if version < _MINIMUM_GUI_PYTHON:
                raise RuntimeError(
                    "the desktop GUI requires Python 3.10 or newer; activate its "
                    "virtual environment or set GRAPE_PARAM_ESTIM_GUI_PYTHON"
                )
            return None
    if not candidate.is_file() or not os.access(str(candidate), os.X_OK):
        raise RuntimeError(
            "GUI Python interpreter is not an executable file: {}".format(
                candidate
            )
        )
    # A virtualenv's python is commonly a symlink to its base interpreter.
    # Invoking that symlink is what makes Python discover pyvenv.cfg, so inode
    # equality (os.path.samefile) must not suppress the re-execution.
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
    package_root=None,
    current_executable=None,
    version_info=None,
    platform_name=None,
    execve=None,
):
    """Re-execute this launcher once under the GUI interpreter when needed."""

    values = os.environ if environment is None else environment
    target = _selected_gui_python(
        values,
        package_root=package_root,
        current_executable=current_executable,
        version_info=version_info,
        platform_name=platform_name,
    )
    if target is None:
        return False
    next_environment = dict(values)
    next_environment[_GUI_REEXEC_GUARD] = "1"
    platform = os.name if platform_name is None else str(platform_name)
    local_target = None
    if package_root is not None:
        local_target = _environment_python(
            Path(package_root).expanduser().resolve() / "gui" / ".venv",
            platform,
        )
    if target == local_target and platform != "nt":
        runtime_root = (
            Path(package_root).expanduser().resolve()
            / "gui"
            / ".venv"
            / "qt-runtime"
            / "usr"
            / "lib"
        )
        runtime_directories = []
        if runtime_root.is_dir():
            runtime_directories.append(runtime_root)
            runtime_directories.extend(
                path for path in sorted(runtime_root.iterdir()) if path.is_dir()
            )
        if runtime_directories:
            existing = str(next_environment.get("LD_LIBRARY_PATH", ""))
            existing_entries = [
                value for value in existing.split(os.pathsep) if value
            ]
            entries = [
                str(path)
                for path in runtime_directories
                if str(path) not in existing_entries
            ]
            entries.extend(existing_entries)
            next_environment["LD_LIBRARY_PATH"] = os.pathsep.join(entries)
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
    package_root = _locate_package_root()
    _reexec_gui_python(sys.argv[1:], package_root=package_root)
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
