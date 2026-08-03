import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


_LAUNCHER_PATH = Path(__file__).resolve().parents[1] / "run_gui.py"
_SPEC = importlib.util.spec_from_file_location(
    "grape_param_estim_gui_launcher_test_target", _LAUNCHER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_LAUNCHER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LAUNCHER)


class GuiLauncherInterpreterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _executable(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def test_explicit_gui_python_is_reexecuted_before_imports(self):
        current = self._executable("current/python")
        target = self._executable("gui/python")
        calls = []
        environment = {
            "GRAPE_PARAM_ESTIM_GUI_PYTHON": str(target),
            "VIRTUAL_ENV": str(self.root / "ignored-venv"),
        }
        changed = _LAUNCHER._reexec_gui_python(
            ("--projects-root", "/tmp/projects"),
            environment,
            current_executable=current,
            version_info=(3, 8),
            platform_name="posix",
            execve=lambda *arguments: calls.append(arguments),
        )
        self.assertTrue(changed)
        self.assertEqual(calls[0][0], str(target.resolve()))
        self.assertEqual(calls[0][1][0], str(target.resolve()))
        self.assertEqual(
            calls[0][1][-2:], ["--projects-root", "/tmp/projects"]
        )
        self.assertEqual(
            calls[0][2][_LAUNCHER._GUI_REEXEC_GUARD], "1"
        )

    def test_active_virtual_environment_is_used_for_catkin_shebang(self):
        current = self._executable("catkin/python3")
        target = self._executable("venv/bin/python")
        selected = _LAUNCHER._selected_gui_python(
            {"VIRTUAL_ENV": str(self.root / "venv")},
            current_executable=current,
            version_info=(3, 8),
            platform_name="posix",
        )
        self.assertEqual(selected, target.resolve())

    def test_current_supported_interpreter_needs_no_reexecution(self):
        current = self._executable("python310")
        self.assertIsNone(
            _LAUNCHER._selected_gui_python(
                {},
                current_executable=current,
                version_info=(3, 10),
                platform_name="posix",
            )
        )

    def test_missing_or_non_executable_interpreter_is_rejected(self):
        current = self._executable("current/python")
        missing = self.root / "missing-python"
        with self.assertRaisesRegex(RuntimeError, "not an executable file"):
            _LAUNCHER._selected_gui_python(
                {"GRAPE_PARAM_ESTIM_GUI_PYTHON": str(missing)},
                current_executable=current,
                version_info=(3, 8),
                platform_name="posix",
            )
        non_executable = self.root / "non-executable"
        non_executable.write_text("python", encoding="utf-8")
        non_executable.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, "not an executable file"):
            _LAUNCHER._selected_gui_python(
                {"GRAPE_PARAM_ESTIM_GUI_PYTHON": str(non_executable)},
                current_executable=current,
                version_info=(3, 8),
                platform_name="posix",
            )

    def test_reexecution_guard_prevents_wrapper_loop(self):
        current = self._executable("current/python")
        target = self._executable("wrapper/python")
        with self.assertRaisesRegex(RuntimeError, "did not select"):
            _LAUNCHER._selected_gui_python(
                {
                    "GRAPE_PARAM_ESTIM_GUI_PYTHON": str(target),
                    _LAUNCHER._GUI_REEXEC_GUARD: "1",
                },
                current_executable=current,
                version_info=(3, 10),
                platform_name="posix",
            )

    def test_python38_without_gui_interpreter_has_actionable_error(self):
        current = self._executable("current/python")
        with self.assertRaisesRegex(
            RuntimeError, "GRAPE_PARAM_ESTIM_GUI_PYTHON"
        ):
            _LAUNCHER._selected_gui_python(
                {},
                current_executable=current,
                version_info=(3, 8),
                platform_name="posix",
            )


if __name__ == "__main__":
    unittest.main()
