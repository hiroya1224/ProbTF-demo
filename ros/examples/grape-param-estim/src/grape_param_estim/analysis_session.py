"""Incremental, automatically persisted failed-bag analysis sessions."""

from datetime import datetime
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from grape_param_estim.automatic_analysis import (
    AutomaticAnalysisConfig,
    analyze_bags,
    merge_analysis_results,
)
from grape_param_estim.effective_estimator import write_result


Analyzer = Callable[
    [Iterable[Path], AutomaticAnalysisConfig],
    Mapping,
]


def default_session_directory(now: Optional[datetime] = None) -> Path:
    """Return a durable, human-readable output path under ROS_HOME."""

    ros_home = Path(
        os.environ.get("ROS_HOME", str(Path.home() / ".ros"))
    ).expanduser()
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return (
        ros_home
        / "grape_param_estim"
        / "failure_analysis"
        / timestamp
    )


class IncrementalAnalysisSession:
    """Track a GUI session and analyze each selected bag at most once."""

    def __init__(
        self,
        config: AutomaticAnalysisConfig,
        output_directory,
        analyzer: Analyzer = analyze_bags,
    ):
        self.config = config
        self.output_directory = (
            Path(output_directory).expanduser().resolve()
        )
        self.analysis_path = self.output_directory / "analysis.json"
        self._analyzer = analyzer
        self._paths = []
        self._completed = set()
        self._result = None

    @property
    def paths(self):
        return tuple(self._paths)

    @property
    def pending_paths(self):
        return tuple(
            path
            for path in self._paths
            if path not in self._completed
        )

    @property
    def completed_paths(self):
        return tuple(
            path for path in self._paths if path in self._completed
        )

    @property
    def result(self):
        return self._result

    def add_bags(self, paths: Iterable) -> tuple:
        """Add existing .bag files, retaining the first selected order."""

        added = []
        known = set(self._paths)
        for value in paths:
            path = Path(value).expanduser().resolve()
            if path.suffix.lower() != ".bag":
                raise ValueError(
                    "not a ROS bag file: {}".format(path)
                )
            if not path.is_file():
                raise FileNotFoundError(str(path))
            if path in known:
                continue
            self._paths.append(path)
            known.add(path)
            added.append(path)
        return tuple(added)

    def analyze(self, path) -> Mapping:
        """Analyze one pending bag, append it, and atomically persist JSON."""

        resolved = Path(path).expanduser().resolve()
        if resolved not in self._paths:
            raise ValueError(
                "bag is not part of this session: {}".format(resolved)
            )
        if resolved in self._completed:
            raise ValueError(
                "bag is already analyzed: {}".format(resolved)
            )
        addition = self._analyzer([resolved], self.config)
        if self._result is None:
            updated = addition
        else:
            updated = merge_analysis_results(
                self._result, addition
            )
        write_result(self.analysis_path, updated, overwrite=True)
        self._result = updated
        self._completed.add(resolved)
        return addition["bags"][0]


__all__ = [
    "IncrementalAnalysisSession",
    "default_session_directory",
]
