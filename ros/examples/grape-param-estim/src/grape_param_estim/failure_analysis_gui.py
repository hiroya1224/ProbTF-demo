"""Tkinter application for interactive, incremental failed-bag analysis."""

import argparse
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

from grape_param_estim.analysis_session import (
    IncrementalAnalysisSession,
    default_session_directory,
)
from grape_param_estim.automatic_analysis import load_automatic_config
from grape_param_estim.interactive_plots import (
    draw_parameter_trace,
    draw_placeholder,
    draw_timeline,
    parameter_rows,
)


def _default_config_path() -> Path:
    source = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "automatic_failure_analysis.yaml"
    )
    if source.is_file():
        return source
    import rospkg

    return (
        Path(rospkg.RosPack().get_path("grape_param_estim"))
        / "config"
        / "automatic_failure_analysis.yaml"
    )


def _allocate_output_directory() -> Path:
    base = default_session_directory()
    for index in range(1000):
        candidate = (
            base
            if index == 0
            else base.with_name("{}-{:02d}".format(base.name, index))
        )
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("could not allocate a GUI session directory")


class FailureAnalysisApp:
    """One-window file selection, analysis and result inspection UI."""

    def __init__(
        self,
        root,
        config,
        output_directory,
        session=None,
    ):
        self.root = root
        self.session = session or IncrementalAnalysisSession(
            config, output_directory
        )
        self._events = queue.Queue()
        self._worker = None
        self._tree_paths = {}
        self._path_items = {}
        self._episode_rows = {}
        self._errors = []
        self.status_text = tk.StringVar(
            value="ROS bag を追加して解析を開始してください。"
        )
        self.progress_text = tk.StringVar(value="待機中")
        self.output_text = tk.StringVar(
            value="結果保存先: {}".format(
                self.session.output_directory
            )
        )
        self._build()
        self.root.after(100, self._drain_events)

    def _build(self) -> None:
        self.root.title("Grape failure-bag parameter analysis")
        self.root.geometry("1280x820")
        self.root.minsize(980, 680)

        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        left = ttk.Frame(container)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(1, weight=1)
        ttk.Label(
            left,
            text="解析する ROS bag",
            font=("", 11, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.bag_tree = ttk.Treeview(
            left,
            columns=("status", "name"),
            show="headings",
            selectmode="extended",
            height=18,
        )
        self.bag_tree.heading("status", text="状態")
        self.bag_tree.heading("name", text="ファイル")
        self.bag_tree.column(
            "status", width=82, stretch=False, anchor="center"
        )
        self.bag_tree.column("name", width=310, stretch=True)
        scrollbar = ttk.Scrollbar(
            left, orient=tk.VERTICAL, command=self.bag_tree.yview
        )
        self.bag_tree.configure(yscrollcommand=scrollbar.set)
        self.bag_tree.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.bag_tree.bind(
            "<<TreeviewSelect>>", self._on_bag_selected
        )

        controls = ttk.Frame(left)
        controls.grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=8
        )
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        self.add_button = ttk.Button(
            controls,
            text="bag を追加…",
            command=self._add_bags,
        )
        self.add_button.grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        self.remove_button = ttk.Button(
            controls,
            text="未解析を削除",
            command=self._remove_selected,
        )
        self.remove_button.grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        self.run_button = ttk.Button(
            controls,
            text="解析を実行",
            command=self._start_analysis,
        )
        self.run_button.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(8, 0),
        )

        ttk.Label(
            left,
            textvariable=self.progress_text,
        ).grid(row=3, column=0, columnspan=2, sticky="w")
        self.progress = ttk.Progressbar(
            left, mode="determinate", maximum=1.0
        )
        self.progress.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(3, 8),
        )
        ttk.Label(
            left,
            textvariable=self.status_text,
            wraplength=390,
        ).grid(row=5, column=0, columnspan=2, sticky="ew")
        ttk.Separator(left).grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=8,
        )
        ttk.Label(
            left,
            textvariable=self.output_text,
            wraplength=390,
        ).grid(row=7, column=0, columnspan=2, sticky="ew")
        self.open_button = ttk.Button(
            left,
            text="結果保存先を開く",
            command=self._open_output_directory,
        )
        self.open_button.grid(
            row=8,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(5, 0),
        )

        self.notebook = ttk.Notebook(container)
        self.notebook.grid(row=0, column=1, sticky="nsew")
        self._build_plot_tab(
            "選択 bag の時系列", "timeline"
        )
        self._build_plot_tab(
            "パラメータ推移", "parameters"
        )
        self._build_details_tab()

        draw_placeholder(
            self.timeline_figure,
            "Select a completed bag from the list",
        )
        self.timeline_canvas.draw_idle()
        draw_placeholder(
            self.parameters_figure,
            "Parameter traces appear after analysis",
        )
        self.parameters_canvas.draw_idle()

    def _build_plot_tab(self, title: str, name: str) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        figure = Figure(figsize=(9.0, 7.0), dpi=100)
        canvas = FigureCanvasTkAgg(figure, master=frame)
        toolbar = NavigationToolbar2Tk(
            canvas, frame, pack_toolbar=False
        )
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True
        )
        setattr(self, "{}_figure".format(name), figure)
        setattr(self, "{}_canvas".format(name), canvas)

    def _build_details_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(frame, text="推定値")
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)

        self.episode_tree = ttk.Treeview(
            frame,
            columns=(
                "bag",
                "episode",
                "status",
                "interval",
                "reason",
            ),
            show="headings",
            selectmode="browse",
        )
        headings = (
            ("bag", "bag"),
            ("episode", "episode"),
            ("status", "状態"),
            ("interval", "区間 [s]"),
            ("reason", "理由"),
        )
        for name, label in headings:
            self.episode_tree.heading(name, text=label)
        self.episode_tree.column("bag", width=220)
        self.episode_tree.column(
            "episode", width=70, stretch=False, anchor="center"
        )
        self.episode_tree.column("status", width=120)
        self.episode_tree.column("interval", width=150)
        self.episode_tree.column("reason", width=270)
        self.episode_tree.grid(row=0, column=0, sticky="nsew")
        self.episode_tree.bind(
            "<<TreeviewSelect>>", self._on_episode_selected
        )

        ttk.Label(
            frame,
            text="選択 episode の最終推定値",
            font=("", 10, "bold"),
        ).grid(row=1, column=0, sticky="w", pady=(8, 3))
        self.parameter_tree = ttk.Treeview(
            frame,
            columns=(
                "axis",
                "parameter",
                "estimate",
                "interval",
                "grade",
            ),
            show="headings",
        )
        parameter_headings = (
            ("axis", "axis"),
            ("parameter", "parameter"),
            ("estimate", "estimate"),
            ("interval", "block-bootstrap 95%"),
            ("grade", "grade"),
        )
        for name, label in parameter_headings:
            self.parameter_tree.heading(name, text=label)
        self.parameter_tree.column("axis", width=160)
        self.parameter_tree.column("parameter", width=260)
        self.parameter_tree.column("estimate", width=110)
        self.parameter_tree.column("interval", width=190)
        self.parameter_tree.column("grade", width=110)
        self.parameter_tree.grid(row=2, column=0, sticky="nsew")

    def _add_bags(self) -> None:
        names = filedialog.askopenfilenames(
            parent=self.root,
            title="解析する ROS bag を選択",
            filetypes=(
                ("ROS bag", "*.bag"),
                ("すべてのファイル", "*"),
            ),
        )
        if not names:
            return
        try:
            added = self.session.add_bags(names)
        except (OSError, ValueError) as error:
            messagebox.showerror(
                "bag を追加できません",
                str(error),
                parent=self.root,
            )
            return
        for path in added:
            item = self.bag_tree.insert(
                "", tk.END, values=("待機", path.name)
            )
            self._tree_paths[item] = path
            self._path_items[path] = item
        if added:
            self.status_text.set(
                "{} 個の bag を追加しました。".format(len(added))
            )
        else:
            self.status_text.set(
                "選択した bag はすでに一覧へ"
                "追加されています。"
            )

    def _remove_selected(self) -> None:
        removed = 0
        blocked = 0
        for item in self.bag_tree.selection():
            path = self._tree_paths[item]
            if path in self.session.completed_paths:
                blocked += 1
                continue
            self.bag_tree.delete(item)
            del self._tree_paths[item]
            del self._path_items[path]
            self.session.remove_bags([path])
            removed += 1
        if blocked:
            self.status_text.set(
                "解析済み bag は累積結果に含まれるため"
                "削除できません。"
            )
        elif removed:
            self.status_text.set(
                "{} 個の未解析 bag を"
                "削除しました。".format(removed)
            )

    def _start_analysis(self) -> None:
        pending = tuple(
            path
            for path in self.session.pending_paths
            if path in self._path_items
        )
        if not pending:
            messagebox.showinfo(
                "解析対象なし",
                "未解析の ROS bag を追加してください。",
                parent=self.root,
            )
            return
        self._errors = []
        self._set_running(True)
        self.progress.configure(maximum=len(pending))
        self.progress["value"] = 0
        self.progress_text.set(
            "0 / {} bag 完了".format(len(pending))
        )
        self.status_text.set("解析を開始しています…")
        self._worker = threading.Thread(
            target=self._analysis_worker,
            args=(pending,),
            daemon=True,
        )
        self._worker.start()

    def _analysis_worker(self, paths) -> None:
        for index, path in enumerate(paths, start=1):
            self._events.put(
                ("started", index, len(paths), path)
            )
            try:
                bag = self.session.analyze(path)
            except Exception as error:
                self._events.put(
                    ("error", index, len(paths), path, error)
                )
            else:
                self._events.put(
                    ("completed", index, len(paths), path, bag)
                )
        self._events.put(("finished", len(paths)))

    def _drain_events(self) -> None:
        try:
            while True:
                event = self._events.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _handle_event(self, event) -> None:
        kind = event[0]
        if kind == "started":
            _, index, total, path = event
            item = self._path_items[path]
            self.bag_tree.set(item, "status", "解析中")
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
            self.progress_text.set(
                "{} / {}: {} を解析中".format(
                    index, total, path.name
                )
            )
            self.status_text.set(
                "読み込み、区間検出、推定を"
                "実行しています。"
            )
        elif kind == "completed":
            _, index, total, path, bag = event
            item = self._path_items[path]
            self.bag_tree.set(item, "status", "完了")
            self.progress.stop()
            self.progress.configure(
                mode="determinate", maximum=total
            )
            self.progress["value"] = index
            self.progress_text.set(
                "{} / {} bag 完了".format(index, total)
            )
            self.status_text.set(
                "{}: {} episode を検出しました。".format(
                    path.name, bag["episode_count"]
                )
            )
            self.bag_tree.selection_set(item)
            self.bag_tree.see(item)
            self._refresh_results(path)
        elif kind == "error":
            _, index, total, path, error = event
            item = self._path_items[path]
            self.bag_tree.set(item, "status", "エラー")
            self.progress.stop()
            self.progress.configure(
                mode="determinate", maximum=total
            )
            self.progress["value"] = index
            self.progress_text.set(
                "{} / {} bag 処理済み".format(index, total)
            )
            self._errors.append((path, error))
            self.status_text.set(
                "{}: {}".format(path.name, error)
            )
        elif kind == "finished":
            self.progress.stop()
            self._set_running(False)
            self._refresh_results()
            completed = len(self.session.completed_paths)
            self.progress_text.set(
                "累計 {} bag の解析が完了".format(completed)
            )
            if self._errors:
                summary = "\n".join(
                    "{}: {}".format(path.name, error)
                    for path, error in self._errors
                )
                messagebox.showwarning(
                    "一部の解析に失敗しました",
                    summary,
                    parent=self.root,
                )
            else:
                self.status_text.set(
                    "解析完了。追加の bag を選び、"
                    "同じ結果へ継続できます。"
                )

    def _set_running(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        self.add_button.configure(state=state)
        self.remove_button.configure(state=state)
        self.run_button.configure(state=state)

    def _result_bag(self, path):
        result = self.session.result
        if result is None:
            return None
        target = str(path)
        return next(
            (
                bag
                for bag in result["bags"]
                if str(Path(bag["path"]).resolve()) == target
            ),
            None,
        )

    def _on_bag_selected(self, _event=None) -> None:
        selection = self.bag_tree.selection()
        if not selection:
            return
        path = self._tree_paths[selection[-1]]
        bag = self._result_bag(path)
        if bag is None:
            return
        draw_timeline(self.timeline_figure, bag)
        self.timeline_canvas.draw_idle()

    def _refresh_results(self, selected_path=None) -> None:
        result = self.session.result
        if result is None:
            return
        draw_parameter_trace(self.parameters_figure, result)
        self.parameters_canvas.draw_idle()
        if selected_path is not None:
            bag = self._result_bag(selected_path)
            if bag is not None:
                draw_timeline(self.timeline_figure, bag)
                self.timeline_canvas.draw_idle()
        self._refresh_episode_table()

    def _refresh_episode_table(self) -> None:
        for item in self.episode_tree.get_children():
            self.episode_tree.delete(item)
        self._episode_rows.clear()
        result = self.session.result
        if result is None:
            return
        for bag in result["bags"]:
            for episode in bag["episodes"]:
                item = self.episode_tree.insert(
                    "",
                    tk.END,
                    values=(
                        Path(bag["path"]).name,
                        episode["episode_index"],
                        episode["status"],
                        "{:.3f}–{:.3f}".format(
                            episode["start_s"],
                            episode["end_s"],
                        ),
                        episode["reason"],
                    ),
                )
                self._episode_rows[item] = episode

    def _on_episode_selected(self, _event=None) -> None:
        for item in self.parameter_tree.get_children():
            self.parameter_tree.delete(item)
        selection = self.episode_tree.selection()
        if not selection:
            return
        episode = self._episode_rows[selection[0]]
        for row in parameter_rows(episode):
            self.parameter_tree.insert("", tk.END, values=row)

    def _open_output_directory(self) -> None:
        try:
            subprocess.Popen(
                ["xdg-open", str(self.session.output_directory)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            messagebox.showerror(
                "保存先を開けません",
                str(error),
                parent=self.root,
            )


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="advanced override for the automatic-analysis YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="advanced override for the automatically allocated session",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = _arguments(argv)
    config_path = (
        _default_config_path()
        if arguments.config is None
        else arguments.config
    )
    config = load_automatic_config(config_path)
    if arguments.output_dir is None:
        output_directory = _allocate_output_directory()
    else:
        output_directory = arguments.output_dir.expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    FailureAnalysisApp(root, config, output_directory)
    root.mainloop()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(
            "{}: {}".format(type(error).__name__, error),
            file=sys.stderr,
        )
        sys.exit(2)
