"""Tkinter application for interactive, incremental failed-bag analysis."""

import argparse
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
import numpy as np

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
        self._enabled_paths = set()
        self._episode_rows = {}
        self._advice_rows = {}
        self._errors = []
        self._run_started = None
        self._run_sizes = {}
        self._run_prefix = {}
        self._run_total_size = 1
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
            columns=("enabled", "status", "name"),
            show="headings",
            selectmode="extended",
            height=18,
        )
        self.bag_tree.heading("enabled", text="解析")
        self.bag_tree.heading("status", text="状態")
        self.bag_tree.heading("name", text="ファイル")
        self.bag_tree.column(
            "enabled", width=48, stretch=False, anchor="center"
        )
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
        self.bag_tree.bind("<Button-1>", self._on_bag_click)
        self.bag_tree.bind("<space>", self._toggle_selected_rows)

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
        self.folder_button = ttk.Button(
            controls,
            text="フォルダを追加…",
            command=self._add_folder,
        )
        self.folder_button.grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        self.select_all_button = ttk.Button(
            controls,
            text="未解析をすべて選択",
            command=lambda: self._select_pending(True),
        )
        self.select_all_button.grid(
            row=1, column=0, sticky="ew", padx=(0, 4), pady=(8, 0)
        )
        self.clear_all_button = ttk.Button(
            controls,
            text="選択を解除",
            command=lambda: self._select_pending(False),
        )
        self.clear_all_button.grid(
            row=1, column=1, sticky="ew", padx=(4, 0), pady=(8, 0)
        )
        self.remove_button = ttk.Button(
            controls,
            text="未解析を一覧から削除",
            command=self._remove_selected,
        )
        self.remove_button.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(8, 0),
        )
        self.run_button = ttk.Button(
            controls,
            text="解析を実行",
            command=self._start_analysis,
        )
        self.run_button.grid(
            row=3,
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
        self._build_advice_tab()
        self._build_plot_tab(
            "実効ゲイン（診断）", "parameters"
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

    def _build_advice_tab(self) -> None:
        self.advice_frame = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(self.advice_frame, text="PID・モデル提案")
        self.advice_frame.rowconfigure(1, weight=1)
        self.advice_frame.rowconfigure(3, weight=1)
        self.advice_frame.columnconfigure(0, weight=1)
        ttk.Label(
            self.advice_frame,
            text=(
                "提案値は記録PIDから求める変更量20%以内の"
                "最初の一手です。機体へ自動適用しません。"
                "係留・安全設備下で一項目ずつ"
                "検証してください。"
            ),
            wraplength=820,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.advice_tree = ttk.Treeview(
            self.advice_frame,
            columns=(
                "bag",
                "episode",
                "group",
                "status",
                "scale",
                "current",
                "proposed",
                "model",
            ),
            show="headings",
            selectmode="browse",
        )
        headings = (
            ("bag", "bag"),
            ("episode", "ep"),
            ("group", "group"),
            ("status", "根拠"),
            ("scale", "応答倍率"),
            ("current", "現在 P/I/D"),
            ("proposed", "提案 P/I/D"),
            ("model", "model 現在→提案"),
        )
        for name, label in headings:
            self.advice_tree.heading(name, text=label)
        self.advice_tree.column("bag", width=175)
        self.advice_tree.column(
            "episode", width=38, stretch=False, anchor="center"
        )
        self.advice_tree.column("group", width=75, stretch=False)
        self.advice_tree.column("status", width=115)
        self.advice_tree.column("scale", width=90)
        self.advice_tree.column("current", width=145)
        self.advice_tree.column("proposed", width=145)
        self.advice_tree.column("model", width=160)
        self.advice_tree.grid(row=1, column=0, sticky="nsew")
        self.advice_tree.bind(
            "<<TreeviewSelect>>", self._on_advice_selected
        )

        ttk.Label(
            self.advice_frame,
            text="識別不能リッジ（選択行）",
            font=("", 10, "bold"),
        ).grid(row=2, column=0, sticky="w", pady=(8, 3))
        self.ridge_text = tk.Text(
            self.advice_frame,
            height=9,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.ridge_text.grid(row=3, column=0, sticky="nsew")

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
        self._register_bags(names)

    def _add_folder(self) -> None:
        directory = filedialog.askdirectory(
            parent=self.root,
            title="ROS bag を含むフォルダを選択",
            mustexist=True,
        )
        if not directory:
            return
        names = tuple(sorted(Path(directory).glob("*.bag")))
        if not names:
            messagebox.showinfo(
                "ROS bag なし",
                "選択したフォルダ直下に .bag がありません。",
                parent=self.root,
            )
            return
        self._register_bags(names)

    def _register_bags(self, names) -> None:
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
                "", tk.END, values=("[x]", "待機", path.name)
            )
            self._tree_paths[item] = path
            self._path_items[path] = item
            self._enabled_paths.add(path)
        if added:
            self.status_text.set(
                "{} 個の bag を追加しました。".format(len(added))
            )
        else:
            self.status_text.set(
                "選択した bag はすでに一覧へ"
                "追加されています。"
            )

    def _set_path_enabled(self, path, enabled: bool) -> None:
        if path in self.session.completed_paths:
            return
        if enabled:
            self._enabled_paths.add(path)
        else:
            self._enabled_paths.discard(path)
        item = self._path_items[path]
        self.bag_tree.set(
            item,
            "enabled",
            "[x]" if path in self._enabled_paths else "[ ]",
        )

    def _on_bag_click(self, event) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        if self.bag_tree.identify_region(
            event.x, event.y
        ) != "cell":
            return
        if self.bag_tree.identify_column(event.x) != "#1":
            return
        item = self.bag_tree.identify_row(event.y)
        if not item:
            return
        path = self._tree_paths[item]
        self._set_path_enabled(
            path, path not in self._enabled_paths
        )

    def _toggle_selected_rows(self, _event=None):
        for item in self.bag_tree.selection():
            path = self._tree_paths[item]
            self._set_path_enabled(
                path, path not in self._enabled_paths
            )
        return "break"

    def _select_pending(self, enabled: bool) -> None:
        for path in self.session.pending_paths:
            if path in self._path_items:
                self._set_path_enabled(path, enabled)
        self.status_text.set(
            "未解析 bag の解析対象を{}しました。".format(
                "選択" if enabled else "解除"
            )
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
            self._enabled_paths.discard(path)
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
            if (
                path in self._path_items
                and path in self._enabled_paths
            )
        )
        if not pending:
            messagebox.showinfo(
                "解析対象なし",
                "解析欄が [x] の未解析 ROS bag を"
                "選んでください。",
                parent=self.root,
            )
            return
        self._errors = []
        self._set_running(True)
        self._run_started = time.monotonic()
        self._run_sizes = {
            path: max(1, path.stat().st_size) for path in pending
        }
        prefix = 0
        self._run_prefix = {}
        for path in pending:
            self._run_prefix[path] = prefix
            prefix += self._run_sizes[path]
        self._run_total_size = max(1, prefix)
        self.progress.configure(
            mode="determinate", maximum=100.0
        )
        self.progress["value"] = 0
        initial_seconds = self._run_total_size / (
            8.0 * 1024.0 * 1024.0
        )
        self.progress_text.set(
            "0% / 残り約 {}".format(
                self._format_duration(initial_seconds)
            )
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
                last_report_time = 0.0

                def report(fraction, phase):
                    nonlocal last_report_time
                    now = time.monotonic()
                    if (
                        now - last_report_time >= 0.08
                        or fraction >= 1.0
                    ):
                        self._events.put(
                            (
                                "progress",
                                index,
                                len(paths),
                                path,
                                float(fraction),
                                str(phase),
                                now,
                            )
                        )
                        last_report_time = now

                bag = self.session.analyze(
                    path, progress_callback=report
                )
            except Exception as error:
                self._events.put(
                    ("error", index, len(paths), path, error)
                )
            else:
                self._events.put(
                    ("completed", index, len(paths), path, bag)
                )
        self._events.put(("finished", len(paths)))

    @staticmethod
    def _format_duration(seconds: float) -> str:
        value = max(0, int(round(seconds)))
        minutes, remainder = divmod(value, 60)
        if minutes:
            return "{}分{:02d}秒".format(minutes, remainder)
        return "{}秒".format(remainder)

    def _update_eta(
        self,
        path,
        bag_fraction: float,
        phase: str,
        now: float,
    ) -> None:
        completed_work = (
            self._run_prefix[path]
            + np.clip(bag_fraction, 0.0, 1.0)
            * self._run_sizes[path]
        )
        overall = float(
            np.clip(
                completed_work / self._run_total_size,
                0.0,
                1.0,
            )
        )
        elapsed = max(0.0, now - self._run_started)
        prior_total = self._run_total_size / (
            8.0 * 1024.0 * 1024.0
        )
        if overall > 1.0e-4:
            observed_total = elapsed / overall
            observed_weight = min(1.0, overall / 0.20)
            predicted_total = (
                (1.0 - observed_weight) * prior_total
                + observed_weight * observed_total
            )
        else:
            predicted_total = prior_total
        remaining = max(0.0, predicted_total - elapsed)
        self.progress["value"] = 100.0 * overall
        self.progress_text.set(
            "{:.0f}% / 残り約 {} / {}".format(
                100.0 * overall,
                self._format_duration(remaining),
                phase,
            )
        )

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
            self._update_eta(
                path,
                0.0,
                "{} / {} bag: 読み込み".format(index, total),
                time.monotonic(),
            )
            self.status_text.set(
                "読み込み、区間検出、推定を"
                "実行しています。"
            )
        elif kind == "progress":
            (
                _,
                index,
                total,
                path,
                fraction,
                phase,
                now,
            ) = event
            self._update_eta(
                path,
                fraction,
                "{} / {} bag: {}".format(index, total, phase),
                now,
            )
        elif kind == "completed":
            _, index, total, path, bag = event
            item = self._path_items[path]
            self._enabled_paths.discard(path)
            self.bag_tree.set(item, "enabled", "済")
            self.bag_tree.set(item, "status", "完了")
            self._update_eta(
                path,
                1.0,
                "{} / {} bag 完了".format(index, total),
                time.monotonic(),
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
            self._update_eta(
                path,
                1.0,
                "{} / {} bag エラー".format(index, total),
                time.monotonic(),
            )
            self._errors.append((path, error))
            self.status_text.set(
                "{}: {}".format(path.name, error)
            )
        elif kind == "finished":
            self._set_running(False)
            self._refresh_results()
            self.progress["value"] = 100.0
            completed = len(self.session.completed_paths)
            self.progress_text.set(
                "100% / 累計 {} bag の解析が完了".format(completed)
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
            if self.session.result is not None:
                self.notebook.select(self.advice_frame)

    def _set_running(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        self.add_button.configure(state=state)
        self.folder_button.configure(state=state)
        self.select_all_button.configure(state=state)
        self.clear_all_button.configure(state=state)
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
        self._refresh_advice_table()
        self._refresh_episode_table()

    @staticmethod
    def _pid_text(gains) -> str:
        if gains is None:
            return "記録なし"
        return "P={:.4g} I={:.4g} D={:.4g}".format(
            gains["p"], gains["i"], gains["d"]
        )

    @staticmethod
    def _advice_status_text(status: str) -> str:
        return {
            "proposal_available": "提案あり",
            "nominal_within_uncertainty": "現状維持",
            "weak_evidence": "根拠不足",
            "not_identifiable": "識別不能",
            "not_available": "記録なし",
        }.get(status, status)

    def _refresh_advice_table(self) -> None:
        for item in self.advice_tree.get_children():
            self.advice_tree.delete(item)
        self._advice_rows.clear()
        result = self.session.result
        if result is None:
            return
        for bag in result["bags"]:
            for episode in bag["episodes"]:
                advice = episode.get("controller_advice", {})
                groups = advice.get("groups", [])
                if not groups:
                    item = self.advice_tree.insert(
                        "",
                        tk.END,
                        values=(
                            Path(bag["path"]).name,
                            episode["episode_index"],
                            "-",
                            self._advice_status_text(
                                advice.get(
                                    "status", "not_available"
                                )
                            ),
                            "-",
                            "-",
                            "-",
                            advice.get("reason", "-"),
                        ),
                    )
                    self._advice_rows[item] = advice
                    continue
                for group in groups:
                    response = group.get("response_scale")
                    response_text = (
                        "-"
                        if response is None
                        else "{:.3g} [{:.3g}, {:.3g}]".format(
                            response["estimate"],
                            response["ci95"][0],
                            response["ci95"][1],
                        )
                    )
                    revision = group.get("minimum_log_change", {})
                    model = group.get("controller_model", {})
                    current_model = model.get("estimate")
                    proposed_model = revision.get(
                        "proposed_controller_model_parameter"
                    )
                    model_text = (
                        "-"
                        if current_model is None
                        else "{:.4g} → {:.4g}".format(
                            current_model, proposed_model
                        )
                    )
                    item = self.advice_tree.insert(
                        "",
                        tk.END,
                        values=(
                            Path(bag["path"]).name,
                            episode["episode_index"],
                            group["group"],
                            self._advice_status_text(
                                group["status"]
                            ),
                            response_text,
                            self._pid_text(
                                group.get("current_pid")
                            ),
                            self._pid_text(
                                revision.get("proposed_pid")
                            ),
                            model_text,
                        ),
                    )
                    self._advice_rows[item] = group
        children = self.advice_tree.get_children()
        if children:
            self.advice_tree.selection_set(children[0])
            self.advice_tree.see(children[0])
            self._on_advice_selected()

    def _on_advice_selected(self, _event=None) -> None:
        selection = self.advice_tree.selection()
        if not selection:
            return
        row = self._advice_rows[selection[0]]
        lines = []
        if (
            "group" not in row
            or "non_identifiability_ridge" not in row
        ):
            lines.append(
                "{}: {}".format(
                    row.get("status", "not_available"),
                    row.get("reason", "提案なし"),
                )
            )
        else:
            lines.append(
                "{} / {}".format(row["group"], row["status"])
            )
            lines.append(
                "観測で一意に決まるのは次の比だけです:"
            )
            ridge = row["non_identifiability_ridge"]
            lines.append("  " + ridge["equation"])
            lines.append(
                "物理parameter倍率 | 物理parameter | "
                "actuator倍率 [95%]"
            )
            for point in ridge["points"]:
                physical = point["physical_parameter"]
                lines.append(
                    "  {:>6.2f} | {:>10} | {:.4g} "
                    "[{:.4g}, {:.4g}]".format(
                        point["physical_parameter_ratio"],
                        (
                            "-"
                            if physical is None
                            else "{:.5g}".format(physical)
                        ),
                        point["actuator_scale"],
                        point["actuator_scale_ci95"][0],
                        point["actuator_scale_ci95"][1],
                    )
                )
            revision = row["minimum_log_change"]
            lines.append("")
            if revision["decision"] == "hold_current_values":
                lines.append(
                    "判断: 95%区間が1を含むか根拠が弱いため、"
                    "現在値を維持します。"
                )
            else:
                lines.append(
                    "最小対数変更の初回ステップ: "
                    "PID ×{:.4g}, model ×{:.4g}; "
                    "変更後予測応答倍率 {:.4g}".format(
                        revision[
                            "recommended_first_step_pid_scale"
                        ],
                        revision[
                            "recommended_first_step_model_scale"
                        ],
                        revision[
                            "predicted_response_scale_after_first_step"
                        ],
                    )
                )
            assumption = row.get("pid_scaling_assumption", {})
            if assumption:
                lines.append(
                    "PID全項同率変更の仮定: feedforward相対RMS "
                    "{:.3g}（feedforward自体は変更しない）".format(
                        assumption[
                            "maximum_feedforward_relative_rms"
                        ]
                    )
                )
        self.ridge_text.configure(state=tk.NORMAL)
        self.ridge_text.delete("1.0", tk.END)
        self.ridge_text.insert("1.0", "\n".join(lines))
        self.ridge_text.configure(state=tk.DISABLED)

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
