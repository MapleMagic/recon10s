#!/usr/bin/env python3
"""
recon10s — PyQt6 front end.

This is the interface for recon10s. The tkinter GUI it replaced was dropped
in 1.2.1; nothing here needs tkinter, including the map, which is pinned to
matplotlib's Qt backend.

  * Live tab: re-reads the IWG1 source on a timer (30 s by default) and plots
    wind, extrapolated MSLP, temperature, dew point, height and static
    pressure at full 1 Hz resolution.
  * Convert tab: writes HDOB, reusing whatever the live tab already holds so
    no second download is needed.

Run:  python recon10s_qt.py
Needs: PyQt6, pyqtgraph, numpy, requests (plus matplotlib/cartopy for the map)
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
import sys
import zipfile
from typing import Any, Dict, Optional

import pyqtgraph as pg
from PyQt6.QtCore import (
    QObject, QRunnable, QThreadPool, QTimer, QUrl, pyqtSignal, pyqtSlot,
)
from PyQt6.QtGui import QAction, QColor, QDesktopServices, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton,
    QSpinBox, QStatusBar, QTabWidget, QVBoxLayout, QWidget,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import recon10s
import recon10s_live
from recon10s_timeseries import TimeSeriesPanel

try:
    # Pin matplotlib to Qt before recon10s_plot imports pyplot. Its default
    # backend is usually TkAgg, and this build no longer assumes tkinter is
    # installed at all.
    import matplotlib
    matplotlib.use("QtAgg")
    import recon10s_plot
except Exception:  # matplotlib/cartopy may not be installed
    recon10s_plot = None

try:
    import requests
except Exception:
    requests = None

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(HERE, "recon10s_settings.json")
GITHUB_LATEST_RELEASE_API = "https://api.github.com/repos/{repo}/releases/latest"

DEFAULTS: Dict[str, Any] = {
    "coord_format": "decimal",
    "gui_theme": "dark",
    "plot_theme": "dark",
    "github_repo": "MapleMagic/recon10s",
    # new in the Qt build
    "iwg1_url": "",
    "iwg1_path": "",
    "mission": "",
    "out_file": "",
    "interval": "30",
    "auto_update": True,
    "auto_update_seconds": 30,
    "plot_window_index": 2,
    "reuse_loaded": True,
    "use_cache": True,
}

_TIME_COLON = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")
_TIME_PLAIN = re.compile(r"^\s*(\d{4}|\d{6})\s*$")


def validate_time_string(s: Optional[str]) -> bool:
    if not s or not s.strip():
        return True
    m = _TIME_COLON.match(s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        ss = int(m.group(3)) if m.group(3) else 0
    else:
        m2 = _TIME_PLAIN.match(s)
        if not m2:
            return False
        tok = m2.group(1)
        hh, mm = int(tok[:2]), int(tok[2:4])
        ss = int(tok[4:6]) if len(tok) == 6 else 0
    return 0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60


# --------------------------------------------------------------- threading

class WorkerSignals(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, object)  # bytes done, bytes total (or None)


class Worker(QRunnable):
    """Runs a callable off the GUI thread and reports back by signal."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            self.signals.done.emit(self.fn(*self.args, **self.kwargs))
        except Exception as exc:
            self.signals.failed.emit(str(exc))


# ------------------------------------------------------------------- window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = dict(DEFAULTS)
        self._load_settings()

        self.pool = QThreadPool.globalInstance()
        self.feed: Optional[recon10s_live.LiveFeed] = None
        self.rows: list = []
        self.series: Optional[dict] = None
        self._polling = False
        self._converting = False
        self._seconds_to_poll = 0

        self.setWindowTitle(f"recon10s {recon10s.VERSION} — IWG1 → HDOB")
        self.resize(1180, 860)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(10)
        root.addWidget(self._build_source_group())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_live_tab(), "Live plot")
        self.tabs.addTab(self._build_convert_tab(), "Convert to HDOB")
        self.tabs.addTab(self._build_settings_tab(), "Settings")
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())
        self.status("Ready.")

        self._build_menu()

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_now)
        self.tick_timer = QTimer(self)
        self.tick_timer.setInterval(1000)
        self.tick_timer.timeout.connect(self._tick)

        self._restore_widgets()
        self._on_auto_toggled(self.chk_auto.isChecked())

    # ------------------------------------------------------------ UI pieces

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        act_settings = QAction("Show settings file", self)
        act_settings.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(SETTINGS_FILE)))
        file_menu.addAction(act_settings)
        file_menu.addSeparator()
        act_quit = QAction("Exit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

    def _build_source_group(self) -> QWidget:
        box = QGroupBox("IWG1 source")
        grid = QGridLayout(box)
        grid.setContentsMargins(12, 10, 12, 12)
        grid.setHorizontalSpacing(10)

        grid.addWidget(QLabel("URL"), 0, 0)
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://…/iwg1/<flight file>")
        self.url_edit.editingFinished.connect(self._on_source_changed)
        grid.addWidget(self.url_edit, 0, 1, 1, 2)

        grid.addWidget(QLabel("Local file"), 1, 0)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("…or a saved IWG1 file (takes priority over the URL)")
        self.path_edit.editingFinished.connect(self._on_source_changed)
        grid.addWidget(self.path_edit, 1, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._choose_iwg1)
        grid.addWidget(browse, 1, 2)

        grid.setColumnStretch(1, 1)
        return box

    def _build_live_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        bar = QHBoxLayout()
        bar.setContentsMargins(10, 0, 10, 0)
        bar.setSpacing(10)

        self.chk_auto = QCheckBox("Auto-update every")
        self.chk_auto.toggled.connect(self._on_auto_toggled)
        bar.addWidget(self.chk_auto)

        self.spin_seconds = QSpinBox()
        self.spin_seconds.setRange(5, 900)
        self.spin_seconds.setValue(30)
        self.spin_seconds.setSuffix(" s")
        self.spin_seconds.valueChanged.connect(self._on_interval_changed)
        bar.addWidget(self.spin_seconds)

        self.btn_refresh = QPushButton("Update now")
        self.btn_refresh.clicked.connect(self.poll_now)
        bar.addWidget(self.btn_refresh)

        self.btn_reload = QPushButton("Reload from start")
        self.btn_reload.setToolTip("Discard what's plotted and re-read the whole file.")
        self.btn_reload.clicked.connect(self.reload_all)
        bar.addWidget(self.btn_reload)

        bar.addStretch(1)
        self.live_status = QLabel("Not started")
        self.live_status.setObjectName("liveStatus")
        bar.addWidget(self.live_status)
        layout.addLayout(bar)

        self.panel = TimeSeriesPanel(theme=self.settings.get("plot_theme", "dark"))
        layout.addWidget(self.panel, 1)
        return page

    def _build_convert_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(10)

        form_box = QWidget()
        form = QFormLayout(form_box)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self.mission_edit = QLineEdit()
        self.mission_edit.setPlaceholderText("AF301 0104A MELISSA — blank to guess from the filename")
        form.addRow("Mission ID", self.mission_edit)

        self.date_edit = QLineEdit(dt.date.today().strftime("%Y%m%d"))
        self.date_edit.setMaximumWidth(140)
        form.addRow("Storm date", self.date_edit)

        self.interval_box = QComboBox()
        self.interval_box.addItems(["5", "10", "30", "60", "120"])
        self.interval_box.setCurrentText("30")
        self.interval_box.setMaximumWidth(140)
        form.addRow("Interval (s)", self.interval_box)

        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        out_row.addWidget(self.out_edit)
        out_btn = QPushButton("Choose…")
        out_btn.clicked.connect(self._choose_output)
        out_row.addWidget(out_btn)
        form.addRow("Output HDOB file", out_row)

        window_row = QHBoxLayout()
        self.start_edit = QLineEdit()
        self.start_edit.setPlaceholderText("HH:MM")
        self.start_edit.setMaximumWidth(120)
        self.end_edit = QLineEdit()
        self.end_edit.setPlaceholderText("HH:MM")
        self.end_edit.setMaximumWidth(120)
        for w in (self.start_edit, self.end_edit):
            w.textChanged.connect(self._validate_times)
        window_row.addWidget(self.start_edit)
        window_row.addWidget(QLabel("to"))
        window_row.addWidget(self.end_edit)
        window_row.addStretch(1)
        form.addRow("UTC window", window_row)

        layout.addWidget(form_box)

        opts = QHBoxLayout()
        self.chk_plot = QCheckBox("Open the map after converting")
        self.chk_plot.setChecked(True)
        self.chk_legend = QCheckBox("Show colour legend")
        self.chk_legend.setChecked(True)
        self.chk_reuse = QCheckBox("Convert the data already loaded in the Live tab")
        self.chk_reuse.setChecked(True)
        self.chk_reuse.setToolTip(
            "Skips the download entirely and converts what is already in memory.\n"
            "Untick to always re-read the source.")
        opts.addWidget(self.chk_plot)
        opts.addWidget(self.chk_legend)
        opts.addWidget(self.chk_reuse)
        opts.addStretch(1)
        layout.addLayout(opts)

        actions = QHBoxLayout()
        self.btn_run = QPushButton("Run conversion")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self.run_conversion)
        actions.addWidget(self.btn_run)
        btn_plot_existing = QPushButton("Plot an existing HDOB…")
        btn_plot_existing.clicked.connect(self._plot_existing)
        actions.addWidget(btn_plot_existing)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(220)
        actions.addWidget(self.progress)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.hdob_text = QPlainTextEdit()
        self.hdob_text.setReadOnly(True)
        mono = QFont()
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setFamilies(["Menlo", "Consolas", "DejaVu Sans Mono", "monospace"])
        mono.setPointSize(10)
        self.hdob_text.setFont(mono)
        self.hdob_text.setPlaceholderText("HDOB output appears here after a conversion.")
        layout.addWidget(self.hdob_text, 1)
        return page

    def _build_settings_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 12, 10, 10)
        layout.setSpacing(14)

        coord_box = QGroupBox("Coordinate display on the map")
        coord_layout = QVBoxLayout(coord_box)
        self.rb_decimal = QRadioButton("Decimal degrees (19.3457)")
        self.rb_dms = QRadioButton("Degrees / minutes / seconds")
        coord_layout.addWidget(self.rb_decimal)
        coord_layout.addWidget(self.rb_dms)
        layout.addWidget(coord_box)

        theme_box = QGroupBox("Appearance")
        theme_layout = QGridLayout(theme_box)
        theme_layout.addWidget(QLabel("Window"), 0, 0)
        self.gui_theme_box = QComboBox()
        self.gui_theme_box.addItems(["dark", "light"])
        self.gui_theme_box.currentTextChanged.connect(self._apply_theme)
        theme_layout.addWidget(self.gui_theme_box, 0, 1)
        theme_layout.addWidget(QLabel("Map plot"), 1, 0)
        self.plot_theme_box = QComboBox()
        self.plot_theme_box.addItems(["dark", "light"])
        self.plot_theme_box.currentTextChanged.connect(
            lambda name: self.panel.set_theme(name))
        theme_layout.addWidget(self.plot_theme_box, 1, 1)
        theme_layout.setColumnStretch(2, 1)
        layout.addWidget(theme_box)

        cache_box = QGroupBox("Download cache")
        cache_layout = QVBoxLayout(cache_box)
        self.chk_cache = QCheckBox("Resume downloads instead of re-fetching the whole file")
        self.chk_cache.setChecked(bool(self.settings.get("use_cache", True)))
        self.chk_cache.setToolTip(
            "IWG1 mission files are appended to as the plane flies, so only the\n"
            "new tail needs downloading. Turn this off if a file is ever rewritten\n"
            "rather than appended to.")
        cache_layout.addWidget(self.chk_cache)
        cache_row = QHBoxLayout()
        self.cache_label = QLabel()
        self.cache_label.setObjectName("hint")
        cache_row.addWidget(self.cache_label)
        cache_row.addStretch(1)
        btn_clear_cache = QPushButton("Clear cache")
        btn_clear_cache.clicked.connect(self._clear_cache)
        cache_row.addWidget(btn_clear_cache)
        cache_layout.addLayout(cache_row)
        layout.addWidget(cache_box)
        self._refresh_cache_label()

        update_box = QGroupBox("Updates")
        update_layout = QHBoxLayout(update_box)
        self.version_label = QLabel(f"Running {self._local_version()} — latest unknown")
        update_layout.addWidget(self.version_label)
        update_layout.addStretch(1)
        btn_check = QPushButton("Check for updates")
        btn_check.clicked.connect(self.check_updates)
        update_layout.addWidget(btn_check)
        layout.addWidget(update_box)

        buttons = QHBoxLayout()
        btn_save = QPushButton("Save settings")
        btn_save.setObjectName("primary")
        btn_save.clicked.connect(self._save_settings_clicked)
        btn_reset = QPushButton("Reset to defaults")
        btn_reset.clicked.connect(self._reset_settings)
        buttons.addWidget(btn_save)
        buttons.addWidget(btn_reset)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        note = QLabel(f"Settings are stored in {SETTINGS_FILE}")
        note.setObjectName("hint")
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    # -------------------------------------------------------------- helpers

    def status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def source(self) -> str:
        path = self.path_edit.text().strip()
        return path or self.url_edit.text().strip()

    def _choose_iwg1(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an IWG1 file", "", "Text files (*.txt *.log *.dat);;All files (*)")
        if path:
            self.path_edit.setText(path)
            self._on_source_changed()

    def _choose_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HDOB output as", self.out_edit.text() or "hdob.txt",
            "Text files (*.txt);;All files (*)")
        if path:
            self.out_edit.setText(path)

    def _validate_times(self):
        ok = True
        for w in (self.start_edit, self.end_edit):
            good = validate_time_string(w.text())
            ok = ok and good
            w.setProperty("invalid", not good)
            w.style().unpolish(w)
            w.style().polish(w)
        self.btn_run.setEnabled(ok and not self._converting)

    def _on_source_changed(self):
        self.feed = None
        self.rows = []
        self.series = None
        self.panel.clear()
        self.live_status.setText("Source changed — press Update now")
        if self.chk_auto.isChecked():
            self.poll_now()

    def _on_interval_changed(self, value: int):
        if self.poll_timer.isActive():
            self.poll_timer.start(value * 1000)
            self._seconds_to_poll = value

    def _on_auto_toggled(self, on: bool):
        if on:
            secs = self.spin_seconds.value()
            self.poll_timer.start(secs * 1000)
            self._seconds_to_poll = secs
            self.tick_timer.start()
            if not self.rows:
                self.poll_now()
        else:
            self.poll_timer.stop()
            self.tick_timer.stop()
            self._update_live_status()

    def _tick(self):
        self._seconds_to_poll = max(0, self._seconds_to_poll - 1)
        self._update_live_status()

    def _update_live_status(self, prefix: Optional[str] = None):
        bits = []
        if prefix:
            bits.append(prefix)
        elif self.rows:
            bits.append(self.panel.latest_text())
        bits.append(f"{len(self.rows):,} obs")
        if self.poll_timer.isActive() and not self._polling:
            bits.append(f"next in {self._seconds_to_poll}s")
        elif self._polling:
            bits.append("updating…")
        self.live_status.setText("   ·   ".join(bits))

    # ------------------------------------------------------------ live poll

    def reload_all(self):
        self.feed = None
        self.rows = []
        self.series = None
        self.panel.clear()
        self.poll_now()

    def poll_now(self):
        if self._polling:
            return
        src = self.source()
        if not src:
            self.live_status.setText("Add a URL or local file above")
            return
        if self.feed is None or self.feed.source != src:
            self.feed = recon10s_live.LiveFeed(
                src, use_cache=self.settings.get("use_cache", True))
            cached = self.feed.cache_file
            if cached and os.path.exists(cached):
                mb = os.path.getsize(cached) / 1e6
                self.status(f"Reusing {mb:.1f} MB already cached for this flight.")

        self._polling = True
        self._update_live_status()
        worker = Worker(self.feed.poll)
        worker.signals.done.connect(self._on_poll_done)
        worker.signals.failed.connect(self._on_poll_failed)
        self.pool.start(worker)

    def _on_poll_done(self, new_rows):
        self._polling = False
        self._seconds_to_poll = self.spin_seconds.value()
        if new_rows:
            self.rows.extend(new_rows)
            self.series = recon10s_live.append_arrays(self.series, new_rows)
            self.panel.set_series(self.series)
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%SZ")
            self.status(f"Added {len(new_rows):,} obs at {stamp}")
        self._update_live_status()

    def _on_poll_failed(self, message: str):
        self._polling = False
        self._seconds_to_poll = self.spin_seconds.value()
        self._update_live_status(prefix="Update failed")
        self.status(f"Could not read the source: {message}")

    # ------------------------------------------------------------- converter

    def run_conversion(self):
        if not self.source():
            QMessageBox.warning(self, "No source",
                                "Add an IWG1 URL or local file at the top of the window.")
            return
        out_file = self.out_edit.text().strip()
        if not out_file:
            QMessageBox.warning(self, "No output file",
                                "Choose where the HDOB text should be written.")
            return
        storm_date = self.date_edit.text().strip()
        if storm_date:
            try:
                dt.datetime.strptime(storm_date, "%Y%m%d")
            except ValueError:
                QMessageBox.warning(self, "Check the storm date",
                                    "Use YYYYMMDD, for example 20250815.")
                return

        path = self.path_edit.text().strip()

        self._converting = True
        self.btn_run.setEnabled(False)

        use_loaded = self.chk_reuse.isChecked() and bool(self.rows)
        if use_loaded:
            self.progress.setRange(0, 0)
            self.status(f"Converting {len(self.rows):,} obs already in memory…")
        else:
            self.progress.setRange(0, 0)
            self.status("Reading the source…")

        params = {
            "rows": list(self.rows) if use_loaded else None,
            "path": path or None,
            "url": None if path else self.url_edit.text().strip(),
            "out_file": out_file,
            "start": self.start_edit.text().strip() or None,
            "end": self.end_edit.text().strip() or None,
            "mission": self.mission_edit.text().strip(),
            "storm_date": storm_date,
            "interval": int(self.interval_box.currentText()),
            "use_cache": self.chk_cache.isChecked(),
            "source_label": self.source(),
        }

        worker = Worker(self._convert, params)
        worker.signals.progress.connect(self._on_download_progress)
        params["progress"] = worker.signals.progress.emit
        worker.signals.done.connect(self._on_convert_done)
        worker.signals.failed.connect(self._on_convert_failed)
        self.pool.start(worker)

    def _on_download_progress(self, done: int, total):
        if total:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(100 * done / total))
            self.status(f"Downloading… {done / 1e6:.1f} of {total / 1e6:.1f} MB")
        else:
            self.status(f"Downloading… {done / 1e6:.1f} MB")

    @staticmethod
    def _convert(params):
        """
        Runs off the GUI thread. Same steps recon10s.main() takes, but able to
        start from rows already in memory and to report download progress.
        """
        start_sec = recon10s._time_input_to_seconds(params["start"]) if params["start"] else None
        end_sec = recon10s._time_input_to_seconds(params["end"]) if params["end"] else None

        rows = params["rows"]
        note = ""
        if rows is None:
            rows = recon10s.read_iwg1(
                params["path"], params["url"], start_sec=start_sec, end_sec=end_sec,
                use_cache=params["use_cache"], progress=params.get("progress"))
        else:
            note = " (no download — reused what the live plot had)"

        rows = recon10s._filter_rows_by_time_of_day(rows, start_sec, end_sec)
        if not rows:
            return {"rc": 3, "text": "", "out_file": params["out_file"],
                    "log": "No observations fall inside that UTC window."}

        mission = params["mission"] or recon10s.auto_mission_from_tail(params["source_label"])
        date = (dt.datetime.strptime(params["storm_date"], "%Y%m%d").date()
                if params["storm_date"] else rows[0].t.date())

        text = recon10s.convert_iwg1_to_hdob(
            rows, mission=mission, storm_date=date, interval_s=params["interval"])
        with open(params["out_file"], "w", encoding="utf-8") as fh:
            fh.write(text)

        return {"rc": 0, "text": text, "out_file": params["out_file"],
                "log": f"Converted {len(rows):,} obs into "
                       f"{len(text.splitlines())} lines{note}."}

    def _on_convert_done(self, result):
        self._converting = False
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self._validate_times()

        if not result["text"]:
            self.status("Conversion produced no output.")
            QMessageBox.warning(self, "Nothing written",
                                result["log"] or "The converter returned no HDOB lines.")
            return

        self.hdob_text.setPlainText(result["text"])
        summary = result["log"].strip().splitlines()
        self.status(summary[-1] if summary else f"Wrote {result['out_file']}")

        if self.chk_plot.isChecked():
            self._plot_hdob(result["out_file"])

    def _on_convert_failed(self, message: str):
        self._converting = False
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self._validate_times()
        self.status("Conversion failed.")
        QMessageBox.critical(self, "Conversion failed", message)

    def _plot_existing(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an HDOB file", "", "Text files (*.txt);;All files (*)")
        if path:
            self._plot_hdob(path)

    def _plot_hdob(self, path: str):
        if recon10s_plot is None:
            QMessageBox.warning(self, "Map unavailable",
                                "recon10s_plot could not be imported. "
                                "Install matplotlib and cartopy to draw the map.")
            return
        try:
            recon10s_plot.main(
                path,
                start_utc=self.start_edit.text().strip() or None,
                end_utc=self.end_edit.text().strip() or None,
                show_legend=self.chk_legend.isChecked(),
                coord_format="dms" if self.rb_dms.isChecked() else "decimal",
                plot_theme=self.plot_theme_box.currentText(),
                show_plot=True,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Map failed", str(exc))

    # --------------------------------------------------------------- updates

    @staticmethod
    def _local_version() -> str:
        """
        Always the version of the code that is actually running. Older builds
        cached this in the settings file, which then went stale and reported
        the wrong version forever; the constant in recon10s.py wins now.
        """
        return recon10s.VERSION

    def _cache_size(self) -> int:
        folder = recon10s.DEFAULT_CACHE_DIR
        if not os.path.isdir(folder):
            return 0
        return sum(os.path.getsize(os.path.join(folder, f))
                   for f in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, f)))

    def _refresh_cache_label(self):
        size = self._cache_size()
        if size:
            self.cache_label.setText(
                f"{size / 1e6:.1f} MB cached in {recon10s.DEFAULT_CACHE_DIR}")
        else:
            self.cache_label.setText(f"Nothing cached yet ({recon10s.DEFAULT_CACHE_DIR})")

    def _clear_cache(self):
        folder = recon10s.DEFAULT_CACHE_DIR
        removed = 0
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                target = os.path.join(folder, name)
                if os.path.isfile(target):
                    try:
                        os.remove(target)
                        removed += 1
                    except OSError:
                        pass
        self.feed = None
        self._refresh_cache_label()
        self.status(f"Cleared {removed} cached file(s).")

    def check_updates(self):
        if requests is None:
            QMessageBox.information(self, "Updates",
                                    "Install the 'requests' package to check for updates.")
            return
        repo = self.settings.get("github_repo", DEFAULTS["github_repo"])
        self.status("Checking GitHub for a newer release…")
        worker = Worker(self._fetch_release, repo)
        worker.signals.done.connect(self._on_release_info)
        worker.signals.failed.connect(
            lambda msg: self.status(f"Update check failed: {msg}"))
        self.pool.start(worker)

    @staticmethod
    def _fetch_release(repo: str):
        r = requests.get(GITHUB_LATEST_RELEASE_API.format(repo=repo), timeout=10,
                         headers={"Accept": "application/vnd.github.v3+json"})
        r.raise_for_status()
        return r.json()

    def _on_release_info(self, data):
        tag = data.get("tag_name") or data.get("name")
        local = self._local_version()
        self.version_label.setText(f"Running {local} — latest {tag or 'unknown'}")
        if not tag:
            self.status("The latest release has no tag.")
            return
        if tag == local:
            self.status(f"Up to date ({local}).")
            return

        answer = QMessageBox.question(
            self, "Update available",
            f"{tag} is out (you have {local}). Download it into a folder you pick?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        page = data.get("html_url")
        if answer != QMessageBox.StandardButton.Yes:
            if page:
                QDesktopServices.openUrl(QUrl(page))
            return

        zip_url = None
        for asset in data.get("assets") or []:
            if str(asset.get("name", "")).lower().endswith(".zip"):
                zip_url = asset.get("browser_download_url")
                break
        zip_url = zip_url or data.get("zipball_url")
        if not zip_url:
            QMessageBox.information(self, "Nothing to download",
                                    "That release has no zip attached.")
            return

        base = QFileDialog.getExistingDirectory(self, f"Where should {tag} go?")
        if not base:
            self.status("Update cancelled.")
            return
        target = os.path.join(base, tag)

        self.status(f"Downloading {tag}…")
        worker = Worker(self._download_release, zip_url, target)
        worker.signals.done.connect(lambda folder: self._on_release_installed(tag, folder))
        worker.signals.failed.connect(
            lambda msg: QMessageBox.critical(self, "Download failed", msg))
        self.pool.start(worker)

    @staticmethod
    def _download_release(zip_url: str, target: str) -> str:
        r = requests.get(zip_url, timeout=180)
        r.raise_for_status()
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(target)
        return target

    def _on_release_installed(self, tag: str, folder: str):
        self.version_label.setText(
            f"Running {self._local_version()} — {tag} downloaded, not yet running")
        self.status(f"{tag} extracted to {folder}")
        QMessageBox.information(self, "Downloaded",
                                f"{tag} is in:\n{folder}\n\nRun it from there to use it.")

    # -------------------------------------------------------------- settings

    def _load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                    stored = json.load(fh)
                for key in DEFAULTS:
                    if key in stored:
                        self.settings[key] = stored[key]
            except (OSError, json.JSONDecodeError):
                pass
        self.settings.pop("current_version", None)  # written by builds <= 1.2.0

    def _write_settings(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
                json.dump(self.settings, fh, indent=2)
        except OSError as exc:
            self.status(f"Could not save settings: {exc}")

    def _collect_settings(self):
        self.settings.update({
            "coord_format": "dms" if self.rb_dms.isChecked() else "decimal",
            "gui_theme": self.gui_theme_box.currentText(),
            "plot_theme": self.plot_theme_box.currentText(),
            "iwg1_url": self.url_edit.text().strip(),
            "iwg1_path": self.path_edit.text().strip(),
            "mission": self.mission_edit.text().strip(),
            "out_file": self.out_edit.text().strip(),
            "interval": self.interval_box.currentText(),
            "auto_update": self.chk_auto.isChecked(),
            "auto_update_seconds": self.spin_seconds.value(),
            "reuse_loaded": self.chk_reuse.isChecked(),
            "use_cache": self.chk_cache.isChecked(),
            "plot_window_index": self.panel.window_box.currentIndex(),
        })

    def _restore_widgets(self):
        s = self.settings
        self.url_edit.setText(s.get("iwg1_url", ""))
        self.path_edit.setText(s.get("iwg1_path", ""))
        self.mission_edit.setText(s.get("mission", ""))
        self.out_edit.setText(s.get("out_file", ""))
        self.interval_box.setCurrentText(str(s.get("interval", "30")))
        self.rb_dms.setChecked(s.get("coord_format") == "dms")
        self.rb_decimal.setChecked(s.get("coord_format") != "dms")
        self.gui_theme_box.setCurrentText(s.get("gui_theme", "dark"))
        self.plot_theme_box.setCurrentText(s.get("plot_theme", "dark"))
        self.spin_seconds.setValue(int(s.get("auto_update_seconds", 30)))
        self.panel.window_box.setCurrentIndex(int(s.get("plot_window_index", 2)))
        self.chk_reuse.setChecked(bool(s.get("reuse_loaded", True)))
        self.chk_cache.setChecked(bool(s.get("use_cache", True)))
        self.chk_auto.setChecked(bool(s.get("auto_update", True)))
        self._validate_times()

    def _save_settings_clicked(self):
        self._collect_settings()
        self._write_settings()
        self.status("Settings saved.")

    def _reset_settings(self):
        if QMessageBox.question(
                self, "Reset settings",
                "Put every setting back to its default?") != QMessageBox.StandardButton.Yes:
            return
        self.settings = dict(DEFAULTS)
        self._write_settings()
        self._restore_widgets()
        self._apply_theme(self.settings["gui_theme"])
        self.status("Settings reset.")

    def closeEvent(self, event):
        self._collect_settings()
        self._write_settings()
        super().closeEvent(event)

    # ----------------------------------------------------------------- theme

    def _apply_theme(self, name: str):
        apply_theme(QApplication.instance(), name)
        self.panel.set_theme(self.plot_theme_box.currentText())


# ------------------------------------------------------------------- styling

DARK_QSS = """
QWidget { background: #0f1216; color: #d5dbe3; font-size: 13px; }
QGroupBox {
    border: 1px solid #232833; border-radius: 6px;
    margin-top: 12px; padding-top: 6px; background: #141821;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #8b95a5; }
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit {
    background: #1a1f28; border: 1px solid #2a303c; border-radius: 4px;
    padding: 5px 7px; selection-background-color: #2f6f7d;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: #4dd0e1; }
QLineEdit[invalid="true"] { border-color: #c0562f; background: #241a17; }
QPushButton {
    background: #212733; border: 1px solid #2f3746; border-radius: 4px;
    padding: 6px 14px;
}
QPushButton:hover { background: #283040; }
QPushButton:disabled { color: #5c6472; background: #191d25; }
QPushButton#primary { background: #1f5c66; border-color: #2b7d8b; color: #eaf7f9; }
QPushButton#primary:hover { background: #256f7b; }
QTabBar::tab {
    background: transparent; padding: 8px 16px; color: #8b95a5;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected { color: #e6edf5; border-bottom: 2px solid #4dd0e1; }
QTabWidget::pane { border: 1px solid #232833; border-radius: 6px; background: #141821; }
QLabel#liveStatus { color: #8b95a5; font-family: Menlo, Consolas, monospace; }
QLabel#hint { color: #6e7787; }
QProgressBar { border: 1px solid #2a303c; border-radius: 4px; background: #1a1f28; height: 8px; }
QProgressBar::chunk { background: #4dd0e1; border-radius: 3px; }
QStatusBar { color: #8b95a5; }
"""

LIGHT_QSS = """
QWidget { background: #f5f6f8; color: #1c2129; font-size: 13px; }
QGroupBox {
    border: 1px solid #d8dbe1; border-radius: 6px; margin-top: 12px;
    padding-top: 6px; background: #ffffff;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #5b6470; }
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit {
    background: #ffffff; border: 1px solid #cfd4dc; border-radius: 4px; padding: 5px 7px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: #1f7f8f; }
QLineEdit[invalid="true"] { border-color: #c0562f; background: #fdf2ee; }
QPushButton { background: #eceef2; border: 1px solid #cfd4dc; border-radius: 4px; padding: 6px 14px; }
QPushButton:hover { background: #e2e5eb; }
QPushButton#primary { background: #1f7f8f; border-color: #1a6b79; color: #ffffff; }
QTabBar::tab { background: transparent; padding: 8px 16px; color: #5b6470; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #12161c; border-bottom: 2px solid #1f7f8f; }
QTabWidget::pane { border: 1px solid #d8dbe1; border-radius: 6px; background: #ffffff; }
QLabel#liveStatus { color: #5b6470; font-family: Menlo, Consolas, monospace; }
QLabel#hint { color: #7a828e; }
"""


def apply_theme(app: QApplication, name: str) -> None:
    app.setStyleSheet(DARK_QSS if name != "light" else LIGHT_QSS)
    palette = app.palette()
    base = QColor("#0f1216") if name != "light" else QColor("#f5f6f8")
    text = QColor("#d5dbe3") if name != "light" else QColor("#1c2129")
    palette.setColor(QPalette.ColorRole.Window, base)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.ToolTipBase, base)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    app.setPalette(palette)


def main() -> int:
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    app.setApplicationName("recon10s")

    stored_theme = "dark"
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                stored_theme = json.load(fh).get("gui_theme", "dark")
        except (OSError, json.JSONDecodeError):
            pass
    apply_theme(app, stored_theme)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
