#!/usr/bin/env python3
"""
recon10s_tdrevopanel — Experimental › TDR evolution.

Every gridded TDR analysis from one or more P-3 missions, measured the same
way and plotted against time: peak winds, the RMW at 2 and 8 km, the vortex
tilt and the inner-core shear (with its uncertainty), against SHIPS. Each
grid is measured once and remembered (recon10s_tdrevo).
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import recon10s_tdr as tdr
import recon10s_tdrevo as evo
import recon10s_tdrgrid as tg

MISSION_COLOURS = ["#4dd0e1", "#ffb74d", "#ba68c8", "#81c784", "#f06292", "#fff176", "#64b5f6"]
COLUMNS = ["Analysis", "Peak 2 km", "Peak < 5 km", "Mean Vt 2 km", "RMW 2 / 8 km", "Tilt", "Inner-core shear",
           "SHIPS (0–500 km)", "Column"]


class _Sig(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)


class _Job(QRunnable):
    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args, self.sig = fn, args, _Sig()

    @pyqtSlot()
    def run(self):
        try:
            self.sig.done.emit(self.fn(self.sig.status.emit, *self.args))
        except Exception as exc:
            self.sig.failed.emit(f"{type(exc).__name__}: {exc}")


class TdrEvolutionPanel(QWidget):
    request_open = pyqtSignal(str, str)          # mission folder, analysis time (HHMM)

    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self.theme = theme
        self.pool = QThreadPool.globalInstance()
        self.missions: List[tdr.TdrMission] = []
        self.metrics: List[evo.PassMetrics] = []
        self.storm_name = ""
        self._busy = False

        self.figure = Figure(figsize=(8, 8), dpi=100, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.addWidget(self._mission_box(), 1)
        lv.addWidget(self._settings_box())
        note = QLabel("Each pass measured the same way: peak winds and the RMW from the analysis, the tilt "
                      "from the vortex column (centroids of vorticity), and the inner-core shear with the "
                      "vortex removed (Experimental › 3D structure explains both). Shear directions are "
                      "headings the shear points toward. Grids are measured once and remembered.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        lv.addWidget(note)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(460)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        plot = QWidget()
        pv = QVBoxLayout(plot)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(NavigationToolbar2QT(self.canvas, plot))
        pv.addWidget(self.canvas, 1)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setToolTip("Double-click a pass to open it in Radar › Airborne TDR.")
        self.table.cellDoubleClicked.connect(self._on_double_click)
        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(plot)
        right_split.addWidget(self.table)
        right_split.setSizes([700, 220])
        self.summary = QLabel("Find missions, tick the ones to follow, and measure.")
        self.summary.setObjectName("probeReadout")
        self.summary.setWordWrap(True)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(right_split, 1)
        rv.addWidget(self.summary)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 1200])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split, 1)
        self._placeholder("No passes measured yet")

    # ---------------------------------------------------------------- building

    def _mission_box(self):
        box = QGroupBox("P-3 missions")
        f = QFormLayout(box)
        self.storm_combo = QComboBox()
        self.storm_combo.addItem("Storm chosen under Satellite", "")
        self.storm_combo.addItem("Every storm", "*")
        f.addRow("Storm", self.storm_combo)
        row = QHBoxLayout()
        self.days = QSpinBox()
        self.days.setRange(1, 60)
        self.days.setValue(7)
        self.days.setSuffix(" days")
        row.addWidget(self.days)
        self.btn_find = QPushButton("Find missions")
        self.btn_find.clicked.connect(self.find)
        row.addWidget(self.btn_find)
        f.addRow("Look back", row)
        self.mission_list = QListWidget()
        self.mission_list.setMinimumHeight(110)
        f.addRow(self.mission_list)
        self.btn_measure = QPushButton("Measure every pass")
        self.btn_measure.setObjectName("primary")
        self.btn_measure.clicked.connect(self.measure)
        f.addRow(self.btn_measure)
        return box

    def _settings_box(self):
        box = QGroupBox("Diagnostics")
        f = QFormLayout(box)
        layer = QHBoxLayout()
        self.z_lo = QDoubleSpinBox()
        self.z_hi = QDoubleSpinBox()
        for w, v in ((self.z_lo, 2.0), (self.z_hi, 8.0)):
            w.setRange(0.5, 16.0)
            w.setSingleStep(0.5)
            w.setValue(v)
            w.setSuffix(" km")
            layer.addWidget(w)
        f.addRow("Tilt && shear layer", layer)
        self.radius = QSpinBox()
        self.radius.setRange(40, 200)
        self.radius.setSingleStep(10)
        self.radius.setValue(120)
        self.radius.setSuffix(" km")
        f.addRow("Shear disc", self.radius)
        self.chk_ships = QCheckBox("SHIPS shear for comparison")
        self.chk_ships.setChecked(True)
        f.addRow(self.chk_ships)
        self.storm_id = QLineEdit()
        self.storm_id.setPlaceholderText("al092026 (for SHIPS)")
        f.addRow("Storm ID", self.storm_id)
        return box

    # ------------------------------------------------------------------ data

    def set_storm(self, name: str, storm_id: str = ""):
        self.storm_name = (name or "").upper()
        self.storm_combo.setItemText(0, f"{self.storm_name.title()} (from Satellite)"
                                     if self.storm_name else "Storm chosen under Satellite")
        if storm_id and not self.storm_id.text().strip():
            self.storm_id.setText(storm_id)

    def find(self):
        name = "" if self.storm_combo.currentData() == "*" else self.storm_name
        self.btn_find.setEnabled(False)
        self.summary.setText("Looking for P-3 missions on HRD's real-time page…")
        job = _Job(lambda st, n, d: tdr.find_missions(n, d, progress=st), name, self.days.value())
        job.sig.status.connect(self.summary.setText)

        def done(ms):
            self.btn_find.setEnabled(True)
            self.set_missions(ms)
            self.summary.setText(f"{len(ms)} mission(s). Tick the ones to follow and press Measure."
                                 if ms else "No P-3 missions in that window.")
        job.sig.done.connect(done)
        job.sig.failed.connect(lambda m: (self.btn_find.setEnabled(True), self.summary.setText(m)))
        self.pool.start(job)

    def set_missions(self, missions):
        self.missions = list(missions)
        self.mission_list.clear()
        for m in sorted(self.missions, key=lambda m: m.folder):
            it = QListWidgetItem(m.label)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            it.setData(Qt.ItemDataRole.UserRole, m.folder)
            self.mission_list.addItem(it)

    def chosen_folders(self) -> List[str]:
        out = []
        for i in range(self.mission_list.count()):
            it = self.mission_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def measure(self, folders: Optional[List[str]] = None):
        folders = folders or self.chosen_folders()
        if not folders or self._busy:
            if not folders:
                self.summary.setText("Tick at least one mission.")
            return
        self._busy = True
        self.btn_measure.setEnabled(False)
        z_lo, z_hi, rad = self.z_lo.value(), self.z_hi.value(), float(self.radius.value())
        sid = self.storm_id.text().strip() if self.chk_ships.isChecked() else ""

        def work(status, folders):
            analyses = []
            for f in folders:
                status(f"Listing {f}’s analyses…")
                analyses += tg.list_analyses(f)
            if not analyses:
                return []
            ms = evo.evolution(analyses, z_lo, z_hi, rad, progress=status)
            if sid:
                evo.add_ships(ms, sid, progress=status)
            return ms
        job = _Job(work, list(folders))
        job.sig.status.connect(self.summary.setText)

        def done(ms):
            self._busy = False
            self.btn_measure.setEnabled(True)
            self.set_metrics(ms)
        job.sig.done.connect(done)
        job.sig.failed.connect(lambda m: (setattr(self, "_busy", False), self.btn_measure.setEnabled(True),
                                          self.summary.setText(m)))
        self.pool.start(job)

    def set_metrics(self, metrics):
        self.metrics = list(metrics)
        self._fill_table()
        self.redraw()
        bad = [m for m in self.metrics if m.error]
        text = evo.summary(self.metrics) or (f"{len(self.metrics)} pass(es) measured." if self.metrics
                                              else "No gridded analyses found for those missions.")
        if bad:
            text += f" {len(bad)} could not be read: {bad[0].error}"
        self.summary.setText(text)

    # --------------------------------------------------------------- table

    def _fill_table(self):
        t = self.table
        t.setRowCount(0)

        def f(v, unit="", fmt="{:.0f}"):
            return "" if v is None else fmt.format(v) + unit
        for m in self.metrics:
            r = t.rowCount()
            t.insertRow(r)
            if m.error:
                cells = [m.label, m.error] + [""] * (len(COLUMNS) - 2)
            else:
                shear = ""
                if m.shear_kt is not None:
                    shear = (f"{m.shear_kt:.0f}±{m.shear_kt_err or 0:.0f} kt toward {m.shear_heading:03.0f}"
                             f"±{m.shear_heading_err or 0:.0f}° ({m.shear_grade})")
                ships = (f"{m.ships_kt:.0f} kt toward {m.ships_heading:03.0f}°" if m.ships_kt is not None
                         else "")
                tilt = (f"{m.tilt_km:.1f} km toward {m.tilt_heading:03.0f}°" if m.tilt_km is not None else "")
                cells = [m.label, f(m.peak_2km_kt, " kt"),
                         f(m.peak_low_kt, " kt") + (f" at {m.peak_low_z:.1f} km" if m.peak_low_z is not None else ""),
                         f(m.vt_max_2km_kt, " kt"), f"{f(m.rmw_2km)} / {f(m.rmw_8km)} km",
                         tilt, shear, ships, f"{m.column_levels} levels"]
            for c, text in enumerate(cells):
                t.setItem(r, c, QTableWidgetItem(text))

    def _on_double_click(self, row, _col):
        if 0 <= row < len(self.metrics):
            m = self.metrics[row]
            hhmm = m.name.split("_")[1] if "_" in m.name else ""
            self.request_open.emit(m.mission, hhmm)

    # -------------------------------------------------------------- drawing

    def _colours(self):
        dark = self.theme != "light"
        return ("#dfe5ec" if dark else "#1c2129"), ("#12151b" if dark else "#ffffff")

    def _placeholder(self, text):
        fg, bg = self._colours()
        self.figure.clear()
        self.figure.patch.set_facecolor(bg)
        self.figure.text(0.5, 0.5, text, ha="center", va="center", color="#8b95a5")
        self.canvas.draw_idle()

    def redraw(self):
        import matplotlib.dates as mdates
        ms = [m for m in self.metrics if not m.error and m.when is not None]
        if not ms:
            self._placeholder("No passes measured yet")
            return
        fg, bg = self._colours()
        fig = self.figure
        fig.clear()
        fig.patch.set_facecolor(bg)
        axes = [fig.add_subplot(4, 1, k + 1) for k in range(4)]
        for ax in axes[1:]:
            ax.sharex(axes[0])
        for ax in axes:
            ax.set_facecolor("#0b0d11" if self.theme != "light" else "#f4f6f8")
            ax.tick_params(colors=fg, labelsize=7.5)
            for s in ax.spines.values():
                s.set_color("#5c6675")
            ax.grid(True, color=fg, alpha=0.12)
        x = np.array([mdates.date2num(m.when) for m in ms])
        missions = []
        for m in ms:
            if m.mission not in missions:
                missions.append(m.mission)
        colour_of = {mi: MISSION_COLOURS[k % len(MISSION_COLOURS)] for k, mi in enumerate(missions)}
        cols = [colour_of[m.mission] for m in ms]

        def series(attr):
            return np.array([np.nan if getattr(m, attr) is None else getattr(m, attr) for m in ms], float)

        ax = axes[0]
        ax.plot(x, series("peak_2km_kt"), color="#ff9800", lw=1.4, marker="o", ms=4, label="peak at 2 km")
        ax.plot(x, series("peak_low_kt"), color="#ff5252", lw=1.0, ls="--", marker=".", label="peak below 5 km")
        ax.plot(x, series("vt_max_2km_kt"), color="#4dd0e1", lw=1.4, marker="s", ms=3.5,
                label="max azimuthal-mean Vt, 2 km")
        ax.set_ylabel("Wind (kt)", color=fg, fontsize=8)
        ax = axes[1]
        ax.plot(x, series("rmw_2km"), color="#ffd54f", lw=1.4, marker="o", ms=4, label="RMW 2 km")
        ax.plot(x, series("rmw_8km"), color="#ba68c8", lw=1.2, marker="o", ms=3, label="RMW 8 km")
        ax.set_ylabel("RMW (km)", color=fg, fontsize=8)
        ax = axes[2]
        tilt = series("tilt_km")
        ax.plot(x, tilt, color="#81c784", lw=1.4, marker="o", ms=4,
                label=f"tilt {self.z_lo.value():g}–{self.z_hi.value():g} km")
        self._headings(ax, x, tilt, series("tilt_heading"), "#81c784")
        ax.set_ylabel("Tilt (km)", color=fg, fontsize=8)
        ax = axes[3]
        sh = series("shear_kt")
        err = series("shear_kt_err")
        ax.errorbar(x, sh, yerr=np.where(np.isfinite(err), err, 0), color="#ff5252", lw=1.4, marker="o", ms=4,
                    capsize=3, label="TDR inner-core shear (jackknife ±)")
        self._headings(ax, x, sh, series("shear_heading"), "#ff5252")
        ships = series("ships_kt")
        if np.isfinite(ships).any():
            ax.plot(x, ships, color="#ffd740", lw=1.2, ls="--", marker="D", ms=3.5, label="SHIPS 0–500 km")
            self._headings(ax, x, ships, series("ships_heading"), "#ffd740", offset=-14)
        ax.set_ylabel("Shear (kt)", color=fg, fontsize=8)
        for ax in axes:
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(0, lo), hi + 0.25 * (hi - lo))
            leg = ax.legend(fontsize=7, loc="upper left", facecolor=bg, edgecolor="#5c6675", ncol=3)
            for t in leg.get_texts():
                t.set_color(fg)
            for xi, c in zip(x, cols):
                ax.axvline(xi, color=c, lw=0.6, alpha=0.25)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%H:%MZ"))
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        storm = ms[0].storm.title() if ms[0].storm else ""
        miss = ", ".join(missions)
        axes[0].set_title(f"{storm} TDR evolution, {len(ms)} passes ({miss}). Arrows: heading, north up",
                          color=fg, fontsize=9)
        self.canvas.draw_idle()

    @staticmethod
    def _headings(ax, x, y, headings, colour, offset: float = 12):
        """A short arrow per point showing a direction (north up), just above the point."""
        ok = np.isfinite(y) & np.isfinite(headings)
        if not ok.any():
            return
        a = np.radians(headings[ok])
        ax.quiver(x[ok], y[ok], np.sin(a), np.cos(a), angles="uv", color=colour, scale=36, width=0.0035,
                  headwidth=4, headlength=4.5, pivot="tail", zorder=5)

    def set_theme(self, theme: str):
        self.theme = theme
        self.redraw()

    def state(self) -> dict:
        return {"days": self.days.value(), "every": self.storm_combo.currentIndex(), "z_lo": self.z_lo.value(),
                "z_hi": self.z_hi.value(), "radius": self.radius.value(), "ships": self.chk_ships.isChecked()}

    def restore(self, st: dict):
        if not st:
            return
        self.days.setValue(int(st.get("days", 7)))
        self.storm_combo.setCurrentIndex(int(st.get("every", 0)))
        self.z_lo.setValue(float(st.get("z_lo", 2.0)))
        self.z_hi.setValue(float(st.get("z_hi", 8.0)))
        self.radius.setValue(int(st.get("radius", 120)))
        self.chk_ships.setChecked(bool(st.get("ships", True)))
