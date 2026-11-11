#!/usr/bin/env python3
"""
recon10s_tdr3dpanel — Experimental › 3D structure.

A gridded TDR analysis in three dimensions -- reflectivity or wind-speed
isosurfaces, the vortex column (the centre at every height), the tilt and
shear vectors -- and the diagnostics behind them: inner-core shear with a
jackknife uncertainty, SHIPS's environmental shear for comparison, and
where the convection and the tilt sit relative to each shear direction.

A ground-based radar volume (Radar > Radar volume) can be shown too, as
reflectivity isosurfaces: one radar measures only the wind along its beam,
so there is no vortex column or shear for it.
"""
from __future__ import annotations

import math
import warnings
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

import recon10s_tdr as tdr
import recon10s_tdr3d as t3
import recon10s_tdrgrid as tg

# A ground-based radar sees the whole rain shield, so its isosurfaces start higher.
RADAR_PRESET = ("Reflectivity", "dBZ", [(30.0, "#3ec1ff"), (40.0, "#fdd835"), (50.0, "#e53935")])
ISO_PRESETS = {
    "REFLECTIVITY": ("Reflectivity", "dBZ", [(20.0, "#3ec1ff"), (35.0, "#fdd835"), (45.0, "#e53935")]),
    "WIND_SPEED": ("Wind speed", "kt", [(64 / t3.MS_TO_KT, "#80cbc4"), (96 / t3.MS_TO_KT, "#ffb74d"),
                                         (137 / t3.MS_TO_KT, "#e040fb")]),
}


class _Sig(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


class _Job(QRunnable):
    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args, self.sig = fn, args, _Sig()

    @pyqtSlot()
    def run(self):
        try:
            self.sig.done.emit(self.fn(*self.args))
        except Exception as exc:
            self.sig.failed.emit(f"{type(exc).__name__}: {exc}")


class Tdr3dPanel(QWidget):
    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self.theme = theme
        self.pool = QThreadPool.globalInstance()
        self.missions: List[tdr.TdrMission] = []
        self.analyses: List[tg.Analysis] = []
        self.grid: Optional[tg.TdrGrid] = None
        self.column = None
        self.shear = None
        self.ships = None
        self.asym = {}
        self.storm_name = ""

        self.fig3d = Figure(figsize=(8, 7), dpi=100)
        self.canvas3d = FigureCanvasQTAgg(self.fig3d)
        self.fig2d = Figure(figsize=(8, 7), dpi=100, layout="constrained")
        self.canvas2d = FigureCanvasQTAgg(self.fig2d)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.addWidget(self._mission_box(), 1)
        lv.addWidget(self._view_box())
        lv.addWidget(self._shear_box())
        note = QLabel("Experimental. The TDR sees ~150 km around the centre, so its shear is the "
                      "inner-core shear, not the 0\u2013500 km environmental shear SHIPS reports; the "
                      "two can differ, and the consistency checks show which one the structure follows. "
                      "Shear directions are the heading the shear vector points toward (westerly shear = "
                      "090\u00b0); SHIPS's text product prints the opposite (where it comes from).")
        note.setObjectName("hint")
        note.setWordWrap(True)
        lv.addWidget(note)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(460)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.tabs = QTabWidget()
        w3 = QWidget()
        v3 = QVBoxLayout(w3)
        v3.setContentsMargins(0, 0, 0, 0)
        v3.addWidget(NavigationToolbar2QT(self.canvas3d, w3))
        v3.addWidget(self.canvas3d, 1)
        self.tabs.addTab(w3, "3D structure")
        w2 = QWidget()
        v2 = QVBoxLayout(w2)
        v2.setContentsMargins(0, 0, 0, 0)
        v2.addWidget(NavigationToolbar2QT(self.canvas2d, w2))
        v2.addWidget(self.canvas2d, 1)
        self.tabs.addTab(w2, "Shear && tilt")
        self.summary = QLabel("Load an analysis to begin.")
        self.summary.setObjectName("probeReadout")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.tabs, 1)
        rv.addWidget(self.summary)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 1200])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split, 1)
        self._placeholder()

    # ---------------------------------------------------------------- building

    def _mission_box(self):
        box = QGroupBox("TDR analysis")
        f = QFormLayout(box)
        self.chk_follow = QCheckBox("Follow Radar \u203a Airborne TDR")
        self.chk_follow.setChecked(True)
        self.chk_follow.setToolTip("Use whichever analysis is open in the Airborne TDR tab.")
        f.addRow(self.chk_follow)
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
        self.btn_find.setObjectName("primary")
        self.btn_find.clicked.connect(self.find)
        row.addWidget(self.btn_find)
        f.addRow("Look back", row)
        self.list = QListWidget()
        self.list.setMinimumHeight(90)
        self.list.currentRowChanged.connect(self._on_mission)
        f.addRow(self.list)
        self.pass_box = QComboBox()
        self.pass_box.currentIndexChanged.connect(lambda _: self.load_selected())
        f.addRow("Analysis", self.pass_box)
        return box

    def _view_box(self):
        box = QGroupBox("3D view")
        f = QFormLayout(box)
        self.field_box = QComboBox()
        for key, (label, _u, _l) in ISO_PRESETS.items():
            self.field_box.addItem(label, key)
        self.field_box.currentIndexChanged.connect(lambda _: self._on_field())
        f.addRow("Isosurfaces of", self.field_box)
        self.level_checks = []
        lv = QHBoxLayout()
        for i in range(3):
            c = QCheckBox("")
            c.setChecked(i < 2)
            c.toggled.connect(lambda _: self.draw3d())
            self.level_checks.append(c)
            lv.addWidget(c)
        f.addRow("Levels", lv)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(10, 90)
        self.opacity.setValue(35)
        self.opacity.sliderReleased.connect(self.draw3d)
        f.addRow("Opacity", self.opacity)
        opts = QHBoxLayout()
        self.chk_column = QCheckBox("Vortex column")
        self.chk_vectors = QCheckBox("Shear && tilt")
        for c in (self.chk_column, self.chk_vectors):
            c.setChecked(True)
            c.toggled.connect(lambda _: self.draw3d())
            opts.addWidget(c)
        f.addRow(opts)
        views = QHBoxLayout()
        for name, (el, az) in (("Oblique", (22, -60)), ("From south", (4, -90)),
                               ("From west", (4, 180)), ("Top", (89, -90))):
            b = QPushButton(name)
            b.setObjectName("smallButton")
            b.clicked.connect(lambda _=False, e=el, a=az: self._set_view(e, a))
            views.addWidget(b)
        f.addRow(views)
        self._on_field(redraw=False)
        return box

    def _shear_box(self):
        box = QGroupBox("Shear and structure")
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
        f.addRow("Layer", layer)
        self.radius = QSpinBox()
        self.radius.setRange(40, 200)
        self.radius.setSingleStep(10)
        self.radius.setValue(120)
        self.radius.setSuffix(" km")
        f.addRow("Averaging disc", self.radius)
        self.z_asym = QDoubleSpinBox()
        self.z_asym.setRange(0.5, 14.0)
        self.z_asym.setSingleStep(0.5)
        self.z_asym.setValue(3.0)
        self.z_asym.setSuffix(" km")
        f.addRow("Asymmetry height", self.z_asym)
        self.storm_id = QLineEdit()
        self.storm_id.setPlaceholderText("e.g. ep172026 (for SHIPS)")
        f.addRow("Storm ID", self.storm_id)
        b = QPushButton("Recompute")
        b.clicked.connect(self.analyse)
        f.addRow(b)
        return box

    # ------------------------------------------------------------------ data

    def set_storm_name(self, name: str):
        self.storm_name = (name or "").upper()
        self.storm_combo.setItemText(0, f"{self.storm_name.title()} (from Satellite)"
                                     if self.storm_name else "Storm chosen under Satellite")

    def find(self):
        name = "" if self.storm_combo.currentData() == "*" else self.storm_name
        self.btn_find.setEnabled(False)
        job = _Job(tdr.find_missions, name, self.days.value())

        def done(ms):
            self.btn_find.setEnabled(True)
            self.missions = ms
            self.list.clear()
            for m in ms:
                self.list.addItem(m.label)
            if ms:
                self.list.setCurrentRow(0)
        job.sig.done.connect(done)
        job.sig.failed.connect(lambda m: (self.btn_find.setEnabled(True), self.summary.setText(m)))
        self.pool.start(job)

    def _on_mission(self, row):
        if not (0 <= row < len(self.missions)):
            return
        job = _Job(tg.list_analyses, self.missions[row].folder)

        def done(analyses):
            self.analyses = analyses
            self.pass_box.blockSignals(True)
            self.pass_box.clear()
            for a in analyses:
                self.pass_box.addItem(f"{a.time[:2]}:{a.time[2:]}Z pass", a.time)
            self.pass_box.blockSignals(False)
            if analyses:
                self.pass_box.setCurrentIndex(len(analyses) - 1)
                self.load_selected()
        job.sig.done.connect(done)
        job.sig.failed.connect(self.summary.setText)
        self.pool.start(job)

    def load_selected(self):
        i = self.pass_box.currentIndex()
        if 0 <= i < len(self.analyses):
            a = self.analyses[i]
            self.summary.setText(f"Reading {a.xy}\u2026")
            job = _Job(lambda a: tg.load_grid(a.url(a.xy)), a)
            job.sig.done.connect(self.set_grid)
            job.sig.failed.connect(self.summary.setText)
            self.pool.start(job)

    def follow(self, grid):
        """An analysis opened in the Airborne TDR tab."""
        if self.chk_follow.isChecked() and grid is not None and grid is not self.grid:
            self.set_grid(grid)

    def set_grid(self, grid):
        self.grid = grid
        radar = not ("VORT" in grid.fields and "U" in grid.fields)
        i = self.field_box.findData("WIND_SPEED")
        item = self.field_box.model().item(i) if i >= 0 else None
        if item is not None:
            item.setEnabled("WIND_SPEED" in grid.fields)
        if radar and self.field_box.currentData() not in grid.fields:
            self.field_box.setCurrentIndex(self.field_box.findData("REFLECTIVITY"))
        if "REFLECTIVITY" not in grid.fields:                 # a synthetic TDR: winds only
            self.field_box.setCurrentIndex(self.field_box.findData("WIND_SPEED"))
        for c, on in zip(self.level_checks, (False, True, True) if radar else (True, True, False)):
            c.blockSignals(True)
            c.setChecked(on)
            c.blockSignals(False)
        self._on_field(redraw=False)
        if not radar and not self.storm_id.text().strip():
            self._guess_storm_id(grid.storm)
        self.analyse()

    @staticmethod
    def _is_radar(g) -> bool:
        return not ("VORT" in g.fields and "U" in g.fields and "V" in g.fields)

    def _guess_storm_id(self, name: str):
        """Storm IDs from NRL's sector file (every basin), for the SHIPS lookup."""
        try:
            import recon10s_jma as jma
            for s in jma.active_storms():
                if s.name.upper() == (name or "").upper():
                    self.storm_id.setText(s.storm_id)
                    return
        except Exception:
            pass

    def analyse(self):
        g = self.grid
        if g is None:
            return
        if self._is_radar(g):
            self.column, self.shear, self.ships, self.asym = None, None, None, {}
            self.draw3d()
            self.draw2d()
            self._write_summary()
            return
        self.summary.setText("Finding the vortex column and the shear\u2026")
        z_lo, z_hi, rad, z_a = self.z_lo.value(), self.z_hi.value(), float(self.radius.value()), self.z_asym.value()
        sid = self.storm_id.text().strip()

        def work():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                col = t3.vortex_column(g)
                sh = t3.inner_core_shear(g, col, z_lo, z_hi, rad)
                ships = t3.ships_shear(sid, g.centre_time) if (sid and g.centre_time) else None
                asym = {}
                for ref, hd in (("TDR", sh.heading if sh else None), ("SHIPS", ships.heading if ships else None)):
                    if hd is not None:
                        asym[ref] = {n: t3.shear_relative(g, col, n, z_a, hd) for n in ("REFLECTIVITY", "W")
                                     if n in g.fields}
                return col, sh, ships, asym
        job = _Job(work)

        def done(res):
            self.column, self.shear, self.ships, self.asym = res
            self.draw3d()
            self.draw2d()
            self._write_summary()
        job.sig.done.connect(done)
        job.sig.failed.connect(self.summary.setText)
        self.pool.start(job)

    @staticmethod
    def _span(g) -> str:
        """Storm, flight and analysis window; files without times just omit the window."""
        when = f" {g.start:%d %b %H:%M}\u2013{g.end:%H:%MZ}" if (g.start and g.end) else ""
        return f"{g.storm.title()} {g.flight}{when}"

    # --------------------------------------------------------------- summary

    def _write_summary(self):
        g, sh, sp, col = self.grid, self.shear, self.ships, self.column
        if self._is_radar(g):
            self.summary.setText(f"{self._span(g)}: a ground-based radar volume ({g.flight}), centred "
                                 f"{g.centre[0]:.2f}, {g.centre[1]:.2f}. Reflectivity isosurfaces only \u2014 one "
                                 "Doppler radar measures just the wind component along its beam, so there is no "
                                 "vortex column, tilt or shear here (those need the airborne TDR's 3-D winds).")
            return
        lines = [f"{self._span(g)}. "
                 f"Vortex centre found at {int(np.isfinite(col.x).sum())} of {len(col.z)} levels."]
        if getattr(g, "source", "") == "synthetic":
            lines.insert(0, "SYNTHETIC TDR — modelled from recon by the TDR-trained model, not a radar "
                            "observation: the column, tilt and shear below are the model's.")
        tl = t3.tilt(col, self.z_lo.value(), self.z_hi.value())
        if sh:
            lines.append(f"TDR inner-core shear: {sh.summary}. Coverage {sh.cover_lo:.0%} / {sh.cover_hi:.0%}.")
        else:
            lines.append("TDR inner-core shear: not enough winds at one of the levels.")
        if sp:
            far = (f" (200\u2013800 km: {sp.kt_200_800:.0f} kt toward {sp.heading_200_800:03.0f}\u00b0)"
                   if sp.kt_200_800 is not None and sp.heading_200_800 is not None else "")
            lines.append(f"SHIPS {sp.cycle:%d/%HZ} (850\u2013200 hPa, 0\u2013500 km, vortex removed): "
                         f"{sp.kt:.0f} kt toward {sp.heading:03.0f}\u00b0{far}"
                         + (f"; {abs(t3.angle_diff(sh.heading, sp.heading)):.0f}\u00b0 from the TDR direction."
                            if sh else "."))
        elif self.storm_id.text().strip():
            lines.append("SHIPS: no cycle within 6 h (Atlantic and eastern/central Pacific only).")
        if tl:
            lines.append(f"Tilt {self.z_lo.value():g}\u2013{self.z_hi.value():g} km: {tl[0]:.1f} km toward {tl[1]:03.0f}\u00b0.")
        checks = []
        for ref, hd in (("TDR", sh.heading if sh else None), ("SHIPS", sp.heading if sp else None)):
            if hd is None or ref not in self.asym or "REFLECTIVITY" not in self.asym[ref]:
                continue
            a = self.asym[ref]["REFLECTIVITY"]
            bits = []
            if a.wn1_phase is not None:
                side = "left" if a.wn1_phase > 0 else "right"
                where = "downshear" if abs(a.wn1_phase) < 90 else "upshear"
                bits.append(f"reflectivity max {abs(a.wn1_phase):.0f}\u00b0 {side} of downshear ({where}-{side})")
            if tl and tl[0] >= 2.0:
                d = -t3.angle_diff(tl[1], hd)
                bits.append(f"tilt {abs(d):.0f}\u00b0 {'left' if d > 0 else 'right'} of downshear")
            n_ok = 0
            if a.wn1_phase is not None and -30 <= a.wn1_phase <= 150:
                n_ok += 1
            if tl and tl[0] >= 2.0 and -45 <= -t3.angle_diff(tl[1], hd) <= 135:
                n_ok += 1
            n_all = int(a.wn1_phase is not None) + int(bool(tl and tl[0] >= 2.0))
            checks.append(f"Relative to {ref} shear: " + "; ".join(bits) +
                          (f" \u2014 {n_ok} of {n_all} match the downshear-to-downshear-left pattern of "
                           "sheared storms." if n_all else ""))
        lines += checks
        if tl and tl[0] < 2.0:
            lines.append("The column is nearly upright (tilt under 2 km), so tilt says little about direction.")
        self.summary.setText("\n".join(lines))

    # --------------------------------------------------------------- drawing

    def _colours(self):
        dark = self.theme != "light"
        return ("#dfe5ec" if dark else "#1c2129"), ("#12151b" if dark else "#ffffff")

    def _placeholder(self):
        fg, bg = self._colours()
        for fig, cv in ((self.fig3d, self.canvas3d), (self.fig2d, self.canvas2d)):
            fig.clear()
            fig.patch.set_facecolor(bg)
            fig.text(0.5, 0.5, "No TDR analysis loaded", ha="center", va="center", color="#8b95a5")
            cv.draw_idle()

    def _preset(self):
        key = self.field_box.currentData()
        if key == "REFLECTIVITY" and self.grid is not None and self._is_radar(self.grid):
            return RADAR_PRESET
        return ISO_PRESETS[key]

    def _on_field(self, redraw=True):
        key = self.field_box.currentData()
        label, unit, levels = self._preset()
        for c, (lv, colour) in zip(self.level_checks, levels):
            shown = lv * t3.MS_TO_KT if key == "WIND_SPEED" else lv
            c.setText(f"{shown:.0f} {unit}")
            c.setStyleSheet(f"color: {colour};")
        if redraw:
            self.draw3d()

    def _set_view(self, el, az):
        for ax in self.fig3d.axes:
            if hasattr(ax, "view_init"):
                ax.view_init(elev=el, azim=az)
        self.canvas3d.draw_idle()

    def _extent(self) -> float:
        g = self.grid
        refl = g.fields.get("REFLECTIVITY", g.fields.get("WIND_SPEED"))
        cols = np.isfinite(refl).any(axis=0)
        if not cols.any():
            return 120.0
        X, Y = np.meshgrid(g.x, g.y)
        far = 80.0 if self._is_radar(g) else 160.0            # a radar volume: the inner core, not all its reach
        return float(min(far, max(60.0, np.hypot(X, Y)[cols].max())))

    def draw3d(self):
        g = self.grid
        if g is None or (self.column is None and not self._is_radar(g)):
            return
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fg, bg = self._colours()
        fig = self.fig3d
        old = [ax for ax in fig.axes if hasattr(ax, "elev")]
        view = (old[0].elev, old[0].azim) if old else (22, -60)
        fig.clear()
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_facecolor(bg)
        lim = self._extent()
        key = self.field_box.currentData()
        label, unit, levels = self._preset()
        alpha = self.opacity.value() / 100.0
        radar = self._is_radar(g)
        step = 2 if lim > 100 else 1
        smooth = (0.6, 1.5, 1.5) if radar else None
        drawn = []
        for c, (lv, colour) in zip(self.level_checks, levels):
            if not c.isChecked():
                continue
            surf = t3.isosurface(g, key, lv, extent_km=lim, step=step, smooth=smooth)
            if surf is None:
                continue
            verts, faces = surf
            poly = Poly3DCollection(verts[faces], facecolor=colour, edgecolor="none", alpha=alpha)
            ax.add_collection3d(poly)
            drawn.append(c.text())
        col = self.column
        ok = np.isfinite(col.x) if col is not None else np.zeros(0, bool)
        if col is not None and self.chk_column.isChecked() and ok.any():
            ax.plot(col.x[ok], col.y[ok], col.z[ok], color="#ffffff", lw=2.4, zorder=10)
            ax.scatter(col.x[ok], col.y[ok], col.z[ok], c=col.z[ok], cmap="viridis", s=10, depthshade=False)
        if self.chk_vectors.isChecked() and col is not None:
            top = float(np.nanmax(col.z[ok])) if ok.any() else 15.0
            L = lim * 0.55
            for hd, kt, colour, name in ((self.shear.heading if self.shear else None,
                                          self.shear.kt if self.shear else 0, "#ff5252", "TDR shear"),
                                         (self.ships.heading if self.ships else None,
                                          self.ships.kt if self.ships else 0, "#ffd740", "SHIPS shear")):
                if hd is None:
                    continue
                a = math.radians(hd)
                ax.quiver(0, 0, top + 1, L * math.sin(a), L * math.cos(a), 0, color=colour, lw=2.4,
                          arrow_length_ratio=0.15)
                ax.text(L * math.sin(a) * 1.3, L * math.cos(a) * 1.3, top + 2.5, f"{name} {kt:.0f} kt",
                        color=colour, fontsize=8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(0, 18)
        ax.set_box_aspect((1, 1, 0.55), zoom=1.3)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.1, 0.12, 0.15, 0.6) if self.theme != "light" else (0.95, 0.96, 0.97, 1))
            axis.label.set_color(fg)
        ax.tick_params(colors=fg, labelsize=7)
        ax.set_xlabel("km east")
        ax.set_ylabel("km north")
        ax.set_zlabel("height km")
        ax.view_init(elev=view[0], azim=view[1])
        ax.set_title(f"{self._span(g)}: "
                     f"{label.lower()} isosurfaces {', '.join(drawn) or '(none above the chosen levels)'}",
                     color=fg, fontsize=9)
        self.canvas3d.draw_idle()

    def draw2d(self):
        g, col, sh, sp = self.grid, self.column, self.shear, self.ships
        if g is not None and self._is_radar(g):
            fg, bg = self._colours()
            self.fig2d.clear()
            self.fig2d.patch.set_facecolor(bg)
            self.fig2d.text(0.5, 0.5, "No shear or tilt for a ground-based radar volume:\none radar measures only "
                            "the wind along its beam.", ha="center", va="center", color="#8b95a5")
            self.canvas2d.draw_idle()
            return
        if g is None or col is None:
            return
        fg, bg = self._colours()
        fig = self.fig2d
        fig.clear()
        fig.patch.set_facecolor(bg)
        gs = fig.add_gridspec(2, 2)
        ax_h = fig.add_subplot(gs[0, 0])
        ax_p = fig.add_subplot(gs[0, 1])
        ax_q = fig.add_subplot(gs[1, :])
        for ax in (ax_h, ax_p, ax_q):
            ax.set_facecolor("#0b0d11" if self.theme != "light" else "#f4f6f8")
            ax.tick_params(colors=fg, labelsize=7)
            for s in ax.spines.values():
                s.set_color("#5c6675")
            ax.grid(True, color=fg, alpha=0.12)

        # hodograph of the vortex-removed disc-mean wind
        prof = t3.mean_wind_profile_removed(g, col, float(self.radius.value()))
        if len(prof):
            u, v = prof[:, 1] * t3.MS_TO_KT, prof[:, 2] * t3.MS_TO_KT
            ax_h.plot(u, v, color="#8b95a5", lw=1)
            sc = ax_h.scatter(u, v, c=prof[:, 0], cmap="viridis", s=18, zorder=3)
            cb = fig.colorbar(sc, ax=ax_h, shrink=0.8)
            cb.set_label("height km", color=fg, fontsize=7)
            cb.ax.tick_params(labelsize=7, colors=fg)
            if sh:
                lo = np.array(sh.mean_lo) * t3.MS_TO_KT
                hi = np.array(sh.mean_hi) * t3.MS_TO_KT
                ax_h.annotate("", xy=hi, xytext=lo, arrowprops=dict(arrowstyle="-|>", color="#ff5252", lw=2))
            ax_h.axhline(0, color=fg, lw=0.6, alpha=0.4)
            ax_h.axvline(0, color=fg, lw=0.6, alpha=0.4)
            ax_h.set_aspect("equal", adjustable="datalim")
        ax_h.set_title("Mean wind by height, vortex removed (kt)", color=fg, fontsize=9)
        ax_h.set_xlabel("u (east)", color=fg, fontsize=8)
        ax_h.set_ylabel("v (north)", color=fg, fontsize=8)

        # plan: centres by height, low-level reflectivity, vectors
        k = g.level(self.z_asym.value())
        lim = self._extent()
        import recon10s_nexrad as nx
        if "REFLECTIVITY" in g.fields:
            cmap, norm, _t = nx.colour_table("reflectivity")
            under, under_name = g.fields["REFLECTIVITY"][k], "reflectivity"
        else:
            import matplotlib
            from matplotlib.colors import Normalize
            cmap, norm = matplotlib.colormaps["turbo"], Normalize(0, 80)
            under, under_name = g.fields["WIND_SPEED"][k], "wind speed"
        ax_p.pcolormesh(g.x, g.y, np.ma.masked_invalid(under), cmap=cmap, norm=norm, shading="nearest", alpha=0.55)
        ok = np.isfinite(col.x)
        if ok.any():
            ax_p.plot(col.x[ok], col.y[ok], color="#ffffff", lw=1)
            sc = ax_p.scatter(col.x[ok], col.y[ok], c=col.z[ok], cmap="viridis", s=16, zorder=4)
        L = lim * 0.45
        for hd, colour, name in ((sh.heading if sh else None, "#ff5252", "TDR shear"),
                                 (sp.heading if sp else None, "#ffd740", "SHIPS shear")):
            if hd is not None:
                a = math.radians(hd)
                ax_p.annotate("", xy=(L * math.sin(a), L * math.cos(a)), xytext=(0, 0),
                              arrowprops=dict(arrowstyle="-|>", color=colour, lw=2))
                ax_p.text(L * math.sin(a) * 1.35, L * math.cos(a) * 1.35, name, color=colour, fontsize=7,
                          ha="center", va="center", fontweight="bold")
        ax_p.set_xlim(-lim, lim)
        ax_p.set_ylim(-lim, lim)
        ax_p.set_aspect("equal")
        ax_p.set_title(f"Vortex centres by height over {g.z[k]:.1f} km {under_name}", color=fg, fontsize=9)

        # shear-relative quadrants for each reference
        refs = [r for r in ("TDR", "SHIPS") if r in self.asym and "REFLECTIVITY" in self.asym[r]]
        width = 0.38
        xs = np.arange(4)
        for i, ref in enumerate(refs):
            a = self.asym[ref]["REFLECTIVITY"]
            vals = [a.quadrant.get(q, np.nan) for q in t3.QUADRANTS]
            bars = ax_q.bar(xs + (i - (len(refs) - 1) / 2) * width, vals, width,
                            color="#ff5252" if ref == "TDR" else "#ffd740", alpha=0.85, label=f"relative to {ref} shear")
            for b, val in zip(bars, vals):
                if np.isfinite(val):
                    ax_q.text(b.get_x() + b.get_width() / 2, val + 0.3, f"{val:.1f}", ha="center",
                              color=fg, fontsize=7)
        ax_q.set_xticks(xs)
        ax_q.set_xticklabels(["downshear-left", "downshear-right", "upshear-right", "upshear-left"], color=fg)
        finite = [v for r in refs for v in self.asym[r]["REFLECTIVITY"].quadrant.values() if np.isfinite(v)]
        if finite:
            ax_q.set_ylim(min(finite) - 3, max(finite) + 2)
        ax_q.set_ylabel("mean dBZ", color=fg, fontsize=8)
        a0 = self.asym[refs[0]]["REFLECTIVITY"] if refs else None
        ax_q.set_title(f"Reflectivity by shear-relative quadrant, {a0.z:.1f} km, {a0.r1:.0f}\u2013{a0.r2:.0f} km"
                       if a0 else "Reflectivity by shear-relative quadrant", color=fg, fontsize=9)
        if refs:
            leg = ax_q.legend(fontsize=7, facecolor=bg, edgecolor="#5c6675")
            for t in leg.get_texts():
                t.set_color(fg)
        self.canvas2d.draw_idle()

    def set_theme(self, theme):
        self.theme = theme
        self.draw3d()
        self.draw2d()
