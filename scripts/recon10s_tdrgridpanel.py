#!/usr/bin/env python3
"""
recon10s_tdrgridpanel — the Airborne TDR tab (Radar section): HRD's gridded
tail Doppler radar analyses, drawn by recon10s rather than shown as HRD's
images.

  * Plan view at any height (0-18 km), any field, with wind vectors and the
    radius of maximum wind at that height.
  * Vertical cross-section through the centre at any bearing -- including
    along each flight leg, whose headings the (otherwise empty) real-time leg
    files record.
  * Azimuthal mean (radius-height), with the RMW's slope with height.
  * Overview: all three at once.
  * "Show on the satellite map" puts the plan view, georeferenced, on the
    Satellite tab's imagery.
  * "Draw a line" cuts the vertical section along any line dragged across
    the plan view, not just through the centre.
The same viewer shows ground-based radar volumes gridded by
recon10s_nexgrid (Radar > Radar volume), whose fields are reflectivity,
radial and storm-relative velocity and the dual-pol fields.
"""
from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QPushButton,
    QScrollArea, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

import recon10s_nexgrid as ng
import recon10s_tdr as tdr
import recon10s_tdrgrid as tg

VIEWS = [("Overview", "overview"), ("Plan view", "plan"), ("Cross-section", "section"),
         ("Azimuthal mean", "rz")]


class TdrLayer:
    """A plan view handed to the Satellite map."""

    def __init__(self, lat, lon, values, cmap, norm, label, time=None):
        self.lat, self.lon, self.values = lat, lon, values
        self.cmap, self.norm, self.label, self.time = cmap, norm, label, time


def field_style(name: str, kt: bool):
    """(cmap, norm, unit label, scale) for a field."""
    from matplotlib.colors import Normalize, TwoSlopeNorm
    import matplotlib
    import recon10s_nexrad as nx

    class cm:                                     # matplotlib.cm.get_cmap is gone from 3.9 on
        @staticmethod
        def get_cmap(name):
            return matplotlib.colormaps[name]
    if name == "REFLECTIVITY":
        cmap, norm, _ = nx.colour_table("reflectivity")
        return cmap, norm, "dBZ", 1.0
    if name in ("RADIAL_VELOCITY", "SR_VELOCITY"):                 # radar volumes: already in kt
        cmap, norm, _ = nx.colour_table("velocity")
        return cmap, norm, "kt", 1.0
    if name == "ZDR":
        cmap, norm, _ = nx.colour_table("zdr")
        return cmap, norm, "dB", 1.0
    if name == "CC":
        cmap, norm, _ = nx.colour_table("cc")
        return cmap, norm, "", 1.0
    scale = tg.MS_TO_KT if (kt and name in tg.WINDS) else 1.0
    unit = "kt" if scale != 1.0 else tg.FIELDS[name][1]
    if name in ("WIND_SPEED", "TANGENTIAL_WIND"):
        return cm.get_cmap("turbo"), Normalize(0, 80 * scale), unit, scale
    if name == "RADIAL_WIND":
        return cm.get_cmap("RdBu_r"), TwoSlopeNorm(0, -20 * scale, 20 * scale), unit, scale
    if name == "W":
        return cm.get_cmap("RdBu_r"), TwoSlopeNorm(0, -6, 6), unit, 1.0
    if name == "WIND_SPREAD":                                          # synthetic TDR uncertainty
        return cm.get_cmap("magma"), Normalize(0, 15 * scale), unit, scale
    return cm.get_cmap("magma"), Normalize(0, 15), unit, 1.0          # vorticity


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


class TdrGridPanel(QWidget):
    layer_changed = pyqtSignal(object)      # TdrLayer or None
    grid_loaded = pyqtSignal(object)        # TdrGrid, for Experimental › 3D structure

    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self.theme = theme
        self.pool = QThreadPool.globalInstance()
        self.missions: List[tdr.TdrMission] = []
        self.analyses: List[tg.Analysis] = []
        self.grid: Optional[tg.TdrGrid] = None
        self.legs = {}
        self.storm_name = ""
        self._roles = {}            # axes -> what it shows, for the probe
        self._markers = {}          # axes -> probe marker (animated)
        self._background = None
        self._pinned = False
        self._line = None               # (x0, y0, x1, y1) km: a section along any line
        self._line_start = None

        self.figure = Figure(figsize=(8, 7), dpi=100, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("figure_leave_event", lambda _e: self._clear_probe())

        box = self._build_source_box()

        view = QGroupBox("Display")
        g = QFormLayout(view)
        self.view_box = QComboBox()
        for label, key in VIEWS:
            self.view_box.addItem(label, key)
        self.view_box.currentIndexChanged.connect(lambda _: self.redraw())
        g.addRow("View", self.view_box)
        self.field_box = QComboBox()
        self._set_fields([n for n in tg.FIELDS if n != "WIND_SPREAD"])
        self.field_box.setCurrentIndex(1)
        self.field_box.currentIndexChanged.connect(lambda _: (self.redraw(), self._emit_layer()))
        g.addRow("Field", self.field_box)
        hrow = QHBoxLayout()
        self.height = QSlider(Qt.Orientation.Horizontal)
        self.height.setRange(0, 36)
        self.height.setValue(4)
        self.height.valueChanged.connect(lambda _: (self._height_label(), self.redraw(), self._emit_layer()))
        self.height_label = QLabel("2.0 km")
        self.height_label.setMinimumWidth(52)
        hrow.addWidget(self.height, 1)
        hrow.addWidget(self.height_label)
        g.addRow("Height", hrow)
        arow = QHBoxLayout()
        self.azimuth = QSpinBox()
        self.azimuth.setRange(0, 359)
        self.azimuth.setWrapping(True)
        self.azimuth.setSuffix("\u00b0")
        self.azimuth.valueChanged.connect(lambda _: self.redraw())
        arow.addWidget(self.azimuth)
        self.btn_in = QPushButton("Inbound leg")
        self.btn_out = QPushButton("Outbound leg")
        for b, leg in ((self.btn_in, "inbound"), (self.btn_out, "outbound")):
            b.setObjectName("smallButton")
            b.setEnabled(False)
            b.clicked.connect(lambda _=False, l=leg: self._use_leg(l))
        g.addRow("Section toward", arow)
        legs = QHBoxLayout()
        legs.addWidget(self.btn_in)
        legs.addWidget(self.btn_out)
        g.addRow("", legs)
        lines = QHBoxLayout()
        self.btn_line = QPushButton("Draw a line")
        self.btn_line.setObjectName("smallButton")
        self.btn_line.setCheckable(True)
        self.btn_line.setToolTip("Drag across the plan view: the cross-section follows that line\n"
                                 "instead of passing through the centre.")
        self.btn_line.toggled.connect(self._on_line_mode)
        self.btn_centre_line = QPushButton("Through centre")
        self.btn_centre_line.setObjectName("smallButton")
        self.btn_centre_line.setToolTip("Back to a section through the centre along the bearing above.")
        self.btn_centre_line.clicked.connect(self.clear_line)
        lines.addWidget(self.btn_line)
        lines.addWidget(self.btn_centre_line)
        g.addRow("Section line", lines)
        self.chk_vectors = QCheckBox("Wind vectors")
        self.chk_vectors.setChecked(True)
        self.chk_vectors.toggled.connect(lambda _: self.redraw())
        g.addRow(self.chk_vectors)
        self.chk_kt = QCheckBox("Winds in knots")
        self.chk_kt.setChecked(True)
        self.chk_kt.toggled.connect(lambda _: (self.redraw(), self._emit_layer()))
        g.addRow(self.chk_kt)
        self.chk_probe = QCheckBox("Probe (hover for values, click to pin)")
        self.chk_probe.setChecked(True)
        self.chk_probe.toggled.connect(lambda on: None if on else self._clear_probe(unpin=True))
        g.addRow(self.chk_probe)
        self.chk_map = QCheckBox("Show on the satellite map")
        self.chk_map.setToolTip("Draw this field at this height, georeferenced, over the\n"
                                "Imagery \u203a Satellite loop.")
        self.chk_map.toggled.connect(lambda _: self._emit_layer())
        g.addRow(self.chk_map)

        self.info = QLabel("")
        self.info.setObjectName("hint")
        self.info.setWordWrap(True)
        about = QLabel("These are HRD's gridded analyses from NOAA's flight data server "
                       "(seb.omao.noaa.gov), the numbers behind the TDR graphics tab.")
        about.setObjectName("hint")
        about.setWordWrap(True)
        self.about = about

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.addWidget(box, 1)
        lv.addWidget(view)
        lv.addWidget(self.info)
        lv.addWidget(about)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(460)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(NavigationToolbar2QT(self.canvas, right))
        rv.addWidget(self.canvas, 1)
        self.readout = QLabel(" ")
        self.readout.setObjectName("probeReadout")
        self.readout.setWordWrap(True)
        self.readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rv.addWidget(self.readout)
        self.status = QLabel("Find missions to begin.")
        self.status.setObjectName("hint")
        rv.addWidget(self.status)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([340, 1200])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split, 1)
        self._placeholder("Find missions to begin.")

    def _build_source_box(self) -> QGroupBox:
        box = QGroupBox("Missions")
        f = QFormLayout(box)
        self.storm_combo = QComboBox()
        self.storm_combo.addItem("Storm chosen under Satellite", "")
        self.storm_combo.addItem("Every storm", "*")
        f.addRow("Storm", self.storm_combo)
        self.days = QSpinBox()
        self.days.setRange(1, 60)
        self.days.setValue(7)
        self.days.setSuffix(" days")
        f.addRow("Look back", self.days)
        self.btn_find = QPushButton("Find TDR missions")
        self.btn_find.setObjectName("primary")
        self.btn_find.clicked.connect(self.find)
        f.addRow(self.btn_find)
        self.list = QListWidget()
        self.list.setMinimumHeight(110)
        self.list.currentRowChanged.connect(self._on_mission)
        f.addRow(self.list)
        self.pass_box = QComboBox()
        self.pass_box.currentIndexChanged.connect(lambda _: self.load_grid())
        f.addRow("Analysis", self.pass_box)
        return box

    def _set_fields(self, names):
        """Offer the fields a grid actually has, in a sensible order, keeping the current choice."""
        keep = self.field_box.currentData() if self.field_box.count() else None
        order = [n for n in dict.fromkeys(list(tg.FIELDS) + list(ng.RADAR_FIELDS)) if n in names]
        self.field_box.blockSignals(True)
        self.field_box.clear()
        for key in order:
            self.field_box.addItem(ng.field_label(key)[0], key)
        i = self.field_box.findData(keep)
        if i < 0:
            i = self.field_box.findData("WIND_SPEED")
        if i < 0:
            i = self.field_box.findData("REFLECTIVITY")
        self.field_box.setCurrentIndex(max(0, i))
        self.field_box.blockSignals(False)

    def show_grid(self, grid, legs=None):
        """Display any TdrGrid-shaped grid (an airborne analysis or a gridded radar volume)."""
        self.grid = grid
        self.legs = legs or {}
        self._line = None
        self._set_fields(list(grid.fields))
        winds = grid.has_winds()
        self.chk_vectors.setEnabled(winds)
        for b, leg in ((self.btn_in, "inbound"), (self.btn_out, "outbound")):
            info = self.legs.get(leg)
            b.setEnabled(bool(info and info.heading is not None))
            b.setToolTip(f"Section along the {leg} leg ({info.heading:.0f}\u00b0)" if info and
                         info.heading is not None else "")
        top = int(round(float(np.nanmax(grid.z)) / 0.5))
        self.height.setRange(0, max(1, top))
        self._describe()
        self.redraw()
        self._emit_layer()
        self.grid_loaded.emit(self.grid)

    # ---------------------------------------------------------------- data

    def set_storm_name(self, name: str):
        self.storm_name = (name or "").upper()
        self.storm_combo.setItemText(0, f"{self.storm_name.title()} (from Satellite)"
                                     if self.storm_name else "Storm chosen under Satellite")

    def find(self):
        name = "" if self.storm_combo.currentData() == "*" else self.storm_name
        self.btn_find.setEnabled(False)
        self.status.setText("Looking for P-3 missions\u2026")
        job = _Job(lambda st, n, d: tdr.find_missions(n, d, progress=st), name, self.days.value())
        job.sig.status.connect(self.status.setText)

        def done(ms):
            self.btn_find.setEnabled(True)
            self.missions = ms
            self.list.clear()
            for m in ms:
                self.list.addItem(m.label)
            self.status.setText(f"{len(ms)} mission(s)." if ms else "No TDR missions in that window.")
            if ms:
                self.list.setCurrentRow(0)
        job.sig.done.connect(done)
        job.sig.failed.connect(lambda m: (self.btn_find.setEnabled(True), self.status.setText(m)))
        self.pool.start(job)

    def _on_mission(self, row: int):
        if not (0 <= row < len(self.missions)):
            return
        m = self.missions[row]
        self.status.setText(f"Listing {m.folder}\u2019s analyses\u2026")
        job = _Job(lambda st, f: tg.list_analyses(f), m.folder)

        def done(analyses):
            self.analyses = analyses
            self.pass_box.blockSignals(True)
            self.pass_box.clear()
            for a in analyses:
                self.pass_box.addItem(f"{a.time[:2]}:{a.time[2:]}Z pass", a.time)
            self.pass_box.blockSignals(False)
            if analyses:
                self.pass_box.setCurrentIndex(len(analyses) - 1)
                self.load_grid()
            else:
                self.status.setText(f"No gridded analyses in {m.folder} on the flight data server.")
        job.sig.done.connect(done)
        job.sig.failed.connect(self.status.setText)
        self.pool.start(job)

    def open_analysis(self, mission: str, hhmm: str):
        """Open one pass of a mission directly (from TDR evolution)."""
        self.status.setText(f"Listing {mission}\u2019s analyses\u2026")
        job = _Job(lambda st, f: tg.list_analyses(f), mission)

        def done(analyses):
            self.analyses = analyses
            self.pass_box.blockSignals(True)
            self.pass_box.clear()
            for a in analyses:
                self.pass_box.addItem(f"{a.time[:2]}:{a.time[2:]}Z pass", a.time)
            i = self.pass_box.findData(hhmm)
            self.pass_box.setCurrentIndex(i if i >= 0 else len(analyses) - 1)
            self.pass_box.blockSignals(False)
            if analyses:
                self.load_grid()
        job.sig.done.connect(done)
        job.sig.failed.connect(self.status.setText)
        self.pool.start(job)

    def load_grid(self):
        i = self.pass_box.currentIndex()
        if not (0 <= i < len(self.analyses)):
            return
        a = self.analyses[i]
        self.status.setText(f"Reading {a.xy}\u2026")

        def work(st, a):
            grid = tg.load_grid(a.url(a.xy))
            legs = {}
            for leg, name in a.legs.items():
                try:
                    legs[leg] = tg.leg_info(a.url(name))
                except Exception:
                    pass
            return grid, legs
        job = _Job(work, a)

        def done(res):
            self.show_grid(*res)
        job.sig.done.connect(done)
        job.sig.failed.connect(self.status.setText)
        self.pool.start(job)

    def _use_leg(self, leg: str):
        info = self.legs.get(leg)
        if info and info.heading is not None:
            self.azimuth.setValue(int(round(info.heading)) % 360)
            if self.view_box.currentData() not in ("section", "overview"):
                self.view_box.setCurrentIndex(self.view_box.findData("section"))

    def _describe(self):
        g = self.grid
        if not g.has_winds():
            refl = g.fields.get("REFLECTIVITY")
            mx = float(np.nanmax(refl)) if refl is not None and np.isfinite(refl).any() else float("nan")
            self.info.setText(f"{g.label}. Centre {g.centre[0]:.2f}, {g.centre[1]:.2f}. Peak reflectivity "
                              f"{mx:.0f} dBZ. One radar measures only the wind along its beam, so there are "
                              "no wind vectors, RMW or vorticity here; velocity sections show that component.")
            self.status.setText(f"{g.name}: {len(g.x)} \u00d7 {len(g.y)} \u00d7 {len(g.z)} grid, "
                                f"{g.x[1] - g.x[0]:.0f} km \u00d7 {g.z[1] - g.z[0]:.1f} km.")
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, rmw = g.rmw()
            ws = g.fields["WIND_SPEED"]
            k = int(np.nanargmax(np.nanmax(ws.reshape(ws.shape[0], -1), axis=1)))
        empty = [l for l, i in self.legs.items() if not i.has_data]
        note = (f" The {' and '.join(empty)} leg file{'s are' if len(empty) > 1 else ' is'} empty in "
                "HRD's real-time output, so leg sections are cut from this grid.") if empty else ""
        self.info.setText(
            f"{g.label}. Centre {g.centre[0]:.2f}, {g.centre[1]:.2f}; storm motion "
            f"{np.hypot(*g.motion):.1f} m/s. Peak wind {np.nanmax(ws) * tg.MS_TO_KT:.0f} kt at "
            f"{g.z[k]:.1f} km. RMW {rmw[g.level(2)]:.0f} km at 2 km, {rmw[g.level(8)]:.0f} km at 8 km.{note}")
        self.status.setText(f"{g.name}: {len(g.x)} \u00d7 {len(g.y)} \u00d7 {len(g.z)} grid, "
                            f"{g.x[1] - g.x[0]:.0f} km \u00d7 {g.z[1] - g.z[0]:.1f} km.")

    def _height_label(self):
        self.height_label.setText(f"{self.height.value() * 0.5:.1f} km")

    def _emit_layer(self):
        g = self.grid
        if g is None or not self.chk_map.isChecked():
            self.layer_changed.emit(None)
            return
        name = self.field_box.currentData()
        km = self.height.value() * 0.5
        cmap, norm, unit, scale = field_style(name, self.chk_kt.isChecked())
        self.layer_changed.emit(TdrLayer(g.lat, g.lon, g.plan(name, km) * scale, cmap, norm,
                                         f"{'TDR' if g.has_winds() else 'Radar'} {ng.field_label(name)[0].lower()} "
                                         f"{km:.1f} km ({unit}), "
                                         f"{g.label}", g.centre_time))

    # ------------------------------------------------------------- drawing

    def _colours(self):
        dark = self.theme != "light"
        return ("#dfe5ec" if dark else "#1c2129"), ("#12151b" if dark else "#ffffff")

    def _placeholder(self, text):
        fg, bg = self._colours()
        self.figure.clear()
        self.figure.patch.set_facecolor(bg)
        self.figure.text(0.5, 0.5, text, ha="center", va="center", color="#8b95a5")
        self.canvas.draw_idle()

    def _style(self, ax, fg, bg):
        ax.set_facecolor("#0b0d11" if self.theme != "light" else "#f4f6f8")
        ax.tick_params(colors=fg, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#5c6675")
        ax.grid(True, color=fg, alpha=0.12)

    def redraw(self):
        g = self.grid
        if g is None:
            return
        fg, bg = self._colours()
        fig = self.figure
        fig.clear()
        self._roles, self._markers, self._background = {}, {}, None
        self._pinned = False
        fig.patch.set_facecolor(bg)
        mode = self.view_box.currentData()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if mode == "overview":
                gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
                self._plan(fig.add_subplot(gs[0, 0]), fg, bg)
                self._section(fig.add_subplot(gs[0, 1]), fg, bg)
                self._rz(fig.add_subplot(gs[1, :]), fg, bg)
            elif mode == "plan":
                self._plan(fig.add_subplot(1, 1, 1), fg, bg)
            elif mode == "section":
                self._section(fig.add_subplot(1, 1, 1), fg, bg)
            else:
                self._rz(fig.add_subplot(1, 1, 1), fg, bg)
        fig.suptitle(f"{g.storm.title()}  \u00b7  {g.flight}  \u00b7  "
                     f"{g.start:%d %b %H:%M}\u2013{g.end:%H:%MZ}" if g.start and g.end else g.label,
                     color=fg, fontsize=10)
        self.canvas.draw_idle()

    def _colorbar(self, im, ax, label, fg):
        cb = self.figure.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
        cb.ax.tick_params(labelsize=7, colors=fg)
        cb.set_label(label, color=fg, fontsize=8)

    def _plan(self, ax, fg, bg):
        g = self.grid
        name = self.field_box.currentData()
        km = self.height.value() * 0.5
        cmap, norm, unit, scale = field_style(name, self.chk_kt.isChecked())
        im = ax.pcolormesh(g.x, g.y, np.ma.masked_invalid(g.plan(name, km) * scale), cmap=cmap,
                           norm=norm, shading="nearest")
        self._style(ax, fg, bg)
        if self.chk_vectors.isChecked() and g.has_winds():
            u, v = self._uv(km)
            if u is not None:
                s = 6
                ax.quiver(g.x[::s], g.y[::s], u[::s, ::s], v[::s, ::s], color="#ffffff", alpha=0.7,
                          scale=1400, width=0.0022)
        z, rmw = g.rmw()
        r = rmw[g.level(km)]
        if np.isfinite(r):
            t = np.linspace(0, 2 * np.pi, 181)
            ax.plot(r * np.sin(t), r * np.cos(t), color="#ffffff", lw=1.0, ls="--", alpha=0.8)
            ax.text(0, r, f" RMW {r:.0f} km", color=fg, fontsize=7, va="bottom")
        if self._line is not None:
            x0, y0, x1, y1 = self._line
            ax.plot([x0, x1], [y0, y1], color="#ff4081", lw=1.6)
            for (xx, yy), lab in (((x0, y0), "A"), ((x1, y1), "B")):
                ax.text(xx, yy, f" {lab}", color="#ff4081", fontsize=9, fontweight="bold")
        else:
            a = np.radians(self.azimuth.value())
            ax.plot([-150 * np.sin(a), 150 * np.sin(a)], [-150 * np.cos(a), 150 * np.cos(a)],
                    color="#ff4081", lw=1.0, alpha=0.8)
        site = getattr(g, "site", None)
        if site is not None:
            import recon10s_windfield as wf
            sx, sy = wf.local_xy(g.centre[0], g.centre[1], site[0], site[1])
            ax.plot(sx, sy, marker="^", ms=7, color="#ffffff", mec="#000000", zorder=6)
        ax.set_aspect("equal")
        lim = self._extent_km()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("km east of centre", color=fg, fontsize=8)
        ax.set_ylabel("km north", color=fg, fontsize=8)
        ax.set_title(f"{ng.field_label(name)[0]} at {km:.1f} km", color=fg, fontsize=9)
        self._colorbar(im, ax, unit, fg)
        self._roles[ax] = ("plan", km)

    def _uv(self, km):
        g = self.grid
        if "U" in g.fields and "V" in g.fields:
            return g.plan("U", km), g.plan("V", km)
        if "TANGENTIAL_WIND" not in g.fields or "RADIAL_WIND" not in g.fields:
            return None, None
        X, Y = np.meshgrid(g.x, g.y)
        th = np.arctan2(Y, X)
        vt, vr = g.plan("TANGENTIAL_WIND", km), g.plan("RADIAL_WIND", km)
        u = vr * np.cos(th) - vt * np.sin(th)
        v = vr * np.sin(th) + vt * np.cos(th)
        return u, v

    def _extent_km(self) -> float:
        g = self.grid
        ws = g.fields.get("REFLECTIVITY")
        if ws is None:
            ws = g.fields.get("WIND_SPEED")
        if ws is None:
            return 150.0
        any_col = np.isfinite(ws).any(axis=0)
        if not any_col.any():
            return 150.0
        X, Y = np.meshgrid(g.x, g.y)
        if not g.has_winds():                      # a radar volume: frame the square of data, not its corners
            return float(min(250.0, max(40.0, np.maximum(np.abs(X), np.abs(Y))[any_col].max() + 5)))
        return float(min(250.0, max(60.0, np.hypot(X, Y)[any_col].max() + 10)))

    def _section(self, ax, fg, bg):
        g = self.grid
        name = self.field_box.currentData()
        cmap, norm, unit, scale = field_style(name, self.chk_kt.isChecked())
        top = float(min(18.0, np.nanmax(g.z) + 0.5))
        if self._line is not None:
            x0, y0, x1, y1 = self._line
            r, z, sec = g.section_line(name, x0, y0, x1, y1)
            im = ax.pcolormesh(r, z, np.ma.masked_invalid(sec * scale), cmap=cmap, norm=norm, shading="nearest")
            self._style(ax, fg, bg)
            ax.set_ylim(0, top)
            brg = (np.degrees(np.arctan2(x1 - x0, y1 - y0)) + 360) % 360
            ax.set_xlabel(f"km along the line A \u2192 B (toward {brg:03.0f}\u00b0)", color=fg, fontsize=8)
            ax.set_ylabel("Height (km)", color=fg, fontsize=8)
            ax.set_title(f"{ng.field_label(name)[0]} section A\u2013B", color=fg, fontsize=9)
            self._colorbar(im, ax, unit, fg)
            self._roles[ax] = ("line", x0, y0, x1, y1)
            return
        lim = self._extent_km()
        r, z, sec = g.section(name, self.azimuth.value(), max_r=lim)
        im = ax.pcolormesh(r, z, np.ma.masked_invalid(sec * scale), cmap=cmap, norm=norm, shading="nearest")
        self._style(ax, fg, bg)
        ax.axvline(0, color=fg, lw=0.8, alpha=0.5)
        ax.set_ylim(0, top)
        ax.set_xlabel(f"km from centre toward {self.azimuth.value():03d}\u00b0", color=fg, fontsize=8)
        ax.set_ylabel("Height (km)", color=fg, fontsize=8)
        ax.set_title(f"{ng.field_label(name)[0]} section", color=fg, fontsize=9)
        self._colorbar(im, ax, unit, fg)
        self._roles[ax] = ("section", float(self.azimuth.value()))

    def _rz(self, ax, fg, bg):
        g = self.grid
        name = self.field_box.currentData()
        cmap, norm, unit, scale = field_style(name, self.chk_kt.isChecked())
        r, z, m = g.azimuthal_mean(name)
        im = ax.pcolormesh(r, z, np.ma.masked_invalid(m * scale), cmap=cmap, norm=norm, shading="nearest")
        zz, rmw = g.rmw()
        if np.isfinite(rmw).any():
            ax.plot(rmw, zz, color="#ffffff", lw=1.4, ls="--", label="RMW")
        self._style(ax, fg, bg)
        ax.set_xlim(0, min(150, self._extent_km()))
        ax.set_ylim(0, float(min(18.0, np.nanmax(g.z) + 0.5)))
        ax.set_xlabel("Radius (km)", color=fg, fontsize=8)
        ax.set_ylabel("Height (km)", color=fg, fontsize=8)
        ax.set_title(f"Azimuthal-mean {ng.field_label(name)[0].lower()}", color=fg, fontsize=9)
        if np.isfinite(rmw).any():
            leg = ax.legend(fontsize=7, loc="upper right", facecolor=bg, edgecolor="#5c6675")
            for t in leg.get_texts():
                t.set_color(fg)
        self._colorbar(im, ax, unit, fg)
        self._roles[ax] = ("rz", r, z, m, zz, rmw)

    # ----------------------------------------------------------------- probe

    def _fmt(self, name: str, v: float) -> str:
        """One field's value in the display units."""
        if not np.isfinite(v):
            return "\u2014"
        if name in tg.WINDS:
            return f"{v * tg.MS_TO_KT:.0f} kt" if self.chk_kt.isChecked() else f"{v:.1f} m/s"
        if name == "REFLECTIVITY":
            return f"{v:.1f} dBZ"
        if name == "W":
            return f"{v:+.1f} m/s"
        if name in ("RADIAL_VELOCITY", "SR_VELOCITY"):
            return f"{v:+.0f} kt"
        if name == "ZDR":
            return f"{v:.2f} dB"
        if name == "CC":
            return f"{v:.3f}"
        return f"{v:.1f} \u00d710\u207b\u00b3 s\u207b\u00b9"

    def _point_text(self, x: float, y: float, z: float) -> Optional[str]:
        vals = self.grid.values_at(x, y, z)
        if vals is None:
            return None
        r = float(np.hypot(x, y))
        brg = (np.degrees(np.arctan2(x, y)) + 360) % 360
        lat, lon = vals["lat"], vals["lon"]
        where = (f"{r:.0f} km from centre toward {brg:03.0f}\u00b0  \u00b7  "
                 f"{abs(lat):.2f}{'N' if lat >= 0 else 'S'} {abs(lon):.2f}{'W' if lon < 0 else 'E'}  \u00b7  "
                 f"{vals['level']:.1f} km")
        order = ("WIND_SPEED", "WIND_SPREAD", "TANGENTIAL_WIND", "RADIAL_WIND", "W", "REFLECTIVITY", "VORT",
                 "RADIAL_VELOCITY", "SR_VELOCITY", "ZDR", "CC")
        short = {"WIND_SPEED": "Wind", "WIND_SPREAD": "\u00b1", "TANGENTIAL_WIND": "Vt", "RADIAL_WIND": "Vr",
                 "W": "w", "REFLECTIVITY": "", "VORT": "\u03b6", "RADIAL_VELOCITY": "Vrad", "SR_VELOCITY": "SRV",
                 "ZDR": "ZDR", "CC": "CC"}
        bits = [f"{short[n]} {self._fmt(n, vals[n])}".strip() for n in order if n in vals]
        if all(not np.isfinite(vals[n]) for n in order if n in vals):
            return where + "  \u00b7  no data here"
        return where + "\n" + "   ".join(bits)

    def probe_text(self, ax, xdata: float, ydata: float) -> Optional[str]:
        """What the probe reports at a point of one of the panels."""
        role = self._roles.get(ax)
        if role is None or self.grid is None:
            return None
        if role[0] == "plan":
            return self._point_text(xdata, ydata, role[1])
        if role[0] == "line":
            _r, x0, y0, x1, y1 = role
            length = float(np.hypot(x1 - x0, y1 - y0)) or 1.0
            px, py = x0 + xdata * (x1 - x0) / length, y0 + xdata * (y1 - y0) / length
            head = f"Section A\u2013B: {xdata:.0f} km from A, height {ydata:.1f} km\n"
            txt = self._point_text(px, py, ydata)
            return head + txt if txt else None
        if role[0] == "section":
            a = np.radians(role[1])
            side = "toward" if xdata >= 0 else "away from"
            head = f"Section: {abs(xdata):.0f} km {side} {role[1]:03.0f}\u00b0, height {ydata:.1f} km\n"
            txt = self._point_text(xdata * np.sin(a), xdata * np.cos(a), ydata)
            return head + txt if txt else None
        _, r, z, m, zz, rmw = role
        if not (r[0] - 1 <= xdata <= r[-1] + 1 and 0 <= ydata <= z[-1]):
            return None
        i = int(np.argmin(np.abs(r - xdata)))
        k = int(np.argmin(np.abs(z - ydata)))
        name = self.field_box.currentData()
        extra = ""
        if name != "TANGENTIAL_WIND" and "TANGENTIAL_WIND" in self.grid.fields:
            _r, _z, vt = self.grid.azimuthal_mean("TANGENTIAL_WIND")
            extra = f"   mean Vt {self._fmt('TANGENTIAL_WIND', vt[k, i])}"
        rm = rmw[k] if k < len(rmw) else np.nan
        return (f"Azimuthal mean at radius {r[i]:.0f} km, height {z[k]:.1f} km\n"
                f"mean {ng.field_label(name)[0].lower()} {self._fmt(name, m[k, i])}{extra}"
                + (f"   RMW at this height {rm:.0f} km" if np.isfinite(rm) else ""))

    def _on_draw(self, _event):
        self._background = self.canvas.copy_from_bbox(self.figure.bbox)
        for ax, mk in self._markers.items():
            if mk.get_visible():
                ax.draw_artist(mk)

    def _marker(self, ax):
        if ax not in self._markers:
            self._markers[ax], = ax.plot([], [], marker="o", ms=9, mfc="none", mec="#ffffff",
                                        mew=1.6, animated=True, zorder=30)
        return self._markers[ax]

    def _show_marker(self, ax, x, y):
        if self._background is None:
            return
        self.canvas.restore_region(self._background)
        for other, mk in self._markers.items():
            mk.set_visible(other is ax)
        mk = self._marker(ax)
        mk.set_data([x], [y])
        mk.set_visible(True)
        ax.draw_artist(mk)
        self.canvas.blit(self.figure.bbox)

    def _clear_probe(self, unpin: bool = False):
        if unpin:
            self._pinned = False
        if self._pinned:
            return
        for mk in self._markers.values():
            mk.set_visible(False)
        if self._background is not None:
            self.canvas.restore_region(self._background)
            self.canvas.blit(self.figure.bbox)
        self.readout.setText(" ")

    def _busy_toolbar(self) -> bool:
        tb = self.findChild(NavigationToolbar2QT)
        return bool(tb and getattr(tb, "mode", ""))

    def _on_move(self, event):
        if not self.chk_probe.isChecked() or self._pinned or self._busy_toolbar():
            return
        if event.inaxes not in self._roles or event.xdata is None:
            self._clear_probe()
            return
        text = self.probe_text(event.inaxes, event.xdata, event.ydata)
        if text is None:
            self._clear_probe()
            return
        self.readout.setText(text)
        self._show_marker(event.inaxes, event.xdata, event.ydata)

    def _on_line_mode(self, on: bool):
        self._line_start = None
        self.status.setText("Drag across the plan view to set the section line." if on else self.status.text())
        if on and self.view_box.currentData() == "section":
            self.view_box.setCurrentIndex(self.view_box.findData("overview"))

    def clear_line(self):
        self._line = None
        self.btn_line.setChecked(False)
        self.redraw()

    def set_line(self, x0, y0, x1, y1):
        """Section along the line from (x0, y0) to (x1, y1) km."""
        self._line = (float(x0), float(y0), float(x1), float(y1))
        self.redraw()

    def _on_release(self, event):
        if not self.btn_line.isChecked() or self._line_start is None or event.xdata is None:
            return
        role = self._roles.get(event.inaxes)
        x0, y0 = self._line_start
        self._line_start = None
        if role is None or role[0] != "plan" or np.hypot(event.xdata - x0, event.ydata - y0) < 5:
            return
        self.btn_line.setChecked(False)
        self.set_line(x0, y0, event.xdata, event.ydata)

    def _on_click(self, event):
        if self.btn_line.isChecked() and event.button == 1 and not self._busy_toolbar():
            role = self._roles.get(event.inaxes)
            if role is not None and role[0] == "plan" and event.xdata is not None:
                self._line_start = (event.xdata, event.ydata)
            return
        if event.button != 1 or not self.chk_probe.isChecked() or self._busy_toolbar():
            return
        if self._pinned:
            self._pinned = False
            self._on_move(event)
            return
        if event.inaxes in self._roles and event.xdata is not None:
            text = self.probe_text(event.inaxes, event.xdata, event.ydata)
            if text:
                self.readout.setText("\U0001F4CC " + text)
                self._show_marker(event.inaxes, event.xdata, event.ydata)
                self._pinned = True

    def set_theme(self, theme: str):
        self.theme = theme
        self.redraw()

    def state(self) -> dict:
        return {"view": self.view_box.currentIndex(), "field": self.field_box.currentIndex(),
                "height": self.height.value(), "kt": self.chk_kt.isChecked(),
                "days": self.days.value(), "every": self.storm_combo.currentIndex()}

    def restore(self, st: dict):
        if not st:
            return
        self.view_box.setCurrentIndex(int(st.get("view", 0)))
        self.field_box.setCurrentIndex(int(st.get("field", 1)))
        self.height.setValue(int(st.get("height", 4)))
        self.chk_kt.setChecked(bool(st.get("kt", True)))
        self.days.setValue(int(st.get("days", 7)))
        self.storm_combo.setCurrentIndex(int(st.get("every", 0)))
