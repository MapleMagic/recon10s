#!/usr/bin/env python3
"""
recon10s_windpanel — Reconnaissance › Wind field.

The loaded recon (shared from Imagery › Satellite) placed relative to the
moving centre: every ob at its storm-relative position, a 2-D analysis of
the flight-level or surface-estimate winds, the RMW, and the 34/50/64-kt
wind radii by quadrant, beside NHC's best-track radii for the nearest
synoptic time.
"""
from __future__ import annotations

import datetime as dt
import math
import warnings
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QRadioButton, QScrollArea, QSlider, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import recon10s_nhc as nhc
import recon10s_surface as surface
import recon10s_windfield as wf

try:
    import recon10s_plot
except Exception:  # pragma: no cover
    recon10s_plot = None

KM_PER_NM = wf.KM_PER_NM
THRESHOLD_COLOURS = {34: "#4dd0e1", 50: "#ffd54f", 64: "#ff5252"}


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


def _wind_cmap():
    """The converter's wind colours (as the map's barbs), as a colormap and norm."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    buckets = recon10s_plot._COLOR_BUCKETS if recon10s_plot else [(0, 1e9, (200, 200, 200))]
    edges = [b[0] for b in buckets] + [400.0]
    colours = [tuple(c / 255.0 for c in b[2]) for b in buckets]
    return ListedColormap(colours), BoundaryNorm(edges, len(colours))


class WindFieldPanel(QWidget):
    request_load = pyqtSignal()

    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self.theme = theme
        self.pool = QThreadPool.globalInstance()
        self.bundle = None
        self.sr: Optional[wf.StormRelative] = None
        self.analysis: Optional[wf.Analysis] = None
        self.radii: List[wf.Radius] = []
        self.track: List[nhc.TrackPoint] = []
        self._track_for = ""
        self._advisory = None
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(120)
        self._redraw_timer.timeout.connect(self.recompute)

        self.figure = Figure(figsize=(8, 7), dpi=100, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.addWidget(self._window_box())
        lv.addWidget(self._wind_box())
        lv.addWidget(self._analysis_box())
        lv.addWidget(self._radii_box())
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(440)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Radius", "Quadrant", "Recon obs", "Analysis", "NHC best track",
                                              "Sampled to", "Obs"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(150)

        plot = QWidget()
        pv = QVBoxLayout(plot)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(NavigationToolbar2QT(self.canvas, plot))
        pv.addWidget(self.canvas, 1)
        self.readout = QLabel(" ")
        self.readout.setObjectName("probeReadout")
        pv.addWidget(self.readout)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(plot)
        right.addWidget(self.table)
        right.setSizes([700, 260])
        self.status = QLabel("Load recon in Imagery › Satellite (or here) to begin.")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        rcol = QWidget()
        rl = QVBoxLayout(rcol)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(right, 1)
        rl.addWidget(self.status)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(rcol)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 1200])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split, 1)
        self._placeholder("No recon loaded yet")

    # ---------------------------------------------------------------- building

    def _window_box(self):
        box = QGroupBox("Time window")
        f = QFormLayout(box)
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.valueChanged.connect(lambda _: (self._time_label(), self._redraw_timer.start()))
        f.addRow("Centre time", self.time_slider)
        self.time_text = QLabel("—")
        self.time_text.setObjectName("probeReadout")
        f.addRow("", self.time_text)
        row = QHBoxLayout()
        self.half = QDoubleSpinBox()
        self.half.setRange(0.5, 12.0)
        self.half.setSingleStep(0.5)
        self.half.setValue(2.0)
        self.half.setPrefix("±")
        self.half.setSuffix(" h")
        self.half.valueChanged.connect(lambda _: (self._time_label(), self._redraw_timer.start()))
        row.addWidget(self.half)
        b = QPushButton("Latest")
        b.setObjectName("smallButton")
        b.setToolTip("Centre the window on the newest fix")
        b.clicked.connect(lambda: self.time_slider.setValue(self.time_slider.maximum()))
        row.addWidget(b)
        f.addRow("Window", row)
        self.aircraft_box = QComboBox()
        self.aircraft_box.addItem("Every aircraft", "")
        self.aircraft_box.currentIndexChanged.connect(lambda _: self._redraw_timer.start())
        f.addRow("Aircraft", self.aircraft_box)
        return box

    def _wind_box(self):
        box = QGroupBox("Winds")
        f = QFormLayout(box)
        self.which_group = QButtonGroup(self)
        row = QHBoxLayout()
        for label, key in (("Surface estimate", "sfc"), ("Flight level", "fl")):
            rb = QRadioButton(label)
            rb.setProperty("key", key)
            rb.setChecked(key == "sfc")
            self.which_group.addButton(rb)
            row.addWidget(rb)
        self.which_group.buttonToggled.connect(lambda _b, on: on and self._redraw_timer.start())
        f.addRow(row)
        self.chk_outer = QCheckBox("Outer-vortex ratio beyond 3×RMW")
        self.chk_outer.setChecked(True)
        self.chk_outer.setToolTip("Surface/flight-level ratios are lower away from the eyewall: NHC gives 0.91\n"
                                  "in the eyewall but 0.78 in the outer vortex at 700 mb. With this on, the\n"
                                  "standard ratio applies inside 1.5×RMW and 0.80 beyond 3×RMW (0.85 at\n"
                                  "1000 ft), blended in between. Wind radii live in the outer vortex.")
        self.chk_outer.toggled.connect(lambda _: self._redraw_timer.start())
        f.addRow(self.chk_outer)
        self.vec_box = QComboBox()
        for label, key in (("Earth-relative winds", "earth"), ("Storm-relative winds", "storm"), ("No wind barbs", "")):
            self.vec_box.addItem(label, key)
        self.vec_box.currentIndexChanged.connect(lambda _: self.redraw())
        f.addRow("Barbs", self.vec_box)
        self.orient_box = QComboBox()
        self.orient_box.addItem("North up", False)
        self.orient_box.addItem("Storm motion up", True)
        self.orient_box.currentIndexChanged.connect(lambda _: self._redraw_timer.start())
        f.addRow("Orientation", self.orient_box)
        mrow = QHBoxLayout()
        self.motion_src = QComboBox()
        for label, key in (("Recon fixes", "fixes"), ("NHC advisory", "nhc"), ("Best track", "track"),
                           ("Manual", "manual")):
            self.motion_src.addItem(label, key)
        self.motion_src.currentIndexChanged.connect(lambda _: self._on_motion_source())
        mrow.addWidget(self.motion_src, 1)
        f.addRow("Motion", mrow)
        mm = QHBoxLayout()
        self.motion_dir = QSpinBox()
        self.motion_dir.setRange(0, 359)
        self.motion_dir.setWrapping(True)
        self.motion_dir.setSuffix("°")
        self.motion_kt = QDoubleSpinBox()
        self.motion_kt.setRange(0.0, 60.0)
        self.motion_kt.setSuffix(" kt")
        for w in (self.motion_dir, self.motion_kt):
            w.valueChanged.connect(lambda _: self._redraw_timer.start() if self.motion_src.currentData() == "manual"
                                   else None)
            mm.addWidget(w)
        f.addRow("toward", mm)
        return box

    def _analysis_box(self):
        box = QGroupBox("Analysis")
        f = QFormLayout(box)
        self.chk_analysis = QCheckBox("Gridded wind field")
        self.chk_analysis.setChecked(True)
        self.chk_analysis.toggled.connect(lambda _: self._redraw_timer.start())
        f.addRow(self.chk_analysis)
        self.radial_km = QDoubleSpinBox()
        self.radial_km.setRange(3.0, 40.0)
        self.radial_km.setValue(10.0)
        self.radial_km.setSuffix(" km")
        self.radial_km.setToolTip("Length scale along a radius (grows to 8% of the radius farther out).")
        self.arc_km = QDoubleSpinBox()
        self.arc_km.setRange(10.0, 150.0)
        self.arc_km.setValue(35.0)
        self.arc_km.setSuffix(" km")
        self.arc_km.setToolTip("Length scale around the circle: how far a leg's winds are spread sideways.")
        for w in (self.radial_km, self.arc_km):
            w.valueChanged.connect(lambda _: self._redraw_timer.start())
        f.addRow("Radial scale", self.radial_km)
        f.addRow("Around-circle scale", self.arc_km)
        self.max_r = QSpinBox()
        self.max_r.setRange(60, 500)
        self.max_r.setSingleStep(20)
        self.max_r.setValue(280)
        self.max_r.setSuffix(" km")
        self.max_r.valueChanged.connect(lambda _: self._redraw_timer.start())
        f.addRow("Out to", self.max_r)
        return box

    def _radii_box(self):
        box = QGroupBox("Wind radii")
        f = QFormLayout(box)
        row = QHBoxLayout()
        self.thr_checks = {}
        for thr in wf.THRESHOLDS:
            c = QCheckBox(f"{thr} kt")
            c.setChecked(True)
            c.setStyleSheet(f"color: {THRESHOLD_COLOURS[thr]};")
            c.toggled.connect(lambda _: self.redraw())
            self.thr_checks[thr] = c
            row.addWidget(c)
        f.addRow(row)
        self.chk_nhc = QCheckBox("Best-track radii (NHC or JTWC, nearest synoptic time)")
        self.chk_nhc.setChecked(True)
        self.chk_nhc.toggled.connect(lambda _: self.redraw())
        f.addRow(self.chk_nhc)
        self.chk_obs = QCheckBox("Obs coloured by speed")
        self.chk_obs.setChecked(True)
        self.chk_obs.toggled.connect(lambda _: self.redraw())
        f.addRow(self.chk_obs)
        note = QLabel("Radii are the farthest surface estimate reaching each speed in each quadrant, along "
                      "the flight track (“+”: still that strong where sampling ended) and in the "
                      "analysis. The aircraft samples only part of each quadrant.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        f.addRow(note)
        return box

    # ------------------------------------------------------------------ data

    def set_bundle(self, bundle):
        self.bundle = bundle
        fix = getattr(bundle, "fix", None) if bundle is not None else None
        self._advisory = fix
        times = self._fix_times()
        keep_latest = self.time_slider.value() >= self.time_slider.maximum()
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0, max(0, len(times) - 1))
        if keep_latest:
            self.time_slider.setValue(self.time_slider.maximum())
        self.time_slider.blockSignals(False)
        cur = self.aircraft_box.currentData()
        self.aircraft_box.blockSignals(True)
        self.aircraft_box.clear()
        self.aircraft_box.addItem("Every aircraft", "")
        if bundle is not None and bundle.overlay is not None:
            seen = []
            for lb in bundle.overlay.column("labels", ""):
                a = str(lb).split(" ")[0]
                if a and a not in seen and a != "converted":
                    seen.append(a)
            for a in seen:
                self.aircraft_box.addItem(a, a)
        i = self.aircraft_box.findData(cur)
        self.aircraft_box.setCurrentIndex(max(0, i))
        self.aircraft_box.blockSignals(False)
        self._time_label()
        sid = getattr(fix, "storm_id", "") if fix else ""
        if sid and sid != self._track_for:
            self._track_for = sid
            job = _Job(nhc.best_track, sid)
            job.sig.done.connect(self._on_track)
            job.sig.failed.connect(lambda m: None)
            self.pool.start(job)
        self._on_motion_source(recompute=False)
        self.recompute()

    def set_track(self, track):
        """NHC's best track, when another panel already has it."""
        self._on_track(track)

    def _on_track(self, track):
        self.track = list(track or [])
        if self.motion_src.currentData() == "track":
            self._on_motion_source(recompute=False)
        self.recompute()

    def _fix_times(self) -> List[dt.datetime]:
        b = self.bundle
        if b is None:
            return []
        times = sorted({f.time for f in b.fixes}) if b.fixes else []
        if not times and b.overlay is not None and len(b.overlay):
            ts = [t for t in b.overlay.times if t]
            if ts:
                t0, t1 = min(ts), max(ts)
                n = int((t1 - t0).total_seconds() // 1800) + 1
                times = [t0 + dt.timedelta(minutes=30 * k) for k in range(n)]
        return times

    def centre_time(self) -> Optional[dt.datetime]:
        times = self._fix_times()
        if not times:
            return None
        return times[min(self.time_slider.value(), len(times) - 1)]

    def _time_label(self):
        t = self.centre_time()
        h = self.half.value()
        self.time_text.setText(f"{t:%d %b %H:%MZ} ±{h:g} h" if t else "—")

    def motion(self):
        src = self.motion_src.currentData()
        if src == "manual":
            return float(self.motion_dir.value()), float(self.motion_kt.value())
        if src == "nhc" and self._advisory is not None and getattr(self._advisory, "motion", None):
            return self._advisory.motion
        if src == "track" and self.track:
            m = nhc.track_motion(self.track, 12.0, self.centre_time())
            if m:
                return m
        if self.bundle is not None and self.bundle.motion:
            return self.bundle.motion
        if self.bundle is not None and self.bundle.fixes:
            return wf.fixes_motion(self.bundle.fixes)
        return None

    def _on_motion_source(self, recompute: bool = True):
        manual = self.motion_src.currentData() == "manual"
        for w in (self.motion_dir, self.motion_kt):
            w.setEnabled(manual)
        if not manual:
            m = self.motion()
            for w, v in ((self.motion_dir, int(round(m[0])) % 360 if m else 0),
                         (self.motion_kt, float(m[1]) if m else 0.0)):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)
        if recompute:
            self._redraw_timer.start()

    def _which(self) -> str:
        b = self.which_group.checkedButton()
        return b.property("key") if b else "sfc"

    def recompute(self):
        b = self.bundle
        if b is None or b.overlay is None or not len(b.overlay) or not b.fixes:
            self.sr = self.analysis = None
            self.radii = []
            self._fill_table()
            self._placeholder("No recon loaded yet" if b is None or b.overlay is None or not len(b.overlay)
                              else "No centre fixes yet — load vortex messages, or tick “Derive from "
                                   "obs” under Imagery › Satellite")
            return
        t = self.centre_time()
        h = dt.timedelta(hours=self.half.value())
        ov = b.overlay
        aircraft = self.aircraft_box.currentData()
        if aircraft:
            import recon10s_satview as satview
            keep = [i for i, lb in enumerate(ov.column("labels", "")) if str(lb).split(" ")[0] == aircraft]
            sub = satview.Overlay()
            for fname in satview.Overlay.FIELDS:
                col = ov.column(fname)
                setattr(sub, fname, [col[i] for i in keep])
            ov = sub
        motion = self.motion()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sr = wf.storm_relative(ov, b.fixes, motion, t - h if t else None, t + h if t else None,
                                   rotate=bool(self.orient_box.currentData()), max_km=float(self.max_r.value()) + 60)
            fl_an = wf.analyse(sr, "fl", max_km=float(self.max_r.value()), radial_km=self.radial_km.value(),
                               arc_km=self.arc_km.value()) if sr is not None and len(sr) else None
            rmw = fl_an.rmw() if fl_an is not None else None
            if rmw is None and sr is not None:
                rmw = sr.rmw_estimate()
            if sr is not None and self.chk_outer.isChecked():
                sr.use_outer_ratio(rmw)
            which = self._which()
            an = None
            if sr is not None and self.chk_analysis.isChecked():
                an = fl_an if which == "fl" else wf.analyse(sr, "sfc", max_km=float(self.max_r.value()),
                                                            radial_km=self.radial_km.value(),
                                                            arc_km=self.arc_km.value())
        self.sr, self.analysis, self.rmw = sr, an, rmw
        self.radii = wf.wind_radii(sr, an, which) if sr is not None else []
        self._fill_table()
        self.redraw()

    # --------------------------------------------------------------- table

    def nhc_point(self):
        t = self.centre_time()
        return nhc.track_at(self.track, t, 3.0) if (t and self.track) else None

    def _fill_table(self):
        tp = self.nhc_point()
        self.table.setRowCount(0)

        def fmt(km, lb=False):
            if km is None:
                return "—"
            return f"{km / KM_PER_NM:.0f}{'+' if lb else ''} nm ({km:.0f} km)"
        for r in self.radii:
            row = self.table.rowCount()
            self.table.insertRow(row)
            nhc_nm = tp.radius(r.threshold, r.quadrant) if tp else None
            cells = [f"{r.threshold} kt", r.quadrant, fmt(r.obs_km, r.obs_lower_bound),
                     fmt(r.analysis_km, r.analysis_lower_bound),
                     (f"{nhc_nm} nm" if nhc_nm else "0" if nhc_nm == 0 else "—"),
                     fmt(r.sampled_km) if r.sampled_km else "not sampled", str(r.n_obs)]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setForeground(_qcolor(THRESHOLD_COLOURS[r.threshold]))
                self.table.setItem(row, c, item)
        ag = nhc.agency(self._track_for)
        if tp is not None:
            self.table.horizontalHeaderItem(4).setText(f"{ag} best track {tp.time:%d/%HZ}")
        else:
            self.table.horizontalHeaderItem(4).setText(f"{ag} best track")

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
        sr = self.sr
        if sr is None or not len(sr):
            if self.bundle is not None and self.bundle.fixes:
                self._placeholder("No obs in this time window")
            return
        fg, bg = self._colours()
        fig = self.figure
        fig.clear()
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor("#0b0d11" if self.theme != "light" else "#f4f6f8")
        ax.tick_params(colors=fg, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#5c6675")
        lim = float(self.max_r.value())
        cmap, norm = _wind_cmap()
        which = self._which()
        an = self.analysis
        if an is not None:
            X, Y = an.xy()
            mesh = ax.pcolormesh(X, Y, np.ma.masked_invalid(an.speed), cmap=cmap, norm=norm,
                                 shading="flat", edgecolors="face", linewidth=0.3, antialiased=True,
                                 rasterized=True, zorder=1)
            cb = fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.01)
            cb.set_label(("Surface estimate" if which == "sfc" else "Flight-level wind") + " (kt)", color=fg,
                         fontsize=8)
            cb.ax.tick_params(labelsize=7, colors=fg)
        # range rings and quadrant lines (geographic quadrants, turned with the frame)
        rot = math.radians(sr.motion[0]) if sr.rotated and sr.motion else 0.0
        for rr in range(50, int(lim) + 1, 50):
            t = np.linspace(0, 2 * np.pi, 181)
            ax.plot(rr * np.sin(t), rr * np.cos(t), color="#8b95a5", lw=0.5, alpha=0.6, zorder=2)
            ax.text(rr * math.sin(math.radians(30)), rr * math.cos(math.radians(30)), f"{rr}", color="#8b95a5",
                    fontsize=6.5)
        for q in (0, 90, 180, 270):
            a = math.radians(q) - rot
            ax.plot([0, lim * math.sin(a)], [0, lim * math.cos(a)], color="#8b95a5", lw=0.6, ls=":", alpha=0.8,
                    zorder=2)
        for k, name in enumerate(wf.QUADRANTS):
            a = math.radians(45 + 90 * k) - rot
            ax.text(0.93 * lim * math.sin(a), 0.93 * lim * math.cos(a), name, color=fg, fontsize=9,
                    ha="center", va="center", alpha=0.8)
        spd = sr.speed(which)
        if self.chk_obs.isChecked():
            ok = np.isfinite(spd)
            for seg in sr.tracks():
                ax.plot(sr.x[seg], sr.y[seg], color="#f2f5f8" if self.theme != "light" else "#333333", lw=0.5,
                        alpha=0.4, zorder=3)
            ax.scatter(sr.x[ok], sr.y[ok], c=spd[ok], cmap=cmap, norm=norm, s=9, edgecolors="#0b0d11",
                       linewidths=0.3, zorder=4)
        vec = self.vec_box.currentData()
        if vec:
            u, v = (sr.u, sr.v) if vec == "earth" else (sr.us, sr.vs)
            step = max(1, len(sr) // 90)
            sel = np.arange(len(sr))[::step]
            sel = sel[np.isfinite(u[sel]) & np.isfinite(v[sel])]
            ax.barbs(sr.x[sel], sr.y[sel], u[sel], v[sel], length=5.2, color="#f2f5f8" if self.theme != "light"
                     else "#222222", linewidth=0.6, zorder=5)
        # radii: arcs over each quadrant
        tp = self.nhc_point()
        for r in self.radii:
            if not self.thr_checks[r.threshold].isChecked():
                continue
            col = THRESHOLD_COLOURS[r.threshold]
            k = wf.QUADRANTS.index(r.quadrant)
            th = np.radians(np.linspace(90 * k, 90 * (k + 1), 31)) - rot
            if r.obs_km:
                ax.plot(r.obs_km * np.sin(th), r.obs_km * np.cos(th), color=col, lw=2.2, zorder=6,
                        ls="-" if not r.obs_lower_bound else (0, (4, 2)))
            if self.chk_nhc.isChecked() and tp is not None:
                nm = tp.radius(r.threshold, r.quadrant)
                if nm:
                    km = nm * KM_PER_NM
                    ax.plot(km * np.sin(th), km * np.cos(th), color=col, lw=1.2, ls=(0, (1, 2)), zorder=6)
        if self.rmw:
            t = np.linspace(0, 2 * np.pi, 181)
            ax.plot(self.rmw * np.sin(t), self.rmw * np.cos(t), color="#ffffff" if self.theme != "light"
                    else "#000000", lw=1.0, ls="--", zorder=6)
            ax.text(0, -self.rmw, f"RMW {self.rmw:.0f} km", color=fg, fontsize=7.5, ha="center", va="top")
        if sr.motion:
            a = math.radians(sr.motion[0]) - rot
            L = 0.22 * lim
            ax.annotate("", xy=(L * math.sin(a), L * math.cos(a)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="#ffca28", lw=2), zorder=7)
            ax.text(1.1 * L * math.sin(a), 1.1 * L * math.cos(a), f"{sr.motion[0]:03.0f}°/{sr.motion[1]:.0f} kt",
                    color="#ffca28", fontsize=8, ha="center", zorder=7)
        ax.plot(0, 0, marker="+", ms=14, mew=2, color="#ff4081", zorder=8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        if sr.rotated:
            ax.set_xlabel("km right of motion", color=fg, fontsize=8)
            ax.set_ylabel("km ahead (along motion)", color=fg, fontsize=8)
        else:
            ax.set_xlabel("km east of centre", color=fg, fontsize=8)
            ax.set_ylabel("km north of centre", color=fg, fontsize=8)
        t = self.centre_time()
        pk = wf.peak(sr, "fl")
        title = (f"{len(sr)} obs, {min(sr.times):%d %H:%M}–{max(sr.times):%H:%MZ}, storm-relative"
                 + (f"  ·  peak FL {pk[0]:.0f} kt at {pk[1]:.0f} km" if pk else ""))
        name = getattr(getattr(self.bundle, "fix", None), "name", "") or ""
        ax.set_title(f"{name}  {title}".strip(), color=fg, fontsize=9)
        legend = (f"Solid arcs: recon radii (dashed: lower bound). Dotted: {nhc.agency(self._track_for)} best track"
                  + (f" {tp.time:%d/%HZ}" if tp else "") + ".")
        ax.text(0.01, 0.01, legend, transform=ax.transAxes, color=fg, fontsize=7, va="bottom",
                bbox=dict(fc=bg, ec="#5c6675", lw=0.5, alpha=0.85))
        self.canvas.draw_idle()
        levels = surface.level_summary(self.bundle.overlay) if self.bundle else []
        lv = ", ".join(f"{lab} ×{f:.2f} ({n})" for lab, f, n in levels if np.isfinite(f))
        self.status.setText(f"Centre from {sum(1 for f in self.bundle.fixes if f.source == 'VDM')} vortex-message "
                            f"and {sum(1 for f in self.bundle.fixes if f.source != 'VDM')} derived fixes. "
                            f"Surface ratios: {lv or 'n/a'}"
                            + ("; eased to the outer-vortex ratio beyond the eyewall." if self.chk_outer.isChecked()
                               else "."))

    def _on_hover(self, event):
        if self.sr is None or event.inaxes is None or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        r = math.hypot(x, y)
        if self.sr.rotated and self.sr.motion:
            b = math.radians(self.sr.motion[0])
            gx, gy = x * math.cos(b) + y * math.sin(b), -x * math.sin(b) + y * math.cos(b)
        else:
            gx, gy = x, y
        brg = (math.degrees(math.atan2(gx, gy)) + 360) % 360
        v = wf.analysis_at(self.analysis, x, y)
        k = int(np.argmin(np.hypot(self.sr.x - x, self.sr.y - y)))
        near = math.hypot(self.sr.x[k] - x, self.sr.y[k] - y)
        ob = ""
        if near <= 6:
            fl = self.sr.wspd[k]
            sf = self.sr.sfc[k]
            ob = (f"  ·  ob {self.sr.times[k]:%H:%M:%SZ} {self.sr.labels[k]}: FL "
                  f"{'%.0f kt' % fl if np.isfinite(fl) else '—'}, surface est. "
                  f"{'%.0f kt' % sf if np.isfinite(sf) else '—'}")
        self.readout.setText(f"{r:.0f} km / {r / KM_PER_NM:.0f} nm toward {brg:03.0f}°"
                             + (f"  ·  analysis {v:.0f} kt" if v is not None else "") + ob)

    def set_theme(self, theme: str):
        self.theme = theme
        self.redraw()

    def state(self) -> dict:
        return {"half": self.half.value(), "which": self._which(), "outer": self.chk_outer.isChecked(),
                "vectors": self.vec_box.currentIndex(), "orient": self.orient_box.currentIndex(),
                "motion": self.motion_src.currentIndex(), "analysis": self.chk_analysis.isChecked(),
                "radial": self.radial_km.value(), "arc": self.arc_km.value(), "max_r": self.max_r.value()}

    def restore(self, st: dict):
        if not st:
            return
        self.half.setValue(float(st.get("half", 2.0)))
        for b in self.which_group.buttons():
            b.setChecked(b.property("key") == st.get("which", "sfc"))
        self.chk_outer.setChecked(bool(st.get("outer", True)))
        self.vec_box.setCurrentIndex(int(st.get("vectors", 0)))
        self.orient_box.setCurrentIndex(int(st.get("orient", 0)))
        self.motion_src.setCurrentIndex(int(st.get("motion", 0)))
        self.chk_analysis.setChecked(bool(st.get("analysis", True)))
        self.radial_km.setValue(float(st.get("radial", 10.0)))
        self.arc_km.setValue(float(st.get("arc", 35.0)))
        self.max_r.setValue(int(st.get("max_r", 280)))


def _qcolor(hex_colour: str):
    from PyQt6.QtGui import QColor
    return QColor(hex_colour)
