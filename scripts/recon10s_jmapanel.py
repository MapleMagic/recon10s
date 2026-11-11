#!/usr/bin/env python3
"""
recon10s_jmapanel — the JMA tab (Radar section).

Typhoons from NRL's sector file; the nearest of JMA's 20 radars within reach
is chosen for one (or pick any radar). Reflectivity, Doppler velocity and
storm-relative velocity in one or two panels, any elevation of the
10-minute cycle, hover for values. "Save as CfRadial" writes the decoded
sweeps for LROSE, Py-ART and friends; "Build 3-D volume" grids the cycle for
Radar › Radar volume. Storm motion comes from JTWC's best track (UCAR's
open mirror) or is typed in.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QDate, QObject, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QButtonGroup, QComboBox, QDateEdit, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

import recon10s_jma as jma
import recon10s_jmagrib as jg
import recon10s_motion as motion_mod
import recon10s_nhc as nhc
import recon10s_nexrad as nx
import recon10s_radarview as rv

PRODUCT_CHOICES = [("Reflectivity", "reflectivity"), ("Doppler velocity", "velocity"),
                   ("Storm-relative velocity", "srv")]
MOTION_SOURCES = [("Best track (last 12 h)", "track"), ("Typed in", "manual")]


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


class JmaPanel(QWidget):
    volume_requested = pyqtSignal(object, object, str, object)    # volume, (lat, lon) or None, storm, motion

    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self.theme = theme
        self.pool = QThreadPool.globalInstance()
        self.storms: List[jma.Storm] = []
        self.cycles: List[jma.Cycle] = []
        self.volume: Optional[jma.JmaVolume] = None
        self.art = None
        self._busy = False
        self.track = []
        self._track_for = ""

        self.figure = Figure(figsize=(8, 7), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.addWidget(self._site_box())
        lv.addWidget(self._data_box())
        lv.addWidget(self._view_box())
        self.motion_box = motion_mod.MotionBox("Storm motion (storm-relative velocity)", MOTION_SOURCES)
        self.motion_box.changed.connect(lambda: self.redraw() if any(
            cb.currentData() == "srv" for cb in self.product_boxes[:self._n()]) else None)
        lv.addWidget(self.motion_box)
        lv.addWidget(self._export_box())
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(440)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        right = QWidget()
        rvl = QVBoxLayout(right)
        rvl.setContentsMargins(0, 0, 0, 0)
        rvl.addWidget(NavigationToolbar2QT(self.canvas, right))
        rvl.addWidget(self.canvas, 1)
        self.readout = QLabel(" ")
        self.readout.setObjectName("probeReadout")
        rvl.addWidget(self.readout)
        self.status = QLabel("Pick a typhoon; the nearest JMA radar in reach is chosen for it.")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        rvl.addWidget(self.status)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([350, 1200])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split, 1)
        self._placeholder("No JMA radar cycle loaded")
        QTimer.singleShot(0, self.refresh_storms)
        QTimer.singleShot(0, self.find_latest_day)

    # ---------------------------------------------------------------- building

    @staticmethod
    def _grow(combo):
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(14)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return combo

    def _site_box(self):
        box = QGroupBox("Typhoon && radar")
        f = QFormLayout(box)
        row = QHBoxLayout()
        self.storm_box = self._grow(QComboBox())
        self.storm_box.currentIndexChanged.connect(self._on_storm)
        row.addWidget(self.storm_box, 1)
        b = QPushButton("\u21bb")
        b.setFixedWidth(34)
        b.setToolTip("Refresh from NRL's sector file")
        b.clicked.connect(self.refresh_storms)
        row.addWidget(b)
        f.addRow("Storm", row)
        self.site_box = self._grow(QComboBox())
        f.addRow("Radar", self.site_box)
        self.site_note = QLabel("")
        self.site_note.setObjectName("hint")
        self.site_note.setWordWrap(True)
        f.addRow(self.site_note)
        return box

    def _data_box(self):
        box = QGroupBox("Cycle")
        f = QFormLayout(box)
        row = QHBoxLayout()
        self.day = QDateEdit()
        self.day.setCalendarPopup(True)
        self.day.setDisplayFormat("yyyy-MM-dd")
        self.day.setDate(QDate.currentDate())
        row.addWidget(self.day, 1)
        self.btn_list = QPushButton("List cycles")
        self.btn_list.setObjectName("primary")
        self.btn_list.clicked.connect(self.list_cycles)
        row.addWidget(self.btn_list)
        f.addRow("Day (UTC)", row)
        self.cycle_box = self._grow(QComboBox())
        self.cycle_box.currentIndexChanged.connect(lambda _: self.load())
        f.addRow("Cycle", self.cycle_box)
        nav = QHBoxLayout()
        for text, step in (("\u25c0 Older", -1), ("Newer \u25b6", 1)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, s=step: self._step(s))
            nav.addWidget(b)
        b = QPushButton("Latest")
        b.clicked.connect(lambda: self.cycle_box.setCurrentIndex(self.cycle_box.count() - 1))
        nav.addWidget(b)
        f.addRow(nav)
        self.archive_note = QLabel("NICT's mirror of JMA data runs about two days behind real time.")
        self.archive_note.setObjectName("hint")
        self.archive_note.setWordWrap(True)
        f.addRow(self.archive_note)
        return box

    def _view_box(self):
        box = QGroupBox("Display")
        f = QFormLayout(box)
        self.elev_box = self._grow(QComboBox())
        self.elev_box.currentIndexChanged.connect(lambda _: self.redraw())
        f.addRow("Elevation", self.elev_box)
        lay = QHBoxLayout()
        self.layout_group = QButtonGroup(self)
        for n in (1, 2):
            rb = QRadioButton(f"{n} panel{'s' if n > 1 else ''}")
            rb.setProperty("n", n)
            self.layout_group.addButton(rb)
            lay.addWidget(rb)
            rb.setChecked(n == 2)
        self.layout_group.buttonToggled.connect(lambda _b, on: on and self._on_layout())
        f.addRow("Layout", lay)
        self.product_boxes = []
        for k in range(2):
            cb = self._grow(QComboBox())
            for label, key in PRODUCT_CHOICES:
                cb.addItem(label, key)
            cb.setCurrentIndex(k)
            cb.currentIndexChanged.connect(lambda _: self.redraw())
            self.product_boxes.append(cb)
            f.addRow(f"Panel {k + 1}", cb)
        centre = QHBoxLayout()
        self.centre_group = QButtonGroup(self)
        for label, key in (("Storm", "storm"), ("Radar", "radar")):
            rb = QRadioButton(label)
            rb.setProperty("key", key)
            self.centre_group.addButton(rb)
            centre.addWidget(rb)
            rb.setChecked(key == "radar")
        self.centre_group.buttonToggled.connect(lambda _b, on: on and self.redraw())
        f.addRow("Centre on", centre)
        self.width = QSpinBox()
        self.width.setRange(40, 400)
        self.width.setSingleStep(20)
        self.width.setValue(250)
        self.width.setSuffix(" km")
        self.width.valueChanged.connect(lambda _: self.redraw())
        f.addRow("Half-width", self.width)
        return box

    def _export_box(self):
        box = QGroupBox("Export")
        v = QVBoxLayout(box)
        self.btn_cf = QPushButton("Save as CfRadial\u2026")
        self.btn_cf.setToolTip("The loaded cycle's reflectivity and velocity, every sweep, as CfRadial\n"
                               "1.4 netCDF: LROSE (RadxConvert, HawkEye), Py-ART and other radar\n"
                               "tools read it. LROSE cannot read JMA's GRIB2 itself.")
        self.btn_cf.clicked.connect(self.export_cfradial)
        v.addWidget(self.btn_cf)
        self.btn_volume = QPushButton("Build 3-D volume \u25b8")
        self.btn_volume.setToolTip("Grid every elevation of this cycle around the typhoon for Radar \u203a Radar\n"
                                   "volume: CAPPIs, cross-sections, azimuthal means and 3-D isosurfaces.")
        self.btn_volume.clicked.connect(self.request_volume)
        v.addWidget(self.btn_volume)
        self.export_note = QLabel("")
        self.export_note.setObjectName("hint")
        self.export_note.setWordWrap(True)
        v.addWidget(self.export_note)
        return box

    # ----------------------------------------------------------------- storms

    def refresh_storms(self):
        job = _Job(jma.active_storms)
        job.sig.done.connect(self._on_storms)
        job.sig.failed.connect(lambda m: self.status.setText(f"Sector file: {m}"))
        self.pool.start(job)

    def _on_storms(self, storms):
        self.storms = storms
        self.storm_box.blockSignals(True)
        self.storm_box.clear()

        def km(s):
            return jma.sites_by_distance(s.lat, s.lon)[0][1]
        ranked = sorted(storms, key=km)
        for s in ranked:
            self.storm_box.addItem(f"{s.label}  ({km(s):.0f} km from a JMA radar)", s)
        self.storm_box.addItem("No typhoon \u2014 choose a radar", None)
        self.storm_box.blockSignals(False)
        self._on_storm()

    def request_volume(self):
        vol = self.volume
        if vol is None:
            self.export_note.setText("Load a cycle first.")
            return
        s = self.storm_box.currentData()
        centre = None
        if s is not None:
            centre = (s.lat, s.lon)
            m = self.motion_box.motion()
            hours = (vol.scan.time - s.time).total_seconds() / 3600.0
            if m and abs(hours) <= 12:
                import recon10s_windfield as wf
                la, lo = wf._destination(s.lat, s.lon, m[0] if hours >= 0 else (m[0] + 180) % 360,
                                         abs(hours) * m[1] * 1.852)
                centre = (la, (lo + 540) % 360 - 180)
        self.volume_requested.emit(vol, centre, s.name if s is not None else "", self.motion_box.motion())

    def _update_track(self):
        s = self.storm_box.currentData()
        sid = s.storm_id if s is not None else ""
        if not sid:
            self.motion_box.set_source("track", None)
            return
        if sid == self._track_for:
            return
        self._track_for = sid
        job = _Job(nhc.best_track, sid)

        def done(track, sid=sid):
            if sid != self._track_for:
                return
            self.track = track
            m = nhc.track_motion(track, 12.0)
            src = "NHC" if sid[:2] in ("al", "ep", "cp") else "JTWC"
            self.motion_box.set_source("track", m, f"{src} best track to {track[-1].time:%d/%HZ}" if track else "")
        job.sig.done.connect(done)
        job.sig.failed.connect(lambda _m: self.motion_box.set_source("track", None))
        self.pool.start(job)

    def _on_storm(self, *_):
        s = self.storm_box.currentData()
        self._update_track()
        self.site_box.clear()
        if s is None:
            for sid in sorted(jg.SITES, key=lambda k: jg.SITE_POS[k][0], reverse=True):
                self.site_box.addItem(f"{jg.SITES[sid][1]}  ({jg.SITES[sid][0]}, {sid})", sid)
            self.site_note.setText("Pick any of JMA's 20 radars.")
            return
        for sid, km in jma.sites_by_distance(s.lat, s.lon):
            self.site_box.addItem(f"{jg.SITES[sid][1]}  \u2014 {km:.0f} km", sid)
        sid, km = jma.nearest_site(s.lat, s.lon)
        if sid is None:
            d0 = jma.sites_by_distance(s.lat, s.lon)[0][1]
            self.site_note.setText(f"{s.name.title()} is {d0:.0f} km from the nearest JMA radar, beyond "
                                   f"its {jma.REACH_KM:.0f} km reach. Position from NRL at {s.time:%d/%H%MZ}.")
        elif km > jma.DOPPLER_KM:
            self.site_note.setText(f"{jg.SITES[sid][1]} is {km:.0f} km from {s.name.title()}: reflectivity "
                                   f"reaches it, velocity ({jma.DOPPLER_KM:.0f} km) does not. "
                                   f"Position from NRL at {s.time:%d/%H%MZ}.")
        else:
            self.site_note.setText(f"{jg.SITES[sid][1]} is {km:.0f} km from {s.name.title()}: reflectivity "
                                   f"and velocity reach it. Position from NRL at {s.time:%d/%H%MZ}.")

    # ----------------------------------------------------------------- cycles

    def find_latest_day(self):
        job = _Job(jma.latest_day)

        def done(day):
            if day is not None:
                self.day.setDate(QDate(day.year, day.month, day.day))
                self.archive_note.setText(f"NICT's mirror runs about two days behind; its newest day is "
                                          f"{day:%d %b %Y}.")
        job.sig.done.connect(done)
        self.pool.start(job)

    def list_cycles(self, keep_latest: bool = False):
        d = self.day.date()
        day = dt.date(d.year(), d.month(), d.day())
        self.btn_list.setEnabled(False)
        self.status.setText(f"Listing JMA cycles for {day:%d %b %Y}\u2026")
        job = _Job(jma.list_cycles, day)

        def done(cycles):
            self.btn_list.setEnabled(True)
            self.cycles = cycles
            self.cycle_box.blockSignals(True)
            self.cycle_box.clear()
            for c in cycles:
                self.cycle_box.addItem(c.label, c.time.isoformat())
            self.cycle_box.blockSignals(False)
            if not cycles:
                self.status.setText(f"No JMA cycles on {day:%d %b %Y} yet.")
                return
            self.status.setText(f"{len(cycles)} cycles, {cycles[0].time:%H:%M}\u2013{cycles[-1].time:%H:%MZ}.")
            self.cycle_box.setCurrentIndex(len(cycles) - 1)
            self.load()
        job.sig.done.connect(done)
        job.sig.failed.connect(lambda m: (self.btn_list.setEnabled(True), self.status.setText(m)))
        self.pool.start(job)

    def _step(self, d):
        i = self.cycle_box.currentIndex() + d
        if 0 <= i < self.cycle_box.count():
            self.cycle_box.setCurrentIndex(i)

    def load(self):
        i = self.cycle_box.currentIndex()
        sid = self.site_box.currentData()
        if not (0 <= i < len(self.cycles)) or sid is None or self._busy:
            return
        cycle = self.cycles[i]
        self._busy = True
        self.status.setText(f"Fetching the {cycle.label} cycle (two TARs, ~5 MB for all 20 radars)\u2026")
        job = _Job(jma.load_volume, cycle, sid)

        def done(vol):
            self._busy = False
            self.volume = vol
            keep = self.elev_box.currentData()
            self.elev_box.blockSignals(True)
            self.elev_box.clear()
            for e in vol.elevations():
                self.elev_box.addItem(f"{e:.1f}\u00b0", e)
            idx = 0
            if keep is not None and self.elev_box.count():
                idx = min(range(self.elev_box.count()), key=lambda k: abs(self.elev_box.itemData(k) - keep))
            else:
                idx = min(range(self.elev_box.count()), key=lambda k: abs(self.elev_box.itemData(k) - 0.5))
            self.elev_box.setCurrentIndex(idx)
            self.elev_box.blockSignals(False)
            self.redraw()
            if self.cycle_box.currentIndex() != i or self.site_box.currentData() != sid:
                self.load()
        job.sig.done.connect(done)
        job.sig.failed.connect(lambda m: (setattr(self, "_busy", False), self.status.setText(m)))
        self.pool.start(job)

    # ---------------------------------------------------------------- drawing

    def _n(self):
        b = self.layout_group.checkedButton()
        return int(b.property("n")) if b else 2

    def _on_layout(self):
        for k, cb in enumerate(self.product_boxes):
            cb.setEnabled(k < self._n())
        self.redraw()

    def _placeholder(self, text):
        self.figure.clear()
        self.figure.patch.set_facecolor("#12151b" if self.theme != "light" else "#ffffff")
        self.figure.text(0.5, 0.5, text, ha="center", va="center", color="#8b95a5")
        self.canvas.draw_idle()

    def redraw(self):
        vol = self.volume
        if vol is None or self.elev_box.currentData() is None:
            return
        import cartopy.crs as ccrs
        s = self.storm_box.currentData()
        centre = (0.0, 0.0)
        if self.centre_group.checkedButton().property("key") == "storm" and s is not None:
            crs = ccrs.AzimuthalEquidistant(central_longitude=vol.site_lon, central_latitude=vol.site_lat)
            sx, sy = crs.transform_point(s.lon, s.lat, ccrs.PlateCarree())
            if np.hypot(sx, sy) < (jma.REACH_KM + self.width.value()) * 1e3:
                centre = (sx, sy)
        products = [cb.currentData() for cb in self.product_boxes[:self._n()]]
        try:
            mot = self.motion_box.motion()
            panels = rv.render_panels(vol, products, self.elev_box.currentData(), centre,
                                      self.width.value() * 1e3, False, 900, motion=mot)
            storm = (s.lat, s.lon, s.name) if s is not None else None
            notes = ["Storm-relative velocity needs a storm motion (left)" if p == "srv" and mot is None
                     else "" for p in products]
            self.art = rv.draw(self.figure, vol, panels, storm=storm, theme=self.theme,
                               rings_km=(50, 100, 150, 200, 250, 300, 400), notes=notes)
        except Exception as exc:
            self.status.setText(f"Could not draw: {exc}")
            return
        info = []
        for p, pim in zip(products, panels):
            if pim is not None:
                info.append(f"{nx.PRODUCTS[p]['label']}: {vol.sweep_info(nx.BASE_PRODUCT.get(p, p), pim.sweep)}")
            else:
                info.append(f"{nx.PRODUCTS[p]['label']}: not scanned at this elevation")
        self.status.setText("  \u00b7  ".join(info))
        self.canvas.draw_idle()

    def _on_hover(self, event):
        if self.art is None or event.inaxes not in self.art.axes or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        km = float(np.hypot(x, y)) / 1e3
        brg = (np.degrees(np.arctan2(x, y)) + 360) % 360
        h = nx.beam_height_km(km, self.elev_box.currentData() or 0.5, self.volume.site_height_m)
        bits = []
        for ax in self.art.axes:
            v = rv.value_at(self.art, ax, x, y)
            if v and np.isfinite(v[1]):
                bits.append(f"{nx.PRODUCTS[v[0]]['label'].split()[0]} {v[1]:.1f} {nx.PRODUCTS[v[0]]['units']}")
        self.readout.setText(f"{km:.0f} km / {km / 1.852:.0f} nm at {brg:03.0f}\u00b0  \u00b7  beam {h:.1f} km"
                             f"  \u00b7  " + ("   ".join(bits) or "no echo"))

    # ----------------------------------------------------------------- export

    def export_cfradial(self):
        vol = self.volume
        if vol is None:
            self.export_note.setText("Load a cycle first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the CfRadial files")
        if not folder:
            return
        written = self.write_cfradial(folder)
        self.export_note.setText("Wrote " + ", ".join(os.path.basename(p) for p in written))

    def write_cfradial(self, folder: str) -> List[str]:
        vol = self.volume
        out = []
        for sf, tag in ((vol.refl, "DBZ"), (vol.vel, "VEL")):
            if sf is None:
                continue
            name = f"cfrad.{vol.scan.time:%Y%m%d_%H%M%S}_JMA_{sf.site_code or sf.site_id}_{tag}.nc"
            out.append(jma.write_cfradial(os.path.join(folder, name), sf))
        return out

    def set_theme(self, theme):
        self.theme = theme
        self.redraw()

    def state(self) -> dict:
        return {"layout": self._n(), "products": [cb.currentData() for cb in self.product_boxes],
                "width": self.width.value(), "centre": self.centre_group.checkedButton().property("key"),
                "motion": self.motion_box.state()}

    def restore(self, st: dict):
        if not st:
            return
        for b in self.layout_group.buttons():
            b.setChecked(b.property("n") == st.get("layout", 2))
        for cb, p in zip(self.product_boxes, st.get("products", [])):
            i = cb.findData(p)
            if i >= 0:
                cb.setCurrentIndex(i)
        self.width.setValue(int(st.get("width", 250)))
        for b in self.centre_group.buttons():
            b.setChecked(b.property("key") == st.get("centre", "radar"))
        self.motion_box.restore(st.get("motion") or {})
