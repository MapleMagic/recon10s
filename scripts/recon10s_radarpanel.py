#!/usr/bin/env python3
"""
recon10s_radarpanel — the NEXRAD tab (Radar section).

Pick a storm and the nearest WSR-88D within reach is chosen for it (or pick
any site); pick a scan, an elevation, and one, two or four panels, each
showing reflectivity, velocity, storm-relative velocity, differential
reflectivity or correlation coefficient. Hover for values, distance and beam
height. Scans can end "now" or at any past time, for landfalls after the
fact. "Build 3-D volume" grids the scan around the storm for Radar › Radar
volume (CAPPIs, cross-sections, azimuthal means) and the 3D structure tab.
"""
from __future__ import annotations

import datetime as dt
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QDateTime, QObject, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDateTimeEdit, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

import recon10s_motion as motion_mod
import recon10s_nexrad as nx
import recon10s_nexradsites as sites
import recon10s_nhc as nhc
import recon10s_radarview as rv

REACH_MI = 286          # NEXRAD reflectivity reaches 460 km
DOPPLER_MI = 186        # velocity and dual-pol reach about 300 km
DEFAULTS = ["reflectivity", "velocity", "zdr", "cc"]
UTC = dt.timezone.utc


def storm_centre_at(when: dt.datetime, fix=None, recon=None) -> Optional[tuple]:
    """
    Best guess of the centre at a scan's time: the recon fixes (within 3 h of
    them), else NHC's position carried along its motion (within 12 h), else None.
    """
    if recon is not None and getattr(recon, "fixes", None):
        import recon10s_windfield as wf
        la, lo = wf.centre_track(recon.fixes, [when], getattr(recon, "motion", None))
        if np.isfinite(la[0]):
            return float(la[0]), float(lo[0])
    if fix is not None and getattr(fix, "lat", None) is not None:
        t0 = getattr(fix, "time", None)
        mot = getattr(fix, "motion", None)
        if t0 is None:
            return fix.lat, fix.lon
        hours = (when - t0).total_seconds() / 3600.0
        if abs(hours) > 12:
            return None
        if mot:
            import recon10s_windfield as wf
            hd, kt = mot
            la, lo = wf._destination(fix.lat, fix.lon, hd if hours >= 0 else (hd + 180) % 360,
                                     abs(hours) * kt * 1.852)
            return la, (lo + 540) % 360 - 180
        return fix.lat, fix.lon
    return None


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


class RadarPanel(QWidget):
    volume_requested = pyqtSignal(object, object, str, object)    # volume, (lat, lon) or None, storm, motion

    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self.theme = theme
        self.pool = QThreadPool.globalInstance()
        self.scans: List[nx.Scan] = []
        self.volume: Optional[nx.Volume] = None
        self.art: Optional[rv.RadarArtists] = None
        self.recon = None
        self._busy = False
        self.track = []
        self._track_for = ""

        self.figure = Figure(figsize=(8, 7), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.follow_timer = QTimer(self)
        self.follow_timer.setInterval(120_000)
        self.follow_timer.timeout.connect(lambda: self.load_scans(keep_latest=True))

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.addWidget(self._build_site_box())
        lv.addWidget(self._build_scan_box())
        lv.addWidget(self._build_view_box())
        self.motion_box = motion_mod.MotionBox("Storm motion (storm-relative velocity)")
        self.motion_box.changed.connect(self._on_motion_changed)
        lv.addWidget(self.motion_box)
        lv.addWidget(self._build_volume_box())
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(430)

        right = QWidget()
        rvl = QVBoxLayout(right)
        rvl.setContentsMargins(0, 0, 0, 0)
        rvl.addWidget(self.toolbar)
        rvl.addWidget(self.canvas, 1)
        self.readout = QLabel(" ")
        self.readout.setObjectName("probeReadout")
        rvl.addWidget(self.readout)
        self.status = QLabel("Pick a storm; the nearest radar in reach is chosen for it.")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        rvl.addWidget(self.status)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([340, 1200])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split, 1)
        self._placeholder()
        QTimer.singleShot(0, self.refresh_storms)

    # -------------------------------------------------------------- building

    @staticmethod
    def _grow(w):
        if isinstance(w, QComboBox):
            w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(14)
        w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return w

    def _build_site_box(self) -> QWidget:
        box = QGroupBox("Storm && radar")
        f = QFormLayout(box)
        row = QHBoxLayout()
        self.storm_box = self._grow(QComboBox())
        self.storm_box.currentIndexChanged.connect(self._on_storm)
        row.addWidget(self.storm_box, 1)
        b = QPushButton("\u21bb")
        b.setFixedWidth(34)
        b.setToolTip("Refresh the active storm list from NHC")
        b.clicked.connect(self.refresh_storms)
        row.addWidget(b)
        f.addRow("Storm", row)
        self.site_box = self._grow(QComboBox())
        self.site_box.setEditable(True)
        self.site_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.site_box.setToolTip("The nearest sites to the storm, or type any four-letter\nNEXRAD identifier (KAMX, TJUA, PGUA...).")
        f.addRow("Radar", self.site_box)
        self.site_note = QLabel("")
        self.site_note.setObjectName("hint")
        self.site_note.setWordWrap(True)
        f.addRow(self.site_note)
        return box

    def _build_scan_box(self) -> QWidget:
        box = QGroupBox("Scans")
        f = QFormLayout(box)
        row = QHBoxLayout()
        self.hours = QSpinBox()
        self.hours.setRange(1, 48)
        self.hours.setValue(3)
        self.hours.setSuffix(" h")
        row.addWidget(QLabel("Last"))
        row.addWidget(self.hours)
        row.addStretch(1)
        self.btn_list = QPushButton("Load scans")
        self.btn_list.setObjectName("primary")
        self.btn_list.clicked.connect(lambda: self.load_scans())
        row.addWidget(self.btn_list)
        f.addRow(row)
        when = QHBoxLayout()
        self.chk_now = QCheckBox("ending now")
        self.chk_now.setChecked(True)
        self.chk_now.setToolTip("Untick to look at any past time -- a landfall after the fact.")
        self.chk_now.toggled.connect(lambda on: self.end_time.setEnabled(not on))
        when.addWidget(self.chk_now)
        self.end_time = QDateTimeEdit(QDateTime.currentDateTimeUtc())
        self.end_time.setTimeSpec(Qt.TimeSpec.UTC)
        self.end_time.setDisplayFormat("yyyy-MM-dd HH:mm 'Z'")
        self.end_time.setCalendarPopup(True)
        self.end_time.setEnabled(False)
        when.addWidget(self.end_time, 1)
        f.addRow("Scans", when)
        self.scan_box = self._grow(QComboBox())
        self.scan_box.currentIndexChanged.connect(lambda _: self.load_volume())
        f.addRow("Scan", self.scan_box)
        nav = QHBoxLayout()
        for text, step in (("\u25c0 Older", -1), ("Newer \u25b6", 1)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, s=step: self._step(s))
            nav.addWidget(b)
        b = QPushButton("Latest")
        b.clicked.connect(lambda: self.scan_box.setCurrentIndex(self.scan_box.count() - 1))
        nav.addWidget(b)
        f.addRow(nav)
        self.chk_follow = QCheckBox("Follow new scans (checks every 2 min)")
        self.chk_follow.toggled.connect(lambda on: self.follow_timer.start() if on else self.follow_timer.stop())
        f.addRow(self.chk_follow)
        return box

    def _build_view_box(self) -> QWidget:
        box = QGroupBox("Display")
        f = QFormLayout(box)
        self.elev_box = self._grow(QComboBox())
        self.elev_box.currentIndexChanged.connect(lambda _: self.redraw())
        f.addRow("Elevation", self.elev_box)
        lay = QHBoxLayout()
        self.layout_group = QButtonGroup(self)
        for n in (1, 2, 4):
            rb = QRadioButton(f"{n} panel{'s' if n > 1 else ''}")
            rb.setProperty("n", n)
            self.layout_group.addButton(rb)
            lay.addWidget(rb)
            if n == 2:
                rb.setChecked(True)
        self.layout_group.buttonToggled.connect(lambda _b, on: on and self._on_layout())
        f.addRow("Layout", lay)
        self.product_boxes: List[QComboBox] = []
        for k in range(4):
            cb = self._grow(QComboBox())
            for key, meta in nx.PRODUCTS.items():
                cb.addItem(meta["label"], key)
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
            if key == "storm":
                rb.setChecked(True)
        self.centre_group.buttonToggled.connect(lambda _b, on: on and self.redraw())
        f.addRow("Centre on", centre)
        self.width = QSpinBox()
        self.width.setRange(40, 460)
        self.width.setSingleStep(20)
        self.width.setValue(220)
        self.width.setSuffix(" km")
        self.width.valueChanged.connect(lambda _: self.redraw())
        f.addRow("Half-width", self.width)
        self.chk_dealias = QCheckBox("Dealias velocity")
        self.chk_dealias.setChecked(True)
        self.chk_dealias.setToolTip("Py-ART region-based dealiasing. Hurricane winds routinely exceed\n"
                                    "the Nyquist velocity (about 47 kt on the low velocity cuts).")
        self.chk_dealias.toggled.connect(lambda _: self.redraw())
        f.addRow(self.chk_dealias)
        self.chk_fixes = QCheckBox("Recon centre fixes within \u00b190 min")
        self.chk_fixes.setChecked(True)
        self.chk_fixes.toggled.connect(lambda _: self.redraw())
        f.addRow(self.chk_fixes)
        self._on_layout(redraw=False)
        return box

    def _build_volume_box(self) -> QWidget:
        box = QGroupBox("3-D volume")
        v = QVBoxLayout(box)
        self.btn_volume = QPushButton("Build 3-D volume \u25b8")
        self.btn_volume.setToolTip("Grid every elevation of this scan around the storm (1 km, 0.5-15 km) and\n"
                                   "open it in Radar \u203a Radar volume: CAPPIs, vertical sections through the\n"
                                   "eye or along any line, azimuthal means, and 3-D isosurfaces.")
        self.btn_volume.clicked.connect(self.request_volume)
        v.addWidget(self.btn_volume)
        self.volume_note = QLabel("Centred on the storm at the scan time (recon fixes, else NHC's position "
                                  "carried along its motion), then on the eye if one shows.")
        self.volume_note.setObjectName("hint")
        self.volume_note.setWordWrap(True)
        v.addWidget(self.volume_note)
        return box

    def request_volume(self):
        vol = self.volume
        if vol is None:
            self.status.setText("Load a scan first.")
            return
        fix = self.storm_box.currentData()
        centre = storm_centre_at(vol.scan.time, fix, self.recon)
        self.volume_requested.emit(vol, centre, fix.name if fix is not None else "", self.motion_box.motion())

    def _on_motion_changed(self):
        if self.volume is not None and any(cb.currentData() == "srv"
                                           for cb in self.product_boxes[:self._n_panels()]):
            self.redraw()

    def _update_motion_sources(self):
        fix = self.storm_box.currentData()
        self.motion_box.set_source("nhc", fix.motion if fix is not None and fix.motion else None,
                                   f"NHC advisory {fix.time:%d/%HZ}" if fix is not None and fix.time else "")
        rec = self.recon.motion if self.recon is not None and getattr(self.recon, "motion", None) else None
        self.motion_box.set_source("fixes", rec, "recon fixes, last 6 h")
        sid = getattr(fix, "storm_id", "") if fix is not None else ""
        if sid and sid != self._track_for:
            self._track_for = sid
            self.track = []
            job = _Job(lambda st, s: nhc.best_track(s), sid)

            def done(track, sid=sid):
                if sid != self._track_for:
                    return
                self.track = track
                m = nhc.track_motion(track, 12.0)
                self.motion_box.set_source("track", m, f"best track to {track[-1].time:%d/%HZ}" if track else "")
            job.sig.done.connect(done)
            self.pool.start(job)
        elif not sid:
            self.motion_box.set_source("track", None)

    # ---------------------------------------------------------------- storms

    def refresh_storms(self):
        self.storm_box.blockSignals(True)
        self.storm_box.clear()
        try:
            storms = nhc.active_storms()
        except Exception as exc:
            storms = []
            self.status.setText(f"Could not reach NHC: {exc}")
        # storms within reach of a radar first
        def dist(f):
            return sites.stations_by_distance(f.lat, f.lon, 1)[0][1]
        for f in sorted(storms, key=dist):
            d = dist(f)
            self.storm_box.addItem(f"{f.label}  ({d:.0f} mi from a radar)", f)
        self.storm_box.addItem("No storm \u2014 choose a radar", None)
        self.storm_box.blockSignals(False)
        self._on_storm()

    def _on_storm(self, *_):
        fix = self.storm_box.currentData()
        self.site_box.blockSignals(True)
        self.site_box.clear()
        if fix is None:
            for s in sorted(sites.STATIONS, key=lambda s: s.icao):
                self.site_box.addItem(f"{s.icao}  {s.name.title()}, {s.state}", s.icao)
            self.site_note.setText("Pick or type any NEXRAD site.")
        else:
            for s, d in sites.stations_by_distance(fix.lat, fix.lon, 10):
                self.site_box.addItem(f"{s.icao}  {s.name.title()}, {s.state}  \u2014 {d:.0f} mi", s.icao)
            near, d = sites.find_nearest_station_in_range(fix.lat, fix.lon, REACH_MI)
            if near is None:
                d0 = sites.stations_by_distance(fix.lat, fix.lon, 1)[0][1]
                self.site_note.setText(f"{fix.name.title()} is {d0:.0f} mi from the nearest radar, beyond "
                                       f"reflectivity's {REACH_MI} mi reach. No radar can see it yet.")
            elif d > DOPPLER_MI:
                self.site_note.setText(f"{near.icao} is {d:.0f} mi from {fix.name.title()}: reflectivity reaches "
                                       f"the centre; velocity and dual-pol only reach ~{DOPPLER_MI} mi.")
            else:
                self.site_note.setText(f"{near.icao} is {d:.0f} mi from {fix.name.title()}: every product "
                                       "reaches the centre.")
        self.site_box.setCurrentIndex(0)
        self.site_box.blockSignals(False)
        self._update_motion_sources()

    def site(self) -> str:
        data = self.site_box.currentData()
        if data and self.site_box.currentText().startswith(str(data)):
            return data
        return self.site_box.currentText().strip().upper()[:4]

    # ----------------------------------------------------------------- scans

    def load_scans(self, keep_latest: bool = False):
        icao = self.site()
        if icao not in sites.STATIONS_BY_ICAO:
            self.status.setText(f"{icao!r} is not a NEXRAD site.")
            return
        at_end = self.scan_box.currentIndex() == self.scan_box.count() - 1
        self.btn_list.setEnabled(False)
        end = None if self.chk_now.isChecked() else \
            dt.datetime.fromtimestamp(self.end_time.dateTime().toSecsSinceEpoch(), tz=UTC)
        job = _Job(lambda st, s, h, e: nx.recent_scans(s, h, now=e), icao, float(self.hours.value()), end)

        def done(scans):
            self.btn_list.setEnabled(True)
            newer = scans and (not self.scans or scans[-1].time > self.scans[-1].time)
            self.scans = scans
            self.scan_box.blockSignals(True)
            self.scan_box.clear()
            for s in scans:
                self.scan_box.addItem(s.label, s.key)
            self.scan_box.blockSignals(False)
            if not scans:
                self.status.setText(f"No {icao} scans in the {self.hours.value()} h ending "
                                    f"{'now' if end is None else end.strftime('%d %b %Y %H:%MZ')}.")
                return
            if not keep_latest or at_end or newer:
                self.scan_box.setCurrentIndex(len(scans) - 1)
                self.load_volume()
            self.status.setText(f"{len(scans)} {icao} scans, {scans[0].time:%H:%M}\u2013{scans[-1].time:%H:%MZ}.")
        job.sig.done.connect(done)
        job.sig.failed.connect(lambda m: (self.btn_list.setEnabled(True), self.status.setText(m)))
        self.pool.start(job)

    def _step(self, d: int):
        i = self.scan_box.currentIndex() + d
        if 0 <= i < self.scan_box.count():
            self.scan_box.setCurrentIndex(i)

    def load_volume(self):
        i = self.scan_box.currentIndex()
        if not (0 <= i < len(self.scans)) or self._busy:
            return
        scan = self.scans[i]
        self._busy = True
        self.status.setText(f"Reading {scan.site} {scan.label} ({scan.size / 1e6:.0f} MB)\u2026")
        job = _Job(lambda st, s: nx.load_volume(s), scan)

        def done(vol):
            self._busy = False
            self.volume = vol
            keep = self.elev_box.currentData()
            self.elev_box.blockSignals(True)
            self.elev_box.clear()
            for e in vol.elevations():
                self.elev_box.addItem(f"{e:.1f}\u00b0", e)
            j = min(range(self.elev_box.count()),
                    key=lambda k: abs(self.elev_box.itemData(k) - keep)) if keep is not None else 0
            self.elev_box.setCurrentIndex(j)
            self.elev_box.blockSignals(False)
            self.redraw()
            if self.scan_box.currentIndex() != i:          # moved on while reading
                self.load_volume()
        job.sig.done.connect(done)
        job.sig.failed.connect(lambda m: (setattr(self, "_busy", False), self.status.setText(m)))
        self.pool.start(job)

    # --------------------------------------------------------------- drawing

    def _n_panels(self) -> int:
        b = self.layout_group.checkedButton()
        return int(b.property("n")) if b else 2

    def _on_layout(self, redraw: bool = True):
        n = self._n_panels()
        for k, cb in enumerate(self.product_boxes):
            cb.setEnabled(k < n)
        if redraw:
            self.redraw()

    def set_recon(self, bundle):
        self.recon = bundle
        self._update_motion_sources()
        if self.volume is not None and self.chk_fixes.isChecked():
            self.redraw()

    def _placeholder(self):
        self.figure.clear()
        self.figure.patch.set_facecolor("#12151b" if self.theme != "light" else "#ffffff")
        self.figure.text(0.5, 0.5, "No radar scan loaded", ha="center", va="center", color="#8b95a5")
        self.canvas.draw_idle()

    def redraw(self):
        vol = self.volume
        if vol is None or self.elev_box.currentData() is None:
            return
        import cartopy.crs as ccrs
        fix = self.storm_box.currentData()
        crs = ccrs.AzimuthalEquidistant(central_longitude=vol.site_lon, central_latitude=vol.site_lat)
        centre = (0.0, 0.0)
        mode = self.centre_group.checkedButton().property("key")
        if mode == "storm" and fix is not None:
            sx, sy = crs.transform_point(fix.lon, fix.lat, ccrs.PlateCarree())
            reach = 460e3
            if np.hypot(sx, sy) < reach + self.width.value() * 1e3:
                centre = (sx, sy)
        products = [cb.currentData() for cb in self.product_boxes[:self._n_panels()]]
        n = 700 if len(products) == 4 else 900
        try:
            mot = self.motion_box.motion()
            panels = rv.render_panels(vol, products, self.elev_box.currentData(), centre,
                                      self.width.value() * 1e3, self.chk_dealias.isChecked(), n, motion=mot)
            notes = ["Storm-relative velocity needs a storm motion (below)" if p == "srv" and mot is None
                     else "" for p in products]
            fixes = []
            if self.chk_fixes.isChecked() and self.recon is not None:
                t = vol.scan.time
                fixes = [f for f in self.recon.fixes if abs((f.time - t).total_seconds()) <= 5400]
            storm = (fix.lat, fix.lon, fix.name) if fix is not None else None
            self.art = rv.draw(self.figure, vol, panels, storm=storm, fixes=fixes, theme=self.theme, notes=notes)
        except Exception as exc:
            self.status.setText(f"Could not draw: {exc}")
            return
        missing = [nx.PRODUCTS[p]["label"] for p, pim in zip(products, panels) if pim is None]
        note = f"  \u00b7  not scanned at this elevation: {', '.join(missing)}" if missing else ""
        self.status.setText(f"{vol.scan.site} {vol.scan.label}, {self.elev_box.currentText()}{note}")
        self.canvas.draw_idle()

    def _on_hover(self, event):
        if self.art is None or event.inaxes not in self.art.axes or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        ground_km = float(np.hypot(x, y)) / 1e3
        brg = (np.degrees(np.arctan2(x, y)) + 360) % 360
        site = sites.STATIONS_BY_ICAO.get(self.volume.scan.site)
        h = nx.beam_height_km(ground_km, self.elev_box.currentData() or 0.5,
                              (site.elev_ft * 0.3048) if site else 0.0)
        bits = []
        for ax in self.art.axes:
            v = rv.value_at(self.art, ax, x, y)
            if v and np.isfinite(v[1]):
                unit = nx.PRODUCTS[v[0]]["units"]
                bits.append(f"{nx.PRODUCTS[v[0]]['label'].split()[0]} {v[1]:.2f}{(' ' + unit) if unit else ''}"
                            if v[0] == "cc" else f"{nx.PRODUCTS[v[0]]['label'].split()[0]} {v[1]:.1f} {unit}")
        self.readout.setText(f"{ground_km:.0f} km / {ground_km / 1.852:.0f} nm at {brg:03.0f}\u00b0  \u00b7  "
                             f"beam {h:.1f} km  \u00b7  " + ("   ".join(bits) or "no echo"))

    def set_theme(self, theme: str):
        self.theme = theme
        if self.volume is not None:
            self.redraw()
        else:
            self._placeholder()

    def state(self) -> dict:
        return {"layout": self._n_panels(), "products": [cb.currentData() for cb in self.product_boxes],
                "width": self.width.value(), "hours": self.hours.value(),
                "dealias": self.chk_dealias.isChecked(),
                "centre": self.centre_group.checkedButton().property("key"),
                "motion": self.motion_box.state()}

    def restore(self, st: dict):
        if not st:
            return
        for b in self.layout_group.buttons():
            b.setChecked(b.property("n") == st.get("layout", 2))
        for cb, p in zip(self.product_boxes, st.get("products", DEFAULTS)):
            i = cb.findData(p)
            if i >= 0:
                cb.setCurrentIndex(i)
        self.width.setValue(int(st.get("width", 220)))
        self.hours.setValue(int(st.get("hours", 3)))
        self.chk_dealias.setChecked(bool(st.get("dealias", True)))
        for b in self.centre_group.buttons():
            b.setChecked(b.property("key") == st.get("centre", "storm"))
        self.motion_box.restore(st.get("motion") or {})
