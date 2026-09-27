#!/usr/bin/env python3
"""
recon10s_satpanel — the Satellite tab.

GOES imagery with recon obs drawn as wind barbs, coloured by speed. Loops of
1, 15 or 30 frames: frames load in parallel and appear as they arrive, so the
slider is usable before the last one lands. Scrubbing reuses the map and only
swaps the image and the barbs (~15 ms a frame against ~60 ms for a full draw).

Obs are synced to the frame: each frame shows the obs up to its scan time,
so moving through the loop replays the flight. The newest frame shows every
ob, because the imagery is always a few minutes behind the aircraft -- the
corner label gives the gap in minutes.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
from typing import Dict, List, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QCursor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMenu, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

import recon10s_drops as drops_mod
import recon10s_nhchdob as nhchdob
import recon10s_goes as goes
from recon10s_dropwin import DropsondeWindow
import recon10s_nhc as nhc
import recon10s_satview as satview

SECTORS = [("Auto (meso if it covers the storm)", "auto"),
           ("Full disk", "F"), ("Mesoscale 1", "M1"), ("Mesoscale 2", "M2")]
SATELLITES = [("Auto by longitude", None), ("GOES-19 (east)", 19), ("GOES-18 (west)", 18)]
FRAME_COUNTS = [("Latest only", 1), ("15 frames", 15), ("30 frames", 30)]
PLAY_MS = 220          # per frame
LAST_FRAME_DWELL = 4   # the newest frame holds this many ticks before looping


class _Signals(QObject):
    frame = pyqtSignal(int, object)      # position in the loop, GoesImage
    started = pyqtSignal(object, str)    # list of scenes, why
    done = pyqtSignal(int)               # frames that could not be used
    failed = pyqtSignal(str)
    status = pyqtSignal(str)


class _LoopWorker(QRunnable):
    """Pick the scene set for a storm, then load every frame in parallel."""

    def __init__(self, lat, lon, band, sector, sat, half_width, count):
        super().__init__()
        self.lat, self.lon = lat, lon
        self.band, self.sector, self.sat = band, sector, sat
        self.half_width, self.count = half_width, count
        self.signals = _Signals()

    @pyqtSlot()
    def run(self):
        try:
            self.signals.status.emit("Looking for the newest scene\u2026")
            newest, why = goes.choose_scene(self.lat, self.lon, self.band,
                                            prefer=self.sector, sat=self.sat)
            if newest is None:
                self.signals.failed.emit("No imagery found for that band in the last few hours.")
                return
            if self.count > 1:
                scenes = goes.recent_scenes(newest.sat, self.band, newest.sector, self.count)
            else:
                scenes = [newest]
            self.signals.started.emit(scenes, why)

            half = self.half_width
            bbox = (self.lat - half, self.lat + half, self.lon - half, self.lon + half)
            meso = newest.sector != "F"
            dropped = 0
            with cf.ThreadPoolExecutor(max_workers=6) as pool:
                futures = {pool.submit(goes.load_image, s, bbox): i for i, s in enumerate(scenes)}
                for fut in cf.as_completed(futures):
                    i = futures[fut]
                    try:
                        img = fut.result()
                    except Exception:
                        dropped += 1
                        continue
                    # A mesoscale sector can be moved mid-loop; frames from
                    # when it pointed elsewhere show nothing useful.
                    if meso and img.extent_lonlat and not goes.covers(img.extent_lonlat,
                                                                      self.lat, self.lon):
                        dropped += 1
                        continue
                    self.signals.frame.emit(i, img)
            self.signals.done.emit(dropped)
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


class _TaskSignals(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)


class _Task(QRunnable):
    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args
        self.signals = _TaskSignals()

    @pyqtSlot()
    def run(self):
        try:
            self.signals.done.emit(self.fn(self.signals.status.emit, *self.args))
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


def _frame_at(lat, lon, when, band=13):
    """
    The full-disk scene closest before `when`, trimmed to the area around a
    drop. Looks in the drop's hour and the hour before, since a drop at
    14:00:30 wants the 13:50 scan, which lives in the previous hour's folder.
    """
    sat = goes.pick_satellite(lat, lon)
    scenes = {}
    for hour in (when, when - dt.timedelta(hours=1)):
        for sc in goes.list_scenes(sat, band, "F", when=hour, hours_back=0):
            scenes[sc.key] = sc
    if not scenes:
        return None
    ordered = sorted(scenes.values(), key=lambda sc: sc.start)
    before = [sc for sc in ordered if sc.start <= when]
    pick = before[-1] if before else ordered[0]
    return goes.load_image(pick, bbox=(lat - 3.2, lat + 3.2, lon - 3.2, lon + 3.2))


class SatellitePanel(QWidget):
    """GOES loop with recon barbs over it."""

    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self.theme = theme
        self.pool = QThreadPool.globalInstance()

        self._frames: List[Optional[object]] = []   # GoesImage per slot, None until loaded
        self._why = ""
        self._fix: Optional[nhc.Fix] = None
        self._overlay: Optional[satview.Overlay] = None
        self._storms: list = []
        self._busy = False
        self._art: Optional[satview.SceneArtists] = None
        self._shown = -1
        self._dwell = 0
        self._drops: List = []
        self._converted: Optional[satview.Overlay] = None   # from the IWG1 converter
        self._batches: List = []                             # NHC HDOB batches
        self._tree_timer = QTimer(self)
        self._tree_timer.setSingleShot(True)
        self._tree_timer.setInterval(150)
        self._tree_timer.timeout.connect(self._rebuild_overlay)
        self._drop_windows: List[DropsondeWindow] = []
        self._inset_cache: Dict[tuple, object] = {}

        self.figure = Figure(figsize=(7, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("resize_event", self._on_resize)

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(PLAY_MS)
        self.play_timer.timeout.connect(self._advance)

        # Controls live in a left column so the map gets the full height; a
        # wide strip of controls across the top squeezed the map into a
        # short band, and a geographic map is limited by its shorter side.
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.setSpacing(10)
        lv.addWidget(self._build_controls())
        lv.addWidget(self._build_layers())
        lv.addWidget(self._build_sidebar(), 1)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(460)

        map_col = QWidget()
        mc = QVBoxLayout(map_col)
        mc.setContentsMargins(0, 0, 0, 0)
        mc.setSpacing(6)
        mc.addWidget(self.toolbar)
        mc.addWidget(self.canvas, 1)
        mc.addLayout(self._build_loop_bar())
        self.status = QLabel("No imagery loaded.")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        mc.addWidget(self.status)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(map_col)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([340, 1200])
        split.setChildrenCollapsible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split, 1)

        self.redraw()

    # -------------------------------------------------------------- controls

    @staticmethod
    def _narrow(combo: QComboBox, chars: int = 16) -> QComboBox:
        """Let a combo box shrink to the column; its drop-down still shows full text."""
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(chars)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return combo

    def _build_controls(self) -> QWidget:
        box = QGroupBox("Storm && satellite")
        form = QFormLayout(box)
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(7)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        storm_row = QHBoxLayout()
        self.storm_box = self._narrow(QComboBox())
        self.storm_box.setToolTip("Active NHC systems. The position decides which\n"
                                  "satellite and which sector to pull.")
        self.storm_box.currentIndexChanged.connect(
            lambda _: self.storm_box.setToolTip(self.storm_box.currentText()))
        storm_row.addWidget(self.storm_box, 1)
        b_storms = QPushButton("\u21bb")
        b_storms.setToolTip("Refresh the active storm list from NHC")
        b_storms.setFixedWidth(34)
        b_storms.clicked.connect(self.refresh_storms)
        storm_row.addWidget(b_storms)
        form.addRow("Storm", storm_row)

        self.band_box = self._narrow(QComboBox())
        for band in sorted(goes.BANDS):
            meta = goes.BANDS[band]
            self.band_box.addItem(f"{band} \u2013 {meta['name']} ({meta['micron']} \u00b5m)", band)
        self.band_box.setCurrentIndex(self.band_box.findData(13))
        self.band_box.setToolTip("Band 2 full disk is ~20 MB a frame even after trimming,\n"
                                 "so a 30-frame visible loop is a large download.")
        form.addRow("Band", self.band_box)

        self.sector_box = self._narrow(QComboBox())
        for label, value in SECTORS:
            self.sector_box.addItem(label, value)
        form.addRow("Sector", self.sector_box)

        self.sat_box = self._narrow(QComboBox())
        for label, value in SATELLITES:
            self.sat_box.addItem(label, value)
        form.addRow("Satellite", self.sat_box)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.5, 15.0)
        self.width_spin.setSingleStep(0.5)
        self.width_spin.setValue(4.0)
        self.width_spin.setSuffix("\u00b0")
        self.width_spin.setToolTip("How far either side of the storm to pull imagery.")
        form.addRow("Half-width", self.width_spin)

        self.count_box = self._narrow(QComboBox(), 10)
        for label, value in FRAME_COUNTS:
            self.count_box.addItem(label, value)
        self.count_box.setCurrentIndex(1)
        self.count_box.setToolTip("Full disk scans every 10 min (30 frames = 5 h);\n"
                                  "mesoscale every minute (30 frames = 30 min).")
        form.addRow("Frames", self.count_box)

        self.btn_fetch = QPushButton("Fetch imagery")
        self.btn_fetch.setObjectName("primary")
        self.btn_fetch.clicked.connect(self.fetch)
        form.addRow(self.btn_fetch)
        return box

    def _build_layers(self) -> QWidget:
        box = QGroupBox("Layers")
        grid = QGridLayout(box)
        grid.setContentsMargins(10, 8, 10, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        self.chk_sat = QCheckBox("Imagery")
        self.chk_barbs = QCheckBox("Wind barbs")
        self.chk_track = QCheckBox("Track")
        self.chk_cbar = QCheckBox("Colour bar")
        self.chk_fix = QCheckBox("NHC fix")
        self.chk_drops = QCheckBox("Dropsondes")
        for chk in (self.chk_sat, self.chk_barbs, self.chk_track, self.chk_cbar,
                    self.chk_fix, self.chk_drops):
            chk.setChecked(True)
            chk.toggled.connect(self.redraw)
        self.chk_declutter = QCheckBox("Declutter barbs")
        self.chk_declutter.setToolTip("Drop barbs that would overlap on screen.\n"
                                      "Zoom in and the hidden ones come back.")
        self.chk_declutter.toggled.connect(self.redraw)
        self.chk_sync = QCheckBox("Obs up to frame time")
        self.chk_sync.setChecked(True)
        self.chk_sync.setToolTip("Each frame shows only the obs taken by its scan time,\n"
                                 "so the loop replays the flight. The newest frame\n"
                                 "always shows every ob.")
        self.chk_sync.toggled.connect(lambda _: self._show_frame(self.slider.value(), force=True))
        self.chk_follow_track = QCheckBox("Zoom to recon")
        self.chk_follow_track.setChecked(True)
        self.chk_follow_track.setToolTip("Frame the map on the recon obs rather than the storm.")
        self.chk_follow_track.toggled.connect(self.redraw)

        order = (self.chk_sat, self.chk_cbar, self.chk_barbs, self.chk_track,
                 self.chk_drops, self.chk_fix, self.chk_declutter, self.chk_follow_track)
        for i, chk in enumerate(order):
            grid.addWidget(chk, i // 2, i % 2)
        grid.addWidget(self.chk_sync, len(order) // 2, 0, 1, 2)
        return box

    def _build_sidebar(self) -> QWidget:
        box = QGroupBox("Recon on the map")
        v = QVBoxLayout(box)
        v.setSpacing(8)

        self.chk_converted = QCheckBox("Converted HDOB (IWG1)")
        self.chk_converted.setEnabled(False)
        self.chk_converted.setChecked(True)
        self.chk_converted.toggled.connect(self._rebuild_overlay)
        v.addWidget(self.chk_converted)
        self.converted_note = QLabel("None yet \u2014 convert one under Reconnaissance.")
        self.converted_note.setObjectName("hint")
        self.converted_note.setWordWrap(True)
        v.addWidget(self.converted_note)

        v.addWidget(QLabel("<b>NHC HDOBs</b> \u2014 USAF and NOAA, 30-s obs"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Last"))
        self.hdob_hours = QSpinBox()
        self.hdob_hours.setRange(1, 72)
        self.hdob_hours.setValue(12)
        self.hdob_hours.setSuffix(" h")
        row.addWidget(self.hdob_hours)
        row.addStretch(1)
        self.btn_hdobs = QPushButton("Load NHC HDOBs")
        self.btn_hdobs.setToolTip("Every HDOB batch NHC has published for this storm in that\n"
                                  "window (AHONT1 Atlantic, AHOPN1 Pacific), grouped by mission.\n"
                                  "Tick the batches to plot.")
        self.btn_hdobs.clicked.connect(self.load_hdobs)
        row.addWidget(self.btn_hdobs)
        v.addLayout(row)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemChanged.connect(lambda *_: self._tree_timer.start())
        v.addWidget(self.tree, 1)
        sel = QHBoxLayout()
        for label, state in (("All", Qt.CheckState.Checked), ("None", Qt.CheckState.Unchecked)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, st=state: self._check_all(st))
            sel.addWidget(b)
        b_last = QPushButton("Newest mission")
        b_last.setToolTip("Tick only the most recent mission")
        b_last.clicked.connect(self._check_newest)
        sel.addWidget(b_last)
        v.addLayout(sel)

        v.addWidget(QLabel("<b>Dropsondes</b>"))
        drow = QHBoxLayout()
        self.btn_drops = QPushButton("Load dropsondes")
        self.btn_drops.setToolTip("TEMP DROP messages from NHC's recon archive for this storm.\n"
                                  "Covers the span of whatever recon is on the map, or the\n"
                                  "last 12 hours. Click a dot for its sounding.")
        self.btn_drops.clicked.connect(self.load_drops)
        drow.addWidget(self.btn_drops)
        self.drop_note = QLabel("")
        self.drop_note.setObjectName("hint")
        drow.addWidget(self.drop_note, 1)
        v.addLayout(drow)
        return box

    # ------------------------------------------------------------ overlays

    def _rebuild_overlay(self):
        """Combine the converted HDOB and every ticked NHC batch into one overlay."""
        parts = []
        if self._converted is not None and self.chk_converted.isChecked():
            parts.append(self._converted)
        ticked = []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                child = top.child(j)
                if child.checkState(0) == Qt.CheckState.Checked:
                    ticked.append(child.data(0, Qt.ItemDataRole.UserRole))
        if ticked:
            parts.append(satview.overlay_from_batches(ticked))
        self._overlay = satview.merge_overlays(parts)
        if self._frames:
            self._show_frame(self.slider.value(), force=True)
        else:
            self.redraw()

    def load_hdobs(self):
        fix = self.current_fix()
        if fix is None:
            self.status.setText("Pick a storm before loading NHC HDOBs.")
            return
        until = dt.datetime.now(dt.timezone.utc)
        since = until - dt.timedelta(hours=int(self.hdob_hours.value()))
        self.btn_hdobs.setEnabled(False)

        def work(status, since, until, fix):
            return nhchdob.fetch_batches(since, until, storm_id=fix.storm_id,
                                         storm_name=fix.name if fix.storm_id else "",
                                         centre=(fix.lat, fix.lon), progress=status)

        task = _Task(work, since, until, fix)
        task.signals.status.connect(self.status.setText)
        task.signals.done.connect(self._on_hdobs)
        task.signals.failed.connect(lambda msg: (self.btn_hdobs.setEnabled(True),
                                                 self.status.setText(f"NHC HDOBs: {msg}")))
        self.pool.start(task)

    def _on_hdobs(self, batches):
        self.btn_hdobs.setEnabled(True)
        self._batches = batches
        self.tree.blockSignals(True)
        self.tree.clear()
        for key, group in nhchdob.group_missions(batches).items():
            nums = [b.number for b in group]
            top = QTreeWidgetItem([f"{' '.join(key)} {group[0].storm}  \u00b7  HDOB "
                                   f"{min(nums):02d}\u2013{max(nums):02d}  \u00b7  "
                                   f"{group[0].start:%H:%M}\u2013{group[-1].end:%H:%MZ}"])
            top.setFlags(top.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
            top.setCheckState(0, Qt.CheckState.Checked)
            for b in group:
                peak = f"max {b.max_wind} kt" if b.max_wind is not None else ""
                sfmr = f", SFMR {b.max_sfmr}" if b.max_sfmr else ""
                child = QTreeWidgetItem([f"HDOB {b.number:02d}  {b.start:%H:%M}\u2013{b.end:%H:%MZ}  {peak}{sfmr}"])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Checked)
                child.setData(0, Qt.ItemDataRole.UserRole, b)
                top.addChild(child)
            self.tree.addTopLevelItem(top)
        self.tree.blockSignals(False)
        missions = self.tree.topLevelItemCount()
        self.status.setText(f"{len(batches)} NHC HDOB batch(es) in {missions} mission(s). "
                            "Untick batches to hide them." if batches else
                            "No NHC HDOBs for this storm in that window.")
        self._rebuild_overlay()

    def _check_all(self, state):
        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setCheckState(0, state)
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                top.child(j).setCheckState(0, state)
        self.tree.blockSignals(False)
        self._rebuild_overlay()

    def _check_newest(self):
        n = self.tree.topLevelItemCount()
        if not n:
            return
        newest = max(range(n), key=lambda i: self.tree.topLevelItem(i).child(
            self.tree.topLevelItem(i).childCount() - 1).data(0, Qt.ItemDataRole.UserRole).end)
        self.tree.blockSignals(True)
        for i in range(n):
            state = Qt.CheckState.Checked if i == newest else Qt.CheckState.Unchecked
            top = self.tree.topLevelItem(i)
            top.setCheckState(0, state)
            for j in range(top.childCount()):
                top.child(j).setCheckState(0, state)
        self.tree.blockSignals(False)
        self._rebuild_overlay()

    def _build_loop_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.btn_prev = QPushButton("\u25c0")
        self.btn_play = QPushButton("\u25b6")
        self.btn_next = QPushButton("\u25b6\u25b6")
        for b in (self.btn_prev, self.btn_play, self.btn_next):
            b.setFixedWidth(42)
        self.btn_prev.setToolTip("Previous frame")
        self.btn_play.setToolTip("Play / pause the loop")
        self.btn_next.setToolTip("Next frame")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(+1))
        self.btn_play.clicked.connect(self.toggle_play)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setPageStep(1)
        self.slider.valueChanged.connect(self._show_frame)

        self.frame_label = QLabel("\u2014")
        self.frame_label.setObjectName("probeReadout")
        self.frame_label.setMinimumWidth(170)

        bar.addWidget(self.btn_prev)
        bar.addWidget(self.btn_play)
        bar.addWidget(self.btn_next)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.frame_label)
        self._set_loop_enabled(False)
        return bar

    def _set_loop_enabled(self, on: bool):
        for w in (self.btn_prev, self.btn_play, self.btn_next, self.slider):
            w.setEnabled(on)

    # ------------------------------------------------------------------ data

    def refresh_storms(self):
        self.storm_box.clear()
        try:
            self._storms = nhc.active_storms()
        except Exception as exc:
            self._storms = []
            self.status.setText(f"Could not reach NHC: {exc}")
        for fix in self._storms:
            self.storm_box.addItem(fix.label, fix)
        self.storm_box.addItem("Use the recon track instead", None)

    def set_overlay(self, overlay: Optional[satview.Overlay]):
        """The HDOB converted from IWG1 (Reconnaissance section)."""
        self._converted = overlay
        has = overlay is not None and len(overlay) > 0
        self.chk_converted.setEnabled(has)
        if has:
            self.converted_note.setText(f"{len(overlay)} obs, "
                                        f"{overlay.times[0]:%d %b %H:%M}\u2013{overlay.times[-1]:%H:%MZ}")
        if has and not self._storms:
            self.refresh_storms()
        self._rebuild_overlay()

    def current_fix(self) -> Optional[nhc.Fix]:
        fix = self.storm_box.currentData()
        if fix is not None:
            return fix
        if self._overlay is not None and len(self._overlay):
            return nhc.fix_from_positions(self._overlay.lats, self._overlay.lons)
        return None

    # -------------------------------------------------------------- fetching

    def fetch(self):
        if self._busy:
            return
        fix = self.current_fix()
        if fix is None:
            self.status.setText("Nothing to centre on yet \u2014 convert an HDOB or pick a storm.")
            return
        self._fix = fix
        self.stop()
        self._busy = True
        self.btn_fetch.setEnabled(False)

        worker = _LoopWorker(fix.lat, fix.lon, self.band_box.currentData(),
                             self.sector_box.currentData(), self.sat_box.currentData(),
                             float(self.width_spin.value()), int(self.count_box.currentData()))
        worker.signals.status.connect(self.status.setText)
        worker.signals.started.connect(self._on_started)
        worker.signals.frame.connect(self._on_frame)
        worker.signals.done.connect(self._on_done)
        worker.signals.failed.connect(self._on_failed)
        self.pool.start(worker)
        if self.chk_drops.isChecked():
            self.load_drops()

    # ---------------------------------------------------------------- drops

    def _drop_window_span(self):
        now = dt.datetime.now(dt.timezone.utc)
        ov = self._overlay
        if ov is not None and len(ov) and ov.times and ov.times[0]:
            return ov.times[0] - dt.timedelta(hours=1), ov.times[-1] + dt.timedelta(hours=1)
        return now - dt.timedelta(hours=12), now

    def load_drops(self):
        fix = self.current_fix()
        if fix is None:
            self.status.setText("Pick a storm or load an HDOB before loading dropsondes.")
            return
        since, until = self._drop_window_span()
        self.btn_drops.setEnabled(False)

        def work(status, since, until, fix):
            return drops_mod.fetch_drops(since, until, storm_id=fix.storm_id,
                                         storm_name=fix.name if fix.storm_id else "",
                                         centre=(fix.lat, fix.lon), progress=status)

        task = _Task(work, since, until, fix)
        task.signals.status.connect(self.status.setText)
        task.signals.done.connect(self._on_drops)
        task.signals.failed.connect(lambda msg: (self.btn_drops.setEnabled(True),
                                                 self.status.setText(f"Dropsondes: {msg}")))
        self.pool.start(task)

    def _on_drops(self, drops):
        self.btn_drops.setEnabled(True)
        self._drops = drops
        if drops:
            first, last = drops[0].release_time, drops[-1].release_time
            span = f" {first:%H:%M}\u2013{last:%H:%MZ}" if first and last else ""
            self.status.setText(f"{len(drops)} dropsonde(s){span} \u2014 click a dot for its sounding.")
            self.drop_note.setText(f"{len(drops)} loaded{span}")
        else:
            self.status.setText("No dropsondes for this storm in that window.")
        self._show_frame(self.slider.value(), force=True) if self._frames else self.redraw()

    def _centre_for(self, drop):
        """
        Storm centre at the time of a drop: the nearest centre/eye drop within
        three hours if there is one (a far better centre than a fix hours
        away), otherwise the NHC position.
        """
        best, gap = None, None
        for d in self._drops:
            if d.location not in ("Center", "Eye") or not (d.release_time and drop.release_time):
                continue
            g = abs((d.release_time - drop.release_time).total_seconds())
            if g <= 3 * 3600 and (gap is None or g < gap):
                lat = d.splash_lat if d.splash_lat is not None else d.release_lat
                lon = d.splash_lon if d.splash_lon is not None else d.release_lon
                best, gap = (lat, lon), g
        if best:
            return best
        fix = self._fix or self.current_fix()
        return (fix.lat, fix.lon) if fix else None

    def _sat_for(self, drop):
        """A loop frame within 15 minutes of the drop, if one is loaded."""
        if not drop.release_time:
            return None
        key = (drop.aircraft, drop.release_time)
        if key in self._inset_cache:
            return self._inset_cache[key]
        near = [f for f in self._frames if f is not None
                and abs((f.start - drop.release_time).total_seconds()) <= 900]
        if near:
            return min(near, key=lambda f: abs((f.start - drop.release_time).total_seconds()))
        return None

    def _open_drop(self, drop):
        idx = self._drops.index(drop) if drop in self._drops else 0
        win = DropsondeWindow(self._drops, idx, self._sat_for, self._centre_for, theme=self.theme)
        win.on_show = lambda d, w=win: self._fetch_inset(d, w)
        self._drop_windows.append(win)
        win.destroyed.connect(lambda *_: self._drop_windows.remove(win)
                              if win in self._drop_windows else None)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        win.show()
        self._fetch_inset(drop, win)   # first drop: on_show was not wired yet during __init__

    def _fetch_inset(self, drop, win):
        """Fetch the frame from the drop's own time for its inset, in the background."""
        key = (drop.aircraft, drop.release_time)
        if not drop.release_time or key in self._inset_cache or self._sat_for(drop) is not None:
            return

        def work(_status, drop):
            return _frame_at(drop.release_lat, drop.release_lon, drop.release_time)

        def done(img):
            self._inset_cache[key] = img
            try:
                if win.isVisible() and win.drops[win.index] is drop:
                    win.show_index(win.index)
            except RuntimeError:
                pass  # window closed meanwhile

        task = _Task(work, drop)
        task.signals.done.connect(done)
        self.pool.start(task)

    def _on_resize(self, _event):
        # Margins are kept in pixels, so they are re-worked out for the new size.
        satview.fit_layout(self.figure, self._art)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.button != 1 or self._art is None or event.inaxes is not self._art.ax:
            return
        if getattr(self.toolbar, "mode", ""):
            return  # pan/zoom owns the mouse
        near = satview.drops_near(self._art, event.x, event.y)
        if not near:
            return
        if len(near) == 1:
            self._open_drop(near[0])
            return
        # Several dots under the cursor (repeat eyewall passes): let the user choose.
        menu = QMenu(self)
        for d in near:
            when = f"{d.release_time:%H:%MZ}" if d.release_time else ""
            wind = f"{d.surface.wspd} kt sfc" if d.surface and d.surface.wspd is not None else ""
            act = QAction(f"#{d.number}  {when}  {d.location or ''}  {wind}".replace("  ", "  "), menu)
            act.triggered.connect(lambda _=False, d=d: self._open_drop(d))
            menu.addAction(act)
        menu.exec(QCursor.pos())

    def _on_started(self, scenes, why):
        self._why = why
        self._frames = [None] * len(scenes)
        self._shown = -1
        self._art = None
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(scenes) - 1))
        self.slider.setValue(len(scenes) - 1)
        self.slider.blockSignals(False)
        self.status.setText(f"Loading {len(scenes)} frame(s) \u2014 {why}\u2026")

    def _on_frame(self, i, img):
        self._frames[i] = img
        loaded = sum(1 for f in self._frames if f is not None)
        self.status.setText(f"{loaded} of {len(self._frames)} frames \u2014 {self._why}")
        self._set_loop_enabled(loaded > 1 or len(self._frames) == 1)
        # Show the newest frame as soon as it lands; otherwise leave the view alone.
        if i == self.slider.value() or self._shown < 0:
            self._show_frame(self.slider.value(), force=True)
        self._update_frame_label()

    def _on_done(self, dropped):
        self._busy = False
        self.btn_fetch.setEnabled(True)
        loaded = [f for f in self._frames if f is not None]
        if not loaded:
            self.status.setText("None of the frames could be used.")
            return
        mb = sum(f.bytes_fetched for f in loaded) / 1e6
        note = f", {dropped} dropped (sector elsewhere or unreadable)" if dropped else ""
        first, last = loaded[0].start, loaded[-1].start
        self.status.setText(f"{len(loaded)} frames {first:%H:%M}\u2013{last:%H:%MZ}{note}  "
                            f"\u00b7  {self._why}  \u00b7  {mb:.0f} MB pulled")
        self._set_loop_enabled(len(loaded) > 1)
        self._show_frame(self.slider.value(), force=True)

    def _on_failed(self, message: str):
        self._busy = False
        self.btn_fetch.setEnabled(True)
        self.status.setText(message)

    # --------------------------------------------------------------- playback

    def _loaded_index(self, i: int) -> Optional[int]:
        """The loaded frame at or nearest before slot i (frames arrive out of order)."""
        for j in range(min(i, len(self._frames) - 1), -1, -1):
            if self._frames[j] is not None:
                return j
        for j in range(i, len(self._frames)):
            if self._frames[j] is not None:
                return j
        return None

    def _step(self, delta: int):
        n = len(self._frames)
        if n:
            self.slider.setValue((self.slider.value() + delta) % n)

    def toggle_play(self):
        if self.play_timer.isActive():
            self.stop()
        else:
            self._dwell = 0
            self.play_timer.start()
            self.btn_play.setText("\u275a\u275a")

    def stop(self):
        self.play_timer.stop()
        self.btn_play.setText("\u25b6")

    def _advance(self):
        n = len(self._frames)
        if n < 2:
            self.stop()
            return
        if self.slider.value() == n - 1:
            self._dwell += 1
            if self._dwell < LAST_FRAME_DWELL:
                return
        self._dwell = 0
        nxt = (self.slider.value() + 1) % n
        while self._frames[nxt] is None and nxt != self.slider.value():
            nxt = (nxt + 1) % n  # skip frames that were dropped or are still loading
        self.slider.setValue(nxt)

    def _update_frame_label(self):
        n = len(self._frames)
        if not n:
            self.frame_label.setText("\u2014")
            return
        j = self._loaded_index(self.slider.value())
        when = self._frames[j].start.strftime("%H:%M:%SZ") if j is not None else "loading"
        self.frame_label.setText(f"{self.slider.value() + 1:>2}/{n}  {when}")

    # --------------------------------------------------------------- drawing

    def set_theme(self, theme: str):
        self.theme = theme
        self.redraw()
        for win in list(self._drop_windows):
            win.theme = theme
            win.show_index(win.index)

    def _extent(self):
        if self._overlay is not None and len(self._overlay) and self.chk_follow_track.isChecked():
            south, north, west, east = self._overlay.bbox
            return (west, east, south, north)
        return None

    def _obs_until(self, j: Optional[int]):
        if j is None or not self.chk_sync.isChecked():
            return None
        newest = self._loaded_index(len(self._frames) - 1)
        if j == newest:
            return None       # the newest frame shows every ob
        return self._frames[j].start

    def _spacing(self) -> float:
        return 22.0 if self.chk_declutter.isChecked() else 0.0

    def _show_frame(self, i: int, force: bool = False):
        self._update_frame_label()
        j = self._loaded_index(i) if self._frames else None
        if j is None:
            if force:
                self.redraw()
            return
        if j == self._shown and not force:
            return
        img = self._frames[j]
        title = satview.scene_title(img, self._fix, self._why)
        swapped = satview.update_frame(
            self._art, img, self._overlay, self._obs_until(j),
            self.chk_barbs.isChecked(), self.chk_track.isChecked(),
            self._spacing(), self.theme, title, drops=self._drops,
            show_drops=self.chk_drops.isChecked()) if (self._art and self.chk_sat.isChecked()) else False
        self._shown = j
        if swapped:
            self.canvas.draw_idle()
        else:
            self.redraw()

    def redraw(self):
        j = self._loaded_index(self.slider.value()) if self._frames else None
        img = self._frames[j] if j is not None else None
        try:
            self._art = satview.draw_scene(
                self.figure, img=img, overlay=self._overlay, fix=self._fix,
                show_satellite=self.chk_sat.isChecked(),
                show_track=self.chk_track.isChecked(),
                show_barbs=self.chk_barbs.isChecked(),
                show_colorbar=self.chk_cbar.isChecked(),
                show_fix=self.chk_fix.isChecked(),
                extent=self._extent(), theme=self.theme,
                title=satview.scene_title(img, self._fix, self._why),
                obs_until=self._obs_until(j), spacing_px=self._spacing(),
                drops=self._drops, show_drops=self.chk_drops.isChecked())
        except Exception as exc:
            self.status.setText(f"Could not draw the map: {exc}")
            return
        self._shown = j if j is not None else -1
        self.canvas.draw_idle()
