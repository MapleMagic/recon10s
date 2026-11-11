#!/usr/bin/env python3
"""
recon10s_satpanel — the Satellite tab.

Left: storm and imagery settings, then the overlays grouped into categories
(a collapsible toolbox): Recon obs, Fixes & dropsondes, Forecast, Lightning,
Live & alerts, Tools. Right: the map and the loop controls.

This panel also owns the storm's recon data (NHC HDOBs, vortex messages,
dropsondes, the converted IWG1 HDOB) and publishes it through
`recon_changed`, so the Analysis and Microwave tabs work from the same set.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
from typing import Dict, List, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QCursor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QMenu, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QSplitter, QToolBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

import recon10s_analysis as analysis
import recon10s_drops as drops_mod
import recon10s_forecast as forecast
import recon10s_glm as glm
import recon10s_goes as goes
import recon10s_nhc as nhc
import recon10s_nhchdob as nhchdob
import recon10s_satview as satview
import recon10s_vdm as vdm
from recon10s_dropwin import DropsondeWindow

SECTORS = [("Auto (meso or target area if over the storm)", "auto"),
           ("Full disk", "F"), ("Mesoscale 1", "M1"), ("Mesoscale 2", "M2"), ("Target area (Himawari)", "T")]
SATELLITES = [("Auto by longitude", None), ("GOES-19 (east)", 19), ("GOES-18 (west)", 18),
              ("Himawari-9 (west Pacific)", 9)]
FRAME_COUNTS = [("Latest only", 1), ("15 frames", 15), ("30 frames", 30)]
FLAG_MODES = [("Fade flagged winds", "fade"), ("Hide flagged winds", "hide"), ("Show all winds", "show")]
PLAY_MS = 220
LAST_FRAME_DWELL = 4


class ReconBundle:
    """What the other tabs need from here, in one object."""

    def __init__(self):
        self.fix: Optional[nhc.Fix] = None
        self.overlay: Optional[satview.Overlay] = None
        self.batches: list = []
        self.drops: list = []
        self.vdms: list = []
        self.fixes: list = []
        self.passes: list = []
        self.tilts: list = []
        self.motion = None


class _Signals(QObject):
    frame = pyqtSignal(int, object)
    started = pyqtSignal(object, str)
    done = pyqtSignal(int)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)


class _LoopWorker(QRunnable):
    """Pick the scans for a storm, then load every frame (single band or product) in parallel."""

    def __init__(self, lat, lon, product, sector, sat, half_width, count, only_after=None, until=None):
        super().__init__()
        self.lat, self.lon, self.product = lat, lon, product
        self.sector, self.sat = sector, sat
        self.half_width, self.count = half_width, count
        self.only_after = only_after
        self.until = until                  # a past moment (case replay); None = now
        self.signals = _Signals()

    def _ref_band(self) -> int:
        if isinstance(self.product, int):
            return self.product
        bands = goes.PRODUCTS[self.product]["bands"]
        return 13 if 13 in bands else bands[0]

    @pyqtSlot()
    def run(self):
        try:
            self.signals.status.emit("Looking for the newest scan\u2026")
            band = self._ref_band()
            sat = self.sat or goes.pick_satellite(self.lat, self.lon)
            note = goes.view_note(sat, self.lat, self.lon)
            if not goes.sees(sat, self.lat, self.lon):
                self.signals.failed.emit(note[:1].upper() + note[1:] + ".")
                return
            newest, why = goes.choose_scene(self.lat, self.lon, band, prefer=self.sector, sat=sat,
                                            when=self.until)
            if note:
                why += f"; {note}"
            if newest is None:
                self.signals.failed.emit("No imagery found for that band in the last few hours.")
                return
            scenes = goes.recent_scenes(newest.sat, band, newest.sector, self.count, when=self.until) \
                if self.count > 1 else [newest]
            if self.until is not None:
                why += f" (up to {self.until:%d %b %H:%MZ})"
            if self.only_after is not None:
                scenes = [s for s in scenes if s.start > self.only_after]
            self.signals.started.emit(scenes, why)
            h = self.half_width
            bbox = (self.lat - h, self.lat + h, self.lon - h, self.lon + h)
            meso = newest.sector != "F"
            dropped = 0
            with cf.ThreadPoolExecutor(max_workers=6) as pool:
                futs = {pool.submit(goes.load_product, self.product, s, bbox): i
                        for i, s in enumerate(scenes)}
                for fut in cf.as_completed(futs):
                    i = futs[fut]
                    try:
                        img = fut.result()
                    except Exception:
                        dropped += 1
                        continue
                    if meso and img.extent_lonlat and not goes.covers(img.extent_lonlat, self.lat, self.lon):
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
    """The full-disk scene closest before `when`, trimmed to the area around a drop."""
    sat = goes.pick_satellite(lat, lon)
    if not goes.sees(sat, lat, lon):
        return None
    scenes = {}
    for hour in (when, (when - dt.timedelta(hours=1)).replace(minute=59, second=59)):
        for sc in goes.list_scenes(sat, band, "F", when=hour, hours_back=0):
            scenes[sc.key] = sc
    if not scenes:
        return None
    ordered = sorted(scenes.values(), key=lambda sc: sc.start)
    before = [sc for sc in ordered if sc.start <= when]
    pick = before[-1] if before else ordered[0]
    return goes.load_image(pick, bbox=(lat - 3.2, lat + 3.2, lon - 3.2, lon + 3.2))


class SatellitePanel(QWidget):
    recon_changed = pyqtSignal(object)      # ReconBundle
    alert = pyqtSignal(str, str)            # title, message

    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self.theme = theme
        self.pool = QThreadPool.globalInstance()
        self._frames: List[Optional[object]] = []
        self._why = ""
        self._fix: Optional[nhc.Fix] = None
        self._overlay: Optional[satview.Overlay] = None
        self._storms: list = []
        self._busy = False
        self._art: Optional[satview.SceneArtists] = None
        self._shown = -1
        self._dwell = 0
        self._append_base = 0
        self._drops: List = []
        self._drop_windows: List[DropsondeWindow] = []
        self._inset_cache: Dict[tuple, object] = {}
        self._converted: Optional[satview.Overlay] = None
        self._batches: List = []
        self._vdms: List = []
        self._bundle = ReconBundle()
        self._forecast = None
        self._models: list = []
        self._glm: Dict[tuple, tuple] = {}
        self._glm_pending: set = set()
        self._loaded = {"hdobs": False, "vdms": False, "drops": False, "forecast": False}
        self._alerted: set = set()
        self._restore_unticked: set = set()
        self._measure = None
        self._measure_artists: list = []
        self._tdr_layer = None
        self.replay_mode = False            # a saved case is being replayed: no live fetching
        self._case_fix = None

        self._tree_timer = QTimer(self)
        self._tree_timer.setSingleShot(True)
        self._tree_timer.setInterval(150)
        self._tree_timer.timeout.connect(self._rebuild_overlay)

        self.figure = Figure(figsize=(7, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        for name, fn in (("button_press_event", self._on_press), ("motion_notify_event", self._on_motion),
                         ("button_release_event", self._on_release), ("resize_event", self._on_resize)):
            self.canvas.mpl_connect(name, fn)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(PLAY_MS)
        self.play_timer.timeout.connect(self._advance)
        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self.refresh_all)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.setSpacing(10)
        lv.addWidget(self._build_controls())
        self.toolbox = QToolBox()
        self.toolbox.addItem(self._page_recon(), "Recon obs")
        self.toolbox.addItem(self._page_fixes(), "Fixes && dropsondes")
        self.toolbox.addItem(self._page_forecast(), "Forecast")
        self.toolbox.addItem(self._page_lightning(), "Lightning")
        self.toolbox.addItem(self._page_live(), "Live && alerts")
        self.toolbox.addItem(self._page_tools(), "Tools")
        lv.addWidget(self.toolbox, 1)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(310)
        scroll.setMaximumWidth(470)

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
        split.setSizes([350, 1200])
        split.setChildrenCollapsible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split, 1)
        self.redraw()
        QTimer.singleShot(0, self.refresh_storms)

    # ---------------------------------------------------------------- building

    @staticmethod
    def _narrow(combo: QComboBox, chars: int = 16) -> QComboBox:
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(chars)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return combo

    @staticmethod
    def _hint(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("hint")
        lab.setWordWrap(True)
        return lab

    def _check(self, text, on=True, tip="", slot=None):
        chk = QCheckBox(text)
        chk.setChecked(on)
        if tip:
            chk.setToolTip(tip)
        chk.toggled.connect(slot or (lambda _: self._refresh_view()))
        return chk

    def _build_controls(self) -> QWidget:
        box = QGroupBox("Storm && imagery")
        form = QFormLayout(box)
        form.setContentsMargins(10, 10, 10, 10)
        form.setVerticalSpacing(7)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        row = QHBoxLayout()
        self.storm_box = self._narrow(QComboBox())
        self.storm_box.currentIndexChanged.connect(self._storm_chosen)
        row.addWidget(self.storm_box, 1)
        b = QPushButton("\u21bb")
        b.setToolTip("Refresh the active storm list: NHC, and JTWC's basins (West Pacific, Indian Ocean,\n"
                     "Southern Hemisphere) from knackwx's ATCF feed or NRL's sector file")
        b.setFixedWidth(34)
        b.clicked.connect(self.refresh_storms)
        row.addWidget(b)
        form.addRow("Storm", row)
        self.product_box = self._narrow(QComboBox())
        for band in goes.SINGLE_BANDS:
            self.product_box.addItem(goes.product_label(band), band)
        self.product_box.insertSeparator(self.product_box.count())
        for key in goes.PRODUCTS:
            self.product_box.addItem(goes.product_label(key), key)
            self.product_box.setItemData(self.product_box.count() - 1, goes.PRODUCTS[key]["note"],
                                         Qt.ItemDataRole.ToolTipRole)
        self.product_box.setCurrentIndex(self.product_box.findData(13))
        form.addRow("Imagery", self.product_box)
        self.band_box = self.product_box
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
        form.addRow("Half-width", self.width_spin)
        self.count_box = self._narrow(QComboBox(), 10)
        for label, value in FRAME_COUNTS:
            self.count_box.addItem(label, value)
        self.count_box.setCurrentIndex(1)
        form.addRow("Frames", self.count_box)
        base = QGridLayout()
        self.chk_sat = self._check("Imagery")
        self.chk_cbar = self._check("Colour bar")
        self.chk_fix = self._check("Storm position", tip="The latest NHC or JTWC position of the storm chosen.")
        self.chk_follow_track = self._check("Zoom to recon", tip="Frame the map on the recon obs.")
        for i, c in enumerate((self.chk_sat, self.chk_cbar, self.chk_fix, self.chk_follow_track)):
            base.addWidget(c, i // 2, i % 2)
        form.addRow(base)
        self.btn_fetch = QPushButton("Fetch imagery")
        self.btn_fetch.setObjectName("primary")
        self.btn_fetch.clicked.connect(self.fetch)
        form.addRow(self.btn_fetch)
        return box

    def _page_recon(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(6)
        self.btn_all = QPushButton("Load all recon for this storm")
        self.btn_all.setObjectName("primary")
        self.btn_all.setToolTip("NHC HDOBs, vortex messages and dropsondes in one go.")
        self.btn_all.clicked.connect(self.load_all_recon)
        v.addWidget(self.btn_all)
        self.chk_converted = QCheckBox("Converted HDOB (IWG1)")
        self.chk_converted.setEnabled(False)
        self.chk_converted.setChecked(True)
        self.chk_converted.toggled.connect(self._rebuild_overlay)
        v.addWidget(self.chk_converted)
        self.converted_note = self._hint("None yet \u2014 convert one under Reconnaissance.")
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
        self.btn_hdobs.clicked.connect(self.load_hdobs)
        row.addWidget(self.btn_hdobs)
        v.addLayout(row)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumHeight(140)
        self.tree.itemChanged.connect(lambda *_: self._tree_timer.start())
        v.addWidget(self.tree, 1)
        sel = QHBoxLayout()
        for label, state in (("All", Qt.CheckState.Checked), ("None", Qt.CheckState.Unchecked)):
            b = QPushButton(label)
            b.setObjectName("smallButton")
            b.clicked.connect(lambda _=False, st=state: self._check_all(st))
            sel.addWidget(b)
        b = QPushButton("Newest mission")
        b.setObjectName("smallButton")
        b.clicked.connect(self._check_newest)
        sel.addWidget(b)
        v.addLayout(sel)
        g = QGridLayout()
        self.chk_barbs = self._check("Wind barbs")
        self.chk_track = self._check("Track")
        self.chk_declutter = self._check("Declutter barbs", on=False,
                                         tip="Drop barbs that would overlap on screen.")
        self.chk_peaks = self._check("Peak wind labels", on=False,
                                     tip="The strongest flight-level wind on each leg of each pass.")
        self.chk_sync = self._check("Obs up to frame time",
                                    tip="Each frame shows only obs taken by its scan time. The\n"
                                        "newest frame always shows every ob.",
                                    slot=lambda _: self._show_frame(self.slider.value(), force=True))
        for i, c in enumerate((self.chk_barbs, self.chk_track, self.chk_declutter, self.chk_peaks)):
            g.addWidget(c, i // 2, i % 2)
        g.addWidget(self.chk_sync, 2, 0, 1, 2)
        v.addLayout(g)
        self.flag_box = QComboBox()
        for label, value in FLAG_MODES:
            self.flag_box.addItem(label, value)
        self.flag_box.setToolTip("HDOB quality flags mark some flight-level winds as questionable.")
        self.flag_box.currentIndexChanged.connect(lambda _: self._refresh_view())
        v.addWidget(self.flag_box)
        return w

    def _page_fixes(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(6)
        row = QHBoxLayout()
        self.btn_vdms = QPushButton("Load vortex messages")
        self.btn_vdms.setToolTip("NHC's official aircraft centre fixes (URNT12/URPN12).")
        self.btn_vdms.clicked.connect(self.load_vdms)
        row.addWidget(self.btn_vdms)
        self.btn_drops = QPushButton("Load dropsondes")
        self.btn_drops.clicked.connect(self.load_drops)
        row.addWidget(self.btn_drops)
        v.addLayout(row)
        g = QGridLayout()
        self.chk_fixes = self._check("Centre fixes", tip="Diamonds: vortex messages. Crosses: derived\n"
                                                         "from obs, for flights without them.")
        self.chk_derived = self._check("Derive from obs", slot=lambda _: self._rebuild_overlay())
        self.chk_motion = self._check("Motion vector")
        self.chk_drops = self._check("Dropsondes")
        for i, c in enumerate((self.chk_fixes, self.chk_derived, self.chk_motion, self.chk_drops)):
            g.addWidget(c, i // 2, i % 2)
        v.addLayout(g)
        self.fix_list = QListWidget()
        self.fix_list.setMinimumHeight(120)
        v.addWidget(self.fix_list, 1)
        self.drop_note = self._hint("Click a dropsonde dot for its sounding.")
        v.addWidget(self.drop_note)
        return w

    def _page_forecast(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.btn_fcst = QPushButton("Load forecast")
        self.btn_fcst.setToolTip("NHC's current cone and track, and the latest model cycle.")
        self.btn_fcst.clicked.connect(self.load_forecast)
        v.addWidget(self.btn_fcst)
        self.chk_cone = self._check("NHC cone")
        self.chk_official = self._check("NHC track")
        self.chk_models = self._check("Model tracks", on=False)
        for c in (self.chk_cone, self.chk_official, self.chk_models):
            v.addWidget(c)
        self.model_groups: Dict[str, QCheckBox] = {}
        grp = QGridLayout()
        for i, name in enumerate(g for g in forecast.MODEL_GROUPS if g != "Official"):
            c = self._check(name)
            c.setStyleSheet(f"color: {forecast.GROUP_COLOURS[name]};")
            self.model_groups[name] = c
            grp.addWidget(c, i // 2, i % 2)
        v.addLayout(grp)
        self.fcst_note = self._hint("")
        v.addWidget(self.fcst_note)
        v.addStretch(1)
        return w

    def _page_lightning(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.chk_glm = self._check("GLM lightning", on=False,
                                   slot=lambda _: self._show_frame(self.slider.value(), force=True))
        v.addWidget(self.chk_glm)
        row = QHBoxLayout()
        row.addWidget(QLabel("Flashes in the last"))
        self.glm_minutes = QSpinBox()
        self.glm_minutes.setRange(1, 30)
        self.glm_minutes.setValue(10)
        self.glm_minutes.setSuffix(" min")
        self.glm_minutes.valueChanged.connect(lambda _: self._show_frame(self.slider.value(), force=True))
        row.addWidget(self.glm_minutes)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(self._hint("Flashes before each frame's scan time, fetched for the frame on "
                               "screen (about 7 MB per 10 minutes) and cached. Eyewall lightning "
                               "bursts often come with rapid changes in intensity. GOES's lightning "
                               "mappers see the Americas and the Pacific to about 170E; Himawari "
                               "carries none."))
        self.glm_note = self._hint("")
        v.addWidget(self.glm_note)
        v.addStretch(1)
        return w

    def _page_live(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        self.chk_live = QCheckBox("Auto-update every")
        self.chk_live.toggled.connect(self._on_live_toggled)
        row.addWidget(self.chk_live)
        self.live_minutes = QSpinBox()
        self.live_minutes.setRange(1, 60)
        self.live_minutes.setValue(5)
        self.live_minutes.setSuffix(" min")
        self.live_minutes.valueChanged.connect(lambda _: self._on_live_toggled(self.chk_live.isChecked()))
        row.addWidget(self.live_minutes)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(self._hint("Adds new scans to the loop and reloads whatever recon, fixes, "
                               "drops and forecast have been loaded."))
        box = QGroupBox("Alerts")
        f = QFormLayout(box)
        self.alert_slp = QSpinBox()
        self.alert_slp.setRange(880, 1010)
        self.alert_slp.setValue(960)
        self.alert_slp.setSuffix(" mb")
        f.addRow("Central pressure \u2264", self.alert_slp)
        self.alert_fall = QSpinBox()
        self.alert_fall.setRange(1, 30)
        self.alert_fall.setValue(4)
        self.alert_fall.setSuffix(" mb")
        f.addRow("Fall between fixes \u2265", self.alert_fall)
        self.alert_wind = QSpinBox()
        self.alert_wind.setRange(34, 200)
        self.alert_wind.setValue(100)
        self.alert_wind.setSuffix(" kt")
        f.addRow("Flight-level wind \u2265", self.alert_wind)
        self.alert_drop = QCheckBox("Every new dropsonde")
        f.addRow(self.alert_drop)
        self.chk_alerts = QCheckBox("Alerts on")
        self.chk_alerts.setChecked(True)
        f.addRow(self.chk_alerts)
        v.addWidget(box)
        self.alert_log = QListWidget()
        v.addWidget(self.alert_log, 1)
        return w

    def _page_tools(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.btn_measure = QPushButton("Measure distance")
        self.btn_measure.setCheckable(True)
        self.btn_measure.setToolTip("Drag across the map for distance and bearing.")
        self.btn_measure.toggled.connect(self._on_measure_toggled)
        v.addWidget(self.btn_measure)
        self.measure_note = self._hint("")
        v.addWidget(self.measure_note)
        self.btn_export = QPushButton("Export loop\u2026")
        self.btn_export.setToolTip("Save the loaded loop as an animated GIF, or MP4 if ffmpeg is installed.")
        self.btn_export.clicked.connect(self.export_loop)
        v.addWidget(self.btn_export)
        self.export_note = self._hint("")
        v.addWidget(self.export_note)
        v.addStretch(1)
        return w

    def _build_loop_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.btn_prev = QPushButton("\u25c0")
        self.btn_play = QPushButton("\u25b6")
        self.btn_next = QPushButton("\u25b6\u25b6")
        for b in (self.btn_prev, self.btn_play, self.btn_next):
            b.setFixedWidth(42)
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(+1))
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._show_frame)
        self.frame_label = QLabel("\u2014")
        self.frame_label.setObjectName("probeReadout")
        self.frame_label.setMinimumWidth(170)
        for wdg in (self.btn_prev, self.btn_play, self.btn_next):
            bar.addWidget(wdg)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.frame_label)
        self._set_loop_enabled(False)
        return bar

    def _set_loop_enabled(self, on: bool):
        for w in (self.btn_prev, self.btn_play, self.btn_next, self.slider):
            w.setEnabled(on)

    # ------------------------------------------------------------------ storms

    def _storm_chosen(self, _index=None):
        """A storm picked (or the list changed under it): the other tabs follow it, recon or not."""
        self.storm_box.setToolTip(self.storm_box.currentText())
        self._update_bundle()

    def refresh_storms(self):
        keep = self.storm_box.currentData()
        self.storm_box.blockSignals(True)
        self.storm_box.clear()
        try:
            self._storms = nhc.all_storms()
        except Exception as exc:
            self._storms = []
            self.status.setText(f"Could not get the storm list: {exc}")
        if nhc.STORM_LIST_NOTES:
            self.status.setText("Storm list: " + "; ".join(nhc.STORM_LIST_NOTES))
        for fix in self._storms:
            self.storm_box.addItem(fix.label, fix)
        cf = self._case_fix
        if cf is not None and cf.storm_id and not any(f.storm_id == cf.storm_id for f in self._storms):
            self.storm_box.insertItem(0, f"{cf.label}  (saved case)", cf)
        self.storm_box.addItem("Use the recon track instead", None)
        if keep is not None:
            self.select_storm(keep.storm_id)
        self.storm_box.blockSignals(False)
        now = self.storm_box.currentData()
        if (getattr(now, "storm_id", None) != getattr(keep, "storm_id", None)) and hasattr(self, "_bundle"):
            self._storm_chosen()

    def current_fix(self) -> Optional[nhc.Fix]:
        fix = self.storm_box.currentData()
        if fix is not None:
            return fix
        if self._overlay is not None and len(self._overlay):
            return nhc.fix_from_positions(self._overlay.lats, self._overlay.lons)
        return None

    def select_storm(self, storm_id: str) -> bool:
        for i in range(self.storm_box.count()):
            d = self.storm_box.itemData(i)
            if d is not None and d.storm_id == storm_id:
                self.storm_box.setCurrentIndex(i)
                return True
        return False

    # -------------------------------------------------------------- recon data

    def set_overlay(self, overlay: Optional[satview.Overlay]):
        """The HDOB converted from IWG1 (Reconnaissance section)."""
        self._converted = overlay
        has = overlay is not None and len(overlay) > 0
        self.chk_converted.setEnabled(has)
        if has:
            self.converted_note.setText(f"{len(overlay)} obs, "
                                        f"{overlay.times[0]:%d %b %H:%M}\u2013{overlay.times[-1]:%H:%MZ}")
        self._rebuild_overlay()

    def _run(self, fn, done, button=None, *args):
        if button is not None:
            button.setEnabled(False)
        task = _Task(fn, *args)
        task.signals.status.connect(self.status.setText)

        def ok(result):
            if button is not None:
                button.setEnabled(True)
            done(result)

        def bad(msg):
            if button is not None:
                button.setEnabled(True)
            self.status.setText(msg)
        task.signals.done.connect(ok)
        task.signals.failed.connect(bad)
        self.pool.start(task)

    def _window(self, hours: Optional[float] = None):
        until = dt.datetime.now(dt.timezone.utc)
        return until - dt.timedelta(hours=hours or float(self.hdob_hours.value())), until

    def _replaying(self) -> bool:
        if self.replay_mode:
            self.status.setText("Replaying a saved case \u2014 exit the replay (bottom bar) to fetch live recon.")
        return self.replay_mode

    def load_all_recon(self):
        if self._replaying():
            return
        self.load_hdobs()
        self.load_vdms()
        self.load_drops()

    def _storm_or_warn(self):
        fix = self.current_fix()
        if fix is None:
            self.status.setText("Pick a storm first.")
        return fix

    def load_hdobs(self):
        if self._replaying():
            return
        fix = self._storm_or_warn()
        if fix is None:
            return
        since, until = self._window()

        def work(status, since, until, fix):
            return nhchdob.fetch_batches(since, until, storm_id=fix.storm_id,
                                         storm_name=fix.name if fix.storm_id else "",
                                         centre=(fix.lat, fix.lon), progress=status)
        self._run(work, self._on_hdobs, self.btn_hdobs, since, until, fix)

    def _on_hdobs(self, batches, rebuild: bool = True):
        first = not self._loaded["hdobs"]
        self._loaded["hdobs"] = True
        old = {(b.aircraft, b.mission, b.number) for b in self._batches}
        unticked = self._unticked_keys() | self._restore_unticked
        self._restore_unticked = set()
        new_batches = [b for b in batches if (b.aircraft, b.mission, b.number) not in old]
        self._batches = batches
        self.tree.blockSignals(True)
        self.tree.clear()
        for key, group in nhchdob.group_missions(batches).items():
            nums = [b.number for b in group]
            top = QTreeWidgetItem([f"{' '.join(key)} {group[0].storm}  \u00b7  HDOB "
                                   f"{min(nums):02d}\u2013{max(nums):02d}  \u00b7  "
                                   f"{group[0].start:%H:%M}\u2013{group[-1].end:%H:%MZ}"])
            top.setFlags(top.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
            for b in group:
                peak = f"max {b.max_wind} kt" if b.max_wind is not None else ""
                child = QTreeWidgetItem([f"HDOB {b.number:02d}  {b.start:%H:%M}\u2013{b.end:%H:%MZ}  {peak}"])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                k = (b.aircraft, b.mission, b.number)
                child.setCheckState(0, Qt.CheckState.Unchecked if k in unticked else Qt.CheckState.Checked)
                child.setData(0, Qt.ItemDataRole.UserRole, b)
                top.addChild(child)
            self.tree.addTopLevelItem(top)
        self.tree.blockSignals(False)
        self.status.setText(f"{len(batches)} NHC HDOB batch(es) in {self.tree.topLevelItemCount()} "
                            "mission(s)." if batches else "No NHC HDOBs for this storm in that window.")
        if not first:
            self._check_wind_alerts(new_batches)
        if rebuild:
            self._rebuild_overlay()

    def _unticked_keys(self) -> set:
        out = set()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                c = top.child(j)
                if c.checkState(0) != Qt.CheckState.Checked:
                    b = c.data(0, Qt.ItemDataRole.UserRole)
                    out.add((b.aircraft, b.mission, b.number))
        return out

    def _check_all(self, state):
        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                top.child(j).setCheckState(0, state)
        self.tree.blockSignals(False)
        self._rebuild_overlay()

    def _check_newest(self):
        n = self.tree.topLevelItemCount()
        if not n:
            return

        def end_of(i):
            top = self.tree.topLevelItem(i)
            return top.child(top.childCount() - 1).data(0, Qt.ItemDataRole.UserRole).end
        newest = max(range(n), key=end_of)
        self.tree.blockSignals(True)
        for i in range(n):
            st = Qt.CheckState.Checked if i == newest else Qt.CheckState.Unchecked
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                top.child(j).setCheckState(0, st)
        self.tree.blockSignals(False)
        self._rebuild_overlay()

    def load_vdms(self):
        if self._replaying():
            return
        fix = self._storm_or_warn()
        if fix is None:
            return
        since, until = self._window(max(24.0, float(self.hdob_hours.value())))

        def work(status, since, until, fix):
            return vdm.fetch_vdms(since, until, storm_id=fix.storm_id,
                                  storm_name=fix.name if fix.storm_id else "",
                                  centre=(fix.lat, fix.lon), progress=status)
        self._run(work, self._on_vdms, self.btn_vdms, since, until, fix)

    def _on_vdms(self, vdms, rebuild: bool = True):
        if self._loaded["vdms"]:
            self._check_pressure_alerts(vdms)
        self._loaded["vdms"] = True
        self._vdms = vdms
        self.status.setText(f"{len(vdms)} vortex message(s)." if vdms else "No vortex messages in that window.")
        if rebuild:
            self._rebuild_overlay()

    def load_drops(self):
        if self._replaying():
            return
        fix = self._storm_or_warn()
        if fix is None:
            return
        since, until = self._window(max(12.0, float(self.hdob_hours.value())))

        def work(status, since, until, fix):
            return drops_mod.fetch_drops(since, until, storm_id=fix.storm_id,
                                         storm_name=fix.name if fix.storm_id else "",
                                         centre=(fix.lat, fix.lon), progress=status)
        self._run(work, self._on_drops, self.btn_drops, since, until, fix)

    def _on_drops(self, drops, rebuild: bool = True):
        if self._loaded["drops"] and self.alert_drop.isChecked():
            known = {(d.aircraft, d.release_time) for d in self._drops}
            for d in drops:
                if (d.aircraft, d.release_time) not in known:
                    sfc = d.surface
                    extra = f", {sfc.p:.0f} mb, {sfc.wspd} kt" if sfc and sfc.wspd is not None else ""
                    self._raise_alert(f"drop{d.aircraft}{d.release_time}", "New dropsonde",
                                      f"#{d.number} {d.aircraft} {d.release_time:%H:%MZ} {d.location}{extra}")
        self._loaded["drops"] = True
        self._drops = drops
        if drops:
            span = f" {drops[0].release_time:%H:%M}\u2013{drops[-1].release_time:%H:%MZ}"
            self.drop_note.setText(f"{len(drops)} dropsonde(s){span}. Click a dot for its sounding.")
        if rebuild:
            self._rebuild_overlay()

    def load_forecast(self):
        if self._replaying():
            return
        fix = self.storm_box.currentData()
        if fix is None or not fix.storm_id:
            self.fcst_note.setText("Forecasts need an NHC storm selected.")
            return
        if nhc.agency(fix.storm_id) != "NHC":
            self.fcst_note.setText("Forecasts come from NHC's cone and a-decks, which cover only NHC's basins: "
                                   "none for JTWC's storms.")
            return

        def work(status, sid):
            status("Loading NHC forecast and model tracks\u2026")
            return forecast.official_forecast(sid), forecast.model_tracks(sid)
        self._run(work, self._on_forecast, self.btn_fcst, fix.storm_id)

    def _on_forecast(self, result):
        self._loaded["forecast"] = True
        self._forecast, self._models = result
        cyc = f", models from {self._models[0].cycle:%d/%HZ}" if self._models else ""
        adv = f"advisory {self._forecast.advisory}" if self._forecast else "no NHC forecast"
        self.fcst_note.setText(f"NHC {adv}{cyc} ({len(self._models)} tracks).")
        self._refresh_view()

    # ------------------------------------------------------------- saved cases

    def set_recon_data(self, batches, vdms, drops):
        """HDOBs, vortex messages and drops from somewhere other than a fetch (a saved case)."""
        self._on_hdobs(list(batches), rebuild=False)
        self._on_vdms(list(vdms), rebuild=False)
        self._on_drops(list(drops), rebuild=False)
        self._rebuild_overlay()

    def set_forecast(self, fc, models):
        self._loaded["forecast"] = fc is not None or bool(models)
        self._forecast, self._models = fc, list(models or [])
        self.fcst_note.setText(f"NHC advisory {fc.advisory} (saved case)" if fc is not None else "")
        self._refresh_view()

    def set_case_storm(self, fix):
        """Put a case's storm in the list (it may no longer be active) and choose it."""
        self._case_fix = fix
        if fix is None:
            return
        if fix.storm_id and self.select_storm(fix.storm_id):
            return
        self.storm_box.insertItem(0, f"{fix.label}  (saved case)", fix)
        self.storm_box.setCurrentIndex(0)

    def converted_overlay(self):
        return self._converted

    def imagery_refs(self) -> dict:
        """What imagery is loaded, by reference (for a saved case)."""
        frames = [f for f in self._frames if f is not None]
        return {"product": self.product_box.currentData(), "sector": self.sector_box.currentData(),
                "satellite": self.sat_box.currentData(), "half_width": self.width_spin.value(),
                "frames": int(self.count_box.currentData()),
                "scenes": [{"sat": f.scene.sat, "band": f.scene.band, "sector": f.scene.sector, "key": f.scene.key,
                            "start": f.scene.start.isoformat()} for f in frames]}

    def fetch_at(self, when):
        """The imagery loop ending at a past moment (a case replay)."""
        self._fetch_frames(until=when)

    def _rebuild_overlay(self):
        parts = []
        if self._converted is not None and self.chk_converted.isChecked():
            parts.append(self._converted)
        ticked = []
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            for j in range(top.childCount()):
                c = top.child(j)
                if c.checkState(0) == Qt.CheckState.Checked:
                    ticked.append(c.data(0, Qt.ItemDataRole.UserRole))
        if ticked:
            parts.append(satview.overlay_from_batches(ticked))
        self._overlay = satview.merge_overlays(parts)
        self._update_bundle()
        self._refresh_view()

    def _update_bundle(self):
        b = ReconBundle()
        b.fix = self.current_fix()
        b.overlay, b.batches, b.drops, b.vdms = self._overlay, self._batches, self._drops, self._vdms
        vf = analysis.fixes_from_vdms(self._vdms)
        derived = []
        if self.chk_derived.isChecked() and self._overlay is not None:
            fallback = (b.fix.lat, b.fix.lon) if b.fix else None
            derived = analysis.derive_fixes(self._overlay, lambda t: analysis.centre_at(vf, t, fallback))
        b.fixes = analysis.combine_fixes(vf, derived)
        b.passes = analysis.passes(self._overlay, b.fixes)
        b.tilts = analysis.tilts(b.fixes, self._drops)
        b.motion = analysis.motion(b.fixes)
        self._bundle = b
        self.fix_list.clear()
        for f in b.fixes:
            self.fix_list.addItem(f.label)
        if b.motion:
            self.fix_list.addItem(f"Motion {b.motion[0]:03.0f}\u00b0 at {b.motion[1]:.0f} kt")
        self.recon_changed.emit(b)

    def bundle(self) -> ReconBundle:
        return self._bundle

    def _extras(self, until=None) -> satview.Extras:
        b = self._bundle
        groups = {g for g, c in self.model_groups.items() if c.isChecked()}
        return satview.Extras(
            fixes=b.fixes, show_fixes=self.chk_fixes.isChecked(),
            motion=b.motion if self.chk_motion.isChecked() else None,
            peaks=analysis.peak_labels(b.passes), show_peaks=self.chk_peaks.isChecked(),
            flagged=self.flag_box.currentData(), forecast=self._forecast,
            show_cone=self.chk_cone.isChecked(), show_official=self.chk_official.isChecked(),
            models=[m for m in self._models if m.group in groups],
            show_models=self.chk_models.isChecked(), tdr=self._tdr_layer,
            frame_time=self._current_frame_time())

    def _current_frame_time(self):
        j = self._loaded_index(self.slider.value()) if self._frames else None
        return self._frames[j].start if j is not None else None

    def set_tdr_layer(self, layer):
        """An airborne TDR plan view from Radar › Airborne TDR (or None to remove it)."""
        self._tdr_layer = layer
        self._refresh_view()

    # ------------------------------------------------------------------ alerts

    def _raise_alert(self, key: str, title: str, message: str):
        if key in self._alerted or not self.chk_alerts.isChecked():
            return
        self._alerted.add(key)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%H:%MZ")
        self.alert_log.insertItem(0, f"{stamp}  {title}: {message}")
        self.alert.emit(title, message)

    def _check_pressure_alerts(self, vdms):
        known = {(v.aircraft, v.time) for v in self._vdms}
        allv = sorted(vdms, key=lambda v: v.time)
        for v in allv:
            if (v.aircraft, v.time) in known or not v.min_slp:
                continue
            if v.min_slp <= self.alert_slp.value():
                self._raise_alert(f"slp{v.time}", "Low central pressure",
                                  f"{v.aircraft} fix {v.time:%H:%MZ}: {v.min_slp:.0f} mb")
            prev = [p for p in allv if p.time < v.time and p.min_slp]
            if prev and prev[-1].min_slp - v.min_slp >= self.alert_fall.value():
                self._raise_alert(f"fall{v.time}", "Pressure falling",
                                  f"{prev[-1].min_slp:.0f} \u2192 {v.min_slp:.0f} mb between "
                                  f"{prev[-1].time:%H:%M} and {v.time:%H:%MZ}")

    def _check_wind_alerts(self, new_batches):
        for b in new_batches:
            if b.max_wind is not None and b.max_wind >= self.alert_wind.value():
                self._raise_alert(f"wind{b.aircraft}{b.number}", "Strong flight-level wind",
                                  f"{b.aircraft} HDOB {b.number}: {b.max_wind} kt "
                                  f"({b.start:%H:%M}\u2013{b.end:%H:%MZ})")

    # -------------------------------------------------------------------- live

    def _on_live_toggled(self, on: bool):
        if on:
            self.live_timer.start(int(self.live_minutes.value()) * 60 * 1000)
        else:
            self.live_timer.stop()

    def refresh_all(self):
        """Auto-update: newer scans onto the loop, and whatever recon is loaded, again."""
        if self.replay_mode:
            return
        loaded = [f for f in self._frames if f is not None]
        if loaded and not self._busy:
            self._fetch_frames(only_after=loaded[-1].start)
        for kind, fn in (("hdobs", self.load_hdobs), ("vdms", self.load_vdms),
                         ("drops", self.load_drops), ("forecast", self.load_forecast)):
            if self._loaded[kind]:
                fn()

    # ----------------------------------------------------------------- imagery

    def fetch(self):
        self._fetch_frames()
        if self.chk_drops.isChecked() and not self._loaded["drops"]:
            self.load_drops()

    def _fetch_frames(self, only_after=None, until=None):
        if self._busy:
            return
        fix = self.current_fix()
        if fix is None:
            self.status.setText("Nothing to centre on yet \u2014 convert an HDOB or pick a storm.")
            return
        self._fix = fix
        if only_after is None:
            self.stop()
        self._busy = True
        self.btn_fetch.setEnabled(False)
        if until is not None and self._bundle.fixes:
            import recon10s_windfield as wf
            la, lo = wf.centre_track(self._bundle.fixes, [until], self._bundle.motion)
            if la[0] == la[0]:                          # centre the loop on the storm at that time
                fix = nhc.Fix(storm_id=fix.storm_id, name=fix.name, lat=float(la[0]), lon=float(lo[0]),
                              time=until, classification=fix.classification, source="recon fixes")
                self._fix = fix
        worker = _LoopWorker(fix.lat, fix.lon, self.product_box.currentData(),
                             self.sector_box.currentData(), self.sat_box.currentData(),
                             float(self.width_spin.value()), int(self.count_box.currentData()),
                             only_after=only_after, until=until)
        worker.signals.status.connect(self.status.setText)
        worker.signals.started.connect(lambda sc, why: self._on_started(sc, why, append=only_after is not None))
        worker.signals.frame.connect(self._on_frame)
        worker.signals.done.connect(self._on_done)
        worker.signals.failed.connect(self._on_failed)
        self.pool.start(worker)

    def _on_started(self, scenes, why, append=False):
        self._why = why
        if append:
            # Newer scans join the end; the oldest drop off to keep the loop's length.
            at_end = self.slider.value() >= self.slider.maximum()
            self._append_base = len(self._frames)
            self._frames += [None] * len(scenes)
            extra = len(self._frames) - max(int(self.count_box.currentData()), 1)
            if extra > 0 and scenes:
                self._frames = self._frames[extra:]
                self._append_base -= extra
            self.slider.blockSignals(True)
            self.slider.setRange(0, max(0, len(self._frames) - 1))
            if at_end:
                self.slider.setValue(len(self._frames) - 1)
            self.slider.blockSignals(False)
            if not scenes:
                self.status.setText("No newer scans yet.")
            return
        self._append_base = 0
        self._frames = [None] * len(scenes)
        self._shown = -1
        self._art = None
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(scenes) - 1))
        self.slider.setValue(len(scenes) - 1)
        self.slider.blockSignals(False)
        self.status.setText(f"Loading {len(scenes)} frame(s) \u2014 {why}\u2026")

    def _on_frame(self, i, img):
        i += self._append_base
        if not (0 <= i < len(self._frames)):
            return
        self._frames[i] = img
        loaded = sum(1 for f in self._frames if f is not None)
        self.status.setText(f"{loaded} of {len(self._frames)} frames \u2014 {self._why}")
        self._set_loop_enabled(loaded > 1 or len(self._frames) == 1)
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
        note = f", {dropped} dropped" if dropped else ""
        self.status.setText(f"{len(loaded)} frames {loaded[0].start:%H:%M}\u2013{loaded[-1].start:%H:%MZ}"
                            f"{note}  \u00b7  {self._why}  \u00b7  {mb:.0f} MB pulled")
        self._set_loop_enabled(len(loaded) > 1)
        self._show_frame(self.slider.value(), force=True)

    def _on_failed(self, message: str):
        self._busy = False
        self.btn_fetch.setEnabled(True)
        self.status.setText(message)

    # --------------------------------------------------------------- lightning

    def _glm_key(self, img):
        """(GOES whose GLM covers the storm, frame time, minutes), or None: Himawari has no lightning
        mapper, so its frames get GOES-18's flashes where GLM reaches (east of about 170E)."""
        sat = img.scene.sat
        if sat not in goes.GOES_SATS:
            fix = self._fix or self.current_fix()
            sat = glm.glm_satellite(fix.lat, fix.lon) if fix is not None else None
            if sat is None:
                return None
        return (sat, img.start, self.glm_minutes.value())

    def _glm_for(self, img):
        """Flashes for a frame from cache; otherwise start fetching them and return None."""
        if not self.chk_glm.isChecked() or img is None:
            return None
        key = self._glm_key(img)
        if key is None:
            self.glm_note.setText("No GLM lightning this far west: GOES's lightning mappers reach to about "
                                  "170E, and Himawari carries none.")
            return None
        if key in self._glm:
            return self._glm[key]
        if key not in self._glm_pending:
            self._glm_pending.add(key)
            fix = self._fix or self.current_fix()
            h = float(self.width_spin.value()) + 1
            bbox = (fix.lat - h, fix.lat + h, fix.lon - h, fix.lon + h) if fix else None

            def work(status, key, bbox):
                la, lo, n = glm.flashes(key[0], key[1], key[2], bbox)
                return key, la, lo, n

            def done(res):
                k, la, lo, n = res
                self._glm_pending.discard(k)
                self._glm[k] = (la, lo, k[2])
                self.glm_note.setText(f"{len(la)} flashes in the {k[2]} min before {k[1]:%H:%MZ} ({n} files).")
                j = self._loaded_index(self.slider.value())
                if j is not None and self._frames[j] is not None and self._frames[j].start == k[1]:
                    satview.draw_glm(self._art, self._glm[k], True)
                    self.canvas.draw_idle()
            self._run(work, done, None, key, bbox)
        return None

    # ------------------------------------------------------------------- drops

    def _centre_for(self, drop):
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
        if drop.release_time and self._bundle.fixes:
            return analysis.centre_at(self._bundle.fixes, drop.release_time)
        fix = self._fix or self.current_fix()
        return (fix.lat, fix.lon) if fix else None

    def _sat_for(self, drop):
        if not drop.release_time:
            return None
        key = (drop.aircraft, drop.release_time)
        if key in self._inset_cache:
            return self._inset_cache[key]
        near = [f for f in self._frames if f is not None and getattr(f, "kind", "") == "brightness"
                and abs((f.start - drop.release_time).total_seconds()) <= 900]
        return min(near, key=lambda f: abs((f.start - drop.release_time).total_seconds())) if near else None

    def _open_drop(self, drop):
        idx = self._drops.index(drop) if drop in self._drops else 0
        win = DropsondeWindow(self._drops, idx, self._sat_for, self._centre_for, theme=self.theme)
        win.on_show = lambda d, w=win: self._fetch_inset(d, w)
        self._drop_windows.append(win)
        win.destroyed.connect(lambda *_: self._drop_windows.remove(win) if win in self._drop_windows else None)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        win.show()
        self._fetch_inset(drop, win)

    def _fetch_inset(self, drop, win):
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
                pass
        self._run(work, done, None, drop)

    # ------------------------------------------------------------------- mouse

    def _geo(self, event):
        if self._art is None or event.inaxes is not self._art.ax or event.xdata is None:
            return None
        import cartopy.crs as ccrs
        lon, lat = ccrs.PlateCarree().transform_point(event.xdata, event.ydata, self._art.ax.projection)
        return lat, lon

    def _on_measure_toggled(self, on: bool):
        self._measure = None
        self._clear_measure()
        self.measure_note.setText("Drag across the map." if on else "")
        self.canvas.draw_idle()

    def _clear_measure(self):
        for a in self._measure_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._measure_artists = []

    def _on_press(self, event):
        if event.button != 1 or getattr(self.toolbar, "mode", ""):
            return
        if self.btn_measure.isChecked():
            self._measure = self._geo(event)
            return
        if self._art is None or event.inaxes is not self._art.ax:
            return
        near = satview.drops_near(self._art, event.x, event.y)
        if not near:
            return
        if len(near) == 1:
            self._open_drop(near[0])
            return
        menu = QMenu(self)
        for d in near:
            when = f"{d.release_time:%H:%MZ}" if d.release_time else ""
            wind = f"{d.surface.wspd} kt sfc" if d.surface and d.surface.wspd is not None else ""
            act = QAction(f"#{d.number}  {when}  {d.location or ''}  {wind}", menu)
            act.triggered.connect(lambda _=False, d=d: self._open_drop(d))
            menu.addAction(act)
        menu.exec(QCursor.pos())

    def measure(self, start, end) -> str:
        """Draw a distance/bearing line between two (lat, lon) points; returns the label."""
        import cartopy.crs as ccrs
        (la1, lo1), (la2, lo2) = start, end
        km = float(analysis.gc_km(la1, lo1, la2, lo2))
        brg = analysis.bearing(la1, lo1, la2, lo2)
        self._clear_measure()
        self._measure_artists += self._art.ax.plot([lo1, lo2], [la1, la2], color="#ffffff", lw=1.6,
                                                   transform=ccrs.Geodetic(), zorder=20)
        text = f"{km:.0f} km  {km / 1.852:.0f} nm  {brg:03.0f}\u00b0"
        self._measure_artists.append(self._art.ax.text(
            lo2, la2, "  " + text, transform=ccrs.PlateCarree(), color="#ffffff", fontsize=9, zorder=21,
            bbox=dict(fc="#0b0d11cc", ec="#5c6675", lw=0.8)))
        self.measure_note.setText(text)
        self.canvas.draw_idle()
        return text

    def _on_motion(self, event):
        if self.btn_measure.isChecked() and self._measure:
            end = self._geo(event)
            if end is not None:
                self.measure(self._measure, end)

    def _on_release(self, event):
        self._measure = None

    def _on_resize(self, _event):
        satview.fit_layout(self.figure, self._art)
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- playback

    def _loaded_index(self, i: int) -> Optional[int]:
        for j in range(min(i, len(self._frames) - 1), -1, -1):
            if self._frames[j] is not None:
                return j
        for j in range(max(i, 0), len(self._frames)):
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
            nxt = (nxt + 1) % n
        self.slider.setValue(nxt)

    def _update_frame_label(self):
        n = len(self._frames)
        if not n:
            self.frame_label.setText("\u2014")
            return
        j = self._loaded_index(self.slider.value())
        when = self._frames[j].start.strftime("%H:%M:%SZ") if j is not None else "loading"
        self.frame_label.setText(f"{self.slider.value() + 1:>2}/{n}  {when}")

    # ----------------------------------------------------------------- drawing

    def set_theme(self, theme: str):
        self.theme = theme
        self.redraw()
        for win in list(self._drop_windows):
            win.theme = theme
            win.show_index(win.index)

    def _extent(self):
        if self._overlay is not None and len(self._overlay) and self.chk_follow_track.isChecked():
            s, n, w, e = self._overlay.bbox
            return (w, e, s, n)
        return None

    def _obs_until(self, j: Optional[int]):
        if j is None or not self.chk_sync.isChecked():
            return None
        if j == self._loaded_index(len(self._frames) - 1):
            return None
        return self._frames[j].start

    def _spacing(self) -> float:
        return 22.0 if self.chk_declutter.isChecked() else 0.0

    def _refresh_view(self):
        if self._frames:
            self._show_frame(self.slider.value(), force=True)
        else:
            self.redraw()

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
        until = self._obs_until(j)
        swapped = False
        if self._art and self.chk_sat.isChecked() and not self._measure_artists:
            swapped = satview.update_frame(
                self._art, img, self._overlay, until, self.chk_barbs.isChecked(),
                self.chk_track.isChecked(), self._spacing(), self.theme,
                satview.scene_title(img, self._fix, self._why), drops=self._drops,
                show_drops=self.chk_drops.isChecked(), extras=self._extras(until))
        self._shown = j
        if swapped:
            satview.draw_glm(self._art, self._glm_for(img), self.chk_glm.isChecked())
            self.canvas.draw_idle()
        else:
            self.redraw()

    def redraw(self):
        j = self._loaded_index(self.slider.value()) if self._frames else None
        img = self._frames[j] if j is not None else None
        until = self._obs_until(j)
        self._clear_measure()
        try:
            self._art = satview.draw_scene(
                self.figure, img=img, overlay=self._overlay, fix=self._fix or self.current_fix(),
                show_satellite=self.chk_sat.isChecked(), show_track=self.chk_track.isChecked(),
                show_barbs=self.chk_barbs.isChecked(), show_colorbar=self.chk_cbar.isChecked(),
                show_fix=self.chk_fix.isChecked(), extent=self._extent(), theme=self.theme,
                title=satview.scene_title(img, self._fix, self._why), obs_until=until,
                spacing_px=self._spacing(), drops=self._drops, show_drops=self.chk_drops.isChecked(),
                extras=self._extras(until))
            satview.draw_glm(self._art, self._glm_for(img), self.chk_glm.isChecked())
        except Exception as exc:
            self.status.setText(f"Could not draw the map: {exc}")
            return
        self._shown = j if j is not None else -1
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ export

    def export_loop(self):
        frames = [f for f in self._frames if f is not None]
        if not frames:
            self.export_note.setText("Fetch a loop first.")
            return
        import matplotlib.animation as anim
        filt = "Animated GIF (*.gif)" + (";;MP4 video (*.mp4)" if anim.writers.is_available("ffmpeg") else "")
        path, _ = QFileDialog.getSaveFileName(self, "Export loop", "recon10s_loop.gif", filt)
        if not path:
            return
        self.export_note.setText(f"Rendering {len(frames)} frames\u2026")
        self.repaint()
        try:
            n = self.render_loop(path, frames)
            self.export_note.setText(f"Saved {n} frames to {path}")
        except Exception as exc:
            self.export_note.setText(f"Export failed: {exc}")

    def render_loop(self, path: str, frames=None) -> int:
        """Draw every frame off-screen with the current layers, then write a GIF or MP4."""
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from PIL import Image
        frames = frames or [f for f in self._frames if f is not None]
        w, h = self.figure.get_size_inches()
        fig = Figure(figsize=(max(w, 6), max(h, 5)), dpi=100)
        FigureCanvasAgg(fig)
        images, art = [], None
        for k, img in enumerate(frames):
            until = None if k == len(frames) - 1 or not self.chk_sync.isChecked() else img.start
            title = satview.scene_title(img, self._fix, self._why)
            ok = art is not None and satview.update_frame(
                art, img, self._overlay, until, self.chk_barbs.isChecked(), self.chk_track.isChecked(),
                self._spacing(), self.theme, title, drops=self._drops,
                show_drops=self.chk_drops.isChecked(), extras=self._extras(until))
            if not ok:
                art = satview.draw_scene(
                    fig, img=img, overlay=self._overlay, fix=self._fix, show_satellite=True,
                    show_track=self.chk_track.isChecked(), show_barbs=self.chk_barbs.isChecked(),
                    show_colorbar=self.chk_cbar.isChecked(), show_fix=self.chk_fix.isChecked(),
                    extent=self._extent(), theme=self.theme, title=title, obs_until=until,
                    spacing_px=self._spacing(), drops=self._drops,
                    show_drops=self.chk_drops.isChecked(), extras=self._extras(until))
            key = self._glm_key(img)
            satview.draw_glm(art, self._glm.get(key) if key else None, self.chk_glm.isChecked())
            fig.canvas.draw()
            images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if path.lower().endswith(".mp4"):
            import matplotlib.animation as anim
            vf = Figure(figsize=fig.get_size_inches(), dpi=100)
            FigureCanvasAgg(vf)
            ax = vf.add_axes([0, 0, 1, 1])
            ax.axis("off")
            shown = ax.imshow(images[0])
            writer = anim.FFMpegWriter(fps=1000 / PLAY_MS)
            with writer.saving(vf, path, dpi=100):
                for im in images + [images[-1]] * LAST_FRAME_DWELL:
                    shown.set_data(im)
                    writer.grab_frame()
        else:
            pil = [Image.fromarray(im) for im in images]
            durations = [PLAY_MS] * (len(pil) - 1) + [PLAY_MS * LAST_FRAME_DWELL]
            pil[0].save(path, save_all=True, append_images=pil[1:], duration=durations, loop=0)
        return len(images)

    # ----------------------------------------------------------------- session

    _CHECKS = ("chk_sat", "chk_cbar", "chk_fix", "chk_follow_track", "chk_barbs", "chk_track",
               "chk_declutter", "chk_peaks", "chk_sync", "chk_fixes", "chk_derived", "chk_motion",
               "chk_drops", "chk_cone", "chk_official", "chk_models", "chk_glm")

    def state(self) -> dict:
        fix = self.storm_box.currentData()
        return {
            "storm_id": fix.storm_id if fix else "",
            "product": self.product_box.currentData(),
            "sector": self.sector_box.currentIndex(), "satellite": self.sat_box.currentIndex(),
            "half_width": self.width_spin.value(), "frames": self.count_box.currentIndex(),
            "hdob_hours": self.hdob_hours.value(), "flagged": self.flag_box.currentIndex(),
            "glm_minutes": self.glm_minutes.value(), "live_minutes": self.live_minutes.value(),
            "alerts": [self.alert_slp.value(), self.alert_fall.value(), self.alert_wind.value(),
                       self.alert_drop.isChecked(), self.chk_alerts.isChecked()],
            "checks": {n: getattr(self, n).isChecked() for n in self._CHECKS},
            "models": {g: c.isChecked() for g, c in self.model_groups.items()},
            "loaded": dict(self._loaded), "had_frames": bool(self._frames),
            "unticked": [list(k) for k in self._unticked_keys()],
            "toolbox": self.toolbox.currentIndex(),
        }

    def restore(self, st: dict, reload_data: bool = False):
        """Put the controls back; with reload_data, fetch again what the session had loaded."""
        if not st:
            return
        i = self.product_box.findData(st.get("product", 13))
        if i >= 0:
            self.product_box.setCurrentIndex(i)
        for name, box in (("sector", self.sector_box), ("satellite", self.sat_box),
                          ("frames", self.count_box), ("flagged", self.flag_box)):
            if 0 <= int(st.get(name, -1)) < box.count():
                box.setCurrentIndex(int(st[name]))
        self.width_spin.setValue(float(st.get("half_width", 4.0)))
        self.hdob_hours.setValue(int(st.get("hdob_hours", 12)))
        self.glm_minutes.setValue(int(st.get("glm_minutes", 10)))
        self.live_minutes.setValue(int(st.get("live_minutes", 5)))
        a = st.get("alerts") or []
        if len(a) == 5:
            self.alert_slp.setValue(a[0])
            self.alert_fall.setValue(a[1])
            self.alert_wind.setValue(a[2])
            self.alert_drop.setChecked(a[3])
            self.chk_alerts.setChecked(a[4])
        for n, on in (st.get("checks") or {}).items():
            if hasattr(self, n):
                w = getattr(self, n)
                w.blockSignals(True)
                w.setChecked(bool(on))
                w.blockSignals(False)
        for g, on in (st.get("models") or {}).items():
            if g in self.model_groups:
                self.model_groups[g].setChecked(bool(on))
        self.toolbox.setCurrentIndex(int(st.get("toolbox", 0)))
        self._restore_unticked = {tuple(k) for k in st.get("unticked", [])}
        if st.get("storm_id"):
            self.select_storm(st["storm_id"])
        if reload_data:
            loaded = st.get("loaded") or {}
            for kind, fn in (("hdobs", self.load_hdobs), ("vdms", self.load_vdms),
                             ("drops", self.load_drops), ("forecast", self.load_forecast)):
                if loaded.get(kind):
                    fn()
            if st.get("had_frames"):
                self._fetch_frames()
        self._refresh_view()
