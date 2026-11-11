#!/usr/bin/env python3
"""
recon10s_tdrpanel — the TDR tab (Imagery section).

NOAA P-3 tail Doppler radar analyses from HRD's real-time page. The swath
(reflectivity and wind speed side by side at one height, 1-18 km) and the
azimuthal mean (radius-height section of reflectivity, tangential or radial
wind) are shown together; plan views, zoom, vortex tilt and TDR centre fixes
are on a second tab. Images are fetched in the background and cached, with
neighbouring heights prefetched so the height slider moves smoothly.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

import recon10s_tdr as tdr

OTHER = [("Plan view: wind + reflectivity", "composites", "ws_dbz_planview"),
         ("Plan view: NHC style", "composites", "ws_nhc_planview"),
         ("Zoomed plan view", "zoom", "planview_zoom"),
         ("Vortex tilt", "tilt", "vort_tilt"),
         ("TDR centre fixes", "centers", "tdrcenters")]


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


class ImageView(QLabel):
    """An image scaled to fit, keeping its aspect; double-click for full size."""

    def __init__(self, placeholder: str):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setObjectName("imageView")
        self._pix: Optional[QPixmap] = None
        self.path = ""
        self.caption = ""

    def set_image(self, path: str, caption: str = ""):
        self.path, self.caption = path, caption
        self._pix = QPixmap(path)
        self.setToolTip(f"{caption}\nDouble-click for full size.")
        self._rescale()

    def set_message(self, text: str):
        self._pix = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def _rescale(self):
        if self._pix is not None and not self._pix.isNull():
            self.setPixmap(self._pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        self._rescale()
        super().resizeEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self._pix is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(self.caption or "TDR")
        lab = QLabel()
        lab.setPixmap(self._pix)
        sc = QScrollArea()
        sc.setWidget(lab)
        v = QVBoxLayout(dlg)
        v.addWidget(sc)
        dlg.resize(min(self._pix.width() + 40, 1600), min(self._pix.height() + 40, 1000))
        dlg.show()


class TdrPanel(QWidget):
    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self.missions: List[tdr.TdrMission] = []
        self.mission: Optional[tdr.TdrMission] = None
        self._paths: Dict[str, str] = {}
        self._pending: set = set()
        self._want: Dict[str, str] = {}      # view name -> url it should show
        self.storm_name = ""

        # ---- left column
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
        self.list.currentRowChanged.connect(self._on_mission)
        f.addRow(self.list)

        view = QGroupBox("Analysis")
        g = QFormLayout(view)
        self.time_combo = QComboBox()
        self.time_combo.currentIndexChanged.connect(lambda _: self._on_time())
        g.addRow("Swath time", self.time_combo)
        lv = QHBoxLayout()
        self.level = QSlider(Qt.Orientation.Horizontal)
        self.level.setRange(0, 0)
        self.level.valueChanged.connect(lambda _: self._show_swath())
        self.level_label = QLabel("\u2014")
        self.level_label.setMinimumWidth(52)
        lv.addWidget(self.level, 1)
        lv.addWidget(self.level_label)
        g.addRow("Height", lv)
        self.window_combo = QComboBox()
        self.window_combo.currentIndexChanged.connect(lambda _: (self._show_rz(), self._show_other()))
        g.addRow("Mean window", self.window_combo)
        rz = QHBoxLayout()
        self.var_group = QButtonGroup(self)
        short = {"dbz": "dBZ", "vt": "Vt", "ur": "Vr", "w": "w"}
        for k, (var, name) in enumerate(tdr.RZ_VARS.items()):
            rb = QRadioButton(short.get(var, var))
            rb.setToolTip(name)
            rb.setProperty("var", var)
            self.var_group.addButton(rb, k)
            rz.addWidget(rb)
            if var == "vt":
                rb.setChecked(True)
        self.var_group.buttonToggled.connect(lambda *_: self._show_rz())
        g.addRow("Mean of", rz)
        self.other_combo = QComboBox()
        for label, _d, _s in OTHER:
            self.other_combo.addItem(label)
        self.other_combo.currentIndexChanged.connect(lambda _: self._show_other())
        g.addRow("Other product", self.other_combo)
        self.note = QLabel("HRD's real-time TDR output is graphics, so these are the analyses as "
                           "HRD draws them. Double-click any image for full size.")
        self.note.setObjectName("hint")
        self.note.setWordWrap(True)

        left = QWidget()
        lvv = QVBoxLayout(left)
        lvv.setContentsMargins(0, 0, 6, 0)
        lvv.addWidget(box, 1)
        lvv.addWidget(view)
        lvv.addWidget(self.note)
        left.setMinimumWidth(300)
        left.setMaximumWidth(440)

        # ---- right: swath above azimuthal mean; other products on a second tab
        self.swath_view = ImageView("Reflectivity and wind swath")
        self.rz_view = ImageView("Azimuthal mean (radius\u2013height)")
        self.other_view = ImageView("Plan views, tilt and centres")
        pair = QSplitter(Qt.Orientation.Vertical)
        pair.addWidget(self.swath_view)
        pair.addWidget(self.rz_view)
        pair.setSizes([480, 520])
        self.tabs = QTabWidget()
        self.tabs.addTab(pair, "Swath + azimuthal mean")
        self.tabs.addTab(self.other_view, "Other products")
        self.status = QLabel("Find missions to begin.")
        self.status.setObjectName("hint")
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.tabs, 1)
        rv.addWidget(self.status)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([340, 1200])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split, 1)

    # ------------------------------------------------------------ missions

    def set_storm_name(self, name: str):
        self.storm_name = (name or "").upper()
        self.storm_combo.setItemText(0, f"{self.storm_name.title()} (from Satellite)"
                                     if self.storm_name else "Storm chosen under Satellite")

    def find(self):
        name = "" if self.storm_combo.currentData() == "*" else self.storm_name
        self.btn_find.setEnabled(False)
        self.status.setText("Reading HRD's real-time TDR folders\u2026")
        job = _Job(lambda status, n, d: tdr.find_missions(n, d, progress=status), name, self.days.value())
        job.sig.status.connect(self.status.setText)
        job.sig.done.connect(self._on_found)
        job.sig.failed.connect(lambda m: (self.btn_find.setEnabled(True), self.status.setText(m)))
        self.pool.start(job)

    def _on_found(self, missions):
        self.btn_find.setEnabled(True)
        self.missions = missions
        self.list.clear()
        for m in missions:
            self.list.addItem(QListWidgetItem(m.label))
        self.status.setText(f"{len(missions)} mission(s) with TDR analyses." if missions else
                            "No TDR missions in that window" + (f" for {self.storm_name}." if self.storm_name else "."))
        if missions:
            self.list.setCurrentRow(0)

    def _on_mission(self, row: int):
        if not (0 <= row < len(self.missions)):
            return
        m = self.mission = self.missions[row]
        for combo, items in ((self.time_combo, m.swath_times()), (self.window_combo, m.windows("rzmeans")
                                                                   or m.windows("composites"))):
            combo.blockSignals(True)
            combo.clear()
            for t in items:
                combo.addItem(t.replace("_", "\u2013") + "Z", t)
            combo.setCurrentIndex(combo.count() - 1)
            combo.blockSignals(False)
        self._on_time()
        self._show_rz()
        self._show_other()

    def _on_time(self):
        m = self.mission
        t = self.time_combo.currentData()
        levels = m.swath_levels(t) if (m and t) else []
        self._levels = levels
        self.level.blockSignals(True)
        self.level.setRange(0, max(0, len(levels) - 1))
        if levels:
            self.level.setValue(min(range(len(levels)), key=lambda i: abs(levels[i] - 2.0)))
        self.level.blockSignals(False)
        self._show_swath()

    # --------------------------------------------------------------- views

    def _load(self, view: ImageView, name: str, url: str, caption: str):
        self._want[name] = url
        if url in self._paths:
            view.set_image(self._paths[url], caption)
            return
        view.set_message("Loading\u2026")
        if url in self._pending:
            return
        self._pending.add(url)
        job = _Job(lambda _s, u: (u, tdr.fetch_image(u)), url)

        def done(res):
            u, path = res
            self._pending.discard(u)
            self._paths[u] = path
            for nm, v in (("swath", self.swath_view), ("rz", self.rz_view), ("other", self.other_view)):
                if self._want.get(nm) == u:
                    v.set_image(path, v.caption or caption)
        job.sig.done.connect(done)
        job.sig.failed.connect(lambda msg, u=url: (self._pending.discard(u), view.set_message(msg)))
        self.pool.start(job)

    def _prefetch(self, urls):
        for u in urls:
            if u not in self._paths and u not in self._pending:
                self._pending.add(u)
                job = _Job(lambda _s, u: (u, tdr.fetch_image(u)), u)
                job.sig.done.connect(lambda res: (self._pending.discard(res[0]),
                                                  self._paths.__setitem__(res[0], res[1])))
                job.sig.failed.connect(lambda _m, u=u: self._pending.discard(u))
                self.pool.start(job)

    def _show_swath(self):
        m, t = self.mission, self.time_combo.currentData()
        levels = getattr(self, "_levels", [])
        if not (m and t and levels):
            self.swath_view.set_message("No swaths for this mission.")
            self.level_label.setText("\u2014")
            return
        i = self.level.value()
        lvl = levels[i]
        self.level_label.setText(f"{lvl:g} km")
        name = m.swath(t, lvl)
        self._load(self.swath_view, "swath", m.url("swaths", name),
                   f"{m.folder} {m.storm} {t}Z swath at {lvl:g} km")
        near = [m.swath(t, levels[j]) for j in (i - 1, i + 1, i + 2) if 0 <= j < len(levels)]
        self._prefetch([m.url("swaths", n) for n in near if n])

    def _var(self) -> str:
        b = self.var_group.checkedButton()
        return b.property("var") if b is not None else "vt"

    def _show_rz(self):
        m, w = self.mission, self.window_combo.currentData()
        if not (m and w):
            self.rz_view.set_message("No azimuthal means for this mission.")
            return
        var = self._var()
        name = m.rzmean(w, var)
        if name is None:
            have = ", ".join(tdr.RZ_VARS[v] for v in m.rz_vars(w)) or "none"
            self.rz_view.set_message(f"No {tdr.RZ_VARS.get(var, var).lower()} mean for this window "
                                     f"(available: {have}).")
            return
        self._load(self.rz_view, "rz", m.url("rzmeans", name),
                   f"{m.folder} {m.storm} {w}Z azimuthal mean {tdr.RZ_VARS[var].lower()}")
        self._prefetch([m.url("rzmeans", m.rzmean(w, v)) for v in m.rz_vars(w) if v != var])

    def _show_other(self):
        m = self.mission
        if m is None:
            return
        label, folder, contains = OTHER[self.other_combo.currentIndex()]
        w = self.window_combo.currentData()
        names = m.product(folder, None, contains)
        pick = [n for n in names if w and f"_{w}_" in n] or names
        if not pick:
            self.other_view.set_message(f"No {label.lower()} for this mission.")
            return
        self._load(self.other_view, "other", m.url(folder, pick[-1]), f"{m.folder} {m.storm} {label}")

    def state(self) -> dict:
        return {"days": self.days.value(), "every": self.storm_combo.currentIndex(),
                "tab": self.tabs.currentIndex(), "var": self._var(),
                "other": self.other_combo.currentIndex()}

    def restore(self, st: dict):
        if not st:
            return
        self.days.setValue(int(st.get("days", 7)))
        self.storm_combo.setCurrentIndex(int(st.get("every", 0)))
        self.tabs.setCurrentIndex(int(st.get("tab", 0)))
        self.other_combo.setCurrentIndex(int(st.get("other", 0)))
        for b in self.var_group.buttons():
            if b.property("var") == st.get("var"):
                b.setChecked(True)
