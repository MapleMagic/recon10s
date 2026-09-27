#!/usr/bin/env python3
"""
recon10s_mwpanel — the Microwave tab.

Left column: sign-in, where to look, which passes exist. Right: two maps side
by side, 89 GHz and 37 GHz, each with small Color / H / V / PCT tabs.

The pixel probe reads the nearest *native* footprint, not the display grid,
so the kelvin value it reports is the value the instrument recorded. It is
drawn with blitting -- only the marker and the label are repainted as the
mouse moves, so it stays smooth even with coastlines and a colour bar.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QRadioButton, QSplitter, QTabBar, QVBoxLayout, QWidget,
)

import recon10s_mw as mw
import recon10s_mwview as mwview
import recon10s_nhc as nhc


# ----------------------------------------------------------------- threads

class _Signals(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)


class _Task(QRunnable):
    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args
        self.signals = _Signals()

    @pyqtSlot()
    def run(self):
        try:
            self.signals.done.emit(self.fn(self.signals.status.emit, *self.args))
        except PermissionError as exc:
            self.signals.failed.emit(str(exc))
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------- one map pane

class BandView(QWidget):
    """A 37 or 89 GHz map with product tabs and a blitted pixel probe."""

    def __init__(self, band: str, owner: "MicrowavePanel"):
        super().__init__()
        self.band = band
        self.owner = owner
        self.pass_ = None
        self.index: Optional[mwview.BandIndex] = None
        self.ax = None
        self._background = None
        self._pinned = False

        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.setDocumentMode(True)
        for name in mwview.PRODUCTS:
            self.tabs.addTab(name)
        self.tabs.currentChanged.connect(lambda _: self.redraw())

        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("axes_leave_event", self._on_leave)
        self.canvas.mpl_connect("draw_event", self._on_draw)

        self.readout = QLabel(" ")
        self.readout.setObjectName("probeReadout")
        self.readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        head = QHBoxLayout()
        title = QLabel(f"{band} GHz")
        title.setObjectName("bandTitle")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self.tabs)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(head)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.readout)
        self.redraw()

    @property
    def product(self) -> str:
        return mwview.PRODUCTS[self.tabs.currentIndex()]

    def set_pass(self, pass_, index):
        self.pass_, self.index = pass_, index
        self._pinned = False
        self.redraw()

    def redraw(self):
        self._background = None
        if self.pass_ is None:
            self.figure.clear()
            self.figure.patch.set_facecolor("#12151b" if self.owner.theme != "light" else "#ffffff")
            self.figure.text(0.5, 0.5, "No pass loaded", ha="center", va="center",
                             color="#8b95a5", fontsize=10)
            self.ax = None
            self.canvas.draw_idle()
            return
        try:
            self.ax = mwview.draw_band(self.figure, self.pass_, self.index, self.band,
                                       self.product, self.owner.extent(), theme=self.owner.theme)
        except Exception as exc:
            self.readout.setText(f"Could not draw: {exc}")
            return

        # Probe furniture, excluded from normal draws and blitted on top.
        self._marker, = self.ax.plot([], [], marker="s", markersize=7, mfc="none",
                                     mec="#ffffff", mew=1.6, animated=True, zorder=20,
                                     transform=self.ax.projection)
        self._label = self.ax.annotate(
            "", xy=(0, 0), xycoords="data", xytext=(12, 12), textcoords="offset points",
            fontsize=8, family="monospace", color="#f2f5f8", animated=True, zorder=21,
            bbox=dict(boxstyle="round,pad=0.35", fc="#12151bdd", ec="#5c6675", lw=0.8))
        self._label.set_visible(False)
        self.canvas.draw_idle()

    # -------------------------------------------------------------- probe

    def _on_draw(self, _event):
        if self.ax is not None:
            self._background = self.canvas.copy_from_bbox(self.figure.bbox)

    def _on_leave(self, _event):
        if not self._pinned:
            self._clear_probe()

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.button != 1:
            return
        if self._pinned:
            self._pinned = False
            self._on_move(event)
        else:
            self._pinned = self._probe_at(event) is not None

    def _on_move(self, event):
        if self._pinned or not self.owner.probe_enabled():
            return
        if event.inaxes is not self.ax:
            return
        self._probe_at(event)

    def _probe_at(self, event):
        if self.index is None or event.xdata is None:
            return None
        # PlateCarree axes: data coordinates are lon/lat already.
        hit = self.index.probe(event.ydata, event.xdata, self.band)
        if hit is None:
            self._clear_probe()
            self.readout.setText("Off the swath")
            return None

        focus = {"Color": None, "H": "H", "V": "V", "PCT": "PCT"}[self.product]
        rows = []
        for key in ("H", "V", "PCT"):
            mark = "\u25b8" if key == focus else " "
            rows.append(f"{mark}{self.band}{key:<3} {hit[key]:6.1f} K")
        self._label.set_text("\n".join(rows))
        self._label.xy = (hit["lon"], hit["lat"])
        # Keep the box inside the map near the right and top edges.
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        dx = -12 if hit["lon"] > x0 + 0.62 * (x1 - x0) else 12
        dy = -12 if hit["lat"] > y0 + 0.75 * (y1 - y0) else 12
        self._label.set_position((dx, dy))
        self._label.set_horizontalalignment("right" if dx < 0 else "left")
        self._label.set_verticalalignment("top" if dy < 0 else "bottom")
        self._label.set_visible(True)
        self._marker.set_data([hit["lon"]], [hit["lat"]])

        ns = "N" if hit["lat"] >= 0 else "S"
        ew = "E" if hit["lon"] >= 0 else "W"
        pin = "   (pinned \u2014 click to release)" if self._pinned else ""
        self.readout.setText(
            f"{abs(hit['lat']):.3f}{ns} {abs(hit['lon']):.3f}{ew}   "
            f"H {hit['H']:.1f} K   V {hit['V']:.1f} K   PCT {hit['PCT']:.1f} K   "
            f"{hit['group']} scan {hit['scan']} pixel {hit['pixel']}{pin}")
        self._blit()
        return hit

    def _clear_probe(self):
        if self.ax is None:
            return
        self._label.set_visible(False)
        self._marker.set_data([], [])
        self.readout.setText(" ")
        self._blit()

    def _blit(self):
        if self._background is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self._background)
        self.ax.draw_artist(self._marker)
        if self._label.get_visible():
            self.ax.draw_artist(self._label)
        self.canvas.blit(self.figure.bbox)


# ------------------------------------------------------------------ the tab

class MicrowavePanel(QWidget):
    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self.theme = theme
        self.pool = QThreadPool.globalInstance()
        self._busy = False
        self._candidates: List[mw.Candidate] = []
        self._pass: Optional[mw.MWPass] = None

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_controls())

        maps = QWidget()
        row = QHBoxLayout(maps)
        row.setContentsMargins(0, 0, 0, 0)
        self.view89 = BandView("89", self)
        self.view37 = BandView("37", self)
        row.addWidget(self.view89, 1)
        row.addWidget(self.view37, 1)
        split.addWidget(maps)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([320, 1100])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(split)

        saved = mw.load_saved_email()
        if saved:
            self.email.setText(saved)
            self.cred_state.setText(f"Loaded from {mw.CREDENTIALS_FILE}")
        self.refresh_storms()

    # ------------------------------------------------------------ controls

    def _build_controls(self) -> QWidget:
        col = QWidget()
        col.setMinimumWidth(290)
        col.setMaximumWidth(420)
        v = QVBoxLayout(col)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(10)

        cred = QGroupBox("PPS near-real-time sign-in")
        cf = QVBoxLayout(cred)
        self.email = QLineEdit()
        self.email.setPlaceholderText("registered email (also the password)")
        cf.addWidget(self.email)
        buttons = QHBoxLayout()
        b_test = QPushButton("Test")
        b_test.clicked.connect(self.test_login)
        b_save = QPushButton("Save to jsimpson.json")
        b_save.clicked.connect(self.save_credentials)
        b_forget = QPushButton("Forget")
        b_forget.clicked.connect(self.forget_credentials)
        for b in (b_test, b_save, b_forget):
            buttons.addWidget(b)
        cf.addLayout(buttons)
        self.cred_state = QLabel("Not saved \u2014 kept for this session only.")
        self.cred_state.setObjectName("hint")
        self.cred_state.setWordWrap(True)
        cf.addWidget(self.cred_state)
        v.addWidget(cred)

        where = QGroupBox("Storm")
        wf = QFormLayout(where)
        self.rb_nhc = QRadioButton("NHC active storm")
        self.rb_nhc.setChecked(True)
        self.rb_manual = QRadioButton("Manual position")
        self.storm_box = QComboBox()
        storm_row = QHBoxLayout()
        storm_row.addWidget(self.storm_box, 1)
        b_storms = QPushButton("\u21bb")
        b_storms.setToolTip("Reload active storms from NHC")
        b_storms.setFixedWidth(32)
        b_storms.clicked.connect(self.refresh_storms)
        storm_row.addWidget(b_storms)
        wf.addRow(self.rb_nhc)
        wf.addRow(storm_row)
        wf.addRow(self.rb_manual)
        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setRange(-60, 60)
        self.lat_spin.setDecimals(2)
        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setRange(-180, 180)
        self.lon_spin.setDecimals(2)
        pos = QHBoxLayout()
        pos.addWidget(QLabel("Lat"))
        pos.addWidget(self.lat_spin)
        pos.addWidget(QLabel("Lon"))
        pos.addWidget(self.lon_spin)
        wf.addRow(pos)
        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(1.0, 8.0)
        self.zoom_spin.setSingleStep(0.5)
        self.zoom_spin.setValue(3.0)
        self.zoom_spin.setSuffix("\u00b0 each side")
        self.zoom_spin.valueChanged.connect(self._redraw_views)
        wf.addRow("View", self.zoom_spin)
        v.addWidget(where)

        find = QGroupBox("Passes")
        ff = QVBoxLayout(find)
        sens = QHBoxLayout()
        self.sensor_boxes: Dict[str, QCheckBox] = {}
        for name in mw.SENSORS:
            chk = QCheckBox(name)
            chk.setChecked(True)
            self.sensor_boxes[name] = chk
            sens.addWidget(chk)
        ff.addLayout(sens)
        hours = QHBoxLayout()
        hours.addWidget(QLabel("Look back"))
        self.hours_spin = QDoubleSpinBox()
        self.hours_spin.setRange(1, 48)
        self.hours_spin.setDecimals(0)
        self.hours_spin.setValue(12)
        self.hours_spin.setSuffix(" h")
        hours.addWidget(self.hours_spin)
        hours.addStretch(1)
        self.btn_find = QPushButton("Find passes")
        self.btn_find.setObjectName("primary")
        self.btn_find.clicked.connect(self.find_passes)
        hours.addWidget(self.btn_find)
        ff.addLayout(hours)
        self.pass_list = QListWidget()
        self.pass_list.itemDoubleClicked.connect(lambda _: self.load_selected())
        ff.addWidget(self.pass_list, 1)
        self.btn_load = QPushButton("Load selected pass")
        self.btn_load.clicked.connect(self.load_selected)
        ff.addWidget(self.btn_load)
        v.addWidget(find, 1)

        self.chk_probe = QCheckBox("Pixel probe (hover for TB, click to pin)")
        self.chk_probe.setChecked(True)
        v.addWidget(self.chk_probe)

        self.status = QLabel("Sign in, pick a storm, then Find passes.")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        return col

    # ---------------------------------------------------------- credentials

    def _email(self) -> Optional[str]:
        email = self.email.text().strip()
        if not email or "@" not in email:
            self.status.setText("Enter the email registered for PPS near-real-time access.")
            return None
        return email

    def test_login(self):
        email = self._email()
        if email:
            ok, msg = mw.check_login(email)
            self.status.setText(msg)

    def save_credentials(self):
        email = self._email()
        if not email:
            return
        answer = QMessageBox.question(
            self, "Save sign-in",
            f"Save this email to\n{mw.CREDENTIALS_FILE}?\n\nIt is stored as plain text, "
            "and on this server the email is also the password.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        path = mw.save_email(email)
        self.cred_state.setText(f"Saved to {path}")

    def forget_credentials(self):
        removed = mw.forget_saved_email()
        self.cred_state.setText("Saved sign-in removed." if removed
                                else "Nothing was saved. Kept for this session only.")

    # --------------------------------------------------------------- storm

    def refresh_storms(self):
        self.storm_box.clear()
        try:
            storms = nhc.active_storms()
        except Exception as exc:
            storms = []
            self.status.setText(f"Could not reach NHC ({exc}). Use a manual position.")
        for fix in storms:
            self.storm_box.addItem(fix.label, fix)
        if not storms:
            self.rb_manual.setChecked(True)

    def centre(self):
        if self.rb_nhc.isChecked():
            fix = self.storm_box.currentData()
            if fix is not None:
                return (fix.lat, fix.lon)
        return (self.lat_spin.value(), self.lon_spin.value())

    def extent(self):
        c = self._pass.centre if self._pass else self.centre()
        return mwview.default_extent(c, self.zoom_spin.value())

    def probe_enabled(self) -> bool:
        return self.chk_probe.isChecked()

    # -------------------------------------------------------------- finding

    def _start(self, fn, on_done, *args):
        if self._busy:
            return
        self._busy = True
        self.btn_find.setEnabled(False)
        self.btn_load.setEnabled(False)
        task = _Task(fn, *args)
        task.signals.status.connect(self.status.setText)
        task.signals.done.connect(lambda result: (self._finish(), on_done(result)))
        task.signals.failed.connect(lambda msg: (self._finish(), self.status.setText(msg)))
        self.pool.start(task)

    def _finish(self):
        self._busy = False
        self.btn_find.setEnabled(True)
        self.btn_load.setEnabled(True)

    def find_passes(self):
        email = self._email()
        if not email:
            return
        sensors = [n for n, chk in self.sensor_boxes.items() if chk.isChecked()]
        if not sensors:
            self.status.setText("Tick at least one sensor.")
            return
        lat, lon = self.centre()
        hours = float(self.hours_spin.value())

        def work(status, sensors, lat, lon, email, hours):
            return mw.find_passes(sensors, lat, lon, email, hours=hours, progress=status)

        self.pass_list.clear()
        self._start(work, self._show_candidates, sensors, lat, lon, email, hours)

    def _show_candidates(self, candidates):
        self._candidates = candidates
        self.pass_list.clear()
        for c in candidates:
            item = QListWidgetItem(c.label)
            item.setData(Qt.ItemDataRole.UserRole, c)
            self.pass_list.addItem(item)
        if candidates:
            self.pass_list.setCurrentRow(0)
            screened = sum(1 for c in candidates if c.screened)
            self.status.setText(f"{len(candidates)} pass(es) near the storm, newest first"
                                + ("" if screened == len(candidates)
                                   else " \u2014 some could not be orbit-screened"))
        else:
            self.status.setText("No passes reached the storm in that window. "
                                "Try a longer look-back.")

    # -------------------------------------------------------------- loading

    def load_selected(self):
        item = self.pass_list.currentItem()
        email = self._email()
        if item is None or not email:
            return
        cand = item.data(Qt.ItemDataRole.UserRole)
        centre = self.centre()
        box = self.zoom_spin.value() + 3.0

        def work(status, cand, centre, email, box):
            g = cand.granule

            def dl(done, total):
                if total:
                    status(f"Downloading {g.sensor}\u2026 {done / 1e6:.1f} of {total / 1e6:.1f} MB")
                else:
                    status(f"Downloading {g.sensor}\u2026 {done / 1e6:.1f} MB")

            path = mw.download_granule(g, email, progress=dl)
            status("Reading channels\u2026")
            p = mw.load_pass(path, g.sensor, centre, box_deg=box, granule=g)
            status("Indexing footprints\u2026")
            idx = {b: (mwview.BandIndex(p.bands[b]) if p.bands[b].lat.size else None)
                   for b in ("89", "37")}
            return p, idx

        self._start(work, self._show_pass, cand, centre, email, box)

    def _show_pass(self, result):
        p, idx = result
        self._pass = p
        self.view89.set_pass(p, idx["89"])
        self.view37.set_pass(p, idx["37"])
        note = "" if p.covers_centre else "  \u2014 swath only clips the storm; the centre is outside it"
        self.status.setText(f"{p.title}{note}")

    def _redraw_views(self):
        for view in (self.view89, self.view37):
            view.redraw()

    def set_theme(self, theme: str):
        self.theme = theme
        self._redraw_views()
