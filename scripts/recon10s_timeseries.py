#!/usr/bin/env python3
"""
recon10s_timeseries — stacked time series for a recon mission.

Three panels, all sharing one time axis:

  Wind & pressure   flight-level wind (kt, left) and extrapolated MSLP
                    (mb, right). MSLP is drawn the normal way up, so a
                    deepening centre dips toward the bottom of the panel.
  Temperature       ambient temperature and dew point (degC). Opens at 0-25
                    and stretches only when the data leaves that range.
  Altitude & press  geopotential height (m, left) and static pressure
                    (mb, right, inverted so it tracks height). Ragged traces
                    here are the turbulence signature.

Move the mouse for a readout at the nearest 1 Hz observation, coloured to
match each trace; click to pin it, click again to release. The crosshair and
the markers follow across every panel, so one cursor reads all of them.

A panel with nothing shown collapses. Every panel keeps fixed-width axes so
the crosshair lines up vertically whichever series are visible.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget,
)

WIND_COLOR = "#4dd0e1"
MSLP_COLOR = "#ffb74d"
TEMP_COLOR = "#ef5350"
DEWPT_COLOR = "#66bb6a"
HEIGHT_COLOR = "#b39ddb"
PRESS_COLOR = "#90a4ae"
MUTED = "#8b95a5"

PLOT_BG_DARK = "#12151b"
PLOT_BG_LIGHT = "#ffffff"

AXIS_WIDTH = 74  # fixed, so the panels line up under one another

WINDOWS = [
    ("Last 5 min", 5 * 60),
    ("Last 15 min", 15 * 60),
    ("Last 30 min", 30 * 60),
    ("Last hour", 3600),
    ("Last 3 hours", 3 * 3600),
    ("Whole flight", None),
]


class Series:
    """One trace: where its numbers come from and how to show them."""

    def __init__(self, key: str, label: str, color: str, unit: str,
                 decimals: int = 0, side: str = "left", default_on: bool = True):
        self.key = key
        self.label = label
        self.color = color
        self.unit = unit
        self.decimals = decimals
        self.side = side
        self.default_on = default_on
        self.checkbox: Optional[QCheckBox] = None
        self.curve: Optional[pg.PlotDataItem] = None
        self.dot: Optional[pg.ScatterPlotItem] = None

    @property
    def visible(self) -> bool:
        return self.checkbox is None or self.checkbox.isChecked()

    def format(self, value: float) -> str:
        return f"{value:.{self.decimals}f} {self.unit}"


class UTCDateAxis(pg.DateAxisItem):
    """Time axis labelled in UTC — recon time is always Z time."""

    def tickStrings(self, values, scale, spacing):
        if spacing < 60:
            fmt = "%H:%M:%S"
        elif spacing < 86400:
            fmt = "%H:%M"
        else:
            fmt = "%d %b %H:%M"
        out = []
        for v in values:
            try:
                out.append(dt.datetime.fromtimestamp(v, tz=dt.timezone.utc).strftime(fmt))
            except (ValueError, OSError, OverflowError):
                out.append("")
        return out


class Panel:
    """
    One stacked plot. Left-axis series live in the PlotItem's own view box;
    a right-axis series gets a second view box locked to the same x range.
    """

    def __init__(self, title: str, series: List[Series], invert_right: bool = False,
                 left_floor: Optional[float] = None, left_hint=None, stretch: int = 2):
        self.title = title
        self.series = series
        self.invert_right = invert_right
        self.left_floor = left_floor
        self.left_hint = left_hint  # range the panel opens at unless data exceeds it
        self.stretch = stretch

        self.left_series = [s for s in series if s.side == "left"]
        self.right_series = [s for s in series if s.side == "right"]

        self.widget = pg.PlotWidget(axisItems={"bottom": UTCDateAxis(orientation="bottom")})
        self.widget.showGrid(x=True, y=True, alpha=0.18)
        self.widget.setMouseEnabled(x=True, y=False)
        self.widget.setMenuEnabled(False)
        self.widget.setMinimumHeight(110)

        self.item = self.widget.getPlotItem()
        self.vb = self.item.vb

        # Both axes always occupy the same width, so panels stay aligned.
        self.item.showAxis("right")
        for name in ("left", "right"):
            self.item.getAxis(name).setWidth(AXIS_WIDTH)

        left_label = " / ".join(s.label for s in self.left_series)
        left_unit = self.left_series[0].unit if self.left_series else ""
        left_color = self.left_series[0].color if len(self.left_series) == 1 else MUTED
        self.item.setLabel("left", f"{left_label} ({left_unit})", color=left_color)
        self.item.getAxis("left").setTextPen(left_color)

        self.vb_right: Optional[pg.ViewBox] = None
        if self.right_series:
            rs = self.right_series[0]
            self.vb_right = pg.ViewBox()
            self.widget.scene().addItem(self.vb_right)
            self.item.getAxis("right").linkToView(self.vb_right)
            self.item.setLabel("right", f"{rs.label} ({rs.unit})", color=rs.color)
            self.item.getAxis("right").setTextPen(rs.color)
            self.vb_right.setXLink(self.vb)
            self.vb_right.setMouseEnabled(x=True, y=False)
            if invert_right:
                self.vb_right.invertY(True)
            self.vb.sigResized.connect(self._sync_right)
            QTimer.singleShot(0, self._sync_right)
        else:
            self.item.getAxis("right").setStyle(showValues=False)
            self.item.getAxis("right").setTextPen(MUTED)

        for s in series:
            s.curve = pg.PlotDataItem(pen=pg.mkPen(s.color, width=1.6), connect="finite",
                                      antialias=True, autoDownsample=True, clipToView=True)
            s.dot = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(s.color),
                                       pen=pg.mkPen(PLOT_BG_DARK, width=1.5))
            s.dot.setZValue(60)
            target = self.vb_right if s.side == "right" else self.vb
            target.addItem(s.curve)
            target.addItem(s.dot)

        self.cursor = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#6b7688", width=1, style=Qt.PenStyle.DashLine))
        self.cursor.setZValue(50)
        self.cursor.hide()
        self.widget.addItem(self.cursor, ignoreBounds=True)

        self.readout = pg.TextItem(html="", anchor=(0, 1))
        self.readout.setZValue(70)
        self.readout.hide()
        self.widget.addItem(self.readout, ignoreBounds=True)

    # -------------------------------------------------------------- geometry

    def _sync_right(self):
        if self.vb_right is not None:
            self.vb_right.setGeometry(self.vb.sceneBoundingRect())
            self.vb_right.linkedViewChanged(self.vb, self.vb_right.XAxis)

    @property
    def any_visible(self) -> bool:
        return any(s.visible for s in self.series)

    def apply_visibility(self):
        for s in self.series:
            on = s.visible
            s.curve.setVisible(on)
            s.dot.setVisible(on)
        if self.left_series:
            self.item.getAxis("left").setStyle(
                showValues=any(s.visible for s in self.left_series))
        if self.right_series:
            self.item.getAxis("right").setStyle(
                showValues=any(s.visible for s in self.right_series))
        self.widget.setVisible(self.any_visible)

    def set_bottom_axis_visible(self, on: bool):
        self.item.getAxis("bottom").setStyle(showValues=on)
        self.item.setLabel("bottom", "Time (UTC)" if on else "", color=MUTED)

    # ------------------------------------------------------------------ data

    def set_data(self, t: np.ndarray, data: Dict[str, np.ndarray]):
        for s in self.series:
            arr = data.get(s.key)
            if arr is None or arr.size == 0:
                s.curve.clear()
            else:
                s.curve.setData(t, arr)

    def autoscale_y(self, t: np.ndarray, data: Dict[str, np.ndarray]):
        if t.size == 0:
            return
        x0, x1 = self.vb.viewRange()[0]
        mask = (t >= x0) & (t <= x1)
        if not mask.any():
            mask = np.ones_like(t, dtype=bool)

        groups = [(self.left_series, self.vb, self.left_floor, self.left_hint)]
        if self.vb_right is not None:
            groups.append((self.right_series, self.vb_right, None, None))

        for series, vb, floor, hint in groups:
            vals = []
            for s in series:
                if not s.visible:
                    continue
                arr = data.get(s.key)
                if arr is None or arr.size == 0:
                    continue
                sel = arr[mask]
                sel = sel[np.isfinite(sel)]
                if sel.size:
                    vals.append(sel)
            if not vals:
                continue
            stacked = np.concatenate(vals)
            lo, hi = float(stacked.min()), float(stacked.max())
            pad = max((hi - lo) * 0.12, 0.5)
            lo, hi = lo - pad, hi + pad
            if hint is not None:
                # Open at the hinted range; stretch only if the data needs it.
                lo, hi = min(lo, hint[0]), max(hi, hint[1])
            if floor is not None:
                lo = max(lo, floor)
            vb.setYRange(lo, hi, padding=0)


class TimeSeriesPanel(QWidget):
    """The whole stack: controls on top, panels below."""

    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)

        self._t = np.empty(0)
        self._data: Dict[str, np.ndarray] = {}
        self._pinned_x: Optional[float] = None
        self._time_color = MUTED

        self.panels: List[Panel] = [
            Panel("Wind & pressure",
                  [Series("wind", "Flight-level wind", WIND_COLOR, "kt", 0, "left"),
                   Series("mslp", "Extrapolated MSLP", MSLP_COLOR, "mb", 0, "right")],
                  left_floor=0.0, stretch=3),
            Panel("Temperature",
                  [Series("temp", "Temperature", TEMP_COLOR, "\u00b0C", 1, "left"),
                   Series("dewpt", "Dew point", DEWPT_COLOR, "\u00b0C", 1, "left")],
                  left_hint=(0.0, 25.0), stretch=2),
            Panel("Altitude & pressure",
                  [Series("height", "Geopotential height", HEIGHT_COLOR, "m", 0, "left"),
                   Series("press", "Static pressure", PRESS_COLOR, "mb", 1, "right")],
                  invert_right=True, left_floor=0.0, stretch=2),
        ]
        self.all_series = [s for p in self.panels for s in p.series]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addLayout(self._build_controls())

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        for panel in self.panels:
            self.splitter.addWidget(panel.widget)
        for i, panel in enumerate(self.panels):
            self.splitter.setStretchFactor(i, panel.stretch)
        layout.addWidget(self.splitter, 1)

        # One time axis for all three.
        base = self.panels[0].vb
        for panel in self.panels[1:]:
            panel.vb.setXLink(base)

        for panel in self.panels:
            panel.widget.scene().sigMouseMoved.connect(
                lambda pos, p=panel: self._on_mouse_moved(p, pos))
            panel.widget.scene().sigMouseClicked.connect(
                lambda ev, p=panel: self._on_mouse_clicked(p, ev))
            panel.vb.sigRangeChangedManually.connect(
                lambda *_: self.chk_follow.setChecked(False))

        self.set_theme(theme)
        self._apply_visibility()

    # -------------------------------------------------------------- controls

    def _build_controls(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 2)
        bar.setSpacing(12)

        for series in self.all_series:
            chk = QCheckBox(series.label)
            chk.setChecked(series.default_on)
            chk.setStyleSheet(f"color: {series.color}; font-weight: 600;")
            chk.toggled.connect(self._apply_visibility)
            series.checkbox = chk
            bar.addWidget(chk)

        bar.addStretch(1)
        bar.addWidget(QLabel("Show"))
        self.window_box = QComboBox()
        for label, _ in WINDOWS:
            self.window_box.addItem(label)
        self.window_box.setCurrentIndex(2)
        self.window_box.currentIndexChanged.connect(lambda _: self._rescale(force=True))
        bar.addWidget(self.window_box)

        self.chk_follow = QCheckBox("Follow latest")
        self.chk_follow.setChecked(True)
        self.chk_follow.setToolTip("Keep the right edge on the newest data.\n"
                                   "Panning or zooming turns this off.")
        self.chk_follow.toggled.connect(lambda on: self._rescale(force=True) if on else None)
        bar.addWidget(self.chk_follow)
        return bar

    def _apply_visibility(self):
        for panel in self.panels:
            panel.apply_visibility()

        visible = [p for p in self.panels if p.any_visible]
        for panel in self.panels:
            panel.set_bottom_axis_visible(bool(visible) and panel is visible[-1])

        self._autoscale_all()
        if self._pinned_x is not None:
            self._update_readout(None, self._pinned_x, keep_label_position=True)

    # ------------------------------------------------------------------ data

    def set_series(self, series: Dict[str, np.ndarray]) -> None:
        """`series` holds a 't' array plus one array per plotted key."""
        t = series.get("t")
        if t is None or len(t) == 0:
            self.clear()
            return
        self._t = t
        self._data = series
        for panel in self.panels:
            panel.set_data(t, series)
        self._rescale()
        if self._pinned_x is not None:
            self._update_readout(None, self._pinned_x, keep_label_position=True)

    def clear(self) -> None:
        self._t = np.empty(0)
        self._data = {}
        self._pinned_x = None
        for panel in self.panels:
            for s in panel.series:
                s.curve.clear()
                s.dot.setData([], [])
            panel.cursor.hide()
            panel.readout.hide()

    def latest_text(self) -> str:
        """One-line summary of the newest observation, for the status bar."""
        if self._t.size == 0:
            return "No data"
        i = self._t.size - 1
        stamp = dt.datetime.fromtimestamp(self._t[i], tz=dt.timezone.utc).strftime("%H:%M:%S")
        bits = [f"{stamp}Z"]
        for key, suffix in (("wind", " kt"), ("mslp", " mb")):
            arr = self._data.get(key)
            if arr is not None and arr.size and np.isfinite(arr[i]):
                bits.append(f"{arr[i]:.0f}{suffix}")
        return "  \u00b7  ".join(bits)

    # --------------------------------------------------------------- display

    def set_theme(self, theme: str) -> None:
        dark = theme != "light"
        bg = PLOT_BG_DARK if dark else PLOT_BG_LIGHT
        fg = "#c9d1d9" if dark else "#222222"
        for panel in self.panels:
            panel.widget.setBackground(bg)
            axis = panel.item.getAxis("bottom")
            axis.setPen(pg.mkPen(fg, width=1))
            axis.setTextPen(fg)
            for s in panel.series:
                s.dot.setPen(pg.mkPen(bg, width=1.5))
            panel.readout.fill = (pg.mkBrush(18, 21, 28, 235) if dark
                                  else pg.mkBrush(255, 255, 255, 240))
            panel.readout.border = pg.mkPen("#39404e" if dark else "#c8ccd4")
            panel.readout.update()
        self._time_color = MUTED if dark else "#555a63"

    def _rescale(self, force: bool = False) -> None:
        if self._t.size == 0:
            return
        if not (self.chk_follow.isChecked() or force):
            self._autoscale_all()
            return
        span = WINDOWS[self.window_box.currentIndex()][1]
        end = float(self._t[-1])
        start = float(self._t[0]) if span is None else max(float(self._t[0]), end - span)
        if end - start < 30:
            end = start + 30
        self.panels[0].widget.setXRange(start, end + (end - start) * 0.02, padding=0)
        self._autoscale_all()

    def _autoscale_all(self) -> None:
        for panel in self.panels:
            if panel.any_visible:
                panel.autoscale_y(self._t, self._data)

    # ---------------------------------------------------------------- cursor

    def _on_mouse_moved(self, panel: Panel, pos) -> None:
        if self._pinned_x is not None:
            return
        if not panel.widget.sceneBoundingRect().contains(pos):
            self._hide_cursor()
            return
        self._update_readout(panel, panel.vb.mapSceneToView(pos).x(), scene_pos=pos)

    def _on_mouse_clicked(self, panel: Panel, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if not panel.widget.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = panel.vb.mapSceneToView(ev.scenePos()).x()
        self._pinned_x = None if self._pinned_x is not None else x
        self._update_readout(panel, x, scene_pos=ev.scenePos())

    def _hide_cursor(self) -> None:
        for panel in self.panels:
            panel.cursor.hide()
            panel.readout.hide()
            for s in panel.series:
                s.dot.setData([], [])

    def _nearest_index(self, x: float) -> Optional[int]:
        if self._t.size == 0:
            return None
        i = int(np.searchsorted(self._t, x))
        if i <= 0:
            return 0
        if i >= self._t.size:
            return self._t.size - 1
        return i if (self._t[i] - x) < (x - self._t[i - 1]) else i - 1

    def _value_at(self, key: str, idx: int) -> float:
        arr = self._data.get(key)
        if arr is None or arr.size <= idx:
            return float("nan")
        return float(arr[idx])

    def _update_readout(self, hovered: Optional[Panel], x: float,
                        scene_pos=None, keep_label_position: bool = False) -> None:
        idx = self._nearest_index(x)
        if idx is None:
            self._hide_cursor()
            return

        t = float(self._t[idx])
        stamp = dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).strftime("%H:%M:%S")

        # Crosshair and markers move on every panel, not just the hovered one.
        for panel in self.panels:
            panel.readout.hide()
            if not panel.any_visible:
                panel.cursor.hide()
                continue
            panel.cursor.setPos(t)
            panel.cursor.show()
            for s in panel.series:
                value = self._value_at(s.key, idx)
                if s.visible and np.isfinite(value):
                    s.dot.setData([t], [value])
                else:
                    s.dot.setData([], [])

        if hovered is None or not hovered.any_visible:
            hovered = next((p for p in self.panels if p.any_visible), None)
            if hovered is None:
                return

        rows = [f'<span style="color:{self._time_color};">{stamp}Z</span>']
        for s in hovered.series:
            if not s.visible:
                continue
            value = self._value_at(s.key, idx)
            shown = s.format(value) if np.isfinite(value) else self._missing_text(s)
            rows.append(f'<span style="color:{s.color};">{s.label}: {shown}</span>')
        if self._pinned_x is not None:
            rows.append(f'<span style="color:{self._time_color};font-size:9pt;">'
                        'pinned — click to release</span>')

        hovered.readout.setHtml(
            '<div style="font-family:Menlo,Consolas,monospace;font-size:11pt;'
            'line-height:150%;white-space:nowrap;padding:2px 4px;">'
            + "<br/>".join(rows) + "</div>")

        if not keep_label_position:
            self._place_label(hovered, t, scene_pos)
        hovered.readout.show()

    @staticmethod
    def _missing_text(series: Series) -> str:
        if series.key == "mslp":
            return "above 550 mb level"
        return "no data"

    def _place_label(self, panel: Panel, t: float, scene_pos) -> None:
        (x0, x1), (y0, y1) = panel.vb.viewRange()
        if scene_pos is not None:
            y = panel.vb.mapSceneToView(scene_pos).y()
        else:
            y = y0 + (y1 - y0) * 0.8

        # Flip the box near the edges so it never runs off the panel.
        right_half = t > x0 + (x1 - x0) * 0.6
        near_top = y > y0 + (y1 - y0) * 0.7
        panel.readout.setAnchor((1 if right_half else 0, 1 if near_top else 0))
        offset = (x1 - x0) * 0.008
        panel.readout.setPos(t - offset if right_half else t + offset, y)
