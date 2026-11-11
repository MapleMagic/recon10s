#!/usr/bin/env python3
"""
recon10s_analysispanel — the Analysis tab (Reconnaissance section).

Works from the recon the Satellite tab has loaded (it publishes a
ReconBundle), so there is one set of data and one place it is fetched.

  * Fixes table: every centre fix (vortex messages and derived), with
    minimum pressure, eye, the peak flight-level wind and RMW on each leg,
    and tilt where it can be measured.
  * Radial profiles: flight-level wind against distance from the centre for
    the passes selected in the table -- inbound legs to the left of zero,
    outbound to the right -- with extrapolated surface pressure (or
    geopotential height) below.
  * Timeline: each aircraft's HDOB batches, dropsondes and fixes on one
    time axis, to see coverage and gaps at a glance.
  * Surface winds: for every dropsonde, the flight-level wind at release
    reduced by the standard ratio beside the sonde's WL150 and MBL estimates
    and its own surface wind -- surface winds without SFMR.
"""
from __future__ import annotations

import datetime as dt
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

import recon10s_surface as surface

try:
    import recon10s_plot
except Exception:  # pragma: no cover
    recon10s_plot = None

PASS_COLOURS = ["#4dd0e1", "#ffb74d", "#ba68c8", "#81c784", "#f06292", "#fff176", "#64b5f6", "#e57373"]
COLUMNS = ["Time", "Source", "Aircraft", "Position", "Min SLP", "Eye",
           "Inbound max", "Outbound max", "Tilt (sfc from FL)"]
SURFACE_COLUMNS = ["#", "Released", "Aircraft", "Where", "Flight level", "FL \u00d7 ratio", "WL150",
                   "\u00d7 0.83", "MBL", "\u00d7 0.80", "Sonde surface", "Surface / FL"]


def _wind_rgb(kt):
    if recon10s_plot is not None and kt is not None:
        return recon10s_plot._speed_to_rgb_normalized(float(kt))
    return (0.6, 0.6, 0.6)


class AnalysisPanel(QWidget):
    request_load = pyqtSignal()        # ask the Satellite tab to load all recon
    request_show = pyqtSignal()        # jump to the Satellite tab
    request_drop = pyqtSignal(object)  # open a dropsonde's sounding

    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self.theme = theme
        self.bundle = None

        top = QHBoxLayout()
        self.summary = QLabel("No recon loaded yet.")
        self.summary.setObjectName("hint")
        top.addWidget(self.summary, 1)
        b_load = QPushButton("Load all recon for the storm")
        b_load.setObjectName("primary")
        b_load.setToolTip("Loads NHC HDOBs, vortex messages and dropsondes for the storm chosen\n"
                          "in Imagery \u203a Satellite (the data is shared between the tabs).")
        b_load.clicked.connect(self.request_load.emit)
        top.addWidget(b_load)
        b_map = QPushButton("Show on the map")
        b_map.clicked.connect(self.request_show.emit)
        top.addWidget(b_map)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.draw_profiles)
        self.motion = QLabel("")
        self.motion.setObjectName("hint")

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.table, 1)
        lv.addWidget(self.motion)

        self.tabs = QTabWidget()
        self.prof_fig = Figure(figsize=(7, 6), dpi=100)
        self.prof_canvas = FigureCanvasQTAgg(self.prof_fig)
        pw = QWidget()
        pv = QVBoxLayout(pw)
        pv.setContentsMargins(0, 0, 0, 0)
        prow = QHBoxLayout()
        prow.addWidget(NavigationToolbar2QT(self.prof_canvas, pw), 1)
        self.chk_sfc = QCheckBox("Surface estimate")
        self.chk_sfc.setChecked(True)
        self.chk_sfc.setToolTip("Dashed: each ob's flight-level wind times the standard ratio for its level\n"
                                "(700 mb 0.90, 850 mb 0.80, 1000 ft 0.85); questionable winds left out.")
        self.chk_sfc.toggled.connect(lambda _: self.draw_profiles())
        prow.addWidget(self.chk_sfc)
        pv.addLayout(prow)
        pv.addWidget(self.prof_canvas, 1)
        self.tabs.addTab(pw, "Radial profiles")
        self.time_fig = Figure(figsize=(7, 6), dpi=100)
        self.time_canvas = FigureCanvasQTAgg(self.time_fig)
        tw = QWidget()
        tv = QVBoxLayout(tw)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.addWidget(NavigationToolbar2QT(self.time_canvas, tw))
        tv.addWidget(self.time_canvas, 1)
        self.tabs.addTab(tw, "Mission timeline")
        sw = QWidget()
        sv = QVBoxLayout(sw)
        sv.setContentsMargins(6, 6, 6, 6)
        self.sfc_note = QLabel("")
        self.sfc_note.setObjectName("hint")
        self.sfc_note.setWordWrap(True)
        sv.addWidget(self.sfc_note)
        self.sfc_table = QTableWidget(0, len(SURFACE_COLUMNS))
        self.sfc_table.setHorizontalHeaderLabels(SURFACE_COLUMNS)
        self.sfc_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sfc_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.sfc_table.verticalHeader().setVisible(False)
        self.sfc_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.sfc_table.horizontalHeader().setStretchLastSection(True)
        self.sfc_table.setToolTip("Double-click a dropsonde for its sounding.")
        self.sfc_table.cellDoubleClicked.connect(self._on_sfc_double_click)
        sv.addWidget(self.sfc_table, 1)
        self.tabs.addTab(sw, "Surface winds")
        self._sfc_rows = []

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setSizes([620, 900])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.addLayout(top)
        layout.addWidget(split, 1)
        self.set_bundle(None)

    # ------------------------------------------------------------------ data

    def set_bundle(self, bundle):
        self.bundle = bundle
        b = bundle
        if b is None or (not b.fixes and (b.overlay is None or not len(b.overlay))):
            self.summary.setText("No recon loaded yet. Load it here or in Imagery \u203a Satellite.")
        else:
            n_obs = len(b.overlay) if b.overlay is not None else 0
            n_vdm = sum(1 for f in b.fixes if f.source == "VDM")
            self.summary.setText(f"{n_obs} obs  \u00b7  {len(b.fixes)} centre fixes ({n_vdm} from vortex "
                                 f"messages)  \u00b7  {len(b.passes)} passes  \u00b7  {len(b.drops)} dropsondes")
        self._fill_table()
        self.motion.setText(f"Storm motion from the fixes: {b.motion[0]:03.0f}\u00b0 at {b.motion[1]:.0f} kt"
                            if b is not None and b.motion else "")
        self.draw_profiles()
        self.draw_timeline()
        self._fill_surface()

    def _fill_surface(self):
        b = self.bundle
        t = self.sfc_table
        t.setRowCount(0)
        rows = surface.drop_comparisons(b.drops, b.overlay) if b is not None else []
        self._sfc_rows = rows

        def kt(v, digits=0):
            return "" if v is None else f"{v:.{digits}f} kt"
        for r in rows:
            k = t.rowCount()
            t.insertRow(k)
            d = r.drop
            fl = ""
            if r.fl_wind is not None:
                fl = f"{r.fl_wind} kt ({r.fl_level or 'no ratio'})"
            elif r.fl_level:
                fl = r.fl_level
            cells = [str(d.number), f"{r.time:%d %H:%M:%SZ}" if r.time else "", d.aircraft, r.location,
                     fl, kt(r.fl_est), kt(r.wl150), kt(r.wl150_est), kt(r.mbl), kt(r.mbl_est), kt(r.sfc_wind),
                     f"{r.observed_ratio:.2f}" if r.observed_ratio is not None and (r.fl_wind or 0) >= 20 else ""]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c in (5, 7, 9) and text:
                    item.setForeground(_qcolour("#ffca28"))
                t.setItem(k, c, item)
        levels = surface.level_summary(b.overlay) if b is not None else []
        used = ", ".join(f"{lab} \u00d7{f:.2f}" for lab, f, n in levels if f == f)
        summ = surface.ratio_summary(rows)
        note = ("Standard surface/flight-level ratios (NHC): 700 mb 0.90, 850 mb 0.80, 1000 ft 0.85 (none above "
                "625 mb, or while the aircraft climbs or descends); dropsonde layer means: lowest 150 m (WL150) "
                "\u00d7 0.83, lowest 500 m (MBL) \u00d7 0.80. No SFMR.")
        if used:
            note += f" Levels flown in this set: {used}."
        if summ:
            note += (f" Here, each sonde's own surface wind over the flight-level wind at its release: median "
                     f"{summ[0]:.2f} (mean {summ[1]:.2f}, {summ[2]} drops with flight-level wind \u2265 50 kt).")
        self.sfc_note.setText(note if rows else "No dropsondes loaded. " + note)

    def _on_sfc_double_click(self, row, _col):
        if 0 <= row < len(self._sfc_rows):
            self.request_drop.emit(self._sfc_rows[row].drop)

    def _fill_table(self):
        b = self.bundle
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        if b is None:
            self.table.blockSignals(False)
            return
        by_fix = {id(p.fix): p for p in b.passes}
        tilt_of = {id(t.fix): t for t in b.tilts}
        for f in b.fixes:
            r = self.table.rowCount()
            self.table.insertRow(r)
            p = by_fix.get(id(f))
            t = tilt_of.get(id(f))
            v = f.vdm

            def leg(L, vmax_pair, at):
                if vmax_pair and at:
                    return f"{vmax_pair[1]} kt @ {at[1] * 1.852:.0f} km (VDM)"
                return f"{L.vmax:.0f} kt @ {L.rmw:.0f} km" if L else ""
            cells = [f.time.strftime("%d %H:%M:%SZ"), f.source, f.aircraft,
                     f"{abs(f.lat):.2f}{'N' if f.lat >= 0 else 'S'} {abs(f.lon):.2f}{'W' if f.lon < 0 else 'E'}",
                     f"{f.min_slp:.0f} mb" if f.min_slp else "",
                     (v.eye_shape or v.eye) if v else "",
                     leg(p.inbound if p else None, v.max_in if v else None, v.max_in_at if v else None),
                     leg(p.outbound if p else None, v.max_out if v else None, v.max_out_at if v else None),
                     f"{t.km:.1f} km toward {t.bearing:03.0f}\u00b0 ({t.source})" if t else ""]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, id(f))
                self.table.setItem(r, c, item)
        self.table.blockSignals(False)
        if self.table.rowCount():
            self.table.selectRow(self.table.rowCount() - 1)

    def _selected_passes(self):
        if self.bundle is None:
            return []
        rows = sorted({i.row() for i in self.table.selectedItems()})
        ids = [self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows]
        return [p for p in self.bundle.passes if id(p.fix) in ids]

    # --------------------------------------------------------------- drawing

    def _style(self, fig):
        dark = self.theme != "light"
        fig.patch.set_facecolor("#12151b" if dark else "#ffffff")
        return ("#dfe5ec" if dark else "#1c2129"), ("#12151b" if dark else "#ffffff")

    def _axes_style(self, ax, fg, bg):
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg, labelsize=8)
        for s in ax.spines.values():
            s.set_color(fg)
        ax.grid(True, alpha=0.2, color=fg)

    def draw_profiles(self):
        fig = self.prof_fig
        fig.clear()
        fg, bg = self._style(fig)
        ps = self._selected_passes()
        if not ps:
            fig.text(0.5, 0.5, "Select one or more fixes in the table\n(ctrl-click to compare passes)",
                     ha="center", va="center", color=fg)
            self.prof_canvas.draw_idle()
            return
        ax = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax)
        use_p = any(np.isfinite(p.pressure).any() for p in ps)
        for k, p in enumerate(ps):
            c = PASS_COLOURS[k % len(PASS_COLOURS)]
            order = np.argsort(p.signed_radius)
            x = p.signed_radius[order]
            ax.plot(x, p.wind[order], color=c, lw=1.6, label=p.label)
            if self.chk_sfc.isChecked() and self.bundle is not None and self.bundle.overlay is not None:
                est = surface.overlay_surface(self.bundle.overlay)
                if est.size:
                    ax.plot(x, est[np.asarray(p.idx)][order], color=c, lw=1.0, ls="--", alpha=0.9,
                            label="surface estimate" if k == 0 else None)
            for L in (p.inbound, p.outbound):
                if L and L.vmax is not None:
                    sx = -L.rmw if L is p.inbound else L.rmw
                    ax.plot(sx, L.vmax, "o", color=c, mec=bg, ms=7)
                    # stagger labels by pass so compared passes do not overprint
                    ax.annotate(f"{L.vmax:.0f} kt\nRMW {L.rmw:.0f} km", (sx, L.vmax), xytext=(0, 8 + 22 * k),
                                textcoords="offset points", ha="center", fontsize=7, color=c)
            y2 = p.pressure[order] if use_p else p.height[order]
            ax2.plot(x, y2, color=c, lw=1.4)
        for a in (ax, ax2):
            self._axes_style(a, fg, bg)
            a.axvline(0, color=fg, lw=0.8, alpha=0.5)
        ax.set_ylabel("Flight-level wind (kt)" + ("\ndashed: surface estimate" if self.chk_sfc.isChecked() else ""),
                      color=fg)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.08 * (hi - lo) + 10 * len(ps))      # room for the staggered RMW labels
        ax2.set_ylabel("Extrapolated SLP (mb)" if use_p else "Geopotential height (m)", color=fg)
        ax2.set_xlabel("Distance from centre (km)   \u2190 inbound  |  outbound \u2192", color=fg)
        ax.set_title("Radial profiles", color=fg, fontsize=10)
        leg = ax.legend(fontsize=7, loc="upper right", facecolor=bg, edgecolor=fg)
        for t in leg.get_texts():
            t.set_color(fg)
        fig.tight_layout()
        self.prof_canvas.draw_idle()

    def draw_timeline(self):
        fig = self.time_fig
        fig.clear()
        fg, bg = self._style(fig)
        b = self.bundle
        if b is None or not (b.batches or b.drops or b.fixes):
            fig.text(0.5, 0.5, "No missions loaded", ha="center", va="center", color=fg)
            self.time_canvas.draw_idle()
            return
        ax = fig.add_subplot(1, 1, 1)
        self._axes_style(ax, fg, bg)
        rows: List[str] = []

        def row(aircraft, mission=""):
            # One row per flight: an aircraft flies several missions a day.
            key = f"{aircraft} {mission}".strip() or "?"
            if key not in rows:
                rows.append(key)
            return rows.index(key)
        import matplotlib.dates as mdates
        for bt in b.batches:
            y = row(bt.aircraft, bt.mission)
            ax.barh(y, mdates.date2num(bt.end) - mdates.date2num(bt.start), left=mdates.date2num(bt.start),
                    height=0.36, color=_wind_rgb(bt.max_wind), edgecolor=bg, linewidth=0.4)
        for k, d in enumerate(sorted((d for d in b.drops if d.release_time), key=lambda d: d.release_time)):
            y = row(d.aircraft or "?", d.mission)
            ax.plot(mdates.date2num(d.release_time), y + 0.33, "v", color="#ffffff", ms=6, mec="#000000")
            ax.text(mdates.date2num(d.release_time), y + (0.47 if k % 2 == 0 else 0.62), str(d.number),
                    fontsize=6.5, color=fg, ha="center", va="top")
        for f in b.fixes:
            mission = f.mission
            y = row(f.aircraft or "?", mission)
            ax.plot(mdates.date2num(f.time), y - 0.33, "D" if f.source == "VDM" else "X",
                    color="#ffca28" if f.source == "VDM" else "#80deea", mec="#000000", ms=6)
            if f.min_slp:
                ax.text(mdates.date2num(f.time), y - 0.58, f"{f.min_slp:.0f}", fontsize=6.5,
                        color="#ffca28", ha="center", va="top")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, color=fg)
        ax.set_ylim(len(rows) - 0.25, -0.9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%H:%MZ"))
        ax.set_title("HDOB batches (coloured by peak wind), dropsondes \u25bc, fixes \u25c6", color=fg, fontsize=10)
        fig.autofmt_xdate()
        fig.tight_layout()
        self.time_canvas.draw_idle()

    def set_theme(self, theme: str):
        self.theme = theme
        self.draw_profiles()
        self.draw_timeline()


def _qcolour(hex_colour: str):
    from PyQt6.QtGui import QColor
    return QColor(hex_colour)
