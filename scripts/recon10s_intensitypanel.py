#!/usr/bin/env python3
"""
recon10s_intensitypanel — Reconnaissance › Intensity.

Timeline: minimum pressure and maximum wind from every source on one time
axis -- vortex messages, centre dropsondes (raw and adjusted), each pass's
flight-level and dropsonde surface-wind estimates, NHC's working best track
and current advisory, CIMSS ADT, and recon10s's own satellite estimates
(Experimental › Satellite intensity).

Verification: the satellite estimates against the aircraft (or against
NHC's best track), matched in time, with bias, MAE and RMSE and every
matched pair listed.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Tuple

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QRadioButton,
    QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

import recon10s_intensity as intensity
import recon10s_nhc as nhc
import recon10s_verify as verify

# source -> (colour, marker, line style, size)
STYLES = {
    "Vortex message": ("#ffca28", "D", "none", 6),
    "Centre drop (splash)": ("#b0bec5", "v", "none", 5),
    "Centre drop − 1 mb/10 kt": ("#ffffff", "o", "none", 4),
    "Flight level × standard ratio": ("#ff9800", "^", "none", 6),
    "Eyewall drop WL150 × 0.83": ("#ffffff", "v", "none", 5),
    "NHC best track": ("#e0e0e0", "o", "-", 4),
    "NHC advisory": ("#ff4081", "*", "none", 11),
    "JTWC best track": ("#e0e0e0", "o", "-", 4),
    "JTWC advisory": ("#ff4081", "*", "none", 11),
    "CIMSS ADT": ("#4dd0e1", "", "-", 0),
    "CIMSS D-PRINT": ("#90a4ae", "", "--", 0),
    "CIMSS D-MINT": ("#aed581", ".", "-", 3),
    "recon10s satellite": ("#ba68c8", "", "-", 0),
}
CIMSS_ML = ("D-PRINT", "D-MINT")          # CIMSS's deep-learning estimates, fetched like ADT
SATELLITE = "recon10s satellite"
ESTIMATOR_COLOURS = ["#ba68c8", "#81c784", "#64b5f6", "#f06292"]


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


class IntensityPanel(QWidget):
    track_loaded = pyqtSignal(object)          # list of nhc.TrackPoint, shared with the Wind field tab
    cimss_loaded = pyqtSignal(object)          # {"CIMSS ADT": [...], "CIMSS D-PRINT": [...], ...} for Satellite intensity

    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self.theme = theme
        self.pool = QThreadPool.globalInstance()
        self.bundle = None
        self.track: List[nhc.TrackPoint] = []
        self.adt: List[verify.Estimate] = []
        self.adt_url = ""
        self.adt_text = ""
        # CIMSS D-PRINT and D-MINT: product -> (estimates, url, raw text)
        self.cimss_ml: Dict[str, Tuple[List[verify.Estimate], str, str]] = {}
        self.sat_estimates: List[verify.Estimate] = []      # from Experimental › Satellite intensity
        self.timeline = intensity.Timeline()
        self._loaded_for = ""
        self._auto_sid = ""                  # the storm ID last filled in from Imagery › Satellite
        self.until = None                    # a case replay's clock
        self.pairs: Dict[str, List[verify.Pair]] = {}

        self.tl_fig = Figure(figsize=(8, 7), dpi=100, layout="constrained")
        self.tl_canvas = FigureCanvasQTAgg(self.tl_fig)
        self.vf_fig = Figure(figsize=(8, 7), dpi=100, layout="constrained")
        self.vf_canvas = FigureCanvasQTAgg(self.vf_fig)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.addWidget(self._sources_box())
        lv.addWidget(self._satellite_box())
        lv.addWidget(self._verify_box())
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(480)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.tabs = QTabWidget()
        tw = QWidget()
        tv = QVBoxLayout(tw)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.addWidget(NavigationToolbar2QT(self.tl_canvas, tw))
        tv.addWidget(self.tl_canvas, 1)
        self.tabs.addTab(tw, "Timeline")
        vw = QWidget()
        vv = QVBoxLayout(vw)
        vv.setContentsMargins(0, 0, 0, 0)
        vv.addWidget(NavigationToolbar2QT(self.vf_canvas, vw))
        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.addWidget(self.vf_canvas)
        self.pair_table = QTableWidget(0, 8)
        self.pair_table.setHorizontalHeaderLabels(["Estimator", "Quantity", "Truth time", "Truth", "Estimate",
                                                   "Estimate − truth", "Matched by", "Truth from"])
        self.pair_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pair_table.verticalHeader().setVisible(False)
        self.pair_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.pair_table.horizontalHeader().setStretchLastSection(True)
        vsplit.addWidget(self.pair_table)
        vsplit.setSizes([620, 260])
        vv.addWidget(vsplit, 1)
        self.tabs.addTab(vw, "Verification")
        self.tabs.currentChanged.connect(lambda _: self.redraw())

        self.status = QLabel("Load recon in Imagery › Satellite to begin; NHC's best track and CIMSS ADT, "
                             "D-PRINT and D-MINT are fetched for the storm.")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.tabs, 1)
        rv.addWidget(self.status)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([370, 1200])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split, 1)
        self.rebuild()

    # ---------------------------------------------------------------- building

    def _sources_box(self):
        box = QGroupBox("Storm && sources")
        f = QFormLayout(box)
        row = QHBoxLayout()
        self.storm_id = QLineEdit()
        self.storm_id.setPlaceholderText("al092026 — filled from Satellite")
        row.addWidget(self.storm_id, 1)
        self.btn_load = QPushButton("Load")
        self.btn_load.setToolTip("NHC's working best track (b-deck) and CIMSS's ADT, D-PRINT and D-MINT\n"
                                 "histories for this storm.")
        self.btn_load.clicked.connect(lambda: self.load_reference(force=True))
        row.addWidget(self.btn_load)
        f.addRow("Storm ID", row)
        self.ref_note = QLabel("")
        self.ref_note.setObjectName("hint")
        self.ref_note.setWordWrap(True)
        f.addRow(self.ref_note)
        self.series_grid = QGridLayout()
        self.series_checks: Dict[str, QCheckBox] = {}
        f.addRow(self.series_grid)
        self.span_box = QComboBox()
        self.span_box.addItem("Recon period (\u00b112 h)", "recon")
        self.span_box.addItem("Whole storm", "all")
        self.span_box.currentIndexChanged.connect(lambda _: self.redraw())
        f.addRow("Show", self.span_box)
        return box

    def _satellite_box(self):
        box = QGroupBox("Satellite estimates")
        f = QVBoxLayout(box)
        self.sat_note = QLabel("CIMSS ADT, D-PRINT and D-MINT are fetched with the best track. recon10s's own "
                               "estimates (IR and microwave, trained on TC PRIMED) come from Experimental › "
                               "Satellite intensity and appear here as “recon10s satellite”.")
        self.sat_note.setObjectName("hint")
        self.sat_note.setWordWrap(True)
        f.addWidget(self.sat_note)
        return box

    def _verify_box(self):
        box = QGroupBox("Verification")
        f = QFormLayout(box)
        self.truth_group = QButtonGroup(self)
        row = QHBoxLayout()
        for label, key in (("Recon", "recon"), ("Best track", "track")):
            rb = QRadioButton(label)
            rb.setProperty("key", key)
            rb.setChecked(key == "recon")
            self.truth_group.addButton(rb)
            row.addWidget(rb)
        self.truth_group.buttonToggled.connect(lambda _b, on: on and self.rebuild())
        f.addRow("Against", row)
        self.max_gap = QSpinBox()
        self.max_gap.setRange(5, 360)
        self.max_gap.setValue(60)
        self.max_gap.setSuffix(" min")
        self.max_gap.setToolTip("Nearest estimate allowed when none bracket the truth time.")
        self.interp_h = QDoubleSpinBox()
        self.interp_h.setRange(0.0, 12.0)
        self.interp_h.setValue(3.0)
        self.interp_h.setSuffix(" h")
        self.interp_h.setToolTip("Interpolate between two estimates this close together that bracket the truth time.")
        for w in (self.max_gap, self.interp_h):
            w.valueChanged.connect(lambda _: self.rebuild())
        f.addRow("Nearest within", self.max_gap)
        f.addRow("Interpolate across", self.interp_h)
        self.scores_label = QLabel("")
        self.scores_label.setObjectName("probeReadout")
        self.scores_label.setWordWrap(True)
        self.scores_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        f.addRow(self.scores_label)
        note = QLabel("Recon truth: each vortex message's minimum pressure, and each pass's surface wind "
                      "(peak flight-level wind × the standard ratio, or an eyewall drop's WL150 × 0.83 "
                      "if higher). The aircraft rarely samples the very strongest wind, so estimates a few "
                      "knots above it are expected. No SFMR.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        f.addRow(note)
        return box

    # ------------------------------------------------------------------ data

    def set_bundle(self, bundle):
        self.bundle = bundle
        fix = getattr(bundle, "fix", None) if bundle is not None else None
        sid = getattr(fix, "storm_id", "") if fix else ""
        # follow the storm chosen in Imagery › Satellite, unless a different one was typed here
        typed = self.storm_id.text().strip().lower()
        if sid and (not typed or typed == self._auto_sid):
            self.storm_id.setText(sid)
            self._auto_sid = sid
        if sid and sid != self._loaded_for and self.storm_id.text().strip().lower() == sid:
            self.load_reference()
        self.rebuild()

    def storm(self) -> str:
        return self.storm_id.text().strip().lower()

    def load_reference(self, force: bool = False):
        sid = self.storm()
        if not sid:
            self.ref_note.setText("Enter a storm ID (e.g. al092026 or wp292026) for the best track and CIMSS's "
                                  "estimates.")
            return
        if sid == self._loaded_for and not force:
            return
        self._loaded_for = sid
        self.btn_load.setEnabled(False)
        self.ref_note.setText(f"Fetching {nhc.agency(sid)}'s best track and CIMSS ADT, D-PRINT and D-MINT for "
                              f"{sid.upper()}…")

        def work(sid):
            out = {"sid": sid}
            try:
                out["track"] = nhc.best_track(sid)
            except Exception as exc:
                out["track_error"] = str(exc)
            try:
                out["adt"] = verify.fetch_adt(sid)
            except Exception as exc:
                out["adt_error"] = str(exc)
            out["ml"] = {}
            for product in CIMSS_ML:
                try:
                    out["ml"][product] = verify.fetch_cimss_ml(sid, product)
                except Exception:
                    out["ml"][product] = ([], "", "")
            return out
        job = _Job(work, sid)
        job.sig.done.connect(self._on_reference)
        job.sig.failed.connect(lambda m: (self.btn_load.setEnabled(True), self.ref_note.setText(m)))
        self.pool.start(job)

    def _on_reference(self, res):
        self.btn_load.setEnabled(True)
        if res.get("sid") != self.storm():
            return
        self.set_reference(res.get("track") or [], *(res.get("adt") or ([], "", "")), ml=res.get("ml") or {})
        bits = []
        ag = nhc.agency(res.get("sid", ""))
        bits.append(f"{ag} best track: {len(self.track)} times" if self.track
                    else f"{ag} best track: {res.get('track_error') or 'none'}")
        bits.append(f"ADT: {len(self.adt)} estimates ({self.adt_url.rsplit('/', 1)[-1]})" if self.adt
                    else f"ADT: {res.get('adt_error') or 'no CIMSS listing for this storm'}")
        for product in CIMSS_ML:
            n = len(self.cimss_ml.get(product, ([],))[0])
            bits.append(f"{product}: {n} estimates" if n else f"{product}: none for this storm")
        self.ref_note.setText("  ·  ".join(bits))

    def set_reference(self, track, adt=(), adt_url: str = "", adt_text: str = "", ml=None):
        """Best track, ADT and D-PRINT/D-MINT ({product: (estimates, url, text)}) handed in directly
        (a fetch, or a saved case)."""
        self.track = list(track or [])
        self.adt = list(adt or [])
        self.adt_url, self.adt_text = adt_url, adt_text
        if ml is not None:
            self.cimss_ml = {k: (list(v[0] or []), v[1], v[2]) for k, v in ml.items() if v and v[0]}
        ref = {"CIMSS ADT": self.adt} if self.adt else {}
        ref.update({"CIMSS " + k: v[0] for k, v in self.cimss_ml.items()})
        self.cimss_loaded.emit(ref)
        if self.track:
            self.track_loaded.emit(self.track)
        self.rebuild()

    def set_satellite_estimates(self, estimates):
        """recon10s's own satellite estimates (recon10s_satint.Estimate), newest last."""
        self.sat_estimates = [verify.Estimate(time=e.time, vmax=e.vmax, mslp=e.mslp, lat=e.centre[0], lon=e.centre[1],
                                              note=("IR+MW" if e.mw_time else "IR"))
                              for e in (estimates or [])]
        n = len(self.sat_estimates)
        self.sat_note.setText((f"{n} recon10s satellite estimate(s), {self.sat_estimates[0].time:%d %b %H:%MZ}–"
                               f"{self.sat_estimates[-1].time:%d %b %H:%MZ}, from Experimental › Satellite "
                               "intensity.") if n else
                              "CIMSS ADT, D-PRINT and D-MINT are fetched with the best track. recon10s's own "
                              "estimates come from Experimental › Satellite intensity.")
        self.rebuild()

    def set_until(self, when):
        """Ignore estimates and the advisory after `when` (a case replay); None for all."""
        self.until = when

    def estimator_series(self) -> Dict[str, List[verify.Estimate]]:
        out = {}
        until = self.until
        if self.adt:
            out["CIMSS ADT"] = [e for e in self.adt if until is None or e.time <= until]
        for product in CIMSS_ML:
            ests = [e for e in self.cimss_ml.get(product, ([],))[0] if until is None or e.time <= until]
            if ests:
                out["CIMSS " + product] = ests
        sat = [e for e in self.sat_estimates if until is None or e.time <= until]
        if sat:
            out[SATELLITE] = sat
        return out

    def rebuild(self):
        extra = {"pressure": {}, "wind": {}}
        for name, ests in self.estimator_series().items():
            extra["pressure"][name] = verify.as_points(ests, "mslp")
            extra["wind"][name] = verify.as_points(ests, "vmax")
        adv = getattr(self.bundle, "fix", None) if self.bundle is not None else None
        if adv is not None and (not getattr(adv, "storm_id", "") or adv.storm_id != self.storm()):
            adv = None
        if adv is not None and self.until is not None and (adv.time is None or adv.time > self.until):
            adv = None                       # replaying: that advisory had not been issued yet
        self.timeline = intensity.build_timeline(self.bundle, self.track, adv, extra, agency=nhc.agency(self.storm()))
        self._sync_series_checks()
        self._verify()
        self.redraw()
        low = intensity.lowest_pressure(self.timeline)
        n_v = len(self.timeline.pressure.get("Vortex message", []))
        n_bt = len(self.track)
        bits = [f"{n_v} vortex message(s)", f"{n_bt} best-track times"]
        if low:
            bits.append(f"lowest recon pressure {low[0].value:.0f} mb at {low[0].time:%d/%H:%MZ} ({low[1].lower()})")
        if self.bundle is not None or self.track:
            self.status.setText("  \u00b7  ".join(bits) + ".")

    def _sync_series_checks(self):
        names = []
        for d in (self.timeline.pressure, self.timeline.wind):
            for s, pts in d.items():
                if pts and s not in names:
                    names.append(s)
        for s in list(self.series_checks):
            if s not in names:
                c = self.series_checks.pop(s)
                self.series_grid.removeWidget(c)
                c.setParent(None)
                c.deleteLater()
        for s in names:
            if s not in self.series_checks:
                c = QCheckBox(s)
                c.setChecked(s != "Centre drop (splash)")
                c.setStyleSheet(f"color: {self._colour(s)};")
                c.toggled.connect(lambda _: self.redraw())
                self.series_checks[s] = c
        for k, s in enumerate(names):
            self.series_grid.addWidget(self.series_checks[s], k // 2, k % 2)

    def _colour(self, source: str) -> str:
        if source in STYLES:
            return STYLES[source][0]
        names = [n for n in self.estimator_series() if n not in STYLES]
        k = names.index(source) if source in names else 0
        return ESTIMATOR_COLOURS[k % len(ESTIMATOR_COLOURS)]

    def _truth(self):
        b = self.truth_group.checkedButton()
        if b is not None and b.property("key") == "track":
            p, w = verify.best_track_truth(self.track)
            return p, w, f"{nhc.agency(self.storm())} best track"
        p, w = verify.recon_truth(self.bundle)
        return p, w, "recon"

    def _verify(self):
        p_truth, w_truth, label = self._truth()
        self.pairs = {}
        lines = []
        for name, ests in self.estimator_series().items():
            pp = verify.match(ests, p_truth, "mslp", self.max_gap.value(), self.interp_h.value())
            wp = verify.match(ests, w_truth, "vmax", self.max_gap.value(), self.interp_h.value())
            self.pairs[name] = [("MSLP", p) for p in pp] + [("Vmax", p) for p in wp]
            lines.append(f"{name} vs {label}\n  MSLP (mb): {verify.scores(pp).text}\n"
                         f"  Vmax (kt): {verify.scores(wp).text}")
        if not lines:
            lines = ["No satellite estimates yet: load CIMSS's (Storm ID › Load) or run Experimental › "
                     "Satellite intensity."]
        elif not (p_truth or w_truth):
            lines.append(f"No {label} truth loaded yet.")
        self.scores_label.setText("\n".join(lines))
        self._fill_pairs()

    def _fill_pairs(self):
        t = self.pair_table
        t.setRowCount(0)
        for name, pairs in self.pairs.items():
            for qty, p in pairs:
                r = t.rowCount()
                t.insertRow(r)
                unit = "mb" if qty == "MSLP" else "kt"
                for c, text in enumerate([name, qty, f"{p.time:%d %b %H:%MZ}", f"{p.truth:.0f} {unit}",
                                          f"{p.estimate:.1f} {unit}", f"{p.diff:+.1f}", p.method, p.label]):
                    t.setItem(r, c, QTableWidgetItem(text))

    # --------------------------------------------------------------- drawing

    def _colours(self):
        dark = self.theme != "light"
        return ("#dfe5ec" if dark else "#1c2129"), ("#12151b" if dark else "#ffffff")

    def _axes(self, ax, fg):
        ax.set_facecolor("#0b0d11" if self.theme != "light" else "#f4f6f8")
        ax.tick_params(colors=fg, labelsize=8)
        for s in ax.spines.values():
            s.set_color("#5c6675")
        ax.grid(True, color=fg, alpha=0.12)

    def _legend(self, ax, fg, bg, ncol=2):
        h, l = ax.get_legend_handles_labels()
        if h:
            leg = ax.legend(fontsize=7, loc="best", facecolor=bg, edgecolor="#5c6675", ncol=ncol, framealpha=0.85)
            for t in leg.get_texts():
                t.set_color(fg)

    def redraw(self):
        if self.tabs.currentIndex() == 1:
            self.draw_verification()
        else:
            self.draw_timeline()

    def _style_of(self, source):
        colour = self._colour(source)
        if source in STYLES:
            _c, marker, ls, ms = STYLES[source]
        else:
            marker, ls, ms = "", "-", 0
        if ls == "-" and self.theme == "light" and colour in ("#e0e0e0", "#ffffff"):
            colour = "#333333"
        if ls == "none" and self.theme == "light" and colour in ("#ffffff", "#b0bec5"):
            colour = "#555555"
        return colour, marker, ls, ms

    def draw_timeline(self):
        import matplotlib.dates as mdates
        fig = self.tl_fig
        fig.clear()
        fg, bg = self._colours()
        fig.patch.set_facecolor(bg)
        tl = self.timeline
        if tl.is_empty():
            fig.text(0.5, 0.5, "No intensity data yet — load recon or NHC's best track, or make satellite estimates",
                     ha="center", va="center", color="#8b95a5")
            self.tl_canvas.draw_idle()
            return
        axp = fig.add_subplot(2, 1, 1)
        axw = fig.add_subplot(2, 1, 2, sharex=axp)
        for ax, series, unit in ((axp, tl.pressure, "mb"), (axw, tl.wind, "kt")):
            self._axes(ax, fg)
            for source, pts in series.items():
                c = self.series_checks.get(source)
                if not pts or (c is not None and not c.isChecked()):
                    continue
                colour, marker, ls, ms = self._style_of(source)
                x = [mdates.date2num(p.time) for p in pts]
                y = [p.value for p in pts]
                ax.plot(x, y, linestyle=ls, marker=marker or None, ms=ms, color=colour,
                        lw=1.3 if ls != "none" else 0, mec="#000000" if marker in ("D", "^", "*") else colour,
                        mew=0.6, label=f"{source} ({len(pts)})", zorder=4 if ls == "none" else 3)
            self._legend(ax, fg, bg)
        low = intensity.lowest_pressure(tl)
        if low:
            p, src = low
            axp.annotate(f"lowest {p.value:.0f} mb {p.time:%d/%H:%MZ}", (mdates.date2num(p.time), p.value),
                         xytext=(-30, 22), textcoords="offset points", ha="right", fontsize=7, color="#ffca28",
                         va="bottom", arrowprops=dict(arrowstyle="-", color="#ffca28", lw=0.6))
        span = self._span()
        if span is not None:
            axw.set_xlim(mdates.date2num(span[0]), mdates.date2num(span[1]))
            for ax, series in ((axp, tl.pressure), (axw, tl.wind)):
                vals = [p.value for s_, pts in series.items() for p in pts
                        if span[0] <= p.time <= span[1] and (self.series_checks.get(s_) is None
                                                             or self.series_checks[s_].isChecked())]
                if vals:
                    pad = max(2.0, 0.08 * (max(vals) - min(vals)))
                    ax.set_ylim(min(vals) - pad, max(vals) + pad)
        axp.tick_params(labelbottom=False)
        axp.set_ylabel("Minimum pressure (mb)", color=fg, fontsize=9)
        axw.set_ylabel("Maximum surface wind (kt)", color=fg, fontsize=9)
        axw.xaxis.set_major_formatter(mdates.DateFormatter("%d/%H:%MZ"))
        for lab in axw.get_xticklabels():
            lab.set_rotation(20)
            lab.set_ha("right")
        name = getattr(getattr(self.bundle, "fix", None), "name", "") if self.bundle is not None else ""
        axp.set_title(f"{name or self.storm().upper()} intensity: recon, {nhc.agency(self.storm())} and satellite "
                      "estimates".strip(),
                      color=fg, fontsize=10)
        self.tl_canvas.draw_idle()

    def _span(self):
        """x-limits: the recon period padded by 12 h, unless the whole storm is asked for."""
        if self.span_box.currentData() == "all" or self.bundle is None:
            return None
        recon = [p.time for s in ("Vortex message", "Flight level \u00d7 standard ratio", "Centre drop (splash)")
                 for p in self.timeline.pressure.get(s, []) + self.timeline.wind.get(s, [])]
        if not recon:
            return None
        pad = dt.timedelta(hours=12)
        return min(recon) - pad, max(recon) + pad

    def draw_verification(self):
        import matplotlib.dates as mdates
        fig = self.vf_fig
        fig.clear()
        fg, bg = self._colours()
        fig.patch.set_facecolor(bg)
        if not any(self.pairs.values()):
            fig.text(0.5, 0.5, "Nothing to verify yet: needs satellite estimates (CIMSS's or recon10s's own)\n"
                     "and recon (or NHC's best track) over the same hours", ha="center", va="center",
                     color="#8b95a5")
            self.vf_canvas.draw_idle()
            return
        gs = fig.add_gridspec(2, 2)
        axes = {"MSLP": fig.add_subplot(gs[0, 0]), "Vmax": fig.add_subplot(gs[0, 1])}
        axd = fig.add_subplot(gs[1, :])
        for ax in list(axes.values()) + [axd]:
            self._axes(ax, fg)
        for name, pairs in self.pairs.items():
            colour = self._colour(name)
            for qty, ax in axes.items():
                pts = [p for q, p in pairs if q == qty]
                if not pts:
                    continue
                s = verify.scores(pts)
                ax.scatter([p.truth for p in pts], [p.estimate for p in pts], s=22, color=colour,
                           edgecolors="#000000", linewidths=0.4, label=f"{name}: bias {s.bias:+.1f}, RMSE {s.rmse:.1f}",
                           zorder=3)
            for qty, marker in (("MSLP", "s"), ("Vmax", "o")):
                pts = [p for q, p in pairs if q == qty]
                if pts:
                    axd.plot([mdates.date2num(p.time) for p in pts], [p.diff for p in pts], marker=marker, ms=4,
                             color=colour, lw=0.8, label=f"{name} {qty} ({'mb' if qty == 'MSLP' else 'kt'})")
        for qty, ax in axes.items():
            lo, hi = ax.get_xlim()
            lo2, hi2 = ax.get_ylim()
            a, b = min(lo, lo2), max(hi, hi2)
            ax.plot([a, b], [a, b], color=fg, lw=0.8, alpha=0.6, ls="--")
            ax.set_xlim(a, b)
            ax.set_ylim(a, b)
            ax.set_aspect("equal", adjustable="box")
            unit = "mb" if qty == "MSLP" else "kt"
            truth = self.truth_group.checkedButton().text()
            ax.set_xlabel(f"{truth} {qty} ({unit})", color=fg, fontsize=8)
            ax.set_ylabel(f"Estimate ({unit})", color=fg, fontsize=8)
            ax.set_title(f"{qty}: estimate against {truth.lower()}", color=fg, fontsize=9)
            self._legend(ax, fg, bg)
        axd.axhline(0, color=fg, lw=0.8, alpha=0.6)
        axd.set_ylabel("Estimate − truth", color=fg, fontsize=8)
        axd.xaxis.set_major_formatter(mdates.DateFormatter("%d/%H:%MZ"))
        self._legend(axd, fg, bg)
        self.vf_canvas.draw_idle()

    def set_theme(self, theme: str):
        self.theme = theme
        self.redraw()

    def state(self) -> dict:
        b = self.truth_group.checkedButton()
        return {"truth": b.property("key") if b else "recon", "max_gap": self.max_gap.value(),
                "interp": self.interp_h.value(),
                "series": {s: c.isChecked() for s, c in self.series_checks.items()}}

    def restore(self, st: dict):
        if not st:
            return
        for b in self.truth_group.buttons():
            b.setChecked(b.property("key") == st.get("truth", "recon"))
        self.max_gap.setValue(int(st.get("max_gap", 60)))
        self.interp_h.setValue(float(st.get("interp", 3.0)))
