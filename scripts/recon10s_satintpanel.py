#!/usr/bin/env python3
"""
recon10s_satintpanel — Experimental › Satellite intensity.

recon10s's own satellite intensity estimates (recon10s_satint): maximum wind
and central pressure, with their 25-75 % and 10-90 % ranges, from GOES or
Himawari-9 IR over the last 12 hours and the most recent microwave pass.
Estimates are made for the latest scan or any past time, and back through the
hours before it; they are drawn against the best track and advisory (NHC's,
or JTWC's west of the dateline and south of the equator), and handed to
Reconnaissance › Intensity for verification against the recon.
"""
from __future__ import annotations

import datetime as dt
import warnings
from typing import Callable, List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import QDateTime, QObject, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDateTimeEdit, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

import recon10s_nhc as nhc
import recon10s_satint as si

BAND_COLOUR = "#ba68c8"
UTC = dt.timezone.utc


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


class SatIntensityPanel(QWidget):
    estimates_changed = pyqtSignal(object)        # List[si.Estimate], for Reconnaissance › Intensity

    def __init__(self, parent=None, theme: str = "dark", email_provider: Optional[Callable[[], Optional[str]]] = None):
        super().__init__(parent)
        self.theme = theme
        self.pool = QThreadPool.globalInstance()
        self.email_provider = email_provider or (lambda: None)
        self.fix = None
        self.bundle = None
        self.track: List[nhc.TrackPoint] = []
        self._track_for = ""
        self.estimates: List[si.Estimate] = []
        self.reference = {}                   # CIMSS ADT, D-PRINT, D-MINT (recon10s_verify.Estimate) from Intensity
        self.last_inputs = None              # (IR grid, MW grid or None, Estimate) of the newest estimate
        self.until: Optional[dt.datetime] = None
        self._busy = False
        self._est = None                      # kept between runs for the same storm: its IR frames are reused
        self._est_key = None
        self.model_error = ""
        try:
            self.model = si.net()
        except Exception as exc:
            self.model = None
            self.model_error = f"{type(exc).__name__}: {exc}"

        self.figure = Figure(figsize=(8, 7), dpi=100, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.addWidget(self._storm_box())
        lv.addWidget(self._run_box())
        lv.addWidget(self._result_box())
        lv.addWidget(self._about_box())
        lv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(460)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(NavigationToolbar2QT(self.canvas, right))
        rv.addWidget(self.canvas, 1)
        self.status = QLabel("Pick a storm under Imagery › Satellite, then Estimate.")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        rv.addWidget(self.status)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1200])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split, 1)

        self.auto_timer = QTimer(self)
        self.auto_timer.setInterval(10 * 60 * 1000)
        self.auto_timer.timeout.connect(self._auto)
        if self.model is None:
            self.btn_run.setEnabled(False)
            self.status.setText(f"The satellite intensity model could not load ({self.model_error}); it ships as "
                                "scripts/models/satint_v1.npz.")
        self.redraw()

    # ---------------------------------------------------------------- building

    def _storm_box(self):
        box = QGroupBox("Storm")
        f = QFormLayout(box)
        self.storm_label = QLabel("None chosen (Imagery › Satellite).")
        self.storm_label.setWordWrap(True)
        f.addRow(self.storm_label)
        self.centre_note = QLabel("")
        self.centre_note.setObjectName("hint")
        self.centre_note.setWordWrap(True)
        f.addRow(self.centre_note)
        return box

    def _run_box(self):
        box = QGroupBox("Estimate")
        f = QFormLayout(box)
        self.when_group = QButtonGroup(self)
        row = QHBoxLayout()
        self.rb_now = QRadioButton("Latest scan")
        self.rb_at = QRadioButton("At")
        self.rb_now.setChecked(True)
        for b in (self.rb_now, self.rb_at):
            self.when_group.addButton(b)
            row.addWidget(b)
        f.addRow("Valid", row)
        self.when_edit = QDateTimeEdit()
        self.when_edit.setDisplayFormat("yyyy-MM-dd HH:mm 'UTC'")
        self.when_edit.setTimeSpec(Qt.TimeSpec.UTC)
        self.when_edit.setDateTime(QDateTime.currentDateTimeUtc())
        self.when_edit.setEnabled(False)
        self.rb_at.toggled.connect(self.when_edit.setEnabled)
        f.addRow("", self.when_edit)
        hrow = QHBoxLayout()
        self.hours = QDoubleSpinBox()
        self.hours.setRange(0.0, 48.0)
        self.hours.setSingleStep(3.0)
        self.hours.setValue(12.0)
        self.hours.setSuffix(" h")
        self.hours.setToolTip("Also estimate back through this many hours before the valid time.")
        self.step = QComboBox()
        for label, h in (("every 30 min", 0.5), ("hourly", 1.0), ("every 3 h", 3.0)):
            self.step.addItem(label, h)
        self.step.setCurrentIndex(1)
        hrow.addWidget(self.hours)
        hrow.addWidget(self.step)
        f.addRow("History", hrow)
        self.chk_mw = QCheckBox("Use the latest microwave pass")
        self.chk_mw.setChecked(True)
        self.chk_mw.setToolTip("GMI, AMSR3 or WSF-M from NASA PPS's near-real-time server, with the sign-in\n"
                               "set up under Imagery › Microwave. Without it the estimates are IR only.")
        f.addRow(self.chk_mw)
        self.mw_note = QLabel("")
        self.mw_note.setObjectName("hint")
        self.mw_note.setWordWrap(True)
        f.addRow(self.mw_note)
        self.btn_run = QPushButton("Estimate")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self.run)
        f.addRow(self.btn_run)
        self.chk_ref = QCheckBox("Draw CIMSS ADT, D-PRINT and D-MINT beside it")
        self.chk_ref.setChecked(True)
        self.chk_ref.setToolTip("CIMSS's estimates for the storm, as fetched by Reconnaissance › Intensity.")
        self.chk_ref.toggled.connect(lambda _: self.redraw())
        f.addRow(self.chk_ref)
        self.chk_auto = QCheckBox("Add an estimate every 30 minutes")
        self.chk_auto.setToolTip("While this is on, each new half hour of full-disk scans adds an estimate.")
        self.chk_auto.toggled.connect(lambda on: self.auto_timer.start() if on else self.auto_timer.stop())
        f.addRow(self.chk_auto)
        return box

    def _result_box(self):
        box = QGroupBox("Latest estimate")
        f = QVBoxLayout(box)
        self.result = QLabel("—")
        self.result.setWordWrap(True)
        self.result.setTextFormat(Qt.TextFormat.RichText)
        self.result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        f.addWidget(self.result)
        self.btn_copy = QPushButton("Copy all estimates (CSV)")
        self.btn_copy.setToolTip("Every estimate shown, with its ranges, the pass it used and the IR-only value,\n"
                                 "to the clipboard as comma-separated text.")
        self.btn_copy.clicked.connect(self.copy_csv)
        f.addWidget(self.btn_copy)
        return box

    def csv_text(self) -> str:
        head = ["time_utc", "vmax_kt_p10", "vmax_kt_p25", "vmax_kt", "vmax_kt_p75", "vmax_kt_p90",
                "mslp_mb_p90", "mslp_mb_p75", "mslp_mb", "mslp_mb_p25", "mslp_mb_p10",
                "ir_only_vmax_kt", "ir_only_mslp_mb", "mw_sensor", "mw_time_utc", "mw_age_h", "satellite",
                "ir_frames", "lat", "lon"]
        rows = [",".join(head)]
        f = lambda x, n=1: "" if x is None else f"{x:.{n}f}"
        for e in self.visible():
            rows.append(",".join([e.time.strftime("%Y-%m-%d %H:%M"), *[f(x) for x in e.vmax_q],
                                  *[f(x) for x in e.mslp_q[::-1]], f(e.ir_vmax), f(e.ir_mslp), e.mw_sensor,
                                  e.mw_time.strftime("%Y-%m-%d %H:%M") if e.mw_time else "", f(e.mw_age_h),
                                  e.satellite, str(e.ir_frames), f(e.centre[0], 2), f(e.centre[1], 2)]))
        return "\n".join(rows) + "\n"

    def copy_csv(self):
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.csv_text())
        self.status.setText(f"Copied {len(self.visible())} estimate(s) to the clipboard.")

    def _about_box(self):
        box = QGroupBox("About")
        f = QVBoxLayout(box)
        info = getattr(self.model, "info", {}) or {}
        val = info.get("validation") or {}
        lines = ["Experimental. A convolutional network trained on TC PRIMED (NOAA): every microwave "
                 "overpass of Atlantic and east/central Pacific storms "
                 f"{info.get('years', [2000, 2025])[0]}–{info.get('years', [2000, 2025])[-1]}, each with the "
                 "geostationary IR at its time and the best track. It sees the IR now and 3–12 h before, "
                 "and the most recent 37/89-GHz pass with its age and the IR at the pass time."]
        if val.get("summary"):
            lines.append(val["summary"])
        lines.append("In JTWC's basins (West Pacific, Indian Ocean, Southern Hemisphere), which it never saw, it runs on "
                     "Himawari-9 IR (GOES-18 east of about 178W), mirrors southern storms into northern ones, and maps "
                     "its estimates onto JTWC's best track; the README has how it scores there.")
        note = QLabel(" ".join(lines))
        note.setObjectName("hint")
        note.setWordWrap(True)
        f.addWidget(note)
        return box

    # ------------------------------------------------------------------ data

    def set_bundle(self, bundle):
        """The storm and any recon centre fixes, from Imagery › Satellite."""
        self.bundle = bundle
        fix = getattr(bundle, "fix", None) if bundle is not None else None
        if fix is not None:
            self.set_storm(fix)

    def set_storm(self, fix):
        changed = self.fix is None or getattr(self.fix, "storm_id", "") != getattr(fix, "storm_id", "")
        self.fix = fix
        sid = getattr(fix, "storm_id", "") or ""
        name = (getattr(fix, "name", "") or sid).title()
        self.storm_label.setText(f"<b>{name}</b> ({sid.upper()}) · {nhc.agency(sid)} {getattr(fix, 'vmax_kt', '?')} kt, "
                                 f"{getattr(fix, 'mslp_mb', '?')} mb at {fix.time:%d %b %H:%MZ}"
                                 if getattr(fix, "time", None) else f"<b>{name}</b> ({sid.upper()})")
        if changed:
            self.estimates = []
            self.last_inputs = None
            self.redraw()
        if sid and sid != self._track_for:
            self._track_for = sid
            job = _Job(lambda _s, s: nhc.best_track(s), sid)
            job.sig.done.connect(self.set_track)
            job.sig.failed.connect(lambda _m: None)
            self.pool.start(job)
        self._mw_note()

    def set_track(self, track):
        """NHC's working best track (fetched here or shared by Reconnaissance › Intensity)."""
        self.track = list(track or [])
        self.redraw()

    def set_reference_estimates(self, ref):
        """CIMSS ADT, D-PRINT and D-MINT for the storm ({name: [recon10s_verify.Estimate]})."""
        self.reference = dict(ref or {})
        self.redraw()

    def set_until(self, when: Optional[dt.datetime]):
        """A case replay's clock: estimates after it are hidden."""
        self.until = when
        self.redraw()
        self.estimates_changed.emit(self.visible())

    def set_estimates(self, estimates):
        """Estimates kept in a saved case."""
        self.estimates = sorted(estimates or [], key=lambda e: e.time)
        self.redraw()
        self._describe()
        self.estimates_changed.emit(self.visible())

    def visible(self) -> List[si.Estimate]:
        return [e for e in self.estimates if self.until is None or e.time <= self.until]

    def _mw_note(self):
        email = self.email_provider()
        self.mw_note.setText("PPS sign-in found: microwave passes will be used." if email else
                             "No PPS sign-in: estimates will be IR only. Sign in under Imagery › Microwave.")

    def source(self) -> si.TrackSource:
        fixes = list(getattr(self.bundle, "fixes", None) or []) if self.bundle is not None else []
        motion = getattr(self.bundle, "motion", None) if self.bundle is not None else None
        return si.TrackSource(track=self.track, advisory=self.fix, fixes=fixes, motion=motion)

    # ------------------------------------------------------------------ run

    def run(self):
        if self.model is None or self._busy:
            return
        if self.fix is None:
            self.status.setText("Pick a storm under Imagery › Satellite first.")
            return
        sid = getattr(self.fix, "storm_id", "")
        if self.rb_at.isChecked():
            end = self.when_edit.dateTime().toPyDateTime().replace(tzinfo=UTC)
        else:
            end = dt.datetime.now(UTC)
        hours, step = self.hours.value(), float(self.step.currentData())
        email = self.email_provider() if self.chk_mw.isChecked() else None
        at_mode = self.rb_at.isChecked()
        self._busy = True
        self.btn_run.setEnabled(False)
        self.status.setText("Estimating…")
        src = self.source()

        key = (sid, email)
        if self._est is None or self._est_key != key:
            self._est, self._est_key = None, key
        cached = self._est

        def work(status, sid, src, email, end, hours, step):
            est = cached if cached is not None else si.Estimator(sid, src, email=email, progress=status)
            est.source, est.progress = src, status
            est.errors = []
            if not at_mode:
                # the newest full-disk scan, not the clock (scans reach S3 a few minutes late)
                end = end - dt.timedelta(minutes=10)
            ests = est.history(end, hours, step) if hours > 0 else [est.estimate(end)]
            inputs = None
            if ests:
                last = ests[-1]
                fr = est.frame(last.time)
                mw = None
                if last.mw_time is not None and est._mw:
                    m = next((p for p in est._mw if p.time == last.mw_time), None)
                    mw = m.grid if m is not None else None
                inputs = (fr[0] if fr else None, mw, last)
            return ests, inputs, est.errors, est

        job = _Job(work, sid, src, email, end, hours, step)
        job.sig.status.connect(self.status.setText)

        def done(res):
            ests, inputs, errors, est = res
            if self._est_key == key:
                self._est = est
            self._busy = False
            self.btn_run.setEnabled(True)
            keep = {e.time: e for e in self.estimates}
            for e in ests:
                keep[e.time] = e
            self.estimates = sorted(keep.values(), key=lambda e: e.time)
            if inputs is not None:
                self.last_inputs = inputs
            self._describe()
            self.redraw()
            self.estimates_changed.emit(self.visible())
            msg = f"{len(ests)} estimate(s)" + (f", {ests[0].time:%d %H:%MZ}–{ests[-1].time:%d %H:%MZ}" if ests else "")
            if errors:
                msg += f". Skipped: {errors[-1]}" + (f" (and {len(errors) - 1} more)" if len(errors) > 1 else "")
            self.status.setText(msg + ".")

        def failed(m):
            self._busy = False
            self.btn_run.setEnabled(True)
            self.status.setText(f"Satellite intensity: {m}")
        job.sig.done.connect(done)
        job.sig.failed.connect(failed)
        self.pool.start(job)

    def _auto(self):
        if self.rb_now.isChecked() and not self._busy and self.fix is not None:
            newest = self.estimates[-1].time if self.estimates else None
            # estimates are valid ~10 min behind the clock (the newest full-disk scan), and the timer
            # looks every 10 min: 35 min since the newest estimate puts them 30 min apart
            if newest is None or dt.datetime.now(UTC) - newest >= dt.timedelta(minutes=35):
                h = self.hours.value()
                self.hours.setValue(0.0)
                try:
                    self.run()
                finally:
                    self.hours.setValue(h)

    def _describe(self):
        ests = self.visible()
        if not ests:
            self.result.setText("—")
            return
        e = ests[-1]
        v, p = e.vmax_q, e.mslp_q
        mw = (f"{e.mw_sensor} pass at {e.mw_time:%H:%MZ}, {e.mw_age_h:.1f} h before ({e.mw_cover * 100:.0f}% of "
              "the inner 200 km)" if e.mw_time else "none (IR only)")
        lines = [f"<span style='font-size:15pt'><b>{v[2]:.0f} kt</b> · <b>{p[2]:.0f} mb</b></span>",
                 f"Vmax 25–75 %: {v[1]:.0f}–{v[3]:.0f} kt; 10–90 %: {v[0]:.0f}–{v[4]:.0f} kt",
                 f"MSLP 25–75 %: {p[3]:.0f}–{p[1]:.0f} mb; 10–90 %: {p[4]:.0f}–{p[0]:.0f} mb",
                 f"Valid {e.time:%d %b %H:%MZ} ({e.satellite}); IR frames {e.ir_frames} of 5",
                 f"Microwave: {mw}"]
        if e.ir_vmax is not None:
            lines.append(f"From the IR alone: {e.ir_vmax:.0f} kt, {e.ir_mslp:.0f} mb (the pass moved it "
                         f"{v[2] - e.ir_vmax:+.0f} kt, {p[2] - e.ir_mslp:+.0f} mb)")
        lines.append(f"Centre {abs(e.centre[0]):.2f}{'N' if e.centre[0] >= 0 else 'S'} "
                     f"{abs(e.centre[1]):.2f}{'W' if e.centre[1] < 0 else 'E'}")
        bt = nhc.track_at(self.track, e.time, 6.0) if self.track else None
        if bt is not None and bt.vmax:
            lines.append(f"{nhc.agency(getattr(self.fix, 'storm_id', ''))} best track at {bt.time:%d/%HZ}: "
                         f"{bt.vmax} kt, {bt.mslp or '?'} mb")
        lines += e.notes and [f"<i>{'; '.join(e.notes)}</i>"] or []
        self.result.setText("<br>".join(lines))

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

    def redraw(self):
        import matplotlib.dates as mdates
        fig = self.figure
        fig.clear()
        fg, bg = self._colours()
        fig.patch.set_facecolor(bg)
        ests = self.visible()
        if not ests:
            fig.text(0.5, 0.5, "No satellite estimates yet", ha="center", va="center", color="#8b95a5")
            self.canvas.draw_idle()
            return
        gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.8, 1.1])
        axv = fig.add_subplot(gs[0, :])
        axp = fig.add_subplot(gs[1, :], sharex=axv)
        t = [mdates.date2num(e.time) for e in ests]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for ax, key, unit in ((axv, "vmax_q", "kt"), (axp, "mslp_q", "mb")):
                self._axes(ax, fg)
                q = np.array([getattr(e, key) for e in ests])
                ax.fill_between(t, q[:, 0], q[:, 4], color=BAND_COLOUR, alpha=0.16, lw=0, label="10–90 %")
                ax.fill_between(t, q[:, 1], q[:, 3], color=BAND_COLOUR, alpha=0.32, lw=0, label="25–75 %")
                ax.plot(t, q[:, 2], color=BAND_COLOUR, lw=1.8, marker="o", ms=3, label="recon10s satellite estimate")
                alone = [(x, getattr(e, "ir_vmax" if key == "vmax_q" else "ir_mslp")) for x, e in zip(t, ests)]
                alone = [(x, y) if y is not None else (x, q[i, 2]) for i, (x, y) in enumerate(alone)]
                if any(getattr(e, "ir_vmax") is not None for e in ests):
                    ax.plot([a for a, _ in alone], [b for _, b in alone], color=BAND_COLOUR, lw=1.0, ls="--",
                            alpha=0.8, label="from the IR alone")
                mw_t = sorted({mdates.date2num(e.mw_time) for e in ests if e.mw_time})
                for k, x in enumerate(mw_t):
                    ax.axvline(x, color="#4dd0e1", lw=0.8, alpha=0.6, ls=":", label="microwave pass" if k == 0 else None)
                lo, hi = min(e.time for e in ests) - dt.timedelta(hours=6), max(e.time for e in ests) + dt.timedelta(hours=6)
                if self.chk_ref.isChecked():
                    attr = "vmax" if key == "vmax_q" else "mslp"
                    for name, colour, ls in (("CIMSS ADT", "#4dd0e1", "-"), ("CIMSS D-PRINT", "#90a4ae", "--"),
                                             ("CIMSS D-MINT", "#aed581", "-")):
                        pts = [r for r in self.reference.get(name, []) if lo <= r.time <= hi
                               and getattr(r, attr) is not None and (self.until is None or r.time <= self.until)]
                        if pts:
                            ax.plot([mdates.date2num(r.time) for r in pts], [getattr(r, attr) for r in pts],
                                    color=colour, lw=0.9, ls=ls, marker="." if name == "CIMSS D-MINT" else None,
                                    ms=3, alpha=0.85, label=name)
                if self.track:
                    pts = [p for p in self.track if lo <= p.time <= hi]
                    attr = "vmax" if key == "vmax_q" else "mslp"
                    pts = [p for p in pts if getattr(p, attr)]
                    if pts:
                        ax.plot([mdates.date2num(p.time) for p in pts], [getattr(p, attr) for p in pts],
                                color="#e0e0e0" if self.theme != "light" else "#333333", marker="o", ms=4, lw=1.2,
                                label=f"{nhc.agency(getattr(self.fix, 'storm_id', ''))} best track")
                adv = self.fix
                if adv is not None and getattr(adv, "time", None) and (self.until is None or adv.time <= self.until):
                    val = getattr(adv, "vmax_kt" if key == "vmax_q" else "mslp_mb", None)
                    if val:
                        ax.plot([mdates.date2num(adv.time)], [val], marker="*", ms=11, color="#ff4081", mec="#000000",
                                lw=0, label=f"{nhc.agency(getattr(self.fix, 'storm_id', ''))} advisory")
                ax.set_ylabel("Maximum wind (kt)" if key == "vmax_q" else "Central pressure (mb)", color=fg, fontsize=8)
                if ax is axv:                      # one legend serves both panels
                    leg = ax.legend(fontsize=7, loc="best", facecolor=bg, edgecolor="#5c6675", ncol=3, framealpha=0.85)
                    for tx in leg.get_texts():
                        tx.set_color(fg)
            axp.invert_yaxis()
            span_h = (max(e.time for e in ests) - min(e.time for e in ests)).total_seconds() / 3600.0 + 12.0
            step = next(h for h in (1, 3, 6, 12, 24, 48) if span_h / h <= 10)
            if step < 24:
                axp.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, step)))
            else:
                axp.xaxis.set_major_locator(mdates.DayLocator(interval=step // 24))
            axp.xaxis.set_major_formatter(mdates.DateFormatter("%d/%H%MZ"))
            axv.tick_params(labelbottom=False)
            name = (getattr(self.fix, "name", "") or "").title() if self.fix is not None else ""
            axv.set_title(f"{name} · satellite intensity (experimental)", color=fg, fontsize=10)
            self._draw_inputs(fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fg, bg)
        self.canvas.draw_idle()

    def _draw_inputs(self, ax_ir, ax_mw, fg, bg):
        inp = self.last_inputs
        for ax in (ax_ir, ax_mw):
            self._axes(ax, fg)
            ax.set_aspect("equal")
        if inp is None or inp[0] is None:
            ax_ir.text(0.5, 0.5, "IR of the newest estimate", transform=ax_ir.transAxes, ha="center", color="#8b95a5")
            ax_mw.text(0.5, 0.5, "No microwave pass", transform=ax_mw.transAxes, ha="center", color="#8b95a5")
            for ax in (ax_ir, ax_mw):
                ax.set_xticks([])
                ax.set_yticks([])
            return
        ir, mw, est = inp
        g = si.grid(si.IR_N, si.IR_DX)
        try:
            import recon10s_irtable as irt
            cmap, norm = irt.mpl_colormap()
        except Exception:
            cmap, norm = None, None
        if cmap is not None:
            im = ax_ir.pcolormesh(g, g, ir - 273.15, cmap=cmap, norm=norm, shading="nearest")
        else:
            im = ax_ir.pcolormesh(g, g, ir - 273.15, cmap="gray_r", vmin=-85, vmax=30, shading="nearest")
        ax_ir.plot(0, 0, "+", color="#ffffff", ms=10)
        ax_ir.set_title(f"IR {est.time:%d %b %H:%MZ} ({est.satellite})", color=fg, fontsize=8)
        cb = self.figure.colorbar(im, ax=ax_ir, shrink=0.85, pad=0.01)
        cb.ax.tick_params(labelsize=6, colors=fg)
        cb.set_label("°C", color=fg, fontsize=7)
        ax_ir.set_xlabel("km east of the centre", color=fg, fontsize=7)
        if mw is None:
            text = ("No microwave pass in the last 12 h" if self.chk_mw.isChecked() and self.email_provider()
                    else "Microwave off: no PPS sign-in" if self.chk_mw.isChecked() else "Microwave off")
            ax_mw.text(0.5, 0.5, text, transform=ax_mw.transAxes, ha="center", color="#8b95a5", fontsize=8)
            ax_mw.set_xticks([])
            ax_mw.set_yticks([])
            return
        gm = si.grid(si.MW_N, si.MW_DX)
        pct = 1.7 * mw[2] - 0.7 * mw[3]
        im2 = ax_mw.pcolormesh(gm, gm, np.ma.masked_invalid(pct), cmap="turbo_r", vmin=180, vmax=290, shading="nearest")
        ax_mw.plot(0, 0, "+", color="#ffffff", ms=10)
        ax_mw.set_title(f"89-GHz PCT, {est.mw_sensor} {est.mw_time:%H:%MZ} ({est.mw_age_h:.1f} h before)", color=fg,
                        fontsize=8)
        cb2 = self.figure.colorbar(im2, ax=ax_mw, shrink=0.85, pad=0.01)
        cb2.ax.tick_params(labelsize=6, colors=fg)
        cb2.set_label("K", color=fg, fontsize=7)

    def set_theme(self, theme: str):
        self.theme = theme
        self.redraw()

    def state(self) -> dict:
        return {"hours": self.hours.value(), "step": self.step.currentIndex(), "mw": self.chk_mw.isChecked(),
                "auto": self.chk_auto.isChecked(), "cimss": self.chk_ref.isChecked()}

    def restore(self, st: dict):
        if not st:
            return
        self.hours.setValue(float(st.get("hours", 12.0)))
        self.step.setCurrentIndex(int(st.get("step", 1)))
        self.chk_mw.setChecked(bool(st.get("mw", True)))
        self.chk_auto.setChecked(bool(st.get("auto", False)))
        self.chk_ref.setChecked(bool(st.get("cimss", True)))
