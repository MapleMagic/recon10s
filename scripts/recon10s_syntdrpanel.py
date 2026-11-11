#!/usr/bin/env python3
"""
recon10s_syntdrpanel — Experimental › Synthetic TDR.

The wind field of the storm as a P-3's tail Doppler radar would have seen
it, built from the recon already loaded (flight-level obs and dropsondes of
any aircraft, USAF or NOAA) by a model of TDR structure learnt from HRD's
analyses (recon10s_syntdr). Every view says that it is modelled, not
observed.

  * Surface (10 m) view: the estimated surface wind, the peak and where it
    is, the RMW, the recon track coloured by its winds, the storm motion
    and shear; anything farther than 80 km (adjustable) from every ob is
    blacked out.
  * The 3-D views of the Airborne TDR tab (plan, sections, azimuthal mean)
    for every height 0.5-12 km, with the uncertainty as its own field.
  * Checks while recon is flying: each pass left out in turn, the previous
    analysis against the obs that arrived since, and any real TDR
    analysis of the same storm opened in Radar › Airborne TDR. Estimates
    are matched with NHC's best track when it reaches their time, and the
    mean error of those pairs corrects future estimates.
"""
from __future__ import annotations

import datetime as dt
import math
import warnings
from typing import List, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider,
    QSpinBox, QVBoxLayout,
)

import recon10s_motion as motion_mod
import recon10s_nhc as nhc
import recon10s_syntdr as st
import recon10s_tdrgrid as tg
import recon10s_windfield as wf
from recon10s_tdrgridpanel import TdrGridPanel, TdrLayer, _Job, field_style

try:
    import recon10s_plot
except Exception:  # pragma: no cover
    recon10s_plot = None

SURFACE_VIEW = ("Surface wind (10 m)", "surface")
CONTOURS = {34: "#4dd0e1", 50: "#ffd54f", 64: "#ff5252"}
SYNTHETIC_NOTE = "SYNTHETIC — modelled from recon, not a radar observation"


def wind_colours():
    """The app's wind colours (as the converter's barbs), as a colormap and norm in kt."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    buckets = recon10s_plot._COLOR_BUCKETS if recon10s_plot else [(0, 1e9, (200, 200, 200))]
    edges = [b[0] for b in buckets] + [200.0]
    colours = [tuple(c / 255.0 for c in b[2]) for b in buckets]
    return ListedColormap(colours), BoundaryNorm(edges, len(colours))


def _compass(deg: float) -> str:
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((deg % 360 + 11.25) // 22.5) % 16]


class SynTdrPanel(TdrGridPanel):
    send_to_3d = pyqtSignal(object)          # the synthetic TdrGrid, for Experimental › 3D structure
    request_load = pyqtSignal()              # "Load all recon" (Imagery › Satellite does the loading)

    # ------------------------------------------------------------ building

    def _build_source_box(self):
        # state first: the base class builds this box before anything else of ours exists
        self.bundle = None
        self.result: Optional[st.Result] = None
        self.track: List[nhc.TrackPoint] = []
        self._track_for = ""
        self._storm_id = ""
        self._advisory = None
        self._ships = None                   # (kt, heading toward, cycle) from SHIPS
        self._ships_key = None
        self._busy = False
        self._pending = False
        self._real_tdr = None                # the latest real TDR analysis opened in Radar › Airborne TDR
        self._last_tdr_check = None
        self._last_verify = None
        self._loo = None
        self.live = st.LiveCalibration()
        self.model_error = ""
        try:
            self.model = st.model()
        except Exception as exc:              # the model file is missing or unreadable
            self.model = None
            self.model_error = f"{type(exc).__name__}: {exc}"

        from PyQt6.QtWidgets import QWidget
        holder = QWidget()
        hv = QVBoxLayout(holder)
        hv.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("Recon and time")
        hv.addWidget(box)
        f = QFormLayout(box)
        self.data_label = QLabel("No recon loaded yet.")
        self.data_label.setObjectName("hint")
        self.data_label.setWordWrap(True)
        f.addRow(self.data_label)
        self.btn_load = QPushButton("Load all recon")
        self.btn_load.setToolTip("Every HDOB, vortex message and dropsonde of the storm chosen under\n"
                                 "Imagery › Satellite (the same as the button there).")
        self.btn_load.clicked.connect(self.request_load.emit)
        f.addRow(self.btn_load)

        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.valueChanged.connect(lambda _: (self._time_label(), self._schedule()))
        f.addRow("Valid at", self.time_slider)
        self.time_text = QLabel("—")
        self.time_text.setObjectName("probeReadout")
        f.addRow("", self.time_text)
        row = QHBoxLayout()
        self.half = QDoubleSpinBox()
        self.half.setRange(0.5, 6.0)
        self.half.setSingleStep(0.5)
        self.half.setValue(2.0)
        self.half.setPrefix("±")
        self.half.setSuffix(" h")
        self.half.setToolTip("Obs from this long before and after the analysis time are used; older ones\n"
                             "are moved with the storm, scaled by the intensity trend and trusted less.")
        self.half.valueChanged.connect(lambda _: (self._time_label(), self._schedule()))
        row.addWidget(self.half)
        b = QPushButton("Latest")
        b.setObjectName("smallButton")
        b.setToolTip("The newest centre fix")
        b.clicked.connect(lambda: self.time_slider.setValue(self.time_slider.maximum()))
        row.addWidget(b)
        f.addRow("Obs window", row)
        self.chk_drops = QCheckBox("Dropsondes (0.3–12 km)")
        self.chk_drops.setChecked(True)
        self.chk_drops.toggled.connect(lambda _: self._schedule())
        f.addRow(self.chk_drops)
        self.chk_trend = QCheckBox("Scale older obs by the intensity trend")
        self.chk_trend.setChecked(True)
        self.chk_trend.setToolTip("The change of the passes' peak flight-level winds (a weighted line through\n"
                                  "them) scales each ob's wind to the analysis time.")
        self.chk_trend.toggled.connect(lambda _: self._schedule())
        f.addRow(self.chk_trend)

        self.motion_box = motion_mod.MotionBox("Storm motion")
        self.motion_box.changed.connect(self._schedule)
        hv.addWidget(self.motion_box)

        sh = QGroupBox("Vertical wind shear")
        sf = QFormLayout(sh)
        self.shear_src = QComboBox()
        for label, key in (("SHIPS (850–200 mb, 0–500 km)", "ships"), ("None", "none"),
                           ("Typed in", "manual")):
            self.shear_src.addItem(label, key)
        self.shear_src.currentIndexChanged.connect(lambda _: (self._shear_widgets(), self._schedule()))
        sf.addRow("From", self.shear_src)
        srow = QHBoxLayout()
        self.shear_kt = QDoubleSpinBox()
        self.shear_kt.setRange(0.0, 80.0)
        self.shear_kt.setSuffix(" kt")
        self.shear_dir = QSpinBox()
        self.shear_dir.setRange(0, 359)
        self.shear_dir.setWrapping(True)
        self.shear_dir.setSuffix("°")
        self.shear_dir.setToolTip("The direction the shear vector points toward (SHIPS SDDC)")
        for w in (self.shear_kt, self.shear_dir):
            w.valueChanged.connect(lambda _: self._schedule() if self.shear_src.currentData() == "manual" else None)
            srow.addWidget(w)
        sf.addRow("Toward", srow)
        self.shear_note = QLabel("")
        self.shear_note.setObjectName("hint")
        self.shear_note.setWordWrap(True)
        sf.addRow(self.shear_note)
        hv.addWidget(sh)

        bb = QGroupBox("Synthetic TDR")
        hv.addWidget(bb)
        f = QFormLayout(bb)
        self.extent = QComboBox()
        for km in (150, 200):
            self.extent.addItem(f"±{km} km", km)
        self.extent.setCurrentIndex(1)
        self.extent.currentIndexChanged.connect(lambda _: self._schedule())
        f.addRow("Grid", self.extent)

        self.btn_build = QPushButton("Build synthetic TDR")
        self.btn_build.setObjectName("primary")
        self.btn_build.clicked.connect(lambda: self.build())
        f.addRow(self.btn_build)
        self.chk_auto = QCheckBox("Rebuild as new recon arrives")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip("After the first build, every new batch of recon (and every change here)\n"
                                 "builds the analysis again; the one before is scored on the obs it had not seen.")
        f.addRow(self.chk_auto)
        actions = QHBoxLayout()
        self.btn_loo = QPushButton("Check: leave each pass out")
        self.btn_loo.setObjectName("smallButton")
        self.btn_loo.setToolTip("Build the analysis again without each flight-level pass in turn and score\n"
                                "it on the pass it did not see: an honest measure of the error between legs.")
        self.btn_loo.clicked.connect(self.leave_one_out)
        self.btn_3d = QPushButton("Open in 3D structure")
        self.btn_3d.setObjectName("smallButton")
        self.btn_3d.clicked.connect(lambda: self.grid is not None and self.send_to_3d.emit(self.grid))
        for w in (self.btn_loo, self.btn_3d):
            w.setEnabled(False)
            actions.addWidget(w)
        f.addRow(actions)
        if self.model is None:
            self.btn_build.setEnabled(False)
            self.data_label.setText(f"The synthetic TDR model could not load ({self.model_error}). It ships as "
                                    "scripts/models/syntdr_v1.npz.")
        self._shear_widgets()
        return holder

    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent, theme)
        self.view_box.blockSignals(True)
        self.view_box.insertItem(0, *SURFACE_VIEW)
        self.view_box.setCurrentIndex(0)
        self.view_box.blockSignals(False)
        self.btn_in.setVisible(False)                   # no flight legs in a synthetic analysis
        self.btn_out.setVisible(False)
        form = self.view_box.parentWidget().layout()
        mrow = QHBoxLayout()
        self.mask_km = QSpinBox()
        self.mask_km.setRange(20, 300)
        self.mask_km.setSingleStep(10)
        self.mask_km.setValue(80)
        self.mask_km.setSuffix(" km")
        self.mask_km.setToolTip("Black out the surface view farther than this from every ob: there the field\n"
                                "is the prior's guess, shaped by the storm's intensity, size, shear and motion.\n"
                                "(Its uncertainty is the \u201cUncertainty (1\u03c3)\u201d field in the other views.)")
        self.mask_km.valueChanged.connect(lambda _: (self.redraw(), self._emit_layer()))
        mrow.addWidget(self.mask_km)
        form.addRow("Black out beyond", mrow)
        self.chk_obs = QCheckBox("Recon obs used (coloured by wind)")
        self.chk_obs.setChecked(True)
        self.chk_obs.toggled.connect(lambda _: self.redraw())
        form.addRow(self.chk_obs)
        self.chk_contours = QCheckBox("34 / 50 / 64-kt contours")
        self.chk_contours.setChecked(True)
        self.chk_contours.toggled.connect(lambda _: self.redraw())
        form.addRow(self.chk_contours)
        self.results = QLabel("")
        self.results.setObjectName("hint")
        self.results.setWordWrap(True)
        self.results.setTextFormat(Qt.TextFormat.RichText)
        self.results.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rv = self.readout.parentWidget().layout()
        rv.insertWidget(rv.indexOf(self.readout) + 1, self.results)
        self.info.setVisible(False)
        self.about.setText(
            "Synthetic TDR is experimental: a statistical model of TDR wind structure, learnt from HRD's "
            "TDR analyses (TC-RADAR 1997–2024 and the real-time analyses since) and fitted to the "
            "recon loaded here. It is an estimate, never a radar observation.")
        self.status.setText("Load recon and press Build synthetic TDR.")
        self._rebuild = QTimer(self)
        self._rebuild.setSingleShot(True)
        self._rebuild.setInterval(1200)
        self._rebuild.timeout.connect(lambda: self.build(auto=True))
        self._placeholder("No synthetic analysis yet")

    # ---------------------------------------------------------------- data

    def set_storm_name(self, name: str):
        self.storm_name = (name or "").upper()

    def set_bundle(self, bundle):
        """New recon (from Imagery › Satellite or a case replay)."""
        self.bundle = bundle
        fix = getattr(bundle, "fix", None) if bundle is not None else None
        self._advisory = fix
        sid = getattr(fix, "storm_id", "") if fix else ""
        if sid != self._storm_id:
            self._storm_id = sid
            self._ships, self._ships_key = None, None
        if fix is not None and getattr(fix, "name", ""):
            self.storm_name = fix.name.upper()
        times = self._fix_times()
        at_latest = self.time_slider.value() >= self.time_slider.maximum()
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0, max(0, len(times) - 1))
        if at_latest:
            self.time_slider.setValue(self.time_slider.maximum())
        self.time_slider.blockSignals(False)
        self._time_label()
        n_obs = len(bundle.overlay) if bundle is not None and bundle.overlay is not None else 0
        n_fix = len(bundle.fixes) if bundle is not None else 0
        n_drop = len(bundle.drops) if bundle is not None else 0
        if self.model is not None:
            self.data_label.setText(f"{n_obs} flight-level obs, {n_fix} centre fixes, {n_drop} dropsondes loaded"
                                    + (f" for {self.storm_name.title()}." if self.storm_name else "."))
        # motion sources, as the radar tabs offer them
        self.motion_box.set_source("nhc", fix.motion if fix is not None and getattr(fix, "motion", None) else None,
                                   "NHC advisory")
        rec = None
        if bundle is not None and bundle.fixes:
            rec = bundle.motion or wf.fixes_motion(bundle.fixes)
        self.motion_box.set_source("fixes", rec, "recon fixes")
        if sid and sid != self._track_for:
            self._track_for = sid
            job = _Job(lambda _s, s: nhc.best_track(s), sid)
            job.sig.done.connect(self.set_track)
            job.sig.failed.connect(lambda _m: None)
            self.pool.start(job)
        if self.chk_auto.isChecked() and self.result is not None:
            self._schedule()

    def set_track(self, track):
        """NHC's working best track (from here or Reconnaissance › Intensity)."""
        self.track = list(track or [])
        if self.track:
            m = nhc.track_motion(self.track, 12.0, self.centre_time())
            self.motion_box.set_source("track", m, f"best track to {self.track[-1].time:%d/%HZ}" if m else "")
        key = self._storm_key()
        if key and self.track and self.live.match_track(key, self.track):
            self._describe_result()

    def set_real_tdr(self, grid):
        """A real TDR analysis opened in Radar › Airborne TDR: compare when it is of this storm and time."""
        if grid is None or getattr(grid, "source", "") == "synthetic" or not grid.has_winds():
            return
        self._real_tdr = grid
        self._compare_real()

    def _compare_real(self):
        g, r = self._real_tdr, self.result
        if g is None or r is None or g.centre_time is None:
            return
        if self.storm_name and g.storm and g.storm.upper() not in (self.storm_name, self.storm_name.title()):
            if g.storm.upper() != self.storm_name.upper():
                return
        if abs((g.centre_time - r.when).total_seconds()) > 3 * 3600:
            return
        stats = st.compare_tdr(r, g, 2.0)
        if stats is None:
            return
        stats["dt_min"] = (g.centre_time - r.when).total_seconds() / 60.0
        self._last_tdr_check = stats
        key = self._storm_key()
        if key:
            self.live.add_tdr(key, r.when, stats)
        self._describe_result()

    def _storm_key(self) -> str:
        return self._storm_id or self.storm_name.lower()

    def _fix_times(self) -> List[dt.datetime]:
        """The centre fixes, and the newest ob when it is more than 15 minutes past the last fix."""
        b = self.bundle
        if b is None:
            return []
        times = sorted({f.time for f in (b.fixes or []) if f.time is not None})
        latest = self._latest_ob()
        if times and latest is not None and (latest - times[-1]).total_seconds() > 900 and \
                (latest - times[-1]).total_seconds() < 3 * 3600:      # the centre track reaches 3 h past a fix
            times.append(latest)
        return times

    def _latest_ob(self) -> Optional[dt.datetime]:
        """The newest flight-level ob within 220 km of the centre (not the ferry home)."""
        b = self.bundle
        if b is None or b.overlay is None or not len(b.overlay) or not b.fixes:
            return None
        last = max(f.time for f in b.fixes if f.time is not None)
        motion = self.motion_box.motion() or b.motion or wf.fixes_motion(b.fixes)
        key = (id(b), len(b.overlay), len(b.fixes), last, motion)
        if getattr(self, "_latest_key", None) == key:
            return self._latest_cached
        self._latest_key, self._latest_cached = key, None
        try:
            sr = wf.storm_relative(b.overlay, b.fixes, motion, last - dt.timedelta(hours=3),
                                   last + dt.timedelta(hours=3), max_km=220.0)
        except Exception:
            sr = None
        if sr is None or not len(sr) or not np.isfinite(sr.t).any():
            return None
        self._latest_cached = dt.datetime.fromtimestamp(float(np.nanmax(sr.t)), dt.timezone.utc).replace(
            second=0, microsecond=0)
        return self._latest_cached

    def centre_time(self) -> Optional[dt.datetime]:
        times = self._fix_times()
        if not times:
            return None
        return times[min(self.time_slider.value(), len(times) - 1)]

    def _time_label(self):
        t = self.centre_time()
        fixes = {f.time for f in (self.bundle.fixes or [])} if self.bundle is not None else set()
        what = "" if t is None or t in fixes else " (newest ob)"
        self.time_text.setText(f"{t:%d %b %H:%MZ}{what}, obs ±{self.half.value():g} h" if t else "— (no centre fixes)")

    def _shear_widgets(self):
        manual = self.shear_src.currentData() == "manual"
        for w in (self.shear_kt, self.shear_dir):
            w.setEnabled(manual)
        src = self.shear_src.currentData()
        if src == "ships":
            self.shear_note.setText(f"SHIPS {self._ships[0]:.0f} kt toward {self._ships[1]:03.0f}° "
                                    f"({self._ships[2]:%d/%HZ} cycle)" if self._ships else
                                    "SHIPS is fetched when the analysis is built.")
        elif src == "none":
            self.shear_note.setText("The prior's unsheared structure; the obs still decide the asymmetry they see.")
        else:
            self.shear_note.setText("")

    def _schedule(self, *_):
        if self.result is not None and self.chk_auto.isChecked():
            self._rebuild.start()

    def shear(self):
        src = self.shear_src.currentData()
        if src == "manual":
            return float(self.shear_kt.value()), float(self.shear_dir.value())
        if src == "ships" and self._ships:
            return self._ships[0], self._ships[1]
        return None

    # --------------------------------------------------------------- build

    def build(self, auto: bool = False):
        if self.model is None:
            return
        if self._busy:
            self._pending = True
            return
        b = self.bundle
        when = self.centre_time()
        motion = self.motion_box.motion()
        if b is None or b.overlay is None or not len(b.overlay):
            self.status.setText("No recon loaded: Load all recon (or load it in Imagery › Satellite).")
            return
        if when is None:
            self.status.setText("No centre fixes yet: the analysis needs the centre (vortex messages or fixes "
                                "derived from the obs).")
            return
        if motion is None:
            self.status.setText("No storm motion: pick a source above or type one in.")
            return
        self._busy = True
        self.btn_build.setEnabled(False)
        self.status.setText(f"Building the synthetic TDR for {when:%d %b %H:%MZ}…")
        sid = self._storm_id
        need_ships = self.shear_src.currentData() == "ships" and sid
        key = (sid, when.replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=when.hour % 6))
        previous = self.result
        offset, n_pairs = self.live.offset()
        args = dict(half_h=self.half.value(), motion=motion, use_drops=self.chk_drops.isChecked(),
                    use_trend=self.chk_trend.isChecked(), storm=self.storm_name, half_km=float(self.extent.currentData()),
                    offset_kt=offset)
        ships_cached = self._ships if self._ships_key == key else None

        def work(status, bundle, when, args, need_ships, ships_cached, previous):
            ships = ships_cached
            if need_ships and ships is None:
                status("Fetching SHIPS shear…")
                try:
                    import recon10s_tdr3d as t3
                    s = t3.ships_shear(sid, when)
                    ships = (float(s.kt), float(s.heading), s.cycle) if s is not None else False
                except Exception:
                    ships = False
            shear = None
            src = args.pop("shear_src")
            if src == "ships" and ships:
                shear = (ships[0], ships[1])
            elif src == "manual":
                shear = args.pop("manual_shear")
            args.pop("manual_shear", None)
            status("Fitting the model to the obs…")
            res = st.synthesize(bundle, when, shear=shear, **args)
            verify = None
            if previous is not None and previous.obs.t.size:
                # the analysis before against the flight-level obs it had not seen
                after = float(np.nanmax(previous.obs.t))
                new = res.obs.subset((res.obs.t > after + 30) & (res.obs.kind == st.KIND_FL)) if res.obs.t.size else None
                if new is not None and len(new) >= 20:
                    fresh = st.flight_level_obs(bundle.overlay, bundle.fixes, previous.motion,
                                                previous.when, args["half_h"] + 3.0)
                    if fresh is not None:
                        fresh = fresh.subset(fresh.t > after + 30)
                        verify = st.verify_against(previous, fresh)
                        if verify is not None:
                            verify.update(since=after, built=previous.when.isoformat())
            return res, ships, verify

        args["shear_src"] = self.shear_src.currentData()
        args["manual_shear"] = (float(self.shear_kt.value()), float(self.shear_dir.value()))
        job = _Job(work, b, when, args, need_ships, ships_cached, previous)
        job.sig.status.connect(self.status.setText)

        def done(out):
            res, ships, verify = out
            self._busy = False
            self.btn_build.setEnabled(True)
            if need_ships:
                self._ships_key = key
                self._ships = ships if ships else None
                self._shear_widgets()
            self.result = res
            self._loo = None
            if verify is not None:
                self._last_verify = verify
                k = self._storm_key()
                if k:
                    self.live.add_pass(k, previous.when, {kk: vv for kk, vv in verify.items()
                                                          if kk in ("n", "mae", "bias", "rmse")})
            k = self._storm_key()
            if k:
                self.live.add_estimate(k, res.when, res.intensity_kt, res.offset_kt, res.peak_1km_kt, res.fl_max_kt)
                if self.track:
                    self.live.match_track(k, self.track)
            self.show_grid(res.grid)
            self.btn_loo.setEnabled(True)
            self.btn_3d.setEnabled(True)
            self._compare_real()
            self._describe_result()
            self.status.setText(f"Synthetic TDR valid {res.when:%d %b %H:%MZ}: {res.n_fl} flight-level obs"
                                f"{f', {res.n_drop} dropsondes' if res.n_drop else ''}"
                                f" ({n_pairs} best-track pair(s) behind the live correction).")
            if self._pending:
                self._pending = False
                self._rebuild.start()

        def failed(msg):
            self._busy = False
            self.btn_build.setEnabled(True)
            self.status.setText(f"Synthetic TDR: {msg}")
            if self._pending:
                self._pending = False
        job.sig.done.connect(done)
        job.sig.failed.connect(failed)
        self.pool.start(job)

    def leave_one_out(self):
        r = self.result
        if r is None:
            return
        self.btn_loo.setEnabled(False)
        self.status.setText("Leaving each pass out in turn…")
        job = _Job(lambda _s, res: st.leave_one_out(res), r)

        def done(out):
            self.btn_loo.setEnabled(True)
            if out is None:
                self.status.setText("Too few separate passes in the window to leave one out.")
                return
            self._loo = out
            self._describe_result()
            self.status.setText(f"Each of {len(out['passes'])} passes left out: wind error MAE {out['mae']:.1f} kt, "
                                f"bias {out['bias']:+.1f} kt.")
        job.sig.done.connect(done)
        job.sig.failed.connect(lambda m: (self.btn_loo.setEnabled(True), self.status.setText(m)))
        self.pool.start(job)

    # ------------------------------------------------------------ describe

    def _describe(self):
        g = self.grid
        self.status.setText(f"{g.name}: {len(g.x)} × {len(g.y)} × {len(g.z)} grid, "
                            f"{g.x[1] - g.x[0]:.0f} km × {g.z[1] - g.z[0]:.1f} km.")
        self._describe_result()

    def _describe_result(self):
        r = self.result
        if r is None:
            return
        f = r.fit
        lines = [f"<b>Intensity estimate {r.intensity_kt:.0f} kt</b> ±{r.intensity_pm_kt:.0f} kt "
                 "(the typical error against NHC's best track on storms the model never saw)"
                 + ("." if r.core_sampled else " — <b>low confidence: the obs do not reach the core</b>."),
                 f"Peak 10-m wind {r.peak_kt:.0f} kt, {r.peak_where}"
                 + (f"; RMW {r.rmw_km:.0f} km at 1 km." if np.isfinite(r.rmw_km) else "."),
                 f"From {r.n_fl} flight-level obs" + (f" and {r.n_drop} dropsondes" if r.n_drop else "")
                 + (f", {r.grid.start:%H:%M}–{r.grid.end:%H:%MZ}" if r.grid.start and r.grid.end else "")
                 + f"; motion {r.motion[0]:03.0f}° {r.motion[1]:.0f} kt"
                 + (f"; shear {r.shear[0]:.0f} kt toward {r.shear[1]:03.0f}°." if r.shear else "; no shear."),
                 f"Fit: Vm {f.vm * st.MS_TO_KT:.0f} kt (2-km mean), R {f.R:.0f} km; the obs are matched to "
                 f"{f.resid_rms * st.MS_TO_KT:.0f} kt rms. Peak 1-km wind {r.peak_1km_kt:.0f} kt"
                 + (f", peak flight-level {r.fl_max_kt:.0f} kt" if r.fl_max_kt else "") + "."]
        if r.trend_kt_h is not None:
            lines.append(f"Trend of the passes' peak winds {r.trend_kt_h:+.1f} kt/h.")
        if abs(r.offset_kt) >= 0.05:
            _o, n = self.live.offset()
            lines.append(f"Live correction {r.offset_kt:+.1f} kt, from {n} estimate(s) matched with the best track.")
        for note in r.notes:
            lines.append(note)
        if self._loo:
            lines.append(f"<b>Passes left out</b> ({len(self._loo['passes'])}): wind MAE {self._loo['mae']:.1f} kt, "
                         f"bias {self._loo['bias']:+.1f} kt.")
        v = self._last_verify
        if v:
            lines.append(f"<b>Previous analysis</b> ({dt.datetime.fromisoformat(v['built']):%H:%MZ}) on the {v['n']} "
                         f"obs that arrived after it: MAE {v['mae']:.1f} kt, bias {v['bias']:+.1f} kt.")
        c = self._last_tdr_check
        if c:
            lines.append(f"<b>Against HRD's TDR</b> ({c['analysis']}, {c['dt_min']:+.0f} min): 2-km wind RMSE "
                         f"{c['rmse']:.0f} kt, bias {c['bias']:+.0f} kt; peaks {c['peak_synthetic']:.0f} vs "
                         f"{c['peak_tdr']:.0f} kt.")
        summ = self.live.summary(self._storm_key())
        if summ:
            lines.append(f"Learnt so far for this storm: {summ}.")
        self.results.setText("<br>".join(lines))

    # ---------------------------------------------------------- satellite map

    def _emit_layer(self):
        g, r = self.grid, self.result
        if g is None or r is None or not self.chk_map.isChecked():
            self.layer_changed.emit(None)
            return
        if self.view_box.currentData() == "surface":
            cmap, norm = wind_colours()
            self.layer_changed.emit(TdrLayer(g.lat, g.lon, self._masked_surface(), cmap, norm,
                                             f"Synthetic TDR surface wind (kt), {r.when:%d %H:%MZ} — modelled",
                                             r.when))
            return
        name = self.field_box.currentData()
        km = self.height.value() * 0.5
        cmap, norm, unit, scale = field_style(name, self.chk_kt.isChecked())
        self.layer_changed.emit(TdrLayer(g.lat, g.lon, g.plan(name, km) * scale, cmap, norm,
                                         f"Synthetic TDR {tg.FIELDS.get(name, (name,))[0].lower()} {km:.1f} km "
                                         f"({unit}), {r.when:%d %H:%MZ} — modelled", r.when))

    def _unsure(self) -> np.ndarray:
        r = self.result
        if r.ob_distance is None:
            return np.zeros(r.surface.shape, bool)
        return np.isfinite(r.surface) & (r.ob_distance > self.mask_km.value())

    def _masked_surface(self) -> np.ndarray:
        out = self.result.surface.astype(float).copy()
        out[self._unsure()] = np.nan
        return out

    # ------------------------------------------------------------- drawing

    def redraw(self):
        g, r = self.grid, self.result
        if g is None or r is None:
            return
        if self.view_box.currentData() != "surface":
            super().redraw()
            self._watermark()
            return
        fg, bg = self._colours()
        fig = self.figure
        fig.clear()
        self._roles, self._markers, self._background = {}, {}, None
        self._pinned = False
        fig.patch.set_facecolor(bg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._surface(fig.add_subplot(1, 1, 1), fg, bg)
        self._watermark()
        self.canvas.draw_idle()

    def _watermark(self):
        fg, _bg = self._colours()
        r = self.result
        title = (f"{(r.storm or self.storm_name or 'Storm').title()}  ·  "
                 f"{'Wind speed at surface (10 m)' if self.view_box.currentData() == 'surface' else 'Synthetic TDR'}"
                 f"  ·  {r.when:%d %b %H:%MZ}")
        self.figure.suptitle(f"{title}\n{SYNTHETIC_NOTE}  ·  model {self._model_tag()}", color=fg, fontsize=10)

    def _model_tag(self) -> str:
        info = getattr(self.model, "info", {}) or {}
        return f"v{info.get('version', 1)}, recon-assimilated"

    def _surface(self, ax, fg, bg):
        r = self.result
        g = r.grid
        cmap, norm = wind_colours()
        vals = r.surface.astype(float)
        unsure = self._unsure()
        im = ax.pcolormesh(g.x, g.y, np.ma.masked_invalid(np.where(unsure, np.nan, vals)), cmap=cmap, norm=norm,
                           shading="nearest")
        from matplotlib.colors import ListedColormap
        if unsure.any():
            ax.pcolormesh(g.x, g.y, np.ma.masked_where(~unsure, np.ones_like(vals)),
                          cmap=ListedColormap(["#000000"]), shading="nearest", alpha=0.78)
        self._style(ax, fg, bg)
        if self.chk_contours.isChecked() and np.isfinite(vals).any():
            levels = [k for k in CONTOURS if np.nanmax(vals) > k]
            if levels:
                ax.contour(g.x, g.y, np.where(np.isfinite(vals), vals, 0.0), levels=levels,
                           colors=[CONTOURS[k] for k in levels], linewidths=0.9)
        if np.isfinite(r.rmw_km):
            t = np.linspace(0, 2 * np.pi, 181)
            ax.plot(r.rmw_km * np.sin(t), r.rmw_km * np.cos(t), color="#ffffff", lw=1.0, ls="--", alpha=0.85)
            ax.text(0, r.rmw_km, f" RMW {r.rmw_km:.0f} km", color="#ffffff", fontsize=7, va="bottom")
        if self.chk_obs.isChecked():
            o = r.obs
            mx, my = st._motion_ms(r.motion)
            spd = np.hypot(o.u + mx, o.v + my) * st.MS_TO_KT
            fl = o.kind == st.KIND_FL
            ax.scatter(o.x[fl], o.y[fl], c=spd[fl], cmap=cmap, norm=norm, s=9, edgecolors="#000000", linewidths=0.3,
                       zorder=5)
            dr = o.kind == st.KIND_DROP
            if dr.any():
                low = dr & (o.z <= 1.0)
                ax.scatter(o.x[low], o.y[low], c=spd[low], cmap=cmap, norm=norm, s=36, marker="v",
                           edgecolors="#ffffff", linewidths=0.6, zorder=6)
        ax.plot(0, 0, marker="+", ms=11, mew=1.6, color="#ffffff", zorder=7)
        if np.isfinite(r.peak_kt):
            px, py = r.peak_xy
            ax.plot(px, py, marker="*", ms=15, color="#ffffff", mec="#000000", mew=0.8, zorder=8)
        lim = min(self._extent_km(), float(self.extent.currentData()))
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("km east of centre", color=fg, fontsize=8)
        ax.set_ylabel("km north", color=fg, fontsize=8)
        # motion and shear arrows, bottom left
        self._arrow(ax, r.motion[0], f"Motion {r.motion[1]:.0f} kt", (0.09, 0.14), "#ffffff")
        if r.shear:
            self._arrow(ax, r.shear[1], f"Shear {r.shear[0]:.0f} kt", (0.25, 0.14), "#ffb74d")
        box = dict(boxstyle="round,pad=0.35", facecolor=bg, edgecolor="#5c6675", alpha=0.85)
        ax.text(0.02, 0.98, f"Max {r.peak_kt:.0f} kt  ·  {r.peak_where}\n"
                            f"Intensity estimate {r.intensity_kt:.0f} ± {r.intensity_pm_kt:.0f} kt",
                transform=ax.transAxes, va="top", ha="left", color=fg, fontsize=9, bbox=box, zorder=9)
        ax.text(0.98, 0.98, "SYNTHETIC", transform=ax.transAxes, va="top", ha="right", color="#ffb74d",
                fontsize=10, fontweight="bold", zorder=9)
        key = []
        if self.chk_obs.isChecked():
            key.append("dots: recon flight-level wind; ▽ dropsonde winds below 1 km (same colours)")
        if unsure.any():
            key.append(f"black: farther than {self.mask_km.value()} km from every ob")
        if not r.core_sampled:
            key.append("the obs do not reach the core: the peak is the model's guess")
        if key:
            ax.text(0.98, 0.02, "\n".join(key), transform=ax.transAxes, va="bottom", ha="right", color="#c0c8d4",
                    fontsize=7, zorder=9, bbox=dict(boxstyle="round,pad=0.25", facecolor=bg, edgecolor="none", alpha=0.7))
        cb = self.figure.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
        cb.ax.tick_params(labelsize=7, colors=fg)
        cb.set_label("10-m wind (kt), modelled", color=fg, fontsize=8)
        self._roles[ax] = ("surface",)

    @staticmethod
    def _arrow(ax, heading: float, label: str, at, colour):
        a = math.radians(heading)
        L = 0.065
        x0, y0 = at
        ax.annotate("", xy=(x0 + L * math.sin(a), y0 + L * math.cos(a)), xytext=(x0 - L * math.sin(a), y0 - L * math.cos(a)),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.6), zorder=9)
        ax.text(x0, y0 - L - 0.03, label, transform=ax.transAxes, ha="center", va="top", color=colour, fontsize=7,
                zorder=9)

    def _plan(self, ax, fg, bg):
        super()._plan(ax, fg, bg)
        r = self.result
        if r is None or not self.chk_obs.isChecked():
            return
        km = self.height.value() * 0.5
        o = r.obs
        near = np.abs(o.z - max(km, 0.5)) <= 0.75
        if near.any():
            ax.scatter(o.x[near], o.y[near], s=6, c="#ffffff", edgecolors="#000000", linewidths=0.3, zorder=5,
                       label="obs within 0.75 km of this height")

    # --------------------------------------------------------------- probe

    def probe_text(self, ax, xdata: float, ydata: float) -> Optional[str]:
        role = self._roles.get(ax)
        if role is None or role[0] != "surface":
            return super().probe_text(ax, xdata, ydata)
        r = self.result
        g = r.grid
        dx = g.x[1] - g.x[0]
        i = int(round((ydata - g.y[0]) / dx))
        j = int(round((xdata - g.x[0]) / dx))
        if not (0 <= i < g.y.size and 0 <= j < g.x.size):
            return None
        d = math.hypot(xdata, ydata)
        brg = (math.degrees(math.atan2(xdata, ydata)) + 360) % 360
        v, s = float(r.surface[i, j]), float(r.surface_spread[i, j])
        lat, lon = float(g.lat[i, j]), float(g.lon[i, j])
        head = (f"{d:.0f} km {_compass(brg)} of centre  ·  {abs(lat):.2f}{'N' if lat >= 0 else 'S'} "
                f"{abs(lon):.2f}{'W' if lon < 0 else 'E'}")
        if not np.isfinite(v):
            return head + "  ·  outside the analysis"
        k = g.level(1.0)
        w1 = float(g.fields["WIND_SPEED"][k, i, j]) * st.MS_TO_KT
        return head + f"\n10-m wind {v:.0f} kt ±{s:.0f} kt (modelled)   1-km wind {w1:.0f} kt"

    # ---------------------------------------------------------------- state

    def state(self) -> dict:
        return {"view": self.view_box.currentIndex(), "field": self.field_box.currentIndex(),
                "height": self.height.value(), "kt": self.chk_kt.isChecked(), "half": self.half.value(),
                "drops": self.chk_drops.isChecked(), "trend": self.chk_trend.isChecked(),
                "motion": self.motion_box.state(), "shear": self.shear_src.currentData(),
                "shear_kt": self.shear_kt.value(), "shear_dir": self.shear_dir.value(),
                "extent": self.extent.currentIndex(), "mask_km": self.mask_km.value(), "auto": self.chk_auto.isChecked(),
                "obs": self.chk_obs.isChecked(), "contours": self.chk_contours.isChecked()}

    def restore(self, s: dict):
        if not s:
            return
        self.view_box.setCurrentIndex(int(s.get("view", 0)))
        self.field_box.setCurrentIndex(int(s.get("field", 0)))
        self.height.setValue(int(s.get("height", 4)))
        self.chk_kt.setChecked(bool(s.get("kt", True)))
        self.half.setValue(float(s.get("half", 2.0)))
        self.chk_drops.setChecked(bool(s.get("drops", True)))
        self.chk_trend.setChecked(bool(s.get("trend", True)))
        self.motion_box.restore(s.get("motion") or {})
        i = self.shear_src.findData(s.get("shear", "ships"))
        self.shear_src.setCurrentIndex(max(0, i))
        self.shear_kt.setValue(float(s.get("shear_kt", 0.0)))
        self.shear_dir.setValue(int(s.get("shear_dir", 0)))
        self.extent.setCurrentIndex(int(s.get("extent", 1)))
        self.mask_km.setValue(int(s.get("mask_km", 80)))
        self.chk_auto.setChecked(bool(s.get("auto", True)))
        self.chk_obs.setChecked(bool(s.get("obs", True)))
        self.chk_contours.setChecked(bool(s.get("contours", True)))
