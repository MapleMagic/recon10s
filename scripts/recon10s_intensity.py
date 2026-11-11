#!/usr/bin/env python3
"""
recon10s_intensity — the storm's intensity over time, from every source on
one chart.

Pressure
  * Vortex messages: the minimum sea-level pressure of each centre fix
    (extrapolated from flight level, or the centre dropsonde when there is
    one -- NHC's own number for the fix).
  * Centre and eye dropsondes: the splash pressure, and the same adjusted by
    NHC's rule of thumb -- 1 mb lower for every 10 kt of surface wind, since
    a sonde that is still in wind has missed the very centre.
  * NHC: the working best track (b-deck) at each synoptic time, and the
    current advisory.
Wind
  * Each pass's peak flight-level wind, reduced to the surface with the
    standard ratio for its level (recon10s_surface), questionable winds left
    out.
  * Eyewall dropsondes: 0.83 x the lowest-150 m mean wind (WL150), else
    0.80 x the lowest-500 m mean (MBL).
  * NHC best track and advisory.
Satellite estimates (CIMSS ADT, the user's own estimator) are added by the
caller as extra series; recon10s_verify lines them up against the recon.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

import recon10s_surface as surface

CENTRE_TAGS = ("Center", "Eye")


@dataclass
class Point:
    time: dt.datetime
    value: float
    label: str = ""                       # aircraft, drop number, note
    lo: Optional[float] = None            # optional uncertainty band
    hi: Optional[float] = None


@dataclass
class Timeline:
    pressure: Dict[str, List[Point]] = field(default_factory=dict)
    wind: Dict[str, List[Point]] = field(default_factory=dict)

    def add(self, kind: str, source: str, points: List[Point]):
        if points:
            getattr(self, kind).setdefault(source, []).extend(points)
            getattr(self, kind)[source].sort(key=lambda p: p.time)

    @property
    def span(self):
        times = [p.time for d in (self.pressure, self.wind) for pts in d.values() for p in pts]
        return (min(times), max(times)) if times else None

    def is_empty(self) -> bool:
        return not any(self.pressure.values()) and not any(self.wind.values())


def splash_adjusted(p_sfc: float, sfc_wind_kt: Optional[float]) -> float:
    """Dropsonde splash pressure lowered by 1 mb per 10 kt of surface wind."""
    return float(p_sfc) - (float(sfc_wind_kt) / 10.0 if sfc_wind_kt else 0.0)


# ------------------------------------------------------------------ recon

def vdm_pressures(vdms: Sequence) -> List[Point]:
    return [Point(v.time, float(v.min_slp), f"{v.aircraft} {v.mission}".strip())
            for v in vdms or [] if v.min_slp]


def drop_pressures(drops: Sequence) -> List[List[Point]]:
    """([raw splash pressures], [wind-adjusted]) for centre and eye sondes."""
    raw, adj = [], []
    for d in drops or []:
        if d.location not in CENTRE_TAGS or d.surface is None or not d.surface.p or d.release_time is None:
            continue
        w = d.surface.wspd
        tag = f"#{d.number} {d.aircraft}" + (f", {w} kt at the surface" if w is not None else "")
        raw.append(Point(d.release_time, float(d.surface.p), tag))
        adj.append(Point(d.release_time, splash_adjusted(d.surface.p, w), tag))
    return [raw, adj]


def pass_peak_wind(ov, p, est: Optional[np.ndarray] = None) -> Optional[Point]:
    """
    One pass's strongest flight-level wind (questionable winds left out),
    reduced to the surface with the ratio for the level it was measured at.
    """
    if ov is None or not len(ov):
        return None
    est = surface.overlay_surface(ov) if est is None else est
    idx = np.asarray(p.idx, int)
    idx = idx[np.isfinite(est[idx])]
    if not idx.size:
        return None
    i = int(idx[np.argmax(est[idx])])
    red = surface.reduction(ov.column("fl_press")[i])
    return Point(ov.times[i], float(est[i]),
                 f"{p.fix.aircraft} pass {p.fix.time:%H:%MZ}: {ov.winds[i]:.0f} kt at "
                 f"{red[1] if red else '?'} × {red[0] if red else '?'}")


def pass_peak_winds(ov, passes: Sequence) -> List[Point]:
    """pass_peak_wind for every pass that has usable winds."""
    if ov is None or not len(ov):
        return []
    est = surface.overlay_surface(ov)
    return [pt for pt in (pass_peak_wind(ov, p, est) for p in passes or []) if pt is not None]


def drop_surface_winds(drops: Sequence) -> List[Point]:
    """Eyewall sondes: 0.83 x WL150 (else 0.80 x MBL)."""
    out = []
    for d in drops or []:
        if d.release_time is None or "Eyewall" not in (d.location or ""):
            continue
        if d.wl150:
            out.append(Point(d.release_time, d.wl150[1] * surface.WL150_FACTOR,
                             f"#{d.number} {d.aircraft} {d.location}: WL150 {d.wl150[1]} kt × 0.83"))
        elif d.mbl:
            out.append(Point(d.release_time, d.mbl[1] * surface.MBL_FACTOR,
                             f"#{d.number} {d.aircraft} {d.location}: MBL {d.mbl[1]} kt × 0.80"))
    return out


# --------------------------------------------------------------------- NHC

def best_track_series(track: Sequence) -> List[List[Point]]:
    """([pressure], [wind]) points from recon10s_nhc.TrackPoint objects."""
    p = [Point(t.time, float(t.mslp), t.kind) for t in track or [] if t.mslp]
    w = [Point(t.time, float(t.vmax), t.kind) for t in track or [] if t.vmax]
    return [p, w]


def build_timeline(bundle=None, track: Sequence = (), advisory=None,
                   extra: Optional[Dict[str, Dict[str, List[Point]]]] = None, agency: str = "NHC") -> Timeline:
    """
    Everything on one timeline. `bundle` is the Satellite tab's ReconBundle,
    `track` the working best track, `advisory` a recon10s_nhc.Fix (the
    current advisory), and `extra` more series as {"pressure"|"wind":
    {source: [Point]}} -- satellite estimates, for one. `agency` ("NHC" or
    "JTWC") names the best-track and advisory series.
    """
    tl = Timeline()
    if bundle is not None:
        tl.add("pressure", "Vortex message", vdm_pressures(bundle.vdms))
        raw, adj = drop_pressures(bundle.drops)
        tl.add("pressure", "Centre drop (splash)", raw)
        tl.add("pressure", "Centre drop − 1 mb/10 kt", adj)
        tl.add("wind", "Flight level × standard ratio", pass_peak_winds(bundle.overlay, bundle.passes))
        tl.add("wind", "Eyewall drop WL150 × 0.83", drop_surface_winds(bundle.drops))
    if track:
        p, w = best_track_series(track)
        tl.add("pressure", f"{agency} best track", p)
        tl.add("wind", f"{agency} best track", w)
    if advisory is not None and advisory.time is not None:
        if advisory.mslp_mb:
            tl.add("pressure", f"{agency} advisory", [Point(advisory.time, float(advisory.mslp_mb), "current")])
        if advisory.vmax_kt:
            tl.add("wind", f"{agency} advisory", [Point(advisory.time, float(advisory.vmax_kt), "current")])
    for kind, series in (extra or {}).items():
        for source, pts in series.items():
            tl.add(kind, source, list(pts))
    return tl


def lowest_pressure(tl: Timeline, sources=("Vortex message", "Centre drop − 1 mb/10 kt")):
    """(Point, source) of the lowest recon pressure, or None."""
    best = None
    for s in sources:
        for p in tl.pressure.get(s, []):
            if best is None or p.value < best[0].value:
                best = (p, s)
    return best
