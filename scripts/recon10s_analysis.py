#!/usr/bin/env python3
"""
recon10s_analysis — turning obs into storm structure.

  * Centre fixes: NHC vortex messages where there are any; otherwise derived
    from the obs. A pass is where the track closes on the centre; the fix is
    the lightest flight-level wind near the lowest geopotential height (on an
    isobaric leg, the lowest height marks the pressure centre).
  * Passes: obs within +-30 min of each fix, with a signed radius from the
    centre (negative inbound, positive outbound), the centre interpolated
    between fixes so storm motion does not smear the profile.
  * Radius of maximum wind and peak wind per leg.
  * Tilt: flight-level centre against the surface centre, from a VDM's
    "SFC CNTR" remark or from a centre dropsonde's splash point.
  * Motion from a least-squares fit of the recent fixes.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

KM_PER_NM = 1.852


def gc_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def destination(lat, lon, brg, km):
    d = km / 6371.0
    b = math.radians(brg)
    la1, lo1 = math.radians(lat), math.radians(lon)
    la2 = math.asin(math.sin(la1) * math.cos(d) + math.cos(la1) * math.sin(d) * math.cos(b))
    lo2 = lo1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(la1), math.cos(d) - math.sin(la1) * math.sin(la2))
    return math.degrees(la2), (math.degrees(lo2) + 540) % 360 - 180


@dataclass
class CentreFix:
    time: dt.datetime
    lat: float
    lon: float
    source: str                     # "VDM" or "derived"
    aircraft: str = ""
    min_slp: Optional[float] = None
    min_height: Optional[float] = None
    min_wind: Optional[int] = None
    vdm: object = None
    mission: str = ""

    @property
    def label(self) -> str:
        bits = [f"{self.time:%H:%MZ}", self.source, self.aircraft]
        if self.min_slp:
            bits.append(f"{self.min_slp:.0f} mb")
        elif self.min_height:
            bits.append(f"{self.min_height:.0f} m")
        return "  ".join(b for b in bits if b)


@dataclass
class Leg:
    radius: np.ndarray       # km, positive
    wind: np.ndarray
    idx: np.ndarray
    rmw: Optional[float] = None
    vmax: Optional[float] = None
    at: Optional[Tuple[float, float, dt.datetime]] = None   # lat, lon, time of vmax


@dataclass
class Pass:
    fix: CentreFix
    signed_radius: np.ndarray
    wind: np.ndarray
    height: np.ndarray
    pressure: np.ndarray
    times: List[dt.datetime]
    idx: np.ndarray
    inbound: Optional[Leg] = None
    outbound: Optional[Leg] = None
    label: str = ""


@dataclass
class Tilt:
    fix: CentreFix
    bearing: float
    km: float
    source: str


# ------------------------------------------------------------------ fixes

def fixes_from_vdms(vdms) -> List[CentreFix]:
    return [CentreFix(time=v.time, lat=v.lat, lon=v.lon, source="VDM", aircraft=v.aircraft,
                      min_slp=v.min_slp, min_height=v.height_m, vdm=v, mission=v.mission) for v in vdms]


def centre_at(fixes: Sequence[CentreFix], t: dt.datetime,
              fallback: Optional[Tuple[float, float]] = None) -> Optional[Tuple[float, float]]:
    """Centre at time t, linear between fixes, extrapolated with motion up to 3 h past the ends."""
    if not fixes:
        return fallback
    fs = sorted(fixes, key=lambda f: f.time)
    if len(fs) == 1:
        return fs[0].lat, fs[0].lon
    ts = np.array([f.time.timestamp() for f in fs])
    x = t.timestamp()
    if x <= ts[0] or x >= ts[-1]:
        a, b = (fs[0], fs[1]) if x <= ts[0] else (fs[-2], fs[-1])
        span = (b.time - a.time).total_seconds()
        if span < 1800 or abs(x - (ts[0] if x <= ts[0] else ts[-1])) > 3 * 3600:
            f = fs[0] if x <= ts[0] else fs[-1]
            return f.lat, f.lon
        frac = (x - a.time.timestamp()) / span
        return a.lat + frac * (b.lat - a.lat), a.lon + frac * (b.lon - a.lon)
    return float(np.interp(x, ts, [f.lat for f in fs])), float(np.interp(x, ts, [f.lon for f in fs]))


def derive_fixes(ov, guess: Callable[[dt.datetime], Optional[Tuple[float, float]]],
                 max_km: float = 120.0, min_gap_min: float = 40.0) -> List[CentreFix]:
    """
    Centre fixes from the obs themselves, for flights with no vortex message
    (converted NOAA IWG1 data). One per close approach to the guessed centre.
    """
    if ov is None or not len(ov):
        return []
    lats, lons = np.asarray(ov.lats, float), np.asarray(ov.lons, float)
    winds = np.array([np.nan if w is None else w for w in ov.winds], float)
    heights = np.array([np.nan if h is None else h for h in ov.column("heights")], float)
    press = np.array([np.nan if p is None else p for p in ov.column("fl_press")], float)
    segs = np.asarray(ov.column("segments", 0))
    labels = ov.column("labels", "")
    stamps = np.array([t.timestamp() if t else np.nan for t in ov.times])
    fixes: List[CentreFix] = []
    for sid in np.unique(segs):
        sel = np.flatnonzero(segs == sid)
        dist = np.full(sel.size, np.inf)
        for k, i in enumerate(sel):
            g = guess(ov.times[i]) if ov.times[i] else None
            if g:
                dist[k] = gc_km(g[0], g[1], lats[i], lons[i])
        order = np.argsort(dist)
        taken: List[float] = []
        for k in order:
            if dist[k] > max_km:
                break
            t0 = stamps[sel[k]]
            if any(abs(t0 - t) < min_gap_min * 60 for t in taken):
                continue
            win = sel[np.abs(stamps[sel] - t0) <= 15 * 60]
            if win.size < 4:
                continue
            h = heights[win]
            isobaric = np.isfinite(press[win]).sum() > 3 and np.nanstd(press[win]) < 6.0
            if isobaric and np.isfinite(h).any():
                anchor = win[int(np.nanargmin(h))]
                near = win[gc_km(lats[anchor], lons[anchor], lats[win], lons[win]) <= 25.0]
            else:
                anchor, near = None, win
            w = winds[near]
            if not np.isfinite(w).any():
                continue
            c = near[int(np.nanargmin(w))]
            # A real centre is calm compared with the winds around it.
            if np.nanmax(winds[win]) < 2.0 * max(np.nanmin(w), 5.0):
                continue
            taken.append(t0)
            fixes.append(CentreFix(time=ov.times[c], lat=float(lats[c]), lon=float(lons[c]),
                                   source="derived", aircraft=str(labels[c]).split(" ")[0],
                                   mission=" ".join(str(labels[c]).split(" ")[1:2]),
                                   min_height=float(np.nanmin(h)) if np.isfinite(h).any() else None,
                                   min_wind=int(np.nanmin(w))))
    return sorted(fixes, key=lambda f: f.time)


def combine_fixes(vdm_fixes: List[CentreFix], derived: List[CentreFix]) -> List[CentreFix]:
    """Official fixes win; a derived fix within 25 min of one from the same aircraft is dropped."""
    out = list(vdm_fixes)
    for d in derived:
        if not any(abs((d.time - v.time).total_seconds()) < 1500 and
                   (not d.aircraft or not v.aircraft or d.aircraft == v.aircraft) for v in vdm_fixes):
            out.append(d)
    return sorted(out, key=lambda f: f.time)


# ---------------------------------------------------------------- passes

def _leg(radius, wind, idx, ov, lats, lons) -> Optional[Leg]:
    ok = np.isfinite(wind) & np.isfinite(radius)
    if ok.sum() < 3:
        return None
    leg = Leg(radius=radius[ok], wind=wind[ok], idx=idx[ok])
    k = int(np.argmax(leg.wind))
    leg.vmax, leg.rmw = float(leg.wind[k]), float(leg.radius[k])
    i = int(leg.idx[k])
    leg.at = (float(lats[i]), float(lons[i]), ov.times[i])
    return leg


def passes(ov, fixes: Sequence[CentreFix], window_min: float = 30.0,
           max_km: float = 250.0) -> List[Pass]:
    """One radial profile per fix, with the centre moving between fixes."""
    if ov is None or not len(ov) or not fixes:
        return []
    lats, lons = np.asarray(ov.lats, float), np.asarray(ov.lons, float)
    winds = np.array([np.nan if w is None else w for w in ov.winds], float)
    heights = np.array([np.nan if h is None else h for h in ov.column("heights")], float)
    press = np.array([np.nan if p is None else p for p in ov.mslps], float)
    stamps = np.array([t.timestamp() if t else np.nan for t in ov.times])
    labels = ov.column("labels", "")
    out = []
    for fx in fixes:
        t0 = fx.time.timestamp()
        idx = np.flatnonzero(np.abs(stamps - t0) <= window_min * 60)
        if fx.aircraft:
            same = [i for i in idx if str(labels[i]).startswith(fx.aircraft)]
            if len(same) >= 6:
                idx = np.array(same)
        if idx.size < 6:
            continue
        r = np.empty(idx.size)
        for k, i in enumerate(idx):
            c = centre_at(fixes, ov.times[i], (fx.lat, fx.lon))
            r[k] = gc_km(c[0], c[1], lats[i], lons[i])
        keep = r <= max_km
        idx, r = idx[keep], r[keep]
        if idx.size < 6:
            continue
        sign = np.where(stamps[idx] < t0, -1.0, 1.0)
        p = Pass(fix=fx, signed_radius=r * sign, wind=winds[idx], height=heights[idx],
                 pressure=press[idx], times=[ov.times[i] for i in idx], idx=idx,
                 label=f"{fx.time:%H:%MZ} {fx.aircraft} ({fx.source})")
        inb, outb = sign < 0, sign > 0
        p.inbound = _leg(r[inb], winds[idx][inb], idx[inb], ov, lats, lons)
        p.outbound = _leg(r[outb], winds[idx][outb], idx[outb], ov, lats, lons)
        out.append(p)
    return out


# ------------------------------------------------------------------ tilt

def tilts(fixes: Sequence[CentreFix], drops=()) -> List[Tilt]:
    """
    Surface centre relative to the flight-level centre. A vortex message's
    "SFC CNTR bbb/nn NM" remark is used when present; otherwise the splash
    point of a centre or eye dropsonde within 20 minutes of the fix.
    """
    out = []
    for fx in fixes:
        v = fx.vdm
        if v is not None and getattr(v, "sfc_centre", None):
            brg, nm = v.sfc_centre
            out.append(Tilt(fx, float(brg), nm * KM_PER_NM, "VDM remark"))
            continue
        best = None
        for d in drops:
            if d.location not in ("Center", "Eye") or d.release_time is None:
                continue
            gap = abs((d.release_time - fx.time).total_seconds())
            if gap > 1200:
                continue
            lat = d.splash_lat if d.splash_lat is not None else d.release_lat
            lon = d.splash_lon if d.splash_lon is not None else d.release_lon
            if best is None or gap < best[0]:
                best = (gap, lat, lon, d.number)
        if best:
            _, lat, lon, n = best
            out.append(Tilt(fx, bearing(fx.lat, fx.lon, lat, lon),
                            float(gc_km(fx.lat, fx.lon, lat, lon)), f"dropsonde #{n} splash"))
    return out


# ---------------------------------------------------------------- motion

def motion(fixes: Sequence[CentreFix], hours: float = 6.0) -> Optional[Tuple[float, float]]:
    """(heading deg, speed kt) from a straight-line fit of the last `hours` of fixes."""
    fs = sorted(fixes, key=lambda f: f.time)
    if len(fs) < 2:
        return None
    end = fs[-1].time
    fs = [f for f in fs if (end - f.time).total_seconds() <= hours * 3600]
    if len(fs) < 2 or (fs[-1].time - fs[0].time).total_seconds() < 3600:
        return None
    t = np.array([(f.time - fs[0].time).total_seconds() / 3600 for f in fs])
    la = np.array([f.lat for f in fs])
    lo = np.array([f.lon for f in fs])
    dlat = np.polyfit(t, la, 1)[0]
    dlon = np.polyfit(t, lo, 1)[0]
    lat0 = math.radians(la.mean())
    vn, ve = dlat * 111.2, dlon * 111.2 * math.cos(lat0)       # km/h
    return (math.degrees(math.atan2(ve, vn)) + 360) % 360, math.hypot(vn, ve) / KM_PER_NM


def peak_labels(pass_list: Sequence[Pass]) -> List[Tuple[float, float, int, dt.datetime]]:
    """(lat, lon, kt, time) of the peak flight-level wind on every leg."""
    out = []
    for p in pass_list:
        for leg in (p.inbound, p.outbound):
            if leg and leg.at and leg.vmax is not None:
                out.append((leg.at[0], leg.at[1], int(round(leg.vmax)), leg.at[2]))
    return out
