#!/usr/bin/env python3
"""
recon10s_windfield — flight-level winds in storm-relative coordinates, a 2-D
wind field from them, and the 34/50/64-kt wind radii by quadrant.

  * Centre track: every ob is placed relative to where the centre was when
    the ob was taken. The track comes from the centre fixes (vortex messages
    and fixes derived from the obs) by a local linear fit in time (Gaussian
    weights, 1 h scale), so a few km of fix-to-fix scatter between aircraft
    does not zigzag the obs; beyond the last fix the track carries on along
    the storm motion for up to `max_extrap_h` hours.
  * Coordinates: km east and north of that centre, or rotated so the storm
    motion points up. Winds are earth-relative (as NHC's wind radii are) or
    storm-relative (motion removed).
  * Analysis: a Gaussian-weighted (single-pass Barnes) analysis on a polar
    grid, with separate radial and along-circle length scales -- the obs
    come along radial legs, so they are dense along a leg and sparse around
    the circle. Points with too little weight nearby are left blank rather
    than invented.
  * Wind radii: in each quadrant, the farthest radius at which a surface
    estimate (flight level x standard ratio, from level flight near a
    standard level only -- not climbs, descents or transits) reaches 34, 50
    or 64 kt, as observed along the flight track and in the analysis. The aircraft only
    sees part of each quadrant, so these are what was sampled; a radius
    reached at the very end of the sampling is flagged as a lower bound.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import recon10s_surface as surface

KM_PER_NM = 1.852
QUADRANTS = ("NE", "SE", "SW", "NW")
THRESHOLDS = (34, 50, 64)
EARTH_KM = 6371.0


# ------------------------------------------------------------- centre track

def centre_track(fixes: Sequence, times: Sequence[dt.datetime], motion: Optional[Tuple[float, float]] = None,
                 sigma_h: float = 1.0, max_extrap_h: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    (lat, lon) of the centre at each time; NaN where no fix is within reach.
    A weighted linear fit of the fixes around each time; past the ends of
    the fixes it continues along `motion` (heading, kt) when given.
    """
    n = len(times)
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    fs = sorted([f for f in fixes or [] if f.time is not None], key=lambda f: f.time)
    if not fs or not n:
        return lat, lon
    ft = np.array([f.time.timestamp() for f in fs]) / 3600.0
    fla = np.array([f.lat for f in fs])
    flo = np.degrees(np.unwrap(np.radians([f.lon for f in fs])))
    tt = np.array([t.timestamp() if t else np.nan for t in times]) / 3600.0
    for i, t in enumerate(tt):
        if not np.isfinite(t):
            continue
        if t < ft[0] - max_extrap_h or t > ft[-1] + max_extrap_h:
            continue
        w = np.exp(-0.5 * ((ft - t) / sigma_h) ** 2)
        if (w > 0.05).sum() >= 2 and ft[0] - 0.25 <= t <= ft[-1] + 0.25:
            sw = w.sum()
            tm = (w * ft).sum() / sw
            var = (w * (ft - tm) ** 2).sum()
            if var > 1e-6:
                for src, dst in ((fla, lat), (flo, lon)):
                    ym = (w * src).sum() / sw
                    slope = (w * (ft - tm) * (src - ym)).sum() / var
                    dst[i] = ym + slope * (t - tm)
                continue
        # one fix in reach, or past an end: nearest fix, carried along the motion
        j = int(np.argmin(np.abs(ft - t)))
        dt_h = t - ft[j]
        mot = motion if motion is not None else fixes_motion(fs)
        if mot is not None and abs(dt_h) > 1e-3:
            hd, kt = mot
            la, lo = _destination(fla[j], flo[j], hd if dt_h > 0 else (hd + 180) % 360,
                                  abs(dt_h) * kt * KM_PER_NM)
            lat[i], lon[i] = la, lo
        elif abs(dt_h) <= 1.0:
            lat[i], lon[i] = fla[j], flo[j]
    lon = (lon + 540.0) % 360.0 - 180.0
    return lat, lon


def fixes_motion(fixes: Sequence) -> Optional[Tuple[float, float]]:
    """(heading, kt) from a straight-line fit of all the fixes, or None."""
    fs = sorted([f for f in fixes if f.time is not None], key=lambda f: f.time)
    if len(fs) < 2 or (fs[-1].time - fs[0].time).total_seconds() < 3600:
        return None
    t = np.array([(f.time - fs[0].time).total_seconds() / 3600 for f in fs])
    la = np.array([f.lat for f in fs])
    lo = np.degrees(np.unwrap(np.radians([f.lon for f in fs])))
    vn = np.polyfit(t, la, 1)[0] * 111.2
    ve = np.polyfit(t, lo, 1)[0] * 111.2 * math.cos(math.radians(la.mean()))
    return (math.degrees(math.atan2(ve, vn)) + 360) % 360, math.hypot(vn, ve) / KM_PER_NM


def _destination(lat, lon, brg, km):
    d = km / EARTH_KM
    b = math.radians(brg)
    la1, lo1 = math.radians(lat), math.radians(lon)
    la2 = math.asin(math.sin(la1) * math.cos(d) + math.cos(la1) * math.sin(d) * math.cos(b))
    lo2 = lo1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(la1), math.cos(d) - math.sin(la1) * math.sin(la2))
    return math.degrees(la2), math.degrees(lo2)


def local_xy(lat0, lon0, lat, lon) -> Tuple[np.ndarray, np.ndarray]:
    """Azimuthal-equidistant km east/north of (lat0, lon0); arrays broadcast."""
    la0, lo0 = np.radians(lat0), np.radians(lon0)
    la, lo = np.radians(lat), np.radians(lon)
    dlo = lo - lo0
    c = np.sin(la0) * np.sin(la) + np.cos(la0) * np.cos(la) * np.cos(dlo)
    rho = np.arccos(np.clip(c, -1, 1))
    az = np.arctan2(np.sin(dlo) * np.cos(la), np.cos(la0) * np.sin(la) - np.sin(la0) * np.cos(la) * np.cos(dlo))
    d = EARTH_KM * rho
    return d * np.sin(az), d * np.cos(az)


# ---------------------------------------------------------- storm-relative obs

@dataclass
class StormRelative:
    """Obs relative to the moving centre."""
    times: List[dt.datetime]
    t: np.ndarray                 # Unix seconds
    x: np.ndarray                 # km east of the centre (or right of motion when rotated)
    y: np.ndarray                 # km north (or along motion)
    r: np.ndarray                 # km
    az: np.ndarray                # bearing from the centre, deg (geographic, even when rotated)
    wspd: np.ndarray              # flight-level wind, kt (NaN: missing or questionable)
    sfc: np.ndarray               # surface estimate, kt (NaN: no standard ratio)
    u: np.ndarray                 # earth-relative wind toward east, kt
    v: np.ndarray
    us: np.ndarray                # storm-relative wind
    vs: np.ndarray
    fl_press: np.ndarray
    labels: List[str]
    idx: np.ndarray               # index into the source Overlay
    motion: Optional[Tuple[float, float]] = None
    rotated: bool = False         # x/y turned so the motion points up
    centre_lat: np.ndarray = field(default_factory=lambda: np.empty(0))
    centre_lon: np.ndarray = field(default_factory=lambda: np.empty(0))
    segments: np.ndarray = field(default_factory=lambda: np.empty(0, int))   # one id per flight
    level: np.ndarray = field(default_factory=lambda: np.empty(0, bool))     # flown level (not climbing)

    def __len__(self):
        return self.t.size

    def speed(self, which: str = "sfc") -> np.ndarray:
        return self.sfc if which == "sfc" else self.wspd

    def tracks(self, max_gap_s: float = 180.0) -> List[np.ndarray]:
        """Index arrays of each flight's continuous stretches, for drawing tracks without false joins."""
        out = []
        segs = self.segments if self.segments.size == self.t.size else np.zeros(self.t.size, int)
        for sid in np.unique(segs):
            sel = np.flatnonzero(segs == sid)
            sel = sel[np.argsort(self.t[sel])]
            cuts = [0] + [k for k in range(1, sel.size) if self.t[sel[k]] - self.t[sel[k - 1]] > max_gap_s] + [sel.size]
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b - a >= 2:
                    out.append(sel[a:b])
        return out

    def use_outer_ratio(self, rmw_km: Optional[float]) -> "StormRelative":
        """Surface estimates eased to the outer-vortex ratio beyond the eyewall (in place)."""
        sfc = surface.radius_dependent(self.wspd, self.fl_press, self.r, rmw_km)
        if self.level.size == sfc.size:
            sfc[~self.level] = np.nan             # climbing or descending: no standard ratio applies
        self.sfc = sfc
        return self

    def rmw_estimate(self) -> Optional[float]:
        """Radius of the strongest flight-level wind in 5 km bins (a robust median of the leg peaks)."""
        ok = np.isfinite(self.wspd) & (self.r <= 200)
        if ok.sum() < 10:
            return None
        k = int(np.nanargmax(np.where(ok, self.wspd, -np.inf)))
        # peak per leg-ish: the strongest ob in each 30 min block, median of their radii
        blocks = np.floor((self.t[ok] - self.t[ok].min()) / 1800.0)
        radii = []
        for bk in np.unique(blocks):
            sel = blocks == bk
            if sel.sum() >= 5:
                radii.append(self.r[ok][sel][int(np.nanargmax(self.wspd[ok][sel]))])
        return float(np.median(radii)) if radii else float(self.r[k])


def storm_relative(ov, fixes: Sequence, motion: Optional[Tuple[float, float]] = None,
                   t0: Optional[dt.datetime] = None, t1: Optional[dt.datetime] = None,
                   rotate: bool = False, max_km: float = 400.0, include_flagged: bool = False,
                   sigma_h: float = 1.0) -> Optional[StormRelative]:
    """The obs between t0 and t1 within max_km of the centre, storm-relative."""
    if ov is None or not len(ov) or not fixes:
        return None
    times = list(ov.times)
    stamps = np.array([t.timestamp() if t else np.nan for t in times])
    keep = np.isfinite(stamps)
    if t0 is not None:
        keep &= stamps >= t0.timestamp()
    if t1 is not None:
        keep &= stamps <= t1.timestamp()
    idx = np.flatnonzero(keep)
    if not idx.size:
        return None
    sel_times = [times[i] for i in idx]
    clat, clon = centre_track(fixes, sel_times, motion, sigma_h=sigma_h)
    lats = np.asarray(ov.lats, float)[idx]
    lons = np.asarray(ov.lons, float)[idx]
    x, y = local_xy(clat, clon, lats, lons)
    ok = np.isfinite(x) & np.isfinite(y)
    r = np.hypot(x, y)
    ok &= r <= max_km
    idx, x, y, r = idx[ok], x[ok], y[ok], r[ok]
    clat, clon = clat[ok], clon[ok]
    az = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    wspd = np.array([np.nan if ov.winds[i] is None else ov.winds[i] for i in idx], float)
    wdir = np.array([np.nan if ov.dirs[i] is None else ov.dirs[i] for i in idx], float)
    flags = np.asarray(ov.column("wind_flagged", False), bool)[idx]
    if not include_flagged:
        wspd[flags] = np.nan
    sfc_all = surface.overlay_surface(ov, include_flagged=include_flagged)
    sfc = sfc_all[idx] if sfc_all.size else np.full(idx.size, np.nan)
    level_all = surface.level_flight(ov)
    level = level_all[idx] if level_all.size else np.ones(idx.size, bool)
    rad = np.radians(wdir)
    u, v = -wspd * np.sin(rad), -wspd * np.cos(rad)        # wind FROM wdir blows toward wdir+180
    mu = mv = 0.0
    if motion is not None:
        mu, mv = motion[1] * math.sin(math.radians(motion[0])), motion[1] * math.cos(math.radians(motion[0]))
    us, vs = u - mu, v - mv
    if rotate and motion is not None:
        b = math.radians(motion[0])
        cb, sb = math.cos(b), math.sin(b)
        x, y = x * cb - y * sb, x * sb + y * cb
        u, v = u * cb - v * sb, u * sb + v * cb
        us, vs = us * cb - vs * sb, us * sb + vs * cb
    press_all = ov.column("fl_press")
    press = np.array([np.nan if press_all[i] is None else press_all[i] for i in idx], float)
    labels = ov.column("labels", "")
    segs = np.asarray(ov.column("segments", 0), int)[idx]
    return StormRelative(times=[times[i] for i in idx], t=stamps[idx], x=x, y=y, r=r, az=az, wspd=wspd,
                         sfc=sfc, u=u, v=v, us=us, vs=vs, fl_press=press, labels=[labels[i] for i in idx],
                         idx=idx, motion=motion, rotated=bool(rotate and motion is not None),
                         centre_lat=clat, centre_lon=clon, segments=segs, level=level)


# ------------------------------------------------------------------- analysis

@dataclass
class Analysis:
    r: np.ndarray                 # ring centres, km
    az: np.ndarray                # bearing of each spoke, deg (in the plot frame)
    speed: np.ndarray             # (n_az, n_r) kt, NaN where unsupported
    weight: np.ndarray            # summed Gaussian weight (data support)
    which: str                    # "sfc" or "fl"
    radial_km: float
    arc_km: float

    def xy(self):
        """Cell-edge X, Y (km) for pcolormesh in the plot frame."""
        dr = self.r[1] - self.r[0] if self.r.size > 1 else 5.0
        da = self.az[1] - self.az[0] if self.az.size > 1 else 5.0
        re = np.concatenate([self.r - dr / 2, [self.r[-1] + dr / 2]]).clip(min=0)
        ae = np.radians(np.concatenate([self.az - da / 2, [self.az[-1] + da / 2]]))
        R, A = np.meshgrid(re, ae)
        return R * np.sin(A), R * np.cos(A)

    def rmw(self) -> Optional[float]:
        """Radius of the strongest azimuthal mean (where at least half the circle has data)."""
        cover = np.isfinite(self.speed).mean(axis=0)
        with np.errstate(invalid="ignore"):
            mean = np.where(cover >= 0.5, np.nanmean(np.where(np.isfinite(self.speed), self.speed, np.nan), axis=0),
                            np.nan) if self.speed.size else np.array([])
        if not np.isfinite(mean).any():
            return None
        return float(self.r[int(np.nanargmax(mean))])


def analyse(sr: StormRelative, which: str = "sfc", max_km: float = 250.0, dr_km: float = 5.0,
            daz: float = 5.0, radial_km: float = 10.0, arc_km: float = 35.0,
            min_weight: float = 0.6) -> Optional[Analysis]:
    """
    Single-pass Gaussian analysis on a polar grid. The radial scale grows
    slowly outward (radial_km, or 8% of the radius if larger); the
    along-circle scale is arc_km. A grid point needs a summed weight of
    `min_weight` -- roughly an ob within one length scale -- or it is blank.
    """
    if sr is None or not len(sr):
        return None
    spd = sr.speed(which)
    ok = np.isfinite(spd)
    if ok.sum() < 5:
        return None
    ox, oy = sr.x[ok], sr.y[ok]
    orr = np.hypot(ox, oy)
    oa = np.arctan2(ox, oy)                              # plot-frame bearing, radians
    ov = spd[ok]
    r = np.arange(dr_km / 2, max_km, dr_km)
    az = np.arange(0.0, 360.0, daz) + daz / 2
    ga = np.radians(az)
    out = np.full((az.size, r.size), np.nan)
    wsum = np.zeros((az.size, r.size))
    for j, rr in enumerate(r):
        lr = max(radial_km, 0.08 * rr)
        near = np.abs(orr - rr) <= 3 * lr
        if not near.any():
            continue
        dr = (orr[near] - rr) / lr                       # (k,)
        mean_r = 0.5 * (orr[near] + rr)
        dth = (oa[near][None, :] - ga[:, None] + np.pi) % (2 * np.pi) - np.pi   # (n_az, k)
        ds = dth * np.maximum(mean_r, 1.0)[None, :] / arc_km
        w = np.exp(-0.5 * (dr[None, :] ** 2 + ds ** 2))
        ws = w.sum(axis=1)
        wsum[:, j] = ws
        good = ws >= min_weight
        out[good, j] = (w[good] * ov[near][None, :]).sum(axis=1) / ws[good]
    return Analysis(r=r, az=az, speed=out, weight=wsum, which=which, radial_km=radial_km, arc_km=arc_km)


def analysis_at(an: Analysis, x: float, y: float) -> Optional[float]:
    """The analysed speed at plot-frame (x, y) km, or None."""
    if an is None:
        return None
    rr = math.hypot(x, y)
    a = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
    j = int(np.argmin(np.abs(an.r - rr)))
    if abs(an.r[j] - rr) > (an.r[1] - an.r[0]):
        return None
    i = int(np.argmin(np.abs(((an.az - a) + 180) % 360 - 180)))
    v = an.speed[i, j]
    return float(v) if np.isfinite(v) else None


# ------------------------------------------------------------------ radii

@dataclass
class Radius:
    threshold: int
    quadrant: str
    obs_km: Optional[float] = None          # farthest ob reaching the threshold
    obs_lower_bound: bool = False           # ... still above it at the end of the sampling
    sampled_km: Optional[float] = None      # farthest ob in the quadrant with a usable wind
    n_obs: int = 0
    analysis_km: Optional[float] = None
    analysis_lower_bound: bool = False

    @staticmethod
    def nm(km: Optional[float]) -> Optional[float]:
        return None if km is None else km / KM_PER_NM


def _quadrant_mask(az_deg: np.ndarray, q: str) -> np.ndarray:
    k = QUADRANTS.index(q)
    return (az_deg >= 90 * k) & (az_deg < 90 * (k + 1))


def wind_radii(sr: StormRelative, an: Optional[Analysis] = None, which: str = "sfc",
               thresholds: Sequence[int] = THRESHOLDS) -> List[Radius]:
    """
    Radii by geographic quadrant (NE, SE, SW, NW, as NHC reports them),
    from the obs and, when given, the analysis. Wind radii are defined on
    earth-relative surface winds; with which="fl" they are flight-level.
    """
    out = []
    spd = sr.speed(which) if sr is not None else np.empty(0)
    for thr in thresholds:
        for q in QUADRANTS:
            rad = Radius(thr, q)
            if sr is not None and len(sr):
                inq = _quadrant_mask(sr.az, q) & np.isfinite(spd)
                rad.n_obs = int(inq.sum())
                if inq.any():
                    rad.sampled_km = float(sr.r[inq].max())
                    hit = inq & (spd >= thr)
                    if hit.any():
                        rad.obs_km = float(sr.r[hit].max())
                        # still at or above the threshold near the end of the sampled leg?
                        beyond = inq & (sr.r > rad.obs_km)
                        rad.obs_lower_bound = (not beyond.any()) or \
                            (rad.sampled_km - rad.obs_km) < max(15.0, 0.1 * rad.obs_km)
            if an is not None:
                geo_az = an.az if not sr.rotated else (an.az + sr.motion[0]) % 360.0
                rows = _quadrant_mask(geo_az, q)
                block = an.speed[rows]
                if block.size and np.isfinite(block).any():
                    reach = np.isfinite(block) & (block >= thr)
                    cols = np.flatnonzero(reach.any(axis=0))
                    if cols.size:
                        rad.analysis_km = float(an.r[cols.max()])
                        valid = np.flatnonzero(np.isfinite(block).any(axis=0))
                        rad.analysis_lower_bound = bool(valid.size and valid.max() <= cols.max() + 1)
            out.append(rad)
    return out


def radii_table(radii: Sequence[Radius]) -> Dict[Tuple[int, str], Radius]:
    return {(r.threshold, r.quadrant): r for r in radii}


def peak(sr: StormRelative, which: str = "fl") -> Optional[Tuple[float, float, float, dt.datetime]]:
    """(speed, r km, bearing, time) of the strongest wind in the window."""
    if sr is None or not len(sr):
        return None
    spd = sr.speed(which)
    if not np.isfinite(spd).any():
        return None
    k = int(np.nanargmax(spd))
    return float(spd[k]), float(sr.r[k]), float(sr.az[k]), sr.times[k]
