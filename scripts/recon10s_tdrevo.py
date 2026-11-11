#!/usr/bin/env python3
"""
recon10s_tdrevo — how the inner core changes from one TDR pass to the next.

For every gridded analysis of one or more P-3 missions (seb.omao.noaa.gov,
recon10s_tdrgrid), the same handful of numbers:

  * peak wind: the strongest wind at 2 km, the strongest anywhere in the
    lowest 5 km (and its height), and the strongest azimuthal-mean
    tangential wind at 2 km;
  * RMW: radius of the azimuthal-mean tangential-wind maximum at 2 and 8 km
    (its outward slope with height);
  * tilt: the vortex column's displacement between two heights
    (recon10s_tdr3d, centroids of vorticity);
  * inner-core shear between the same heights, vortex removed, with its
    jackknife uncertainty and confidence grade, and optionally SHIPS's
    environmental shear for the nearest cycle.

A grid is ~200 MB in memory, so each is loaded, measured and dropped; the
numbers are cached on disk by file name and settings, so a mission is only
ever measured once.
"""
from __future__ import annotations

import datetime as dt
import gc
import hashlib
import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import recon10s_tdr3d as t3
import recon10s_tdrgrid as tg

CACHE_DIR = os.path.join(tg.CACHE_DIR, "evolution")
MS_TO_KT = tg.MS_TO_KT
VERSION = 1            # bump when a metric's definition changes, to invalidate the cache


@dataclass
class PassMetrics:
    mission: str
    name: str                                 # analysis file name
    time: Optional[str] = None                # ISO, analysis centre time
    start: Optional[str] = None
    end: Optional[str] = None
    storm: str = ""
    centre_lat: Optional[float] = None
    centre_lon: Optional[float] = None
    peak_2km_kt: Optional[float] = None
    peak_low_kt: Optional[float] = None       # strongest in the lowest 5 km
    peak_low_z: Optional[float] = None
    vt_max_2km_kt: Optional[float] = None     # strongest azimuthal-mean tangential wind at 2 km
    rmw_2km: Optional[float] = None
    rmw_8km: Optional[float] = None
    tilt_km: Optional[float] = None
    tilt_heading: Optional[float] = None
    column_levels: int = 0
    shear_kt: Optional[float] = None
    shear_kt_err: Optional[float] = None
    shear_heading: Optional[float] = None
    shear_heading_err: Optional[float] = None
    shear_grade: str = ""
    shear_cover: Optional[float] = None
    ships_kt: Optional[float] = None
    ships_heading: Optional[float] = None
    ships_cycle: Optional[str] = None
    z_lo: float = 2.0
    z_hi: float = 8.0
    radius: float = 120.0
    error: str = ""
    available: Optional[str] = None           # ISO: when the analysis reached the server

    @property
    def when(self) -> Optional[dt.datetime]:
        return dt.datetime.fromisoformat(self.time) if self.time else None

    @property
    def label(self) -> str:
        w = self.when
        return f"{self.mission} {w:%d/%H:%MZ}" if w else f"{self.mission} {self.name}"


def _f(x) -> Optional[float]:
    return float(x) if x is not None and np.isfinite(x) else None


def measure(grid, z_lo: float = 2.0, z_hi: float = 8.0, radius: float = 120.0) -> PassMetrics:
    """Every number for one loaded TdrGrid."""
    g = grid
    m = PassMetrics(mission=g.flight, name=g.name, storm=g.storm, z_lo=z_lo, z_hi=z_hi, radius=radius)
    when = g.centre_time or g.start
    m.time = when.isoformat() if when else None
    m.start = g.start.isoformat() if g.start else None
    m.end = g.end.isoformat() if g.end else None
    m.centre_lat, m.centre_lon = float(g.centre[0]), float(g.centre[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws = g.fields.get("WIND_SPEED")
        if ws is not None:
            k2 = g.level(2.0)
            m.peak_2km_kt = _f(np.nanmax(ws[k2]) * MS_TO_KT) if np.isfinite(ws[k2]).any() else None
            low = g.z <= 5.0
            sub = ws[low]
            if np.isfinite(sub).any():
                flat = np.nanargmax(sub)
                kk = np.unravel_index(flat, sub.shape)[0]
                m.peak_low_kt = _f(sub.flat[flat] * MS_TO_KT)
                m.peak_low_z = float(g.z[low][kk])
        if "TANGENTIAL_WIND" in g.fields:
            r, z, vt = g.azimuthal_mean("TANGENTIAL_WIND")
            k2, k8 = g.level(2.0), g.level(8.0)
            if np.isfinite(vt[k2]).any():
                m.vt_max_2km_kt = _f(np.nanmax(vt[k2]) * MS_TO_KT)
                m.rmw_2km = float(r[int(np.nanargmax(vt[k2]))])
            if np.isfinite(vt[k8]).any():
                m.rmw_8km = float(r[int(np.nanargmax(vt[k8]))])
        if "VORT" in g.fields:
            col = t3.vortex_column(g)
            m.column_levels = int(np.isfinite(col.x).sum())
            tl = t3.tilt(col, z_lo, z_hi)
            if tl:
                m.tilt_km, m.tilt_heading = float(tl[0]), float(tl[1])
            if "U" in g.fields and "V" in g.fields:
                sh = t3.inner_core_shear(g, col, z_lo, z_hi, radius)
                if sh:
                    m.shear_kt, m.shear_heading = float(sh.kt), float(sh.heading)
                    m.shear_kt_err, m.shear_heading_err = _f(sh.kt_err), _f(sh.heading_err)
                    m.shear_grade, m.shear_cover = sh.grade, float(min(sh.cover_lo, sh.cover_hi))
    return m


def _key(url: str, z_lo: float, z_hi: float, radius: float) -> str:
    raw = f"{VERSION}|{t3.ALGORITHM_VERSION}|{url}|{z_lo:g}|{z_hi:g}|{radius:g}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def cached(url: str, z_lo: float, z_hi: float, radius: float) -> Optional[PassMetrics]:
    path = os.path.join(CACHE_DIR, _key(url, z_lo, z_hi, radius) + ".json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return PassMetrics(**json.load(fh))
    except (OSError, ValueError, TypeError):
        return None


def _store(url: str, m: PassMetrics):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _key(url, m.z_lo, m.z_hi, m.radius) + ".json")
    with open(path + ".part", "w", encoding="utf-8") as fh:
        json.dump(asdict(m), fh)
    os.replace(path + ".part", path)


def pass_metrics(analysis, z_lo: float = 2.0, z_hi: float = 8.0, radius: float = 120.0,
                 use_cache: bool = True, loader: Callable = None) -> PassMetrics:
    """Metrics for one tg.Analysis (download, measure, forget the grid)."""
    url = analysis.url(analysis.xy)
    avail = getattr(analysis, "available", None)
    if use_cache:
        hit = cached(url, z_lo, z_hi, radius)
        if hit is not None:
            hit.available = hit.available or (avail.isoformat() if avail else None)
            return hit
    try:
        grid = (loader or tg.load_grid)(url)
    except Exception as exc:
        return PassMetrics(mission=analysis.mission, name=analysis.xy, z_lo=z_lo, z_hi=z_hi, radius=radius,
                           error=f"{type(exc).__name__}: {exc}")
    m = measure(grid, z_lo, z_hi, radius)
    m.mission = analysis.mission
    m.available = avail.isoformat() if avail else None
    del grid
    gc.collect()
    if use_cache and not m.error:
        try:
            _store(url, m)
        except OSError:
            pass
    return m


def evolution(analyses: Sequence, z_lo: float = 2.0, z_hi: float = 8.0, radius: float = 120.0,
              progress: Optional[Callable[[str], None]] = None, use_cache: bool = True,
              loader: Callable = None) -> List[PassMetrics]:
    """Metrics for every analysis, oldest first."""
    out = []
    for k, a in enumerate(analyses, 1):
        if progress:
            progress(f"Measuring {a.mission} {a.time[:2]}:{a.time[2:]}Z ({k} of {len(analyses)})…")
        out.append(pass_metrics(a, z_lo, z_hi, radius, use_cache, loader))
    out.sort(key=lambda m: (m.time or "", m.name))
    return out


def add_ships(metrics: Sequence[PassMetrics], storm_id: str,
              progress: Optional[Callable[[str], None]] = None) -> None:
    """SHIPS hour-0 shear for each pass's nearest cycle (fetched once per cycle)."""
    seen: Dict[str, object] = {}
    for m in metrics:
        w = m.when
        if w is None:
            continue
        base = w.replace(minute=0, second=0, microsecond=0)
        base -= dt.timedelta(hours=base.hour % 6)
        key = base.isoformat()
        if key not in seen:
            if progress:
                progress(f"SHIPS for {base:%d/%HZ}…")
            try:
                seen[key] = t3.ships_shear(storm_id, w)
            except Exception:
                seen[key] = None
        sp = seen[key]
        if sp is not None:
            m.ships_kt, m.ships_heading, m.ships_cycle = sp.kt, sp.heading, sp.cycle.isoformat()


def summary(metrics: Sequence[PassMetrics]) -> str:
    """One line on the change from the first pass to the last."""
    ok = [m for m in metrics if not m.error and m.when]
    if len(ok) < 2:
        return ""
    a, b = ok[0], ok[-1]
    hours = (b.when - a.when).total_seconds() / 3600
    bits = []
    if a.peak_2km_kt and b.peak_2km_kt:
        bits.append(f"2 km peak wind {a.peak_2km_kt:.0f} → {b.peak_2km_kt:.0f} kt")
    if a.rmw_2km and b.rmw_2km:
        bits.append(f"RMW {a.rmw_2km:.0f} → {b.rmw_2km:.0f} km")
    if a.tilt_km is not None and b.tilt_km is not None:
        bits.append(f"tilt {a.tilt_km:.0f} → {b.tilt_km:.0f} km")
    if a.shear_kt is not None and b.shear_kt is not None:
        bits.append(f"inner-core shear {a.shear_kt:.0f} → {b.shear_kt:.0f} kt")
    return f"Over {hours:.0f} h and {len(ok)} passes: " + ", ".join(bits) + "."
