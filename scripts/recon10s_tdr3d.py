#!/usr/bin/env python3
"""
recon10s_tdr3d — three-dimensional structure from a gridded TDR analysis
(experimental).

  * Vortex column: the centre at each height, as the centroid of positive
    relative vorticity within 30 km of the centre found one level below
    (iterated to convergence, starting from the analysis origin at the
    lowest level with data). The line through them is the vortex tilt.
  * Inner-core shear: the vector difference between the area-mean
    horizontal wind at two heights inside a disc around the centre, after
    removing the axisymmetric vortex at each height (otherwise uneven radar
    coverage lets the vortex itself leak into the mean as false shear). The
    TDR sees only ~150 km out, so this is vortex-scale shear, not the
    0-500 km (or 200-800 km) environmental shear SHIPS reports. Uncertainty is estimated by
    jackknife: the shear recomputed eight times, each leaving out one 45 deg
    sector of the disc; the spread says how much the answer depends on
    where the radar happened to see.
  * Shear-relative structure: reflectivity and vertical velocity in the
    four shear-relative quadrants, and the phase of the azimuthal
    wavenumber-1 asymmetry relative to the shear heading. Sheared storms
    usually put their convection downshear and downshear-left, and tilt
    downshear to downshear-left; agreement between these is a physical
    consistency check on the shear direction.
  * SHIPS: NHC's large-scale diagnostics for the nearest cycle
    (atcf/lsdiag/<cycle><basin><nn><yy>_lsdiag.dat), as an independent
    reference: SHDC/SDDC, the 850-200 hPa shear averaged 0-500 km with the
    vortex removed -- the shear printed on SHIPS's text product -- and
    SHRD/SHTD, the same over 200-800 km without vortex removal. The lsdiag
    headings are the direction the shear vector points toward ("westerly
    shear = 90 deg"). The text product's "SHEAR DIR" is the opposite,
    meteorological convention: the direction the shear comes FROM (checked:
    Isaias 8 Oct 12Z, SDDC 70 deg and SHEAR DIR 250; Polo 25 Sep 18Z, SDDC
    246 and SHEAR DIR 66). recon10s 2.2.1 read SHEAR DIR as a heading toward,
    which turned every SHIPS shear direction around by 180 deg; when lsdiag is
    not there the text product is used, converted. Atlantic and eastern/
    central Pacific only.
"""
from __future__ import annotations

import datetime as dt
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

MS_TO_KT = 1.943844
# Bumped whenever a diagnostic's definition changes; caches of measured passes key on it.
ALGORITHM_VERSION = "2.3.0-column2"
SHIPS_URL = "https://ftp.nhc.noaa.gov/atcf/stext/{stamp}{basin}{num}{yy}_ships.txt"
LSDIAG_URL = "https://ftp.nhc.noaa.gov/atcf/lsdiag/{stamp}{basin}{num}{yy}_lsdiag.dat"


def heading(u: float, v: float) -> float:
    """Direction a vector points, degrees clockwise from north (SHIPS lsdiag's SDDC/SHTD convention)."""
    return (math.degrees(math.atan2(u, v)) + 360.0) % 360.0


def angle_diff(a: float, b: float) -> float:
    """a - b, wrapped to (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


# -------------------------------------------------------------- vortex column

@dataclass
class Column:
    z: np.ndarray                 # km
    x: np.ndarray                 # centre, km east of the analysis origin (NaN where not found)
    y: np.ndarray


def _centroid(X, Y, layer, x0, y0, radius_km, max_iter=12):
    """Iterated centroid of positive vorticity within radius_km of (x0, y0); None if too little."""
    for _ in range(max_iter):
        near = (np.hypot(X - x0, Y - y0) <= radius_km) & np.isfinite(layer) & (layer > 0)
        if near.sum() < 20:
            return None
        w = layer[near]
        nx_, ny_ = float((w * X[near]).sum() / w.sum()), float((w * Y[near]).sum() / w.sum())
        moved = math.hypot(nx_ - x0, ny_ - y0)
        x0, y0 = nx_, ny_
        if moved < 0.25:
            break
    return x0, y0


def _cover(X, Y, layer, x0, y0, radius_km) -> float:
    disc = np.hypot(X - x0, Y - y0) <= radius_km
    return float(np.isfinite(layer[disc]).mean()) if disc.any() else 0.0


def vortex_column(grid, radius_km: float = 30.0, max_iter: int = 12,
                  max_step_km: float = 10.0, start_km: Tuple[float, float] = (1.5, 3.0),
                  max_origin_km: float = 20.0, min_cover: float = 0.3) -> Column:
    """
    The centre at each height. It starts at the best-sampled level between
    1.5 and 3 km (the lowest levels are often sparse near the centre), where
    the centroid may move at most `max_origin_km` from the analysis origin --
    HRD's own centre -- so it cannot wander off to a data-rich patch at the
    edge. From there it climbs and descends a level at a time, each search
    starting from the last centre found. A level is accepted when 30 km
    around its centre is at least 30% sampled and the centre moved no more
    than `max_step_km` from the last accepted one; a level that fails is
    skipped, and three failures in a row end the column in that direction.
    """
    X, Y = np.meshgrid(grid.x, grid.y)
    vort = grid.fields["VORT"]
    z = grid.z
    cx = np.full(z.size, np.nan)
    cy = np.full(z.size, np.nan)

    starts = [k for k in range(z.size) if start_km[0] - 1e-6 <= z[k] <= start_km[1] + 1e-6]
    starts.sort(key=lambda k: -_cover(X, Y, vort[k], 0.0, 0.0, radius_km))
    starts += [k for k in range(z.size) if k not in starts]            # any level, as a last resort
    k0 = None
    for k in starts:
        layer = vort[k]
        if np.isfinite(layer).sum() < 50:
            continue
        c = _centroid(X, Y, layer, 0.0, 0.0, radius_km, max_iter)
        if c is None or math.hypot(*c) > max_origin_km or _cover(X, Y, layer, *c, radius_km) < min_cover:
            continue
        k0 = k
        cx[k], cy[k] = c
        break
    if k0 is None:
        return Column(z=z, x=cx, y=cy)

    for step in (1, -1):
        px, py = cx[k0], cy[k0]
        misses = 0
        k = k0 + step
        while 0 <= k < z.size and misses < 3:
            layer = vort[k]
            c = _centroid(X, Y, layer, px, py, radius_km, max_iter) if np.isfinite(layer).sum() >= 50 else None
            # A real vortex does not jump tens of km between levels 0.5 km apart;
            # a big jump means scattered echo (usually near the top of the data).
            if c is not None and _cover(X, Y, layer, *c, radius_km) >= min_cover and \
                    math.hypot(c[0] - px, c[1] - py) <= max_step_km * (1 + misses):
                cx[k], cy[k] = c
                px, py = c
                misses = 0
            else:
                misses += 1
            k += step
    return Column(z=z, x=cx, y=cy)


def tilt(col: Column, z_lo: float, z_hi: float, slack_km: float = 1.0) -> Optional[Tuple[float, float]]:
    """
    (magnitude km, heading deg) of the upper centre relative to the lower, or
    None. Where a level has no centre, the nearest one within slack_km is used.
    """
    def at(zz):
        ok = np.flatnonzero(np.isfinite(col.x))
        if not ok.size:
            return None
        k = ok[int(np.argmin(np.abs(col.z[ok] - zz)))]
        return k if abs(col.z[k] - zz) <= slack_km + 1e-6 else None
    k0, k1 = at(z_lo), at(z_hi)
    if k0 is None or k1 is None or k0 == k1:
        return None
    dx, dy = col.x[k1] - col.x[k0], col.y[k1] - col.y[k0]
    return math.hypot(dx, dy), heading(dx, dy)


# --------------------------------------------------------------------- shear

@dataclass
class Shear:
    z_lo: float
    z_hi: float
    radius: float
    u: float                      # m/s, upper minus lower
    v: float
    kt: float
    heading: float
    kt_err: float                 # jackknife standard error
    heading_err: float
    cover_lo: float               # fraction of the disc with winds
    cover_hi: float
    grade: str
    mean_lo: Tuple[float, float]
    mean_hi: Tuple[float, float]

    @property
    def summary(self) -> str:
        return (f"{self.kt:.0f} \u00b1 {self.kt_err:.0f} kt toward {self.heading:03.0f}\u00b0 "
                f"\u00b1 {self.heading_err:.0f}\u00b0 ({self.z_lo:g}\u2013{self.z_hi:g} km, "
                f"{self.radius:.0f} km disc; confidence {self.grade})")


_RESIDUAL_CACHE: Dict[tuple, tuple] = {}


def residual_wind(grid, k, cx, cy, ring_km: float = 2.0):
    """
    U and V at level k with the axisymmetric vortex removed: the ring-mean
    tangential and radial winds about (cx, cy) are subtracted point by point.
    Averaging the raw winds instead lets a 70 m/s vortex leak into the mean
    wherever the radar's coverage is uneven, which masquerades as shear.
    """
    key = (id(grid), k, round(cx, 2), round(cy, 2))
    if key in _RESIDUAL_CACHE:
        return _RESIDUAL_CACHE[key]
    X, Y = np.meshgrid(grid.x, grid.y)
    dx, dy = X - cx, Y - cy
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    u, v = grid.fields["U"][k], grid.fields["V"][k]
    vt = -u * np.sin(th) + v * np.cos(th)
    vr = u * np.cos(th) + v * np.sin(th)
    ring = np.floor(r / ring_km).astype(int)
    ok = np.isfinite(vt) & np.isfinite(vr)
    nring = int(ring.max()) + 1
    cnt = np.bincount(ring[ok], minlength=nring)
    mvt = np.bincount(ring[ok], weights=vt[ok], minlength=nring) / np.maximum(cnt, 1)
    mvr = np.bincount(ring[ok], weights=vr[ok], minlength=nring) / np.maximum(cnt, 1)
    total = np.bincount(ring.ravel(), minlength=nring)
    usable = cnt >= np.maximum(8, 0.25 * total)          # a ring mean needs a quarter of the ring
    sym_vt = np.where(usable[ring], mvt[ring], np.nan)
    sym_vr = np.where(usable[ring], mvr[ring], np.nan)
    ur = u - (sym_vr * np.cos(th) - sym_vt * np.sin(th))
    vr_ = v - (sym_vr * np.sin(th) + sym_vt * np.cos(th))
    _RESIDUAL_CACHE[key] = (ur, vr_)
    if len(_RESIDUAL_CACHE) > 200:
        _RESIDUAL_CACHE.pop(next(iter(_RESIDUAL_CACHE)))
    return ur, vr_


def _disc_mean(grid, k, cx, cy, radius, sector=None, vortex_removed: bool = True):
    X, Y = np.meshgrid(grid.x, grid.y)
    if vortex_removed:
        u, v = residual_wind(grid, k, cx, cy)
    else:
        u, v = grid.fields["U"][k], grid.fields["V"][k]
    r = np.hypot(X - cx, Y - cy)
    disc = r <= radius
    if sector is not None:
        az = (np.degrees(np.arctan2(X - cx, Y - cy)) + 360) % 360
        disc &= ~((az >= sector * 45) & (az < (sector + 1) * 45))
    ok = disc & np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 30:
        return None
    return float(u[ok].mean()), float(v[ok].mean()), ok.sum() / max(disc.sum(), 1)


def mean_wind_profile(grid, col: Column, radius: float = 120.0, min_cover: float = 0.25):
    """(z, u, v, coverage) of the vortex-removed disc-mean wind where at least
    `min_cover` of the disc has winds (thinner levels are noise)."""
    out = []
    for k, z in enumerate(grid.z):
        cx = col.x[k] if np.isfinite(col.x[k]) else 0.0
        cy = col.y[k] if np.isfinite(col.y[k]) else 0.0
        m = _disc_mean(grid, k, cx, cy, radius)
        if m and m[2] >= min_cover:
            out.append((z, m[0], m[1], m[2]))
    return np.array(out) if out else np.empty((0, 4))


mean_wind_profile_removed = mean_wind_profile      # (vortex removal is the default)


def inner_core_shear(grid, col: Column, z_lo: float = 2.0, z_hi: float = 8.0,
                     radius: float = 120.0) -> Optional[Shear]:
    k0, k1 = int(np.argmin(abs(grid.z - z_lo))), int(np.argmin(abs(grid.z - z_hi)))

    def centre(k):
        return ((col.x[k], col.y[k]) if np.isfinite(col.x[k]) else (0.0, 0.0))

    lo = _disc_mean(grid, k0, *centre(k0), radius)
    hi = _disc_mean(grid, k1, *centre(k1), radius)
    if lo is None or hi is None:
        return None
    du, dv = hi[0] - lo[0], hi[1] - lo[1]
    # Jackknife over eight 45-degree sectors.
    reps = []
    for s in range(8):
        a = _disc_mean(grid, k0, *centre(k0), radius, sector=s)
        b = _disc_mean(grid, k1, *centre(k1), radius, sector=s)
        if a and b:
            reps.append((b[0] - a[0], b[1] - a[1]))
    reps = np.array(reps)
    n = len(reps)
    if n >= 4:
        mags = np.hypot(reps[:, 0], reps[:, 1]) * MS_TO_KT
        heads = np.array([heading(x, y) for x, y in reps])
        dh = np.array([angle_diff(h, heading(du, dv)) for h in heads])
        kt_err = float(np.sqrt((n - 1) / n * ((mags - mags.mean()) ** 2).sum()))
        hd_err = float(np.sqrt((n - 1) / n * ((dh - dh.mean()) ** 2).sum()))
    else:
        kt_err, hd_err = float("nan"), float("nan")
    kt = math.hypot(du, dv) * MS_TO_KT
    cover = min(lo[2], hi[2])
    if cover >= 0.6 and np.isfinite(kt_err) and kt_err < 0.35 * max(kt, 1) and hd_err < 30:
        grade = "high"
    elif cover >= 0.35 and np.isfinite(kt_err) and kt_err < 0.7 * max(kt, 1) and hd_err < 60:
        grade = "moderate"
    else:
        grade = "low"
    return Shear(z_lo=float(grid.z[k0]), z_hi=float(grid.z[k1]), radius=radius, u=du, v=dv, kt=kt,
                 heading=heading(du, dv), kt_err=kt_err, heading_err=hd_err, cover_lo=lo[2],
                 cover_hi=hi[2], grade=grade, mean_lo=(lo[0], lo[1]), mean_hi=(hi[0], hi[1]))


# ------------------------------------------------------ shear-relative layout

QUADRANTS = ("DL", "DR", "UR", "UL")        # downshear-left, downshear-right, upshear-right, upshear-left


@dataclass
class Asymmetry:
    field: str
    z: float
    quadrant: Dict[str, float]
    wn1_phase: Optional[float]      # deg from downshear, positive = to the left (counterclockwise)
    wn1_amplitude: Optional[float]  # fraction of the ring mean
    r1: float
    r2: float


def shear_relative(grid, col: Column, name: str, z: float, shear_heading: float,
                   r1: float = 10.0, r2: float = 100.0) -> Asymmetry:
    k = int(np.argmin(abs(grid.z - z)))
    cx = col.x[k] if np.isfinite(col.x[k]) else 0.0
    cy = col.y[k] if np.isfinite(col.y[k]) else 0.0
    X, Y = np.meshgrid(grid.x, grid.y)
    r = np.hypot(X - cx, Y - cy)
    az = (np.degrees(np.arctan2(X - cx, Y - cy)) + 360) % 360
    rel = np.array(angle_diff_vec(az, shear_heading))          # >0 clockwise (right of shear)
    vals = grid.fields[name][k].copy()
    if name == "REFLECTIVITY":
        vals = 10 ** (vals / 10.0)                              # average in linear Z, report dBZ
    ring = (r >= r1) & (r <= r2) & np.isfinite(vals)
    quads = {}
    for q, (a, b) in {"DR": (0, 90), "UR": (90, 180), "UL": (-180, -90), "DL": (-90, 0)}.items():
        sel = ring & (rel >= a) & (rel < b)
        m = float(vals[sel].mean()) if sel.sum() > 20 else float("nan")
        quads[q] = 10 * math.log10(m) if (name == "REFLECTIVITY" and m > 0) else m
    # Wavenumber-1 from 16 azimuth bins, each bin's mean weighted equally.
    phase = amp = None
    bins = np.full(16, np.nan)
    for i in range(16):
        sel = ring & (az >= i * 22.5) & (az < (i + 1) * 22.5)
        if sel.sum() > 10:
            bins[i] = vals[sel].mean()
    if np.isfinite(bins).sum() >= 10:
        th = np.radians(np.arange(16) * 22.5 + 11.25)
        good = np.isfinite(bins)
        a1 = (bins[good] * np.sin(th[good])).sum() * 2 / good.sum()
        b1 = (bins[good] * np.cos(th[good])).sum() * 2 / good.sum()
        a0 = bins[good].mean()
        max_heading = (math.degrees(math.atan2(a1, b1)) + 360) % 360
        phase = -angle_diff(max_heading, shear_heading)          # positive = left of downshear
        amp = float(math.hypot(a1, b1) / abs(a0)) if a0 else None
    return Asymmetry(field=name, z=float(grid.z[k]), quadrant=quads, wn1_phase=phase,
                     wn1_amplitude=amp, r1=r1, r2=r2)


def angle_diff_vec(a, b):
    return (np.asarray(a) - b + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- isosurfaces

def isosurface(grid, name: str, level: float, extent_km: float = 150.0, step: int = 1,
               smooth: Optional[Tuple[float, float, float]] = None):
    """
    (vertices (x, y, z km), faces) of one isosurface, or None; NaN treated as
    below the level. `smooth` is a Gaussian sigma in grid points (z, y, x),
    for ground-based radar volumes whose single-beam detail makes ragged
    surfaces.
    """
    from skimage.measure import marching_cubes
    vol = grid.fields[name]
    xs = np.flatnonzero(np.abs(grid.x) <= extent_km)[::step]
    ys = np.flatnonzero(np.abs(grid.y) <= extent_km)[::step]
    sub = vol[:, ys][:, :, xs]
    if not np.isfinite(sub).any() or np.nanmax(sub) < level:
        return None
    filled = np.where(np.isfinite(sub), sub, np.nanmin(sub) - 10)
    if smooth is not None:
        from scipy.ndimage import gaussian_filter
        filled = gaussian_filter(filled, smooth)
        if filled.max() < level:
            return None
    dz = float(grid.z[1] - grid.z[0])
    dy = float(grid.y[ys[1]] - grid.y[ys[0]])
    dx = float(grid.x[xs[1]] - grid.x[xs[0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        verts, faces, _n, _v = marching_cubes(filled, level=level, spacing=(dz, dy, dx))
    xyz = np.column_stack([verts[:, 2] + grid.x[xs[0]], verts[:, 1] + grid.y[ys[0]], verts[:, 0] + grid.z[0]])
    return xyz, faces


# ---------------------------------------------------------------------- SHIPS

@dataclass
class ShipsShear:
    cycle: dt.datetime
    kt: float                      # SHDC: 850-200 hPa, 0-500 km, vortex removed
    heading: float                 # direction the shear vector points toward
    url: str
    kt_200_800: Optional[float] = None        # SHRD
    heading_200_800: Optional[float] = None   # SHTD


def parse_ships(text: str) -> Optional[Tuple[float, float]]:
    """
    (shear kt, heading toward deg) at hour 0 from a SHIPS text product.
    The text's SHEAR DIR is where the shear comes from, so it is turned
    around (+180) into the heading the vector points toward.
    """
    kt = frm = None
    for line in text.splitlines():
        m = re.match(r"\s*SHEAR \(KT\)\s+(-?\d+)", line)
        if m:
            kt = float(m.group(1))
        m = re.match(r"\s*SHEAR DIR\s+(-?\d+)", line)
        if m:
            frm = float(m.group(1))
    return (kt, (frm + 180.0) % 360.0) if kt is not None and frm is not None else None


def parse_lsdiag(text: str) -> Optional[Dict[str, float]]:
    """
    Hour-0 SHDC, SDDC, SHRD and SHTD from a SHIPS lsdiag file: fixed 5-character
    columns from -12 h, so hour 0 is the third; shears in kt x 10, headings in
    degrees toward. Missing values are 9999.
    """
    out: Dict[str, float] = {}
    for line in text.splitlines():
        name = line[115:].strip() if len(line) > 115 else line.split()[-1] if line.split() else ""
        name = name.split()[0] if name else ""
        if name not in ("SHDC", "SDDC", "SHRD", "SHTD"):
            continue
        cell = line[10:15].strip()
        if not cell or not cell.lstrip("-").isdigit() or int(cell) == 9999:
            continue
        v = float(cell)
        out[name] = v / 10.0 if name in ("SHDC", "SHRD") else v % 360.0
    return out if "SHDC" in out and "SDDC" in out else None


def ships_shear(storm_id: str, when: dt.datetime, max_hours: float = 6.0) -> Optional[ShipsShear]:
    """SHIPS hour-0 shear from the synoptic cycle nearest `when` (within max_hours)."""
    m = re.fullmatch(r"(al|ep|cp)(\d{2})(\d{4})", storm_id.lower())
    if not m:
        return None
    basin, num, year = m.group(1).upper(), m.group(2), m.group(3)[2:]
    base = when.replace(minute=0, second=0, microsecond=0)
    base -= dt.timedelta(hours=base.hour % 6)
    for cyc in sorted((base, base + dt.timedelta(hours=6), base - dt.timedelta(hours=6)),
                      key=lambda c: abs((c - when).total_seconds())):
        if abs((cyc - when).total_seconds()) > max_hours * 3600:
            continue
        stamp = cyc.strftime("%y%m%d%H")
        url = LSDIAG_URL.format(stamp=stamp, basin=basin, num=num, yy=year)
        try:
            r = requests.get(url, timeout=20)
        except requests.RequestException:
            r = None
        if r is not None and r.ok:
            got = parse_lsdiag(r.text)
            if got:
                return ShipsShear(cycle=cyc, kt=got["SHDC"], heading=got["SDDC"], url=url,
                                  kt_200_800=got.get("SHRD"), heading_200_800=got.get("SHTD"))
        url = SHIPS_URL.format(stamp=stamp, basin=basin, num=num, yy=year)
        try:
            r = requests.get(url, timeout=20)
        except requests.RequestException:
            continue
        if r.ok:
            got = parse_ships(r.text)
            if got:
                return ShipsShear(cycle=cyc, kt=got[0], heading=got[1], url=url)
    return None
