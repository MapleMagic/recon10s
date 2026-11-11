#!/usr/bin/env python3
"""
recon10s_nexgrid — a ground-based radar volume (NEXRAD Level II, or a JMA
cycle) on a storm-centred 3-D grid, so the airborne-TDR tools work on it:
CAPPIs at any height, vertical sections through the centre or along any
line, azimuthal means around the eye, and 3-D isosurfaces.

Method (the usual one behind CAPPIs and volume renderers):
  1. Each grid column's ground range and azimuth from the radar, from the
     great-circle geometry (the grid is azimuthal-equidistant about the
     storm centre).
  2. For every elevation angle, the nearest ray and gate to each column at
     that column's ground range (slant range = ground range / cos(elevation)),
     and the beam's height there (4/3-earth model, plus the antenna height).
  3. At each grid height, linear interpolation between the two elevation
     angles whose beams bracket it. Below the lowest beam and above the
     highest, a value is kept only within half a beamwidth of the beam
     centre; where two beams are more than `max_gap_km` apart between their
     edges, only their own half-beamwidths are filled -- nothing is invented
     in the gap. NEXRAD's repeated low-level cuts (SAILS/MESO-SAILS) are
     counted once.
Reflectivity is interpolated in dBZ. Radial velocity is the component along
each beam, so it is gridded for display (sections, plan views) but cannot
make a 3-D wind; storm-relative velocity removes the storm motion first.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import recon10s_nexrad as nx
import recon10s_tdrgrid as tg

EARTH_KM = 6371.0
AE_KM = 4.0 / 3.0 * EARTH_KM

# grid field name -> (product, label, units)
RADAR_FIELDS: Dict[str, Tuple[str, str, str]] = {
    "REFLECTIVITY": ("reflectivity", "Reflectivity", "dBZ"),
    "RADIAL_VELOCITY": ("velocity", "Radial velocity", "kt"),
    "SR_VELOCITY": ("srv", "Storm-relative velocity", "kt"),
    "ZDR": ("zdr", "Differential reflectivity", "dB"),
    "CC": ("cc", "Correlation coefficient", ""),
}
PRODUCT_FIELD = {v[0]: k for k, v in RADAR_FIELDS.items()}


def field_label(name: str) -> Tuple[str, str]:
    """(label, units) for a TDR or radar grid field."""
    if name in RADAR_FIELDS:
        return RADAR_FIELDS[name][1], RADAR_FIELDS[name][2]
    return tg.FIELDS.get(name, (name.title(), ""))


def _destinations(lat0: float, lon0: float, x_km: np.ndarray, y_km: np.ndarray):
    """lat/lon of points x km east, y km north of (lat0, lon0), azimuthal-equidistant."""
    d = np.hypot(x_km, y_km) / EARTH_KM
    b = np.arctan2(x_km, y_km)
    la1, lo1 = math.radians(lat0), math.radians(lon0)
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(b))
    lo2 = lo1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(la1), np.cos(d) - np.sin(la1) * np.sin(la2))
    return np.degrees(la2), (np.degrees(lo2) + 540.0) % 360.0 - 180.0


def range_azimuth(site_lat: float, site_lon: float, lat: np.ndarray, lon: np.ndarray):
    """Great-circle ground range (km) and azimuth (deg) from the radar to each point."""
    p1, p2 = math.radians(site_lat), np.radians(lat)
    dl = np.radians(lon - site_lon)
    a = np.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    s = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    az = np.degrees(np.arctan2(np.sin(dl) * np.cos(p2),
                               math.cos(p1) * np.sin(p2) - math.sin(p1) * np.cos(p2) * np.cos(dl)))
    return s, (az + 360.0) % 360.0


def beam_height(s_km: np.ndarray, elevation: float, site_alt_km: float = 0.0) -> np.ndarray:
    """Beam-centre height (km above sea level) at ground range s for an elevation angle."""
    r = s_km / math.cos(math.radians(elevation))
    return np.sqrt(r ** 2 + AE_KM ** 2 + 2 * r * AE_KM * math.sin(math.radians(elevation))) - AE_KM + site_alt_km


def sample_sweep(az: np.ndarray, rng_m: np.ndarray, vals: np.ndarray, elevation: float,
                 s_km: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Nearest ray and gate of one sweep at each (ground range, azimuth); NaN outside it."""
    order = np.argsort(az)
    az_sorted = az[order]
    idx = np.searchsorted(az_sorted, phi)
    lo = (idx - 1) % az_sorted.size
    hi = idx % az_sorted.size
    d_lo = np.abs(((phi - az_sorted[lo]) + 180) % 360 - 180)
    d_hi = np.abs(((az_sorted[hi] - phi) + 180) % 360 - 180)
    ray = order[np.where(d_lo <= d_hi, lo, hi)]
    beam = np.minimum(d_lo, d_hi)
    spacing = 360.0 / max(az.size, 1)
    slant_m = s_km * 1e3 / math.cos(math.radians(elevation))
    dr = float(rng_m[1] - rng_m[0]) if rng_m.size > 1 else 250.0
    gate = np.rint((slant_m - rng_m[0]) / dr).astype(int)
    ok = (gate >= 0) & (gate < rng_m.size) & (beam <= 1.5 * spacing)
    out = np.full(s_km.shape, np.nan)
    out[ok] = vals[ray[ok], gate[ok]]
    return out


def sweeps_for(vol, product: str) -> List[Tuple[float, int]]:
    """(elevation, sweep) for each distinct elevation that carries the product, lowest first."""
    base = "velocity" if product == "srv" else product
    out = []
    for el in vol.elevations():
        i = vol.sweep_for(base, el)
        if i is not None:
            out.append((el, i))
    return out


def srv_values(az: np.ndarray, vals_kt: np.ndarray, motion: Optional[Tuple[float, float]]) -> np.ndarray:
    """Radial velocity (kt) with the storm motion's component along each ray removed."""
    if motion is None:
        return vals_kt
    hd, kt = motion
    return vals_kt - kt * np.cos(np.radians(az - hd))[:, None]


def vertical_interp(V: np.ndarray, H: np.ndarray, half: np.ndarray, z: np.ndarray,
                    max_gap_km: float = 2.0) -> np.ndarray:
    """
    V, H, half: (n_sweeps, ny, nx) values, beam heights and half-beamwidths
    (km), sweeps ordered by elevation. Returns (nz, ny, nx).
    """
    n = V.shape[0]
    out = np.full((z.size,) + V.shape[1:], np.nan)
    if n == 0:
        return out
    Hs = np.where(np.isfinite(H), H, np.inf)
    for k, zz in enumerate(z):
        above = (Hs > zz).sum(axis=0)                  # sweeps whose beam centre is above zz
        k_hi = n - above                               # index of the first sweep above (n = none)
        k_lo = k_hi - 1
        kh = np.clip(k_hi, 0, n - 1)
        kl = np.clip(k_lo, 0, n - 1)
        vh = np.take_along_axis(V, kh[None], 0)[0]
        vl = np.take_along_axis(V, kl[None], 0)[0]
        hh = np.take_along_axis(H, kh[None], 0)[0]
        hl = np.take_along_axis(H, kl[None], 0)[0]
        wh = np.take_along_axis(half, kh[None], 0)[0]
        wl = np.take_along_axis(half, kl[None], 0)[0]
        res = np.full(V.shape[1:], np.nan)
        inside = (k_lo >= 0) & (k_hi <= n - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.clip((zz - hl) / (hh - hl), 0.0, 1.0)
            both = inside & np.isfinite(vl) & np.isfinite(vh) & ((hh - wh) - (hl + wl) <= max_gap_km)
            res[both] = (vl + f * (vh - vl))[both]
            near_lo = inside & ~both & np.isfinite(vl) & (zz <= hl + wl)
            res[near_lo] = vl[near_lo]
            near_hi = inside & ~both & np.isfinite(vh) & (zz >= hh - wh)
            res[near_hi] = vh[near_hi]
            below = (k_hi == 0) & np.isfinite(vh) & (zz >= hh - wh)
            res[below] = vh[below]
            top = (k_lo == n - 1) & np.isfinite(vl) & (zz <= hl + wl)
            res[top] = vl[top]
        out[k] = res
    return out


def grid_volume(vol, centre: Optional[Tuple[float, float]] = None, half_width_km: float = 150.0,
                dx_km: float = 1.0, z_km: Sequence[float] = tuple(np.arange(0.5, 15.01, 0.5)),
                products: Sequence[str] = ("reflectivity", "velocity", "srv", "zdr", "cc"),
                motion: Optional[Tuple[float, float]] = None, dealias: bool = True,
                site_alt_m: Optional[float] = None, beamwidth: float = 0.95, storm: str = "",
                max_gap_km: float = 2.0, progress=None) -> tg.TdrGrid:
    """
    Grid a NEXRAD Volume or JmaVolume about `centre` (lat, lon; the radar if
    None) into a recon10s_tdrgrid.TdrGrid whose fields are named as in
    RADAR_FIELDS. `motion` (heading, kt) is needed for storm-relative velocity.
    """
    lat0, lon0 = centre if centre is not None else (vol.site_lat, vol.site_lon)
    x = np.arange(-half_width_km, half_width_km + dx_km / 2, dx_km)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    lat, lon = _destinations(lat0, lon0, X, Y)
    s, phi = range_azimuth(vol.site_lat, vol.site_lon, lat, lon)
    if site_alt_m is None:
        site_alt_m = _site_altitude(vol)
    alt_km = site_alt_m / 1000.0
    z = np.asarray(z_km, float)
    half_bw = math.radians(beamwidth / 2.0)
    fields: Dict[str, np.ndarray] = {}
    for product in products:
        if product == "srv" and motion is None:
            continue
        sw = sweeps_for(vol, product)
        if not sw:
            continue
        if progress:
            progress(f"Gridding {nx.PRODUCTS['velocity' if product == 'srv' else product]['label'].lower()}"
                     f"{' (storm-relative)' if product == 'srv' else ''}: {len(sw)} elevations…")
        V = np.full((len(sw),) + X.shape, np.nan)
        H = np.full((len(sw),) + X.shape, np.nan)
        W = np.full((len(sw),) + X.shape, np.nan)
        base = "velocity" if product == "srv" else product
        for k, (el, i) in enumerate(sw):
            az, rng, vals, el_true, _nyq = vol.sweep_data(base, i, dealias=dealias)
            if product == "srv":
                vals = srv_values(az, vals, motion)
            V[k] = sample_sweep(az, rng, vals, el_true, s, phi)
            H[k] = beam_height(s, el_true, alt_km)
            W[k] = s / math.cos(math.radians(el_true)) * math.tan(half_bw)
        fields[PRODUCT_FIELD[product]] = vertical_interp(V, H, W, z, max_gap_km)
    t = vol.scan.time
    mot = (0.0, 0.0)
    if motion is not None:
        ms = motion[1] / nx.MS_TO_KT
        mot = (ms * math.sin(math.radians(motion[0])), ms * math.cos(math.radians(motion[0])))
    site = getattr(vol, "site_label", None) or vol.scan.site
    grid = tg.TdrGrid(name=f"{vol.scan.site} {t:%Y%m%d %H%M%S} volume", storm=storm or "", flight=str(site),
                      x=x, y=y, z=z, lat=lat, lon=lon, fields=fields, centre=(float(lat0), float(lon0)),
                      start=t, end=t + dt.timedelta(minutes=5), centre_time=t, motion=mot)
    grid.source = "radar"
    grid.site = (float(vol.site_lat), float(vol.site_lon))
    return grid


EYE_LEVELS_KM = (1.0, 1.5, 2.0, 2.5, 3.0)


def _smoothed(layer: np.ndarray, dx: float, smooth_km: float) -> np.ndarray:
    from scipy import ndimage
    n = max(3, int(round(smooth_km / dx)) | 1)
    # no echo counts as 0 dBZ, so the eye shows as a hole rather than as missing data
    return ndimage.uniform_filter(np.where(np.isfinite(layer), layer, 0.0), n)


def _enclosure(sm: np.ndarray, ci: np.ndarray, cj: np.ndarray, dx: float, dy: float, reach_km: float,
               n_az: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each candidate (row ci, column cj): the strongest echo met along each
    of n_az rays out to reach_km, and the score -- that strongest echo in the
    weaker quarter of directions, less the echo at the candidate itself. An
    eye scores high (a wall nearly all the way round); a clear patch outside
    the eyewall does not (the rays leading away from the storm meet nothing).
    Returns (score, wall level).
    """
    from scipy import ndimage
    az = np.radians(np.arange(0.0, 360.0, 360.0 / n_az))
    rr = np.arange(max(dx, 1.0), reach_km + 1e-6, max(dx, 1.0))
    rows = ci[:, None, None] + rr[None, None, :] * np.cos(az)[None, :, None] / dy
    cols = cj[:, None, None] + rr[None, None, :] * np.sin(az)[None, :, None] / dx
    vals = ndimage.map_coordinates(sm, [rows.ravel(), cols.ravel()], order=1, mode="constant", cval=0.0)
    wall = np.percentile(vals.reshape(ci.size, az.size, rr.size).max(axis=2), 25, axis=1)
    return wall - sm[ci, cj], wall


def find_eye(grid, levels: Sequence[float] = EYE_LEVELS_KM, search_km: float = 40.0, smooth_km: float = 7.0,
             depth_db: float = 10.0, reach_km: float = 60.0) -> Optional[Tuple[float, float]]:
    """
    (x, y) km of the eye within `search_km` of the grid origin, or None.

    Every 2 km across the search area, how well echo encloses the point is
    scored on each low level (smoothed reflectivity at 1-3 km; see
    `_enclosure`). The eye is the point with the best score averaged over
    the levels -- an eye is enclosed at every level, a gap in a rainband at
    one only -- provided that average is at least `depth_db`. Its position
    is then the centroid of the clear area around that point, on the level
    where it is clearest; if that area opens out (an eye open on one side,
    or a moat), the smoothed minimum itself.
    """
    from scipy import ndimage
    if "REFLECTIVITY" not in grid.fields or grid.x.size < 2:
        return None
    vol = grid.fields["REFLECTIVITY"]
    ks = sorted({grid.level(z) for z in levels if grid.z.min() - 0.25 <= z <= grid.z.max() + 0.25})
    ks = [k for k in ks if np.isfinite(vol[k]).mean() > 0.2]
    if not ks:
        return None
    dx = float(grid.x[1] - grid.x[0])
    dy = float(grid.y[1] - grid.y[0])
    X, Y = np.meshgrid(grid.x, grid.y)
    step = max(1, int(round(2.0 / dx)))
    cand = np.zeros(X.shape, bool)
    cand[::step, ::step] = True
    cand &= np.hypot(X, Y) <= search_km
    ci, cj = np.nonzero(cand)
    if ci.size == 0:
        return None
    smooth = {k: _smoothed(vol[k], dx, smooth_km) for k in ks}
    scores = np.array([_enclosure(smooth[k], ci, cj, dx, dy, reach_km)[0] for k in ks])
    mean = scores.mean(axis=0)
    b = int(np.argmax(mean))
    if mean[b] < depth_db:
        return None
    k = ks[int(np.argmax(scores[:, b]))]
    sm = smooth[k]
    _s, wall = _enclosure(sm, ci[b:b + 1], cj[b:b + 1], dx, dy, reach_km)
    X0, Y0 = X[ci[b], cj[b]], Y[ci[b], cj[b]]
    local = np.hypot(X - X0, Y - Y0) <= 2.0 * step * dx
    i, j = np.unravel_index(np.argmin(np.where(local, sm, np.inf)), sm.shape)
    labels, _n = ndimage.label(sm < sm[i, j] + 0.5 * (wall[0] - sm[i, j]))
    hole = labels == labels[i, j]
    if np.hypot(X[hole] - X[i, j], Y[hole] - Y[i, j]).max() > reach_km / 2:
        return float(X[i, j]), float(Y[i, j])
    return float(X[hole].mean()), float(Y[hole].mean())


def recentre(grid, x_km: float, y_km: float):
    """A view of the grid with its origin moved to (x_km, y_km): sections and means about the new centre."""
    import copy
    g = copy.copy(grid)
    g.x = grid.x - x_km
    g.y = grid.y - y_km
    i = int(np.argmin(np.abs(grid.y - y_km)))
    j = int(np.argmin(np.abs(grid.x - x_km)))
    g.centre = (float(grid.lat[i, j]), float(grid.lon[i, j]))
    return g


def _site_altitude(vol) -> float:
    if getattr(vol, "site_height_m", None) is not None:
        return float(vol.site_height_m)
    radar = getattr(vol, "radar", None)
    try:
        alt = float(radar.altitude["data"][0])
        tower = 0.0
        return alt + tower
    except Exception:
        return 0.0


def is_radar_grid(grid) -> bool:
    return getattr(grid, "source", "") == "radar"
