#!/usr/bin/env python3
"""
recon10s_glm — GOES Geostationary Lightning Mapper flashes.

GLM-L2-LCFA files in the same public buckets as the ABI imagery, one every
20 seconds (~240 KB each). A frame shows the flashes in the ten minutes up
to its scan time: thirty files, fetched in parallel, each cached so that
stepping back and forth through a loop does not fetch anything twice.

Eyewall lightning bursts are a known signal of intensity change, which is
the reason to put this on the map.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import io
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import requests

import recon10s_goes as goes

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

PRODUCT = "GLM-L2-LCFA"
REACH_DEG = 55.0      # GLM's field of view, as great-circle distance from the sub-point
_CACHE: "OrderedDict[str, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_CACHE_MAX = 2000   # files, i.e. ~11 hours of flashes


def glm_satellite(lat: float, lon: float) -> Optional[int]:
    """The GOES whose GLM sees (lat, lon) best, or None (Himawari carries no lightning mapper)."""
    best = min(goes.GOES_SATS, key=lambda s: goes.arc_from_subpoint(lat, lon, goes.SUBPOINT[s]))
    return best if goes.arc_from_subpoint(lat, lon, goes.SUBPOINT[best]) <= REACH_DEG else None


def _files(sat: int, start: dt.datetime, end: dt.datetime) -> List[Tuple[str, dt.datetime]]:
    out = []
    t = start.replace(minute=0, second=0, microsecond=0)
    while t <= end:
        prefix = f"{PRODUCT}/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"
        for key, _size in goes._list_keys(goes.BUCKETS[sat], prefix):
            name = key.rsplit("/", 1)[-1]
            try:
                s = goes._parse_start(name.split("_s", 1)[1][:14])
            except (IndexError, ValueError):
                continue
            if start <= s <= end:
                out.append((key, s))
        t += dt.timedelta(hours=1)
    return sorted(out, key=lambda x: x[1])


def _read(sat: int, key: str) -> Tuple[np.ndarray, np.ndarray]:
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit
    r = requests.get(f"https://{goes.BUCKETS[sat]}.s3.amazonaws.com/{key}", timeout=30)
    r.raise_for_status()
    with h5py.File(io.BytesIO(r.content), "r") as h:
        out = []
        for name in ("flash_lat", "flash_lon"):
            if name not in h:
                out.append(np.empty(0))
                continue
            d = h[name]
            v = d[...].astype(np.float64)
            sc = goes._attr(d.attrs, "scale_factor")
            of = goes._attr(d.attrs, "add_offset")
            if sc is not None:
                v = v * float(sc)
            if of is not None:
                v = v + float(of)
            out.append(v)
    _CACHE[key] = (out[0], out[1])
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return out[0], out[1]


def flashes(sat: int, end: dt.datetime, minutes: float = 10.0,
            bbox: Optional[Tuple[float, float, float, float]] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    (lats, lons, files read) of every flash in the `minutes` before `end`,
    optionally trimmed to a (south, north, west, east) box.
    """
    if h5py is None:
        raise RuntimeError("h5py is required for GLM (pip install h5py)")
    start = end - dt.timedelta(minutes=minutes)
    files = _files(sat, start, end)
    lats, lons = [], []
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for la, lo in ex.map(lambda f: _read(sat, f[0]), files):
            lats.append(la)
            lons.append(lo)
    la = np.concatenate(lats) if lats else np.empty(0)
    lo = np.concatenate(lons) if lons else np.empty(0)
    if bbox is not None and la.size:
        s, n, w, e = bbox
        mid = (w + e) / 2.0                 # longitudes about the box's middle, so a box across 180 works
        rel = (lo - mid + 180.0) % 360.0 - 180.0
        keep = (la >= s) & (la <= n) & (rel >= w - mid) & (rel <= e - mid)
        la, lo = la[keep], lo[keep]
    return la, lo, len(files)
