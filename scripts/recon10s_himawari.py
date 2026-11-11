#!/usr/bin/env python3
"""
recon10s_himawari — Himawari-9 AHI imagery from NOAA's public bucket, for
storms west of the dateline: the West Pacific, the South Pacific, Australia
and the eastern Indian Ocean.

NOAA re-serves JMA's Himawari-9 data on AWS (s3://noaa-himawari9, open, no
sign-in) as AWIPS "sectorized" netCDF4 tiles under
AHI-L2-FLDK-ISatSS/YYYY/MM/DD/HHMM/ (from December 2022):

  * Full disk every 10 minutes, in the HH:M0 folders:
        OR_HFD-<res>-B<bits>-M1C<ch>-T<tile>_GH9_s<YYYYDDDHHMMSSt>_c<...>.nc
    The disk is cut into a 10 x 10 grid of tiles: 550 x 550 pixels at 2 km
    (res 020), 1100 at 1 km (010), 2200 at 0.5 km (005). Tiles wholly off
    the Earth are not made, so the 88 that are are numbered row by row: rows
    0 and 9 hold columns 2-7, rows 1 and 8 columns 1-8, rows 2-7 all ten. A
    storm-sized view needs one to four tiles, about 0.3 MB each at 2 km.
  * Target area every 2.5 minutes, in the other four folders of each 10
    minutes: OR_HR3-..., one 500 x 500-pixel tile (1000 km at 2 km) over the
    box JMA points wherever it is watching -- usually the strongest typhoon.
    Like a GOES mesoscale sector, where it is lives only in the file.

Every tile carries its fixed-grid scan angles x/y (16-bit, microradians),
the geostationary projection (sub-point 140.7E, sweep axis y) and the
brightness temperature (K) or reflectance as scaled 16-bit integers;
negative counts (-1076) mark space.

Band numbers here are the app's -- GOES's -- for the same wavelength, so the
Satellite tab's choices mean the same on either satellite: 2 is AHI's 0.64-um
band 3; 7-10 and 12-15 are AHI's bands of the same numbers (13 is 10.4 um,
ABI's is 10.3).

Images come back as recon10s_goes.GoesImage, so the map, loops, products,
drop insets and the satellite intensity estimator take them unchanged.
recon10s_goes hands everything for satellite 9 to this module.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import io
import math
import re
import threading
import time as _time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

import recon10s_goes as goes

SAT = 9
NAME = "Himawari-9"
BUCKET = "noaa-himawari9"
BASE = f"https://{BUCKET}.s3.amazonaws.com/"
PREFIX = "AHI-L2-FLDK-ISatSS"
SUBPOINT = 140.7
HEIGHT = 35785831.0
SEMI_MAJOR, SEMI_MINOR = 6378137.0, 6356752.31414
SWEEP = "y"
FIRST = dt.datetime(2022, 12, 13, tzinfo=dt.timezone.utc)      # the archive starts with Himawari-9

# app band (GOES numbering) -> AHI channel of the same wavelength
CHANNEL = {2: 3, 7: 7, 8: 8, 9: 9, 10: 10, 12: 12, 13: 13, 14: 14, 15: 15}
# bits per pixel in the file names
BITS = {1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 14, 8: 11, 9: 11, 10: 12, 11: 12, 12: 12, 13: 12, 14: 12,
        15: 12, 16: 11}
SECTOR_NAMES = {"F": "Full disk", "T": "Target area"}

GRID = 10
TILE_COLS = [(2, 7), (1, 8)] + [(0, 9)] * 6 + [(1, 8), (2, 7)]   # first and last column made, per tile row
# scan angle (radians) of each full-disk grid's first column -- minus that of its first row -- and the
# step between pixels, as the tiles give them: x = (column - (N/2 + 0.5)) * step on every grid
GRID_X0 = {"020": -153719.91730804e-6, "010": -153705.94906557e-6, "005": -153698.96118914e-6}
GRID_STEP = {"020": 55.8879903e-6, "010": 27.94399583e-6, "005": 13.97199774e-6}
MIN_KELVIN = 100.0                      # anything colder is not a real scene
WHOLE_FILE_MB = 1.5                     # tiles smaller than this are fetched whole

_S3 = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_FNAME = re.compile(r"OR_(?P<sector>HFD|HR3)-(?P<res>\d{3})-B(?P<bits>\d{2})-M1C(?P<ch>\d{2})-T(?P<tile>\d{3})"
                    r"_GH9_s(?P<start>\d{14})_c\d+\.nc$")


def res_code(channel: int) -> str:
    return {1: "010", 2: "010", 3: "005", 4: "010"}.get(int(channel), "020")


def full_pixels(res: str) -> int:
    return {"020": 5500, "010": 11000, "005": 22000}[res]


def tile_number(row: int, col: int) -> Optional[int]:
    """1-based number of the tile in grid row/column, or None where none is made (space)."""
    if not (0 <= row < GRID and 0 <= col < GRID):
        return None
    lo, hi = TILE_COLS[row]
    if not lo <= col <= hi:
        return None
    return sum(b - a + 1 for a, b in TILE_COLS[:row]) + (col - lo) + 1


def tile_position(n: int) -> Tuple[int, int]:
    """(row, column) of tile number n."""
    if n < 1:
        raise ValueError("no such tile")
    for row, (lo, hi) in enumerate(TILE_COLS):
        count = hi - lo + 1
        if n <= count:
            return row, lo + n - 1
        n -= count
    raise ValueError("no such tile")


# ------------------------------------------------------------------ listing

_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = _local.session = requests.Session()
    return s


_LIST_CACHE: Dict[Tuple[str, bool], Tuple[float, list, list]] = {}
_LIST_LOCK = threading.Lock()


def _list(prefix: str, delimiter: bool = False, timeout: float = 30.0, final: bool = False):
    """
    (keys with sizes, sub-folders) under a prefix, following continuation. `final` says the
    folder can no longer change (its time is well past), so its listing is kept.
    """
    with _LIST_LOCK:
        hit = _LIST_CACHE.get((prefix, delimiter))
    if hit and (hit[0] == float("inf") or _time.time() - hit[0] < 60.0):
        return hit[1], hit[2]
    keys, folders, token = [], [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            params["delimiter"] = "/"
        if token:
            params["continuation-token"] = token
        r = _session().get(BASE, params=params, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for node in root.findall(f"{_S3}Contents"):
            keys.append((node.find(f"{_S3}Key").text, int(node.find(f"{_S3}Size").text)))
        for node in root.findall(f"{_S3}CommonPrefixes"):
            folders.append(node.find(f"{_S3}Prefix").text)
        nxt = root.find(f"{_S3}NextContinuationToken")
        if nxt is None or not nxt.text:
            break
        token = nxt.text
    with _LIST_LOCK:
        if len(_LIST_CACHE) > 4000:
            _LIST_CACHE.clear()
        _LIST_CACHE[(prefix, delimiter)] = (float("inf") if final else _time.time(), keys, folders)
    return keys, folders


def _is_final(t: dt.datetime, minutes: float) -> bool:
    return dt.datetime.now(dt.timezone.utc) - t > dt.timedelta(minutes=minutes)


def _hour_folders(t: dt.datetime) -> List[Tuple[dt.datetime, str]]:
    """(folder time, folder prefix) of every time folder in the hour of t, oldest first."""
    hour = t.replace(minute=0, second=0, microsecond=0)
    _keys, folders = _list(f"{PREFIX}/{hour:%Y/%m/%d/%H}", delimiter=True,
                           final=_is_final(hour, 90))
    out = []
    for f in folders:
        m = re.search(r"/(\d{2})(\d{2})/$", f)
        if m and m.group(1) == f"{hour:%H}":
            out.append((hour.replace(minute=int(m.group(2))), f))
    return sorted(out)


def _scene_in(folder: str, folder_time: dt.datetime, band: int, sector: str):
    """The scene of one band in one time folder, or None if it is not (yet) there."""
    ch = CHANNEL[band]
    res = res_code(ch)
    tag = "HFD" if sector == "F" else "HR3"
    final = _is_final(folder_time, 30)
    stem = f"{folder}OR_{tag}-{res}-B{BITS[ch]}-M1C{ch:02d}-T"
    found = _list(stem, final=final)[0]
    if not found:                      # in case the bit depth in the names ever changes
        found = [(k, s) for k, s in _list(f"{folder}OR_{tag}-{res}-", final=final)[0] if f"-M1C{ch:02d}-T" in k]
    tiles, start = {}, None
    for key, size in found:
        m = _FNAME.search(key)
        if not m or int(m.group("ch")) != ch:
            continue
        tiles[int(m.group("tile"))] = (key, size)
        st = goes._parse_start(m.group("start"))
        start = st if start is None or st < start else start
    if not tiles:
        return None
    if sector == "F" and len(tiles) < 88 and not final:
        return None                    # still arriving: the storm's tiles may not be in yet
    return goes.Scene(sat=SAT, band=band, sector=sector, key=stem[:-2],
                      size=sum(s for _k, s in tiles.values()), start=start, tiles=tiles)


def list_scenes(band: int, sector: str = "F", when: Optional[dt.datetime] = None, hours_back: int = 3,
                workers: int = 8) -> list:
    """Scenes of one band and sector up to `when`, newest last, walking back an hour at a time until some turn up."""
    if band not in CHANNEL:
        raise ValueError(f"band {band} has no Himawari equivalent here")
    sector = "F" if sector == "F" else "T"
    when = when or dt.datetime.now(dt.timezone.utc)
    for back in range(hours_back + 1):
        t = when - dt.timedelta(hours=back)
        if t < FIRST - dt.timedelta(hours=1):
            break
        folders = [(ft, f) for ft, f in _hour_folders(t)
                   if ft <= when + dt.timedelta(minutes=1) and (ft.minute % 10 == 0) == (sector == "F")]
        if not folders:
            continue
        errors = []

        def one(f):
            try:
                return _scene_in(f[1], f[0], band, sector)
            except Exception as exc:          # one folder's listing failing should not lose the rest
                errors.append(exc)
                return None
        with cf.ThreadPoolExecutor(max_workers=min(workers, len(folders))) as ex:
            scenes = [s for s in ex.map(one, folders) if s is not None]
        if errors and len(errors) == len(folders):
            raise errors[0]
        scenes = [s for s in scenes if s.start <= when + dt.timedelta(minutes=1)]
        if scenes:
            return sorted(scenes, key=lambda s: s.start)
    return []


def recent_scenes(band: int, sector: str, count: int, when: Optional[dt.datetime] = None,
                  max_hours: int = 8) -> list:
    """The newest `count` scenes up to `when`, oldest first -- the frames of a loop."""
    when = when or dt.datetime.now(dt.timezone.utc)
    found: Dict[str, object] = {}
    cursor = when
    for _ in range(max_hours + 1):
        for s in list_scenes(band, sector, cursor, hours_back=0):
            if s.start <= when + dt.timedelta(minutes=1):
                found[s.key] = s
        if len(found) >= count:
            break
        cursor = cursor.replace(minute=59, second=59, microsecond=0) - dt.timedelta(hours=1)
    return sorted(found.values(), key=lambda s: s.start)[-count:]


def latest_scene(band: int, sector: str = "F", when: Optional[dt.datetime] = None):
    scenes = list_scenes(band, sector, when)
    return scenes[-1] if scenes else None


# --------------------------------------------------------------- navigation

def transformer(inverse: bool = False):
    """lon/lat -> projection metres (or back) for Himawari-9's fixed grid; one per thread (pyproj's
    transformers are not to be shared between threads, and frames load in parallel)."""
    cache = getattr(_local, "tf", None)
    if cache is None:
        cache = _local.tf = {}
    if inverse not in cache:
        from pyproj import CRS, Transformer
        crs = CRS.from_proj4(f"+proj=geos +h={HEIGHT} +lon_0={SUBPOINT} +sweep={SWEEP} "
                             f"+a={SEMI_MAJOR} +b={SEMI_MINOR} +units=m +no_defs")
        cache[inverse] = (Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if inverse
                          else Transformer.from_crs("EPSG:4326", crs, always_xy=True))
    return cache[inverse]


def scan_angles(lats, lons):
    """Fixed-grid (x, y) scan angles in radians; NaN off the disk."""
    x, y = transformer().transform(np.asarray(lons, float), np.asarray(lats, float))
    x, y = np.asarray(x, float), np.asarray(y, float)
    bad = ~np.isfinite(x) | ~np.isfinite(y) | (np.abs(x) > 1e20) | (np.abs(y) > 1e20)
    return np.where(bad, np.nan, x / HEIGHT), np.where(bad, np.nan, y / HEIGHT)


def arc_deg(lat: float, lon: float) -> float:
    """Great-circle distance (degrees) of a point from the sub-satellite point."""
    return goes.arc_from_subpoint(lat, lon, SUBPOINT)


def _box_points(bbox, n: int = 9):
    south, north, west, east = bbox
    if east < west:                      # across the dateline
        east += 360.0
    la, lo = np.meshgrid(np.linspace(south, north, n), np.linspace(west, east, n), indexing="ij")
    return la.ravel(), ((lo.ravel() + 180.0) % 360.0) - 180.0


def pixel_window(bbox, res: str, pad: int = 8) -> Optional[Tuple[int, int, int, int]]:
    """(row0, row1, col0, col1) of the full-disk grid at `res` covering a (S, N, W, E) box; None if off the disk."""
    la, lo = _box_points(bbox)
    x, y = scan_angles(la, lo)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return None
    n = full_pixels(res)
    x0, step = GRID_X0[res], GRID_STEP[res]
    cols = (x[ok] - x0) / step
    rows = (-x0 - y[ok]) / step
    c0, c1 = int(math.floor(cols.min())) - pad, int(math.ceil(cols.max())) + pad + 1
    r0, r1 = int(math.floor(rows.min())) - pad, int(math.ceil(rows.max())) + pad + 1
    c0, r0, c1, r1 = max(0, c0), max(0, r0), min(n, c1), min(n, r1)
    if c1 - c0 < 2 or r1 - r0 < 2:
        return None
    return r0, r1, c0, c1


# ------------------------------------------------------------------ reading

def _axis(d) -> np.ndarray:
    v = d[:].astype(np.float64)
    s, o = goes._attr(d.attrs, "scale_factor"), goes._attr(d.attrs, "add_offset")
    if s is not None:
        v = v * float(s)
    if o is not None:
        v = v + float(o)
    return v * 1e-6                       # microradians -> radians


class _Tile:
    """One open tile file: whole in memory when small, over HTTP ranges when large."""

    def __init__(self, key: str, size: int):
        import h5py
        url = BASE + key
        self.bytes = 0
        self.handle = None
        if size <= WHOLE_FILE_MB * 1e6:
            r = _session().get(url, timeout=60)
            r.raise_for_status()
            self.bytes = len(r.content)
            self.h = h5py.File(io.BytesIO(r.content), "r")
        else:
            self.handle = goes.HttpRangeFile(url, block=1 << 18)
            self.h = h5py.File(self.handle, "r")

    def read(self, rows: Optional[slice] = None, cols: Optional[slice] = None):
        """(values, x, y, global row of rows.start, global column of cols.start, kind)."""
        h = self.h
        x, y = _axis(h["x"]), _axis(h["y"])
        d = h["Sectorized_CMI"]
        rows = rows or slice(0, d.shape[0])
        cols = cols or slice(0, d.shape[1])
        if self.handle is not None:
            goes._prefetch_rows(self.handle, d, rows, cols)
        raw = d[rows, cols]
        vals = (raw.astype(np.float32) * float(goes._attr(d.attrs, "scale_factor", 1.0))
                + float(goes._attr(d.attrs, "add_offset", 0.0)))
        units = str(goes._attr(d.attrs, "units", "") or "").lower()
        kind = "brightness" if units in ("k", "kelvin") else "reflectance"
        bad = raw < 0
        if kind == "brightness":
            bad |= vals < MIN_KELVIN
        vals[bad] = np.nan
        r_off = int(goes._attr(h.attrs, "tile_row_offset", 0))
        c_off = int(goes._attr(h.attrs, "tile_column_offset", 0))
        return vals, x[cols], y[rows], r_off + rows.start, c_off + cols.start, kind

    def projection(self) -> Tuple[float, float]:
        p = self.h["fixedgrid_projection"].attrs
        return (float(goes._attr(p, "longitude_of_projection_origin", SUBPOINT)),
                float(goes._attr(p, "perspective_point_height", HEIGHT)))

    def close(self):
        try:
            self.h.close()
        finally:
            if self.handle is not None:
                self.handle.close()
                self.bytes = self.handle.bytes_fetched


def _read(key: str, size: int, rows=None, cols=None):
    t = _Tile(key, size)
    try:
        out = t.read(rows, cols)
        lon0, height = t.projection()
    finally:
        t.close()
    return out + (lon0, height, t.bytes)


def scene_extent_lonlat(scene) -> Optional[Tuple[float, float, float, float]]:
    """(south, north, west, east) a target-area scene covers, from its file (where JMA pointed it)."""
    if scene.sector == "F":
        return None
    try:
        key, size = next(iter(scene.tiles.values()))
        t = _Tile(key, size)
        try:
            x, y = _axis(t.h["x"]), _axis(t.h["y"])
        finally:
            t.close()
        return extent_of(x, y)
    except Exception:
        return None


def extent_of(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """(S, N, W, E) of the part of the Earth inside a fixed-grid x/y box (radians)."""
    gx = np.linspace(x[0], x[-1], 25)
    gy = np.linspace(y[0], y[-1], 25)
    xs = np.concatenate([gx, gx, np.full(gy.size, gx[0]), np.full(gy.size, gx[-1]), [gx[12]]])
    ys = np.concatenate([np.full(gx.size, gy[0]), np.full(gx.size, gy[-1]), gy, gy, [gy[12]]])
    lo, la = transformer(inverse=True).transform(xs * HEIGHT, ys * HEIGHT)
    lo, la = np.asarray(lo, float), np.asarray(la, float)
    ok = np.isfinite(lo) & np.isfinite(la) & (np.abs(la) <= 90) & (np.abs(lo) <= 360)
    if not ok.any():
        return None
    lo = lo[ok]
    # the box may straddle the dateline: measure longitudes around the box's own middle
    mid = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(lo))), np.mean(np.cos(np.radians(lo))))))
    rel = ((lo - mid + 180.0) % 360.0) - 180.0
    west = ((mid + rel.min() + 180.0) % 360.0) - 180.0
    east = ((mid + rel.max() + 180.0) % 360.0) - 180.0
    return float(la[ok].min()), float(la[ok].max()), float(west), float(east)


def _fill_axis(a: np.ndarray) -> np.ndarray:
    """Scan angles where no tile supplied any (space), extended linearly from those known."""
    ok = np.flatnonzero(np.isfinite(a))
    if 2 <= ok.size < a.size:
        step = (a[ok[-1]] - a[ok[0]]) / (ok[-1] - ok[0])
        a = a[ok[0]] + (np.arange(a.size) - ok[0]) * step
    return a


def load_image(scene, bbox: Optional[Tuple[float, float, float, float]] = None, max_pixels: int = 2400):
    """
    A cut of a scene as a GoesImage (degC for IR, reflectance 0-1 for band 2): the tiles a
    (south, north, west, east) box touches, read in parallel and stitched on the full-disk pixel
    grid. A target area is one tile, trimmed to the box when one is given; a full disk needs one.
    """
    if scene.sector != "F":
        key, size = next(iter(scene.tiles.values()))
        vals, x, y, _r, _c, kind, lon0, height, nbytes = _read(key, size)
        ext = extent_of(x, y)
        if bbox is not None:
            la, lo = _box_points(bbox)
            bx, by = scan_angles(la, lo)
            ok = np.isfinite(bx) & np.isfinite(by)
            if ok.any():
                pad = 8 * abs(x[1] - x[0])
                ci = np.flatnonzero((x >= np.min(bx[ok]) - pad) & (x <= np.max(bx[ok]) + pad))
                ri = np.flatnonzero((y >= np.min(by[ok]) - pad) & (y <= np.max(by[ok]) + pad))
                if ci.size > 1 and ri.size > 1:
                    vals = vals[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1]
                    x, y = x[ci[0]:ci[-1] + 1], y[ri[0]:ri[-1] + 1]
        return _image(scene, vals, x, y, kind, lon0, height, nbytes, ext, max_pixels)

    if bbox is None:
        raise ValueError("a full-disk Himawari scene needs a box to read")
    res = res_code(CHANNEL[scene.band])
    win = pixel_window(bbox, res)
    if win is None:
        raise RuntimeError(f"that area is beyond the edge of {NAME}'s disk")
    r0, r1, c0, c1 = win
    tile = full_pixels(res) // GRID
    jobs = []
    for tr in range(r0 // tile, (r1 - 1) // tile + 1):
        for tc in range(c0 // tile, (c1 - 1) // tile + 1):
            n = tile_number(tr, tc)
            if n is None or n not in scene.tiles:
                continue
            rr = slice(max(r0, tr * tile) - tr * tile, min(r1, (tr + 1) * tile) - tr * tile)
            cc = slice(max(c0, tc * tile) - tc * tile, min(c1, (tc + 1) * tile) - tc * tile)
            jobs.append((scene.tiles[n][0], scene.tiles[n][1], rr, cc))
    if not jobs:
        raise RuntimeError(f"no {NAME} tiles cover that area in the {scene.start:%H:%MZ} scan")
    with cf.ThreadPoolExecutor(max_workers=min(4, len(jobs))) as ex:
        parts = list(ex.map(lambda j: _read(*j), jobs))
    mosaic = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
    xs = np.full(c1 - c0, np.nan)
    ys = np.full(r1 - r0, np.nan)
    nbytes = 0
    for vals, x, y, ro, co, _kind, _lon0, _h, nb in parts:
        mosaic[ro - r0:ro - r0 + vals.shape[0], co - c0:co - c0 + vals.shape[1]] = vals
        xs[co - c0:co - c0 + x.size] = x
        ys[ro - r0:ro - r0 + y.size] = y
        nbytes += nb
    kind, lon0, height = parts[0][5], parts[0][6], parts[0][7]
    return _image(scene, mosaic, _fill_axis(xs), _fill_axis(ys), kind, lon0, height, nbytes, None, max_pixels)


def _image(scene, vals, x, y, kind, lon0, height, nbytes, extent_ll, max_pixels):
    if max(vals.shape) > max_pixels:
        step = int(math.ceil(max(vals.shape) / max_pixels))
        vals, x, y = vals[::step, ::step], x[::step], y[::step]
    if kind == "brightness":
        vals = vals - 273.15                                  # kelvin -> degC, as for GOES
    dx = (x[1] - x[0]) if x.size > 1 else 0.0
    dy = (y[1] - y[0]) if y.size > 1 else 0.0
    extent_m = ((x[0] - dx / 2) * height, (x[-1] + dx / 2) * height,
                (y[-1] + dy / 2) * height, (y[0] - dy / 2) * height)
    return goes.GoesImage(data=vals, extent_m=extent_m, lon0=lon0, sat_height=height, sweep=SWEEP,
                          scene=scene, kind=kind, bytes_fetched=nbytes, extent_lonlat=extent_ll)


# ----------------------------------------------------------------- choosing

def choose_scene(lat: float, lon: float, band: int, prefer: str = "auto",
                 when: Optional[dt.datetime] = None) -> Tuple[Optional[object], str]:
    """The scene to show for a storm: the target area when JMA has it over the storm, else the full disk."""
    if prefer == "F":
        return latest_scene(band, "F", when), "full disk (forced)"
    if prefer in ("T", "M1", "M2"):
        return latest_scene(band, "T", when), "target area (forced)"
    target = latest_scene(band, "T", when)
    if target is not None:
        ext = scene_extent_lonlat(target)
        fresh = (when or dt.datetime.now(dt.timezone.utc)) - target.start <= dt.timedelta(minutes=20)
        if ext and fresh and goes.covers(ext, lat, lon, margin=-0.5):
            return target, "the target area (2.5-minute scans) is over the storm"
    return latest_scene(band, "F", when), "the target area is elsewhere: full disk"


# --------------------------------------------------------------- self-check

def self_check() -> str:
    """For --smoke-test: the tile layout, and navigation (pyproj) of the sub-point and of a target area."""
    try:
        ok = tile_position(45) == (5, 0) and tile_number(9, 7) == 88
        x, y = scan_angles(np.array([0.0]), np.array([SUBPOINT]))
        ok = ok and abs(x[0]) < 1e-9 and abs(y[0]) < 1e-9 and pixel_window((9.5, 17.5, 137.4, 145.4), "020") is not None
        ext = extent_of(np.array([-0.0123233, 0.0155648]), np.array([0.0550776, 0.0271895]))  # 18 Oct 2026's, over Champi
        ok = ok and ext is not None and 8.5 < ext[0] < 9.2 and 18.0 < ext[1] < 18.7
        return f"Himawari-9 {'ok' if ok else 'SUSPECT'} (tiles, navigation)"
    except Exception as exc:
        return f"Himawari-9 FAILED ({type(exc).__name__}: {exc})"
