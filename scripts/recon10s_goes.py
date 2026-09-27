#!/usr/bin/env python3
"""
recon10s_goes — pulls GOES-18/19 ABI imagery from the NOAA public S3 buckets.

No credentials, no boto3: the buckets are public, so listing is a plain HTTP
GET and reading is HTTP Range requests fed to h5py (ABI netCDF4 files are
HDF5 underneath).

That matters more than it sounds. A full-disk band 2 file is ~230 MB, which
is not something to download to draw one storm. Opening it over Range costs
about 1 MB, and a storm-sized subset costs ~35 MB, because ABI chunks are
full image rows -- so trimming rows helps and trimming columns does not.

    sat   = pick_satellite(lat, lon)          # 19 east, 18 west
    scene = latest_scene(sat, band=13, prefer_meso_at=(lat, lon))
    img   = load_image(scene, bbox=(south, north, west, east))
    # img.data is degC for IR bands, reflectance 0-1 for band 2
"""
from __future__ import annotations

import datetime as dt
import concurrent.futures as cf
import io
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
BUCKETS = {18: "noaa-goes18", 19: "noaa-goes19"}
SUBPOINT = {18: -137.0, 19: -75.2}

# Bands this tool offers, with how each is normally shown.
BANDS: Dict[int, Dict[str, object]] = {
    2:  {"name": "Red visible", "micron": 0.64, "kind": "reflectance", "res_km": 0.5},
    7:  {"name": "Shortwave IR", "micron": 3.9, "kind": "brightness", "res_km": 2.0},
    9:  {"name": "Mid-level water vapour", "micron": 6.9, "kind": "brightness", "res_km": 2.0},
    13: {"name": "Clean longwave IR", "micron": 10.3, "kind": "brightness", "res_km": 2.0},
}

DEFAULT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "recon10s_cache", "goes")
_FNAME_RE = re.compile(
    r"OR_ABI-L2-CMIP(?P<sector>F|M1|M2)-M(?P<mode>\d)C(?P<band>\d{2})_G(?P<sat>\d{2})"
    r"_s(?P<start>\d{14})_e\d{14}_c\d{14}\.nc$")


# ------------------------------------------------------------------ scenes

@dataclass
class Scene:
    """One ABI file on S3, before anything is read from it."""
    sat: int
    band: int
    sector: str          # "F", "M1", "M2"
    key: str
    size: int
    start: dt.datetime

    @property
    def bucket(self) -> str:
        return BUCKETS[self.sat]

    @property
    def url(self) -> str:
        return f"https://{self.bucket}.s3.amazonaws.com/{self.key}"

    @property
    def filename(self) -> str:
        return self.key.rsplit("/", 1)[-1]

    def __str__(self) -> str:
        return (f"GOES-{self.sat} band {self.band} {self.sector} "
                f"{self.start:%H:%M:%SZ} ({self.size / 1e6:.0f} MB)")


@dataclass
class GoesImage:
    """A loaded (and usually subsetted) patch of ABI imagery."""
    data: np.ndarray          # degC for IR bands, 0-1 reflectance for band 2
    extent_m: Tuple[float, float, float, float]  # x0, x1, y0, y1 in projection metres
    lon0: float
    sat_height: float
    sweep: str
    scene: Scene
    kind: str                 # "brightness" or "reflectance"
    bytes_fetched: int = 0
    extent_lonlat: Optional[Tuple[float, float, float, float]] = None  # S, N, W, E of the scene

    @property
    def start(self) -> dt.datetime:
        return self.scene.start


def _parse_start(stamp: str) -> dt.datetime:
    """ABI start stamps are YYYYDDDHHMMSSt (tenths of a second on the end)."""
    year = int(stamp[0:4])
    doy = int(stamp[4:7])
    base = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=doy - 1)
    return base + dt.timedelta(hours=int(stamp[7:9]), minutes=int(stamp[9:11]),
                               seconds=int(stamp[11:13]), milliseconds=100 * int(stamp[13]))


def pick_satellite(lat: float, lon: float) -> int:
    """
    GOES-19 sits at 75.2W and GOES-18 at 137W. Both see the east Pacific, so
    pick whichever has the smaller viewing angle -- which for practical
    purposes is whichever subpoint the storm is nearer in longitude.
    """
    return min(BUCKETS, key=lambda s: abs(((lon - SUBPOINT[s]) + 180) % 360 - 180))


def _list_keys(bucket: str, prefix: str, timeout: float = 30.0) -> List[Tuple[str, int]]:
    r = requests.get(f"https://{bucket}.s3.amazonaws.com/",
                     params={"list-type": "2", "prefix": prefix, "max-keys": 1000},
                     timeout=timeout)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    out = []
    for node in root.findall(".//s3:Contents", S3_NS):
        key = node.find("s3:Key", S3_NS).text
        size = int(node.find("s3:Size", S3_NS).text)
        out.append((key, size))
    return out


def list_scenes(sat: int, band: int, sector: str = "F",
                when: Optional[dt.datetime] = None, hours_back: int = 3) -> List[Scene]:
    """
    Scenes for one band and sector, newest last. Walks back an hour at a time
    until something turns up, since the current hour is empty right after it
    rolls over.
    """
    if band not in BANDS:
        raise ValueError(f"band {band} is not one of {sorted(BANDS)}")
    when = when or dt.datetime.now(dt.timezone.utc)
    bucket = BUCKETS[sat]
    product = "ABI-L2-CMIPF" if sector == "F" else "ABI-L2-CMIPM"

    scenes: List[Scene] = []
    for back in range(hours_back + 1):
        t = when - dt.timedelta(hours=back)
        # Narrow the prefix past the band: a meso hour holds ~1900 keys, which
        # would otherwise need paging.
        # The sector is part of the filename stem (CMIPF, CMIPM1, CMIPM2)
        # even though meso sectors share one ABI-L2-CMIPM directory.
        folder = f"{product}/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"
        stem = f"OR_ABI-L2-CMIP{sector}-M6C{band:02d}_G{sat}_s"
        found = _list_keys(bucket, folder + stem)
        if not found:  # scan modes other than 6 (rare now, but cheap to allow)
            found = [(k, sz) for k, sz in _list_keys(bucket, folder + f"OR_ABI-L2-CMIP{sector}-M")
                     if f"C{band:02d}_G{sat}_s" in k]
        for key, size in found:
            m = _FNAME_RE.search(key)
            if not m:
                continue
            scenes.append(Scene(sat=sat, band=band, sector=m.group("sector"), key=key,
                                size=size, start=_parse_start(m.group("start"))))
        if scenes:
            break

    scenes.sort(key=lambda s: s.start)
    return scenes


def recent_scenes(sat: int, band: int, sector: str, count: int,
                  when: Optional[dt.datetime] = None, max_hours: int = 8) -> List[Scene]:
    """
    The newest `count` scenes for one band and sector, oldest first -- the
    frames of a loop. Walks back an hour at a time, since a 30-frame full
    disk loop reaches five hours into the past.
    """
    when = when or dt.datetime.now(dt.timezone.utc)
    found: Dict[str, Scene] = {}
    cursor = when
    for _ in range(max_hours + 1):
        for scene in list_scenes(sat, band, sector, cursor, hours_back=0):
            found[scene.key] = scene
        if len(found) >= count:
            break
        cursor -= dt.timedelta(hours=1)
    frames = sorted(found.values(), key=lambda s: s.start)
    return frames[-count:]


def latest_scene(sat: int, band: int, sector: str = "F",
                 when: Optional[dt.datetime] = None) -> Optional[Scene]:
    scenes = list_scenes(sat, band, sector, when)
    return scenes[-1] if scenes else None


# ------------------------------------------------------- remote file access

class HttpRangeFile(io.RawIOBase):
    """
    A seekable read-only file over HTTP Range requests, block-cached.

    h5py accepts any file-like object with readinto/seek/tell, which is what
    makes subsetting a 230 MB full disk affordable.
    """

    def __init__(self, url: str, block: int = 1 << 20, timeout: float = 60.0):
        self.url, self.block, self.timeout = url, block, timeout
        self.pos = 0
        self.session = requests.Session()
        self._cache: Dict[int, bytes] = {}
        self.bytes_fetched = 0
        head = self.session.head(url, timeout=timeout)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])

    def seekable(self) -> bool: return True
    def readable(self) -> bool: return True

    def seek(self, off, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = off
        elif whence == io.SEEK_CUR:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def tell(self) -> int:
        return self.pos

    def _block_at(self, idx: int) -> bytes:
        blk = self._cache.get(idx)
        if blk is None:
            start = idx * self.block
            end = min(start + self.block, self.size) - 1
            r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                 timeout=self.timeout)
            r.raise_for_status()
            blk = r.content
            self._cache[idx] = blk
            self.bytes_fetched += len(blk)
        return blk

    def prefetch(self, spans, workers: int = 8) -> None:
        """
        Pull every block touched by the given (offset, size) byte spans in
        parallel. h5py holds a global lock while it reads, so letting it fetch
        chunk by chunk serialises the network; fetching the blocks up front
        from plain threads, then letting h5py decode from cache, does not.
        """
        wanted = set()
        for offset, size in spans:
            if size <= 0:
                continue
            wanted.update(range(offset // self.block, (offset + size - 1) // self.block + 1))
        missing = sorted(i for i in wanted if i not in self._cache)
        if not missing:
            return

        def grab(idx):
            start = idx * self.block
            end = min(start + self.block, self.size) - 1
            r = requests.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                             timeout=self.timeout)
            r.raise_for_status()
            return idx, r.content

        with cf.ThreadPoolExecutor(max_workers=min(workers, len(missing))) as ex:
            for idx, blob in ex.map(grab, missing):
                self._cache[idx] = blob
                self.bytes_fetched += len(blob)

    def readinto(self, buf) -> int:
        end = min(self.pos + len(buf), self.size)
        written = 0
        while self.pos < end:
            idx = self.pos // self.block
            off = self.pos - idx * self.block
            chunk = self._block_at(idx)[off: off + (end - self.pos)]
            if not chunk:
                break
            buf[written: written + len(chunk)] = chunk
            written += len(chunk)
            self.pos += len(chunk)
        return written

    def close(self):
        self._cache.clear()
        try:
            self.session.close()
        finally:
            super().close()


# -------------------------------------------------------------- navigation

def _proj_transformer(lon0: float, height: float, sweep: str,
                      semi_major: float, semi_minor: float):
    from pyproj import CRS, Transformer
    crs = CRS.from_proj4(
        f"+proj=geos +h={height} +lon_0={lon0} +sweep={sweep} "
        f"+a={semi_major} +b={semi_minor} +units=m +no_defs")
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def _attr(attrs, name, default=None):
    """h5py hands back 0-d arrays, 1-element arrays or bytes depending on the
    attribute; flatten all of that to a plain Python value."""
    if name not in attrs:
        return default
    val = attrs[name]
    if isinstance(val, bytes):
        return val.decode()
    arr = np.asarray(val)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.ravel()[0].item()
    return arr


def _read_scaled(dset, rows: slice, cols: slice) -> np.ndarray:
    """Apply scale_factor / add_offset / _FillValue the way netCDF would."""
    raw = dset[rows, cols]
    out = raw.astype(np.float32)
    fill = _attr(dset.attrs, "_FillValue")
    if fill is not None:
        out[raw == fill] = np.nan
    scale = _attr(dset.attrs, "scale_factor")
    offset = _attr(dset.attrs, "add_offset")
    if scale is not None:
        out *= float(scale)
    if offset is not None:
        out += float(offset)
    return out


def _axis(dset) -> np.ndarray:
    vals = dset[:].astype(np.float64)
    scale = _attr(dset.attrs, "scale_factor")
    offset = _attr(dset.attrs, "add_offset")
    if scale is not None:
        vals = vals * float(scale)
    if offset is not None:
        vals = vals + float(offset)
    return vals


def scene_extent_lonlat(scene: Scene) -> Optional[Tuple[float, float, float, float]]:
    """
    (south, north, west, east) for a scene, read from its header only.
    Used to find out where a mesoscale sector is currently pointed, which is
    nowhere in the filename -- it has to come from the file.
    """
    handle = HttpRangeFile(scene.url)
    try:
        with h5py.File(handle, "r") as h:
            ext = h["geospatial_lat_lon_extent"].attrs
            return (float(_attr(ext, "geospatial_southbound_latitude")),
                    float(_attr(ext, "geospatial_northbound_latitude")),
                    float(_attr(ext, "geospatial_westbound_longitude")),
                    float(_attr(ext, "geospatial_eastbound_longitude")))
    except Exception:
        return None
    finally:
        handle.close()


def covers(extent: Sequence[float], lat: float, lon: float, margin: float = 0.0) -> bool:
    """
    Is (lat, lon) inside a (south, north, west, east) box? Handles boxes that
    straddle the dateline, where west is numerically greater than east --
    GOES-18's own full disk runs from 141.7E round to 55.7W, and a mesoscale
    sector over the Central Pacific can do the same.
    """
    south, north, west, east = extent
    if not (south - margin <= lat <= north + margin):
        return False
    lon = (lon + 180.0) % 360.0 - 180.0
    west = (west + 180.0) % 360.0 - 180.0
    east = (east + 180.0) % 360.0 - 180.0
    if west <= east:
        return west - margin <= lon <= east + margin
    return lon >= west - margin or lon <= east + margin   # wraps through 180


def load_image(scene: Scene, bbox: Optional[Tuple[float, float, float, float]] = None,
               max_pixels: int = 2400) -> GoesImage:
    """
    Read a scene, optionally cut down to a (south, north, west, east) box.

    The box is converted to fixed-grid scan angles and turned into row/column
    slices, so only the rows that matter come across the wire. Rows are the
    only axis worth trimming -- ABI chunks span the full image width.
    `max_pixels` decimates anything still larger than that, which keeps a
    band 2 full disk from becoming a 20k-square array in memory.
    """
    if h5py is None:
        raise RuntimeError("h5py is required to read GOES imagery (pip install h5py)")

    handle = HttpRangeFile(scene.url)
    try:
        with h5py.File(handle, "r") as h:
            proj = h["goes_imager_projection"].attrs
            lon0 = float(_attr(proj, "longitude_of_projection_origin"))
            height = float(_attr(proj, "perspective_point_height"))
            sweep = str(_attr(proj, "sweep_angle_axis", "x"))
            semi_major = float(_attr(proj, "semi_major_axis"))
            semi_minor = float(_attr(proj, "semi_minor_axis"))

            x = _axis(h["x"])   # scan angle, radians
            y = _axis(h["y"])
            cmi = h["CMI"]

            rows = slice(0, y.size)
            cols = slice(0, x.size)
            if bbox is not None:
                tf = _proj_transformer(lon0, height, sweep, semi_major, semi_minor)
                south, north, west, east = bbox
                lons = [west, east, west, east, (west + east) / 2, (west + east) / 2]
                lats = [south, south, north, north, south, north]
                xs, ys = tf.transform(lons, lats)
                xs = np.array(xs, dtype=float) / height   # metres -> scan radians
                ys = np.array(ys, dtype=float) / height
                good = np.isfinite(xs) & np.isfinite(ys)
                if good.any():
                    rows = _slice_for(y, ys[good])
                    cols = _slice_for(x, xs[good])

            step = 1
            nrows = rows.stop - rows.start
            ncols = cols.stop - cols.start
            if max(nrows, ncols) > max_pixels:
                step = int(np.ceil(max(nrows, ncols) / max_pixels))

            _prefetch_rows(handle, cmi, rows, cols)
            data = _read_scaled(cmi, rows, cols)[::step, ::step]

            extent_ll = None
            if "geospatial_lat_lon_extent" in h:
                ext = h["geospatial_lat_lon_extent"].attrs
                try:
                    extent_ll = (float(_attr(ext, "geospatial_southbound_latitude")),
                                 float(_attr(ext, "geospatial_northbound_latitude")),
                                 float(_attr(ext, "geospatial_westbound_longitude")),
                                 float(_attr(ext, "geospatial_eastbound_longitude")))
                except (TypeError, ValueError):
                    extent_ll = None
            xs_sel = x[cols][::step]
            ys_sel = y[rows][::step]

        kind = str(BANDS[scene.band]["kind"])
        if kind == "brightness":
            data = data - 273.15  # kelvin -> degC, what the IR table expects

        # Half-pixel out to the cell edges, and into metres for cartopy.
        dx = (xs_sel[1] - xs_sel[0]) if xs_sel.size > 1 else 0.0
        dy = (ys_sel[1] - ys_sel[0]) if ys_sel.size > 1 else 0.0
        extent_m = ((xs_sel[0] - dx / 2) * height, (xs_sel[-1] + dx / 2) * height,
                    (ys_sel[-1] + dy / 2) * height, (ys_sel[0] - dy / 2) * height)

        return GoesImage(data=data, extent_m=extent_m, lon0=lon0, sat_height=height,
                         sweep=sweep, scene=scene, kind=kind,
                         bytes_fetched=handle.bytes_fetched, extent_lonlat=extent_ll)
    finally:
        handle.close()


def _prefetch_rows(handle: HttpRangeFile, dset, rows: slice, cols: slice) -> None:
    """Fetch the chunks covering rows x cols in parallel before h5py asks."""
    try:
        chunks = dset.chunks
        if not chunks:
            return
        spans = []
        for r in range(rows.start - rows.start % chunks[0], rows.stop, chunks[0]):
            for c in range(cols.start - cols.start % chunks[1], cols.stop, chunks[1]):
                info = dset.id.get_chunk_info_by_coord((r, c))
                if info.byte_offset is not None:
                    spans.append((int(info.byte_offset), int(info.size)))
        handle.prefetch(spans)
    except Exception:
        pass  # purely an optimisation; h5py will fetch what it needs anyway


def _slice_for(axis: np.ndarray, wanted: np.ndarray, pad: int = 8) -> slice:
    """Index range of `axis` covering `wanted`; axis may ascend or descend."""
    lo, hi = float(np.min(wanted)), float(np.max(wanted))
    if axis[0] <= axis[-1]:
        i0, i1 = np.searchsorted(axis, [lo, hi])
    else:
        rev = axis[::-1]
        j0, j1 = np.searchsorted(rev, [lo, hi])
        i0, i1 = axis.size - j1, axis.size - j0
    i0 = int(max(0, min(axis.size - 2, i0 - pad)))
    i1 = int(max(i0 + 2, min(axis.size, i1 + pad)))
    return slice(i0, i1)


# ------------------------------------------------------- sector selection

def choose_scene(lat: float, lon: float, band: int, prefer: str = "auto",
                 when: Optional[dt.datetime] = None,
                 sat: Optional[int] = None) -> Tuple[Optional[Scene], str]:
    """
    Pick the scene to show for a storm position. Returns (scene, why).

    Mesoscale sectors are movable and their location lives only inside the
    file, so "is there a meso over this storm" means reading the header of the
    newest M1 and M2 (a few hundred KB each). Worth it: a meso sector refreshes
    every minute against the full disk's ten.
    """
    sat = sat or pick_satellite(lat, lon)

    if prefer in ("M1", "M2"):
        return latest_scene(sat, band, prefer, when), f"mesoscale {prefer} (forced)"
    if prefer == "F":
        return latest_scene(sat, band, "F", when), "full disk (forced)"

    for sector in ("M1", "M2"):
        scene = latest_scene(sat, band, sector, when)
        if scene is None:
            continue
        extent = scene_extent_lonlat(scene)
        if extent and covers(extent, lat, lon, margin=-0.5):
            return scene, f"mesoscale {sector} is over the storm"

    return latest_scene(sat, band, "F", when), "no mesoscale sector covers the storm"
