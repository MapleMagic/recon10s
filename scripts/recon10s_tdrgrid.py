#!/usr/bin/env python3
"""
recon10s_tdrgrid — the gridded tail Doppler radar analyses behind HRD's
real-time graphics.

NOAA's flight data server keeps, per P-3 mission,

    https://seb.omao.noaa.gov/pub/flight/radar/<mission>/

  <mid>_<HHMM>_xy.nc.gz            3-D analysis for one pass: storm-centred,
                                    2 km grid (250 x 250 = 500 km across),
                                    37 levels every 0.5 km from 0 to 18 km;
                                    U, V, W, vorticity, reflectivity, wind speed,
                                    radial and tangential wind, with latitude and
                                    longitude for every column
  <mid>_<HHMM>_vert_{inbound,outbound}[_fall|_rel].nc.gz
                                    along-leg sections (1.5 km x 150 m). In the
                                    real-time output these are placeholders --
                                    every value is the -999.9 fill -- so the leg
                                    section is cut from the 3-D grid instead,
                                    along the heading the file records
  *.w.gz                           the same analyses in HRD's older binary form
  <mid>_<t1>_<t2>_radials.so.gz    ASCII radial-velocity super-observations,
                                    prepared for data assimilation in forecast
                                    models rather than for display
  *_analysis.tar, *_jobfile.tar.gz processing configuration

Checked against HRD's own graphic for 260925I1 2211Z: the grid's 2 km
maximum wind is 76.1 m/s = 147.9 kt, exactly the "Max. Wind Speed" printed on
HRD's 2 km swath image. Winds are m/s; vorticity 10^-3 s^-1; missing -999.9.
The files are classic netCDF, so SciPy reads them (no netCDF4 needed).
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

BASE = "https://seb.omao.noaa.gov/pub/flight/radar"
CACHE_DIR = os.path.join(tempfile.gettempdir(), "recon10s_cache", "tdrgrid")
MS_TO_KT = 1.943844

FIELDS = {
    "REFLECTIVITY": ("Reflectivity", "dBZ"),
    "WIND_SPEED": ("Wind speed", "m/s"),
    "TANGENTIAL_WIND": ("Tangential wind", "m/s"),
    "RADIAL_WIND": ("Radial wind", "m/s"),
    "W": ("Vertical velocity", "m/s"),
    "VORT": ("Relative vorticity", "10\u207b\u00b3 s\u207b\u00b9"),
    "WIND_SPREAD": ("Uncertainty (1\u03c3)", "m/s"),          # synthetic TDR only
}
WINDS = {"WIND_SPEED", "TANGENTIAL_WIND", "RADIAL_WIND", "WIND_SPREAD"}
_HREF = re.compile(r'href="([^"?/]+)"')
_XY = re.compile(r"^(?P<mid>\d{6}[A-Z]\d)_(?P<t>\d{4})_xy\.nc\.gz$")
_VERT = re.compile(r"^(?P<mid>\d{6}[A-Z]\d)_(?P<t>\d{4})_vert_(?P<leg>inbound|outbound)(?P<var>_fall|_rel)?\.nc\.gz$")


@dataclass
class Analysis:
    mission: str
    time: str                      # "2211"
    xy: str                        # file name
    legs: Dict[str, str] = field(default_factory=dict)   # "inbound" -> file name
    available: Optional[dt.datetime] = None   # when HRD's job for it ran (its job file's stamp)

    def url(self, name: str) -> str:
        return f"{BASE}/{self.mission}/{name}"


_JOB = re.compile(r"^(?P<made>\d{14})_\d{8}[A-Z]\d_(?P<t>\d{4})\d{2}_jobfile\.tar\.gz$")


def url_for(name: str) -> str:
    """The server URL of an analysis file from its name alone (261008I1_1602_xy.nc.gz)."""
    return f"{BASE}/20{name[:8]}/{name}"


def list_analyses(mission: str, timeout: float = 45.0) -> List[Analysis]:
    r = requests.get(f"{BASE}/{mission}/", timeout=timeout)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    names = _HREF.findall(r.text)
    out: Dict[str, Analysis] = {}
    for n in names:
        m = _XY.match(n)
        if m:
            out.setdefault(m.group("t"), Analysis(mission, m.group("t"), n)).xy = n
    for n in names:
        m = _VERT.match(n)
        if m and m.group("t") in out and not m.group("var"):
            out[m.group("t")].legs[m.group("leg")] = n
        j = _JOB.match(n)
        if j and j.group("t") in out:
            out[j.group("t")].available = dt.datetime.strptime(j.group("made"), "%Y%m%d%H%M%S").replace(
                tzinfo=dt.timezone.utc)
    return [out[t] for t in sorted(out)]


def _fetch(url: str, timeout: float = 120.0) -> bytes:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, hashlib.sha1(url.encode()).hexdigest()[:16] + ".nc")
    if not os.path.exists(path):
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = gzip.decompress(r.content) if r.content[:2] == b"\x1f\x8b" else r.content
        with open(path + ".part", "wb") as fh:
            fh.write(data)
        os.replace(path + ".part", path)
    with open(path, "rb") as fh:
        return fh.read()


def _open(data: bytes):
    from scipy.io import netcdf_file
    return netcdf_file(io.BytesIO(data), "r", mmap=False)


def _clean(a) -> np.ndarray:
    a = np.array(a, dtype=float)
    a[(a < -900) | (a > 1e6)] = np.nan
    return a


def _attr(nc, name, default=None):
    v = nc._attributes.get(name, default)
    return v.decode() if isinstance(v, bytes) else v


@dataclass
class TdrGrid:
    name: str
    storm: str
    flight: str
    x: np.ndarray                 # km east of the centre
    y: np.ndarray                 # km north
    z: np.ndarray                 # km
    lat: np.ndarray               # (ny, nx)
    lon: np.ndarray
    fields: Dict[str, np.ndarray]  # name -> (nz, ny, nx)
    centre: Tuple[float, float]
    start: Optional[dt.datetime] = None
    end: Optional[dt.datetime] = None
    centre_time: Optional[dt.datetime] = None
    motion: Tuple[float, float] = (0.0, 0.0)    # east, north m/s

    @property
    def label(self) -> str:
        span = (f"{self.start:%H:%M}\u2013{self.end:%H:%MZ}" if self.start and self.end else "")
        return f"{self.flight} {self.storm} {span}".strip()

    def level(self, km: float) -> int:
        return int(np.argmin(np.abs(self.z - km)))

    def plan(self, name: str, km: float) -> np.ndarray:
        return self.fields[name][self.level(km)]

    def section(self, name: str, azimuth: float, max_r: float = 150.0, dr: float = 1.0):
        """
        Vertical section through the centre along `azimuth` (deg, towards):
        (distance km, negative behind the centre; heights km; values (nz, nr)).
        """
        from scipy.ndimage import map_coordinates
        r = np.arange(-max_r, max_r + dr / 2, dr)
        a = np.radians(azimuth)
        xs, ys = r * np.sin(a), r * np.cos(a)
        dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        cols = (xs - self.x[0]) / dx
        rows = (ys - self.y[0]) / dy
        vol = self.fields[name]
        out = np.full((vol.shape[0], r.size), np.nan)
        for k in range(vol.shape[0]):
            layer = vol[k]
            valid = np.isfinite(layer).astype(float)
            filled = np.where(np.isfinite(layer), layer, 0.0)
            num = map_coordinates(filled, [rows, cols], order=1, mode="constant", cval=0.0)
            den = map_coordinates(valid, [rows, cols], order=1, mode="constant", cval=0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[k] = np.where(den > 0.5, num / den, np.nan)
        return r, self.z, out

    def section_line(self, name: str, x0: float, y0: float, x1: float, y1: float, dr: float = 1.0):
        """
        Vertical section along any straight line from (x0, y0) to (x1, y1) km:
        (distance along the line km, heights km, values (nz, nr)).
        """
        from scipy.ndimage import map_coordinates
        length = float(np.hypot(x1 - x0, y1 - y0))
        s = np.arange(0.0, length + dr / 2, dr) if length > 0 else np.array([0.0])
        ux, uy = ((x1 - x0) / length, (y1 - y0) / length) if length > 0 else (0.0, 0.0)
        xs, ys = x0 + s * ux, y0 + s * uy
        dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        cols = (xs - self.x[0]) / dx
        rows = (ys - self.y[0]) / dy
        vol = self.fields[name]
        out = np.full((vol.shape[0], s.size), np.nan)
        for k in range(vol.shape[0]):
            layer = vol[k]
            valid = np.isfinite(layer).astype(float)
            filled = np.where(np.isfinite(layer), layer, 0.0)
            num = map_coordinates(filled, [rows, cols], order=1, mode="constant", cval=0.0)
            den = map_coordinates(valid, [rows, cols], order=1, mode="constant", cval=0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[k] = np.where(den > 0.5, num / den, np.nan)
        return s, self.z, out

    def has_winds(self) -> bool:
        return "U" in self.fields and "V" in self.fields or \
            ("TANGENTIAL_WIND" in self.fields and "RADIAL_WIND" in self.fields)

    def azimuthal_mean(self, name: str, dr: float = 2.0, max_r: float = 150.0, min_cover: float = 0.25):
        """
        Mean around the centre in rings dr wide: (radius km, heights km,
        values (nz, nr)). A ring needs at least `min_cover` of its points to
        have data, so a lone arc of echo is not reported as a mean.
        """
        X, Y = np.meshgrid(self.x, self.y)
        R = np.hypot(X, Y)
        edges = np.arange(0, max_r + dr, dr)
        centres = (edges[:-1] + edges[1:]) / 2
        vol = self.fields[name]
        out = np.full((vol.shape[0], centres.size), np.nan)
        for i in range(centres.size):
            ring = (R >= edges[i]) & (R < edges[i + 1])
            n = ring.sum()
            if not n:
                continue
            vals = vol[:, ring]
            good = np.isfinite(vals)
            cover = good.sum(axis=1) / n
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmean(np.where(good, vals, np.nan), axis=1)
            out[:, i] = np.where(cover >= min_cover, m, np.nan)
        return centres, self.z, out

    def values_at(self, x_km: float, y_km: float, z_km: float) -> Optional[dict]:
        """
        Every field at the grid point nearest (x, y, z), with its latitude and
        longitude, or None outside the grid.
        """
        dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        i = int(round((y_km - self.y[0]) / dy))
        j = int(round((x_km - self.x[0]) / dx))
        if not (0 <= i < self.y.size and 0 <= j < self.x.size):
            return None
        k = self.level(z_km)
        out = {name: float(vol[k, i, j]) for name, vol in self.fields.items()}
        out.update(lat=float(self.lat[i, j]), lon=float(self.lon[i, j]), level=float(self.z[k]))
        return out

    def rmw(self) -> Tuple[np.ndarray, np.ndarray]:
        """Radius of the maximum azimuthal-mean tangential wind at each height (NaN without winds)."""
        if "TANGENTIAL_WIND" not in self.fields:
            return self.z, np.full(self.z.size, np.nan)
        r, z, vt = self.azimuthal_mean("TANGENTIAL_WIND")
        out = np.full(z.size, np.nan)
        for k in range(z.size):
            if np.isfinite(vt[k]).any():
                out[k] = r[int(np.nanargmax(vt[k]))]
        return z, out


def load_grid(url: str) -> TdrGrid:
    nc = _open(_fetch(url))
    v = nc.variables
    x = np.array(v["x"][:], float)
    y = np.array(v["y"][:], float)
    z = np.array(v["level"][:], float)
    fields = {}
    for name in list(FIELDS) + ["U", "V"]:
        if name in v:
            # stored (x, y, level, time): make it (level, y, x) so rows run north-south
            fields[name] = np.transpose(_clean(v[name][..., 0]), (2, 1, 0))
    lat = np.transpose(np.array(v["LATITUDE"][..., 0], float))
    lon = np.transpose(np.array(v["LONGITUDE"][..., 0], float))
    day = None
    try:
        day = dt.date(int(_attr(nc, "FLIGHT YEAR")), int(_attr(nc, "FLIGHT MONTH")), int(_attr(nc, "FLIGHT DAY")))
    except (TypeError, ValueError):
        pass

    def when(prefix):
        try:
            t = dt.time(int(_attr(nc, f"{prefix} HOUR")), int(_attr(nc, f"{prefix} MINUTE")),
                        int(_attr(nc, f"{prefix} SECOND")))
            return dt.datetime.combine(day, t, tzinfo=dt.timezone.utc)
        except (TypeError, ValueError):
            return None
    start, end, mid = when("ANALYSIS START"), when("ANALYSIS END"), when("ANALYSIS CENTER")
    if start and end and end < start:
        end += dt.timedelta(days=1)
    if start and mid and mid < start:
        mid += dt.timedelta(days=1)
    try:
        motion = (float(_attr(nc, "EASTWARD STORM MOTION (METERS PER SECOND)")),
                  float(_attr(nc, "NORTHWARD STORM MOTION (METERS PER SECOND)")))
    except (TypeError, ValueError):
        motion = (0.0, 0.0)
    grid = TdrGrid(name=url.rsplit("/", 1)[-1], storm=str(_attr(nc, "STMNAME", "")).strip(),
                   flight=str(_attr(nc, "FLTNAME", "")).strip(), x=x, y=y, z=z, lat=lat, lon=lon,
                   fields=fields, centre=(float(_attr(nc, "ORIGIN_LATITUDE")), float(_attr(nc, "ORIGIN_LONGITUDE"))),
                   start=start, end=end, centre_time=mid, motion=motion)
    nc.close()
    return grid


@dataclass
class LegInfo:
    heading: Optional[float]
    has_data: bool


def leg_info(url: str) -> LegInfo:
    """The heading of a leg file, and whether it holds any data at all."""
    nc = _open(_fetch(url))
    v = nc.variables
    heading = float(np.array(v["heading"][:], float).ravel()[0]) if "heading" in v else None
    has = any(np.isfinite(_clean(v[k][:])).any() for k in ("Wind_Speed", "REFLECTIVITY") if k in v)
    nc.close()
    return LegInfo(heading, has)
