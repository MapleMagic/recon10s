#!/usr/bin/env python3
"""
recon10s_jma — JMA's 20 C-band Doppler radars, for typhoons near Japan.

  * Archive: NICT's mirror of JMA's per-radar polar GRIB2,
      https://pawr.nict.go.jp/jmadata/JMA-PolarCoordsRadar/<yyyy>/<mm>/<dd>/
    one TAR per 10-minute cycle for all 20 radars:
      Z__C_RJTD_<yyyyMMddhhmmss>_RDR_JMAGPV_N5_grib2.tar   reflectivity (Pze)
      Z__C_RJTD_<yyyyMMddhhmmss>_RDR_JMAGPV_N6_grib2.tar   Doppler velocity (Pvr)
    The mirror runs about two days behind real time.
  * Typhoon positions: knackwx's ATCF feed (recon10s_nhc.knack_storms), else
    NRL's GeoIPS sector file (one line per active system: id, name, date,
    time, lat, lon, basin, class, wind kt, pressure mb).
  * Decoding: recon10s_jmagrib (JMA's local GRIB2 templates).
  * JmaVolume has the same shape as the NEXRAD Volume, so the NEXRAD tab's
    resampling and drawing work unchanged.
  * write_cfradial: the decoded sweeps as CfRadial 1.4 netCDF, readable by
    LROSE (RadxConvert, HawkEye, RadxPrint), Py-ART and other radar tools.
    LROSE has no JMA GRIB2 reader of its own (checked in lrose-core-20260425:
    neither Radx nor its grib2 library knows templates 3.50120, 4.51022 or
    5.200), so this is the bridge.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import math
import os
import re
import tarfile
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

import recon10s_jmagrib as jg

ARCHIVE = "https://pawr.nict.go.jp/jmadata/JMA-PolarCoordsRadar/{y:04d}/{m:02d}/{d:02d}/"
SECTOR_FILE = "https://science.nrlmry.navy.mil/geoips/tcdat/sectors/updated_sector_file"
CACHE_DIR = os.path.join(tempfile.gettempdir(), "recon10s_cache", "jma")
_TAR_RE = re.compile(r"Z__C_RJTD_(\d{14})_RDR_JMAGPV_(N5|N6)_grib2\.tar")
REACH_KM = 400.0        # the long-range reflectivity sweeps
DOPPLER_KM = 250.0      # the long velocity sweeps
MS_TO_KT = 1.943844
HEADERS = {"User-Agent": "recon10s (hurricane reconnaissance viewer)"}


# ------------------------------------------------------------------ storms

@dataclass
class Storm:
    storm_id: str
    name: str
    time: dt.datetime
    lat: float
    lon: float
    basin: str
    kind: str
    wind_kt: Optional[int]
    pressure: Optional[int]

    @property
    def label(self) -> str:
        return (f"{self.kind} {self.name.title()} ({self.storm_id.upper()})  {abs(self.lat):.1f}"
                f"{'N' if self.lat >= 0 else 'S'} {abs(self.lon):.1f}{'E' if self.lon >= 0 else 'W'}  "
                f"{self.wind_kt or '?'} kt  {self.time:%d/%H%MZ}")


def parse_sectors(text: str) -> List[Storm]:
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 10:
            continue
        try:
            t = dt.datetime.strptime(p[2] + p[3], "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc)
            out.append(Storm(p[0].lower(), p[1], t, float(p[4]), float(p[5]), p[6], p[7],
                             int(p[8]) if p[8].isdigit() else None,
                             int(p[9]) if p[9].isdigit() else None))
        except ValueError:
            continue
    return out


def sector_storms(timeout: float = 30.0) -> List[Storm]:
    """Every system in NRL's sector file."""
    r = requests.get(SECTOR_FILE, timeout=timeout, headers=HEADERS)
    r.raise_for_status()
    return parse_sectors(r.text)


def active_storms(timeout: float = 30.0) -> List[Storm]:
    """Every active system: from knackwx's ATCF feed (fresher), else NRL's sector file."""
    try:
        import recon10s_nhc as nhc
        fixes = nhc.knack_storms(timeout=min(timeout, 20.0))
    except Exception:
        fixes = []
    out = []
    for f in fixes:
        if f.time is None:
            continue
        basin = (nhc.short_id(f.storm_id) or "  ")[-1]
        out.append(Storm(f.storm_id, f.name, f.time, f.lat, f.lon, basin, f.classification, f.vmax_kt, f.mslp_mb))
    return out or sector_storms(timeout)


# ------------------------------------------------------------------- sites

def _km(lat1, lon1, lat2, lon2) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(min(1.0, a)))


def sites_by_distance(lat: float, lon: float) -> List[Tuple[str, float]]:
    """(WMO number, km) for every JMA radar, nearest first."""
    return sorted(((sid, _km(lat, lon, *pos)) for sid, pos in jg.SITE_POS.items()), key=lambda x: x[1])


def nearest_site(lat: float, lon: float, max_km: float = REACH_KM) -> Tuple[Optional[str], Optional[float]]:
    """The nearest radar within reach, or (None, None) -- no far-away site for a storm out of range."""
    sid, d = sites_by_distance(lat, lon)[0]
    return (sid, d) if d <= max_km else (None, None)


# ----------------------------------------------------------------- archive

@dataclass
class Cycle:
    time: dt.datetime
    n5: Optional[str] = None
    n6: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.time:%d %b %H:%MZ}"


def list_cycles(day: dt.date, timeout: float = 45.0) -> List[Cycle]:
    base = ARCHIVE.format(y=day.year, m=day.month, d=day.day)
    r = requests.get(base, timeout=timeout, headers=HEADERS)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    cycles: Dict[str, Cycle] = {}
    for stamp, kind in set(_TAR_RE.findall(r.text)):
        c = cycles.setdefault(stamp, Cycle(dt.datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(
            tzinfo=dt.timezone.utc)))
        url = base + f"Z__C_RJTD_{stamp}_RDR_JMAGPV_{kind}_grib2.tar"
        if kind == "N5":
            c.n5 = url
        else:
            c.n6 = url
    return [cycles[k] for k in sorted(cycles)]


def latest_day(back: int = 6) -> Optional[dt.date]:
    """The newest day the mirror has (it runs a couple of days behind)."""
    today = dt.datetime.now(dt.timezone.utc).date()
    for k in range(back + 1):
        d = today - dt.timedelta(days=k)
        r = requests.get(ARCHIVE.format(y=d.year, m=d.month, d=d.day), timeout=30, headers=HEADERS)
        if r.ok and _TAR_RE.search(r.text):
            return d
    return None


def fetch_tar(url: str, timeout: float = 180.0) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, hashlib.sha1(url.encode()).hexdigest()[:16] + ".tar")
    if not os.path.exists(path):
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        r.raise_for_status()
        with open(path + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(path + ".part", path)
    return path


def site_file(tar_path: str, site_id: str) -> Optional[jg.SiteFile]:
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            if f"_RS{site_id}_" in m.name:
                return jg.decode(t.extractfile(m).read())
    return None


def sites_in(tar_path: str) -> List[str]:
    with tarfile.open(tar_path) as t:
        return sorted({m.name.split("_RS")[1][:5] for m in t.getmembers() if "_RS" in m.name})


# ------------------------------------------------------------------ volume

@dataclass
class JmaScan:
    site: str
    time: dt.datetime
    key: str
    size: int = 0

    @property
    def label(self) -> str:
        return f"{self.time:%d %b %H:%M:%SZ}"


class JmaVolume:
    """
    A radar's reflectivity and velocity for one cycle, in the NEXRAD Volume's
    shape (elevations, sweep_for, sweep_data, site_lat/lon, scan, vcp).
    """

    def __init__(self, refl: Optional[jg.SiteFile], vel: Optional[jg.SiteFile], cycle_time: dt.datetime):
        self.refl, self.vel = refl, vel
        base = refl or vel
        self.site_lat, self.site_lon = base.lat, base.lon
        self.site_id = base.site_id
        self.vcp = None
        self.scan = JmaScan(site=base.site_code or base.site_id, time=cycle_time,
                            key=f"{base.site_id}-{cycle_time:%Y%m%d%H%M}")
        self.site_label = f"{base.name} ({base.site_code}, {base.site_id}), JMA"
        self.site_height_m = base.height_m

    def _file(self, product):
        return self.refl if product == "reflectivity" else self.vel if product == "velocity" else None

    def elevations(self) -> List[float]:
        seen: List[float] = []
        for f in (self.refl, self.vel):
            for s in (f.sweeps if f else []):
                if not any(abs(s.elevation - e) < 0.05 for e in seen):
                    seen.append(s.elevation)
        return sorted(seen)

    def sweep_for(self, product: str, elevation: float) -> Optional[int]:
        """The longest-range sweep at that elevation, the later one when equal."""
        f = self._file(product)
        if f is None:
            return None
        best = None
        for i, s in enumerate(f.sweeps):
            if abs(s.elevation - elevation) < 0.05 and np.isfinite(s.values).any():
                if best is None or (s.range_m[-1], s.start) >= (f.sweeps[best].range_m[-1], f.sweeps[best].start):
                    best = i
        if best is None:          # nothing but "no echo" at this elevation: still show the sweep
            for i, s in enumerate(f.sweeps):
                if abs(s.elevation - elevation) < 0.05:
                    best = i
        return best

    def sweep_data(self, product: str, sweep: int, dealias: bool = True):
        """(azimuths, ranges m, values, elevation, Nyquist m/s or None). Velocity in knots."""
        s = self._file(product).sweeps[sweep]
        vals = s.values.copy()
        nyq = None
        if product == "velocity":
            vals = vals * MS_TO_KT                     # already dual-PRF unfolded by JMA
        return s.azimuth, s.range_m, vals, s.elevation, nyq

    def sweep_info(self, product: str, sweep: int) -> str:
        s = self._file(product).sweeps[sweep]
        rng = f"{s.range_m[-1] / 1e3:.0f} km"
        if not s.prf:
            # JMA merges several scans into one sweep here; its stored times overlap.
            return f"{min(s.start, s.end):%H:%M:%SZ}, {rng}, composite of several scans"
        prf = "/".join(f"{p:.0f}" for p in s.prf)
        return f"{s.start:%H:%M:%S}\u2013{s.end:%H:%M:%SZ}, {rng}, PRF {prf} Hz"


_VOLUMES: "OrderedDict[str, JmaVolume]" = OrderedDict()


def load_volume(cycle: Cycle, site_id: str) -> JmaVolume:
    key = f"{site_id}-{cycle.time:%Y%m%d%H%M}"
    if key in _VOLUMES:
        _VOLUMES.move_to_end(key)
        return _VOLUMES[key]
    refl = site_file(fetch_tar(cycle.n5), site_id) if cycle.n5 else None
    vel = site_file(fetch_tar(cycle.n6), site_id) if cycle.n6 else None
    if refl is None and vel is None:
        raise RuntimeError(f"Radar {site_id} is not in the {cycle.label} cycle.")
    vol = JmaVolume(refl, vel, cycle.time)
    _VOLUMES[key] = vol
    while len(_VOLUMES) > 6:
        _VOLUMES.popitem(last=False)
    return vol


# ---------------------------------------------------------------- CfRadial

def write_cfradial(path: str, sf: jg.SiteFile, field_name: Optional[str] = None) -> str:
    """
    One site file (all its sweeps) as CfRadial 1.4, classic netCDF, readable by
    LROSE's Radx tools and Py-ART. Sweeps with fewer gates are padded with the
    fill value, as CfRadial requires a single range dimension.
    """
    from scipy.io import netcdf_file
    prod = sf.sweeps[0].product
    name = field_name or {"reflectivity": "DBZ", "velocity": "VEL"}.get(prod, prod.upper())
    units = {"reflectivity": "dBZ", "velocity": "m/s"}.get(prod, "")
    std = {"reflectivity": "equivalent_reflectivity_factor",
           "velocity": "radial_velocity_of_scatterers_away_from_instrument"}.get(prod, prod)
    nr = [s.values.shape[0] for s in sf.sweeps]
    ngates = max(s.values.shape[1] for s in sf.sweeps)
    dr = float(sf.sweeps[0].range_m[1] - sf.sweeps[0].range_m[0])
    first = float(sf.sweeps[0].range_m[0])
    t0 = min(s.start for s in sf.sweeps)
    t1 = max(s.end for s in sf.sweeps)
    fill = -9999.0

    times, az, el, data = [], [], [], []
    starts, ends, fixed = [], [], []
    k = 0
    for s in sf.sweeps:
        n = s.values.shape[0]
        span = max((s.end - s.start).total_seconds(), 0.0)
        # Rays are in azimuth order from north; time them in scan order from the start azimuth.
        seq = np.argsort(np.argsort((s.azimuth - s.azimuth[0]) % 360.0))
        times.append((s.start - t0).total_seconds() + span * seq / max(n - 1, 1))
        az.append(s.azimuth)
        el.append(np.where(np.isfinite(s.ray_elevation), s.ray_elevation, s.elevation))
        block = np.full((n, ngates), fill, dtype=np.float32)
        v = s.values.astype(np.float32)
        block[:, :v.shape[1]] = np.where(np.isfinite(v), v, fill)
        data.append(block)
        starts.append(k)
        ends.append(k + n - 1)
        fixed.append(s.elevation)
        k += n

    nc = netcdf_file(path, "w", version=2)
    nc.Conventions = "CF/Radial instrument_parameters"
    nc.version = "1.4"
    nc.title = f"JMA {sf.name} radar {prod}"
    nc.institution = "Japan Meteorological Agency (via NICT); converted by recon10s"
    nc.source = "JMA per-radar polar-coordinate GRIB2 (templates 3.50120 / 4.51022 / 5.200)"
    nc.instrument_name = sf.site_code or sf.site_id
    nc.site_name = sf.name
    nc.platform_type = "fixed"
    nc.primary_axis = "axis_z"
    nc.time_coverage_start = t0.strftime("%Y-%m-%dT%H:%M:%SZ")
    nc.time_coverage_end = t1.strftime("%Y-%m-%dT%H:%M:%SZ")
    nc.history = f"recon10s: decoded from JMA GRIB2 at {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}"

    nc.createDimension("time", k)
    nc.createDimension("range", ngates)
    nc.createDimension("sweep", len(nr))
    nc.createDimension("string_length", 32)

    def var(name, typ, dims, values, **attrs):
        v = nc.createVariable(name, typ, dims)
        if dims:
            v[:] = values
        else:
            v.data[()] = values               # scalars (assignValue fails on 0-d arrays)
        for a, val in attrs.items():
            setattr(v, a, val)
        return v

    var("volume_number", "i", (), 0)
    var("time_coverage_start", "c", ("string_length",),
        np.array(list(nc.time_coverage_start.ljust(32)), dtype="S1"))
    var("time_coverage_end", "c", ("string_length",),
        np.array(list(nc.time_coverage_end.ljust(32)), dtype="S1"))
    var("latitude", "d", (), sf.lat, units="degrees_north", long_name="latitude")
    var("longitude", "d", (), sf.lon, units="degrees_east", long_name="longitude")
    var("altitude", "d", (), sf.height_m, units="meters", long_name="altitude", positive="up")
    var("time", "d", ("time",), np.concatenate(times), standard_name="time",
        long_name="time in seconds since volume start",
        units=f"seconds since {t0:%Y-%m-%dT%H:%M:%SZ}", calendar="gregorian")
    var("range", "f", ("range",), (first + np.arange(ngates) * dr).astype(np.float32),
        standard_name="projection_range_coordinate", long_name="range_to_center_of_measurement_volume",
        units="meters", axis="radial_range_coordinate", meters_to_center_of_first_gate=first,
        meters_between_gates=dr)
    var("azimuth", "f", ("time",), np.concatenate(az).astype(np.float32),
        standard_name="ray_azimuth_angle", units="degrees", axis="radial_azimuth_coordinate")
    var("elevation", "f", ("time",), np.concatenate(el).astype(np.float32),
        standard_name="ray_elevation_angle", units="degrees", axis="radial_elevation_coordinate",
        positive="up")
    var("sweep_number", "i", ("sweep",), np.arange(len(nr), dtype=np.int32), long_name="sweep_index_number_0_based")
    mode = np.array([list("azimuth_surveillance".ljust(32))] * len(nr), dtype="S1")
    var("sweep_mode", "c", ("sweep", "string_length"), mode, long_name="scan_mode_for_sweep")
    var("fixed_angle", "f", ("sweep",), np.array(fixed, dtype=np.float32),
        long_name="ray_target_fixed_angle", units="degrees")
    var("sweep_start_ray_index", "i", ("sweep",), np.array(starts, dtype=np.int32),
        long_name="index_of_first_ray_in_sweep")
    var("sweep_end_ray_index", "i", ("sweep",), np.array(ends, dtype=np.int32),
        long_name="index_of_last_ray_in_sweep")
    if sf.frequency_mhz:
        var("frequency", "f", (), np.float32(sf.frequency_mhz * 1e6), units="s-1",
            long_name="radiation_frequency", meta_group="instrument_parameters")
    var(name, "f", ("time", "range"), np.concatenate(data), long_name=prod, standard_name=std,
        units=units, _FillValue=np.float32(fill), coordinates="elevation azimuth range")
    nc.close()
    return path
