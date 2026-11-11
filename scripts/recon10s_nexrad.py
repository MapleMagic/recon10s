#!/usr/bin/env python3
"""
recon10s_nexrad — NEXRAD Level II radar for storms near land.

Carried over from MWSynth 0.169's radar_ingest.py (never run there, as it
was written without network or Py-ART):

  * Archive: s3://unidata-nexrad-level2/<YYYY>/<MM>/<DD>/<SITE>/<SITE><YYYYMMDD>_<HHMMSS>_V06
    (the bucket renamed from noaa-nexrad-level2 in 2025). Public, anonymous;
    listed here over plain HTTPS like the GOES buckets, so boto3 is not
    needed. *_MDM objects are metadata, not scans, and are skipped. Files
    before mid-2016 are gzipped; those are unpacked.
  * Reading with Py-ART (arm_pyart), as MWSynth chose.

Corrected from MWSynth after reading real files: a NEXRAD volume uses split
cuts at the low elevations -- at 0.5 deg one sweep carries reflectivity,
differential reflectivity and correlation coefficient, and a second sweep at
the same angle carries velocity. "Sweep 0" (MWSynth's choice) has no velocity
at all, so each product here takes the lowest sweep at the chosen elevation
that actually contains it. Velocity can be dealiased with Py-ART's
region-based method (~0.3 s a sweep); hurricane winds routinely exceed the
Nyquist velocity (~24 m/s on the low velocity cuts).

For display a sweep is resampled onto a square grid in an azimuthal
equidistant projection centred on the radar, so the map draws an image
rather than a million-cell mesh.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

os.environ.setdefault("PYART_QUIET", "1")    # Py-ART prints a banner on import otherwise

BUCKET_URL = "https://unidata-nexrad-level2.s3.amazonaws.com/"
CACHE_DIR = os.path.join(tempfile.gettempdir(), "recon10s_cache", "nexrad")
_S3 = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_KEY_RE = re.compile(r"([A-Z]{4})(\d{8})_(\d{6})(?:_V0\d)?(?:\.gz)?$")

PRODUCTS = {
    "reflectivity": {"label": "Reflectivity", "units": "dBZ", "field": "reflectivity"},
    "velocity": {"label": "Radial velocity", "units": "kt", "field": "velocity"},
    "zdr": {"label": "Differential reflectivity", "units": "dB", "field": "differential_reflectivity"},
    "cc": {"label": "Correlation coefficient", "units": "", "field": "cross_correlation_ratio"},
    # radial velocity with the storm's motion along each beam removed (needs the motion)
    "srv": {"label": "Storm-relative velocity", "units": "kt", "field": "velocity"},
}
BASE_PRODUCT = {"srv": "velocity"}
MS_TO_KT = 1.943844


@dataclass
class Scan:
    site: str
    time: dt.datetime
    key: str
    size: int = 0

    @property
    def url(self) -> str:
        return BUCKET_URL + self.key

    @property
    def label(self) -> str:
        return f"{self.time:%d %b %H:%M:%SZ}"


def parse_key(key: str) -> Optional[Tuple[str, dt.datetime]]:
    """'2026/09/27/PHWA/PHWA20260927_121858_V06' -> ('PHWA', datetime)."""
    name = key.rsplit("/", 1)[-1]
    if name.endswith("_MDM"):
        return None
    m = _KEY_RE.search(name)
    if not m:
        return None
    return m.group(1), dt.datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S").replace(
        tzinfo=dt.timezone.utc)


def list_scans(site: str, day: dt.date, timeout: float = 30.0) -> List[Scan]:
    prefix = f"{day:%Y/%m/%d}/{site}/"
    out, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET_URL, params=params, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for c in root.findall("s3:Contents", _S3):
            key = c.find("s3:Key", _S3).text
            parsed = parse_key(key)
            if parsed:
                out.append(Scan(site=parsed[0], time=parsed[1], key=key,
                                size=int(c.find("s3:Size", _S3).text)))
        if root.findtext("s3:IsTruncated", default="false", namespaces=_S3) != "true":
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=_S3)
    return sorted(out, key=lambda s: s.time)


def recent_scans(site: str, hours: float = 6.0, now: Optional[dt.datetime] = None) -> List[Scan]:
    """Scans from the last `hours`, oldest first (across a date change if needed)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(hours=hours)
    scans: List[Scan] = []
    day = since.date()
    while day <= now.date():
        scans += list_scans(site, day)
        day += dt.timedelta(days=1)
    return [s for s in scans if since <= s.time <= now]


def nearest_scan(scans: List[Scan], when: dt.datetime) -> Optional[Scan]:
    return min(scans, key=lambda s: abs((s.time - when).total_seconds())) if scans else None


def storm_relative(az: np.ndarray, vals_kt: np.ndarray, motion: Optional[Tuple[float, float]]) -> np.ndarray:
    """
    Radial velocity (kt, positive away from the radar) minus the storm
    motion's component along each ray: V_sr = V_r - M cos(azimuth - heading).
    `motion` is (heading toward, kt).
    """
    if motion is None:
        return vals_kt
    hd, kt = motion
    return vals_kt - kt * np.cos(np.radians(np.asarray(az, float) - hd))[:, None]


def download(scan: Scan, timeout: float = 90.0, tries: int = 3) -> str:
    """
    Fetch a scan once and cache it. S3 now and then answers a GET with a
    transient error (even "NoSuchBucket") that works a moment later, so a
    failed request is retried, and nothing but a real volume is cached.
    """
    import time
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, hashlib.sha1(scan.key.encode()).hexdigest()[:16])
    if not os.path.exists(path):
        data = None
        for attempt in range(tries):
            try:
                r = requests.get(scan.url, timeout=timeout)
                if r.ok and (r.content[:4] == b"AR2V" or r.content[:2] == b"\x1f\x8b" or
                             r.content[:4].isalnum()):
                    data = r.content
                    break
                last = f"HTTP {r.status_code}"
            except requests.RequestException as exc:
                last = str(exc)
            time.sleep(1.5 * (attempt + 1))
        if data is None:
            raise RuntimeError(f"Could not download {scan.key} ({last}).")
        if scan.key.endswith(".gz") or data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)                # pre-2016 archive files
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    return path


# ------------------------------------------------------------------ volumes

@dataclass
class Volume:
    scan: Scan
    radar: object                                   # pyart.core.Radar
    site_lat: float
    site_lon: float
    vcp: Optional[int] = None
    _dealiased: Dict[int, np.ndarray] = field(default_factory=dict)

    def elevations(self) -> List[float]:
        """Distinct elevation angles, lowest first (split cuts counted once)."""
        seen: List[float] = []
        for a in self.radar.fixed_angle["data"]:
            if not any(abs(a - s) < 0.1 for s in seen):
                seen.append(float(a))
        return sorted(seen)

    def sweep_for(self, product: str, elevation: float) -> Optional[int]:
        """The first sweep at `elevation` whose data actually include the product."""
        product = BASE_PRODUCT.get(product, product)
        fname = PRODUCTS[product]["field"]
        if fname not in self.radar.fields:
            return None
        data = self.radar.fields[fname]["data"]
        for i, a in enumerate(self.radar.fixed_angle["data"]):
            if abs(a - elevation) < 0.1:
                sl = self.radar.get_slice(i)
                if np.ma.count(data[sl]) > 0:
                    return i
        return None

    def sweep_data(self, product: str, sweep: int, dealias: bool = True):
        """(azimuths deg, ranges m, values 2-D float with NaN, elevation, nyquist or None)."""
        r = self.radar
        sl = r.get_slice(sweep)
        fname = PRODUCTS[product]["field"]
        nyq = None
        if product == "velocity":
            ip = r.instrument_parameters or {}
            if "nyquist_velocity" in ip:
                nyq = float(ip["nyquist_velocity"]["data"][sl.start])
            if dealias:
                if sweep not in self._dealiased:
                    import pyart
                    sw = r.extract_sweeps([sweep])
                    try:
                        dv = pyart.correct.dealias_region_based(sw, vel_field=fname)
                        self._dealiased[sweep] = np.ma.filled(dv["data"].astype(float), np.nan)
                    except Exception:
                        self._dealiased[sweep] = np.ma.filled(sw.fields[fname]["data"].astype(float), np.nan)
                vals = self._dealiased[sweep].copy()
            else:
                vals = np.ma.filled(r.fields[fname]["data"][sl].astype(float), np.nan)
            vals *= MS_TO_KT
        else:
            vals = np.ma.filled(r.fields[fname]["data"][sl].astype(float), np.nan)
        return (np.asarray(r.azimuth["data"][sl], float), np.asarray(r.range["data"], float), vals,
                float(r.fixed_angle["data"][sweep]), nyq)


def read_volume(path: str, scan: Scan) -> Volume:
    try:
        import pyart
    except ImportError as exc:
        raise RuntimeError("NEXRAD needs Py-ART: pip install arm_pyart") from exc
    radar = pyart.io.read_nexrad_archive(path)
    vcp = radar.metadata.get("vcp_pattern")
    return Volume(scan=scan, radar=radar, site_lat=float(radar.latitude["data"][0]),
                  site_lon=float(radar.longitude["data"][0]), vcp=int(vcp) if vcp is not None else None)


_VOLUMES: "OrderedDict[str, Volume]" = OrderedDict()


def load_volume(scan: Scan) -> Volume:
    """Download and read a scan, keeping the last few volumes in memory."""
    if scan.key in _VOLUMES:
        _VOLUMES.move_to_end(scan.key)
        return _VOLUMES[scan.key]
    vol = read_volume(download(scan), scan)
    _VOLUMES[scan.key] = vol
    while len(_VOLUMES) > 4:
        _VOLUMES.popitem(last=False)
    return vol


# ---------------------------------------------------------------- regridding

def to_grid(az: np.ndarray, rng: np.ndarray, vals: np.ndarray, elevation: float,
            centre_xy: Tuple[float, float] = (0.0, 0.0), half_width_m: float = 250e3,
            n: int = 800) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Resample one sweep onto an n x n grid of ground distance (metres) from the
    radar, in an azimuthal-equidistant frame. Each pixel takes the nearest
    ray and gate. Slant range is ground range / cos(elevation), which is
    within a gate of the full 4/3-earth geometry out to ~200 km at low tilts.
    Returns (image with row 0 at the top, (x0, x1, y0, y1) extent).
    """
    cx, cy = centre_xy
    xs = np.linspace(cx - half_width_m, cx + half_width_m, n)
    ys = np.linspace(cy + half_width_m, cy - half_width_m, n)
    X, Y = np.meshgrid(xs, ys)
    ground = np.hypot(X, Y)
    slant = ground / np.cos(np.radians(elevation))
    pix_az = (np.degrees(np.arctan2(X, Y)) + 360.0) % 360.0

    order = np.argsort(az)
    az_sorted = az[order]
    # circular nearest ray
    idx = np.searchsorted(az_sorted, pix_az)
    lo = (idx - 1) % az_sorted.size
    hi = idx % az_sorted.size
    d_lo = np.abs(((pix_az - az_sorted[lo]) + 180) % 360 - 180)
    d_hi = np.abs(((az_sorted[hi] - pix_az) + 180) % 360 - 180)
    ray = order[np.where(d_lo <= d_hi, lo, hi)]
    beam = np.minimum(d_lo, d_hi)
    spacing = 360.0 / max(az.size, 1)

    dr = float(rng[1] - rng[0]) if rng.size > 1 else 250.0
    gate = np.rint((slant - rng[0]) / dr).astype(int)
    ok = (gate >= 0) & (gate < rng.size) & (beam <= 1.5 * spacing)
    img = np.full(X.shape, np.nan)
    img[ok] = vals[ray[ok], gate[ok]]
    return img, (xs[0], xs[-1], ys[-1], ys[0])


def beam_height_km(ground_km: float, elevation: float, site_elev_m: float = 0.0) -> float:
    """Height of the beam centre above sea level (4/3 earth radius model)."""
    ae = 4.0 / 3.0 * 6371.0
    r = ground_km / np.cos(np.radians(elevation))
    h = np.sqrt(r ** 2 + ae ** 2 + 2 * r * ae * np.sin(np.radians(elevation))) - ae
    return float(h + site_elev_m / 1000.0)


# ------------------------------------------------------------------ colours

def colour_table(product: str):
    """(cmap, norm, ticks) approximating the familiar operational tables."""
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
    if product == "reflectivity":
        # The NWS default table (GR2Analyst .pal, bundled in palettes/): smooth
        # shading within each band, sharp steps at 10/20/35/50/65/75 dBZ.
        import recon10s_palette
        return recon10s_palette.reflectivity()
    if product in ("velocity", "srv"):
        stops = [(0.0, "#00ffff"), (0.12, "#00e000"), (0.35, "#006400"), (0.47, "#374037"),
                 (0.5, "#555555"), (0.53, "#403737"), (0.65, "#7f0000"), (0.88, "#ff2020"),
                 (1.0, "#ffff66")]
        return LinearSegmentedColormap.from_list("vel", stops), Normalize(-120, 120), [-120, -80, -40, 0, 40, 80, 120]
    if product == "zdr":
        cols = ["#5a5a5a", "#8c8c8c", "#3a64c8", "#34b4e6", "#3cc83c", "#b4e632", "#ffff3c",
                "#ffb400", "#ff5a00", "#e61e1e", "#c80096", "#ff96ff"]
        bounds = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6]
        return ListedColormap(cols), BoundaryNorm(bounds, len(cols)), [-2, 0, 1, 2, 3, 4, 6]
    if product == "cc":
        cols = ["#15137a", "#2b4bd1", "#3b8fe8", "#39c6e0", "#3fd07a", "#8adc3a", "#e8e83a",
                "#f5b02c", "#f06b24", "#d8262a", "#a4103e", "#e8b4f0"]
        bounds = [0.2, 0.45, 0.65, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.05]
        return ListedColormap(cols), BoundaryNorm(bounds, len(cols)), [0.2, 0.65, 0.8, 0.9, 0.95, 0.98, 1.0]
    raise ValueError(product)
