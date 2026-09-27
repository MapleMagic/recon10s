#!/usr/bin/env python3
"""
recon10s_mw — real passive-microwave passes from the PPS near-real-time server.

Server: https://jsimpsonhttps.pps.eosdis.nasa.gov/1C/<SENSOR>/, HTTP Basic auth,
where the username and the password are both the registered email address.
Each sensor directory is a flat Apache listing of 1C granules named like

    1C.WSFM1.MWI.XCAL2026-V.20260814-S230958-E231956.V08A.RT-NC

Three things this module does that are not obvious:

  1. It does not look inside every granule to find one over the storm. GMI NRT
     granules are five minutes long, so a twelve-hour lookback is ~140 files.
     Instead it propagates each satellite's orbit (TLE + SGP4) across every
     granule's time window and only downloads the ones whose ground track
     comes within reach of the storm. Typically that is one to three files.

  2. It never hardcodes which array slot holds 37 or 89 GHz. 1C files split
     channels across swath groups S1..Sn, and the group layout differs by
     sensor -- WSF-M/MWI puts each frequency pair in its own group. Every
     Tc variable carries a LongName like
         "Intercalibrated Tb for channels 1) 10.85 GHz V-Pol and 2) 10.85 GHz H-Pol"
     and the channel map is parsed out of that. This is the approach MWSynth
     confirmed against a real MWI file. If a file does not follow it, loading
     fails loudly with a dump of what the file does contain, rather than
     quietly labelling some other channel as 89 GHz.

  3. Credentials are never written anywhere unless you press Save. Then they
     go to jsimpson.json beside the scripts, in plain text.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

HERE = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(HERE, "jsimpson.json")
BASE_URL = "https://jsimpsonhttps.pps.eosdis.nasa.gov/1C"
CACHE_DIR = os.path.join(tempfile.gettempdir(), "recon10s_cache", "mw")
TLE_CACHE = os.path.join(CACHE_DIR, "tle.json")
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=TLE"

# reach_km is how far from the sub-satellite point the swath can see the
# storm. Deliberately generous: a false positive costs one download, a false
# negative silently hides a pass. Conical scanners also look ahead of the
# satellite, which the time padding below covers.
SENSORS: Dict[str, Dict[str, object]] = {
    "GMI":   {"dir": "GMI",   "platform": "GPM",      "catnr": 39574, "reach_km": 650},
    "AMSR3": {"dir": "AMSR3", "platform": "GOSAT-GW", "catnr": 64694, "reach_km": 950},
    "WSFM":  {"dir": "WSFM",  "platform": "WSF-M",    "catnr": 59481, "reach_km": 950},
}
TIME_PAD = dt.timedelta(minutes=3)

# Last-resort orbits, bundled with the release (epoch 2026 day 267). Used only
# when CelesTrak cannot be reached and nothing is cached. SGP4 drift is a few
# km per day for these orbits, so they stay good enough to screen passes for
# several weeks -- the swath reach above has plenty of slack for that.
BUNDLED_TLES: Dict[int, Tuple[str, str]] = {
    39574: ("1 39574U 14009C   26267.89861186  .00010417  00000+0  22356-3 0  9995",
            "2 39574  64.9723 334.0014 0010914 265.1835  94.8059 15.45102901713218"),
    64694: ("1 64694U 25141A   26267.93965205  .00000473  00000+0  91836-4 0  9998",
            "2 64694  98.0613 203.6461 0001444  84.8919 275.2450 14.67553185 66475"),
    59481: ("1 59481U 24070A   26267.87594000  .00000037  00000+0  37364-4 0  9992",
            "2 59481  98.7203 274.4853 0005494  45.2895 314.8728 14.20751088127252"),
}

_GRANULE_RE = re.compile(r"(?P<date>\d{8})-S(?P<s>\d{6})-E(?P<e>\d{6})")
_HREF_RE = re.compile(r'href="(1C\.[^"/?#]+)"', re.IGNORECASE)
_CHANNEL_RE = re.compile(r"(\d+)\)\s*([\d.]+)\s*GHz\s*([VH])-?Pol", re.IGNORECASE)


# ------------------------------------------------------------- credentials

def load_saved_email() -> Optional[str]:
    """The email from jsimpson.json, if the user ever chose to save one."""
    try:
        with open(CREDENTIALS_FILE, "r", encoding="utf-8") as fh:
            email = json.load(fh).get("email", "").strip()
        return email or None
    except (OSError, ValueError):
        return None


def save_email(email: str) -> str:
    with open(CREDENTIALS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"email": email.strip()}, fh, indent=2)
    try:
        os.chmod(CREDENTIALS_FILE, 0o600)  # owner-only where the OS supports it
    except OSError:
        pass
    return CREDENTIALS_FILE


def forget_saved_email() -> bool:
    try:
        os.remove(CREDENTIALS_FILE)
        return True
    except FileNotFoundError:
        return False


def _auth(email: str) -> Tuple[str, str]:
    """PPS NRT convention: the email is both the username and the password."""
    email = email.strip()
    return (email, email)


def check_login(email: str, timeout: float = 20.0) -> Tuple[bool, str]:
    """True if the server accepts these credentials."""
    try:
        r = requests.head(f"{BASE_URL}/GMI/", auth=_auth(email), timeout=timeout)
    except requests.RequestException as exc:
        return False, f"Could not reach the PPS server: {exc}"
    if r.status_code == 200:
        return True, "Signed in."
    if r.status_code == 401:
        return False, ("The server rejected that email. It has to be registered for "
                       "near-real-time access at registration.pps.eosdis.nasa.gov.")
    return False, f"Unexpected reply from the server (HTTP {r.status_code})."


# ---------------------------------------------------------------- granules

@dataclass
class Granule:
    sensor: str
    filename: str
    start: dt.datetime
    end: dt.datetime

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{SENSORS[self.sensor]['dir']}/{self.filename}"


@dataclass
class Candidate:
    """A granule that is predicted to see the storm."""
    granule: Granule
    closest_km: Optional[float] = None
    closest_time: Optional[dt.datetime] = None
    screened: bool = True  # False if there was no orbit to screen with

    @property
    def label(self) -> str:
        g = self.granule
        when = (self.closest_time or g.start).strftime("%d %b %H:%MZ")
        if self.closest_km is None:
            where = "not orbit-screened"
        else:
            where = f"track {self.closest_km:.0f} km from centre"
        return f"{g.sensor:<5}  {when}  \u00b7  {where}"


def _parse_granule(sensor: str, filename: str) -> Optional[Granule]:
    m = _GRANULE_RE.search(filename)
    if not m:
        return None
    day = dt.datetime.strptime(m.group("date"), "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    start = day + dt.timedelta(hours=int(m.group("s")[:2]), minutes=int(m.group("s")[2:4]),
                               seconds=int(m.group("s")[4:6]))
    end = day + dt.timedelta(hours=int(m.group("e")[:2]), minutes=int(m.group("e")[2:4]),
                             seconds=int(m.group("e")[4:6]))
    if end < start:  # granule crosses midnight; the date is the start date
        end += dt.timedelta(days=1)
    return Granule(sensor=sensor, filename=filename, start=start, end=end)


def list_granules(sensor: str, email: str, since: dt.datetime,
                  until: Optional[dt.datetime] = None, timeout: float = 60.0) -> List[Granule]:
    """Granules for one sensor whose time window overlaps [since, until]."""
    until = until or dt.datetime.now(dt.timezone.utc)
    r = requests.get(f"{BASE_URL}/{SENSORS[sensor]['dir']}/", auth=_auth(email), timeout=timeout)
    if r.status_code == 401:
        raise PermissionError("PPS rejected the credentials (HTTP 401).")
    r.raise_for_status()

    seen, out = set(), []
    for name in _HREF_RE.findall(r.text):
        if name in seen:
            continue
        seen.add(name)
        g = _parse_granule(sensor, name)
        if g and g.end >= since and g.start <= until:
            out.append(g)
    out.sort(key=lambda g: g.start)
    return out


# --------------------------------------------------------------- orbits

def _load_tle_cache() -> dict:
    try:
        with open(TLE_CACHE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def get_tle(catnr: int, max_age_hours: float = 24.0) -> Optional[Tuple[str, str]]:
    """
    TLE for a catalog number. Cached on disk: CelesTrak is not always quick to
    answer, and a day-old TLE is still good to a few km for this purpose. A
    stale cached copy beats none if the fetch fails.
    """
    cache = _load_tle_cache()
    hit = cache.get(str(catnr))
    now = time.time()
    if hit and now - hit.get("fetched", 0) < max_age_hours * 3600:
        return hit["l1"], hit["l2"]

    for attempt in range(3):
        try:
            r = requests.get(TLE_URL.format(catnr=catnr), timeout=25)
            lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
            l1 = next((ln for ln in lines if ln.startswith("1 ")), None)
            l2 = next((ln for ln in lines if ln.startswith("2 ")), None)
            if l1 and l2:
                cache[str(catnr)] = {"l1": l1, "l2": l2, "fetched": now}
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(TLE_CACHE, "w", encoding="utf-8") as fh:
                    json.dump(cache, fh)
                return l1, l2
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))

    if hit:
        return hit["l1"], hit["l2"]
    return BUNDLED_TLES.get(int(catnr))


def tle_age_days(tle: Tuple[str, str]) -> float:
    """Days since the TLE epoch, so stale orbits can be flagged."""
    epoch = tle[0][18:32]
    year = 2000 + int(epoch[:2])
    t = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=float(epoch[2:]) - 1)
    return (dt.datetime.now(dt.timezone.utc) - t).total_seconds() / 86400.0


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def subsatellite_track(tle: Tuple[str, str], times: List[dt.datetime]):
    """(lats, lons) of the sub-satellite point at each time, via SGP4."""
    from sgp4.api import Satrec, jday

    sat = Satrec.twoline2rv(*tle)
    lats, lons = [], []
    for t in times:
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)
        err, r, _ = sat.sgp4(jd, fr)
        if err:
            lats.append(np.nan); lons.append(np.nan)
            continue
        # TEME -> Earth-fixed by rotating through Greenwich sidereal time.
        # Ignoring polar motion and the TEME/PEF distinction costs well under
        # a kilometre, which is noise next to a several-hundred-km swath.
        d = jd - 2451545.0 + fr
        gmst = math.radians((280.46061837 + 360.98564736629 * d) % 360.0)
        x = r[0] * math.cos(gmst) + r[1] * math.sin(gmst)
        y = -r[0] * math.sin(gmst) + r[1] * math.cos(gmst)
        z = r[2]
        lon = math.degrees(math.atan2(y, x))
        # Geodetic latitude on WGS-84, one Bowring-style iteration is plenty.
        e2 = 6.69437999014e-3
        p = math.hypot(x, y)
        lat = math.atan2(z, p * (1 - e2))
        for _ in range(3):
            n = 6378.137 / math.sqrt(1 - e2 * math.sin(lat) ** 2)
            lat = math.atan2(z + e2 * n * math.sin(lat), p)
        lats.append(math.degrees(lat))
        lons.append(lon)
    return np.array(lats), np.array(lons)


def find_passes(sensors: List[str], lat: float, lon: float, email: str,
                hours: float = 12.0, now: Optional[dt.datetime] = None,
                progress=None) -> List[Candidate]:
    """
    Granules from the last `hours` whose ground track reaches the storm,
    newest first. `progress` is an optional callable taking a status string.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(hours=hours)
    out: List[Candidate] = []

    for sensor in sensors:
        meta = SENSORS[sensor]
        if progress:
            progress(f"Listing {sensor} granules\u2026")
        granules = list_granules(sensor, email, since, now)
        if not granules:
            continue

        tle = get_tle(int(meta["catnr"]))
        if tle is None:
            # No orbit to screen with: offer the newest few rather than all.
            for g in granules[-6:]:
                out.append(Candidate(granule=g, screened=False))
            continue

        if progress:
            progress(f"Screening {len(granules)} {sensor} granules against the orbit\u2026")
        reach = float(meta["reach_km"])
        for g in granules:
            t0, t1 = g.start - TIME_PAD, g.end + TIME_PAD
            n = max(3, int((t1 - t0).total_seconds() // 20) + 1)
            times = [t0 + (t1 - t0) * (i / (n - 1)) for i in range(n)]
            slat, slon = subsatellite_track(tle, times)
            dist = _haversine_km(slat, slon, lat, lon)
            if not np.isfinite(dist).any():
                continue
            i = int(np.nanargmin(dist))
            if dist[i] <= reach:
                out.append(Candidate(granule=g, closest_km=float(dist[i]),
                                     closest_time=times[i]))

    out.sort(key=lambda c: c.closest_time or c.granule.start, reverse=True)
    return out


# ---------------------------------------------------------------- download

def download_granule(granule: Granule, email: str, progress=None,
                     timeout: float = 180.0) -> str:
    """Fetch a granule to the cache (once) and return the local path."""
    folder = os.path.join(CACHE_DIR, granule.sensor)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, granule.filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    tmp = path + ".part"
    with requests.get(granule.url, auth=_auth(email), stream=True, timeout=timeout) as r:
        if r.status_code == 401:
            raise PermissionError("PPS rejected the credentials (HTTP 401).")
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0) or None
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
    os.replace(tmp, path)  # a half-written file never looks like a cached one
    return path


# ------------------------------------------------------ channel discovery

def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return " ".join(_text(v) for v in value.ravel())
    return str(value)


def parse_channel_longname(long_name: str) -> Dict[int, Tuple[float, str]]:
    """'channels 1) 10.85 GHz V-Pol and 2) 10.85 GHz H-Pol' -> {0: (10.85,'V'), 1: (10.85,'H')}"""
    return {int(i) - 1: (float(f), p.upper()) for i, f, p in _CHANNEL_RE.findall(long_name)}


def _band_of(freq: float) -> Optional[str]:
    if 35.0 <= freq <= 38.5:   # GMI 36.64, AMSR 36.5, MWI 36.75
        return "37"
    if 85.0 <= freq <= 92.5:   # 89.0 everywhere; 91.665 on SSMIS if ever added
        return "89"
    return None


def discover_channels(h) -> Dict[str, Tuple[str, int]]:
    """
    {"37V": (group, index), "37H": ..., "89V": ..., "89H": ...} from the Tc
    LongName of each swath group. First match wins, so on AMSR, which carries
    89 GHz twice (A and B horns), the first group is used.
    """
    found: Dict[str, Tuple[str, int]] = {}
    seen: Dict[str, str] = {}
    for name in sorted(h.keys(), key=lambda s: (len(s), s)):
        grp = h[name]
        if not hasattr(grp, "keys") or "Tc" not in grp:
            continue
        attrs = grp["Tc"].attrs
        long_name = _text(attrs.get("LongName", attrs.get("long_name", "")))
        seen[name] = long_name
        for idx, (freq, pol) in parse_channel_longname(long_name).items():
            band = _band_of(freq)
            if band and f"{band}{pol}" not in found:
                found[f"{band}{pol}"] = (name, idx)

    missing = {"37V", "37H", "89V", "89H"} - set(found)
    if missing:
        detail = "; ".join(f"{g}: {ln[:160] or '(no LongName)'}" for g, ln in seen.items()) or "no Tc variables"
        raise RuntimeError(
            f"Could not identify channels {sorted(missing)} in this file. "
            f"Swath groups and their Tc LongName: {detail}")
    return found


# ------------------------------------------------------------------ swaths

@dataclass
class BandSwath:
    """One frequency's V and H, flattened to the pixels near the storm."""
    lat: np.ndarray
    lon: np.ndarray
    v: np.ndarray
    h: np.ndarray
    scan: np.ndarray
    pixel: np.ndarray
    group: str


@dataclass
class MWPass:
    sensor: str
    start: dt.datetime
    end: dt.datetime
    filename: str
    centre: Tuple[float, float]
    bands: Dict[str, BandSwath] = field(default_factory=dict)
    covers_centre: bool = True

    @property
    def title(self) -> str:
        platform = SENSORS.get(self.sensor, {}).get("platform", "")
        return f"{platform} {self.sensor}  {self.start:%d %b %Y %H:%M}\u2013{self.end:%H:%MZ}"


def _read_tc(grp, idx: int) -> np.ndarray:
    tc = grp["Tc"]
    arr = tc[..., idx].astype(np.float32) if tc.ndim == 3 else tc[...].astype(np.float32)
    fill = tc.attrs.get("_FillValue")
    if fill is not None:
        arr[arr == np.asarray(fill).ravel()[0]] = np.nan
    arr[(arr < 20.0) | (arr > 400.0)] = np.nan  # PPS uses -9999.9 and friends
    return arr


def load_pass(path: str, sensor: str, centre: Tuple[float, float],
              box_deg: float = 6.0, granule: Optional[Granule] = None) -> MWPass:
    """Read 37 and 89 GHz near the storm from a downloaded 1C granule."""
    if h5py is None:
        raise RuntimeError("h5py is required to read microwave granules (pip install h5py)")
    clat, clon = centre
    g = granule or _parse_granule(sensor, os.path.basename(path))

    with h5py.File(path, "r") as h:
        chans = discover_channels(h)
        mw = MWPass(sensor=sensor, start=g.start if g else None, end=g.end if g else None,
                    filename=os.path.basename(path), centre=centre)

        for band in ("89", "37"):
            gv, iv = chans[f"{band}V"]
            gh, ih = chans[f"{band}H"]
            grp = h[gv]
            lat = grp["Latitude"][...].astype(np.float64)
            lon = grp["Longitude"][...].astype(np.float64)
            v = _read_tc(grp, iv)
            hh = _read_tc(h[gh], ih)
            if hh.shape != v.shape:
                raise RuntimeError(f"{band} GHz V and H are on different grids "
                                   f"({gv} {v.shape} vs {gh} {hh.shape}); not supported yet.")

            dlon = (lon - clon + 180.0) % 360.0 - 180.0
            near = ((np.abs(lat - clat) <= box_deg) & (np.abs(dlon) <= box_deg)
                    & np.isfinite(v) & np.isfinite(hh) & (np.abs(lat) <= 90))
            scans, pixels = np.nonzero(near)
            mw.bands[band] = BandSwath(lat=lat[near], lon=clon + dlon[near], v=v[near],
                                       h=hh[near], scan=scans, pixel=pixels, group=gv)

    # Does the swath actually contain the centre, or only clip the edge?
    b89 = mw.bands.get("89")
    if b89 is None or b89.lat.size == 0:
        mw.covers_centre = False
    else:
        d = _haversine_km(b89.lat, b89.lon, clat, clon)
        mw.covers_centre = bool(np.nanmin(d) < 25.0)
    return mw


# ----------------------------------------------------- brightness products

# NRL's definitions, from GeoIPS. The colour composites and the standalone PCT
# products use DIFFERENT coefficients on purpose: the composites keep the
# classic Spencer et al. (1989) style weights, the standalone products use
# Cecil & Chronis (2018).
def pct89(v, h):
    return 1.7 * v - 0.7 * h


def pct37(v, h):
    return 2.15 * v - 1.15 * h


def pct(band: str, v, h):
    return pct89(v, h) if band == "89" else pct37(v, h)
