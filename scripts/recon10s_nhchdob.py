#!/usr/bin/env python3
"""
recon10s_nhchdob — HDOBs as NHC publishes them.

NHC's recon archive carries every aircraft's high-density obs in batches of
twenty 30-second lines (ten minutes of flight), one file per message:

    AHONT1  Atlantic                      (URNT15)
    AHOPN1  East and Central Pacific      (URPN15)

This is where USAF Reserve data come from -- the NOAA IWG1 feed only has
NOAA's own aircraft -- and NOAA flights appear here too.

    AF302 0515E NOLO               HDOB 27 20260926
    165630 1540N 15519W 6973 03136 0042 +088 //// 264059 061 /// /// 05

Obs are decoded by position: time, lat, lon, flight-level pressure,
geopotential height, extrapolated surface pressure (or D-value above
550 mb), temperature, dew point, wind, peak 10-s wind, SFMR, rain rate,
quality flags. Each batch's date is the date of its first ob (NHC's rule,
the same one the converter now follows), so a batch that crosses 00Z moves
to the next day part way through.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

ARCHIVE = "https://www.nhc.noaa.gov/archive/recon/{year}/{product}/"
_FILE_RE = re.compile(r"(AHO(?:NT|PN|PA)1-K[A-Z]{3}\.(\d{12})\.txt)")
_MISSION_RE = re.compile(r"^(\S+)\s+(\S+)\s+(.*?)\s+HDOB\s+(\d+)\s+(\d{8})\s*$")
_LAT_RE = re.compile(r"^(\d{2})(\d{2})([NS])$")
_LON_RE = re.compile(r"^(\d{3})(\d{2})([EW])$")

_CACHE: Dict[str, str] = {}   # url -> message text; batches never change once issued


@dataclass
class Ob:
    time: dt.datetime
    lat: float
    lon: float
    wdir: Optional[int] = None
    wspd: Optional[int] = None       # flight-level 30-s mean, kt
    peak: Optional[int] = None       # peak 10-s flight-level wind, kt
    sfmr: Optional[int] = None       # SFMR surface wind, kt
    fl_press: Optional[float] = None
    surface_press: Optional[float] = None   # extrapolated, when below 550 mb flight level


@dataclass
class Batch:
    aircraft: str
    mission: str
    storm: str
    number: int
    date: dt.date
    obs: List[Ob] = field(default_factory=list)
    url: str = ""
    message_time: Optional[dt.datetime] = None

    @property
    def key(self) -> Tuple[str, str]:
        return (self.aircraft, self.mission)

    @property
    def start(self) -> Optional[dt.datetime]:
        return self.obs[0].time if self.obs else None

    @property
    def end(self) -> Optional[dt.datetime]:
        return self.obs[-1].time if self.obs else None

    @property
    def max_wind(self) -> Optional[int]:
        w = [o.wspd for o in self.obs if o.wspd is not None]
        return max(w) if w else None

    @property
    def max_sfmr(self) -> Optional[int]:
        w = [o.sfmr for o in self.obs if o.sfmr is not None]
        return max(w) if w else None


def _num(tok: str) -> Optional[int]:
    return int(tok) if tok and "/" not in tok and tok.lstrip("+-").isdigit() else None


def parse_message(text: str, url: str = "") -> Optional[Batch]:
    lines = [ln.strip() for ln in text.replace("\r", "").splitlines() if ln.strip()]
    batch = None
    last_time = None
    for ln in lines:
        m = _MISSION_RE.match(ln)
        if m:
            date = dt.datetime.strptime(m.group(5), "%Y%m%d").date()
            batch = Batch(aircraft=m.group(1), mission=m.group(2), storm=m.group(3).strip(),
                          number=int(m.group(4)), date=date, url=url)
            continue
        if batch is None:
            continue
        parts = ln.split()
        if len(parts) < 9 or not (len(parts[0]) == 6 and parts[0].isdigit()):
            continue
        lat_m, lon_m = _LAT_RE.match(parts[1]), _LON_RE.match(parts[2])
        if not (lat_m and lon_m):
            continue
        lat = (int(lat_m.group(1)) + int(lat_m.group(2)) / 60.0) * (1 if lat_m.group(3) == "N" else -1)
        lon = (int(lon_m.group(1)) + int(lon_m.group(2)) / 60.0) * (1 if lon_m.group(3) == "E" else -1)

        hh, mm, ss = int(parts[0][:2]), int(parts[0][2:4]), int(parts[0][4:6])
        t = dt.datetime.combine(batch.date, dt.time(hh, mm, ss), tzinfo=dt.timezone.utc)
        if last_time is not None and t < last_time - dt.timedelta(hours=6):
            t += dt.timedelta(days=1)          # crossed 00Z inside the batch
        last_time = t

        fl = _num(parts[3])
        fl_press = (fl / 10.0 + (1000.0 if fl < 1000 else 0.0)) if fl is not None else None
        xxxx = _num(parts[5])
        sfc = None
        if xxxx is not None and fl_press is not None and fl_press >= 550.0:
            sfc = xxxx / 10.0 + (1000.0 if xxxx < 5000 else 0.0)
        wind = parts[8]
        wdir = wspd = None
        if len(wind) == 6 and wind.isdigit():
            wdir, wspd = int(wind[:3]), int(wind[3:])
        batch.obs.append(Ob(time=t, lat=lat, lon=lon, wdir=wdir, wspd=wspd,
                            peak=_num(parts[9]) if len(parts) > 9 else None,
                            sfmr=_num(parts[10]) if len(parts) > 10 else None,
                            fl_press=fl_press, surface_press=sfc))
    return batch if batch and batch.obs else None


def _products(storm_id: str, lon: Optional[float]) -> List[str]:
    sid = (storm_id or "").lower()
    if sid.startswith("al"):
        return ["AHONT1"]
    if sid.startswith(("ep", "cp")):
        return ["AHOPN1"]
    if lon is not None and lon < -100:
        return ["AHOPN1"]
    return ["AHONT1", "AHOPN1"]


def list_messages(product: str, since: dt.datetime, until: dt.datetime,
                  timeout: float = 45.0) -> List[Tuple[str, dt.datetime]]:
    out = []
    for year in sorted({since.year, until.year}):
        base = ARCHIVE.format(year=year, product=product)
        r = requests.get(base, timeout=timeout)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        for name, stamp in set(_FILE_RE.findall(r.text)):
            t = dt.datetime.strptime(stamp, "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc)
            if since <= t <= until:
                out.append((base + name, t))
    out.sort(key=lambda x: x[1])
    return out


def fetch_batches(since: dt.datetime, until: Optional[dt.datetime] = None,
                  storm_id: str = "", storm_name: str = "",
                  centre: Optional[Tuple[float, float]] = None, max_km: float = 900.0,
                  progress=None) -> List[Batch]:
    """
    Every HDOB batch for one storm in a time window, oldest first.

    A batch is kept if its mission line names the storm, or -- for systems
    without a name on the mission line -- if it came within `max_km` of the
    centre. Re-sent batches (same aircraft, mission and number) keep the
    newest copy.
    """
    import math
    until = until or dt.datetime.now(dt.timezone.utc)
    lon = centre[1] if centre else None
    urls = []
    for product in _products(storm_id, lon):
        if progress:
            progress(f"Listing {product} HDOB messages\u2026")
        urls += list_messages(product, since, until + dt.timedelta(minutes=20))
    if progress:
        progress(f"Reading {len(urls)} HDOB message(s)\u2026")

    def grab(item):
        url, stamp = item
        if url not in _CACHE:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            _CACHE[url] = r.text
        return url, stamp, _CACHE[url]

    def near(b: Batch) -> bool:
        if not centre:
            return False
        la0, lo0 = map(math.radians, centre)
        for o in b.obs[:: max(1, len(b.obs) // 4)]:
            la, lo = math.radians(o.lat), math.radians(o.lon)
            a = (math.sin((la - la0) / 2) ** 2
                 + math.cos(la0) * math.cos(la) * math.sin((lo - lo0) / 2) ** 2)
            if 6371.0 * 2 * math.asin(math.sqrt(min(1.0, a))) <= max_km:
                return True
        return False

    kept: Dict[Tuple[str, str, int], Batch] = {}
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for url, stamp, text in ex.map(grab, urls):
            b = parse_message(text, url)
            if b is None or b.end < since or b.start > until:
                continue
            b.message_time = stamp
            named = bool(storm_name) and b.storm.upper() == storm_name.upper()
            if storm_name and not named and not near(b):
                continue
            if not storm_name and centre and not near(b):
                continue
            key = (b.aircraft, b.mission, b.number)
            if key not in kept or stamp > kept[key].message_time:
                kept[key] = b
    return sorted(kept.values(), key=lambda b: (b.start, b.aircraft))


def group_missions(batches: List[Batch]) -> Dict[Tuple[str, str], List[Batch]]:
    """Batches grouped by (aircraft, mission), each group in batch order."""
    out: Dict[Tuple[str, str], List[Batch]] = {}
    for b in batches:
        out.setdefault(b.key, []).append(b)
    for k in out:
        out[k].sort(key=lambda b: (b.start, b.number))
    return out
