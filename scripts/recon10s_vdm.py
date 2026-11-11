#!/usr/bin/env python3
"""
recon10s_vdm — vortex data messages (URNT12 / URPN12), NHC's official
aircraft centre fixes.

Archive: REPNT2 (Atlantic) and REPPN2 (Pacific) under
https://www.nhc.noaa.gov/archive/recon/<year>/. Current format, letters:

  A  fix time          B  centre position     C  level and height
  D  minimum pressure  E  dropsonde surface wind at the centre
  F  eye character     G  eye shape / diameters (e.g. E09/36/30, C20)
  J/K inbound max flight-level wind, and its bearing/range/time from centre
  N/O outbound max flight-level wind, same
  P/Q temperature and altitude outside / inside the eye
  U  mission line;  free-text remarks follow (MAX FL WIND ..., SFC CNTR ...)
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

ARCHIVE = "https://www.nhc.noaa.gov/archive/recon/{year}/{product}/"
_FILE_RE = re.compile(r"(REP(?:NT|PN)2-K[A-Z]{3}\.(\d{12})\.txt)")
_CACHE: Dict[str, str] = {}


@dataclass
class VortexFix:
    time: dt.datetime
    lat: float
    lon: float
    storm_id: str = ""
    aircraft: str = ""
    mission: str = ""
    storm: str = ""
    ob: str = ""
    level: str = ""                      # "700 mb", "925 mb", ...
    height_m: Optional[int] = None
    min_slp: Optional[float] = None      # mb
    sfc_wind: Optional[Tuple[int, int]] = None
    eye: str = ""                        # "CLOSED", "OPEN SW", ...
    eye_shape: str = ""                  # "E09/36/30", "C20"
    max_in: Optional[Tuple[int, int]] = None           # dir, kt
    max_in_at: Optional[Tuple[int, int, str]] = None   # bearing, nm, time
    max_out: Optional[Tuple[int, int]] = None
    max_out_at: Optional[Tuple[int, int, str]] = None
    max_fl_wind: Optional[Tuple[int, int, int, str]] = None   # kt, bearing, nm, time
    temp_out: str = ""
    temp_in: str = ""
    sfc_centre: Optional[Tuple[int, float]] = None     # bearing, nm from the FL centre
    remarks: str = ""
    url: str = ""
    raw: str = ""

    @property
    def label(self) -> str:
        bits = [f"{self.time:%H:%MZ}", self.aircraft]
        if self.min_slp:
            bits.append(f"{self.min_slp:.0f} mb")
        if self.eye_shape:
            bits.append(self.eye_shape)
        return "  ".join(b for b in bits if b)

    @property
    def eye_diameter_nm(self) -> Optional[int]:
        """Largest diameter in the G. group: C20 -> 20, E09/36/30 -> 36."""
        m = re.findall(r"\d+", self.eye_shape.split("/", 1)[-1] if "/" in self.eye_shape
                       else self.eye_shape[1:])
        return max(int(x) for x in m) if m else None

    @property
    def peak_inbound(self) -> Optional[int]:
        return self.max_in[1] if self.max_in else None

    @property
    def peak_outbound(self) -> Optional[int]:
        return self.max_out[1] if self.max_out else None


def _field(text: str, letter: str) -> str:
    m = re.search(rf"^\s*{letter}\.\s*(.*?)\s*$", text, flags=re.M)
    return m.group(1).strip() if m else ""


def _dir_kt(s: str) -> Optional[Tuple[int, int]]:
    m = re.match(r"(\d{1,3})\s*deg\s+(\d{1,3})\s*kt", s, flags=re.I)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _brg_rng(s: str) -> Optional[Tuple[int, int, str]]:
    m = re.match(r"(\d{1,3})\s*deg\s+(\d{1,3})\s*nm(?:\s+(\d{2}:\d{2}:\d{2})Z)?", s, flags=re.I)
    return (int(m.group(1)), int(m.group(2)), m.group(3) or "") if m else None


def parse_vdm(text: str, stamp: Optional[dt.datetime] = None, url: str = "") -> Optional[VortexFix]:
    if "VORTEX DATA MESSAGE" not in text.upper():
        return None
    body = text.replace("\r", "")
    a = _field(body, "A")
    b = _field(body, "B")
    ma = re.match(r"(\d{2})/(\d{2}):(\d{2}):(\d{2})Z", a)
    mb = re.match(r"([\d.]+)\s*deg\s*([NS])\s+([\d.]+)\s*deg\s*([EW])", b, flags=re.I)
    if not (ma and mb):
        return None
    stamp = stamp or dt.datetime.now(dt.timezone.utc)
    day, hh, mm, ss = map(int, ma.groups())
    t = stamp.replace(day=1, hour=hh, minute=mm, second=ss, microsecond=0)
    # the day belongs to the message month, or the month before near a boundary
    try:
        t = t.replace(day=day)
    except ValueError:
        pass
    if t > stamp + dt.timedelta(hours=2):
        prev = (stamp.replace(day=1) - dt.timedelta(days=1))
        t = prev.replace(day=min(day, prev.day), hour=hh, minute=mm, second=ss, microsecond=0)
    lat = float(mb.group(1)) * (1 if mb.group(2).upper() == "N" else -1)
    lon = float(mb.group(3)) * (1 if mb.group(4).upper() == "E" else -1)

    fix = VortexFix(time=t, lat=lat, lon=lon, url=url, raw=body.strip())
    m_id = re.search(r"VORTEX DATA MESSAGE\s+(\w+)", body, flags=re.I)
    fix.storm_id = m_id.group(1).lower() if m_id else ""
    c = _field(body, "C")
    m_c = re.match(r"(\d+\s*mb)\s+(\d+)\s*m", c, flags=re.I)
    if m_c:
        fix.level, fix.height_m = m_c.group(1), int(m_c.group(2))
    d = re.search(r"(\d{3,4})\s*mb", _field(body, "D"), flags=re.I)
    fix.min_slp = float(d.group(1)) if d else None
    fix.sfc_wind = _dir_kt(_field(body, "E"))
    fix.eye = _field(body, "F") if _field(body, "F") != "NA" else ""
    fix.eye_shape = _field(body, "G") if _field(body, "G") != "NA" else ""
    fix.max_in, fix.max_in_at = _dir_kt(_field(body, "J")), _brg_rng(_field(body, "K"))
    fix.max_out, fix.max_out_at = _dir_kt(_field(body, "N")), _brg_rng(_field(body, "O"))
    fix.temp_out, fix.temp_in = _field(body, "P"), _field(body, "Q")
    u = _field(body, "U")
    m_u = re.match(r"(\S+)\s+(\S+)\s+(.*?)\s+OB\s+(\d+)", u)
    if m_u:
        fix.aircraft, fix.mission, fix.storm, fix.ob = (m_u.group(1), m_u.group(2),
                                                        m_u.group(3).strip(), m_u.group(4))
    tail = body.split(u, 1)[-1] if u else ""
    fix.remarks = " ".join(ln.strip() for ln in tail.splitlines()
                           if ln.strip() and ln.strip() not in (";", "$$"))
    m_mx = re.search(r"MAX FL WIND\s+(\d+)\s*KT\s+(\d{1,3})\s*/\s*(\d+)\s*NM\s+(\d{2}:\d{2}:\d{2})Z",
                     fix.remarks)
    if m_mx:
        fix.max_fl_wind = (int(m_mx.group(1)), int(m_mx.group(2)), int(m_mx.group(3)), m_mx.group(4))
    m_s = re.search(r"SFC\s+(?:CNTR|CENTER)\s+(\d{1,3})\s*/\s*([\d.]+)\s*NM", fix.remarks, flags=re.I)
    if m_s:
        fix.sfc_centre = (int(m_s.group(1)), float(m_s.group(2)))
    return fix


def _products(storm_id: str, lon: Optional[float]) -> List[str]:
    sid = (storm_id or "").lower()
    if sid.startswith("al"):
        return ["REPNT2"]
    if sid.startswith(("ep", "cp")):
        return ["REPPN2"]
    return ["REPPN2"] if (lon is not None and lon < -100) else ["REPNT2", "REPPN2"]


def fetch_vdms(since: dt.datetime, until: Optional[dt.datetime] = None, storm_id: str = "",
               storm_name: str = "", centre: Optional[Tuple[float, float]] = None,
               max_km: float = 600.0, progress=None) -> List[VortexFix]:
    """Vortex messages for one storm in a window, oldest first."""
    import math
    until = until or dt.datetime.now(dt.timezone.utc)
    items = []
    for product in _products(storm_id, centre[1] if centre else None):
        if progress:
            progress(f"Listing {product} vortex messages\u2026")
        for year in sorted({since.year, until.year}):
            base = ARCHIVE.format(year=year, product=product)
            r = requests.get(base, timeout=45)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            for name, stamp in set(_FILE_RE.findall(r.text)):
                t = dt.datetime.strptime(stamp, "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc)
                if since <= t <= until + dt.timedelta(minutes=30):
                    items.append((base + name, t))

    def grab(item):
        url, stamp = item
        if url not in _CACHE:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            _CACHE[url] = r.text
        return url, stamp, _CACHE[url]

    out: Dict[Tuple, VortexFix] = {}
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        for url, stamp, text in ex.map(grab, items):
            fix = parse_vdm(text, stamp, url)
            if fix is None:
                continue
            ok = False
            if storm_name and fix.storm.upper() == storm_name.upper():
                ok = True
            if storm_id and fix.storm_id and storm_id.lower()[2:4] == fix.storm_id[2:4] \
                    and storm_id.lower()[-4:] == fix.storm_id[-4:]:
                ok = True
            if not ok and centre:
                la0, lo0, la, lo = map(math.radians, (centre[0], centre[1], fix.lat, fix.lon))
                h = math.sin((la - la0) / 2) ** 2 + math.cos(la0) * math.cos(la) * math.sin((lo - lo0) / 2) ** 2
                ok = 6371 * 2 * math.asin(math.sqrt(h)) <= max_km and not storm_name
            if ok:
                out[(fix.aircraft, fix.time)] = fix
    return sorted(out.values(), key=lambda f: f.time)
