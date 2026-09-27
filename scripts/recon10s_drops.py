#!/usr/bin/env python3
"""
recon10s_drops — dropsondes from NHC's TEMP DROP messages.

Source: https://www.nhc.noaa.gov/archive/recon/<year>/<product>/, where
REPNT3 carries Atlantic drops (UZNT13) and REPPN3 East/Central Pacific drops
(UZPN13), one small text file per message.

Decoding follows WMO FM 37 TEMP DROP, with the details that matter in
practice:

  * Part A (XXAA) gives the surface and mandatory levels. The last digit of
    the first group (Id) says which is the last mandatory level that carries
    a wind group -- levels above it have only two groups, so reading three
    per level silently misaligns everything after 850 mb.
  * Part B (XXBB) gives significant temperature levels, then after 21212 the
    significant wind levels -- the detailed wind profile.
  * Winds are dddff with direction to 5 degrees; speeds of 100 kt and up are
    carried in the direction's units digit (04605 = 045 deg at 105 kt).
  * The 62626 remarks (location tag, MBL/WL150 layer winds, release and
    splash points) are wrapped at a fixed column, often mid-token
    ("WL150 3" / "1001"), so they are re-joined before being read.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

ARCHIVE = "https://www.nhc.noaa.gov/archive/recon/{year}/{product}/"
PRODUCTS = {"atlantic": "REPNT3", "pacific": "REPPN3"}
_FILE_RE = re.compile(r"(REP(?:NT|PN)3-KNHC\.(\d{12})\.txt)")

# Surface wind from layer-mean dropsonde winds (Franklin et al. 2003 style):
# the factors that reproduce NHC's usual quick reductions.
MBL_FACTOR = 0.80      # mean wind, lowest 500 m
WL150_FACTOR = 0.83    # mean wind, lowest 150 m

_STD_LEVELS = {"00": 1000, "92": 925, "85": 850, "70": 700, "50": 500, "40": 400,
               "30": 300, "25": 250, "20": 200, "15": 150, "10": 100}
# Id digit -> lowest-pressure mandatory level that still has a wind group.
_ID_LAST_WIND = {"0": 1000, "9": 925, "8": 850, "7": 700, "5": 500, "4": 400,
                 "3": 300, "2": 200, "1": 100}


@dataclass
class Level:
    p: float                        # hPa
    z: Optional[float] = None       # m
    t: Optional[float] = None       # degC
    td: Optional[float] = None      # degC
    wdir: Optional[int] = None      # deg
    wspd: Optional[int] = None      # kt

    @property
    def rh(self) -> Optional[float]:
        if self.t is None or self.td is None:
            return None
        return relative_humidity(self.t, self.td)


@dataclass
class Drop:
    number: int = 0                 # 1..N by release time within a loaded set
    aircraft: str = ""
    mission: str = ""
    storm: str = ""
    ob: str = ""
    release_lat: Optional[float] = None
    release_lon: Optional[float] = None
    release_time: Optional[dt.datetime] = None
    splash_lat: Optional[float] = None
    splash_lon: Optional[float] = None
    splash_time: Optional[dt.datetime] = None
    location: str = ""              # "NW Eyewall", "Center", ...
    surface: Optional[Level] = None
    mandatory: List[Level] = field(default_factory=list)
    sig_temp: List[Level] = field(default_factory=list)
    sig_wind: List[Level] = field(default_factory=list)
    mbl: Optional[Tuple[int, int]] = None        # (dir, kt), lowest 500 m
    wl150: Optional[Tuple[int, int]] = None      # (dir, kt), lowest 150 m
    dlm: Optional[Tuple[int, int, int, int]] = None  # dir, kt, bottom hPa, top hPa
    message_time: Optional[dt.datetime] = None
    source: str = ""
    raw: str = ""

    @property
    def lat(self) -> Optional[float]:
        return self.release_lat

    @property
    def lon(self) -> Optional[float]:
        return self.release_lon

    @property
    def thermo_levels(self) -> List[Level]:
        """Every level with a temperature, surface first, deduplicated by pressure."""
        seen: Dict[float, Level] = {}
        for lev in ([self.surface] if self.surface else []) + self.mandatory + self.sig_temp:
            if lev is None or lev.t is None:
                continue
            seen.setdefault(round(lev.p, 1), lev)
        return sorted(seen.values(), key=lambda l: -l.p)

    @property
    def wind_levels(self) -> List[Level]:
        """The detailed wind profile: significant levels, else mandatory ones."""
        src = self.sig_wind or [l for l in ([self.surface] if self.surface else []) + self.mandatory]
        levs = [l for l in src if l is not None and l.wspd is not None]
        return sorted(levs, key=lambda l: -l.p)

    @property
    def title_name(self) -> str:
        return f"{self.aircraft} Dropsonde" if self.aircraft else "Dropsonde"


# ------------------------------------------------------------ small helpers

def relative_humidity(t: float, td: float) -> float:
    """RH (%) from temperature and dew point, Bolton (1980)."""
    es = 6.112 * math.exp(17.67 * t / (t + 243.5))
    e = 6.112 * math.exp(17.67 * td / (td + 243.5))
    return max(0.0, min(100.0, 100.0 * e / es))


def compass(az: float, points: int = 8) -> str:
    names8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    names16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
               "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    names = names16 if points == 16 else names8
    return names[int((az % 360) / (360 / len(names)) + 0.5) % len(names)]


def bearing_distance(lat1, lon1, lat2, lon2) -> Tuple[float, float]:
    """Bearing (deg) and great-circle distance (km) from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    brg = (math.degrees(math.atan2(y, x)) + 360) % 360
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return brg, 6371.0 * 2 * math.asin(math.sqrt(min(1.0, a)))


def _wind(group: str) -> Tuple[Optional[int], Optional[int]]:
    """dddff -> (direction, speed kt). Speeds >= 100 ride in the direction's units digit."""
    if not group or "/" in group or len(group) != 5 or not group.isdigit():
        return None, None
    d, ff = int(group[:3]), int(group[3:])
    hundreds = d % 5
    return (d - hundreds) % 360, hundreds * 100 + ff


def _temp_dd(group: str) -> Tuple[Optional[float], Optional[float]]:
    """TTTaDD -> (T, Td). Tenths digit odd means below zero; DD 56-99 are whole degrees + 50."""
    if not group or len(group) != 5:
        return None, None
    tt = group[:3]
    t = None
    if "/" not in tt and tt.isdigit():
        t = int(tt) / 10.0
        if int(tt[2]) % 2 == 1:
            t = -t
    dd = group[3:]
    if t is None or "/" in dd or not dd.isdigit():
        return t, None
    dd_i = int(dd)
    if dd_i <= 50:
        dep = dd_i / 10.0
    elif dd_i >= 56:
        dep = float(dd_i - 50)
    else:
        return t, None
    return t, t - dep


def _std_height(p: int, hhh: str) -> Optional[float]:
    if not hhh.isdigit():
        return None
    h = int(hhh)
    if p == 1000:
        return float(-(h - 500)) if h >= 500 else float(h)
    if p == 925:
        return float(h)
    if p == 850:
        return float(1000 + h)
    if p == 700:
        return float(h + (3000 if h < 500 else 2000))
    if p in (500, 400):
        return float(h * 10)
    val = h * 10                      # 300 hPa and up: decametres
    return float(val + 10000 if val < 5000 else val)


def _surface_pressure(ppp: str) -> Optional[float]:
    if not ppp.isdigit():
        return None
    p = int(ppp)
    return float(p + 1000 if p < 100 else p)


def _latlon(token: str) -> Optional[Tuple[float, float]]:
    """'1706N15590W' -> (17.06, -155.90)."""
    m = re.fullmatch(r"(\d{4})([NS])(\d{5})([EW])", token)
    if not m:
        return None
    lat = int(m.group(1)) / 100.0 * (1 if m.group(2) == "N" else -1)
    lon = int(m.group(3)) / 100.0 * (1 if m.group(4) == "E" else -1)
    return lat, lon


# ---------------------------------------------------------------- decoding

def _sections(text: str) -> Dict[str, List[str]]:
    """Split a message into its coded parts as token lists."""
    body = text.replace("\r", "")
    parts: Dict[str, List[str]] = {}
    for key in ("XXAA", "XXBB"):
        m = re.search(key + r"(.*?)(?=XXAA|XXBB|XXCC|XXDD|;|$)", body, flags=re.S)
        if m:
            parts[key] = m.group(1).split()
    return parts


def _remarks(text: str) -> str:
    """
    The 62626 section, un-wrapped. The remarks are broken at a fixed width
    regardless of token boundaries, so lines are joined with no separator
    and every field is read with regexes that tolerate a missing space.
    """
    m = re.search(r"62626(.*?)=", text.replace("\r", ""), flags=re.S)
    if not m:
        return ""
    return "".join(m.group(1).split("\n"))


def _coded_until_remarks(tokens: List[str]) -> List[str]:
    out = []
    for tok in tokens:
        if tok in ("31313", "51515", "61616", "62626"):
            break
        out.append(tok)
    return out


def parse_tempdrop(text: str, source: str = "") -> Optional[Drop]:
    if "XXAA" not in text and "XXBB" not in text:
        return None
    drop = Drop(source=source, raw=text.strip())
    parts = _sections(text)

    m = re.search(r"U\w{5}\s+KNHC\s+(\d{6})", text)
    m_mission = re.search(r"61616\s+(\S+)\s+(\S+)\s+(.*?)\s+OB\s+(\d+)", text)
    if m_mission:
        drop.aircraft, drop.mission = m_mission.group(1), m_mission.group(2)
        drop.storm = m_mission.group(3).strip()
        drop.ob = m_mission.group(4)

    id_digit = None
    day = hour = None
    a = parts.get("XXAA")
    if a and len(a) >= 4:
        yyggi = a[0]
        if yyggi.isdigit() or yyggi[:4].isdigit():
            day = int(yyggi[:2]) - 50 if int(yyggi[:2]) > 50 else int(yyggi[:2])
            hour = int(yyggi[2:4])
            id_digit = yyggi[4]
        toks = _coded_until_remarks(a[4:])   # skip YYGGId, 99LaLaLa, QcLo, MMMU
        i = 0
        while i < len(toks):
            tok = toks[i]
            if tok.startswith("99") and i == 0:
                p = _surface_pressure(tok[2:])
                t, td = _temp_dd(toks[i + 1]) if i + 1 < len(toks) else (None, None)
                d, s = _wind(toks[i + 2]) if i + 2 < len(toks) else (None, None)
                drop.surface = Level(p=p, z=0.0, t=t, td=td, wdir=d, wspd=s) if p else None
                i += 3
                continue
            ident = tok[:2]
            if ident in _STD_LEVELS:
                p = _STD_LEVELS[ident]
                last_wind = _ID_LAST_WIND.get(id_digit or "", 0)
                has_wind = p >= last_wind if last_wind else False
                t, td = _temp_dd(toks[i + 1]) if i + 1 < len(toks) else (None, None)
                d = s = None
                if has_wind and i + 2 < len(toks):
                    d, s = _wind(toks[i + 2])
                lev = Level(p=float(p), z=_std_height(p, tok[2:]), t=t, td=td, wdir=d, wspd=s)
                # A mandatory level below the surface (1000 mb over a 985 mb
                # surface) carries only an extrapolated height; skip it.
                if not (drop.surface and drop.surface.p and p > drop.surface.p):
                    drop.mandatory.append(lev)
                i += 3 if has_wind else 2
                continue
            if ident in ("88", "77", "66"):
                # Tropopause / max wind: 88999 / 77999 means none. Otherwise
                # skip its groups (tropopause 3, max wind 3 plus optional shear).
                if tok[2:] == "999":
                    i += 1
                else:
                    i += 3
                    if ident in ("77", "66") and i < len(toks) and toks[i].startswith("4"):
                        i += 1
                continue
            i += 1

    b = parts.get("XXBB")
    if b and len(b) >= 4:
        toks = _coded_until_remarks(b[4:])
        section = "temp"
        i = 0
        while i + 1 < len(toks) or (i < len(toks) and toks[i] == "21212"):
            tok = toks[i]
            if tok == "21212":
                section = "wind"
                i += 1
                continue
            if len(tok) == 5 and tok[:2].isdigit() and tok[:2][0] == tok[:2][1]:
                p = _surface_pressure(tok[2:])
                if p is None:
                    i += 2
                    continue
                if section == "temp":
                    t, td = _temp_dd(toks[i + 1])
                    drop.sig_temp.append(Level(p=p, t=t, td=td))
                else:
                    d, s = _wind(toks[i + 1])
                    drop.sig_wind.append(Level(p=p, wdir=d, wspd=s))
                i += 2
                continue
            i += 1

    rem = _remarks(text)
    if rem:
        m_loc = re.search(r"^\s*(EYEWALL\s*(\d{3})|EYE|CENTER|RAINBAND)", rem)
        if m_loc:
            if m_loc.group(2):
                drop.location = f"{compass(int(m_loc.group(2)))} Eyewall"
            else:
                drop.location = m_loc.group(1).title()
        m_mbl = re.search(r"MBL\s*WND\s*(\d{5})", rem)
        if m_mbl:
            d, s = _wind(m_mbl.group(1))
            drop.mbl = (d, s) if s is not None else None
        m_wl = re.search(r"WL\s*150\s*(\d{5})", rem)
        if m_wl:
            d, s = _wind(m_wl.group(1))
            drop.wl150 = (d, s) if s is not None else None
        m_dlm = re.search(r"DLM\s*WND\s*(\d{5})\s*(\d{3})(\d{3})", rem)
        if m_dlm:
            d, s = _wind(m_dlm.group(1))
            if s is not None:
                drop.dlm = (d, s, int(m_dlm.group(2)), int(m_dlm.group(3)))
        m_rel = re.search(r"REL\s*(\d{4}[NS]\d{5}[EW])\s*(\d{6})", rem)
        if m_rel:
            ll = _latlon(m_rel.group(1))
            if ll:
                drop.release_lat, drop.release_lon = ll
            drop._rel_hms = m_rel.group(2)
        m_spg = re.search(r"(?:SPG|SPL)\s*(\d{4}[NS]\d{5}[EW])\s*(\d{4,6})", rem)
        if m_spg:
            ll = _latlon(m_spg.group(1))
            if ll:
                drop.splash_lat, drop.splash_lon = ll
            drop._spg_hms = m_spg.group(2)

    # Fall back to the coded position for the release point.
    if drop.release_lat is None and a and len(a) >= 3:
        try:
            la = int(a[1][2:]) / 10.0
            q = a[2][0]
            lo = int(a[2][1:]) / 10.0
            drop.release_lat = la if q in "17" else -la
            drop.release_lon = -lo if q in "57" else lo
        except (ValueError, IndexError):
            pass

    # Times: message header gives day/hour; REL/SPG and 31313 give clock time.
    header_time = None
    if m:
        header_time = m.group(1)
    drop.message_time = None
    drop._day = day
    drop._launch = None
    m_launch = re.search(r"31313\s+\S+\s+8(\d{4})", text)
    if m_launch:
        drop._launch = m_launch.group(1)
    drop._header = header_time
    return drop


def _resolve_times(drop: Drop, stamp: dt.datetime):
    """
    Turn clock times into datetimes using the message's own file timestamp.
    A release clock time later than the message time belongs to the day
    before (a drop at 23:58 reported at 00:03).
    """
    drop.message_time = stamp

    def at(hms: Optional[str]):
        if not hms:
            return None
        hh, mm = int(hms[:2]), int(hms[2:4])
        ss = int(hms[4:6]) if len(hms) >= 6 else 0
        t = stamp.replace(hour=hh, minute=mm, second=ss, microsecond=0)
        if t > stamp + dt.timedelta(minutes=10):
            t -= dt.timedelta(days=1)
        return t

    drop.release_time = at(getattr(drop, "_rel_hms", None) or getattr(drop, "_launch", None))
    drop.splash_time = at(getattr(drop, "_spg_hms", None))


# ----------------------------------------------------------------- fetching

def basin_product(storm_id: str = "", lon: Optional[float] = None) -> List[str]:
    sid = (storm_id or "").lower()
    if sid.startswith("al"):
        return [PRODUCTS["atlantic"]]
    if sid.startswith(("ep", "cp")):
        return [PRODUCTS["pacific"]]
    if lon is not None and lon < -100:
        return [PRODUCTS["pacific"]]
    return [PRODUCTS["atlantic"], PRODUCTS["pacific"]]


def list_messages(product: str, since: dt.datetime, until: dt.datetime,
                  timeout: float = 30.0) -> List[Tuple[str, dt.datetime]]:
    """(url, message time) for every TEMP DROP file issued in the window."""
    out = []
    years = sorted({since.year, until.year})
    for year in years:
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


def fetch_drops(since: dt.datetime, until: Optional[dt.datetime] = None,
                storm_id: str = "", storm_name: str = "",
                centre: Optional[Tuple[float, float]] = None,
                max_km: float = 500.0, progress=None) -> List[Drop]:
    """
    Every drop for one storm in a time window, numbered 1..N by release time.

    A drop is kept if its mission line names the storm, or -- when it does
    not (unnamed systems, name changes) -- if it was released within
    `max_km` of the storm centre.
    """
    until = until or dt.datetime.now(dt.timezone.utc)
    lon = centre[1] if centre else None
    urls: List[Tuple[str, dt.datetime]] = []
    for product in basin_product(storm_id, lon):
        if progress:
            progress(f"Listing {product} dropsonde messages\u2026")
        urls += list_messages(product, since, until + dt.timedelta(minutes=30))

    if progress:
        progress(f"Reading {len(urls)} dropsonde message(s)\u2026")

    def grab(item):
        url, stamp = item
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return url, stamp, r.text

    drops: List[Drop] = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for url, stamp, text in ex.map(grab, urls):
            drop = parse_tempdrop(text, source=url)
            if drop is None or drop.release_lat is None:
                continue
            _resolve_times(drop, stamp)
            if drop.release_time and not (since <= drop.release_time <= until):
                continue
            name_ok = bool(storm_name) and drop.storm.upper() == storm_name.upper()
            near = False
            if centre:
                _, km = bearing_distance(centre[0], centre[1], drop.release_lat, drop.release_lon)
                near = km <= max_km
            if storm_name and not name_ok and not near:
                continue
            if not storm_name and centre and not near:
                continue
            drops.append(drop)

    # One drop can be re-sent (corrections); keep the latest copy of each.
    unique: Dict[Tuple, Drop] = {}
    for d in drops:
        key = (d.aircraft, d.release_time, d.release_lat, d.release_lon)
        if key not in unique or (d.message_time or 0) > (unique[key].message_time or 0):
            unique[key] = d
    ordered = sorted(unique.values(), key=lambda d: d.release_time or d.message_time)
    for n, d in enumerate(ordered, 1):
        d.number = n
    return ordered


def describe_location(drop: Drop, centre: Optional[Tuple[float, float]]) -> str:
    """The message's own tag if it has one, else bearing and distance from the centre."""
    if drop.location:
        return drop.location
    if centre and drop.release_lat is not None:
        brg, km = bearing_distance(centre[0], centre[1], drop.release_lat, drop.release_lon)
        return f"{compass(brg)}, {km:.0f} km from centre"
    return ""


def surface_estimates(drop: Drop) -> List[Tuple[str, int, float]]:
    """(label, layer mean kt, reduced to surface kt) for MBL and WL150."""
    out = []
    if drop.mbl:
        out.append(("lowest 500m", drop.mbl[1], drop.mbl[1] * MBL_FACTOR))
    if drop.wl150:
        out.append(("lowest 150m", drop.wl150[1], drop.wl150[1] * WL150_FACTOR))
    return out
