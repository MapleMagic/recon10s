#!/usr/bin/env python3
"""
recon10s_nhc — where is the storm right now.

The position decides which satellite to use and where to centre the imagery.
Two sources, in order:

  1. CurrentStorms.json, which carries every active NHC system with a position
     and the synoptic time it belongs to. One request, no parsing of fixed
     width text.
  2. The real-time ATCF best track (b-deck) for a storm id, used when
     CurrentStorms is unreachable or the storm has been dropped from it.

NHC's published best track is a post-season product; the b-decks under
/atcf/btk/ are the working file updated each cycle, which is the closest
thing to a live best track fix. `best_track` reads the whole working best
track (intensity, pressure, wind radii and RMW at each synoptic time).

JTWC's basins -- West Pacific, North Indian Ocean, Southern Hemisphere --
are not NHC's. `all_storms` adds them from knackwx's open ATCF feed
(api.knackwx.com/atcf/v2: every active system in every basin, invests too,
with motion), falling back to NRL's GeoIPS sector file. Their best tracks
come from knackwx (the active storm's working b-deck, which needs JTWC's
short id such as 29W or 01P), else UCAR's open mirror of JTWC's b-decks,
else natyphoon.top's copy.

Storm motion: CurrentStorms.json gives the advisory motion as movementDir
(degrees the storm is heading toward) and movementSpeed -- in mph, not knots
(checked against the forecast advisory: 10 mph there is "9 KT" in the TCM).
"""
from __future__ import annotations

import datetime as dt
import math
import re
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

CURRENT_STORMS_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"
BDECK_URL = "https://ftp.nhc.noaa.gov/atcf/btk/b{storm_id}.dat"
JTWC_BDECK_URL = "https://hurricanes.ral.ucar.edu/repository/data/bdecks_open/{year}/b{storm_id}.dat"
KNACK_API = "https://api.knackwx.com/atcf/v2"
KNACK_TRACK_URL = KNACK_API + "/track/history/text?stormID={short}"
NATYPHOON_BDECK_URL = "https://www.natyphoon.top/atcf/temp/b{storm_id}.dat"
NHC_BASINS = ("al", "ep", "cp")
BASIN_ORDER = ("al", "ep", "cp", "wp", "io", "sh")
STORM_LIST_NOTES: List[str] = []          # what went wrong fetching the last storm list, if anything
MPH_TO_KT = 0.868976
KM_PER_NM = 1.852
QUADRANTS = ("NE", "SE", "SW", "NW")

_CLASSIFICATIONS = {
    "TD": "Tropical Depression", "TS": "Tropical Storm", "HU": "Hurricane",
    "PTC": "Potential Tropical Cyclone", "STD": "Subtropical Depression",
    "STS": "Subtropical Storm", "PC": "Post-tropical Cyclone",
    "EX": "Post-tropical Cyclone", "LO": "Low",
    # JTWC's (b-deck) natures
    "TY": "Typhoon", "ST": "Super Typhoon", "TC": "Tropical Cyclone", "SD": "Subtropical Depression",
    "SS": "Subtropical Storm", "DB": "Disturbance", "MD": "Monsoon Depression", "WV": "Tropical Wave",
    "IN": "Inland", "XX": "",
}


@dataclass
class Fix:
    """One storm position."""
    storm_id: str
    name: str
    lat: float
    lon: float
    time: Optional[dt.datetime] = None
    classification: str = ""
    vmax_kt: Optional[int] = None
    mslp_mb: Optional[int] = None
    source: str = ""
    motion_dir: Optional[float] = None     # degrees the storm is heading toward
    motion_kt: Optional[float] = None

    @property
    def motion(self) -> Optional[Tuple[float, float]]:
        """(heading toward, kt) from the advisory, or None."""
        if self.motion_dir is None or self.motion_kt is None:
            return None
        return float(self.motion_dir) % 360.0, float(self.motion_kt)

    @property
    def label(self) -> str:
        kind = _CLASSIFICATIONS.get(self.classification, self.classification)
        name = self.name
        if name.lower().startswith("invest"):
            kind = ""                                   # "Invest 93S" says it all
        elif self.storm_id[:2] not in NHC_BASINS and self.storm_id:
            name = f"{name} ({short_id(self.storm_id) or self.storm_id.upper()})"
        bits = [f"{kind} {name}".strip(), f"{abs(self.lat):.1f}{'N' if self.lat >= 0 else 'S'}",
                f"{abs(self.lon):.1f}{'W' if self.lon < 0 else 'E'}"]
        if self.vmax_kt:
            bits.append(f"{self.vmax_kt} kt")
        if self.mslp_mb:
            bits.append(f"{self.mslp_mb} mb")
        if self.time:
            bits.append(self.time.strftime("%d/%H%MZ"))
        return "  ".join(bits)


def _as_int(value) -> Optional[int]:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return n if n else None


def active_storms(timeout: float = 20.0) -> List[Fix]:
    """Every active NHC system, newest fix each. Empty list if none are up."""
    r = requests.get(CURRENT_STORMS_URL, timeout=timeout)
    r.raise_for_status()
    payload = r.json()

    fixes = []
    for storm in payload.get("activeStorms") or []:
        lat = storm.get("latitudeNumeric")
        lon = storm.get("longitudeNumeric")
        if lat is None or lon is None:
            continue
        when = None
        stamp = storm.get("lastUpdate")
        if stamp:
            try:
                when = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except ValueError:
                when = None
        mdir, mkt = storm.get("movementDir"), storm.get("movementSpeed")
        try:
            mdir = float(mdir) if mdir is not None else None
            mkt = float(mkt) * MPH_TO_KT if mkt is not None else None
        except (TypeError, ValueError):
            mdir = mkt = None
        fixes.append(Fix(storm_id=str(storm.get("id", "")).lower(),
                         name=str(storm.get("name", "")).title(),
                         lat=float(lat), lon=float(lon), time=when,
                         classification=str(storm.get("classification", "")),
                         vmax_kt=_as_int(storm.get("intensity")),
                         mslp_mb=_as_int(storm.get("pressure")),
                         source="NHC CurrentStorms", motion_dir=mdir, motion_kt=mkt))
    return fixes


_BDECK_LAT = re.compile(r"^(\d+)([NS])$")
_BDECK_LON = re.compile(r"^(\d+)([EW])$")


def latest_bdeck_fix(storm_id: str, timeout: float = 20.0) -> Optional[Fix]:
    """
    Newest line of the working best track for e.g. 'al062026'.

    ATCF columns: basin, number, YYYYMMDDHH, tech num, tech, tau, lat, lon,
    vmax, mslp, type. Latitude and longitude are tenths of a degree with a
    hemisphere letter stuck on the end.
    """
    if storm_id.lower()[:2] not in NHC_BASINS:
        pts = best_track(storm_id, timeout=timeout)
        if not pts:
            return None
        p = pts[-1]
        return Fix(storm_id=storm_id.lower(), name=_display_name(p.name, storm_id.lower()), lat=p.lat, lon=p.lon,
                   time=p.time, classification=_jtwc_nature(storm_id.lower(), p.kind), vmax_kt=p.vmax,
                   mslp_mb=p.mslp, source="ATCF b-deck")
    r = requests.get(BDECK_URL.format(storm_id=storm_id.lower()), timeout=timeout)
    if r.status_code != 200 or not r.text.strip():
        return None

    best = None
    for line in r.text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 11:
            continue
        lat_m = _BDECK_LAT.match(parts[6])
        lon_m = _BDECK_LON.match(parts[7])
        if not (lat_m and lon_m):
            continue
        lat = int(lat_m.group(1)) / 10.0 * (1 if lat_m.group(2) == "N" else -1)
        lon = int(lon_m.group(1)) / 10.0 * (1 if lon_m.group(2) == "E" else -1)
        try:
            when = dt.datetime.strptime(parts[2], "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        if best is None or when >= best.time:
            best = Fix(storm_id=storm_id.lower(), name=(parts[27].title() if len(parts) > 27 else ""),
                       lat=lat, lon=lon, time=when, classification=parts[10],
                       vmax_kt=_as_int(parts[8]), mslp_mb=_as_int(parts[9]),
                       source="ATCF b-deck")
    return best


def latest_fix(storm_id: Optional[str] = None, timeout: float = 20.0) -> Optional[Fix]:
    """
    Best available current position. With a storm id, that storm; without one,
    whichever active system NHC updated most recently.
    """
    try:
        fixes = active_storms(timeout=timeout)
    except Exception:
        fixes = []

    if storm_id:
        storm_id = storm_id.lower()
        for fix in fixes:
            if fix.storm_id == storm_id:
                return fix
        try:
            return latest_bdeck_fix(storm_id, timeout=timeout)
        except Exception:
            return None

    if not fixes:
        return None
    return max(fixes, key=lambda f: f.time or dt.datetime.min.replace(tzinfo=dt.timezone.utc))


# ------------------------------------------------------------- every basin

_SHORT_IDS: Dict[str, str] = {}        # "wp292026" -> "29W", learnt from the storm lists


def register_short_id(storm_id: str, short: str) -> None:
    if storm_id and short and re.fullmatch(r"\d{2}[A-Z]", short.upper()):
        _SHORT_IDS[storm_id.lower()] = short.upper()


def short_id(storm_id: str) -> Optional[str]:
    """JTWC/NHC short id ('29W', '01P', '09L') of an ATCF id: known from a storm list, else worked out
    where the basin settles the letter (not the Southern Hemisphere's S/P or the Indian Ocean's A/B)."""
    sid = (storm_id or "").lower()
    if sid in _SHORT_IDS:
        return _SHORT_IDS[sid]
    m = re.fullmatch(r"([a-z]{2})(\d{2})\d{4}", sid)
    if not m:
        return None
    letter = {"al": "L", "ep": "E", "cp": "C", "wp": "W"}.get(m.group(1))
    return m.group(2) + letter if letter else None


def known_short_id(storm_id: str) -> Optional[str]:
    """The short id only if a storm list said so (for the Southern Hemisphere and Indian Ocean)."""
    return _SHORT_IDS.get((storm_id or "").lower())


def agency(storm_id: str) -> str:
    """Whose best track and advisories: NHC for its basins (or no id), JTWC elsewhere."""
    sid = (storm_id or "").lower()
    return "NHC" if not sid or sid[:2] in NHC_BASINS else "JTWC"


def is_invest(storm_id: str) -> bool:
    m = re.fullmatch(r"[a-z]{2}(\d{2})\d{4}", (storm_id or "").lower())
    return bool(m) and int(m.group(1)) >= 90


def _norm_lon(lon: float) -> float:
    lon = float(lon)
    while lon >= 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return lon


def _jtwc_nature(storm_id: str, nature: str) -> str:
    nature = (nature or "").upper()
    if storm_id[:2] in ("sh", "io") and nature in ("TY", "ST"):
        return "TC"                                   # a typhoon by any other name
    return nature


def _display_name(name: str, storm_id: str) -> str:
    name = (name or "").strip()
    if not name or name.upper() in ("NONAME", "UNNAMED"):
        return short_id(storm_id) or storm_id.upper()
    if name.upper() == "INVEST":
        return f"Invest {short_id(storm_id) or storm_id.upper()}"
    return name.title()


def parse_knack(payload) -> List[Fix]:
    """knackwx's ATCF v2 storm list (JSON) as Fixes, every basin."""
    out = []
    for st in payload or []:
        try:
            sid = str(st.get("long_atcf_id") or "").lower()
            if not re.fullmatch(r"[a-z]{2}\d{6}", sid):
                continue
            register_short_id(sid, str(st.get("atcf_id") or ""))
            lat, lon = st.get("latitude"), st.get("longitude")
            if lat is None or lon is None:
                continue
            when = None
            if st.get("analysis_time"):
                when = dt.datetime.fromisoformat(str(st["analysis_time"]).replace("Z", "+00:00"))
            mdir, mkt = st.get("movedir"), st.get("movespeed")
            out.append(Fix(storm_id=sid, name=_display_name(st.get("storm_name"), sid), lat=float(lat),
                           lon=_norm_lon(lon), time=when,
                           classification=_jtwc_nature(sid, st.get("cyclone_nature")),
                           vmax_kt=_as_int(st.get("winds")), mslp_mb=_as_int(st.get("pressure")),
                           source="ATCF via knackwx",
                           motion_dir=float(mdir) if mdir is not None else None,
                           motion_kt=float(mkt) if mkt is not None else None))
        except (TypeError, ValueError, AttributeError):
            continue
    return out


_KNACK_CACHE: List = [0.0, None]           # (fetched at, payload): several tabs ask for the list at once


def knack_storms(timeout: float = 20.0) -> List[Fix]:
    """Every active system from knackwx's ATCF feed (kept for a minute)."""
    if _KNACK_CACHE[1] is None or _time.time() - _KNACK_CACHE[0] > 60.0:
        r = requests.get(KNACK_API, timeout=timeout, headers={"User-Agent": "recon10s"})
        r.raise_for_status()
        _KNACK_CACHE[:] = [_time.time(), r.json()]
    return parse_knack(_KNACK_CACHE[1])


def sector_storms(timeout: float = 30.0) -> List[Fix]:
    """NRL's GeoIPS sector file (every basin; positions only, no motion) as Fixes."""
    import recon10s_jma as jma
    out = []
    for s in jma.sector_storms(timeout=timeout):
        sid = s.storm_id.lower()
        if not re.fullmatch(r"[a-z]{2}\d{6}", sid):
            continue
        if re.fullmatch(r"[A-Z]", s.basin or ""):
            register_short_id(sid, sid[2:4] + s.basin)
        out.append(Fix(storm_id=sid, name=_display_name(s.name, sid), lat=s.lat,
                       lon=_norm_lon(s.lon), time=s.time,
                       classification=_jtwc_nature(sid, s.kind), vmax_kt=s.wind_kt, mslp_mb=s.pressure,
                       source="NRL sector file"))
    return out


def all_storms(timeout: float = 20.0) -> List[Fix]:
    """
    Every active storm worldwide: NHC's (CurrentStorms.json, the authority for its basins), then
    JTWC's basins from knackwx -- or NRL's sector file if knackwx is down -- then invests anywhere.
    If NHC is unreachable its storms come from knackwx too. What failed goes in STORM_LIST_NOTES.
    """
    notes = []
    try:
        nhc_fixes: Optional[List[Fix]] = active_storms(timeout=timeout)
    except Exception as exc:
        nhc_fixes = None
        notes.append(f"NHC unreachable ({type(exc).__name__})")
    others: List[Fix] = []
    try:
        others = knack_storms(timeout=min(timeout, 10.0))
    except Exception as exc:
        notes.append(f"knackwx unreachable ({type(exc).__name__})")
        try:
            others = sector_storms(timeout=min(timeout, 10.0))
        except Exception as exc2:
            notes.append(f"NRL sector file unreachable ({type(exc2).__name__})")
    out = list(nhc_fixes or [])
    seen = {f.storm_id for f in out}
    storms, invests = [], []
    for f in others:
        if f.storm_id in seen:
            continue
        if f.storm_id[:2] in NHC_BASINS and nhc_fixes is not None and not is_invest(f.storm_id):
            continue                      # NHC's own list decides which of its storms are active
        seen.add(f.storm_id)
        (invests if is_invest(f.storm_id) else storms).append(f)
    order = lambda f: (BASIN_ORDER.index(f.storm_id[:2]) if f.storm_id[:2] in BASIN_ORDER else 9,
                       -(f.vmax_kt or 0))
    STORM_LIST_NOTES[:] = notes
    return out + sorted(storms, key=order) + sorted(invests, key=order)


def fix_from_positions(lats, lons, times=None) -> Optional[Fix]:
    """
    Fallback when NHC has nothing: use the aircraft itself. The plane is in
    the storm, so the midpoint of a pass is a serviceable centre for deciding
    which satellite and sector to pull.
    """
    lats = [v for v in lats if v is not None]
    lons = [v for v in lons if v is not None]
    if not lats or not lons:
        return None
    when = None
    if times:
        stamps = [t for t in times if t is not None]
        when = max(stamps) if stamps else None
    return Fix(storm_id="", name="aircraft track", lat=sum(lats) / len(lats),
               lon=sum(lons) / len(lons), time=when, source="HDOB positions")


# ------------------------------------------------------------- best track

@dataclass
class TrackPoint:
    """One synoptic time of a working best track."""
    time: dt.datetime
    lat: float
    lon: float
    vmax: Optional[int] = None              # kt
    mslp: Optional[int] = None              # mb
    kind: str = ""                          # HU, TS, TD, EX, ...
    radii: Dict[int, Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]] = \
        field(default_factory=dict)        # 34/50/64 -> (NE, SE, SW, NW) nm
    rmw_nm: Optional[int] = None
    name: str = ""

    def radius(self, threshold: int, quadrant: str) -> Optional[int]:
        q = self.radii.get(threshold)
        return q[QUADRANTS.index(quadrant)] if q else None


def _latlon(lat_tok: str, lon_tok: str) -> Optional[Tuple[float, float]]:
    lat_m = _BDECK_LAT.match(lat_tok)
    lon_m = _BDECK_LON.match(lon_tok)
    if not (lat_m and lon_m):
        return None
    lat = int(lat_m.group(1)) / 10.0 * (1 if lat_m.group(2) == "N" else -1)
    lon = int(lon_m.group(1)) / 10.0 * (1 if lon_m.group(2) == "E" else -1)
    return lat, lon


def parse_bdeck(text: str, tech: str = "BEST") -> List[TrackPoint]:
    """
    Every synoptic time of an ATCF b-deck, oldest first. A time with wind
    radii has one line per threshold (34, 50, 64 kt), each carrying the four
    quadrant radii in nm (NEQ: NE, SE, SW, NW); they are merged into one
    point. A radius of 0 means the winds do not reach that threshold there.
    """
    points: Dict[dt.datetime, TrackPoint] = {}
    for line in text.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 11 or (tech and p[4] != tech):
            continue
        ll = _latlon(p[6], p[7])
        if ll is None:
            continue
        try:
            when = dt.datetime.strptime(p[2], "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        if len(p) > 3 and p[3].isdigit() and len(p[3]) <= 2:          # minutes for off-synoptic lines
            when += dt.timedelta(minutes=int(p[3]))
        pt = points.get(when)
        if pt is None:
            pt = points[when] = TrackPoint(time=when, lat=ll[0], lon=ll[1], vmax=_as_int(p[8]),
                                           mslp=_as_int(p[9]), kind=p[10])
        if len(p) > 16 and p[11].isdigit() and int(p[11]) in (34, 50, 64):
            code = p[12].upper()
            vals = [int(x) if x.isdigit() else None for x in p[13:17]]
            if code == "AAA":
                vals = [vals[0]] * 4
            if code in ("NEQ", "AAA"):
                pt.radii[int(p[11])] = tuple(vals)
        if len(p) > 19 and p[19].isdigit() and int(p[19]) > 0:
            pt.rmw_nm = int(p[19])
        if len(p) > 27 and p[27] and not pt.name:
            pt.name = p[27].title()
    return [points[k] for k in sorted(points)]


_TEXT_CACHE: Dict[str, Tuple[float, str]] = {}


def fetch_text(url: str, timeout: float = 30.0, max_age_s: float = 600.0, headers=None) -> Optional[str]:
    """GET with a short-lived in-memory cache; None for a 404 or an empty file."""
    hit = _TEXT_CACHE.get(url)
    if hit and _time.time() - hit[0] < max_age_s:
        return hit[1]
    r = requests.get(url, timeout=timeout, headers=headers or {"User-Agent": "recon10s"})
    if r.status_code == 404:
        return None
    r.raise_for_status()
    text = r.text if r.text.strip() else None
    if text is not None:
        _TEXT_CACHE[url] = (_time.time(), text)
    return text


def bdeck_url(storm_id: str) -> str:
    sid = storm_id.lower()
    if sid[:2] in NHC_BASINS:
        return BDECK_URL.format(storm_id=sid)
    return JTWC_BDECK_URL.format(year=sid[-4:], storm_id=sid)


def track_urls(storm_id: str) -> List[str]:
    """Where to look for a storm's working best track, in order."""
    sid = storm_id.lower()
    if sid[:2] in NHC_BASINS:
        return [BDECK_URL.format(storm_id=sid)]
    ucar = JTWC_BDECK_URL.format(year=sid[-4:], storm_id=sid)
    nat = NATYPHOON_BDECK_URL.format(storm_id=sid)
    short = short_id(sid)
    knack = [KNACK_TRACK_URL.format(short=short)] if short else []
    if known_short_id(sid):                   # active now: knackwx has the newest cycle
        return knack + [ucar, nat]
    return [ucar] + knack + [nat]


def season_window(storm_id: str) -> Optional[Tuple[dt.datetime, dt.datetime]]:
    """The dates a storm id can belong to: its year, or for the Southern Hemisphere the season (Jul-Jun)
    ending in that year."""
    m = re.fullmatch(r"([a-z]{2})\d{2}(\d{4})", (storm_id or "").lower())
    if not m:
        return None
    y = int(m.group(2))
    if m.group(1) == "sh":
        return (dt.datetime(y - 1, 7, 1, tzinfo=dt.timezone.utc), dt.datetime(y, 7, 1, tzinfo=dt.timezone.utc))
    return dt.datetime(y, 1, 1, tzinfo=dt.timezone.utc), dt.datetime(y + 1, 1, 1, tzinfo=dt.timezone.utc)


def bdeck_matches(text: str, storm_id: str) -> bool:
    """Is a b-deck this storm's: basin, number and dates all agree (knackwx's short ids carry no year)?"""
    m = re.fullmatch(r"([a-z]{2})(\d{2})\d{4}", (storm_id or "").lower())
    win = season_window(storm_id)
    if not m or not win:
        return False
    for line in (text or "").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 8 or len(p[2]) < 10 or not p[2][:10].isdigit():
            continue
        try:
            when = dt.datetime.strptime(p[2][:10], "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        return p[0].lower() == m.group(1) and p[1].zfill(2) == m.group(2) and win[0] <= when < win[1]
    return False


def best_track(storm_id: str, timeout: float = 30.0) -> List[TrackPoint]:
    """
    The working best track for any storm id: NHC's b-deck for its basins; for JTWC's, the first of
    knackwx, UCAR's mirror and natyphoon.top that has it (checked to be the same storm).
    """
    sid = storm_id.lower()
    if sid[:2] in NHC_BASINS:
        text = fetch_text(bdeck_url(sid), timeout=timeout)
        return parse_bdeck(text) if text else []
    for url in track_urls(sid):
        try:
            text = fetch_text(url, timeout=timeout)
        except Exception:
            continue
        if text and bdeck_matches(text, sid):
            pts = parse_bdeck(text)
            if pts:
                return pts
    return []


def track_motion(points: List[TrackPoint], hours: float = 12.0,
                 until: Optional[dt.datetime] = None) -> Optional[Tuple[float, float]]:
    """(heading toward, kt) over the last `hours` of a track (ending at `until`)."""
    pts = [p for p in points if until is None or p.time <= until]
    if len(pts) < 2:
        return None
    end = pts[-1]
    start = next((p for p in pts if (end.time - p.time).total_seconds() <= hours * 3600), None)
    if start is None or start is end:
        start = pts[-2]
    h = (end.time - start.time).total_seconds() / 3600.0
    if h <= 0:
        return None
    lat0 = math.radians((start.lat + end.lat) / 2)
    dn = (end.lat - start.lat) * 111.2
    de = (((end.lon - start.lon) + 180) % 360 - 180) * 111.2 * math.cos(lat0)
    kt = math.hypot(dn, de) / KM_PER_NM / h
    return (math.degrees(math.atan2(de, dn)) + 360.0) % 360.0, kt


def track_at(points: List[TrackPoint], when: dt.datetime, max_gap_h: float = 6.0) -> Optional[TrackPoint]:
    """The track point nearest `when`, within max_gap_h hours."""
    if not points:
        return None
    best = min(points, key=lambda p: abs((p.time - when).total_seconds()))
    return best if abs((best.time - when).total_seconds()) <= max_gap_h * 3600 else None
