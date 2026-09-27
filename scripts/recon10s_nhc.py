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
thing to a live best track fix.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import List, Optional

import requests

CURRENT_STORMS_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"
BDECK_URL = "https://ftp.nhc.noaa.gov/atcf/btk/b{storm_id}.dat"

_CLASSIFICATIONS = {
    "TD": "Tropical Depression", "TS": "Tropical Storm", "HU": "Hurricane",
    "PTC": "Potential Tropical Cyclone", "STD": "Subtropical Depression",
    "STS": "Subtropical Storm", "PC": "Post-tropical Cyclone",
    "EX": "Post-tropical Cyclone", "LO": "Low",
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

    @property
    def label(self) -> str:
        kind = _CLASSIFICATIONS.get(self.classification, self.classification)
        bits = [f"{kind} {self.name}".strip(), f"{abs(self.lat):.1f}{'N' if self.lat >= 0 else 'S'}",
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
        fixes.append(Fix(storm_id=str(storm.get("id", "")).lower(),
                         name=str(storm.get("name", "")).title(),
                         lat=float(lat), lon=float(lon), time=when,
                         classification=str(storm.get("classification", "")),
                         vmax_kt=_as_int(storm.get("intensity")),
                         mslp_mb=_as_int(storm.get("pressure")),
                         source="NHC CurrentStorms"))
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
