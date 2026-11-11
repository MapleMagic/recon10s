#!/usr/bin/env python3
"""
recon10s_tdr — NOAA tail Doppler radar analyses from HRD's real-time page.

https://www.aoml.noaa.gov/ftp/pub/hrd/data/RTradar/<year>/<mission>/<product>/

The real-time output is graphics, one PNG per product, so this is a
catalogue and viewer rather than a gridded-data reader. Mission folders are
YYYYMMDD plus the aircraft letter and flight number (H = NOAA42 Kermit,
I = NOAA43 Miss Piggy, N = NOAA49 Gonzo). Files are named

    260925I1_POLO_2211_dbz_ws_2.0km.png          swath: reflectivity + wind at 2 km
    260925I1_POLO_2154_2304_vt_rzmean.png        azimuthal mean (radius-height)
    260925I1_POLO_2154_2304_ws_dbz_planview.png  plan-view composite
    260925I1_POLO_2154_2304_vort_tilt.png        vortex tilt
    260925I1_POLO_tdrcenters.png                 TDR centre fixes
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

BASE = "https://www.aoml.noaa.gov/ftp/pub/hrd/data/RTradar"
PRODUCT_DIRS = ("swaths", "rzmeans", "composites", "zoom", "tilt", "centers")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "recon10s_cache", "tdr")
AIRCRAFT = {"H": "NOAA42", "I": "NOAA43", "N": "NOAA49"}
RZ_VARS = {"dbz": "Reflectivity", "vt": "Tangential wind", "ur": "Radial wind", "w": "Vertical velocity"}

_HREF = re.compile(r'href="([^"?/]+/?)"')
_MISSION = re.compile(r"^(\d{8})([A-Z])(\d)/$")
_FILE = re.compile(r"^(?P<mid>\d{6}[A-Z]\d)_(?P<storm>[A-Za-z0-9-]+)_(?P<rest>.+)\.png$")


@dataclass
class TdrMission:
    folder: str                 # "20260925I1"
    year: int
    date: dt.date
    aircraft: str
    storm: str = ""
    files: Dict[str, List[str]] = field(default_factory=dict)   # product dir -> filenames

    @property
    def label(self) -> str:
        return f"{self.date:%d %b} {self.aircraft} ({self.folder}) {self.storm}".strip()

    def url(self, product: str, name: str) -> str:
        return f"{BASE}/{self.year}/{self.folder}/{product}/{name}"

    # ---- swaths ---------------------------------------------------------
    def swath_times(self) -> List[str]:
        return sorted({m.group(1) for n in self.files.get("swaths", [])
                       if (m := re.search(r"_(\d{4})_dbz_ws_", n))})

    def swath_levels(self, time: str) -> List[float]:
        out = set()
        for n in self.files.get("swaths", []):
            m = re.search(rf"_{time}_dbz_ws_([\d.]+)km\.png$", n)
            if m:
                out.add(float(m.group(1)))
        return sorted(out)

    def swath(self, time: str, level: float) -> Optional[str]:
        for n in self.files.get("swaths", []):
            m = re.search(rf"_{time}_dbz_ws_([\d.]+)km\.png$", n)
            if m and abs(float(m.group(1)) - level) < 1e-6:
                return n
        return None

    # ---- windowed products ---------------------------------------------
    def windows(self, product: str) -> List[str]:
        """Analysis windows like '2154_2219', newest last."""
        return sorted({m.group(1) for n in self.files.get(product, [])
                       if (m := re.search(r"_(\d{4}_\d{4})_", n))})

    def rzmean(self, window: str, var: str) -> Optional[str]:
        return next((n for n in self.files.get("rzmeans", [])
                     if f"_{window}_{var}_rzmean" in n), None)

    def rz_vars(self, window: str) -> List[str]:
        return [v for v in RZ_VARS if self.rzmean(window, v)]

    def product(self, product: str, window: Optional[str] = None, contains: str = "") -> List[str]:
        return [n for n in self.files.get(product, [])
                if (window is None or f"_{window}_" in n) and contains in n]


def _listing(url: str, timeout: float = 30.0, tries: int = 4) -> List[str]:
    """
    Folder listing. AOML's server answers 503 when pressed with parallel
    requests, so back off and retry; a folder that still fails is treated as
    empty rather than failing the whole mission.
    """
    import time
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
        except requests.RequestException:
            r = None
        if r is not None and r.status_code == 404:
            return []
        if r is not None and r.ok:
            return [h for h in _HREF.findall(r.text) if not h.startswith(("mailto", "/"))]
        time.sleep(1.5 * (attempt + 1))
    return []


def list_missions(year: int, since: Optional[dt.date] = None, progress=None) -> List[TdrMission]:
    """Mission folders for a year (optionally from a date on), without their contents."""
    out = []
    for h in _listing(f"{BASE}/{year}/"):
        m = _MISSION.match(h)
        if not m:
            continue
        d = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
        if since and d < since:
            continue
        out.append(TdrMission(folder=h.rstrip("/"), year=year, date=d,
                              aircraft=AIRCRAFT.get(m.group(2), m.group(2))))
    return sorted(out, key=lambda x: x.folder)


def load_contents(mission: TdrMission) -> TdrMission:
    """Fill in a mission's file lists and its storm name."""
    base = f"{BASE}/{mission.year}/{mission.folder}"

    def one(product):
        return product, [n for n in _listing(f"{base}/{product}/") if n.lower().endswith(".png")]

    with cf.ThreadPoolExecutor(max_workers=2) as ex:     # AOML throttles bursts
        for product, names in ex.map(one, PRODUCT_DIRS):
            mission.files[product] = sorted(names)
    for names in mission.files.values():
        for n in names:
            m = _FILE.match(n)
            if m:
                mission.storm = m.group("storm").upper()
                break
        if mission.storm:
            break
    return mission


def find_missions(storm_name: str = "", days: int = 14, progress=None) -> List[TdrMission]:
    """
    Missions from the last `days` days, contents loaded, optionally only
    those for one storm. Newest first.
    """
    today = dt.datetime.now(dt.timezone.utc).date()
    since = today - dt.timedelta(days=days)
    missions = []
    for year in sorted({since.year, today.year}):
        missions += list_missions(year, since)
    if progress:
        progress(f"Reading {len(missions)} TDR mission folder(s)\u2026")
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        missions = list(ex.map(load_contents, missions))
    if storm_name:
        missions = [m for m in missions if m.storm == storm_name.upper()]
    return sorted([m for m in missions if any(m.files.values())],
                  key=lambda m: m.folder, reverse=True)


def fetch_image(url: str, timeout: float = 45.0) -> str:
    """Download a PNG once; return its cached path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, hashlib.sha1(url.encode()).hexdigest()[:16] + ".png")
    if not os.path.exists(path):
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            fh.write(r.content)
        os.replace(tmp, path)
    return path
