#!/usr/bin/env python3
"""
recon10s_forecast — where NHC and the models say the storm is going.

  * Official track and cone: the KMZ files CurrentStorms.json links for each
    storm (forecastTrack, trackCone). A KMZ is a zipped KML; the cone is a
    polygon, the track a line plus a point per forecast hour.
  * Model tracks: the ATCF a-deck for the storm
    (ftp.nhc.noaa.gov/atcf/aid_public/a<id>.dat.gz), latest cycle only.
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

CURRENT_STORMS = "https://www.nhc.noaa.gov/CurrentStorms.json"
ADECK = "https://ftp.nhc.noaa.gov/atcf/aid_public/a{sid}.dat.gz"
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _iter(root, name):
    """Elements by local name: NHC's KMZs use the old earth.google.com/kml/2.1
    namespace, others 2.2, so matching on the namespace would miss one of them."""
    return (e for e in root.iter() if _local(e.tag) == name)


def _child(e, name):
    return next((c for c in e.iter() if _local(c.tag) == name), None)

# Models worth showing by default, and their families for colouring.
MODEL_GROUPS = {
    "Official": ["OFCL"],
    "Consensus": ["TVCN", "TVCA", "TVCE", "HCCA", "FSSE", "IVCN"],
    "Global": ["AVNO", "AVNI", "EMXI", "EMX", "CMC", "CMCI", "UKX", "UKXI", "EGRI", "NVGI", "NGX"],
    "Hurricane": ["HFAI", "HFBI", "HFSA", "HFSB", "HWFI", "HMNI", "CTCI"],
}
GROUP_COLOURS = {"Official": "#ffffff", "Consensus": "#ffd54f", "Global": "#64b5f6",
                 "Hurricane": "#f06292"}


@dataclass
class OfficialForecast:
    cone: List[List[Tuple[float, float]]] = field(default_factory=list)   # polygons of (lon, lat)
    track: List[Tuple[float, float]] = field(default_factory=list)        # (lon, lat) line
    points: List[Tuple[float, float, str]] = field(default_factory=list)  # lon, lat, label
    advisory: str = ""


@dataclass
class ModelTrack:
    model: str
    group: str
    cycle: dt.datetime
    points: List[Tuple[int, float, float, Optional[int]]]   # tau h, lat, lon, vmax kt


def _kml_from_kmz(url: str) -> Optional[ET.Element]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
        return ET.fromstring(z.read(name)) if name else None


def _coords(text: str) -> List[Tuple[float, float]]:
    out = []
    for tok in (text or "").split():
        parts = tok.split(",")
        if len(parts) >= 2:
            try:
                out.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return out


def official_forecast(storm_id: str) -> Optional[OfficialForecast]:
    """NHC's current cone and track for an active storm, or None."""
    data = requests.get(CURRENT_STORMS, timeout=30).json()
    storm = next((s for s in data.get("activeStorms", []) if s.get("id", "").lower() == storm_id.lower()), None)
    if storm is None:
        return None
    fc = OfficialForecast(advisory=str((storm.get("forecastTrack") or {}).get("advNum", "")))
    cone_url = (storm.get("trackCone") or {}).get("kmzFile")
    track_url = (storm.get("forecastTrack") or {}).get("kmzFile")
    if cone_url:
        root = _kml_from_kmz(cone_url)
        if root is not None:
            for poly in _iter(root, "Polygon"):
                outer = _child(poly, "outerBoundaryIs")
                ring = _child(outer if outer is not None else poly, "coordinates")
                if ring is not None:
                    fc.cone.append(_coords(ring.text))
    if track_url:
        root = _kml_from_kmz(track_url)
        if root is not None:
            for line in _iter(root, "LineString"):
                c = _child(line, "coordinates")
                if c is not None and len(_coords(c.text)) > len(fc.track):
                    fc.track = _coords(c.text)
            for pm in _iter(root, "Placemark"):
                point = _child(pm, "Point")
                if point is None:
                    continue
                xy = _coords(_child(point, "coordinates").text)
                name_el = next((c for c in pm if _local(c.tag) == "name"), None)
                name = (name_el.text or "").strip() if name_el is not None else ""
                if xy:
                    fc.points.append((xy[0][0], xy[0][1], name))
    return fc


def _group(model: str) -> Optional[str]:
    for g, names in MODEL_GROUPS.items():
        if model in names:
            return g
    return None


def model_tracks(storm_id: str, groups=("Official", "Consensus", "Global", "Hurricane")) -> List[ModelTrack]:
    """Latest-cycle tracks from the storm's a-deck, for the chosen model groups."""
    r = requests.get(ADECK.format(sid=storm_id.lower()), timeout=45)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    text = gzip.decompress(r.content).decode("utf-8", errors="replace")
    rows: Dict[Tuple[str, str], Dict[int, Tuple[float, float, Optional[int]]]] = {}
    for line in text.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 9:
            continue
        cycle, model = p[2], p[4]
        g = _group(model)
        if g is None or g not in groups:
            continue
        m_lat = re.fullmatch(r"(\d+)([NS])", p[6])
        m_lon = re.fullmatch(r"(\d+)([EW])", p[7])
        if not (m_lat and m_lon and p[5].lstrip("-").isdigit()):
            continue
        lat = int(m_lat.group(1)) / 10 * (1 if m_lat.group(2) == "N" else -1)
        lon = int(m_lon.group(1)) / 10 * (1 if m_lon.group(2) == "E" else -1)
        v = int(p[8]) if p[8].isdigit() and int(p[8]) > 0 else None
        rows.setdefault((cycle, model), {})[int(p[5])] = (lat, lon, v)
    if not rows:
        return []
    latest = max(c for c, _ in rows)
    out = []
    for (cycle, model), pts in rows.items():
        # Interpolated models (the ...I names) arrive six hours after their
        # parents, so take each model's own newest cycle within 12 h of the latest.
        newest = max(c for c, m in rows if m == model)
        if cycle != newest or (dt.datetime.strptime(latest, "%Y%m%d%H") -
                                dt.datetime.strptime(cycle, "%Y%m%d%H")).total_seconds() > 12 * 3600:
            continue
        out.append(ModelTrack(model=model, group=_group(model),
                              cycle=dt.datetime.strptime(cycle, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc),
                              points=sorted((t, la, lo, v) for t, (la, lo, v) in pts.items())))
    return sorted(out, key=lambda m: (list(MODEL_GROUPS).index(m.group), m.model))
