#!/usr/bin/env python3
"""
recon10s_case — save a whole case to one file, open it later, and replay
the mission as it unfolded.

A case file (.r10case) is a zip archive:

    case.json            manifest: storm, time span, what is inside, and the
                         settings that go with it (imagery choices), with
                         references to imagery
    hdob/*.txt           every NHC HDOB message, exactly as issued
    vdm/*.txt            vortex data messages, as issued
    drops/*.txt          TEMP DROP messages, as issued
    nhc/best_track.json  NHC's (or JTWC's) working best track, as parsed
    nhc/forecast.json    the official cone and track, and the model tracks
    adt/<code>-list.txt  CIMSS ADT history, as published
    cimss/*.txt          CIMSS D-PRINT and D-MINT histories, as published (2.5.0 on)
    satint.json          recon10s's satellite intensity estimates (2.5.0 on)
    tdr/*.nc.gz          gridded TDR analyses (optional, ~5 MB each)
    tdr/evolution.json   TDR pass-by-pass measurements
    converted.json       an HDOB converted from IWG1 in this session, if any

Messages are kept as raw text and decoded again on opening, so a case saved
now is read by whatever decoders recon10s has then. Imagery is not copied --
GOES, Himawari-9, NEXRAD and JMA stay in their public archives -- the case
records which scans were used, and a replay fetches imagery for its own clock.

Replay: every item has the time it became available -- an HDOB batch or a
message at its file time on NHC's server, a TDR analysis when HRD's job
file was written (else 30 minutes after the analysis ended), best-track
points and satellite estimates at their valid time -- and `Replay.at(t)`
returns what had arrived by t.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import re
import zipfile
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import recon10s_drops as drops_mod
import recon10s_nhc as nhc
import recon10s_nhchdob as nhchdob
import recon10s_vdm as vdm

CASE_VERSION = 1
EXTENSION = ".r10case"
UTC = dt.timezone.utc
_STAMP = re.compile(r"\.(\d{12})\.txt$")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _iso(t: Optional[dt.datetime]) -> Optional[str]:
    return t.isoformat() if t is not None else None


def _time(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _stamp_from(url: str) -> Optional[dt.datetime]:
    m = _STAMP.search(url or "")
    return dt.datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=UTC) if m else None


_HEADER = re.compile(r"^\s*[A-Z]{4}\d{2}\s+[A-Z]{4}\s+(\d{2})(\d{2})(\d{2})\s*$", re.M)


def _issued(raw: str, after: Optional[dt.datetime]) -> Optional[dt.datetime]:
    """
    When a message was issued, from its WMO header ("URNT12 KNHC 081235"),
    for messages kept without NHC's file name: the day/hour/minute in the
    month nearest `after` (the message's own last time), and never before it.
    """
    if not raw or after is None:
        return None
    m = _HEADER.search(raw[:300])
    if not m:
        return None
    day, hh, mm = map(int, m.groups())
    best = None
    for k in (-1, 0, 1):
        y, mo = after.year + (after.month - 1 + k) // 12, (after.month - 1 + k) % 12 + 1
        try:
            t = dt.datetime(y, mo, day, hh, mm, tzinfo=UTC)
        except ValueError:
            continue
        if best is None or abs(t - after) < abs(best - after):
            best = t
    if best is None or abs(best - after) > dt.timedelta(days=2):
        return None
    return max(best, after)


def _member_name(prefix: str, url: str, fallback: str) -> str:
    base = (url or "").rsplit("/", 1)[-1] or fallback
    return f"{prefix}/{_SAFE.sub('_', base)}"


@dataclass
class Case:
    """Everything a saved case holds, decoded."""
    path: str = ""
    manifest: Dict = field(default_factory=dict)
    fix: Optional[nhc.Fix] = None
    batches: List = field(default_factory=list)
    vdms: List = field(default_factory=list)
    drops: List = field(default_factory=list)
    converted: object = None                      # satview.Overlay
    track: List[nhc.TrackPoint] = field(default_factory=list)
    forecast: object = None                       # forecast.OfficialForecast
    models: List = field(default_factory=list)
    adt_text: str = ""
    adt_url: str = ""
    cimss_ml: Dict[str, Tuple[str, str]] = field(default_factory=dict)   # "D-PRINT"/"D-MINT" -> (url, text)
    satint: List[Dict] = field(default_factory=list)   # recon10s_satint.Estimate.to_dict() rows
    tdr_paths: Dict[str, str] = field(default_factory=dict)   # analysis URL -> local .nc
    tdr_available: Dict[str, str] = field(default_factory=dict)  # analysis URL -> ISO time
    evolution: List[Dict] = field(default_factory=list)
    imagery: Dict = field(default_factory=dict)
    notes: str = ""

    @property
    def name(self) -> str:
        m = self.manifest
        return m.get("title") or os.path.basename(self.path)

    def span(self) -> Optional[Tuple[dt.datetime, dt.datetime]]:
        times = [b.start for b in self.batches if b.start] + [b.end for b in self.batches if b.end]
        times += [v.time for v in self.vdms] + [d.release_time for d in self.drops if d.release_time]
        return (min(times), max(times)) if times else None


# --------------------------------------------------------------------- save

def _fix_dict(fix) -> Optional[Dict]:
    if fix is None:
        return None
    return {"storm_id": fix.storm_id, "name": fix.name, "lat": fix.lat, "lon": fix.lon, "time": _iso(fix.time),
            "classification": fix.classification, "vmax_kt": fix.vmax_kt, "mslp_mb": fix.mslp_mb,
            "source": fix.source, "motion_dir": getattr(fix, "motion_dir", None),
            "motion_kt": getattr(fix, "motion_kt", None)}


def _overlay_dict(ov) -> Optional[Dict]:
    if ov is None or not len(ov):
        return None
    out = {}
    for f in ov.FIELDS:
        col = ov.column(f)
        out[f] = [(_iso(v) if f == "times" else v) for v in col]
    return out


def _track_dicts(track) -> List[Dict]:
    return [{"time": _iso(p.time), "lat": p.lat, "lon": p.lon, "vmax": p.vmax, "mslp": p.mslp, "kind": p.kind,
             "radii": {str(k): list(v) for k, v in p.radii.items()}, "rmw_nm": p.rmw_nm, "name": p.name}
            for p in track or []]


def _forecast_dict(fc, models) -> Optional[Dict]:
    if fc is None and not models:
        return None
    out = {"models": [{"model": m.model, "group": m.group, "cycle": _iso(m.cycle),
                       "points": [list(p) for p in m.points]} for m in models or []]}
    if fc is not None:
        out["official"] = {"cone": [[list(xy) for xy in poly] for poly in fc.cone],
                           "track": [list(xy) for xy in fc.track], "points": [list(p) for p in fc.points],
                           "advisory": fc.advisory}
    return out


def save_case(path: str, *, fix=None, batches: Sequence = (), vdms: Sequence = (), drops: Sequence = (),
              converted=None, track: Sequence = (), forecast=None, models: Sequence = (),
              adt_url: str = "", adt_text: str = "", cimss_ml: Optional[Dict[str, Tuple[str, str]]] = None,
              satint: Sequence[Dict] = (),
              tdr_grids: Optional[Dict[str, str]] = None, tdr_available: Optional[Dict[str, str]] = None,
              evolution: Sequence = (), imagery: Optional[Dict] = None, notes: str = "",
              title: str = "", version: str = "", progress=None) -> Dict:
    """
    Write a case file. `tdr_grids` maps an analysis URL to its local
    (uncompressed) netCDF file; each is stored gzipped. Returns the manifest.
    """
    import recon10s_nhchdob as H
    if not path.lower().endswith(EXTENSION):
        path += EXTENSION
    man: Dict = {"recon10s_case": CASE_VERSION, "saved": _iso(dt.datetime.now(UTC)), "recon10s": version,
                 "title": title, "storm": _fix_dict(fix), "notes": notes, "hdob": [], "vdm": [], "drops": [],
                 "tdr": [], "imagery": imagery or {}}
    tmp = path + ".part"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        used = set()

        def put(prefix, url, fallback, text):
            name = _member_name(prefix, url, fallback)
            k = 1
            while name in used:
                name = f"{name.rsplit('.', 1)[0]}_{k}.txt"
                k += 1
            used.add(name)
            z.writestr(name, text)
            return name

        for i, b in enumerate(batches or []):
            text = b.raw or H._CACHE.get(b.url, "")
            if not text:
                continue
            issued = b.message_time or _stamp_from(b.url) or _issued(text, b.end)
            man["hdob"].append({"file": put("hdob", b.url, f"hdob_{i}.txt", text), "url": b.url,
                                "message_time": _iso(issued)})
        for i, v in enumerate(vdms or []):
            text = v.raw or vdm._CACHE.get(v.url, "")
            if not text:
                continue
            # the file time on NHC's server; else the message header; else unknown (a replay then
            # allows ten minutes after the fix)
            stamp = _stamp_from(v.url) or getattr(v, "message_time", None) or _issued(text, v.time)
            man["vdm"].append({"file": put("vdm", v.url, f"vdm_{i}.txt", text), "url": v.url,
                               "message_time": _iso(stamp), "fix_time": _iso(v.time)})
        for i, d in enumerate(drops or []):
            text = d.raw or drops_mod._CACHE.get(d.source, "")
            if not text:
                continue
            issued = d.message_time or _stamp_from(d.source) or _issued(text, d.release_time)
            man["drops"].append({"file": put("drops", d.source, f"drop_{i}.txt", text), "url": d.source,
                                 "message_time": _iso(issued), "release_time": _iso(d.release_time)})
        ov = _overlay_dict(converted)
        if ov:
            z.writestr("converted.json", json.dumps(ov))
            man["converted"] = "converted.json"
        if track:
            z.writestr("nhc/best_track.json", json.dumps(_track_dicts(track)))
            man["best_track"] = "nhc/best_track.json"
        fc = _forecast_dict(forecast, models)
        if fc:
            z.writestr("nhc/forecast.json", json.dumps(fc))
            man["forecast"] = "nhc/forecast.json"
        if adt_text:
            name = _member_name("adt", adt_url, "adt-list.txt")
            z.writestr(name, adt_text)
            man["adt"] = {"file": name, "url": adt_url}
        for product, (url, text) in (cimss_ml or {}).items():
            if text:
                name = _member_name("cimss", url, f"{product.replace('-', '')}hist.txt")
                z.writestr(name, text)
                man.setdefault("cimss_ml", {})[product] = {"file": name, "url": url}
        if satint:
            z.writestr("satint.json", json.dumps(list(satint)))
            man["satint"] = "satint.json"
        for k, (url, local) in enumerate((tdr_grids or {}).items(), 1):
            if progress:
                progress(f"Compressing TDR analysis {k} of {len(tdr_grids)}…")
            try:
                with open(local, "rb") as fh:
                    data = gzip.compress(fh.read(), compresslevel=6)
            except OSError:
                continue
            name = _member_name("tdr", url, f"tdr_{k}.nc.gz")
            if not name.endswith(".gz"):
                name += ".gz"
            z.writestr(name, data)
            man["tdr"].append({"file": name, "url": url, "available": (tdr_available or {}).get(url)})
        if evolution:
            rows = [asdict(m) if hasattr(m, "__dataclass_fields__") else dict(m) for m in evolution]
            z.writestr("tdr/evolution.json", json.dumps(rows))
            man["evolution"] = "tdr/evolution.json"
        span = None
        times = [b.start for b in batches if b.start] + [b.end for b in batches if b.end] + \
                [v.time for v in vdms] + [d.release_time for d in drops if d.release_time]
        if times:
            span = (min(times), max(times))
        man["span"] = [_iso(span[0]), _iso(span[1])] if span else None
        z.writestr("case.json", json.dumps(man, indent=1))
    os.replace(tmp, path)
    man["path"] = path
    return man


# --------------------------------------------------------------------- open

def _fix_from(d: Optional[Dict]):
    if not d:
        return None
    return nhc.Fix(storm_id=d.get("storm_id", ""), name=d.get("name", ""), lat=float(d["lat"]),
                   lon=float(d["lon"]), time=_time(d.get("time")), classification=d.get("classification", ""),
                   vmax_kt=d.get("vmax_kt"), mslp_mb=d.get("mslp_mb"), source=(d.get("source") or "") + " (case)",
                   motion_dir=d.get("motion_dir"), motion_kt=d.get("motion_kt"))


def _overlay_from(d: Dict):
    import recon10s_satview as satview
    ov = satview.Overlay()
    for f in ov.FIELDS:
        col = d.get(f) or []
        setattr(ov, f, [(_time(v) if f == "times" else v) for v in col])
    return ov


def _read_text(z: zipfile.ZipFile, name: str) -> str:
    return z.read(name).decode("utf-8", errors="replace")


def load_case(path: str, cache_dir: Optional[str] = None, progress=None) -> Case:
    """Open a case file; messages are decoded again with this version's decoders."""
    with zipfile.ZipFile(path) as z:
        man = json.loads(_read_text(z, "case.json"))
        if int(man.get("recon10s_case", 0)) > CASE_VERSION:
            raise ValueError("This case was saved by a newer recon10s; update to open it.")
        case = Case(path=path, manifest=man, fix=_fix_from(man.get("storm")), notes=man.get("notes", ""),
                    imagery=man.get("imagery") or {})
        if progress:
            progress(f"Decoding {len(man.get('hdob', []))} HDOB message(s)…")
        kept: Dict[Tuple, object] = {}
        for item in man.get("hdob", []):
            b = nhchdob.parse_message(_read_text(z, item["file"]), item.get("url", ""))
            if b is None:
                continue
            b.message_time = _time(item.get("message_time"))
            key = (b.aircraft, b.mission, b.number)
            if key not in kept or (b.message_time or dt.datetime.min.replace(tzinfo=UTC)) > \
                    (kept[key].message_time or dt.datetime.min.replace(tzinfo=UTC)):
                kept[key] = b
        case.batches = sorted(kept.values(), key=lambda b: (b.start, b.aircraft))
        fixes = {}
        for item in man.get("vdm", []):
            stamp = _time(item.get("message_time"))
            fixed = _time(item.get("fix_time"))
            # the message's day of month is resolved against when it was issued (or fixed), not today
            v = vdm.parse_vdm(_read_text(z, item["file"]), stamp or (fixed + dt.timedelta(minutes=30) if fixed
                                                                     else None), item.get("url", ""))
            if v is not None:
                v.message_time = stamp
                fixes[(v.aircraft, v.time)] = v
        case.vdms = sorted(fixes.values(), key=lambda v: v.time)
        drops = {}
        for item in man.get("drops", []):
            d = drops_mod.parse_tempdrop(_read_text(z, item["file"]), source=item.get("url", ""))
            stamp = _time(item.get("message_time"))
            if d is None or d.release_lat is None:
                continue
            released = _time(item.get("release_time"))
            if stamp is not None or released is not None:
                drops_mod._resolve_times(d, stamp or released + dt.timedelta(minutes=15))
                if stamp is None:
                    d.message_time = None
            key = (d.aircraft, d.release_time, d.release_lat, d.release_lon)
            if key not in drops or (d.message_time or dt.datetime.min.replace(tzinfo=UTC)) > \
                    (drops[key].message_time or dt.datetime.min.replace(tzinfo=UTC)):
                drops[key] = d
        ordered = sorted(drops.values(), key=lambda d: d.release_time or d.message_time)
        for n, d in enumerate(ordered, 1):
            d.number = n
        case.drops = ordered
        if man.get("converted"):
            case.converted = _overlay_from(json.loads(_read_text(z, man["converted"])))
        if man.get("best_track"):
            pts = []
            for p in json.loads(_read_text(z, man["best_track"])):
                pts.append(nhc.TrackPoint(time=_time(p["time"]), lat=p["lat"], lon=p["lon"], vmax=p.get("vmax"),
                                          mslp=p.get("mslp"), kind=p.get("kind", ""),
                                          radii={int(k): tuple(v) for k, v in (p.get("radii") or {}).items()},
                                          rmw_nm=p.get("rmw_nm"), name=p.get("name", "")))
            case.track = pts
        if man.get("forecast"):
            import recon10s_forecast as fcst
            fd = json.loads(_read_text(z, man["forecast"]))
            off = fd.get("official")
            if off:
                case.forecast = fcst.OfficialForecast(cone=[[tuple(xy) for xy in poly] for poly in off["cone"]],
                                                      track=[tuple(xy) for xy in off["track"]],
                                                      points=[tuple(p) for p in off["points"]],
                                                      advisory=off.get("advisory", ""))
            case.models = [fcst.ModelTrack(model=m["model"], group=m["group"], cycle=_time(m["cycle"]),
                                           points=[tuple(p) for p in m["points"]]) for m in fd.get("models", [])]
        if man.get("adt"):
            case.adt_text = _read_text(z, man["adt"]["file"])
            case.adt_url = man["adt"].get("url", "")
        for product, item in (man.get("cimss_ml") or {}).items():
            case.cimss_ml[product] = (item.get("url", ""), _read_text(z, item["file"]))
        if man.get("satint"):
            case.satint = list(json.loads(_read_text(z, man["satint"])))
        # cases from 2.3.0-2.4.0 may hold an imported estimator file: recon10s no longer imports
        # estimates, so it is left in the archive unread
        if man.get("tdr"):
            import recon10s_tdrgrid as tg
            folder = cache_dir or tg.CACHE_DIR
            os.makedirs(folder, exist_ok=True)
            for item in man["tdr"]:
                url = item.get("url", "")
                # same cache name the TDR reader uses, so tdrgrid.load_grid(url) finds it
                local = os.path.join(folder, hashlib.sha1(url.encode()).hexdigest()[:16] + ".nc")
                if not os.path.exists(local):
                    data = gzip.decompress(z.read(item["file"]))
                    with open(local + ".part", "wb") as fh:
                        fh.write(data)
                    os.replace(local + ".part", local)
                case.tdr_paths[url] = local
                if item.get("available"):
                    case.tdr_available[url] = item["available"]
        if man.get("evolution"):
            case.evolution = json.loads(_read_text(z, man["evolution"]))
    return case


# ------------------------------------------------------------------- replay

def tdr_available_time(url: str, end: Optional[dt.datetime], jobfiles: Sequence[str] = ()) -> Optional[dt.datetime]:
    """
    When an analysis reached the server: HRD's job file for it carries the
    processing time ("20261008162748_20261008I1_160258_jobfile.tar.gz" for
    the 1602 analysis); otherwise 30 minutes after the analysis window ended.
    """
    m = re.search(r"/(\d{6}[A-Z]\d)_(\d{4})_xy", url or "")
    if m:
        for name in jobfiles:
            j = re.match(r"(\d{14})_\d{8}[A-Z]\d_(\d{4})\d{2}_jobfile", name)
            if j and j.group(2) == m.group(2):
                return dt.datetime.strptime(j.group(1), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return end + dt.timedelta(minutes=30) if end else None


class Replay:
    """What had arrived by any moment of a case."""

    def __init__(self, case: Case):
        self.case = case

        def avail_batch(b):
            return b.message_time or (b.end + dt.timedelta(minutes=1) if b.end else None)

        def avail_vdm(v):
            return getattr(v, "message_time", None) or _stamp_from(v.url) or v.time + dt.timedelta(minutes=10)

        def avail_drop(d):
            return d.message_time or ((d.release_time + dt.timedelta(minutes=15)) if d.release_time else None)
        self._b = [(avail_batch(b), b) for b in case.batches]
        self._v = [(avail_vdm(v), v) for v in case.vdms]
        self._d = [(avail_drop(d), d) for d in case.drops]
        times = [t for t, _ in self._b + self._v + self._d if t is not None]
        sp = case.span()
        if sp:
            times += list(sp)
        self.start = min(times) if times else None
        self.end = max(times) if times else None

    def at(self, t: dt.datetime):
        """(batches, vdms, drops) available at time t, in the case's order."""
        def upto(items):
            return [x for when, x in items if when is not None and when <= t]
        drops = upto(self._d)
        return upto(self._b), upto(self._v), drops

    def counts(self, t: dt.datetime) -> Tuple[int, int, int]:
        b, v, d = self.at(t)
        return len(b), len(v), len(d)

    def track_at(self, t: dt.datetime):
        return [p for p in self.case.track if p.time <= t]

    def tdr_at(self, t: dt.datetime) -> List[str]:
        """Analysis URLs (with a saved grid) available by t, oldest first."""
        out = []
        for url, when in self.case.tdr_available.items():
            w = _time(when)
            if w is not None and w <= t and url in self.case.tdr_paths:
                out.append((w, url))
        return [u for _w, u in sorted(out)]

    def evolution_at(self, t: dt.datetime) -> List[Dict]:
        """TDR pass measurements whose analysis had reached the server by t."""
        out = []
        for row in self.case.evolution:
            when = _time(row.get("available"))
            if when is None:
                end = _time(row.get("end")) or _time(row.get("time"))
                when = end + dt.timedelta(minutes=30) if end is not None else None
            if when is not None and when <= t:
                out.append(row)
        return out
