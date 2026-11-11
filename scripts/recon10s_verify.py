#!/usr/bin/env python3
"""
recon10s_verify — satellite intensity estimates against the aircraft.

Estimates
  * CIMSS ADT: the real-time history list for a storm,
        https://tropic.ssec.wisc.edu/real-time/adt/<NN><B>-list.txt
    (09L, 18E, 27W, ...): every 30-minute estimate with CI number, MSLP
    (the CKZ wind-pressure relationship), Vmax, scene type and position.
    Fixes over land (scene LAND, every value 0.0) carry no estimate and are
    left out, as are zeros and -999s in any other file.
  * CIMSS D-PRINT (IR) and D-MINT (IR + microwave), the deep-learning
    estimates, from their history files,
        https://tropic.ssec.wisc.edu/real-time/DPRINT/<year>/DPRINThist_<year>_<NN><B>.txt
        https://tropic.ssec.wisc.edu/real-time/DMINT/<year>/DMINThist_<year>_<NN><B>.txt
    (time, position, [microwave sensor,] MSLP, Vmax and its 25th and 75th
    percentiles; Vmax is their 30-70th percentile mean).
  * recon10s's own satellite estimates (recon10s_satint), handed in by
    Experimental › Satellite intensity. Nothing is imported from files.

Truth
  * Pressure: each vortex message's minimum pressure, at the fix time.
  * Wind: each pass's surface wind from the aircraft -- the larger of the
    peak flight-level wind times the standard ratio and the eyewall
    dropsondes' WL150 estimates within 45 minutes of the fix. No SFMR.
  * Or NHC's working best track, which is recon-aided while aircraft are in
    the storm.

Matching: an estimate is interpolated to each truth time when estimates
bracket it within `interp_span_h`, else the nearest within `max_gap_min` is
used, else the truth time is skipped. Scores are estimate minus truth.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

import recon10s_intensity as intensity

UTC = dt.timezone.utc
ADT_URL = "https://tropic.ssec.wisc.edu/real-time/adt/{code}-list.txt"
ADT_PAGE = "https://tropic.ssec.wisc.edu/real-time/adt/adt.html"
CIMSS_ML_URL = {"D-PRINT": "https://tropic.ssec.wisc.edu/real-time/DPRINT/{year}/DPRINThist_{year}_{code}.txt",
                "D-MINT": "https://tropic.ssec.wisc.edu/real-time/DMINT/{year}/DMINThist_{year}_{code}.txt"}


# Outside these an estimate is a placeholder, not a value: ADT writes CI 0.0 / 0.0 mb / 0.0 kt for
# every fix over land, and other files use 0 or -999 for missing.
VMAX_RANGE = (5.0, 250.0)          # kt
MSLP_RANGE = (850.0, 1060.0)       # mb


def plausible(value: Optional[float], bounds: Tuple[float, float]) -> Optional[float]:
    """The value if it lies within bounds, else None."""
    return value if value is not None and np.isfinite(value) and bounds[0] <= value <= bounds[1] else None


@dataclass
class Estimate:
    time: dt.datetime
    vmax: Optional[float] = None
    mslp: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    note: str = ""                       # scene type, member, CI ...


# --------------------------------------------------------------------- ADT

def adt_codes(storm_id: str) -> List[str]:
    """CIMSS's storm codes to try for an ATCF id: al092026 -> ['09L'], io012026 -> ['01A', '01B'] -- or just
    the one a storm list gave (sh012027 -> ['01P'] once knackwx has said it is 01P)."""
    m = re.fullmatch(r"([a-z]{2})(\d{2})(\d{4})", (storm_id or "").lower())
    if not m:
        return []
    import recon10s_nhc as nhc
    known = nhc.known_short_id(storm_id)
    if known:
        return [known]
    basin, num = m.group(1), m.group(2)
    letters = {"al": ["L"], "ep": ["E"], "cp": ["C"], "wp": ["W"], "io": ["A", "B"],
               "sh": ["S", "P"], "ls": ["Q"]}.get(basin, [])
    return [num + x for x in letters]


_ADT_LINE = re.compile(
    r"^(?P<date>\d{4}[A-Z]{3}\d{2})\s+(?P<time>\d{6})\s+(?P<ci>\d+\.\d)\s+(?P<mslp>\d+\.\d)\s+"
    r"(?P<vmax>\d+\.\d)\b(?P<rest>.*)$")
# position, fix method and satellite near the end of the line: "22.08   95.34  FCST    GOES19 34.4";
# the cloud temperatures before the scene type look alike, so the last match is the one
_ADT_POS = re.compile(r"\s(-?\d{1,2}\.\d{2})\s+(-?\d{1,3}\.\d{2})\s+([A-Z]{3,8})\s+([A-Z]+-?\d+[A-Z]?)\b")
_ADT_SCENE = re.compile(r"\b(EYE|PINHOLE|LARGE|CDO|EMBC|IRRCDO|CRVBND|SHEAR|UNIFRM|CLRMSK)\b")


def parse_adt(text: str) -> List[Estimate]:
    """Every line of a CIMSS ADT history listing (ADT 8.x/9.x), oldest first."""
    out = []
    for line in (text or "").splitlines():
        m = _ADT_LINE.match(line.strip())
        if not m:
            continue
        try:
            when = dt.datetime.strptime(m.group("date") + m.group("time"), "%Y%b%d%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            continue
        vmax = plausible(float(m.group("vmax")), VMAX_RANGE)
        mslp = plausible(float(m.group("mslp")), MSLP_RANGE)
        if vmax is None and mslp is None:
            continue                                   # over land (scene LAND, CI 0.0): no estimate
        est = Estimate(when, vmax=vmax, mslp=mslp, note=f"CI {m.group('ci')}")
        rest = " " + m.group("rest")
        found = list(_ADT_POS.finditer(rest))
        pos = found[-1] if found else None
        if pos:
            est.lat = float(pos.group(1))
            est.lon = -float(pos.group(2))            # ADT longitudes are positive west
        sc = _ADT_SCENE.search(rest)
        if sc:
            est.note += f", {sc.group(1)}"
        out.append(est)
    out.sort(key=lambda e: e.time)
    return out


def fetch_adt(storm_id: str, fetch=None) -> Tuple[List[Estimate], str, str]:
    """
    (estimates, url, raw text) for a storm, trying each of its CIMSS codes; empty if none. The real-time
    lists carry no year, so a list left over from an earlier storm of the same code (last season's 01S)
    is recognised by its dates and passed over.
    """
    import recon10s_nhc as nhc
    fetch = fetch or (lambda url: nhc.fetch_text(url, timeout=30, max_age_s=300))
    win = nhc.season_window(storm_id)
    for code in adt_codes(storm_id):
        url = ADT_URL.format(code=code)
        try:
            text = fetch(url)
        except Exception:
            text = None
        if text:
            ests = parse_adt(text)
            if win:
                ests = [e for e in ests if win[0] <= e.time < win[1]]
            if ests:
                return ests, url, text
    return [], "", ""


# --------------------------------------------------------- D-PRINT / D-MINT

_MONTHS = {m: i + 1 for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
                                           "NOV", "DEC"))}


def parse_cimss_ml(text: str) -> List[Estimate]:
    """A CIMSS D-PRINT or D-MINT history file, oldest first (repeated lines once)."""
    out = {}
    for line in (text or "").splitlines():
        tok = line.split()
        if len(tok) not in (8, 9) or len(tok[0]) != 9 or not tok[0][:4].isdigit() or tok[0][4:7] not in _MONTHS:
            continue
        try:
            when = dt.datetime(int(tok[0][:4]), _MONTHS[tok[0][4:7]], int(tok[0][7:9]), int(tok[1][:2]),
                               int(tok[1][2:4]), tzinfo=UTC)
            lat, lon = float(tok[2]), float(tok[3])
            mslp = plausible(float(tok[-4]), MSLP_RANGE) if tok[-4].lower() != "nan" else None
            vmax = plausible(float(tok[-3]), VMAX_RANGE)
            q25, q75 = float(tok[-2]), float(tok[-1])
        except ValueError:
            continue
        if vmax is None and mslp is None:
            continue
        sensor = tok[4] if len(tok) == 9 else ""
        note = (sensor + ", " if sensor else "") + f"25-75 %: {q25:.0f}-{q75:.0f} kt"
        out[(when, sensor)] = Estimate(when, vmax=vmax, mslp=mslp, lat=lat, lon=lon, note=note)
    return sorted(out.values(), key=lambda e: e.time)


def fetch_cimss_ml(storm_id: str, product: str, fetch=None) -> Tuple[List[Estimate], str, str]:
    """(estimates, url, raw text) of CIMSS's "D-PRINT" or "D-MINT" for a storm; empty if none."""
    import recon10s_nhc as nhc
    fetch = fetch or (lambda url: nhc.fetch_text(url, timeout=30, max_age_s=300))
    m = re.fullmatch(r"([a-z]{2})\d{2}(\d{4})", (storm_id or "").lower())
    if not m or product not in CIMSS_ML_URL:
        return [], "", ""
    year = int(m.group(2))
    years = [year, year - 1] if m.group(1) == "sh" else [year]     # CIMSS files a southern season by its end year
    for code in adt_codes(storm_id):
        for y in years:
            url = CIMSS_ML_URL[product].format(year=y, code=code)
            try:
                text = fetch(url)
            except Exception:
                text = None
            if text:
                ests = parse_cimss_ml(text)
                if ests:
                    return ests, url, text
    return [], "", ""


# ------------------------------------------------------------------ truth

def recon_truth(bundle, drop_window_min: float = 45.0) -> Tuple[List[intensity.Point], List[intensity.Point]]:
    """
    (pressure truths, wind truths) from the recon. Pressure: vortex-message
    minimum pressure. Wind: per pass, the larger of the flight-level estimate
    and eyewall dropsonde WL150 estimates near the fix.
    """
    if bundle is None:
        return [], []
    import recon10s_surface as surface
    pres = intensity.vdm_pressures(bundle.vdms)
    drops = intensity.drop_surface_winds(bundle.drops)
    ov = bundle.overlay
    est = surface.overlay_surface(ov) if ov is not None and len(ov) else None
    winds = []
    for p in bundle.passes or []:
        cands = []
        f = intensity.pass_peak_wind(ov, p, est) if est is not None else None
        if f is not None:
            cands.append(f)
        cands += [d for d in drops if abs((d.time - p.fix.time).total_seconds()) <= drop_window_min * 60]
        if cands:
            best = max(cands, key=lambda c: c.value)
            winds.append(intensity.Point(p.fix.time, best.value, best.label))
    return pres, winds


def best_track_truth(track) -> Tuple[List[intensity.Point], List[intensity.Point]]:
    p, w = intensity.best_track_series(track)
    return p, w


# --------------------------------------------------------------- matching

@dataclass
class Pair:
    time: dt.datetime
    truth: float
    estimate: float
    method: str                          # "interpolated" or "nearest (+12 min)"
    label: str = ""

    @property
    def diff(self) -> float:
        return self.estimate - self.truth


@dataclass
class Scores:
    n: int
    bias: float
    mae: float
    rmse: float

    @property
    def text(self) -> str:
        if not self.n:
            return "no matches"
        return f"n={self.n}  bias {self.bias:+.1f}  MAE {self.mae:.1f}  RMSE {self.rmse:.1f}"


def match(estimates: Sequence[Estimate], truths: Sequence[intensity.Point], what: str,
          max_gap_min: float = 60.0, interp_span_h: float = 3.0) -> List[Pair]:
    """Each truth with the estimate at its time (`what` is 'vmax' or 'mslp')."""
    ests = [(e.time.timestamp(), getattr(e, what)) for e in estimates if getattr(e, what) is not None]
    if not ests:
        return []
    ests.sort()
    ts = np.array([e[0] for e in ests])
    vs = np.array([e[1] for e in ests], float)
    out = []
    for tr in truths:
        x = tr.time.timestamp()
        k = int(np.searchsorted(ts, x))
        if 0 < k < len(ts) and ts[k] - ts[k - 1] <= interp_span_h * 3600:
            f = (x - ts[k - 1]) / (ts[k] - ts[k - 1]) if ts[k] > ts[k - 1] else 0.0
            out.append(Pair(tr.time, tr.value, float(vs[k - 1] + f * (vs[k] - vs[k - 1])), "interpolated", tr.label))
            continue
        if k < len(ts) and ts[k] == x:
            out.append(Pair(tr.time, tr.value, float(vs[k]), "same time", tr.label))
            continue
        j = int(np.argmin(np.abs(ts - x)))
        gap = (ts[j] - x) / 60.0
        if abs(gap) <= max_gap_min:
            out.append(Pair(tr.time, tr.value, float(vs[j]), f"nearest ({gap:+.0f} min)", tr.label))
    return out


def scores(pairs: Sequence[Pair]) -> Scores:
    if not pairs:
        return Scores(0, float("nan"), float("nan"), float("nan"))
    d = np.array([p.diff for p in pairs], float)
    return Scores(len(d), float(d.mean()), float(np.abs(d).mean()), float(np.sqrt((d ** 2).mean())))


def as_points(estimates: Sequence[Estimate], what: str) -> List[intensity.Point]:
    """Estimates as timeline points."""
    return [intensity.Point(e.time, float(getattr(e, what)), e.note) for e in estimates
            if getattr(e, what) is not None]
