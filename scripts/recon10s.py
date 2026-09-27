#!/usr/bin/env python3
"""
IWG1 -> HDOB converter

- Reads IWG1 ASCII stream/file/URL (per UCAR IWG1 packet spec).
- Aggregates to HDOB intervals (supports 10, 30, 60, 120-second intervals).
- Optional time-of-day window filtering (--start/--end).
- Caches downloads and resumes them with HTTP Range, so re-reading a mission
  that is still flying only transfers the part that is new.

Speed note (changed in 1.2.0): parsing used to run through a ThreadPool. It is
pure Python and GIL-bound, so 150k submissions cost more than the work itself
(9.4s with 4 workers vs 2.1s with 1 on a 42 MB file). It is now a single pass
with a fast timestamp reader, which does the same file in about 1s. --workers
is still accepted so existing scripts keep working, but it no longer changes
anything.
"""
from __future__ import annotations
import argparse
import datetime as dt
import io
import math
import re
import sys
import os
from collections import deque
from typing import List, Optional, Tuple

import hashlib
import tempfile
from typing import Callable, Iterator

try:
    import requests
except Exception:
    requests = None

# ----------------------------- constants & basic defs -----------------------------
R_EARTH = 6371000.0
G0 = 9.80665
R_D = 287.05
LAPSE = 0.0065  # K/m
P0_STD = 1013.25  # hPa
T0_STD = 288.15  # K
KTS_PER_MPS = 1.9438444924406
MSLP_BIAS_CORRECTION = -2.4  # mb
VERSION = "v1.7.1"
UTC = dt.timezone.utc
DEFAULT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "recon10s_cache")

BASE_FIELDS = [
    "Lat","Lon","GPS_MSL_Alt","WGS_84_Alt","Press_Alt","Radar_Alt","Grnd_Spd",
    "True_Airspeed","Indicated_Airspeed","Mach_Number","Vert_Velocity",
    "True_Hdg","Track","Drift","Pitch","Roll","Side_slip","Angle_of_Attack",
    "Ambient_Temp","Dew_Point","Total_Temp","Static_Press","Dynamic_Press",
    "Cabin_Pressure","Wind_Speed","Wind_Dir","Vert_Wind_Spd","Solar_Zenith",
    "Sun_Elev_AC","Sun_Az_Grd","Sun_Az_AC",
]

class IWG1Row:
    __slots__ = ("t","lat","lon","ps_hpa","ga_m","temp_c","td_c","wspd_ms","wdir_deg")
    def __init__(self, t: dt.datetime, lat: Optional[float], lon: Optional[float], ps_hpa: Optional[float],
                 ga_m: Optional[float], temp_c: Optional[float], td_c: Optional[float],
                 wspd_ms: Optional[float], wdir_deg: Optional[float]):
        self.t = t
        self.lat = lat
        self.lon = lon
        self.ps_hpa = ps_hpa
        self.ga_m = ga_m
        self.temp_c = temp_c
        self.td_c = td_c
        self.wspd_ms = wspd_ms
        self.wdir_deg = wdir_deg

# ----------------------------- parsing helpers -----------------------------
def parse_float(x: str) -> Optional[float]:
    try:
        x = x.strip()
        if x == "" or x.lower() in {"nan","inf","+inf","-inf"}:
            return None
        return float(x)
    except Exception:
        return None

def parse_float_fast(x: str) -> Optional[float]:
    """Same result as parse_float, minus the strip() and the set lookup."""
    if not x:
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    if v != v or v in (_INF, _NEG_INF):  # NaN / inf
        return None
    return v


_INF = float("inf")
_NEG_INF = float("-inf")


def parse_time_fast(s: str) -> Optional[dt.datetime]:
    """
    IWG1 timestamps are almost always YYYY-MM-DDTHH:MM:SS. Building the
    datetime straight from the digits is ~30x quicker than strptime, which
    is the single biggest cost when reading a long mission. Anything that
    does not match falls through to the tolerant parser below.
    """
    if len(s) >= 19 and s[4] == "-" and s[7] == "-" and s[13] == ":" and s[16] == ":":
        try:
            return dt.datetime(int(s[0:4]), int(s[5:7]), int(s[8:10]),
                               int(s[11:13]), int(s[14:16]), int(s[17:19]), tzinfo=UTC)
        except ValueError:
            pass
    try:
        return parse_time(s)
    except ValueError:
        return None


def seconds_of_day(s: str) -> Optional[int]:
    """UTC seconds-past-midnight straight off the timestamp text."""
    if len(s) >= 19 and s[13] == ":" and s[16] == ":":
        try:
            return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])
        except ValueError:
            return None
    return None


def parse_time(s: str) -> dt.datetime:
    s = s.strip()
    fmt_variants = [
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%dT%H%M%S", "%Y%m%d %H%M%S",
        "%Y%m%dT%H%M%S.%f", "%Y-%m-%dT%H:%M:%S.%f",
    ]
    for f in fmt_variants:
        try:
            return dt.datetime.strptime(s, f).replace(tzinfo=dt.timezone.utc)
        except Exception:
            pass
    try:
        t = dt.datetime.fromisoformat(s)
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t.astimezone(dt.timezone.utc)
    except Exception:
        raise ValueError(f"Unrecognized time format: {s}")

def iwg1_iter_lines_from_text(text: str):
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        if not raw.startswith("IWG1,"):
            continue
        parts = raw.split(",")
        if len(parts) < 3:
            continue
        yield parts

def parse_iwg1_row(parts: List[str]) -> Optional[IWG1Row]:
    try:
        t = parse_time(parts[1])
        get = lambda idx: parse_float(parts[idx]) if idx < len(parts) else None
        lat = get(2)
        lon = get(3)
        gps_msl_alt_m = get(4)
        ga_m = gps_msl_alt_m if gps_msl_alt_m is not None else get(5)
        ps_hpa = get(23)
        temp_c = get(20)
        td_c = get(21)
        wspd_ms = get(26)
        wdir_deg = get(27)
        return IWG1Row(t, lat, lon, ps_hpa, ga_m, temp_c, td_c, wspd_ms, wdir_deg)
    except Exception:
        return None

def parse_iwg1_line(line: str) -> Optional[IWG1Row]:
    """
    Parse one raw IWG1 line. Splits only as far as the last field we use
    (wind direction, index 27) so the AOC-specific columns NOAA appends after
    the standard packet are never touched.
    """
    if not line.startswith("IWG1,"):
        return None
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]

    p = line.split(",", 28)
    if len(p) < 28:
        return None

    t = parse_time_fast(p[1])
    if t is None:
        return None

    ga = parse_float_fast(p[4])
    if ga is None:
        ga = parse_float_fast(p[5])

    return IWG1Row(t, parse_float_fast(p[2]), parse_float_fast(p[3]),
                   parse_float_fast(p[23]), ga, parse_float_fast(p[20]),
                   parse_float_fast(p[21]), parse_float_fast(p[26]),
                   parse_float_fast(p[27]))


# ----------------------------- physics helpers -----------------------------
def isa_z_from_p(ps_hpa: float) -> float:
    expo = (R_D * LAPSE) / G0
    z = (T0_STD / LAPSE) * (1.0 - (ps_hpa / P0_STD) ** expo)
    return max(0.0, z)

def d_value_m(ga_m: float, ps_hpa: float) -> Optional[int]:
    if ga_m is None or ps_hpa is None:
        return None
    z_std = isa_z_from_p(ps_hpa)
    d = ga_m - z_std
    return int(round(d))

def extrapolate_surface_pressure(ps_hpa: float, z_m: float, t_c: Optional[float]) -> Optional[float]:
    if ps_hpa is None or z_m is None:
        return None
    if t_c is None:
        z_est = isa_z_from_p(ps_hpa)
        T_z = T0_STD - LAPSE * z_est
    else:
        T_z = t_c + 273.15
    T_bar = T_z + 0.5 * LAPSE * z_m
    if T_bar <= 0:
        return None
    p0 = ps_hpa * math.exp(G0 * z_m / (R_D * T_bar))
    return float(p0 + MSLP_BIAS_CORRECTION)

# ----------------------------- HDOB encoding helpers -----------------------------
def lat_to_LLLLH(lat: float) -> str:
    hemi = "N" if lat >= 0 else "S"
    lat_abs = abs(lat)
    deg = int(math.floor(lat_abs))
    minutes = int(round((lat_abs - deg) * 60.0))
    if minutes == 60:
        deg += 1
        minutes = 0
    return f"{deg:02d}{minutes:02d}{hemi}"

def lon_to_NNNNNH(lon: float) -> str:
    hemi = "E" if lon >= 0 else "W"
    lon_abs = abs(lon)
    deg = int(math.floor(lon_abs))
    minutes = int(round((lon_abs - deg) * 60.0))
    if minutes == 60:
        deg += 1
        minutes = 0
    return f"{deg:03d}{minutes:02d}{hemi}"

def encode_PPPP(ps_hpa: Optional[float]) -> str:
    if ps_hpa is None:
        return "////"
    tenths = int(round(ps_hpa * 10.0))
    if tenths >= 10000:
        tenths -= 10000
    return f"{tenths:04d}"

def encode_GGGGG(z_m: Optional[float]) -> str:
    if z_m is None:
        return "/////"
    val = int(round(z_m))
    return f"{val:05d}"

def encode_XXXX(ps_hpa: Optional[float], z_m: Optional[float], t_c: Optional[float]) -> str:
    if ps_hpa is None or z_m is None:
        return "////"
    if ps_hpa >= 550.0:
        p0 = extrapolate_surface_pressure(ps_hpa, z_m, t_c)
        return encode_PPPP(p0)
    else:
        d = d_value_m(z_m, ps_hpa)
        if d is None:
            return "////"
        if d < 0:
            d = 5000 + d
        return f"{int(round(d))%10000:04d}"

def encode_sxxx(val_c: Optional[float]) -> str:
    if val_c is None:
        return "///"
    sign = "+" if val_c >= 0 else "-"
    mag = int(round(abs(val_c) * 10.0))
    return f"{sign}{mag:03d}"

def encode_wwwSSS(wdir_deg: Optional[float], wspd_ms: Optional[float]) -> str:
    if wdir_deg is None or wspd_ms is None:
        return "//////"
    www = int(round(wdir_deg)) % 360
    sss = int(round(wspd_ms * KTS_PER_MPS))
    return f"{www:03d}{sss:03d}"

def encode_TTT(val: Optional[int]) -> str:
    if val is None:
        return "///"
    return f"{int(round(val)):03d}"

# ----------------------------- wind mean/peak logic -----------------------------
def vector_mean_wind(dir_deg_list: List[float], spd_ms_list: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not dir_deg_list or not spd_ms_list:
        return (None, None)
    u = 0.0; v = 0.0; n = 0
    for d, s in zip(dir_deg_list, spd_ms_list):
        if d is None or s is None:
            continue
        rad = math.radians(d)
        u += -s * math.sin(rad)
        v += -s * math.cos(rad)
        n += 1
    if n == 0:
        return (None, None)
    u /= n; v /= n
    spd = math.hypot(u, v)
    dir_rad = math.atan2(-u, -v)
    deg = (math.degrees(dir_rad) + 360.0) % 360.0
    return (deg, spd)

def compute_peak10s(times: List[dt.datetime], spd_ms_list: List[Optional[float]]) -> Optional[float]:
    if not times:
        return None
    samples = [(t, s) for t, s in zip(times, spd_ms_list) if s is not None]
    if not samples:
        return None
    dq = deque()
    sum_s = 0.0
    count = 0
    best_mean = 0.0
    for t, s in samples:
        dq.append((t, s))
        sum_s += s; count += 1
        tmin = t - dt.timedelta(seconds=10)
        while dq and dq[0][0] < tmin:
            _, s0 = dq.popleft()
            sum_s -= s0; count -= 1
        if count > 0:
            best_mean = max(best_mean, sum_s/count)
    if best_mean == 0.0:
        best = max(s for _, s in samples)
        return best * KTS_PER_MPS
    return best_mean * KTS_PER_MPS

# ----------------------------- time input helper -----------------------------
def _time_input_to_seconds(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    s2 = s.strip()
    if s2 == "":
        return None
    if ":" in s2:
        parts = s2.split(":")
        if len(parts) == 2:
            hh, mm = int(parts[0]), int(parts[1]); ss = 0
        elif len(parts) == 3:
            hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            raise ValueError(f"Invalid time string: {s}")
    else:
        if not s2.isdigit():
            raise ValueError(f"Invalid time string: {s}")
        if len(s2) == 4:
            hh, mm, ss = int(s2[0:2]), int(s2[2:4]), 0
        elif len(s2) == 6:
            hh, mm, ss = int(s2[0:2]), int(s2[2:4]), int(s2[4:6])
        else:
            raise ValueError(f"Invalid time string: {s}")
    if not (0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60):
        raise ValueError(f"Time out of range: {s}")
    return hh*3600 + mm*60 + ss

# ----------------------------- conversion to HDOB (core) -----------------------------
def convert_iwg1_to_hdob(rows: List[IWG1Row], mission: str, storm_date: Optional[dt.date] = None,
                         interval_s: int = 30, lines_per_msg: int = 20,
                         header_center: str = "KNHC", wmo_header: str = "URNT15",
                         default_flags: str = "00") -> str:
    """
    Encode rows as HDOB messages of `lines_per_msg` obs each.

    Dating follows NHC's own practice, checked against their archive: each
    message's mission line carries the date of its FIRST ob, and its WMO
    header DDHHMM is the time of its LAST ob. So on a mission that crosses
    00Z the date on the mission line rolls over with the data. (Before 1.7.0
    every message repeated one date and one header time.)

    `storm_date` is kept for compatibility and no longer used: the IWG1
    timestamps already carry the date.
    """
    if not rows:
        return ""
    rows = sorted(rows, key=lambda r: r.t)
    start = rows[0].t
    out_lines: List[str] = []
    out_times: List[dt.datetime] = []

    # align first bin to multiple of interval_s
    bin_start = dt.datetime.fromtimestamp((start.timestamp() // interval_s) * interval_s, tz=dt.timezone.utc)
    i = 0
    while i < len(rows):
        cur_bin_end = bin_start + dt.timedelta(seconds=interval_s)
        bin_rows: List[IWG1Row] = []
        while i < len(rows) and rows[i].t < cur_bin_end:
            bin_rows.append(rows[i])
            i += 1

        if bin_rows:
            mid_time = bin_start + dt.timedelta(seconds=interval_s // 2)
            lat_vals = [r.lat for r in bin_rows if r.lat is not None]
            lon_vals = [r.lon for r in bin_rows if r.lon is not None]
            lat = sum(lat_vals)/len(lat_vals) if lat_vals else None
            lon = sum(lon_vals)/len(lon_vals) if lon_vals else None

            ps_vals = [r.ps_hpa for r in bin_rows if r.ps_hpa is not None]
            ps = sum(ps_vals)/len(ps_vals) if ps_vals else None

            ga_vals = [r.ga_m for r in bin_rows if r.ga_m is not None]
            ga = sum(ga_vals)/len(ga_vals) if ga_vals else None

            t_vals = [r.temp_c for r in bin_rows if r.temp_c is not None]
            t_c = sum(t_vals)/len(t_vals) if t_vals else None

            td_vals = [r.td_c for r in bin_rows if r.td_c is not None]
            td_c = sum(td_vals)/len(td_vals) if td_vals else None

            d_list = [r.wdir_deg for r in bin_rows if r.wdir_deg is not None and r.wspd_ms is not None]
            s_list = [r.wspd_ms for r in bin_rows if r.wdir_deg is not None and r.wspd_ms is not None]
            mean_dir, mean_spd = vector_mean_wind(d_list, s_list)

            times = [r.t for r in bin_rows]
            spds = [r.wspd_ms for r in bin_rows]
            peak10 = compute_peak10s(times, spds)

            hhmmss = mid_time.strftime("%H%M%S")
            lat_str = lat_to_LLLLH(lat) if lat is not None else "/////"
            lon_str = lon_to_NNNNNH(lon) if lon is not None else "//////"
            pppp = encode_PPPP(ps)
            ggggg = encode_GGGGG(ga)
            xxxx = encode_XXXX(ps, ga, t_c)
            sTTT = encode_sxxx(t_c)
            sddd = encode_sxxx(td_c)
            wwwSSS = encode_wwwSSS(mean_dir, mean_spd)

            if peak10 is None or (isinstance(peak10, float) and math.isnan(peak10)):
                MMM = "///"
            else:
                try:
                    MMM = encode_TTT(int(round(float(peak10))))
                except Exception:
                    MMM = "///"

            KKK = "///"
            ppp = "///"
            FF = default_flags

            line = f"{hhmmss} {lat_str} {lon_str} {pppp} {ggggg} {xxxx} {sTTT} {sddd} {wwwSSS} {MMM} {KKK} {ppp} {FF}"
            out_lines.append(line)
            out_times.append(mid_time)

        bin_start = cur_bin_end
        if i < len(rows) and rows[i].t >= bin_start + dt.timedelta(seconds=interval_s):
            next_ts = rows[i].t
            bin_start = dt.datetime.fromtimestamp((next_ts.timestamp() // interval_s) * interval_s, tz=dt.timezone.utc)

    msgs: List[str] = []
    for obnum, i in enumerate(range(0, len(out_lines), lines_per_msg), start=1):
        lines = out_lines[i:i + lines_per_msg]
        first_t = out_times[i]
        last_t = out_times[min(i + lines_per_msg, len(out_times)) - 1]
        header = f"{wmo_header} {header_center} {last_t.strftime('%d%H%M')}"
        mission_line = f"{mission} HDOB {obnum:02d} {first_t.strftime('%Y%m%d')}"
        msg = "\n".join([header, mission_line] + lines + ["$$"])
        msgs.append(msg)
    return "\n\n".join(msgs) + ("\n" if msgs else "")

# ----------------------------- download cache + reading -----------------------------
ProgressFn = Optional[Callable[[int, Optional[int]], None]]


def cache_path_for(url: str, cache_dir: Optional[str] = None) -> str:
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{digest}.iwg1")


def download_iwg1(url: str, cache_dir: Optional[str] = None, use_cache: bool = True,
                  timeout: float = 180.0, progress: ProgressFn = None) -> str:
    """
    Put the IWG1 file on disk and return its path.

    With the cache on, a file we already hold is topped up with a Range
    request instead of downloaded again. Mission files are appended to as the
    plane flies, so a re-read part way through a flight transfers only the
    minutes that are new rather than the whole thing.
    """
    if requests is None:
        raise RuntimeError("'requests' is required to read from URL; install it or use --path.")

    path = cache_path_for(url, cache_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    have = os.path.getsize(path) if (use_cache and os.path.exists(path)) else 0

    headers = {"Accept-Encoding": "gzip"}
    if have:
        headers["Range"] = f"bytes={have}-"

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=timeout)
    except Exception as exc:
        if have:
            print(f"Download failed ({exc}); using the cached copy.", file=sys.stderr)
            return path
        raise

    with resp:
        if resp.status_code == 416:  # cache already holds the whole file
            if progress:
                progress(have, have)
            return path

        resp.raise_for_status()
        appending = resp.status_code == 206
        if have and not appending:
            # Server ignored the Range header, so we are getting it all again.
            have = 0

        total = resp.headers.get("Content-Length")
        total = (int(total) + have) if total and total.isdigit() else None

        done = have
        mode = "ab" if appending and have else "wb"
        with open(path, mode) as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)

    return path


def iter_iwg1_rows(path: str, start_sec: Optional[int] = None,
                   end_sec: Optional[int] = None,
                   start_dt: Optional[dt.datetime] = None,
                   end_dt: Optional[dt.datetime] = None) -> Iterator[IWG1Row]:
    """
    Stream rows from a file. When a UTC window is given, rows outside it are
    rejected on the timestamp text before any float parsing happens, which is
    most of the cost — pulling 20 minutes out of a 10-hour mission gets much
    cheaper than reading the lot and filtering afterwards.
    """
    dated = start_dt is not None or end_dt is not None
    windowed = (start_sec is not None or end_sec is not None) and not dated
    wraps = (start_sec is not None and end_sec is not None and start_sec > end_sec)

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("IWG1,"):
                continue
            if windowed:
                sec = seconds_of_day(line[5:34].split(",", 1)[0])
                if sec is not None:
                    if start_sec is not None and end_sec is not None:
                        keep = (start_sec <= sec <= end_sec) if not wraps else (sec >= start_sec or sec <= end_sec)
                    elif start_sec is not None:
                        keep = sec >= start_sec
                    else:
                        keep = sec <= end_sec
                    if not keep:
                        continue
            if dated:
                t = parse_time_fast(line[5:34].split(",", 1)[0])
                if t is not None and ((start_dt is not None and t < start_dt)
                                      or (end_dt is not None and t > end_dt)):
                    continue
            row = parse_iwg1_line(line)
            if row is not None:
                yield row


def read_iwg1(path: Optional[str], url: Optional[str], workers: int = 1,
              start_sec: Optional[int] = None, end_sec: Optional[int] = None,
              start_dt: Optional[dt.datetime] = None, end_dt: Optional[dt.datetime] = None,
              use_cache: bool = True, cache_dir: Optional[str] = None,
              progress: ProgressFn = None) -> List[IWG1Row]:
    """
    Read IWG1 rows from a path or URL.

    `workers` is accepted for backwards compatibility and ignored: parsing is
    a single pass now, which is several times quicker than the old thread pool.
    """
    if path:
        src = path
    elif url:
        src = download_iwg1(url, cache_dir=cache_dir, use_cache=use_cache, progress=progress)
    else:
        raise ValueError("Provide --path or --url")

    return list(iter_iwg1_rows(src, start_sec, end_sec, start_dt, end_dt))


# ----------------------------- filtering by time of day -----------------------------
def _filter_rows_by_time_of_day(rows: List[IWG1Row], start_sec: Optional[int], end_sec: Optional[int]) -> List[IWG1Row]:
    if start_sec is None and end_sec is None:
        return rows
    out = []
    for r in rows:
        t = r.t.astimezone(dt.timezone.utc)
        sec = t.hour * 3600 + t.minute * 60 + t.second
        if start_sec is not None and end_sec is not None:
            if start_sec <= end_sec:
                keep = (start_sec <= sec <= end_sec)
            else:
                keep = (sec >= start_sec or sec <= end_sec)
        elif start_sec is not None:
            keep = (sec >= start_sec)
        else:
            keep = (sec <= end_sec)
        if keep:
            out.append(r)
    return out

_DATE_TIME_RE = re.compile(
    r"^\s*(\d{4})-?(\d{2})-?(\d{2})[ T_]+(\d{1,2}):?(\d{2})(?::?(\d{2}))?\s*$")


def parse_window_bound(text: Optional[str], date: Optional[dt.date] = None):
    """
    One end of a UTC window. Returns (datetime or None, seconds-of-day or None):

      "2026-09-26 17:45", "20260926 1745", "2026-09-26T17:45:30"  -> a datetime
      "17:45" or "1745" with `date` given                          -> a datetime
      "17:45" alone                                                -> seconds of day
      blank                                                        -> (None, None)

    A bare time-of-day window repeats every day, which is wrong for a mission
    spanning 00Z -- it would take 12-14Z from both days. A date pins it.
    """
    if text is None or not text.strip():
        return None, None
    m = _DATE_TIME_RE.match(text)
    if m:
        y, mo, d, hh, mm, ss = m.groups()
        return dt.datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss or 0),
                           tzinfo=dt.timezone.utc), None
    sec = _time_input_to_seconds(text)
    if date is not None:
        return (dt.datetime.combine(date, dt.time(), tzinfo=dt.timezone.utc)
                + dt.timedelta(seconds=sec)), None
    return None, sec


def filter_rows_window(rows: List[IWG1Row], start_dt: Optional[dt.datetime] = None,
                       end_dt: Optional[dt.datetime] = None,
                       start_sec: Optional[int] = None, end_sec: Optional[int] = None) -> List[IWG1Row]:
    """Full-datetime bounds where given; bare time-of-day bounds fall back to the old rule."""
    if start_dt is not None or end_dt is not None:
        return [r for r in rows
                if (start_dt is None or r.t >= start_dt) and (end_dt is None or r.t <= end_dt)]
    return _filter_rows_by_time_of_day(rows, start_sec, end_sec)


def auto_mission_from_tail(url_or_path: str, fallback: str = "AFXXX 0000A INVEST") -> str:
    tail = url_or_path.split("/")[-1]
    hint = tail.replace("_", " ")
    tokens = hint.split()
    for i in range(len(tokens)-1):
        if len(tokens[i]) == 5 and tokens[i][0:4].isdigit() and tokens[i][4] in ("A","B"):
            return f"AFXXX {tokens[i]} {tokens[i+1].upper()}"
    return fallback

# ----------------------------- CLI entrypoint -----------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Convert IWG1 to HDOB (supports time-window filtering and multithreaded parsing).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--path", help="Path to local IWG1 file")
    src.add_argument("--url", help="URL to IWG1 file")
    ap.add_argument("--mission", help="Mission identifier line prefix")
    ap.add_argument("--storm-date", required=False,
                    help="Accepted for compatibility and ignored: each message now carries "
                         "the date of its first ob, as NHC's do")
    ap.add_argument("--interval", type=int, choices=(5,10,30,60,120), default=30, help="HDOB time resolution (s)")
    ap.add_argument("--lines-per-message", type=int, default=20, help="Number of lines per HDOB message")
    ap.add_argument("--out", help="Output file path for HDOB text; default prints to stdout")
    ap.add_argument("--start", default=None,
                    help="UTC start: 'YYYY-MM-DD HH:MM', or a time (HH:MM, HHMM, HH:MM:SS) "
                         "with --start-date. A bare time repeats every day, as before.")
    ap.add_argument("--end", default=None,
                    help="UTC end: 'YYYY-MM-DD HH:MM', or a time with --end-date")
    ap.add_argument("--start-date", default=None, help="YYYYMMDD or YYYY-MM-DD date for --start")
    ap.add_argument("--end-date", default=None, help="YYYYMMDD or YYYY-MM-DD date for --end")
    ap.add_argument("--workers", type=int, default=1, help="Accepted for compatibility; parsing is single-pass and ignores it")
    ap.add_argument("--no-cache", action="store_true", help="Always download the whole file instead of resuming a cached copy")
    ap.add_argument("--cache-dir", help=f"Where to keep downloaded IWG1 files (default: {DEFAULT_CACHE_DIR})")
    args = ap.parse_args(argv)

    # Validate start/end before doing any work.
    def _date(text, flag):
        if not text:
            return None
        return dt.datetime.strptime(text.replace("-", ""), "%Y%m%d").date()

    try:
        start_dt, start_sec = parse_window_bound(args.start, _date(args.start_date, "--start-date"))
    except ValueError as e:
        print(f"Invalid --start value: {e}", file=sys.stderr)
        return 4
    try:
        end_dt, end_sec = parse_window_bound(args.end, _date(args.end_date, "--end-date"))
    except ValueError as e:
        print(f"Invalid --end value: {e}", file=sys.stderr)
        return 5
    # A dated bound paired with a bare time: put the time on the dated bound's day.
    if start_dt is not None and end_dt is None and end_sec is not None:
        end_dt = dt.datetime.combine(start_dt.date(), dt.time(), tzinfo=dt.timezone.utc) + dt.timedelta(seconds=end_sec)
        if end_dt < start_dt:
            end_dt += dt.timedelta(days=1)
        end_sec = None
    if end_dt is not None and start_dt is None and start_sec is not None:
        start_dt = dt.datetime.combine(end_dt.date(), dt.time(), tzinfo=dt.timezone.utc) + dt.timedelta(seconds=start_sec)
        if start_dt > end_dt:
            start_dt -= dt.timedelta(days=1)
        start_sec = None
    if start_dt and end_dt and end_dt < start_dt:
        print(f"--end ({end_dt:%Y-%m-%d %H:%M}Z) is before --start ({start_dt:%Y-%m-%d %H:%M}Z).", file=sys.stderr)
        return 6

    def _progress(done, total):
        if total:
            print(f"\rDownloading… {done/1e6:.1f} of {total/1e6:.1f} MB", end="", file=sys.stderr)
        else:
            print(f"\rDownloading… {done/1e6:.1f} MB", end="", file=sys.stderr)

    rows = read_iwg1(args.path, args.url, start_sec=start_sec, end_sec=end_sec,
                     start_dt=start_dt, end_dt=end_dt, use_cache=not args.no_cache, cache_dir=args.cache_dir,
                     progress=_progress if args.url else None)
    if args.url:
        print("", file=sys.stderr)
    if not rows:
        print("No IWG1 rows parsed.", file=sys.stderr)
        return 2

    print(f"Read {len(rows)} rows from source (after parsing).")
    if any(v is not None for v in (start_dt, end_dt, start_sec, end_sec)):
        rows_filtered = filter_rows_window(rows, start_dt, end_dt, start_sec, end_sec)
        shown_start = f"{start_dt:%Y-%m-%d %H:%M:%S}Z" if start_dt else args.start
        shown_end = f"{end_dt:%Y-%m-%d %H:%M:%S}Z" if end_dt else args.end
        print(f"{len(rows_filtered)} rows remain after applying time window (start={shown_start}, end={shown_end}).")
        rows = rows_filtered
        if not rows:
            print("No rows after filtering -- no HDOB will be produced.", file=sys.stderr)

    first_date = rows[0].t.date() if rows else dt.datetime.utcnow().date()
    storm_date = dt.datetime.strptime(args.storm_date, "%Y%m%d").date() if args.storm_date else first_date

    src_label = args.url or args.path or ""
    mission = args.mission or auto_mission_from_tail(src_label)

    text = convert_iwg1_to_hdob(rows, mission=mission, storm_date=storm_date,
                                interval_s=args.interval, lines_per_msg=args.lines_per_message)
    if not text:
        print("No HDOB output generated.", file=sys.stderr)
        return 3

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {args.out} ({len(text.splitlines())} lines)")
    else:
        print(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


