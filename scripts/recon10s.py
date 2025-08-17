#!/usr/bin/env python3
"""
IWG1 → HDOB converter (multithreaded parsing)

- Reads IWG1 ASCII stream/file/URL (per UCAR IWG1 packet spec).
- Aggregates to HDOB intervals (supports 10, 30, 60, 120-second intervals).
- Optional time-of-day window filtering (--start/--end).
- Multithreaded parsing via --workers (defaults to 4).
"""
from __future__ import annotations
import argparse
import datetime as dt
import io
import math
import sys
import os
from collections import deque
from typing import List, Optional, Tuple

# concurrency
import concurrent.futures

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
    return float(p0)

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
def convert_iwg1_to_hdob(rows: List[IWG1Row], mission: str, storm_date: dt.date,
                         interval_s: int = 30, lines_per_msg: int = 20,
                         header_center: str = "KNHC", wmo_header: str = "URNT15",
                         default_flags: str = "00") -> str:
    if not rows:
        return ""
    rows = sorted(rows, key=lambda r: r.t)
    start = rows[0].t
    out_lines: List[str] = []

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

        bin_start = cur_bin_end
        if i < len(rows) and rows[i].t >= bin_start + dt.timedelta(seconds=interval_s):
            next_ts = rows[i].t
            bin_start = dt.datetime.fromtimestamp((next_ts.timestamp() // interval_s) * interval_s, tz=dt.timezone.utc)

    msgs: List[str] = []
    ddhhmm = (start + dt.timedelta(seconds=interval_s // 2)).strftime("%d%H%M") if out_lines else dt.datetime.now(dt.timezone.utc).strftime("%d%H%M")
    for obnum, i in enumerate(range(0, len(out_lines), lines_per_msg), start=1):
        lines = out_lines[i:i + lines_per_msg]
        header = f"{wmo_header} {header_center} {ddhhmm}"
        mission_line = f"{mission} HDOB {obnum:02d} {storm_date.strftime('%Y%m%d')}"
        msg = "\n".join([header, mission_line] + lines + ["$$"])
        msgs.append(msg)
    return "\n\n".join(msgs) + ("\n" if msgs else "")

# ----------------------------- I/O and multithreaded parsing -----------------------------
def read_iwg1(path: Optional[str], url: Optional[str], workers: int = 4) -> List[IWG1Row]:
    """
    Read IWG1 rows from path or URL and parse into IWG1Row objects using a thread pool.
    workers: number of threads to use (>=1). If 1 -> single-threaded.
    """
    if path:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif url:
        if requests is None:
            raise RuntimeError("'requests' is required to read from URL; install it or use --path.")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        text = resp.text
    else:
        raise ValueError("Provide --path or --url")

    parts_list = []
    for parts in iwg1_iter_lines_from_text(text):
        parts_list.append(parts)

    if not parts_list:
        return []

    # If workers == 1, do simple serial parse (avoid thread overhead)
    if workers is None or workers <= 1:
        rows = [parse_iwg1_row(p) for p in parts_list]
    else:
        # cap workers to reasonable number
        max_workers = max(1, min(workers, (os.cpu_count() or 4) * 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            rows = list(ex.map(parse_iwg1_row, parts_list))
    # filter Nones
    rows = [r for r in rows if r is not None]
    return rows

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
    ap.add_argument("--storm-date", required=False, help="YYYYMMDD date for mission line; default from first record UTC date")
    ap.add_argument("--interval", type=int, choices=(10,30,60,120), default=30, help="HDOB time resolution (s)")
    ap.add_argument("--lines-per-message", type=int, default=20, help="Number of lines per HDOB message")
    ap.add_argument("--out", help="Output file path for HDOB text; default prints to stdout")
    ap.add_argument("--start", help="UTC start time-of-day (HH:MM or HHMM or HH:MM:SS or HHMMSS)", default=None)
    ap.add_argument("--end", help="UTC end time-of-day (HH:MM or HHMM or HH:MM:SS or HHMMSS)", default=None)
    ap.add_argument("--workers", type=int, default=4, help="Number of worker threads for parsing (default: 4)")
    args = ap.parse_args(argv)

    rows = read_iwg1(args.path, args.url, workers=args.workers)
    if not rows:
        print("No IWG1 rows parsed.", file=sys.stderr)
        return 2

    # Validate start/end
    try:
        start_sec = _time_input_to_seconds(args.start) if args.start else None
    except ValueError as e:
        print(f"Invalid --start value: {e}", file=sys.stderr)
        return 4
    try:
        end_sec = _time_input_to_seconds(args.end) if args.end else None
    except ValueError as e:
        print(f"Invalid --end value: {e}", file=sys.stderr)
        return 5

    print(f"Read {len(rows)} rows from source (after parsing).")
    if start_sec is not None or end_sec is not None:
        rows_filtered = _filter_rows_by_time_of_day(rows, start_sec, end_sec)
        print(f"{len(rows_filtered)} rows remain after applying time window (start={args.start}, end={args.end}).")
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
