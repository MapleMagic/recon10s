"""
Fetch the in-situ flight-level data (IWG1, 1 Hz) of NOAA P-3 TDR missions for the times of their
TDR passes only, by HTTP range requests with a binary search on the timestamps, and reduce it to
10-s means. One output file per mission: data/iwg1/<mission>.npz.

Usage: python fetch_iwg1.py windows.json OUTDIR [workers]
windows.json: {"20261008I1": [[t0_epoch, t1_epoch], ...], ...}
"""
import concurrent.futures as cf
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import recon10s as R  # noqa: E402

BASE = "https://seb.omao.noaa.gov/pub/flight/aamps_ingest/iwg1/{m}/{m}_iwg1.txt"
UTC = dt.timezone.utc


def _get(s, url, a, b, tries=6):
    for k in range(tries):
        try:
            r = s.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=90)
            if r.status_code in (200, 206):
                return r.content
            if r.status_code in (404, 416):
                return None
        except requests.RequestException:
            pass
        time.sleep(2 + 3 * k)
    raise IOError(f"{url} {a}-{b} failed")


def _first_time(chunk: bytes, skip_partial=True):
    text = chunk.decode("ascii", "replace")
    lines = text.split("\n")
    if skip_partial:
        lines = lines[1:]
    for ln in lines[:-1]:
        row = R.parse_iwg1_line(ln)
        if row is not None:
            return row.t.timestamp()
    return None


def size_of(s, url):
    for k in range(6):
        try:
            h = s.head(url, timeout=60)
            if h.status_code == 404:
                return None
            h.raise_for_status()
            return int(h.headers["Content-Length"])
        except requests.RequestException:
            time.sleep(2 + 3 * k)
    return None


def find_offset(s, url, size, t_target):
    """Largest offset (line-aligned approx.) whose first full line is at or before t_target."""
    lo, hi = 0, size
    while hi - lo > 65536:
        mid = (lo + hi) // 2
        c = _get(s, url, mid, min(size - 1, mid + 16383))
        t = _first_time(c) if c else None
        if t is None:
            hi = mid
        elif t <= t_target:
            lo = mid
        else:
            hi = mid
    return lo


def read_window(s, url, size, t0, t1):
    a = find_offset(s, url, size, t0)
    rows = []
    pos = a
    step = 4 << 20
    first = True
    while pos < size:
        c = _get(s, url, pos, min(size - 1, pos + step - 1))
        if not c:
            break
        text = c.decode("ascii", "replace")
        cut = text.rfind("\n")
        if cut < 0:
            break
        body = text[:cut]
        lines = body.split("\n")
        if first and pos > 0:
            lines = lines[1:]
        first = False
        done = False
        for ln in lines:
            row = R.parse_iwg1_line(ln)
            if row is None:
                continue
            ts = row.t.timestamp()
            if ts < t0:
                continue
            if ts > t1:
                done = True
                break
            rows.append((ts, row.lat, row.lon, row.ga_m, row.ps_hpa, row.wspd_ms, row.wdir_deg, row.temp_c))
        if done:
            break
        pos += len(body.encode("ascii", "replace")) + 1
    return rows


def reduce10(rows):
    if not rows:
        return None
    a = np.array([[np.nan if v is None else v for v in r] for r in rows], float)
    t, lat, lon, z, p, ws, wd, tc = a.T
    u = -ws * np.sin(np.radians(wd))
    v = -ws * np.cos(np.radians(wd))
    b = np.floor(t / 10.0)
    keys, inv = np.unique(b, return_inverse=True)
    out = {}
    for name, arr in (("t", t), ("lat", lat), ("lon", lon), ("z", z), ("p", p), ("u", u), ("v", v), ("temp", tc)):
        s = np.bincount(inv, weights=np.where(np.isfinite(arr), arr, 0.0), minlength=keys.size)
        n = np.bincount(inv, weights=np.isfinite(arr).astype(float), minlength=keys.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(n > 0, s / n, np.nan)
    mx = np.full(keys.size, np.nan)
    np.fmax.at(mx, inv, ws)
    out["wspd_max1s"] = mx
    out["n"] = np.bincount(inv, minlength=keys.size)
    return out


def one(mission, windows, outdir):
    path = os.path.join(outdir, f"{mission}.npz")
    if os.path.exists(path) or os.path.exists(path + ".missing"):
        return mission, "cached"
    s = requests.Session()
    url = BASE.format(m=mission)
    size = size_of(s, url)
    if size is None:
        open(path + ".missing", "w").close()
        return mission, "no IWG1"
    rows = []
    for t0, t1 in windows:
        rows += read_window(s, url, size, t0, t1)
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda r: r[0]):
        if r[0] not in seen:
            seen.add(r[0])
            uniq.append(r)
    red = reduce10(uniq)
    if red is None:
        open(path + ".missing", "w").close()
        return mission, "empty"
    np.savez_compressed(path, **red)
    return mission, f"{red['t'].size} records"


def merge_windows(ws, pad=600):
    ws = sorted((a - pad, b + pad) for a, b in ws)
    out = []
    for a, b in ws:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


if __name__ == "__main__":
    windows = json.load(open(sys.argv[1]))
    outdir = sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(outdir, exist_ok=True)
    t = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, m, merge_windows(w), outdir): m for m, w in windows.items()}
        for k, f in enumerate(cf.as_completed(futs), 1):
            try:
                m, msg = f.result()
            except Exception as exc:
                m, msg = futs[f], f"FAILED {exc}"
            print(f"[{k}/{len(futs)}] {m}: {msg} ({time.time() - t:.0f}s)", flush=True)
