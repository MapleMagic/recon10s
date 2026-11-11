"""
HRD's real-time TDR analyses (seb.omao.noaa.gov/pub/flight/radar/<mission>/*_xy.nc.gz) for given
years, reduced like TC-RADAR: re-centred on the 2-km circulation, motion-relative tangential and
radial winds in 2-km x 11.25-deg polar bins at 0-14 km. The real-time U and V are earth-relative
(checked against TC-RADAR); the motion is the analysis file's own.

Usage: python fetch_realtime.py OUTDIR YEAR [YEAR ...]   (downloads go to a scratch folder and are deleted)
"""
import concurrent.futures as cf
import datetime as dt
import gzip
import json
import os
import re
import sys
import tempfile
import time

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import recon10s_tdrgrid as tg  # noqa: E402

BASE = "https://seb.omao.noaa.gov/pub/flight/radar/"
NR, NPHI, DR, ZMAX = 100, 32, 2.0, 14.0
UTC = dt.timezone.utc


def get(url, tries=6, timeout=120):
    for k in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException:
            time.sleep(2 + 3 * k)
    raise IOError(url)


def missions(year):
    r = get(BASE)
    return sorted(set(m for m in re.findall(r'href="(\d{8}[A-Z]\d)/"', r.text) if m.startswith(str(year))))


def analyses(mission):
    r = get(BASE + mission + "/")
    if r is None:
        return []
    return sorted(set(re.findall(r'href="(\d{6}[A-Z]\d_\d{4}_xy\.nc\.gz)"', r.text)))


def recentre(x, y, u, v, r_max=25.0):
    """2-km circulation centre: maximise the mean tangential wind over 0.5-1.5 RMW (coarse, then fine)."""
    X, Y = np.meshgrid(x, y)
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 200:
        return None
    xs, ys, us, vs = X[ok], Y[ok], u[ok], v[ok]

    def prof_rmw(x0, y0):
        dx, dy = xs - x0, ys - y0
        r = np.hypot(dx, dy)
        vt = (-us * dy + vs * dx) / np.maximum(r, 1e-6)
        rb = np.floor(r / 4.0).astype(int)
        nb = 40
        sel = rb < nb
        s = np.bincount(rb[sel], vt[sel], nb)
        n = np.bincount(rb[sel], minlength=nb)
        p = np.where(n > 10, s / np.maximum(n, 1), np.nan)
        if not np.isfinite(p).any():
            return 40.0
        return (np.nanargmax(p) + 0.5) * 4.0

    R = prof_rmw(0.0, 0.0)

    def score(x0, y0):
        dx, dy = xs - x0, ys - y0
        r = np.hypot(dx, dy)
        sel = (r > 0.5 * R) & (r < 1.5 * R)
        if sel.sum() < 50:
            return -np.inf
        vt = (-us[sel] * dy[sel] + vs[sel] * dx[sel]) / r[sel]
        # mean over azimuth sectors so that an unevenly sampled ring is not biased
        az = np.arctan2(dx[sel], dy[sel])
        sec = ((az + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        m = [vt[sec == k].mean() for k in range(8) if (sec == k).sum() >= 5]
        return np.mean(m) if len(m) >= 3 else -np.inf

    best = (0.0, 0.0, score(0.0, 0.0))
    for step, span in ((2.0, r_max), (0.5, 3.0)):
        cx, cy = best[0], best[1]
        for x0 in np.arange(cx - span, cx + span + 1e-6, step):
            for y0 in np.arange(cy - span, cy + span + 1e-6, step):
                if np.hypot(x0, y0) > r_max:
                    continue
                s = score(x0, y0)
                if s > best[2]:
                    best = (x0, y0, s)
    return best[0], best[1], R


def reduce_file(data):
    nc = tg._open(data)
    v = nc.variables
    a = nc._attributes

    def att(k, d=None):
        val = a.get(k, d)
        return val.decode() if isinstance(val, bytes) else val
    x = np.array(v["x"][:], float)
    y = np.array(v["y"][:], float)
    lev = np.array(v["level"][:], float)
    nz = int(np.searchsorted(lev, ZMAX + 1e-6))

    def field(name):
        f = np.array(v[name][..., 0], np.float32)          # (x, y, level)
        f[f < -900] = np.nan
        return np.transpose(f, (2, 1, 0))[:nz]              # (level, y, x)
    U, V = field("U"), field("V")
    mx = float(att("EASTWARD STORM MOTION (METERS PER SECOND)", 0.0))
    my = float(att("NORTHWARD STORM MOTION (METERS PER SECOND)", 0.0))
    Ur, Vr_ = U - mx, V - my
    k2 = int(np.argmin(np.abs(lev[:nz] - 2.0)))
    c = recentre(x, y, Ur[k2], Vr_[k2])
    if c is None:
        return None
    x0, y0, R0 = c
    X, Y = np.meshgrid(x - x0, y - y0)
    r = np.hypot(X, Y)
    az = (np.degrees(np.arctan2(X, Y)) + 360.0) % 360.0
    ir = np.floor(r / DR).astype(int)
    ip = np.floor(az / (360.0 / NPHI)).astype(int) % NPHI
    keep = ir < NR
    idx = (ir * NPHI + ip)[keep]
    sn, cs = X / np.maximum(r, 1e-6), Y / np.maximum(r, 1e-6)     # sin, cos of geographic azimuth
    vt_all = np.full((nz, NR * NPHI), np.nan, np.float32)
    vr_all = np.full((nz, NR * NPHI), np.nan, np.float32)
    cnt = np.zeros((nz, NR * NPHI), np.uint16)
    for k in range(nz):
        # tangential (cyclonic positive) and radial (outward positive) about the new centre:
        # e_r = (sin az, cos az), e_t = (-cos az, sin az)
        vt = -Ur[k] * cs + Vr_[k] * sn
        vrad = Ur[k] * sn + Vr_[k] * cs
        good = (np.isfinite(vt) & np.isfinite(vrad))[keep]
        tv, rv = np.where(good, vt[keep], 0.0), np.where(good, vrad[keep], 0.0)
        n = np.bincount(idx, weights=good, minlength=NR * NPHI)
        with np.errstate(invalid="ignore", divide="ignore"):
            vt_all[k] = np.where(n > 0, np.bincount(idx, weights=tv, minlength=NR * NPHI) / np.maximum(n, 1), np.nan)
            vr_all[k] = np.where(n > 0, np.bincount(idx, weights=rv, minlength=NR * NPHI) / np.maximum(n, 1), np.nan)
        cnt[k] = n
    yr, mo, dy = int(att("FLIGHT YEAR")), int(att("FLIGHT MONTH")), int(att("FLIGHT DAY"))
    day = dt.datetime(yr, mo, dy, tzinfo=UTC)

    def at(prefix):
        h, m, s = (int(att(f"ANALYSIS {prefix} HOUR")), int(att(f"ANALYSIS {prefix} MINUTE")),
                   int(att(f"ANALYSIS {prefix} SECOND")))
        return day + dt.timedelta(hours=h, minutes=m, seconds=s)
    t_start, t_end, t_c = at("START"), at("END"), at("CENTER")
    # analyses that run past midnight: times earlier than the start belong to the next day
    if t_end < t_start:
        t_end += dt.timedelta(days=1)
    if t_c < t_start:
        t_c += dt.timedelta(days=1)
    lat0, lon0 = float(att("ORIGIN_LATITUDE")), float(att("ORIGIN_LONGITUDE"))
    clat = lat0 + y0 / 111.2
    clon = lon0 + x0 / (111.2 * np.cos(np.radians(lat0)))
    md = {"storm": att("STMNAME", ""), "flight": att("FLTNAME", ""), "start": t_start.timestamp(),
          "end": t_end.timestamp(), "centre_time": t_c.timestamp(), "origin": [lat0, lon0], "offset_km": [x0, y0],
          "centre": [clat, clon], "motion": [mx, my], "rmw_guess": R0}
    return (vt_all.reshape(nz, NR, NPHI).astype(np.float16), vr_all.reshape(nz, NR, NPHI).astype(np.float16),
            np.minimum(cnt, 255).astype(np.uint8).reshape(nz, NR, NPHI), lev[:nz], md)


def one(mission, name, outdir, scratch):
    out = os.path.join(outdir, name.replace(".nc.gz", ".npz"))
    if os.path.exists(out) or os.path.exists(out + ".bad"):
        return name, "cached"
    r = get(BASE + mission + "/" + name)
    if r is None:
        return name, "missing"
    raw = r.content
    data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    try:
        res = reduce_file(data)
    except Exception as exc:
        open(out + ".bad", "w").write(str(exc))
        return name, f"bad: {exc}"
    if res is None:
        open(out + ".bad", "w").write("no centre")
        return name, "no centre"
    vt, vr, cnt, lev, md = res
    md["mission"] = mission
    md["name"] = name
    np.savez_compressed(out, vt=vt, vr=vr, count=cnt, z=lev)
    with open(out.replace(".npz", ".json"), "w") as fh:
        json.dump(md, fh)
    return name, f"ok offset {md['offset_km'][0]:+.0f},{md['offset_km'][1]:+.0f} km"


if __name__ == "__main__":
    outdir = sys.argv[1]
    years = sys.argv[2:]
    os.makedirs(outdir, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="seb_", dir=os.path.dirname(os.path.abspath(outdir)))
    jobs = []
    for yv in years:
        for m in missions(yv):
            for name in analyses(m):
                jobs.append((m, name))
        print(f"{yv}: {len(jobs)} analyses listed so far", flush=True)
    t = time.time()
    with cf.ThreadPoolExecutor(2) as ex:
        futs = [ex.submit(one, m, n, outdir, scratch) for m, n in jobs]
        for k, f in enumerate(cf.as_completed(futs), 1):
            try:
                name, msg = f.result()
            except Exception as exc:
                name, msg = "?", f"FAILED {exc}"
            if k % 10 == 0 or "bad" in msg or "FAIL" in msg:
                print(f"[{k}/{len(jobs)}] {name}: {msg} ({time.time() - t:.0f}s)", flush=True)
