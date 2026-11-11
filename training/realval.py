"""
Real-data validation: synthetic TDR from the P-3's own flight-level data (IWG1, as 30-s means)
against the TDR analysis of the same pass (TC-RADAR swaths) or mission (merged).
"""
import datetime as dt
import glob
import json
import math
import os
import warnings

import numpy as np

import synmodel as sm

warnings.simplefilter("ignore")
UTC = dt.timezone.utc
DATA = "data/tcradar"
IWG = "data/iwg1"
KM_PER_DEG = 111.2


def tcradar_cases(kind="swath", min_year=2018):
    """(meta dict, vt, vr, z) for TC-RADAR analyses of the given kind with IWG1 on disk."""
    out = []
    for path in sorted(glob.glob(f"{DATA}/*_{kind}_ships_[0-9][0-9][0-9][0-9].npz")):
        md = json.load(open(path.replace(".npz", ".json")))
        grid = json.load(open(path.rsplit("_", 1)[0] + "_grid.json"))
        d = None
        for i, mis in enumerate(md["mission"]):
            if md["year"][i] < min_year:
                continue
            mission = ("20" + mis) if len(mis) == 8 else mis
            if not os.path.exists(f"{IWG}/{mission}.npz"):
                continue
            if d is None:
                d = np.load(path)
            case = {k: (v[i] if isinstance(v, list) else v) for k, v in md.items()}
            case["mission_dir"] = mission
            case["kind"] = kind
            case["z"] = grid["z"]
            out.append((case, d["vt"][i].astype(np.float32), d["vr"][i].astype(np.float32)))
    return out


def window(case):
    """Seconds since epoch: the pass (swath) or the whole mission (merged: centre +- 3 h)."""
    if "t_start" in case:                         # real-time analyses: their own start and end
        f = lambda s: dt.datetime.fromtimestamp(s, UTC)
        return f(case["t_centre"]), f(case["t_start"]), f(case["t_end"])
    tc = dt.datetime(int(case["year"]), int(case["month"]), int(case["day"]), int(case["hour"]),
                     int(case["minute"]), tzinfo=UTC)
    if case["kind"] == "swath":
        def around(hh, mm, before):
            t = tc.replace(hour=int(hh), minute=int(mm))
            if before:
                while t > tc:
                    t -= dt.timedelta(days=1)
            else:
                while t < tc:
                    t += dt.timedelta(days=1)
            return t
        return tc, around(case["beg_hour"], case["beg_min"], True), around(case["end_hour"], case["end_min"], False)
    return tc, tc - dt.timedelta(hours=3), tc + dt.timedelta(hours=3)


def iwg1_obs(case, every_s=30.0, max_r=160.0, sigma=2.0, centre=None):
    """Flight-level obs (motion-relative) relative to the moving 2-km centre."""
    tc, t0, t1 = window(case)
    if not os.path.exists(f"{IWG}/{case['mission_dir']}.npz"):
        return None, None
    d = np.load(f"{IWG}/{case['mission_dir']}.npz")
    t = d["t"]
    sel = (t >= t0.timestamp()) & (t <= t1.timestamp()) & np.isfinite(d["u"]) & np.isfinite(d["lat"]) & np.isfinite(d["z"])
    if sel.sum() < 30:
        return None, None
    t, lat, lon, z, u, v = (d[k][sel] for k in ("t", "lat", "lon", "z", "u", "v"))
    # 30-s means (HDOB-like)
    b = np.floor((t - t[0]) / every_s).astype(int)
    n = np.bincount(b)
    keep = n > 0

    def mean(a):
        return (np.bincount(b, weights=a) / np.maximum(n, 1))[keep]
    t, lat, lon, z, u, v = (mean(a) for a in (t, lat, lon, z, u, v))
    mx, my = case.get("motion_x"), case.get("motion_y")
    if mx is None or not np.isfinite(mx):
        return None, None
    k2 = 4                                            # 2-km level in the TC-RADAR height list
    if centre is None and "centre_2km" in case:
        clat, clon = case["centre_2km"]
    elif centre is None:
        clat = case["tc_center_lat"][k2]
        clon = case["tc_center_lon"][k2]
    else:
        clat, clon = centre
    if clat is None or not np.isfinite(clat):
        return None, None
    dtc = t - tc.timestamp()
    # centre moves with the storm
    cx_km = mx * dtc / 1000.0
    cy_km = my * dtc / 1000.0
    x = (lon - clon) * KM_PER_DEG * math.cos(math.radians(clat)) - cx_km
    y = (lat - clat) * KM_PER_DEG - cy_km
    r = np.hypot(x, y)
    ok = (r <= max_r) & (z > 300) & (z < 12000)
    if ok.sum() < 20:
        return None, None
    obs = sm.Obs(x=x[ok], y=y[ok], z=z[ok] / 1000.0, u=u[ok] - mx, v=v[ok] - my, sigma=np.full(ok.sum(), sigma),
                 kind=np.zeros(ok.sum(), int))
    earth = np.hypot(u[ok], v[ok])
    return obs, {"fl_max_earth": float(np.max(earth)), "z_med": float(np.median(z[ok])), "n": int(ok.sum()),
                 "motion": (mx, my)}


def heading_of(case):
    return math.degrees(math.atan2(case["motion_x"], case["motion_y"])) % 360.0


def cov_extra(case):
    hd = heading_of(case)
    s, sd = case.get("shdc"), case.get("sddc")
    if s is not None and sd is not None and np.isfinite(s) and np.isfinite(sd):
        rel = math.radians(sd - hd)
        sp, sq = s * math.cos(rel), s * math.sin(rel)
    else:
        sp = sq = 0.0
    lat = abs(case["lat"]) if case.get("lat") is not None and np.isfinite(case["lat"]) else 25.0
    return dict(s_par=sp, s_perp=sq, speed=math.hypot(case["motion_x"], case["motion_y"]), lat=lat)


def score_case(model, fit, vt, vr, z_src, max_r=150.0, levels=None):
    r_src = (np.arange(vt.shape[1]) + 0.5) * 2.0
    az_src = (np.arange(vt.shape[2]) + 0.5) * (360.0 / vt.shape[2])
    Rg, Ag = np.meshgrid(r_src, az_src, indexing="ij")
    sel = Rg <= max_r
    x = (Rg * np.sin(np.radians(Ag)))[sel]
    y = (Rg * np.cos(np.radians(Ag)))[sel]
    st = model.state(fit)
    res = {}
    allt, allr = [], []
    for k, zz in enumerate(z_src):
        if zz < 0.5 or zz > 12.0 or (levels is not None and zz not in levels):
            continue
        tv, rv = vt[k][sel], vr[k][sel]
        ok = np.isfinite(tv) & np.isfinite(rv)
        if ok.sum() < 20:
            continue
        pt = model.sample_points(fit, x[ok], y[ok], np.full(ok.sum(), zz), 0, st)
        pr = model.sample_points(fit, x[ok], y[ok], np.full(ok.sum(), zz), 1, st)
        allt.append(pt - tv[ok]); allr.append(pr - rv[ok])
        res[float(zz)] = (float(np.sqrt(np.mean((pt - tv[ok]) ** 2))), int(ok.sum()),
                          float(np.max(np.hypot(tv[ok], rv[ok]))), float(np.max(np.hypot(pt, pr))))
    if not allt:
        return None
    et, er = np.concatenate(allt), np.concatenate(allr)
    return {"rmse_vt": float(np.sqrt(np.mean(et ** 2))), "rmse_vr": float(np.sqrt(np.mean(er ** 2))),
            "bias_vt": float(et.mean()), "by_z": res}
