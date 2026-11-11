"""
Reduce TC PRIMED overpass files to what the satellite-intensity model learns from, reading only
the needed variables over HTTP range requests:

  * IR (10.3-um window, CIRA/RAMMB TC IR archive) at the overpass time: a 128 x 128 grid, 4 km apart, centred on the
    interpolated best-track centre (uint8: Tb = 163 K + 0.6 K * value, 255 = missing).
  * Imagers (SSM/I, SSMIS, TMI, AMSR-E, AMSR2, GMI): 37 and 85-92 GHz V and H, gridded to a
    64 x 64 grid 8 km apart around the centre at the pass time (uint8: Tb = 100 K + 0.85 K *
    value, 255 = missing), and how much of the inner 200 km they cover.
  * The storm metadata interpolated to the pass: best-track intensity and pressure, position,
    motion, distance to land, development level.

Usage: python fetch_tcprimed.py INDEX.json OUTDIR [workers] [year0] [year1]
One npz + json per storm in OUTDIR (restartable: finished storms are skipped).
"""
import concurrent.futures as cf
import json
import math
import os
import re
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = "https://noaa-nesdis-tcprimed-pds.s3.amazonaws.com/"
IMAGERS = {"SSMI", "SSMIS", "TMI", "AMSRE", "AMSR2", "GMI"}
IR_N, IR_DX = 128, 4.0
MW_N, MW_DX = 64, 8.0
EARTH_KM = 6371.0088
F37 = re.compile(r"^TB_(36\.5|36\.64|37\.0)([VH])$")
F89 = re.compile(r"^TB_(A89\.0|89\.0|85\.5|91\.665)([VH])$")


def grid(n, dx):
    return (np.arange(n) - (n - 1) / 2.0) * dx


def destinations(lat0, lon0, x, y):
    """lat/lon of points x km east, y km north of (lat0, lon0) (azimuthal equidistant)."""
    d = np.hypot(x, y) / EARTH_KM
    b = np.arctan2(x, y)
    la1, lo1 = math.radians(lat0), math.radians(lon0)
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(b))
    lo2 = lo1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(la1), np.cos(d) - np.sin(la1) * np.sin(la2))
    return np.degrees(la2), np.degrees(lo2)


def local_xy(lat0, lon0, lat, lon):
    la0, lo0 = math.radians(lat0), math.radians(lon0)
    la, lo = np.radians(lat), np.radians(lon)
    dlo = lo - lo0
    c = np.sin(la0) * np.sin(la) + np.cos(la0) * np.cos(la) * np.cos(dlo)
    rho = np.arccos(np.clip(c, -1, 1))
    k = np.where(rho > 1e-12, rho / np.maximum(np.sin(rho), 1e-12), 1.0)
    x = EARTH_KM * k * np.cos(la) * np.sin(dlo)
    y = EARTH_KM * k * (np.cos(la0) * np.sin(la) - np.sin(la0) * np.cos(la) * np.cos(dlo))
    return x, y


def encode(tb, lo, step):
    q = np.round((tb - lo) / step)
    q = np.where(np.isfinite(q), np.clip(q, 0, 254), 255)
    return q.astype(np.uint8)


def regrid_ir(f):
    ir = f["infrared"]
    if "infrared_availability_flag" in ir and int(ir["infrared_availability_flag"][0]) == 0:
        return None
    tb = ir["IRWIN"][:].astype(np.float64)
    tb[(tb < 150) | (tb > 340)] = np.nan
    lat = ir["latitude"][:]
    lon = ir["longitude"][:]
    lat0 = float(ir["storm_latitude"][0])
    lon0 = float(ir["storm_longitude"][0])
    lat_v = lat[:, 0].astype(np.float64)
    lon_v = ((lon[0, :].astype(np.float64) - lon0 + 180.0) % 360.0) - 180.0
    g = grid(IR_N, IR_DX)
    X, Y = np.meshgrid(g, g)
    la, lo = destinations(lat0, 0.0, X, Y)          # lon relative to the centre
    if lat_v[0] > lat_v[-1]:
        lat_v, tb = lat_v[::-1], tb[::-1]
    if lon_v[0] > lon_v[-1]:
        lon_v, tb = lon_v[::-1], tb[:, ::-1]
    rows = np.interp(la, lat_v, np.arange(lat_v.size), left=np.nan, right=np.nan)
    cols = np.interp(lo, lon_v, np.arange(lon_v.size), left=np.nan, right=np.nan)
    from scipy.ndimage import map_coordinates
    ok = np.isfinite(rows) & np.isfinite(cols)
    valid = np.isfinite(tb).astype(np.float64)
    filled = np.where(np.isfinite(tb), tb, 0.0)
    rr, cc = np.where(ok, rows, 0), np.where(ok, cols, 0)
    num = map_coordinates(filled, [rr, cc], order=1, mode="nearest")
    den = map_coordinates(valid, [rr, cc], order=1, mode="nearest")
    out = np.where(ok & (den > 0.5), num / np.maximum(den, 1e-6), np.nan)
    return {"ir": encode(out, 163.0, 0.6), "ir_t": int(ir["time"][0]), "ir_c": (lat0, lon0)}


def regrid_mw(f, lat0, lon0):
    from scipy.spatial import cKDTree
    pm = f["passive_microwave"]
    found = {}
    b89 = {}
    for gname in pm:
        grp = pm[gname]
        if not hasattr(grp, "keys"):
            continue
        for v in grp:
            if v in ("TB_B89.0V", "TB_B89.0H"):
                b89["89" + v[-1]] = (gname, v)
                continue
            for tag, rx in (("37", F37), ("89", F89)):
                m = rx.match(v)
                if m and (tag + m.group(2)) not in found:
                    found[tag + m.group(2)] = (gname, v)
    # AMSR-E lost its 89-GHz A horn in 2004: use the B scan when the A scan is empty
    if b89 and "89V" in found:
        gname, v = found["89V"]
        a = pm[gname][v][:]
        if not np.isfinite(np.where((a > 50) & (a < 340), a, np.nan)).any():
            found.update(b89)
    for k in ("89V", "89H"):
        if k not in found and k in b89:
            found[k] = b89[k]
    if not all(k in found for k in ("37V", "37H", "89V", "89H")):
        return None
    g = grid(MW_N, MW_DX)
    X, Y = np.meshgrid(g, g)
    targets = np.c_[X.ravel(), Y.ravel()]
    out = np.full((4, MW_N, MW_N), 255, np.uint8)
    cover = 0.0
    cache = {}
    for i, key in enumerate(("37V", "37H", "89V", "89H")):
        gname, v = found[key]
        if gname not in cache:
            lat = pm[gname]["latitude"][:].astype(np.float64)
            lon = pm[gname]["longitude"][:].astype(np.float64)
            x, y = local_xy(lat0, lon0, lat, lon)
            near = (np.abs(x) < 300) & (np.abs(y) < 300) & np.isfinite(x) & np.isfinite(y)
            cache[gname] = (near, cKDTree(np.c_[x[near], y[near]]) if near.any() else None)
        near, tree = cache[gname]
        if tree is None:
            return None
        tb = pm[gname][v][:].astype(np.float64)[near]
        tb[(tb < 50) | (tb > 340)] = np.nan
        radius = 7.0 if key.startswith("89") else 11.0
        k = min(24, tree.n)
        d, ii = tree.query(targets, k=k, distance_upper_bound=radius)
        d, ii = d.reshape(len(targets), -1), ii.reshape(len(targets), -1)
        hit = ii < tree.n
        t = np.where(hit, tb[np.minimum(ii, tree.n - 1)], np.nan)
        w = np.where(hit & np.isfinite(t), np.exp(-d ** 2 / (2 * (radius / 2) ** 2)), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = np.where(w.sum(1) > 0, np.nansum(w * np.nan_to_num(t), 1) / w.sum(1), np.nan)
        out[i] = encode(vals.reshape(MW_N, MW_N), 100.0, 0.85)
        if key == "89H":
            inner = (np.hypot(X, Y) <= 200).ravel()
            cover = float(np.isfinite(vals[inner]).mean())
    return {"mw": out, "mw_cov": cover, "mw_t": int(f["passive_microwave/time"][0]) if "time" in pm else None,
            "mw_freq": [found[k][1] for k in ("37V", "89V")]}


def metadata(f):
    sm = f["overpass_storm_metadata"]
    out = {}
    for k in ("intensity", "central_min_pressure", "storm_latitude", "storm_longitude", "storm_speed",
              "storm_heading", "distance_to_land", "time"):
        if k in sm:
            v = sm[k][()]
            v = np.asarray(v).ravel()
            out[k] = float(v[0]) if v.size else None
    if "development_level" in sm:
        dl = np.asarray(sm["development_level"][()]).ravel()
        out["development_level"] = (dl[0].decode() if isinstance(dl[0], bytes) else str(dl[0])) if dl.size else ""
    if "intensity_change" in sm:
        out["intensity_change"] = [int(x) for x in sm["intensity_change"][:]]
    return out


def one(row):
    import h5py
    from s3range import RangeFile
    t0 = time.time()
    try:
        rf = RangeFile(BASE + row["key"], size=row["size"])
        with h5py.File(rf, "r") as f:
            md = metadata(f)
            res = {"meta": md, "row": row}
            ir = regrid_ir(f) if "infrared" in f else None
            if ir:
                res.update(ir)
            if row["sensor"] in IMAGERS and "passive_microwave" in f and md.get("storm_latitude") is not None:
                mw = regrid_mw(f, md["storm_latitude"], md["storm_longitude"])
                if mw:
                    res.update(mw)
        res["bytes"] = rf.fetched
        res["secs"] = time.time() - t0
        return res
    except Exception as exc:
        return {"row": row, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=2)}


def write_storm(outdir, storm, results):
    results = [r for r in results if "error" not in r]
    results.sort(key=lambda r: r["row"]["time"])
    irs = [r for r in results if "ir" in r]
    mws = [r for r in results if "mw" in r]
    meta = []
    for r in results:
        meta.append({"key": r["row"]["key"].rsplit("/", 1)[-1], "sensor": r["row"]["sensor"],
                     "platform": r["row"]["platform"], "time": r["row"]["time"], **r["meta"],
                     "ir": irs.index(r) if "ir" in r else -1, "ir_t": r.get("ir_t"), "ir_c": r.get("ir_c"),
                     "mw": mws.index(r) if "mw" in r else -1, "mw_t": r.get("mw_t"), "mw_cov": r.get("mw_cov"),
                     "mw_freq": r.get("mw_freq")})
    np.savez_compressed(os.path.join(outdir, storm + ".npz"),
                        ir=np.array([r["ir"] for r in irs], np.uint8).reshape(-1, IR_N, IR_N),
                        mw=np.array([r["mw"] for r in mws], np.uint8).reshape(-1, 4, MW_N, MW_N))
    with open(os.path.join(outdir, storm + ".json"), "w") as fh:
        json.dump(meta, fh)


def main(index, outdir, workers=16, y0=2000, y1=2025):
    os.makedirs(outdir, exist_ok=True)
    rows = json.load(open(index))
    # one copy of each pass: the final version where both exist
    best = {}
    for r in rows:
        if not (y0 <= int(r["storm"][-4:]) <= y1):
            continue
        k = (r["storm"], r["sensor"], r["platform"], int(r["time"]))
        if k not in best or r["kind"] == "final":
            best[k] = r
    by_storm = {}
    for r in best.values():
        by_storm.setdefault(r["storm"], []).append(r)
    todo = sorted((s for s in by_storm if not os.path.exists(os.path.join(outdir, s + ".npz"))),
                  key=lambda s: (-int(s[-4:]), s))
    print(f"{len(by_storm)} storms, {len(todo)} to do, {sum(len(by_storm[s]) for s in todo)} files", flush=True)
    t0 = time.time()
    done_files = 0
    nbytes = 0
    errors = 0
    pending = {}
    with cf.ProcessPoolExecutor(workers) as ex:
        futs = {}
        for s in todo:
            pending[s] = []
            for r in by_storm[s]:
                futs[ex.submit(one, r)] = s
        left = {s: len(by_storm[s]) for s in todo}
        for fut in cf.as_completed(futs):
            s = futs[fut]
            res = fut.result()
            pending[s].append(res)
            done_files += 1
            if "error" in res:
                errors += 1
                if errors <= 20 or errors % 100 == 0:
                    print("ERROR", res["row"]["key"].rsplit("/", 1)[-1], res["error"], flush=True)
            else:
                nbytes += res.get("bytes", 0)
            left[s] -= 1
            if left[s] == 0:
                write_storm(outdir, s, pending.pop(s))
            if done_files % 500 == 0:
                el = time.time() - t0
                print(f"{done_files} files, {nbytes / 1e9:.1f} GB, {errors} errors, {el / 60:.0f} min, "
                      f"{done_files / el:.1f} files/s", flush=True)
    print("done", done_files, "files", errors, "errors")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]) if len(a) > 2 else 16, int(a[3]) if len(a) > 3 else 2000,
         int(a[4]) if len(a) > 4 else 2025)
