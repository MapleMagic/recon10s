"""
Reduce TC-RADAR v3m (HRD/AOML) to compact storm-centred polar grids, reading only what is needed
over HTTP range requests (the files are 6-11 GB each).

For every swath and merged analysis: the motion-relative tangential and radial winds about the
2-km centre ("recentered_*"), averaged into 2-km radial x 11.25-degree azimuthal bins (geographic
azimuth, clockwise from north) at heights 0-14 km, with the number of valid 2-km cells in each
bin; plus metadata (times, mission, storm, RMW by height, SHIPS at lag 0).

Usage: python extract_tcradar.py OUTDIR  (restartable: finished groups are skipped)
"""
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rangefile import RangeFile  # noqa: E402

BASE = "https://www.aoml.noaa.gov/ftp/pub/hrd/data/radar/level3/"
FILES = [("merge", "tc_radar_v3m_1997_2019_xy_rel_merge_ships.nc"),
         ("merge", "tc_radar_v3m_2020_2024_xy_rel_merge_ships.nc"),
         ("swath", "tc_radar_v3m_1997_2019_xy_rel_swath_ships.nc"),
         ("swath", "tc_radar_v3m_2020_2024_xy_rel_swath_ships.nc")]
NR, NPHI = 100, 32
DR = 2.0
ZMAX = 14.0
SHIPS_VARS = ("vmax", "pres", "motion_x", "motion_y", "shdc", "sddc", "shrd", "shtd", "shrs", "shts", "lat", "lon",
              "rhmd", "sst", "mpi", "dtl")


def _text(a):
    out = []
    for v in a:
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        out.append(str(v).strip())
    return out


def polar_index(x, y):
    """For the (y, x) grid: radial bin, azimuth bin (geographic) and validity."""
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X, Y)
    az = (np.degrees(np.arctan2(X, Y)) + 360.0) % 360.0
    ir = np.floor(r / DR).astype(int)
    ip = np.floor(az / (360.0 / NPHI)).astype(int) % NPHI
    ok = ir < NR
    flat = np.where(ok, ir * NPHI + ip, -1)
    return flat


def reduce_group(data, flat):
    """data: (n, ny, nx, nz) -> means (n, nz, NR, NPHI) float16 and counts uint8."""
    n, ny, nx, nz = data.shape
    keep = flat.ravel() >= 0
    idx = flat.ravel()[keep]
    means = np.full((n, nz, NR * NPHI), np.nan, np.float32)
    counts = np.zeros((n, nz, NR * NPHI), np.uint16)
    for i in range(n):
        block = data[i].reshape(ny * nx, nz)[keep]           # (cells, nz)
        good = np.isfinite(block)
        vals = np.where(good, block, 0.0)
        for k in range(nz):
            c = np.bincount(idx, weights=good[:, k], minlength=NR * NPHI)
            s = np.bincount(idx, weights=vals[:, k], minlength=NR * NPHI)
            with np.errstate(invalid="ignore", divide="ignore"):
                means[i, k] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            counts[i, k] = c
    return (means.reshape(n, nz, NR, NPHI).astype(np.float16),
            np.minimum(counts, 255).astype(np.uint8).reshape(n, nz, NR, NPHI))


def metadata(f, kind, i0, i1, lag0):
    md = {"mission": _text(f["mission_ID"][i0:i1]), "storm": _text(f["storm_name"][i0:i1])}
    if kind == "merge":
        for k in ("year", "month", "day", "hour", "minute"):
            md[k] = f[f"merge_{k}"][i0:i1].tolist()
        md["number_of_swaths"] = f["number_of_swaths"][i0:i1].tolist()
    else:
        for k, src in (("year", "swath_year"), ("month", "swath_month"), ("day", "swath_day"),
                       ("hour", "swath_hour"), ("minute", "swath_min")):
            md[k] = f[src][i0:i1].tolist()
        for k in ("year", "month", "day", "hour", "min"):
            md[f"beg_{k}"] = f[f"beg_swath_{k}"][i0:i1].tolist()
            md[f"end_{k}"] = f[f"end_swath_{k}"][i0:i1].tolist()
        md["swath_coverage"] = np.asarray(f["swath_coverage"][i0:i1]).tolist() if "swath_coverage" in f else None
    md["tc_rmw"] = np.asarray(f["tc_rmw"][i0:i1]).tolist()
    md["tc_center_lat"] = np.asarray(f["tc_center_latitudes"][i0:i1]).tolist()
    md["tc_center_lon"] = np.asarray(f["tc_center_longitudes"][i0:i1]).tolist()
    md["fix_lat"] = np.asarray(f["original_fix_latitude"][i0:i1]).tolist()
    md["fix_lon"] = np.asarray(f["original_fix_longitude"][i0:i1]).tolist()
    for v in SHIPS_VARS:
        name = f"{v}_ships"
        if name in f:
            md[v] = np.asarray(f[name][i0:i1, lag0], float).tolist()
    md["tcid"] = _text(f["tcid_ships"][i0:i1, lag0]) if "tcid_ships" in f else None
    return md


def main(out):
    os.makedirs(out, exist_ok=True)
    for kind, name in FILES:
        t0 = time.time()
        rf = RangeFile(BASE + name, block=4 << 20, max_blocks=48)
        f = h5py.File(rf, "r")
        n = f["num_cases"].shape[0]
        z = np.asarray(f["height"][:], float)
        nz = int(np.searchsorted(z, ZMAX + 1e-6))
        x = np.asarray(f["eastward_distance"][:], float)
        y = np.asarray(f["northward_distance"][:], float)
        lags = np.asarray(f["ships_lag_times"][:], float)
        lag0 = int(np.argmin(np.abs(lags)))
        flat = polar_index(x, y)
        step = f["recentered_tangential_wind"].chunks[0]
        stem = name.replace(".nc", "")
        with open(os.path.join(out, stem + "_grid.json"), "w") as fh:
            json.dump({"kind": kind, "n": n, "z": z[:nz].tolist(), "nr": NR, "nphi": NPHI, "dr": DR,
                       "x_range": [float(x[0]), float(x[-1])], "lag0": float(lags[lag0])}, fh)
        print(f"{name}: {n} cases, levels to {z[nz - 1]} km, groups of {step}", flush=True)
        for i0 in range(0, n, step):
            i1 = min(n, i0 + step)
            path = os.path.join(out, f"{stem}_{i0:04d}.npz")
            if os.path.exists(path):
                continue
            t = time.time()
            res = {}
            for var, key in (("recentered_tangential_wind", "vt"), ("recentered_radial_wind", "vr")):
                data = np.asarray(f[var][i0:i1, :, :, :nz], np.float32)
                data[data < -900] = np.nan
                m, c = reduce_group(data, flat)
                res[key] = m
                if key == "vt":
                    res["count"] = c
                del data
            md = metadata(f, kind, i0, i1, lag0)
            np.savez_compressed(path + ".part.npz", **res)
            os.replace(path + ".part.npz", path)
            with open(path.replace(".npz", ".json"), "w") as fh:
                json.dump(md, fh)
            print(f"  {i0}-{i1}: {time.time() - t:.0f}s, fetched {rf.fetched / 1e6:.0f} MB so far", flush=True)
        f.close()
        print(f"{name} done in {time.time() - t0:.0f}s, {rf.fetched / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
