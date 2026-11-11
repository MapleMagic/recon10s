"""
The real-time HRD TDR analyses of 2025-2026 (reduced by fetch_seb.py) as training rows and as
validation cases in TC-RADAR's case format: storm id from the best tracks by name and date,
best-track intensity interpolated to the analysis time, SHIPS shear (NHC's lsdiag files, which
exist for the current season only; 2025's are left missing here).

Usage: python realtime_dataset.py SEBDIR BDECKS.json OUT_PREFIX
writes OUT_PREFIX_cases.json (one dict per usable analysis) and OUT_PREFIX_X.npy/.json (normalised).
"""
import datetime as dt
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import dataset as ds  # noqa: E402

UTC = dt.timezone.utc
SKIP = {"TRAIN", "SURV", "TDR", "TEST", "CYCLONE"}


def storm_for(name, when, bdecks):
    """(storm id, track rows) of the named storm whose best track spans `when` (+-1 day)."""
    name = name.strip().upper()
    for sid, d in bdecks.items():
        if name not in [n.upper() for n in d["names"]]:
            continue
        ts = [dt.datetime.fromisoformat(r[0]) for r in d["track"]]
        if ts and ts[0] - dt.timedelta(days=1) <= when <= ts[-1] + dt.timedelta(days=1):
            return sid, d["track"]
    return None, None


def bt_at(track, when):
    rows = [r for r in track if r[3] is not None]
    if len(rows) < 2:
        return None, None, None
    t = np.array([dt.datetime.fromisoformat(r[0]).timestamp() for r in rows])
    order = np.argsort(t)
    t = t[order]
    v = np.array([rows[i][3] for i in order], float)
    la = np.array([rows[i][1] for i in order], float)
    s = when.timestamp()
    if s < t[0] - 3 * 3600 or s > t[-1] + 3 * 3600:
        return None, None, None
    return float(np.interp(s, t, v)), float(np.interp(s, t, la)), (t[0], t[-1])


def main(sebdir, bdeck_path, prefix):
    bdecks = json.load(open(bdeck_path))
    import recon10s_tdr3d as t3
    ships_cache = {}
    cases, rows, meta = [], [], []
    for path in sorted(glob.glob(os.path.join(sebdir, "*_xy.npz"))):
        md = json.load(open(path.replace(".npz", ".json")))
        if md["storm"].upper() in SKIP:
            continue
        tc = dt.datetime.fromtimestamp(md["centre_time"], UTC)
        sid, track = storm_for(md["storm"], tc, bdecks)
        if sid is None:
            print("no best track for", md["storm"], md["name"])
            continue
        vbt, bt_lat, _span = bt_at(track, tc)
        d = np.load(path)
        vt, vr, z = d["vt"].astype(np.float32), d["vr"].astype(np.float32), d["z"]
        vm, R = ds.vm_and_rmw(vt, z)
        if vm is None or vm < 10.0:
            continue
        mx, my = md["motion"]
        if not (np.isfinite(mx) and np.isfinite(my)) or (mx == 0 and my == 0):
            continue
        # SHIPS: lsdiag of the nearest synoptic cycle (current season only on NHC's server)
        cyc = tc.replace(minute=0, second=0, microsecond=0)
        cyc -= dt.timedelta(hours=cyc.hour % 6)
        if tc - cyc > dt.timedelta(hours=3):
            cyc += dt.timedelta(hours=6)
        key = (sid, cyc.isoformat())
        if key not in ships_cache:
            got = None
            if sid.endswith("2026"):
                for _ in range(3):
                    try:
                        got = t3.ships_shear(sid, cyc, max_hours=3.0)
                        break
                    except Exception:
                        time.sleep(3)
            ships_cache[key] = (got.kt, got.heading) if got is not None else None
        sh = ships_cache[key]
        case = {"kind": "seb", "mission": md["mission"], "mission_dir": md["mission"], "storm": md["storm"].upper(),
                "storm_id": sid, "year": tc.year, "month": tc.month, "day": tc.day, "hour": tc.hour,
                "minute": tc.minute, "t_centre": md["centre_time"], "t_start": md["start"], "t_end": md["end"],
                "centre_2km": md["centre"], "motion_x": mx, "motion_y": my, "vmax": vbt,
                "shdc": sh[0] if sh else None, "sddc": sh[1] if sh else None,
                "lat": md["centre"][0], "lon": md["centre"][1], "name": md["name"], "z": [float(x) for x in z],
                "src": os.path.basename(path)}
        cases.append(case)
        heading, cov = ds.covariates(case, vm, R)
        a = ds.to_normalised(vt, z, R, heading) / vm
        b = ds.to_normalised(vr, z, R, heading) / vm
        rows.append(np.stack([a, b]).astype(np.float32))
        meta.append({"kind": "seb", "mission": md["mission"], "storm": case["storm"], "tcid": sid,
                     "time": (tc.year, tc.month, tc.day, tc.hour, tc.minute), "heading": heading, "vbt_kt": vbt,
                     "src": case["src"], "index": 0, **cov})
        print(f"{md['name']} {case['storm']:10s} {sid} Vm {vm:5.1f} R {R:5.1f} BT {vbt} shear {sh}", flush=True)
    json.dump(cases, open(prefix + "_cases.json", "w"))
    np.save(prefix + "_X.npy", np.array(rows))
    json.dump(meta, open(prefix + "_X.json", "w"))
    print(len(cases), "cases;", sum(1 for c in cases if c["shdc"] is not None), "with SHIPS shear")


if __name__ == "__main__":
    main(*sys.argv[1:4])
