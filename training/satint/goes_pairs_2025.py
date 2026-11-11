"""
The 2025 test samples again, with the IR taken through the app's own GOES pipeline
(recon10s_satint.Estimator.frame) instead of TC PRIMED's archive images: the frames at t, t-3, t-6,
t-9 and t-12 h and at the time of the sample's microwave pass. Lets the evaluation compare the
network on archive IR (what it was trained on) and on live-style GOES IR for the same cases.

Usage: python goes_pairs_2025.py DATA_PREFIX STORMDIR OUTDIR [n_samples] [workers]
One npz per storm: idx (sample indices into DATA_PREFIX), ir (n, 6, 128, 128) float16 K (NaN where
no frame), t (n, 6) scan times, sat (n, 6).
"""
import concurrent.futures as cf
import datetime as dt
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
UTC = dt.timezone.utc
LAGS = (0.0, 3.0, 6.0, 9.0, 12.0)


def select(S, storms, n_target, rng):
    """2025 samples at least 6 h apart within each storm; all hurricanes, the rest at random."""
    pick = []
    for si in np.unique(S["storm"][S["year"] == 2025]):
        idx = np.flatnonzero((S["storm"] == si) & (S["year"] == 2025))
        idx = idx[np.argsort(S["time"][idx])]
        last = -1e18
        for i in idx:
            if S["time"][i] - last >= 6 * 3600:
                pick.append(i)
                last = S["time"][i]
    pick = np.array(pick)
    strong = pick[S["vmax"][pick] >= 64]
    weak = pick[S["vmax"][pick] < 64]
    n_weak = max(0, n_target - len(strong))
    weak = rng.permutation(weak)[:n_weak]
    out = np.sort(np.concatenate([strong, weak]))
    return out


def storm_job(args):
    name, idx, times, ages, stormdir, outdir = args
    import recon10s_nhc as nhc
    import recon10s_satint as si
    out = os.path.join(outdir, name + ".npz")
    if os.path.exists(out):
        return name, "cached"
    meta = json.load(open(os.path.join(stormdir, name + ".json")))
    pts = {}
    for m in meta:
        if m.get("ir_t") and m.get("ir_c"):
            la, lo = m["ir_c"]
            pts[int(m["ir_t"])] = (la, ((lo + 180) % 360) - 180)
    track = [nhc.TrackPoint(time=dt.datetime.fromtimestamp(t, UTC), lat=la, lon=lo) for t, (la, lo) in sorted(pts.items())]
    src = si.TrackSource(track=track)
    est = si.Estimator(name.lower(), src)
    n = len(idx)
    ir = np.full((n, 6, 128, 128), np.nan, np.float16)
    ts = np.zeros((n, 6))
    sats = np.zeros((n, 6), int)
    for k, (t, age) in enumerate(zip(times, ages)):
        want = [t - lag * 3600 for lag in LAGS] + [t - max(age, 0) * 3600 if age >= 0 else None]
        for j, tw in enumerate(want):
            if tw is None:
                continue
            when = dt.datetime.fromtimestamp(tw, UTC)
            try:
                fr = est.frame(when)
            except Exception:
                fr = None
            if fr is not None and abs((fr[1] - when).total_seconds()) <= 1800:
                ir[k, j] = fr[0].astype(np.float16)
                ts[k, j] = fr[1].timestamp()
                sats[k, j] = fr[2]
    np.savez_compressed(out, idx=np.array(idx), ir=ir, t=ts, sat=sats)
    got = int(np.isfinite(ir[:, 0, 64, 64]).sum())
    return name, f"{n} samples, {got} with a current frame ({len(est.errors)} errors)"


def main(prefix, stormdir, outdir, n_target=160, workers=3):
    os.makedirs(outdir, exist_ok=True)
    S = dict(np.load(prefix + "_samples.npz"))
    storms = json.load(open(prefix + "_meta.json"))["storms"]
    rng = np.random.RandomState(7)
    pick = select(S, storms, n_target, rng)
    print(len(pick), "samples;", int((S["vmax"][pick] >= 64).sum()), "hurricanes", flush=True)
    jobs = []
    for si in np.unique(S["storm"][pick]):
        idx = pick[S["storm"][pick] == si]
        jobs.append((storms[si], idx.tolist(), S["time"][idx].tolist(), S["age"][idx].tolist(), stormdir, outdir))
    t0 = time.time()
    with cf.ProcessPoolExecutor(workers) as ex:
        for name, msg in ex.map(storm_job, jobs):
            print(f"{name}: {msg} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 160, int(a[5]) if len(a) > 5 else 3)
