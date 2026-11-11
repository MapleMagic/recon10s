"""
West Pacific / Southern Hemisphere / North Indian Ocean samples again, with the IR taken through the
app's own pipeline (recon10s_satint.Estimator.frame: Himawari-9 west of about 178W, GOES-18 east of
it, from NOAA's buckets) instead of TC PRIMED's archive images -- the frames at t, t-3, t-6, t-9 and
t-12 h and at the time of the sample's microwave pass. Himawari-9's archive on AWS starts in
December 2022.

Usage: python himawari_pairs.py DATA_PREFIX STORMDIR OUTDIR [years 2023-2025] [n_samples 300] [workers 3]
                                 [zenith range, e.g. 55-80: only samples the app sees that obliquely, its
                                 70-degree limit lifted to measure what lies beyond it]
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
H9_START = dt.datetime(2022, 12, 14, tzinfo=UTC).timestamp() + 12 * 3600


def sample_lon(S, i):
    t = dt.datetime.fromtimestamp(float(S["time"][i]), UTC)
    lon = (S["solar"][i] - (t.hour + t.minute / 60 + t.second / 3600)) * 15.0
    return (lon + 540.0) % 360.0 - 180.0


def select(S, years, n_target, rng, zenith=None):
    """Samples at least 6 h apart within each storm, that the app can estimate (or within a zenith-angle
    range); strong ones first."""
    import recon10s_goes as goes
    import recon10s_satint as si
    y0, y1 = years
    z0, z1 = zenith or (0.0, si.MAX_ZENITH)
    pick = []
    sel = (S["year"] >= y0) & (S["year"] <= y1) & (S["time"] >= H9_START)
    for s_ in np.unique(S["storm"][sel]):
        idx = np.flatnonzero((S["storm"] == s_) & sel)
        idx = idx[np.argsort(S["time"][idx])]
        last = -1e18
        for i in idx:
            la, lo = float(S["lat"][i]), sample_lon(S, i)
            if not z0 <= goes.zenith_angle(la, lo, goes.pick_satellite(la, lo)) <= z1:
                continue
            if S["time"][i] - last >= 6 * 3600:
                pick.append(i)
                last = S["time"][i]
    pick = np.array(pick)
    strong = pick[S["vmax"][pick] >= 64]
    weak = pick[S["vmax"][pick] < 64]
    n_strong = min(len(strong), int(n_target * 2 / 3))
    strong = rng.permutation(strong)[:n_strong]
    weak = rng.permutation(weak)[:max(0, n_target - n_strong)]
    return np.sort(np.concatenate([strong, weak]))


def storm_job(args):
    name, idx, times, ages, stormdir, outdir, max_zenith = args
    import recon10s_nhc as nhc
    import recon10s_satint as si
    si.MAX_ZENITH = max_zenith
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
    est = si.Estimator(name.lower(), si.TrackSource(track=track))
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
    errs = "; ".join(sorted(set(e.split(":")[0] for e in est.errors))[:3])
    return name, f"{n} samples, {got} with a current frame ({len(est.errors)} errors {errs})"


def main(prefix, stormdir, outdir, years="2023-2025", n_target=300, workers=3, zenith=None):
    import recon10s_satint as si
    os.makedirs(outdir, exist_ok=True)
    S = dict(np.load(prefix + "_samples.npz"))
    storms = json.load(open(prefix + "_meta.json"))["storms"]
    rng = np.random.RandomState(7)
    zr = tuple(float(x) for x in zenith.split("-")) if zenith else None
    max_zenith = max(si.MAX_ZENITH, zr[1]) if zr else si.MAX_ZENITH
    pick = select(S, tuple(int(x) for x in years.split("-")), n_target, rng, zr)
    print(len(pick), "samples;", int((S["vmax"][pick] >= 64).sum()), "typhoon-strength", flush=True)
    jobs = []
    for s_ in np.unique(S["storm"][pick]):
        idx = pick[S["storm"][pick] == s_]
        jobs.append((storms[s_], idx.tolist(), S["time"][idx].tolist(), S["age"][idx].tolist(), stormdir, outdir,
                     max_zenith))
    t0 = time.time()
    with cf.ProcessPoolExecutor(workers) as ex:
        for name, msg in ex.map(storm_job, jobs):
            print(f"{name}: {msg} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], a[4] if len(a) > 4 else "2023-2025", int(a[5]) if len(a) > 5 else 300,
         int(a[6]) if len(a) > 6 else 3, a[7] if len(a) > 7 else None)
