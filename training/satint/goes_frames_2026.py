"""
GOES band-13 frames of the 2026 Atlantic and east/central Pacific storms, every 3 hours along NHC's
best track, through the app's own pipeline (recon10s_satint.Estimator.frame): the independent
check of the satellite intensity model on a season TC PRIMED does not have yet.

Usage: python goes_frames_2026.py BDECKS.json OUTDIR [workers]
One npz per storm: ir (n,128,128) uint8 as in training, times, centres, satellites.
"""
import concurrent.futures as cf
import datetime as dt
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))

TROPICAL = {"TD", "TS", "HU", "TY", "ST", "SD", "SS"}


def storm_frames(args):
    sid, rows, outdir = args
    import recon10s_nhc as nhc
    import recon10s_satint as si
    out = os.path.join(outdir, sid + ".npz")
    if os.path.exists(out):
        return sid, "cached"
    track = [nhc.TrackPoint(time=dt.datetime.fromisoformat(r[0]), lat=r[1], lon=r[2], vmax=r[3], mslp=r[4],
                            kind=r[5] if len(r) > 5 else "") for r in rows]
    track.sort(key=lambda p: p.time)
    if len(track) < 2:
        return sid, "short"
    src = si.TrackSource(track=track)
    est = si.Estimator(sid, src)
    t = track[0].time - dt.timedelta(hours=12)
    end = track[-1].time
    frames, times, centres, sats = [], [], [], []
    while t <= end:
        try:
            fr = est.frame(t)
        except Exception as exc:
            fr = None
        if fr is not None:
            c = src.centre_at(fr[1])
            q = np.round((fr[0] - 163.0) / 0.6)
            q = np.where(np.isfinite(q), np.clip(q, 0, 254), 255).astype(np.uint8)
            frames.append(q)
            times.append(fr[1].timestamp())
            centres.append(c)
            sats.append(fr[2])
        t += dt.timedelta(hours=3)
    np.savez_compressed(out, ir=np.array(frames, np.uint8).reshape(-1, 128, 128), time=np.array(times),
                        centre=np.array(centres, float).reshape(-1, 2), sat=np.array(sats, int))
    return sid, f"{len(frames)} frames ({len(est.errors)} errors)"


def main(bdecks, outdir, workers=4):
    os.makedirs(outdir, exist_ok=True)
    d = json.load(open(bdecks))
    jobs = []
    for sid, s in d.items():
        if not sid.endswith("2026") or not sid[:2] in ("al", "ep", "cp") or int(sid[2:4]) >= 90:
            continue
        rows = [r for r in s["track"] if r[3] is not None]
        if len(rows) >= 4:
            jobs.append((sid, rows, outdir))
    print(len(jobs), "storms", flush=True)
    t0 = time.time()
    with cf.ProcessPoolExecutor(workers) as ex:
        for sid, msg in ex.map(storm_frames, jobs):
            print(f"{sid}: {msg} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
