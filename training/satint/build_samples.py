"""
Training samples for the satellite intensity model, from the reduced TC PRIMED storms.

A sample is one IR frame (time t) of a storm with:
  * the same storm's IR frames nearest t-3, t-6, t-9 and t-12 h (within 1 h; -1 when none);
  * the most recent imager pass (37 + 89 GHz) from t-12 h to t (+15 min), covering at least half
    of the inner 200 km, its age, and the IR frame taken with it; and also the youngest pass in each
    of the age bins 0-2, 2-4, 4-7 and 7-12 h, so training can show the network older passes than
    TC PRIMED's dense constellation usually leaves (the app sees fewer sensors);
  * best-track intensity and pressure at t (TC PRIMED interpolates them to each overpass);
  * scalars known in real time: latitude, translation speed, basin, local solar hour.

Usage: python build_samples.py STORMDIR OUT_PREFIX
Writes OUT_PREFIX_ir.npy (all IR frames, uint8), OUT_PREFIX_mw.npy (all passes, uint8),
OUT_PREFIX_samples.npz (index arrays and scalars) and OUT_PREFIX_meta.json.
"""
import glob
import json
import math
import os
import sys

import numpy as np

LAGS_H = (3.0, 6.0, 9.0, 12.0)
MW_MAX_AGE_H = 12.0
MW_MIN_COVER = 0.5
TROPICAL = {"TD", "TS", "HU", "TY", "ST", "SD", "SS", "TC"}
AGE_BINS = ((-0.25, 2.0), (2.0, 4.0), (4.0, 7.0), (7.0, 12.0))
SENSOR_RES = {"SSMI": 15.0, "SSMIS": 14.0, "TMI": 7.0, "AMSRE": 5.0, "AMSR2": 4.0, "GMI": 6.0}


def solar_hour(t, lon):
    h = (t % 86400) / 3600.0 + (((lon + 180) % 360) - 180) / 15.0
    return h % 24.0


def main(stormdir, prefix):
    files = sorted(glob.glob(os.path.join(stormdir, "*.npz")))
    # the frame stores are written straight to disk (all of 2000-2025 is a few GB)
    tot_ir = tot_mw = 0
    for path in files:
        with np.load(path) as d:
            tot_ir += d["ir"].shape[0]
            tot_mw += d["mw"].shape[0]
    ir_out = np.lib.format.open_memmap(prefix + "_ir.npy", mode="w+", dtype=np.uint8, shape=(tot_ir, 128, 128))
    mw_out = np.lib.format.open_memmap(prefix + "_mw.npy", mode="w+", dtype=np.uint8, shape=(tot_mw, 4, 64, 64))
    n_ir = n_mw = 0
    S = {k: [] for k in ("ir", "h3", "h6", "h9", "h12", "mw", "mw_ir", "age", "vmax", "mslp", "lat", "speed",
                         "basin", "solar", "mw_cov", "mw_res", "year", "time", "storm",
                         "mwc", "mwirc", "agec", "covc", "resc", "dv12", "dv24",
                         "mwg", "mwirg", "ageg", "covg")}
    storms = []
    for si, path in enumerate(files):
        storm = os.path.basename(path)[:-4]
        meta = json.load(open(path[:-4] + ".json"))
        d = np.load(path)
        ir, mw = d["ir"], d["mw"]
        if ir.shape[0] == 0:
            continue
        ir_out[n_ir:n_ir + ir.shape[0]] = ir
        if mw.shape[0]:
            mw_out[n_mw:n_mw + mw.shape[0]] = mw
        # frames of this storm: (time, global index, metadata row)
        frames = [(m["ir_t"], n_ir + m["ir"], m) for m in meta if m["ir"] >= 0 and m.get("ir_t")]
        frames.sort(key=lambda x: x[0])
        # overpasses minutes apart share one image of CIRA's IR archive: one sample per image
        seen, uniq = set(), []
        for f in frames:
            if f[0] not in seen:
                seen.add(f[0])
                uniq.append(f)
        frames = uniq
        passes = [(m["mw_t"] or m["time"], n_mw + m["mw"], m) for m in meta
                  if m["mw"] >= 0 and (m.get("mw_cov") or 0) >= MW_MIN_COVER]
        passes.sort(key=lambda x: x[0])
        ft = np.array([f[0] for f in frames], float)
        pt = np.array([p[0] for p in passes], float)
        for t, gi, m in frames:
            dev = (m.get("development_level") or "").strip().upper()
            vmax, mslp = m.get("intensity"), m.get("central_min_pressure")
            if dev not in TROPICAL or vmax is None or vmax < 15 or vmax > 200:
                continue
            hist = []
            for lag in LAGS_H:
                target = t - lag * 3600
                j = int(np.argmin(np.abs(ft - target))) if ft.size else -1
                hist.append(frames[j][1] if j >= 0 and abs(ft[j] - target) <= 3600 else -1)
            k = -1
            cand = [(-1, -1, -1.0, 0.0, 0.0)] * len(AGE_BINS)
            if pt.size:
                ok = np.flatnonzero((pt <= t + 900) & (pt >= t - MW_MAX_AGE_H * 3600))
                if ok.size:
                    k = int(ok[-1])
                for b, (a0, a1) in enumerate(AGE_BINS):
                    age_h = (t - pt) / 3600.0
                    inb = np.flatnonzero((age_h >= a0) & (age_h < a1) & (pt <= t + 900))
                    if inb.size:
                        kb = int(inb[-1])
                        pb = passes[kb][2]
                        cand[b] = (passes[kb][1], n_ir + pb["ir"] if pb["ir"] >= 0 else -1,
                                   max(0.0, (t - pt[kb]) / 3600.0), pb["mw_cov"], SENSOR_RES.get(pb["sensor"], 10.0))
            # the newest GMI pass alone: what the app's sensors (GMI, AMSR3, WSF-M) leave in a season
            # TC PRIMED has only GMI of them for
            kg = -1
            if pt.size:
                okg = [q for q in np.flatnonzero((pt <= t + 900) & (pt >= t - MW_MAX_AGE_H * 3600))
                       if passes[q][2]["sensor"] == "GMI"]
                if okg:
                    kg = int(okg[-1])
            if kg >= 0:
                pg = passes[kg][2]
                S["mwg"].append(passes[kg][1]); S["mwirg"].append(n_ir + pg["ir"] if pg["ir"] >= 0 else -1)
                S["ageg"].append(max(0.0, (t - pt[kg]) / 3600.0)); S["covg"].append(pg["mw_cov"])
            else:
                S["mwg"].append(-1); S["mwirg"].append(-1); S["ageg"].append(-1.0); S["covg"].append(0.0)
            S["mwc"].append([c[0] for c in cand]); S["mwirc"].append([c[1] for c in cand])
            S["agec"].append([c[2] for c in cand]); S["covc"].append([c[3] for c in cand])
            S["resc"].append([c[4] for c in cand])
            if k >= 0:
                pm = passes[k][2]
                mw_ir = n_ir + pm["ir"] if pm["ir"] >= 0 else -1
                S["mw"].append(passes[k][1]); S["mw_ir"].append(mw_ir)
                S["age"].append(max(0.0, (t - pt[k]) / 3600.0)); S["mw_cov"].append(pm["mw_cov"])
                S["mw_res"].append(SENSOR_RES.get(pm["sensor"], 10.0))
            else:
                S["mw"].append(-1); S["mw_ir"].append(-1); S["age"].append(-1.0); S["mw_cov"].append(0.0)
                S["mw_res"].append(0.0)
            S["ir"].append(gi)
            for name, h in zip(("h3", "h6", "h9", "h12"), hist):
                S[name].append(h)
            S["vmax"].append(vmax)
            # best-track change over the 12 and 24 h before (TC PRIMED: periods -24..24 h, -999 missing)
            ch = m.get("intensity_change") or []
            S["dv24"].append(ch[0] if len(ch) == 9 and ch[0] > -900 else np.nan)
            S["dv12"].append(ch[2] if len(ch) == 9 and ch[2] > -900 else np.nan)
            S["mslp"].append(mslp if mslp is not None and 850 < mslp < 1030 else np.nan)
            lat = m.get("storm_latitude") or 0.0
            lon = m.get("storm_longitude") or 0.0
            S["lat"].append(lat)
            # storm_speed is packed: int16, scale 0.1, m/s, fill -9990 (h5py does not unpack it)
            sp = m.get("storm_speed")
            S["speed"].append(sp * 0.1 / 0.514444 if sp is not None and sp > -9000 else np.nan)
            S["basin"].append(0 if storm.startswith("AL") else 1)
            S["solar"].append(solar_hour(t, lon))
            S["year"].append(int(storm[-4:]))
            S["time"].append(t)
            S["storm"].append(si)
        storms.append(storm)
        n_ir += ir.shape[0]
        n_mw += mw.shape[0]
        if si % 100 == 0:
            print(si, storm, n_ir, n_mw, len(S["ir"]), flush=True)
    ir_out.flush()
    mw_out.flush()
    assert n_ir <= tot_ir and n_mw <= tot_mw
    arrays = {k: np.array(v) for k, v in S.items()}
    np.savez(prefix + "_samples.npz", **arrays)
    json.dump({"storms": [os.path.basename(f)[:-4] for f in files]}, open(prefix + "_meta.json", "w"))
    n = len(S["ir"])
    print(f"{n} samples from {len(files)} storms; {n_ir} IR frames, {n_mw} passes; "
          f"with a pass {np.mean(arrays['mw'] >= 0):.2f}; history -3h {np.mean(arrays['h3'] >= 0):.2f}, "
          f"-12h {np.mean(arrays['h12'] >= 0):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
