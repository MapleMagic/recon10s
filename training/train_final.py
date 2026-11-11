"""
The shipped model: the prior trained on every TC-RADAR analysis (1997-2024) and every usable
real-time HRD analysis of 2025-2026, with the settings tuned on real flight-level data (IWG1) and
the intensity calibration and uncertainty profile measured out of sample.

Usage: python train_final.py OUT.npz
"""
import datetime as dt
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import dataset as ds  # noqa: E402
import export  # noqa: E402
import synmodel as sm  # noqa: E402

KT = 1.943844


def seb_rows():
    X = np.load("data/seb_X.npy")
    meta = json.load(open("data/seb_X.json"))
    keep = []
    for i, m in enumerate(meta):
        mission = m["mission"]
        if mission[8] == "N":                       # the G-IV's radar sees little of the low levels
            continue
        ratio = m["vm"] * KT / m["vbt_kt"] if m.get("vbt_kt") else 1.0
        if not 0.45 <= ratio <= 1.3:                # the 2-km peak far from the best track: a partial analysis
            continue
        keep.append(i)
    return X[keep], [meta[i] for i in keep]


def impute_shear(X, meta, B):
    """Shear along/across the motion that best explains each analysis's structure (for rows without SHIPS)."""
    out = 0
    for i, m in enumerate(meta):
        if m.get("has_shear"):
            continue
        c = sm.cov_vector(m["vm"], m["R"], 0.0, 0.0, m["speed"], m["lat"])
        a = c[1]
        mu0 = c @ B
        g1 = B[3] + a * B[7]
        g2 = B[4] + a * B[8]
        x = X[i].reshape(-1).astype(np.float64)
        ok = np.isfinite(x)
        G = np.c_[g1[ok], g2[ok]]
        lam = 1.0
        cd = np.linalg.solve(G.T @ G + lam * np.eye(2), G.T @ (x[ok] - mu0[ok]))
        cd = np.clip(cd, -2.0, 2.0)
        m["s_par"], m["s_perp"] = float(20 * cd[0]), float(20 * cd[1])
        m["shear_imputed"] = True
        out += 1
    return out


def main(out):
    t0 = time.time()
    X1 = np.load("data/norm_X.npy")
    m1 = json.load(open("data/norm_X.json"))
    X2, m2 = seb_rows()
    n25 = sum(1 for m in m2 if m["time"][0] == 2025)
    print(f"TC-RADAR {len(m1)}, real-time {len(m2)} ({n25} of 2025, no SHIPS archive)", flush=True)
    # 1) the regression on the analyses with every predictor, to infer 2025's shear
    complete = np.array([True] * len(m1) + [bool(m.get("has_shear")) for m in m2])
    X = np.concatenate([X1, X2])
    meta = m1 + m2
    grid = sm.Grid(ds.RHO, ds.PHI, ds.Z)
    C = export.covariates(meta)
    pre = sm.SynModel.train(X[complete], C[complete], grid, k=60, iters=40, progress=None)
    n_imp = impute_shear(X2, m2, pre.B)
    print(f"shear inferred for {n_imp} analyses ({time.time() - t0:.0f}s)", flush=True)
    meta = m1 + m2
    cal = json.load(open("data/intensity_calibration.json"))
    hyper = {
        "kappa": 0.25, "gamma": 6.0, "sigma_fl": 2.0, "sigma_drop": 5.0,
        "R_prior": [math.log(35.0), 0.7], "R_obs_sd": 0.3,
        "surface_level_km": 1.0, "surface_ratio": round(float(cal["pk_ratio_coef"]) if "pk_ratio_coef" in cal else 0.983, 3),
        "intensity": {"hinge_kt": cal["hinge_kt"], "full": cal["full"]["coef"], "pk_only": cal["pk_only"]["coef"]},
        "intensity_rmse_kt": cal["full"]["loso_all"]["rmse"],
        # posterior spread x factor(r / R): errors against independent 2025-2026 TDR analyses
        "spread_by_rho": [[0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.75, 2.25, 2.75, 3.5, 4.5, 6.5],
                          [2.82, 2.09, 1.71, 1.46, 1.34, 1.25, 1.07, 0.96, 0.86, 0.87, 0.88, 0.95]],
    }
    val = json.load(open("data/validation_summary.json"))
    years = sorted({int(m["time"][0]) for m in meta if m.get("time")})
    info = {
        "version": 1, "trained": dt.date.today().isoformat(), "n_analyses": int(X.shape[0]),
        "n_tcradar": len(m1), "n_realtime": len(m2), "years": [years[0], years[-1]], "eofs": 60,
        "sources": [
            "TC-RADAR v3m (NOAA AOML/HRD, Fischer et al. 2022): merged and swath analyses 1997-2024",
            "HRD real-time TDR analyses 2025-2026 (seb.omao.noaa.gov/pub/flight/radar)",
            "NOAA P-3 IWG1 flight-level data 2018-2026 (seb.omao.noaa.gov/pub/flight/aamps_ingest/iwg1, CUI): "
            "tuning, intensity calibration and validation only",
            "SHIPS shear (TC-RADAR; NHC lsdiag for 2026), NHC best tracks",
        ],
        "validation": val,
    }
    export.train_and_save(X, meta, out, k=60, hyper=hyper, info=info)
    print(f"saved {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
