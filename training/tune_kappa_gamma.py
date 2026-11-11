"""
Tune the prior weight (kappa) and the flight-level error inflation (gamma) on real flight-level
data: the prior trained without 20% of the storms, then fitted to those storms' IWG1 obs pass by
pass and scored against their own TDR analyses. Shipped: kappa 0.25, gamma 6.

Usage (from this folder, after extract_tcradar.py, dataset.py and fetch_iwg1.py):
    python tune_kappa_gamma.py
"""
import json
import os
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import numpy as np  # noqa: E402

import evaluate as ev  # noqa: E402
import export  # noqa: E402
import realval as rv  # noqa: E402
import recon10s_syntdr as st  # noqa: E402

HELD = "data/heldout_k60.npz"
if not os.path.exists(HELD):
    X = np.load("data/norm_X.npy")
    meta = json.load(open("data/norm_X.json"))
    keep = np.array([not ev.held_out(m["storm"], m["time"][0]) for m in meta])
    export.train_and_save(X, meta, HELD, k=60, keep=keep)
cases = {kind: [c for c in rv.tcradar_cases(kind) if ev.held_out(c[0]["storm"], c[0]["year"])]
         for kind in ("swath", "merge")}
print({k: len(v) for k, v in cases.items()}, flush=True)
m = st.Model(HELD)
best = None
for kappa in (0.125, 0.25, 0.5, 1.0):
    for gamma in (1.5, 3.0, 6.0, 12.0):
        out = {}
        for kind in ("swath", "merge"):
            rows = ev.evaluate(m, cases=cases[kind], kappa=kappa, gamma=gamma)
            ev.summarize(rows, f"kappa {kappa} gamma {gamma} {kind}")
            out[kind] = rows
        score = np.mean([r["rmse_vt"] for r in out["swath"]]) + np.mean([r["rmse_vt"] for r in out["merge"]])
        if best is None or score < best[0]:
            best = (score, kappa, gamma, out)
print("best: kappa", best[1], "gamma", best[2])
json.dump(best[3], open("data/eval_best_rows.json", "w"))
