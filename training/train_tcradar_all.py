"""
The prior trained on TC-RADAR alone (1997-2024), for the independent check on the real-time
analyses of 2025-2026 (score_realtime.py, calibrate_spread.py).

Usage: python train_tcradar_all.py   (writes data/model_tcradar_all_k60.npz)
"""
import json
import time

import numpy as np

import export

X = np.load("data/norm_X.npy")
meta = json.load(open("data/norm_X.json"))
t = time.time()
export.train_and_save(X, meta, "data/model_tcradar_all_k60.npz", k=60, hyper={"kappa": 0.25, "gamma": 6.0},
                      info={"version": 1, "trained_on": "TC-RADAR v3m 1997-2024", "n": int(X.shape[0])})
print("trained", X.shape, f"{time.time() - t:.0f}s")
