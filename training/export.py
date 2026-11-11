"""Train the prior on a subset of the normalised TDR analyses and write it in the app's npz format."""
import json
import math
import sys

import numpy as np

import dataset as ds
import synmodel as sm


def covariates(meta):
    return np.array([sm.cov_vector(m["vm"], m["R"], m["s_par"], m["s_perp"], m["speed"], m["lat"]) for m in meta])


def train_and_save(X, meta, out, k=40, hyper=None, info=None, iters=40, keep=None):
    C = covariates(meta)
    if keep is not None:
        X, C = X[keep], C[keep]
    grid = sm.Grid(ds.RHO, ds.PHI, ds.Z)
    model = sm.SynModel.train(X, C, grid, k=k, iters=iters, progress=None)
    hyper = dict({"kappa": 0.5, "gamma": 3.0, "sigma_fl": 2.0, "sigma_drop": 3.0,
                  "R_prior": [math.log(35.0), 0.7], "R_obs_sd": 0.3, "surface_ratio": 0.91,
                  "surface_level_km": 1.0, "intensity_rmse_kt": 8.0}, **(hyper or {}))
    np.savez_compressed(out, rho=ds.RHO.astype(np.float32), phi=ds.PHI.astype(np.float32), z=ds.Z.astype(np.float32),
                        B=model.B.astype(np.float32), E=model.E.astype(np.float16), lam=model.lam.astype(np.float64),
                        s2=model.s2.astype(np.float32), hyper=np.array(json.dumps(hyper)),
                        info=np.array(json.dumps(info or {})))
    return out


if __name__ == "__main__":
    X = np.load("data/norm_X.npy")
    meta = json.load(open("data/norm_X.json"))
    train_and_save(X, meta, sys.argv[1], k=int(sys.argv[2]) if len(sys.argv) > 2 else 40)
