"""5-fold cross-validation over TC-RADAR storms: the prior trained without each fold, evaluated on the
fold's passes with their IWG1 data. Gives out-of-sample rows for every 2018-2024 storm with IWG1."""
import hashlib, json, os, sys, time, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import numpy as np
import export, realval as rv
import recon10s_syntdr as st
from evalcase import eval_case

K = 5


def fold(storm, year):
    return int(hashlib.md5(f"{storm}{year}".encode()).hexdigest()[:8], 16) % K


X = np.load("data/norm_X.npy")
meta = json.load(open("data/norm_X.json"))
fold_of_row = np.array([fold(m["storm"], m["time"][0]) for m in meta])
cases = rv.tcradar_cases("swath") + rv.tcradar_cases("merge")
print(len(cases), "cases with IWG1", flush=True)
hyper = {"kappa": 0.25, "gamma": 6.0}
rows, zs = [], []
t0 = time.time()
for f in range(K):
    path = f"data/cv_fold{f}.npz"
    export.train_and_save(X, meta, path, k=60, hyper=hyper, keep=fold_of_row != f)
    m = st.Model(path)
    mine = [c for c in cases if fold(c[0]["storm"], c[0]["year"]) == f]
    for case, vt, vr in mine:
        r = eval_case(m, case, vt, vr, with_z=True)
        if r is None:
            continue
        zs.append(r.pop("_z"))
        r["fold"] = f
        rows.append(r)
    print(f"fold {f}: {len(mine)} cases, {len(rows)} rows so far ({time.time() - t0:.0f}s)", flush=True)
json.dump(rows, open("data/cv_rows.json", "w"))
np.save("data/cv_z.npy", np.concatenate(zs))
print("done", len(rows))
