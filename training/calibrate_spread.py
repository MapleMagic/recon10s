"""
How the synthetic field's error depends on the distance to the nearest ob and on r / R, on the
independent 2025-2026 analyses, and the factors by r / R that make the model's spread match it
(train_final.py's spread_by_rho).

Usage: python calibrate_spread.py data/model_tcradar_all_k60.npz
"""
import os, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import numpy as np
from scipy.spatial import cKDTree
import recon10s_syntdr as st
import dataset as ds, evaluate as ev, realval as rv
import realtime_eval as seb_eval

m = st.Model(sys.argv[1])
cases = seb_eval.load_cases()
D, E, S, RHO, VM = [], [], [], [], []
for case, vt, vr in cases:
    o, info = rv.iwg1_obs(case)
    if o is None:
        continue
    z_src = np.array(case["z"])
    vm_t, R_t = ds.vm_and_rmw(vt, z_src)
    if vm_t is None:
        continue
    obs = ev.as_app_obs(o)
    try:
        fit = m.fit(obs, rv.heading_of(case), rv.cov_extra(case), R_obs=st.observed_rmw(obs))
    except ValueError:
        continue
    r_src = (np.arange(vt.shape[1]) + 0.5) * 2.0
    az_src = (np.arange(vt.shape[2]) + 0.5) * (360.0 / vt.shape[2])
    Rg, Ag = np.meshgrid(r_src, az_src, indexing="ij")
    sel = Rg <= 150
    x = (Rg * np.sin(np.radians(Ag)))[sel]
    y = (Rg * np.cos(np.radians(Ag)))[sel]
    tree = cKDTree(np.c_[obs.x, obs.y])
    dist, _ = tree.query(np.c_[x, y])
    s, sp = m.state(fit), m.spread(fit)
    for zz in (1.0, 2.0):
        k = int(np.argmin(np.abs(z_src - zz)))
        tv, rvv = vt[k][sel], vr[k][sel]
        ok = np.isfinite(tv) & np.isfinite(rvv)
        pt, pr, st_, sr_ = m.sample_many(fit, [(s, 0), (s, 1), (sp, 0), (sp, 1)], x[ok], y[ok], np.full(ok.sum(), zz))
        err = np.hypot(pt - tv[ok], pr - rvv[ok]) / np.sqrt(2)       # rms of the two components
        D.append(dist[ok]); E.append(err); S.append(np.hypot(st_, sr_) / np.sqrt(2)); RHO.append(np.hypot(x[ok], y[ok]) / fit.R)
        VM.append(np.full(ok.sum(), fit.vm))
D, E, S, RHO, VM = map(np.concatenate, (D, E, S, RHO, VM))
print(f"{D.size} points")
for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 35), (35, 50), (50, 75), (75, 100), (100, 150)):
    b = (D >= lo) & (D < hi)
    if b.sum() < 50:
        continue
    print(f"nearest ob {lo:3d}-{hi:3d} km: n {b.sum():6d}  rms err {np.sqrt(np.mean(E[b] ** 2)) * 1.944:5.1f} kt   "
          f"rms err/Vm {np.sqrt(np.mean((E[b] / VM[b]) ** 2)):.3f}   model spread rms {np.sqrt(np.mean(S[b] ** 2)) * 1.944:5.1f} kt")
for lo, hi in ((0, 0.5), (0.5, 1.5), (1.5, 3), (3, 8)):
    b = (RHO >= lo) & (RHO < hi)
    print(f"r/R {lo}-{hi}: rms err {np.sqrt(np.mean(E[b] ** 2)) * 1.944:5.1f} kt, err/Vm {np.sqrt(np.mean((E[b] / VM[b]) ** 2)):.3f}")
np.savez("data/err_vs_dist.npz", D=D, E=E, S=S, RHO=RHO, VM=VM)

print("spread factors by r / R (rms error / rms spread):")
edges = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]
for lo, hi in zip(edges[:-1], edges[1:]):
    b = (RHO >= lo) & (RHO < hi)
    if b.sum() >= 100:
        print(f"  {(lo + hi) / 2:5.3f}: {np.sqrt(np.mean(E[b] ** 2)) / np.sqrt(np.mean(S[b] ** 2)):.2f}")
