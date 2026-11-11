"""
Evaluate a trained prior on held-out storms with their real flight-level data (IWG1, 30-s means):
field errors against the TDR analysis of the same pass/mission, and the 1-km earth-relative peak
for the intensity calibration against the best track. Uses the app's reconstruction code.
"""
import hashlib
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import recon10s_syntdr as st  # noqa: E402
import dataset as ds  # noqa: E402
import realval as rv  # noqa: E402


def held_out(storm, year, frac=0.2):
    h = int(hashlib.md5(f"{storm}{year}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def as_app_obs(o):
    return st.Obs(x=o.x, y=o.y, z=np.maximum(o.z, 0.5), u=o.u, v=o.v, sigma=o.sigma, kind=o.kind,
                  t=np.zeros(o.x.size), group=np.zeros(o.x.size, int))


def peak_1km_earth(m, fit, motion_ms, half_km=150.0, dx=2.0, z=1.0):
    x = np.arange(-half_km, half_km + dx / 2, dx)
    X, Y = np.meshgrid(x, x)
    s = m.state(fit)
    Z = np.full(X.shape, z)
    vt = m.sample(fit, s, X, Y, Z, 0)
    vr = m.sample(fit, s, X, Y, Z, 1)
    th = np.arctan2(Y, X)
    u = vr * np.cos(th) - vt * np.sin(th) + motion_ms[0]
    v = vr * np.sin(th) + vt * np.cos(th) + motion_ms[1]
    sp = np.hypot(u, v)
    sp[np.hypot(X, Y) > 8 * fit.R] = np.nan
    return float(np.nanmax(sp))


def evaluate(m, kinds=("swath", "merge"), kappa=None, gamma=None, centre="tdr", limit=None, verbose=False,
             cases=None):
    rows = []
    if cases is None:
        cases = []
        for kind in kinds:
            cases += [c for c in rv.tcradar_cases(kind) if held_out(c[0]["storm"], c[0]["year"])]
    if limit:
        cases = cases[:limit]
    for case, vt, vr in cases:
        cen = None
        if centre == "fix":
            cen = (case["fix_lat"], case["fix_lon"])
        o, info = rv.iwg1_obs(case, centre=cen)
        if o is None:
            continue
        z_src = np.array(case["z"])
        vm_t, R_t = ds.vm_and_rmw(vt, z_src)
        if vm_t is None:
            continue
        obs = as_app_obs(o)
        hd = rv.heading_of(case)
        cx = rv.cov_extra(case)
        try:
            fit = m.fit(obs, hd, cx, R_obs=st.observed_rmw(obs), kappa=kappa, gamma=gamma)
        except ValueError:
            continue
        sc = score(m, fit, vt, vr, z_src)
        if sc is None:
            continue
        pk = peak_1km_earth(m, fit, (case["motion_x"], case["motion_y"]))
        row = dict(storm=case["storm"], mission=case["mission"], kind=case["kind"], year=case["year"],
                   vm=vm_t, R=R_t, fitvm=fit.vm, fitR=fit.R, vbt=case.get("vmax"), fl_max=info["fl_max_earth"],
                   z_med=info["z_med"], n=info["n"], peak1=pk, **sc)
        rows.append(row)
        if verbose:
            print(f"{row['storm']:10s} {row['kind']} Vm {vm_t:5.1f}->{fit.vm:5.1f} R {R_t:5.1f}->{fit.R:5.1f} "
                  f"Vt {sc['rmse_vt']:.2f} fl {sc['rmse_fl']:.2f} pk1 {pk:.1f} BT {row['vbt']}", flush=True)
    return rows


def score(m, fit, vt, vr, z_src, max_r=150.0):
    r_src = (np.arange(vt.shape[1]) + 0.5) * 2.0
    az_src = (np.arange(vt.shape[2]) + 0.5) * (360.0 / vt.shape[2])
    Rg, Ag = np.meshgrid(r_src, az_src, indexing="ij")
    sel = Rg <= max_r
    x = (Rg * np.sin(np.radians(Ag)))[sel]
    y = (Rg * np.cos(np.radians(Ag)))[sel]
    s = m.state(fit)
    et, er, efl, eb = [], [], [], []
    for k, zz in enumerate(z_src):
        if zz < 0.5 or zz > 12.0:
            continue
        tv, rvv = vt[k][sel], vr[k][sel]
        ok = np.isfinite(tv) & np.isfinite(rvv)
        if ok.sum() < 20:
            continue
        pt = m.sample(fit, s, x[ok], y[ok], np.full(ok.sum(), zz), 0)
        pr = m.sample(fit, s, x[ok], y[ok], np.full(ok.sum(), zz), 1)
        et.append(pt - tv[ok]); er.append(pr - rvv[ok])
        if 2.5 <= zz <= 3.5:
            efl.append(pt - tv[ok])
        if zz <= 1.0:
            eb.append(pt - tv[ok])
    if not et:
        return None
    cat = np.concatenate
    out = {"rmse_vt": float(np.sqrt(np.mean(cat(et) ** 2))), "rmse_vr": float(np.sqrt(np.mean(cat(er) ** 2))),
           "bias_vt": float(np.mean(cat(et)))}
    out["rmse_fl"] = float(np.sqrt(np.mean(cat(efl) ** 2))) if efl else float("nan")
    out["rmse_bl"] = float(np.sqrt(np.mean(cat(eb) ** 2))) if eb else float("nan")
    return out


def summarize(rows, label=""):
    a = lambda k: np.array([r[k] for r in rows], float)
    vbt = a("vbt") * 0.514444
    pk = a("peak1")
    ok = np.isfinite(vbt) & np.isfinite(pk) & (vbt > 17)
    ratio = float(np.sum(vbt[ok] * pk[ok]) / np.sum(pk[ok] ** 2)) if ok.any() else float("nan")
    err = (ratio * pk[ok] - vbt[ok]) / 0.514444
    fl = a("fl_max")[ok]
    ratio_fl = float(np.sum(vbt[ok] * fl) / np.sum(fl ** 2))
    err_fl = (ratio_fl * fl - vbt[ok]) / 0.514444
    print(f"{label}: {len(rows)} cases  Vt RMSE {np.nanmean(a('rmse_vt')):.2f}  flight level {np.nanmean(a('rmse_fl')):.2f}  "
          f"BL {np.nanmean(a('rmse_bl')):.2f}  Vr {np.nanmean(a('rmse_vr')):.2f}  |Vm err| med "
          f"{np.median(np.abs(a('fitvm') - a('vm'))):.2f}  |lnR err| med {np.median(np.abs(np.log(a('fitR') / a('R')))):.3f}"
          f"  | intensity: ratio {ratio:.3f}, MAE {np.mean(np.abs(err)):.1f} kt, RMSE {np.sqrt(np.mean(err ** 2)):.1f} kt "
          f"(flight-level max x {ratio_fl:.3f}: MAE {np.mean(np.abs(err_fl)):.1f} kt)", flush=True)
    return ratio, err


if __name__ == "__main__":
    m = st.Model(sys.argv[1])
    for kind in ("swath", "merge"):
        rows = evaluate(m, kinds=(kind,), verbose=False)
        summarize(rows, kind)
