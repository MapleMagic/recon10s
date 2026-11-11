"""Out-of-sample check on the real-time TDR analyses of 2025-2026 with the P-3's own IWG1 data."""
import json, os, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import numpy as np
import recon10s_syntdr as st
import dataset as ds, evaluate as ev, realval as rv

KT = 1.943844


def load_cases(path="data/seb_cases.json"):
    out = []
    for c in json.load(open(path)):
        d = np.load("data/seb/" + c["src"])
        out.append((c, d["vt"].astype(np.float32), d["vr"].astype(np.float32)))
    return out


def spread_z(m, fit, vt, vr, z_src, levels=(1.0, 2.0, 3.0, 5.0), max_r=150.0):
    r_src = (np.arange(vt.shape[1]) + 0.5) * 2.0
    az_src = (np.arange(vt.shape[2]) + 0.5) * (360.0 / vt.shape[2])
    Rg, Ag = np.meshgrid(r_src, az_src, indexing="ij")
    sel = Rg <= max_r
    x = (Rg * np.sin(np.radians(Ag)))[sel]
    y = (Rg * np.cos(np.radians(Ag)))[sel]
    s, sp = m.state(fit), m.spread(fit)
    out = []
    for zz in levels:
        k = int(np.argmin(np.abs(np.asarray(z_src) - zz)))
        tv = vt[k][sel]
        ok = np.isfinite(tv)
        if ok.sum() < 20:
            continue
        p, q = m.sample_many(fit, [(s, 0), (sp, 0)], x[ok], y[ok], np.full(ok.sum(), zz))
        out.append((p - tv[ok]) / np.maximum(q, 0.3))
    return np.concatenate(out) if out else np.empty(0)


def run(model_path, cases, label, kappa=None, gamma=None, verbose=False):
    m = st.Model(model_path)
    rows, zs = [], []
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
            fit = m.fit(obs, rv.heading_of(case), rv.cov_extra(case), R_obs=st.observed_rmw(obs), kappa=kappa, gamma=gamma)
        except ValueError:
            continue
        sc = ev.score(m, fit, vt, vr, z_src)
        if sc is None:
            continue
        zs.append(spread_z(m, fit, vt, vr, z_src))
        pk = ev.peak_1km_earth(m, fit, (case["motion_x"], case["motion_y"]))
        rows.append(dict(storm=case["storm"], mission=case["mission"], name=case["name"], year=case["year"],
                         vm=vm_t, R=R_t, fitvm=fit.vm, fitR=fit.R, vbt=case.get("vmax"), fl_max=info["fl_max_earth"],
                         z_med=info["z_med"], n=info["n"], peak1=pk, shear=case.get("shdc") is not None, **sc))
        if verbose:
            r = rows[-1]
            print(f"{r['name']} {r['storm']:9s} Vm {vm_t:5.1f}->{fit.vm:5.1f} R {R_t:5.1f}->{fit.R:5.1f} Vt {sc['rmse_vt']:.2f} "
                  f"pk1 {pk * KT:5.1f} fl {info['fl_max_earth'] * KT:5.1f} BT {r['vbt']}", flush=True)
    a = lambda k: np.array([r[k] for r in rows], float)
    z = np.concatenate(zs) if zs else np.empty(0)
    print(f"{label}: {len(rows)} analyses ({len(set(r['storm'] + str(r['year']) for r in rows))} storms)  "
          f"Vt RMSE {np.nanmean(a('rmse_vt')):.2f} m/s (flight level {np.nanmean(a('rmse_fl')):.2f}, below 1 km "
          f"{np.nanmean(a('rmse_bl')):.2f}), Vr {np.nanmean(a('rmse_vr')):.2f}; |Vm err| median "
          f"{np.median(np.abs(a('fitvm') - a('vm'))):.2f} m/s; |lnR err| median {np.median(np.abs(np.log(a('fitR') / a('R')))):.3f}; "
          f"spread z rms {np.sqrt(np.mean(z ** 2)):.2f}, |z|<1 {np.mean(np.abs(z) < 1):.2f}, |z|<2 {np.mean(np.abs(z) < 2):.2f}",
          flush=True)
    return rows, z


def intensity(rows, coef, ratio_pk=0.987, ratio_fl=0.924, label=""):
    r = [x for x in rows if x["vbt"] is not None and x["vbt"] > 33]
    vbt = np.array([x["vbt"] for x in r])
    pk = np.array([x["peak1"] for x in r]) * KT
    fl = np.array([x["fl_max"] for x in r]) * KT
    out = {}
    for name, est in (("calibrated", coef[0] * pk + coef[1] * fl + coef[2]), ("1-km peak x %.3f" % ratio_pk, ratio_pk * pk),
                      ("flight-level max x %.3f" % ratio_fl, ratio_fl * fl)):
        e = est - vbt
        out[name] = e
        print(f"  {label} {name:26s} n {len(e)}  bias {e.mean():+.1f}  MAE {np.abs(e).mean():.1f}  RMSE {np.sqrt((e ** 2).mean()):.1f}  "
              f"within 5 kt {np.mean(np.abs(e) <= 5) * 100:.0f}%  within 10 kt {np.mean(np.abs(e) <= 10) * 100:.0f}%")
    return out


if __name__ == "__main__":
    cases = load_cases()
    rows, z = run(sys.argv[1], cases, "2025-2026 real-time", verbose="-v" in sys.argv)
    json.dump(rows, open("data/seb_eval_rows.json", "w"))
    np.save("data/seb_eval_z.npy", z)
    intensity(rows, [0.508, 0.394, 7.794], label="all")
    for yr in (2025, 2026):
        intensity([r for r in rows if r["year"] == yr], [0.508, 0.394, 7.794], label=str(yr))
