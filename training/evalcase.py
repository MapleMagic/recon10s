"""One validation case: fit the app's model to the IWG1 obs of a TDR pass and measure it against the TDR."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import recon10s_syntdr as st
import dataset as ds, evaluate as ev, realval as rv

KT = 1.943844


def core_coverage(obs, R):
    """Smallest ob radius, and how many of 8 sectors have an ob within 0.5-1.5 R."""
    r = np.hypot(obs.x, obs.y)
    az = (np.degrees(np.arctan2(obs.x, obs.y)) + 360) % 360
    ring = (r >= 0.5 * R) & (r <= 1.5 * R)
    return float(r.min()), int(len(set((az[ring] // 45).astype(int).tolist())))


def eval_case(m, case, vt, vr, kappa=None, gamma=None, with_z=False):
    o, info = rv.iwg1_obs(case)
    if o is None:
        return None
    z_src = np.array(case["z"])
    vm_t, R_t = ds.vm_and_rmw(vt, z_src)
    if vm_t is None:
        return None
    obs = ev.as_app_obs(o)
    try:
        fit = m.fit(obs, rv.heading_of(case), rv.cov_extra(case), R_obs=st.observed_rmw(obs), kappa=kappa, gamma=gamma)
    except ValueError:
        return None
    sc = ev.score(m, fit, vt, vr, z_src)
    if sc is None:
        return None
    pk = ev.peak_1km_earth(m, fit, (case["motion_x"], case["motion_y"]))
    rmin, sectors = core_coverage(obs, fit.R)
    row = dict(storm=case["storm"], year=int(case["year"]), kind=case["kind"], mission=case["mission"],
               name=case.get("name", ""), vm=vm_t, R=R_t, fitvm=fit.vm, fitR=fit.R, vbt=case.get("vmax"),
               fl_max=info["fl_max_earth"], z_med=info["z_med"], n=info["n"], peak1=pk, rmin=rmin,
               core_sectors=sectors, **sc)
    if with_z:
        import realtime_eval as seb_eval
        row["_z"] = seb_eval.spread_z(m, fit, vt, vr, z_src)
    return row
