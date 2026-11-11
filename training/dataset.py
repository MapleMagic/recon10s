"""
Build the normalised training set from the reduced TC-RADAR polar grids.

Each analysis becomes one row: motion-relative tangential and radial wind divided by Vm (the
strongest azimuthal-mean tangential wind at 2 km), on nodes of rho = r / R (R: radius of Vm),
azimuth relative to the storm motion (clockwise, 0 = direction of motion) and height.
"""
import glob
import json
import math
import os

import numpy as np

RHO = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5,
                4.0, 5.0, 6.0, 8.0])
NPHI_N = 16
PHI = np.arange(NPHI_N) * (360.0 / NPHI_N)
Z = np.arange(0.5, 12.01, 0.5)
FIELDS = ("vt", "vr")
SRC_DR = 2.0
SRC_NPHI = 32


def src_radii(nr=100):
    return (np.arange(nr) + 0.5) * SRC_DR


def azimuthal_profile(vt_level, min_cover=0.25):
    """Azimuthal mean by radius, NaN where under min_cover of the azimuth bins have data."""
    good = np.isfinite(vt_level)
    cover = good.mean(axis=1)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(np.where(good, vt_level, np.nan), axis=1)
    m[cover < min_cover] = np.nan
    return m


def vm_and_rmw(vt, z_src, min_cover=0.25, height=2.0):
    k = int(np.argmin(np.abs(z_src - height)))
    prof = azimuthal_profile(vt[k], min_cover)
    if not np.isfinite(prof).any():
        return None, None
    j = int(np.nanargmax(prof))
    r = src_radii(prof.size)
    # refine the radius of the maximum with a parabola through the neighbours
    rr = r[j]
    if 0 < j < prof.size - 1 and np.isfinite(prof[j - 1]) and np.isfinite(prof[j + 1]):
        a, b, c = prof[j - 1], prof[j], prof[j + 1]
        den = a - 2 * b + c
        if den < 0:
            rr = r[j] + SRC_DR * 0.5 * (a - c) / den
    return float(prof[j]), float(max(rr, 3.0))


def to_normalised(field, z_src, R, heading, rho=RHO, phi=PHI, z=Z):
    """
    field: (nz_src, nr, 32) on 2-km x 11.25-deg bins, geographic azimuth. Returns (len(z), len(rho), len(phi))
    sampled at r = rho * R, azimuth = phi + heading, by bilinear interpolation with NaN-aware weights.
    """
    nz_s, nr, nph = field.shape
    kz = [int(np.argmin(np.abs(z_src - zz))) for zz in z]
    r_t = rho[:, None] * R + 0 * phi[None, :]
    az_t = (phi[None, :] + heading) % 360.0 + 0 * rho[:, None]
    fr = r_t / SRC_DR - 0.5
    fr = np.clip(fr, 0.0, nr - 1.000001)
    i0 = np.floor(fr).astype(int)
    wr = fr - i0
    fa = az_t / (360.0 / nph) - 0.5
    j0 = np.floor(fa).astype(int)
    wa = fa - j0
    j0 %= nph
    j1 = (j0 + 1) % nph
    i1 = np.minimum(i0 + 1, nr - 1)
    out = np.full((len(z), len(rho), len(phi)), np.nan, np.float32)
    beyond = rho[:, None] * R + 0 * phi[None, :] > nr * SRC_DR
    for n, k in enumerate(kz):
        f = field[k]
        num = np.zeros(r_t.shape)
        den = np.zeros(r_t.shape)
        for ii, wi in ((i0, 1 - wr), (i1, wr)):
            for jj, wj in ((j0, 1 - wa), (j1, wa)):
                v = f[ii, jj]
                w = wi * wj
                ok = np.isfinite(v)
                num += np.where(ok, v * w, 0.0)
                den += np.where(ok, w, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            val = np.where(den >= 0.5, num / den, np.nan)
        val[beyond] = np.nan
        out[n] = val
    return out


def covariates(md_case, vm, R):
    """Predictors known at analysis time (live: from recon, NHC and SHIPS)."""
    mx, my = md_case.get("motion_x"), md_case.get("motion_y")
    ok_m = bool(mx is not None and my is not None and np.isfinite(mx) and np.isfinite(my))
    heading = math.degrees(math.atan2(mx, my)) % 360.0 if ok_m else 0.0
    speed = math.hypot(mx, my) if ok_m else 5.0
    s, sd = md_case.get("shdc"), md_case.get("sddc")
    ok_s = bool(s is not None and sd is not None and np.isfinite(s) and np.isfinite(sd) and s >= 0)
    if ok_s:
        rel = math.radians(sd - heading)
        s_par, s_perp = s * math.cos(rel), s * math.sin(rel)
    else:
        s_par = s_perp = 0.0
    lat = md_case.get("lat")
    lat = abs(lat) if lat is not None and np.isfinite(lat) else 25.0
    return heading, {"vm": float(vm), "R": float(R), "speed": float(speed), "s_par": float(s_par),
                     "s_perp": float(s_perp), "lat": float(lat),
                     "has_motion": ok_m, "has_shear": ok_s}


def load_cases(folder):
    """Every reduced TC-RADAR analysis with valid 2-km data, normalised."""
    rows, meta = [], []
    for path in sorted(glob.glob(os.path.join(folder, "*_[0-9][0-9][0-9][0-9].npz"))):
        stem = os.path.basename(path)
        kind = "merge" if "_merge_" in stem else "swath"
        grid = json.load(open(path.rsplit("_", 1)[0] + "_grid.json"))
        z_src = np.array(grid["z"])
        d = np.load(path)
        md = json.load(open(path.replace(".npz", ".json")))
        vt_all = d["vt"].astype(np.float32)
        vr_all = d["vr"].astype(np.float32)
        for i in range(vt_all.shape[0]):
            vt, vr = vt_all[i], vr_all[i]
            vm, R = vm_and_rmw(vt, z_src)
            if vm is None or vm < 10.0:
                continue
            case = {k: (v[i] if isinstance(v, list) else v) for k, v in md.items()}
            heading, cov = covariates(case, vm, R)
            if not cov["has_motion"]:
                continue
            a = to_normalised(vt, z_src, R, heading) / vm
            b = to_normalised(vr, z_src, R, heading) / vm
            rows.append(np.stack([a, b]).astype(np.float32))
            when = (case.get("year"), case.get("month"), case.get("day"), case.get("hour"), case.get("minute"))
            meta.append({"kind": kind, "mission": case.get("mission"), "storm": case.get("storm"),
                         "tcid": case.get("tcid"), "time": when, "heading": heading, "vbt_kt": case.get("vmax"),
                         "pres": case.get("pres"), "src": stem, "index": i, **cov})
    return np.array(rows), meta


if __name__ == "__main__":
    import sys
    X, meta = load_cases(sys.argv[1])
    print(X.shape, len(meta))
    np.save(sys.argv[2], X)
    with open(sys.argv[2].replace(".npy", ".json"), "w") as fh:
        json.dump(meta, fh)
