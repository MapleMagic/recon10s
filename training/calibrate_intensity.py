"""Final intensity calibration: fitted on every out-of-sample row whose obs reached the core; LOSO scores."""
import json
import numpy as np
KT = 1.943844
HINGE = 96.0
cv = json.load(open("data/cv_rows.json"))
seb = json.load(open("data/seb_rows2.json"))
for r in cv: r["set"] = "tcradar"
for r in seb: r["set"] = "seb"
rows = [r for r in cv + seb if r["vbt"] is not None and r["vbt"] >= 34]
core = np.array([r["rmin"] <= min(1.5 * r["fitR"], 50.0) and r["core_sectors"] >= 2 for r in rows])
vbt = np.array([r["vbt"] for r in rows], float)
pk = np.array([r["peak1"] for r in rows]) * KT
fl = np.array([r["fl_max"] for r in rows]) * KT
sset = np.array([r["set"] for r in rows])
key = np.array([r["storm"] + str(r["year"]) for r in rows])
one = np.ones_like(pk)
hinge = np.maximum(pk - HINGE, 0.0)
full = np.c_[pk, fl, one, hinge]
pkonly = np.c_[pk, one, hinge]


def loso(A, sel):
    pred = np.full(vbt.size, np.nan)
    for s in sorted(set(key[sel])):
        tr = sel & (key != s)
        te = sel & (key == s)
        coef, *_ = np.linalg.lstsq(A[tr], vbt[tr], rcond=None)
        pred[te] = A[te] @ coef
    coef, *_ = np.linalg.lstsq(A[sel], vbt[sel], rcond=None)
    return pred, coef


def stats(e):
    return {"n": int(e.size), "bias": round(float(e.mean()), 1), "mae": round(float(np.abs(e).mean()), 1),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 1), "median_abs": round(float(np.median(np.abs(e))), 1),
            "within5": round(float(np.mean(np.abs(e) <= 5)) * 100), "within10": round(float(np.mean(np.abs(e) <= 10)) * 100)}


out = {"hinge_kt": HINGE}
for name, A in (("full", full), ("pk_only", pkonly)):
    pred, coef = loso(A, core)
    e = pred - vbt
    out[name] = {"coef": [round(float(c), 4) for c in coef],
                 "loso_all": stats(e[core]), "loso_tcradar": stats(e[core & (sset == "tcradar")]),
                 "loso_2025_26": stats(e[core & (sset == "seb")])}
    print(name, out[name])
# rows whose obs did not reach the core, with the full formula fitted on the core rows
coef = np.array(out["full"]["coef"])
e_nc = full[~core] @ coef - vbt[~core]
out["not_core"] = stats(e_nc)
print("not core:", out["not_core"])
# 1-km peak alone through the origin (the simplest reference) and flight-level max x 0.9 (the old rule)
for nm, est in (("pk_ratio", pk * (np.sum(pk[core] * vbt[core]) / np.sum(pk[core] ** 2))), ("fl_x0.9", 0.9 * fl)):
    out[nm] = stats(est[core] - vbt[core])
    print(nm, out[nm])
json.dump(out, open("data/intensity_calibration.json", "w"), indent=1)

# by intensity, core-sampled rows, LOSO predictions of the full formula
pred, _ = loso(full, core)
e = pred - vbt
print("\nby best-track intensity (core sampled, LOSO):")
bins = []
for lo, hi in ((34, 64), (64, 96), (96, 137), (137, 200)):
    sel = core & (vbt >= lo) & (vbt < hi)
    st_ = stats(e[sel]); st_["range"] = [lo, hi]
    bins.append(st_)
    print(f"  {lo:3d}-{hi:3d} kt: {st_}")
out["by_intensity"] = bins
json.dump(out, open("data/intensity_calibration.json", "w"), indent=1)
