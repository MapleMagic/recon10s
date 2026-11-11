"""The validation numbers kept in the model file (train_final.py) and quoted in the README."""
import json
import numpy as np
KT = 1.943844
cv = json.load(open("data/cv_rows.json"))
seb = json.load(open("data/seb_rows2.json"))
cal = json.load(open("data/intensity_calibration.json"))


def field(rows):
    a = lambda k: np.array([r[k] for r in rows], float)
    return {"n": len(rows), "storms": len({r["storm"] + str(r["year"]) for r in rows}),
            "vt_rmse_kt": round(float(np.nanmean(a("rmse_vt"))) * KT, 1),
            "vt_rmse_flight_level_kt": round(float(np.nanmean(a("rmse_fl"))) * KT, 1),
            "vt_rmse_below_1km_kt": round(float(np.nanmean(a("rmse_bl"))) * KT, 1),
            "vr_rmse_kt": round(float(np.nanmean(a("rmse_vr"))) * KT, 1),
            "vm_abs_err_median_kt": round(float(np.median(np.abs(a("fitvm") - a("vm")))) * KT, 1),
            "rmw_abs_err_median_pct": round(float(np.median(np.abs(a("fitR") / a("R") - 1))) * 100)}


z_cv = np.load("data/cv_z.npy")
out = {
    "fields_tcradar_cv_swath": field([r for r in cv if r["kind"] == "swath"]),
    "fields_tcradar_cv_merge": field([r for r in cv if r["kind"] == "merge"]),
    "fields_2025_2026": field(seb),
    "intensity_core_sampled_loso": cal["full"]["loso_all"],
    "intensity_core_sampled_loso_2025_2026": cal["full"]["loso_2025_26"],
    "intensity_core_not_sampled": cal["not_core"],
    "intensity_flight_level_x0.9_rule": cal["fl_x0.9"],
}
for k, v in out.items():
    print(k, v)
json.dump(out, open("data/validation_summary.json", "w"), indent=1)
# keep the 1-km-peak-only ratio for the fallback field
core = [r for r in cv + seb if r["vbt"] and r["vbt"] >= 34 and r["rmin"] <= min(1.5 * r["fitR"], 50) and r["core_sectors"] >= 2]
pk = np.array([r["peak1"] for r in core]) * KT
bt = np.array([r["vbt"] for r in core])
cal["pk_ratio_coef"] = float(np.sum(pk * bt) / np.sum(pk * pk))
json.dump(cal, open("data/intensity_calibration.json", "w"), indent=1)
print("pk ratio", round(cal["pk_ratio_coef"], 3))
