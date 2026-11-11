"""
An adjustment of the network's estimates to JTWC's best track, for storms of JTWC's basins: the network
learnt NHC's best tracks, whose pressures come mostly from aircraft, while JTWC's come from its wind-
pressure relationship, in basins with lower environmental pressure. Fitted on the decision years only
(evaluate_wsi.py --decide) and judged on the reported years, which it never saw:

  * MSLP: the median moved by c0 + c1 * (median - 1000 mb) [+ c2 * (|lat| - 20) / 10];
  * Vmax: likewise, if it helps;
  * the 10-90 % (and 25-75 %) ranges widened about the median by one factor each for Vmax and MSLP,
    so the 10-90 % range holds the best track 80 % of the time.

Usage: python fit_jtwc_adjust.py EVAL_WSI.json DATA_PREFIX OUT.json [CIMSS_DIR] [--no-vmax-shift]
With CIMSS_DIR, D-PRINT and D-MINT are compared again with the adjusted estimates. --no-vmax-shift keeps
the Vmax median as the network gives it (only its ranges widened): what the app ships, since a Vmax
shift fitted on 2020-2023 helped weak systems but hurt tropical storms and typhoons in 2024-2025.
"""
import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import evaluate_satint as ev  # noqa: E402


def lsq(X, y):
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    c, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return c


def design(med, lat, kind, terms):
    cols = [np.ones_like(med)]
    if "level" in terms:
        cols.append((med - (1000.0 if kind == "p" else 65.0)) / (50.0 if kind == "v" else 1.0))
    if "lat" in terms:
        cols.append((np.abs(lat) - 20.0) / 10.0)
    return np.stack(cols, 1)


def apply(q, coef, lat, kind, terms, scale=1.0):
    med = q[:, 2]
    shift = design(med, lat, kind, terms) @ coef if coef is not None else 0.0
    out = q - med[:, None]
    out = out * scale + (med + shift)[:, None]
    return out


def coverage(q, t, lo=0, hi=4):
    ok = np.isfinite(t)
    return float(np.mean((t[ok] >= np.minimum(q[ok, lo], q[ok, hi])) & (t[ok] <= np.maximum(q[ok, lo], q[ok, hi]))))


def fit_scale(q, t, target=0.8):
    """Factor on the 10-90 % half-widths that makes the range hold `target` of the truths."""
    lo, hi = 0.5, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2
        c = coverage(apply(q, None, None, None, (), mid), t)
        lo, hi = (mid, hi) if c < target else (lo, mid)
    return (lo + hi) / 2


def main(path, prefix, out, cdir=None, no_vmax_shift=False):
    e = json.load(open(path))
    S = dict(np.load(prefix + "_samples.npz"))
    res = {"decide_years": e["decide"]["years"], "report_years": e["report"]["years"]}
    adjusted = {}
    for key, label in (("", "newest pass"), ("_ir", "IR only")):
        dp, rp = e["decide_pred"], e["pred"]
        di, ri = np.array(dp["idx"]), np.array(rp["idx"])
        # the reported cases the app can estimate (zenith within its limit)
        reach = np.array(rp["zenith"]) <= 70.0
        ri_r = ri[reach]
        dv, dpp = np.array(dp["v" + key]), np.array(dp["p" + key])
        rv, rpp = np.array(rp["v" + key])[reach], np.array(rp["p" + key])[reach]
        d_lat, r_lat = S["lat"][di], S["lat"][ri_r]
        fits = {}
        for kind, dq, rq, dt_, rt_ in (("p", dpp, rpp, S["mslp"][di], S["mslp"][ri_r]),
                                       ("v", dv, rv, S["vmax"][di], S["vmax"][ri_r])):
            best = None
            for terms in ((), ("level",), ("level", "lat")):
                X = design(dq[:, 2], d_lat, kind, terms)
                coef = lsq(X, dt_ - dq[:, 2])
                adj_d = apply(dq, coef, d_lat, kind, terms)
                mae_d = float(np.nanmean(np.abs(adj_d[:, 2] - dt_)))
                if best is None or mae_d < best[0] - 0.05:
                    best = (mae_d, terms, coef)
            mae0_d = float(np.nanmean(np.abs(dq[:, 2] - dt_)))
            mae_d, terms, coef = best
            use = mae0_d - mae_d >= 0.2 and not (kind == "v" and no_vmax_shift)
            if not use:
                terms, coef = (), None
            scale = fit_scale(apply(dq, coef, d_lat, kind, terms), dt_)
            adj_r = apply(rq, coef, r_lat, kind, terms, scale)
            adjusted[(label, kind)] = adj_r
            ts_ = rt_ if kind == "v" else S["vmax"][ri_r]
            strong = ts_ >= 34
            by_int = {}
            for lo_, hi_ in ((0, 34), (34, 64), (64, 96), (96, 137), (137, 999)):
                mm = (S["vmax"][ri_r] >= lo_) & (S["vmax"][ri_r] < hi_)
                if mm.sum() >= 5:
                    by_int[f"{lo_}-{hi_}"] = {"n": int(mm.sum()),
                                              "mae": [round(float(np.nanmean(np.abs(rq[mm, 2] - rt_[mm]))), 2),
                                                      round(float(np.nanmean(np.abs(adj_r[mm, 2] - rt_[mm]))), 2)],
                                              "bias": [round(float(np.nanmean(rq[mm, 2] - rt_[mm])), 2),
                                                       round(float(np.nanmean(adj_r[mm, 2] - rt_[mm])), 2)]}
            fits_extra = {"report_34kt_up": {"n": int(strong.sum()),
                                             "mae": [round(float(np.nanmean(np.abs(rq[strong, 2] - rt_[strong]))), 2),
                                                     round(float(np.nanmean(np.abs(adj_r[strong, 2] - rt_[strong]))), 2)],
                                             "rmse": round(float(np.sqrt(np.nanmean((adj_r[strong, 2] - rt_[strong]) ** 2))), 2),
                                             "within10": round(float(np.nanmean(np.abs(adj_r[strong, 2] - rt_[strong]) <= 10)), 3),
                                             "cover_10_90": round(coverage(adj_r[strong], rt_[strong]), 3)},
                          "report_by_intensity": by_int,
                          "report_rmse": round(float(np.sqrt(np.nanmean((adj_r[:, 2] - rt_) ** 2))), 2),
                          "report_within10": round(float(np.nanmean(np.abs(adj_r[:, 2] - rt_) <= 10)), 3)}
            fits[kind] = {"terms": list(terms), "coef": None if coef is None else [round(float(c), 3) for c in coef],
                          "scale": round(float(scale), 3),
                          "decide_mae": [round(mae0_d, 2), round(mae_d, 2)],
                          "report_mae": [round(float(np.nanmean(np.abs(rq[:, 2] - rt_))), 2),
                                         round(float(np.nanmean(np.abs(adj_r[:, 2] - rt_))), 2)],
                          "report_bias": [round(float(np.nanmean(rq[:, 2] - rt_)), 2),
                                          round(float(np.nanmean(adj_r[:, 2] - rt_)), 2)],
                          "report_cover_10_90": [round(coverage(rq, rt_), 3), round(coverage(adj_r, rt_), 3)],
                          "report_cover_25_75": [round(coverage(rq, rt_, 1, 3), 3), round(coverage(adj_r, rt_, 1, 3), 3)],
                          **fits_extra}
            print(f"{label} {kind}: terms {terms} coef {fits[kind]['coef']} scale {scale:.2f} | decide MAE "
                  f"{mae0_d:.2f} -> {mae_d:.2f} | report MAE {fits[kind]['report_mae']} bias {fits[kind]['report_bias']} "
                  f"in80 {fits[kind]['report_cover_10_90']} in50 {fits[kind]['report_cover_25_75']} | 34 kt+ "
                  f"{fits[kind]['report_34kt_up']}")
        res[label] = fits
        res["report_idx"] = ri_r.tolist()
    if cdir:
        import json as _json
        import evaluate_wsi as ew
        storms = _json.load(open(prefix + "_meta.json"))["storms"]
        ri_r = np.array(res["report_idx"])
        c = ew.compare_cimss(S, storms, ri_r, adjusted[("newest pass", "v")], adjusted[("newest pass", "p")],
                             adjusted[("IR only", "v")], adjusted[("IR only", "p")], cdir)
        res["cimss_adjusted"] = c
        for kind, o in c.items():
            print(f"adjusted vs {kind} ({o['n']} cases, {o['storms']} storms): {kind} {ev.short(o[kind])}")
            print(f"    newest pass: {ev.short(o['this model, newest pass'])}")
            print(f"    IR only:     {ev.short(o['this model, IR only'])}")
            print(f"    paired: {o['paired']}")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*args[:4], no_vmax_shift="--no-vmax-shift" in sys.argv)
