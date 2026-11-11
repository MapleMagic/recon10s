"""
The info block shipped inside satint_v1.npz (shown in the panel's About box), from the training
data and the evaluation reports.

Usage: python make_info.py DATA_PREFIX TRAIN_YEARS EVAL_HELDOUT.json EVAL_GOES.json OUT.json
"""
import datetime as dt
import json
import sys

import numpy as np


def main(prefix, years, heldout, goes, out):
    S = np.load(prefix + "_samples.npz")
    y0, y1 = (int(x) for x in years.split("-"))
    m = (S["year"] >= y0) & (S["year"] <= y1)
    h = json.load(open(heldout))
    g = json.load(open(goes))
    a = h["archive"]["all"]
    a0 = h["archive"]["all_ir_only"]
    hy = h["archive"]["years"]
    lines = [f"Held out {hy[0]}–{hy[1]} ({a['n']:,} estimates, {h['archive']['storms']} storms, model trained without them): "
             f"Vmax mean error {a['vmax_mae']:.1f} kt (RMSE {a['vmax_rmse']:.1f}; IR only {a0['vmax_mae']:.1f}), "
             f"MSLP {a['mslp_mae']:.1f} mb."]
    cm = h.get("archive_vs_cimss", {})
    bits = []
    for kind, label in (("DPRINT", "D-PRINT"), ("DMINT", "D-MINT")):
        if kind in cm:
            o = cm[kind]
            bits.append(f"at {label}'s times {o['this model, newest pass']['vmax_mae']:.1f} kt against {label}'s "
                        f"{o[kind]['vmax_mae']:.1f}")
    if bits:
        lines.append("On the same cases, " + "; ".join(bits) + ".")
    g26 = g.get("goes26", {})
    if g26.get("all", {}).get("n"):
        s = g26["all"]
        txt = (f"2026 through the app's GOES pipeline, IR only ({s['n']:,} estimates, {g26['storms']} storms): "
               f"{s['vmax_mae']:.1f} kt against NHC's working best track")
        c26 = g.get("goes26_vs_cimss", {}).get("DPRINT")
        if c26:
            txt += (f"; at D-PRINT's times {c26['this model, IR only']['vmax_mae']:.1f} kt against D-PRINT's "
                    f"{c26['DPRINT']['vmax_mae']:.1f}")
        if "same_times_as_adt" in g26:
            mm, ad = g26["same_times_as_adt"]["model"], g26["same_times_as_adt"]["adt"]
            txt += f"; at ADT's times {mm['vmax_mae']:.1f} kt against ADT's {ad['vmax_mae']:.1f}"
        lines.append(txt + ".")
    info = {"version": 1, "years": [y0, y1], "samples": int(m.sum()), "storms": int(len(np.unique(S["storm"][m]))),
            "sources": ["TC PRIMED v01r01 (NOAA NESDIS / CIRA), https://doi.org/10.25921/dmy1-0595",
                        "best track as interpolated by TC PRIMED"],
            "trained": dt.date.today().isoformat(),
            "validation": {"summary": " ".join(lines), "heldout": {k: a[k] for k in a if not isinstance(a[k], (list, dict))},
                           "heldout_ir_only": {k: a0[k] for k in a0 if not isinstance(a0[k], (list, dict))},
                           "goes2026": {k: v for k, v in g26.get("all", {}).items()}}}
    json.dump(info, open(out, "w"), indent=1)
    print(info["validation"]["summary"])


if __name__ == "__main__":
    main(*sys.argv[1:6])
