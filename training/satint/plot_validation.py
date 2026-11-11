"""
The validation figure: mean wind error by intensity against CIMSS D-PRINT, D-MINT and ADT on the
same cases, held-out 2024-2025 (TC PRIMED archive inputs) and 2026 (the app's GOES pipeline).

Usage: python plot_validation.py EVAL_HELDOUT.json EVAL_2026.json OUT.png
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BINS = ["34-63 kt", "64-95 kt", "96-136 kt", "137+ kt"]
COL = {"this model, newest pass": "#8e24aa", "this model, GMI passes only": "#ba68c8", "this model, IR only": "#f06292",
       "DPRINT": "#607d8b", "DMINT": "#455a64", "ADT": "#9e9e9e"}
LABEL = {"DPRINT": "CIMSS D-PRINT", "DMINT": "CIMSS D-MINT", "ADT": "CIMSS ADT"}


def bars(ax, groups, title, note):
    names = list(groups)
    w = 0.8 / len(names)
    x = np.arange(len(BINS) + 1)
    for k, name in enumerate(names):
        allv, by = groups[name]
        vals = [allv.get("vmax_mae", np.nan)] + [by.get(b, {}).get("vmax_mae", np.nan) for b in BINS]
        ax.bar(x + (k - (len(names) - 1) / 2) * w, vals, w, color=COL.get(name, "#888"), label=LABEL.get(name, name))
    ns = [groups[names[0]][0].get("n", 0)] + [groups[names[0]][1].get(b, {}).get("n", 0) for b in BINS]
    ax.set_xticks(x)
    ax.set_xticklabels([f"all\n(n={ns[0]})"] + [f"{b}\n(n={n})" for b, n in zip(BINS, ns[1:])], fontsize=8)
    ax.set_ylabel("mean |error| in Vmax (kt)")
    ax.set_title(f"{title}\n{note}", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)


def main(heldout, y2026, out):
    h = json.load(open(heldout))
    g = json.load(open(y2026))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    c = h.get("archive_vs_cimss", {})
    for ax, kind in zip(axs[:2], ("DPRINT", "DMINT")):
        if kind not in c:
            ax.axis("off")
            continue
        out_ = c[kind]
        groups = {}
        for name in ("this model, newest pass", "this model, GMI passes only", "this model, IR only"):
            if name in out_:
                by = out_["by_intensity"].get(name)
                if by is None and name == "this model, GMI passes only":
                    continue
                groups[name] = (out_[name], by or {})
        groups[kind] = (out_[kind], out_["by_intensity"][kind])
        bars(ax, groups, f"Held out 2024-2025: at {LABEL[kind]}'s times",
             f"{out_['n']} cases, {out_['storms']} storms; truth: best track (TC PRIMED)")
    c26 = g.get("goes26_vs_cimss", {}).get("DPRINT")
    g26 = g.get("goes26", {})
    ax = axs[2]
    if c26:
        groups = {"this model, IR only": (c26["this model, IR only"], c26["by_intensity"]["this model, IR only"]),
                  "DPRINT": (c26["DPRINT"], c26["by_intensity"]["DPRINT"])}
        bars(ax, groups, "2026 through the app's GOES pipeline, IR only",
             f"{c26['n']} cases, {c26['storms']} storms; truth: NHC working best track")
    else:
        ax.axis("off")
    fig.suptitle("recon10s satellite intensity (experimental): errors against best track on the same cases", fontsize=11)
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
