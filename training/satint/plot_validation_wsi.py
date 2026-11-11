"""
The JTWC-basin validation figure: mean wind error by intensity against CIMSS D-PRINT and D-MINT on the
same West Pacific / Southern Hemisphere / North Indian Ocean cases of the reported years (never seen in
training), with the app's mapping onto JTWC's best track (fit_jtwc_adjust.py), and wind and pressure
errors by region.

Usage: python plot_validation_wsi.py EVAL_WSI.json JTWC_ADJUST.json DATA_PREFIX OUT.png
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import recon10s_satint as si  # noqa: E402

BINS = ["34-63 kt", "64-95 kt", "96-136 kt", "137+ kt"]
COL = {"this model, newest pass": "#8e24aa", "this model, IR only": "#f06292", "DPRINT": "#607d8b", "DMINT": "#455a64"}
LABEL = {"DPRINT": "CIMSS D-PRINT", "DMINT": "CIMSS D-MINT"}


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


def main(path, adjust, prefix, out):
    e = json.load(open(path))
    a = json.load(open(adjust))
    S = dict(np.load(prefix + "_samples.npz"))
    storms = json.load(open(prefix + "_meta.json"))["storms"]
    years = e["report"]["years"]
    c = a.get("cimss_adjusted", {})
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, kind in zip(axs[:2], ("DPRINT", "DMINT")):
        if kind not in c:
            ax.axis("off")
            continue
        o = c[kind]
        groups = {name: (o[name], o["by_intensity"][name]) for name in ("this model, newest pass", "this model, IR only")}
        groups[kind] = (o[kind], o["by_intensity"][kind])
        bars(ax, groups, f"{years[0]}-{years[1]}, JTWC's basins: at {LABEL[kind]}'s times",
             f"{o['n']} cases, {o['storms']} storms; truth: JTWC's best track (TC PRIMED)")
    # by region: wind as the network gives it, pressure mapped onto JTWC's best track
    rp = e["pred"]
    reach = np.array(rp["zenith"]) <= si.MAX_ZENITH
    ri = np.array(rp["idx"])[reach]
    v, p = np.array(rp["v"])[reach], np.array(rp["p"])[reach]
    mapped = np.array([si.to_jtwc(v[k], p[k], float(S["lat"][i]))[1][2] for k, i in enumerate(ri)])
    lon = [(S["solar"][i] - ((S["time"][i] % 86400) / 3600.0)) * 15.0 for i in ri]
    lon = (np.array(lon) + 540.0) % 360.0 - 180.0
    reg = []
    for i, lo in zip(ri, lon):
        b = storms[S["storm"][i]][:2]
        reg.append("West Pacific" if b == "WP" else "North Indian Ocean" if b == "IO" else
                   ("Southern Hemisphere,\nAustralia & S Pacific" if (lo >= 90 or lo < -100) else
                    "Southern Hemisphere,\nSW Indian Ocean"))
    reg = np.array(reg)
    names = [r for r in ("West Pacific", "Southern Hemisphere,\nAustralia & S Pacific", "North Indian Ocean",
                         "Southern Hemisphere,\nSW Indian Ocean") if (reg == r).sum() >= 20]
    ax = axs[2]
    x = np.arange(len(names))
    bt_v, bt_p = S["vmax"][ri], S["mslp"][ri]
    vm = [np.mean(np.abs(v[reg == r, 2] - bt_v[reg == r])) for r in names]
    pm = [np.nanmean(np.abs(mapped[reg == r] - bt_p[reg == r])) for r in names]
    ax.bar(x - 0.2, vm, 0.4, color=COL["this model, newest pass"], label="Vmax (kt)")
    ax.bar(x + 0.2, pm, 0.4, color="#26a69a", label="MSLP (mb), mapped onto JTWC's")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={(reg == r).sum()})" for r in names], fontsize=7.5)
    ax.set_ylabel("mean |error|")
    ax.set_title(f"{years[0]}-{years[1]} by region, every overpass image\n(cases the app can estimate; newest pass)",
                 fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle("recon10s satellite intensity (experimental) in basins it never saw: errors against JTWC's best "
                 "track", fontsize=11)
    fig.savefig(out, dpi=110)
    print("wrote", out, "| by region Vmax", dict(zip([n.replace(chr(10), ' ') for n in names], np.round(vm, 2))),
          "MSLP", dict(zip([n.replace(chr(10), ' ') for n in names], np.round(pm, 2))))


if __name__ == "__main__":
    main(*sys.argv[1:5])
