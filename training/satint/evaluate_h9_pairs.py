"""
The app's Himawari-9 (and GOES-18) IR against TC PRIMED's archive IR for the same West Pacific /
Southern Hemisphere / Indian Ocean samples (himawari_pairs.py): time offsets, Tb differences (all,
cold cloud, warmest eye pixel), the shift that lines them up, and the network on both for the same
cases, run as the app runs it (southern storms mirrored, four turns).

Usage: python evaluate_h9_pairs.py CHECKPOINT.pt DATA_PREFIX PAIRS_DIR OUT.json [--flag 1] [--png OUT.png]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_wsi  # noqa: E402,F401  (2.6.0's satint first, and satnet.assemble mirroring southern storms)
import evaluate_satint as ev  # noqa: E402
import satnet as sn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("prefix")
    ap.add_argument("pairs")
    ap.add_argument("out")
    ap.add_argument("--flag", type=int, default=1)
    ap.add_argument("--png")
    a = ap.parse_args()
    ev.ROT = 4
    S = dict(np.load(a.prefix + "_samples.npz"))
    S["basin"] = np.full_like(S["basin"], a.flag)
    storms = json.load(open(a.prefix + "_meta.json"))["storms"]
    ir_store = np.load(a.prefix + "_ir.npy", mmap_mode="r")
    mw_store = np.load(a.prefix + "_mw.npy", mmap_mode="r")
    net = ev.TorchNet(a.checkpoint, threads=2)
    res = ev.goes25(net, S, ir_store, mw_store, a.pairs)
    # the scans' timing and the eye, case by case
    rows, pairs = [], []
    rr = np.hypot(*np.meshgrid(np.arange(128) - 63.5, np.arange(128) - 63.5)) * 4.0
    for path in sorted(glob.glob(os.path.join(a.pairs, "*.npz"))):
        d = np.load(path)
        for k, i in enumerate(d["idx"]):
            g = d["ir"][k, 0].astype(float)
            ar = sn.ir_tb(ir_store[S["ir"][i]]).astype(float)
            ok = np.isfinite(g) & np.isfinite(ar)
            if ok.mean() < 0.8:
                continue
            rows.append(dict(storm=storms[S["storm"][i]], vmax=float(S["vmax"][i]), sat=int(d["sat"][k, 0]),
                             dt_min=float(d["t"][k, 0] - S["time"][i]) / 60.0,
                             bias=float(np.mean(g[ok] - ar[ok])), rmsd=float(np.sqrt(np.mean((g[ok] - ar[ok]) ** 2))),
                             eye_g=float(np.nanmax(np.where(rr < 30, g, np.nan))),
                             eye_a=float(np.nanmax(np.where(rr < 30, ar, np.nan))),
                             min_g=float(np.nanmin(np.where(rr < 100, g, np.nan))),
                             min_a=float(np.nanmin(np.where(rr < 100, ar, np.nan)))))
            pairs.append((g, ar, rows[-1]))
    R = {k: np.array([r[k] for r in rows], float) for k in rows[0] if k != "storm"}
    res["cases"] = len(rows)
    res["storms"] = len({r["storm"] for r in rows})
    res["by_satellite"] = {str(s): int((R["sat"] == s).sum()) for s in np.unique(R["sat"])}
    res["scan_minus_archive_min"] = {"median": float(np.median(R["dt_min"])), "p90_abs": float(np.percentile(np.abs(R["dt_min"]), 90))}
    res["tb"] = {"bias_median": float(np.median(R["bias"])), "rmsd_median": float(np.median(R["rmsd"])),
                 "eye_max_diff_median": float(np.nanmedian(R["eye_g"] - R["eye_a"])),
                 "eye_max_diff_mean": float(np.nanmean(R["eye_g"] - R["eye_a"])),
                 "coldest_diff_median": float(np.nanmedian(R["min_g"] - R["min_a"]))}
    strong = R["vmax"] >= 64
    res["tb_typhoons"] = {"n": int(strong.sum()), "eye_max_diff_median": float(np.nanmedian((R["eye_g"] - R["eye_a"])[strong])),
                          "rmsd_median": float(np.median(R["rmsd"][strong]))}
    json.dump(res, open(a.out, "w"), indent=1)
    for k in ("cases", "storms", "by_satellite", "scan_minus_archive_min", "tb", "tb_typhoons"):
        print(k, res[k])
    print("app pipeline:", ev.short(res["goes_pipeline"]))
    print("archive IR:  ", ev.short(res["archive_ir"]))
    print("pipeline minus archive Vmax:", res["goes_minus_archive_vmax"], "Tb:", res["goes_minus_archive_tb"],
          "shift:", res["best_shift_km"])
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel = sorted(pairs, key=lambda p: -p[2]["vmax"])[:3] + [pairs[len(pairs) // 2]]
        fig, axs = plt.subplots(len(sel), 3, figsize=(10, 3.2 * len(sel)))
        for row, (g, ar, r) in zip(axs, sel):
            for ax, im, title in ((row[0], ar, f"archive {r['storm']} {r['vmax']:.0f} kt"),
                                  (row[1], g, f"app, sat {r['sat']} (dt {r['dt_min']:.0f} min)"),
                                  (row[2], g - ar, f"diff bias {r['bias']:.1f} rms {r['rmsd']:.1f}")):
                h = ax.imshow(im, cmap="gray_r" if ax is not row[2] else "RdBu_r", vmin=180 if ax is not row[2] else -20,
                              vmax=300 if ax is not row[2] else 20, origin="lower")
                ax.set_title(title, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(h, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(a.png, dpi=80)


if __name__ == "__main__":
    main()
