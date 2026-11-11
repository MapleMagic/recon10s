"""
The app's GOES IR against TC PRIMED's archive IR for the same 2025 samples (goes_pairs_2025.py):
time offsets, mean and rms differences, the coldest and warmest pixels, and the shift that lines them up;
a figure of a few pairs.

Usage: python cmp_goes_archive.py DATA_PREFIX GOES_PAIRS_DIR OUT.png
"""
import glob, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import satnet as sn
pre, gdir, png = sys.argv[1], sys.argv[2], sys.argv[3]
S = dict(np.load(pre + "_samples.npz")); ir_store = np.load(pre + "_ir.npy", mmap_mode="r")
meta = json.load(open(pre + "_meta.json"))["storms"]
rows = []
pairs = []
for path in sorted(glob.glob(os.path.join(gdir, "*.npz"))):
    d = np.load(path)
    for k, i in enumerate(d["idx"]):
        g = d["ir"][k, 0].astype(float)
        a = sn.ir_tb(ir_store[S["ir"][i]]).astype(float)
        ok = np.isfinite(g) & np.isfinite(a)
        if ok.mean() < 0.8:
            continue
        dt_s = d["t"][k, 0] - S["time"][i]
        best = (1e9, 0, 0)
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                b = np.roll(np.roll(g, dy, 0), dx, 1)[8:-8, 8:-8]
                aa = a[8:-8, 8:-8]
                m = np.isfinite(b) & np.isfinite(aa)
                r = float(np.mean((b[m] - aa[m]) ** 2))
                if r < best[0]:
                    best = (r, dy, dx)
        c = 64
        rr = np.hypot(*np.meshgrid(np.arange(128) - 63.5, np.arange(128) - 63.5)) * 4
        inner = rr < 100
        rows.append(dict(storm=meta[S["storm"][i]], vmax=float(S["vmax"][i]), dt_min=dt_s / 60,
                         bias=float(np.mean(g[ok] - a[ok])), rmsd=float(np.sqrt(np.mean((g[ok] - a[ok]) ** 2))),
                         cold_bias=float(np.mean((g - a)[ok & (a < 215)])) if (ok & (a < 215)).sum() > 30 else np.nan,
                         eye_g=float(np.nanmax(np.where(rr < 30, g, np.nan))), eye_a=float(np.nanmax(np.where(rr < 30, a, np.nan))),
                         min_g=float(np.nanmin(np.where(inner, g, np.nan))), min_a=float(np.nanmin(np.where(inner, a, np.nan))),
                         shift_y=best[1] * 4.0, shift_x=best[2] * 4.0, rms_shifted=math.sqrt(best[0])))
        pairs.append((g, a, rows[-1]))
R = {k: np.array([r[k] for r in rows], float) for k in rows[0] if k != "storm"}
print(len(rows), "pairs; scan minus archive time (min): median", np.median(R["dt_min"]).round(1), "p90 |dt|", np.percentile(np.abs(R["dt_min"]), 90).round(1))
for k in ("bias", "rmsd", "cold_bias", "rms_shifted"):
    print(f"{k}: median {np.nanmedian(R[k]):.2f} mean {np.nanmean(R[k]):.2f}")
print("eye max Tb: GOES-archive median", np.nanmedian(R["eye_g"] - R["eye_a"]).round(2), "mean", np.nanmean(R["eye_g"] - R["eye_a"]).round(2))
print("coldest inner Tb: GOES-archive median", np.nanmedian(R["min_g"] - R["min_a"]).round(2))
print("best shift km (y north-down rows, x): median", np.median(R["shift_y"]), np.median(R["shift_x"]), "mean |shift|", np.mean(np.hypot(R["shift_y"], R["shift_x"])).round(1))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sel = sorted(pairs, key=lambda p: -p[2]["vmax"])[:3] + [pairs[len(pairs) // 2]]
fig, axs = plt.subplots(len(sel), 3, figsize=(10, 3.2 * len(sel)))
for row, (g, a, r) in zip(axs, sel):
    for ax, im, title in ((row[0], a, f"archive {r['storm']} {r['vmax']:.0f} kt"), (row[1], g, f"GOES pipeline (dt {r['dt_min']:.0f} min)"), (row[2], g - a, f"diff bias {r['bias']:.1f} rms {r['rmsd']:.1f}")):
        h = ax.imshow(im, cmap="gray_r" if ax is not row[2] else "RdBu_r", vmin=180 if ax is not row[2] else -20, vmax=300 if ax is not row[2] else 20, origin="lower")
        ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(h, ax=ax, shrink=0.7)
fig.tight_layout(); fig.savefig(png, dpi=80)
