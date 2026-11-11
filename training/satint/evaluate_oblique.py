"""
How the estimates fare when Himawari-9 sees a storm obliquely: the network on the app's IR (himawari_pairs.py
with a zenith range, the app's limit lifted) against the same network on TC PRIMED's archive IR for the same
cases -- which for the south-west Indian Ocean and the Arabian Sea is Meteosat's, seen squarely -- by the
zenith angle Himawari sees them at. Southern storms mirrored and four turns averaged, as in the app.

Usage: python evaluate_oblique.py CHECKPOINT.pt DATA_PREFIX PAIRS_DIR OUT.json [--flag 1]
"""
import datetime as dt
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_wsi as ew  # noqa: E402  (2.6.0's satint and goes first; satnet.assemble mirrors southern storms)
import evaluate_satint as ev  # noqa: E402
import satnet as sn  # noqa: E402

UTC = dt.timezone.utc


def main():
    a = sys.argv[1:]
    flag = 1
    if "--flag" in a:
        k = a.index("--flag")
        flag = int(a[k + 1])
        a = a[:k] + a[k + 2:]
    ckpt, prefix, pdir, out = a[:4]
    ev.ROT = 4
    S = dict(np.load(prefix + "_samples.npz"))
    S["basin"] = np.full_like(S["basin"], flag)
    ir_store = np.load(prefix + "_ir.npy", mmap_mode="r")
    mw_store = np.load(prefix + "_mw.npy", mmap_mode="r")
    net = ev.TorchNet(ckpt, threads=2)
    rows, xs = [], []
    for path in sorted(glob.glob(os.path.join(pdir, "*.npz"))):
        d = np.load(path)
        for k, i in enumerate(d["idx"]):
            g = d["ir"][k].astype(np.float64)
            if not np.isfinite(g[0]).mean() > 0.8:
                continue
            t = dt.datetime.fromtimestamp(float(S["time"][i]), UTC)
            frames = [g[0]] + [g[j] if np.isfinite(g[j]).mean() > 0.8 else None for j in range(1, 5)]
            mw = mw_ir = None
            if S["mw"][i] >= 0:
                mw = sn.mw_tb(mw_store[S["mw"][i]])
                mw_ir = g[5] if np.isfinite(g[5]).mean() > 0.8 else None
            lon = ew.sample_lon(S, i)
            xs.append(ew.si.network_inputs(frames, mw, mw_ir, max(S["age"][i], 0.0), S["mw_res"][i], S["mw_cov"][i],
                                           S["lat"][i], S["speed"][i], True, t, lon))
            lat = float(S["lat"][i])
            sat = ew.goes.pick_satellite(lat, lon)
            arch = sn.ir_tb(ir_store[S["ir"][i]])
            ok = np.isfinite(arch) & np.isfinite(g[0])
            rows.append({"i": int(i), "zen": ew.goes.zenith_angle(lat, lon, sat), "sat": sat,
                         "tb_bias": float(np.mean(g[0][ok] - arch[ok])),
                         "cold_bias": float(np.mean((g[0] - arch)[ok & (arch < 220)])) if (ok & (arch < 220)).sum() > 50 else None})
    idx = np.array([r["i"] for r in rows])
    zen = np.array([r["zen"] for r in rows])
    v_g, p_g = ev.run(net, np.concatenate([x[0] for x in xs]), np.concatenate([x[1] for x in xs]),
                      np.concatenate([x[2] for x in xs]))
    a_ir, a_mw, a_sc = ev.archive_inputs(S, idx, ir_store, mw_store)
    v_a, p_a = ev.run(net, a_ir, a_mw, a_sc)
    bt_v = S["vmax"][idx]
    res = {"cases": len(rows), "by_zenith": {}}
    for lo, hi in ((55, 62), (62, 66), (66, 70), (70, 74), (74, 78), (78, 86)):
        m = (zen >= lo) & (zen < hi)
        if m.sum() < 8:
            continue
        tb = [r["tb_bias"] for r, mm in zip(rows, m) if mm]
        cb = [r["cold_bias"] for r, mm in zip(rows, m) if mm and r["cold_bias"] is not None]
        res["by_zenith"][f"{lo}-{hi}"] = {
            "n": int(m.sum()), "app_ir_vmax_mae": float(np.mean(np.abs(v_g[m, 2] - bt_v[m]))),
            "archive_ir_vmax_mae": float(np.mean(np.abs(v_a[m, 2] - bt_v[m]))),
            "app_minus_archive_vmax_mean": float(np.mean(v_g[m, 2] - v_a[m, 2])),
            "app_minus_archive_tb_mean": float(np.mean(tb)), "app_minus_archive_cold_tb_mean": float(np.mean(cb)) if cb else None}
        z = res["by_zenith"][f"{lo}-{hi}"]
        print(f"zenith {lo}-{hi}: n={z['n']:3d}  Vmax MAE app IR {z['app_ir_vmax_mae']:.1f} vs archive IR "
              f"{z['archive_ir_vmax_mae']:.1f}  (app - archive {z['app_minus_archive_vmax_mean']:+.1f} kt; Tb "
              f"{z['app_minus_archive_tb_mean']:+.1f} K, cold cloud {z['app_minus_archive_cold_tb_mean']})", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
