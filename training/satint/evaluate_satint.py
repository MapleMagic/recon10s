"""
Evaluate an exported satellite intensity model (the app's numpy format) the way the app runs it.

  archive  TC PRIMED samples of some years (inputs as in training), with and without microwave
  goes25   the same 2025 samples with IR through the app's GOES pipeline (goes_pairs_2025.py)
  goes26   2026 storms, IR only, frames every 3 h through the app's pipeline (goes_frames_2026.py),
           against NHC's best track and against CIMSS ADT at the same times

Usage: python evaluate_satint.py MODEL.npz DATA_PREFIX OUT.json [--years 2024-2025] [--goes25 DIR]
                                 [--goes26 DIR --bdecks FILE] [--adt]
"""
import argparse
import datetime as dt
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import recon10s_satint as si  # noqa: E402
import satnet as sn  # noqa: E402

UTC = dt.timezone.utc
TROPICAL = {"TD", "TS", "HU", "TY", "ST", "SD", "SS", "TC"}


# ------------------------------------------------------------------ scores

def scores(v, p, bt_v, bt_p):
    """v, p: (N, 5) quantiles; bt_v, bt_p: truth (NaN where unknown)."""
    out = {"n": int(len(bt_v))}
    if not len(bt_v):
        return out
    e = v[:, 2] - bt_v
    ok = np.isfinite(e)
    e = e[ok]
    if not len(e):
        return out
    cover = lambda lo, hi, t: float(np.mean((t >= lo) & (t <= hi))) if np.isfinite(lo).all() and np.isfinite(hi).all() \
        else None
    out.update(vmax_mae=float(np.mean(np.abs(e))), vmax_rmse=float(np.sqrt(np.mean(e ** 2))),
               vmax_bias=float(np.mean(e)), vmax_median_ae=float(np.median(np.abs(e))),
               vmax_within10=float(np.mean(np.abs(e) <= 10)), vmax_within5=float(np.mean(np.abs(e) <= 5)),
               vmax_cover_10_90=cover(v[ok, 0], v[ok, 4], bt_v[ok]),
               vmax_cover_25_75=cover(v[ok, 1], v[ok, 3], bt_v[ok]),
               vmax_width_10_90=float(np.mean(v[ok, 4] - v[ok, 0])) if np.isfinite(v[ok, 0]).all() else None)
    if bt_p is not None:
        okp = np.isfinite(bt_p) & np.isfinite(p[:, 2])
        if okp.any():
            ep = p[okp, 2] - bt_p[okp]
            out.update(n_mslp=int(okp.sum()), mslp_mae=float(np.mean(np.abs(ep))),
                       mslp_rmse=float(np.sqrt(np.mean(ep ** 2))), mslp_bias=float(np.mean(ep)),
                       mslp_cover_10_90=cover(p[okp, 0], p[okp, 4], bt_p[okp]))
    return out


def by_bins(v, p, bt_v, bt_p, key, edges, labels):
    out = {}
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = (key >= lo) & (key < hi)
        if m.sum() >= 5:
            out[lab] = scores(v[m], p[m], bt_v[m], bt_p[m] if bt_p is not None else None)
    return out


VBINS = ([0, 34, 64, 96, 137, 999], ["<34 kt", "34-63 kt", "64-95 kt", "96-136 kt", "137+ kt"])


def short(d):
    if "vmax_mae" not in d:
        return f"n={d.get('n', 0)}"
    s = (f"n={d['n']} Vmax MAE {d['vmax_mae']:.1f} RMSE {d['vmax_rmse']:.1f} bias {d['vmax_bias']:+.1f} "
         f"<=10kt {100 * d['vmax_within10']:.0f}%")
    if d.get("vmax_cover_10_90") is not None:
        s += f" in80 {100 * d['vmax_cover_10_90']:.0f}%"
    if d.get("vmax_cover_25_75") is not None:
        s += f" in50 {100 * d['vmax_cover_25_75']:.0f}%"
    if "mslp_mae" in d:
        s += f" | MSLP (n={d['n_mslp']}) MAE {d['mslp_mae']:.1f} RMSE {d['mslp_rmse']:.1f} bias {d['mslp_bias']:+.1f}"
    return s


# ------------------------------------------------------------ the network

ROT = 1


class TorchNet:
    """The PyTorch checkpoint behind an exported model, with the numpy Net's predict(): the same
    numbers (check_export.py) many times faster, for evaluating thousands of cases."""

    def __init__(self, pt, info=None, threads=2):
        import torch
        torch.set_num_threads(threads)
        self.models = []
        for path in pt.split(","):                    # several checkpoints: their average, as the app does
            ck = torch.load(path, map_location="cpu", weights_only=False)
            m = sn.SatNet(ck.get("width", 1.0))
            m.load_state_dict(ck["state"])
            m.eval()
            self.models.append(m.to(memory_format=torch.channels_last))
        self.info = info or {}
        self.torch = torch

    def predict(self, ir, mw, sc, rotations=1):
        torch = self.torch
        outs = []
        with torch.no_grad():
            a, b, c = (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)) for x in (ir, mw, sc))
            for r in range(max(1, int(rotations))):
                ar = torch.rot90(a, r, dims=(2, 3)) if r else a
                br = torch.rot90(b, r, dims=(2, 3)) if r else b
                ar = ar.contiguous(memory_format=torch.channels_last)
                br = br.contiguous(memory_format=torch.channels_last)
                for m in self.models:
                    outs.append(m(ar, br, c).numpy())
        o = np.mean(outs, 0)
        return o[:, :5] * sn.V_SCALE + sn.V_MEAN, o[:, 5:] * sn.P_SCALE + sn.P_MEAN


def run(net, ir, mw, sc, bs=64):
    vs, ps = [], []
    for k in range(0, len(ir), bs):
        v, p = net.predict(ir[k:k + bs], mw[k:k + bs], sc[k:k + bs], ROT)
        vs.append(v)
        ps.append(p)
    return np.concatenate(vs), np.concatenate(ps)


def archive_inputs(S, idx, ir_store, mw_store, ir_only=False):
    ir, mw, sc = sn.assemble(S, np.asarray(idx), ir_store, mw_store)
    if ir_only:
        mw[:] = 0.0
        sc[:, 0:4] = 0.0
    return ir, mw, sc


def predict_samples(net, S, idx, ir_store, mw_store, cand=None, ir_only=False, chunk=256):
    """The network on archive samples, a chunk at a time (all of 2024-2025 at once is gigabytes)."""
    idx = np.asarray(idx)
    vs, ps = [], []
    for k in range(0, len(idx), chunk):
        part = idx[k:k + chunk]
        c = None if cand is None else cand[k:k + chunk]
        ir, mw, sc = sn.assemble(S, part, ir_store, mw_store, c)
        if ir_only:
            mw[:] = 0.0
            sc[:, 0:4] = 0.0
        v, p = run(net, ir, mw, sc)
        vs.append(v)
        ps.append(p)
    return np.concatenate(vs), np.concatenate(ps)


def consistency(S, idx, ir_store, mw_store):
    """The training inputs (satnet.assemble) against the app's (recon10s_satint.network_inputs)."""
    worst = 0.0
    for i in idx:
        a = sn.assemble(S, np.array([i]), ir_store, mw_store)
        cur = sn.ir_tb(ir_store[S["ir"][i]])
        frames = [cur] + [sn.ir_tb(ir_store[S[h][i]]) if S[h][i] >= 0 else None for h in ("h3", "h6", "h9", "h12")]
        mw = None
        mw_ir = None
        if S["mw"][i] >= 0:
            mw = sn.mw_tb(mw_store[S["mw"][i]])
            mw_ir = sn.ir_tb(ir_store[S["mw_ir"][i]]) if S["mw_ir"][i] >= 0 else None
        t = dt.datetime.fromtimestamp(float(S["time"][i]), UTC)
        lon = (S["solar"][i] - (t.hour + t.minute / 60 + t.second / 3600)) * 15.0
        b = si.network_inputs(frames, mw, mw_ir, max(S["age"][i], 0.0), S["mw_res"][i], S["mw_cov"][i], S["lat"][i],
                              S["speed"][i], bool(S["basin"][i]), t, lon)
        for x, y in zip(a, b):
            worst = max(worst, float(np.max(np.abs(x - y))))
    return worst


# ---------------------------------------------------------------- archive

def archive(net, S, ir_store, mw_store, years):
    y0, y1 = years
    idx = np.flatnonzero((S["year"] >= y0) & (S["year"] <= y1))
    bt_v, bt_p = S["vmax"][idx], S["mslp"][idx]
    res = {"years": [y0, y1], "storms": int(len(np.unique(S["storm"][idx])))}
    v, p = predict_samples(net, S, idx, ir_store, mw_store)
    v0, p0 = predict_samples(net, S, idx, ir_store, mw_store, ir_only=True)
    has = S["mw"][idx] >= 0
    if "mwg" in S:
        # as the app would have it: the newest GMI pass only (of its three sensors, TC PRIMED has GMI)
        vg, pg = predict_samples(net, S, idx, ir_store, mw_store, np.array(["gmi"] * len(idx), dtype=object))
        res["all_gmi_only"] = scores(vg, pg, bt_v, bt_p)
        hg = S["mwg"][idx] >= 0
        res["with_gmi_pass"] = scores(vg[hg], pg[hg], bt_v[hg], bt_p[hg])
        res["with_gmi_pass_ir_only"] = scores(v0[hg], p0[hg], bt_v[hg], bt_p[hg])
        res["pred_gmi"] = {"v": vg.round(2).tolist(), "p": pg.round(2).tolist()}
    res["all"] = scores(v, p, bt_v, bt_p)
    res["all_ir_only"] = scores(v0, p0, bt_v, bt_p)
    res["with_pass"] = scores(v[has], p[has], bt_v[has], bt_p[has])
    res["with_pass_ir_only"] = scores(v0[has], p0[has], bt_v[has], bt_p[has])
    res["by_intensity"] = by_bins(v, p, bt_v, bt_p, bt_v, *VBINS)
    res["by_intensity_ir_only"] = by_bins(v0, p0, bt_v, bt_p, bt_v, *VBINS)
    age = np.where(has, S["age"][idx], 99)
    res["by_pass_age"] = by_bins(v, p, bt_v, bt_p, age, [0, 1, 3, 6, 12.01, 100],
                                 ["0-1 h", "1-3 h", "3-6 h", "6-12 h", "no pass"])
    # the same samples given an older pass on purpose (the youngest in each age bin), against IR only
    res["by_forced_pass_age"] = {}
    if "mwc" in S:
        for b, lab in enumerate(["0-2 h", "2-4 h", "4-7 h", "7-12 h"]):
            sub = np.flatnonzero(S["mwc"][idx, b] >= 0)
            if len(sub) < 20:
                continue
            ii = idx[sub]
            vb, pb = predict_samples(net, S, ii, ir_store, mw_store, np.full(len(ii), b))
            res["by_forced_pass_age"][lab] = {"with_pass": scores(vb, pb, S["vmax"][ii], S["mslp"][ii]),
                                              "ir_only": scores(v0[sub], p0[sub], S["vmax"][ii], S["mslp"][ii])}
    if "dv12" in S:
        dv = np.nan_to_num(S["dv12"][idx], nan=-999)
        res["by_change_12h"] = by_bins(v, p, bt_v, bt_p, dv, [-998, -10, 10, 20, 999],
                                       ["weakening 10+ kt", "steady", "strengthening 10-19 kt", "strengthening 20+ kt"])
    res["by_basin"] = {name: scores(v[m], p[m], bt_v[m], bt_p[m])
                       for name, m in (("Atlantic", S["basin"][idx] == 0), ("east/central Pacific", S["basin"][idx] == 1))}
    res["by_year"] = {int(y): scores(v[S["year"][idx] == y], p[S["year"][idx] == y], bt_v[S["year"][idx] == y],
                                     bt_p[S["year"][idx] == y]) for y in np.unique(S["year"][idx])}
    # intensity change over the previous 12 h (from the samples themselves, where there is one)
    res["pred"] = {"idx": idx.tolist(), "v": v.round(2).tolist(), "p": p.round(2).tolist(),
                   "v_ir": v0.round(2).tolist(), "p_ir": p0.round(2).tolist()}
    return res


# ------------------------------------------------------- D-PRINT / D-MINT

MONTHS = {m: i + 1 for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}


def read_cimss(path):
    """[(epoch, vmax, mslp, sensor)] from a CIMSS D-PRINT or D-MINT history file."""
    out = []
    for line in open(path, encoding="latin-1"):
        tok = line.split()
        if len(tok) not in (8, 9) or len(tok[0]) != 9 or not tok[0][:4].isdigit():
            continue
        try:
            t = dt.datetime(int(tok[0][:4]), MONTHS[tok[0][4:7]], int(tok[0][7:9]), int(tok[1][:2]), int(tok[1][2:4]),
                            tzinfo=UTC)
            sensor = tok[4] if len(tok) == 9 else ""
            p = float(tok[-4]) if tok[-4].lower() != "nan" else np.nan
            v, q25, q75 = float(tok[-3]), float(tok[-2]), float(tok[-1])
        except (ValueError, KeyError):
            continue
        if 10 <= v <= 200:
            out.append((t.timestamp(), v, p if 850 < p < 1030 else np.nan, sensor, q25, q75))
    return out


def cimss_q(rows, k_v=0, k_25=None, k_75=None):
    """(N, 5) quantile array for CIMSS values: 25th, median (their 30-70th mean) and 75th; no 10/90."""
    out = np.full((len(rows), 5), np.nan)
    for i, r in enumerate(rows):
        out[i, 2] = r[k_v]
        if k_25 is not None:
            out[i, 1], out[i, 3] = r[k_25], r[k_75]
    return out


def cimss_file(cdir, kind, sid):
    sid = sid.upper()
    code = sid[2:4] + {"AL": "L", "EP": "E", "CP": "C"}[sid[:2]]
    return os.path.join(cdir, f"{kind}hist_{sid[4:8]}_{code}.txt")


def paired(err_a, err_b, groups, n_boot=2000, seed=0):
    """Mean |err_a| - |err_b| with a 95 % interval from resampling whole storms (cases within a storm
    are not independent), and how often a is closer."""
    da = np.abs(err_a) - np.abs(err_b)
    ok = np.isfinite(da)
    da, groups = da[ok], np.asarray(groups)[ok]
    if not len(da):
        return {}
    uniq = np.unique(groups)
    idx = [np.flatnonzero(groups == g) for g in uniq]
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[k] for k in pick])
        boots.append(da[sel].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_abs_diff": float(da.mean()), "ci95": [float(lo), float(hi)],
            "share_closer": float(np.mean(np.abs(err_a[ok]) < np.abs(err_b[ok]))), "storms": int(len(uniq))}


def compare_archive(S, storms, pred, cdir, tol_min=45.0):
    """D-PRINT and D-MINT against this model on the held-out samples nearest their times."""
    idx = np.array(pred["idx"])
    cols = {"latest": np.array(pred["v"]), "ir": np.array(pred["v_ir"]),
            "gmi": np.array(pred["pred_gmi"]["v"]) if "pred_gmi" in pred else None}
    pcols = {"latest": np.array(pred["p"]), "ir": np.array(pred["p_ir"]),
             "gmi": np.array(pred["pred_gmi"]["p"]) if "pred_gmi" in pred else None}
    res = {}
    for kind in ("DPRINT", "DMINT"):
        rows = []
        for si_ in np.unique(S["storm"][idx]):
            sid = storms[si_]
            path = cimss_file(cdir, kind, sid)
            if not os.path.exists(path):
                continue
            mine = np.flatnonzero(S["storm"][idx] == si_)
            ts = S["time"][idx][mine]
            order = np.argsort(ts)
            mine, ts = mine[order], ts[order]
            bt_v_s, bt_p_s = S["vmax"][idx][mine], S["mslp"][idx][mine]
            for t, v, p, _sensor, q25, q75 in read_cimss(path):
                j = int(np.argmin(np.abs(ts - t)))
                if abs(ts[j] - t) > tol_min * 60:
                    continue
                # truth at the CIMSS time, between the samples either side (TC PRIMED's best track)
                k = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
                if not (ts[k - 1] <= t <= ts[k]) or ts[k] - ts[k - 1] > 6 * 3600:
                    bt_v_c, bt_p_c = bt_v_s[j], bt_p_s[j]
                else:
                    f = (t - ts[k - 1]) / max(ts[k] - ts[k - 1], 1.0)
                    bt_v_c = bt_v_s[k - 1] + f * (bt_v_s[k] - bt_v_s[k - 1])
                    bt_p_c = bt_p_s[k - 1] + f * (bt_p_s[k] - bt_p_s[k - 1])
                i = mine[j]
                rows.append((v, p, bt_v_c, bt_p_c, i, sid, q25, q75))
        if len(rows) < 10:
            continue
        cv = np.array([r[0] for r in rows]); cp = np.array([r[1] for r in rows])
        btc_v = np.array([r[2] for r in rows]); btc_p = np.array([r[3] for r in rows])
        ii = np.array([r[4] for r in rows])
        bt_v, bt_p = S["vmax"][idx][ii], S["mslp"][idx][ii]
        rep = lambda a: np.repeat(a[:, None], 5, 1)
        cq = cimss_q(rows, 0, 6, 7)
        out = {"n": len(rows), "storms": len({r[5] for r in rows}),
               kind: scores(cq, rep(cp), btc_v, btc_p)}
        for name in ("ir", "latest", "gmi"):
            if cols[name] is not None:
                out["this model, " + {"ir": "IR only", "latest": "newest pass", "gmi": "GMI passes only"}[name]] = \
                    scores(cols[name][ii], pcols[name][ii], bt_v, bt_p)
        out["by_intensity"] = {kind: by_bins(cq, rep(cp), btc_v, btc_p, btc_v, *VBINS),
                               "this model, IR only": by_bins(cols["ir"][ii], pcols["ir"][ii], bt_v, bt_p, bt_v, *VBINS),
                               "this model, newest pass": by_bins(cols["latest"][ii], pcols["latest"][ii], bt_v, bt_p,
                                                                  bt_v, *VBINS)}
        if cols["gmi"] is not None:
            out["by_intensity"]["this model, GMI passes only"] = by_bins(cols["gmi"][ii], pcols["gmi"][ii], bt_v, bt_p,
                                                                         bt_v, *VBINS)
        groups = [r[5] for r in rows]
        out["paired_vs_" + kind] = {name: paired(cols[c][ii][:, 2] - bt_v, cv - btc_v, groups)
                                    for name, c in (("IR only", "ir"), ("newest pass", "latest"), ("GMI passes only", "gmi"))
                                    if cols[c] is not None}
        res[kind] = out
    return res


def compare_goes26(pred, cdir, bdecks, tol_min=30.0):
    """D-PRINT against this model's IR-only estimates at the same 3-hourly times of 2026; D-MINT at
    its own times against the nearest of them (this model without microwave there)."""
    d_all = json.load(open(bdecks))
    by_sid = {}
    for r in pred:
        by_sid.setdefault(r["sid"], []).append(r)
    res = {}
    for kind, tol in (("DPRINT", tol_min), ("DMINT", 90.0)):
        rows = []
        for sid, rs in by_sid.items():
            path = cimss_file(cdir, kind, sid)
            if not os.path.exists(path):
                continue
            ts = np.array([r["t"] for r in rs])
            track = [x for x in d_all[sid]["track"] if x[3] is not None]
            for t, v, p, _sensor, q25, q75 in read_cimss(path):
                j = int(np.argmin(np.abs(ts - t)))
                if abs(ts[j] - t) > tol * 60:
                    continue
                tr = track_at(track, dt.datetime.fromtimestamp(t, UTC))
                if tr is None or str(tr[2]).upper() not in TROPICAL:
                    continue
                rows.append((v, p, tr[0], tr[1], rs[j], q25, q75))
        if len(rows) < 10:
            continue
        rep = lambda a: np.repeat(np.asarray(a, float)[:, None], 5, 1)
        cv = cimss_q(rows, 0, 5, 6); cp = np.array([r[1] for r in rows])
        btc_v = np.array([r[2] for r in rows], float); btc_p = np.array([r[3] for r in rows], float)
        mv = np.array([r[4]["v"] for r in rows]); mp = np.array([r[4]["p"] for r in rows])
        bt_v = np.array([r[4]["bt_v"] for r in rows], float); bt_p = np.array([r[4]["bt_p"] for r in rows], float)
        res[kind] = {"n": len(rows), "storms": len({r[4]["sid"] for r in rows}),
                     kind: scores(cv, rep(cp), btc_v, btc_p),
                     "this model, IR only": scores(mv, mp, bt_v, bt_p),
                     "paired_vs_" + kind: {"IR only": paired(mv[:, 2] - bt_v, cv[:, 2] - btc_v,
                                                             [r[4]["sid"] for r in rows])},
                     "by_intensity": {kind: by_bins(cv, rep(cp), btc_v, btc_p, btc_v, *VBINS),
                                      "this model, IR only": by_bins(mv, mp, bt_v, bt_p, bt_v, *VBINS)}}
    return res


# ----------------------------------------------------------------- goes25

def goes25(net, S, ir_store, mw_store, gdir):
    import glob
    rows = []
    diffs = {"all": [], "cold": [], "warm": [], "shift": []}
    for path in sorted(glob.glob(os.path.join(gdir, "*.npz"))):
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
            lon = (S["solar"][i] - (t.hour + t.minute / 60 + t.second / 3600)) * 15.0
            x = si.network_inputs(frames, mw, mw_ir, max(S["age"][i], 0.0), S["mw_res"][i], S["mw_cov"][i],
                                  S["lat"][i], S["speed"][i], bool(S["basin"][i]), t, lon)
            rows.append((i, x))
            a = sn.ir_tb(ir_store[S["ir"][i]])
            ok = np.isfinite(a) & np.isfinite(g[0])
            dd = g[0][ok] - a[ok]
            diffs["all"].append(float(np.mean(dd)))
            cold = ok & (a < 220)
            if cold.sum() > 50:
                diffs["cold"].append(float(np.mean(g[0][cold] - a[cold])))
            warm = ok & (a > 280)
            if warm.sum() > 50:
                diffs["warm"].append(float(np.mean(g[0][warm] - a[warm])))
            # the shift (km) that lines the two up best
            best = (1e9, 0, 0)
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    b = np.roll(np.roll(g[0], dy, 0), dx, 1)[8:-8, 8:-8]
                    aa = a[8:-8, 8:-8]
                    m = np.isfinite(b) & np.isfinite(aa)
                    if m.mean() > 0.8:
                        r = float(np.mean((b[m] - aa[m]) ** 2))
                        if r < best[0]:
                            best = (r, dy, dx)
            diffs["shift"].append([best[1] * 4.0, best[2] * 4.0, math.sqrt(best[0])])
    if not rows:
        return {"n": 0}
    idx = np.array([r[0] for r in rows])
    ir = np.concatenate([r[1][0] for r in rows])
    mw = np.concatenate([r[1][1] for r in rows])
    sc = np.concatenate([r[1][2] for r in rows])
    v_g, p_g = run(net, ir, mw, sc)
    a_ir, a_mw, a_sc = archive_inputs(S, idx, ir_store, mw_store)
    v_a, p_a = run(net, a_ir, a_mw, a_sc)
    bt_v, bt_p = S["vmax"][idx], S["mslp"][idx]
    sh = np.array(diffs["shift"])
    res = {"goes_pipeline": scores(v_g, p_g, bt_v, bt_p), "archive_ir": scores(v_a, p_a, bt_v, bt_p),
           "goes_minus_archive_vmax": {"mean": float(np.mean(v_g[:, 2] - v_a[:, 2])),
                                       "mae": float(np.mean(np.abs(v_g[:, 2] - v_a[:, 2]))),
                                       "p90_abs": float(np.percentile(np.abs(v_g[:, 2] - v_a[:, 2]), 90))},
           "goes_minus_archive_tb": {k: float(np.mean(diffs[k])) if diffs[k] else None for k in ("all", "cold", "warm")},
           "best_shift_km": {"median_dy": float(np.median(sh[:, 0])), "median_dx": float(np.median(sh[:, 1])),
                             "mean_abs": float(np.mean(np.hypot(sh[:, 0], sh[:, 1]))), "rms_tb_after": float(np.median(sh[:, 2]))},
           "goes_by_intensity": by_bins(v_g, p_g, bt_v, bt_p, bt_v, *VBINS),
           "archive_by_intensity": by_bins(v_a, p_a, bt_v, bt_p, bt_v, *VBINS)}
    return res


# ----------------------------------------------------------------- goes26

def track_at(rows, t):
    """(vmax, mslp, lat, lon, kind, speed kt) from 6-hourly b-deck rows at time t, or None."""
    ts = np.array([dt.datetime.fromisoformat(r[0]).timestamp() for r in rows])
    x = t.timestamp()
    if x < ts[0] or x > ts[-1]:
        return None
    j = int(np.clip(np.searchsorted(ts, x), 1, len(ts) - 1))
    a, b = rows[j - 1], rows[j]
    f = (x - ts[j - 1]) / (ts[j] - ts[j - 1]) if ts[j] > ts[j - 1] else 0.0
    if a[3] is None or b[3] is None:
        return None
    v = a[3] + f * (b[3] - a[3])
    p = (a[4] + f * (b[4] - a[4])) if a[4] and b[4] else np.nan
    kind = (a[5] if f < 0.5 else b[5]) or ""
    xk, yk = si.local_xy(a[1], a[2], b[1], b[2])
    hrs = (ts[j] - ts[j - 1]) / 3600.0
    sp = math.hypot(xk, yk) / max(hrs, 1e-6) / 1.852
    return v, p, kind, sp


def goes26(net, gdir, bdecks, adt=False):
    import glob
    d_all = json.load(open(bdecks))
    out_rows = []
    adt_cache = {}
    for path in sorted(glob.glob(os.path.join(gdir, "*.npz"))):
        sid = os.path.basename(path)[:-4]
        d = np.load(path)
        if not len(d["time"]):
            continue
        rows = [r for r in d_all[sid]["track"] if r[3] is not None]
        times = d["time"]
        grids = d["ir"]
        if adt and sid not in adt_cache:
            try:
                import recon10s_verify as vf
                ests, _url, _txt = vf.fetch_adt(sid)
            except Exception:
                ests = []
            adt_cache[sid] = [(e.time.timestamp(), e.vmax, e.mslp) for e in ests if e.time is not None]
        for k, t0 in enumerate(times):
            t = dt.datetime.fromtimestamp(float(t0), UTC)
            tr = track_at(rows, t)
            if tr is None or str(tr[2]).upper() not in TROPICAL:
                continue
            cur = sn.ir_tb(grids[k])
            if not np.isfinite(cur).mean() > 0.8:
                continue
            frames = [cur]
            for lag in (3, 6, 9, 12):
                j = int(np.argmin(np.abs(times - (t0 - lag * 3600))))
                ok = abs(times[j] - (t0 - lag * 3600)) <= 1800
                frames.append(sn.ir_tb(grids[j]) if ok else None)
            lat, lon = d["centre"][k]
            x = si.network_inputs(frames, None, None, 0.0, 0.0, 0.0, lat, tr[3], not sid.startswith("al"), t, lon)
            a_v = a_p = np.nan
            if adt:
                cands = [e for e in adt_cache.get(sid, []) if abs(e[0] - t0) <= 1800]
                if cands:
                    e = min(cands, key=lambda e: abs(e[0] - t0))
                    a_v = e[1] if e[1] is not None else np.nan
                    a_p = e[2] if e[2] is not None else np.nan
            out_rows.append((sid, t0, x, tr[0], tr[1], a_v, a_p, sum(f is not None for f in frames[1:])))
    if not out_rows:
        return {"n": 0}
    ir = np.concatenate([r[2][0] for r in out_rows])
    mw = np.concatenate([r[2][1] for r in out_rows])
    sc = np.concatenate([r[2][2] for r in out_rows])
    v, p = run(net, ir, mw, sc)
    bt_v = np.array([r[3] for r in out_rows], float)
    bt_p = np.array([r[4] for r in out_rows], float)
    a_v = np.array([r[5] for r in out_rows], float)
    a_p = np.array([r[6] for r in out_rows], float)
    res = {"storms": len({r[0] for r in out_rows}), "all": scores(v, p, bt_v, bt_p),
           "by_intensity": by_bins(v, p, bt_v, bt_p, bt_v, *VBINS),
           "by_basin": {name: scores(v[m], p[m], bt_v[m], bt_p[m]) for name, m in
                        (("Atlantic", np.array([r[0].startswith("al") for r in out_rows])),
                         ("east/central Pacific", np.array([not r[0].startswith("al") for r in out_rows])))}}
    if adt:
        m = np.isfinite(a_v)
        if m.sum() >= 10:
            av = np.full((int(m.sum()), 5), np.nan)
            ap = np.full((int(m.sum()), 5), np.nan)
            av[:, 2], ap[:, 2] = a_v[m], a_p[m]
            res["same_times_as_adt"] = {"model": scores(v[m], p[m], bt_v[m], bt_p[m]),
                                        "adt": scores(av, ap, bt_v[m], bt_p[m])}
    res["per_storm"] = {}
    for sid in sorted({r[0] for r in out_rows}):
        m = np.array([r[0] == sid for r in out_rows])
        res["per_storm"][sid] = scores(v[m], p[m], bt_v[m], bt_p[m])
    res["pred"] = [{"sid": r[0], "t": r[1], "v": v[i].round(1).tolist(), "p": p[i].round(1).tolist(), "bt_v": r[3],
                    "bt_p": r[4], "adt_v": None if not np.isfinite(r[5]) else r[5]} for i, r in enumerate(out_rows)]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prefix")
    ap.add_argument("out")
    ap.add_argument("--years", default="2024-2025")
    ap.add_argument("--goes25")
    ap.add_argument("--goes26")
    ap.add_argument("--bdecks")
    ap.add_argument("--adt", action="store_true")
    ap.add_argument("--skip-archive", action="store_true")
    ap.add_argument("--cimss", help="folder of CIMSS D-PRINT/D-MINT history files (fetch_cimss.py)")
    ap.add_argument("--rotations", type=int, default=1, help="average over this many 90-degree turns")
    ap.add_argument("--torch", help="the model's PyTorch checkpoint: the same numbers, faster")
    a = ap.parse_args()
    global ROT
    ROT = a.rotations
    net = si.Net(a.model)
    if a.torch:
        net = TorchNet(a.torch, net.info)
    report = {"model": a.model, "info": net.info, "rotations": ROT, "torch": bool(a.torch)}
    S = dict(np.load(a.prefix + "_samples.npz"))
    ir_store = np.load(a.prefix + "_ir.npy", mmap_mode="r")
    mw_store = np.load(a.prefix + "_mw.npy", mmap_mode="r")
    rng = np.random.RandomState(0)
    pick = rng.choice(len(S["ir"]), min(40, len(S["ir"])), replace=False)
    report["input_consistency_max_abs"] = consistency(S, pick, ir_store, mw_store)
    print("training vs app inputs, max abs difference:", report["input_consistency_max_abs"], flush=True)
    if not a.skip_archive:
        y = tuple(int(x) for x in a.years.split("-"))
        r = archive(net, S, ir_store, mw_store, y)
        report["archive"] = r
        print("archive", a.years, short(r["all"]), flush=True)
        print("  IR only:", short(r["all_ir_only"]), flush=True)
        print("  with a pass:", short(r["with_pass"]), "| same, IR only:", short(r["with_pass_ir_only"]), flush=True)
        if "all_gmi_only" in r:
            print("  GMI passes only:", short(r["all_gmi_only"]), flush=True)
            print("  with a GMI pass:", short(r["with_gmi_pass"]), "| same, IR only:", short(r["with_gmi_pass_ir_only"]), flush=True)
        for k, s in r["by_intensity"].items():
            print("  ", k, short(s), flush=True)
        for k, s in r["by_pass_age"].items():
            print("  pass", k, short(s), flush=True)
        for k, s in r.get("by_change_12h", {}).items():
            print("  12-h change", k, short(s), flush=True)
        for k, s in r["by_forced_pass_age"].items():
            print("  given a pass", k, "old:", short(s["with_pass"]), "| IR only:", short(s["ir_only"]), flush=True)
        if a.cimss:
            storms = json.load(open(a.prefix + "_meta.json"))["storms"]
            pred = dict(r["pred"])
            if "pred_gmi" in r:
                pred["pred_gmi"] = r["pred_gmi"]
            c = compare_archive(S, storms, pred, a.cimss)
            report["archive_vs_cimss"] = c
            for kind, out in c.items():
                print(f"  {kind} on the same cases ({out['n']}, {out['storms']} storms):", flush=True)
                for name, sc_ in out.items():
                    if isinstance(sc_, dict) and "vmax_mae" in sc_:
                        print(f"     {name:28s}", short(sc_), flush=True)
                print_paired(out)
    if a.goes25:
        r = goes25(net, S, ir_store, mw_store, a.goes25)
        report["goes25"] = r
        print("2025 via GOES pipeline:", short(r.get("goes_pipeline", {})), flush=True)
        print("   same with archive IR:", short(r.get("archive_ir", {})), flush=True)
        print("   ", {k: r.get(k) for k in ("goes_minus_archive_vmax", "goes_minus_archive_tb", "best_shift_km")}, flush=True)
    if a.goes26:
        r = goes26(net, a.goes26, a.bdecks, a.adt)
        report["goes26"] = r
        print("2026 IR only via GOES:", short(r.get("all", {})), flush=True)
        for k, s in r.get("by_intensity", {}).items():
            print("  ", k, short(s), flush=True)
        if "same_times_as_adt" in r:
            print("   model at ADT times:", short(r["same_times_as_adt"]["model"]), flush=True)
            print("   ADT              :", short(r["same_times_as_adt"]["adt"]), flush=True)
        if a.cimss and r.get("pred"):
            c = compare_goes26(r["pred"], a.cimss, a.bdecks)
            report["goes26_vs_cimss"] = c
            for kind, out in c.items():
                print(f"   {kind} on the same 2026 cases ({out['n']}, {out['storms']} storms):", flush=True)
                for name, sc_ in out.items():
                    if isinstance(sc_, dict) and "vmax_mae" in sc_:
                        print(f"     {name:28s}", short(sc_), flush=True)
                print_paired(out)
    json.dump(report, open(a.out, "w"), indent=1, default=float)


def print_paired(out):
    for key, d in out.items():
        if key.startswith("paired_vs_"):
            for name, p in d.items():
                if p:
                    print(f"     this model ({name}) minus {key[10:]}, mean |error|: {p['mean_abs_diff']:+.2f} kt "
                          f"(95 % {p['ci95'][0]:+.2f} to {p['ci95'][1]:+.2f}, by storm); closer in "
                          f"{100 * p['share_closer']:.0f} % of cases", flush=True)


if __name__ == "__main__":
    main()
