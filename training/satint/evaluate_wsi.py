"""
The satellite intensity model on storms of JTWC's basins -- West Pacific, Southern Hemisphere and North
Indian Ocean -- none of which it saw in training (it learnt Atlantic and east/central Pacific storms).
TC PRIMED samples with their archive IR, truth JTWC's best track, run as the app runs them: southern
storms mirrored north-south with |latitude|, four 90-degree turns averaged.

The basin input (0 Atlantic, 1 east/central Pacific) has no right value here, so it is fixed on one set
of years (--decide) and reported on another (--report). D-PRINT and D-MINT on the same cases.

Usage: python evaluate_wsi.py CHECKPOINT.pt DATA_PREFIX OUT.json [--decide 2020-2023] [--report 2024-2025]
                              [--cimss DIR] [--rotations 4] [--threads 4]
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import recon10s_goes as goes  # noqa: E402  (2.6.0: zenith angles, satellite choice)
import recon10s_satint as si  # noqa: E402  (2.6.0, mirroring southern storms)
import evaluate_satint as ev  # noqa: E402
import satnet as sn  # noqa: E402

UTC = dt.timezone.utc
_assemble = sn.assemble


def assemble_mirrored(S, idx, ir_store, mw_store, cand=None):
    """satnet.assemble, with southern samples mirrored north-south as recon10s_satint.network_inputs does."""
    ir, mw, sc = _assemble(S, idx, ir_store, mw_store, cand)
    south = np.asarray(S["lat"])[np.asarray(idx)] < 0
    if south.any():
        ir[south] = ir[south][:, :, ::-1, :]
        mw[south] = mw[south][:, :, ::-1, :]
        sc[south, 4] = -sc[south, 4]
    return ir, mw, sc


sn.assemble = assemble_mirrored


def sample_lon(S, i):
    t = dt.datetime.fromtimestamp(float(S["time"][i]), UTC)
    lon = (S["solar"][i] - (t.hour + t.minute / 60 + t.second / 3600)) * 15.0
    return (lon + 540.0) % 360.0 - 180.0


def regions(S, storms, idx):
    out = []
    for i in idx:
        name = storms[S["storm"][i]]
        b = name[:2]
        if b == "SH":
            out.append("SH Australia/S Pacific" if sample_lon(S, i) >= 90.0 or sample_lon(S, i) < -100 else
                       "SH SW Indian Ocean")
        else:
            out.append({"WP": "West Pacific", "IO": "North Indian Ocean"}.get(b, b))
    return np.array(out)


def app_view(S, idx):
    """(satellite the app picks, zenith angle) per sample."""
    sats, zen = [], []
    for i in idx:
        la, lo = float(S["lat"][i]), sample_lon(S, i)
        s = goes.pick_satellite(la, lo)
        sats.append(s)
        zen.append(goes.zenith_angle(la, lo, s))
    return np.array(sats), np.array(zen)


def consistency(S, idx, ir_store, mw_store):
    """The evaluation's inputs against the app's (recon10s_satint.network_inputs) for southern samples."""
    worst = 0.0
    for i in idx:
        a = sn.assemble(S, np.array([i]), ir_store, mw_store)
        cur = sn.ir_tb(ir_store[S["ir"][i]])
        frames = [cur] + [sn.ir_tb(ir_store[S[h][i]]) if S[h][i] >= 0 else None for h in ("h3", "h6", "h9", "h12")]
        mw = mw_ir = None
        if S["mw"][i] >= 0:
            mw = sn.mw_tb(mw_store[S["mw"][i]])
            mw_ir = sn.ir_tb(ir_store[S["mw_ir"][i]]) if S["mw_ir"][i] >= 0 else None
        t = dt.datetime.fromtimestamp(float(S["time"][i]), UTC)
        b = si.network_inputs(frames, mw, mw_ir, max(S["age"][i], 0.0), S["mw_res"][i], S["mw_cov"][i], S["lat"][i],
                              S["speed"][i], bool(S["basin"][i]), t, sample_lon(S, i))
        for x, y in zip(a, b):
            worst = max(worst, float(np.max(np.abs(x - y))))
    return worst


def cimss_path(cdir, kind, sid):
    sid = sid.upper()
    hits = sorted(glob.glob(os.path.join(cdir, f"{kind}hist_{sid[4:8]}_{sid[2:4]}[A-Z].txt")))
    letters = {"WP": "W", "SH": "SP", "IO": "AB"}.get(sid[:2], "")
    hits = [h for h in hits if h[-5] in letters]
    return hits[0] if hits else None


def compare_cimss(S, storms, idx, v, p, v0, p0, cdir, tol_min=45.0):
    """D-PRINT and D-MINT against this model at the samples nearest their times (as for the Atlantic)."""
    res = {}
    for kind in ("DPRINT", "DMINT"):
        rows = []
        for s_ in np.unique(S["storm"][idx]):
            sid = storms[s_]
            path = cimss_path(cdir, kind, sid)
            if not path:
                continue
            mine = np.flatnonzero(S["storm"][idx] == s_)
            ts = S["time"][idx][mine]
            order = np.argsort(ts)
            mine, ts = mine[order], ts[order]
            bt_v_s, bt_p_s = S["vmax"][idx][mine], S["mslp"][idx][mine]
            for t, cv, cp, _sensor, q25, q75 in ev.read_cimss(path):
                j = int(np.argmin(np.abs(ts - t)))
                if abs(ts[j] - t) > tol_min * 60:
                    continue
                k = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
                if not (ts[k - 1] <= t <= ts[k]) or ts[k] - ts[k - 1] > 6 * 3600:
                    bv, bp = bt_v_s[j], bt_p_s[j]
                else:
                    f = (t - ts[k - 1]) / max(ts[k] - ts[k - 1], 1.0)
                    bv = bt_v_s[k - 1] + f * (bt_v_s[k] - bt_v_s[k - 1])
                    bp = bt_p_s[k - 1] + f * (bt_p_s[k] - bt_p_s[k - 1])
                rows.append((cv, cp, bv, bp, mine[j], sid, q25, q75))
        if len(rows) < 10:
            continue
        cv = np.array([r[0] for r in rows]); cp = np.array([r[1] for r in rows])
        bv = np.array([r[2] for r in rows]); bp = np.array([r[3] for r in rows])
        jj = np.array([r[4] for r in rows])
        tv, tp = S["vmax"][idx][jj], S["mslp"][idx][jj]
        rep = lambda a: np.repeat(a[:, None], 5, 1)
        cq = ev.cimss_q(rows, 0, 6, 7)
        groups = [r[5] for r in rows]
        res[kind] = {"n": len(rows), "storms": len(set(groups)), kind: ev.scores(cq, rep(cp), bv, bp),
                     "this model, newest pass": ev.scores(v[jj], p[jj], tv, tp),
                     "this model, IR only": ev.scores(v0[jj], p0[jj], tv, tp),
                     "by_intensity": {kind: ev.by_bins(cq, rep(cp), bv, bp, bv, *ev.VBINS),
                                      "this model, newest pass": ev.by_bins(v[jj], p[jj], tv, tp, tv, *ev.VBINS),
                                      "this model, IR only": ev.by_bins(v0[jj], p0[jj], tv, tp, tv, *ev.VBINS)},
                     "paired": {"newest pass": ev.paired(v[jj][:, 2] - tv, cv - bv, groups),
                                "IR only": ev.paired(v0[jj][:, 2] - tv, cv - bv, groups)}}
    return res


def evaluate(net, S, storms, idx, ir_store, mw_store, flag):
    S = dict(S)
    S["basin"] = np.full_like(S["basin"], flag)
    v, p = ev.predict_samples(net, S, idx, ir_store, mw_store)
    v0, p0 = ev.predict_samples(net, S, idx, ir_store, mw_store, ir_only=True)
    return v, p, v0, p0


def summary(v, p, v0, p0, S, storms, idx, zen):
    bt_v, bt_p = S["vmax"][idx], S["mslp"][idx]
    has = S["mw"][idx] >= 0
    reg = regions(S, storms, idx)
    out = {"n": int(len(idx)), "storms": int(len(np.unique(S["storm"][idx]))),
           "all": ev.scores(v, p, bt_v, bt_p), "all_ir_only": ev.scores(v0, p0, bt_v, bt_p),
           "with_pass": ev.scores(v[has], p[has], bt_v[has], bt_p[has]),
           "with_pass_ir_only": ev.scores(v0[has], p0[has], bt_v[has], bt_p[has]),
           "by_intensity": ev.by_bins(v, p, bt_v, bt_p, bt_v, *ev.VBINS),
           "by_intensity_ir_only": ev.by_bins(v0, p0, bt_v, bt_p, bt_v, *ev.VBINS),
           "by_region": {r: ev.scores(v[reg == r], p[reg == r], bt_v[reg == r], bt_p[reg == r]) for r in np.unique(reg)},
           "by_region_ir_only": {r: ev.scores(v0[reg == r], p0[reg == r], bt_v[reg == r], bt_p[reg == r])
                                 for r in np.unique(reg)},
           "by_zenith": ev.by_bins(v, p, bt_v, bt_p, zen, [0, 40, 55, 62, 70, 90],
                                   ["<40", "40-55", "55-62", "62-70", "70+ (no estimate in the app)"]),
           "by_year": {int(y): ev.scores(v[S["year"][idx] == y], p[S["year"][idx] == y], bt_v[S["year"][idx] == y],
                                         bt_p[S["year"][idx] == y]) for y in np.unique(S["year"][idx])}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("prefix")
    ap.add_argument("out")
    ap.add_argument("--decide", default="2020-2023")
    ap.add_argument("--report", default="2024-2025")
    ap.add_argument("--cimss")
    ap.add_argument("--rotations", type=int, default=4, help="90-degree turns averaged for the reported years")
    ap.add_argument("--decide-rotations", type=int, default=1, help="... and for choosing the basin input")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    S = dict(np.load(a.prefix + "_samples.npz"))
    storms = json.load(open(a.prefix + "_meta.json"))["storms"]
    ir_store = np.load(a.prefix + "_ir.npy", mmap_mode="r")
    mw_store = np.load(a.prefix + "_mw.npy", mmap_mode="r")
    net = ev.TorchNet(a.checkpoint, threads=a.threads)
    south = np.flatnonzero(S["lat"] < 0)
    rng = np.random.RandomState(0)
    chk = consistency(S, rng.choice(south, min(40, len(south)), replace=False), ir_store, mw_store) if len(south) else 0.0
    print(f"evaluation vs app inputs, southern samples: max |diff| {chk:.2e}", flush=True)
    out = {"consistency_max_abs": chk}
    span = lambda s: tuple(int(x) for x in s.split("-"))
    d0, d1 = span(a.decide)
    r0, r1 = span(a.report)
    sats, zen = app_view(S, np.arange(len(S["lat"])))
    reach = zen <= si.MAX_ZENITH
    out["app_reach"] = {"share_within_limit": float(reach.mean()), "limit_deg": si.MAX_ZENITH}
    # 1. the basin input, chosen on the decision years (cases the app can estimate)
    idx = np.flatnonzero((S["year"] >= d0) & (S["year"] <= d1) & reach)
    ev.ROT = a.decide_rotations
    dec = {}
    for flag in (0, 1):
        v, p, v0, p0 = evaluate(net, S, storms, idx, ir_store, mw_store, flag)
        dec[flag] = {"newest pass": ev.scores(v, p, S["vmax"][idx], S["mslp"][idx]),
                     "IR only": ev.scores(v0, p0, S["vmax"][idx], S["mslp"][idx])}
        print(f"decide {d0}-{d1}, basin input {flag}: {ev.short(dec[flag]['newest pass'])}", flush=True)
        print(f"    IR only: {ev.short(dec[flag]['IR only'])}", flush=True)
    score = {f: dec[f]["newest pass"]["vmax_mae"] + 0.5 * dec[f]["newest pass"]["mslp_mae"] for f in (0, 1)}
    flag = min(score, key=score.get)
    out["decide"] = {"years": [d0, d1], "n": int(len(idx)), "flag0": dec[0], "flag1": dec[1], "chosen_flag": int(flag),
                     "rotations": a.decide_rotations}
    print("chosen basin input:", flag, flush=True)
    # the decision years again with the chosen input, as the app runs it: for fitting any adjustment to JTWC's
    # best track (fit_jtwc_adjust.py) on years kept apart from the reported ones
    ev.ROT = a.rotations
    v, p, v0, p0 = evaluate(net, S, storms, idx, ir_store, mw_store, flag)
    out["decide_pred"] = {"idx": idx.tolist(), "v": v.round(2).tolist(), "p": p.round(2).tolist(),
                          "v_ir": v0.round(2).tolist(), "p_ir": p0.round(2).tolist()}
    # 2. the report years, with the chosen input (and the other, to show how much it matters)
    idx = np.flatnonzero((S["year"] >= r0) & (S["year"] <= r1))
    ev.ROT = a.rotations
    rep = {}
    for f in (flag, 1 - flag):
        v, p, v0, p0 = evaluate(net, S, storms, idx, ir_store, mw_store, f)
        m = reach[idx]
        rep[f"flag{f}"] = {"reachable": summary(v[m], p[m], v0[m], p0[m], S, storms, idx[m], zen[idx][m]),
                           "all": summary(v, p, v0, p0, S, storms, idx, zen[idx])}
        print(f"report {r0}-{r1}, basin input {f}, reachable: {ev.short(rep[f'flag{f}']['reachable']['all'])}", flush=True)
        for r, sc in rep[f"flag{f}"]["reachable"]["by_region"].items():
            print(f"    {r}: {ev.short(sc)}", flush=True)
        if f == flag:
            keep = (v, p, v0, p0)
            if a.cimss:
                c = compare_cimss(S, storms, idx[m], v[m], p[m], v0[m], p0[m], a.cimss)
                rep["cimss"] = c
                for kind, o in c.items():
                    print(f"  vs {kind} ({o['n']} cases, {o['storms']} storms): {kind} {ev.short(o[kind])}", flush=True)
                    print(f"      this model newest pass: {ev.short(o['this model, newest pass'])}", flush=True)
                    print(f"      this model IR only:     {ev.short(o['this model, IR only'])}", flush=True)
                    print(f"      paired: {o['paired']}", flush=True)
    out["report"] = {"years": [r0, r1], **rep}
    v, p, v0, p0 = keep
    out["pred"] = {"idx": idx.tolist(), "v": v.round(2).tolist(), "p": p.round(2).tolist(),
                   "v_ir": v0.round(2).tolist(), "p_ir": p0.round(2).tolist(), "zenith": zen[idx].round(1).tolist(),
                   "sat": sats[idx].tolist()}
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
