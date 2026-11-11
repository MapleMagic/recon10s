"""
Train the satellite intensity network.

Usage: python train_cnn.py DATA_PREFIX OUT.pt [--epochs N] [--train-years 2000-2021] [--val-years 2022-2023]
                            [--width 1.0] [--seed 0] [--batch 64] [--limit N]
"""
import argparse
import json
import os
import queue
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import satnet as sn  # noqa: E402


def years(spec):
    a, b = spec.split("-")
    return int(a), int(b)


def load(prefix):
    S = dict(np.load(prefix + "_samples.npz"))
    ir = np.load(prefix + "_ir.npy")
    mw = np.load(prefix + "_mw.npy")
    return S, ir, mw


def batches(S, idx, ir, mw, bs, shuffle, rng, p_latest=None):
    order = rng.permutation(idx) if shuffle else np.asarray(idx)
    for k in range(0, len(order), bs):
        part = order[k:k + bs]
        cand = sn.choose_passes(S, part, rng, p_latest) if p_latest is not None else None
        x = sn.assemble(S, part, ir, mw, cand)
        yv = (S["vmax"][part] - sn.V_MEAN) / sn.V_SCALE
        yp = (S["mslp"][part] - sn.P_MEAN) / sn.P_SCALE
        yield part, x, yv.astype(np.float32), yp.astype(np.float32)


def prefetch(gen, n=4):
    q = queue.Queue(n)

    def run():
        for item in gen:
            q.put(item)
        q.put(None)
    threading.Thread(target=run, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        yield item


def predict(model, S, idx, ir, mw, bs=256, ir_only=False):
    model.eval()
    out = np.zeros((len(idx), 2 * len(sn.QUANTILES)), np.float32)
    pos = 0
    with torch.no_grad():
        for part, (a, b, c), _yv, _yp in prefetch(batches(S, idx, ir, mw, bs, False, None)):
            if ir_only:
                b[:] = 0.0
                c[:, 0:4] = 0.0
            o = model(torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(c)).numpy()
            out[pos:pos + len(part)] = o
            pos += len(part)
    v = out[:, :5] * sn.V_SCALE + sn.V_MEAN
    p = out[:, 5:] * sn.P_SCALE + sn.P_MEAN
    return v, p


def scores(S, idx, v, p):
    bt = S["vmax"][idx]
    e = v[:, 2] - bt
    out = {"n": int(len(idx)), "vmax_mae": float(np.mean(np.abs(e))), "vmax_rmse": float(np.sqrt(np.mean(e ** 2))),
           "vmax_bias": float(np.mean(e)),
           "vmax_cover_10_90": float(np.mean((bt >= v[:, 0]) & (bt <= v[:, 4])))}
    pt = S["mslp"][idx]
    ok = np.isfinite(pt)
    ep = p[ok, 2] - pt[ok]
    out.update(mslp_mae=float(np.mean(np.abs(ep))), mslp_rmse=float(np.sqrt(np.mean(ep ** 2))), mslp_bias=float(np.mean(ep)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-years", default="2000-2021")
    ap.add_argument("--val-years", default="2022-2023")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--mw-drop", type=float, default=0.3)
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast for the training passes")
    ap.add_argument("--resume", action="store_true", help="carry on from OUT.state (written after every epoch)")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--p-latest", type=float, default=0.4,
                    help="share of training samples given their latest pass; the rest get an older one")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    brng = np.random.RandomState(a.seed + 1)      # the batch thread's own
    S, ir, mw = load(a.prefix)
    yr = S["year"]
    t0, t1 = years(a.train_years)
    v0, v1 = years(a.val_years)
    tr = np.flatnonzero((yr >= t0) & (yr <= t1))
    va = np.flatnonzero((yr >= v0) & (yr <= v1))
    if a.limit:
        tr = rng.permutation(tr)[:a.limit]
    print(f"train {len(tr)} samples ({t0}-{t1}), validation {len(va)} ({v0}-{v1})", flush=True)
    model = sn.SatNet(a.width)
    if a.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(tr) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.15)
    best = None
    history = []
    start = 0
    state_path = a.out + ".state"
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best, history = st["epoch"], st["best"], st["history"]
        rng.set_state(st["rng"])
        brng.set_state(st["brng"])
        torch.set_rng_state(st["torch_rng"])
        print(f"resumed after epoch {start}", flush=True)
    for ep in range(start, a.epochs):
        model.train()
        t = time.time()
        tot, n = 0.0, 0
        for part, (xi, xm, xs), yv, yp in prefetch(batches(S, tr, ir, mw, a.batch, True, brng, a.p_latest)):
            # microwave dropout: the same network must work from IR alone (no pass, no PPS sign-in)
            drop = rng.rand(len(part)) < a.mw_drop
            xm[drop] = 0.0
            xs[drop, 0:4] = 0.0
            xi, xm = sn.augment(torch.from_numpy(xi), torch.from_numpy(xm), rng=rng)
            if a.channels_last:
                xi = xi.contiguous(memory_format=torch.channels_last)
                xm = xm.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=a.bf16):
                out = model(xi, xm, torch.from_numpy(xs))
            out = out.float()
            yv_t = torch.from_numpy(yv)
            yp_t = torch.from_numpy(np.nan_to_num(yp))
            mask_p = torch.from_numpy(np.isfinite(yp).astype(np.float32))
            loss = sn.pinball(out[:, :5], yv_t, torch.ones(len(part))) + \
                0.5 * sn.pinball(out[:, 5:], yp_t, mask_p)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(part)
            n += len(part)
        v, p = predict(model, S, va, ir, mw) if len(va) else (None, None)
        sc = scores(S, va, v, p) if len(va) else {}
        if len(va):
            v0, p0 = predict(model, S, va, ir, mw, ir_only=True)
            s0 = scores(S, va, v0, p0)
            sc.update(ir_only_vmax_mae=s0["vmax_mae"], ir_only_mslp_mae=s0["mslp_mae"])
        sc.update(epoch=ep + 1, loss=tot / max(n, 1), secs=time.time() - t)
        history.append(sc)
        print(json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in sc.items()}), flush=True)
        key = sc.get("vmax_mae", -ep)
        if best is None or key < best[0]:
            best = (key, ep + 1)
            torch.save({"state": model.state_dict(), "width": a.width, "epoch": ep + 1, "scores": sc,
                        "history": history, "args": vars(a)}, a.out)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1, "best": best, "history": history, "rng": rng.get_state(),
                    "brng": brng.get_state(), "torch_rng": torch.get_rng_state()}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
    torch.save({"state": model.state_dict(), "width": a.width, "epoch": a.epochs, "history": history,
                "args": vars(a)}, a.out.replace(".pt", "_last.pt"))
    print("best epoch", best)


if __name__ == "__main__":
    main()
