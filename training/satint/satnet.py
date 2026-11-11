"""
The satellite intensity network (training side, PyTorch) and the batch assembly shared by training
and evaluation.

Inputs
  ir  (B, 5, 128, 128): IR at t, t-3, t-6, t-9, t-12 h on a 4-km storm-centred grid (missing history
      frames repeat the current one); (Tb - 240) / 40.
  mw  (B, 6, 64, 64): the latest imager pass on an 8-km grid -- 37V, 37H, 89V, 89H ((Tb - 230) / 40),
      a mask of where the swath has data, and the IR at the pass time; zeros when there is no pass.
  sc  (B, 13): has a pass, age / 12 h, 89-GHz footprint / 15 km, coverage, lat / 30, speed / 20 kt,
      east Pacific, sin and cos of the local solar hour, and which history frames exist.
Outputs (B, 10): quantiles 10, 25, 50, 75, 90 % of Vmax ((kt - 65) / 35) and of MSLP ((mb - 985) / 25).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
V_MEAN, V_SCALE = 65.0, 35.0
P_MEAN, P_SCALE = 985.0, 25.0
N_SCALAR = 13


def block(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class SatNet(nn.Module):
    def __init__(self, width=1.0):
        super().__init__()
        w = lambda c: int(round(c * width))
        self.ir = nn.Sequential(block(5, w(16), 2), block(w(16), w(32), 2), block(w(32), w(32), 1),
                                block(w(32), w(64), 2), block(w(64), w(96), 2), block(w(96), w(128), 2))
        self.mw = nn.Sequential(block(6, w(24), 2), block(w(24), w(48), 2), block(w(48), w(64), 2),
                                block(w(64), w(96), 2))
        self.ir_fc = nn.Sequential(nn.Flatten(), nn.Linear(w(128) * 16, 192), nn.ReLU(), nn.Dropout(0.25))
        self.mw_fc = nn.Sequential(nn.Flatten(), nn.Linear(w(96) * 16, 96), nn.ReLU(), nn.Dropout(0.25))
        self.head = nn.Sequential(nn.Linear(192 + 96 + N_SCALAR, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                  nn.Linear(64, 2 * len(QUANTILES)))

    def forward(self, ir, mw, sc):
        a = self.ir_fc(self.ir(ir))
        b = self.mw_fc(self.mw(mw)) * sc[:, :1]
        out = self.head(torch.cat([a, b, sc], 1))
        # quantiles kept in order: the median plus cumulative positive gaps either side
        n = len(QUANTILES)
        res = []
        for k in range(2):
            o = out[:, k * n:(k + 1) * n]
            mid = o[:, 2:3]
            lo = mid - torch.cumsum(F.softplus(o[:, :2].flip(1)), 1).flip(1)
            hi = mid + torch.cumsum(F.softplus(o[:, 3:]), 1)
            res.append(torch.cat([lo, mid, hi], 1))
        return torch.cat(res, 1)


def pinball(pred, target, mask):
    q = torch.tensor(QUANTILES, dtype=pred.dtype)
    e = target[:, None] - pred
    loss = torch.maximum(q * e, (q - 1) * e)
    return (loss.mean(1) * mask).sum() / mask.sum().clamp(min=1)


# --------------------------------------------------------------- batches

def ir_tb(q):
    tb = 163.0 + 0.6 * q.astype(np.float32)
    tb[q == 255] = np.nan
    return tb


def mw_tb(q):
    tb = 100.0 + 0.85 * q.astype(np.float32)
    tb[q == 255] = np.nan
    return tb


def pass_fields(S, i, c=-1):
    """(pass index, IR-at-pass index, age h, coverage, footprint km) of sample i: the latest pass
    (c = -1) or the youngest in age bin c (see build_samples.AGE_BINS); pass index -1 if none."""
    if c == "gmi":
        return S["mwg"][i], S["mwirg"][i], S["ageg"][i], S["covg"][i], 6.0
    if c is None or c < 0 or "mwc" not in S:
        return S["mw"][i], S["mw_ir"][i], S["age"][i], S["mw_cov"][i], S["mw_res"][i]
    return S["mwc"][i, c], S["mwirc"][i, c], S["agec"][i, c], S["covc"][i, c], S["resc"][i, c]


def choose_passes(S, idx, rng, p_latest=0.4):
    """Training: the latest pass for some samples, else one of the age bins that has a pass."""
    out = np.full(len(idx), -1)
    if "mwc" not in S:
        return out
    for b, i in enumerate(idx):
        if rng.rand() < p_latest:
            continue
        avail = np.flatnonzero(S["mwc"][i] >= 0)
        if avail.size:
            out[b] = int(rng.choice(avail))
    return out


def assemble(S, idx, ir_store, mw_store, cand=None):
    """Numpy inputs for the samples idx (no augmentation); cand: per-sample pass choice (pass_fields)."""
    B = len(idx)
    ir = np.zeros((B, 5, 128, 128), np.float32)
    mw = np.zeros((B, 6, 64, 64), np.float32)
    sc = np.zeros((B, N_SCALAR), np.float32)
    for b, i in enumerate(idx):
        cur = ir_tb(ir_store[S["ir"][i]])
        frames = [cur]
        avail = []
        for name in ("h3", "h6", "h9", "h12"):
            j = S[name][i]
            avail.append(1.0 if j >= 0 else 0.0)
            frames.append(ir_tb(ir_store[j]) if j >= 0 else cur)
        x = (np.stack(frames) - 240.0) / 40.0
        ir[b] = np.nan_to_num(x, nan=0.0)
        k, j, age, cov, res = pass_fields(S, i, None if cand is None else cand[b])
        if k >= 0:
            t = mw_tb(mw_store[k])
            valid = np.isfinite(t).all(0)
            mw[b, :4] = np.nan_to_num((t - 230.0) / 40.0, nan=0.0) * valid
            mw[b, 4] = valid
            src = ir_tb(ir_store[j]) if j >= 0 else cur
            small = np.nanmean(src.reshape(64, 2, 64, 2), axis=(1, 3))
            mw[b, 5] = np.nan_to_num((small - 240.0) / 40.0, nan=0.0)
            sc[b, 0] = 1.0
            sc[b, 1] = age / 12.0
            sc[b, 2] = res / 15.0
            sc[b, 3] = cov
        sc[b, 4] = S["lat"][i] / 30.0
        sp = S["speed"][i]
        sc[b, 5] = (sp if np.isfinite(sp) else 10.0) / 20.0
        sc[b, 6] = S["basin"][i]
        a = 2 * math.pi * S["solar"][i] / 24.0
        sc[b, 7], sc[b, 8] = math.sin(a), math.cos(a)
        sc[b, 9:13] = avail
    return ir, mw, sc


def augment(ir, mw, jitter_km=6.0, rng=np.random):
    """Rotate each sample by a random angle about the centre and shift it a few km (torch tensors)."""
    B = ir.shape[0]
    th = torch.tensor(rng.uniform(0, 2 * math.pi, B), dtype=torch.float32)
    c, s = torch.cos(th), torch.sin(th)
    # grid coordinates are in units of half the image width
    shift_ir = torch.tensor(rng.normal(0, jitter_km, (B, 2)) / (128 * 4.0 / 2), dtype=torch.float32)
    shift_mw = shift_ir * (128 * 4.0) / (64 * 8.0) + torch.tensor(rng.normal(0, jitter_km, (B, 2)) / (64 * 8.0 / 2),
                                                                  dtype=torch.float32)
    out = []
    for x, sh in ((ir, shift_ir), (mw, shift_mw)):
        A = torch.zeros(B, 2, 3)
        A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1] = c, -s, s, c
        A[:, :, 2] = sh
        grid = F.affine_grid(A, x.shape, align_corners=False)
        out.append(F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False))
    return out[0], out[1]
