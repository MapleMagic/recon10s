"""
Export a trained SatNet (PyTorch checkpoint) to the app's numpy format: batch norm folded into
each convolution, the layers listed in a JSON spec.

Usage: python export_net.py MODEL.pt[,MODEL2.pt,...] OUT.npz [info.json]
Several checkpoints (comma-separated) go into one file as members whose estimates the app averages.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import satnet as sn  # noqa: E402


def fold(conv, bn):
    w = conv.weight.detach().numpy().astype(np.float64)
    g = bn.weight.detach().numpy()
    b = bn.bias.detach().numpy()
    m = bn.running_mean.detach().numpy()
    v = bn.running_var.detach().numpy()
    s = g / np.sqrt(v + bn.eps)
    return (w * s[:, None, None, None]).astype(np.float32), (b - m * s).astype(np.float32)


def _arrays(model, pre=""):
    model.eval()
    arrays, spec = {}, {}
    for name, seq, fc in (("ir", model.ir, model.ir_fc), ("mw", model.mw, model.mw_fc)):
        layers = []
        for i, blk in enumerate(seq):
            conv, bn = blk[0], blk[1]
            w, b = fold(conv, bn)
            arrays[f"{pre}{name}{i}_w"], arrays[f"{pre}{name}{i}_b"] = w, b
            layers.append({"stride": int(conv.stride[0]), "out": int(w.shape[0])})
        spec[name] = layers
        lin = fc[1]
        arrays[f"{pre}{name}_fc_w"] = np.ascontiguousarray(lin.weight.detach().numpy(), dtype=np.float32)
        arrays[f"{pre}{name}_fc_b"] = np.ascontiguousarray(lin.bias.detach().numpy(), dtype=np.float32)
    lins = [m for m in model.head if isinstance(m, torch.nn.Linear)]
    for i, lin in enumerate(lins):
        arrays[f"{pre}head{i}_w"] = np.ascontiguousarray(lin.weight.detach().numpy(), dtype=np.float32)
        arrays[f"{pre}head{i}_b"] = np.ascontiguousarray(lin.bias.detach().numpy(), dtype=np.float32)
    spec["head"] = len(lins)
    return arrays, spec


def export(model, out, info=None):
    """One network, or a list of them (members, averaged by the app)."""
    spec = {"quantiles": list(sn.QUANTILES),
            "norm": {"vmax": [sn.V_MEAN, sn.V_SCALE], "mslp": [sn.P_MEAN, sn.P_SCALE]}}
    if isinstance(model, (list, tuple)) and len(model) > 1:
        arrays, spec["members"] = {}, []
        for k, m in enumerate(model):
            a, sp = _arrays(m, f"m{k}_")
            arrays.update(a)
            spec["members"].append(sp)
    else:
        arrays, sp = _arrays(model[0] if isinstance(model, (list, tuple)) else model)
        spec.update(sp)
    np.savez_compressed(out, spec=np.array(json.dumps(spec)), info=np.array(json.dumps(info or {})), **arrays)
    return out


if __name__ == "__main__":
    models = []
    for pt in sys.argv[1].split(","):
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        m = sn.SatNet(ck.get("width", 1.0))
        m.load_state_dict(ck["state"])
        models.append(m)
    info = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
    export(models, sys.argv[2], info)
    print("exported", sys.argv[2], f"({len(models)} network(s))", os.path.getsize(sys.argv[2]) / 1e6, "MB")
