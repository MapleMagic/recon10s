"""
Check an exported model: the app's numpy network against the PyTorch checkpoint on real samples.

Usage: python check_export.py MODEL.pt[,MODEL2.pt,...] MODEL.npz DATA_PREFIX
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import recon10s_satint as si  # noqa: E402
import satnet as sn  # noqa: E402


def main(pt, npz, prefix):
    S = dict(np.load(prefix + "_samples.npz"))
    ir = np.load(prefix + "_ir.npy", mmap_mode="r")
    mw = np.load(prefix + "_mw.npy", mmap_mode="r")
    idx = np.linspace(0, len(S["ir"]) - 1, 32).astype(int)
    a, b, c = sn.assemble(S, idx, ir, mw)
    outs = []
    for path in pt.split(","):                         # an ensemble file: the members' average
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = sn.SatNet(ck.get("width", 1.0))
        m.load_state_dict(ck["state"])
        m.eval()
        with torch.no_grad():
            outs.append(m(torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(c)).numpy())
    o = np.mean(outs, 0)
    tv = o[:, :5] * sn.V_SCALE + sn.V_MEAN
    tp = o[:, 5:] * sn.P_SCALE + sn.P_MEAN
    n = si.Net(npz)
    t = time.time()
    nv, npp = n.forward(a, b, c)
    dtm = time.time() - t
    dv, dp = np.abs(nv - tv).max(), np.abs(npp - tp).max()
    print(f"max |numpy - torch|: Vmax {dv:.4f} kt, MSLP {dp:.4f} mb; numpy {dtm / len(idx) * 1000:.1f} ms/sample")
    assert dv < 0.05 and dp < 0.05, "the export does not reproduce the network"


if __name__ == "__main__":
    main(*sys.argv[1:4])
