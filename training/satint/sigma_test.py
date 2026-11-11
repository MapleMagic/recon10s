"""How much to smooth ABI's 2-km band 13 to match the 4-km archive IR (GOES frames vs TC PRIMED).

Usage: python sigma_test.py DATA_PREFIX GOES_PAIRS_DIR [STORMDIR]
"""
import glob, json, math, os, sys, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import numpy as np
import satnet as sn
import recon10s_goes as goes
import recon10s_satint as si
from scipy.ndimage import gaussian_filter
UTC = dt.timezone.utc
pre, gdir = sys.argv[1], sys.argv[2]
stormdir = sys.argv[3] if len(sys.argv) > 3 else "data/storms"
S = dict(np.load(pre + "_samples.npz")); ir_store = np.load(pre + "_ir.npy", mmap_mode="r")
meta = json.load(open(pre + "_meta.json"))["storms"]
rng = np.random.RandomState(3)
cases = []
for path in sorted(glob.glob(os.path.join(gdir, "*.npz"))):
    d = np.load(path)
    for k, i in enumerate(d["idx"]):
        if d["t"][k, 0] > 0:
            cases.append((i, d["t"][k, 0], int(d["sat"][k, 0])))
cases = [cases[j] for j in rng.choice(len(cases), 12, replace=False)]
imgs = []
for i, ts, sat in cases:
    when = dt.datetime.fromtimestamp(ts, UTC)
    js = json.load(open(os.path.join(stormdir, meta[S["storm"][i]] + ".json")))
    # centre as the archive frame has it
    row = min(js, key=lambda m: abs((m.get("ir_t") or 0) - S["time"][i]))
    la, lo = row["ir_c"]; lo = ((lo + 180) % 360) - 180
    scenes = goes.list_scenes(sat, 13, "F", when.replace(minute=59, second=59), hours_back=0)
    sc = min(scenes, key=lambda s: abs((s.start - when).total_seconds()))
    half = 3.4
    bbox = (la - half, la + half, lo - half / math.cos(math.radians(la)), lo + half / math.cos(math.radians(la)))
    img = goes.load_image(sc, bbox=bbox)
    imgs.append((i, img, (la, lo)))
print(len(imgs), "images", flush=True)
orig = si.gaussian_filter if hasattr(si, "gaussian_filter") else None
import scipy.ndimage as nd
for sigma in (0.0, 0.5, 0.7, 0.85, 1.0, 1.2):
    res = []
    eyes = []
    for i, img, c in imgs:
        data = np.asarray(img.data, float)
        class I: pass
        im2 = I(); im2.data = data; im2.sat_height = img.sat_height; im2.lon0 = img.lon0; im2.sweep = img.sweep; im2.extent_m = img.extent_m
        real = nd.gaussian_filter
        nd.gaussian_filter = (lambda x, s, _sig=sigma: x if _sig == 0 else real(x, _sig))
        try:
            g = si.ir_from_goes(im2, c)
        finally:
            nd.gaussian_filter = real
        a = sn.ir_tb(ir_store[S["ir"][i]])
        ok = np.isfinite(g) & np.isfinite(a)
        res.append(np.sqrt(np.mean((g[ok] - a[ok]) ** 2)))
        rr = np.hypot(*np.meshgrid(np.arange(128) - 63.5, np.arange(128) - 63.5)) * 4
        eyes.append(np.nanmax(np.where(rr < 30, g, np.nan)) - np.nanmax(np.where(rr < 30, a, np.nan)))
    print(f"sigma {sigma}: rms {np.mean(res):.3f} K, eye max diff {np.mean(eyes):+.2f} K", flush=True)
