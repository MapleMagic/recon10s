"""Storm names and tracks from NHC b-deck files (plain or .gz) in a folder.

Usage: python best_tracks.py FOLDER OUT.json
"""
import glob, gzip, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import recon10s_nhc as nhc
folder = sys.argv[1]
out = {}
for p in sorted(glob.glob(os.path.join(folder, "b*.dat*"))):
    raw = open(p, "rb").read()
    text = (gzip.decompress(raw) if p.endswith(".gz") else raw).decode("ascii", "replace")
    sid = os.path.basename(p)[1:9]
    pts = nhc.parse_bdeck(text)
    names = sorted({pt.name for pt in pts if pt.name and pt.name not in ("INVEST", "")})
    out[sid] = {"names": names, "track": [(pt.time.isoformat(), pt.lat, pt.lon, pt.vmax, pt.mslp) for pt in pts]}
json.dump(out, open(sys.argv[2], "w"))
print({k: v["names"] for k, v in out.items()})
