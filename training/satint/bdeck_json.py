"""
NHC b-decks (b*2026.dat files in a folder) to one JSON: per storm, its names and track rows
(time, lat, lon, vmax, mslp, kind), for goes_frames_2026.py and evaluate_satint.py.

Usage: python bdeck_json.py BDECK_FOLDER OUT.json
"""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import recon10s_nhc as nhc
out = {}
for p in sorted(glob.glob(os.path.join(sys.argv[1], "b*2026.dat"))):
    sid = os.path.basename(p)[1:9]
    pts = nhc.parse_bdeck(open(p, encoding="ascii", errors="replace").read())
    out[sid] = {"names": sorted({pt.name for pt in pts if pt.name}),
                "track": [(pt.time.isoformat(), pt.lat, pt.lon, pt.vmax, pt.mslp, pt.kind) for pt in pts]}
json.dump(out, open(sys.argv[2], "w"))
print({k: (v["names"], len(v["track"])) for k, v in out.items()})
