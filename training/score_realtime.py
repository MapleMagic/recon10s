"""Score a model on every 2025-2026 real-time analysis with IWG1 (rows for calibrate_intensity.py).

Usage: python score_realtime.py MODEL.npz OUT.json
"""
import json, os, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import recon10s_syntdr as st
import realtime_eval as seb_eval
from evalcase import eval_case
m = st.Model(sys.argv[1])
rows = []
for case, vt, vr in seb_eval.load_cases():
    r = eval_case(m, case, vt, vr)
    if r is not None:
        r["storm_id"] = case["storm_id"]
        rows.append(r)
json.dump(rows, open(sys.argv[2], "w"))
print(len(rows))
