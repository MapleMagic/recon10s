"""
The time windows of every TDR analysis, for fetch_iwg1.py: each TC-RADAR swath's own start and
end (merged analyses: centre +- 3 h), 2018 on, when the IWG1 archive begins; and each real-time
analysis's start and end. Five minutes are added either side.

Usage: python iwg1_windows.py   (writes data/iwg1_windows_tcradar.json and data/iwg1_windows_seb.json)
"""
import collections
import glob
import json
import os

import realval as rv

SKIP = {"TRAIN", "SURV", "TDR", "TEST"}


def tcradar(min_year=2018):
    win = collections.defaultdict(list)
    for path in sorted(glob.glob("data/tcradar/*_[0-9][0-9][0-9][0-9].npz")):
        md = json.load(open(path.replace(".npz", ".json")))
        kind = "merge" if "_merge_" in os.path.basename(path) else "swath"
        for i, mis in enumerate(md["mission"]):
            if md["year"][i] < min_year:
                continue
            case = {k: (v[i] if isinstance(v, list) else v) for k, v in md.items()}
            case["kind"] = kind
            _tc, t0, t1 = rv.window(case)
            mission = ("20" + mis) if len(mis) == 8 else mis
            win[mission].append([t0.timestamp() - 300, t1.timestamp() + 300])
    return win


def realtime():
    win = collections.defaultdict(list)
    for path in sorted(glob.glob("data/seb/*.json")):
        m = json.load(open(path))
        if m["storm"].upper() not in SKIP:
            win[m["mission"]].append([m["start"] - 300, m["end"] + 300])
    return win


if __name__ == "__main__":
    for name, w in (("tcradar", tcradar()), ("seb", realtime())):
        json.dump(w, open(f"data/iwg1_windows_{name}.json", "w"))
        print(name, len(w), "missions")
