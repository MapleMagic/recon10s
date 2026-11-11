"""
CIMSS's D-PRINT and D-MINT history files for a list of storms, for the head-to-head comparison.

Usage: python fetch_cimss.py OUTDIR STORM_ID [STORM_ID ...]     (e.g. AL052025 ep122026)
Writes OUTDIR/{DPRINT,DMINT}hist_{year}_{code}.txt where they exist.
"""
import os
import sys
import time

import requests

BASE = "https://tropic.ssec.wisc.edu/real-time/{kind}/{year}/{kind}hist_{year}_{code}.txt"
BASIN = {"AL": "L", "EP": "E", "CP": "C"}


def code(sid):
    sid = sid.upper()
    return f"{sid[2:4]}{BASIN[sid[:2]]}", sid[4:8]


def main(outdir, sids):
    os.makedirs(outdir, exist_ok=True)
    got = 0
    for sid in sids:
        if sid[:2].upper() not in BASIN or int(sid[2:4]) >= 90:
            continue
        c, year = code(sid)
        for kind in ("DPRINT", "DMINT"):
            out = os.path.join(outdir, f"{kind}hist_{year}_{c}.txt")
            if os.path.exists(out):
                continue
            url = BASE.format(kind=kind, year=year, code=c)
            for attempt in range(3):
                try:
                    r = requests.get(url, timeout=30)
                    break
                except requests.RequestException:
                    time.sleep(2 + 3 * attempt)
            else:
                print("failed", url, flush=True)
                continue
            if r.status_code == 200 and "VMAX" in r.text[:400]:
                open(out, "w").write(r.text)
                got += 1
            else:
                print(r.status_code, url, flush=True)
    print("downloaded", got, "files", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
