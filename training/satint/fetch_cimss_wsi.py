"""
CIMSS's D-PRINT and D-MINT history files for West Pacific, Southern Hemisphere and North Indian
Ocean storms (JTWC's basins), for the head-to-head comparison.

Usage: python fetch_cimss_wsi.py OUTDIR INDEX.json [INDEX.json ...] --years 2024-2025
Storm ids come from TC PRIMED indexes (WP012024, SH052025, IO012024). The Southern Hemisphere's
letter (S west of 135E, P east) and the Indian Ocean's (A Arabian Sea, B Bay of Bengal) are not in
the id, so both are tried; JTWC numbers each season's storms once, so only one exists.
Writes OUTDIR/{DPRINT,DMINT}hist_{year}_{code}.txt where they exist.
"""
import json
import os
import sys
import time

import requests

BASE = "https://tropic.ssec.wisc.edu/real-time/{kind}/{year}/{kind}hist_{year}_{code}.txt"
LETTERS = {"WP": "W", "SH": "SP", "IO": "AB"}


def main():
    a = sys.argv[1:]
    years = (2024, 2025)
    if "--years" in a:
        k = a.index("--years")
        years = tuple(int(x) for x in a[k + 1].split("-"))
        a = a[:k] + a[k + 2:]
    outdir, indexes = a[0], a[1:]
    os.makedirs(outdir, exist_ok=True)
    sids = set()
    for path in indexes:
        for r in json.load(open(path)):
            s = r["storm"].upper()
            if s[:2] in LETTERS and years[0] <= int(s[-4:]) <= years[1] and int(s[2:4]) < 90:
                sids.add(s)
    got = 0
    sess = requests.Session()
    for sid in sorted(sids):
        year, num = sid[-4:], sid[2:4]
        for letter in LETTERS[sid[:2]]:
            code = num + letter
            found = False
            for kind in ("DPRINT", "DMINT"):
                out = os.path.join(outdir, f"{kind}hist_{year}_{code}.txt")
                if os.path.exists(out):
                    found = True
                    continue
                url = BASE.format(kind=kind, year=year, code=code)
                r = None
                for attempt in range(3):
                    try:
                        r = sess.get(url, timeout=30)
                        break
                    except requests.RequestException:
                        time.sleep(2 + 3 * attempt)
                if r is not None and r.status_code == 200 and "VMAX" in r.text[:300]:
                    with open(out, "w") as fh:
                        fh.write(r.text)
                    got += 1
                    found = True
            if found:
                break
    print(len(sids), "storms;", got, "files fetched;", len(os.listdir(outdir)), "in", outdir)


if __name__ == "__main__":
    main()
