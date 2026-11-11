"""
List TC PRIMED (s3://noaa-nesdis-tcprimed-pds, v01r01) overpass files for some basins and years.

Usage: python index_tcprimed.py OUT.json BASIN[,BASIN...] YEAR0 YEAR1
Each entry: key, size, storm (e.g. AL102023), sensor, platform, time (UTC epoch), kind (final/preliminary).
"""
import concurrent.futures as cf
import datetime as dt
import json
import re
import sys
import time
import xml.etree.ElementTree as ET

import requests

BASE = "https://noaa-nesdis-tcprimed-pds.s3.amazonaws.com/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
NAME = re.compile(r"TCPRIMED_v01r01-(final|preliminary)_([A-Z]{2}\d{6})_([A-Z0-9]+)_([A-Z0-9]+)_(\d+)_(\d{14})\.nc$")


def list_prefix(prefix, delimiter=True):
    keys, prefixes, token = [], [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = "/"
        if token:
            params["continuation-token"] = token
        for attempt in range(6):
            try:
                r = requests.get(BASE, params=params, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(2 + 2 * attempt)
        else:
            raise IOError(prefix)
        root = ET.fromstring(r.content)
        for c in root.findall(NS + "Contents"):
            keys.append((c.find(NS + "Key").text, int(c.find(NS + "Size").text)))
        for p in root.findall(NS + "CommonPrefixes"):
            prefixes.append(p.find(NS + "Prefix").text)
        if root.find(NS + "IsTruncated").text == "true":
            token = root.find(NS + "NextContinuationToken").text
        else:
            return keys, prefixes


def storm_files(prefix):
    keys, _ = list_prefix(prefix, delimiter=False)
    out = []
    for key, size in keys:
        m = NAME.search(key)
        if not m:
            continue
        kind, storm, sensor, platform, _orbit, stamp = m.groups()
        t = dt.datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc).timestamp()
        out.append({"key": key, "size": size, "storm": storm, "sensor": sensor, "platform": platform, "time": t,
                    "kind": kind})
    return out


def main(out, basins, y0, y1):
    storms = []
    for kind in ("final", "preliminary"):
        for year in range(y0, y1 + 1):
            for basin in basins:
                _k, prefixes = list_prefix(f"v01r01/{kind}/{year}/{basin}/")
                storms += prefixes
    print(len(storms), "storm folders", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(16) as ex:
        for k, res in enumerate(ex.map(storm_files, storms), 1):
            rows += res
            if k % 200 == 0:
                print(k, len(rows), flush=True)
    json.dump(rows, open(out, "w"))
    print(len(rows), "files")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4]))
