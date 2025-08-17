#!/usr/bin/env python3

from __future__ import annotations
import argparse
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Regexes (lat/lon/wind)
_RE_LAT = re.compile(r"^(\d{2})(\d{2})(\d{2})?([NS])$")
_RE_LON = re.compile(r"^(\d{3})(\d{2})(\d{2})?([EW])$")
_RE_WIND = re.compile(r"^(\d{3})(\d{3})$")
_RE_PPPP = re.compile(r"^\d{4}$")

# Color buckets (knots) user-specified; RGB normalized below
_COLOR_BUCKETS = [
    (0, 10, (255, 255, 255)),
    (10, 20, (0, 255, 255)),
    (20, 30, (0, 128, 128)),
    (30, 34, (0, 0, 255)),
    (34, 40, (0, 128, 0)),
    (40, 45, (0, 255, 0)),
    (45, 50, (255, 255, 0)),
    (50, 55, (255, 128, 0)),
    (55, 60, (255, 0, 0)),
    (60, 64, (200, 0, 0)),
    (64, 83, (170, 1, 255)),
    (83, 96, (213, 106, 255)),
    (96, 113, (255, 212, 255)),
    (113, 137, (255, 166, 193)),
    (137, float("inf"), (253, 105, 110)),
]


def _tok_to_latlon(lat_tok: str, lon_tok: str) -> Optional[Tuple[float, float]]:
    mlat = _RE_LAT.match(lat_tok)
    mlon = _RE_LON.match(lon_tok)
    if not (mlat and mlon):
        return None
    deg = int(mlat.group(1)); minutes = int(mlat.group(2)); secs = int(mlat.group(3)) if mlat.group(3) else 0
    lat = deg + minutes/60.0 + secs/3600.0
    if mlat.group(4) == "S": lat = -lat
    deg_lon = int(mlon.group(1)); minutes_lon = int(mlon.group(2)); secs_lon = int(mlon.group(3)) if mlon.group(3) else 0
    lon = deg_lon + minutes_lon/60.0 + secs_lon/3600.0
    if mlon.group(4) == "W": lon = -lon
    return lat, lon


def _find_mslp_and_wind(parts: List[str]):
    pppp_idx = None
    wind_tuple = None
    for idx, tok in enumerate(parts):
        t = tok.strip()
        m = _RE_WIND.match(t)
        if m:
            d = int(m.group(1)); s = int(m.group(2))
            if 0 <= d <= 360 and 0 <= s <= 300:
                wind_tuple = (d, s)
                continue
        if _RE_PPPP.match(t):
            try:
                val = int(t)
                if 8000 <= val <= 11000:
                    pppp_idx = idx
                    continue
            except Exception:
                pass
    return pppp_idx, wind_tuple


def _parse_time_token(tstr: str):
    t = tstr.strip()
    if not t.isdigit():
        return None
    if len(t) == 6: return int(t[0:2]), int(t[2:4]), int(t[4:6])
    if len(t) == 4: return int(t[0:2]), int(t[2:4]), 0
    return None


def _hhmm_to_seconds(hhmm: Optional[str]) -> Optional[int]:
    if hhmm is None: return None
    s = hhmm.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2: hh, mm, ss = int(parts[0]), int(parts[1]), 0
        else: hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        if len(s) == 4: hh, mm, ss = int(s[0:2]), int(s[2:4]), 0
        elif len(s) == 6: hh, mm, ss = int(s[0:2]), int(s[2:4]), int(s[4:6])
        else: return None
    return hh*3600 + mm*60 + ss


def _speed_to_rgb_normalized(kts: float):
    for lo, hi, rgb in _COLOR_BUCKETS:
        if lo <= kts < hi:
            return (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
    return (0.8, 0.8, 0.8)


def _wind_to_uv_knots(dir_deg: int, spd_kt: int) -> Tuple[float, float]:
    rad = np.deg2rad(dir_deg)
    u = -spd_kt * np.sin(rad)
    v = -spd_kt * np.cos(rad)
    return float(u), float(v)


def parse_hdob_file(hdob_file: str):
    times = []; times_tup = []; lats = []; lons = []; mslps = []; wind_dirs = []; wind_spds = []
    with open(hdob_file, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line: continue
            if any(line.startswith(p) for p in ("URNT", "KNHC", "$$")): continue
            parts = line.split()
            if len(parts) < 3: continue
            time_tok = parts[0]; lat_tok = parts[1]; lon_tok = parts[2]
            latlon = _tok_to_latlon(lat_tok, lon_tok)
            if latlon is None:
                found = False
                for i in range(1, len(parts)-1):
                    maybe = _tok_to_latlon(parts[i], parts[i+1])
                    if maybe:
                        latlon = maybe; found = True; break
                if not found: continue
            lat_dec, lon_dec = latlon
            pppp_idx, wind_tok = _find_mslp_and_wind(parts)
            mslp_val = None
            if pppp_idx is not None:
                try: mslp_val = int(parts[pppp_idx]) / 10.0
                except: mslp_val = None
            else:
                for alt in (3,4,5,6):
                    if alt < len(parts) and _RE_PPPP.match(parts[alt]):
                        try:
                            vv = int(parts[alt])
                            if 8000 <= vv <= 11000:
                                mslp_val = vv / 10.0; break
                        except Exception: pass
            dir_deg = None; spd_kt = None
            if wind_tok is not None:
                dir_deg, spd_kt = wind_tok
            ttt = _parse_time_token(time_tok)
            if ttt is None: continue
            times.append(time_tok); times_tup.append(ttt)
            lats.append(lat_dec); lons.append(lon_dec); mslps.append(mslp_val)
            wind_dirs.append(dir_deg); wind_spds.append(spd_kt)
    return times, times_tup, lats, lons, mslps, wind_dirs, wind_spds


def main(hdob_file: str, start_utc: Optional[str] = None, end_utc: Optional[str] = None,
         show_legend: bool = False, plot_flag: bool = True):
    if not os.path.exists(hdob_file): raise FileNotFoundError(hdob_file)
    times, times_tup, lats, lons, mslps, wind_dirs, wind_spds = parse_hdob_file(hdob_file)
    if not lats:
        print("No valid HDOB points found."); return

    secs = [hh*3600 + mm*60 + ss for (hh,mm,ss) in times_tup]
    start_sec = _hhmm_to_seconds(start_utc) if start_utc else None
    end_sec = _hhmm_to_seconds(end_utc) if end_utc else None
    idxs = []
    for i, s in enumerate(secs):
        keep = True
        if start_sec is not None and end_sec is not None:
            if start_sec <= end_sec: keep = (start_sec <= s <= end_sec)
            else: keep = (s >= start_sec or s <= end_sec)
        elif start_sec is not None: keep = (s >= start_sec)
        elif end_sec is not None: keep = (s <= end_sec)
        if keep: idxs.append(i)

    print(f"Total records: {len(lats)}; filtered for window: {len(idxs)}")
    if not idxs:
        print("No records in requested window."); return

    px=[]; py=[]; uu=[]; vv=[]; colors=[]; pm=[]
    for i in idxs:
        d = wind_dirs[i]; s = wind_spds[i]
        if d is None or s is None: continue
        lon = lons[i]; lat = lats[i]
        u, v = _wind_to_uv_knots(d, s)
        c = _speed_to_rgb_normalized(s)
        px.append(lon); py.append(lat); uu.append(u); vv.append(v); colors.append(c); pm.append(mslps[i])
    if not px:
        print("No wind-bearing points to plot in the selected window."); return

    px = np.array(px); py = np.array(py); uu = np.array(uu); vv = np.array(vv); pm = np.array(pm)
    color_groups = {}
    for i, c in enumerate(colors):
        color_groups.setdefault(c, []).append(i)

    if not plot_flag:
        print("Plot flag false; skipping plotting.")
        return

    # Dark style: black figure & axes background; white text & gridlines
    fig = plt.figure(figsize=(11, 8), facecolor="black")
    ax = plt.axes(projection=ccrs.Mercator(), facecolor="black")
    lonmin, lonmax = float(np.min(px)), float(np.max(px))
    latmin, latmax = float(np.min(py)), float(np.max(py))
    margin_lon = max(0.5, (lonmax - lonmin) * 0.12)
    margin_lat = max(0.5, (latmax - latmin) * 0.12)
    ax.set_extent([lonmin - margin_lon, lonmax + margin_lon, latmin - margin_lat, latmax + margin_lat], crs=ccrs.PlateCarree())

    # Map features tuned for dark background
    land = cfeature.LAND.with_scale("50m")
    ocean = cfeature.OCEAN.with_scale("50m")
    ax.add_feature(land, facecolor="#222222", edgecolor="#222222")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#888888", linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#666666", linestyle=":", linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=0.4, color="white", alpha=0.3, linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    # make grid label colors white
    try:
        gl.xlabel_style = {"color": "white"}
        gl.ylabel_style = {"color": "white"}
    except Exception:
        pass

    # Plot barbs grouped by color
    for color, inds in color_groups.items():
        xs = px[inds]; ys = py[inds]; us = uu[inds]; vs = vv[inds]
        ax.barbs(xs, ys, us, vs, length=6, transform=ccrs.PlateCarree(), color=color, linewidth=0.8)

    # MSLP labels — white text with small red halo for clarity
    for x, y, m in zip(px, py, pm):
        if m is not None:
            ax.text(x + 0.02, y + 0.02, f"{m:.1f}", color="white", fontsize=8, transform=ccrs.PlateCarree(),
                    ha="left", va="bottom", bbox={"facecolor":"none","edgecolor":"none","pad":0})

    # Legend (drawn as colored swatches); on dark background use white text
    if show_legend:
        patches = []
        for lo, hi, rgb in _COLOR_BUCKETS:
            if hi == float("inf"):
                label = f"{int(lo)}+ kt"
            else:
                label = f"{int(lo)}–{int(hi)} kt"
            color = (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
            patches.append(mpatches.Patch(color=color, label=label))
        leg = ax.legend(handles=patches, title="Wind speed (kt)", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
        leg.get_frame().set_facecolor("#111111")
        leg.get_frame().set_edgecolor("white")
        for text in leg.get_texts():
            text.set_color("white")
        if leg.get_title():
            leg.get_title().set_color("white")

    # Title and labels in white
    title_time = ""
    if start_utc or end_utc:
        title_time = f" ({start_utc or '00:00'}–{end_utc or '23:59'} UTC)"
    ax.set_title(f"HDOB: {os.path.basename(hdob_file)} — Flight-level winds & MSLP{title_time}", color="white")

    out_png = os.path.splitext(hdob_file)[0] + ".png"
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved plot: {out_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot HDOB wind barbs and MSLP (dark theme).")
    parser.add_argument("hdob_file", help="Path to HDOB text file")
    parser.add_argument("--start", help="UTC start time-of-day (HH:MM or HH:MM:SS or HHMM or HHMMSS)", default=None)
    parser.add_argument("--end", help="UTC end time-of-day (HH:MM or HH:MM:SS or HHMM or HHMMSS)", default=None)
    parser.add_argument("--legend", action="store_true", help="Show color legend on the plot")
    parser.add_argument("--no-plot", action="store_true", help="Do not display/save the plot")
    args = parser.parse_args()
    main(args.hdob_file, start_utc=args.start, end_utc=args.end, show_legend=args.legend, plot_flag=(not args.no_plot))
