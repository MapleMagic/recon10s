#!/usr/bin/env python3
"""
recon10s_plot.py — HDOB plotter with coord_format and plot_theme support

Functions:
 - parse_hdob_file(hdob_file) -> times, times_tup, lats, lons, mslps, wind_dirs, wind_spds
 - _time_input_to_seconds(s) -> seconds-of-day or None
 - main(hdob_file, start_utc=None, end_utc=None, show_legend=False,
        coord_format='decimal', plot_theme='dark', show_plot=True)

Plot themes:
 - 'dark': black background, white labels/lines
 - 'light': white background, dark labels/lines

Coordinate formats:
 - 'decimal' : 4 decimal places (e.g. 19.3457)
 - 'dms'     : degrees°minutes'seconds" + hemisphere (e.g. 19°20'44"N)

Saved PNG is written next to the HDOB file with the same basename (e.g. myfile.png).
"""
from __future__ import annotations
import os
import re
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Regex tokens for HDOB-style lat/lon tokens (LLMMH / LLLMMH)
_RE_LAT = re.compile(r"^(\d{2})(\d{2})(\d{2})?([NS])$")
_RE_LON = re.compile(r"^(\d{3})(\d{2})(\d{2})?([EW])$")
_RE_WIND = re.compile(r"^(\d{3})(\d{3})$")
_RE_PPPP = re.compile(r"^\d{4}$")

# Color buckets in knots as specified by user (RGB tuples)
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
    """Parse HDOB tokens like 1917N, 05758W or 191704N, 0575804W (supports optional seconds)"""
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
    """Try to locate PPPP and WWWSSS tokens in the token list."""
    pppp_idx = None
    wind_tuple = None
    for idx, tok in enumerate(parts):
        t = tok.strip()
        m = _RE_WIND.match(t)
        if m:
            d = int(m.group(1)); s = int(m.group(2))
            if 0 <= d <= 360 and 0 <= s <= 300:
                wind_tuple = (d, s)
                # don't return early — prefer to find explicit PPPP too
                continue
        if _RE_PPPP.match(t):
            try:
                val = int(t)
                # PPPP is tenths of hPa; restrict to plausible range
                if 8000 <= val <= 11000:
                    pppp_idx = idx
                    continue
            except Exception:
                pass
    return pppp_idx, wind_tuple

def _parse_time_token(tstr: str):
    """Parse hhmmss or hhmm token into (hh,mm,ss)."""
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
    """
    Parse the HDOB file and return:
    times (list of raw time tokens), times_tup ([(hh,mm,ss)...]),
    lats, lons, mslps (hPa), wind_dirs (deg), wind_spds (kt)
    """
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
                # try to find lat/lon somewhere else in the tokens
                found = False
                for i in range(1, len(parts)-1):
                    maybe = _tok_to_latlon(parts[i], parts[i+1])
                    if maybe:
                        latlon = maybe; found = True; break
                if not found:
                    continue
            lat_dec, lon_dec = latlon
            pppp_idx, wind_tok = _find_mslp_and_wind(parts)
            mslp_val = None
            if pppp_idx is not None:
                try:
                    mslp_val = int(parts[pppp_idx]) / 10.0
                except Exception:
                    mslp_val = None
            else:
                # fallback scanning other likely spots
                for alt in (3,4,5,6):
                    if alt < len(parts) and _RE_PPPP.match(parts[alt]):
                        try:
                            vv = int(parts[alt])
                            if 8000 <= vv <= 11000:
                                mslp_val = vv / 10.0; break
                        except Exception:
                            pass
            dir_deg = None; spd_kt = None
            if wind_tok is not None:
                dir_deg, spd_kt = wind_tok
            ttt = _parse_time_token(time_tok)
            if ttt is None: continue
            times.append(time_tok); times_tup.append(ttt)
            lats.append(lat_dec); lons.append(lon_dec); mslps.append(mslp_val)
            wind_dirs.append(dir_deg); wind_spds.append(spd_kt)
    return times, times_tup, lats, lons, mslps, wind_dirs, wind_spds

# expose helper for GUI usage
_time_input_to_seconds = _hhmm_to_seconds

def _decimal_to_dms_str(val: float, is_lat: bool) -> str:
    """Convert decimal degrees to DMS string with hemisphere."""
    hemi = ""
    if is_lat:
        hemi = "N" if val >= 0 else "S"
    else:
        hemi = "E" if val >= 0 else "W"
    aval = abs(val)
    deg = int(aval)
    minutes = int((aval - deg) * 60)
    seconds = int(round((aval - deg - minutes/60.0) * 3600.0))
    # fix rollovers
    if seconds == 60:
        seconds = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        deg += 1
    return f"{deg}°{minutes:02d}'{seconds:02d}\"{hemi}"

def _format_coord_for_display(lat: float, lon: float, coord_format: str = "decimal") -> str:
    if coord_format == "dms":
        return f"{_decimal_to_dms_str(lat, True)}, {_decimal_to_dms_str(lon, False)}"
    else:
        return f"{lat:.4f}, {lon:.4f}"

def main(hdob_file: str,
         start_utc: Optional[str] = None,
         end_utc: Optional[str] = None,
         show_legend: bool = False,
         coord_format: str = "decimal",
         plot_theme: str = "dark",
         show_plot: bool = True):
    """
    Create a plot for HDOB file.

    coord_format: 'decimal' or 'dms'
    plot_theme: 'dark' or 'light'
    show_plot: if False, only save PNG and return
    """
    if not os.path.exists(hdob_file):
        raise FileNotFoundError(hdob_file)

    times, times_tup, lats, lons, mslps, wind_dirs, wind_spds = parse_hdob_file(hdob_file)
    if not lats:
        print("No valid HDOB points found.")
        return

    secs = [hh*3600 + mm*60 + ss for (hh, mm, ss) in times_tup]
    start_sec = _hhmm_to_seconds(start_utc) if start_utc else None
    end_sec = _hhmm_to_seconds(end_utc) if end_utc else None
    idxs = []
    for i, s in enumerate(secs):
        keep = True
        if start_sec is not None and end_sec is not None:
            if start_sec <= end_sec:
                keep = (start_sec <= s <= end_sec)
            else:
                keep = (s >= start_sec or s <= end_sec)
        elif start_sec is not None:
            keep = (s >= start_sec)
        elif end_sec is not None:
            keep = (s <= end_sec)
        if keep:
            idxs.append(i)

    print(f"Total records: {len(lats)}; filtered for window: {len(idxs)}")
    if not idxs:
        print("No records in requested window.")
        return

    px=[]; py=[]; uu=[]; vv=[]; colors=[]; pm=[]
    for i in idxs:
        d = wind_dirs[i]; s = wind_spds[i]
        if d is None or s is None:
            continue
        lon = lons[i]; lat = lats[i]
        u, v = _wind_to_uv_knots(d, s)
        c = _speed_to_rgb_normalized(s)
        px.append(lon); py.append(lat); uu.append(u); vv.append(v); colors.append(c); pm.append(mslps[i])
    if not px:
        print("No wind-bearing points to plot in the selected window.")
        return

    px = np.array(px); py = np.array(py); uu = np.array(uu); vv = np.array(vv); pm = np.array(pm)
    color_groups = {}
    for i, c in enumerate(colors):
        color_groups.setdefault(c, []).append(i)

    # Choose theme options
    dark = (plot_theme == "dark")
    if dark:
        fig = plt.figure(figsize=(11, 8), facecolor="black")
        ax = plt.axes(projection=ccrs.Mercator(), facecolor="black")
        land_face = "#222222"; coast_color = "#cccccc"; border_color = "#888888"
        grid_color = "white"; grid_alpha = 0.25; text_color = "white"
    else:
        fig = plt.figure(figsize=(11, 8), facecolor="white")
        ax = plt.axes(projection=ccrs.Mercator(), facecolor="white")
        land_face = "#eaeaea"; coast_color = "#222222"; border_color = "#333333"
        grid_color = "black"; grid_alpha = 0.2; text_color = "black"

    lonmin, lonmax = float(np.min(px)), float(np.max(px))
    latmin, latmax = float(np.min(py)), float(np.max(py))
    margin_lon = max(0.5, (lonmax - lonmin) * 0.12)
    margin_lat = max(0.5, (latmax - latmin) * 0.12)
    ax.set_extent([lonmin - margin_lon, lonmax + margin_lon, latmin - margin_lat, latmax + margin_lat], crs=ccrs.PlateCarree())

    # Map features
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=land_face, edgecolor=land_face)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=coast_color, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=border_color, linestyle=":", linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linewidth=0.4, color=grid_color, alpha=grid_alpha, linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    try:
        gl.xlabel_style = {"color": text_color}
        gl.ylabel_style = {"color": text_color}
    except Exception:
        pass

    # Plot barbs grouped by color
    for color, inds in color_groups.items():
        xs = px[inds]; ys = py[inds]; us = uu[inds]; vs = vv[inds]
        # matplotlib barbs interpret u/v in data units; we used knots for u/v
        ax.barbs(xs, ys, us, vs, length=6, transform=ccrs.PlateCarree(), color=color, linewidth=0.8)

    # MSLP labels — choose text color vs theme
    for x, y, m in zip(px, py, pm):
        if m is not None:
            text = f"{m:.1f}"
            # place label slightly offset
            ax.text(x + 0.02, y + 0.02, text, color=text_color, fontsize=8, transform=ccrs.PlateCarree(),
                    ha="left", va="bottom", bbox={"facecolor":"none","edgecolor":"none","pad":0})

    # If requested, annotate coordinates next to the first point (small) or on hover — keep simple: label each barb with lat/lon small
    # To avoid clutter, we annotate only every Nth point based on number of points
    n_points = len(px)
    step = max(1, n_points // 40)  # aim for up to ~40 labels
    for i in range(0, n_points, step):
        x = px[i]; y = py[i]
        coord_text = _format_coord_for_display(y, x, coord_format)  # lat, lon
        ax.text(x - 0.02, y - 0.02, coord_text, color=text_color, fontsize=7, transform=ccrs.PlateCarree(), ha="right", va="top", alpha=0.9)

    # Legend
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
        # adjust legend aesthetics for theme
        if dark:
            leg.get_frame().set_facecolor("#111111")
            leg.get_frame().set_edgecolor("white")
            for text in leg.get_texts(): text.set_color("white")
            if leg.get_title(): leg.get_title().set_color("white")
        else:
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("black")
            for text in leg.get_texts(): text.set_color("black")
            if leg.get_title(): leg.get_title().set_color("black")

    # Title and final touches
    title_time = ""
    if start_utc or end_utc:
        title_time = f" ({start_utc or '00:00'}–{end_utc or '23:59'} UTC)"
    ax.set_title(f"HDOB: {os.path.basename(hdob_file)} — Flight-level winds & MSLP{title_time}", color=text_color)

    out_png = os.path.splitext(hdob_file)[0] + ".png"
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved plot: {out_png}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
