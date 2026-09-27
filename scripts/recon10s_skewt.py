#!/usr/bin/env python3
"""
recon10s_skewt — the dropsonde window's figure.

Skew-T on the left (temperature red, dew point green, isotherms grey, dry
adiabats red dashed, moist adiabats blue dashed), wind barbs beside it,
then on the right: mandatory-level table, significant-wind table, and a
satellite inset centred on the release point.

Plain matplotlib, no MetPy: the skew is x = T + SKEW * ln(P_REF / p) on a
log-pressure axis, which is all a skew-T is.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

import recon10s_drops as drops_mod

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover
    ccrs = None
    cfeature = None

try:
    import recon10s_plot
except Exception:  # pragma: no cover
    recon10s_plot = None

SKEW = 38.0          # degC per unit ln(p); isotherms at ~45 degrees
P_REF = 1050.0
RD, CP, LV, EPS = 287.04, 1005.7, 2.501e6, 0.622


# ------------------------------------------------------------- thermodynamics

def _es(t_c):
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def _moist_adiabat(t0_c: float, p: np.ndarray) -> np.ndarray:
    """Pseudo-adiabat through (t0_c, p[0]), integrated with small RK2 steps."""
    out = np.empty_like(p, dtype=float)
    t = t0_c + 273.15
    out[0] = t0_c
    for i in range(1, len(p)):
        p0, p1 = p[i - 1], p[i]
        n = max(1, int(abs(p1 - p0) / 2.0))
        dp = (p1 - p0) / n
        pp = p0
        for _ in range(n):
            def slope(tk, pk):
                rs = EPS * _es(tk - 273.15) / (pk - _es(tk - 273.15))
                return (RD * tk + LV * rs) / (pk * (CP + LV * LV * rs * EPS / (RD * tk * tk)))
            k1 = slope(t, pp)
            k2 = slope(t + dp * k1, pp + dp)
            t += dp * 0.5 * (k1 + k2)
            pp += dp
        out[i] = t - 273.15
    return out


def _x(t, p):
    return np.asarray(t, dtype=float) + SKEW * np.log(P_REF / np.asarray(p, dtype=float))


# ------------------------------------------------------------------- colours

def _wind_rgb(kt: Optional[float]):
    if kt is None:
        return (0.85, 0.85, 0.85)
    if recon10s_plot is not None:
        return recon10s_plot._speed_to_rgb_normalized(float(kt))
    return (1.0, 1.0, 1.0)


def _ramp(value, lo, hi, cmap_name):
    import matplotlib
    cmap = matplotlib.colormaps[cmap_name]
    frac = 0.0 if value is None else float(np.clip((value - lo) / (hi - lo), 0, 1))
    return cmap(0.25 + 0.7 * frac)


def _text_on(rgb) -> str:
    r, g, b = rgb[:3]
    return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "#ffffff"


# ------------------------------------------------------------------ drawing

def _style_table(tbl, fg, edge, fontsize=8.5):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(edge)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(color=fg, weight="bold")


def _thin_wind_levels(levels, ax, min_px=26.0):
    """Keep barbs at least min_px apart vertically; surface and top always stay."""
    if len(levels) <= 2:
        return levels
    ys = ax.transData.transform(np.column_stack([np.zeros(len(levels)),
                                                 [l.p for l in levels]]))[:, 1]
    keep = [0]
    for i in range(1, len(levels) - 1):
        if abs(ys[i] - ys[keep[-1]]) >= min_px and abs(ys[i] - ys[-1]) >= min_px:
            keep.append(i)
    keep.append(len(levels) - 1)
    return [levels[i] for i in keep]


def render(fig, drop: "drops_mod.Drop", sat_img=None, centre=None,
           theme: str = "dark", credit: str = "recon10s"):
    """Draw the whole dropsonde window into `fig`."""
    import matplotlib.ticker as mticker

    dark = theme != "light"
    bg = "#12151b" if dark else "#ffffff"
    fg = "#e6edf5" if dark else "#111111"
    muted = "#8b95a5" if dark else "#555a63"
    grid = "#6b7688" if dark else "#9aa0a8"
    edge = "#3a4150" if dark else "#555555"
    fig.clear()
    fig.patch.set_facecolor(bg)

    levels = drop.thermo_levels
    winds = drop.wind_levels
    ps = [l.p for l in levels] + [l.p for l in winds]
    if not ps:
        fig.text(0.5, 0.5, "This message has no decodable levels.", ha="center", color=fg)
        return
    p_bot = max(1010.0, max(ps) + 20.0)
    p_top = max(100.0, min(ps) - 60.0)

    # ------------------------------------------------------------- skew-T
    ax = fig.add_axes([0.075, 0.08, 0.50, 0.80])
    ax.set_facecolor(bg)
    ax.set_yscale("log")
    ax.set_ylim(p_bot, p_top)
    xs = [x for l in levels for x in (_x(l.t, l.p), _x(l.td, l.p) if l.td is not None else _x(l.t, l.p))]
    x_lo = min(-10.0, (min(xs) if xs else 0) - 8)
    x_hi = max(45.0, (max(xs) if xs else 30) + 8)
    if x_hi - x_lo < 55:
        x_hi = x_lo + 55
    ax.set_xlim(x_lo, x_hi)

    pgrid = np.geomspace(p_bot, p_top, 60)
    for t0 in range(-120, 70, 10):
        ax.plot(_x(t0, pgrid), pgrid, color=grid, lw=1.1, alpha=0.75, zorder=1)
    for theta in range(250, 480, 10):
        t = theta * (pgrid / 1000.0) ** (RD / CP) - 273.15
        ax.plot(_x(t, pgrid), pgrid, color="#ff5a5a", lw=0.8, ls="--", alpha=0.55, zorder=1)
    pm = np.linspace(p_bot, max(p_top, 200.0), 60)
    for t0 in range(-20, 45, 5):
        ax.plot(_x(_moist_adiabat(t0, pm), pm), pm, color="#5a6bff", lw=0.8, ls="--",
                alpha=0.6, zorder=1)

    for tick in (1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100):
        if p_top <= tick <= p_bot:
            ax.axhline(tick, color=grid, lw=1.0, alpha=0.9, zorder=1)
    ax.yaxis.set_major_locator(mticker.FixedLocator(
        [t for t in (1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100) if p_top <= t <= p_bot]))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("Pressure (hPa)", color=fg)
    ax.set_xlabel("Temperature (\u00b0C)", color=fg)
    ax.tick_params(colors=fg, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(fg)

    tp = [l.p for l in levels]
    ax.plot(_x([l.t for l in levels], tp), tp, color="#ff2a2a", lw=3.0, marker="o",
            ms=6, zorder=5, label="Temperature")
    dlev = [l for l in levels if l.td is not None]
    if dlev:
        ax.plot(_x([l.td for l in dlev], [l.p for l in dlev]), [l.p for l in dlev],
                color="#12a53a", lw=3.0, marker="o", ms=6, zorder=6, label="Dew point")

    lines = []
    for label, mean, reduced in drops_mod.surface_estimates(drop):
        lines.append(f"Mean wind in {label}: {mean} kt")
        lines.append(f"Reduction to surface: {reduced:.1f} kt")
    if drop.surface and drop.surface.p:
        lines.append(f"Surface pressure: {drop.surface.p:.0f} mb")
    if lines:
        ax.text(0.015, 0.985, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
                fontsize=10.5, color=fg, zorder=10, linespacing=1.45,
                bbox=dict(boxstyle="round,pad=0.45", fc=bg, ec=muted, lw=1.2))

    # ------------------------------------------------------------- barbs
    bx = fig.add_axes([0.585, 0.08, 0.045, 0.80], sharey=ax)
    bx.set_facecolor(bg)
    bx.set_xlim(-1, 1)
    bx.axis("off")
    fig.canvas.draw()  # transforms must be current for spacing
    shown = _thin_wind_levels(winds, ax)
    if shown:
        spd = np.array([l.wspd for l in shown], dtype=float)
        rad = np.deg2rad([l.wdir or 0 for l in shown])
        bx.barbs(np.zeros(len(shown)), [l.p for l in shown], -spd * np.sin(rad),
                 -spd * np.cos(rad), length=7, color=fg, linewidth=1.0, zorder=5)

    # ------------------------------------------------------------- titles
    lat, lon = drop.release_lat, drop.release_lon
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    when = drop.release_time
    fig.text(0.075, 0.955, f"#{drop.number}  {drop.title_name}, at {abs(lat):.2f}\u00b0{ns}, "
                           f"{abs(lon):.2f}\u00b0{ew}", fontsize=13, color=fg, ha="left")
    if when is not None:
        fig.text(0.075, 0.922, f"Dropped at {when:%H:%M:%S}z, {when:%b %d, %Y}",
                 fontsize=13, color=fg, ha="left")
    bits = [b for b in (drop.storm.title() if drop.storm else "", drop.mission,
                        f"OB {drop.ob}" if drop.ob else "") if b]
    fig.text(0.63, 0.922, "  \u00b7  ".join(bits), fontsize=10.5, color=muted, ha="right")
    fig.text(0.985, 0.012, credit, fontsize=8, color=muted, ha="right")

    # ---------------------------------------------------- mandatory table
    rows, colours = [], []
    table_levels = [l for l in [drop.surface] + drop.mandatory
                    if l is not None and (l.t is not None or l.wspd is not None)]
    table_levels.sort(key=lambda l: l.p)
    for l in table_levels:
        name = f"{l.p:.0f}mb"
        height = "" if l.z is None else f"{l.z:.0f}m"
        wind = "" if l.wspd is None else f"{l.wspd} kts"
        temp = "" if l.t is None else f"{l.t:.1f}\u00b0C"
        rh = "" if l.rh is None else f"{l.rh:.0f}%"
        rows.append([name, height, wind, temp, rh])
        colours.append([bg, bg, _wind_rgb(l.wspd) if l.wspd is not None else bg,
                        _ramp(l.t, 0, 32, "Oranges") if l.t is not None else bg,
                        _ramp(l.rh, 40, 100, "Purples") if l.rh is not None else bg])
    top = 0.905
    if rows:
        h = 0.028 * (len(rows) + 1)
        tax = fig.add_axes([0.68, top - h, 0.305, h])
        tax.axis("off")
        tbl = tax.table(cellText=rows, colLabels=["Pressure", "Height", "Wind", "Temp", "RH"],
                        cellColours=colours, colColours=[bg] * 5, loc="center",
                        cellLoc="center", bbox=[0, 0, 1, 1])
        _style_table(tbl, fg, edge)
        for (r, c), cell in tbl.get_celld().items():
            if r > 0:
                cell.set_text_props(color=_text_on(cell.get_facecolor()) if c >= 2 else fg)
        top -= h + 0.035

    # --------------------------------------------- significant wind table
    wl = sorted(winds, key=lambda l: l.p)
    if len(wl) > 17:
        # Long profiles (high-altitude drops) are thinned, but never at the
        # expense of the top, the surface, or the strongest wind.
        strongest = max(range(len(wl)), key=lambda i: wl[i].wspd or 0)
        pick = set(np.linspace(0, len(wl) - 1, 15).round().astype(int)) | {0, len(wl) - 1, strongest}
        wl = [wl[i] for i in sorted(pick)]
    if wl:
        rows = []
        colours = []
        sfc_p = drop.surface.p if drop.surface else None
        for l in wl:
            label = f"{l.p:.0f}mb" + (" (Surface)" if sfc_p and abs(l.p - sfc_p) < 0.5 else "")
            rows.append([label, f"{l.wspd} kts"])
            colours.append([bg, _wind_rgb(l.wspd)])
        h = min(0.022 * (len(rows) + 1), top - 0.35)
        wax = fig.add_axes([0.68, top - h, 0.22, h])
        wax.axis("off")
        tbl = wax.table(cellText=rows, colLabels=["Pressure", "Wind"], cellColours=colours,
                        colColours=[bg, bg], loc="center", bbox=[0, 0, 1, 1])
        _style_table(tbl, fg, edge, fontsize=8.5)
        for (r, c), cell in tbl.get_celld().items():
            if r > 0:
                cell._loc = "right"
                cell.set_text_props(color=_text_on(cell.get_facecolor()) if c == 1 else fg)

    # --------------------------------------------------- satellite inset
    _inset(fig, drop, sat_img, centre, dark, fg, muted)
    loc = drops_mod.describe_location(drop, centre)
    if loc:
        fig.text(0.68, 0.035, f"Location: {loc}", fontsize=11.5, color=fg, ha="left")


def _inset(fig, drop, sat_img, centre, dark, fg, muted):
    if ccrs is None:
        return
    half = 2.4
    lat, lon = drop.release_lat, drop.release_lon
    extent = (lon - half, lon + half, lat - half, lat + half)
    rect = [0.68, 0.07, 0.22, 0.26]
    if sat_img is not None:
        crs = ccrs.Geostationary(central_longitude=sat_img.lon0,
                                 satellite_height=sat_img.sat_height, sweep_axis=sat_img.sweep)
        ax = fig.add_axes(rect, projection=crs)
        if sat_img.kind == "brightness":
            ax.imshow(np.ma.masked_invalid(sat_img.data), origin="upper", extent=sat_img.extent_m,
                      transform=crs, cmap="gray_r", vmin=-85, vmax=30, interpolation="nearest")
        else:
            ax.imshow(np.sqrt(np.clip(np.nan_to_num(sat_img.data), 0, 1)), origin="upper",
                      extent=sat_img.extent_m, transform=crs, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
        ax.text(0.02, 0.02, f"GOES-{sat_img.scene.sat} B{sat_img.scene.band} "
                            f"{sat_img.start:%H:%MZ}", transform=ax.transAxes, fontsize=7,
                color="#ffffff", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="#000000aa", ec="none"))
    else:
        ax = fig.add_axes(rect, projection=ccrs.PlateCarree())
        ax.set_facecolor("#0b0d11" if dark else "#dfe5ec")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#ffffff", linewidth=0.6)
    pc = ccrs.PlateCarree()
    if drop.splash_lat is not None:
        ax.plot([lon, drop.splash_lon], [lat, drop.splash_lat], transform=pc,
                color="#ffffff", lw=1.0, zorder=5)
    if centre is not None:
        ax.plot(centre[1], centre[0], marker="+", ms=9, mew=1.6, color="#ff5252",
                transform=pc, zorder=6)
    ax.plot(lon, lat, marker="o", ms=7, mfc="#ffffff", mec="#000000", mew=1.2,
            transform=pc, zorder=7)
    ax.set_extent(extent, crs=pc)
    for spine in ax.spines.values():
        spine.set_edgecolor(fg)
