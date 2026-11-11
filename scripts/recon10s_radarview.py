#!/usr/bin/env python3
"""
recon10s_radarview — draws NEXRAD sweeps: one, two or four map panels, each
with its own product, sharing the same view. The map projection is an
azimuthal equidistant one centred on the radar, which is exactly the frame
the sweep is resampled into, so the image needs no warping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import recon10s_nexrad as nx

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover
    ccrs = cfeature = None

LAYOUTS = {1: (1, 1), 2: (1, 2), 4: (2, 2)}


@dataclass
class PanelImage:
    product: str
    img: np.ndarray
    extent: Tuple[float, float, float, float]
    elevation: float
    nyquist: Optional[float] = None
    sweep: Optional[int] = None


@dataclass
class RadarArtists:
    axes: list = field(default_factory=list)
    panels: List[Optional[PanelImage]] = field(default_factory=list)
    crs: object = None


def render_panels(vol: "nx.Volume", products: List[str], elevation: float, view_centre=(0.0, 0.0),
                  half_width_m: float = 250e3, dealias: bool = True, n: int = 800,
                  motion: Optional[Tuple[float, float]] = None) -> List[Optional[PanelImage]]:
    """
    One resampled sweep per product. "srv" (storm-relative velocity) is the
    radial velocity with `motion` (heading toward, kt) removed along each
    ray; without a motion it is left out.
    """
    out = []
    for p in products:
        base = nx.BASE_PRODUCT.get(p, p)
        sweep = vol.sweep_for(base, elevation)
        if sweep is None or (p == "srv" and motion is None):
            out.append(None)
            continue
        az, rng, vals, el, nyq = vol.sweep_data(base, sweep, dealias=dealias)
        if p == "srv":
            vals = nx.storm_relative(az, vals, motion)
        img, extent = nx.to_grid(az, rng, vals, el, view_centre, half_width_m, n)
        pim = PanelImage(p, img, extent, el, nyq, sweep)
        pim.motion = motion if p == "srv" else None
        out.append(pim)
    return out


def draw(fig, vol: "nx.Volume", panels: List[Optional[PanelImage]], storm=None, fixes=(),
         theme: str = "dark", rings_km=(50, 100, 150, 200, 250), notes=None) -> RadarArtists:
    """Lay out the panels and draw them. `storm` is (lat, lon, label); `notes` replaces the empty-panel text."""
    fig.clear()
    dark = theme != "light"
    fg = "#dfe5ec" if dark else "#1c2129"
    bg = "#12151b" if dark else "#ffffff"
    fig.patch.set_facecolor(bg)
    crs = ccrs.AzimuthalEquidistant(central_longitude=vol.site_lon, central_latitude=vol.site_lat)
    rows, cols = LAYOUTS.get(len(panels), (1, 1))
    art = RadarArtists(crs=crs, panels=panels)
    pc = ccrs.PlateCarree()
    site = nx_site_label(vol)
    for k, pim in enumerate(panels):
        ax = fig.add_subplot(rows, cols, k + 1, projection=crs)
        ax.set_facecolor("#0b0d11" if dark else "#f4f6f8")
        art.axes.append(ax)
        if pim is None:
            ax.set_extent(_extent_of(panels), crs=crs) if any(panels) else None
            ax.text(0.5, 0.5, (notes[k] if notes and k < len(notes) and notes[k] else
                               "Not in this scan at this elevation"), transform=ax.transAxes,
                    ha="center", va="center", color="#8b95a5", fontsize=9)
            continue
        cmap, norm, ticks = nx.colour_table(pim.product)
        im = ax.imshow(np.ma.masked_invalid(pim.img), extent=pim.extent, origin="upper",
                       transform=crs, cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
        ax.set_extent(pim.extent, crs=crs)
        ax.set_autoscale_on(False)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7,
                       edgecolor="#e8edf2" if dark else "#222222", zorder=3)
        ax.add_feature(cfeature.STATES.with_scale("10m"), linewidth=0.35,
                       edgecolor="#8b95a5", zorder=3)
        for r in rings_km:
            t = np.linspace(0, 2 * np.pi, 181)
            ax.plot(r * 1e3 * np.sin(t), r * 1e3 * np.cos(t), color="#8b95a5", lw=0.5, alpha=0.5,
                    transform=crs, zorder=2)
        ax.plot(0, 0, marker="^", ms=6, color="#ffffff", mec="#000000", transform=crs, zorder=5)
        if storm is not None:
            ax.plot(storm[1], storm[0], marker="+", ms=14, mew=2.2, color="#ff4081", transform=pc, zorder=6)
        for f in fixes:
            ax.plot(f.lon, f.lat, marker="D", ms=6, mfc="#ffca28", mec="#000000", transform=pc, zorder=6)
        info = nx.PRODUCTS[pim.product]
        extra = f"  (Nyquist {pim.nyquist * nx.MS_TO_KT:.0f} kt)" if pim.nyquist else ""
        if getattr(pim, "motion", None):
            extra = f"  (motion {pim.motion[0]:03.0f}\u00b0/{pim.motion[1]:.0f} kt removed)"
        ax.set_title(f"{info['label']} {pim.elevation:.1f}\u00b0{extra}", color=fg, fontsize=9, pad=3)
        cax = ax.inset_axes([1.015, 0.0, 0.025, 1.0])
        cb = fig.colorbar(im, cax=cax, ticks=ticks)
        cb.ax.tick_params(labelsize=7, colors=fg)
        if info["units"]:
            cb.set_label(info["units"], color=fg, fontsize=7)
    fig.suptitle(f"{site}  \u00b7  {vol.scan.time:%d %b %Y %H:%M:%SZ}"
                 + (f"  \u00b7  VCP {vol.vcp}" if vol.vcp else ""), color=fg, fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.93, bottom=0.03, top=0.91, wspace=0.18, hspace=0.12)
    return art


def _extent_of(panels):
    for p in panels:
        if p is not None:
            return p.extent
    return None


def nx_site_label(vol) -> str:
    if getattr(vol, "site_label", None):          # JMA volumes name themselves
        return vol.site_label
    try:
        import recon10s_nexradsites as sites
        s = sites.STATIONS_BY_ICAO.get(vol.scan.site)
        if s:
            return f"{s.icao} {s.name.title()}, {s.state}"
    except Exception:
        pass
    return vol.scan.site


def value_at(art: RadarArtists, ax, x, y) -> Optional[Tuple[str, float]]:
    """(product, value) of the panel `ax` at projection coordinates x, y."""
    if ax not in art.axes:
        return None
    pim = art.panels[art.axes.index(ax)]
    if pim is None:
        return None
    x0, x1, y0, y1 = pim.extent
    n = pim.img.shape[0]
    col = int(round((x - x0) / (x1 - x0) * (n - 1)))
    row = int(round((y1 - y) / (y1 - y0) * (n - 1)))
    if 0 <= row < n and 0 <= col < n:
        return pim.product, float(pim.img[row, col])
    return None
