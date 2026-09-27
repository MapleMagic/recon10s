#!/usr/bin/env python3
"""
recon10s_mwview — turning a microwave pass into pictures, and pixels into numbers.

Every product definition here is NRL's, taken from GeoIPS source rather than
from memory, so the imagery matches what forecasters are used to:

  color89   R = 1.818V - 0.818H over 220-310 K, inverted
            G = H over 240-300 K,  B = V over 270-290 K
  color37   R = 2.181V - 1.181H over 260-280 K, inverted
            G = V over 180-300 K,  B = H over 160-300 K
  89PCT     1.7V - 0.7H      (standalone product; Cecil & Chronis 2018)
  37PCT     2.15V - 1.15H    (standalone product)

The composite and standalone PCTs deliberately differ -- NRL kept the older
weights inside the colour recipes. Standalone H and V use NRL's 89H and 37H
colour tables.

Display goes through a regular lat/lon grid (nearest neighbour, no smoothing,
so no values are invented), but the pixel probe never reads that grid: it
finds the nearest native footprint and reports its actual TB.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

import recon10s_mw as mw

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover
    ccrs = None
    cfeature = None

PRODUCTS = ("Color", "H", "V", "PCT")


# ------------------------------------------------------------- colour maps

def _segmented(name, vmin, vmax, transitions, colours):
    """
    Rebuild a GeoIPS create_linear_segmented_colormap: each (lo, hi) span
    blends its own colour pair, and a new span may jump to a new colour --
    those jumps are what give the NRL tables their hard edges.
    """
    span = float(vmax - vmin)
    red, green, blue = [], [], []
    for i, ((lo, hi), (c0, c1)) in enumerate(zip(transitions, colours)):
        a, b = to_rgb(c0), to_rgb(c1)
        x0, x1 = (lo - vmin) / span, (hi - vmin) / span
        prev = to_rgb(colours[i - 1][1]) if i else a
        for chan, lst in ((0, red), (1, green), (2, blue)):
            lst.append((x0, prev[chan], a[chan]))
            if i == len(transitions) - 1:
                lst.append((x1, b[chan], b[chan]))
    cmap = LinearSegmentedColormap(name, {"red": red, "green": green, "blue": blue}, N=1024)
    cmap.set_bad(alpha=0.0)
    return cmap


def colormap_for(band: str, product: str):
    """(cmap, norm, ticks) for a standalone H, V or PCT product."""
    if band == "89" and product in ("H", "V"):
        vmin, vmax = 105.0, 305.0
        tr = [(vmin, 180), (180, 212), (212, 228), (228.1, 254), (254.1, 280), (280, vmax)]
        cl = [("white", "black"), ("#A4641A", "#FC0603"), ("#F4CD03", "#F2F403"),
              ("#8CF303", "#0FB503"), ("#06DCFD", "#0708B5"), ("navy", "white")]
        ticks = [105, 150, 180, 212, 228, 254, 280, 305]
    elif band == "89":
        vmin, vmax = 105.0, 280.0
        tr = [(vmin, 125), (125, 150), (150, 175), (175, 212), (212, 230),
              (230, 250), (250, 265), (265, vmax)]
        cl = [("orange", "chocolate"), ("chocolate", "indianred"), ("indianred", "firebrick"),
              ("firebrick", "red"), ("gold", "yellow"), ("lime", "limegreen"),
              ("deepskyblue", "blue"), ("navy", "slateblue")]
        ticks = [105, 125, 150, 175, 212, 230, 250, 265, 280]
    elif product in ("H", "V"):
        vmin, vmax = 125.0, 310.0
        tr = [(vmin, 180), (180, 195), (195, 210), (210, 220), (220, 230),
              (230, 240), (240, 260), (260, 280), (280, vmax)]
        cl = [("lightyellow", "darkmagenta"), ("#80007F", "#0080FF"), ("#0080FF", "#3AB9FF"),
              ("#3AB9FF", "#7DFDFF"), ("#7DFDFF", "#80FF82"), ("#80FF82", "#FFFF80"),
              ("#FFFF80", "#FF8000"), ("#FF8000", "#800000"), ("silver", "black")]
        ticks = [125, 150, 180, 200, 220, 240, 260, 280, 300, 310]
    else:
        vmin, vmax = 230.0, 280.0
        tr = [(vmin, 240), (240, 260), (260, vmax)]
        cl = [("cyan", "yellow"), ("yellow", "red"), ("red", "darkred")]
        ticks = [230, 240, 250, 260, 270, 280]

    cmap = _segmented(f"nrl_{band}{product}", vmin, vmax, tr, cl)
    cmap.set_under(to_rgb(cl[0][0]))
    cmap.set_over(to_rgb(cl[-1][1]))
    return cmap, Normalize(vmin=vmin, vmax=vmax), ticks


def _scale(x, lo, hi, inverse=False):
    out = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - out if inverse else out


def color_composite(band: str, v: np.ndarray, h: np.ndarray) -> np.ndarray:
    """NRL colour composite as an (..., 4) RGBA float array; NaN -> transparent."""
    if band == "89":
        r = _scale(1.818 * v - 0.818 * h, 220.0, 310.0, inverse=True)
        g = _scale(h, 240.0, 300.0)
        b = _scale(v, 270.0, 290.0)
    else:
        r = _scale(2.181 * v - 1.181 * h, 260.0, 280.0, inverse=True)
        g = _scale(v, 180.0, 300.0)
        b = _scale(h, 160.0, 300.0)
    a = (np.isfinite(v) & np.isfinite(h)).astype(float)
    rgba = np.stack([np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b), a], axis=-1)
    return rgba


def product_values(band: str, product: str, v, h):
    if product == "H":
        return h
    if product == "V":
        return v
    return mw.pct(band, v, h)


# ------------------------------------------------------------- regridding

@dataclass
class Grid:
    lats: np.ndarray       # 1-D, south to north
    lons: np.ndarray       # 1-D, west to east
    index: np.ndarray      # (ny, nx) native pixel index, -1 where no data
    spacing_km: float

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        dlon = self.lons[1] - self.lons[0] if self.lons.size > 1 else 0.05
        dlat = self.lats[1] - self.lats[0] if self.lats.size > 1 else 0.05
        return (self.lons[0] - dlon / 2, self.lons[-1] + dlon / 2,
                self.lats[0] - dlat / 2, self.lats[-1] + dlat / 2)


def _unit_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack((np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)))


class BandIndex:
    """KD-tree over one band's native footprints: regridding and the probe."""

    def __init__(self, swath: "mw.BandSwath"):
        if cKDTree is None:
            raise RuntimeError("scipy is required for the microwave display (pip install scipy)")
        self.swath = swath
        self.tree = cKDTree(_unit_xyz(swath.lat, swath.lon))
        # Native spacing, from each footprint's nearest neighbour.
        sample = self.tree.data[:: max(1, len(self.tree.data) // 4000)]
        d, _ = self.tree.query(sample, k=2)
        self.spacing_km = float(np.median(d[:, 1])) * 6371.0

    def grid(self, extent: Tuple[float, float, float, float]) -> Grid:
        """
        Nearest-neighbour grid at about half the native spacing. Cells further
        than 1.2 footprints from any pixel stay empty, so the swath edge and
        any gaps show as gaps instead of smeared data.
        """
        west, east, south, north = extent
        res = float(np.clip(self.spacing_km / 2.0 / 111.0, 0.015, 0.1))
        lats = np.arange(south, north + res / 2, res)
        lons = np.arange(west, east + res / 2, res)
        glon, glat = np.meshgrid(lons, lats)
        dist, idx = self.tree.query(_unit_xyz(glat.ravel(), glon.ravel()))
        idx = idx.astype(np.int64)
        idx[dist * 6371.0 > 1.2 * self.spacing_km] = -1
        return Grid(lats=lats, lons=lons, index=idx.reshape(glat.shape),
                    spacing_km=self.spacing_km)

    def probe(self, lat: float, lon: float, band: str) -> Optional[Dict[str, float]]:
        """The native footprint nearest (lat, lon), or None if off the swath."""
        dist, i = self.tree.query(_unit_xyz(np.array([lat]), np.array([lon]))[0])
        km = float(dist) * 6371.0
        if km > 1.5 * self.spacing_km:
            return None
        s = self.swath
        v, h = float(s.v[i]), float(s.h[i])
        return {"lat": float(s.lat[i]), "lon": float(s.lon[i]), "km": km,
                "V": v, "H": h, "PCT": float(mw.pct(band, v, h)),
                "scan": int(s.scan[i]), "pixel": int(s.pixel[i]), "group": s.group}


def gridded_image(index: BandIndex, grid: Grid, band: str, product: str):
    """What imshow needs: an RGBA array for Color, else a masked value array."""
    s = index.swath
    valid = grid.index >= 0
    safe = np.where(valid, grid.index, 0)
    v = np.where(valid, s.v[safe], np.nan)
    h = np.where(valid, s.h[safe], np.nan)
    if product == "Color":
        return color_composite(band, v, h)
    return np.ma.masked_invalid(product_values(band, product, v, h))


# ------------------------------------------------------------------ drawing

def default_extent(centre: Tuple[float, float], half_deg: float = 3.0):
    lat, lon = centre
    return (lon - half_deg, lon + half_deg, lat - half_deg, lat + half_deg)


def draw_band(fig, pass_: "mw.MWPass", index: Optional[BandIndex], band: str, product: str,
              extent, theme: str = "dark", show_centre: bool = True, colorbar: bool = True):
    """Draw one panel into `fig` and return its axes."""
    if ccrs is None:
        raise RuntimeError("cartopy is required for the map (pip install cartopy)")
    fig.clear()
    dark = theme != "light"
    fg = "#dfe5ec" if dark else "#1c2129"
    bg = "#12151b" if dark else "#ffffff"
    fig.patch.set_facecolor(bg)

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_facecolor("#05070a" if dark else "#d9dee5")

    freq = {"89": "89 GHz", "37": "37 GHz"}[band]
    label = {"Color": "colour", "H": "H-pol", "V": "V-pol", "PCT": "PCT"}[product]

    if index is None or index.swath.lat.size == 0:
        ax.text(0.5, 0.5, f"No {freq} data over the storm in this pass",
                transform=ax.transAxes, ha="center", va="center", color=fg, fontsize=9)
    else:
        grid = index.grid(extent)
        img = gridded_image(index, grid, band, product)
        if product == "Color":
            ax.imshow(img, origin="lower", extent=grid.extent, transform=ccrs.PlateCarree(),
                      interpolation="nearest", zorder=1)
        else:
            cmap, norm, ticks = colormap_for(band, product)
            im = ax.imshow(img, origin="lower", extent=grid.extent, transform=ccrs.PlateCarree(),
                           cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
            if colorbar:
                cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045,
                                  pad=0.06, ticks=ticks, extend="both")
                cb.set_label("Brightness temperature (K)", color=fg, fontsize=8)
                cb.ax.tick_params(colors=fg, labelsize=7)
                cb.outline.set_edgecolor(fg)

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#ffffff" if dark else "#222222", linewidth=0.6, zorder=3)
    if show_centre:
        ax.plot(pass_.centre[1], pass_.centre[0], marker="+", markersize=12, mew=2.0,
                color="#ffffff", transform=ccrs.PlateCarree(), zorder=5)
        ax.plot(pass_.centre[1], pass_.centre[0], marker="+", markersize=12, mew=0.9,
                color="#ff5252", transform=ccrs.PlateCarree(), zorder=6)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    grid_lines = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.3,
                              color="#ffffff" if dark else "#888888")
    grid_lines.top_labels = grid_lines.right_labels = False
    for attr in ("xlabel_style", "ylabel_style"):
        getattr(grid_lines, attr).update({"size": 7, "color": fg})

    ax.set_title(f"{pass_.title}\n{freq} {label}", color=fg, fontsize=9)
    return ax
