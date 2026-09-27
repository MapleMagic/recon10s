#!/usr/bin/env python3
"""
recon10s_satview — the map that lives beside the HDOB text.

Split in two on purpose: `draw_scene` is plain matplotlib and can be rendered
and checked without a GUI, while `SatellitePanel` is the Qt wrapper that adds
the controls and does the fetching off the main thread.

The satellite image is drawn in its own geostationary projection with no
resampling: ABI fixed-grid scan angles times the satellite height give
projection metres, which is exactly what cartopy's Geostationary CRS wants.
The recon track goes on top in PlateCarree and cartopy handles the rest.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

import recon10s_irtable as irtable

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

TRACK_OUTLINE = "#101318"


@dataclass
class Overlay:
    """Recon observations drawn over the imagery, with real datetimes."""
    lats: List[float] = field(default_factory=list)
    lons: List[float] = field(default_factory=list)
    winds: List[Optional[int]] = field(default_factory=list)
    dirs: List[Optional[int]] = field(default_factory=list)
    mslps: List[Optional[float]] = field(default_factory=list)
    times: List[Optional[dt.datetime]] = field(default_factory=list)
    segments: List[int] = field(default_factory=list)   # one id per track (flight)

    def __len__(self) -> int:
        return len(self.lats)

    @property
    def bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """(south, north, west, east) around the track, with a little room."""
        if not self.lats:
            return None
        pad = 1.0
        return (min(self.lats) - pad, max(self.lats) + pad,
                min(self.lons) - pad, max(self.lons) + pad)

    def upto(self, when: Optional[dt.datetime]) -> np.ndarray:
        """Indices of obs at or before `when` (all of them if `when` is None)."""
        idx = np.arange(len(self.lats))
        if when is None:
            return idx
        stamps = np.array([t.timestamp() if t else np.nan for t in self.times])
        return idx[np.isfinite(stamps) & (stamps <= when.timestamp())]


def overlay_from_batches(batches, segment: int = 0) -> Overlay:
    """NHC HDOB batches (recon10s_nhchdob.Batch) as one Overlay, one segment per mission."""
    ov = Overlay()
    seg_of = {}
    for b in sorted(batches, key=lambda b: b.start):
        sid = seg_of.setdefault(b.key, segment + len(seg_of))
        for o in b.obs:
            ov.lats.append(o.lat); ov.lons.append(o.lon)
            ov.winds.append(o.wspd); ov.dirs.append(o.wdir)
            ov.mslps.append(o.surface_press); ov.times.append(o.time); ov.segments.append(sid)
    return ov


def merge_overlays(overlays) -> Optional[Overlay]:
    """
    Several sources as one Overlay, time-ordered, each keeping its own
    segment ids so tracks from different aircraft are never joined.
    """
    parts = [o for o in overlays if o is not None and len(o)]
    if not parts:
        return None
    rows = []
    next_seg = 0
    for o in parts:
        segs = o.segments if len(o.segments) == len(o) else [0] * len(o)
        remap = {}
        for i in range(len(o)):
            sid = remap.setdefault(segs[i], next_seg + len(remap))
            rows.append((o.times[i], o.lats[i], o.lons[i], o.winds[i], o.dirs[i], o.mslps[i], sid))
        next_seg += len(remap)
    far = dt.datetime.max.replace(tzinfo=dt.timezone.utc)
    rows.sort(key=lambda r: r[0] or far)
    out = Overlay()
    for t, la, lo, w, d, p, sid in rows:
        out.times.append(t); out.lats.append(la); out.lons.append(lo)
        out.winds.append(w); out.dirs.append(d); out.mslps.append(p); out.segments.append(sid)
    return out


_MISSION_DATE = re.compile(r"\bHDOB\s+\d+\s+(\d{8})\b")


def overlay_from_hdob(path: str, fallback_date: Optional[dt.date] = None) -> Optional[Overlay]:
    """
    Read an HDOB file into an Overlay with full datetimes.

    Each HDOB message opens with a mission line ending "HDOB nn YYYYMMDD".
    That date cannot be taken at face value: the converter (like many real
    HDOB streams) repeats the storm date on every message even after 00Z. So
    the header only seeds the date, and each ob is placed on whichever day
    keeps the sequence moving forward -- a jump back of more than six hours
    means midnight was crossed. Values are decoded with the converter's own
    token parsers so they match the text exactly.
    """
    if recon10s_plot is None:
        return None
    P = recon10s_plot
    ov = Overlay()
    date = fallback_date or dt.datetime.now(dt.timezone.utc).date()
    last_when: Optional[dt.datetime] = None
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            m = _MISSION_DATE.search(line)
            if m:
                try:
                    date = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
                except ValueError:
                    pass
                continue
            if line.startswith(("URNT", "KNHC", "$$")):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            latlon = P._tok_to_latlon(parts[1], parts[2])
            hms = P._parse_time_token(parts[0])
            if latlon is None or hms is None:
                continue
            secs = hms[0] * 3600 + hms[1] * 60 + hms[2]
            when = dt.datetime.combine(date, dt.time(), tzinfo=dt.timezone.utc) + dt.timedelta(seconds=secs)
            if last_when is not None:
                while when < last_when - dt.timedelta(hours=6):
                    when += dt.timedelta(days=1)       # crossed midnight
            last_when = when

            pppp_idx, wind = P._find_mslp_and_wind(parts)
            mslp = None
            if pppp_idx is not None:
                try:
                    mslp = int(parts[pppp_idx]) / 10.0
                except ValueError:
                    mslp = None
            d, k = (wind if wind is not None else (None, None))
            ov.lats.append(latlon[0]); ov.lons.append(latlon[1])
            ov.dirs.append(d); ov.winds.append(k); ov.mslps.append(mslp); ov.times.append(when)
    return ov if len(ov) else None


def _wind_cmap():
    """The converter's wind buckets as a colormap + norm, so each barb can be
    coloured by its own speed in a single draw call."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    buckets = recon10s_plot._COLOR_BUCKETS if recon10s_plot else [(0, 1e9, (255, 255, 255))]
    edges = [b[0] for b in buckets] + [400.0]
    colours = [tuple(c / 255.0 for c in b[2]) for b in buckets]
    return ListedColormap(colours), BoundaryNorm(edges, len(colours))


@dataclass
class SceneArtists:
    """Handles on what changes between frames, so scrubbing can skip a full redraw."""
    ax: object
    crs: object
    image: object = None
    barbs: list = field(default_factory=list)
    track: list = field(default_factory=list)
    stamp: object = None
    title: object = None
    cbar_ax: object = None
    drop_artists: list = field(default_factory=list)
    drop_points: list = field(default_factory=list)   # (drop, lon, lat) as drawn, for clicks


def _declutter(ax, x, y, min_px: float) -> List[int]:
    """
    Indices to keep so no two barbs sit closer than min_px on screen. Done in
    display space, so it adapts to zoom: zoom in and more barbs come back.
    Walks newest-first, so the latest ob is always kept.
    """
    pc = ccrs.PlateCarree()
    pts = ax.projection.transform_points(pc, np.asarray(x), np.asarray(y))[:, :2]
    pix = ax.transData.transform(pts)
    kept: List[int] = []
    for i in range(len(pix) - 1, -1, -1):
        if not np.all(np.isfinite(pix[i])):
            continue
        if all(np.hypot(*(pix[i] - pix[j])) >= min_px for j in kept[-60:]):
            kept.append(i)
    return sorted(kept)


def _draw_obs(art: SceneArtists, overlay: Optional[Overlay], until, show_barbs, show_track,
              spacing_px: float, dark: bool):
    for a in art.barbs + art.track:
        try:
            a.remove()
        except Exception:
            pass
    art.barbs, art.track = [], []
    if overlay is None or not len(overlay):
        return None

    idx = overlay.upto(until)
    if idx.size == 0:
        return None
    pc = ccrs.PlateCarree()
    lons = np.asarray(overlay.lons, dtype=float)
    lats = np.asarray(overlay.lats, dtype=float)

    if show_track:
        # One line per flight, broken wherever obs are missing for more than
        # a few minutes (a skipped batch), so gaps are not drawn as straight
        # lines across the storm.
        segs = np.asarray(overlay.segments if len(overlay.segments) == len(overlay)
                          else [0] * len(overlay))
        stamps = np.array([t.timestamp() if t else np.nan for t in overlay.times])
        for sid in np.unique(segs[idx]):
            sel = idx[segs[idx] == sid]
            ts = stamps[sel]
            cuts = [0] + [k for k in range(1, len(sel))
                          if not np.isfinite(ts[k]) or ts[k] - ts[k - 1] > 180] + [len(sel)]
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b - a < 2:
                    continue
                line, = art.ax.plot(lons[sel[a:b]], lats[sel[a:b]], transform=pc,
                                    color="#f2f5f8" if dark else "#333333",
                                    linewidth=0.8, alpha=0.55, zorder=4)
                art.track.append(line)

    if show_barbs:
        keep = [int(i) for i in idx
                if overlay.dirs[i] is not None and overlay.winds[i] is not None]
        if keep and spacing_px > 0:
            sub = _declutter(art.ax, lons[keep], lats[keep], spacing_px)
            keep = [keep[j] for j in sub]
        if keep:
            spd = np.array([overlay.winds[i] for i in keep], dtype=float)
            rad = np.deg2rad([overlay.dirs[i] for i in keep])
            u, v = -spd * np.sin(rad), -spd * np.cos(rad)
            x, y = lons[keep], lats[keep]
            cmap, norm = _wind_cmap()
            # Dark under-stroke first so pale barbs (calm, 96-113 kt) read
            # over white cloud tops, then the coloured barb on top.
            art.barbs.append(art.ax.barbs(x, y, u, v, transform=pc, length=6.2,
                                          color="#0b0d11", linewidth=2.2, zorder=6))
            art.barbs.append(art.ax.barbs(x, y, u, v, spd, transform=pc, length=6.2,
                                          cmap=cmap, norm=norm, linewidth=1.0, zorder=7))
    return overlay.times[idx[-1]]


def _draw_drops(art: SceneArtists, drops, until, dark: bool):
    """
    Numbered dots at each dropsonde's release point, with a short line to
    where it splashed. Numbers run 1..N by release time across the whole
    loaded set, so a drop keeps its number as frames advance.
    """
    import matplotlib.patheffects as pe
    for a in art.drop_artists:
        try:
            a.remove()
        except Exception:
            pass
    art.drop_artists, art.drop_points = [], []
    if not drops:
        return
    pc = ccrs.PlateCarree()
    halo = [pe.withStroke(linewidth=2.6, foreground="#0b0d11")]
    for d in drops:
        if d.release_lat is None:
            continue
        if until is not None and d.release_time is not None and d.release_time > until:
            continue
        if d.splash_lat is not None:
            ln, = art.ax.plot([d.release_lon, d.splash_lon], [d.release_lat, d.splash_lat],
                              transform=pc, color="#ffffff", lw=0.9, alpha=0.8, zorder=10)
            art.drop_artists.append(ln)
        dot, = art.ax.plot(d.release_lon, d.release_lat, transform=pc, marker="o", ms=9,
                           mfc="#ffffff", mec="#0b0d11", mew=1.4, zorder=11, picker=True)
        label = art.ax.text(d.release_lon, d.release_lat, f" {d.number}", transform=pc,
                            fontsize=8.5, fontweight="bold", color="#ffffff", ha="left",
                            va="bottom", zorder=12, path_effects=halo, clip_on=True)
        art.drop_artists += [dot, label]
        art.drop_points.append((d, d.release_lon, d.release_lat))


def drops_near(art: SceneArtists, x_px: float, y_px: float, radius_px: float = 12.0) -> list:
    """
    Every dropsonde drawn within radius_px of a click, nearest first.
    Repeat passes through the same eyewall often put drops a few pixels
    apart, so a click can genuinely mean more than one.
    """
    if art is None or not art.drop_points:
        return []
    pc = ccrs.PlateCarree()
    lons = np.array([p[1] for p in art.drop_points])
    lats = np.array([p[2] for p in art.drop_points])
    pts = art.ax.projection.transform_points(pc, lons, lats)[:, :2]
    pix = art.ax.transData.transform(pts)
    dist = np.hypot(pix[:, 0] - x_px, pix[:, 1] - y_px)
    order = np.argsort(dist)
    return [art.drop_points[i][0] for i in order if np.isfinite(dist[i]) and dist[i] <= radius_px]


def drop_at(art: SceneArtists, x_px: float, y_px: float, radius_px: float = 12.0):
    """The single nearest dropsonde to a click, or None."""
    near = drops_near(art, x_px, y_px, radius_px)
    return near[0] if near else None


def _stamp_text(frame_time, latest_ob):
    frame = frame_time.strftime("%d %b %H:%M:%SZ") if frame_time else "\u2014"
    if latest_ob is None:
        ob = "none yet"
    else:
        ob = latest_ob.strftime("%H:%M:%SZ")
        if frame_time is not None:
            lag = (latest_ob - frame_time).total_seconds() / 60.0
            if abs(lag) >= 1:
                ob += f"  ({lag:+.0f} min)"
    return f"Frame      {frame}\nLatest ob  {ob}"


# Space kept around the map, in screen pixels rather than figure fractions, so
# the three-line title and the axis labels fit at any window size without the
# fixed 10-12% matplotlib default eating a small canvas.
MARGIN_PX = {"left": 54, "bottom": 30, "top": 64, "right": 14, "colorbar": 88}


def fit_layout(fig, art: Optional["SceneArtists"] = None) -> None:
    """Re-apply pixel margins; call again whenever the canvas is resized."""
    w, h = fig.get_size_inches() * fig.dpi
    if w < 50 or h < 50:
        return
    right = MARGIN_PX["right"] + (MARGIN_PX["colorbar"] if art is not None and art.cbar_ax is not None else 0)
    fig.subplots_adjust(left=min(0.3, MARGIN_PX["left"] / w),
                        right=max(0.7, 1 - right / w),
                        bottom=min(0.3, MARGIN_PX["bottom"] / h),
                        top=max(0.7, 1 - MARGIN_PX["top"] / h))


def draw_scene(fig, img=None, overlay: Optional[Overlay] = None, fix=None,
               show_satellite: bool = True, show_track: bool = True,
               show_barbs: bool = True, show_colorbar: bool = True, show_fix: bool = True,
               extent: Optional[Tuple[float, float, float, float]] = None,
               theme: str = "dark", title: Optional[str] = None,
               obs_until: Optional[dt.datetime] = None, spacing_px: float = 0.0,
               drops=None, show_drops: bool = True) -> SceneArtists:
    """
    Draw one frame into `fig`. Returns the artists that change between
    frames; pass them to update_frame to move through a loop cheaply.
    `extent` is (west, east, south, north) in degrees; `obs_until` hides obs
    after that time.
    """
    if ccrs is None:
        raise RuntimeError("cartopy is required for the map (pip install cartopy)")

    fig.clear()
    dark = theme != "light"
    fg = "#dfe5ec" if dark else "#1c2129"
    fig.patch.set_facecolor("#12151b" if dark else "#ffffff")

    if img is not None and show_satellite:
        crs = ccrs.Geostationary(central_longitude=img.lon0, satellite_height=img.sat_height,
                                 sweep_axis=img.sweep)
    else:
        centre = 0.0
        if extent:
            centre = (extent[0] + extent[1]) / 2
        elif overlay and len(overlay):
            centre = float(np.mean(overlay.lons))
        crs = ccrs.PlateCarree(central_longitude=centre)

    ax = fig.add_subplot(1, 1, 1, projection=crs)
    ax.set_facecolor("#0b0d11" if dark else "#e9edf2")
    art = SceneArtists(ax=ax, crs=crs)

    mappable = None
    if img is not None and show_satellite:
        art.image = ax.imshow(_frame_rgb(img), origin="upper", extent=img.extent_m,
                              transform=crs, interpolation="nearest", zorder=1)
        if show_colorbar and img.kind == "brightness":
            import matplotlib.cm as cm
            cmap, norm = irtable.mpl_colormap()
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#ffffff" if dark else "#333333", linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor="#cccccc" if dark else "#666666", linewidth=0.4, zorder=3)

    if fix is not None and show_fix:
        ax.plot(fix.lon, fix.lat, transform=ccrs.PlateCarree(), marker="+",
                markersize=15, markeredgewidth=2.2, color="#ffffff", zorder=8)
        ax.plot(fix.lon, fix.lat, transform=ccrs.PlateCarree(), marker="+",
                markersize=15, markeredgewidth=1.0, color="#ff5252", zorder=9)

    if extent is None:
        if overlay is not None and overlay.bbox:
            south, north, west, east = overlay.bbox
            extent = (west, east, south, north)
        elif fix is not None:
            extent = (fix.lon - 4, fix.lon + 4, fix.lat - 4, fix.lat + 4)
            # Don't frame empty space past the edge of a mesoscale sector.
            cov = getattr(img, "extent_lonlat", None) if img is not None else None
            if cov is not None and cov[2] < cov[3]:          # skip dateline-wrapping boxes
                w, e = max(extent[0], cov[2]), min(extent[1], cov[3])
                so, no = max(extent[2], cov[0]), min(extent[3], cov[1])
                if e - w > 1.0 and no - so > 1.0:
                    extent = (w, e, so, no)
    if extent:
        try:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        except Exception:
            pass

    grid = ax.gridlines(draw_labels=True, linewidth=0.4,
                        color="#ffffff" if dark else "#888888", alpha=0.3)
    grid.top_labels = grid.right_labels = False
    for attr in ("xlabel_style", "ylabel_style"):
        getattr(grid, attr).update({"size": 8, "color": fg})

    if mappable is not None:
        # Attached to the map's own box, so it sits against the map's real
        # edge wherever the aspect ratio puts it. fig.colorbar(ax=...) would
        # instead pin the map to the right of the figure.
        art.cbar_ax = ax.inset_axes([1.02, 0.0, 0.028, 1.0])
        cbar = fig.colorbar(mappable, cax=art.cbar_ax, ticks=list(irtable.TICK_TEMPS))
        cbar.set_label("Brightness temperature (\u00b0C)", color=fg, fontsize=8)
        cbar.ax.tick_params(colors=fg, labelsize=7)
        cbar.outline.set_edgecolor(fg)

    fit_layout(fig, art)   # before the barbs: decluttering measures screen pixels
    latest = _draw_obs(art, overlay, obs_until, show_barbs, show_track, spacing_px, dark)
    if show_drops:
        _draw_drops(art, drops, obs_until, dark)
    frame_time = img.start if img is not None else None
    art.stamp = ax.text(0.012, 0.988, _stamp_text(frame_time, latest), transform=ax.transAxes,
                        ha="left", va="top", fontsize=8.5, family="monospace", color="#f2f5f8",
                        zorder=12, bbox=dict(boxstyle="round,pad=0.4", fc="#0b0d11cc",
                                             ec="#5c6675", lw=0.8))
    if title:
        art.title = ax.set_title(title, color=fg, fontsize=9)
    return art


def _frame_rgb(img):
    if img.kind == "brightness":
        return irtable.temperature_to_rgb(img.data, missing=(0, 0, 0))
    vis = np.clip(img.data, 0.0, 1.0)
    grey = np.sqrt(np.where(np.isfinite(vis), vis, 0.0))
    return np.repeat((grey * 255).astype(np.uint8)[..., None], 3, axis=2)


def update_frame(art: SceneArtists, img, overlay: Optional[Overlay], obs_until,
                 show_barbs: bool, show_track: bool, spacing_px: float = 0.0, theme: str = "dark",
                 title: Optional[str] = None, drops=None, show_drops: bool = True) -> bool:
    """
    Swap in another frame of the same loop without rebuilding the map.
    Returns False when a full redraw is needed instead (no image artist yet,
    or the frame is from a different projection).
    """
    if art is None or art.image is None or img is None:
        return False
    crs = art.crs
    if (getattr(crs, "proj4_params", {}).get("lon_0") != img.lon0):
        return False
    art.image.set_data(_frame_rgb(img))
    art.image.set_extent(img.extent_m)
    latest = _draw_obs(art, overlay, obs_until, show_barbs, show_track, spacing_px, theme != "light")
    _draw_drops(art, drops if show_drops else None, obs_until, theme != "light")
    art.stamp.set_text(_stamp_text(img.start, latest))
    if title and art.title is not None:
        art.title.set_text(title)
    return True


def scene_title(img=None, fix=None, why: str = "") -> str:
    bits = []
    if img is not None:
        band = img.scene.band
        from recon10s_goes import BANDS
        name = BANDS.get(band, {}).get("name", "")
        sector = {"F": "Full disk", "M1": "Meso 1", "M2": "Meso 2"}.get(img.scene.sector,
                                                                       img.scene.sector)
        bits.append(f"GOES-{img.scene.sat} band {band} ({name}) \u00b7 {sector} "
                    f"\u00b7 {img.start:%d %b %H:%M:%S}Z")
    if fix is not None:
        bits.append(fix.label)
    if why:
        bits.append(why)
    return "\n".join(bits)
