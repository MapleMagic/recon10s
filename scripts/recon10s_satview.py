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
    heights: List[Optional[float]] = field(default_factory=list)   # geopotential, m
    fl_press: List[Optional[float]] = field(default_factory=list)  # flight level, mb
    peaks: List[Optional[int]] = field(default_factory=list)       # peak 10-s FL wind, kt
    wind_flagged: List[bool] = field(default_factory=list)         # HDOB QC: FL wind suspect
    labels: List[str] = field(default_factory=list)                # aircraft/mission per ob

    FIELDS = ("times", "lats", "lons", "winds", "dirs", "mslps", "segments",
              "heights", "fl_press", "peaks", "wind_flagged", "labels")

    def column(self, name, default=None):
        col = getattr(self, name)
        return col if len(col) == len(self.lats) else [default] * len(self.lats)

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
            ov.heights.append(o.height); ov.fl_press.append(o.fl_press); ov.peaks.append(o.peak)
            ov.wind_flagged.append(o.wind_suspect); ov.labels.append(f"{b.aircraft} {b.mission}")
    return ov


def merge_overlays(overlays) -> Optional[Overlay]:
    """
    Several sources as one Overlay, time-ordered, each keeping its own
    segment ids so tracks from different aircraft are never joined.
    """
    parts = [o for o in overlays if o is not None and len(o)]
    if not parts:
        return None
    defaults = {"segments": 0, "wind_flagged": False, "labels": ""}
    rows = []
    next_seg = 0
    for o in parts:
        cols = {f: o.column(f, defaults.get(f)) for f in Overlay.FIELDS}
        remap = {}
        for i in range(len(o)):
            row = {f: cols[f][i] for f in Overlay.FIELDS}
            row["segments"] = remap.setdefault(row["segments"], next_seg + len(remap))
            rows.append(row)
        next_seg += len(remap)
    far = dt.datetime.max.replace(tzinfo=dt.timezone.utc)
    rows.sort(key=lambda r: r["times"] or far)
    out = Overlay()
    for r in rows:
        for f in Overlay.FIELDS:
            getattr(out, f).append(r[f])
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
            if abs(latlon[0]) < 0.01 and abs(latlon[1]) < 0.01:
                continue                      # no position fix yet (0N 0E)
            ov.lats.append(latlon[0]); ov.lons.append(latlon[1])
            ov.dirs.append(d); ov.winds.append(k); ov.mslps.append(mslp); ov.times.append(when)
            fl = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
            ov.fl_press.append((fl / 10.0 + (1000.0 if fl < 1000 else 0.0)) if fl is not None else None)
            ov.heights.append(float(parts[4]) if len(parts) > 4 and parts[4].isdigit() else None)
            ov.peaks.append(int(parts[9]) if len(parts) > 9 and parts[9].isdigit() else None)
            flags = parts[12] if len(parts) > 12 else ""
            ov.wind_flagged.append(len(flags) == 2 and flags[1] in "2469")
            ov.labels.append("converted IWG1")
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
class Extras:
    """Everything on the map besides imagery, obs and drops."""
    fixes: list = field(default_factory=list)          # recon10s_analysis.CentreFix
    show_fixes: bool = True
    motion: Optional[tuple] = None                     # (heading, kt)
    peaks: list = field(default_factory=list)          # (lat, lon, kt, time)
    show_peaks: bool = False
    flagged: str = "fade"                              # "show", "fade" or "hide"
    forecast: object = None                            # recon10s_forecast.OfficialForecast
    show_cone: bool = True
    show_official: bool = True
    models: list = field(default_factory=list)         # recon10s_forecast.ModelTrack
    show_models: bool = False
    glm: Optional[tuple] = None                        # (lats, lons, minutes)
    show_glm: bool = True
    tdr: object = None                                 # recon10s_tdrgridpanel.TdrLayer
    tdr_alpha: float = 0.8
    frame_time: object = None                          # to say how old a TDR analysis is


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
    extra_artists: list = field(default_factory=list)
    glm_artists: list = field(default_factory=list)
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
              spacing_px: float, dark: bool, flagged: str = "fade"):
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
        flags = overlay.column("wind_flagged", False)
        keep = [int(i) for i in idx
                if overlay.dirs[i] is not None and overlay.winds[i] is not None
                and not (flagged == "hide" and flags[i])]
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
            good = np.array([not flags[i] for i in keep]) if flagged == "fade" else np.ones(len(keep), bool)
            if good.any():
                art.barbs.append(art.ax.barbs(x[good], y[good], u[good], v[good], spd[good], transform=pc,
                                              length=6.2, cmap=cmap, norm=norm, linewidth=1.0, zorder=7))
            if (~good).any():
                # HDOB quality flags mark these flight-level winds as questionable.
                art.barbs.append(art.ax.barbs(x[~good], y[~good], u[~good], v[~good], transform=pc,
                                              length=6.2, color="#8b95a5", alpha=0.45,
                                              linewidth=0.9, zorder=7))
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


def _clear(artists):
    for a in artists:
        try:
            a.remove()
        except Exception:
            pass


def _unwrap(lons):
    """Longitudes made continuous across the dateline (179 -> 181, not -179)."""
    return list(np.degrees(np.unwrap(np.radians(np.asarray(lons, float))))) if len(lons) else []


def _draw_extras(art: SceneArtists, ex: Optional[Extras], until, dark: bool):
    """Centre fixes, motion, peak winds and forecast -- redrawn on every frame."""
    import matplotlib.patheffects as pe
    _clear(art.extra_artists)
    art.extra_artists = []
    if ex is None:
        return
    pc = ccrs.PlateCarree()
    halo = [pe.withStroke(linewidth=2.4, foreground="#0b0d11")]
    A = art.extra_artists

    if ex.tdr is not None:
        # Airborne TDR plan view, georeferenced by its own latitude/longitude grid.
        t = ex.tdr
        A.append(art.ax.pcolormesh(t.lon, t.lat, np.ma.masked_invalid(t.values), cmap=t.cmap,
                                   norm=t.norm, alpha=ex.tdr_alpha, shading="nearest",
                                   transform=pc, zorder=3))
        label = t.label
        if t.time is not None and ex.frame_time is not None:
            gap = (ex.frame_time - t.time).total_seconds() / 3600
            if abs(gap) >= 1.5:
                span = f"{abs(gap):.0f} h" if abs(gap) < 48 else f"{abs(gap) / 24:.1f} days"
                label += f"  \u2014 {span} {'before' if gap > 0 else 'after'} this frame"
        A.append(art.ax.text(0.01, 0.01, label, transform=art.ax.transAxes, fontsize=7.5,
                             color="#f2f5f8", va="bottom", zorder=15,
                             bbox=dict(fc="#0b0d11cc", ec="#5c6675", lw=0.6)))

    fc = ex.forecast
    if fc is not None and ex.show_cone:
        for poly in fc.cone:
            if len(poly) > 2:
                xs, ys = zip(*poly)
                xs = _unwrap(xs)
                A += art.ax.fill(xs, ys, transform=pc, facecolor="#ffffff", alpha=0.10,
                                 edgecolor="#ffffff", linewidth=1.0, zorder=2)
    if ex.show_models:
        import recon10s_forecast as fcst
        for m in ex.models:
            if m.group == "Official":
                continue
            xs = _unwrap([p[2] for p in m.points])
            ys = [p[1] for p in m.points]
            A += art.ax.plot(xs, ys, transform=pc, color=fcst.GROUP_COLOURS.get(m.group, "#cccccc"),
                             linewidth=1.0, alpha=0.8, zorder=3)
            if xs:
                A.append(art.ax.text(xs[-1], ys[-1], m.model, transform=pc, fontsize=6.5, clip_on=True,
                                     color=fcst.GROUP_COLOURS.get(m.group, "#cccccc"), zorder=3,
                                     path_effects=halo))
    if fc is not None and ex.show_official and fc.track:
        xs, ys = zip(*fc.track)
        xs = _unwrap(xs)
        A += art.ax.plot(xs, ys, transform=pc, color="#ffffff", linewidth=2.0, zorder=4,
                         path_effects=[pe.withStroke(linewidth=3.6, foreground="#0b0d11")])
        if fc.points:
            A += art.ax.plot([p[0] for p in fc.points], [p[1] for p in fc.points], "o", ms=5,
                             mfc="#ffffff", mec="#0b0d11", transform=pc, zorder=4)

    if ex.show_fixes:
        for f in ex.fixes:
            if until is not None and f.time > until:
                continue
            vdm = f.source == "VDM"
            A += art.ax.plot(f.lon, f.lat, marker="D" if vdm else "X", ms=7 if vdm else 8,
                             mfc="#ffca28" if vdm else "#80deea", mec="#0b0d11", mew=1.0,
                             transform=pc, zorder=13)
            if vdm and f.min_slp:
                A.append(art.ax.text(f.lon, f.lat, f"  {f.min_slp:.0f}", transform=pc, fontsize=7.5,
                                     color="#ffca28", va="top", zorder=13, clip_on=True,
                                     path_effects=halo))
        shown = [f for f in ex.fixes if until is None or f.time <= until]
        if ex.motion and shown:
            import recon10s_analysis as an
            last = shown[-1]
            heading, kt = ex.motion
            lat2, lon2 = an.destination(last.lat, last.lon, heading, max(kt, 2) * 1.852 * 6)
            A.append(art.ax.annotate("", xy=(lon2, lat2), xytext=(last.lon, last.lat),
                                     xycoords=pc._as_mpl_transform(art.ax),
                                     arrowprops=dict(arrowstyle="-|>", color="#ffca28", lw=1.6),
                                     zorder=13))
            A.append(art.ax.text(lon2, lat2, f" {heading:03.0f}\u00b0/{kt:.0f} kt", transform=pc,
                                 fontsize=7.5, color="#ffca28", zorder=13, clip_on=True,
                                 path_effects=halo))
    if ex.show_peaks:
        for lat, lon, kt, t in ex.peaks:
            if until is not None and t is not None and t > until:
                continue
            A.append(art.ax.text(lon, lat, f"{kt}", transform=pc, fontsize=8, fontweight="bold",
                                 color="#ffffff", ha="center", va="bottom", zorder=14, clip_on=True,
                                 path_effects=[pe.withStroke(linewidth=2.8, foreground="#b0003a")]))


def draw_glm(art: SceneArtists, glm, show: bool):
    """Lightning flashes; kept separate because they arrive after the frame does."""
    _clear(art.glm_artists)
    art.glm_artists = []
    if not show or glm is None or art is None:
        return
    lats, lons = glm[0], glm[1]
    if len(lats):
        art.glm_artists = art.ax.plot(lons, lats, "+", ms=5, mew=1.1, color="#ffee58",
                                      transform=ccrs.PlateCarree(), zorder=9, alpha=0.9)


def subset_overlay(ov: Optional[Overlay], t0, t1) -> Optional[Overlay]:
    """The obs taken between t0 and t1."""
    if ov is None or not len(ov):
        return None
    keep = [i for i, t in enumerate(ov.times) if t is not None and t0 <= t <= t1]
    if not keep:
        return None
    out = Overlay()
    for f in Overlay.FIELDS:
        col = ov.column(f)
        setattr(out, f, [col[i] for i in keep])
    return out


def draw_recon_on_axes(ax, bundle, when, minutes: float = 90.0, dark: bool = True,
                       spacing_px: float = 16.0):
    """
    Recon barbs, dropsondes and centre fixes within +-`minutes` of `when` on
    any PlateCarree-based cartopy axes -- used for the microwave passes.
    """
    if bundle is None or when is None:
        return None
    t0, t1 = when - dt.timedelta(minutes=minutes), when + dt.timedelta(minutes=minutes)
    art = SceneArtists(ax=ax, crs=ax.projection)
    _draw_obs(art, subset_overlay(bundle.overlay, t0, t1), None, True, True, spacing_px, dark)
    near = [d for d in (bundle.drops or []) if d.release_time and t0 <= d.release_time <= t1]
    _draw_drops(art, near, None, dark)
    fixes = [f for f in (bundle.fixes or []) if t0 <= f.time <= t1]
    _draw_extras(art, Extras(fixes=fixes, flagged="fade"), None, dark)
    return art


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
               drops=None, show_drops: bool = True, extras: Optional[Extras] = None) -> SceneArtists:
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
        if show_colorbar and img.kind in ("brightness", "difference"):
            import matplotlib.cm as cm
            cmap, norm = irtable.mpl_colormap() if img.kind == "brightness" else difference_cmap()
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable._recon_kind = img.kind

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
    # Freeze the view. Anything drawn later -- forecast cones and model
    # tracks reaching thousands of km away -- would otherwise autoscale the
    # map out to fit them.
    ax.set_autoscale_on(False)

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
        if getattr(mappable, "_recon_kind", "") == "difference":
            cbar = fig.colorbar(mappable, cax=art.cbar_ax)
            cbar.set_label("Band 8 \u2212 band 13 (\u00b0C)", color=fg, fontsize=8)
        else:
            cbar = fig.colorbar(mappable, cax=art.cbar_ax, ticks=list(irtable.TICK_TEMPS))
            cbar.set_label("Brightness temperature (\u00b0C)", color=fg, fontsize=8)
        cbar.ax.tick_params(colors=fg, labelsize=7)
        cbar.outline.set_edgecolor(fg)

    fit_layout(fig, art)   # before the barbs: decluttering measures screen pixels
    _draw_extras(art, extras, obs_until, dark)
    latest = _draw_obs(art, overlay, obs_until, show_barbs, show_track, spacing_px, dark,
                       extras.flagged if extras else "fade")
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


def difference_cmap():
    """
    WV-IR: greys up to 0 C, then yellow to red above it. Deep convection sits
    just below zero across a whole CDO; the positive values are the
    overshooting tops, so only they are coloured.
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    z = 40.0 / 45.0
    stops = [(0.0, "#101010"), (0.6, "#8a8a8a"), (z, "#f5f5f5"), (z, "#ffee58"),
             (z + 0.05, "#ff9800"), (1.0, "#d50000")]
    return LinearSegmentedColormap.from_list("wv_ir", stops), Normalize(-40.0, 5.0)


def _frame_rgb(img):
    if img.kind == "rgb":
        return (np.clip(np.nan_to_num(img.data), 0, 1) * 255).astype(np.uint8)
    if img.kind == "difference":
        cmap, norm = difference_cmap()
        return (cmap(norm(np.nan_to_num(img.data, nan=-40.0)))[..., :3] * 255).astype(np.uint8)
    if img.kind == "brightness":
        return irtable.temperature_to_rgb(img.data, missing=(0, 0, 0))
    vis = np.clip(img.data, 0.0, 1.0)
    grey = np.sqrt(np.where(np.isfinite(vis), vis, 0.0))
    return np.repeat((grey * 255).astype(np.uint8)[..., None], 3, axis=2)


def update_frame(art: SceneArtists, img, overlay: Optional[Overlay], obs_until,
                 show_barbs: bool, show_track: bool, spacing_px: float = 0.0, theme: str = "dark",
                 title: Optional[str] = None, drops=None, show_drops: bool = True,
                 extras: Optional[Extras] = None) -> bool:
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
    _draw_extras(art, extras, obs_until, theme != "light")
    latest = _draw_obs(art, overlay, obs_until, show_barbs, show_track, spacing_px, theme != "light",
                       extras.flagged if extras else "fade")
    _draw_drops(art, drops if show_drops else None, obs_until, theme != "light")
    art.stamp.set_text(_stamp_text(img.start, latest))
    if title and art.title is not None:
        art.title.set_text(title)
    return True


def scene_title(img=None, fix=None, why: str = "") -> str:
    bits = []
    if img is not None:
        band = img.scene.band
        from recon10s_goes import BANDS, PRODUCTS, SECTOR_NAMES, sat_name
        name = BANDS.get(band, {}).get("name", "")
        key = getattr(img, "product", None)
        if key in PRODUCTS:
            name, band = PRODUCTS[key]["name"], "/".join(str(b) for b in PRODUCTS[key]["bands"])
        sector = SECTOR_NAMES.get(img.scene.sector, img.scene.sector)
        bits.append(f"{sat_name(img.scene.sat)} band {band} ({name}) \u00b7 {sector} "
                    f"\u00b7 {img.start:%d %b %H:%M:%S}Z")
    if fix is not None:
        bits.append(fix.label)
    if why:
        bits.append(why)
    return "\n".join(bits)
