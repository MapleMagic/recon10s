#!/usr/bin/env python3
"""
recon10s_satint — satellite intensity estimates for Experimental › Satellite
intensity: a storm's maximum sustained wind and central pressure from
geostationary infrared and the most recent passive-microwave pass, by a
convolutional network trained on TC PRIMED (models/satint_v1.npz).

Like CIMSS's D-PRINT (IR only) and D-MINT (IR and microwave) it looks at the
storm, not at a Dvorak tree: the 10.3-um window IR now and at 3, 6, 9 and 12
hours before, on a 4-km grid centred on the storm, and 37 and 89 GHz on an
8-km grid. Unlike D-MINT it does not wait for a pass: every estimate takes
the most recent pass of the last 12 hours, how old it is, and the IR from the
time of the pass, so the network can weigh the microwave structure against
how the IR has changed since. With no pass (or no PPS sign-in) the same
network works from IR alone, as it was trained to.

Outputs are quantiles (10, 25, 50, 75, 90 %) of the best-track maximum wind
and central pressure; the estimate is the median.

Inputs at run time
  * IR: band 13 of GOES-East or -West ABI (10.3 um) or Himawari-9 AHI
    (10.4 um), whichever sees the storm at the smaller angle, cut around the
    storm and resampled to the training grid.
  * Microwave: the newest GMI, AMSR3 or WSF-M MWI pass from PPS's near-real-
    time server (the Microwave tab's sign-in), gridded like the training data.
  * Centre: recon fixes when there are any near the time, else the working
    best track (NHC's or JTWC's) and advisory, interpolated (and carried on
    along the motion for a few hours past the last position).

Southern Hemisphere storms turn the other way. The network learnt northern
storms only, so their grids are mirrored north-south (poleward stays up)
and the latitude given as its size: a mirrored southern storm is a northern
one in every respect the network can see.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "satint_v1.npz")
UTC = dt.timezone.utc
EARTH_KM = 6371.0088
IR_N, IR_DX = 128, 4.0
MW_N, MW_DX = 64, 8.0
LAGS_H = (3.0, 6.0, 9.0, 12.0)
MW_MAX_AGE_H = 12.0
MW_MIN_COVER = 0.5
MAX_ZENITH = 70.0                  # no estimate from IR seen more obliquely than this (degrees)
OBLIQUE_ZENITH = 62.0              # beyond this an estimate says the view is oblique (degrades by 66-70)
JTWC_BASINS = ("wp", "io", "sh")
# Storms of JTWC's basins: the network learnt NHC's best tracks, whose pressures come mostly from aircraft;
# JTWC's come from its wind-pressure relationship, in basins of lower environmental pressure, and its
# intensities from satellite analyses. So their estimates are mapped onto JTWC's best track: the MSLP median
# moved by c0 + c1 * (median - 1000 mb) + c2 * (|lat| - 20) / 10, and the ranges of both widened about their
# medians by `scale` (the Vmax median is left as it is: a shift fitted the same way helped weak systems but
# hurt tropical storms and typhoons). Fitted on TC PRIMED's storms of 2020-2023 in those basins
# (training/satint/fit_jtwc_adjust.py --no-vmax-shift) and checked on 2024-2025, which it never saw (README).
JTWC_ADJUST = {
    "pass": {"vmax": {"terms": [], "coef": None, "scale": 1.504},
             "mslp": {"terms": ["level", "lat"], "coef": [-4.8, -0.036, -1.824], "scale": 1.663}},
    "ir": {"vmax": {"terms": [], "coef": None, "scale": 1.436},
           "mslp": {"terms": ["level", "lat"], "coef": [-5.396, -0.01, -1.921], "scale": 1.715}},
}
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
ROTATIONS = 4                      # each estimate averages the network over four 90-degree turns
# nominal 89-GHz footprint (km), an input of the network; MWI and AMSR3 are new sensors not in
# the training data, placed by their resolution beside the ones that are
SENSOR_RES_KM = {"SSMI": 15.0, "SSMIS": 14.0, "TMI": 7.0, "AMSRE": 5.0, "AMSR2": 4.0, "AMSR3": 4.0, "GMI": 6.0,
                 "WSFM": 10.0, "MWI": 10.0}


# ------------------------------------------------------------------ geometry

def grid(n: int, dx: float) -> np.ndarray:
    return (np.arange(n) - (n - 1) / 2.0) * dx


def destinations(lat0: float, lon0: float, x, y):
    """lat/lon of points x km east, y km north of (lat0, lon0) (azimuthal equidistant)."""
    d = np.hypot(x, y) / EARTH_KM
    b = np.arctan2(x, y)
    la1, lo1 = math.radians(lat0), math.radians(lon0)
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(b))
    lo2 = lo1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(la1), np.cos(d) - np.sin(la1) * np.sin(la2))
    return np.degrees(la2), (np.degrees(lo2) + 540.0) % 360.0 - 180.0


def local_xy(lat0: float, lon0: float, lat, lon):
    la0, lo0 = math.radians(lat0), math.radians(lon0)
    la, lo = np.radians(lat), np.radians(lon)
    dlo = lo - lo0
    c = np.sin(la0) * np.sin(la) + np.cos(la0) * np.cos(la) * np.cos(dlo)
    rho = np.arccos(np.clip(c, -1, 1))
    k = np.where(rho > 1e-12, rho / np.maximum(np.sin(rho), 1e-12), 1.0)
    x = EARTH_KM * k * np.cos(la) * np.sin(dlo)
    y = EARTH_KM * k * (np.cos(la0) * np.sin(la) - np.sin(la0) * np.cos(la) * np.cos(dlo))
    return x, y


# ------------------------------------------------------------------- the network

def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int) -> np.ndarray:
    """3x3 convolution, padding 1, as PyTorch computes it: x (B, C, H, W), w (O, C, 3, 3)."""
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))[:, :, ::stride, ::stride]
    out = np.tensordot(win, w, axes=([1, 4, 5], [1, 2, 3]))          # (B, H', W', O)
    return np.ascontiguousarray(out.transpose(0, 3, 1, 2)) + b[None, :, None, None]


def _softplus(x):
    return np.logaddexp(0.0, x)


class Net:
    """
    The trained network (batch norm folded into the convolutions), in numpy. A file may hold several
    networks trained alike ("members"; arrays prefixed m0_, m1_, ...): each estimate is then their
    average.
    """

    def __init__(self, path: Optional[str] = None):
        path = path or MODEL_PATH
        d = np.load(path, allow_pickle=False)
        self.path = path
        self.spec = json.loads(str(d["spec"]))
        self.info = json.loads(str(d["info"])) if "info" in d else {}
        self.arrays = {k: d[k].astype(np.float32) for k in d.files if k not in ("spec", "info")}
        self.quantiles = tuple(self.spec.get("quantiles", QUANTILES))
        self.norm = self.spec["norm"]
        # (array-name prefix, layer spec) of each member
        if "members" in self.spec:
            self.members = [(f"m{k}_", m) for k, m in enumerate(self.spec["members"])]
        else:
            self.members = [("", {"ir": self.spec["ir"], "mw": self.spec["mw"], "head": self.spec["head"]})]

    def _branch(self, pre: str, spec: dict, name: str, x: np.ndarray) -> np.ndarray:
        A = self.arrays
        for i, layer in enumerate(spec[name]):
            x = _conv3x3(x, A[f"{pre}{name}{i}_w"], A[f"{pre}{name}{i}_b"], int(layer["stride"]))
            np.maximum(x, 0.0, out=x)
        x = x.reshape(x.shape[0], -1)
        x = x @ A[f"{pre}{name}_fc_w"].T + A[f"{pre}{name}_fc_b"]
        return np.maximum(x, 0.0)

    def forward(self, ir: np.ndarray, mw: np.ndarray, sc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantiles of Vmax (kt) and MSLP (mb), each (B, 5): the members' average."""
        outs = [self._member(pre, spec, ir, mw, sc) for pre, spec in self.members]
        return np.mean([o[0] for o in outs], 0), np.mean([o[1] for o in outs], 0)

    def _member(self, pre: str, spec: dict, ir: np.ndarray, mw: np.ndarray, sc: np.ndarray):
        a = self._branch(pre, spec, "ir", ir.astype(np.float32))
        b = self._branch(pre, spec, "mw", mw.astype(np.float32)) * sc[:, :1]
        h = np.concatenate([a, b, sc.astype(np.float32)], 1)
        n_head = int(spec["head"])
        for i in range(n_head):
            h = h @ self.arrays[f"{pre}head{i}_w"].T + self.arrays[f"{pre}head{i}_b"]
            if i < n_head - 1:
                h = np.maximum(h, 0.0)
        n = len(self.quantiles)
        res = []
        for k in range(2):
            o = h[:, k * n:(k + 1) * n]
            mid = o[:, 2:3]
            lo = mid - np.cumsum(_softplus(o[:, :2][:, ::-1]), 1)[:, ::-1]
            hi = mid + np.cumsum(_softplus(o[:, 3:]), 1)
            res.append(np.concatenate([lo, mid, hi], 1))
        nv, npr = self.norm["vmax"], self.norm["mslp"]
        return res[0] * nv[1] + nv[0], res[1] * npr[1] + npr[0]

    def predict(self, ir: np.ndarray, mw: np.ndarray, sc: np.ndarray, rotations: int = 1
                ) -> Tuple[np.ndarray, np.ndarray]:
        """forward, averaged over 90-degree turns of the grids (exact on these grids; the network
        learnt every orientation, so the turns differ only by its noise)."""
        vs, ps = [], []
        for r in range(max(1, int(rotations))):
            a = np.ascontiguousarray(np.rot90(ir, r, axes=(2, 3))) if r else ir
            b = np.ascontiguousarray(np.rot90(mw, r, axes=(2, 3))) if r else mw
            v, p = self.forward(a, b, sc)
            vs.append(v)
            ps.append(p)
        return np.mean(vs, 0), np.mean(ps, 0)


_NET: Optional[Net] = None


def net(path: Optional[str] = None) -> Net:
    global _NET
    path = path or MODEL_PATH
    if _NET is None or _NET.path != path:
        _NET = Net(path)
    return _NET


# ------------------------------------------------------------- the inputs

def ir_from_lonlat_grid(tb: np.ndarray, lat_v: np.ndarray, lon_v: np.ndarray, centre: Tuple[float, float]) -> np.ndarray:
    """IR on a regular lat/lon grid (rows lat_v, columns lon_v) to the 4-km storm grid (K)."""
    lat0, lon0 = centre
    g = grid(IR_N, IR_DX)
    X, Y = np.meshgrid(g, g)
    la, lo = destinations(lat0, lon0, X, Y)
    lon_r = ((np.asarray(lon_v, float) - lon0 + 180.0) % 360.0) - 180.0
    lo_r = ((lo - lon0 + 180.0) % 360.0) - 180.0
    lat_v = np.asarray(lat_v, float)
    t = np.asarray(tb, float)
    if lat_v[0] > lat_v[-1]:
        lat_v, t = lat_v[::-1], t[::-1]
    if lon_r[0] > lon_r[-1]:
        lon_r, t = lon_r[::-1], t[:, ::-1]
    rows = np.interp(la, lat_v, np.arange(lat_v.size), left=np.nan, right=np.nan)
    cols = np.interp(lo_r, lon_r, np.arange(lon_r.size), left=np.nan, right=np.nan)
    return _sample(t, rows, cols)


def _sample(t: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates
    ok = np.isfinite(rows) & np.isfinite(cols)
    valid = np.isfinite(t).astype(np.float64)
    filled = np.where(np.isfinite(t), t, 0.0)
    rr, cc = np.where(ok, rows, 0.0), np.where(ok, cols, 0.0)
    num = map_coordinates(filled, [rr, cc], order=1, mode="nearest")
    den = map_coordinates(valid, [rr, cc], order=1, mode="nearest")
    return np.where(ok & (den > 0.5), num / np.maximum(den, 1e-6), np.nan)


def ir_from_goes(img, centre: Tuple[float, float]) -> np.ndarray:
    """A loaded GOES band-13 cut (recon10s_goes.GoesImage, degC) to the 4-km storm grid (K)."""
    from pyproj import CRS, Transformer
    from scipy.ndimage import gaussian_filter
    data = np.asarray(img.data, float) + 273.15
    # ABI's 2-km pixels, smoothed toward the 4-km remapped archive imagery the network learnt from:
    # 0.7 pixel matched the archive best on 2025 storms (1.35 K rms, no bias in the warmest eye pixel)
    valid = np.isfinite(data)
    if valid.mean() > 0.5:
        filled = np.where(valid, data, np.nanmean(data))
        data = np.where(valid, gaussian_filter(filled, 0.7), np.nan)
    crs = CRS.from_proj4(f"+proj=geos +h={img.sat_height} +lon_0={img.lon0} +sweep={img.sweep} "
                         "+a=6378137 +b=6356752.31414 +units=m +no_defs")
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lat0, lon0 = centre
    g = grid(IR_N, IR_DX)
    X, Y = np.meshgrid(g, g)
    la, lo = destinations(lat0, lon0, X, Y)
    xm, ym = tf.transform(lo, la)
    x0, x1, yb, yt = img.extent_m
    nr, nc = data.shape
    dxm = (x1 - x0) / nc
    dym = (yb - yt) / nr                               # rows run north to south
    cols = (np.asarray(xm) - (x0 + dxm / 2)) / dxm
    rows = (np.asarray(ym) - (yt + dym / 2)) / dym
    out = _sample(data, rows, cols)
    out[(out < 150) | (out > 340)] = np.nan
    return out


def mw_grid(bands: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
            centre: Tuple[float, float]) -> Tuple[np.ndarray, float]:
    """
    37 and 89 GHz near the storm -- {'37': (lat, lon, V, H), '89': (...)} -- to the 8-km storm grid:
    (4, 64, 64) Tb in K (37V, 37H, 89V, 89H; NaN without data) and the coverage of the inner 200 km.
    Each cell is a Gaussian-weighted mean of the footprints within 7 km (89) or 11 km (37).
    """
    from scipy.spatial import cKDTree
    lat0, lon0 = centre
    g = grid(MW_N, MW_DX)
    X, Y = np.meshgrid(g, g)
    targets = np.c_[X.ravel(), Y.ravel()]
    out = np.full((4, MW_N, MW_N), np.nan)
    cover = 0.0
    for bi, band in enumerate(("37", "89")):
        if band not in bands:
            continue
        lat, lon, v, h = (np.asarray(a, float).ravel() for a in bands[band])
        x, y = local_xy(lat0, lon0, lat, lon)
        near = (np.abs(x) < 300) & (np.abs(y) < 300) & np.isfinite(x) & np.isfinite(y)
        if near.sum() < 10:
            continue
        tree = cKDTree(np.c_[x[near], y[near]])
        radius = 11.0 if band == "37" else 7.0
        k = min(24, tree.n)
        d, ii = tree.query(targets, k=k, distance_upper_bound=radius)
        d, ii = d.reshape(len(targets), -1), ii.reshape(len(targets), -1)
        hit = ii < tree.n
        for ci, arr in enumerate((v, h)):
            t = arr[near]
            t = np.where((t > 50) & (t < 340), t, np.nan)
            tv = np.where(hit, t[np.minimum(ii, tree.n - 1)], np.nan)
            w = np.where(hit & np.isfinite(tv), np.exp(-d ** 2 / (2 * (radius / 2) ** 2)), 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                vals = np.where(w.sum(1) > 0, np.nansum(w * np.nan_to_num(tv), 1) / w.sum(1), np.nan)
            out[2 * bi + ci] = vals.reshape(MW_N, MW_N)
        if band == "89":
            inner = np.hypot(X, Y) <= 200
            cover = float(np.isfinite(out[3][inner]).mean())
    return out, cover


def solar_hour(t: dt.datetime, lon: float) -> float:
    h = t.hour + t.minute / 60.0 + t.second / 3600.0 + (((lon + 180) % 360) - 180) / 15.0
    return h % 24.0


def network_inputs(frames: Sequence[Optional[np.ndarray]], mw: Optional[np.ndarray], mw_ir: Optional[np.ndarray],
                   age_h: float, mw_res_km: float, mw_cover: float, lat: float, speed_kt: Optional[float],
                   east_pacific: bool, when: dt.datetime, lon: float):
    """
    The network's three inputs for one estimate, normalised as in training. `frames` holds the IR
    grids (K) at t, t-3, t-6, t-9 and t-12 h (None where missing: the current frame stands in).
    A southern storm (lat < 0) is mirrored north-south first, into the northern storm it looks like.
    """
    # the training grids were stored as bytes: IR 163-315.4 K, microwave 100-315.9 K
    clip_ir = lambda a: None if a is None else np.clip(np.asarray(a, float), 163.0, 315.4)
    frames = [clip_ir(f) for f in frames]
    mw_ir = clip_ir(mw_ir)
    if mw is not None:
        mw = np.clip(np.asarray(mw, float), 100.0, 315.9)
    if lat < 0:
        mirror = lambda a: None if a is None else np.ascontiguousarray(np.asarray(a)[..., ::-1, :])
        frames = [mirror(f) for f in frames]
        mw_ir, mw = mirror(mw_ir), mirror(mw)
        lat = -lat
    cur = frames[0]
    ir = np.zeros((1, 5, IR_N, IR_N), np.float32)
    mwx = np.zeros((1, 6, MW_N, MW_N), np.float32)
    sc = np.zeros((1, 13), np.float32)
    avail = []
    for k, f in enumerate(frames):
        if k > 0:
            avail.append(1.0 if f is not None else 0.0)
        x = (np.asarray(f if f is not None else cur, float) - 240.0) / 40.0
        ir[0, k] = np.nan_to_num(x, nan=0.0)
    if mw is not None:
        valid = np.isfinite(mw).all(0)
        mwx[0, :4] = np.nan_to_num((mw - 230.0) / 40.0, nan=0.0) * valid
        mwx[0, 4] = valid
        src = mw_ir if mw_ir is not None else cur
        small = np.nanmean(np.asarray(src, float).reshape(MW_N, 2, MW_N, 2), axis=(1, 3))
        mwx[0, 5] = np.nan_to_num((small - 240.0) / 40.0, nan=0.0)
        sc[0, 0] = 1.0
        sc[0, 1] = age_h / 12.0
        sc[0, 2] = mw_res_km / 15.0
        sc[0, 3] = mw_cover
    sc[0, 4] = lat / 30.0
    sc[0, 5] = (speed_kt if speed_kt is not None and np.isfinite(speed_kt) else 10.0) / 20.0
    sc[0, 6] = 1.0 if east_pacific else 0.0
    a = 2 * math.pi * solar_hour(when, lon) / 24.0
    sc[0, 7], sc[0, 8] = math.sin(a), math.cos(a)
    sc[0, 9:13] = avail
    return ir, mwx, sc


# --------------------------------------------------------------- the centre

@dataclass
class TrackSource:
    """Where the storm was: recon fixes, NHC's working best track and the advisory."""
    track: list = field(default_factory=list)          # recon10s_nhc.TrackPoint
    advisory: object = None                             # recon10s_nhc.Fix
    fixes: list = field(default_factory=list)           # recon10s_analysis.CentreFix
    motion: Optional[Tuple[float, float]] = None        # heading toward, kt

    def points(self) -> List[Tuple[float, float, float]]:
        pts = {}
        for p in self.track or []:
            if p.time is not None and p.lat is not None and p.lon is not None:
                pts[p.time.timestamp()] = (float(p.lat), float(p.lon))
        a = self.advisory
        if a is not None and getattr(a, "time", None) is not None and getattr(a, "lat", None) is not None:
            pts[a.time.timestamp()] = (float(a.lat), float(a.lon))
        return sorted((t, la, lo) for t, (la, lo) in pts.items())

    def heading_speed(self) -> Optional[Tuple[float, float]]:
        if self.motion:
            return self.motion
        a = self.advisory
        if a is not None and getattr(a, "motion", None):
            return a.motion
        pts = self.points()
        if len(pts) >= 2:
            (t0, la0, lo0), (t1, la1, lo1) = pts[-2], pts[-1]
            if t1 > t0:
                x, y = local_xy(la0, lo0, la1, lo1)
                kt = math.hypot(x, y) / ((t1 - t0) / 3600.0) / 1.852
                return (math.degrees(math.atan2(x, y)) % 360.0, kt)
        return None

    def centre_at(self, when: dt.datetime, max_extrap_h: float = 6.0) -> Optional[Tuple[float, float]]:
        fixes = [f for f in self.fixes or [] if f.time is not None and abs((f.time - when).total_seconds()) <= 5400]
        if fixes:
            try:
                import recon10s_windfield as wf
                la, lo = wf.centre_track(self.fixes, [when], self.heading_speed())
                if np.isfinite(la[0]):
                    return float(la[0]), float(lo[0])
            except Exception:
                pass
        pts = self.points()
        if not pts:
            return None
        t = when.timestamp()
        ts = np.array([p[0] for p in pts])
        if ts[0] <= t <= ts[-1] and len(pts) > 1:
            j = int(np.clip(np.searchsorted(ts, t), 1, len(pts) - 1))
            (t0, la0, lo0), (t1, la1, lo1) = pts[j - 1], pts[j]
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            dlon = ((lo1 - lo0 + 180) % 360) - 180
            return la0 + f * (la1 - la0), ((lo0 + f * dlon + 180) % 360) - 180
        if t >= ts[-1] and t - ts[-1] <= max_extrap_h * 3600:
            hs = self.heading_speed()
            la, lo = pts[-1][1], pts[-1][2]
            if hs is None:
                return la, lo
            d = hs[1] * 1.852 * (t - ts[-1]) / 3600.0
            la2, lo2 = destinations(la, lo, d * math.sin(math.radians(hs[0])), d * math.cos(math.radians(hs[0])))
            return float(la2), float(lo2)
        if t < ts[0] and ts[0] - t <= 3 * 3600:
            return pts[0][1], pts[0][2]
        return None


# ------------------------------------------------------------- estimates

@dataclass
class Estimate:
    time: dt.datetime
    vmax_q: Tuple[float, ...]            # kt at QUANTILES
    mslp_q: Tuple[float, ...]            # mb
    centre: Tuple[float, float]
    satellite: str = ""                  # "GOES-19", "Himawari-9"
    ir_frames: int = 1                   # of 5
    mw_sensor: str = ""
    mw_time: Optional[dt.datetime] = None
    mw_cover: float = 0.0
    notes: List[str] = field(default_factory=list)
    ir_vmax: Optional[float] = None      # the median from IR alone, when a pass was used
    ir_mslp: Optional[float] = None

    @property
    def vmax(self) -> float:
        return float(self.vmax_q[2])

    @property
    def mslp(self) -> float:
        return float(self.mslp_q[2])

    @property
    def mw_age_h(self) -> Optional[float]:
        return (self.time - self.mw_time).total_seconds() / 3600.0 if self.mw_time else None

    @property
    def label(self) -> str:
        mw = f"{self.mw_sensor} {self.mw_age_h:.1f} h old" if self.mw_time else "IR only"
        return f"{self.time:%d %b %H:%MZ}: {self.vmax:.0f} kt, {self.mslp:.0f} mb ({mw})"

    def to_dict(self) -> dict:
        return {"time": self.time.isoformat(), "vmax_q": [round(float(v), 2) for v in self.vmax_q],
                "mslp_q": [round(float(v), 2) for v in self.mslp_q], "centre": list(self.centre),
                "satellite": self.satellite, "ir_frames": self.ir_frames, "mw_sensor": self.mw_sensor,
                "mw_time": self.mw_time.isoformat() if self.mw_time else None, "mw_cover": self.mw_cover,
                "notes": list(self.notes),
                "ir_vmax": None if self.ir_vmax is None else round(float(self.ir_vmax), 2),
                "ir_mslp": None if self.ir_mslp is None else round(float(self.ir_mslp), 2)}

    @staticmethod
    def from_dict(d: dict) -> "Estimate":
        return Estimate(time=dt.datetime.fromisoformat(d["time"]), vmax_q=tuple(d["vmax_q"]),
                        mslp_q=tuple(d["mslp_q"]), centre=tuple(d.get("centre") or (0.0, 0.0)),
                        satellite=d.get("satellite", ""), ir_frames=int(d.get("ir_frames", 1)),
                        mw_sensor=d.get("mw_sensor", ""),
                        mw_time=dt.datetime.fromisoformat(d["mw_time"]) if d.get("mw_time") else None,
                        mw_cover=float(d.get("mw_cover") or 0.0), notes=list(d.get("notes") or []),
                        ir_vmax=d.get("ir_vmax"), ir_mslp=d.get("ir_mslp"))


def _adjust(q, fit: dict, lat: float, kind: str) -> Tuple[float, ...]:
    """Quantiles q (5) moved and widened as one JTWC_ADJUST entry says."""
    med = float(q[2])
    shift = 0.0
    if fit.get("coef"):
        x = [1.0]
        if "level" in fit.get("terms", ()):
            x.append((med - 65.0) / 50.0 if kind == "vmax" else med - 1000.0)
        if "lat" in fit.get("terms", ()):
            x.append((abs(lat) - 20.0) / 10.0)
        shift = float(sum(c * v for c, v in zip(fit["coef"], x)))
    scale = float(fit.get("scale", 1.0))
    return tuple(med + shift + scale * (float(v) - med) for v in q)


def to_jtwc(vq, pq, lat: float, ir_only: bool = False):
    """(Vmax, MSLP) quantiles mapped onto JTWC's best track (JTWC_ADJUST)."""
    a = JTWC_ADJUST["ir" if ir_only else "pass"]
    return _adjust(vq, a["vmax"], lat, "vmax"), _adjust(pq, a["mslp"], lat, "mslp")


def estimate_from_grids(frames: Sequence[Optional[np.ndarray]], when: dt.datetime, centre: Tuple[float, float],
                        speed_kt: Optional[float], east_pacific: bool, mw: Optional[np.ndarray] = None,
                        mw_ir: Optional[np.ndarray] = None, mw_time: Optional[dt.datetime] = None,
                        mw_sensor: str = "", mw_cover: float = 0.0, satellite: str = "",
                        model_path: Optional[str] = None, jtwc: bool = False) -> Estimate:
    """One estimate from grids already on the network's storm-centred grids; `jtwc` maps it onto JTWC's
    best track (a storm of JTWC's basins)."""
    n = net(model_path)
    if frames[0] is None or not np.isfinite(frames[0]).mean() > 0.8:
        raise ValueError("the current IR frame does not cover the storm")
    use_mw = mw is not None and mw_time is not None and mw_cover >= MW_MIN_COVER
    age = (when - mw_time).total_seconds() / 3600.0 if use_mw else 0.0
    if use_mw and not (-0.25 <= age <= MW_MAX_AGE_H):
        use_mw = False
    ir, mwx, sc = network_inputs(frames, mw if use_mw else None, mw_ir if use_mw else None, max(age, 0.0),
                                 SENSOR_RES_KM.get(mw_sensor.upper(), 10.0), mw_cover, centre[0], speed_kt,
                                 east_pacific, when, centre[1])
    v, p = n.predict(ir, mwx, sc, ROTATIONS)
    vq, pq = tuple(float(x) for x in v[0]), tuple(float(x) for x in p[0])
    if jtwc:
        vq, pq = to_jtwc(vq, pq, centre[0], ir_only=not use_mw)
    notes = []
    ir_v = ir_p = None
    if use_mw:
        # the same estimate from IR alone, to show what the pass added
        mw0, sc0 = np.zeros_like(mwx), sc.copy()
        sc0[:, 0:4] = 0.0
        v0, p0 = n.predict(ir, mw0, sc0, ROTATIONS)
        v0q, p0q = v0[0], p0[0]
        if jtwc:
            v0q, p0q = to_jtwc(v0q, p0q, centre[0], ir_only=True)
        ir_v, ir_p = float(v0q[2]), float(p0q[2])
    else:
        notes.append("IR only")
    if jtwc:
        notes.append("mapped onto JTWC's best track")
    return Estimate(time=when, vmax_q=vq, mslp_q=pq,
                    centre=(float(centre[0]), float(centre[1])), satellite=satellite,
                    ir_frames=1 + sum(1 for f in frames[1:] if f is not None),
                    mw_sensor=mw_sensor if use_mw else "", mw_time=mw_time if use_mw else None,
                    mw_cover=mw_cover if use_mw else 0.0, notes=notes, ir_vmax=ir_v, ir_mslp=ir_p)


# ------------------------------------------------- fetching for a live storm

@dataclass
class MwInput:
    time: dt.datetime
    sensor: str
    grid: np.ndarray                     # (4, 64, 64) K
    cover: float
    centre: Tuple[float, float]
    label: str = ""


class Estimator:
    """
    Estimates for one storm at any times, fetching and caching what each needs: band-13 frames
    from GOES or Himawari-9 (one per 10-minute full-disk scan, shared between estimates) and
    microwave passes. `email` signs in to PPS for microwave; without it every estimate is IR only.
    """

    def __init__(self, storm_id: str, source: TrackSource, email: Optional[str] = None,
                 sensors: Sequence[str] = ("GMI", "AMSR3", "WSFM"), progress: Optional[Callable[[str], None]] = None,
                 model_path: Optional[str] = None):
        self.storm_id = (storm_id or "").lower()
        self.source = source
        self.email = email
        self.sensors = list(sensors)
        self.progress = progress or (lambda _m: None)
        self.model_path = model_path
        self._frames: Dict[str, Tuple[np.ndarray, dt.datetime, int]] = {}
        self._scenes: Dict[Tuple[int, str], list] = {}
        self._mw: Optional[List[MwInput]] = None
        self._mw_searched: Optional[Tuple[dt.datetime, dt.datetime, Optional[dt.datetime]]] = None
        self.errors: List[str] = []

    @property
    def east_pacific(self) -> bool:
        """The network's basin input: 0 for the Atlantic, 1 elsewhere -- JTWC's basins included, which
        scored better so (on 2020-2023, before 2024-2025 were looked at)."""
        return not self.storm_id.startswith("al")

    @property
    def jtwc(self) -> bool:
        return self.storm_id[:2] in JTWC_BASINS

    # ---- IR
    def _scene_near(self, sat: int, when: dt.datetime, tol_min: float = 16.0):
        import recon10s_goes as goes
        key = (sat, when.strftime("%Y%m%d%H"))
        if key not in self._scenes:
            self._scenes[key] = goes.list_scenes(sat, 13, "F", when.replace(minute=59, second=59), hours_back=0)
        cands = list(self._scenes[key])
        # the neighbouring hour, when the time is near its edge
        for off in (-1, 1):
            t2 = when + dt.timedelta(hours=off)
            k2 = (sat, t2.strftime("%Y%m%d%H"))
            if abs((when - when.replace(minute=30, second=0, microsecond=0)).total_seconds()) > 1200:
                if k2 not in self._scenes:
                    try:
                        self._scenes[k2] = goes.list_scenes(sat, 13, "F", t2.replace(minute=59, second=59), hours_back=0)
                    except Exception:
                        self._scenes[k2] = []
                cands += self._scenes[k2]
        if not cands:
            return None
        best = min(cands, key=lambda s: abs((s.start - when).total_seconds()))
        return best if abs((best.start - when).total_seconds()) <= tol_min * 60 else None

    def frame(self, when: dt.datetime, centre: Optional[Tuple[float, float]] = None
              ) -> Optional[Tuple[np.ndarray, dt.datetime, int]]:
        """(4-km IR grid in K, scan time, satellite) nearest `when`, or None."""
        import recon10s_goes as goes
        centre = centre or self.source.centre_at(when)
        if centre is None:
            return None
        sat = goes.pick_satellite(centre[0], centre[1])
        zen = goes.zenith_angle(centre[0], centre[1], sat)
        if zen > MAX_ZENITH:
            msg = (f"{goes.sat_name(sat)} sees {abs(centre[0]):.1f}{'N' if centre[0] >= 0 else 'S'} "
                   f"{abs(centre[1]):.1f}{'E' if centre[1] >= 0 else 'W'} at a zenith angle of {zen:.0f}\u00b0, "
                   f"too obliquely for an estimate (limit {MAX_ZENITH:.0f}\u00b0)")
            if msg not in self.errors:
                self.errors.append(msg)
            return None
        try:
            scene = self._scene_near(sat, when)
        except Exception as exc:
            self.errors.append(f"{goes.sat_name(sat)} listing: {exc}")
            return None
        if scene is None:
            return None
        if scene.key in self._frames:
            return self._frames[scene.key]
        self.progress(f"{goes.sat_name(sat)} band 13, {scene.start:%d %b %H:%MZ}…")
        lat0, lon0 = self.source.centre_at(scene.start) or centre
        half = 3.4
        bbox = (lat0 - half, lat0 + half, lon0 - half / max(0.3, math.cos(math.radians(lat0))),
                lon0 + half / max(0.3, math.cos(math.radians(lat0))))
        try:
            img = goes.load_image(scene, bbox=bbox)
        except Exception as exc:
            self.errors.append(f"{goes.sat_name(sat)} {scene.start:%H:%MZ}: {exc}")
            return None
        out = (ir_from_goes(img, (lat0, lon0)), scene.start, sat)
        self._frames[scene.key] = out
        while len(self._frames) > 400:                 # kept across runs: drop the oldest scans
            self._frames.pop(min(self._frames, key=lambda k: self._frames[k][1]))
        return out

    # ---- microwave
    def passes(self, start: dt.datetime, end: dt.datetime, needed_from: Optional[dt.datetime] = None) -> List[MwInput]:
        """
        Microwave passes over the storm between start and end (gridded), oldest first. With
        needed_from, of the passes before it only the newest is fetched: estimates from then on
        never use the older ones.
        """
        if not self.email:
            return []
        if self._mw is not None and self._mw_searched:
            s0, e0, n0 = self._mw_searched
            if s0 <= start and e0 >= end and (n0 is None or (needed_from is not None and needed_from >= n0)):
                return [m for m in self._mw if start <= m.time <= end]
        import recon10s_mw as mwmod
        c = self.source.centre_at(end) or self.source.centre_at(start)
        if c is None:
            return []
        hours = (end - start).total_seconds() / 3600.0
        self.progress("Looking for microwave passes…")
        try:
            cands = mwmod.find_passes(self.sensors, c[0], c[1], self.email, hours=hours, now=end, progress=self.progress)
        except Exception as exc:
            self.errors.append(f"microwave: {exc}")
            return []
        when = lambda c: c.closest_time or c.granule.start
        cands = sorted(cands, key=when)

        def grid_of(cand) -> Optional[MwInput]:
            t = when(cand)
            centre = self.source.centre_at(t)
            if centre is None:
                return None
            try:
                self.progress(f"{cand.granule.sensor} {t:%d %b %H:%MZ}…")
                path = mwmod.download_granule(cand.granule, self.email)
                mp = mwmod.load_pass(path, cand.granule.sensor, centre, box_deg=4.0, granule=cand.granule)
            except Exception as exc:
                self.errors.append(f"{cand.granule.sensor} {t:%H:%MZ}: {exc}")
                return None
            bands = {b: (s.lat, s.lon, s.v, s.h) for b, s in mp.bands.items()}
            g, cover = mw_grid(bands, centre)
            if cover < MW_MIN_COVER:
                return None
            return MwInput(time=t, sensor=cand.granule.sensor, grid=g, cover=cover, centre=centre, label=cand.label)

        out = []
        if needed_from is not None:
            # before the first estimate only the newest pass that covers the storm is needed
            for cand in reversed([c for c in cands if when(c) < needed_from]):
                m = grid_of(cand)
                if m is not None:
                    out.append(m)
                    break
            cands = [c for c in cands if when(c) >= needed_from]
        for cand in cands:
            m = grid_of(cand)
            if m is not None:
                out.append(m)
        out.sort(key=lambda m: m.time)
        self._mw = out
        self._mw_searched = (start, end, needed_from)
        return out

    def latest_pass(self, when: dt.datetime, passes: Sequence[MwInput]) -> Optional[MwInput]:
        ok = [m for m in passes if when - dt.timedelta(hours=MW_MAX_AGE_H) <= m.time <= when + dt.timedelta(minutes=15)]
        return ok[-1] if ok else None

    # ---- estimates
    def estimate(self, when: dt.datetime, passes: Optional[Sequence[MwInput]] = None) -> Estimate:
        import recon10s_goes as goes
        centre = self.source.centre_at(when)
        if centre is None:
            raise ValueError(f"no storm position for {when:%d %b %H:%MZ}")
        cur = self.frame(when, centre)
        if cur is None:
            raise ValueError(f"no geostationary IR of the storm near {when:%d %b %H:%MZ}")
        t0 = cur[1]
        frames = [cur[0]]
        for lag in LAGS_H:
            f = self.frame(t0 - dt.timedelta(hours=lag))
            frames.append(f[0] if f is not None and abs((f[1] - (t0 - dt.timedelta(hours=lag))).total_seconds()) <= 3600
                          else None)
        if passes is None:
            passes = self.passes(t0 - dt.timedelta(hours=MW_MAX_AGE_H), t0, needed_from=t0)
        m = self.latest_pass(t0, passes)
        mw_ir = None
        if m is not None:
            f = self.frame(m.time)
            mw_ir = f[0] if f is not None else None
        hs = self.source.heading_speed()
        est = estimate_from_grids(frames, t0, self.source.centre_at(t0) or centre, hs[1] if hs else None,
                                  self.east_pacific, mw=m.grid if m else None, mw_ir=mw_ir,
                                  mw_time=m.time if m else None, mw_sensor=m.sensor if m else "",
                                  mw_cover=m.cover if m else 0.0, satellite=goes.sat_name(cur[2]),
                                  model_path=self.model_path, jtwc=self.jtwc)
        if m is None:
            est.notes = [("IR only: no microwave pass over the storm in the last 12 h" if self.email else
                          "IR only: no PPS sign-in (Imagery › Microwave)")] + [x for x in est.notes if x != "IR only"]
        zen = goes.zenith_angle(est.centre[0], est.centre[1], cur[2])
        if zen > OBLIQUE_ZENITH:
            est.notes.append(f"{goes.sat_name(cur[2])} sees the storm obliquely (zenith angle {zen:.0f}\u00b0): "
                             "less reliable")
        return est

    def history(self, end: dt.datetime, hours: float = 12.0, step_h: float = 1.0) -> List[Estimate]:
        """Estimates every step_h hours over the last `hours` (oldest first)."""
        n = int(round(hours / step_h))
        times = [end - dt.timedelta(hours=step_h * k) for k in range(n, -1, -1)]
        passes = self.passes(times[0] - dt.timedelta(hours=MW_MAX_AGE_H), end, needed_from=times[0])
        out = []
        for t in times:
            try:
                out.append(self.estimate(t, passes))
            except ValueError as exc:
                self.errors.append(str(exc))
        return out


# --------------------------------------------------------------- self-check

def self_check(path: Optional[str] = None) -> str:
    """Load the model and run it on a synthetic eye and on clear air: for --smoke-test."""
    n = net(path)
    g = grid(IR_N, IR_DX)
    X, Y = np.meshgrid(g, g)
    r = np.hypot(X, Y)
    eye = np.where(r < 12, 290.0, np.where(r < 120, 192.0 + 0.2 * r, 270.0))
    flat = np.full_like(eye, 285.0)
    t = dt.datetime(2026, 9, 1, 18, tzinfo=UTC)
    a = estimate_from_grids([eye] * 5, t, (20.0, -60.0), 10.0, False, model_path=path)
    b = estimate_from_grids([flat] * 5, t, (20.0, -60.0), 10.0, False, model_path=path)
    ok = a.vmax > b.vmax
    return (f"satellite intensity {'ok' if ok else 'SUSPECT'} (model v{n.info.get('version', 1)}; "
            f"synthetic eye {a.vmax:.0f} kt vs clear air {b.vmax:.0f} kt)")
