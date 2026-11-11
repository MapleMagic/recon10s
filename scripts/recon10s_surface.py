#!/usr/bin/env python3
"""
recon10s_surface — surface wind estimates without SFMR.

Two independent routes from what the aircraft measures to a 10 m wind:

  * Flight-level winds, reduced by the standard ratios NHC uses (NHC's
    "About the wind profile" note, after Franklin et al. 2003):
        700 mb                0.90  (eyewall; the outer vortex runs nearer 0.8)
        850 mb (and 925 mb)   0.80
        1000-1500 ft          0.85  (low-level invest flights)
    No ratio is offered above 625 mb (G-IV and other high flights, P-3
    transits), where the flight-level wind says little about the surface,
    nor while the aircraft is climbing or descending (the pressure changing
    by more than 10 mb a minute): a ratio belongs to a level.
  * Dropsonde layer means from the TEMP DROP 62626 remarks: the mean wind in
    the lowest 150 m (WL150) times 0.83, and over the lowest 500 m (MBL)
    times 0.80 -- plus the sonde's own last wind near the surface.

`drop_comparisons` lines the two up for every dropsonde: the flight-level
wind the aircraft measured when it released the sonde, its reduction, the
WL150 and MBL estimates, and the measured surface wind. The observed
surface/flight-level ratio for each drop shows how well the standard ratio
holds in this storm on this day.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

import recon10s_drops as drops_mod

# (lowest flight-level pressure mb, highest mb, ratio, which standard level)
FL_REDUCTIONS: Tuple[Tuple[float, float, float, str], ...] = (
    (625.0, 775.0, 0.90, "700 mb"),
    (775.0, 950.0, 0.80, "850 mb"),
    (950.0, 1100.0, 0.85, "1000 ft"),
)
# Faster than this the aircraft is changing level (~300 ft a minute near 700 mb; descents run
# 1000-1500 ft a minute), and its wind belongs to no standard level.
LEVEL_RATE_MB_MIN = 10.0
WL150_FACTOR = drops_mod.WL150_FACTOR       # 0.83
MBL_FACTOR = drops_mod.MBL_FACTOR           # 0.80

# Away from the eyewall the 700 mb ratio is lower: NHC's note gives 0.91 in
# the eyewall but 0.78 in the outer vortex (Franklin et al. 2003). Used for
# wind radii, which live in the outer vortex: full standard ratio inside
# 1.5 x RMW, the outer ratio beyond 3 x RMW, linear in between.
OUTER_RATIO = {"700 mb": 0.80, "850 mb": 0.80, "1000 ft": 0.85}
OUTER_BLEND = (1.5, 3.0)


def radius_dependent(wspd: np.ndarray, fl_press: np.ndarray, r_km: np.ndarray,
                     rmw_km: Optional[float]) -> np.ndarray:
    """
    Surface estimates with the ratio easing from the eyewall value to the
    outer-vortex value with distance from the centre (needs the RMW).
    """
    wspd = np.asarray(wspd, float)
    press = np.asarray(fl_press, float)
    r = np.asarray(r_km, float)
    inner = np.full(wspd.shape, np.nan)
    outer = np.full(wspd.shape, np.nan)
    for lo, hi, factor, label in FL_REDUCTIONS:
        sel = (press >= lo) & (press < hi)
        inner[sel] = factor
        outer[sel] = OUTER_RATIO.get(label, factor)
    if not rmw_km or not np.isfinite(rmw_km) or rmw_km <= 0:
        return wspd * inner
    a, b = OUTER_BLEND
    f = np.clip((r / rmw_km - a) / (b - a), 0.0, 1.0)
    return wspd * (inner + f * (outer - inner))


def reduction(fl_press: Optional[float]) -> Optional[Tuple[float, str]]:
    """(ratio, standard level) for a flight-level pressure in mb, or None above 625 mb."""
    if fl_press is None or not np.isfinite(fl_press):
        return None
    for lo, hi, factor, label in FL_REDUCTIONS:
        if lo <= fl_press < hi:
            return factor, label
    return None


def surface_estimate(wspd: Optional[float], fl_press: Optional[float]) -> Optional[float]:
    """Flight-level wind (kt) reduced to the surface, or None."""
    red = reduction(fl_press)
    if wspd is None or red is None or not np.isfinite(wspd):
        return None
    return float(wspd) * red[0]


def level_flight(ov, max_rate: float = LEVEL_RATE_MB_MIN, max_gap_s: float = 120.0) -> np.ndarray:
    """
    True for each ob of an Overlay flown level: its flight-level pressure
    changing by less than `max_rate` mb a minute between the neighbouring
    obs of the same flight. An ob with no neighbour within `max_gap_s`
    cannot be judged and counts as level.
    """
    if ov is None or not len(ov):
        return np.empty(0, bool)
    press = np.array([np.nan if p is None else p for p in ov.column("fl_press")], float)
    t = np.array([tt.timestamp() if tt else np.nan for tt in ov.times], float)
    segs = np.asarray(ov.column("segments", 0), int)
    ok = np.ones(press.size, bool)
    for sid in np.unique(segs):
        sel = np.flatnonzero(segs == sid)
        sel = sel[np.argsort(t[sel])]
        if sel.size < 2:
            continue
        p, tt = press[sel], t[sel]
        dt_ = np.diff(tt)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.abs(np.diff(p)) / np.where(dt_ > 0, dt_, np.nan) * 60.0
        near = (dt_ > 0) & (dt_ <= max_gap_s) & np.isfinite(rate)
        fast = near & (rate > max_rate)
        # in transit: the pressure changes fast on either side of the ob
        bad = np.zeros(sel.size, bool)
        bad[:-1] |= fast
        bad[1:] |= fast
        ok[sel] = ~bad
    return ok


def overlay_surface(ov, include_flagged: bool = False, level_only: bool = True) -> np.ndarray:
    """
    Surface estimate (kt) for every ob of an Overlay; NaN where there is no
    wind, no standard ratio for its level, the aircraft is climbing or
    descending (unless level_only is False), or (by default) the HDOB flags
    call the flight-level wind questionable.
    """
    if ov is None or not len(ov):
        return np.empty(0)
    winds = np.array([np.nan if w is None else w for w in ov.winds], float)
    press = np.array([np.nan if p is None else p for p in ov.column("fl_press")], float)
    factors = np.full(len(ov), np.nan)
    for lo, hi, factor, _label in FL_REDUCTIONS:
        factors[(press >= lo) & (press < hi)] = factor
    out = winds * factors
    if level_only:
        out[~level_flight(ov)] = np.nan
    if not include_flagged:
        flags = np.asarray(ov.column("wind_flagged", False), bool)
        out[flags] = np.nan
    return out


def level_summary(ov) -> List[Tuple[str, float, int]]:
    """(standard level, ratio, number of obs) for the flight levels in an Overlay."""
    if ov is None or not len(ov):
        return []
    press = np.array([np.nan if p is None else p for p in ov.column("fl_press")], float)
    out = []
    for lo, hi, factor, label in FL_REDUCTIONS:
        n = int(((press >= lo) & (press < hi)).sum())
        if n:
            out.append((label, factor, n))
    high = int((press < FL_REDUCTIONS[0][0]).sum())
    if high:
        out.append((f"above {FL_REDUCTIONS[0][0]:.0f} mb (no ratio)", float("nan"), high))
    return out


# ------------------------------------------------------------ dropsondes

@dataclass
class DropSurface:
    drop: object                          # recon10s_drops.Drop
    time: Optional[dt.datetime]
    location: str
    sfc_wind: Optional[int] = None        # the sonde's own wind at the surface level, kt
    sfc_press: Optional[float] = None
    wl150: Optional[int] = None           # lowest-150 m mean, kt
    wl150_est: Optional[float] = None
    mbl: Optional[int] = None             # lowest-500 m mean, kt
    mbl_est: Optional[float] = None
    fl_wind: Optional[int] = None         # flight-level wind when the sonde left the aircraft
    fl_press: Optional[float] = None
    fl_level: str = ""
    fl_est: Optional[float] = None
    fl_gap_s: Optional[float] = None      # seconds between release and the matched ob

    @property
    def best_estimate(self) -> Optional[float]:
        """The dropsonde-based estimate NHC would lean on first: WL150, then MBL."""
        return self.wl150_est if self.wl150_est is not None else self.mbl_est

    @property
    def observed_ratio(self) -> Optional[float]:
        """Measured surface wind / flight-level wind for this drop."""
        if self.sfc_wind is None or not self.fl_wind:
            return None
        return self.sfc_wind / float(self.fl_wind)


def _aircraft_of(label: str) -> str:
    return str(label or "").split(" ")[0]


def flight_level_at(ov, when: dt.datetime, aircraft: str = "", max_gap_s: float = 90.0):
    """(wind kt, FL pressure mb, gap s) of the ob nearest `when` from that aircraft, or None."""
    if ov is None or not len(ov) or when is None:
        return None
    stamps = np.array([t.timestamp() if t else np.nan for t in ov.times])
    gap = np.abs(stamps - when.timestamp())
    if aircraft:
        same = np.array([_aircraft_of(lb) == aircraft for lb in ov.column("labels", "")])
        if same.any():
            gap = np.where(same, gap, np.inf)
    if not np.isfinite(gap).any():
        return None
    i = int(np.nanargmin(gap))
    if gap[i] > max_gap_s:
        return None
    flags = ov.column("wind_flagged", False)
    wind = ov.winds[i] if not flags[i] else None
    return wind, ov.column("fl_press")[i], float(gap[i])


def drop_comparisons(drops: Sequence, ov=None) -> List[DropSurface]:
    """Every dropsonde's surface-wind estimates side by side, in release order."""
    out = []
    for d in drops or []:
        sfc = d.surface
        row = DropSurface(drop=d, time=d.release_time, location=d.location or "",
                          sfc_wind=sfc.wspd if sfc else None, sfc_press=sfc.p if sfc else None)
        if d.wl150:
            row.wl150, row.wl150_est = d.wl150[1], d.wl150[1] * WL150_FACTOR
        if d.mbl:
            row.mbl, row.mbl_est = d.mbl[1], d.mbl[1] * MBL_FACTOR
        fl = flight_level_at(ov, d.release_time, d.aircraft) if d.release_time else None
        if fl is not None:
            wind, press, gap = fl
            row.fl_wind, row.fl_press, row.fl_gap_s = wind, press, gap
            red = reduction(press)
            if wind is not None and red is not None:
                row.fl_est, row.fl_level = wind * red[0], red[1]
            elif red is None and press is not None:
                row.fl_level = f"{press:.0f} mb (no ratio)"
        out.append(row)
    return out


def ratio_summary(rows: Sequence[DropSurface]) -> Optional[Tuple[float, float, int]]:
    """(median, mean, n) of the observed surface/flight-level ratios, eyewall-strength winds only."""
    vals = [r.observed_ratio for r in rows
            if r.observed_ratio is not None and r.fl_wind is not None and r.fl_wind >= 50]
    if not vals:
        return None
    return float(np.median(vals)), float(np.mean(vals)), len(vals)
