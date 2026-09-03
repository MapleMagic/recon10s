#!/usr/bin/env python3
"""
recon10s_live — incremental IWG1 reader for the live plot.

Wraps the parsing and physics in recon10s.py so the plotted extrapolated MSLP
is the same number that lands in the HDOB output (bias correction and the
550 hPa cutoff from encode_XXXX included).

Downloads land in the same cache file the converter uses, so a mission is only
ever pulled across the wire once. Starting the live plot on a flight you have
already converted reads from disk and is effectively instant; after that each
poll is an HTTP Range request for the handful of KB that are new.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List, Optional

import numpy as np

import recon10s

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

# Below this flight-level pressure, recon10s encodes a D-value instead of an
# extrapolated surface pressure, so there is nothing to plot.
MIN_PRESS_FOR_EXTRAP = 550.0


class LiveFeed:
    """Reads an IWG1 file (URL or local path) and yields only what is new."""

    def __init__(self, source: str, timeout: float = 60.0,
                 cache_dir: Optional[str] = None, use_cache: bool = True):
        self.source = source.strip()
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.is_url = self.source.lower().startswith(("http://", "https://"))
        self._offset = 0
        self._tail = ""
        self._primed = False
        self._last_t: Optional[dt.datetime] = None
        self._session = requests.Session() if (self.is_url and requests) else None

    def reset(self) -> None:
        self._offset = 0
        self._tail = ""
        self._primed = False
        self._last_t = None

    @property
    def cache_file(self) -> Optional[str]:
        if not self.is_url:
            return None
        return recon10s.cache_path_for(self.source, self.cache_dir)

    # ------------------------------------------------------------------ poll

    def poll(self) -> List["recon10s.IWG1Row"]:
        rows = self._poll_url() if self.is_url else self._poll_file()

        fresh = []
        for row in rows:
            if self._last_t is not None and row.t <= self._last_t:
                continue  # already have it (re-sent chunk, or a duplicate line)
            fresh.append(row)

        fresh.sort(key=lambda r: r.t)
        if fresh:
            self._last_t = fresh[-1].t
        return fresh

    def _poll_url(self) -> List["recon10s.IWG1Row"]:
        if requests is None:
            raise RuntimeError("The 'requests' package is needed to poll a URL.")

        path = self.cache_file
        rows: List["recon10s.IWG1Row"] = []

        # First poll: everything already on disk is free.
        if not self._primed:
            self._primed = True
            if self.use_cache and path and os.path.exists(path):
                rows.extend(recon10s.iter_iwg1_rows(path))
                self._offset = os.path.getsize(path)

        headers = {"Accept-Encoding": "gzip"}
        if self._offset:
            headers["Range"] = f"bytes={self._offset}-"

        with self._session.get(self.source, headers=headers, stream=True,
                               timeout=self.timeout) as resp:
            if resp.status_code == 416:
                return rows  # nothing appended since the last poll
            resp.raise_for_status()

            appending = resp.status_code == 206
            body = resp.content

        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if appending and self._offset:
                with open(path, "ab") as fh:
                    fh.write(body)
                self._offset += len(body)
            else:
                # Server ignored the Range header and sent the lot.
                with open(path, "wb") as fh:
                    fh.write(body)
                self._offset = len(body)
                self._tail = ""
                rows = list(recon10s.iter_iwg1_rows(path))
                return rows
        else:
            self._offset += len(body)

        rows.extend(self._rows_from_text(body.decode("utf-8", errors="replace")))
        return rows

    def _poll_file(self) -> List["recon10s.IWG1Row"]:
        with open(self.source, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            if size < self._offset:  # file rolled over or was replaced
                self.reset()
            fh.seek(self._offset)
            chunk = fh.read()
            self._offset = fh.tell()
        return self._rows_from_text(chunk.decode("utf-8", errors="replace"))

    def _rows_from_text(self, text: str) -> List["recon10s.IWG1Row"]:
        """Parse whole lines, holding back a partial trailing line for later."""
        if not text:
            return []
        buf = self._tail + text
        if buf.endswith("\n"):
            self._tail = ""
        else:
            buf, _, self._tail = buf.rpartition("\n")

        rows = []
        for line in buf.splitlines():
            row = recon10s.parse_iwg1_line(line)
            if row is not None:
                rows.append(row)
        return rows


# --------------------------------------------------------------- derived data

def extrapolated_mslp(row: "recon10s.IWG1Row") -> Optional[float]:
    """Extrapolated surface pressure for one row, or None where undefined."""
    if row.ps_hpa is None or row.ga_m is None:
        return None
    if row.ps_hpa < MIN_PRESS_FOR_EXTRAP:
        return None
    return recon10s.extrapolate_surface_pressure(row.ps_hpa, row.ga_m, row.temp_c)


SERIES_KEYS = ("t", "wind", "wdir", "mslp", "temp", "dewpt", "height", "press")


def rows_to_arrays(rows) -> Dict[str, np.ndarray]:
    """
    Everything the plot draws, as float arrays keyed by series name:

        t       epoch seconds        wind    flight-level wind, kt
        wdir    wind direction, deg  mslp    extrapolated surface pressure, mb
        temp    ambient temp, degC   dewpt   dew point, degC
        height  geopotential height, m
        press   static pressure at flight level, mb

    Missing values are NaN so the plot breaks the trace rather than drawing a
    straight line through a gap.
    """
    n = len(rows)
    out = {k: np.full(n, np.nan) for k in SERIES_KEYS}
    t = np.empty(n, dtype=float)

    wind, wdir, mslp = out["wind"], out["wdir"], out["mslp"]
    temp, dewpt, height, press = out["temp"], out["dewpt"], out["height"], out["press"]
    kts = recon10s.KTS_PER_MPS

    for i, r in enumerate(rows):
        t[i] = r.t.timestamp()
        if r.wspd_ms is not None:
            wind[i] = r.wspd_ms * kts
        if r.wdir_deg is not None:
            wdir[i] = r.wdir_deg
        if r.temp_c is not None:
            temp[i] = r.temp_c
        if r.td_c is not None:
            dewpt[i] = r.td_c
        if r.ga_m is not None:
            height[i] = r.ga_m
        if r.ps_hpa is not None:
            press[i] = r.ps_hpa
        p0 = extrapolated_mslp(r)
        if p0 is not None:
            mslp[i] = p0

    out["t"] = t
    return out


def append_arrays(existing: Optional[Dict[str, np.ndarray]], rows) -> Dict[str, np.ndarray]:
    """
    Add newly polled rows to arrays we already hold, instead of rebuilding the
    whole mission every 30 s. On a long flight that is the difference between
    a visible hitch on each update and none.
    """
    fresh = rows_to_arrays(rows)
    if not existing or existing.get("t") is None or len(existing["t"]) == 0:
        return fresh
    return {k: np.concatenate((existing[k], fresh[k])) for k in SERIES_KEYS}


def filter_rows_by_time_of_day(rows, start: Optional[str], end: Optional[str]):
    """Same UTC time-of-day window the converter uses."""
    start_sec = recon10s._time_input_to_seconds(start) if start else None
    end_sec = recon10s._time_input_to_seconds(end) if end else None
    return recon10s._filter_rows_by_time_of_day(rows, start_sec, end_sec)
