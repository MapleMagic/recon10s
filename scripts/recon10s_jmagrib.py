#!/usr/bin/env python3
"""
recon10s_jmagrib — JMA's per-radar polar-coordinate GRIB2.

JMA's 20 C-band radars, one GRIB2 message per radar per 10-minute cycle,
holding every elevation of the cycle. The format uses three templates that
standard GRIB2 software (ecCodes, wgrib2, pygrib) and LROSE's Radx and grib2
libraries do not implement:

  3.50120  azimuth-range grid: bins per ray Nb, rays Nr (512, 0.703 deg),
           radar latitude/longitude (1e-6 deg), bin spacing (1e-3 m), first
           ray's starting azimuth (1e-2 deg)
  4.51022  radar product by elevation: site, frequency, operating mode, the
           elevation set value, scan start/end in seconds before the
           reference time, then for each ray its measured elevation and PRF
  5.200    run-length packing with level values (WMO standard), and 7.200
           its data: level bytes <= V, run-length digits above V

Layout, octet by octet, from JMA's "per-radar polar-coordinate echo
intensity GPV format (GRIB2, Ver. 2.00)" (MRI, SitePolar.pdf). Two things a
generic decoder would get wrong, both from that document:

  * Negative numbers are sign-and-magnitude (top bit set), not two's
    complement -- elevations, magnetic declination, scan times and the
    representative values of velocity levels.
  * Level 0 is "outside the range or missing"; in reflectivity, level 1 is
    "no echo". In the Doppler velocity files level 1 is a real 0 m/s (its
    gates lie inside the echo), so it is kept. Level values map to physical
    values through the table in section 5 -- velocity's interleaves signs
    (+0.5, -0.5, +1.0, ...) out to +-70 m/s, already unfolded by JMA's
    dual-PRF processing.

Rays are stored clockwise from the first ray's starting azimuth, so each
sweep is rotated back to north here. Files are named
Z__C_RJTD_<yyyyMMddhhmmss>_RDR_JMAGPV_RS<WMO#>_Gar0p5km0p7deg_<Pze|Pvr>_ANAL_grib2.bin
and ship as one TAR per 10 minutes for all radars (N5 reflectivity, N6
Doppler velocity).
"""
from __future__ import annotations

import datetime as dt
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Site names from the same specification (table 4); coordinates come from the files.
SITES = {
    "47415": ("SAPP", "Sapporo"), "47419": ("KUSH", "Kushiro"), "47432": ("HAKO", "Hakodate"),
    "47590": ("SEND", "Sendai"), "47582": ("AKIT", "Akita"), "47695": ("KASH", "Tokyo (Kashiwa)"),
    "47572": ("YAHI", "Niigata (Yahiko)"), "47705": ("TOJI", "Fukui (Tojimbo)"),
    "47611": ("KURU", "Nagano (Kurumayama)"), "47659": ("MAKI", "Shizuoka (Makinohara)"),
    "47636": ("NAGO", "Nagoya"), "47773": ("TAKA", "Osaka (Takayasu)"),
    "47791": ("MISA", "Matsue (Misaka)"), "47792": ("HAIG", "Hiroshima (Haigamine)"),
    "47899": ("MURO", "Muroto"), "47806": ("SEFU", "Fukuoka (Sefurisan)"),
    "47869": ("TANE", "Tanegashima"), "47909": ("FUNC", "Naze (Amami)"),
    "47937": ("ITOK", "Okinawa (Itokazu)"), "47920": ("ISHI", "Ishigakijima"),
}
# Positions (deg) from the specification's table, for choosing a site before any
# file is read; the files' own values are used for drawing.
SITE_POS = {
    "47415": (43.13889, 141.00972), "47419": (42.96083, 144.5175), "47432": (41.93361, 140.78139),
    "47590": (38.26222, 140.89667), "47582": (39.71778, 140.09944), "47695": (35.85972, 139.95972),
    "47572": (37.71861, 138.81611), "47705": (36.2375, 136.1425), "47611": (36.10306, 138.19583),
    "47659": (34.74278, 138.13361), "47636": (35.16833, 136.96472), "47773": (34.61611, 135.65611),
    "47791": (35.54167, 133.10333), "47792": (34.27028, 132.59333), "47899": (33.25222, 134.17722),
    "47806": (33.43472, 130.35694), "47869": (30.63944, 130.97861), "47909": (28.39333, 129.55083),
    "47937": (26.15333, 127.76444), "47920": (24.42667, 124.18222),
}
PARAMS = {(15, 1): "reflectivity", (15, 2): "velocity"}


def _sm(raw: int, bits: int) -> int:
    """Sign-and-magnitude integer, as JMA encodes negatives."""
    top = 1 << (bits - 1)
    return -(raw & (top - 1)) if raw & top else raw


def _u(b, o, n):
    return int.from_bytes(b[o:o + n], "big")


def _missing(v, n):
    return v == (1 << (8 * n)) - 1


@dataclass
class Sweep:
    product: str
    elevation: float                 # set value, deg
    ray_elevation: np.ndarray        # measured, per ray (north-first order)
    azimuth: np.ndarray              # ray centres, deg, north-first
    range_m: np.ndarray              # bin centres
    values: np.ndarray               # (rays, bins) float, NaN = missing or no echo
    start: dt.datetime
    end: dt.datetime
    prf: List[float] = field(default_factory=list)
    mode: int = 0
    no_echo: Optional[np.ndarray] = None   # True where the radar saw nothing


@dataclass
class SiteFile:
    site_id: str                    # WMO number
    site_code: str                  # e.g. FUNC
    lat: float
    lon: float
    height_m: float
    reference: dt.datetime
    frequency_mhz: Optional[float]
    sweeps: List[Sweep]

    @property
    def name(self) -> str:
        return SITES.get(self.site_id, (self.site_code, self.site_code))[1]


def decode_rle(data: bytes, nbit: int, maxv: int, npts: int) -> np.ndarray:
    """Template 7.200: bytes <= maxv are levels; bytes above are run-length digits."""
    lngu = (1 << nbit) - 1 - maxv
    out = np.zeros(npts, dtype=np.uint16)
    pos = 0
    i, n = 0, len(data)
    while i < n and pos < npts:
        level = data[i]
        i += 1
        run, power = 0, 1
        while i < n and data[i] > maxv:
            run += (data[i] - maxv - 1) * power
            power *= lngu
            i += 1
        run += 1
        out[pos:pos + run] = level
        pos += run
    return out


def decode(buf: bytes) -> SiteFile:
    """Decode one per-radar GRIB2 file (all its sweeps)."""
    if buf[:4] != b"GRIB" or buf[7] != 2:
        raise ValueError("not a GRIB2 message")
    total = struct.unpack(">Q", buf[8:16])[0]
    p = 16
    ref = None
    grid = None
    prod = None
    rep = None
    rle = None
    sweeps: List[Sweep] = []
    site = {}
    while p < total - 4:
        ln = _u(buf, p, 4)
        num = buf[p + 4]
        s = buf[p:p + ln]             # s[k] is octet k+1 of the section
        if num == 1:
            ref = dt.datetime(_u(s, 12, 2), s[14], s[15], s[16], s[17], s[18], tzinfo=dt.timezone.utc)
        elif num == 3:
            tmpl = _u(s, 12, 2)
            if tmpl != 50120:
                raise ValueError(f"grid template {tmpl}, expected 3.50120")
            grid = {"nb": _u(s, 14, 4), "nr": _u(s, 18, 4),
                    "lat": _sm(_u(s, 22, 4), 32) / 1e6, "lon": _sm(_u(s, 26, 4), 32) / 1e6,
                    "dx": _u(s, 30, 4) / 1e3, "dstart": _u(s, 34, 4) / 1e3,
                    "scan": s[38], "azi0": _u(s, 39, 2) / 100.0}
        elif num == 4:
            tmpl = _u(s, 7, 2)
            if tmpl != 51022:
                raise ValueError(f"product template {tmpl}, expected 4.51022")
            cat, par = s[9], s[10]
            elev_raw = _u(s, 41, 2)
            nprf = s[43]
            prfs = [_u(s, 44 + 2 * k, 2) for k in range(3)]
            t0, t1 = _sm(_u(s, 50, 2), 16), _sm(_u(s, 52, 2), 16)
            nr = grid["nr"]
            per = np.frombuffer(s[60:60 + 4 * nr], dtype=">u2").reshape(nr, 2).astype(np.int64)
            e = per[:, 0]
            ray_el = np.where(e == 0xFFFF, np.nan,
                              np.where(e & 0x8000, -(e & 0x7FFF), e) / 100.0)
            freq = _u(s, 32, 4)
            site = {"lat": _sm(_u(s, 14, 4), 32) / 1e6, "lon": _sm(_u(s, 18, 4), 32) / 1e6,
                    "height": _u(s, 22, 2) / 10.0, "code": s[24:28].decode("ascii", "replace").strip(),
                    "id": str(_u(s, 28, 2)), "freq": None if _missing(freq, 4) else freq / 1e3}
            prod = {"product": PARAMS.get((cat, par), f"param{cat}.{par}"),
                    "elevation": _sm(elev_raw, 16) / 100.0, "mode": s[37],
                    "prf": [v / 10.0 for v in prfs[:nprf] if not _missing(v, 2)] if nprf != 255 else [],
                    "t0": t0, "t1": t1, "ray_el": ray_el}
        elif num == 5:
            tmpl = _u(s, 9, 2)
            if tmpl != 200:
                raise ValueError(f"data representation template {tmpl}, expected 5.200")
            nbit, maxv, mvl, scale = s[11], _u(s, 12, 2), _u(s, 14, 2), s[16]
            vals = np.frombuffer(s[17:17 + 2 * mvl], dtype=">u2").astype(np.int64)
            vals = np.where(vals & 0x8000, -(vals & 0x7FFF), vals) / (10.0 ** scale)
            rep = {"npts": _u(s, 5, 4), "nbit": nbit, "maxv": maxv,
                   "table": np.concatenate([[np.nan], vals])}      # index = level; level 0 missing
        elif num == 7:
            levels = decode_rle(s[5:], rep["nbit"], rep["maxv"], rep["npts"])
            nb, nr = grid["nb"], grid["nr"]
            lv = levels.reshape(nr, nb)
            table = rep["table"]
            phys = table[np.clip(lv, 0, len(table) - 1)]
            if prod["product"] == "reflectivity":
                # Level 1 is "no echo" (the specification's table 3), not 0 dBZ.
                no_echo = lv == 1
                phys[no_echo] = np.nan
            else:
                # In the velocity files level 1 is a real 0 m/s: those gates sit
                # inside the echo, where the wind is across the beam; "nothing
                # there" is level 0. (Checked against same-sweep reflectivity.)
                no_echo = lv == 0
            # Rays start at the first ray's starting azimuth; turn the sweep so row 0 is north.
            dth = 360.0 / nr
            centres = (grid["azi0"] + (np.arange(nr) + 0.5) * dth) % 360.0
            order = np.argsort(centres)
            rng = grid["dstart"] + (np.arange(nb) + 0.5) * grid["dx"]
            sweeps.append(Sweep(
                product=prod["product"], elevation=prod["elevation"],
                ray_elevation=prod["ray_el"][order], azimuth=centres[order], range_m=rng,
                values=phys[order], start=ref + dt.timedelta(seconds=prod["t0"]),
                end=ref + dt.timedelta(seconds=prod["t1"]), prf=prod["prf"], mode=prod["mode"],
                no_echo=no_echo[order]))
        p += ln
    lat = site.get("lat", grid["lat"] if grid else np.nan)
    lon = site.get("lon", grid["lon"] if grid else np.nan)
    return SiteFile(site_id=site.get("id", ""), site_code=site.get("code", ""), lat=lat, lon=lon,
                    height_m=site.get("height", 0.0), reference=ref,
                    frequency_mhz=site.get("freq"), sweeps=sweeps)
