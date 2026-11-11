#!/usr/bin/env python3
"""
recon10s_palette — GR2Analyst .pal colour tables.

    product: BR
    units: dBZ
    step: 5
    color: 10 0 165 255 0 8 197
    color: 20 16 255 8 10 126 3

Each "color:" line is a value and a colour. With a second colour on the line,
the shade blends from the first colour at that value to the second just below
the next line's value, then jumps to the next line's colour; with one colour,
it blends straight into the next line's colour. Values below the first line
are left clear, values above the last take its colour. "color4:" lines carry
an alpha channel (0-255) after each RGB. "step" is the legend's spacing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PALETTE_DIR = os.path.join(HERE, "palettes")
REFLECTIVITY_PAL = "NWS-Default-Color-Table.pal"


@dataclass
class Palette:
    stops: List[Tuple[float, tuple, Optional[tuple]]] = field(default_factory=list)  # value, start, end
    units: str = ""
    step: float = 10.0
    product: str = ""

    @property
    def vmin(self) -> float:
        return self.stops[0][0]

    @property
    def vmax(self) -> float:
        return self.stops[-1][0]

    def cmap(self, name: str = "pal", resolution: float = 0.05):
        """
        A matplotlib colormap over [vmin, vmax] that follows the table exactly:
        bins `resolution` wide, aligned with the table's values so every jump
        falls on a bin edge, each coloured by the table's own rule. (An
        interpolated colormap blurs the jumps by a bin either way.)
        """
        import numpy as np
        from matplotlib.colors import ListedColormap
        n = max(2, int(round((self.vmax - self.vmin) / resolution)))
        edges = self.vmin + np.arange(n) * (self.vmax - self.vmin) / n
        cm = ListedColormap([self.colour_at(v + 1e-9) for v in edges], name=name)
        over = self.stops[-1][2] or self.stops[-1][1]
        if hasattr(cm, "with_extremes"):              # matplotlib >= 3.4; set_* are being deprecated
            return cm.with_extremes(under=(0, 0, 0, 0), over=over, bad=(0, 0, 0, 0))
        cm.set_under((0, 0, 0, 0))
        cm.set_over(over)
        cm.set_bad((0, 0, 0, 0))
        return cm

    def norm(self):
        from matplotlib.colors import Normalize
        return Normalize(self.vmin, self.vmax, clip=False)

    def ticks(self, max_ticks: int = 12) -> List[float]:
        step = self.step or 10.0
        while (self.vmax - self.vmin) / step > max_ticks:
            step *= 2
        out, v = [], self.vmin
        while v <= self.vmax + 1e-9:
            out.append(round(v, 6))
            v += step
        return out

    def colour_at(self, value: float) -> tuple:
        """RGBA (0-1) the table gives a value, straight from the table's rules."""
        if value < self.vmin:
            return (0.0, 0.0, 0.0, 0.0)
        for i, (v, c0, c1) in enumerate(self.stops):
            nxt = self.stops[i + 1] if i + 1 < len(self.stops) else None
            if nxt is None or value < nxt[0]:
                if nxt is None:
                    return c1 or c0
                end = c1 if c1 is not None else nxt[1]
                f = (value - v) / (nxt[0] - v)
                return tuple(a + (b - a) * f for a, b in zip(c0, end))
        return self.stops[-1][1]


def _rgba(nums: List[float], alpha: bool) -> tuple:
    if alpha:
        r, g, b, a = nums[:4]
        return (r / 255, g / 255, b / 255, a / 255)
    r, g, b = nums[:3]
    return (r / 255, g / 255, b / 255, 1.0)


def parse_pal(text: str) -> Palette:
    pal = Palette()
    for raw in text.replace("\r", "").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip().lower()
        parts = rest.split()
        if key in ("color", "color4"):
            alpha = key == "color4"
            n = 4 if alpha else 3
            try:
                nums = [float(x) for x in parts]
            except ValueError:
                continue
            if len(nums) < 1 + n:
                continue
            value = nums[0]
            c0 = _rgba(nums[1:1 + n], alpha)
            c1 = _rgba(nums[1 + n:1 + 2 * n], alpha) if len(nums) >= 1 + 2 * n else None
            pal.stops.append((value, c0, c1))
        elif key == "units":
            pal.units = rest.strip()
        elif key == "step":
            try:
                pal.step = float(parts[0])
            except (IndexError, ValueError):
                pass
        elif key == "product":
            pal.product = rest.strip()
    pal.stops.sort(key=lambda s: s[0])
    if len(pal.stops) < 2:
        raise ValueError("A palette needs at least two colour lines.")
    return pal


def load(name: str = REFLECTIVITY_PAL) -> Palette:
    path = name if os.path.isabs(name) else os.path.join(PALETTE_DIR, name)
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse_pal(fh.read())


_CACHE = {}


def reflectivity():
    """(cmap, norm, ticks) from the bundled NWS table; cached."""
    if "refl" not in _CACHE:
        pal = load()
        _CACHE["refl"] = (pal.cmap("nws_reflectivity"), pal.norm(), pal.ticks())
    return _CACHE["refl"]
