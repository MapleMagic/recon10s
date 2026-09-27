#!/usr/bin/env python3
"""
recon10s_irtable — the infrared enhancement curve used for GOES imagery.

Sampled straight from the reference colour bar rather than re-typed, so the
steps land where they do on the original. Two things about it are easy to get
wrong:

  * It is NOT linear in temperature. The bar carries 20 degC per label step
    from +40 down to -20, then 10 degC per step from -20 down to -90, so the
    cold end is stretched to twice the resolution of the warm end. Feeding it
    a plain linear Normalize puts every colour in the wrong place.
  * Below about -19 degC it is stepped, not a gradient: roughly 1 degC blocks
    of flat colour. Interpolating between samples would smear those edges, so
    lookups are nearest-sample.

Above -19 degC it is a smooth grey ramp, warm = dark, running to near-white
just before the cyan step. Then cyan, blue, green, yellow, orange, red to
black at -70, a second grey ramp, and magenta below -80.

    rgb = temperature_to_rgb(bt_celsius)     # (..., 3) uint8
    rgb = kelvin_to_rgb(abi_cmi)             # ABI IR bands are in kelvin
    cmap, norm = mpl_colormap()              # for colorbars
"""
from __future__ import annotations

import base64

import numpy as np

# Temperatures of the labelled ticks, warm end first. Consecutive entries are
# one equal step of bar length apart -- which is what makes the scale
# non-linear, since the steps are 20 degC at the warm end and 10 at the cold.
TICK_TEMPS = np.array([40.0, 20.0, 0.0, -20.0, -30.0, -40.0,
                       -50.0, -60.0, -70.0, -80.0, -90.0])

# 775 RGB triples sampled down the bar, warm end first.
_LUT_B64 = (
    "Xl5eAQEBAgICBQUFAAAABgYGAgICBAQEAQEBAQEBAQEBAAAAAQEBAQEBAAAAAAAAAAAAAAAAAQEBAAAAAgICCAgICgoK"
    "CQkJCgoKCgoKCAgICgoKEhISFRUVFBQUFBQUFBQUFBQUFRUVHBwcHh4eHBwcHBwcHR0dHBwcHh4eHR0dKCgoKSkpKCgo"
    "KSkpKCgoKCgoKCgoMjIyMzMzMzMzMjIyNDQ0MjIyNDQ0QEBAQEBAQEBAQUFBPz8/QUFBQEBAQEBAS0tLS0tLS0tLSkpK"
    "S0tLSkpKTExMVFRUVlZWVlZWVlZWVlZWVlZWWFhYXV1dYGBgXl5eX19fX19fX19fXl5eYGBgZmZmampqaGhoaGhoaGho"
    "Z2dnaWlpbm5ucHBwb29vcHBwcHBwb29vcnJyeHh4enp6enp6enp6enp6e3t7eXl5fHx8gYGBg4ODg4ODg4ODg4ODgoKC"
    "hISEjIyMjo6OjY2NjY2NjY2Njo6OjY2NlJSUlpaWlZWVlZWVlZWVlZWVlZWVlZWVoKCgoKCgoKCgoaGhoKCgoKCgoKCg"
    "q6urrKysqqqqq6urrKysqqqqrKystLS0tbW1tra2tbW1tbW1tra2tLS0tra2vr6+v7+/vr6+v7+/v7+/vr6+wMDAxcXF"
    "x8fHxsbGx8fHxsbGxMTEyMjIzc3Nz8/Pzs7Ozs7Ozs7Ozs7OzMzMz8/P1NTU1tbW1dXV1dXV1tbW1NTU1tbW3Nzc3d3d"
    "3Nzc3Nzc3Nzc3Nzc3d3d4ODg4uLi4uLi4uLi4uLi4uLi4eHh4+Pj6Ojo6urq6enp6Ojo6enp6enp6urq7e3t7+/v7u7u"
    "7u7u7u7u7u7u7u7u8vLy9PT08vLy8vLy8/Pz8/Pz9PT08/Pz9vb2+Pj49vb29/f39/f39vb29/f3+/v7/Pz8+/v7+/v7"
    "+/v7+vr6/Pz8/v///P///P7////6//35//z+6P//uv//QOPiGPb0Af39Av3/APz/A/7/Bvv9AOzwAOrwAOryAOryAujy"
    "BObyB+bzANblANbmANXnANXoANTnAtPoBNLoCNDpAMDbAMDbAMDcAL/cAL3cBL3cBrrcAKzPAKzQAKvQAarQAqrQBKnQ"
    "BqbQAJfCAJXCAJXDAZTDApLDAJLDBpTGCozEAG+qAGyqAGuqAWqrAmmrBGiuBmWsAFaiAFOgAlWiAFOhAFaiAFSiB1Cj"
    "AEKWAEGWAEKXAECWAECXAz6XBjyXCzuYACaEACGBACOEASCDAyGEAiCDBxyEAAJyAgByAQFyAABwAAJuAANsAARqBhBy"
    "BBFuAhJsARRqABVmABhlABZiABdhBiZqASpjASpgAihhBCliAilcAC5aBDtaAT9YAD5UAEFWAD5WAkBXAEBSAEROBFFR"
    "AlRNBVROAlRMBFZKAFZHAFpKBmROBWdIAmlEAGpHAGpGAG5BAHM2B4w3BZIwAZIxAJIyApIxAZMwAJUrAJYpB6QwBKYu"
    "AqgqAqonAKolAKshAK4hBLklBL0kAL0eAL8eAL4ZAL4cAMIXB9gaA94UAN4TAN4RAeARAN4QAOATAOQLCPoIAv4BAf8A"
    "AP8AAf8EAv0BCPwAEP8CEv8CEv4AFv4AGP4AGvwAHPwAJP8CJv8BKP8BKf8AKv8AK/4ALP4AMPwAP/8CQv4CPv8CPv8C"
    "QP8CQv4ASPwAUP8DVf4CVf0CVv4BVf4AVvwAXP0AZP4Dav8Eaf0CbP8DaP0AbP4AbvsAdfoAiP8Gkf8Glv4EmP4Dlv4A"
    "lvwAmv4Aov8Cqf8Dqv8CrP4CrP4Cqv4Ar/8Auv8CwP4Cwv0Aw/4BwP8AwP8Aw/wAyvwA1/8E3f8E3P8C3f8B3v8C4fwA"
    "6/wA9v8D/P8C/v8A/f8A/v8A//4C//wD/O4A/+oA/+oB/+gC/+oE/+gE/+gG/+YG+doA+NYA/tcA/tMA/9MD/9IG/9AI"
    "+8EA/MAA/sAA/r4A/r4A/74B/7wE+a0A/KoB/6gB/6gB/6kC/qoD/qoE/6gG+pcA/pUA/pQA/5UE/pIE/5QI/40G9nIA"
    "/GwA/GgB/GwA+mwA/WwA/2gC+1oA/lYC/1UG/1QD/1cB/VQA/1YC/1IF/EQA/kAC/kIB/kAC/0AE/z0I/zsL9iUA+iQA"
    "/CIA/yIA/yAA/x4E/xwI9gYA+QQA/AIB/QEC/QEE+wAE+gIF9gIE8AAA7AAA6wAC6AAC5wID5gMF4gIE3AAA1gAA1AAA"
    "0gIC0gEC0AEEzAACwwAAvgAAvAEAvAIBvgACvgACvgEEuQECsAAAqgAAqAEAqAIBqAACqAECogMGmAAAkwAAlAAAlAAA"
    "kgAAkAICiQQEcgAAawAAaQIBaQECawACagACaAIAZAQCVwAAUgABUgECUAACUgABUgECTgICRQAAPgAAPAIAQAECQAAC"
    "QAAEOgEEKAAAIQAAIAIAIAAAIgACIQABIAIEGAIECAAAAgAAAAIAAAMAAAEAAQAAAwABGBYWGBgYFxgYFxcXFxcXFhYW"
    "FxcXLS0tLi4uLi4uLi4uLi4uLi4uLi4uLy8vRERERkZGRUVFRUVFRkZGRUVFRUVFW1tbXFxcXFxcW1tbXFxcXFxcWlpa"
    "Xl5eioqKioqKioqKioqKioqKioqKioqKoqKioKCgoKCgoqKioKCgoqKioqKitra2urq6uLi4ubm5t7e3uLi4t7i4tri4"
    "zM/O0NHR0M7Q1szR1MvQ0c/Pzs/O5ufm5efm5Ofm5Obn5eXq8uHw/9Tw0IKx5Xa77XDA7nDB7XDC7XDC73HE7G7C5We8"
    "5me85Ga75me+5Ga+5mjA5Ga+3F222ly12ly22ly22ly121222lu01FWw0FKs0lOv0VKw0FSy0FGw0lS0zlGyxkipw0ao"
    "xUioxEioxEeoxUiqw0aovD+juj2guz6iuz6kvD6mvD6mvD6nqCqWqCqWqCmVqSqWqCqXpyqYqCyZpSeWniGQnR+OniCQ"
    "niCQniCRniCSnB6QlBaKkhSIlBaKkxSKlBaLkhSKkhaKig2CiAqBiAuEiAuEiAqFiAyGhgqHiAyIegB+fAJ8egF8fQJ8"
    "eAF2egJ6dwB1dQNxcAJsdAN2eAJ4bARudgh3cgRzegV4dARweQhyeAxxcw9ukleO"
)

_LUT = np.frombuffer(base64.b64decode("".join(_LUT_B64.split())),
                     dtype=np.uint8).reshape(-1, 3)
N = _LUT.shape[0]

# Geometry measured off the reference bar, in image rows: the bar spans 774
# rows, the warmest label sits 20.5 rows below the top, and labels are 73.4
# rows apart. The labels are inset from the ends, so the bar runs a little
# past +40 and -90 -- that inset is why the ends need extrapolating.
_BAR_ROWS = 774.0
_FIRST_TICK_ROW = 20.5
_TICK_SPACING_ROWS = 73.4

# Position of each tick along the bar: 0 at the warm end, 1 at the cold end.
_TICK_POS = (_FIRST_TICK_ROW + _TICK_SPACING_ROWS * np.arange(len(TICK_TEMPS))) / _BAR_ROWS
_STEP = _TICK_SPACING_ROWS / _BAR_ROWS

_warm_end = TICK_TEMPS[0] + (_TICK_POS[0] / _STEP) * (TICK_TEMPS[0] - TICK_TEMPS[1])
_cold_end = TICK_TEMPS[-1] - ((1.0 - _TICK_POS[-1]) / _STEP) * (TICK_TEMPS[-2] - TICK_TEMPS[-1])

_CAL_POS = np.concatenate(([0.0], _TICK_POS, [1.0]))
_CAL_TEMPS = np.concatenate(([_warm_end], TICK_TEMPS, [_cold_end]))

# Ends of the bar, in degC.
TEMP_MAX = float(_CAL_TEMPS[0])
TEMP_MIN = float(_CAL_TEMPS[-1])

# np.interp needs ascending x, so keep a cold-to-warm copy for lookups.
_CAL_TEMPS_ASC = _CAL_TEMPS[::-1]
_CAL_POS_DESC = _CAL_POS[::-1]


def temperature_to_position(temp_c):
    """Brightness temperature (degC) to 0-1 along the bar, 0 = warm end."""
    t = np.asarray(temp_c, dtype=float)
    return np.interp(t, _CAL_TEMPS_ASC, _CAL_POS_DESC,
                     left=_CAL_POS_DESC[0], right=_CAL_POS_DESC[-1])


def temperature_to_rgb(temp_c, missing=(0, 0, 0)):
    """
    Brightness temperature in degC to RGB. Any array shape in; same shape out
    with a trailing axis of 3, dtype uint8. NaN becomes `missing`.
    Nearest-sample, so the stepped blocks stay sharp.
    """
    t = np.asarray(temp_c, dtype=float)
    idx = np.clip(np.rint(temperature_to_position(t) * (N - 1)), 0, N - 1).astype(np.intp)
    rgb = _LUT[idx]
    bad = ~np.isfinite(t)
    if bad.any():
        rgb = rgb.copy()
        rgb[bad] = np.asarray(missing, dtype=np.uint8)
    return rgb


def kelvin_to_rgb(temp_k, missing=(0, 0, 0)):
    """ABI CMI for the IR bands is in kelvin."""
    return temperature_to_rgb(np.asarray(temp_k, dtype=float) - 273.15, missing=missing)


def mpl_colormap():
    """
    (colormap, norm) for matplotlib, so a colorbar puts the right temperatures
    against the right colours. The norm carries the non-linear stretch.
    """
    from matplotlib.colors import FuncNorm, ListedColormap

    cmap = ListedColormap(_LUT[::-1] / 255.0, name="recon10s_ir")

    def forward(value):
        # Norms run cold-to-warm; the bar is stored warm-to-cold.
        return 1.0 - temperature_to_position(value)

    def inverse(value):
        return np.interp(1.0 - np.asarray(value, dtype=float), _CAL_POS, _CAL_TEMPS)

    return cmap, FuncNorm((forward, inverse), vmin=TEMP_MIN, vmax=TEMP_MAX)
