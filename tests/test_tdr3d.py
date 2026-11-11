"""3D structure diagnostics on a synthetic storm with known shear, tilt and asymmetry."""
import math

import numpy as np
import pytest

from conftest import fixture_text
import recon10s_tdr3d as t3
import recon10s_tdrgrid as tg

KT = t3.MS_TO_KT


def make_storm(shear_heading=90.0, shear_ms=8.0, tilt_km=6.0, tilt_heading=45.0,
               wedge=None, conv_heading=None):
    """
    A 60 m/s Rankine vortex (RMW 20 km) over 0-12 km whose centre drifts
    linearly with height toward `tilt_heading`; an environmental wind that
    grows linearly with height toward `shear_heading` (shear_ms between
    2 and 8 km); reflectivity enhanced toward `conv_heading`; optionally a
    missing wedge of coverage (degrees, (start, end)).
    """
    x = np.arange(-149.0, 150.0, 2.0)
    y = np.arange(-149.0, 150.0, 2.0)
    z = np.arange(0.0, 12.5, 0.5)
    X, Y = np.meshgrid(x, y)
    U = np.zeros((z.size, y.size, x.size))
    V = np.zeros_like(U)
    dbz = np.zeros_like(U)
    vort = np.zeros_like(U)
    a_s = math.radians(shear_heading)
    a_t = math.radians(tilt_heading)
    for k, h in enumerate(z):
        f = (h - 2.0) / 6.0
        cx, cy = tilt_km * (h / 12.0) * math.sin(a_t), tilt_km * (h / 12.0) * math.cos(a_t)
        dx, dy = X - cx, Y - cy
        r = np.hypot(dx, dy)
        vt = np.where(r <= 20, 60 * r / 20, 60 * 20 / np.maximum(r, 1e-6))
        th = np.arctan2(dy, dx)
        U[k] = -vt * np.sin(th) + f * shear_ms * math.sin(a_s)
        V[k] = vt * np.cos(th) + f * shear_ms * math.cos(a_s)
        vort[k] = np.where(r <= 20, 6.0, 0.1)
        az = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
        base = np.where(r <= 100, 30.0, np.nan)
        if conv_heading is not None:
            base = base + 8.0 * np.cos(np.radians(az - conv_heading))
        dbz[k] = base
        if wedge:
            gone = (az >= wedge[0]) & (az < wedge[1])
            for arr in (U, V, dbz, vort):
                arr[k][gone] = np.nan
    lat = 17 + Y / 111.0
    lon = -109 + X / 106.0
    fields = {"U": U, "V": V, "REFLECTIVITY": dbz, "VORT": vort, "W": np.zeros_like(U),
              "WIND_SPEED": np.hypot(U, V), "TANGENTIAL_WIND": np.zeros_like(U), "RADIAL_WIND": np.zeros_like(U)}
    return tg.TdrGrid(name="synthetic", storm="TEST", flight="000000X1", x=x, y=y, z=z, lat=lat, lon=lon,
                      fields=fields, centre=(17.0, -109.0))


def test_heading_convention():
    assert t3.heading(1, 0) == pytest.approx(90)       # westerly shear (vector east) is 90, as SHIPS
    assert t3.heading(0, -1) == pytest.approx(180)
    assert t3.angle_diff(10, 350) == pytest.approx(20)


def test_column_recovers_tilt():
    g = make_storm(tilt_km=6.0, tilt_heading=45.0)
    col = t3.vortex_column(g)
    mag, hd = t3.tilt(col, 2.0, 8.0)
    assert mag == pytest.approx(6.0 * 6 / 12, abs=0.6)          # 3 km between 2 and 8 km
    assert abs(t3.angle_diff(hd, 45.0)) < 10


def test_shear_recovered_despite_missing_coverage():
    g = make_storm(shear_heading=250.0, shear_ms=8.0, wedge=(0, 90))
    col = t3.vortex_column(g)
    sh = t3.inner_core_shear(g, col, 2.0, 8.0, 120.0)
    assert sh.kt == pytest.approx(8.0 * KT, rel=0.15)
    assert abs(t3.angle_diff(sh.heading, 250.0)) < 10
    # Averaging the raw winds instead lets the 60 m/s vortex leak through the
    # coverage gap; the vortex-removed answer is the one to trust.
    k0, k1 = g.level(2.0), g.level(8.0)
    raw = [t3._disc_mean(g, k, col.x[k], col.y[k], 120.0, vortex_removed=False) for k in (k0, k1)]
    leak_free = [t3._disc_mean(g, k, col.x[k], col.y[k], 120.0) for k in (k0, k1)]
    raw_mean = math.hypot(raw[0][0], raw[0][1])
    removed_mean = math.hypot(leak_free[0][0], leak_free[0][1])
    assert raw_mean > 3 * removed_mean + 1.0                       # the raw 2 km mean is vortex-contaminated


def test_asymmetry_phase_and_quadrants():
    # convection toward 20 deg with shear toward 90 deg: 70 deg left of downshear
    g = make_storm(shear_heading=90.0, conv_heading=20.0)
    col = t3.vortex_column(g)
    a = t3.shear_relative(g, col, "REFLECTIVITY", 3.0, 90.0)
    assert a.wn1_phase == pytest.approx(70.0, abs=8)
    assert a.quadrant["DL"] > a.quadrant["UR"]


def test_isosurface_in_km():
    g = make_storm()
    verts, faces = t3.isosurface(g, "REFLECTIVITY", 25.0, extent_km=120)
    assert len(faces) > 100
    assert verts[:, 0].min() >= -121 and verts[:, 0].max() <= 121 and verts[:, 2].max() <= 12.5


def test_ships_text_direction_is_where_the_shear_comes_from():
    # The text product prints SHEAR DIR 66 for Polo 25 Sep 18Z; NHC's lsdiag file
    # for the same cycle gives the heading SDDC = 246 (westerly shear = 90). So the
    # text direction is the meteorological "from" and is turned around on reading.
    kt, hd = t3.parse_ships(fixture_text("ships_polo_2609251800.txt"))
    assert (kt, hd) == (9.0, 246.0)
    diag = t3.parse_lsdiag(fixture_text("lsdiag_polo_2609251800.txt"))
    assert diag["SDDC"] == hd and abs(diag["SHDC"] - kt) < 0.5          # the text shear is SHDC
    assert (diag["SHRD"], diag["SHTD"]) == (13.2, 256.0)
