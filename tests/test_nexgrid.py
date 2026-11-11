"""Ground radar volumes on a storm-centred 3-D grid; storm-relative velocity; the eye finder. Offline."""
import datetime as dt
import math
import os
import types

import numpy as np
import pytest

from conftest import FIXTURES
import recon10s_nexgrid as ng
import recon10s_nexrad as nx
import recon10s_tdrgrid as tg

UTC = dt.timezone.utc
T = dt.datetime(2026, 10, 9, 18, 2, 11, tzinfo=UTC)
SITE = (27.0, -82.0)
ELEVATIONS = [0.5, 0.9, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1, 6.4, 8.0, 10.0, 12.5, 15.6, 19.5]   # VCP 212


def layered(h_km):
    """Reflectivity falling off 2.5 dB per km of height."""
    return 45.0 - 2.5 * h_km


class FakeVolume:
    """The NEXRAD Volume interface over analytic fields: value = fn(product, azimuth, slant range, beam height)."""

    def __init__(self, fn, elevations=ELEVATIONS, products=("reflectivity", "velocity"), height_m=0.0):
        self.fn, self.els, self.products = fn, list(elevations), products
        self.site_lat, self.site_lon = SITE
        self.site_height_m = height_m
        self.scan = types.SimpleNamespace(site="KTST", time=T, key="2026/10/09/KTST/KTST20261009_180211_V06")
        self.vcp = 212

    def elevations(self):
        return list(self.els)

    def sweep_for(self, product, elevation):
        product = nx.BASE_PRODUCT.get(product, product)
        if product not in self.products:
            return None
        for i, e in enumerate(self.els):
            if abs(e - elevation) < 0.1:
                return i
        return None

    def sweep_data(self, product, sweep, dealias=True):
        el = self.els[sweep]
        az = np.arange(0.5, 360.0, 1.0)
        rng = np.arange(2125.0, 330e3, 250.0)
        h = ng.beam_height(rng / 1e3 * math.cos(math.radians(el)), el, self.site_height_m / 1e3)
        vals = self.fn(product, az[:, None], rng[None, :], h[None, :]) + 0 * az[:, None]
        return az, rng, vals.astype(float), el, None


def refl_only(product, az, rng, h):
    return layered(h)


def test_beam_geometry():
    assert ng.beam_height(np.array(0.0), 0.5) == pytest.approx(0.0, abs=1e-9)
    # 4/3-earth beam: s sin(el) plus s^2 / (2 * 4/3 * 6371) at 100 km and 0.5 deg
    assert float(ng.beam_height(np.array(100.0), 0.5)) == pytest.approx(1.46, abs=0.01)
    assert float(ng.beam_height(np.array(100.0), 0.5, 0.03)) == pytest.approx(1.49, abs=0.01)
    s, az = ng.range_azimuth(27.0, -82.0, np.array([28.0, 27.0]), np.array([-82.0, -81.0]))
    assert s[0] == pytest.approx(111.2, abs=0.2) and az[0] == pytest.approx(0.0, abs=0.01)
    assert s[1] == pytest.approx(99.1, abs=0.3) and az[1] == pytest.approx(89.8, abs=0.3)


def test_vertical_interpolation_never_fills_wide_gaps():
    # one column, three beams at 1, 2 and 8 km, each 0.4 km either side of its centre
    V = np.array([10.0, 20.0, 80.0]).reshape(3, 1, 1)
    H = np.array([1.0, 2.0, 8.0]).reshape(3, 1, 1)
    half = np.full((3, 1, 1), 0.4)
    z = np.array([0.5, 0.7, 1.5, 2.3, 3.0, 5.0, 7.7, 8.3, 8.5])
    out = ng.vertical_interp(V, H, half, z, max_gap_km=2.0)[:, 0, 0]
    assert np.isnan(out[0])                        # more than half a beam below the lowest beam
    assert out[1] == pytest.approx(10.0)           # inside the lowest beam
    assert out[2] == pytest.approx(15.0)           # between close beams: interpolated
    assert out[3] == pytest.approx(20.0)           # just above 2 km: still inside that beam
    assert np.isnan(out[4]) and np.isnan(out[5])   # 5.2 km between the beams' edges: left empty
    assert out[6] == pytest.approx(80.0)           # inside the 8 km beam, from below
    assert out[7] == pytest.approx(80.0) and np.isnan(out[8])


def test_layered_atmosphere_is_recovered():
    vol = FakeVolume(refl_only)
    z = np.arange(0.5, 12.01, 0.5)
    g = ng.grid_volume(vol, centre=(27.0, -81.0), half_width_km=40, dx_km=2, z_km=z, products=("reflectivity",))
    assert set(g.fields) == {"REFLECTIVITY"} and g.source == "radar" and g.site == SITE
    assert g.centre == (27.0, -81.0) and g.start == T and g.name.startswith("KTST 20261009 180211")
    dbz = g.fields["REFLECTIVITY"]
    truth = np.broadcast_to(layered(z)[:, None, None], dbz.shape)
    s, _az = ng.range_azimuth(*SITE, g.lat, g.lon)                  # 59-145 km from the radar
    low = ng.beam_height(s, ELEVATIONS[0])[None]                    # lowest beam centre over each column
    half = (s / math.cos(math.radians(0.5)) * math.tan(math.radians(0.475)))[None]
    Z = z[:, None, None]
    between = (Z >= low + 0.01) & (Z <= 11.0)
    assert np.isfinite(dbz[between]).all()
    assert np.max(np.abs(dbz - truth)[between]) < 0.1             # linear between beams, nearest gate in range
    # under the lowest beam: its own value within half a beamwidth of its centre, nothing further down
    under = (Z < low - 0.01) & (Z > low - half + 0.01)
    lowest = np.broadcast_to(layered(low), dbz.shape)
    assert under.any() and np.max(np.abs(dbz - lowest)[under]) < 0.1
    assert np.isnan(dbz[Z < low - half - 0.01]).all()


def test_wide_beam_spacing_is_left_empty():
    # a sparse scan (VCP 31-like top at 4.5 deg): nothing is invented between 2.5 and 4.5 deg far out
    vol = FakeVolume(refl_only, elevations=[0.5, 1.5, 2.5, 3.5, 4.5])
    z = np.arange(0.5, 15.01, 0.5)
    g = ng.grid_volume(vol, centre=(27.0, -80.6), half_width_km=10, dx_km=2, z_km=z, products=("reflectivity",))
    col = g.fields["REFLECTIVITY"][:, g.y.size // 2, g.x.size // 2]
    s = float(ng.range_azimuth(*SITE, np.array(g.centre[0]), np.array(g.centre[1]))[0])
    top = float(ng.beam_height(np.array(s), 4.5)) + s * math.tan(math.radians(0.475))
    assert np.all(np.isnan(col[z > top + 0.01]))         # above the top beam's upper edge: empty
    assert np.all(np.isfinite(col[(z > 3.0) & (z < top - 0.6)]))


def test_storm_relative_velocity_removes_the_motion():
    motion = (300.0, 12.0)

    def flow(product, az, rng, h):
        # the whole atmosphere moving with the storm: radial component of 12 kt toward 300
        if product == "velocity":
            return 12.0 * np.cos(np.radians(az - 300.0)) + 0 * h
        return layered(h)
    vol = FakeVolume(flow)
    g = ng.grid_volume(vol, centre=(27.0, -81.0), half_width_km=40, dx_km=2, z_km=[1.0, 3.0, 6.0], motion=motion)
    assert {"REFLECTIVITY", "RADIAL_VELOCITY", "SR_VELOCITY"} <= set(g.fields)
    rv, sr = g.fields["RADIAL_VELOCITY"], g.fields["SR_VELOCITY"]
    ok = np.isfinite(rv) & np.isfinite(sr)
    assert ok.mean() > 0.9
    assert np.nanmax(rv[ok]) < -5.0                    # looking east into a 12 kt flow toward the WNW
    assert np.max(np.abs(sr[ok])) < 0.3
    ms = 12.0 / nx.MS_TO_KT
    assert g.motion == pytest.approx((ms * math.sin(math.radians(300)), ms * math.cos(math.radians(300))))
    # no motion, no storm-relative field
    g2 = ng.grid_volume(vol, centre=(27.0, -81.0), half_width_km=20, dx_km=4, z_km=[2.0])
    assert "SR_VELOCITY" not in g2.fields and "RADIAL_VELOCITY" in g2.fields


def test_nexrad_srv_panels():
    import recon10s_radarview as rv
    motion = (45.0, 10.0)
    vol = FakeVolume(lambda p, az, rng, h: 10.0 * np.cos(np.radians(az - 45.0)) + 0 * h)
    assert nx.storm_relative(np.array([45.0, 225.0]), np.array([[10.0], [-10.0]]), motion) == \
        pytest.approx(np.zeros((2, 1)))
    without = rv.render_panels(vol, ["velocity", "srv"], 0.5, half_width_m=100e3, n=101)
    assert without[0] is not None and without[1] is None      # SRV needs a motion
    panels = rv.render_panels(vol, ["velocity", "srv"], 0.5, half_width_m=100e3, n=101, motion=motion)
    assert panels[1].motion == motion and panels[0].motion is None
    assert np.nanmax(np.abs(panels[1].img)) < 0.1 and np.nanmax(panels[0].img) > 9.0


def ring_grid(cx=8.0, cy=-5.0, r_eye=25.0, base=5.0):
    x = np.arange(-80.0, 81.0, 1.0)
    y = x.copy()
    z = np.arange(0.5, 6.01, 0.5)
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X - cx, Y - cy)
    layer = base + 45.0 * np.exp(-((R - r_eye) / 8.0) ** 2)
    dbz = np.repeat(layer[None], z.size, axis=0)
    lat = 25.0 + Y / 111.0
    lon = -80.0 + X / 100.6
    return tg.TdrGrid(name="ring", storm="TEST", flight="KTST", x=x, y=y, z=z, lat=lat, lon=lon,
                      fields={"REFLECTIVITY": dbz}, centre=(25.0, -80.0))


def test_find_eye_and_recentre():
    # clear air outside the eyewall too, inside the search radius: the eye is the clear area echo encloses
    g = ring_grid()
    found = ng.find_eye(g)
    assert found is not None and math.hypot(found[0] - 8.0, found[1] + 5.0) < 1.5
    opened = ring_grid()
    X, Y = np.meshgrid(opened.x, opened.y)
    toward = (np.degrees(np.arctan2(X - 8.0, Y + 5.0)) + 360) % 360
    opened.fields["REFLECTIVITY"][:, (toward > 180) & (toward < 270)] = 5.0      # eyewall open to the SW
    near = ng.find_eye(opened)
    assert near is not None and math.hypot(near[0] - 8.0, near[1] + 5.0) < 3.0
    moved = ng.recentre(g, *found)
    assert moved.x[0] == pytest.approx(g.x[0] - found[0]) and g.x[0] == -80.0      # the original is untouched
    assert moved.centre[0] == pytest.approx(25.0 - 5.0 / 111.0, abs=0.01)
    assert moved.centre[1] == pytest.approx(-80.0 + 8.0 / 100.6, abs=0.01)
    r, _z, mean = moved.azimuthal_mean("REFLECTIVITY", dr=2.0, max_r=60)
    assert r[int(np.nanargmax(mean[2]))] == pytest.approx(25.0, abs=2.0)      # the eyewall, all round
    flat = ring_grid()
    flat.fields["REFLECTIVITY"][:] = 30.0
    assert ng.find_eye(flat) is None                          # no hole
    flat.fields = {}
    assert ng.find_eye(flat) is None


def test_section_along_a_line():
    g = ring_grid()
    X, Y = np.meshgrid(g.x, g.y)
    g.fields["LINEAR"] = np.repeat((X + 2 * Y)[None], g.z.size, axis=0)
    s, z, sec = g.section_line("LINEAR", -20.0, -10.0, 30.0, 20.0)
    length = math.hypot(50.0, 30.0)
    assert s[0] == 0.0 and s[-1] == pytest.approx(length, abs=1.0) and z.size == g.z.size
    xs = -20.0 + s * 50.0 / length
    ys = -10.0 + s * 30.0 / length
    assert np.allclose(sec[3], xs + 2 * ys, atol=1e-6)


def test_field_labels():
    assert ng.field_label("SR_VELOCITY") == ("Storm-relative velocity", "kt")
    assert ng.field_label("REFLECTIVITY") == ("Reflectivity", "dBZ")
    assert ng.field_label("WIND_SPEED") == ("Wind speed", "m/s")


def test_real_jma_cycle():
    """Naze, 25 Sep 2026 08:40Z: both products gridded, and SRV differs from radial velocity by the motion."""
    import recon10s_jma as jma
    import recon10s_jmagrib as jg

    def read(name):
        with open(os.path.join(FIXTURES, name), "rb") as fh:
            return jg.decode(fh.read())
    vol = jma.JmaVolume(read("jma_naze_Pze.bin"), read("jma_naze_Pvr.bin"), dt.datetime(2026, 9, 25, 8, 40, tzinfo=UTC))
    motion = (330.0, 15.0)
    b = math.radians(112.8)                              # the cycle's one shower: 148 km toward 113 deg
    la, lo = ng._destinations(vol.site_lat, vol.site_lon, np.array(148.0 * math.sin(b)),
                              np.array(148.0 * math.cos(b)))
    g = ng.grid_volume(vol, centre=(float(la), float(lo)), half_width_km=30, dx_km=1, z_km=[1.0, 2.0, 3.0],
                       motion=motion, dealias=False)
    assert {"REFLECTIVITY", "RADIAL_VELOCITY", "SR_VELOCITY"} <= set(g.fields)
    assert np.isfinite(g.fields["REFLECTIVITY"]).any() and np.isfinite(g.fields["RADIAL_VELOCITY"]).any()
    _s, phi = ng.range_azimuth(vol.site_lat, vol.site_lon, g.lat, g.lon)
    rv, sr = g.fields["RADIAL_VELOCITY"], g.fields["SR_VELOCITY"]
    ok = np.isfinite(rv) & np.isfinite(sr)
    want = -15.0 * np.cos(np.radians(phi - 330.0))
    assert np.max(np.abs((sr - rv) - want[None])[ok]) < 0.3
