"""NEXRAD sites, keys, resampling and sweep choice; gridded TDR reading. Offline."""
import datetime as dt
import types

import numpy as np
import pytest

import recon10s_nexrad as nx
import recon10s_nexradsites as sites
import recon10s_tdrgrid as tg

UTC = dt.timezone.utc


# ---- MWSynth's own checks for the station module, carried over unchanged
def test_station_table_and_mwsynth_rules():
    assert len(sites.STATIONS) == 163                       # NEXRAD only, 47 TDWR left out
    assert sites.find_nearest_station_in_range(25.5, -80.0)[0].icao == "KAMX"
    assert sites.find_nearest_station_in_range(18.2, -66.5)[0].icao == "TJUA"
    assert sites.find_nearest_station_in_range(15.0, -140.0) == (None, None)
    ranked = sites.stations_by_distance(16.5, -157.3, 3)
    assert ranked[0][0].icao == "PHWA" and ranked[0][1] < ranked[1][1] < ranked[2][1]


@pytest.mark.parametrize("key,site,when", [
    ("2026/09/27/PHWA/PHWA20260927_121858_V06", "PHWA", dt.datetime(2026, 9, 27, 12, 18, 58, tzinfo=UTC)),
    ("2013/05/20/KTLX/KTLX20130520_201643_V06.gz", "KTLX", dt.datetime(2013, 5, 20, 20, 16, 43, tzinfo=UTC)),
    ("2005/08/29/KLIX/KLIX20050829_120345.gz", "KLIX", dt.datetime(2005, 8, 29, 12, 3, 45, tzinfo=UTC)),
])
def test_parse_key_across_archive_eras(key, site, when):
    assert nx.parse_key(key) == (site, when)


def test_parse_key_skips_metadata():
    assert nx.parse_key("2026/09/27/PHWA/PHWA20260927_121858_V06_MDM") is None


def test_to_grid_geometry():
    az = np.arange(0.25, 360, 0.5)
    rng = np.arange(2125.0, 460e3, 250.0)
    vals = az[:, None] + 0 * rng[None, :]                  # value = azimuth
    img, ext = nx.to_grid(az, rng, vals, 0.5, (0, 0), 200e3, n=401)
    mid = 200
    east = img[mid, mid + 100]                             # 100 km due east
    north = img[mid - 100, mid]                            # 100 km due north
    assert abs(east - 90) <= 0.5 and (north <= 0.5 or north >= 359.5)
    ranges = np.broadcast_to(rng, (az.size, rng.size))
    img_r, _ = nx.to_grid(az, rng, ranges, 0.5, (0, 0), 200e3, n=401)
    assert abs(img_r[mid, mid + 100] - 100e3 / np.cos(np.radians(0.5))) <= 250   # within a gate


def test_sweep_choice_follows_split_cuts():
    """At 0.5 deg velocity lives only in the second sweep; sweep 0 has none."""
    n = 4
    vel = np.ma.masked_all((n * 2, 3))
    vel[n:2 * n] = 10.0
    refl = np.ma.array(np.ones((n * 2, 3)))
    radar = types.SimpleNamespace(
        fixed_angle={"data": np.array([0.48, 0.48])},
        fields={"reflectivity": {"data": refl}, "velocity": {"data": vel}},
        get_slice=lambda i: slice(i * n, (i + 1) * n))
    vol = nx.Volume(scan=None, radar=radar, site_lat=0, site_lon=0)
    assert vol.elevations() == [pytest.approx(0.48)]
    assert vol.sweep_for("reflectivity", 0.48) == 0
    assert vol.sweep_for("velocity", 0.48) == 1
    assert vol.sweep_for("cc", 0.48) is None


def test_beam_height():
    assert nx.beam_height_km(0, 0.5) == pytest.approx(0.0, abs=1e-6)
    assert 3.0 < nx.beam_height_km(200, 0.5) < 4.2         # ~3.5 km at 200 km, 0.5 deg


# ---- gridded TDR
@pytest.fixture
def tdr_file(tmp_path, monkeypatch):
    """A small analysis in HRD's layout: (x, y, level, time), -999.9 missing."""
    from conftest import write_hrd_analysis
    path = tmp_path / "t_xy.nc"
    write_hrd_analysis(str(path))
    data = path.read_bytes()
    monkeypatch.setattr(tg, "_fetch", lambda url, timeout=120.0: data)
    return "https://example/t_xy.nc.gz"


def test_grid_orientation_and_metadata(tdr_file):
    g = tg.load_grid(tdr_file)
    assert g.storm == "TEST" and g.start == dt.datetime(2026, 9, 25, 21, 54, tzinfo=UTC)
    plan = g.plan("REFLECTIVITY", 0.0)                     # value = x (km east)
    row = np.argmin(abs(g.y - 1)); col = np.argmin(abs(g.x - 29))
    assert plan[row, col] == pytest.approx(29)             # rows are north-south, columns east-west
    assert g.lat[-1, 0] > g.lat[0, 0] and g.lon[0, -1] > g.lon[0, 0]
    assert np.isnan(g.fields["WIND_SPEED"][0, 0, 0])       # -999.9 read as missing


def test_azimuthal_mean_and_section(tdr_file):
    g = tg.load_grid(tdr_file)
    z, rmw = g.rmw()
    assert abs(rmw[0] - 20) <= 2
    r, zz, sec = g.section("REFLECTIVITY", 90.0, max_r=40)  # east-west line: value = x
    k = np.argmin(abs(r - 21))
    assert sec[0, k] == pytest.approx(r[k], abs=1.0)


# ---- GR2Analyst palette (bundled NWS reflectivity table)
def test_palette_follows_gr2_rules():
    import recon10s_palette as P
    pal = P.load()
    assert (pal.units, pal.step, pal.vmin, pal.vmax) == ("dBZ", 5.0, 0.0, 95.0)
    cm, norm, ticks = P.reflectivity()
    rgb = lambda v: tuple(round(c * 255) for c in cm(norm(v))[:3])
    assert rgb(0) == (0, 160, 255)            # a band's first colour at its value
    assert rgb(9.99) == (0, 120, 255)         # ... blending to its second colour
    assert rgb(10) == (0, 165, 255)           # ... then a sharp step to the next line
    assert rgb(34.99) == (10, 126, 3) and rgb(35) == (251, 238, 0)
    assert rgb(85) == (184, 184, 184)
    assert cm(norm(-5))[3] == 0 and cm(np.nan)[3] == 0     # below 0 dBZ and missing: clear


def test_palette_single_colour_and_alpha_lines():
    import recon10s_palette as P
    pal = P.parse_pal("units: dBZ\ncolor: 0 0 0 0\ncolor: 10 255 255 255\n"
                      "color4: 20 255 0 0 128\n; a comment\n")
    assert pal.colour_at(5)[:3] == pytest.approx((0.5, 0.5, 0.5))      # one colour: blend to next
    assert pal.colour_at(20)[3] == pytest.approx(128 / 255)


def test_values_at(tdr_file):
    g = tg.load_grid(tdr_file)
    v = g.values_at(29.0, 1.0, 0.0)
    assert v["REFLECTIVITY"] == pytest.approx(29)          # the field is x itself
    assert v["lat"] == pytest.approx(17 + 1 / 111.0, abs=1e-3)
    assert g.values_at(500, 0, 0) is None


def test_tdr_probe(tdr_file, qtbot):
    pytest.importorskip("pytestqt")
    import recon10s_tdrgridpanel as P
    w = P.TdrGridPanel()
    qtbot.addWidget(w)
    w.grid = tg.load_grid(tdr_file)
    w.field_box.setCurrentIndex(w.field_box.findData("REFLECTIVITY"))
    w.height.setValue(0)
    w.view_box.setCurrentIndex(w.view_box.findData("overview"))
    w.redraw()
    roles = {v[0]: ax for ax, v in w._roles.items()}
    assert set(roles) == {"plan", "section", "rz"}
    text = w.probe_text(roles["plan"], 29.0, 1.0)
    assert "29.0 dBZ" in text and "29 km from centre toward 088" in text
    sec = w.probe_text(roles["section"], -10.0, 0.0)        # section toward 000: behind = south
    assert "toward 180" in sec
    import re
    rz = w.probe_text(roles["rz"], 20.0, 0.0)
    assert abs(int(re.search(r"RMW at this height (\d+) km", rz).group(1)) - 20) <= 2   # 2 km rings
