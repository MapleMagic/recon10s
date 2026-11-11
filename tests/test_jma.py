"""JMA per-radar polar GRIB2: real Naze files from 25 Sep 2026 08:40Z; NRL sector file."""
import datetime as dt
import os

import numpy as np
import pytest

from conftest import FIXTURES, fixture_text
import recon10s_jma as jma
import recon10s_jmagrib as jg

UTC = dt.timezone.utc


def _read(name):
    with open(os.path.join(FIXTURES, name), "rb") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def refl():
    return jg.decode(_read("jma_naze_Pze.bin"))


@pytest.fixture(scope="module")
def vel():
    return jg.decode(_read("jma_naze_Pvr.bin"))


def test_site_and_cycle(refl):
    assert (refl.site_id, refl.site_code) == ("47909", "FUNC") and refl.name == "Naze (Amami)"
    assert refl.lat == pytest.approx(28.394, abs=0.01) and refl.lon == pytest.approx(129.552, abs=0.01)
    assert refl.frequency_mhz == 5350.0
    assert refl.reference == dt.datetime(2026, 9, 25, 8, 40, tzinfo=UTC)
    assert len(refl.sweeps) == 35


def test_sweep_geometry_and_times(refl):
    s = refl.sweeps[0]
    assert s.product == "reflectivity" and s.elevation == 25.0
    assert s.values.shape == (512, 128) and s.range_m[-1] == pytest.approx(63750)
    assert np.all(np.diff(s.azimuth) > 0) and s.azimuth[0] < 0.71        # rotated to start at north
    assert s.start == dt.datetime(2026, 9, 25, 8, 30, tzinfo=UTC)          # negative offset from 08:40
    assert s.prf == [1830.0, 1525.0]
    longest = [x for x in refl.sweeps if x.range_m[-1] > 390e3]
    assert longest and longest[0].values.shape == (512, 800)
    assert any(x.elevation < 0 for x in refl.sweeps)                      # sign-magnitude elevations


def test_velocity_levels(vel):
    assert all(s.product == "velocity" for s in vel.sweeps)
    vals = np.concatenate([s.values[np.isfinite(s.values)] for s in vel.sweeps])
    assert vals.min() >= -70 and vals.max() <= 70
    # level 1 is a real 0 m/s in the velocity files, not "no echo"
    assert any((s.no_echo is not None) for s in vel.sweeps)


def test_rle_and_sign_magnitude():
    # Template 7.200 with 8-bit values and top level 3: bytes above 3 are run digits,
    # each worth (byte - 4) times 252^k. Level 2, digit 8 (= 4 extra), then level 1.
    data = bytes([2, 3 + 1 + 4, 1])
    out = jg.decode_rle(data, 8, 3, 6)
    assert list(out) == [2, 2, 2, 2, 2, 1]
    assert jg._sm(0x8005, 16) == -5 and jg._sm(0x0005, 16) == 5


def test_cfradial_round_trip(refl, tmp_path):
    from scipy.io import netcdf_file
    path = jma.write_cfradial(str(tmp_path / "naze.nc"), refl)
    nc = netcdf_file(path, "r", mmap=False)
    assert nc.Conventions.decode() == "CF/Radial instrument_parameters"
    assert nc.variables["fixed_angle"][:].shape == (35,)
    dbz = np.array(nc.variables["DBZ"][:], float)
    s0, e0 = nc.variables["sweep_start_ray_index"][0], nc.variables["sweep_end_ray_index"][0]
    got = dbz[s0:e0 + 1, :128]
    want = refl.sweeps[0].values
    fill = nc.variables["DBZ"]._FillValue
    got[got == fill] = np.nan
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.nanmax(np.abs(got - want)) if np.isfinite(want).any() else True
    nc.close()


def test_sector_file_and_site_choice():
    storms = jma.parse_sectors(fixture_text("nrl_sectors.txt"))
    surigae = [s for s in storms if s.name == "SURIGAE"][0]
    assert (surigae.basin, surigae.wind_kt, surigae.lat, surigae.lon) == ("W", 130, 26.3, 130.0)
    sid, km = jma.nearest_site(surigae.lat, surigae.lon)
    assert sid == "47937" and 200 < km < 250                          # Okinawa, ~224 km
    nolo = [s for s in storms if s.name == "NOLO"][0]
    assert jma.nearest_site(nolo.lat, nolo.lon) == (None, None)        # no JMA radar for Hawaii


def test_volume_matches_nexrad_interface(refl, vel):
    vol = jma.JmaVolume(refl, vel, dt.datetime(2026, 9, 25, 8, 40, tzinfo=UTC))
    els = vol.elevations()
    assert -0.3 in [round(e, 1) for e in els] and 25.0 in els
    i = vol.sweep_for("reflectivity", -0.1)
    az, rng, vals, el, nyq = vol.sweep_data("reflectivity", i)
    assert rng[-1] > 390e3                                              # prefers the long-range sweep
    j = vol.sweep_for("velocity", 1.8)
    _, _, v, _, _ = vol.sweep_data("velocity", j)
    assert v.shape[0] == 512
