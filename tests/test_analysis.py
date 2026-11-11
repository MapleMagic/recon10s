"""Recon analysis on a real eye pass: derived fix vs NHC's vortex message."""
import datetime as dt

import pytest

from conftest import fixture_text
import recon10s_analysis as an
import recon10s_nhchdob as nhchdob
import recon10s_satview as satview
import recon10s_vdm as vdm

UTC = dt.timezone.utc


@pytest.fixture(scope="module")
def pass_data():
    batches = [nhchdob.parse_message(fixture_text(f"hdob_af302_{n}.txt")) for n in (29, 30, 31, 32)]
    ov = satview.merge_overlays([satview.overlay_from_batches(batches)])
    v = vdm.parse_vdm(fixture_text("vdm_af302_1739.txt"), dt.datetime(2026, 9, 26, 18, 6, tzinfo=UTC))
    return ov, v


def test_derived_fix_matches_vortex_message(pass_data):
    ov, v = pass_data
    fixes = an.derive_fixes(ov, lambda t: (16.8, -155.7))
    assert len(fixes) == 1
    f = fixes[0]
    assert float(an.gc_km(f.lat, f.lon, v.lat, v.lon)) < 5.0
    assert abs((f.time - v.time).total_seconds()) < 120


def test_radial_profile_matches_vortex_message(pass_data):
    ov, v = pass_data
    p = an.passes(ov, an.fixes_from_vdms([v]))[0]
    assert abs(p.outbound.vmax - v.max_out[1]) <= 3          # 83 kt in the VDM
    assert abs(p.inbound.vmax - v.max_in[1]) <= 3            # 63 kt
    assert abs(p.outbound.rmw - v.max_out_at[1] * 1.852) < 10


def test_motion_and_geometry():
    t0 = dt.datetime(2026, 9, 26, 12, tzinfo=UTC)
    fixes = [an.CentreFix(t0 + dt.timedelta(hours=h), 16.8, -155.0 - 0.1 * h, "VDM") for h in range(4)]
    heading, kt = an.motion(fixes)
    assert heading == pytest.approx(270, abs=1)
    assert kt == pytest.approx(0.1 * 111.2 * 0.957 / 1.852, rel=0.03)
    assert an.bearing(0, 0, 1, 0) == pytest.approx(0)
    lat, lon = an.destination(16.8, -155.0, 90, 100)
    assert float(an.gc_km(16.8, -155.0, lat, lon)) == pytest.approx(100, rel=1e-3)
