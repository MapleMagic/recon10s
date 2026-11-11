"""Decoders against real NHC messages (Hurricane Nolo, 26 Sep 2026)."""
import datetime as dt

import pytest

from conftest import fixture_text
import recon10s_drops as drops
import recon10s_nhchdob as nhchdob
import recon10s_vdm as vdm

UTC = dt.timezone.utc


def test_hdob_batch_decodes_by_position():
    b = nhchdob.parse_message(fixture_text("hdob_af302_27.txt"))
    assert (b.aircraft, b.mission, b.storm, b.number) == ("AF302", "0515E", "NOLO", 27)
    assert len(b.obs) == 20
    o = b.obs[0]
    assert o.time == dt.datetime(2026, 9, 26, 16, 56, 30, tzinfo=UTC)
    assert o.lat == pytest.approx(15 + 40 / 60)          # 1540N is degrees-minutes
    assert o.lon == pytest.approx(-(155 + 19 / 60))
    assert (o.wdir, o.wspd, o.peak) == (264, 59, 61)
    assert o.fl_press == pytest.approx(697.3)
    assert o.surface_press == pytest.approx(1004.2)      # "0042" -> 1004.2, not missed
    assert o.height == 3136


def test_hdob_position_flag_drops_zero_position():
    b = nhchdob.parse_message(fixture_text("hdob_af306_01.txt"))
    assert b is not None
    assert all(abs(o.lat) > 1 and abs(o.lon) > 1 for o in b.obs), "0000N 00000E line with flags 39 kept"


@pytest.mark.parametrize("flags,wind,pos", [("00", False, False), ("05", False, False),
                                            ("02", True, False), ("09", True, False),
                                            ("39", True, True), ("10", False, True)])
def test_hdob_quality_flags(flags, wind, pos):
    assert nhchdob.flags_wind_suspect(flags) is wind
    assert nhchdob.flags_position_suspect(flags) is pos


def test_vdm_fields():
    f = vdm.parse_vdm(fixture_text("vdm_af302_1739.txt"), dt.datetime(2026, 9, 26, 18, 6, tzinfo=UTC))
    assert f.time == dt.datetime(2026, 9, 26, 17, 39, tzinfo=UTC)
    assert (f.lat, f.lon) == (16.79, -155.66)
    assert f.min_slp == 977 and f.height_m == 2912 and f.level == "700 mb"
    assert f.eye == "CLOSED" and f.eye_shape == "E09/36/30" and f.eye_diameter_nm == 36
    assert f.max_in == (202, 63) and f.max_in_at == (106, 18, "17:33:30")
    assert f.max_out == (54, 83) and f.max_out_at == (319, 22, "17:45:30")
    assert (f.aircraft, f.mission, f.storm, f.ob) == ("AF302", "0515E", "NOLO", "19")
    assert f.max_fl_wind == (83, 319, 22, "14:38:30")


def test_vdm_month_boundary():
    text = fixture_text("vdm_af302_1739.txt").replace("A. 26/17:39:00Z", "A. 30/23:50:00Z")
    f = vdm.parse_vdm(text, dt.datetime(2026, 10, 1, 0, 10, tzinfo=UTC))
    assert f.time == dt.datetime(2026, 9, 30, 23, 50, tzinfo=UTC)


def test_tempdrop_centre_drop():
    d = drops.parse_tempdrop(fixture_text("tempdrop_centre.txt"))
    assert d.location == "Center"
    assert d.surface.p == 977 and d.surface.wspd == 1
    lv = {m.p: m for m in d.mandatory}
    assert lv[925].z == 487 and lv[925].t == pytest.approx(24.4)
    assert lv[850].z == 1227 and lv[700].z == 2894


PART_CD = """XXAA 76171 99171 70820 04700 99010 28000 09010 00084 26400 09512 92770 20800 10015
85503 16800 11020 70159 08400 13025 50590 07560 16030 40760 18160 17535 30966 32960 19040
25090 42760 20045 20240 55560 21050 15420 66560 22055 10673 75560 23060 88999 77999
61616 NASA872 WXWXA GUSTAV OB 12
62626 REL 1710N07000W 171231 SPG 1712N07005W 173100 =
XXCC 76171 99171 70820 04700 70870 73760 24060 50060 70560 25055 30380 64560
26040 20640 58560 27030 10090 47560 28020 88999 77999
61616 NASA872 WXWXA GUSTAV OB 12
62626 REL 1710N07000W 171231 SPG 1712N07005W 173100 =
XXDD 76178 99171 70820 04700 11850 75160 22400 72960 33200 60360
21212 11850 24058 22400 25050
61616 NASA872 WXWXA GUSTAV OB 12
62626 REL 1710N07000W 171231 SPG 1712N07005W 173100 ="""


def test_tempdrop_parts_c_and_d():
    d = drops.parse_tempdrop(PART_CD)
    high = {m.p: m for m in d.mandatory if m.p < 100}
    assert sorted(high) == [10, 20, 30, 50, 70]
    assert high[70].z == 18700 and high[50].z == 20600 and high[10].z == 30900
    assert high[70].t == pytest.approx(-73.7) and (high[30].wdir, high[30].wspd) == (260, 40)
    assert [(lv.p, lv.t) for lv in d.sig_temp if lv.p < 100] == [(85.0, -75.1), (40.0, -72.9), (20.0, -60.3)]
    assert [(lv.p, lv.wspd) for lv in d.sig_wind if lv.p < 100] == [(85.0, 58), (40.0, 50)]
