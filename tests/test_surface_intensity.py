"""Surface-wind reductions, the intensity timeline, best track, ADT and estimator verification. Offline."""
import datetime as dt
import json

import numpy as np
import pytest

from conftest import fixture_text
import recon10s_analysis as an
import recon10s_drops as drops
import recon10s_intensity as intensity
import recon10s_nhc as nhc
import recon10s_nhchdob as nhchdob
import recon10s_satview as satview
import recon10s_surface as surface
import recon10s_vdm as vdm
import recon10s_verify as verify

UTC = dt.timezone.utc


# ------------------------------------------------------------- reductions

@pytest.mark.parametrize("press,factor,label", [(696.5, 0.90, "700 mb"), (751.9, 0.90, "700 mb"),
                                                (843.0, 0.80, "850 mb"), (925.0, 0.80, "850 mb"),
                                                (985.0, 0.85, "1000 ft")])
def test_standard_ratios(press, factor, label):
    assert surface.reduction(press) == (factor, label)
    assert surface.surface_estimate(100, press) == pytest.approx(100 * factor)


def test_no_ratio_for_high_flights():
    assert surface.reduction(179.6) is None             # a G-IV at 180 mb
    assert surface.surface_estimate(80, 179.6) is None
    assert surface.reduction(595.0) is None             # a P-3 transiting at 14,000 ft


def test_no_ratio_while_changing_level():
    """A P-3 descending from 595 to 700 mb at ~45 mb a minute, then level: only the level obs get a ratio."""
    t0 = dt.datetime(2026, 10, 8, 15, 20, tzinfo=UTC)
    press = [595.0, 618.5, 641.4, 664.1, 684.5, 699.0, 699.4, 698.8, 699.1, 699.3]
    ov = satview.Overlay()
    for k, p in enumerate(press):
        ov.lats.append(25.0); ov.lons.append(-88.0 + 0.01 * k); ov.times.append(t0 + dt.timedelta(seconds=30 * k))
        ov.winds.append(50); ov.dirs.append(180); ov.mslps.append(None); ov.segments.append(0)
        ov.heights.append(3000.0); ov.fl_press.append(p); ov.peaks.append(None); ov.wind_flagged.append(False)
        ov.labels.append("NOAA3 0809A")
    level = surface.level_flight(ov)
    assert not level[:6].any() and level[6:].all()      # 684.5 -> 699.0 is still 29 mb a minute
    est = surface.overlay_surface(ov)
    assert np.isnan(est[:6]).all() and est[6:] == pytest.approx(np.full(4, 45.0))
    assert not np.isnan(surface.overlay_surface(ov, level_only=False)[3])


def test_outer_vortex_ratio_blends_with_radius():
    w = np.array([80.0, 80.0, 80.0, 80.0])
    p = np.full(4, 700.0)
    r = np.array([20.0, 30.0, 45.0, 200.0])             # RMW 20 km: at it, 1.5x, halfway (2.25x), beyond 3x
    est = surface.radius_dependent(w, p, r, 20.0)
    assert est[0] == pytest.approx(72.0) and est[1] == pytest.approx(72.0)     # 0.90 out to 1.5 x RMW
    assert est[2] == pytest.approx(80 * 0.85)           # halfway between 0.90 and 0.80
    assert est[3] == pytest.approx(64.0)                # 0.80 in the outer vortex


@pytest.fixture(scope="module")
def isaias():
    """AF305's eyewall pass in Hurricane Isaias, 8 Oct 2026, with drop #92 released at 12:01:55Z."""
    batch = nhchdob.parse_message(fixture_text("hdob_af305_isaias.txt"))
    ov = satview.merge_overlays([satview.overlay_from_batches([batch])])
    d = drops.parse_tempdrop(fixture_text("tempdrop_isaias_92.txt"))
    drops._resolve_times(d, dt.datetime(2026, 10, 8, 12, 24, tzinfo=UTC))
    v = vdm.parse_vdm(fixture_text("vdm_af305_1156.txt"), dt.datetime(2026, 10, 8, 12, 35, tzinfo=UTC))
    return batch, ov, d, v


def test_drop_beside_flight_level(isaias):
    batch, ov, d, v = isaias
    assert d.location == "SE Eyewall" and d.wl150 == (200, 53) and d.mbl == (205, 58)
    row = surface.drop_comparisons([d], ov)[0]
    assert row.fl_level == "700 mb" and row.fl_gap_s <= 30
    assert row.fl_wind == 73 and row.fl_est == pytest.approx(73 * 0.9)
    assert row.wl150_est == pytest.approx(53 * 0.83) and row.mbl_est == pytest.approx(58 * 0.80)
    assert row.sfc_wind == 42 and row.observed_ratio == pytest.approx(42 / 73)
    assert row.best_estimate == row.wl150_est


def test_kwbc_messages_are_listed():
    """NHC's backup issues messages as KWBC; most of Isaias's drops came that way."""
    listing = ('<a href="REPNT3-KNHC.202610081224.txt">x</a> <a href="REPNT3-KWBC.202610081845.txt">y</a>')
    assert len(drops._FILE_RE.findall(listing)) == 2
    assert len(vdm._FILE_RE.findall(listing.replace("REPNT3", "REPNT2"))) == 2
    assert len(nhchdob._FILE_RE.findall(listing.replace("REPNT3", "AHONT1"))) == 2


# ---------------------------------------------------------------- timeline

def test_splash_adjustment():
    assert intensity.splash_adjusted(976.0, 10) == pytest.approx(975.0)
    assert intensity.splash_adjusted(976.0, None) == pytest.approx(976.0)


def test_timeline_from_recon(isaias):
    batch, ov, d, v = isaias
    centre = drops.parse_tempdrop(fixture_text("tempdrop_centre.txt"))
    drops._resolve_times(centre, dt.datetime(2026, 9, 26, 17, 50, tzinfo=UTC))

    class B:
        pass
    b = B()
    b.overlay, b.vdms, b.drops = ov, [v], [d, centre]
    b.fixes = an.fixes_from_vdms([v])
    b.passes = an.passes(ov, b.fixes)
    tl = intensity.build_timeline(b)
    assert [p.value for p in tl.pressure["Vortex message"]] == [978.0]
    raw = tl.pressure["Centre drop (splash)"][0]
    adj = tl.pressure["Centre drop − 1 mb/10 kt"][0]
    assert raw.value - adj.value == pytest.approx((centre.surface.wspd or 0) / 10.0)
    wl = tl.wind["Eyewall drop WL150 × 0.83"]
    assert len(wl) == 1 and wl[0].value == pytest.approx(53 * 0.83)
    fl = tl.wind["Flight level × standard ratio"]
    assert len(fl) == 1 and fl[0].value == pytest.approx(0.9 * max(o.wspd for o in batch.obs if o.wspd))


def test_best_track_parse_and_motion():
    track = nhc.parse_bdeck(fixture_text("bdeck_al092026.dat"))
    last = track[-1]
    assert last.time == dt.datetime(2026, 10, 8, 18, tzinfo=UTC) and (last.vmax, last.mslp) == (75, 977)
    assert last.radii == {34: (110, 80, 70, 110), 50: (40, 30, 30, 40), 64: (15, 15, 15, 15)}
    assert last.radius(34, "SW") == 70 and last.rmw_nm == 15 and last.name == "Isaias"
    hd, kt = nhc.track_motion(track, 12.0)
    assert hd == pytest.approx(60, abs=3) and kt == pytest.approx(9, abs=1)    # TCM: 60 deg at 9 kt
    assert nhc.track_at(track, dt.datetime(2026, 10, 8, 16, 30, tzinfo=UTC)).time.hour == 18


def test_currentstorms_motion_is_mph(monkeypatch):
    payload = {"activeStorms": [{"id": "al092026", "name": "Isaias", "classification": "HU", "intensity": "75",
                                 "pressure": "975", "latitudeNumeric": 24.0, "longitudeNumeric": -89.8,
                                 "movementDir": 60, "movementSpeed": 10,
                                 "lastUpdate": "2026-10-08T18:00:00.000Z"}]}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return payload
    monkeypatch.setattr(nhc.requests, "get", lambda *a, **k: R())
    fix = nhc.active_storms()[0]
    assert fix.motion[0] == 60 and fix.motion[1] == pytest.approx(8.69, abs=0.01)   # 10 mph


# --------------------------------------------------------------- estimates

def test_adt_listing():
    ests = verify.parse_adt(fixture_text("adt_09L_list.txt"))
    first = ests[0]
    assert first.time == dt.datetime(2026, 10, 7, 1, 30, 21, tzinfo=UTC)
    assert (first.vmax, first.mslp) == (30.0, 1004.5)
    assert first.lat == pytest.approx(22.08) and first.lon == pytest.approx(-95.34)    # ADT: west positive
    assert "CRVBND" in first.note and "CI 2.0" in first.note
    assert verify.adt_codes("al092026") == ["09L"] and verify.adt_codes("io012026") == ["01A", "01B"]


def test_adt_over_land_and_missing_values_are_not_estimates():
    listing = ("2026OCT09 234021  4.9  960.6  87.4  4.6 4.3 4.3  NO LIMIT   ON  OFF  OFF  OFF -67.30 -66.65  "
               "UNIFRM   N/A   48.2   29.82   86.90  FCST    GOES19 37.0\n"
               "2026OCT10 004021  0.0    0.0   0.0  0.0 0.0 0.0            N/A  N/A  N/A  OFF  99.50  99.50  "
               "LAND     N/A    N/A   30.78   87.07  ARCHER  GOES19 38.1\n")
    ests = verify.parse_adt(listing)
    assert len(ests) == 1 and ests[0].vmax == pytest.approx(87.4) and ests[0].lon == pytest.approx(-86.90)
    assert verify.plausible(0.0, verify.VMAX_RANGE) is None and verify.plausible(-999.0, verify.MSLP_RANGE) is None
    assert verify.plausible(72.0, verify.VMAX_RANGE) == 72.0


def test_dprint_and_dmint_histories():
    dp = verify.parse_cimss_ml(fixture_text("dprint_2026_09L.txt"))
    dm = verify.parse_cimss_ml(fixture_text("dmint_2026_09L.txt"))
    assert len(dp) == 93 and dp[0].time == dt.datetime(2026, 10, 7, 0, 0, tzinfo=UTC)
    assert dp[0].vmax == 32.0 and dp[0].mslp is None and dp[0].lon == pytest.approx(-95.40)   # MSLP NaN
    assert dp[1].mslp == 1004.0 and "25-75 %: 29-37 kt" in dp[1].note
    first = dm[0]
    assert first.time == dt.datetime(2026, 10, 7, 1, 26, tzinfo=UTC) and first.note.startswith("SSMISF16")
    assert (first.vmax, first.mslp) == (31.0, 1006.0)
    assert len({(e.time, e.note.split(",")[0]) for e in dm}) == len(dm)        # the doubled ATMS line kept once
    assert all(dm[i].time <= dm[i + 1].time for i in range(len(dm) - 1))
    urls = []

    def fetch(url):
        urls.append(url)
        return fixture_text("dmint_2026_09L.txt") if "DMINT" in url else None
    ests, url, text = verify.fetch_cimss_ml("al092026", "D-MINT", fetch=fetch)
    assert url == "https://tropic.ssec.wisc.edu/real-time/DMINT/2026/DMINThist_2026_09L.txt" and len(ests) == len(dm)
    assert verify.fetch_cimss_ml("al092026", "D-PRINT", fetch=fetch) == ([], "", "")
    assert verify.fetch_cimss_ml("nonsense", "D-PRINT", fetch=fetch) == ([], "", "") and len(urls) == 2


def test_matching_interpolates_and_scores():
    t0 = dt.datetime(2026, 10, 8, 12, tzinfo=UTC)
    ests = [verify.Estimate(t0 + dt.timedelta(minutes=30 * k), vmax=60 + 2 * k, mslp=990 - k) for k in range(5)]
    truths = [intensity.Point(t0 + dt.timedelta(minutes=45), 60.0),          # between estimates
              intensity.Point(t0 + dt.timedelta(hours=2, minutes=40), 70.0),  # 40 min past the last
              intensity.Point(t0 + dt.timedelta(hours=6), 75.0)]              # too far from any
    pairs = verify.match(ests, truths, "vmax", max_gap_min=60, interp_span_h=3)
    assert len(pairs) == 2
    assert pairs[0].estimate == pytest.approx(63.0) and pairs[0].method == "interpolated"
    assert pairs[1].estimate == pytest.approx(68.0) and pairs[1].method.startswith("nearest")
    s = verify.scores(pairs)
    assert s.n == 2 and s.bias == pytest.approx((3.0 - 2.0) / 2)
    assert s.rmse == pytest.approx(np.sqrt((9 + 4) / 2))
