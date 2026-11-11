"""
Storms in every basin (recon10s_nhc): knackwx's ATCF feed, NRL's sector file as the fallback, the
merge with NHC's list, and best tracks for JTWC's basins from knackwx, UCAR or natyphoon -- all
from fixtures, offline.
"""
import datetime as dt
import json

import pytest

from conftest import fixture_text
import recon10s_jma as jma
import recon10s_nhc as nhc
import recon10s_verify as verify

UTC = dt.timezone.utc
TRUDY = nhc.Fix(storm_id="ep212026", name="Trudy", lat=19.0, lon=-106.7, classification="HU", vmax_kt=90,
                mslp_mb=970, time=dt.datetime(2026, 10, 18, 15, tzinfo=UTC), source="NHC CurrentStorms")


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(nhc, "_SHORT_IDS", {})
    monkeypatch.setattr(nhc, "_TEXT_CACHE", {})


def knack():
    return json.loads(fixture_text("knackwx_storms.json"))


def test_knackwx_storm_list():
    fixes = {f.storm_id: f for f in nhc.parse_knack(knack())}
    assert sorted(fixes) == ["ep212026", "io962026", "sh012027", "sh932027", "wp292026"]
    champi, wati = fixes["wp292026"], fixes["sh012027"]
    assert (champi.lat, champi.lon, champi.vmax_kt, champi.mslp_mb) == (13.5, 141.4, 120, 943)
    assert champi.time == dt.datetime(2026, 10, 18, 12, tzinfo=UTC) and champi.motion is not None
    assert champi.label.startswith("Typhoon Champi (29W)  13.5N  141.4E  120 kt")
    assert wati.label.startswith("Tropical Storm Wati (01P)  17.4S  161.8W")
    assert fixes["sh932027"].label.startswith("Invest 93S  2.0S  60.6E")
    # the short ids come from the feed: the Southern Hemisphere's P/S and the Indian Ocean's A/B can't be guessed
    assert nhc.short_id("sh012027") == "01P" and nhc.short_id("io962026") == "96A"
    assert nhc.agency("wp292026") == "JTWC" and nhc.agency("ep212026") == "NHC"


def test_unregistered_short_ids():
    assert nhc.short_id("wp252026") == "25W" and nhc.short_id("al092026") == "09L"
    assert nhc.short_id("sh052027") is None and nhc.known_short_id("wp252026") is None


def test_a_southern_typhoon_is_a_tropical_cyclone():
    d = knack()
    for s in d:
        if s["atcf_id"] == "01P":
            s["cyclone_nature"], s["winds"] = "TY", 70
    wati = next(f for f in nhc.parse_knack(d) if f.storm_id == "sh012027")
    assert wati.classification == "TC" and wati.label.startswith("Tropical Cyclone Wati (01P)")


def test_all_storms_nhc_first_then_jtwc_then_invests(monkeypatch):
    monkeypatch.setattr(nhc, "active_storms", lambda *a, **k: [TRUDY])
    monkeypatch.setattr(nhc, "knack_storms", lambda *a, **k: nhc.parse_knack(knack()))
    ids = [f.storm_id for f in nhc.all_storms()]
    assert ids == ["ep212026", "wp292026", "sh012027", "io962026", "sh932027"]
    assert nhc.all_storms()[0].source == "NHC CurrentStorms" and nhc.STORM_LIST_NOTES == []


def test_all_storms_when_nhc_is_down(monkeypatch):
    def down(*a, **k):
        raise ConnectionError("no route")
    monkeypatch.setattr(nhc, "active_storms", down)
    monkeypatch.setattr(nhc, "knack_storms", lambda *a, **k: nhc.parse_knack(knack()))
    fixes = nhc.all_storms()
    assert "ep212026" in [f.storm_id for f in fixes]           # Trudy from knackwx instead
    assert any("NHC unreachable" in n for n in nhc.STORM_LIST_NOTES)


def test_all_storms_fall_back_to_the_sector_file(monkeypatch):
    def down(*a, **k):
        raise ConnectionError("no route")
    monkeypatch.setattr(nhc, "active_storms", lambda *a, **k: [])
    monkeypatch.setattr(nhc, "knack_storms", down)
    monkeypatch.setattr(jma, "sector_storms", lambda *a, **k: jma.parse_sectors(fixture_text("nrl_sectors.txt")))
    fixes = {f.storm_id: f for f in nhc.all_storms()}
    # NHC answered (with nothing), so its basins' storms are not taken from the sector file -- invests are
    assert "wp252026" in fixes and "al072026" not in fixes and "al912026" in fixes
    assert fixes["wp252026"].label.startswith("Super Typhoon Surigae (25W)") and nhc.short_id("wp252026") == "25W"
    assert fixes["wp922026"].name == "Invest 92W"
    assert any("knackwx unreachable" in n for n in nhc.STORM_LIST_NOTES)


def test_best_track_sources_in_order(monkeypatch):
    nhc.parse_knack(knack())                                 # registers 29W as active
    asked = []
    track = fixture_text("knackwx_track_29W.txt")

    def fetch(url, *a, **k):
        asked.append(url)
        return track if "knackwx" in url else None
    monkeypatch.setattr(nhc, "fetch_text", fetch)
    pts = nhc.best_track("wp292026")
    assert asked == ["https://api.knackwx.com/atcf/v2/track/history/text?stormID=29W"]
    assert len(pts) == 30 and pts[0].time < pts[-1].time            # knackwx sends newest first
    assert pts[-1].vmax == 120 and pts[-1].radii[64] == (50, 45, 40, 50) and pts[-1].name == "Champi"


def test_best_track_skips_another_storms_track(monkeypatch):
    """knackwx's short ids carry no year: a 29W from another season must not pass for this one."""
    nhc.register_short_id("wp292027", "29W")
    track = fixture_text("knackwx_track_29W.txt")              # 2026's 29W
    asked = []

    def fetch(url, *a, **k):
        asked.append(url)
        return track if "knackwx" in url else None
    monkeypatch.setattr(nhc, "fetch_text", fetch)
    assert nhc.best_track("wp292027") == []
    assert [u.split("/")[2] for u in asked] == ["api.knackwx.com", "hurricanes.ral.ucar.edu", "www.natyphoon.top"]


def test_best_track_falls_back_to_natyphoon(monkeypatch):
    track = fixture_text("knackwx_track_29W.txt")

    def fetch(url, *a, **k):
        if "natyphoon" in url:
            return track
        raise ConnectionError("down")
    monkeypatch.setattr(nhc, "fetch_text", fetch)
    assert len(nhc.best_track("wp292026")) == 30


def test_season_windows():
    lo, hi = nhc.season_window("sh012027")
    assert lo == dt.datetime(2026, 7, 1, tzinfo=UTC) and hi == dt.datetime(2027, 7, 1, tzinfo=UTC)
    assert nhc.season_window("wp292026")[0].year == 2026 and nhc.season_window("x") is None


def test_cimss_codes_and_stale_lists(monkeypatch):
    nhc.parse_knack(knack())
    assert verify.adt_codes("sh012027") == ["01P"]            # not 01S as well
    adt = fixture_text("adt_09L_list.txt")                      # a 2026 list
    monkeypatch.setattr(nhc, "fetch_text", lambda url, *a, **k: adt)
    assert verify.fetch_adt("al092026")[0]
    assert verify.fetch_adt("al092027")[0] == []                # same code, wrong year: last season's list
    asked = []
    monkeypatch.setattr(nhc, "fetch_text", lambda url, *a, **k: asked.append(url))
    verify.fetch_cimss_ml("sh012027", "D-PRINT")
    assert asked == ["https://tropic.ssec.wisc.edu/real-time/DPRINT/2027/DPRINThist_2027_01P.txt",
                     "https://tropic.ssec.wisc.edu/real-time/DPRINT/2026/DPRINThist_2026_01P.txt"]


def test_latest_fix_of_a_jtwc_storm(monkeypatch):
    monkeypatch.setattr(nhc, "active_storms", lambda *a, **k: [])
    monkeypatch.setattr(nhc, "fetch_text", lambda url, *a, **k: fixture_text("knackwx_track_29W.txt"))
    f = nhc.latest_fix("wp292026")
    assert f.lat == 13.5 and f.lon == 141.4 and f.vmax_kt == 120 and f.name == "Champi"


def test_jma_storms_from_knackwx_else_the_sector_file(monkeypatch):
    monkeypatch.setattr(nhc, "knack_storms", lambda *a, **k: nhc.parse_knack(knack()))
    storms = {s.storm_id: s for s in jma.active_storms()}
    assert storms["wp292026"].name.upper() == "CHAMPI" and storms["wp292026"].basin == "W"
    assert storms["wp292026"].time == dt.datetime(2026, 10, 18, 12, tzinfo=UTC)

    def down(*a, **k):
        raise ConnectionError("no route")
    monkeypatch.setattr(nhc, "knack_storms", down)
    monkeypatch.setattr(jma, "sector_storms", lambda *a, **k: jma.parse_sectors(fixture_text("nrl_sectors.txt")))
    assert "wp252026" in {s.storm_id for s in jma.active_storms()}
