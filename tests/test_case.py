"""Saved cases: everything in, everything out, decoded again; replay availability. Offline."""
import dataclasses
import datetime as dt
import json
import os
import zipfile

import pytest

from conftest import fixture_text, write_hrd_analysis
import recon10s_case as case_mod
import recon10s_drops as drops
import recon10s_nhc as nhc
import recon10s_nhchdob as nhchdob
import recon10s_tdrevo as evo
import recon10s_tdrgrid as tg
import recon10s_vdm as vdm
import recon10s_verify as verify

UTC = dt.timezone.utc
ISAIAS = nhc.Fix(storm_id="al092026", name="ISAIAS", lat=23.4, lon=-90.6, classification="HU", vmax_kt=70,
                 mslp_mb=976, time=dt.datetime(2026, 10, 8, 12, tzinfo=UTC), motion_dir=60, motion_kt=8.7)
TDR_URL = tg.url_for("260925I1_2211_xy.nc.gz")
SATINT = [{"time": "2026-10-08T12:00:00+00:00", "vmax_q": [60.0, 65.0, 69.0, 74.0, 79.0],
           "mslp_q": [970.0, 975.0, 980.0, 984.0, 988.0], "centre": [23.4, -90.6], "satellite": "GOES-19",
           "ir_frames": 5, "mw_sensor": "GMI", "mw_time": "2026-10-08T09:30:00+00:00", "mw_cover": 0.9,
           "notes": []}]


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """A case built from the Isaias fixtures, a TDR grid and satellite estimates; returns (path, inputs)."""
    batch = nhchdob.parse_message(fixture_text("hdob_af305_isaias.txt"))
    v = vdm.parse_vdm(fixture_text("vdm_af305_1156.txt"), dt.datetime(2026, 10, 8, 12, 35, tzinfo=UTC))
    d = drops.parse_tempdrop(fixture_text("tempdrop_isaias_92.txt"))
    drops._resolve_times(d, dt.datetime(2026, 10, 8, 12, 24, tzinfo=UTC))
    track = nhc.parse_bdeck(fixture_text("bdeck_al092026.dat"))
    nc = tmp_path / "grid.nc"
    write_hrd_analysis(str(nc))
    metrics = [evo.PassMetrics(mission="20260925I1", name="260925I1_2211_xy.nc.gz",
                               time="2026-09-25T22:11:00+00:00", end="2026-09-25T22:19:00+00:00",
                               peak_2km_kt=117.0, available="2026-09-25T22:31:00+00:00")]
    inputs = dict(fix=ISAIAS, batches=[batch], vdms=[v], drops=[d], track=track,
                  adt_url=verify.ADT_URL.format(code="09L"), adt_text=fixture_text("adt_09L_list.txt"),
                  cimss_ml={"D-MINT": (verify.CIMSS_ML_URL["D-MINT"].format(year="2026", code="09L"),
                                       fixture_text("dmint_2026_09L.txt"))},
                  satint=SATINT,
                  tdr_grids={TDR_URL: str(nc)}, tdr_available={TDR_URL: "2026-09-25T22:31:00+00:00"},
                  evolution=metrics, imagery={"goes": {"product": "ir", "scenes": []}})
    path = str(tmp_path / "isaias")
    man = case_mod.save_case(path, title="Isaias AL092026", version="v2.3.0", **inputs)
    return man, inputs, nc


def test_round_trip(saved, tmp_path):
    man, inputs, nc = saved
    path = man["path"]
    assert path.endswith(".r10case") and os.path.exists(path) and not os.path.exists(path + ".part")
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        stored = json.loads(z.read("case.json"))
    assert {"case.json", "nhc/best_track.json", "tdr/evolution.json"} <= names
    assert any(n.startswith("hdob/") for n in names) and any(n.startswith("adt/") for n in names)
    assert stored["title"] == "Isaias AL092026" and stored["recon10s"] == "v2.3.0"
    assert stored["span"][0].startswith("2026-09-26") or stored["span"][0].startswith("2026-10-08")

    c = case_mod.load_case(path, cache_dir=str(tmp_path / "cache"))
    b0, b1 = inputs["batches"][0], c.batches[0]
    assert (b1.aircraft, b1.mission, b1.number) == (b0.aircraft, b0.mission, b0.number)
    assert [o.wspd for o in b1.obs] == [o.wspd for o in b0.obs]                 # decoded again, the same
    assert c.vdms[0].min_slp == 978 and c.vdms[0].time == inputs["vdms"][0].time
    d = c.drops[0]
    assert d.wl150 == (200, 53) and d.release_time == inputs["drops"][0].release_time and d.number == 1
    assert len(c.track) == len(inputs["track"]) and c.track[-1].radii[34] == (110, 80, 70, 110)
    assert c.track[-1].rmw_nm == 15 and c.track[-1].name == "Isaias"
    assert c.fix.storm_id == "al092026" and c.fix.motion == (60, 8.7) and "(case)" in c.fix.source
    assert verify.parse_adt(c.adt_text)[0].vmax == 30.0 and c.adt_url.endswith("09L-list.txt")
    assert c.satint == SATINT
    url, text = c.cimss_ml["D-MINT"]
    assert url.endswith("DMINThist_2026_09L.txt") and text == fixture_text("dmint_2026_09L.txt")
    assert c.evolution[0]["peak_2km_kt"] == 117.0
    assert c.imagery["goes"]["product"] == "ir"
    local = c.tdr_paths[TDR_URL]
    with open(local, "rb") as fh, open(nc, "rb") as orig:
        assert fh.read() == orig.read()                                          # stored gzipped, intact


def test_saved_grid_opens_without_the_network(saved, tmp_path, monkeypatch):
    man, _inputs, _nc = saved
    cache = tmp_path / "tdrcache"
    case_mod.load_case(man["path"], cache_dir=str(cache))
    monkeypatch.setattr(tg, "CACHE_DIR", str(cache))
    monkeypatch.setattr(tg.requests, "get", lambda *a, **k: pytest.fail("tried the network"))
    g = tg.load_grid(TDR_URL)
    assert g.storm == "TEST" and g.flight == "260925I1"


def test_replay_reveals_items_as_they_arrived(saved, tmp_path):
    man, inputs, _nc = saved
    c = case_mod.load_case(man["path"], cache_dir=str(tmp_path / "cache"))
    r = case_mod.Replay(c)
    b, v, d = c.batches[0], c.vdms[0], c.drops[0]
    # kept without NHC's file names, so issued at their headers' times -- but never before their own data
    batch_at = b.end                                            # header 081208, last ob 12:10:30
    vdm_at = dt.datetime(2026, 10, 8, 12, 35, tzinfo=UTC)       # URNT12 KNHC 081235, fix at 11:56
    assert v.message_time == vdm_at and d.message_time == dt.datetime(2026, 10, 8, 12, 24, tzinfo=UTC)
    assert r.start <= min(b.start, v.time, d.release_time) and r.end >= max(batch_at, vdm_at)
    assert r.counts(batch_at - dt.timedelta(seconds=1))[0] == 0 and r.counts(batch_at)[0] == 1
    assert r.counts(vdm_at - dt.timedelta(seconds=1))[1] == 0 and r.counts(vdm_at)[1] == 1
    assert r.counts(d.message_time)[2] == 1 and r.counts(d.message_time - dt.timedelta(seconds=1))[2] == 0
    # best track: only points valid by then
    t = dt.datetime(2026, 10, 8, 12, tzinfo=UTC)
    assert all(p.time <= t for p in r.track_at(t)) and len(r.track_at(t)) < len(c.track)
    # TDR: the grid and its measurements appear when HRD's job made them, not at the analysis time
    avail = dt.datetime(2026, 9, 25, 22, 31, tzinfo=UTC)
    assert r.tdr_at(avail - dt.timedelta(minutes=1)) == [] and r.tdr_at(avail) == [TDR_URL]
    assert r.evolution_at(avail - dt.timedelta(minutes=1)) == [] and len(r.evolution_at(avail)) == 1
    # without a recorded time, the evolution row arrives 30 minutes after its analysis ended
    c.evolution[0]["available"] = None
    end = dt.datetime(2026, 9, 25, 22, 19, tzinfo=UTC)
    assert r.evolution_at(end + dt.timedelta(minutes=29)) == [] and r.evolution_at(end + dt.timedelta(minutes=30))


def test_issue_time_from_the_header():
    after = dt.datetime(2026, 9, 30, 23, 58, tzinfo=UTC)
    assert case_mod._issued("URNT12 KNHC 010003\nVORTEX DATA MESSAGE", after) == \
        dt.datetime(2026, 10, 1, 0, 3, tzinfo=UTC)                     # across the month boundary
    assert case_mod._issued("URNT15 KNHC 302355\n", after) == after   # never before the message's own data
    assert case_mod._issued("no header here", after) is None
    assert case_mod._issued("URNT15 KNHC 151200\n", after) is None    # two weeks off: not this message's


def test_tdr_available_time_from_job_files():
    url = tg.url_for("261008I1_1602_xy.nc.gz")
    jobs = ["20261008162748_20261008I1_160258_jobfile.tar.gz", "20261008172001_20261008I1_170200_jobfile.tar.gz"]
    assert case_mod.tdr_available_time(url, None, jobs) == dt.datetime(2026, 10, 8, 16, 27, 48, tzinfo=UTC)
    end = dt.datetime(2026, 10, 8, 16, 14, tzinfo=UTC)
    assert case_mod.tdr_available_time(url, end, []) == end + dt.timedelta(minutes=30)


def test_newer_case_is_refused(tmp_path):
    path = tmp_path / "future.r10case"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("case.json", json.dumps({"recon10s_case": case_mod.CASE_VERSION + 1}))
    with pytest.raises(ValueError, match="newer recon10s"):
        case_mod.load_case(str(path))


def test_nothing_private_is_saved(saved):
    """No credentials or settings, and nothing from the restricted IWG1 directory."""
    man, _inputs, _nc = saved
    with zipfile.ZipFile(man["path"]) as z:
        blob = b"".join(z.read(n) for n in z.namelist() if not n.endswith(".gz")).lower()
    for word in (b"jsimpson", b"password", b"aamps_ingest", b"settings"):
        assert word not in blob
