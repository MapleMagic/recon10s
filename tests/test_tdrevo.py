"""TDR evolution: one pass's numbers, the disk cache, ordering, SHIPS per cycle. Offline."""
import dataclasses
import datetime as dt

import numpy as np
import pytest

import recon10s_tdr3d as t3
import recon10s_tdrevo as evo
import recon10s_tdrgrid as tg
from test_tdr3d import make_storm

UTC = dt.timezone.utc
KT = tg.MS_TO_KT


def storm_pass(when: dt.datetime, **kw) -> tg.TdrGrid:
    """make_storm's tilted, sheared vortex with its tangential wind filled in, at a given time."""
    g = make_storm(**kw)
    X, Y = np.meshgrid(g.x, g.y)
    th = np.arctan2(Y, X)
    g.fields["TANGENTIAL_WIND"] = -g.fields["U"] * np.sin(th) + g.fields["V"] * np.cos(th)
    g.start, g.end = when - dt.timedelta(minutes=12), when + dt.timedelta(minutes=12)
    g.centre_time = when
    g.flight, g.name = "20261008I1", f"261008I1_{when:%H%M}_xy.nc.gz"
    return g


T1 = dt.datetime(2026, 10, 8, 16, 2, tzinfo=UTC)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(evo, "CACHE_DIR", str(tmp_path / "evolution"))
    return tmp_path


def test_measure_one_pass():
    g = storm_pass(T1, shear_heading=250.0, shear_ms=8.0, tilt_km=6.0, tilt_heading=45.0)
    m = evo.measure(g)
    assert m.time == T1.isoformat() and m.mission == "20261008I1" and m.centre_lat == pytest.approx(17.0)
    # 60 m/s at the RMW plus the environmental wind at 2 km (none: it starts there)
    assert m.peak_2km_kt == pytest.approx(60 * KT, abs=3)
    assert m.peak_low_kt >= m.peak_2km_kt and 0.0 <= m.peak_low_z <= 5.0
    assert m.rmw_2km == pytest.approx(20.0, abs=3.0)
    assert m.vt_max_2km_kt == pytest.approx(60 * KT, abs=8)         # a ring mean, a little off-centre
    # the column drifts 6 km over 12 km toward 045: 3 km between 2 and 8 km
    assert m.tilt_km == pytest.approx(3.0, abs=1.0) and m.tilt_heading == pytest.approx(45.0, abs=20.0)
    assert m.shear_kt == pytest.approx(8.0 * KT, abs=3.0) and m.shear_heading == pytest.approx(250.0, abs=10.0)
    assert m.shear_grade and m.column_levels > 10 and not m.error


def test_metrics_cached_and_ordered(cache):
    grids = {f"261008I1_{t:%H%M}_xy.nc.gz": storm_pass(t) for t in (T1, T1 + dt.timedelta(hours=1))}
    calls = []

    def loader(url):
        calls.append(url)
        return grids[url.rsplit("/", 1)[-1]]
    avail = T1 + dt.timedelta(minutes=26)
    later = tg.Analysis("20261008I1", "1702", "261008I1_1702_xy.nc.gz")
    first = tg.Analysis("20261008I1", "1602", "261008I1_1602_xy.nc.gz", available=avail)
    ms = evo.evolution([later, first], loader=loader)
    assert [m.time for m in ms] == [T1.isoformat(), (T1 + dt.timedelta(hours=1)).isoformat()]   # oldest first
    assert ms[0].available == avail.isoformat() and ms[1].available is None
    assert len(calls) == 2
    again = evo.evolution([later, first], loader=loader)
    assert len(calls) == 2                                       # measured once, read back from disk
    assert [dataclasses.asdict(m) for m in again] == [dataclasses.asdict(m) for m in ms]
    # a different shear layer is a different measurement
    evo.evolution([first], z_lo=1.0, loader=loader)
    assert len(calls) == 3


def test_unreadable_pass_is_reported_not_cached(cache):
    a = tg.Analysis("20261008I1", "1602", "261008I1_1602_xy.nc.gz")

    def broken(url):
        raise OSError("truncated download")
    m = evo.pass_metrics(a, loader=broken)
    assert "truncated download" in m.error and m.mission == "20261008I1"
    assert evo.cached(a.url(a.xy), 2.0, 8.0, 120.0) is None


def test_ships_fetched_once_per_cycle(monkeypatch):
    asked = []

    def fake(storm_id, when):
        asked.append(when)
        return t3.ShipsShear(cycle=dt.datetime(2026, 10, 8, 12, tzinfo=UTC), kt=14.0, heading=70.0, url="lsdiag")
    monkeypatch.setattr(t3, "ships_shear", fake)
    ms = [evo.PassMetrics(mission="m", name=f"n{k}", time=(T1 + dt.timedelta(minutes=50 * k)).isoformat())
          for k in range(3)]                                    # 16:02, 16:52, 17:42: one 12Z cycle
    evo.add_ships(ms, "al092026")
    assert len(asked) == 1 and all(m.ships_kt == 14.0 and m.ships_heading == 70.0 for m in ms)


def test_summary_line():
    a = evo.PassMetrics(mission="m", name="a", time=T1.isoformat(), peak_2km_kt=48, rmw_2km=40, tilt_km=16,
                        shear_kt=10)
    b = evo.PassMetrics(mission="m", name="b", time=(T1 + dt.timedelta(hours=26)).isoformat(), peak_2km_kt=85,
                        rmw_2km=24, tilt_km=26, shear_kt=18)
    text = evo.summary([a, b])
    assert text.startswith("Over 26 h and 2 passes") and "48 → 85 kt" in text and "RMW 40 → 24 km" in text
    assert evo.summary([a]) == ""
