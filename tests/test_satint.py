"""
Satellite intensity (recon10s_satint): the numpy network against a plain convolution, the IR and
microwave gridding against analytic fields (including a GOES fixed-grid image built here), the
centre from best track, advisory and recon fixes, and the shipped model.
"""
import datetime as dt
import json
import math
import types

import numpy as np
import pytest

import recon10s_satint as si

UTC = dt.timezone.utc


def naive_conv(x, w, b, stride):
    B, C, H, W = x.shape
    O = w.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    Ho, Wo = (H - 1) // stride + 1, (W - 1) // stride + 1
    out = np.zeros((B, O, Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            patch = xp[:, :, i * stride:i * stride + 3, j * stride:j * stride + 3]
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


@pytest.mark.parametrize("stride", [1, 2])
def test_convolution_matches_a_plain_one(stride):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 9, 8)).astype(np.float32)
    w = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    b = rng.normal(size=4).astype(np.float32)
    assert np.allclose(si._conv3x3(x, w, b, stride), naive_conv(x, w, b, stride), atol=1e-4)


def tiny_model(path):
    """A random network in the app's format, with the shipped layer layout but few channels."""
    rng = np.random.default_rng(1)
    arrays, spec = {}, {"quantiles": [0.1, 0.25, 0.5, 0.75, 0.9], "norm": {"vmax": [65.0, 35.0], "mslp": [985.0, 25.0]}}
    for name, cin, strides, size in (("ir", 5, [2, 2, 1, 2, 2, 2], 128), ("mw", 6, [2, 2, 2, 2], 64)):
        layers, c = [], cin
        for i, s in enumerate(strides):
            arrays[f"{name}{i}_w"] = rng.normal(0, 0.3, (4, c, 3, 3)).astype(np.float32)
            arrays[f"{name}{i}_b"] = np.zeros(4, np.float32)
            layers.append({"stride": s, "out": 4})
            c = 4
            size = (size - 1) // s + 1
        spec[name] = layers
        arrays[f"{name}_fc_w"] = rng.normal(0, 0.1, (8, 4 * size * size)).astype(np.float32)
        arrays[f"{name}_fc_b"] = np.zeros(8, np.float32)
    arrays["head0_w"] = rng.normal(0, 0.1, (12, 8 + 8 + 13)).astype(np.float32)
    arrays["head0_b"] = np.zeros(12, np.float32)
    arrays["head1_w"] = rng.normal(0, 0.1, (10, 12)).astype(np.float32)
    arrays["head1_b"] = np.zeros(10, np.float32)
    spec["head"] = 2
    np.savez(path, spec=np.array(json.dumps(spec)), info=np.array(json.dumps({"version": 0})), **arrays)
    return str(path)


def ring_storm(lat0=20.0, lon0=-60.0, eye_km=15.0):
    """IR on a 0.03-degree lat/lon grid: a warm eye, a cold ring, warming outward."""
    lat_v = np.arange(lat0 - 4, lat0 + 4, 0.03)
    lon_v = np.arange(lon0 - 4, lon0 + 4, 0.03)
    LA, LO = np.meshgrid(lat_v, lon_v, indexing="ij")
    x, y = si.local_xy(lat0, lon0, LA, LO)
    r = np.hypot(x, y)
    tb = np.where(r < eye_km, 290.0, 195.0 + 0.25 * np.clip(r - eye_km, 0, None))
    return lat_v, lon_v, np.minimum(tb, 295.0)


def test_ir_grid_is_centred():
    lat_v, lon_v, tb = ring_storm()
    g = si.ir_from_lonlat_grid(tb, lat_v, lon_v, (20.0, -60.0))
    assert g.shape == (si.IR_N, si.IR_N) and np.isfinite(g).all()
    c = si.IR_N // 2
    assert g[c - 1:c + 1, c - 1:c + 1].mean() > 280                      # the eye in the middle
    ring = si.grid(si.IR_N, si.IR_DX)
    X, Y = np.meshgrid(ring, ring)
    r = np.hypot(X, Y)
    assert abs(g[(r > 40) & (r < 50)].std()) < 1.5                       # round: same Tb all around
    # descending latitudes (as some files store them) give the same grid
    g2 = si.ir_from_lonlat_grid(tb[::-1], lat_v[::-1], lon_v, (20.0, -60.0))
    assert np.allclose(g, g2, atol=1e-6)


def test_ir_from_a_goes_fixed_grid_image():
    """A band-13 cut in the geostationary projection, as recon10s_goes loads it, back to the storm grid."""
    from pyproj import CRS, Transformer
    h, lon_sat = 35786023.0, -75.2
    crs = CRS.from_proj4(f"+proj=geos +h={h} +lon_0={lon_sat} +sweep=x +a=6378137 +b=6356752.31414 +units=m +no_defs")
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lat0, lon0 = 22.0, -70.0
    cx, cy = fwd.transform(lon0, lat0)
    dx = 2000.0                                   # ~2 km pixels
    n = 400
    xs = cx + (np.arange(n) - n / 2 + 0.5) * dx
    ys = cy - (np.arange(n) - n / 2 + 0.5) * dx   # rows run north to south
    XM, YM = np.meshgrid(xs, ys)
    lo, la = inv.transform(XM, YM)
    x, y = si.local_xy(lat0, lon0, la, lo)
    r = np.hypot(x, y)
    tb_k = np.where(r < 12, 288.0, 196.0 + 0.3 * np.clip(r - 12, 0, None))
    img = types.SimpleNamespace(data=tb_k - 273.15, sat_height=h, lon0=lon_sat, sweep="x",
                                extent_m=(xs[0] - dx / 2, xs[-1] + dx / 2, ys[-1] - dx / 2, ys[0] + dx / 2))
    g = si.ir_from_goes(img, (lat0, lon0))
    ring = si.grid(si.IR_N, si.IR_DX)
    X, Y = np.meshgrid(ring, ring)
    rr = np.hypot(X, Y)
    truth = np.where(rr < 12, 288.0, 196.0 + 0.3 * np.clip(rr - 12, 0, None))
    away = (rr > 30) & (rr < 240)
    assert np.isfinite(g).all() and np.abs(g[away] - truth[away]).mean() < 1.0
    c = si.IR_N // 2
    assert g[c - 1:c + 1, c - 1:c + 1].mean() > 260                      # the eye lands in the middle


def test_microwave_gridding_and_coverage():
    rng = np.random.default_rng(3)
    lat0, lon0 = 18.0, -100.0
    # footprints every ~5 km over the west half of the storm only (the swath edge cuts it)
    xs = rng.uniform(-300, 0, 12000)
    ys = rng.uniform(-300, 300, 12000)
    lat, lon = si.destinations(lat0, lon0, xs, ys)
    r = np.hypot(xs, ys)
    v89 = np.where(r < 20, 270.0, 200.0 + 0.3 * r)
    bands = {"89": (lat, lon, v89, v89 - 10), "37": (lat, lon, v89 + 20, v89 + 5)}
    g, cover = si.mw_grid(bands, (lat0, lon0))
    assert g.shape == (4, si.MW_N, si.MW_N)
    assert 0.4 < cover < 0.6                                              # half the inner 200 km
    gx = si.grid(si.MW_N, si.MW_DX)
    X, Y = np.meshgrid(gx, gx)
    rr = np.hypot(X, Y)
    west = (X < -20) & (rr > 40) & (rr < 200)
    assert np.nanmean(np.abs(g[2][west] - (200.0 + 0.3 * rr[west]))) < 2.5
    assert np.isnan(g[2][(X > 30)]).all()                                 # nothing where the swath was not


def _track(t0):
    import recon10s_nhc as nhc
    return [nhc.TrackPoint(time=t0 + dt.timedelta(hours=6 * k), lat=20.0 + 0.5 * k, lon=-60.0 - 1.0 * k, vmax=60 + 5 * k)
            for k in range(4)]


def test_centre_from_track_advisory_and_fixes():
    import recon10s_analysis as an
    import recon10s_nhc as nhc
    t0 = dt.datetime(2026, 9, 1, 0, tzinfo=UTC)
    src = si.TrackSource(track=_track(t0))
    la, lo = src.centre_at(t0 + dt.timedelta(hours=3))
    assert la == pytest.approx(20.25) and lo == pytest.approx(-60.5)
    # past the last point it is carried along the motion from the last two positions
    la2, lo2 = src.centre_at(t0 + dt.timedelta(hours=21))
    assert la2 > 21.5 and lo2 < -63.0
    assert src.centre_at(t0 + dt.timedelta(hours=40)) is None             # too far beyond the track
    adv = nhc.Fix(storm_id="al052026", name="TEST", lat=21.9, lon=-63.4, time=t0 + dt.timedelta(hours=21),
                  classification="HU", vmax_kt=80, mslp_mb=970)
    src2 = si.TrackSource(track=_track(t0), advisory=adv)
    assert src2.centre_at(t0 + dt.timedelta(hours=21)) == pytest.approx((21.9, -63.4))
    fix = an.CentreFix(time=t0 + dt.timedelta(hours=12), lat=21.3, lon=-62.1, source="VDM", aircraft="AF301")
    src3 = si.TrackSource(track=_track(t0), fixes=[fix])
    la3, lo3 = src3.centre_at(t0 + dt.timedelta(hours=12))
    assert la3 == pytest.approx(21.3, abs=0.05) and lo3 == pytest.approx(-62.1, abs=0.05)


def test_network_inputs_and_estimate(tmp_path):
    path = tiny_model(tmp_path / "tiny.npz")
    lat_v, lon_v, tb = ring_storm()
    cur = si.ir_from_lonlat_grid(tb, lat_v, lon_v, (20.0, -60.0))
    t = dt.datetime(2026, 9, 1, 18, tzinfo=UTC)
    ir, mw, sc = si.network_inputs([cur, None, cur, None, cur], None, None, 0.0, 6.0, 0.0, 20.0, 12.0, True, t, -60.0)
    assert ir.shape == (1, 5, 128, 128) and np.allclose(ir[0, 1], ir[0, 0])   # a missing frame repeats the current
    assert list(sc[0, 9:13]) == [0.0, 1.0, 0.0, 1.0] and sc[0, 0] == 0.0 and sc[0, 6] == 1.0
    assert not mw.any()
    mwg = np.full((4, si.MW_N, si.MW_N), 250.0)
    e = si.estimate_from_grids([cur] * 5, t, (20.0, -60.0), 12.0, False, mw=mwg, mw_ir=cur,
                               mw_time=t - dt.timedelta(hours=2), mw_sensor="GMI", mw_cover=0.9, satellite="GOES-19",
                               model_path=path)
    assert e.mw_sensor == "GMI" and e.mw_age_h == pytest.approx(2.0) and e.ir_frames == 5
    assert e.ir_vmax is not None and e.ir_mslp is not None              # what the IR alone gives, beside it
    assert list(e.vmax_q) == sorted(e.vmax_q) and list(e.mslp_q) == sorted(e.mslp_q)
    old = si.estimate_from_grids([cur] * 5, t, (20.0, -60.0), 12.0, False, mw=mwg, mw_ir=cur,
                                 mw_time=t - dt.timedelta(hours=13), mw_sensor="GMI", mw_cover=0.9, model_path=path)
    assert old.mw_time is None and "IR only" in old.notes                 # a pass older than 12 h is not used
    assert old.ir_vmax is None and old.vmax == pytest.approx(e.ir_vmax, abs=1e-6)
    back = si.Estimate.from_dict(json.loads(json.dumps(e.to_dict())))
    assert back.time == e.time and back.vmax_q == pytest.approx(e.vmax_q, abs=0.01) and back.mw_time == e.mw_time
    assert back.ir_vmax == pytest.approx(e.ir_vmax, abs=0.01)
    assert "kt" in e.label and "GMI" in e.label
    with pytest.raises(ValueError):
        si.estimate_from_grids([np.full_like(cur, np.nan)] * 5, t, (20.0, -60.0), 12.0, False, model_path=path)


def test_passes_fetches_only_what_the_estimates_use(monkeypatch):
    """Of the passes before the first estimate only the newest is downloaded; each is gridded."""
    import recon10s_mw as mwmod
    import recon10s_nhc as nhc
    t0 = dt.datetime(2026, 9, 1, 12, tzinfo=UTC)
    track = [nhc.TrackPoint(time=t0 + dt.timedelta(hours=6 * k), lat=18.0 + 0.3 * k, lon=-60.0 - 0.5 * k)
             for k in range(-5, 2)]
    ages = (20, 15, 11, 9, 2)
    cands = [types.SimpleNamespace(granule=types.SimpleNamespace(sensor="GMI", start=t0 - dt.timedelta(hours=h)),
                                   closest_time=t0 - dt.timedelta(hours=h), label=f"GMI -{h} h") for h in ages]
    monkeypatch.setattr(mwmod, "find_passes", lambda *a, **k: list(reversed(cands)))
    fetched = []

    def download(granule, email, *a, **k):
        fetched.append(granule.start)
        return "granule.h5"

    def load(path, sensor, centre, box_deg=4.0, granule=None):
        g = np.arange(-300, 301, 5.0)
        X, Y = np.meshgrid(g, g)
        lat, lon = si.destinations(centre[0], centre[1], X.ravel(), Y.ravel())
        tb = 200.0 + 0.2 * np.hypot(X, Y).ravel()
        band = types.SimpleNamespace(lat=lat, lon=lon, v=tb, h=tb - 8)
        return types.SimpleNamespace(bands={"37": band, "89": band})
    monkeypatch.setattr(mwmod, "download_granule", download)
    monkeypatch.setattr(mwmod, "load_pass", load)
    est = si.Estimator("al052026", si.TrackSource(track=track), email="someone@example.com")
    got = est.passes(t0 - dt.timedelta(hours=22), t0, needed_from=t0 - dt.timedelta(hours=10))
    assert fetched == [t0 - dt.timedelta(hours=h) for h in (11, 9, 2)]
    assert [m.time for m in got] == fetched and all(m.cover > 0.99 for m in got)
    assert est.latest_pass(t0, got).time == t0 - dt.timedelta(hours=2)
    assert est.latest_pass(t0 - dt.timedelta(hours=10), got).time == t0 - dt.timedelta(hours=11)
    # without a sign-in there is nothing to fetch
    assert si.Estimator("al052026", si.TrackSource(track=track)).passes(t0 - dt.timedelta(hours=12), t0) == []


def test_rotation_averaging(tmp_path):
    n = si.Net(tiny_model(tmp_path / "tiny.npz"))
    rng = np.random.default_rng(5)
    ir = rng.normal(size=(2, 5, 128, 128)).astype(np.float32)
    mw = rng.normal(size=(2, 6, 64, 64)).astype(np.float32)
    sc = rng.normal(size=(2, 13)).astype(np.float32)
    v1, p1 = n.predict(ir, mw, sc, 1)
    v0, p0 = n.forward(ir, mw, sc)
    assert np.allclose(v1, v0) and np.allclose(p1, p0)
    v4, p4 = n.predict(ir, mw, sc, 4)
    turned = [n.forward(np.ascontiguousarray(np.rot90(ir, r, axes=(2, 3))),
                        np.ascontiguousarray(np.rot90(mw, r, axes=(2, 3))), sc)[0] for r in range(4)]
    assert np.allclose(v4, np.mean(turned, 0), atol=1e-4)
    assert (np.diff(v4, axis=1) >= 0).all() and (np.diff(p4, axis=1) >= 0).all()
    # a field the same under every turn gives the same answer every way
    g = si.grid(si.IR_N, si.IR_DX)
    X, Y = np.meshgrid(g, g)
    sym = np.broadcast_to(np.hypot(X, Y)[None, None] / 100.0, (1, 5, 128, 128)).astype(np.float32)
    zero = np.zeros((1, 6, 64, 64), np.float32)
    assert np.allclose(n.predict(sym, zero, sc[:1], 4)[0], n.forward(sym, zero, sc[:1])[0], atol=1e-3)


def test_a_file_of_several_networks_averages_them(tmp_path):
    """Members (m0_, m1_, ... arrays) in one file: each estimate is their mean."""
    one = np.load(tiny_model(tmp_path / "a.npz"))
    spec = json.loads(str(one["spec"]))
    arrays = {}
    for k, scale in enumerate((1.0, 0.5)):
        for name in one.files:
            if name not in ("spec", "info"):
                arrays[f"m{k}_{name}"] = one[name] * (scale if name.endswith("_w") else 1.0)
    ens = {"quantiles": spec["quantiles"], "norm": spec["norm"],
           "members": [{"ir": spec["ir"], "mw": spec["mw"], "head": spec["head"]}] * 2}
    np.savez(tmp_path / "ens.npz", spec=np.array(json.dumps(ens)), info=np.array("{}"), **arrays)
    half = {k: (one[k] * 0.5 if k.endswith("_w") else one[k]) for k in one.files if k not in ("spec", "info")}
    np.savez(tmp_path / "half.npz", spec=one["spec"], info=one["info"], **half)
    rng = np.random.default_rng(2)
    x = (rng.normal(size=(2, 5, 128, 128)).astype(np.float32), rng.normal(size=(2, 6, 64, 64)).astype(np.float32),
         rng.normal(size=(2, 13)).astype(np.float32))
    e, a, b = si.Net(str(tmp_path / "ens.npz")), si.Net(str(tmp_path / "a.npz")), si.Net(str(tmp_path / "half.npz"))
    assert len(e.members) == 2 and len(a.members) == 1
    ve, pe = e.forward(*x)
    assert np.allclose(ve, (a.forward(*x)[0] + b.forward(*x)[0]) / 2, atol=1e-4)
    assert np.allclose(pe, (a.forward(*x)[1] + b.forward(*x)[1]) / 2, atol=1e-4)


def test_the_shipped_model():
    """The model in scripts/models loads, says what it learnt from, and ranks an eye above clear air."""
    n = si.Net()
    assert n.info.get("version") == 1 and n.info.get("samples", 0) > 20000
    assert "TC PRIMED" in " ".join(n.info.get("sources", []))
    assert si.self_check().startswith("satellite intensity ok")
