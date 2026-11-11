"""
Synthetic TDR (recon10s_syntdr): the reconstruction on a tiny hand-made model,
built end to end from a synthetic flight through a moving vortex; the live
calibration; the pass splitting, coverage and comparison helpers; and the
shipped model file itself.
"""
import datetime as dt
import json
import math
import types

import numpy as np
import pytest

import recon10s_syntdr as st
import recon10s_tdrgrid as tg
import test_windfield as twf

UTC = dt.timezone.utc
RHO = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
PHI = np.arange(8) * 45.0
Z = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
CAL = {"hinge_kt": 96.0, "full": [0.4, 0.4, 14.0, 0.26], "pk_only": [0.75, 17.0, 0.37]}


def decay(z):
    return np.clip(1.0 - 0.04 * np.clip(z - 2.0, 0.0, None), 0.3, 1.0)


def make_model(path) -> str:
    """A Rankine-like mean (Vt/Vm = rho inside, rho^-0.6 outside, weakening above 2 km) and three modes."""
    nz, nr, npf = Z.size, RHO.size, PHI.size
    prof = np.where(RHO <= 1.0, RHO, np.maximum(RHO, 1.0) ** -0.6)
    ones = np.ones((nz, nr, npf))
    vt = decay(Z)[:, None, None] * prof[None, :, None] * ones
    mean = np.stack([vt, np.zeros_like(vt)]).reshape(-1)
    B = np.zeros((10, mean.size))
    B[0] = mean
    bell = np.exp(-((RHO - 1.0) / 0.7) ** 2)[None, :, None]
    zero = np.zeros((nz, nr, npf))
    e1 = np.stack([bell * np.cos(np.radians(PHI))[None, None, :] * ones, zero]).reshape(-1)
    e2 = np.stack([bell * np.sin(np.radians(PHI))[None, None, :] * ones, zero]).reshape(-1)
    e3 = np.stack([zero, -(Z <= 1.5).astype(float)[:, None, None] * bell * ones]).reshape(-1)
    hyper = {"kappa": 1.0, "gamma": 1.0, "sigma_fl": 2.0, "sigma_drop": 3.0, "R_prior": [math.log(35.0), 0.7],
             "R_obs_sd": 0.3, "surface_level_km": 1.0, "surface_ratio": 0.95, "intensity": CAL,
             "intensity_rmse_kt": 7.7, "spread_by_rho": [[0.5, 2.0], [2.0, 1.0]]}
    np.savez_compressed(path, rho=RHO, phi=PHI, z=Z, B=B, E=np.c_[e1, e2, e3], lam=np.array([0.01, 0.01, 0.01]),
                        s2=np.full(mean.size, 1e-4), hyper=np.array(json.dumps(hyper)),
                        info=np.array(json.dumps({"version": 1, "n_analyses": 3})))
    return str(path)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_model(tmp_path_factory.mktemp("syntdr") / "tiny.npz")


def legs(m, vm=50.0, R=30.0, z=3.0, headings=(0.0, 45.0, 90.0, 135.0), half_km=150.0):
    """Obs along straight legs through a vortex drawn from the model's own mean."""
    k = m.E.shape[1]
    cov = dict(s_par=0.0, s_perp=0.0, speed=5.0, lat=20.0)
    truth = st.Fit(R=R, vm=vm, b=np.r_[vm, np.zeros(k)], P=np.zeros((k + 1, k + 1)),
                   mu=st.cov_vector(vm, R, **cov) @ m.B, log_evidence=0.0, heading=0.0, n_obs=0, resid_rms=0.0)
    s = m.state(truth)
    d = np.arange(-half_km, half_km + 1, 3.0)
    x = np.concatenate([d * math.sin(math.radians(h)) for h in headings])
    y = np.concatenate([d * math.cos(math.radians(h)) for h in headings])
    vt, vr = m.sample_many(truth, [(s, 0), (s, 1)], x, y, np.full(x.size, z))
    th = np.arctan2(y, x)
    u, v = vr * np.cos(th) - vt * np.sin(th), vr * np.sin(th) + vt * np.cos(th)
    n = x.size
    t = np.arange(n) * 30.0
    return st.Obs(x, y, np.full(n, z), u, v, np.full(n, 2.0), np.full(n, st.KIND_FL), t, np.zeros(n, int)), cov


def test_cov_vector():
    c = st.cov_vector(60.0, 40.0, s_par=20.0, s_perp=-20.0, speed=8.0, lat=33.0)
    assert c.shape == (10,) and c[0] == 1.0
    assert c[1] == pytest.approx(1.0) and c[2] == pytest.approx(0.0) and c[3] == 1.0 and c[4] == -1.0
    assert c[7] == pytest.approx(c[1] * c[3]) and c[9] == pytest.approx(c[1] * c[2])


def test_fit_recovers_the_vortex(tiny):
    m = st.Model(tiny)
    obs, cov = legs(m, vm=50.0, R=30.0)
    f = m.fit(obs, 0.0, cov, R_obs=st.observed_rmw(obs))
    assert f.vm == pytest.approx(50.0, rel=0.05)
    assert f.R == pytest.approx(30.0, rel=0.12)
    assert f.resid_rms < 1.5
    # the uncertainty profile multiplies the spread by r / R: twice at the centre, once beyond 2 R
    sp = m.spread(f)
    assert np.all(np.isfinite(sp)) and sp[0, 2, 0, 0] > sp[0, 2, -1, 0]


def test_intensity_calibration_formula(tiny):
    m = st.Model(tiny)
    assert m.intensity(80.0, 90.0) == pytest.approx(0.4 * 80 + 0.4 * 90 + 14.0)
    assert m.intensity(120.0, 130.0) == pytest.approx(0.4 * 120 + 0.4 * 130 + 14.0 + 0.26 * 24)
    assert m.intensity(80.0, None) == pytest.approx(0.75 * 80 + 17.0)


def _bundle():
    ov, fixes, _truth = twf.fly(headings=(45.0, 135.0))
    return types.SimpleNamespace(overlay=ov, fixes=fixes, drops=[], passes=[], motion=None, fix=None)


def test_synthesize_from_a_flight(tiny):
    b = _bundle()
    when = b.fixes[0].time + dt.timedelta(minutes=45)
    res = st.synthesize(b, when, half_h=2.0, motion=twf.MOTION, shear=(15.0, 90.0), storm="TEST", model_path=tiny,
                        half_km=150.0)
    g = res.grid
    assert g.source == "synthetic" and g.flight == "SYNTHETIC"
    assert {"WIND_SPEED", "TANGENTIAL_WIND", "RADIAL_WIND", "U", "V", "VORT", "WIND_SPREAD"} <= set(g.fields)
    assert g.has_winds() and g.fields["U"].shape == (Z.size, g.y.size, g.x.size)
    # the flown vortex: 100 kt at 3 km with a 30-km RMW
    assert res.fit.R == pytest.approx(twf.RM, rel=0.25)
    assert res.fit.vm * st.MS_TO_KT == pytest.approx(twf.VM / decay(3.0), rel=0.15)
    assert res.core_sampled and res.n_fl > 100 and res.n_drop == 0
    assert res.fl_max_kt == pytest.approx(twf.VM + twf.MOTION[1], abs=3)
    # the intensity is the calibrated one, and the surface map peaks at it
    m = st.model(tiny)
    assert res.intensity_kt == pytest.approx(m.intensity(res.peak_1km_kt, res.fl_max_kt), abs=0.6)
    assert float(np.nanmax(res.surface)) == pytest.approx(res.intensity_kt, abs=0.01)
    assert res.peak_where.endswith("of centre") and res.ob_distance.shape == res.surface.shape
    assert res.shear == (15.0, 90.0) and res.rmw_km == pytest.approx(twf.RM, rel=0.3)
    # vorticity is positive (cyclonic) inside the RMW
    k = g.level(2.0)
    c = g.y.size // 2
    assert np.nanmean(g.fields["VORT"][k, c - 3:c + 4, c - 3:c + 4]) > 0
    # each leg left out in turn
    loo = st.leave_one_out(res, model_path=tiny)
    assert loo is not None and len(loo["passes"]) == 2 and loo["mae"] < 25
    # the analysis against new obs, and against a "real" TDR grid identical to it
    v = st.verify_against(res, res.obs)
    assert v is None or v["n"] > 0
    real = tg.TdrGrid(name="261008I1_1200_xy.nc.gz", storm="TEST", flight="261008I1", x=g.x, y=g.y, z=g.z,
                      lat=g.lat, lon=g.lon, fields=dict(g.fields), centre=g.centre, start=g.start, end=g.end,
                      centre_time=g.centre_time, motion=g.motion)
    cmp_ = st.compare_tdr(res, real, 2.0)
    assert cmp_["rmse"] < 0.5 and abs(cmp_["bias"]) < 0.1 and cmp_["n"] > 1000


def test_synthesize_needs_motion_fixes_and_obs(tiny):
    b = _bundle()
    when = b.fixes[0].time
    with pytest.raises(ValueError, match="motion"):
        st.synthesize(b, when, motion=None, model_path=tiny)
    with pytest.raises(ValueError, match="fixes"):
        st.synthesize(types.SimpleNamespace(overlay=b.overlay, fixes=[], drops=[], passes=[]), when,
                      motion=twf.MOTION, model_path=tiny)
    with pytest.raises(ValueError, match="observations"):
        st.synthesize(b, when + dt.timedelta(hours=12), half_h=0.5, motion=twf.MOTION, model_path=tiny)


def test_split_passes_at_the_outer_turns():
    """A continuous figure-four (no time gaps): one pass per centre crossing."""
    xs, ys = [], []
    for k in range(3):
        a = math.radians(45.0 + 90.0 * k)
        d = np.arange(-150.0, 151.0, 4.0)
        xs.append(d * math.sin(a)); ys.append(d * math.cos(a))
        # the crosswind leg from the end of this pass to the start of the next
        p0 = 150.0 * np.array([math.sin(a), math.cos(a)])
        p1 = -150.0 * np.array([math.sin(a + math.pi / 2), math.cos(a + math.pi / 2)])
        f = np.linspace(0.0, 1.0, 30)[1:-1]
        xs.append(p0[0] + f * (p1[0] - p0[0])); ys.append(p0[1] + f * (p1[1] - p0[1]))
    x, y = np.concatenate(xs), np.concatenate(ys)
    n = x.size
    obs = st.Obs(x, y, np.full(n, 3.0), np.zeros(n), np.zeros(n), np.full(n, 2.0), np.full(n, st.KIND_FL),
                 np.arange(n) * 30.0, np.zeros(n, int))
    passes = st.split_passes(obs)
    assert sum(p.size for p in passes) == n

    def crossings(p):
        inner = np.hypot(x[np.sort(p)], y[np.sort(p)]) < 10.0
        return int(np.sum(np.diff(np.r_[0, inner.astype(int)]) == 1))
    assert [crossings(p) for p in passes].count(1) == 3 and max(crossings(p) for p in passes) == 1


def test_core_coverage_and_distance():
    far = st.Obs(np.array([90.0, -90.0, 0.0]), np.array([0.0, 0.0, 95.0]), np.ones(3), np.zeros(3), np.zeros(3),
                 np.ones(3), np.zeros(3, int))
    assert st.core_coverage(far, 30.0)[2] is False
    near = st.Obs(np.array([-30.0, 0.0, 30.0, 0.0]), np.array([0.0, 30.0, 0.0, -2.0]), np.ones(4), np.zeros(4),
                  np.zeros(4), np.ones(4), np.zeros(4, int))
    rmin, sectors, ok = st.core_coverage(near, 30.0)
    assert ok and rmin == pytest.approx(2.0) and sectors >= 3
    d = st.nearest_ob_km(near, np.array([0.0, 40.0]), np.array([0.0, 30.0]))
    assert d.shape == (2, 2) and d[0, 0] == pytest.approx(2.0) and d[1, 1] == pytest.approx(math.hypot(10, 30))


def test_live_calibration(tmp_path):
    import recon10s_nhc as nhc
    path = str(tmp_path / "live.json")
    live = st.LiveCalibration(path)
    t0 = dt.datetime(2026, 10, 8, 12, tzinfo=UTC)
    live.add_estimate("al092026", t0 + dt.timedelta(minutes=10), 80.0, 0.0, 82.0, 85.0)
    live.add_estimate("al092026", t0 + dt.timedelta(minutes=20), 81.0, 0.0, 82.0, 85.0)   # same half hour: replaced
    live.add_estimate("al092026", t0 + dt.timedelta(hours=3), 90.0, 0.0, 92.0, 95.0)
    live.add_estimate("al092026", t0 + dt.timedelta(hours=7), 95.0, 0.0, 97.0, 99.0)      # beyond the track
    assert len(live.data["intensity"]) == 3 and live.offset() == (0.0, 0)
    track = [nhc.TrackPoint(time=t0, lat=23.0, lon=-90.0, vmax=85),
             nhc.TrackPoint(time=t0 + dt.timedelta(hours=6), lat=24.0, lon=-89.0, vmax=97)]
    assert live.match_track("al092026", track) == 2
    rows = live.matched()
    assert sorted(r["best_track"] for r in rows) == pytest.approx([85.0 + 12 * 20 / 360, 91.0])
    off, n = live.offset()
    errs = [r["best_track"] - r["synthetic"] for r in rows]
    assert n == 2 and off == pytest.approx(2 / 22 * np.mean(errs))
    live.add_pass("al092026", t0, {"n": 40, "mae": 6.0, "bias": -1.0, "rmse": 7.5})
    live.add_tdr("al092026", t0, {"analysis": "261008I1_1200_xy.nc.gz", "rmse": 9.0, "bias": 1.0, "level_km": 2.0})
    again = st.LiveCalibration(path)                     # it persists
    assert len(again.matched()) == 2 and len(again.data["passes"]) == 1 and len(again.data["tdr"]) == 1
    text = again.summary("al092026")
    assert "matched with NHC's best track" in text and "new pass" in text and "HRD's TDR" in text


def test_self_check(tiny):
    assert st.self_check(tiny).startswith("synthetic TDR ok")


def test_the_shipped_model():
    """The model in scripts/models loads, says what it was trained on, and recovers a vortex."""
    m = st.Model()
    assert m.info.get("version") == 1 and m.info.get("n_analyses", 0) > 1500
    assert m.intensity_full and len(m.intensity_full) == 4 and m.intensity_pk_only
    assert any("TC-RADAR" in s for s in m.info.get("sources", []))
    assert any("IWG1" in s for s in m.info.get("sources", []))
    assert m.E.shape[1] >= 20 and m.B.shape == (10, m.E.shape[0])
    assert st.self_check().startswith("synthetic TDR ok")
