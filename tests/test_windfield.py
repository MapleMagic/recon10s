"""Storm-relative wind analysis and wind radii, on a synthetic moving vortex flown by a synthetic aircraft."""
import datetime as dt
import math

import numpy as np
import pytest

import recon10s_analysis as an
import recon10s_satview as satview
import recon10s_windfield as wf

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 10, 8, 12, tzinfo=UTC)
VM, RM = 100.0, 30.0                 # kt, km
MOTION = (45.0, 10.0)                # toward NE at 10 kt


def vt(r):
    return np.where(r <= RM, VM * r / RM, VM * (RM / np.maximum(r, 1e-6)) ** 0.6)


def centre(t: dt.datetime):
    h = (t - T0).total_seconds() / 3600.0
    return wf._destination(25.0, -85.0, MOTION[0], h * MOTION[1] * 1.852)


def fly(headings=(45.0, 135.0), fix_times=(T0, T0 + dt.timedelta(hours=1, minutes=30)), half_km=180.0):
    """Straight legs through the moving centre along each heading, 30 s obs at 130 m/s."""
    ov = satview.Overlay()
    truth = []
    step_km = 0.13 * 30
    for hd, tf in zip(headings, fix_times):
        n = int(2 * half_km / step_km)
        for k in range(n + 1):
            s = -half_km + k * step_km                      # km along the leg, centre at s = 0
            t = tf + dt.timedelta(seconds=s / 0.13)
            x, y = s * math.sin(math.radians(hd)), s * math.cos(math.radians(hd))
            clat, clon = centre(t)
            r = math.hypot(x, y)
            brg = math.degrees(math.atan2(x, y))
            lat, lon = wf._destination(clat, clon, brg, r) if r > 0 else (clat, clon)
            th = math.atan2(y, x)
            v = float(vt(np.array(r)))
            u_ = -v * math.sin(th) + MOTION[1] * math.sin(math.radians(MOTION[0]))
            v_ = v * math.cos(th) + MOTION[1] * math.cos(math.radians(MOTION[0]))
            spd = math.hypot(u_, v_)
            wdir = (math.degrees(math.atan2(-u_, -v_)) + 360) % 360       # direction it blows from
            ov.lats.append(lat); ov.lons.append(lon); ov.times.append(t)
            ov.winds.append(int(round(spd))); ov.dirs.append(int(round(wdir)) % 360)
            ov.mslps.append(None); ov.segments.append(0); ov.heights.append(3000.0)
            ov.fl_press.append(700.0); ov.peaks.append(None); ov.wind_flagged.append(False)
            ov.labels.append("AF300 0109A")
            truth.append((x, y, spd))
    fixes = [an.CentreFix(time=tf, lat=centre(tf)[0], lon=centre(tf)[1], source="VDM", aircraft="AF300")
             for tf in fix_times]
    return satview.merge_overlays([ov]), fixes, np.array(truth)


@pytest.fixture(scope="module")
def flight():
    return fly()


def test_centre_track_through_fixes(flight):
    ov, fixes, _ = flight
    mid = T0 + dt.timedelta(minutes=45)
    la, lo = wf.centre_track(fixes, [mid, T0 + dt.timedelta(hours=3)], MOTION)
    want = centre(mid)
    assert float(an.gc_km(la[0], lo[0], *want)) < 0.5
    beyond = centre(T0 + dt.timedelta(hours=3))                 # 1.5 h past the last fix, along the motion
    assert float(an.gc_km(la[1], lo[1], *beyond)) < 1.0


def test_storm_relative_positions(flight):
    ov, fixes, truth = flight
    sr = wf.storm_relative(ov, fixes, MOTION, max_km=400)
    assert len(sr) == len(truth)
    assert np.max(np.hypot(sr.x - truth[:, 0], sr.y - truth[:, 1])) < 0.6
    # storm-relative winds have the motion taken out: a pure vortex again
    assert np.allclose(np.hypot(sr.us, sr.vs), vt(sr.r), atol=1.0)


def test_analysis_finds_the_rmw(flight):
    ov, fixes, _ = flight
    sr = wf.storm_relative(ov, fixes, MOTION)
    a = wf.analyse(sr, "fl", max_km=200, radial_km=5.0, arc_km=60.0)
    assert a.rmw() == pytest.approx(RM, abs=5.0)
    assert np.isfinite(a.speed).mean() > 0.5


def test_radii_follow_the_vortex(flight):
    ov, fixes, _ = flight
    sr = wf.storm_relative(ov, fixes, MOTION)
    radii = {(r.threshold, r.quadrant): r for r in wf.wind_radii(sr, None, "sfc")}
    # 50 kt at the surface = 55.6 kt at 700 mb; motion adds 10 kt on the NE leg's right... check each quadrant
    for q, brg in (("NE", 45.0), ("SE", 135.0), ("SW", 225.0), ("NW", 315.0)):
        r_axis = np.linspace(RM, 300, 3000)
        th = np.radians(90 - brg)                                  # math angle of the leg direction
        u = -vt(r_axis) * np.sin(th) + MOTION[1] * math.sin(math.radians(MOTION[0]))
        v = vt(r_axis) * np.cos(th) + MOTION[1] * math.cos(math.radians(MOTION[0]))
        sfc = 0.9 * np.hypot(u, v)
        want = r_axis[np.flatnonzero(sfc >= 50)].max()
        got = radii[(50, q)].obs_km
        assert got == pytest.approx(want, abs=5.0), q
    # stronger to the right of motion: SE (right-rear of a NE-moving storm) beats NW (left)
    assert radii[(50, "SE")].obs_km > radii[(50, "NW")].obs_km
    assert radii[(34, "NE")].obs_lower_bound            # 34 kt still blowing where the leg ended


def test_unsampled_quadrant():
    ov, fixes, _ = fly(headings=(45.0,), fix_times=(T0,))
    sr = wf.storm_relative(ov, fixes, MOTION)
    radii = {(r.threshold, r.quadrant): r for r in wf.wind_radii(sr, None, "sfc")}
    assert radii[(34, "SE")].n_obs == 0 and radii[(34, "SE")].obs_km is None
    assert radii[(34, "NE")].n_obs > 0


def test_motion_up_puts_the_strong_side_right(flight):
    ov, fixes, _ = flight
    sr = wf.storm_relative(ov, fixes, MOTION, rotate=True)
    near = (sr.r > 20) & (sr.r < 60)
    right = near & (sr.x > 10)
    left = near & (sr.x < -10)
    assert np.nanmean(sr.wspd[right]) > np.nanmean(sr.wspd[left]) + 10       # earth-relative winds add the motion
    assert sr.rotated


def test_tracks_never_join_flights():
    ov1, fixes, _ = fly(headings=(45.0,), fix_times=(T0,))
    ov2, _f, _t = fly(headings=(135.0,), fix_times=(T0,))
    both = satview.merge_overlays([ov1, ov2])                  # two aircraft at once, interleaved in time
    sr = wf.storm_relative(both, fixes, MOTION)
    for seg in sr.tracks():
        ids = {sr.segments[i] for i in seg}
        assert len(ids) == 1
