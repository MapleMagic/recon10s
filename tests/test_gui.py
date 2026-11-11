"""
GUI smoke test (pytest-qt), fully offline: the network is refused, NHC's
storm list, best track and ADT come from fixtures/, and real recon
messages are fed straight into the panels.

    QT_QPA_PLATFORM=offscreen python -m pytest tests
"""
import datetime as dt
import math

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("cartopy")

from conftest import fixture_text, write_hrd_analysis
import recon10s_drops as drops
import recon10s_nhc as nhc
import recon10s_nhchdob as nhchdob
import recon10s_vdm as vdm
import recon10s_verify as verify

UTC = dt.timezone.utc
NOLO = nhc.Fix(storm_id="ep152026", name="NOLO", lat=16.7, lon=-155.8, classification="HU",
               vmax_kt=80, mslp_mb=972, time=dt.datetime(2026, 9, 26, 18, tzinfo=UTC))
ISAIAS = nhc.Fix(storm_id="al092026", name="ISAIAS", lat=23.4, lon=-90.6, classification="HU", vmax_kt=70,
                 mslp_mb=976, time=dt.datetime(2026, 10, 8, 12, tzinfo=UTC), motion_dir=60, motion_kt=8.7)
CHAMPI = nhc.Fix(storm_id="wp292026", name="Champi", lat=13.5, lon=141.4, classification="TY", vmax_kt=120,
                 mslp_mb=943, time=dt.datetime(2026, 10, 18, 12, tzinfo=UTC), source="ATCF via knackwx",
                 motion_dir=281, motion_kt=9.5)


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    import requests
    import recon10s_qt as Q
    import recon10s_jma as jma
    import recon10s_tdr3d as t3
    tried = []

    def refuse(*a, **k):
        url = k.get("url") or next((x for x in a if isinstance(x, str) and "://" in x), "?")
        tried.append(url)
        raise requests.ConnectionError(f"offline test: {url}")
    for name in ("get", "head", "post"):
        monkeypatch.setattr(requests, name, refuse)
    monkeypatch.setattr(requests.Session, "request", lambda self, *a, **k: refuse(*a, **k))
    monkeypatch.setattr(nhc, "active_storms", lambda *a, **k: [NOLO, ISAIAS])
    monkeypatch.setattr(nhc, "knack_storms", lambda *a, **k: [CHAMPI])             # JTWC's basins
    track = nhc.parse_bdeck(fixture_text("bdeck_al092026.dat"))
    monkeypatch.setattr(nhc, "best_track", lambda sid, *a, **k: list(track) if sid == "al092026" else [])
    adt = fixture_text("adt_09L_list.txt")
    monkeypatch.setattr(verify, "fetch_adt", lambda sid, *a, **k: (
        (verify.parse_adt(adt), verify.ADT_URL.format(code="09L"), adt) if sid == "al092026" else ([], "", "")))
    ml = {"D-PRINT": fixture_text("dprint_2026_09L.txt"), "D-MINT": fixture_text("dmint_2026_09L.txt")}
    monkeypatch.setattr(verify, "fetch_cimss_ml", lambda sid, product, *a, **k: (
        (verify.parse_cimss_ml(ml[product]), verify.CIMSS_ML_URL[product].format(year="2026", code="09L"), ml[product])
        if sid == "al092026" else ([], "", "")))
    monkeypatch.setattr(t3, "ships_shear", lambda *a, **k: None)
    monkeypatch.setattr(jma, "active_storms", lambda *a, **k: [])      # keep the JMA tab offline too
    monkeypatch.setattr(jma, "latest_day", lambda *a, **k: None)
    monkeypatch.setattr(Q, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    w = Q.MainWindow()
    qtbot.addWidget(w)
    w.resize(1500, 950)
    w.show()
    qtbot.waitExposed(w)
    w.tried_urls = tried
    return w


def _feed(sp):
    """Real Nolo batches and a vortex message, as if they had just been downloaded."""
    batches = [nhchdob.parse_message(fixture_text(f"hdob_af302_{n}.txt")) for n in (29, 30, 31, 32)]
    sp._on_hdobs(batches)
    sp._on_vdms([vdm.parse_vdm(fixture_text("vdm_af302_1739.txt"), dt.datetime(2026, 9, 26, 18, 6, tzinfo=UTC))])


def _feed_isaias(w, qtbot):
    """AF305's Isaias eyewall pass, its vortex message and SE-eyewall drop, with Isaias chosen."""
    sp = w.sat_panel
    assert sp.select_storm("al092026")
    d = drops.parse_tempdrop(fixture_text("tempdrop_isaias_92.txt"))
    drops._resolve_times(d, dt.datetime(2026, 10, 8, 12, 24, tzinfo=UTC))
    sp.set_recon_data([nhchdob.parse_message(fixture_text("hdob_af305_isaias.txt"))],
                      [vdm.parse_vdm(fixture_text("vdm_af305_1156.txt"), dt.datetime(2026, 10, 8, 12, 35, tzinfo=UTC))],
                      [d])
    ip = w.intensity_panel
    qtbot.waitUntil(lambda: len(ip.track) == 6 and bool(ip.adt) and len(ip.cimss_ml) == 2, timeout=5000)  # "fetched"
    return sp.bundle()


def test_every_section_and_tab_opens(window, qtbot):
    w = window
    assert [w.recon_tabs.tabText(i) for i in range(w.recon_tabs.count())] == \
        ["Live plot", "Convert IWG1 to HDOB", "Analysis", "Wind field", "Intensity"]
    assert [w.imagery_tabs.tabText(i) for i in range(w.imagery_tabs.count())] == \
        ["Satellite", "Microwave", "TDR"]
    assert [w.radar_tabs.tabText(i) for i in range(w.radar_tabs.count())] == \
        ["NEXRAD", "JMA", "Airborne TDR", "Radar volume"]
    assert [w.experimental_tabs.tabText(i) for i in range(w.experimental_tabs.count())] == \
        ["3D structure", "TDR evolution", "Synthetic TDR", "Satellite intensity"]
    for go in (w.go_recon, w.go_imagery, w.go_radar, w.go_experimental, w.go_settings, w.go_home):
        go()
        qtbot.wait(20)
    for tabs in (w.recon_tabs, w.imagery_tabs, w.radar_tabs, w.experimental_tabs):
        for i in range(tabs.count()):
            tabs.setCurrentIndex(i)
            qtbot.wait(20)
    sp = w.sat_panel
    for i in range(sp.toolbox.count()):
        sp.toolbox.setCurrentIndex(i)
        qtbot.wait(10)
    assert sp.storm_box.itemData(0).name == "NOLO"
    assert w.tried_urls == []


def test_recon_flows_to_map_analysis_and_microwave(window, qtbot):
    w = window
    sp = w.sat_panel
    _feed(sp)
    b = sp.bundle()
    assert len(b.overlay) == 80 and len(b.fixes) == 1 and len(b.passes) == 1
    assert w.analysis_panel.table.rowCount() == 1
    assert "83 kt" in w.analysis_panel.table.item(0, 7).text()     # outbound max from the VDM
    assert w.mw_panel.recon is b
    assert w.wind_panel.bundle is b and w.intensity_panel.bundle is b
    for chk in (sp.chk_peaks, sp.chk_declutter, sp.chk_models, sp.chk_motion):
        chk.setChecked(True)
    sp.flag_box.setCurrentIndex(1)
    qtbot.wait(50)
    assert sp._art is not None and sp._art.extra_artists          # fix drawn


def test_tree_unticking_removes_batches(window, qtbot):
    sp = window.sat_panel
    _feed(sp)
    top = sp.tree.topLevelItem(0)
    from PyQt6.QtCore import Qt
    top.child(0).setCheckState(0, Qt.CheckState.Unchecked)
    qtbot.waitUntil(lambda: len(sp.bundle().overlay) == 60, timeout=2000)


def test_measure_tool(window):
    sp = window.sat_panel
    sp.btn_measure.setChecked(True)
    assert sp.measure((16.0, -155.0), (17.0, -155.0)).startswith("111 km")


def test_convert_window_validation(window, qtbot):
    w = window
    w.go_recon(w.convert_page)
    w.start_edit.setText("23:30")
    w.end_edit.setText("00:30")          # same date on both ends: ends before it starts
    qtbot.wait(20)
    assert not w.btn_run.isEnabled() and "before it starts" in w.window_hint.text()
    w.end_date.setDate(w.start_date.date().addDays(1))
    qtbot.wait(20)
    assert w.btn_run.isEnabled() and "(1.0 h)" in w.window_hint.text()


def test_session_round_trip(window, qtbot):
    w = window
    sp = w.sat_panel
    sp.product_box.setCurrentIndex(sp.product_box.findData("airmass"))
    sp.chk_glm.setChecked(True)
    sp.hdob_hours.setValue(20)
    _feed(sp)
    w._collect_settings()
    ses = w.settings["session"]["satellite"]
    assert ses["storm_id"] == "ep152026" and ses["product"] == "airmass" and ses["loaded"]["vdms"]
    for key in ("wind", "intensity", "radarvol", "tdrevo"):
        assert key in w.settings["session"]
    sp.product_box.setCurrentIndex(0)
    sp.chk_glm.setChecked(False)
    sp.hdob_hours.setValue(3)
    sp.restore(ses, reload_data=False)
    assert sp.product_box.currentData() == "airmass" and sp.chk_glm.isChecked() and sp.hdob_hours.value() == 20


def test_radar_section(window, qtbot):
    w = window
    w.go_radar()
    qtbot.wait(20)
    rp = w.radar_panel
    # the faked Nolo (16.7N 155.8W) is 166 mi from South Shore (PHWA): inside Doppler reach
    qtbot.waitUntil(lambda: rp.site_box.count() > 0, timeout=3000)
    assert rp.site() == "PHWA" and "166 mi" in rp.site_note.text() and "every product" in rp.site_note.text()
    for b in rp.layout_group.buttons():
        if b.property("n") == 4:
            b.setChecked(True)
    assert all(cb.isEnabled() for cb in rp.product_boxes)
    assert w.tdrgrid_panel.view_box.count() == 4


def test_experimental_3d(window, qtbot):
    import test_tdr3d
    w = window
    w.go_experimental()
    p = w.tdr3d_panel
    grid = test_tdr3d.make_storm(shear_heading=250.0, shear_ms=8.0, conv_heading=160.0)
    p.follow(grid)                                      # as when the Airborne TDR tab loads one
    qtbot.waitUntil(lambda: p.shear is not None, timeout=20000)
    assert abs(p.shear.heading - 250) < 10
    assert "downshear-left" in p.summary.text()
    assert any(hasattr(ax, "elev") for ax in p.fig3d.axes)   # a 3D axes was drawn


# ------------------------------------------------------------------ 2.3.0

def test_isaias_wind_field_surface_winds_and_timeline(window, qtbot):
    w = window
    b = _feed_isaias(w, qtbot)
    assert len(b.fixes) == 1 and len(b.drops) == 1
    # Analysis > Surface winds: the drop beside the flight level at its release
    an = w.analysis_panel
    assert an.sfc_table.rowCount() == 1
    row = [an.sfc_table.item(0, c).text() if an.sfc_table.item(0, c) else "" for c in range(an.sfc_table.columnCount())]
    assert "SE Eyewall" in " ".join(row) and any("73" in c for c in row)
    # Wind field: storm-relative radii, with NHC's best track beside them
    wp = w.wind_panel
    qtbot.waitUntil(lambda: wp.table.rowCount() == 12 and bool(wp.track), timeout=5000)
    assert wp.sr is not None and len(wp.sr) > 10
    assert wp.table.horizontalHeaderItem(4).text().startswith("NHC best track 08/12Z")
    cells = {(wp.table.item(r, 0).text(), wp.table.item(r, 1).text()): wp.table.item(r, 4).text()
             for r in range(wp.table.rowCount())}
    assert cells[("34 kt", "NE")] == "80 nm"                    # b-deck 12Z: 34-kt NE radius
    for src in ("manual", "track", "nhc"):
        wp.motion_src.setCurrentIndex(wp.motion_src.findData(src))
        qtbot.wait(50)
    assert wp.motion() == (60.0, 8.7)                         # the advisory's motion
    # Intensity: vortex-message pressure, best track and ADT on one chart
    ip = w.intensity_panel
    tl = ip.timeline
    assert [p.value for p in tl.pressure["Vortex message"]] == [978.0]
    assert len(tl.pressure["NHC best track"]) == 6 and tl.wind["CIMSS ADT"]
    assert "vortex message" in ip.status.text() and "978 mb" in ip.status.text()
    assert w.tried_urls == []


def _sat_estimates():
    """recon10s satellite estimates around Isaias's eyewall pass (as the Satellite intensity tab makes them)."""
    import recon10s_satint as satint
    t0 = dt.datetime(2026, 10, 8, 10, tzinfo=UTC)
    out = []
    for k in range(7):
        v = 66.0 + k
        out.append(satint.Estimate(time=t0 + dt.timedelta(minutes=30 * k), vmax_q=(v - 12, v - 6, v, v + 6, v + 12),
                                   mslp_q=(992 - k, 986 - k, 981 - k, 976 - k, 971 - k), centre=(23.4, -90.6),
                                   satellite="GOES-19", ir_frames=5, mw_sensor="GMI" if k < 4 else "",
                                   mw_time=t0 if k < 4 else None, mw_cover=0.9 if k < 4 else 0.0))
    return out


def test_satellite_estimates_feed_verification(window, qtbot):
    w = window
    _feed_isaias(w, qtbot)
    sp = w.satint_panel
    sp.set_estimates(_sat_estimates())                        # as after Estimate (or from a saved case)
    ip = w.intensity_panel
    assert len(ip.sat_estimates) == 7 and "7 recon10s satellite estimate" in ip.sat_note.text()
    text = ip.scores_label.text()
    assert "recon10s satellite vs recon" in text and "CIMSS ADT vs recon" in text
    assert "CIMSS D-PRINT vs recon" in text and "CIMSS D-MINT vs recon" in text
    assert "D-PRINT: 93 estimates" in ip.ref_note.text()
    assert "MSLP (mb)" in text and ip.pair_table.rowCount() >= 2
    assert "recon10s satellite" in ip.series_checks
    for b in ip.truth_group.buttons():
        if b.property("key") == "track":
            b.setChecked(True)
    qtbot.wait(50)
    assert "vs NHC best track" in ip.scores_label.text()
    ip.tabs.setCurrentIndex(1)                                # the verification chart draws
    qtbot.wait(50)
    assert ip.vf_fig.axes
    # the panel's own chart and readout, with CIMSS's estimates beside it
    assert sp.figure.axes and "kt" in sp.result.text() and "GMI" not in sp.result.text()   # the newest is IR only
    assert sorted(sp.reference) == ["CIMSS ADT", "CIMSS D-MINT", "CIMSS D-PRINT"]
    labels = [t.get_text() for t in sp.figure.axes[0].get_legend().get_texts()]
    assert "CIMSS D-PRINT" in labels and "CIMSS D-MINT" in labels and "CIMSS ADT" in labels
    sp.set_until(dt.datetime(2026, 10, 8, 11, 15, tzinfo=UTC))                              # a replay's clock
    assert len(sp.visible()) == 3 and len(ip.estimator_series()["recon10s satellite"]) == 3
    assert w.tried_urls == []


def test_choosing_a_storm_without_recon_reaches_the_other_tabs(window, qtbot):
    w = window
    sp, si_panel, ip = w.sat_panel, w.satint_panel, w.intensity_panel
    qtbot.waitUntil(lambda: si_panel.fix is not None, timeout=5000)        # the first storm in NHC's list
    assert si_panel.fix.storm_id == "ep152026" and ip.storm() == "ep152026"
    assert sp.select_storm("al092026")
    qtbot.waitUntil(lambda: si_panel.fix.storm_id == "al092026" and len(ip.track) == 6, timeout=5000)
    assert ip.storm() == "al092026" and "Isaias" in si_panel.storm_label.text()
    ip.storm_id.setText("al102026")                          # typed by hand: choosing a storm leaves it be
    assert sp.select_storm("ep152026")
    qtbot.waitUntil(lambda: si_panel.fix.storm_id == "ep152026", timeout=5000)
    assert ip.storm() == "al102026"
    assert w.tried_urls == []


def test_a_west_pacific_storm(window, qtbot):
    """JTWC's basins reach the storm lists: the Satellite tab offers Himawari-9, and the other tabs follow."""
    import recon10s_goes as goes
    w = window
    sp, si_panel, ip = w.sat_panel, w.satint_panel, w.intensity_panel
    labels = [sp.storm_box.itemText(i) for i in range(sp.storm_box.count())]
    assert labels[:3] == [NOLO.label, ISAIAS.label, CHAMPI.label] and labels[2].startswith("Typhoon Champi (29W)")
    assert sp.sat_box.findData(9) >= 0 and sp.sector_box.findData("T") >= 0
    assert sp.select_storm("wp292026")
    qtbot.waitUntil(lambda: si_panel.fix is not None and si_panel.fix.storm_id == "wp292026", timeout=5000)
    assert "JTWC 120 kt" in si_panel.storm_label.text() and ip.storm() == "wp292026"
    # no lightning mapper reaches 141E: no GLM key for a Himawari frame there
    sc = goes.Scene(sat=9, band=13, sector="F", key="k", size=0, start=dt.datetime(2026, 10, 18, 12, tzinfo=UTC))
    img = goes.GoesImage(data=np.zeros((2, 2)), extent_m=(0.0, 1.0, 0.0, 1.0), lon0=140.7, sat_height=35785831.0,
                         sweep="y", scene=sc, kind="brightness")
    assert sp._glm_key(img) is None
    if w.mw_panel is not None:
        w.mw_panel.refresh_storms()
        assert any("Champi" in w.mw_panel.storm_box.itemText(i) for i in range(w.mw_panel.storm_box.count()))
    assert w.tried_urls == []


def test_satellite_intensity_estimate_button(window, qtbot, monkeypatch):
    """Estimate in Experimental > Satellite intensity, with GOES and PPS replaced by synthetic grids
    so the shipped network runs on a ring-shaped storm (IR now and earlier, and a GMI pass)."""
    import recon10s_satint as si
    w = window
    _feed_isaias(w, qtbot)
    g = si.grid(si.IR_N, si.IR_DX)
    X, Y = np.meshgrid(g, g)
    r = np.hypot(X, Y)
    ring = np.where(r < 14, 289.0, np.where(r < 140, 190.0 + 0.25 * r, 268.0))
    gm = si.grid(si.MW_N, si.MW_DX)
    XM, YM = np.meshgrid(gm, gm)
    rm = np.hypot(XM, YM)
    mwg = np.stack([250 + 0 * rm, 230 + 0 * rm, np.where((rm > 15) & (rm < 45), 205.0, 270.0),
                    np.where((rm > 15) & (rm < 45), 200.0, 262.0)])
    calls = []

    class FakeEstimator:
        def __init__(self, sid, source, email=None, progress=None, **k):
            calls.append((sid, email))
            self.source = source
            self.errors = []
            self._mw = [si.MwInput(time=dt.datetime(2026, 10, 8, 9, 40, tzinfo=UTC), sensor="GMI", grid=mwg, cover=0.95,
                                   centre=(23.3, -90.8))]

        def frame(self, when, centre=None):
            return ring, when, 19

        def estimate(self, when, passes=None):
            m = self._mw[0]
            return si.estimate_from_grids([ring] * 5, when, self.source.centre_at(when) or (23.4, -90.6), 8.7, False,
                                          mw=m.grid, mw_ir=ring, mw_time=m.time, mw_sensor="GMI", mw_cover=m.cover,
                                          satellite="GOES-19")

        def history(self, end, hours=12.0, step_h=1.0):
            n = int(round(hours / step_h))
            return [self.estimate(end - dt.timedelta(hours=step_h * k)) for k in range(n, -1, -1)]

    monkeypatch.setattr(si, "Estimator", FakeEstimator)
    sp = w.satint_panel
    assert sp.model is not None and sp.btn_run.isEnabled()
    assert sp.fix is not None and sp.fix.storm_id == "al092026"
    sp.rb_at.setChecked(True)
    from PyQt6.QtCore import QDate, QDateTime, Qt, QTime
    sp.when_edit.setDateTime(QDateTime(QDate(2026, 10, 8), QTime(12, 0), Qt.TimeSpec.UTC))
    sp.hours.setValue(3.0)
    sp.btn_run.click()
    qtbot.waitUntil(lambda: len(sp.estimates) == 4 and not sp._busy, timeout=30000)
    assert calls and calls[0][0] == "al092026"
    e = sp.estimates[-1]
    assert e.time == dt.datetime(2026, 10, 8, 12, tzinfo=UTC) and e.mw_sensor == "GMI"
    assert list(e.vmax_q) == sorted(e.vmax_q) and 20 < e.vmax < 180 and 880 < e.mslp < 1015
    assert "kt" in sp.result.text() and "GMI pass at 09:40Z" in sp.result.text()
    assert len(sp.figure.axes) >= 4                             # the two series and the IR and 89-GHz images
    csv = sp.csv_text().splitlines()
    assert len(csv) == 5 and csv[0].startswith("time_utc,vmax_kt_p10") and csv[-1].startswith("2026-10-08 12:00,")
    assert ",GMI,2026-10-08 09:40," in csv[-1]
    ip = w.intensity_panel
    assert len(ip.sat_estimates) == 4 and "recon10s satellite vs recon" in ip.scores_label.text()
    assert w.tried_urls == []


def test_case_save_open_and_replay(window, qtbot, tmp_path, monkeypatch):
    import recon10s_qt as Q
    w = window
    _feed_isaias(w, qtbot)
    w.satint_panel.set_estimates(_sat_estimates())
    contents = w._case_contents()
    assert contents["fix"].storm_id == "al092026" and len(contents["track"]) == 6
    assert len(contents["satint"]) == 7 and contents["adt_text"]
    assert sorted(contents["cimss_ml"]) == ["D-MINT", "D-PRINT"] and "DPRINThist_2026_09L" in contents["cimss_ml"]["D-PRINT"][0]
    path = str(tmp_path / "isaias.r10case")
    monkeypatch.setattr(Q.QFileDialog, "getSaveFileName", lambda *a, **k: (path, ""))
    w.save_case()                                            # File > Save case
    qtbot.waitUntil(lambda: w.statusBar().currentMessage().startswith("Saved isaias.r10case"), timeout=10000)
    assert "1 HDOBs, 1 vortex messages, 1 drops, 0 TDR analyses" in w.statusBar().currentMessage()
    w.open_case(path)                                        # File > Open case
    qtbot.waitUntil(lambda: w.replay is not None, timeout=10000)
    assert w.replay_bar.isVisibleTo(w) and "ISAIAS AL092026" in w.replay_title.text()
    assert w.sat_panel.replay_mode and not w.sat_panel.chk_live.isChecked()
    r = w.replay
    w._replay_set(r.start, force=True)
    empty = w.sat_panel.bundle()
    assert (empty.overlay is None or not len(empty.overlay)) and "0 HDOBs" in w.replay_clock_label.text()
    ip = w.intensity_panel
    assert ip.until == r.start and all(p.time <= r.start for p in ip.track)
    series = ip.estimator_series()
    assert all(e.time <= r.start for k in ("CIMSS D-PRINT", "CIMSS D-MINT") for e in series.get(k, []))
    assert len(w.satint_panel.estimates) == 7                # the case's satellite estimates came back
    assert all(e.time <= r.start for e in w.satint_panel.visible())
    w._replay_set(r.end)
    assert len(ip.estimator_series()["CIMSS D-MINT"]) > 10   # the case's D-MINT came back for the replay
    b = w.sat_panel.bundle()
    assert len(b.overlay) == 20 and len(b.fixes) == 1 and len(b.drops) == 1
    assert "1 fixes" in w.replay_clock_label.text()
    for _ in range(3):
        w._replay_tick()                                     # the clock stops at the end
    assert w._replay_clock == r.end
    w.exit_replay()
    assert not w.replay_bar.isVisibleTo(w) and ip.until is None and not w.sat_panel.replay_mode
    assert w.tried_urls == []


def test_radar_volume_to_3d(window, qtbot):
    import numpy as np
    import test_nexgrid as tn
    w = window
    motion = (300.0, 12.0)

    def fn(product, az, rng, h):
        if product == "velocity":
            return 12.0 * np.cos(np.radians(az - 300.0)) + 0 * h
        return tn.layered(h)
    vol = tn.FakeVolume(fn)
    p = w.radarvol_panel
    p.half_width.setValue(60)
    p.resolution.setCurrentIndex(p.resolution.findData(2.0))
    w._on_volume_requested(vol, (27.0, -81.0), "TEST", motion)
    assert w.radar_tabs.currentWidget() is p
    qtbot.waitUntil(lambda: p.grid is not None, timeout=20000)
    assert {"REFLECTIVITY", "RADIAL_VELOCITY", "SR_VELOCITY"} <= set(p.grid.fields)
    assert "removed" in p.source_label.text() and p.btn_3d.isEnabled()
    assert w.statusBar().currentMessage().startswith("Radar volume ready: KTST 20261009 180211 volume")
    fields = [p.field_box.itemData(i) for i in range(p.field_box.count())]
    assert len(fields) == len(set(fields)) and "SR_VELOCITY" in fields
    p.field_box.setCurrentIndex(p.field_box.findData("SR_VELOCITY"))
    p.redraw()
    p.btn_3d.click()                                          # hand it to Experimental > 3D structure
    t3p = w.tdr3d_panel
    qtbot.waitUntil(lambda: t3p.grid is p.grid, timeout=5000)
    assert w.experimental_tabs.currentWidget() is t3p and not t3p.chk_follow.isChecked()
    qtbot.waitUntil(lambda: any(hasattr(ax, "elev") for ax in t3p.fig3d.axes), timeout=20000)
    assert w.tried_urls == []


def test_tdr_evolution_opens_a_pass(window, qtbot, tmp_path, monkeypatch):
    import recon10s_tdrevo as evo
    import recon10s_tdrgrid as tg
    w = window
    nc = tmp_path / "t_xy.nc"
    write_hrd_analysis(str(nc))
    data = nc.read_bytes()
    monkeypatch.setattr(tg, "list_analyses", lambda mission, *a, **k: [
        tg.Analysis(mission, "2211", "260925I1_2211_xy.nc.gz")])
    monkeypatch.setattr(tg, "_fetch", lambda url, timeout=120.0: data)
    ev = w.tdrevo_panel
    rows = [evo.PassMetrics(mission="20260925I1", name=f"260925I1_{hhmm}_xy.nc.gz",
                            time=f"2026-09-25T{hhmm[:2]}:{hhmm[2:]}:00+00:00", peak_2km_kt=pk, rmw_2km=rmw,
                            tilt_km=tilt, tilt_heading=45.0, shear_kt=sh, shear_heading=250.0, shear_kt_err=2.0,
                            shear_heading_err=10.0, shear_grade="good", column_levels=20)
            for hhmm, pk, rmw, tilt, sh in (("2131", 90.0, 30.0, 8.0, 12.0), ("2211", 101.0, 26.0, 5.0, 10.0))]
    ev.set_metrics(rows)
    assert ev.table.rowCount() == 2 and "Over 1 h and 2 passes" in ev.summary.text()
    ev._on_double_click(1, 0)
    assert w.radar_tabs.currentWidget() is w.tdrgrid_panel
    qtbot.waitUntil(lambda: w.tdrgrid_panel.grid is not None, timeout=10000)
    assert w.tdrgrid_panel.grid.flight == "260925I1"
    assert w.tried_urls == []


# ------------------------------------------------------------------ 2.4.0

def test_synthetic_tdr(window, qtbot, tmp_path):
    """Isaias's eyewall pass in, a synthetic TDR out: the surface view, the 3-D views, the satellite layer, 3-D."""
    import recon10s_syntdr as st
    w = window
    syn = w.syntdr_panel
    syn.live = st.LiveCalibration(str(tmp_path / "syntdr_live.json"))      # not the user's own record
    _feed_isaias(w, qtbot)
    assert "flight-level obs" in syn.data_label.text() and syn.centre_time() is not None
    assert syn.motion_box.motion() is not None
    w.go_experimental(syn)
    layers = []
    syn.layer_changed.connect(layers.append)
    syn.build()
    qtbot.waitUntil(lambda: syn.result is not None and not syn._busy, timeout=60000)
    r = syn.result
    assert r.grid.source == "synthetic" and np.isfinite(r.intensity_kt) and 20 < r.intensity_kt < 160
    assert "Intensity estimate" in syn.results.text() and "SYNTHETIC" in syn.figure._suptitle.get_text()
    assert any(role[0] == "surface" for role in syn._roles.values())
    ax = next(a for a, role in syn._roles.items() if role[0] == "surface")
    assert "10-m wind" in syn.probe_text(ax, 20.0, 10.0)
    for key in ("overview", "plan", "section", "rz"):
        syn.view_box.setCurrentIndex(syn.view_box.findData(key))
        qtbot.wait(20)
    syn.field_box.setCurrentIndex(syn.field_box.findData("WIND_SPREAD"))
    syn.chk_map.setChecked(True)
    assert layers and layers[-1] is not None and "Synthetic TDR" in layers[-1].label
    # the 3-D tab takes the synthetic grid and says what it is
    t3p = w.tdr3d_panel
    syn.btn_3d.click()
    assert w.experimental_tabs.currentWidget() is t3p and t3p.grid is r.grid
    qtbot.waitUntil(lambda: "SYNTHETIC" in t3p.summary.text(), timeout=30000)
    syn.restore(syn.state())
    assert w.tried_urls == []
