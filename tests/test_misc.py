"""Smaller pieces: colour table, sector coverage, KML parsing, TDR names, overlays."""
import datetime as dt
import xml.etree.ElementTree as ET

import recon10s_forecast as fc
import recon10s_goes as goes
import recon10s_irtable as irtable
import recon10s_satview as satview
import recon10s_tdr as tdr


def test_ir_table_endpoints():
    cmap, norm = irtable.mpl_colormap()
    assert norm.vmin < -80 and norm.vmax > 30


def test_meso_coverage_across_dateline():
    assert goes.covers((10, 25, 170, -170), 17.0, 179.5)
    assert goes.covers((10, 25, 170, -170), 17.0, -175.0)
    assert not goes.covers((10, 25, 170, -170), 17.0, -150.0)


def test_kml_parsing_ignores_namespace():
    for ns in ("http://earth.google.com/kml/2.1", "http://www.opengis.net/kml/2.2"):
        root = ET.fromstring(f"<kml xmlns='{ns}'><Placemark><Polygon><outerBoundaryIs><LinearRing>"
                             "<coordinates>-150,15,0 -151,16,0 -152,15,0</coordinates>"
                             "</LinearRing></outerBoundaryIs></Polygon></Placemark></kml>")
        polys = [fc._coords(fc._child(p, "coordinates").text) for p in fc._iter(root, "Polygon")]
        assert polys == [[(-150.0, 15.0), (-151.0, 16.0), (-152.0, 15.0)]]


def test_tdr_file_names():
    m = tdr.TdrMission(folder="20260925I1", year=2026, date=dt.date(2026, 9, 25), aircraft="NOAA43")
    m.files = {"swaths": [f"260925I1_POLO_2211_dbz_ws_{lv}km.png" for lv in ("1.0", "2.0", "10.0")],
               "rzmeans": ["260925I1_POLO_2154_2304_vt_rzmean.png", "260925I1_POLO_2154_2304_dbz_rzmean.png"]}
    assert m.swath_times() == ["2211"] and m.swath_levels("2211") == [1.0, 2.0, 10.0]
    assert m.swath("2211", 10.0).endswith("_10.0km.png")
    assert m.windows("rzmeans") == ["2154_2304"] and m.rz_vars("2154_2304") == ["dbz", "vt"]
    assert m.url("swaths", "x.png").endswith("/2026/20260925I1/swaths/x.png")


def test_merge_keeps_flights_apart():
    t = dt.datetime(2026, 9, 26, 12, tzinfo=dt.timezone.utc)
    a, b = satview.Overlay(), satview.Overlay()
    for k, ov in enumerate((a, b)):
        for i in range(3):
            ov.lats.append(16.0 + k); ov.lons.append(-155.0); ov.winds.append(50); ov.dirs.append(90)
            ov.mslps.append(None); ov.times.append(t + dt.timedelta(minutes=2 * i + k))
    m = satview.merge_overlays([a, b])
    assert len(m) == 6 and sorted(set(m.segments)) == [0, 1]
    assert all(x <= y for x, y in zip(m.times, m.times[1:]))
