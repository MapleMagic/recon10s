"""
Himawari-9 (recon10s_himawari) offline: synthetic AWIPS ISatSS tiles laid out like NOAA's bucket,
served by a fake S3, so listing, tile choice, stitching, navigation and the target area are
checked without the network.
"""
import datetime as dt
import io
import math

import numpy as np
import pytest

import recon10s_goes as goes
import recon10s_himawari as hw

h5py = pytest.importorskip("h5py")
pytest.importorskip("pyproj")

UTC = dt.timezone.utc
STEP = 55.8879903                     # microradians per 2-km pixel
X0 = -153719.91730804                 # first column's scan angle (microradians)
SCALE, OFFSET = 0.06420898, 69.0


def tb_at(x, y):
    """The synthetic scene: brightness temperature (K) as a smooth function of the scan angles (rad)."""
    return 250.0 + 200.0 * x + 100.0 * y


def tile_bytes(rows, cols, x_raw, y_raw, x_off, y_off, row_off=0, col_off=0, space=None):
    """An ISatSS-like tile: 16-bit counts, scaled x/y in microradians, the fixed-grid projection."""
    x = (x_raw * STEP + x_off) * 1e-6
    y = (y_raw * -STEP + y_off) * 1e-6
    X, Y = np.meshgrid(x, y)
    raw = np.round((tb_at(X, Y) - OFFSET) / SCALE).astype(np.int16)
    if space is not None:
        raw[space] = -1076
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        d = f.create_dataset("Sectorized_CMI", data=raw, chunks=raw.shape, compression="gzip")
        d.attrs["units"] = np.bytes_("kelvin")
        d.attrs["scale_factor"] = np.array([SCALE], np.float32)
        d.attrs["add_offset"] = np.array([OFFSET], np.float32)
        for name, vals, off, sc in (("x", x_raw, x_off, STEP), ("y", y_raw, y_off, -STEP)):
            v = f.create_dataset(name, data=vals.astype(np.int16))
            v.attrs["scale_factor"] = np.array([sc])
            v.attrs["add_offset"] = np.array([off])
        p = f.create_dataset("fixedgrid_projection", data=np.int32(0))
        p.attrs["longitude_of_projection_origin"] = np.array([140.7])
        p.attrs["perspective_point_height"] = np.array([35785831.0])
        p.attrs["sweep_angle_axis"] = np.bytes_("y")
        f.attrs["tile_row_offset"] = np.array([row_off])
        f.attrs["tile_column_offset"] = np.array([col_off])
    return bio.getvalue()


class _Resp:
    def __init__(self, text="", content=b"", status=200):
        self.text, self.content, self.status_code = text, content, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise IOError(self.status_code)


class FakeS3:
    """Enough of S3's ListObjectsV2 and GET for the module: prefixes, delimiters, objects."""

    def __init__(self):
        self.objects = {}
        self.gets = []

    def get(self, url, params=None, timeout=None, headers=None):
        if params:
            prefix, delim = params["prefix"], params.get("delimiter")
            contents, folders = [], set()
            for k in sorted(self.objects):
                if not k.startswith(prefix):
                    continue
                rest = k[len(prefix):]
                if delim and "/" in rest:
                    folders.add(prefix + rest.split("/")[0] + "/")
                else:
                    contents.append(k)
            ns = "http://s3.amazonaws.com/doc/2006-03-01/"
            body = "".join(f"<Contents><Key>{k}</Key><Size>{len(self.objects[k])}</Size></Contents>" for k in contents)
            body += "".join(f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in sorted(folders))
            return _Resp(text=f'<ListBucketResult xmlns="{ns}"><Prefix>{prefix}</Prefix>{body}</ListBucketResult>')
        key = url[len(hw.BASE):]
        self.gets.append(key)
        if key not in self.objects:
            return _Resp(status=404)
        return _Resp(content=self.objects[key])


def full_disk_tiles(s3, t, numbers, ch=13):
    folder = f"{hw.PREFIX}/{t:%Y/%m/%d/%H%M}/"
    stamp = f"{t:%Y}{t.timetuple().tm_yday:03d}{t:%H%M%S}0"
    for n in numbers:
        r, c = hw.tile_position(n)
        rows = np.arange(r * 550, (r + 1) * 550)
        cols = np.arange(c * 550, (c + 1) * 550)
        space = None
        if n == numbers[0]:
            space = (slice(0, 3), slice(0, 3))           # a corner of space in the first tile
        key = f"{folder}OR_HFD-020-B{hw.BITS[ch]}-M1C{ch:02d}-T{n:03d}_GH9_s{stamp}_c{stamp}.nc"
        s3.objects[key] = tile_bytes(550, 550, cols, rows, X0, -X0, r * 550, c * 550, space)
    return folder


def target_tile(s3, t, col0=2530, row0=1765, ch=13):
    folder = f"{hw.PREFIX}/{t:%Y/%m/%d/%H%M}/"
    stamp = f"{t:%Y}{t.timetuple().tm_yday:03d}{t:%H%M%S}0"
    key = f"{folder}OR_HR3-020-B{hw.BITS[ch]}-M1C{ch:02d}-T001_GH9_s{stamp}_c{stamp}.nc"
    i = np.arange(500)
    s3.objects[key] = tile_bytes(500, 500, i, i, X0 + col0 * STEP, -X0 - row0 * STEP)
    return key


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(hw, "_session", lambda: fake)
    hw._LIST_CACHE.clear()
    return fake


def test_tile_numbering_matches_the_bucket():
    # checked against real tiles' row/column offsets
    known = {1: (0, 2), 2: (0, 3), 7: (1, 1), 8: (1, 2), 15: (2, 0), 45: (5, 0), 81: (8, 7), 88: (9, 7)}
    for n, rc in known.items():
        assert hw.tile_position(n) == rc and hw.tile_number(*rc) == n
    assert hw.tile_number(0, 0) is None and hw.tile_number(9, 9) is None
    assert sum(1 for r in range(10) for c in range(10) if hw.tile_number(r, c)) == 88


def test_pixel_window_and_bands():
    # the sub-satellite point is pixel N/2 + 0.5 of every grid (as the real tiles' x/y have it)
    for res, n in (("020", 5500), ("010", 11000), ("005", 22000)):
        r0, r1, c0, c1 = hw.pixel_window((-0.05, 0.05, 140.65, 140.75), res, pad=0)
        assert abs((r0 + r1 - 1) / 2 - (n / 2 + 0.5)) <= 1 and abs((c0 + c1 - 1) / 2 - (n / 2 + 0.5)) <= 1
    assert hw.pixel_window((40, 45, -40, -30), "020") is None          # the far side of the Earth
    assert hw.CHANNEL[2] == 3 and hw.res_code(3) == "005" and hw.res_code(13) == "020"


def test_satellite_choice_and_view():
    assert goes.pick_satellite(13.5, 141.4) == 9          # Champi: Himawari
    assert goes.pick_satellite(-17.4, -161.8) == 18       # Wati, east of the dateline: GOES-18
    assert goes.pick_satellite(15.0, 179.0) == 9 and goes.pick_satellite(15.0, -177.0) == 18
    assert goes.sees(9, 13.5, 141.4) and not goes.sees(9, 7.1, 58.7)     # an Arabian Sea invest is off every disk
    assert "beyond the edge" in goes.view_note(9, 7.1, 58.7)
    assert "obliquely" in goes.view_note(9, -20.0, 75.0) and goes.view_note(9, 13.5, 141.4) == ""
    assert abs(goes.zenith_angle(0.0, 140.7, 9)) < 1e-6
    assert goes.sat_name(9) == "Himawari-9" and goes.sat_name(18) == "GOES-18"


def test_full_disk_listing_and_stitching(s3):
    t = dt.datetime(2025, 9, 1, 12, 0, tzinfo=UTC)
    lat, lon = 13.5, 141.4
    bbox = (lat - 4, lat + 4, lon - 4, lon + 4)
    r0, r1, c0, c1 = hw.pixel_window(bbox, "020")
    need = sorted({hw.tile_number(r // 550, c // 550) for r in (r0, r1 - 1) for c in (c0, c1 - 1)})
    assert len(need) == 4                                  # the box straddles four tiles
    full_disk_tiles(s3, t, need)
    full_disk_tiles(s3, t + dt.timedelta(minutes=10), need)
    target_tile(s3, t + dt.timedelta(minutes=2, seconds=17))
    scenes = goes.list_scenes(9, 13, "F", when=t + dt.timedelta(minutes=15), hours_back=0)
    assert [s.start for s in scenes] == [t, t + dt.timedelta(minutes=10)]
    sc = scenes[-1]
    assert sc.sat == 9 and sc.sector == "F" and sorted(sc.tiles) == need and "Himawari-9" in str(sc)
    img = goes.load_image(sc, bbox=bbox)
    assert img.lon0 == 140.7 and img.sweep == "y" and img.kind == "brightness"
    assert img.data.shape == (r1 - r0, c1 - c0) and np.isfinite(img.data).all()
    # every pixel where the synthetic scene puts it: navigation and stitching across the tile edges
    x0, x1, yb, yt = img.extent_m
    nr, nc = img.data.shape
    xs = (x0 + (np.arange(nc) + 0.5) * (x1 - x0) / nc) / hw.HEIGHT
    ys = (yt - (np.arange(nr) + 0.5) * (yt - yb) / nr) / hw.HEIGHT
    X, Y = np.meshgrid(xs, ys)
    assert np.nanmax(np.abs(img.data + 273.15 - tb_at(X, Y))) < 0.05
    # only the four tiles were read, and nothing else
    assert sorted(k.split("-T")[1][:3] for k in s3.gets) == [f"{n:03d}" for n in need]


def test_space_is_missing(s3):
    t = dt.datetime(2025, 9, 1, 12, 0, tzinfo=UTC)
    n = hw.tile_number(5, 0)                               # the western limb
    full_disk_tiles(s3, t, [n])
    sc = goes.latest_scene(9, 13, "F", t + dt.timedelta(minutes=5))
    r, c = hw.tile_position(n)
    vals, *_ = hw._read(sc.tiles[n][0], sc.tiles[n][1])
    assert np.isnan(vals[:3, :3]).all() and np.isfinite(vals[3:, 3:]).all()


def test_a_scene_still_arriving_is_not_offered(s3, monkeypatch):
    now = dt.datetime.now(UTC).replace(second=0, microsecond=0)
    t = now - dt.timedelta(minutes=now.minute % 10 + 10)
    full_disk_tiles(s3, t, [44, 45])                       # 2 of 88 tiles so far
    assert all(s.start != t for s in goes.list_scenes(9, 13, "F", when=now, hours_back=1))


def test_target_area(s3):
    t = dt.datetime(2025, 9, 1, 12, 2, 17, tzinfo=UTC)
    target_tile(s3, t)
    full_disk_tiles(s3, dt.datetime(2025, 9, 1, 12, 0, tzinfo=UTC), [hw.tile_number(3, 4)])
    sc = goes.latest_scene(9, 13, "T", t + dt.timedelta(minutes=1))
    assert sc.sector == "T" and sc.start == t
    ext = goes.scene_extent_lonlat(sc)
    assert ext[0] < 13.5 < ext[1] and ext[2] < 141.3 < ext[3] and 8 < ext[1] - ext[0] < 11
    chosen, why = goes.choose_scene(13.5, 141.3, 13, sat=9, when=t + dt.timedelta(minutes=1))
    assert chosen.sector == "T" and "target area" in why
    far, why = goes.choose_scene(25.0, 125.0, 13, sat=9, when=t + dt.timedelta(minutes=1))
    assert far.sector == "F" and "elsewhere" in why
    img = goes.load_image(sc, bbox=(11.5, 15.5, 139.3, 143.3))
    assert img.extent_lonlat == ext and img.data.shape[0] < 500 and np.isfinite(img.data).all()
    # the map title names the satellite and the sector
    import recon10s_satview as satview
    assert "Himawari-9 band 13" in satview.scene_title(img) and "Target area" in satview.scene_title(img)


def test_products_from_matching_scans(s3):
    t = dt.datetime(2025, 9, 1, 12, 0, tzinfo=UTC)
    n = [hw.tile_number(3, 4), hw.tile_number(3, 5), hw.tile_number(4, 4), hw.tile_number(4, 5)]
    for ch in (8, 13):
        full_disk_tiles(s3, t, n, ch=ch)
    ref = goes.latest_scene(9, 13, "F", t + dt.timedelta(minutes=5))
    img = goes.load_product("wv_ir", ref, (12.0, 15.0, 140.0, 143.0))
    assert img.kind == "difference" and np.allclose(img.data, 0.0, atol=0.1)     # same synthetic scene in both


def test_satellite_intensity_takes_himawari_frames():
    """ir_from_goes reads any fixed grid: a Himawari cut (sweep y) lands on the storm grid like a GOES one."""
    import recon10s_satint as si
    lat, lon = 13.5, 141.4
    x0, y0 = hw.scan_angles(np.array([lat]), np.array([lon]))
    n = 400
    xs = x0[0] + (np.arange(n) - n / 2) * STEP * 1e-6
    ys = y0[0] - (np.arange(n) - n / 2) * STEP * 1e-6
    X, Y = np.meshgrid(xs, ys)
    data = tb_at(X, Y) - 273.15
    ext = ((xs[0] - STEP * 5e-7) * hw.HEIGHT, (xs[-1] + STEP * 5e-7) * hw.HEIGHT,
           (ys[-1] - STEP * 5e-7) * hw.HEIGHT, (ys[0] + STEP * 5e-7) * hw.HEIGHT)
    sc = goes.Scene(sat=9, band=13, sector="F", key="k", size=0, start=dt.datetime(2025, 9, 1, tzinfo=UTC))
    img = goes.GoesImage(data=data, extent_m=ext, lon0=140.7, sat_height=hw.HEIGHT, sweep="y", scene=sc,
                         kind="brightness")
    g = si.ir_from_goes(img, (lat, lon))
    assert g.shape == (si.IR_N, si.IR_N) and np.isfinite(g).mean() > 0.99
    assert abs(g[63:65, 63:65].mean() - tb_at(x0[0], y0[0])) < 0.3


def test_southern_storms_are_mirrored_into_northern_ones():
    import recon10s_satint as si
    g = si.grid(si.IR_N, si.IR_DX)
    X, Y = np.meshgrid(g, g)
    r = np.hypot(X, Y)
    storm = np.where(r < 12, 290.0, np.where(r < 120, 192.0 + 0.2 * r, 270.0)) + np.where((X > 0) & (Y > 0) & (r < 200), -20, 0)
    t = dt.datetime(2026, 3, 1, 6, tzinfo=UTC)
    north = si.network_inputs([storm] * 5, None, None, 0.0, 0.0, 0.0, 18.0, 10.0, True, t, 150.0)
    south = si.network_inputs([storm[::-1]] * 5, None, None, 0.0, 0.0, 0.0, -18.0, 10.0, True, t, 150.0)
    for a, b in zip(north, south):
        assert np.array_equal(a, b)
    assert south[2][0, 4] > 0                              # latitude by its size


def test_no_estimate_from_an_edge_on_view():
    import recon10s_satint as si
    src = si.TrackSource(advisory=type("A", (), {"time": dt.datetime(2026, 10, 18, 12, tzinfo=UTC), "lat": -2.0,
                                                 "lon": 60.6, "motion": None})())
    est = si.Estimator("sh932027", src)
    assert est.frame(dt.datetime(2026, 10, 18, 12, tzinfo=UTC)) is None
    assert "too obliquely" in est.errors[-1]


def test_estimates_mapped_onto_jtwcs_best_track(monkeypatch):
    import recon10s_satint as si
    fit = {"pass": {"vmax": {"terms": ["level"], "coef": [-2.0, 3.0], "scale": 1.25},
                    "mslp": {"terms": ["level", "lat"], "coef": [-3.0, 0.0, -2.0], "scale": 1.5}},
           "ir": {"vmax": {"terms": [], "coef": None, "scale": 1.2}, "mslp": {"terms": [], "coef": [-4.0], "scale": 1.6}}}
    monkeypatch.setattr(si, "JTWC_ADJUST", fit)
    v, p = si.to_jtwc((90, 95, 100, 105, 110), (960, 965, 970, 975, 980), 30.0)
    assert v[2] == pytest.approx(100 - 2.0 + 3.0 * (100 - 65) / 50) and v[4] - v[2] == pytest.approx(12.5)
    assert p[2] == pytest.approx(970 - 3.0 - 2.0 * (30 - 20) / 10) and p[2] - p[0] == pytest.approx(15.0)
    assert list(v) == sorted(v) and list(p) == sorted(p)
    v0, p0 = si.to_jtwc((90, 95, 100, 105, 110), (960, 965, 970, 975, 980), -30.0, ir_only=True)
    assert v0[2] == 100 and p0[2] == pytest.approx(966.0)
    g = si.grid(si.IR_N, si.IR_DX)
    X, Y = np.meshgrid(g, g)
    r = np.hypot(X, Y)
    storm = np.where(r < 12, 290.0, np.where(r < 120, 192.0 + 0.2 * r, 270.0))
    t = dt.datetime(2026, 10, 18, 6, tzinfo=UTC)
    a = si.estimate_from_grids([storm] * 5, t, (13.5, 141.4), 10.0, True)
    b = si.estimate_from_grids([storm] * 5, t, (13.5, 141.4), 10.0, True, jtwc=True)
    exp_v, exp_p = si.to_jtwc(a.vmax_q, a.mslp_q, 13.5, ir_only=True)
    assert b.vmax_q == pytest.approx(exp_v) and b.mslp_q == pytest.approx(exp_p)
    assert "mapped onto JTWC's best track" in b.notes and "mapped onto JTWC's best track" not in a.notes
    assert si.Estimator("wp292026", si.TrackSource()).jtwc and not si.Estimator("ep212026", si.TrackSource()).jtwc
