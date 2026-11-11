"""IWG1 -> HDOB conversion: NHC dating across 00Z, and dated windows."""
import datetime as dt

import recon10s

UTC = dt.timezone.utc


def _rows(start, hours, step=1):
    out = []
    for i in range(0, int(hours * 3600), step):
        t = start + dt.timedelta(seconds=i)
        f = [""] * 33
        f[0], f[1] = "IWG1", t.strftime("%Y-%m-%dT%H:%M:%S")
        f[2], f[3] = "16.5", "-155.5"
        f[4] = f[5] = f[6] = "3050"
        f[23], f[26], f[27] = "700", "30", "90"
        out.append(recon10s.parse_iwg1_line(",".join(f)))
    return [r for r in out if r is not None]


def _messages(text):
    return [m.strip().splitlines() for m in text.split("\n\n") if m.strip()]


def test_messages_dated_like_nhc_across_midnight():
    rows = _rows(dt.datetime(2026, 9, 26, 23, 0, tzinfo=UTC), 2)
    msgs = _messages(recon10s.convert_iwg1_to_hdob(rows, mission="AF302 0515E NOLO"))
    by_first = {m[2][:6]: m for m in msgs}
    before = by_first["235015"]           # batch starting before 00Z
    after = by_first["000015"]            # first batch after
    assert before[1].endswith("20260926") and before[0].endswith("262359")
    assert after[1].endswith("20260927") and after[0].endswith("270009")
    assert len({m[0] for m in msgs}) == len(msgs), "every message needs its own header time"


def test_bare_window_repeats_daily_but_dated_window_does_not():
    rows = _rows(dt.datetime(2026, 9, 26, 6, 0, tzinfo=UTC), 36, step=60)
    bare = recon10s.filter_rows_window(rows, None, None, 12 * 3600, 14 * 3600)
    s, _ = recon10s.parse_window_bound("2026-09-27 12:00")
    e, _ = recon10s.parse_window_bound("14:00", dt.date(2026, 9, 27))
    dated = recon10s.filter_rows_window(rows, s, e)
    assert len(bare) == 2 * len(dated)
    assert {r.t.date() for r in dated} == {dt.date(2026, 9, 27)}


def test_parse_window_bound_forms():
    assert recon10s.parse_window_bound("") == (None, None)
    assert recon10s.parse_window_bound("17:45") == (None, 17 * 3600 + 45 * 60)
    want = dt.datetime(2026, 9, 26, 17, 45, tzinfo=UTC)
    for text in ("2026-09-26 17:45", "20260926 1745", "2026-09-26T17:45:00"):
        assert recon10s.parse_window_bound(text)[0] == want
    assert recon10s.parse_window_bound("1745", dt.date(2026, 9, 26))[0] == want
