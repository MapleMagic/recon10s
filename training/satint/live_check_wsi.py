"""
Live check of the app's satellite intensity on current JTWC storms, through the app's own code: storm
list and best track (knackwx), Himawari-9 / GOES-18 IR, IR only (no PPS sign-in here). Prints the
estimates beside JTWC's best track and CIMSS's ADT and D-PRINT at the nearest times.

Usage: python live_check_wsi.py OUT.json STORM_ID [STORM_ID ...] [--hours 12]
"""
import datetime as dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import recon10s_nhc as nhc  # noqa: E402
import recon10s_satint as si  # noqa: E402
import recon10s_verify as verify  # noqa: E402

UTC = dt.timezone.utc


def nearest(ests, t, tol_min=45):
    best = min(ests, key=lambda e: abs((e.time - t).total_seconds()), default=None)
    return best if best is not None and abs((best.time - t).total_seconds()) <= tol_min * 60 else None


def main():
    a = sys.argv[1:]
    hours = 12.0
    if "--hours" in a:
        k = a.index("--hours")
        hours = float(a[k + 1])
        a = a[:k] + a[k + 2:]
    out_path, sids = a[0], a[1:]
    storms = {f.storm_id: f for f in nhc.all_storms()}
    print("storm list:", ", ".join(f"{f.storm_id} {f.name}" for f in storms.values()), "| notes", nhc.STORM_LIST_NOTES)
    result = {}
    for sid in sids:
        fix = storms.get(sid)
        track = nhc.best_track(sid)
        print(f"\n{sid}: {fix.label if fix else '(not in the list)'}; best track {len(track)} points to "
              f"{track[-1].time:%d/%HZ} ({track[-1].vmax} kt)" if track else f"\n{sid}: no best track")
        src = si.TrackSource(track=track, advisory=fix)
        est = si.Estimator(sid, src, progress=lambda m: None)
        end = dt.datetime.now(UTC) - dt.timedelta(minutes=10)
        t0 = time.time()
        ests = est.history(end, hours, 1.0)
        print(f"  {len(ests)} estimates in {time.time() - t0:.0f} s; errors: {est.errors[:3]}")
        adt = verify.fetch_adt(sid)[0]
        dp = verify.fetch_cimss_ml(sid, "D-PRINT")[0]
        rows = []
        for e in ests:
            bt = nhc.track_at(track, e.time, 3.0)
            a_ = nearest(adt, e.time)
            d_ = nearest(dp, e.time)
            rows.append({"time": e.time.isoformat(), "vmax": round(e.vmax, 1), "mslp": round(e.mslp, 1),
                         "q10_90": [round(e.vmax_q[0]), round(e.vmax_q[4])], "sat": e.satellite, "frames": e.ir_frames,
                         "centre": [round(c, 2) for c in e.centre],
                         "jtwc_bt": [bt.vmax, bt.mslp, bt.time.isoformat()] if bt else None,
                         "adt": [a_.vmax, a_.mslp] if a_ else None, "dprint": [d_.vmax, d_.mslp] if d_ else None})
            print(f"  {e.time:%d %H:%MZ} {e.satellite:10s} {e.vmax:5.0f} kt {e.mslp:6.0f} mb  (10-90 % "
                  f"{e.vmax_q[0]:.0f}-{e.vmax_q[4]:.0f}; {e.ir_frames} frames)  JTWC "
                  f"{(str(bt.vmax) + ' kt ' + str(bt.mslp) + ' mb @' + bt.time.strftime('%HZ')) if bt else '-':>18}  "
                  f"ADT {a_.vmax if a_ else '-'}  D-PRINT {d_.vmax if d_ else '-'}")
        result[sid] = rows
    json.dump(result, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
