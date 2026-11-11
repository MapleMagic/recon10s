"""Shared test setup: scripts/ on the path, headless Qt and matplotlib, fixture helpers."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
FIXTURES = os.path.join(HERE, "fixtures")


def fixture_text(name: str) -> str:
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as fh:
        return fh.read()


def write_hrd_analysis(path: str, storm: str = "TEST", flight: str = "260925I1") -> None:
    """
    A small gridded TDR analysis in HRD's layout -- (x, y, level, time),
    -999.9 missing -- holding a Rankine vortex (60 m/s, RMW 20 km) about
    17N 109W and a reflectivity equal to x (km east).
    """
    import numpy as np
    from scipy.io import netcdf_file
    x = np.arange(-49.0, 50.0, 2.0)
    y = np.arange(-49.0, 50.0, 2.0)
    z = np.arange(0.0, 3.0, 0.5)
    f = netcdf_file(str(path), "w")
    for name, n in (("x", x.size), ("y", y.size), ("level", z.size), ("time", 1)):
        f.createDimension(name, n)
    for name, vals in (("x", x), ("y", y), ("level", z)):
        v = f.createVariable(name, "f4", (name,))
        v[:] = vals
    X, Y = np.meshgrid(x, y, indexing="ij")                # (x, y) like HRD
    R = np.hypot(X, Y)
    vt = np.where(R <= 20, 60 * R / 20, 60 * 20 / np.maximum(R, 1e-6))   # Rankine, RMW 20 km
    fields = {"TANGENTIAL_WIND": vt, "WIND_SPEED": vt, "RADIAL_WIND": 0 * vt,
              "REFLECTIVITY": X.copy(), "W": 0 * vt, "VORT": 0 * vt, "U": -vt * Y / np.maximum(R, 1e-6),
              "V": vt * X / np.maximum(R, 1e-6)}
    for name, a in fields.items():
        v = f.createVariable(name, "f4", ("x", "y", "level", "time"))
        vol = np.repeat(a[:, :, None], z.size, axis=2)[..., None].astype("f4")
        vol[0, 0, :, 0] = -999.9
        v[:] = vol
    for name, a in (("LATITUDE", 17 + Y / 111.0), ("LONGITUDE", -109 + X / 106.0)):
        v = f.createVariable(name, "f4", ("x", "y", "time"))
        v[:] = a[..., None]
    for k, val in {"STMNAME": storm, "FLTNAME": flight, "ORIGIN_LATITUDE": "17.0",
                   "ORIGIN_LONGITUDE": "-109.0", "FLIGHT YEAR": "2026", "FLIGHT MONTH": "9",
                   "FLIGHT DAY": "25", "ANALYSIS START HOUR": "21", "ANALYSIS START MINUTE": "54",
                   "ANALYSIS START SECOND": "0", "ANALYSIS END HOUR": "22", "ANALYSIS END MINUTE": "19",
                   "ANALYSIS END SECOND": "0", "ANALYSIS CENTER HOUR": "22", "ANALYSIS CENTER MINUTE": "11",
                   "ANALYSIS CENTER SECOND": "0"}.items():
        setattr(f, k, val)
    f.close()
