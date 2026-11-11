#!/usr/bin/env python3
"""
recon10s_syntdr — synthetic TDR: the 3-D wind field of a hurricane as the
P-3's tail Doppler radar would show it, built from the flight-level obs and
dropsondes of any aircraft (USAF or NOAA), using a model of TDR structure
learned from HRD's analyses. It is a statistical estimate, not a radar
observation, and says so everywhere it is drawn.

The model (scripts/models/syntdr_v1.npz, trained by training/train_syntdr.py)
  * State: storm-motion-relative tangential and radial wind, divided by Vm
    (the strongest azimuthal-mean tangential wind at 2 km), on nodes of
    rho = r / R (R: the radius of Vm), azimuth clockwise from the direction of
    motion, and height 0.5-12 km.
  * Prior learned from TDR analyses: the mean structure as a linear function
    of intensity, size, shear (along and across the motion), translation
    speed and latitude, plus EOFs of the departures from it (EM-PCA, which
    copes with the gaps every TDR analysis has), and the variance the EOFs
    leave unexplained at every node.
  * Observations y = Vm H(R) (mu + E a) + noise, H interpolating the state to
    each ob's storm-relative position and height. For a given R the problem
    is linear in b = Vm [1, a] and solved exactly (Gaussian posterior, so
    every grid point also gets an uncertainty); R is chosen by the marginal
    likelihood, with a prior centred on the radius of the observed peak.
  * Older obs are moved to the analysis time with the storm motion and scaled
    by the intensity trend of the recon's own passes, and count for less.
  * The intensity estimate is a regression on the peak 1-km wind of the
    synthetic field and the strongest flight-level wind the recon measured,
    fitted to NHC's best track on storms the prior never saw; the surface
    (10 m) map is the 1-km wind scaled so that its peak is that estimate.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "syntdr_v1.npz")
MS_TO_KT = 1.943844
KT_TO_MS = 1.0 / MS_TO_KT
UTC = dt.timezone.utc
KIND_FL, KIND_DROP = 0, 1


# ------------------------------------------------------------------ model

def cov_vector(vm: float, R: float, s_par: float = 0.0, s_perp: float = 0.0, speed: float = 5.0,
               lat: float = 25.0) -> np.ndarray:
    """Predictors of the prior mean: Vm (m/s), R (km), shear along/across the motion (kt), motion (m/s)."""
    a = (vm - 40.0) / 20.0
    b = (math.log(max(R, 3.0)) - math.log(40.0)) / 0.6
    c = s_par / 20.0
    d = s_perp / 20.0
    return np.array([1.0, a, b, c, d, (speed - 5.0) / 3.0, (abs(lat) - 25.0) / 8.0, a * c, a * d, a * b])


@dataclass
class Obs:
    """Observations relative to the moving centre: km east/north, height km, motion-relative wind (m/s)."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    u: np.ndarray
    v: np.ndarray
    sigma: np.ndarray
    kind: np.ndarray
    t: np.ndarray = field(default_factory=lambda: np.empty(0))      # epoch seconds
    group: np.ndarray = field(default_factory=lambda: np.empty(0, int))   # pass (flight level) or drop number

    def __len__(self):
        return int(self.x.size)

    def subset(self, keep: np.ndarray) -> "Obs":
        g = self.group if self.group.size == self.x.size else np.zeros(self.x.size, int)
        t = self.t if self.t.size == self.x.size else np.zeros(self.x.size)
        return Obs(self.x[keep], self.y[keep], self.z[keep], self.u[keep], self.v[keep], self.sigma[keep],
                   self.kind[keep], t[keep], g[keep])

    @staticmethod
    def concat(parts: Sequence["Obs"]) -> "Obs":
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            e = np.empty(0)
            return Obs(e, e, e, e, e, e, np.empty(0, int), e, np.empty(0, int))
        cat = lambda name: np.concatenate([getattr(p, name) for p in parts])
        return Obs(cat("x"), cat("y"), cat("z"), cat("u"), cat("v"), cat("sigma"), cat("kind"), cat("t"),
                   cat("group"))

    def vt_vr(self):
        r = np.hypot(self.x, self.y)
        th = np.arctan2(self.y, self.x)
        vt = -self.u * np.sin(th) + self.v * np.cos(th)
        vr = self.u * np.cos(th) + self.v * np.sin(th)
        return r, vt, vr


@dataclass
class Fit:
    R: float
    vm: float
    b: np.ndarray
    P: np.ndarray
    mu: np.ndarray
    log_evidence: float
    heading: float
    n_obs: int
    resid_rms: float
    evidence_by_R: Dict[float, float] = field(default_factory=dict)


class Model:
    """The trained prior and the reconstruction."""

    def __init__(self, path: Optional[str] = None):
        path = path or MODEL_PATH
        d = np.load(path, allow_pickle=False)
        self.path = path
        self.rho = d["rho"].astype(float)
        self.phi = d["phi"].astype(float)
        self.z = d["z"].astype(float)
        self.B = d["B"].astype(np.float64)
        self.E = d["E"].astype(np.float64)
        self.lam = d["lam"].astype(np.float64)
        self.s2 = d["s2"].astype(np.float64)
        hyper = json.loads(str(d["hyper"]))
        self.kappa = float(hyper.get("kappa", 0.5))
        self.gamma = float(hyper.get("gamma", 3.0))
        self.sigma_fl = float(hyper.get("sigma_fl", 2.0))
        self.sigma_drop = float(hyper.get("sigma_drop", 3.0))
        self.R_prior = tuple(hyper.get("R_prior", (math.log(35.0), 0.7)))
        self.R_obs_sd = float(hyper.get("R_obs_sd", 0.3))
        self.surface_ratio = float(hyper.get("surface_ratio", 0.91))
        self.surface_level = float(hyper.get("surface_level_km", 1.0))
        self.intensity_rmse_kt = float(hyper.get("intensity_rmse_kt", 8.0))
        # the posterior spread times these factors (by r / R) matches the errors seen on independent TDR analyses
        prof = hyper.get("spread_by_rho")
        if prof:
            self.spread_factor = np.interp(self.rho, np.asarray(prof[0], float), np.asarray(prof[1], float))
        else:
            self.spread_factor = np.full(self.rho.size, float(hyper.get("spread_scale", 1.0)))
        # best track ~ a * (peak 1-km wind) + b * (peak flight-level wind) + c + d * max(peak 1-km wind - hinge, 0)
        cal = hyper.get("intensity") or {}
        self.hinge_kt = float(cal.get("hinge_kt", 1e9))
        self.intensity_full = tuple(float(c) for c in cal.get("full", ())) or None
        self.intensity_pk_only = tuple(float(c) for c in cal.get("pk_only", ())) or None
        self.info = json.loads(str(d["info"])) if "info" in d else {}
        self.shape = (2, self.z.size, self.rho.size, self.phi.size)
        self._M_cache = None

    # ---- geometry
    def _interp_index(self, f: np.ndarray, z: np.ndarray, rho: np.ndarray, phi: np.ndarray):
        nz, nr, npf = self.z.size, self.rho.size, self.phi.size
        zz = np.clip(z, self.z[0], self.z[-1])
        kz = np.clip(np.searchsorted(self.z, zz) - 1, 0, nz - 2)
        wz = (zz - self.z[kz]) / (self.z[kz + 1] - self.z[kz])
        rr = np.clip(rho, self.rho[0], self.rho[-1])
        kr = np.clip(np.searchsorted(self.rho, rr) - 1, 0, nr - 2)
        wr = (rr - self.rho[kr]) / (self.rho[kr + 1] - self.rho[kr])
        dphi = 360.0 / npf
        fp = (np.asarray(phi) % 360.0) / dphi
        kp = np.floor(fp).astype(int) % npf
        wp = fp - np.floor(fp)
        kp1 = (kp + 1) % npf
        idx, wts = [], []
        for a, wa in ((kz, 1 - wz), (kz + 1, wz)):
            for b, wb in ((kr, 1 - wr), (kr + 1, wr)):
                for c, wc in ((kp, 1 - wp), (kp1, wp)):
                    idx.append(((f * nz + a) * nr + b) * npf + c)
                    wts.append(wa * wb * wc)
        return np.stack(idx, 1), np.stack(wts, 1)

    def _design(self, obs: Obs, R: float, heading: float, mu: np.ndarray, gamma: float):
        r, vt, vr = obs.vt_vr()
        az = (np.degrees(np.arctan2(obs.x, obs.y)) + 360.0) % 360.0
        phi = (az - heading) % 360.0
        y = np.concatenate([vt, vr])
        f = np.concatenate([np.zeros(len(obs), int), np.ones(len(obs), int)])
        idx, w = self._interp_index(f, np.concatenate([obs.z, obs.z]), np.concatenate([r, r]) / R,
                                    np.concatenate([phi, phi]))
        Hmu = (mu[idx] * w).sum(1)
        HE = np.einsum("nk,nkj->nj", w, self.E[idx])
        Hs2 = (self.s2[idx] * w).sum(1)
        sig = np.concatenate([obs.sigma, obs.sigma])
        kk = np.concatenate([obs.kind, obs.kind])
        sig = np.where(kk == KIND_FL, sig * math.sqrt(gamma), sig)
        ok = np.isfinite(y) & np.isfinite(Hmu)
        return y[ok], Hmu[ok], HE[ok], Hs2[ok], sig[ok]

    # ---- fitting
    def fit_R(self, obs: Obs, R: float, heading: float, cov_extra: dict, vm0: float, R_prior=None,
              kappa: Optional[float] = None, gamma: Optional[float] = None, iters: int = 4) -> Fit:
        kappa = self.kappa if kappa is None else kappa
        gamma = self.gamma if gamma is None else gamma
        vm = vm0
        k = self.E.shape[1]
        for _ in range(iters):
            c = cov_vector(vm, R, **cov_extra)
            mu = c @ self.B
            y, Hmu, HE, Hs2, sig = self._design(obs, R, heading, mu, gamma)
            A = np.column_stack([Hmu, HE])
            noise = sig ** 2 + (vm ** 2) * Hs2
            Winv = 1.0 / noise
            prior_var = np.concatenate([[(0.6 * vm) ** 2], (vm ** 2) * kappa * self.lam])
            prior_mean = np.concatenate([[vm], np.zeros(k)])
            Pinv = (A.T * Winv) @ A + np.diag(1.0 / prior_var)
            P = np.linalg.inv(Pinv)
            b = P @ ((A.T * Winv) @ y + prior_mean / prior_var)
            vm_new = float(max(5.0, b[0]))
            done = abs(vm_new - vm) < 0.2
            vm = vm_new
            if done:
                break
        r0 = y - A @ prior_mean
        Wr = r0 * Winv
        AtWr = A.T @ Wr
        quad = float(r0 @ Wr - AtWr @ P @ AtWr)
        _sign, logdet_Pinv = np.linalg.slogdet(Pinv)
        logdet = logdet_Pinv + np.sum(np.log(prior_var)) + np.sum(np.log(noise))
        loge = -0.5 * (quad + logdet + y.size * math.log(2 * math.pi))
        mu_lr, sd_lr = R_prior if R_prior is not None else self.R_prior
        loge += -0.5 * ((math.log(R) - mu_lr) / sd_lr) ** 2
        res = y - A @ b
        return Fit(R=R, vm=vm, b=b, P=P, mu=mu, log_evidence=loge, heading=heading, n_obs=int(y.size),
                   resid_rms=float(np.sqrt(np.mean(res ** 2))) if res.size else float("nan"))

    def fit(self, obs: Obs, heading: float, cov_extra: dict, R_obs: Optional[float] = None,
            vm0: Optional[float] = None, kappa: Optional[float] = None, gamma: Optional[float] = None) -> Fit:
        if len(obs) < 5:
            raise ValueError("too few observations for a synthetic analysis")
        _r, vt, _vr = obs.vt_vr()
        if vm0 is None:
            vm0 = float(np.nanpercentile(vt, 95)) if np.isfinite(vt).any() else 30.0
        vm0 = max(vm0, 8.0)
        R_prior = None
        if R_obs is not None and np.isfinite(R_obs) and R_obs > 0:
            m0, s0 = self.R_prior
            w0, w1 = 1 / s0 ** 2, 1 / self.R_obs_sd ** 2
            R_prior = ((m0 * w0 + math.log(R_obs) * w1) / (w0 + w1), (w0 + w1) ** -0.5)
        grid = np.exp(np.linspace(math.log(6.0), math.log(160.0), 28))
        fits = [self.fit_R(obs, float(R), heading, cov_extra, vm0, R_prior, kappa, gamma) for R in grid]
        j = int(np.argmax([f.log_evidence for f in fits]))
        best = fits[j]
        lo, hi = grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]
        for R in np.exp(np.linspace(math.log(lo), math.log(hi), 7)):
            f = self.fit_R(obs, float(R), heading, cov_extra, best.vm, R_prior, kappa, gamma)
            if f.log_evidence > best.log_evidence:
                best = f
        best.evidence_by_R = {round(float(f.R), 2): float(f.log_evidence) for f in fits}
        return best

    # ---- results
    def _M(self, fit: Fit) -> np.ndarray:
        return np.column_stack([fit.mu, self.E])

    def state(self, fit: Fit) -> np.ndarray:
        """Posterior mean, m/s, shape (2, nz, nrho, nphi): motion-relative Vt and Vr."""
        return (self._M(fit) @ fit.b).reshape(self.shape)

    def spread(self, fit: Fit) -> np.ndarray:
        """Posterior standard deviation (m/s) at every node, including what the EOFs leave out."""
        M = self._M(fit)
        var = np.einsum("ij,jk,ik->i", M, fit.P, M) + (fit.vm ** 2) * self.s2
        sd = np.sqrt(np.maximum(var, 0.0)).reshape(self.shape)
        return sd * self.spread_factor[None, None, :, None]

    def sample_many(self, fit: Fit, arrays: Sequence[Tuple[np.ndarray, int]], x: np.ndarray, y: np.ndarray,
                    z: np.ndarray) -> List[np.ndarray]:
        """Several (state-shaped array, field) pairs at the same points, interpolating once."""
        r = np.hypot(x, y)
        az = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
        phi = (az - fit.heading) % 360.0
        idx, w = self._interp_index(np.zeros(np.size(x), int), np.broadcast_to(z, np.shape(x)).ravel(),
                                    np.ravel(r) / fit.R, np.ravel(phi))
        per_field = self.z.size * self.rho.size * self.phi.size
        out = []
        for values, fld in arrays:
            out.append((values.ravel()[idx + fld * per_field] * w).sum(1).reshape(np.shape(x)))
        return out

    def intensity(self, peak_1km_kt: float, fl_max_kt: Optional[float]) -> float:
        """The calibrated intensity estimate (kt) from the synthetic 1-km peak and the recon's flight-level peak."""
        above = max(peak_1km_kt - self.hinge_kt, 0.0)
        if fl_max_kt is not None and np.isfinite(fl_max_kt) and self.intensity_full:
            a, b, c, d = self.intensity_full
            return float(a * peak_1km_kt + b * fl_max_kt + c + d * above)
        if self.intensity_pk_only:                      # no flight-level obs near the storm: the 1-km peak alone
            a, c, d = self.intensity_pk_only
            return float(a * peak_1km_kt + c + d * above)
        return float(self.surface_ratio * peak_1km_kt)

    def sample(self, fit: Fit, values: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
               fld: int) -> np.ndarray:
        """A state-shaped array (e.g. the posterior mean or spread) at storm-relative points (km)."""
        r = np.hypot(x, y)
        az = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
        phi = (az - fit.heading) % 360.0
        idx, w = self._interp_index(np.full(np.size(x), fld), np.broadcast_to(z, np.shape(x)).ravel(),
                                    np.ravel(r) / fit.R, np.ravel(phi))
        return (values.ravel()[idx] * w).sum(1).reshape(np.shape(x))


_MODEL: Optional[Model] = None


def model(path: Optional[str] = None) -> Model:
    global _MODEL
    path = path or MODEL_PATH
    if _MODEL is None or _MODEL.path != path:
        _MODEL = Model(path)
    return _MODEL


def observed_rmw(obs: Obs, bin_km: float = 4.0, sectors: int = 8) -> Optional[float]:
    """Median over azimuth sectors of the radius of the strongest smoothed tangential wind at flight level."""
    r, vt, _vr = obs.vt_vr()
    sel = np.isfinite(vt) & (obs.kind == KIND_FL)
    if sel.sum() < 10:
        return None
    az = (np.degrees(np.arctan2(obs.x, obs.y)) + 360.0) % 360.0
    sec = (az // (360.0 / sectors)).astype(int)
    out = []
    for s in range(sectors):
        m = sel & (sec == s)
        if m.sum() < 6:
            continue
        rb = np.floor(r[m] / bin_km).astype(int)
        nb = int(rb.max()) + 1
        sums = np.bincount(rb, weights=vt[m], minlength=nb)
        cnts = np.bincount(rb, minlength=nb)
        prof = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        if np.isfinite(prof).sum() < 5:
            continue
        smooth = np.convolve(np.nan_to_num(prof, nan=0.0), np.ones(3) / 3, "same")
        j = int(np.nanargmax(np.where(np.isfinite(prof), smooth, -np.inf)))
        if 0 < j < nb - 1:
            out.append((j + 0.5) * bin_km)
    return float(np.median(out)) if out else None


# ------------------------------------------------------------ observations

def _motion_ms(motion) -> Tuple[float, float]:
    if not motion:
        return 0.0, 0.0
    hd, kt = motion
    ms = kt * KT_TO_MS
    return ms * math.sin(math.radians(hd)), ms * math.cos(math.radians(hd))


def flight_level_obs(ov, fixes, motion, t0: dt.datetime, half_h: float, max_km: float = 220.0,
                     sigma: float = 2.0) -> Optional[Obs]:
    """Every flight-level ob within half_h of t0, relative to the moving centre, motion removed."""
    import recon10s_windfield as wf
    h = dt.timedelta(hours=half_h)
    sr = wf.storm_relative(ov, fixes, motion, t0 - h, t0 + h, max_km=max_km)
    if sr is None or not len(sr):
        return None
    heights = ov.column("heights")
    z = np.array([np.nan if heights[i] is None else heights[i] for i in sr.idx], float) / 1000.0
    ok = np.isfinite(sr.us) & np.isfinite(sr.vs) & np.isfinite(z) & (z >= 0.25) & (z <= 12.5) & np.isfinite(sr.x)
    if ok.sum() < 5:
        return None
    sig = np.full(ok.sum(), sigma)
    zz = z[ok]
    sig = np.where(zz < 0.5, sigma * 1.5, sig)          # a 1000-ft leg stands for the 0.5 km level
    segs = sr.segments if sr.segments.size == sr.t.size else np.zeros(sr.t.size, int)
    return Obs(x=sr.x[ok], y=sr.y[ok], z=np.maximum(zz, 0.5), u=sr.us[ok] * KT_TO_MS, v=sr.vs[ok] * KT_TO_MS,
               sigma=sig, kind=np.full(ok.sum(), KIND_FL), t=sr.t[ok], group=segs[ok])


def _drop_levels(d) -> List[Tuple[float, float, float]]:
    """(height m, direction, speed kt) of every wind level of a TEMP DROP, heights filled in from pressure."""
    known = [(lv.p, lv.z) for lv in (d.mandatory or []) if lv.z is not None and lv.p]
    if d.surface is not None and d.surface.p:
        known.append((d.surface.p, d.surface.z if d.surface.z is not None else 10.0))
    known.sort()
    out = []
    for lv in list(d.mandatory or []) + list(d.sig_wind or []):
        if lv.wspd is None or lv.wdir is None or not lv.p:
            continue
        z = lv.z
        if z is None and len(known) >= 2:
            ps = np.log([k[0] for k in known])
            zs = np.array([k[1] for k in known], float)
            order = np.argsort(ps)
            z = float(np.interp(math.log(lv.p), ps[order], zs[order]))
            if lv.p < known[0][0] or lv.p > known[-1][0]:
                z = None                                  # beyond the levels we can place
        if z is not None:
            out.append((float(z), float(lv.wdir), float(lv.wspd)))
    return out


def dropsonde_obs(drops, fixes, motion, t0: dt.datetime, half_h: float, sigma: float = 3.0) -> Optional[Obs]:
    """Dropsonde winds 0.3-12 km, each at its interpolated position along the fall."""
    import recon10s_windfield as wf
    mx, my = _motion_ms(motion)
    parts = []
    for n, d in enumerate(drops or []):
        if d.release_time is None or d.release_lat is None or abs((d.release_time - t0).total_seconds()) > half_h * 3600:
            continue
        lv = [l for l in _drop_levels(d) if 300.0 <= l[0] <= 12500.0]
        if not lv:
            continue
        z_rel = max(l[0] for l in lv) + 100.0
        s_lat = d.splash_lat if d.splash_lat is not None else d.release_lat
        s_lon = d.splash_lon if d.splash_lon is not None else d.release_lon
        xs, ys, zs, us, vs, ts = [], [], [], [], [], []
        for z, wdir, wspd in lv:
            f = min(1.0, max(0.0, z / z_rel))
            lat = s_lat + f * (d.release_lat - s_lat)
            lon = s_lon + f * (d.release_lon - s_lon)
            when = d.release_time + dt.timedelta(seconds=(z_rel - z) / 12.0)
            clat, clon = wf.centre_track(fixes, [when], motion)
            if not np.isfinite(clat[0]):
                continue
            x, y = wf.local_xy(clat[0], clon[0], lat, lon)
            spd = wspd * KT_TO_MS
            u = -spd * math.sin(math.radians(wdir)) - mx
            v = -spd * math.cos(math.radians(wdir)) - my
            xs.append(float(x)); ys.append(float(y)); zs.append(z / 1000.0); us.append(u); vs.append(v)
            ts.append(when.timestamp())
        if xs:
            k = len(xs)
            parts.append(Obs(np.array(xs), np.array(ys), np.maximum(np.array(zs), 0.5), np.array(us), np.array(vs),
                             np.full(k, sigma), np.full(k, KIND_DROP), np.array(ts), np.full(k, 1000 + n)))
    return Obs.concat(parts) if parts else None


def pass_peaks(bundle, t0: dt.datetime, half_h: float = 6.0) -> List[Tuple[float, float]]:
    """(epoch s, kt) of each recon pass's peak flight-level wind near t0."""
    out = []
    for p in getattr(bundle, "passes", None) or []:
        if p.wind is None or not np.isfinite(p.wind).any() or not p.times:
            continue
        k = int(np.nanargmax(p.wind))
        t = p.times[k]
        if abs((t - t0).total_seconds()) <= half_h * 3600:
            out.append((t.timestamp(), float(p.wind[k])))
    return sorted(out)


def intensity_trend(bundle, t0: dt.datetime) -> Optional[float]:
    """Change of the passes' peak flight-level wind, kt per hour (a weighted line), or None."""
    pk = pass_peaks(bundle, t0)
    if len(pk) < 2 or pk[-1][0] - pk[0][0] < 3600:
        return None
    t = np.array([p[0] for p in pk]); v = np.array([p[1] for p in pk])
    w = np.exp(-0.5 * ((t - t0.timestamp()) / (3 * 3600)) ** 2)
    tm = np.sum(w * t) / w.sum()
    vm = np.sum(w * v) / w.sum()
    den = np.sum(w * (t - tm) ** 2)
    if den <= 0:
        return None
    return float(np.sum(w * (t - tm) * (v - vm)) / den * 3600.0)


def apply_trend(obs: Obs, t0: dt.datetime, rate_kt_h: Optional[float], ref_kt: float, tau_h: float = 1.5):
    """Scale each ob's wind to the analysis time by the intensity trend, and trust older obs less."""
    if not len(obs):
        return obs
    dt_h = (obs.t - t0.timestamp()) / 3600.0
    if rate_kt_h is not None and ref_kt > 10:
        then = ref_kt + rate_kt_h * dt_h
        f = np.clip(ref_kt / np.maximum(then, 5.0), 0.85, 1.2)
        obs.u = obs.u * f
        obs.v = obs.v * f
    obs.sigma = obs.sigma * np.sqrt(1.0 + (dt_h / tau_h) ** 2)
    return obs


# --------------------------------------------------------------- products

@dataclass
class Result:
    grid: object                         # recon10s_tdrgrid.TdrGrid: WIND_SPEED, TANGENTIAL/RADIAL_WIND, U, V, VORT, WIND_SPREAD
    surface: np.ndarray                  # (ny, nx) 10-m wind estimate, kt
    surface_spread: np.ndarray           # (ny, nx) kt
    fit: Fit
    obs: Obs
    when: dt.datetime
    motion: Tuple[float, float]          # heading toward, kt
    shear: Optional[Tuple[float, float]]  # kt, heading toward
    peak_kt: float                       # peak of the surface map = the intensity estimate
    peak_xy: Tuple[float, float]
    intensity_kt: float
    intensity_pm_kt: float
    trend_kt_h: Optional[float]
    peak_1km_kt: float = float("nan")    # the earth-relative 1-km peak of the synthetic field
    fl_max_kt: Optional[float] = None    # the strongest flight-level wind the recon measured (earth-relative)
    surface_scale: float = float("nan")  # surface map / 1-km wind
    rmw_km: float = float("nan")         # radius of the azimuthal-mean 1-km tangential wind maximum
    offset_kt: float = 0.0               # correction learnt from this user's matched best-track pairs
    core_sampled: bool = True            # an ob within 1.5 RMW (and 50 km) of the centre, the RMW ring seen from 2+ sides
    ob_distance: Optional[np.ndarray] = None   # (ny, nx) km from each grid point to the nearest ob
    model_path: str = ""                 # the model file the analysis was built with
    storm: str = ""
    cov_extra: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    verification: Optional[dict] = None

    @property
    def peak_where(self) -> str:
        x, y = self.peak_xy
        d = math.hypot(x, y)
        brg = (math.degrees(math.atan2(x, y)) + 360) % 360
        names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        return f"{d:.0f} km {names[int((brg + 11.25) // 22.5) % 16]} of centre"

    @property
    def n_fl(self) -> int:
        return int((self.obs.kind == KIND_FL).sum())

    @property
    def n_drop(self) -> int:
        return int(len(set(self.obs.group[self.obs.kind == KIND_DROP].tolist()))) if self.obs.group.size else 0


MAX_RADIUS_KM = 200.0                    # the TDR analyses the prior learnt from reach 200 km


def _grid_from_fit(m: Model, fit: Fit, centre: Tuple[float, float], when: dt.datetime, motion,
                   half_km: float = 200.0, dx: float = 2.0, storm: str = "", window=None):
    import recon10s_tdrgrid as tg
    x = np.arange(-half_km, half_km + dx / 2, dx)
    y = x.copy()
    z = m.z.copy()
    X, Y = np.meshgrid(x, y)
    st = m.state(fit)
    sp = m.spread(fit)
    shape = (z.size, y.size, x.size)
    Xb = np.broadcast_to(X, shape)
    Yb = np.broadcast_to(Y, shape)
    Zb = np.broadcast_to(z[:, None, None], shape)
    vt, vr, svt, svr = m.sample_many(fit, [(st, 0), (st, 1), (sp, 0), (sp, 1)], Xb, Yb, Zb)
    R = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    us = vr * np.cos(th) - vt * np.sin(th)
    vs = vr * np.sin(th) + vt * np.cos(th)
    mx, my = _motion_ms(motion)
    U, V = us + mx, vs + my
    # vertical vorticity, 10^-3 s^-1 (m/s per km)
    vort = np.gradient(V, dx, axis=2) - np.gradient(U, dx, axis=1)
    # beyond the radius the model was trained to (8 R, and the analyses' 200 km) there is nothing: blank it
    far = ((R > 8.0 * fit.R) | (R > MAX_RADIUS_KM))[None].repeat(z.size, 0)
    for a in (vt, vr, U, V, svt, svr, vort):
        a[far] = np.nan
    lat0, lon0 = centre
    lat = lat0 + Y / 111.2
    lon = lon0 + X / (111.2 * math.cos(math.radians(lat0)))
    fields = {"WIND_SPEED": np.hypot(U, V), "TANGENTIAL_WIND": vt, "RADIAL_WIND": vr, "U": U, "V": V,
              "VORT": vort, "WIND_SPREAD": np.hypot(svt, svr) / math.sqrt(2.0)}
    start, end = window if window else (when, when)
    g = tg.TdrGrid(name=f"SYNTHETIC {storm} {when:%Y%m%d %H%M}", storm=storm or "", flight="SYNTHETIC", x=x, y=y,
                   z=z, lat=lat, lon=lon, fields={k: v.astype(np.float32) for k, v in fields.items()},
                   centre=(float(lat0), float(lon0)), start=start, end=end, centre_time=when, motion=(mx, my))
    g.source = "synthetic"
    return g


def fl_peak_kt(obs: Obs, motion, max_km: float = 160.0) -> Optional[float]:
    """The strongest earth-relative flight-level wind among the obs (kt), as the calibration used it."""
    if obs is None or not len(obs):
        return None
    mx, my = _motion_ms(motion)
    sel = (obs.kind == KIND_FL) & (np.hypot(obs.x, obs.y) <= max_km)
    if not sel.any():
        return None
    sp = np.hypot(obs.u[sel] + mx, obs.v[sel] + my) * MS_TO_KT
    return float(np.nanmax(sp)) if np.isfinite(sp).any() else None


def synthesize(bundle, when: dt.datetime, half_h: float = 2.0, motion=None, shear=None, lat: Optional[float] = None,
               use_drops: bool = True, use_trend: bool = True, storm: str = "", model_path: Optional[str] = None,
               half_km: float = 200.0, offset_kt: float = 0.0) -> Result:
    """
    The synthetic TDR analysis valid at `when` from the recon in `bundle` (a ReconBundle: overlay,
    fixes, drops, passes). `motion` (heading toward, kt) is required; `shear` (kt, heading toward)
    is optional (SHIPS), as is the latitude (taken from the fixes). `offset_kt` is added to the
    calibrated intensity (the correction LiveCalibration has learnt).
    """
    import recon10s_windfield as wf
    m = model(model_path)
    if motion is None:
        raise ValueError("the storm motion is needed (recon fixes, NHC advisory, best track or typed in)")
    fixes = list(getattr(bundle, "fixes", None) or [])
    if not fixes:
        raise ValueError("no centre fixes: load vortex messages or derive fixes from the obs")
    if getattr(bundle, "overlay", None) is None:
        raise ValueError("no flight-level obs loaded")
    notes = []
    fl = flight_level_obs(bundle.overlay, fixes, motion, when, half_h)
    dr = dropsonde_obs(getattr(bundle, "drops", None), fixes, motion, when, half_h) if use_drops else None
    obs = Obs.concat([fl, dr])
    if len(obs) < 10:
        raise ValueError(f"only {len(obs)} usable observations within {half_h:g} h of {when:%d/%H%MZ}")
    fl_max = fl_peak_kt(obs, motion)
    rate = intensity_trend(bundle, when) if use_trend else None
    peaks = pass_peaks(bundle, when)
    ref = float(np.median([p[1] for p in peaks])) if peaks else 0.0
    obs = apply_trend(obs, when, rate, ref)
    if rate is not None:
        notes.append(f"Older obs scaled by the passes' trend, {rate:+.1f} kt/h.")
    clat, clon = wf.centre_track(fixes, [when], motion)
    if not np.isfinite(clat[0]):
        raise ValueError("no centre position at that time")
    lat = abs(clat[0]) if lat is None else abs(lat)
    hd = float(motion[0]) % 360.0
    mx, my = _motion_ms(motion)
    if shear is not None and shear[0] is not None:
        rel = math.radians(shear[1] - hd)
        s_par, s_perp = shear[0] * math.cos(rel), shear[0] * math.sin(rel)
    else:
        s_par = s_perp = 0.0
        notes.append("No shear given: the prior's asymmetry is the unsheared one.")
    cov_extra = dict(s_par=s_par, s_perp=s_perp, speed=math.hypot(mx, my), lat=lat)
    fit = m.fit(obs, hd, cov_extra, R_obs=observed_rmw(obs))
    times = obs.t[np.isfinite(obs.t)]
    window = ((dt.datetime.fromtimestamp(times.min(), UTC), dt.datetime.fromtimestamp(times.max(), UTC))
              if times.size else (when, when))
    grid = _grid_from_fit(m, fit, (float(clat[0]), float(clon[0])), when, motion, half_km=half_km, storm=storm,
                          window=window)
    k = grid.level(m.surface_level)
    ws1 = grid.fields["WIND_SPEED"][k] * MS_TO_KT
    if not np.isfinite(ws1).any():
        raise ValueError("the synthetic field is empty at 1 km")
    i = np.unravel_index(int(np.nanargmax(ws1)), ws1.shape)
    pk1 = float(ws1[i])
    peak_xy = (float(grid.x[i[1]]), float(grid.y[i[0]]))
    intensity = m.intensity(pk1, fl_max) + offset_kt
    scale = float(np.clip(intensity / pk1, 0.6, 1.25)) if pk1 > 0 else m.surface_ratio
    surface = scale * ws1
    spread = scale * grid.fields["WIND_SPREAD"][k] * MS_TO_KT
    _z, rmw = grid.rmw()
    core = core_coverage(obs, fit.R)
    ob_dist = nearest_ob_km(obs, grid.x, grid.y)
    if not core[2]:
        notes.append(f"The obs come no closer than {core[0]:.0f} km to the centre (RMW {fit.R:.0f} km, seen from "
                     f"{core[1]} of 8 sides): the core and the peak are the model's guess.")
    if fl_max is None:
        notes.append("No flight-level obs within 160 km: the intensity comes from the synthetic field alone.")
    res = Result(grid=grid, surface=surface.astype(np.float32), surface_spread=spread.astype(np.float32), fit=fit,
                 obs=obs, when=when, motion=(float(motion[0]), float(motion[1])),
                 shear=(float(shear[0]), float(shear[1])) if shear is not None and shear[0] is not None else None,
                 peak_kt=float(np.nanmax(surface)), peak_xy=peak_xy, intensity_kt=float(np.nanmax(surface)),
                 intensity_pm_kt=m.intensity_rmse_kt, trend_kt_h=rate, peak_1km_kt=pk1, fl_max_kt=fl_max,
                 surface_scale=scale, rmw_km=float(rmw[k]) if np.isfinite(rmw[k]) else float("nan"),
                 offset_kt=offset_kt, core_sampled=core[2], ob_distance=ob_dist, model_path=m.path, storm=storm,
                 cov_extra=cov_extra, notes=notes)
    return res


def nearest_ob_km(obs: Obs, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance (km) from every point of the (y, x) grid to the nearest ob, storm-relative."""
    X, Y = np.meshgrid(x, y)
    if obs is None or not len(obs):
        return np.full(X.shape, np.inf, np.float32)
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(np.c_[obs.x, obs.y]).query(np.c_[X.ravel(), Y.ravel()])
        return d.reshape(X.shape).astype(np.float32)
    except ImportError:                                   # pragma: no cover - scipy ships with the app
        best = np.full(X.size, np.inf)
        for k in range(0, len(obs), 256):
            dx = X.ravel()[:, None] - obs.x[None, k:k + 256]
            dy = Y.ravel()[:, None] - obs.y[None, k:k + 256]
            best = np.minimum(best, np.sqrt(dx * dx + dy * dy).min(1))
        return best.reshape(X.shape).astype(np.float32)


def core_coverage(obs: Obs, R: float, limit_km: float = 50.0) -> Tuple[float, int, bool]:
    """
    (closest ob, km; sectors of 8 with an ob within 0.5-1.5 R; whether the core counts as sampled:
    an ob within 1.5 R and 50 km of the centre, and the RMW ring seen from two sides or more).
    On storms the model had not seen, the calibrated intensity was off by 5.7 kt on average when the
    core was sampled and 8.9 kt when it was not.
    """
    if obs is None or not len(obs):
        return float("inf"), 0, False
    r = np.hypot(obs.x, obs.y)
    az = (np.degrees(np.arctan2(obs.x, obs.y)) + 360.0) % 360.0
    ring = (r >= 0.5 * R) & (r <= 1.5 * R)
    sectors = len(set((az[ring] // 45.0).astype(int).tolist()))
    rmin = float(r.min())
    return rmin, sectors, bool(rmin <= min(1.5 * R, limit_km) and sectors >= 2)


# ------------------------------------------------------------- verification

def predict_obs(m: Model, fit: Fit, obs: Obs, motion) -> np.ndarray:
    """Earth-relative wind speed (m/s) the synthetic field gives at each ob's place."""
    st = m.state(fit)
    vt = m.sample(fit, st, obs.x, obs.y, obs.z, 0)
    vr = m.sample(fit, st, obs.x, obs.y, obs.z, 1)
    th = np.arctan2(obs.y, obs.x)
    us = vr * np.cos(th) - vt * np.sin(th)
    vs = vr * np.sin(th) + vt * np.cos(th)
    mx, my = _motion_ms(motion)
    return np.hypot(us + mx, vs + my)


def leave_one_out(result: Result, model_path: Optional[str] = None, min_obs: int = 15) -> Optional[dict]:
    """
    Each flight-level pass (aircraft track segment) left out in turn: how well the field built from
    everything else predicts it. Returns MAE, bias and RMSE (kt) of the wind speed, overall and per pass.
    """
    m = model(model_path or result.model_path or None)
    obs = result.obs
    fl = obs.kind == KIND_FL
    if obs.t.size != len(obs) or fl.sum() < 2 * min_obs:
        return None
    passes = split_passes(obs, min_obs)
    if len(passes) < 2:
        return None
    mx, my = _motion_ms(result.motion)
    hd = result.fit.heading
    rows, all_err = [], []
    for p in passes:
        keep = np.ones(len(obs), bool)
        keep[p] = False
        train = obs.subset(keep)
        test = obs.subset(~keep)
        try:
            f = m.fit(train, hd, result.cov_extra, R_obs=observed_rmw(train))
        except ValueError:
            continue
        pred = predict_obs(m, f, test, result.motion)
        truth = np.hypot(test.u + mx, test.v + my)
        err = (pred - truth) * MS_TO_KT
        err = err[np.isfinite(err)]
        if not err.size:
            continue
        all_err.append(err)
        rows.append({"start": float(obs.t[p].min()), "end": float(obs.t[p].max()), "n": int(err.size),
                     "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err)),
                     "peak_obs": float(np.nanmax(truth) * MS_TO_KT), "peak_pred": float(np.nanmax(pred) * MS_TO_KT)})
    if not all_err:
        return None
    e = np.concatenate(all_err)
    return {"passes": rows, "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "n": int(e.size)}


def split_passes(obs: Obs, min_obs: int = 15) -> List[np.ndarray]:
    """
    The flight-level obs (indices) cut into passes: at gaps over 20 minutes, changes of aircraft
    segment, and the outer turns of the pattern (the farthest points between centre crossings), so
    that each pass is an inbound and an outbound leg.
    """
    fl = np.flatnonzero(obs.kind == KIND_FL)
    if obs.t.size != len(obs) or fl.size < 2 * min_obs:
        return []
    order = fl[np.argsort(obs.t[fl], kind="stable")]
    t = obs.t[order]
    g = obs.group[order] if obs.group.size == len(obs) else np.zeros(order.size, int)
    r = np.hypot(obs.x[order], obs.y[order])
    cuts = {0, order.size}
    cuts |= set((np.flatnonzero((np.diff(t) > 1200) | (np.diff(g) != 0)) + 1).tolist())
    if r.size > 20:
        rs = np.convolve(r, np.ones(5) / 5.0, "same")
        w, top = 8, float(rs.max())
        for i in range(w, rs.size - w):
            if rs[i] >= 0.5 * top and rs[i] == rs[i - w:i + w + 1].max():
                cuts.add(i)
    cuts = sorted(cuts)
    out: List[np.ndarray] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = order[a:b]
        if not seg.size:
            continue
        if out and seg.size < min_obs:
            out[-1] = np.concatenate([out[-1], seg])
        else:
            out.append(seg)
    return [seg for seg in out if seg.size >= min_obs]


def verify_against(previous: Result, new_obs: Obs) -> Optional[dict]:
    """How the earlier analysis does on obs it had not seen (a new pass): MAE and bias in kt."""
    if previous is None or new_obs is None or not len(new_obs):
        return None
    m = model(previous.model_path or None)
    pred = predict_obs(m, previous.fit, new_obs, previous.motion)
    mx, my = _motion_ms(previous.motion)
    truth = np.hypot(new_obs.u + mx, new_obs.v + my)
    err = (pred - truth) * MS_TO_KT
    err = err[np.isfinite(err)]
    if not err.size:
        return None
    return {"n": int(err.size), "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err)),
            "rmse": float(np.sqrt(np.mean(err ** 2)))}


# --------------------------------------------------- learning from live recon

class LiveCalibration:
    """
    What the user's own sessions teach the model, kept in syntdr_live.json
    in the user folder:
      * every intensity estimate, later matched with NHC's best track at the
        same time (interpolated between its 6-hourly points, never
        extrapolated); the mean error of the matched pairs becomes a
        correction added to future estimates, shrunk toward zero so that 20
        pairs count as much as the training;
      * every new pass scored against the analysis made before it (a
        forecast of winds the model had not seen);
      * every comparison with a real HRD TDR analysis of the same storm.
    """

    PRIOR_PAIRS = 20.0

    def __init__(self, path: Optional[str] = None):
        if path is None:
            try:
                import recon10s
                path = os.path.join(recon10s.user_dir(), "syntdr_live.json")
            except Exception:
                path = os.path.join(HERE, "syntdr_live.json")
        self.path = path
        self.data = {"intensity": [], "passes": [], "tdr": []}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                got = json.load(fh)
            if isinstance(got, dict):
                for k in self.data:
                    if isinstance(got.get(k), list):
                        self.data[k] = got[k]
        except (OSError, ValueError):
            pass

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path + ".part", "w", encoding="utf-8") as fh:
                json.dump(self.data, fh)
            os.replace(self.path + ".part", self.path)
        except OSError:
            pass

    @staticmethod
    def _slot(when: dt.datetime) -> str:
        """Estimates are kept one per storm per half hour."""
        t = when.astimezone(UTC).replace(second=0, microsecond=0)
        return t.replace(minute=30 * (t.minute // 30)).isoformat()

    def add_estimate(self, storm: str, when: dt.datetime, intensity_kt: float, offset_kt: float,
                     peak_1km_kt: float, fl_max_kt: Optional[float]):
        """A synthetic intensity estimate, to be matched with the best track when it reaches that time."""
        if not storm or not np.isfinite(intensity_kt):
            return
        key = (storm, self._slot(when))
        old = next((r for r in self.data["intensity"] if (r["storm"], r["slot"]) == key), None)
        row = {"storm": storm, "slot": key[1], "time": when.astimezone(UTC).isoformat(),
               "synthetic": float(intensity_kt), "offset": float(offset_kt), "peak_1km": float(peak_1km_kt),
               "fl_max": None if fl_max_kt is None else float(fl_max_kt),
               "best_track": old.get("best_track") if old else None}
        self.data["intensity"] = [r for r in self.data["intensity"] if (r["storm"], r["slot"]) != key] + [row]
        self.data["intensity"] = self.data["intensity"][-5000:]
        self.save()

    def match_track(self, storm: str, track) -> int:
        """Fill in the best-track intensity of this storm's estimates that the track now brackets."""
        pts = sorted((p.time, p.vmax) for p in (track or []) if getattr(p, "vmax", None) is not None)
        if len(pts) < 2:
            return 0
        ts = np.array([p[0].timestamp() for p in pts])
        vs = np.array([p[1] for p in pts], float)
        n = 0
        for r in self.data["intensity"]:
            if r["storm"] != storm:
                continue
            t = dt.datetime.fromisoformat(r["time"]).timestamp()
            j = int(np.searchsorted(ts, t))
            if j == 0 or j >= ts.size or ts[j] - ts[j - 1] > 6.5 * 3600:
                if not (j < ts.size and ts[j] == t):
                    continue
            bt = float(np.interp(t, ts, vs))
            if r.get("best_track") != bt:
                r["best_track"] = bt
                n += 1
        if n:
            self.save()
        return n

    def matched(self, storm: str = "") -> List[dict]:
        return [r for r in self.data["intensity"] if r.get("best_track") is not None and (not storm or r["storm"] == storm)]

    def offset(self) -> Tuple[float, int]:
        """The correction (kt) to add to the calibrated estimate, and the number of matched pairs behind it."""
        rows = self.matched()
        if not rows:
            return 0.0, 0
        raw_err = np.array([r["best_track"] - (r["synthetic"] - r.get("offset", 0.0)) for r in rows])
        n = len(rows)
        return float(n / (self.PRIOR_PAIRS + n) * raw_err.mean()), n

    def add_pass(self, storm: str, when: dt.datetime, stats: dict):
        self.data["passes"].append({"storm": storm, "time": when.astimezone(UTC).isoformat(), **stats})
        self.data["passes"] = self.data["passes"][-2000:]
        self.save()

    def add_tdr(self, storm: str, when: dt.datetime, stats: dict):
        key = (storm, stats.get("analysis", ""))
        self.data["tdr"] = [r for r in self.data["tdr"] if (r["storm"], r.get("analysis", "")) != key]
        self.data["tdr"].append({"storm": storm, "time": when.astimezone(UTC).isoformat(), **stats})
        self.data["tdr"] = self.data["tdr"][-2000:]
        self.save()

    def summary(self, storm: str = "") -> str:
        bits = []
        rows = self.matched(storm)
        if rows:
            e = np.array([r["synthetic"] - r["best_track"] for r in rows])
            bits.append(f"{len(rows)} estimate(s) matched with NHC's best track: bias {e.mean():+.1f} kt, "
                        f"MAE {np.abs(e).mean():.1f} kt, {np.mean(np.abs(e) <= 10) * 100:.0f}% within 10 kt")
        ps = [r for r in self.data["passes"] if not storm or r["storm"] == storm]
        if ps:
            bits.append(f"{len(ps)} new pass(es) scored against the analysis before them: MAE "
                        f"{np.mean([p['mae'] for p in ps]):.1f} kt, bias {np.mean([p['bias'] for p in ps]):+.1f} kt")
        ts = [r for r in self.data["tdr"] if not storm or r["storm"] == storm]
        if ts:
            bits.append(f"{len(ts)} comparison(s) with HRD's TDR: wind RMSE "
                        f"{np.mean([t['rmse'] for t in ts]):.1f} kt at {ts[-1].get('level_km', 2):g} km")
        return "; ".join(bits)


# ------------------------------------------------------- against real TDR

def compare_tdr(result: Result, grid, level_km: float = 2.0) -> Optional[dict]:
    """
    The synthetic field against a real HRD TDR analysis (a TdrGrid with winds): the wind speed at
    one height on the synthetic grid's points (placed by latitude and longitude, so each grid keeps
    its own centre), RMSE and bias in kt, and both grids' peaks.
    """
    import recon10s_windfield as wf
    if grid is None or not grid.has_winds() or "WIND_SPEED" not in grid.fields:
        return None
    s = result.grid
    ks, kr = s.level(level_km), grid.level(level_km)
    if abs(grid.z[kr] - level_km) > 0.3:
        return None
    xs, ys = wf.local_xy(grid.centre[0], grid.centre[1], s.lat, s.lon)
    dxr, dyr = grid.x[1] - grid.x[0], grid.y[1] - grid.y[0]
    ii = np.rint((np.asarray(ys) - grid.y[0]) / dyr).astype(int)
    jj = np.rint((np.asarray(xs) - grid.x[0]) / dxr).astype(int)
    inside = (ii >= 0) & (ii < grid.y.size) & (jj >= 0) & (jj < grid.x.size)
    real = np.full(s.lat.shape, np.nan)
    real[inside] = grid.fields["WIND_SPEED"][kr][ii[inside], jj[inside]]
    syn = s.fields["WIND_SPEED"][ks]
    ok = np.isfinite(real) & np.isfinite(syn)
    if ok.sum() < 50:
        return None
    err = (syn[ok] - real[ok]) * MS_TO_KT
    return {"analysis": getattr(grid, "name", ""), "level_km": float(s.z[ks]), "n": int(ok.sum()),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(np.mean(err)),
            "peak_synthetic": float(np.nanmax(syn[ok]) * MS_TO_KT), "peak_tdr": float(np.nanmax(real[ok]) * MS_TO_KT),
            "corr": float(np.corrcoef(syn[ok], real[ok])[0, 1]) if ok.sum() > 2 else float("nan")}


# --------------------------------------------------------------- self-check

def self_check(path: Optional[str] = None) -> str:
    """Load the model and recover a vortex drawn from its own prior (a figure-four at 3 km): for --smoke-test."""
    m = Model(path)
    R, vm = 30.0, 40.0
    cov = dict(s_par=0.0, s_perp=0.0, speed=5.0, lat=20.0)
    k = m.E.shape[1]
    truth = Fit(R=R, vm=vm, b=np.r_[vm, np.zeros(k)], P=np.zeros((k + 1, k + 1)), mu=cov_vector(vm, R, **cov) @ m.B,
                log_evidence=0.0, heading=0.0, n_obs=0, resid_rms=0.0)
    s = m.state(truth)
    d = np.arange(-150.0, 151.0, 3.0)
    xs, ys = [], []
    for hd in (0.0, 45.0, 90.0, 135.0):
        a = math.radians(hd)
        xs.append(d * math.sin(a))
        ys.append(d * math.cos(a))
    x, y = np.concatenate(xs), np.concatenate(ys)
    z = np.full(x.size, 3.0)
    vt, vr = m.sample_many(truth, [(s, 0), (s, 1)], x, y, z)
    th = np.arctan2(y, x)
    u = vr * np.cos(th) - vt * np.sin(th)
    v = vr * np.sin(th) + vt * np.cos(th)
    ok = np.isfinite(u) & np.isfinite(v)
    obs = Obs(x[ok], y[ok], z[ok], u[ok], v[ok], np.full(ok.sum(), 2.0), np.full(ok.sum(), KIND_FL),
              np.zeros(ok.sum()), np.zeros(ok.sum(), int))
    f = m.fit(obs, 0.0, cov, R_obs=observed_rmw(obs))
    n = m.info.get("n_analyses") or m.info.get("n")
    return (f"synthetic TDR ok (model v{m.info.get('version', 1)}{f', {n} TDR analyses' if n else ''}; "
            f"test vortex Vm {vm:.0f} -> {f.vm:.1f} m/s, R {R:.0f} -> {f.R:.0f} km)")
