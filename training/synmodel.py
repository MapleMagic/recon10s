"""
Synthetic TDR: a statistical model of the 3-D vortex learned from TDR analyses, constrained by
in-situ observations.

State: motion-relative tangential and radial wind / Vm on (field, z, rho, phi) nodes, rho = r / R,
phi = azimuth clockwise from the storm motion. Prior: x = mu(c) + E a, mu(c) = c . B (covariate
regression per node), a ~ N(0, diag(lam) * kappa); unexplained variance s2 per node.
Observations: y = Vm * H(R) x + noise. With b = Vm * [1, a] the problem is linear in b for a given
R; R is chosen by the marginal likelihood (times a broad prior), Vm and the covariates iterated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

COV_NAMES = ("one", "vm", "lnR", "s_par", "s_perp", "speed", "lat", "vm_s_par", "vm_s_perp", "vm_lnR")


def cov_vector(vm: float, R: float, s_par: float = 0.0, s_perp: float = 0.0, speed: float = 5.0,
               lat: float = 25.0) -> np.ndarray:
    a = (vm - 40.0) / 20.0
    b = (math.log(max(R, 3.0)) - math.log(40.0)) / 0.6
    c = s_par / 20.0
    d = s_perp / 20.0
    return np.array([1.0, a, b, c, d, (speed - 5.0) / 3.0, (lat - 25.0) / 8.0, a * c, a * d, a * b])


@dataclass
class Grid:
    rho: np.ndarray
    phi: np.ndarray
    z: np.ndarray
    nf: int = 2

    @property
    def shape(self):
        return (self.nf, self.z.size, self.rho.size, self.phi.size)

    @property
    def size(self):
        return int(np.prod(self.shape))


def _interp_index(grid: Grid, f: np.ndarray, z: np.ndarray, rho: np.ndarray, phi: np.ndarray):
    """Trilinear (z, rho) x circular (phi) weights: (n, 8) indices into the flat state and weights."""
    nz, nr, npf = grid.z.size, grid.rho.size, grid.phi.size
    zz = np.clip(z, grid.z[0], grid.z[-1])
    kz = np.clip(np.searchsorted(grid.z, zz) - 1, 0, nz - 2)
    wz = (zz - grid.z[kz]) / (grid.z[kz + 1] - grid.z[kz])
    rr = np.clip(rho, grid.rho[0], grid.rho[-1])
    kr = np.clip(np.searchsorted(grid.rho, rr) - 1, 0, nr - 2)
    wr = (rr - grid.rho[kr]) / (grid.rho[kr + 1] - grid.rho[kr])
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


@dataclass
class Obs:
    """Storm-relative observations: position (km east/north of the centre, height km) and
    motion-relative wind components (m/s), with an error (m/s)."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    u: np.ndarray
    v: np.ndarray
    sigma: np.ndarray
    kind: Optional[np.ndarray] = None       # 0 flight level, 1 dropsonde, ...

    def __len__(self):
        return self.x.size

    def vt_vr(self):
        r = np.hypot(self.x, self.y)
        th = np.arctan2(self.y, self.x)                 # math angle
        vt = -self.u * np.sin(th) + self.v * np.cos(th)
        vr = self.u * np.cos(th) + self.v * np.sin(th)
        return r, vt, vr


@dataclass
class Fit:
    R: float
    vm: float
    b: np.ndarray                     # posterior mean of Vm*[1, a]
    P: np.ndarray                     # posterior covariance of b
    mu: np.ndarray                    # prior mean (normalised) used
    log_evidence: float
    heading: float
    cov: np.ndarray
    n_obs: int
    resid_rms: float
    evidence_by_R: Dict[float, float] = field(default_factory=dict)


class SynModel:
    def __init__(self, grid: Grid, B: np.ndarray, E: np.ndarray, lam: np.ndarray, s2: np.ndarray,
                 kappa: float = 1.0, sigma_fl: float = 2.0, sigma_drop: float = 3.0, R_prior=(math.log(35.0), 0.7)):
        self.grid, self.B, self.E, self.lam, self.s2 = grid, B, E, lam, s2
        self.kappa, self.sigma_fl, self.sigma_drop = kappa, sigma_fl, sigma_drop
        self.R_prior = R_prior
        self.gamma = 1.0           # error-variance inflation for flight-level obs (correlated along a leg)

    # ------------------------------------------------------------------ fitting the prior
    @classmethod
    def train(cls, X: np.ndarray, C: np.ndarray, grid: Grid, k: int = 60, ridge: float = 3.0, iters: int = 40,
              seed: int = 0, progress=print) -> "SynModel":
        n = X.shape[0]
        Xf = X.reshape(n, -1).astype(np.float64)
        M = np.isfinite(Xf)
        X0 = np.where(M, Xf, 0.0)
        q = C.shape[1]
        A = np.einsum("ip,iq,ij->jpq", C, C, M.astype(np.float64), optimize=True)
        reg = ridge * np.eye(q)
        reg[0, 0] = 1e-6
        A += reg[None]
        rhs = np.einsum("ip,ij->jp", C, X0, optimize=True)
        B = np.linalg.solve(A, rhs[..., None])[..., 0].T          # (q, p)
        cnt = M.sum(0)
        B[:, cnt < 20] = 0.0                       # too few samples: leave the node to the EOFs
        R = np.where(M, Xf - C @ B, 0.0)
        # EM-PCA on the residuals with missing values
        rng = np.random.default_rng(seed)
        F = R.copy()
        prev = None
        for it in range(iters):
            U, S, Vt = _rsvd(F, k, rng)
            approx = (U * S) @ Vt
            F = np.where(M, R, approx)
            err = float(np.sqrt(np.mean((R - approx)[M] ** 2)))
            if progress and (it % 5 == 0 or it == iters - 1):
                progress(f"  EM-PCA iter {it}: rms misfit on observed {err:.4f}")
            if prev is not None and abs(prev - err) < 1e-5:
                break
            prev = err
        E = Vt.T                                       # (p, k), orthonormal columns
        coef = U * S                                   # (n, k)
        lam = (coef ** 2).mean(0)
        resid = np.where(M, R - coef @ Vt, np.nan)
        with np.errstate(invalid="ignore"):
            s2 = np.nanmean(resid ** 2, axis=0)
        s2 = np.where(np.isfinite(s2), s2, np.nanmedian(s2))
        return cls(grid, B.astype(np.float32), E.astype(np.float32), lam.astype(np.float64), s2.astype(np.float32))

    # ------------------------------------------------------------------ reconstruction
    def prior_mean(self, c: np.ndarray) -> np.ndarray:
        return (c @ self.B).astype(np.float64)

    def _design(self, obs: Obs, R: float, heading: float, mu: np.ndarray):
        r, vt, vr = obs.vt_vr()
        rho = r / R
        az = (np.degrees(np.arctan2(obs.x, obs.y)) + 360.0) % 360.0
        phi = (az - heading) % 360.0
        y = np.concatenate([vt, vr])
        f = np.concatenate([np.zeros(len(obs), int), np.ones(len(obs), int)])
        zz = np.concatenate([obs.z, obs.z])
        idx, w = _interp_index(self.grid, f, zz, np.concatenate([rho, rho]), np.concatenate([phi, phi]))
        Hmu = (mu[idx] * w).sum(1)
        HE = np.einsum("nk,nkj->nj", w, self.E[idx].astype(np.float64))
        Hs2 = (self.s2[idx] * w).sum(1)
        sig = np.concatenate([obs.sigma, obs.sigma])
        if obs.kind is not None:
            kk = np.concatenate([obs.kind, obs.kind])
            sig = np.where(kk == 0, sig * math.sqrt(self.gamma), sig)
        ok = np.isfinite(y)
        return y[ok], Hmu[ok], HE[ok], Hs2[ok], sig[ok]

    def fit_R(self, obs: Obs, R: float, heading: float, cov_extra: dict, vm0: float, iters: int = 3,
              R_prior=None) -> Fit:
        vm = vm0
        for _ in range(iters):
            c = cov_vector(vm, R, **cov_extra)
            mu = self.prior_mean(c)
            y, Hmu, HE, Hs2, sig = self._design(obs, R, heading, mu)
            A = np.column_stack([Hmu, HE])                 # m x (k+1)
            noise = sig ** 2 + (vm ** 2) * Hs2
            Winv = 1.0 / noise
            k = self.E.shape[1]
            prior_var = np.concatenate([[(0.6 * vm) ** 2], (vm ** 2) * self.kappa * self.lam])
            prior_mean = np.concatenate([[vm], np.zeros(k)])
            Pinv = (A.T * Winv) @ A + np.diag(1.0 / prior_var)
            P = np.linalg.inv(Pinv)
            b = P @ ((A.T * Winv) @ y + prior_mean / prior_var)
            vm_new = float(max(5.0, b[0]))
            if abs(vm_new - vm) < 0.2:
                vm = vm_new
                break
            vm = vm_new
        # evidence with the final linearisation (Woodbury)
        r0 = y - A @ prior_mean
        Wr = r0 * Winv
        AtWr = A.T @ Wr
        quad = float(r0 @ Wr - AtWr @ P @ AtWr)
        sign, logdet_Pinv = np.linalg.slogdet(Pinv)
        logdet = logdet_Pinv + np.sum(np.log(prior_var)) + np.sum(np.log(noise))
        loge = -0.5 * (quad + logdet + y.size * math.log(2 * math.pi))
        mu_lr, sd_lr = R_prior if R_prior is not None else self.R_prior
        lr = (math.log(R) - mu_lr) / sd_lr
        loge += -0.5 * lr * lr
        res = y - A @ b
        return Fit(R=R, vm=vm, b=b, P=P, mu=mu, log_evidence=loge, heading=heading, cov=c, n_obs=y.size,
                   resid_rms=float(np.sqrt(np.mean(res ** 2))))

    def fit(self, obs: Obs, heading: float, cov_extra: dict, R_grid: Sequence[float] = None,
            vm0: Optional[float] = None, R_obs: Optional[float] = None, R_obs_sd: float = 0.3) -> Fit:
        r, vt, vr = obs.vt_vr()
        if vm0 is None:
            vm0 = float(np.nanpercentile(vt, 95)) if np.isfinite(vt).any() else 30.0
        if R_grid is None:
            R_grid = np.exp(np.linspace(math.log(6.0), math.log(160.0), 28))
        R_prior = None
        if R_obs is not None and np.isfinite(R_obs) and R_obs > 0:
            # combine the climatological prior on ln R with the radius of the observed peak
            m0, s0 = self.R_prior
            w0, w1 = 1 / s0 ** 2, 1 / R_obs_sd ** 2
            R_prior = ((m0 * w0 + math.log(R_obs) * w1) / (w0 + w1), (w0 + w1) ** -0.5)
        fits = [self.fit_R(obs, float(R), heading, cov_extra, vm0, R_prior=R_prior) for R in R_grid]
        best = max(fits, key=lambda f: f.log_evidence)
        # refine around the best R
        j = int(np.argmax([f.log_evidence for f in fits]))
        lo = R_grid[max(0, j - 1)]
        hi = R_grid[min(len(R_grid) - 1, j + 1)]
        for R in np.exp(np.linspace(math.log(lo), math.log(hi), 7)):
            f = self.fit_R(obs, float(R), heading, cov_extra, best.vm, R_prior=R_prior)
            if f.log_evidence > best.log_evidence:
                best = f
        best.evidence_by_R = {round(float(f.R), 2): float(f.log_evidence) for f in fits}
        return best

    def state(self, fit: Fit) -> np.ndarray:
        """Posterior mean in physical units (m/s), shape grid.shape."""
        M = np.column_stack([fit.mu, self.E.astype(np.float64)])
        return (M @ fit.b).reshape(self.grid.shape)

    def spread(self, fit: Fit) -> np.ndarray:
        """Posterior standard deviation (m/s) at every node, including what the EOFs cannot represent."""
        M = np.column_stack([fit.mu, self.E.astype(np.float64)])
        var = np.einsum("ij,jk,ik->i", M, fit.P, M) + (fit.vm ** 2) * self.s2
        return np.sqrt(np.maximum(var, 0.0)).reshape(self.grid.shape)

    def sample_points(self, fit: Fit, x: np.ndarray, y: np.ndarray, z: np.ndarray, fld: int, values=None):
        """Values of a state (default: posterior mean) at physical points (km); fld 0 = Vt, 1 = Vr."""
        st = self.state(fit).ravel() if values is None else values.ravel()
        r = np.hypot(x, y)
        az = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
        phi = (az - fit.heading) % 360.0
        idx, w = _interp_index(self.grid, np.full(x.size, fld), z, r / fit.R, phi)
        return (st[idx] * w).sum(1)


def _rsvd(A: np.ndarray, k: int, rng, oversample: int = 15, power: int = 2):
    n, p = A.shape
    G = rng.standard_normal((p, k + oversample))
    Y = A @ G
    for _ in range(power):
        Y, _r = np.linalg.qr(Y)
        Y = A @ (A.T @ Y)
    Q, _r = np.linalg.qr(Y)
    Bm = Q.T @ A
    Ub, S, Vt = np.linalg.svd(Bm, full_matrices=False)
    U = Q @ Ub
    return U[:, :k], S[:k], Vt[:k]


def observed_rmw(obs: Obs, kind: int = 0, bin_km: float = 4.0, sectors: int = 8) -> Optional[float]:
    """Median over azimuth sectors of the radius of the strongest (smoothed) tangential wind seen."""
    r, vt, _vr = obs.vt_vr()
    sel = np.isfinite(vt) & ((obs.kind == kind) if obs.kind is not None else True)
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
        nb = rb.max() + 1
        sums = np.bincount(rb, weights=vt[m], minlength=nb)
        cnts = np.bincount(rb, minlength=nb)
        prof = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        if np.isfinite(prof).sum() < 5:
            continue
        sm_ = np.convolve(np.nan_to_num(prof, nan=0.0), np.ones(3) / 3, "same")
        j = int(np.nanargmax(np.where(np.isfinite(prof), sm_, -np.inf)))
        if 0 < j < nb - 1:                          # an interior maximum, not the end of the leg
            out.append((j + 0.5) * bin_km)
    return float(np.median(out)) if out else None
