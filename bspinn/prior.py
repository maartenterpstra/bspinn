"""Sparsity priors for the image reconstruction with fixed coil maps (2-D):

    x = argmin_x 1/2 ||M F (C x) - y||^2 + beta_tv TV(x) + beta_w ||W x||_1 + lam/2 ||x||^2

TV: isotropic total variation of the complex image (periodic forward differences). W: orthonormal periodic
Daubechies wavelets (db4 default; haar, db2), ``levels`` levels on the image zero-padded to a multiple of
2^levels; the coarsest approximation band is not penalised. Solved by ADMM (scaled form): the image update
is a few warm-started CG steps on (A^H A + rho (D^H D + I_W) + lam) x = A^H y + rho (...), the splits are
shrunk in closed form. All of it is batched over a leading axis of candidate weights, so k-space
cross-validation over a weight grid (the same withheld phase-encode lines as bspinn.senselam) costs about
one batched solve per fold.

Weights are relative: the data are scaled so that a lightly regularised SENSE image has its 99th-percentile
magnitude at 1, and rho is relative to the sampling fraction. ``prior='tv+wavelet'`` uses one weight beta for
TV and ``ratio`` x beta for the wavelets.
"""
import numpy as np
import torch

from .senselam import cv_masks, heldout_error

WAVELETS = {
    "haar": [0.7071067811865476, 0.7071067811865476],
    "db2": [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    "db4": [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114,
            -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
}
PRIORS = ("tv", "wavelet", "tv+wavelet")


def _fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def _ifft2c(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def _analysis(n, h):
    """One-level periodic orthonormal analysis matrix (n even): rows = n/2 lowpass, then n/2 highpass."""
    h = np.asarray(h, float)
    g = np.array([(-1) ** j * h[len(h) - 1 - j] for j in range(len(h))])
    W = np.zeros((n, n))
    for k in range(n // 2):
        for j in range(len(h)):
            W[k, (2 * k + j) % n] += h[j]
            W[n // 2 + k, (2 * k + j) % n] += g[j]
    return W


class Wavelet2D:
    """Orthonormal multi-level 2-D DWT (Mallat; periodic) on images zero-padded to multiples of 2^levels."""

    def __init__(self, shape, levels=4, wave="db4", device="cpu"):
        H, W = shape
        q = 2 ** levels
        self.H, self.W = H, W
        self.Hp, self.Wp = -(-H // q) * q, -(-W // q) * q
        self.levels = levels
        self.mats = []
        h, w = self.Hp, self.Wp
        for _ in range(levels):
            self.mats.append((torch.tensor(_analysis(h, WAVELETS[wave]), dtype=torch.complex64, device=device),
                              torch.tensor(_analysis(w, WAVELETS[wave]), dtype=torch.complex64, device=device)))
            h //= 2; w //= 2
        self.detail = torch.ones(self.Hp, self.Wp, dtype=torch.bool, device=device)
        self.detail[:h, :w] = False                          # coarsest approximation: not penalised

    def fwd(self, x):
        c = torch.zeros(*x.shape[:-2], self.Hp, self.Wp, dtype=x.dtype, device=x.device)
        c[..., :self.H, :self.W] = x
        h, w = self.Hp, self.Wp
        for Wr, Wc in self.mats:
            blk = Wr @ c[..., :h, :w] @ Wc.transpose(0, 1)
            c = c.clone(); c[..., :h, :w] = blk
            h //= 2; w //= 2
        return c

    def adj(self, c):
        c = c.clone()
        sizes = [(self.Hp >> i, self.Wp >> i) for i in range(self.levels)]
        for (Wr, Wc), (h, w) in zip(reversed(self.mats), reversed(sizes)):
            c[..., :h, :w] = Wr.transpose(0, 1) @ c[..., :h, :w] @ Wc
        return c[..., :self.H, :self.W]


def _grad(x):
    return torch.stack([torch.roll(x, -1, -2) - x, torch.roll(x, -1, -1) - x], -3)


def _grad_adj(z):
    return (torch.roll(z[..., 0, :, :], 1, -2) - z[..., 0, :, :]) + (torch.roll(z[..., 1, :, :], 1, -1) - z[..., 1, :, :])


def _bcg(apply, rhs, x, iters):
    """Batched CG over a leading axis (independent systems), fixed number of steps."""
    dims = tuple(range(1, rhs.ndim))
    r = rhs - apply(x); p = r
    rs = (r.conj() * r).real.sum(dims, keepdim=True)
    for _ in range(iters):
        Ap = apply(p)
        a = rs / (p.conj() * Ap).real.sum(dims, keepdim=True).clamp_min(1e-30)
        x = x + a * p; r = r - a * Ap
        rn = (r.conj() * r).real.sum(dims, keepdim=True)
        p = r + (rn / rs.clamp_min(1e-30)) * p
        rs = rn
    return x


def admm(y, C, m, beta, prior="tv", ratio=1.0, iters=50, cg_iters=5, rho=0.5, lam=1e-6, levels=4, wave="db4", x0=None):
    """y (N,H,W) complex tensor (zero outside m), C (N,H,W) RSS-normalised maps, m (H,W) float mask,
    beta (L,) weights -> images (L,H,W)."""
    if prior not in PRIORS:
        raise ValueError(f"prior must be one of {PRIORS}")
    L = beta.numel()
    fill = float(m.mean())
    r_ = rho * fill
    b_tv = beta.view(L, 1, 1) if "tv" in prior else None
    b_w = (beta * (ratio if prior == "tv+wavelet" else 1.0)).view(L, 1, 1) if "wavelet" in prior else None
    Wt = Wavelet2D(m.shape, levels, wave, y.device) if b_w is not None else None

    def AHA(x):
        return (C.conj() * _ifft2c(m * _fft2c(C * x[:, None]))).sum(1)

    def lhs(x):
        out = AHA(x) + lam * x
        if b_tv is not None:
            out = out + r_ * _grad_adj(_grad(x))
        if Wt is not None:
            out = out + r_ * x
        return out

    AHy = (C.conj() * _ifft2c(m * y)).sum(0)[None].expand(L, -1, -1)
    x = AHy.clone() if x0 is None else x0.expand(L, -1, -1).clone()
    z1 = _grad(x) if b_tv is not None else None; u1 = torch.zeros_like(z1) if z1 is not None else None
    z2 = Wt.fwd(x) if Wt is not None else None; u2 = torch.zeros_like(z2) if z2 is not None else None
    for _ in range(iters):
        rhs = AHy.clone()
        if z1 is not None:
            rhs = rhs + r_ * _grad_adj(z1 - u1)
        if z2 is not None:
            rhs = rhs + r_ * Wt.adj(z2 - u2)
        x = _bcg(lhs, rhs, x, cg_iters)
        if z1 is not None:
            v = _grad(x) + u1
            mag = torch.sqrt((v.abs() ** 2).sum(-3, keepdim=True)).clamp_min(1e-12)
            z1 = v * torch.clamp(1 - (b_tv / r_)[:, None] / mag, min=0)
            u1 = v - z1
        if z2 is not None:
            v = Wt.fwd(x) + u2
            shr = torch.clamp(1 - (b_w / r_) / v.abs().clamp_min(1e-12), min=0)
            z2 = torch.where(Wt.detail, v * shr, v)
            u2 = v - z2
    return x


def _setup(ksp, C, mask, device):
    C = np.asarray(C)
    C = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)
    m = np.asarray(mask, bool)
    Ct = torch.from_numpy(C.astype(np.complex64)).to(device)
    mt = torch.from_numpy(m.astype(np.float32)).to(device)
    yt = torch.from_numpy((np.asarray(ksp) * m[None]).astype(np.complex64)).to(device)
    # scale: lightly regularised SENSE image (Tikhonov 1e-2 x fill, CG) at p99 magnitude 1; also the ADMM start
    fill = float(mt.mean())
    AHA = lambda x: (Ct.conj() * _ifft2c(mt * _fft2c(Ct * x[:, None]))).sum(1) + 1e-2 * fill * x  # noqa: E731
    b = (Ct.conj() * _ifft2c(mt * yt)).sum(0)[None]
    x0 = _bcg(AHA, b, torch.zeros_like(b), 30)
    s = float(torch.quantile(x0.abs().flatten(), 0.99).clamp_min(1e-12))
    return C, m, Ct, mt, yt / s, x0 / s, s


AUTO_SCALE = 0.5      # the CV folds withhold 10 % of the lines, a harder problem that wants more regularisation
FALLBACK_WEIGHT = 2e-3  # used when the CV curve has no interior minimum
FLAT_CV = 0.01        # CV error range below which the curve is uninformative (noise-dominated slice)


def select_weight(grid, err, scale=AUTO_SCALE, fallback=FALLBACK_WEIGHT, flat=FLAT_CV, refine=True):
    """Weight from a CV error curve over ``grid`` (v46 rule, calibrated on 20 dataloader slices x 2 map sets x
    R = 4, 6; README 8.9):
      flat      max/min error < 1 + flat: held-out k-space is noise-dominated, the data cannot tell -> the
                largest grid weight;
      interior  the lowest interior local minimum, refined by a parabola in log weight, times ``scale``;
                minima at the grid edges are ignored (the lowest-weight edge was a spurious global minimum
                in 20 % of the cases, at 1.2-3x the oracle NRMSE);
      fallback  no interior minimum -> ``fallback``.
    Returns (weight, rule)."""
    grid = np.asarray(grid, float); e = np.asarray(err, float) / max(float(np.min(err)), 1e-30)
    if e.max() < 1 + flat:
        return float(grid[-1]), "flat"
    mins = [i for i in range(1, len(e) - 1) if e[i] <= e[i - 1] and e[i] <= e[i + 1]]
    if not mins:
        return float(fallback), "fallback"
    i = min(mins, key=lambda j: e[j])
    w = float(grid[i])
    if refine:
        lx = np.log(grid[i - 1:i + 2])
        c2, c1, _ = np.polyfit(lx, e[i - 1:i + 2], 2)
        if c2 > 0:
            w = float(np.exp(np.clip(-c1 / (2 * c2), lx[0], lx[2])))
    return w * scale, "interior"


def choose_weight(ksp, C, mask, prior="tv", grid=None, device="cpu", holdout=0.1, folds=2, calib=24, seed=0,
                  scale=AUTO_SCALE, fallback=FALLBACK_WEIGHT, flat=FLAT_CV, refine=True, **kw):
    """k-space cross-validation over ``grid`` (relative weights), weight by ``select_weight``: (weight, info)."""
    grid = np.asarray(grid if grid is not None else np.logspace(-4, -1, 10))
    C, m, Ct, mt, yt, x0, s = _setup(ksp, C, mask, device)
    err = np.zeros(len(grid))
    for tr, ho in cv_masks(m, holdout, folds, calib, seed):
        trt = torch.from_numpy(tr.astype(np.float32)).to(device)
        with torch.no_grad():
            X = admm(yt * trt, Ct, trt, torch.tensor(grid, dtype=torch.float32, device=device), prior, x0=x0, **kw)
        err += heldout_error(X.cpu().numpy(), C, np.asarray(ksp) / s, ho)
    i = int(np.argmin(err))
    w, rule = select_weight(grid, err, scale, fallback, flat, refine)
    return w, {"grid": grid.tolist(), "cv_error": err.tolist(), "weight_grid": float(grid[i]), "weight": w,
               "rule": rule, "scale": scale, "at_grid_edge": i in (0, len(grid) - 1)}


def prior_image(ksp, C, mask, prior="tv", weight="auto", device="cpu", report=None, grid=None, cv_options=None,
                scale=AUTO_SCALE, **kw):
    """Image with a sparsity prior; ``weight`` a number (relative) or 'auto' (k-space cross-validation and
    ``select_weight``; ``scale`` multiplies a CV-selected interior minimum). kw: ratio, iters, cg_iters, rho,
    lam, levels, wave (see admm). ``cv_options``: holdout, folds, calib, seed, fallback, flat, refine."""
    if isinstance(weight, str):
        weight, info = choose_weight(ksp, C, mask, prior, grid=grid, device=device, scale=scale,
                                     **(cv_options or {}), **kw)
        if report is not None:
            report.update(info, method="cv")
    elif report is not None:
        report.update(method="manual", weight=float(weight))
    _, _, Ct, mt, yt, x0, s = _setup(ksp, C, mask, device)
    with torch.no_grad():
        x = admm(yt, Ct, mt, torch.tensor([float(weight)], device=device), prior, x0=x0, **kw)[0] * s
    return x.cpu().numpy().astype(np.complex64)
