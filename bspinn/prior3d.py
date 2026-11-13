"""Regularised 3-D SENSE image: TGV^2 (default) or TV, ADMM, weight by k-space cross-validation.

    from bspinn.prior3d import prior_image_3d
    x = prior_image_3d(ksp, C, mask, prior="tgv", weight="auto", device="cuda")   # ksp, C (N, D, H, W)

Tikhonov CG-SENSE (``sense_image(..., prior=None)``) has one knob that trades noise for blur everywhere at once;
at the g-factors of 2-D-accelerated 3-D acquisitions (R = 4..8 over two phase-encode axes) the weight that
keeps the image sharp amplifies noise in the centre of the volume. TGV^2 (the 2-D default, bspinn.prior) removes
that noise while keeping edges and smooth shading.

Cartesian 3-D data with a fully sampled readout axis are first inverse-FFT'd along it (exact: hybrid space), so
the encoding operator is a 2-D FFT over the two phase-encode axes for every readout position, and the regulariser
couples all three axes. The weight is cross-validated on a slab of ``cv_slab`` readout positions through the
object (withheld: random phase-encode positions outside the calibration box, all readout positions of the slab),
then the whole volume is reconstructed with it. Without a fully sampled axis the operator is the 3-D FFT and the
cross-validation runs on the whole volume (slower).

Weights are relative: data scaled so the lightly regularised SENSE image has p99 magnitude 1 (as bspinn.prior).
"""
from __future__ import annotations

import numpy as np
import torch

from .prior import AUTO_SCALE, FALLBACK_WEIGHT, FLAT_CV, select_weight

_S2 = 2 ** -0.5
_AX = (-3, -2, -1)
_OFF = ((0, 1), (0, 2), (1, 2))


def _fftc(x, dims):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)


def _ifftc(x, dims):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)


# ------------------------------------------------------------------ finite differences (periodic), 3 axes
def _grad(x):                       # (..., D, H, W) -> (..., 3, D, H, W), forward differences
    return torch.stack([torch.roll(x, -1, a) - x for a in _AX], -4)


def _grad_adj(z):
    return sum(torch.roll(z[..., i, :, :, :], 1, a) - z[..., i, :, :, :] for i, a in enumerate(_AX))


def _dfb(x, a):                     # backward difference
    return x - torch.roll(x, 1, a)


def _dfb_adj(z, a):
    return z - torch.roll(z, -1, a)


def _sym_grad(v):
    """Symmetrised gradient of a vector field v (..., 3, D, H, W) -> (..., 6, D, H, W): 3 diagonal terms, then the
    3 off-diagonal ones scaled so the 2-norm over the 6 components is the Frobenius norm (backward differences)."""
    c = [v[..., i, :, :, :] for i in range(3)]
    diag = [_dfb(c[i], _AX[i]) for i in range(3)]
    off = [_S2 * (_dfb(c[i], _AX[j]) + _dfb(c[j], _AX[i])) for i, j in _OFF]
    return torch.stack(diag + off, -4)


def _sym_grad_adj(z):
    e = [z[..., i, :, :, :] for i in range(6)]
    out = [_dfb_adj(e[i], _AX[i]) for i in range(3)]
    for k, (i, j) in enumerate(_OFF):
        out[i] = out[i] + _S2 * _dfb_adj(e[3 + k], _AX[j])
        out[j] = out[j] + _S2 * _dfb_adj(e[3 + k], _AX[i])
    return torch.stack(out, -4)


def _shrink(v, t):
    mag = torch.sqrt((v.abs() ** 2).sum(-4, keepdim=True)).clamp_min(1e-12)
    return v * torch.clamp(1 - t / mag, min=0)


def _bcg(apply, rhs, x, iters):
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


# ------------------------------------------------------------------ ADMM
def admm3d(y, C, m, beta, fdims, prior="tgv", ratio=2.0, iters=40, cg_iters=6, rho=0.5, lam=1e-6, x0=None):
    """min_x 1/2 ||M F (C x) - y||^2 + beta R(x), batched over L weights.
    y, C (N, D, H, W); m (D, H, W) float (or broadcastable); beta (L,); F = centred FFT over ``fdims`` (the
    phase-encode axes in hybrid space, or all three). R = TV (||grad x||_{2,1}) or TGV^2
    (min_v ||grad x - v||_{2,1} + ratio ||E v||_{2,1}). Returns x (L, D, H, W)."""
    if prior not in ("tv", "tgv"):
        raise ValueError("prior must be 'tv' or 'tgv'")
    L = beta.numel()
    r_ = rho * float(m.mean())
    a1 = beta.view(L, 1, 1, 1, 1).to(y.device)
    a0 = a1 * ratio
    fd = fdims

    def AHA(v):                                           # (L, D, H, W)
        return (C.conj() * _ifftc(m * _fftc(C * v[:, None], fd), fd)).sum(1)

    AHy = (C.conj() * _ifftc(m * y, fd)).sum(0)[None].expand(L, -1, -1, -1)
    x = (x0 if x0 is not None else AHy).expand(L, -1, -1, -1).clone()
    z1 = _grad(x); u1 = torch.zeros_like(z1)
    if prior == "tv":
        def lhs(v):
            return AHA(v) + lam * v + r_ * _grad_adj(_grad(v))
        for _ in range(iters):
            x = _bcg(lhs, AHy + r_ * _grad_adj(z1 - u1), x, cg_iters)
            q = _grad(x) + u1
            z1 = _shrink(q, a1 / r_); u1 = q - z1
        return x

    def lhs(w):                                           # w (L, 4, D, H, W): x, then v (3 components)
        xx, v = w[:, 0], w[:, 1:]
        gx = _grad(xx) - v
        ox = AHA(xx) + lam * xx + r_ * _grad_adj(gx)
        ov = -r_ * gx + r_ * _sym_grad_adj(_sym_grad(v)) + lam * v
        return torch.cat([ox[:, None], ov], 1)

    w = torch.cat([x[:, None], torch.zeros(L, 3, *x.shape[1:], dtype=x.dtype, device=x.device)], 1)
    z2 = torch.zeros(L, 6, *x.shape[1:], dtype=x.dtype, device=x.device); u2 = torch.zeros_like(z2)
    for _ in range(iters):
        rx = AHy + r_ * _grad_adj(z1 - u1)
        rv = -r_ * (z1 - u1) + r_ * _sym_grad_adj(z2 - u2)
        w = _bcg(lhs, torch.cat([rx[:, None], rv], 1), w, cg_iters)
        xx, v = w[:, 0], w[:, 1:]
        q = _grad(xx) - v + u1
        z1 = _shrink(q, a1 / r_); u1 = q - z1
        q = _sym_grad(v) + u2
        z2 = _shrink(q, a0 / r_); u2 = q - z2
    return w[:, 0]


# ------------------------------------------------------------------ setup, cross-validation, driver
def readout_axis(m):
    """Spatial axis (0..2) along which every acquired line is complete (Cartesian readout), or None."""
    m = np.asarray(m, bool)
    for a in range(3):
        if m.shape[a] > 1 and bool((m == m.any(axis=a, keepdims=True)).all()) and not m.all():
            return a
    return 0 if m.all() else None


def _prepare(ksp, C, mask):
    """RSS-normalised maps, hybrid-space data (readout inverse-FFT'd) and the FFT axes of the operator."""
    C = np.asarray(C, np.complex64)
    C = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)
    m = np.asarray(mask, bool)
    y = np.asarray(ksp, np.complex64) * m[None]
    ro = readout_axis(m)
    if ro is None:
        return C, m.astype(np.float32), y, (-3, -2, -1), None
    a = 1 + ro                                                     # axis in (N, D, H, W)
    y = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(y, axes=a), axis=a, norm="ortho"), axes=a).astype(np.complex64)
    mpe = m.any(axis=ro, keepdims=True).astype(np.float32)        # broadcast along the readout
    fdims = tuple(d for d in (-3, -2, -1) if d != ro - 3)
    return C, mpe, y, fdims, ro


def _scale(Ct, mt, yt, fdims, iters=20):
    fill = float(mt.mean())
    AHA = lambda v: (Ct.conj() * _ifftc(mt * _fftc(Ct * v[:, None], fdims), fdims)).sum(1) + 1e-2 * fill * v  # noqa: E731
    b = (Ct.conj() * _ifftc(mt * yt, fdims)).sum(0)[None]
    x0 = _bcg(AHA, b, torch.zeros_like(b), iters)
    sub = x0.abs().flatten()
    sub = sub[torch.randperm(sub.numel(), device=sub.device)[:2_000_000]]
    return x0, float(torch.quantile(sub, 0.99).clamp_min(1e-12))


def _cv_mask(mpe, ro, holdout, calib, rng):
    """Withhold a random ``holdout`` fraction of the acquired phase-encode positions outside the central
    calibration box (all readout positions). Returns (train, heldout) masks shaped like ``mpe``."""
    m = mpe.astype(bool)
    pe = [a for a in range(3) if a != ro] if ro is not None else [0, 1, 2]
    box = np.ones(m.shape, bool)
    for a in pe:
        n = m.shape[a]; c = min(int(calib), n)
        keep = np.zeros(n, bool); keep[n // 2 - c // 2: n // 2 - c // 2 + c] = True
        box &= keep.reshape([-1 if i == a else 1 for i in range(3)])
    cand = np.flatnonzero((m & ~box).ravel())
    ho = np.zeros(m.size, bool)
    if cand.size:
        ho[rng.choice(cand, max(1, int(round(holdout * cand.size))), replace=False)] = True
    ho = ho.reshape(m.shape)
    return (m & ~ho), ho


def choose_weight_3d(ksp, C, mask, prior="tgv", grid=None, device="cpu", holdout=0.1, folds=2, calib=24, seed=0,
                     cv_slab=24, scale=AUTO_SCALE, fallback=FALLBACK_WEIGHT, flat=FLAT_CV, **kw):
    """k-space cross-validation of the relative weight on a slab of readout positions; (weight, info)."""
    grid = np.asarray(grid if grid is not None else np.logspace(-4, -1, 8))
    Cn, mpe, y, fdims, ro = _prepare(ksp, C, mask)
    dev = torch.device(device)
    if ro is not None and cv_slab and Cn.shape[1 + ro] > cv_slab:
        e = (np.abs(y) ** 2).sum(axis=tuple(i for i in range(4) if i != 1 + ro))       # signal along the readout
        c = int(np.argmax(np.convolve(e, np.ones(cv_slab), "valid"))) if e.size >= cv_slab else 0
        sl = [slice(None)] * 4; sl[1 + ro] = slice(c, c + cv_slab)
        Cn, y = Cn[tuple(sl)], y[tuple(sl)]
    Ct = torch.from_numpy(np.ascontiguousarray(Cn)).to(dev)
    yt = torch.from_numpy(np.ascontiguousarray(y)).to(dev)
    rng = np.random.default_rng(seed)
    err = np.zeros(len(grid))
    with torch.no_grad():
        for _ in range(folds):
            tr, ho = _cv_mask(mpe, ro, holdout, calib, rng)
            trt = torch.from_numpy(tr.astype(np.float32)).to(dev)
            hot = torch.from_numpy(ho.astype(np.float32)).to(dev)
            x0, s = _scale(Ct, trt, yt * trt, fdims)
            for i, b in enumerate(grid):                 # one weight at a time: bounded memory
                x = admm3d(yt * trt / s, Ct, trt, torch.tensor([float(b)], device=dev), fdims, prior, x0=x0 / s,
                           **kw)[0]
                P = hot * _fftc(Ct * x[None], fdims)
                Y = hot * yt / s
                pp = float((P.abs() ** 2).sum()); py = complex((P.conj() * Y).sum())
                err[i] += float((Y.abs() ** 2).sum()) - abs(py) ** 2 / max(pp, 1e-30)   # best global scale
    i = int(np.argmin(err))
    w, rule = select_weight(grid, err, scale, fallback, flat, True)
    return w, {"grid": grid.tolist(), "cv_error": err.tolist(), "weight_grid": float(grid[i]), "weight": w,
               "rule": rule, "scale": scale, "readout_axis": ro, "cv_slab": cv_slab}


def prior_image_3d(ksp, C, mask=None, prior="tgv", weight="auto", device="cpu", report=None, cv_options=None,
                   iters=40, cg_iters=6, ratio=2.0, rho=0.5, lam=1e-6):
    """3-D image (D, H, W) with a TGV^2 / TV prior. ksp, C (N, D, H, W) (pre-whitened, maps of the same channels);
    mask (D, H, W) bool (default: non-zero samples). ``weight``: relative number or 'auto' (cross-validation,
    ``cv_options``: holdout, folds, calib, seed, cv_slab, grid, scale). The image is in the scale of
    sum_j C_j^* x_j for RSS-normalised maps."""
    ksp = np.asarray(ksp)
    m = (np.abs(ksp).sum(0) > 0) if mask is None else np.asarray(mask, bool)
    kw = dict(ratio=ratio, iters=iters, cg_iters=cg_iters, rho=rho, lam=lam)
    if isinstance(weight, str):
        weight, info = choose_weight_3d(ksp, C, m, prior, device=device, **{**(cv_options or {})},
                                        **{k: v for k, v in kw.items() if k != "iters"}, iters=max(20, iters // 2))
        if report is not None:
            report.update(info, method="cv", prior=prior)
    elif report is not None:
        report.update(method="manual", weight=float(weight), prior=prior)
    Cn, mpe, y, fdims, ro = _prepare(ksp, C, m)
    dev = torch.device(device)
    with torch.no_grad():
        Ct = torch.from_numpy(Cn).to(dev); mt = torch.from_numpy(mpe).to(dev); yt = torch.from_numpy(y).to(dev)
        x0, s = _scale(Ct, mt, yt, fdims)
        x = admm3d(yt / s, Ct, mt, torch.tensor([float(weight)], device=dev), fdims, prior, x0=x0 / s, **kw)[0] * s
    return x.cpu().numpy().astype(np.complex64)
