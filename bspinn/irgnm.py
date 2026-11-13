"""PINN-anchored IRGNM refinement of 2-D or 3-D coil maps (NLINV's Gauss-Newton solver, started from the PINN).

Unknowns: the image x and a Sobolev-weighted free correction d of every map,

    C_j = C0_j + W d_j,   W d = iFFT(w * d),   w(k) = (1 + a |k|^2)^(-b/2)      (k in cycles / FOV-pixel)

and the model y_j = P F (C_j x). Iteratively regularised Gauss-Newton (as BART nlinv): at step n, with
alpha_n = alpha0 * q^n, solve by CG the linearised problem

    min_{dx, dd} || DG(dx, dd) - (y - G(x_n, d_n)) ||^2 + alpha_n (|| x_n + dx - x0 ||^2 + || d_n + dd ||^2)

i.e. the penalty pulls towards the PINN solution (x0, d = 0) instead of towards NLINV's (1, 0): for large alpha
nothing changes, as alpha falls the data take over. Gauge (scale) and data scaling follow BART nlinv
(||y|| = 100, image RMS 1), so alpha0 = 1, q = 1/2 are its defaults.

Stopping: with ``holdout > 0`` a random fraction of the acquired k-space locations (all coils) is withheld,
the path is run on the rest, the withheld samples are predicted after every step (step -1 = the PINN maps, with
the image re-solved from the training samples), and the full-data path is then run up to the best step.
``steps`` without held-out samples is a fixed schedule.

    from bspinn.irgnm import irgnm_refine
    C, x, info = irgnm_refine(ksp, C_pinn, mask)          # ksp pre-whitened, (N, H, W); mask (H, W) bool
    C, x, info = irgnm_refine(ksp, C_pinn, mask)          # 3-D: ksp (N, D, H, W), mask (D, H, W)

3-D memory: about 10 coil-volume arrays in complex64 (32 coils x 256 x 256 x 128: ~20 GB).
"""
import math

import numpy as np
import torch


_DIMS = [(-2, -1)]            # spatial dims of the current problem (set by irgnm_refine: 2-D or 3-D)


def _F(x):
    d = _DIMS[0]
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=d), dim=d, norm="ortho"), dim=d)


def _Fh(k):
    d = _DIMS[0]
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(k, dim=d), dim=d, norm="ortho"), dim=d)


def _dot(a, b):
    return sum(torch.vdot(u.reshape(-1), v.reshape(-1)).real for u, v in zip(a, b))


def _cg(A, b, iters):
    x = [torch.zeros_like(t) for t in b]
    r = [t.clone() for t in b]; p = [t.clone() for t in b]; rs = _dot(r, r); b0 = rs
    for _ in range(iters):
        Ap = A(p); al = rs / _dot(p, Ap)
        x = [u + al * v for u, v in zip(x, p)]; r = [u - al * v for u, v in zip(r, Ap)]
        rn = _dot(r, r)
        if float(rn) < 1e-12 * float(b0):
            break
        p = [u + (rn / rs) * v for u, v in zip(r, p)]; rs = rn
    return x


def sobolev_weight(*shape, a=220.0, b=32.0, device="cpu"):
    """(1 + a |k|^2)^(-b/2) on a grid of ``shape`` (2-D or 3-D), k in cycles / FOV-pixel."""
    k2 = 0.0
    for i, n in enumerate(shape):
        k = (torch.arange(n, device=device, dtype=torch.float32) - n // 2) / n
        k2 = k2 + (k ** 2).view([-1 if j == i else 1 for j in range(len(shape))])
    return (1.0 + a * k2) ** (-b / 2)


def _sense(C, y, M, lam, iters=60):
    """Tikhonov CG-SENSE: (C^H F^H M F C + lam) x = C^H F^H M y."""
    def A(v):
        return [(C.conj() * _Fh(M * _F(C * v[0][None]))).sum(0) + lam * v[0]]
    return _cg(A, [(C.conj() * _Fh(M * y)).sum(0)], iters)[0]


def _path(y, M, C0, x0, w, alphas, cg_iters, x_weight, Mho=None, xref=None):
    """Run the IRGNM path; returns (C, x, held-out errors per step [step -1 first] or None)."""
    d = torch.zeros_like(C0); x = x0.clone()
    xr = x0 if xref is None else xref
    Wd = lambda dd: _Fh(w * dd)                              # noqa: E731
    WdH = lambda c: w * _F(c)                                # noqa: E731  (w real)
    errs = []

    def heldout(C, x):
        if Mho is None:
            return None
        r = Mho * (y - _F(C * x[None]))
        return float((r.abs() ** 2).sum())
    C = C0
    errs.append(heldout(C, x))
    for al in alphas:
        r = M * (y - _F(C * x[None]))
        Cc, xc = C, x

        def DG(dx, dd):
            return M * _F(Cc * dx[None] + Wd(dd) * xc[None])

        def DGH(k):
            z = _Fh(M * k)
            return (Cc.conj() * z).sum(0), WdH(xc.conj()[None] * z)

        def A(v):
            gx, gd = DGH(DG(v[0], v[1]))
            return [gx + al * x_weight * v[0], gd + al * v[1]]
        gx, gd = DGH(r)
        rhs = [gx + al * x_weight * (xr - x), gd - al * d]
        dx, dd = _cg(A, rhs, cg_iters)
        x = x + dx; d = d + dd
        C = C0 + Wd(d)
        errs.append(heldout(C, x))
    return C, x, (errs if Mho is not None else None)


@torch.no_grad()
def irgnm_refine(ksp, C0, mask, x0=None, *, steps=8, alpha0=1.0, q=0.5, cg_iters=30, sobolev=(220.0, 32.0),
                 x_weight=1.0, holdout=0.0, folds=1, max_steps=12, lam_x0=1e-2, seed=0, device="cpu",
                 x_ref="pinn"):
    """Refine maps C0 (N, H, W) or (N, D, H, W) on the (pre-whitened) undersampled k-space ksp of the same shape,
    mask (H, W) / (D, H, W).

    holdout > 0: choose the number of Gauss-Newton steps (0..max_steps) by held-out prediction of that fraction
    of the acquired locations (``folds`` disjoint draws, errors summed), then run the full-data path that far.
    x_ref: the image the penalty pulls towards: 'pinn' (the SENSE image with C0) or 'zero' (BART nlinv's
    Tikhonov image term; the maps stay anchored to C0).
    holdout = 0: run ``steps`` steps. Returns (C, x, info); C in the scale of C0 (RSS-normalised per pixel by the
    caller if wanted), x the joint image in the matching scale (y ~ F(C x)).
    """
    dev = torch.device(device)
    y = torch.as_tensor(np.asarray(ksp), dtype=torch.complex64, device=dev)
    mask = np.asarray(mask, bool)
    Mfull = torch.as_tensor(mask, device=dev).to(torch.complex64)[None]
    C = torch.as_tensor(np.asarray(C0), dtype=torch.complex64, device=dev)
    N, sp = C.shape[0], tuple(C.shape[1:])
    if len(sp) not in (2, 3):
        raise ValueError("irgnm_refine: maps must be (N, H, W) or (N, D, H, W)")
    _DIMS[0] = tuple(range(-len(sp), 0))
    rss = (C.abs() ** 2).sum(0).sqrt()
    rs_in = rss.clone()
    Cn = C / torch.clamp(rss, min=1e-12 * float(rss.max()))[None]          # RSS 1 (unit-norm channel vectors)
    sy = 100.0 / float((y * Mfull).abs().pow(2).sum().sqrt())
    y = y * Mfull * sy
    w = sobolev_weight(*sp, a=sobolev[0], b=sobolev[1], device=dev)
    info = {"alpha0": alpha0, "q": q, "sobolev": list(sobolev), "cg_iters": cg_iters}

    def start(M):
        xs = _sense(Cn, y, M, lam_x0)
        g = math.sqrt(float((xs.abs() ** 2).mean()))            # gauge: image RMS 1 (BART's x0 = 1)
        return xs / g, Cn * g

    xr = (lambda xs: None) if x_ref == "pinn" else (lambda xs: torch.zeros_like(xs))   # noqa: E731
    info["x_ref"] = x_ref
    if holdout > 0:
        rng = np.random.default_rng(seed)
        acq = np.flatnonzero(mask.ravel())
        tot = np.zeros(max_steps + 1)
        alphas = [alpha0 * q ** n for n in range(max_steps)]
        perm = rng.permutation(acq)
        nho = int(round(holdout * acq.size))
        for f in range(folds):
            ho = np.zeros(mask.size, bool); ho[perm[f * nho:(f + 1) * nho]] = True
            ho = ho.reshape(mask.shape)
            Mho = torch.as_tensor(ho, device=dev).to(torch.complex64)[None]
            Mtr = Mfull * (1 - Mho)
            xs, Cs = start(Mtr)
            _, _, errs = _path(y, Mtr, Cs, xs, w, alphas, cg_iters, x_weight, Mho=Mho, xref=xr(xs))
            tot += np.asarray(errs)
        n_best = int(np.argmin(tot))               # number of steps (0 = keep the PINN maps)
        info["heldout_err"] = (tot / tot[0]).round(5).tolist()
        steps = n_best
    info["steps"] = steps
    xs, Cs = start(Mfull)
    if steps == 0:
        C_out, x_out = Cs, xs
    else:
        C_out, x_out = _path(y, Mfull, Cs, xs, w, [alpha0 * q ** n for n in range(steps)], cg_iters, x_weight,
                             xref=xr(xs))[:2]
    # back to the caller's scale: C_out * x_out ~ y * ... ; keep the map scale of C0 where it was RSS-normalised
    g = float(Cs.abs().pow(2).sum(0).sqrt().mean())             # = image gauge factor applied to Cn
    C_ret = C_out / g * rs_in[None]
    x_ret = x_out * g / sy
    x_ret = x_ret / torch.where(rs_in > 0, rs_in, torch.ones_like(rs_in))
    info["data_residual"] = float(((Mfull * (y - _F(C_out * x_out[None]))).abs() ** 2).sum() / (y.abs() ** 2).sum())
    return C_ret.cpu().numpy(), x_ret.cpu().numpy(), info
