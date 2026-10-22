"""Cine (x, y, t) image priors with fixed coil maps: a generic multi-split ADMM and the candidates of the r2-dyn
study (README 8.13).

    x = argmin_x 1/2 sum_t ||M_t F (C_t x_t) - y_t||^2 + sum_i beta_i R_i(Psi_i x - b_i)

Splits Psi_i (all with closed-form proximal maps): temporal TV (cyclic), spatial TV (isotropic), temporal Fourier
L1 (k-t SPARSE-SENSE; the temporal mean optionally free), temporal second differences, a reference image
(||x_t - x_ref||_1), global and locally low rank (singular-value thresholding of Casorati blocks). L+S
(Otazo et al. 2015) has its own iteration. The image update is a few warm-started CG steps on
(A^H A + rho sum_i Psi_i^H Psi_i + lam) x = A^H y + rho sum_i Psi_i^H (z_i + b_i - u_i), as bspinn.prior.

Weights are relative to the data scaled as in bspinn.prior (99th percentile of a light Tikhonov image = 1);
singular-value thresholds are weight x sqrt(rows of the Casorati block), so one weight scale fits all."""
from __future__ import annotations

import numpy as np
import torch

from .prior import _bcg, _dyn_setup, _grad, _grad_adj, _tgrad, _tgrad_adj
from .senselam import cv_masks


def _sh(x):
    return torch.fft.ifftshift(x, dim=(-2, -1))


def _ush(x):
    return torch.fft.fftshift(x, dim=(-2, -1))


def _F(x):
    return torch.fft.fft2(x, norm="ortho")


def _Fi(x):
    return torch.fft.ifft2(x, norm="ortho")


class DynProblem:
    """ksp (N,H,W,T), C (N,H,W) or (N,H,W,T), mask (H,W,T) -> T-first tensors, scaled data.
    Internally everything (images, maps, k-space, masks) is ifftshifted once, so the operator uses plain FFTs
    (the centred FFT's shifts cost 6x the transform on the CPU); all priors here are shift-invariant (cyclic),
    except the LLR block grid, which is shifted at random anyway. ``out(x)`` undoes the shift; masks passed to
    solvers must be shifted with ``shift_mask``."""

    def __init__(self, ksp, C, mask, device="cpu"):
        k, m, Cn, Ct, mt, yt, x0, s = _dyn_setup(ksp, C, mask, device)
        self.k, self.mnp, self.Cn, self.s = k, m, Cn, s
        self.C, self.m, self.y, self.x0 = _sh(Ct), _sh(mt), _sh(yt), _sh(x0)
        self.device = device
        self.T = m.shape[0]

    def shift_mask(self, m):
        return _sh(m)

    def out(self, x):
        return _ush(x)

    def AHA(self, v, m):
        return (self.C.conj() * _Fi(m[:, None] * _F(self.C * v[:, None]))).sum(-3)

    def AHy(self, y, m):
        return (self.C.conj() * _Fi(m[:, None] * y)).sum(-3)

    def start(self, y, m, iters=30):
        fill = float(m.mean())
        b = self.AHy(y, m)
        return _bcg(lambda v: (self.AHA(v[0], m) + 1e-2 * fill * v[0])[None], b[None], torch.zeros_like(b)[None], iters)[0]


# ---------------------------------------------------------------------------------------- proximal maps
def soft(v, t):
    return v * torch.clamp(1 - t / v.abs().clamp_min(1e-12), min=0)


def soft_group(v, t, dim):
    mag = torch.sqrt((v.abs() ** 2).sum(dim, keepdim=True)).clamp_min(1e-12)
    return v * torch.clamp(1 - t / mag, min=0)


def svt_casorati(x, t):
    """x (T,H,W): nuclear-norm prox of the (HW, T) Casorati matrix, threshold t * sqrt(HW)."""
    T, H, W = x.shape
    M = x.reshape(T, H * W).T
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    S = torch.clamp(S - t * np.sqrt(H * W), min=0)
    return ((U * S.to(U.dtype)) @ Vh).T.reshape(T, H, W)


def svt_local(x, t, b=8, shift=(0, 0)):
    """Locally low rank: SVT of (b*b, T) Casorati blocks on a grid shifted by ``shift`` (cyclic), threshold t * b."""
    T, H, W = x.shape
    xs = torch.roll(x, shift, (1, 2))
    Hb, Wb = H // b * b, W // b * b
    core = xs[:, :Hb, :Wb].reshape(T, Hb // b, b, Wb // b, b).permute(1, 3, 2, 4, 0).reshape(-1, b * b, T)
    U, S, Vh = torch.linalg.svd(core, full_matrices=False)
    S = torch.clamp(S - t * b, min=0)
    core = (U * S.to(U.dtype)[:, None, :]) @ Vh
    out = xs.clone()
    out[:, :Hb, :Wb] = core.reshape(Hb // b, Wb // b, b, b, T).permute(4, 0, 2, 1, 3).reshape(T, Hb, Wb)
    return torch.roll(out, (-shift[0], -shift[1]), (1, 2))


def _tfft(x):
    return torch.fft.fft(x, dim=0, norm="ortho")


def _itfft(z):
    return torch.fft.ifft(z, dim=0, norm="ortho")


def _t2(x):
    return _tgrad(_tgrad(x))


def _t2_adj(z):
    return _tgrad_adj(_tgrad_adj(z))


class Split:
    def __init__(self, fwd, adj, prox, beta, gram=None, offset=None):
        self.fwd, self.adj, self.prox, self.beta, self.gram, self.offset = fwd, adj, prox, beta, gram, offset


def split(kind, beta, **kw):
    """One regulariser: 'tt' temporal TV, 'ts' spatial TV, 'tf' temporal Fourier L1 (dc=False: mean free),
    't2' temporal 2nd difference, 'ref' ||x - ref||_1 (ref (T|1,H,W)), 'lr' global low rank, 'llr' local (b)."""
    ident = lambda x: x  # noqa: E731
    if kind == "tt":
        return Split(_tgrad, _tgrad_adj, lambda v, t: soft(v, t), beta)
    if kind == "ts":
        return Split(_grad, _grad_adj, lambda v, t: soft_group(v, t, -3), beta)
    if kind == "t2":
        return Split(_t2, _t2_adj, lambda v, t: soft(v, t), beta)
    if kind == "tf":
        dc = kw.get("dc", False)

        def prox(v, t):
            out = soft(v, t)
            if not dc:
                out[0] = v[0]
            return out
        return Split(_tfft, _itfft, prox, beta, gram=ident)
    if kind == "ref":
        return Split(ident, ident, lambda v, t: soft(v, t), beta, gram=ident, offset=kw["ref"])
    if kind == "lr":
        return Split(ident, ident, svt_casorati, beta, gram=ident)
    if kind == "llr":
        b = kw.get("b", 8)
        rng = np.random.default_rng(kw.get("seed", 0))

        def prox(v, t):
            return svt_local(v, t, b, tuple(int(s) for s in rng.integers(0, b, 2)))
        return Split(ident, ident, prox, beta, gram=ident)
    raise ValueError(kind)


def admm_multi(prob: DynProblem, y, m, splits, iters=50, cg_iters=5, rho=0.5, lam=1e-6, x0=None):
    r_ = rho * float(m.mean())
    AHy = prob.AHy(y, m)
    x = (prob.start(y, m) if x0 is None else x0).clone()

    def lhs(v):
        out = prob.AHA(v, m) + lam * v
        for sp in splits:
            out = out + r_ * (sp.gram(v) if sp.gram is not None else sp.adj(sp.fwd(v)))
        return out

    zs, us = [], []
    for sp in splits:
        z = sp.fwd(x) - (sp.offset if sp.offset is not None else 0)
        zs.append(z); us.append(torch.zeros_like(z))
    for _ in range(iters):
        rhs = AHy
        for sp, z, u in zip(splits, zs, us):
            rhs = rhs + r_ * sp.adj(z + (sp.offset if sp.offset is not None else 0) - u)
        x = _bcg(lambda v: lhs(v[0])[None], rhs[None], x[None], cg_iters)[0]
        for i, sp in enumerate(splits):
            v = sp.fwd(x) - (sp.offset if sp.offset is not None else 0) + us[i]
            zs[i] = sp.prox(v, sp.beta / r_)
            us[i] = v - zs[i]
    return x


def lplus_s(prob: DynProblem, y, m, lam_l, lam_s, iters=50, x0=None, dc=True):
    """L+S (Otazo 2015): L low rank (SVT, threshold lam_l sqrt(HW)), S sparse in temporal Fourier (lam_s);
    proximal gradient with unit step (||A|| <= 1: RSS-normalised maps, binary mask), FISTA momentum."""
    M = (prob.start(y, m) if x0 is None else x0).clone()
    S = torch.zeros_like(M)
    AHy = prob.AHy(y, m)
    for _ in range(iters):
        L = svt_casorati(M - S, lam_l)
        Z = _tfft(M - L)
        Zs = soft(Z, lam_s)
        if not dc:
            Zs[0] = Z[0]
        S = _itfft(Zs)
        X = L + S
        M = X - (prob.AHA(X, m) - AHy)
    return L + S


def tgv_t(prob: DynProblem, y, m, a1, a0, beta_s, iters=50, cg_iters=5, rho=0.5, lam=1e-6, x0=None):
    """Temporal TGV2 (+ spatial TV beta_s): min 1/2||Ax-y||^2 + a1 ||D_t x - w||_1 + a0 ||D_t w||_1 + beta_s TV(x),
    ADMM over (x, w) jointly (CG on the stacked variable)."""
    r_ = rho * float(m.mean())
    T = y.shape[0]
    AHy = prob.AHy(y, m)
    x = (prob.start(y, m) if x0 is None else x0).clone()
    w = _tgrad(x).clone()

    def lhs(v):
        xx, ww = v[0, :T], v[0, T:]
        e = _tgrad(xx) - ww
        ox = prob.AHA(xx, m) + lam * xx + r_ * _tgrad_adj(e)
        ow = -r_ * e + r_ * _tgrad_adj(_tgrad(ww)) + lam * ww
        if beta_s > 0:
            ox = ox + r_ * _grad_adj(_grad(xx))
        return torch.cat([ox, ow])[None]
    z1 = _tgrad(x) - w; u1 = torch.zeros_like(z1)
    z2 = _tgrad(w); u2 = torch.zeros_like(z2)
    zs = _grad(x) if beta_s > 0 else None
    us = torch.zeros_like(zs) if zs is not None else None
    for _ in range(iters):
        a = z1 - u1
        rx = AHy + r_ * _tgrad_adj(a)
        rw = -r_ * a + r_ * _tgrad_adj(z2 - u2)
        if zs is not None:
            rx = rx + r_ * _grad_adj(zs - us)
        v = _bcg(lhs, torch.cat([rx, rw])[None], torch.cat([x, w])[None], cg_iters)[0]
        x, w = v[:T], v[T:]
        q = _tgrad(x) - w + u1; z1 = soft(q, a1 / r_); u1 = q - z1
        q = _tgrad(w) + u2; z2 = soft(q, a0 / r_); u2 = q - z2
        if zs is not None:
            q = _grad(x) + us; zs = soft_group(q, beta_s / r_, -3); us = q - zs
    return x


def view_share(prob: DynProblem, y, m):
    """x0 from view sharing: each frame's missing k-space filled from the nearest frame (cyclic) that sampled it,
    then the light-Tikhonov SENSE of that full-ish k-space with the frame's maps."""
    T = y.shape[0]
    yf, mf = y.clone(), m.clone()
    for d in range(1, T // 2 + 1):
        for sgn in (1, -1):
            src = (torch.arange(T) + sgn * d) % T
            take = (mf == 0) & (m[src] > 0)
            yf = torch.where(take[:, None], y[src], yf)
            mf = torch.where(take, torch.ones_like(mf), mf)
    return prob.start(yf, mf)


# ---------------------------------------------------------------------------------------- method specs
def make_solver(spec: str, prob: DynProblem, iters=50, cg_iters=5):
    """spec 'name[:p1[:p2]]' -> solver(y, m, w) returning x (T,H,W) for the weight w. Prefix 'vs+' starts every
    solve from the view-shared image (instead of the light-Tikhonov one). Names:
      ttv:r          temporal TV w + spatial TV r w (r = 0.25: the current default)
      t2:r           temporal 2nd difference w + spatial TV r w
      ttvt2:q:r      temporal TV w + 2nd difference q w + spatial TV r w (a TGV-like mix)
      tf:r           temporal Fourier L1 w (mean free) + spatial TV r w;  tfdc:r the mean penalised too
      ttvtf:q:r      temporal TV w + temporal Fourier q w + spatial TV r w
      llr:b:r        locally low rank (blocks b) w + spatial TV r w;  ttvllr:q:r temporal TV w + LLR(8) q w + sTV r w
      lr:r           global low rank w + spatial TV r w
      tgv:q:r        temporal TGV2 (a1 = w, a0 = q w) + spatial TV r w
      ls:q           L+S: S temporal Fourier w, L threshold q w
      ref:q:r        temporal TV w + spatial TV r w + q w ||x_t - x_ref||_1, x_ref the time-averaged SENSE-1 image
      refts:q        temporal TV w + q w TV(x_t - x_ref) (spatial TV of the difference from the reference)"""
    vs = spec.startswith("vs+")
    if vs:
        spec = spec[3:]
    parts = spec.split(":")
    name, p = parts[0], [float(v) for v in parts[1:]]
    cache = {}

    def x0_for(y, m):
        if not vs:
            return None
        key = (id(y), id(m))
        if key not in cache:
            cache.clear(); cache[key] = view_share(prob, y, m)
        return cache[key]

    def ref_image(y, m):
        key = ("ref", id(y), id(m))
        if key not in cache:
            # time-averaged k-space (mean over the frames sampling each point) -> SENSE with the mean maps
            cnt = m.sum(0)
            ya = (y * m[:, None]).sum(0) / cnt.clamp_min(1)[None]
            ma = (cnt > 0).float()
            Cm = prob.C if prob.C.ndim == 3 else prob.C.mean(0)
            AHA = lambda v: (Cm.conj() * _Fi(ma * _F(Cm * v))).sum(0) + 1e-3 * v  # noqa: E731
            b = (Cm.conj() * _Fi(ma * ya)).sum(0)
            xr = _bcg(lambda v: AHA(v[0])[None], b[None], torch.zeros_like(b)[None], 30)[0]
            cache[key] = xr[None]
        return cache[key]

    def solve(y, m, w):
        sp = []
        if name == "ttv":
            sp = [split("tt", w), split("ts", w * p[0])]
        elif name == "t2":
            sp = [split("t2", w), split("ts", w * p[0])]
        elif name == "ttvt2":
            sp = [split("tt", w), split("t2", w * p[0]), split("ts", w * p[1])]
        elif name in ("tf", "tfdc"):
            sp = [split("tf", w, dc=name == "tfdc"), split("ts", w * p[0])]
        elif name == "ttvtf":
            sp = [split("tt", w), split("tf", w * p[0]), split("ts", w * p[1])]
        elif name == "llr":
            sp = [split("llr", w, b=int(p[0])), split("ts", w * p[1])]
        elif name == "ttvllr":
            sp = [split("tt", w), split("llr", w * p[0]), split("ts", w * p[1])]
        elif name == "lr":
            sp = [split("lr", w), split("ts", w * p[0])]
        elif name == "ref":
            sp = [split("tt", w), split("ts", w * p[1]), split("ref", w * p[0], ref=ref_image(y, m))]
        elif name == "refts":
            r = ref_image(y, m)
            sp = [split("tt", w), Split(_grad, _grad_adj, lambda v, t: soft_group(v, t, -3), w * p[0], offset=_grad(r))]
        elif name == "tgv":
            with torch.no_grad():
                return tgv_t(prob, y, m, w, w * p[0], w * p[1], iters, cg_iters, x0=x0_for(y, m))
        elif name == "ls":
            with torch.no_grad():
                return lplus_s(prob, y, m, w * p[0], w, iters, x0_for(y, m))
        else:
            raise ValueError(f"unknown method {name}")
        sp = [s for s in sp if s.beta > 0]
        with torch.no_grad():
            return admm_multi(prob, y, m, sp, iters, cg_iters, x0=x0_for(y, m))
    return solve


def cv_curve(prob: DynProblem, solver, grid, holdout=0.1, folds=2, seed=0):
    """Scale-invariant held-out k-space error per grid weight: whole lines withheld in every frame (as
    bspinn.prior.dynamic_prior_image)."""
    T = prob.T
    m = prob.mnp
    err = np.zeros(len(grid))
    yk = prob.k / prob.s
    for f in range(folds):
        tr_t, ho_t = [], []
        for t in range(T):
            fm = cv_masks(m[t], holdout, folds, 0, seed + 1000 * t)
            tr, ho = fm[f % len(fm)] if fm else (m[t], np.zeros_like(m[t]))
            tr_t.append(tr); ho_t.append(ho)
        tr, ho = np.stack(tr_t), np.stack(ho_t)
        trt = prob.shift_mask(torch.from_numpy(tr.astype(np.float32)).to(prob.device))
        ytr = prob.y * trt[:, None]
        for gi, b in enumerate(grid):
            X = prob.out(solver(ytr, trt, float(b))).cpu().numpy()
            Cn = prob.Cn
            P = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Cn * X[:, None] if Cn.ndim == 4 else Cn[None] * X[:, None],
                                                             axes=(-2, -1)), norm="ortho"), axes=(-2, -1))
            ph = P.transpose(0, 2, 3, 1)[ho]; yh = yk.transpose(0, 2, 3, 1)[ho]
            al = (ph.conj() * yh).sum() / max((np.abs(ph) ** 2).sum(), 1e-30)
            err[gi] += float((np.abs(al * ph - yh) ** 2).sum())
    return err.tolist()
