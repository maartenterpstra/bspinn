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


class ADMM:
    """Scaled-form ADMM for 1/2 ||M F (C x) - y||^2 + b_tv TV(x) + b_w ||W x||_1 + lam/2 ||x||^2, batched over
    a leading axis of L candidate weights (or L slices). The splits and duals persist between ``run`` calls,
    so it can be warm-started repeatedly while the maps change (the joint fit's image update)."""

    def __init__(self, shape, beta, prior="tv", ratio=1.0, rho=0.5, lam=1e-6, levels=4, wave="db4", fill=1.0,
                 device="cpu"):
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}")
        beta = torch.as_tensor(beta, dtype=torch.float32, device=device).reshape(-1)
        L = beta.numel()
        self.r_ = rho * float(fill)
        self.lam = lam
        self.b_tv = beta.view(L, 1, 1) if "tv" in prior else None
        self.b_w = (beta * (ratio if prior == "tv+wavelet" else 1.0)).view(L, 1, 1) if "wavelet" in prior else None
        self.Wt = Wavelet2D(shape, levels, wave, device) if self.b_w is not None else None
        self.z1 = self.u1 = self.z2 = self.u2 = None

    def run(self, x, C, m, y, iters=50, cg_iters=5):
        """x (L,H,W) start; C (N,H,W) or (L,N,H,W) RSS-normalised maps; m (H,W) float; y like C."""
        r_ = self.r_

        def AHA(v):
            return (C.conj() * _ifft2c(m * _fft2c(C * v[:, None]))).sum(-3)

        def lhs(v):
            out = AHA(v) + self.lam * v
            if self.b_tv is not None:
                out = out + r_ * _grad_adj(_grad(v))
            if self.Wt is not None:
                out = out + r_ * v
            return out

        AHy = (C.conj() * _ifft2c(m * y)).sum(-3)
        if AHy.dim() == 2:
            AHy = AHy[None].expand(x.shape[0], -1, -1)
        if self.b_tv is not None and self.z1 is None:
            self.z1 = _grad(x); self.u1 = torch.zeros_like(self.z1)
        if self.Wt is not None and self.z2 is None:
            self.z2 = self.Wt.fwd(x); self.u2 = torch.zeros_like(self.z2)
        for _ in range(iters):
            rhs = AHy.clone()
            if self.z1 is not None:
                rhs = rhs + r_ * _grad_adj(self.z1 - self.u1)
            if self.z2 is not None:
                rhs = rhs + r_ * self.Wt.adj(self.z2 - self.u2)
            x = _bcg(lhs, rhs, x, cg_iters)
            if self.z1 is not None:
                v = _grad(x) + self.u1
                mag = torch.sqrt((v.abs() ** 2).sum(-3, keepdim=True)).clamp_min(1e-12)
                self.z1 = v * torch.clamp(1 - (self.b_tv / r_)[:, None] / mag, min=0)
                self.u1 = v - self.z1
            if self.z2 is not None:
                v = self.Wt.fwd(x) + self.u2
                shr = torch.clamp(1 - (self.b_w / r_) / v.abs().clamp_min(1e-12), min=0)
                self.z2 = torch.where(self.Wt.detail, v * shr, v)
                self.u2 = v - self.z2
        return x


def admm(y, C, m, beta, prior="tv", ratio=1.0, iters=50, cg_iters=5, rho=0.5, lam=1e-6, levels=4, wave="db4", x0=None,
         **ext):
    """y (N,H,W) complex tensor (zero outside m), C (N,H,W) RSS-normalised maps, m (H,W) float mask,
    beta (L,) weights -> images (L,H,W). ``prior`` also takes the opt-in EXTRA_PRIORS (r2-image): 'tgv'
    (second-order total generalised variation, ext ``tgv_ratio`` = alpha0 / alpha1, default 2), 'pnp:bm3d' /
    'pnp:nlm' (plug-and-play ADMM, beta = denoiser sigma^2 / rho). ``ext`` (opt-in): phase (H,W) +
    phase_weight (soft real-valued image after removing a smooth phase), w_map (H,W spatial TV weight),
    bregman (int, outer Bregman / add-back-the-residual iterations); see ADMMx, tgv_admm, pnp_admm."""
    L = beta.numel()
    if x0 is None:
        x = (C.conj() * _ifft2c(m * y)).sum(0)[None].expand(L, -1, -1).clone()
    else:
        x = x0.expand(L, -1, -1).clone()
    nb = int(ext.pop("bregman", 0) or 0)
    yk = y
    for b in range(nb + 1):
        if prior == "tgv":
            e = dict(ext); tr = float(e.pop("tgv_ratio", 2.0))
            x = tgv_admm(yk, C, m, beta, ratio=tr, iters=iters, cg_iters=cg_iters, rho=rho, lam=lam, x0=x, **e)
        elif prior.startswith("pnp"):
            if torch.is_tensor(ext.get("phase")):
                ext["phase"] = ext["phase"].cpu().numpy()
            x = pnp_admm(yk, C, m, beta, denoiser=prior.split(":")[1] if ":" in prior else "bm3d", iters=iters,
                         cg_iters=cg_iters, rho=rho, lam=lam, x0=x, **ext)
        elif ext:
            solver = ADMMx(m.shape, beta, prior, ratio, rho, lam, levels, wave, float(m.mean()), y.device, **ext)
            x = solver.run(x, C, m, yk, iters, cg_iters)
        else:
            solver = ADMM(m.shape, beta, prior, ratio, rho, lam, levels, wave, float(m.mean()), y.device)
            x = solver.run(x, C, m, yk, iters, cg_iters)
        if b < nb:                      # Bregman: add the data residual back (debiasing; contrast restored)
            yk = yk + m * (y - _fft2c(C * x[:, None])) if yk.dim() == 4 else \
                yk[None] + m * (y[None] - _fft2c(C * x[:, None]))
    return x


# ----------------------------------------------------------------------------- r2-image: opt-in extensions
EXTRA_PRIORS = ("tgv", "pnp:bm3d", "pnp:nlm")


def _shrink_iso(v, t, dim=-3):
    mag = torch.sqrt((v.abs() ** 2).sum(dim, keepdim=True)).clamp_min(1e-12)
    return v * torch.clamp(1 - t / mag, min=0)


def _dfb(x, ax):           # backward difference (periodic) = -(forward difference)^H
    return x - torch.roll(x, 1, ax)


def _dfb_adj(z, ax):
    return z - torch.roll(z, -1, ax)


_S2 = 2 ** -0.5


def _sym_grad(v):
    """Symmetrised gradient of a vector field v (...,2,H,W) -> (...,3,H,W), the off-diagonal scaled by sqrt 2 so
    the Frobenius norm is the plain 2-norm over the 3 components (backward differences: TGV convention)."""
    a, b = v[..., 0, :, :], v[..., 1, :, :]
    return torch.stack([_dfb(a, -2), _dfb(b, -1), _S2 * (_dfb(a, -1) + _dfb(b, -2))], -3)


def _sym_grad_adj(z):
    e0, e1, e2 = z[..., 0, :, :], z[..., 1, :, :], z[..., 2, :, :]
    return torch.stack([_dfb_adj(e0, -2) + _S2 * _dfb_adj(e2, -1), _dfb_adj(e1, -1) + _S2 * _dfb_adj(e2, -2)], -3)


def tgv_admm(y, C, m, beta, ratio=2.0, iters=50, cg_iters=8, rho=0.5, lam=1e-6, x0=None, w_map=None, phase=None,
             phase_weight=0.0):
    """TGV^2: min_x,v 1/2||M F C x - y||^2 + beta ||grad x - v||_{2,1} + ratio beta ||E v||_{2,1} (isotropic),
    ADMM with splits z1 = grad x - v, z2 = E v; the (x, v) update is a joint CG (packed (L,3,H,W)).
    beta (L,) -> x (L,H,W). w_map: optional (H,W) spatial weight on both terms; phase / phase_weight as ADMMx."""
    L = beta.numel()
    r_ = rho * float(m.mean())
    ph = None
    if phase is not None and phase_weight > 0:
        ph = torch.exp(1j * torch.as_tensor(phase, device=y.device).float()).to(torch.complex64)
        g = float(phase_weight) * float(m.mean())
    a1 = beta.view(L, 1, 1, 1).to(y.device)
    a0 = a1 * ratio
    if w_map is not None:
        a1 = a1 * w_map; a0 = a0 * w_map

    def AHA(v):
        return (C.conj() * _ifft2c(m * _fft2c(C * v[:, None]))).sum(-3)

    def lhs(w):
        x, v = w[:, 0], w[:, 1:]
        gx = _grad(x) - v
        ox = AHA(x) + lam * x + r_ * _grad_adj(gx)
        if ph is not None:
            ox = ox + g * ph * (1j * (ph.conj() * x).imag)
        ov = -r_ * gx + r_ * _sym_grad_adj(_sym_grad(v)) + lam * v
        return torch.cat([ox[:, None], ov], 1)

    AHy = (C.conj() * _ifft2c(m * y)).sum(-3)
    if AHy.dim() == 2:
        AHy = AHy[None].expand(L, -1, -1)
    x = x0.clone() if x0 is not None else AHy.clone()
    w = torch.cat([x[:, None], torch.zeros(L, 2, *x.shape[-2:], dtype=x.dtype, device=x.device)], 1)
    z1 = _grad(x); u1 = torch.zeros_like(z1)
    z2 = torch.zeros(L, 3, *x.shape[-2:], dtype=x.dtype, device=x.device); u2 = torch.zeros_like(z2)
    for _ in range(iters):
        rx = AHy + r_ * _grad_adj(z1 - u1)
        rv = -r_ * (z1 - u1) + r_ * _sym_grad_adj(z2 - u2)
        w = _bcg(lhs, torch.cat([rx[:, None], rv], 1), w, cg_iters)
        x, v = w[:, 0], w[:, 1:]
        q = _grad(x) - v + u1
        z1 = _shrink_iso(q, a1 / r_); u1 = q - z1
        q = _sym_grad(v) + u2
        z2 = _shrink_iso(q, a0 / r_); u2 = q - z2
    return w[:, 0]


class ADMMx(ADMM):
    """ADMM with opt-in extras (r2-image):
      w_map      (H,W) spatial multiplier of the TV weight (e.g. the g-factor map, see noise_map);
      phase      (H,W) real tensor phi: + phase_weight/2 ||Im(e^{-i phi} x)||^2 (a soft real-valued image
                 after removing a smooth phase; 'virtual conjugate coils'); phase_weight relative to fill."""

    def __init__(self, shape, beta, prior="tv", ratio=1.0, rho=0.5, lam=1e-6, levels=4, wave="db4", fill=1.0,
                 device="cpu", w_map=None, phase=None, phase_weight=0.0):
        super().__init__(shape, beta, prior, ratio, rho, lam, levels, wave, fill, device)
        self.fill = float(fill)
        if w_map is not None and self.b_tv is not None:     # (L,1,H,W) for the (L,2,H,W) shrink
            self.b_tv = self.b_tv.view(-1, 1, 1, 1) * torch.as_tensor(w_map, device=device).float()
        self.ph = None
        if phase is not None and phase_weight > 0:
            self.ph = torch.exp(1j * torch.as_tensor(phase, device=device).float()).to(torch.complex64)
            self.g = float(phase_weight) * self.fill

    def run(self, x, C, m, y, iters=50, cg_iters=5):
        r_ = self.r_
        ph = self.ph

        def AHA(v):
            return (C.conj() * _ifft2c(m * _fft2c(C * v[:, None]))).sum(-3)

        def lhs(v):
            out = AHA(v) + self.lam * v
            if ph is not None:            # real-linear, self-adjoint in Re<.,.>: CG with real inner products
                out = out + self.g * ph * (1j * (ph.conj() * v).imag)
            if self.b_tv is not None:
                out = out + r_ * _grad_adj(_grad(v))
            if self.Wt is not None:
                out = out + r_ * v
            return out

        AHy = (C.conj() * _ifft2c(m * y)).sum(-3)
        if AHy.dim() == 2:
            AHy = AHy[None].expand(x.shape[0], -1, -1)
        if self.b_tv is not None and self.z1 is None:
            self.z1 = _grad(x); self.u1 = torch.zeros_like(self.z1)
        if self.Wt is not None and self.z2 is None:
            self.z2 = self.Wt.fwd(x); self.u2 = torch.zeros_like(self.z2)
        bt = self.b_tv if self.b_tv is None or self.b_tv.dim() == 4 else self.b_tv[:, None]
        for _ in range(iters):
            rhs = AHy.clone()
            if self.z1 is not None:
                rhs = rhs + r_ * _grad_adj(self.z1 - self.u1)
            if self.z2 is not None:
                rhs = rhs + r_ * self.Wt.adj(self.z2 - self.u2)
            x = _bcg(lhs, rhs, x, cg_iters)
            if self.z1 is not None:
                v = _grad(x) + self.u1
                self.z1 = _shrink_iso(v, bt / r_); self.u1 = v - self.z1
            if self.z2 is not None:
                v = self.Wt.fwd(x) + self.u2
                shr = torch.clamp(1 - (self.b_w / r_) / v.abs().clamp_min(1e-12), min=0)
                self.z2 = torch.where(self.Wt.detail, v * shr, v)
                self.u2 = v - self.z2
        return x


def _denoise(v, sigma, kind, phase=None):
    """Complex image denoiser for PnP: real and imaginary parts (after removing ``phase``) separately."""
    import numpy as _np
    z = v.cpu().numpy()
    if phase is not None:
        z = z * _np.exp(-1j * phase)
    out = _np.empty_like(z)
    for i in range(z.shape[0]):
        parts = []
        for p in (z[i].real, z[i].imag):
            if kind == "bm3d":
                import bm3d
                parts.append(bm3d.bm3d(p.astype(_np.float64), float(sigma[i]),
                                       stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING))
            else:
                from skimage.restoration import denoise_nl_means
                parts.append(denoise_nl_means(p, h=0.8 * float(sigma[i]), sigma=float(sigma[i]), fast_mode=True,
                                              patch_size=5, patch_distance=6))
        out[i] = parts[0] + 1j * parts[1]
    if phase is not None:
        out = out * _np.exp(1j * phase)
    return torch.from_numpy(out.astype(_np.complex64)).to(v.device)


def pnp_admm(y, C, m, beta, denoiser="bm3d", iters=30, cg_iters=5, rho=0.5, lam=1e-6, x0=None, phase=None):
    """Plug-and-play ADMM: x-step CG on (A^H A + r) x = A^H y + r (z - u), z = D(x + u; sigma = sqrt(beta / r)).
    beta (L,) relative weights as for TV (data scaled to p99 magnitude 1)."""
    L = beta.numel()
    r_ = rho * float(m.mean())
    sig = torch.sqrt(beta / r_).cpu().numpy()

    def lhs(v):
        return (C.conj() * _ifft2c(m * _fft2c(C * v[:, None]))).sum(-3) + (lam + r_) * v

    AHy = (C.conj() * _ifft2c(m * y)).sum(-3)
    if AHy.dim() == 2:
        AHy = AHy[None].expand(L, -1, -1)
    x = x0.clone() if x0 is not None else AHy.clone()
    z = x.clone(); u = torch.zeros_like(x)
    for _ in range(iters):
        x = _bcg(lhs, AHy + r_ * (z - u), x, cg_iters)
        z = _denoise(x + u, sig, denoiser, phase)
        u = u + x - z
    return x


def smooth_phase(x, width=24):
    """Low-resolution phase of x (H,W) or (L,H,W): Hann window of ``width`` k-space lines in both directions."""
    xx = x if x.dim() == 2 else x[0]
    H, W = xx.shape
    width = int(max(2, min(width, H, W)))
    wh = torch.zeros(H); wh[H // 2 - width // 2:H // 2 - width // 2 + width] = torch.hann_window(width, periodic=False)
    ww = torch.zeros(W); ww[W // 2 - width // 2:W // 2 - width // 2 + width] = torch.hann_window(width, periodic=False)
    k = _fft2c(xx) * (wh[:, None] * ww[None]).to(xx.device)
    return torch.angle(_ifft2c(k)).float()


def noise_map(C, m, lam=1e-2, probes=4, smooth=9, seed=0):
    """Relative noise standard deviation of the lightly regularised SENSE image (Monte Carlo, white k-space
    noise), smoothed and normalised to median 1 where |C| > 0: a g-factor-like map for spatial TV weights."""
    g = torch.Generator().manual_seed(seed)
    fill = float(m.mean())
    AHA = lambda v: (C.conj() * _ifft2c(m * _fft2c(C * v[:, None]))).sum(-3) + lam * fill * v  # noqa: E731
    n = (torch.randn((probes,) + tuple(C.shape), generator=g) + 1j * torch.randn((probes,) + tuple(C.shape), generator=g)).to(C.dtype).to(C.device)
    b = (C.conj() * _ifft2c(m * n)).sum(-3)
    xn = _bcg(AHA, b, torch.zeros_like(b), 30)
    v = (xn.abs() ** 2).mean(0)
    k = torch.ones(1, 1, smooth, smooth, device=C.device) / smooth ** 2
    v = torch.nn.functional.conv2d(v[None, None].float(), k, padding=smooth // 2)[0, 0]
    sd = v.sqrt()
    sup = (C.abs() ** 2).sum(0) > 0.5
    return (sd / sd[sup].median().clamp_min(1e-12)).clamp(0.25, 4.0)


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


AUTO_SCALE = 0.35     # the CV folds withhold 10 % of the lines, a harder problem that wants more regularisation;
#                       v47: 0.35 centres the chosen weight on the oracle (median 0.92-0.96 x on two 20-slice runs)
#                       at the same NRMSE as 0.5 (which chose 1.3 x the oracle: visibly over-smoothed)
FALLBACK_WEIGHT = 2e-3  # used when the CV curve has no interior minimum
FLAT_CV = 0.04        # CV error range below which the curve is uninformative (noise-dominated slice); v47: 0.01
#                       missed a 2-4 % plateau (held-out worst case 1.97 -> 1.24 x the oracle NRMSE)


def select_weight(grid, err, scale=AUTO_SCALE, fallback=FALLBACK_WEIGHT, flat=FLAT_CV, refine=True):
    """Weight from a CV error curve over ``grid`` (rule of v46, constants of v47; calibrated on two disjoint
    sets of 20 dataloader slices x 2 map sets x R = 4, 6; README 8.9):
      flat      max/min error < 1 + flat: held-out k-space is noise-dominated, the data cannot tell -> the
                largest grid weight;
      interior  the lowest interior local minimum, refined by a parabola in log weight, times ``scale``;
                minima at the grid edges are ignored (the lowest-weight edge was a spurious global minimum
                in 20 % of the cases, at 1.2-3x the oracle NRMSE);
      upper     the largest grid weight is the lowest minimum (v49; the CV says: more regularisation) -> it;
      fallback  no interior minimum -> ``fallback``.
    Returns (weight, rule)."""
    grid = np.asarray(grid, float); e = np.asarray(err, float) / max(float(np.min(err)), 1e-30)
    if e.max() < 1 + flat:
        return float(grid[-1]), "flat"
    mins = [i for i in range(1, len(e) - 1) if e[i] <= e[i - 1] and e[i] <= e[i + 1]]
    up = len(e) - 1
    if e[up] <= e[up - 1] and (not mins or e[up] < min(e[j] for j in mins)):
        return float(grid[up]), "upper"        # the strongest weight is best (unscaled: the optimum may lie beyond)
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
    kw = resolve_options(kw, Ct, mt, x0, calib)
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
    kw = resolve_options(kw, Ct, mt, x0, int((cv_options or {}).get("calib", 24)))
    with torch.no_grad():
        x = admm(yt, Ct, mt, torch.tensor([float(weight)], device=device), prior, x0=x0, **kw)[0] * s
    return x.cpu().numpy().astype(np.complex64)


# ----------------------------------------------------------------------------------- dynamic (x, y, t)
def _tgrad(x):
    return torch.roll(x, -1, 0) - x                       # cyclic in time (cine: one cardiac cycle)


def _tgrad_adj(z):
    return torch.roll(z, 1, 0) - z


def dynamic_admm(y, C, m, beta_t, beta_s, iters=50, cg_iters=5, rho=0.5, lam=1e-6, x0=None, ref=None):
    """min 1/2 sum_t ||M_t F (C_t x_t) - y_t||^2 + beta_t sum |x_{t+1} - x_t| + beta_s sum_t TV(x_t - ref), ADMM.
    y (T,N,H,W), C (T,N,H,W) or (N,H,W) RSS-normalised maps, m (T,H,W) float masks, ref None (spatial TV of the
    frames) or (H,W) a reference image (v57: spatial TV of each frame's difference from it) -> x (T,H,W)."""
    # v57: everything ifftshifted once, plain FFTs inside (the centred FFT's shifts cost ~5x the transform on a
    # CPU); temporal and spatial TV are cyclic, so the solution is the same, shifted back at the end
    sh = lambda a: torch.fft.ifftshift(a, dim=(-2, -1))  # noqa: E731
    y, C, m = sh(y), sh(C), sh(m)
    x0 = None if x0 is None else sh(x0)
    gref = 0 if ref is None else _grad(sh(ref))[None]
    mm = m[:, None]                                       # (T,1,H,W)
    r_ = rho * float(m.mean())
    F = lambda a: torch.fft.fft2(a, norm="ortho")  # noqa: E731
    Fi = lambda a: torch.fft.ifft2(a, norm="ortho")  # noqa: E731

    def AHA(v):
        return (C.conj() * Fi(mm * F(C * v[:, None]))).sum(-3)

    def lhs(v):
        out = AHA(v) + lam * v + r_ * _tgrad_adj(_tgrad(v))
        if beta_s > 0:
            out = out + r_ * _grad_adj(_grad(v))
        return out

    AHy = (C.conj() * Fi(mm * y)).sum(-3)
    x = AHy.clone() if x0 is None else x0.clone()
    zt, ut = _tgrad(x), None
    ut = torch.zeros_like(zt)
    zs = _grad(x) - gref if beta_s > 0 else None
    us = torch.zeros_like(zs) if zs is not None else None
    for _ in range(iters):
        rhs = AHy + r_ * _tgrad_adj(zt - ut)
        if zs is not None:
            rhs = rhs + r_ * _grad_adj(zs + gref - us)
        # one batch of T coupled frames: CG over all of them at once (dims 0..2 summed)
        x = _bcg(lambda v: lhs(v[0])[None], rhs[None], x[None], cg_iters)[0]
        v = _tgrad(x) + ut
        zt = v * torch.clamp(1 - (beta_t / r_) / v.abs().clamp_min(1e-12), min=0)
        ut = v - zt
        if zs is not None:
            v = _grad(x) - gref + us
            mag = torch.sqrt((v.abs() ** 2).sum(-3, keepdim=True)).clamp_min(1e-12)
            zs = v * torch.clamp(1 - (beta_s / r_) / mag, min=0)
            us = v - zs
    return torch.fft.fftshift(x, dim=(-2, -1))


def _dyn_setup(ksp, C, mask, device):
    """ksp (N,H,W,T), C (N,H,W) or (N,H,W,T), mask (H,W,T) -> tensors (T-first), scale, light-Tikhonov start."""
    k = np.moveaxis(np.asarray(ksp), -1, 0)                                   # (T,N,H,W)
    m = np.moveaxis(np.asarray(mask, bool), -1, 0)                            # (T,H,W)
    C = np.asarray(C)
    C = np.moveaxis(C, -1, 0) if C.ndim == 4 else C                           # (T,N,H,W) | (N,H,W)
    C = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(-3, keepdims=True)), 1e-12)
    Ct = torch.from_numpy(C.astype(np.complex64)).to(device)
    mt = torch.from_numpy(m.astype(np.float32)).to(device)
    yt = torch.from_numpy((k * m[:, None]).astype(np.complex64)).to(device)
    fill = float(mt.mean())
    AHA = lambda v: (Ct.conj() * _ifft2c(mt[:, None] * _fft2c(Ct * v[:, None]))).sum(-3) + 1e-2 * fill * v  # noqa: E731
    b = (Ct.conj() * _ifft2c(mt[:, None] * yt)).sum(-3)
    x0 = _bcg(lambda v: AHA(v[0])[None], b[None], torch.zeros_like(b)[None], 30)[0]
    s = float(torch.quantile(x0.abs().flatten()[:: max(1, x0.numel() // 2_000_000)], 0.99).clamp_min(1e-12))
    return k, m, C, Ct, mt, yt / s, x0 / s, s


DYN_SCALE = 1.0   # dynamic: the CV interior minimum sat at the oracle weight on a synthetic GRO cine (R = 6);
#                   the 2-D AUTO_SCALE (0.35) halved the accuracy there. Calibrate with scripts/ocmr_eval.py --oracle


def dynamic_prior_image(ksp, C, mask, weight="auto", spatial_ratio=0.25, device="cpu", grid=None, report=None,
                        holdout=0.1, folds=2, calib=0, seed=0, iters=50, cg_iters=5, scale=DYN_SCALE,
                        oracle_ref=None, oracle_fn=None, spatial_on="image"):
    """Cine image series with temporal TV (+ spatial_ratio x the weight of spatial TV). ksp (N,H,W,T)
    acquired k-space (zeros elsewhere), C static (N,H,W) or per-frame (N,H,W,T) maps, mask (H,W,T).
    weight 'auto': k-space cross-validation with whole phase-encode lines withheld from every frame,
    ``select_weight`` rule. ``oracle_fn(x (T,H,W)) -> error`` optionally scores every grid weight (report).
    spatial_on: 'image' (default: TV of every frame) | 'difference' (v57: TV of every frame's difference from the
    time-averaged image, the SENSE-1 combination of the time-averaged k-space with the mean maps; README 8.13).
    Returns x (H,W,T) complex64."""
    k, m, Cn, Ct, mt, yt, x0, s = _dyn_setup(ksp, C, mask, device)
    T = k.shape[0]
    grid = np.asarray(grid if grid is not None else np.logspace(-4, 0, 9))      # dynamic optima reach ~0.1
    if spatial_on not in ("image", "difference"):
        raise ValueError("spatial_on must be 'image' or 'difference'")
    refs = {}

    def ref_of(yy, mm):                       # time-averaged reference of the samples in use (CV: training lines)
        if spatial_on == "image":
            return None
        key = id(mm)
        if key not in refs:
            refs.clear()
            cnt = mm.sum(0)
            ya = (yy * mm[:, None]).sum(0) / cnt.clamp_min(1)[None]
            Cm = Ct if Ct.ndim == 3 else Ct.mean(0)
            ma = (cnt > 0).float()
            AHA = lambda v: (Cm.conj() * _ifft2c(ma * _fft2c(Cm * v))).sum(0) + 1e-3 * v  # noqa: E731
            b = (Cm.conj() * _ifft2c(ma * ya)).sum(0)
            refs[key] = _bcg(lambda v: AHA(v[0])[None], b[None], torch.zeros_like(b)[None], 30)[0]
        return refs[key]
    solve = lambda yy, mm, b: dynamic_admm(yy, Ct, mm, b, b * spatial_ratio, iters, cg_iters, x0=x0,  # noqa: E731
                                           ref=ref_of(yy, mm))
    info = {"spatial_ratio": spatial_ratio, "spatial_on": spatial_on}
    if isinstance(weight, str):
        err = np.zeros(len(grid))
        rng_seed = seed
        tr_all, ho_all = [], []
        for f in range(folds):
            tr_t, ho_t = [], []
            for t in range(T):
                fm = cv_masks(m[t], holdout, folds, calib, rng_seed + 1000 * t)
                tr, ho = fm[f % len(fm)] if fm else (m[t], np.zeros_like(m[t]))
                tr_t.append(tr); ho_t.append(ho)
            tr_all.append(np.stack(tr_t)); ho_all.append(np.stack(ho_t))
        for gi, b in enumerate(grid):
            for tr, ho in zip(tr_all, ho_all):
                trt = torch.from_numpy(tr.astype(np.float32)).to(device)
                with torch.no_grad():
                    X = solve(yt * trt[:, None], trt, float(b)).cpu().numpy()          # (T,H,W)
                # scale-invariant held-out error over all frames jointly
                P = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Cn * X[:, None] if Cn.ndim == 4 else Cn[None] * X[:, None],
                                                                 axes=(-2, -1)), norm="ortho"), axes=(-2, -1))
                ph = P.transpose(0, 2, 3, 1)[ho]; yh = (k / s).transpose(0, 2, 3, 1)[ho]
                al = (ph.conj() * yh).sum() / max((np.abs(ph) ** 2).sum(), 1e-30)
                err[gi] += float((np.abs(al * ph - yh) ** 2).sum())
        w, rule = select_weight(grid, err, scale)
        info.update(grid=grid.tolist(), cv_error=err.tolist(), weight=w, rule=rule, method="cv")
        if oracle_fn is not None:
            curve = []
            for b in grid:
                with torch.no_grad():
                    curve.append(float(oracle_fn(np.moveaxis(solve(yt, mt, float(b)).cpu().numpy() * s, 0, -1))))
            info["grid_error"] = curve
    else:
        w = float(weight); info.update(weight=w, method="manual")
    with torch.no_grad():
        x = solve(yt, mt, w).cpu().numpy() * s
    if report is not None:
        report.update(info)
    return np.moveaxis(x, 0, -1).astype(np.complex64)


def resolve_options(kw, Ct, mt, x0, calib=24):
    """Turn string options into tensors: phase='lowres' -> smooth_phase of the SENSE start (Hann, ``calib`` lines),
    w_map='noise' -> noise_map (g-factor-like) ** w_power (default 1)."""
    kw = dict(kw)
    if isinstance(kw.get("phase"), str):
        kw["phase"] = smooth_phase(x0, int(kw.pop("phase_width", calib)))
    else:
        kw.pop("phase_width", None)
    if isinstance(kw.get("w_map"), str):
        kw["w_map"] = noise_map(Ct, mt) ** float(kw.pop("w_power", 1.0))
    else:
        kw.pop("w_power", None)
    return kw
