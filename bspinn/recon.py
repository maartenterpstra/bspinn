"""Reconstruction utilities: CG-SENSE, baseline (low-resolution division) maps, metrics."""
from __future__ import annotations

import math
import numpy as np
import torch

from .geometry import crop_center, fft2c, ifft2c


def normalize_maps(C: torch.Tensor, eps=1e-12) -> torch.Tensor:
    """ESPIRiT/RSS-style normalisation sum_j |C_j|^2 = 1 (keeps inter-coil phase)."""
    return C / torch.sqrt((C.abs() ** 2).sum(1, keepdim=True)).clamp_min(eps)


def lowres_division_maps(kspace: torch.Tensor, calib, window=True) -> torch.Tensor:
    """Classical self-calibrated maps (McKenzie / mSENSE): Hann-windowed calibration
    region zero-padded to full size, divided by its root-sum-of-squares."""
    S, N, H, W = kspace.shape
    h, w = calib
    kc = crop_center(kspace, (h, w))
    if window:
        wh = torch.hann_window(h + 2, periodic=False)[1:-1]
        ww = torch.hann_window(w + 2, periodic=False)[1:-1]
        kc = kc * (wh[:, None] * ww[None, :])
    kp = torch.zeros_like(kspace)
    r0, c0 = H // 2 - h // 2, W // 2 - w // 2
    kp[..., r0:r0 + h, c0:c0 + w] = kc
    return normalize_maps(ifft2c(kp))


def projection_residual(C: torch.Tensor, I: torch.Tensor, support: torch.Tensor) -> float:
    """Fraction of coil-image energy outside span{C(r)} at each pixel (ESPIRiT Eq. 20),
    summed over the support. Gauge-invariant (insensitive to per-pixel complex scaling)."""
    num = (C.conj() * I).sum(1).abs() ** 2
    den = (C.abs() ** 2).sum(1).clamp_min(1e-20)
    e = (I.abs() ** 2).sum(1)
    res = (e - num / den) * support
    return float(res.sum() / (e * support).sum())


def coil_combine(C, I):
    return (C.conj() * I).sum(1) / (C.abs() ** 2).sum(1).clamp_min(1e-12)


def cg_sense(op, y, C, lam=1e-3, iters=40, weights=None):
    """Tikhonov-regularised (optionally density-weighted) SENSE by conjugate gradients.
    op: encoding operator with forward (S,N,h,w)->data and adjoint; C: (S,N,h,w) maps."""
    wt = 1.0 if weights is None else weights

    def AhA(x):
        return (C.conj() * op.adjoint(wt * op.forward(C * x[:, None]))).sum(1) + lam * x

    b = (C.conj() * op.adjoint(wt * y)).sum(1)
    x = torch.zeros_like(b); r = b.clone(); p = r.clone()
    rs = (r.conj() * r).real.sum()
    for _ in range(iters):
        Ap = AhA(p)
        alpha = rs / (p.conj() * Ap).real.sum()
        x = x + alpha * p; r = r - alpha * Ap
        rs_new = (r.conj() * r).real.sum()
        if rs_new.sqrt() < 1e-6 * b.abs().norm():
            break
        p = r + (rs_new / rs) * p; rs = rs_new
    return x


def cg_sense_sets(op, y, Ks, lam=1e-3, iters=40, weights=None, ridge=1e-2):
    """SENSE with K map sets: y = E sum_k C_k x_k. Ks: (S,K,N,H,W) -> x (S,K,H,W).
    Sets >= 2 get an extra ridge (ENLIVE-style low-rank surrogate)."""
    wt = 1.0 if weights is None else weights
    K = Ks.shape[1]
    lamv = torch.full((K,), lam + ridge); lamv[0] = lam
    lamv = lamv[None, :, None, None].to(Ks.real.dtype)

    def AhA(x):
        r = op.adjoint(wt * op.forward((Ks * x[:, :, None]).sum(1)))
        return (Ks.conj() * r[:, None]).sum(2) + lamv * x

    b = (Ks.conj() * op.adjoint(wt * y)[:, None]).sum(2)
    x = torch.zeros_like(b); r = b.clone(); p = r.clone()
    rs = (r.conj() * r).real.sum()
    for _ in range(iters):
        Ap = AhA(p)
        alpha = rs / (p.conj() * Ap).real.sum()
        x = x + alpha * p; r = r - alpha * Ap
        rs_new = (r.conj() * r).real.sum()
        if rs_new.sqrt() < 1e-6 * b.abs().norm():
            break
        p = r + (rs_new / rs) * p; rs = rs_new
    return x


def upsample_lowres_division(I_low, shape):
    """Low-res division maps from (gridded) low-res coil images, Hann-windowed and
    sinc-interpolated to ``shape``."""
    S, N, h, w = I_low.shape
    k = fft2c(I_low)
    wh = torch.hann_window(h + 2, periodic=False)[1:-1]
    ww = torch.hann_window(w + 2, periodic=False)[1:-1]
    k = k * (wh[:, None] * ww[None, :])
    H, W = shape
    kp = torch.zeros(S, N, H, W, dtype=k.dtype)
    r0, c0 = H // 2 - h // 2, W // 2 - w // 2
    kp[..., r0:r0 + h, c0:c0 + w] = k
    return normalize_maps(ifft2c(kp))


def subspace_error(C_est, C_true, support):
    """1 - |<c_est,c_true>|^2/(|c_est|^2|c_true|^2), averaged over support (per-pixel gauge-free)."""
    num = (C_est.conj() * C_true).sum(1).abs() ** 2
    den = (C_est.abs() ** 2).sum(1) * (C_true.abs() ** 2).sum(1)
    e = 1 - num / den.clamp_min(1e-20)
    return float((e * support).sum() / support.sum())


def absolute_error(C_est, C_true, support):
    """Relative L2 error after one optimal *global* complex scale (no per-pixel freedom).
    Only physics-constrained maps can do well here; RSS-normalised maps cannot."""
    s = support[:, None]
    a = (C_est.conj() * C_true * s).sum() / (C_est.abs() ** 2 * s).sum()
    return float(((a * C_est - C_true).abs() ** 2 * s).sum().sqrt() / ((C_true.abs() ** 2 * s).sum().sqrt()))


def virtual_body_coil(C0, sup=None, robust=True, iters=300, restarts=8, seed=0):
    """Coil-combination weights d of a virtual body coil T(r) = d^H C(r) (Inati et al.).

    robust=False: dominant eigenvector of sum_r C C^H (Inati's choice, maximises mean |T|^2). It can
                  pass through zero inside the object and then plants a phase pole in every map.
    robust=True : maximise the geometric mean of |T|^2/||C||^2 over the object (log-barrier), from the
                  eigenvector and ``restarts`` random starts (the objective has local optima in which T
                  winds around the object and must vanish inside); keep the largest minimum strength.
    """
    S, N, H, W = C0.shape
    w = torch.ones(S, H, W) if sup is None else sup.reshape(S, H, W).float()
    X = C0.permute(1, 0, 2, 3).reshape(N, -1)[:, w.flatten() > 0]
    _, V = torch.linalg.eigh(X @ X.conj().T)
    d_eig = V[:, -1]
    if not robust:
        return d_eig
    g = torch.Generator().manual_seed(seed)
    Xn = X / X.abs().pow(2).sum(0).sqrt().clamp_min(1e-12)
    if Xn.shape[1] > 20000:
        Xn = Xn[:, torch.randperm(Xn.shape[1], generator=g)[:20000]]
    strength = lambda d: (d.conj() @ Xn).abs().min()
    best, best_s = d_eig, float(strength(d_eig))
    starts = [d_eig] + [torch.randn(N, dtype=torch.complex64, generator=g) for _ in range(restarts)]
    # all starts optimised together (Adam is element-wise, the summed losses are independent: identical
    # to one loop per start, ~(restarts + 1)x fewer operations)
    D0 = torch.stack([d0 / d0.norm() for d0 in starts])                         # (R,N)
    with torch.enable_grad():
        p = torch.nn.Parameter(torch.view_as_real(D0.clone()))
        opt = torch.optim.Adam([p], lr=2e-2)
        for _ in range(iters):
            dc = torch.view_as_complex(p); dc = dc / dc.norm(dim=1, keepdim=True)
            loss = -torch.log((dc.conj() @ Xn).abs() ** 2 + 1e-6).mean(1).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    D = torch.view_as_complex(p.detach()); D = D / D.norm(dim=1, keepdim=True)
    st = (D.conj() @ Xn).abs().min(1).values
    for r in range(D.shape[0]):
        if float(st[r]) > best_s:
            best, best_s = D[r], float(st[r])
    return best


def phase_reference(C, mode="inati", support=None, images=None, d=None):
    """Fix the per-pixel phase common to all coils (the gauge that SENSE cannot see).

    C: (S,N,H,W) or (S,K,N,H,W) complex maps; images (S,H,W) or (S,K,H,W) are counter-rotated
    so that C*m is unchanged. Inter-coil phases C_j C_k^* are never altered.

    'model' / 'none' : keep the phase of the model (Biot–Savart: B1- transverse-field angle).
    'inati'          : Inati et al. (ISMRM 2013/2014): reference all coils to a *virtual body
                       coil* T(r) = d^H C(r); afterwards T(r) is real positive and the remaining
                       phase goes into the image. d is chosen to keep T away from zero
                       (see virtual_body_coil), so no phase singularity is introduced.
    'inati-eig'      : the same with Inati's dominant-eigenvector weights (may be singular).
    'coil'           : reference to the strongest coil (ESPIRiT convention; singular where it is weak).
    Returns (C', images', info, Phi) with Phi (S,H,W) the unit-modulus phase removed from the maps
    (None for 'model').
    """
    if mode in ("model", "none"):
        return C, images, {"phase_ref": mode}, None
    sets = C.dim() == 5
    C0 = C[:, 0] if sets else C                                       # primary set defines the gauge
    S, N, H, W = C0.shape
    sup = torch.ones(S, H, W) if support is None else support.reshape(S, H, W).float()
    if mode in ("inati", "inati-eig"):
        d = virtual_body_coil(C0, sup, robust=(mode == "inati")) if d is None else d
        T = torch.einsum("n,snhw->shw", d.conj(), C0)
        if mode == "inati":
            rel0 = T.abs() / (C0.abs() ** 2).sum(1).sqrt().clamp_min(1e-12)
            s0 = float(rel0[sup > 0].min()) if (sup > 0).any() else 1.0
            if s0 < 0.05:
                # no safe virtual body coil found: keep the physical (model) phase rather than
                # introducing a reference-induced phase pole
                return C, images, {"phase_ref": "model", "requested": "inati",
                                   "note": f"virtual body coil too weak inside the object (min {s0:.3f}); "
                                           "kept the model phase to avoid a phase singularity"}, None
    elif mode == "coil":
        k = int(((C0.abs() ** 2) * sup[:, None]).sum((0, 2, 3)).argmax())
        T = C0[:, k]
    else:
        raise ValueError(f"unknown phase reference '{mode}'")
    mag = T.abs()
    Phi = T / mag.clamp_min(1e-12)
    rel = mag / ((C0.abs() ** 2).sum(1).sqrt().clamp_min(1e-12))
    smin = float(rel[sup > 0].min()) if (sup > 0).any() else None
    info = {"phase_ref": mode, "reference_strength_min_on_support": smin}
    if smin is not None and smin < 0.02:
        info["warning"] = ("phase reference nearly vanishes inside the object: the exported maps carry a "
                           "phase singularity there (try --phase-ref inati or model)")
    Cr = C * (Phi.conj()[:, None, None] if sets else Phi.conj()[:, None])
    if images is not None:
        images = images * (Phi[:, None] if images.dim() == 4 else Phi)
    return Cr, images, info, Phi


def phase_vortices(C: torch.Tensor, support: torch.Tensor, rel_mag: float = 0.0):
    """Count phase singularities (vortices) of every coil map inside the support.

    A plaquette (2x2 pixel square) is a vortex when the wrapped phase differences around it sum
    to +/-2*pi. Plaquettes touching pixels outside ``support`` or with |C| below ``rel_mag`` x the
    coil's 99th percentile are ignored (phase undefined in noise). C: (S,N,H,W); support: (S,H,W)
    or (H,W). Returns per-coil totals (positive + negative charge) and the net winding."""
    C = torch.as_tensor(C)
    if C.dim() == 3:
        C = C[None]
    sup = torch.as_tensor(support).bool()
    if sup.dim() == 2:
        sup = sup[None].expand(C.shape[0], -1, -1)
    ph = torch.angle(C)
    mag = C.abs()
    thr = rel_mag * torch.quantile(mag.flatten(2).transpose(0, 1).reshape(C.shape[1], -1).float(), 0.99, dim=1)
    ok = sup[:, None] & (mag > thr[None, :, None, None])                           # (S,N,H,W)

    def w(a):
        return torch.remainder(a + math.pi, 2 * math.pi) - math.pi

    p00, p01, p11, p10 = ph[..., :-1, :-1], ph[..., :-1, 1:], ph[..., 1:, 1:], ph[..., 1:, :-1]
    circ = w(p01 - p00) + w(p11 - p01) + w(p10 - p11) + w(p00 - p10)
    q = torch.round(circ / (2 * math.pi))
    valid = ok[..., :-1, :-1] & ok[..., :-1, 1:] & ok[..., 1:, 1:] & ok[..., 1:, :-1]
    q = q * valid
    per_coil = (q.abs().sum((0, 2, 3))).long().tolist()
    net = q.sum((0, 2, 3)).long().tolist()
    return {"total": int(sum(per_coil)), "per_coil": per_coil, "net_winding": net,
            "coils_with_vortices": int(sum(1 for c in per_coil if c > 0))}
