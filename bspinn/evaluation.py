"""Metrics for comparing coil-sensitivity estimators (used by ``scripts/evaluate_methods.py``).

All functions take maps C (N,H,W) or, for 3-D encoded data, (N,D,H,W) as returned by
:mod:`bspinn.methods` and an :class:`EvalCase`.
Data are pre-whitened (noise variance sigma^2 per k-space sample, independent between channels),
so "noise floor" values are exact expectations, not estimates.

Groups
------
1. k-space consistency of *combine -> expand* (``kspace_consistency``): coil images are combined with
   the maps (per-pixel least squares, m = C^H I / C^H C), re-expanded (C m) and transformed back to
   k-space; the residual is measured at the acquired samples, split into the calibration (ACS) region,
   which the estimators saw, and the rest of k-space, which they did not. For perfect maps the residual
   is pure noise with expected energy (N-1) sigma^2 per sample, reported as ``floor``.
2. Reference-image quality (``reference_image``): SENSE-1 combination of the fully sampled data with
   RSS-normalised maps; noise-debiased SNR efficiency |c^H i| / ||i|| relative to the optimum (=1 for
   the true maps), and, with ground truth, NRMSE / SSIM / PSNR against |rho| ||c_true||.
3. Parallel imaging (``sense_at_R``): retrospective uniform undersampling (+ACS lines), CG-SENSE,
   image metrics against a method-independent reference (RSS or ground truth) and against the
   method's own fully sampled combination, error of the *predicted missing k-space*, analytic g-factor.
4. Map properties: ground-truth errors (subspace, magnitude, relative phase, absolute), coverage,
   gauge-invariant roughness, phase vortices.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .methods import EvalCase, fft2c, ifft2c


# ----------------------------------------------------------------------------------- helpers
def normalise(C, eps=1e-12):
    s = np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True))
    return np.where(s > eps, C / np.maximum(s, eps), 0)


def object_mask(case: EvalCase, frac=0.05, noise_factor=1.3, smooth=9):
    """Synthetic: the ground-truth object. Real: smoothed RSS above ``noise_factor`` x the expected
    RSS of pure noise (sigma sqrt(N fill)) and above ``frac`` x its 99th percentile; opened, holes
    filled, components smaller than 1 % of the image dropped."""
    from scipy.ndimage import binary_fill_holes, binary_opening, label, uniform_filter
    if case.gt_object is not None:
        return np.abs(case.gt_object) > 1e-6
    if case.fully_sampled:
        I = ifft2c(case.ksp * case.sampled[None]); fill = case.sampled.mean()
    else:   # prospective (2-D): acquired samples of the central 64x64 (variable density, mostly sampled there)
        H, W = case.shape
        box = np.zeros(case.shape, bool); box[H // 2 - 32:H // 2 + 32, W // 2 - 32:W // 2 + 32] = True
        I = ifft2c(case.ksp * (case.sampled & box)[None]); fill = (case.sampled & box).mean()
        frac = max(frac, 0.15)
    rss = uniform_filter(np.sqrt((np.abs(I) ** 2).sum(0)), smooth)
    thr = max(frac * np.quantile(rss, 0.99), noise_factor * case.sigma * math.sqrt(case.N * fill))
    m = binary_fill_holes(binary_opening(rss > thr, iterations=2))
    lab, n = label(m)
    keep = np.zeros_like(m)
    for i in range(1, n + 1):
        if (lab == i).sum() > 0.01 * m.size:
            keep |= lab == i
    return keep


def combine(C, I):
    """Per-pixel least-squares combination (SENSE-1). Zero where the maps vanish."""
    d = (np.abs(C) ** 2).sum(0)
    return np.where(d > 1e-20, (C.conj() * I).sum(0) / np.maximum(d, 1e-20), 0)


def expand(C, m):
    return C * m[None]


def _radius(shape):
    ks = np.meshgrid(*[(np.arange(n) - n // 2) / (n / 2) for n in shape], indexing="ij")
    return np.sqrt(sum(k ** 2 for k in ks))


# ----------------------------------------------------------------------------------- 1. k-space
def kspace_consistency(C, case: EvalCase, n_bins=10):
    """Combine the fully sampled coil images with C, expand, compare with the acquired k-space."""
    y = case.ksp * case.sampled[None]
    I = ifft2c(y)
    r = (y - fft2c(expand(C, combine(C, I)))) * case.sampled[None]
    acs = case.acs_mask()
    out = {}
    N, s2 = case.N, case.sigma ** 2
    # noise floor: the same combine/expand applied to pure (whitened) noise, Monte Carlo. Exactly the
    # residual perfect maps would leave; ~ (N-1) sigma^2 per sample (less with partial-Fourier gaps).
    rng = np.random.default_rng(1234)
    fl = np.zeros(case.shape)
    for _ in range(2):
        n = case.sigma * (rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape)) / math.sqrt(2) * case.sampled[None]
        rn = (n - fft2c(expand(C, combine(C, ifft2c(n))))) * case.sampled[None]
        fl += (np.abs(rn) ** 2).sum(0) / 2
    for name, reg in (("all", case.sampled), ("acs", acs), ("outside_acs", case.sampled & ~acs)):
        er = float((np.abs(r[:, reg]) ** 2).sum())
        ey = float((np.abs(y[:, reg]) ** 2).sum())
        floor = float(fl[reg].sum())
        out[name] = {"nrmse": math.sqrt(er / ey), "residual_over_noise_floor": er / floor,
                     "excess_rel": (er - floor) / ey, "noise_floor_nrmse": math.sqrt(floor / ey)}
    rad = _radius(case.shape)
    edges = np.linspace(0, rad[case.sampled].max() + 1e-9, n_bins + 1)
    prof = []
    for a, b in zip(edges[:-1], edges[1:]):
        reg = case.sampled & (rad >= a) & (rad < b)
        er = float((np.abs(r[:, reg]) ** 2).sum()); ey = float((np.abs(y[:, reg]) ** 2).sum())
        f = float(fl[reg].sum())
        prof.append({"r0": float(a), "r1": float(b), "rel_residual": er / max(ey, 1e-30),
                     "rel_noise_floor": f / max(ey, 1e-30), "residual_over_noise_floor": er / max(f, 1e-30)})
    out["radial_profile"] = prof
    # per-coil (outside ACS): where the maps fail
    reg = case.sampled & ~acs
    out["per_coil_nrmse_outside_acs"] = [float(np.sqrt((np.abs(r[j, reg]) ** 2).sum() / (np.abs(y[j, reg]) ** 2).sum()))
                                         for j in range(N)]
    return out


# ----------------------------------------------------------------------------------- 2. reference image
def image_metrics(x, ref, obj):
    """Magnitude images; x is scaled by the least-squares factor on the object first."""
    from skimage.metrics import structural_similarity
    a, b = np.abs(x), np.abs(ref)
    s = (a[obj] * b[obj]).sum() / max((a[obj] ** 2).sum(), 1e-30)
    a = a * s
    err = a[obj] - b[obj]
    nrmse = float(np.linalg.norm(err) / np.linalg.norm(b[obj]))
    psnr = float(20 * np.log10(b[obj].max() / max(np.sqrt((err ** 2).mean()), 1e-30)))
    _, smap = structural_similarity(a, b, data_range=float(b[obj].max()), full=True)
    return {"nrmse": nrmse, "psnr_db": psnr, "ssim": float(smap[obj].mean())}


def reference_image(C, case: EvalCase, obj):
    Cn = normalise(C)
    I = ifft2c(case.ksp * case.sampled[None])
    m = (Cn.conj() * I).sum(0)                                   # SENSE-1 with RSS-normalised maps
    fill = case.sampled.mean()
    s2 = case.sigma ** 2 * fill                                  # image-domain noise variance per coil
    num = np.abs(m) ** 2 - s2
    den = (np.abs(I) ** 2).sum(0) - case.N * s2
    eff = np.sqrt(np.clip(num, 0, None) / np.clip(den, 1e-30, None))
    good = obj & (den > 10 * case.N * s2)                        # debiasing is unreliable at very low SNR
    out = {"snr_efficiency_mean": float(np.clip(eff[good], 0, 1.2).mean()),
           "snr_efficiency_p05": float(np.quantile(eff[good], 0.05)),
           "snr_loss_db_mean": float(-20 * np.log10(np.clip(eff[good], 1e-3, None)).mean())}
    rss = np.sqrt((np.abs(I) ** 2).sum(0))
    out["vs_rss"] = image_metrics(m, rss, obj)
    if case.gt_maps is not None:
        truth = np.abs(case.gt_object) * np.sqrt((np.abs(case.gt_maps) ** 2).sum(0))
        out["vs_truth"] = image_metrics(m, truth, obj)
        # bias-field-free image: absolute maps (only meaningful if the maps carry the true |C|)
        out["absolute_combination_vs_object"] = image_metrics(combine(C, I), case.gt_object, obj)
    return out, m


# ----------------------------------------------------------------------------------- 3. parallel imaging
def undersampling_mask(case: EvalCase, R, acs_lines=None):
    """2-D: R lines along ``case.pe_axis`` + the ACS lines. 3-D: R = (R1, R2) along ``case.pe_axes``
    (readout fully sampled) + the ACS block in the phase-encoding plane."""
    if case.ndim == 3:
        a1, a2 = case.pe_axes
        L1, L2 = case.shape[a1], case.shape[a2]
        R1, R2 = (R, 1) if np.isscalar(R) else R
        p = np.zeros((L1, L2), bool)
        p[(L1 // 2) % R1::R1, (L2 // 2) % R2::R2] = True
        c1, c2 = case.acs[a1], case.acs[a2]
        p[L1 // 2 - c1 // 2: L1 // 2 - c1 // 2 + c1, L2 // 2 - c2 // 2: L2 // 2 - c2 // 2 + c2] = True
        ro = ({0, 1, 2} - {a1, a2}).pop()
        m = np.expand_dims(p, ro)
        return np.broadcast_to(m, case.shape) & case.sampled
    H, W = case.shape
    L = case.shape[case.pe_axis]
    lines = np.zeros(L, bool)
    lines[(L // 2) % R::R] = True
    a = acs_lines if acs_lines is not None else case.acs[case.pe_axis]
    lines[L // 2 - a // 2: L // 2 - a // 2 + a] = True
    m = lines[None, :] if case.pe_axis == 1 else lines[:, None]
    return np.broadcast_to(m, (H, W)) & case.sampled


def cg_sense(y, mask, C, lam=1e-3, iters=40, tol=1e-6):
    """argmin_x ||M F (C x) - y||^2 + lam ||x||^2 by CG (maps RSS-normalised by the caller)."""
    def AhA(x):
        return (C.conj() * ifft2c(mask[None] * fft2c(C * x[None]))).sum(0) + lam * x
    b = (C.conj() * ifft2c(mask[None] * y)).sum(0)
    x = np.zeros_like(b); r = b.copy(); p = r.copy(); rs = np.vdot(r, r).real; b0 = rs
    for _ in range(iters):
        Ap = AhA(p)
        al = rs / np.vdot(p, Ap).real
        x += al * p; r -= al * Ap
        rn = np.vdot(r, r).real
        if rn < tol * b0:
            break
        p = r + (rn / rs) * p; rs = rn
    return x


def gfactor(Cn, R, axis, obj):
    """Analytic SENSE g-factor for uniform undersampling by R along ``axis`` (2-D), or by
    R = (R1, R2) along axis = (a1, a2) (3-D, CAIPI-free). Exact when each matrix size is divisible by
    its factor. Whitened data, RSS-normalised maps."""
    Rs = (R,) if np.isscalar(R) else tuple(R)
    axs = (axis,) if np.isscalar(axis) else tuple(axis)
    Ls = [Cn.shape[1 + a] for a in axs]
    if any(L % r for L, r in zip(Ls, Rs)):
        return None
    k = len(axs)
    C = np.moveaxis(Cn, [1 + a for a in axs], list(range(1, 1 + k)))      # (N, L1[, L2], other...)
    rest = C.shape[1 + k:]
    C = C.reshape(C.shape[0], *Ls, -1)                                     # (N, L1[, L2], P)
    steps = [L // r for L, r in zip(Ls, Rs)]
    g = np.zeros(C.shape[1:])
    import itertools
    for res in itertools.product(*[range(s) for s in steps]):
        sets = list(itertools.product(*[[p + j * s for j in range(r)] for p, s, r in zip(res, steps, Rs)]))
        S = np.stack([C[(slice(None),) + idx] for idx in sets], -1)        # (N, P, nR)
        S = np.transpose(S, (1, 0, 2))                                     # (P, N, nR)
        A = np.conj(np.transpose(S, (0, 2, 1))) @ S
        Ai = np.linalg.pinv(A, rcond=1e-6, hermitian=True)
        gg = np.sqrt(np.clip(np.real(np.diagonal(Ai, axis1=1, axis2=2) * np.diagonal(A, axis1=1, axis2=2)), 0, None))
        for i, idx in enumerate(sets):
            g[idx] = gg[:, i]
    g = g.reshape(*Ls, *rest)
    g = np.moveaxis(g, list(range(k)), list(axs))
    v = g[obj & (g > 0)]
    if v.size == 0:
        return None
    return {"mean": float(v.mean()), "p95": float(np.quantile(v, 0.95)), "max": float(v.max())}, g


def sense_at_R(C, case: EvalCase, obj, R, ref_rss, lam=1e-3, iters=40, support=None):
    """``support``: optional (H,W) mask; maps are zeroed outside it (the image is then only solved
    for inside: a common support constraint, removing the effect of each method's background)."""
    Cn = normalise(C)
    if support is not None:
        Cn = Cn * support[None]
    y_full = case.ksp * case.sampled[None]
    m = undersampling_mask(case, R)
    x = cg_sense(y_full * m[None], m, Cn, lam, iters)
    own = (Cn.conj() * ifft2c(y_full)).sum(0)
    out = {"R_nominal": R if np.isscalar(R) else list(R), "R_effective": float(case.sampled.sum() / m.sum()),
           "vs_reference": image_metrics(x, ref_rss, obj), "vs_own_full_combination": image_metrics(x, own, obj)}
    # predicted k-space at the samples that were dropped (known from the fully sampled data)
    pred = fft2c(Cn * x[None])
    miss = case.sampled & ~m
    er = float((np.abs(pred[:, miss] - y_full[:, miss]) ** 2).sum())
    ey = float((np.abs(y_full[:, miss]) ** 2).sum())
    out["missing_kspace_nrmse"] = math.sqrt(er / ey)
    out["missing_kspace_noise_floor_nrmse"] = math.sqrt(case.N * case.sigma ** 2 * miss.sum() / ey)
    # data consistency at the acquired samples
    er2 = float((np.abs(pred[:, m] - y_full[:, m]) ** 2).sum()); ey2 = float((np.abs(y_full[:, m]) ** 2).sum())
    out["acquired_kspace_nrmse"] = math.sqrt(er2 / ey2)
    g = gfactor(Cn, R, case.pe_axis if case.ndim == 2 else case.pe_axes, obj)
    out["gfactor"] = g[0] if g else None
    return out, x


# ----------------------------------------------------------------------------------- 4. maps
def map_properties(C, case: EvalCase, obj):
    from .recon import phase_vortices
    Cn = normalise(C)
    pw = (np.abs(C) ** 2).sum(0)
    med = np.median(pw[obj]) if obj.any() else 1.0
    out = {"coverage": float((pw[obj] > 1e-3 * med).mean())}
    rough = []
    for ax in range(1, Cn.ndim):
        a = np.take(Cn, range(Cn.shape[ax] - 1), axis=ax)
        b = np.take(Cn, range(1, Cn.shape[ax]), axis=ax)
        ov = np.abs((a.conj() * b).sum(0)) ** 2
        o2 = np.take(obj, range(obj.shape[ax - 1] - 1), axis=ax - 1) & np.take(obj, range(1, obj.shape[ax - 1]), axis=ax - 1)
        rough.append(np.sqrt(np.clip(2 - 2 * ov, 0, None))[o2])
    out["projector_roughness"] = float(np.concatenate(rough).mean())
    if C.ndim == 4:      # 3-D: count in every slice along the first array axis
        v = phase_vortices(torch.from_numpy(np.ascontiguousarray(np.moveaxis(C, 1, 0))), torch.from_numpy(obj))
    else:
        v = phase_vortices(torch.from_numpy(C[None]), torch.from_numpy(obj))
    out["phase_vortices"] = v["total"]
    out["coils_with_vortices"] = v["coils_with_vortices"]
    if case.gt_maps is not None:
        G = case.gt_maps; Gn = normalise(G)
        ov = np.abs((Cn.conj() * Gn).sum(0)) ** 2
        out["gt_subspace_err"] = float((1 - ov)[obj].mean())
        out["gt_magnitude_err"] = float(np.linalg.norm((np.abs(Cn) - np.abs(Gn))[:, obj]) / np.linalg.norm(Gn[:, obj]))
        ph = np.angle((Cn.conj() * Gn).sum(0))                   # remove the common (unidentifiable) phase
        d = np.angle(Cn * np.exp(1j * ph)[None] * Gn.conj())
        w = np.abs(Gn) ** 2
        out["gt_rel_phase_err_deg"] = float(np.degrees((np.abs(d) * w)[:, obj].sum() / w[:, obj].sum()))
        a = (C[:, obj].conj() * G[:, obj]).sum() / max((np.abs(C[:, obj]) ** 2).sum(), 1e-30)
        out["gt_absolute_err"] = float(np.linalg.norm(a * C[:, obj] - G[:, obj]) / np.linalg.norm(G[:, obj]))
    return out


# ----------------------------------------------------------------------------------- prospective data
def heldout_prospective(C, case: EvalCase, obj, frac=0.2, seed=0, lam=1e-3, iters=40, folds=2):
    """No fully sampled reference: hold out a random fraction of acquired samples outside the ACS,
    reconstruct from the rest, predict the held-out samples. Also the data consistency (combine via
    CG-SENSE, expand) at the samples used."""
    Cn = normalise(C)
    rng = np.random.default_rng(seed)
    cand = case.sampled & ~case.acs_mask()
    idx = np.flatnonzero(cand)
    rng.shuffle(idx)
    res = []
    for f in range(folds):
        ho = np.zeros(case.shape, bool)
        sel = idx[int(f * frac * len(idx)): int((f + 1) * frac * len(idx))]
        ho.flat[sel] = True
        tr = case.sampled & ~ho
        x = cg_sense(case.ksp * tr[None], tr, Cn, lam, iters)
        pred = fft2c(Cn * x[None])
        y = case.ksp
        e_ho = float((np.abs(pred[:, ho] - y[:, ho]) ** 2).sum()); y_ho = float((np.abs(y[:, ho]) ** 2).sum())
        e_tr = float((np.abs(pred[:, tr] - y[:, tr]) ** 2).sum()); y_tr = float((np.abs(y[:, tr]) ** 2).sum())
        res.append({"heldout_nrmse": math.sqrt(e_ho / y_ho), "heldout_noise_floor_nrmse": math.sqrt(case.N * case.sigma ** 2 * ho.sum() / y_ho),
                    "acquired_nrmse": math.sqrt(e_tr / y_tr)})
    out = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    x = cg_sense(case.ksp * case.sampled[None], case.sampled, Cn, lam, iters)
    return out, x
