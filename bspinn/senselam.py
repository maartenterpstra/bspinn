"""Automatic Tikhonov weight for the CG-SENSE image: argmin ||M F (C x) - y||^2 + lam ||x||^2.

All rules work on the data actually acquired (no reference image) and return ``lam`` in the convention of
``api.sense_image`` (relative to mean_r sum_j |C_j(r)|^2, i.e. absolute for RSS-normalised maps). The
weight is dimensionless: scaling the data scales both terms alike.

Rules
-----
'cv+map'       default of 'auto': max(cv, map). Cross-validation picks the weight; the noise-based Wiener
               value is a floor against its rare collapses to tiny weights (few / noisy withheld lines).
               On 20 slices of fastMRI knee/brain and Calgary-Campinas (R=4 vd, ACS 24): NRMSE 1.007x
               the oracle (median), worst 1.06x, 95 % of slices within 5 % (cv alone: 1.011x, worst 1.47x,
               85 %; fixed 1e-2: 1.029x, worst 1.56x, 60 %; fixed 3e-2: 1.067x, worst 1.17x, 45 %).
'cv'           k-space cross-validation: a random ``holdout`` fraction of the acquired
               phase-encode lines (Cartesian line masks; points otherwise) outside the calibration box is
               withheld (``folds`` disjoint folds), the image is
               reconstructed from the rest for every lam on a log grid, and the withheld samples are
               predicted; the lam with the lowest prediction error wins. Predicting samples that were not
               used is predicting k-space the reconstruction had to unfold, so this tracks the image error
               (noise amplification AND residual aliasing / map errors), which data-space criteria (GCV,
               discrepancy) do not: they only see the fit to the samples used. Scale-invariant (best
               global scale per lam), like the image metrics. No noise level needed.
'map'          Gaussian MAP / Wiener: lam = sigma^2 / s^2, sigma^2 the noise variance per k-space sample
               (per channel, after whitening if noise samples are given; otherwise estimated from the
               outermost acquired k-space, robustly), s^2 the mean signal power per image pixel, from
               the density-compensated energy of the acquired samples minus the noise energy. Closed form,
               no extra reconstructions; optimal for an i.i.d. Gaussian image prior (real images are not,
               hence ``scale``).
'gcv'          Generalised cross-validation: lam minimising n ||r_lam||^2 / (n - tr H_lam)^2 over a log
               grid (default 1e-5..1, 16 values), tr H_lam = tr((A^H A + lam)^-1 A^H A) by Hutchinson (``probes`` Rademacher vectors,
               the same for every lam). Needs no noise level. Costs (1 + probes) batched CG solves.
'discrepancy'  Morozov: the lam where ||r_lam||^2 = sigma^2 (n - tr H_lam) (the expected residual of a
               linear estimator that only removes noise), interpolated on the grid; needs sigma^2. Map
               errors add to the residual, so with imperfect maps it can sit at the grid floor.
'oracle'       not a rule (``scripts/sense_lambda_eval.py``): the grid lam closest to a fully sampled
               reference, to calibrate the rules.

``choose_sense_lambda`` returns (lam, info) with every rule's value in ``info`` (cheap to compute together
once the grid solves are done).
"""
from __future__ import annotations

import numpy as np
import torch

RULES = ("cv+map", "cv", "map", "gcv", "discrepancy")


def _fft(x, dims):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)


def _ifft(x, dims):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)


def noise_variance(ksp, m, outer=0.85):
    """Per-sample complex noise variance from the acquired samples in the outer k-space shell
    (normalised Chebyshev radius > ``outer`` of the acquired extent), robust to residual signal:
    |n| is Rayleigh, P(|n| < q) = 1 - exp(-q^2 / sigma^2), so sigma^2 = q25^2 / -ln(0.75), per channel,
    averaged over channels. Falls back to the lowest-energy 5 % of acquired samples if the shell is empty."""
    sp = m.shape
    grids = np.meshgrid(*[np.abs(np.arange(n) - n // 2) / (n / 2) for n in sp], indexing="ij")
    r = np.maximum.reduce(grids)
    rmax = r[m].max() if m.any() else 1.0
    shell = m & (r > outer * rmax)
    if shell.sum() < 64:
        shell = m & (r > 0.7 * rmax)
    a = np.abs(ksp[:, shell]) if shell.sum() >= 16 else np.sort(np.abs(ksp[:, m]), axis=1)[:, : max(16, m.sum() // 20)]
    q = np.quantile(a, 0.25, axis=1)
    return float(np.mean(q ** 2 / -np.log(0.75)))


def density(m, size=9):
    """Local sampling density of the mask (box average), for density-compensated energies."""
    from scipy.ndimage import uniform_filter
    return np.clip(uniform_filter(m.astype(float), size=size, mode="nearest"), 1.0 / size ** m.ndim, 1.0)


def lambda_map(ksp, C, m, sigma2):
    """Wiener / Gaussian-MAP weight sigma^2 / s^2 (see module docstring)."""
    N = ksp.shape[0]
    d = density(m)
    e_tot = float(((np.abs(ksp[:, m]) ** 2) / d[m][None]).sum())        # density-compensated energy
    n_ext = float(_extent(m).sum())
    floor = e_tot - sigma2 * N * n_ext < 0.05 * e_tot      # (almost) no signal above the noise estimate
    e_sig = max(e_tot - sigma2 * N * n_ext, 1e-3 * e_tot)
    s2 = e_sig / m.size                                                  # per pixel (maps RSS-normalised)
    den = float((np.abs(C) ** 2).sum(0).mean())
    return sigma2 / s2 / max(den, 1e-12), {"sigma2": sigma2, "signal_power": s2, "map_signal_floor": bool(floor)}


def _extent(m):
    from .api import acquired_extent
    return acquired_extent(m)


def _solver(ksp, C, m, device):
    dev = torch.device(device)
    y = torch.from_numpy(np.ascontiguousarray(ksp, np.complex64)).to(dev)
    Ct = torch.from_numpy(np.ascontiguousarray(C, np.complex64)).to(dev)
    M = torch.from_numpy(np.ascontiguousarray(m)).to(dev)
    box = _extent(m)
    B = None if box.all() else torch.from_numpy(box).to(dev)
    sp = tuple(range(1, y.ndim))                 # spatial dims of (N, ...)
    spb = tuple(range(-len(sp), 0))              # spatial dims of a batch (L, ...)
    band = (lambda x: x) if B is None else (lambda x: _ifft(_fft(x, spb) * B, spb))

    def A(x):                                    # (L, ...) -> (L, N, ...)
        return M * _fft(Ct * band(x)[:, None], tuple(d + 1 for d in sp))

    def Ah(k):                                   # (L, N, ...) -> (L, ...)
        return band((Ct.conj() * _ifft(M * k, tuple(d + 1 for d in sp))).sum(1))

    def cg(lams, b, iters):
        """Batched CG: (A^H A + lam_l) x_l = b_l for every l (lams (L,), b (L, ...))."""
        lv = lams.view(-1, *([1] * (b.ndim - 1)))
        x = torch.zeros_like(b); r = b.clone(); p = r.clone()
        rs = (r.conj() * r).real.flatten(1).sum(1)
        for _ in range(iters):
            Ap = Ah(A(p)) + lv * p
            al = rs / (p.conj() * Ap).real.flatten(1).sum(1).clamp_min(1e-30)
            x = x + al.view(-1, *([1] * (b.ndim - 1))) * p
            r = r - al.view(-1, *([1] * (b.ndim - 1))) * Ap
            rn = (r.conj() * r).real.flatten(1).sum(1)
            p = r + (rn / rs.clamp_min(1e-30)).view(-1, *([1] * (b.ndim - 1))) * p
            rs = rn
        return band(x)
    return y, A, Ah, cg, dev


def grid_solutions(ksp, C, m, lams, device="cpu", iters=60):
    """CG-SENSE images for every relative lam in ``lams`` (batched): (L, ...) complex numpy."""
    y, A, Ah, cg, dev = _solver(ksp, C, m, device)
    den = float((np.abs(C) ** 2).sum(0).mean())
    lt = torch.tensor(np.asarray(lams) * den, dtype=torch.float32, device=dev)
    b = Ah(y[None]).expand(len(lams), *y.shape[1:]).clone()
    return cg(lt, b, iters).cpu().numpy()


def cv_masks(m, holdout=0.1, folds=2, calib=24, seed=0):
    """Hold-out folds for k-space cross-validation: [(train mask, held-out mask)]. Cartesian line sampling
    (2-D: every acquired line fully sampled along the readout): whole phase-encode lines outside the
    calibration region are withheld, so the reconstruction must unfold them as it unfolds the missing lines
    (withheld single points would be interpolated along the readout and hide the aliasing); otherwise points."""
    sp = m.shape
    box = np.zeros(sp, bool)
    box[tuple(slice(n // 2 - calib // 2, n // 2 - calib // 2 + calib) for n in sp)] = True
    rng = np.random.default_rng(seed)
    ro = [a for a in range(m.ndim) if m.ndim == 2 and bool((m == m.any(axis=a, keepdims=True)).all()) and not m.all()]
    if ro:
        pe = 1 - ro[0]
        lines = m.any(axis=ro[0])
        cbox = np.zeros(sp[pe], bool); cbox[sp[pe] // 2 - calib // 2: sp[pe] // 2 - calib // 2 + calib] = True
        units = np.flatnonzero(lines & ~cbox)
    else:
        units = np.flatnonzero(m & ~box)
    rng.shuffle(units)
    nho = max(1, int(holdout * units.size))
    out = []
    for f in range(folds):
        sel = units[(f * nho) % max(units.size, 1): (f * nho) % max(units.size, 1) + nho]
        if sel.size == 0:
            continue
        if ro:
            hl = np.zeros(sp[pe], bool); hl[sel] = True
            ho = m & (hl[None, :] if pe == 1 else hl[:, None])
        else:
            ho = np.zeros(m.size, bool); ho[sel] = True; ho = ho.reshape(sp)
        out.append((m & ~ho, ho))
    return out


def heldout_error(X, C, ksp, ho):
    """Scale-invariant prediction error of images X (L, ...) on the held-out samples ``ho``: the best global
    (complex) scale per candidate first, as image metrics compare up to a scale; otherwise the plain
    shrinkage of a stronger regulariser is penalised as if it were an error."""
    ax = tuple(range(1, X.ndim))
    P = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(C[None] * X[:, None], axes=tuple(a + 1 for a in ax)),
                                    axes=tuple(a + 1 for a in ax), norm="ortho"), axes=tuple(a + 1 for a in ax))
    Ph = P[:, :, ho].reshape(X.shape[0], -1); yh = ksp[:, ho].reshape(-1)[None]
    al = (Ph.conj() * yh).sum(1) / np.maximum((np.abs(Ph) ** 2).sum(1), 1e-30)
    return (np.abs(al[:, None] * Ph - yh) ** 2).sum(1)


def lambda_cv(ksp, C, m, grid, device="cpu", iters=60, holdout=0.1, folds=2, calib=24, seed=0):
    """k-space cross-validation (see module docstring): (best lam, prediction errors per lam)."""
    err = np.zeros(len(grid))
    for tr, ho in cv_masks(m, holdout, folds, calib, seed):
        X = grid_solutions(ksp * tr[None], C, tr, grid, device=device, iters=iters)    # (L, ...)
        err += heldout_error(X, C, ksp, ho)
    return float(grid[int(np.argmin(err))]), err


def choose_sense_lambda(ksp, C, mask=None, noise_var=None, method="cv", scale=1.0, device="cpu", grid=None,
                        iters=60, probes=2, seed=0, calib=24, holdout=0.1, folds=2, ceiling=None):
    """Automatic relative Tikhonov weight for api.sense_image (see module docstring). ``ksp`` (N, ...)
    acquired k-space (zeros elsewhere), ``C`` (N, ...) maps, ``mask`` acquired samples (default: non-zero),
    ``noise_var`` per-sample noise variance if known (1 after whitening). Returns (lam, info); lam is None
    when the problem needs no regularisation (everything inside the acquired extent is sampled: SENSE-1)."""
    ksp = np.asarray(ksp); C = np.asarray(C)
    m = (np.abs(ksp).sum(0) > 0) if mask is None else np.asarray(mask, bool)
    info = {"method": method, "scale": scale}
    if (m | ~_extent(m)).all():
        info["note"] = "fully sampled inside the acquired extent: closed-form SENSE-1, no regularisation"
        return None, info
    sigma2 = float(noise_var) if noise_var is not None else noise_variance(ksp, m)
    lam_map, mi = lambda_map(ksp, C, m, sigma2)
    info.update(mi, lam_map=lam_map)
    if method == "map":
        return lam_map * scale, info
    grid = np.asarray(grid if grid is not None else np.logspace(-5, 0, 16))   # relative weights
    if method in ("cv", "cv+map"):
        lam_cv, err = lambda_cv(ksp, C, m, grid, device=device, iters=iters, holdout=holdout, folds=folds,
                                calib=calib, seed=seed)
        info.update(grid=grid.tolist(), cv_error=err.tolist(), lam_cv=lam_cv)
        if method == "cv":
            return lam_cv * scale, info
        if mi.get("map_signal_floor") or lam_map > grid.max():
            # the Wiener value is unreliable when the signal estimate collapses (noise estimate >= ~ all the
            # energy; one of 20 real slices gave lam_map = 113, an empty image): cross-validation alone
            info["map_unreliable"] = True
            return lam_cv * scale, info
        info["floor"] = "map" if lam_map > lam_cv else "cv"
        lam = max(lam_cv, lam_map)
        if ceiling is not None and lam > ceiling * lam_map:              # optional: at most ceiling x Wiener
            lam, info["ceiling_applied"] = ceiling * lam_map, True
        return lam * scale, info
    y, A, Ah, cg, dev = _solver(ksp, C, m, device)
    den = float((np.abs(C) ** 2).sum(0).mean())
    lt = torch.tensor(grid * den, dtype=torch.float32, device=dev)
    L = len(grid)
    b = Ah(y[None]).expand(L, *y.shape[1:]).clone()
    x = cg(lt, b, iters)
    res = ((A(x) - y[None]).abs() ** 2).flatten(1).sum(1)                 # ||r_lam||^2, (L,)
    g = torch.Generator(device=dev).manual_seed(seed)
    tr = torch.zeros(L, device=dev)
    for _ in range(probes):                                               # Hutchinson, complex Rademacher
        z = (torch.randint(0, 2, y.shape[1:], generator=g, device=dev) * 2 - 1).to(torch.complex64)
        z = z * torch.exp(0.5j * np.pi * torch.randint(0, 2, y.shape[1:], generator=g, device=dev))
        AhAz = Ah(A(z[None]))[0]
        w = cg(lt, AhAz[None].expand(L, *AhAz.shape).clone(), iters)
        tr += (z.conj()[None] * w).real.flatten(1).sum(1) / probes
    n = float(m.sum()) * ksp.shape[0]
    res, tr = res.cpu().numpy(), tr.cpu().numpy()
    gcv = n * res / np.maximum(n - tr, 1.0) ** 2
    lam_gcv = float(grid[int(np.argmin(gcv))])
    target = sigma2 * (n - tr)
    f = res - target                                                      # increasing in lam
    lam_disc = float(grid[0] if f[0] >= 0 else grid[-1])
    for i in range(L - 1):
        if f[i] < 0 <= f[i + 1]:
            t = f[i] / (f[i] - f[i + 1])
            lam_disc = float(np.exp(np.log(grid[i]) + t * (np.log(grid[i + 1]) - np.log(grid[i]))))
            break
    info.update(grid=grid.tolist(), residual=res.tolist(), trace=tr.tolist(), gcv=gcv.tolist(), n=n,
                lam_gcv=lam_gcv, lam_discrepancy=lam_disc)
    lam = {"gcv": lam_gcv, "discrepancy": lam_disc}[method]
    return lam * scale, info
