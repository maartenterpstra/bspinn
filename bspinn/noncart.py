"""Non-Cartesian 2-D coil maps and images with MRI-NUFFT (https://mind-inria.github.io/mri-nufft/).

    maps, image, info = estimate_maps_noncart(kspace, traj, (256, 256), fov_mm=220, plane="axial",
                                              return_image=True, device="cuda")
    image = sense_image_noncart(kspace, maps, traj, (256, 256), prior="tv")

* ``kspace`` (coils, samples) or (coils, shots, samples_per_shot); ``traj`` (samples, 2) or (shots, samples, 2)
  in MRI-NUFFT units (each axis in [-0.5, 0.5), row then column of the image), or ``traj_units='cycles'``
  (cycles/FOV, BART / bspinn convention) / 'radians' ([-pi, pi)).
* Maps: the Biot-Savart PINN is fitted to the samples inside the calibration box with bspinn's exact NUDFT
  (small grid, no gridding error) and evaluated at full resolution; then the common phase and RSS
  normalisation of ``estimate_maps``. The B0 plane must be given (the automatic plane search and the
  fold-over / IRGNM steps of the Cartesian path assume a Cartesian grid).
* Images: CG-SENSE (density-weighted least squares, Tikhonov) or isotropic TV by ADMM, with the MRI-NUFFT
  operator (backend 'finufft' on the CPU, 'cufinufft' / 'gpunufft' on a GPU, any MRI-NUFFT backend name).
  TV weight 'auto': cross-validation on withheld shots (spokes / interleaves), as for Cartesian data.
Install: ``pip install mri-nufft finufft`` (GPU: ``cufinufft`` or ``gpunufft``).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import torch


def to_mrinufft(traj, shape, units="mrinufft"):
    """Trajectory -> (M, 2) float32 in MRI-NUFFT units ([-0.5, 0.5) per axis)."""
    t = np.asarray(traj, np.float64).reshape(-1, 2)
    if units == "cycles":
        t = t / np.asarray(shape, np.float64)[None]
    elif units == "radians":
        t = t / (2 * np.pi)
    elif units != "mrinufft":
        raise ValueError("traj_units must be 'mrinufft', 'cycles' or 'radians'")
    if np.abs(t).max() > 0.5 + 1e-6:
        raise ValueError(f"trajectory exceeds the grid (max |k| {np.abs(t).max():.3f} > 0.5 in MRI-NUFFT units): "
                         "wrong traj_units or shape?")
    return np.ascontiguousarray(np.clip(t, -0.5, 0.5 - 1e-6).astype(np.float32))


def _backend(backend, device):
    if backend != "auto":
        return backend
    if str(device).startswith("cuda"):
        from mrinufft import list_backends
        av = list_backends(available_only=True)
        for b in ("cufinufft", "gpunufft"):
            if b in av:
                return b
    return "finufft"


def nufft_operator(traj, shape, n_coils, smaps=None, density=True, backend="finufft"):
    """MRI-NUFFT operator: op(x (H,W)) -> (N, M) with smaps, adj_op the density-weighted adjoint."""
    from mrinufft import get_operator
    sm = None if smaps is None else np.ascontiguousarray(np.asarray(smaps, np.complex64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get_operator(backend)(np.ascontiguousarray(traj, np.float32), shape=tuple(int(s) for s in shape),
                                     n_coils=int(n_coils), smaps=sm, density=density)


def _shots(kspace, traj):
    """(N, M) k-space, (M, 2) trajectory and the shot length (samples per spoke/interleave, or None)."""
    k = np.asarray(kspace)
    ns = k.shape[-1] if k.ndim == 3 else None
    t = np.asarray(traj)
    if ns is None and t.ndim == 3:
        ns = t.shape[1]
    return np.ascontiguousarray(k.reshape(k.shape[0], -1).astype(np.complex64)), ns


def _support_noncart(kspace, traj, shape, calib, backend, threshold=0.1, dilate=2):
    """Low-resolution RSS support: density-compensated adjoint of the central samples, smoothed, thresholded."""
    from scipy import ndimage
    sel = (np.abs(traj[:, 0]) * shape[0] < calib / 2) & (np.abs(traj[:, 1]) * shape[1] < calib / 2)
    op = nufft_operator(traj[sel], shape, kspace.shape[0], density=True, backend=backend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = np.asarray(op.adj_op(np.ascontiguousarray(kspace[:, sel])))
    rss = ndimage.uniform_filter(np.sqrt((np.abs(img) ** 2).sum(0)), 5)
    sup = rss > threshold * np.percentile(rss, 99)
    sup = ndimage.binary_fill_holes(sup)
    return ndimage.binary_dilation(sup, iterations=dilate) if dilate else sup


def estimate_maps_noncart(kspace, traj, shape, *, traj_units="mrinufft", fov_mm=None, spacing=None, plane="axial",
                          noise=None, calib=24, quality="default", device="cpu", seed=0, verbose=True,
                          normalise="rss", phase_ref="inati", return_image=False, image_prior="auto",
                          image_prior_weight="auto", image_options=None, backend="auto", **cfg_overrides):
    """Coil maps (N, H, W) from non-Cartesian 2-D k-space (see the module docstring); with ``return_image`` also
    the image (sense_image_noncart, ``image_prior`` 'auto' = TV with a cross-validated weight, None = Tikhonov).
    ``spacing`` (mm per pixel) or ``fov_mm`` (square FOV) set the physical scale of the Biot-Savart model
    (default 1 mm pixels); ``cfg_overrides``: FitConfig fields, as in estimate_maps."""
    from .api import PRESETS, _finish
    from .data import MultiCoilData
    from .fit import FitConfig, fit
    from .geometry import SliceGeometry
    if plane == "auto":
        raise ValueError("estimate_maps_noncart needs the B0 plane ('axial', 'coronal', 'coronal_t', 'sagittal', ...)")
    shape = tuple(int(s) for s in shape)
    k, ns = _shots(kspace, traj)
    N = k.shape[0]
    t = to_mrinufft(traj, shape, traj_units)
    if t.shape[0] != k.shape[1]:
        raise ValueError(f"{k.shape[1]} k-space samples per coil but {t.shape[0]} trajectory points")
    if spacing is None:
        spacing = (fov_mm / shape[0], fov_mm / shape[1]) if fov_mm else (1.0, 1.0)
    log = print if verbose else (lambda *a, **kw: None)
    t0 = time.time()
    ia, iw, ip = PRESETS["2d"][quality]
    cfg = FitConfig(model="pinn", phase_ref="model", device=device, seed=seed, iters_auto=ia, iters_wires=iw,
                    iters_pinn=ip, log_every=max(100, ip // 5))
    cfg.calib = int(calib)
    cfg.learn_mixing, cfg.resid_order, cfg.resid_order_final, cfg.optimizer = True, 3, 8, "adam+lbfgs"
    for kk, v in cfg_overrides.items():
        if not hasattr(cfg, kk):
            raise TypeError(f"unknown option {kk!r} (not a FitConfig field)")
        setattr(cfg, kk, v)
    geom = SliceGeometry(shape, tuple(float(s) for s in spacing), np.zeros(1), plane)
    tc = (t.astype(np.float64) * np.asarray(shape)[None]).astype(np.float32)          # cycles/FOV for bspinn
    data = MultiCoilData(k[None], geom, noise=None if noise is None else np.asarray(noise).reshape(-1, N), traj=tc)
    log(f"non-Cartesian: {N} coils, {t.shape[0]} samples, grid {shape}, "
        f"{int(((np.abs(tc) < cfg.calib / 2).all(1)).sum())} samples in the {cfg.calib}x{cfg.calib} calibration box")
    model, prob, hist, extra = fit(data, cfg, log=log)
    pts = geom.grid().reshape(-1, 3)
    model.to(device)
    with torch.no_grad():
        C = prob.mixC(model.sensitivity_batched(pts), -2).detach().cpu().numpy().reshape(N, *shape)
    model.cpu()
    M = prob.data.meta.get("mix")
    if M is not None:
        C = np.einsum("ij,j...->i...", np.linalg.inv(np.asarray(M)), C)
    be = _backend(backend, device)
    sup = _support_noncart(k, t, shape, cfg.calib, be)
    C = _finish(C, sup, "2d", phase_ref, normalise, device).astype(np.complex64)
    info = {"handedness": extra.get("handedness"), "support": sup, "backend": be, "seconds": time.time() - t0,
            "model": model, "problem": prob, "traj_mrinufft": t}
    log(f"maps {C.shape} in {info['seconds']:.0f}s")
    if not return_image:
        return C, info
    rep = {}
    img = sense_image_noncart(k, C, t, shape, noise=noise, prior=image_prior, prior_weight=image_prior_weight,
                              backend=be, device=device, report=rep, shot_length=ns, **(image_options or {}))
    info["image"] = rep
    info["seconds"] = time.time() - t0
    return C, img, info


# ------------------------------------------------------------------------------------------- images
def _grad(x):
    return np.stack([np.roll(x, -1, 0) - x, np.roll(x, -1, 1) - x])


def _grad_adj(z):
    return (np.roll(z[0], 1, 0) - z[0]) + (np.roll(z[1], 1, 1) - z[1])


def _cg(apply, b, x, iters, tol=1e-6):
    r = b - apply(x); p = r.copy(); rs = np.vdot(r, r).real; b0 = max(np.vdot(b, b).real, 1e-30)
    for _ in range(iters):
        if rs < tol ** 2 * b0:
            break
        Ap = apply(p)
        a = rs / max(np.vdot(p, Ap).real, 1e-30)
        x = x + a * p; r = r - a * Ap
        rn = np.vdot(r, r).real
        p = r + (rn / rs) * p; rs = rn
    return x


class _Normal:
    """Density-weighted SENSE normal operator A^H D A, scaled so that its largest eigenvalue is ~1."""

    def __init__(self, k, C, traj, shape, density, backend):
        self.op = nufft_operator(traj, shape, C.shape[0], smaps=C, density=density, backend=backend)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.b = np.asarray(self.op.adj_op(np.ascontiguousarray(k)))
            v = np.random.default_rng(0).standard_normal(shape).astype(np.complex64)
            for _ in range(8):                                   # power iteration: lambda_max
                v = self.raw(v); lam = float(np.linalg.norm(v)); v = v / max(lam, 1e-30)
        self.lam = max(lam, 1e-30)
        self.b = self.b / self.lam

    def raw(self, x):
        return np.asarray(self.op.adj_op(np.asarray(self.op.op(x.astype(np.complex64)))))

    def __call__(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.raw(x) / self.lam

    def forward(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(self.op.op(x.astype(np.complex64)))


def _tv_admm(Nop, betas, iters=40, cg_iters=4, rho=0.5, x0=None):
    out = []
    for beta in betas:
        x = Nop.b.copy() if x0 is None else x0.copy()
        z = _grad(x); u = np.zeros_like(z)
        A = lambda v: Nop(v) + rho * _grad_adj(_grad(v))      # noqa: E731
        for _ in range(iters):
            x = _cg(A, Nop.b + rho * _grad_adj(z - u), x, cg_iters)
            g = _grad(x) + u
            mag = np.sqrt((np.abs(g) ** 2).sum(0, keepdims=True))
            z = g * np.maximum(1 - (beta / rho) / np.maximum(mag, 1e-12), 0)
            u = g - z
        out.append(x)
    return out


def sense_image_noncart(kspace, maps, traj, shape, *, traj_units="mrinufft", noise=None, prior="auto",
                        prior_weight="auto", lam=1e-5, iters=100, density=True, backend="auto", device="cpu",
                        report=None, shot_length=None, grid=None, holdout=0.1, folds=1, admm_iters=40, seed=0):
    """Image (H, W) from non-Cartesian k-space and maps. prior None: CG-SENSE (density-weighted least squares +
    ``lam`` x lambda_max Tikhonov, ``iters`` CG steps; radial CG converges slowly: 100 steps reach the error
    floor of the unsampled k-space corners on a 64^2 phantom, 30 steps had 3x that error); 'tv' / 'auto': isotropic TV by ADMM, weight relative to
    the normalised image (``prior_weight`` number, or 'auto' = k-space cross-validation on withheld shots of
    ``shot_length`` samples, weight by bspinn.prior.select_weight). ``noise`` (Ns, N) noise samples whiten data
    and maps alike."""
    k, ns = _shots(kspace, traj)
    ns = shot_length or ns
    C = np.asarray(maps, np.complex64)
    t = to_mrinufft(traj, shape, traj_units)
    be = _backend(backend, device)
    if noise is not None:
        n = np.asarray(noise).reshape(-1, k.shape[0])
        L = np.linalg.inv(np.linalg.cholesky((n.conj().T @ n) / n.shape[0]))
        k = (L @ k).astype(np.complex64)
        C = np.einsum("ij,j...->i...", L, C).astype(np.complex64)
    rep = {} if report is None else report
    Nop = _Normal(k, C, t, shape, density, be)
    x0 = _cg(lambda v: Nop(v) + lam * v, Nop.b, np.zeros(shape, np.complex64), iters)
    if prior is None:
        rep.update(method="cg-sense", lam=lam, backend=be)
        return x0.astype(np.complex64)
    if prior not in ("tv", "auto"):
        raise ValueError("non-Cartesian prior: None (Tikhonov) or 'tv' / 'auto'")
    s = float(np.percentile(np.abs(x0), 99)) or 1.0
    grid = np.asarray(grid if grid is not None else np.logspace(-5, -1, 9))
    if isinstance(prior_weight, str):
        # withheld shots (whole spokes / interleaves: neighbouring samples on a shot are not independent)
        M = k.shape[1]
        L_ = ns if ns and M % ns == 0 else 64
        groups = np.arange(M) // L_
        rng = np.random.default_rng(seed)
        err = np.zeros(len(grid))
        ng = groups.max() + 1
        for f in range(folds):
            ho_g = rng.choice(ng, max(1, int(round(holdout * ng))), replace=False)
            ho = np.isin(groups, ho_g)
            Ntr = _Normal(k[:, ~ho], C, t[~ho], shape, density, be)
            Ntr_s = float(np.percentile(np.abs(_cg(lambda v: Ntr(v) + lam * v, Ntr.b, np.zeros(shape, np.complex64),
                                                   iters)), 99)) or 1.0
            Ntr.b = Ntr.b / Ntr_s
            xs = _tv_admm(Ntr, grid, admm_iters)
            Aho = nufft_operator(t[ho], shape, C.shape[0], smaps=C, density=False, backend=be)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i, xx in enumerate(xs):
                    pred = np.asarray(Aho.op((xx * Ntr_s).astype(np.complex64)))
                    # the operator's scale is the same with and without density weighting: compare directly
                    err[i] += float(np.linalg.norm(pred - k[:, ho]) ** 2 / max(np.linalg.norm(k[:, ho]) ** 2, 1e-30))
        # plain minimum of the held-out error (edges allowed; the Cartesian interior rule of select_weight was
        # calibrated on fastMRI slices, where the lowest-weight edge was spurious), refined by a parabola in log w
        i = int(np.argmin(err)); w, rule = float(grid[i]), "edge" if i in (0, len(grid) - 1) else "interior"
        if rule == "interior":
            lx = np.log(grid[i - 1:i + 2]); c2, c1, _ = np.polyfit(lx, err[i - 1:i + 2], 2)
            if c2 > 0:
                w = float(np.exp(np.clip(-c1 / (2 * c2), lx[0], lx[2])))
        rep.update(method="cv", grid=grid.tolist(), cv_error=err.tolist(), weight=w, rule=rule, shot_length=int(L_))
    else:
        w = float(prior_weight)
        rep.update(method="manual", weight=w)
    Nop.b = Nop.b / s
    x = _tv_admm(Nop, [w], admm_iters, x0=x0 / s)[0] * s
    rep.update(prior="tv", backend=be)
    return x.astype(np.complex64)
