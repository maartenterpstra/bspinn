"""One-call interface: k-space array in, coil sensitivity maps out.

    from bspinn.api import estimate_maps
    maps, info = estimate_maps(ksp, layout="2d")            # ksp (coils, x, y)
    maps, info = estimate_maps(ksp, layout="3d")            # ksp (coils, x, y, z)
    maps, info = estimate_maps(ksp, layout="dynamic")       # ksp (coils, x, y, time)

``maps`` has the same axis order and shape as ``ksp`` (coils first): (N,X,Y), (N,X,Y,Z) or
(N,X,Y,T). Unsampled k-space entries must be exactly zero (the sampling mask is taken from the
non-zero entries unless ``mask`` is given). Cartesian data only; the centre of k-space
(``calib`` points per axis) should be fully sampled or at least densely sampled.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from .fit import FitConfig

PRESETS = {      # iterations: (handedness/orientation probe, wire stage, PINN stage)
    "2d": {"fast": (100, 500, 800), "default": (300, 1500, 2500)},
    "3d": {"fast": (80, 300, 300), "default": (150, 600, 600)},
    "dynamic": {"fast": (100, 500, 800), "default": (300, 1500, 2500)},
}


def estimate_maps(ksp, layout="2d", *, coil_axis=0, mask=None, spacing=None, plane="axial", axes="auto",
                  calib=None, noise=None, physics="laplace", b0_tesla=3.0, model="pinn", rank=2,
                  iters_dynamic=None, quality="default", normalise="rss", phase_ref="inati", device="cpu",
                  threads=0, seed=0, verbose=True, **cfg_overrides):
    """Estimate coil sensitivity maps from Cartesian multi-coil k-space.

    Parameters
    ----------
    ksp : complex array, (coils, x, y) | (coils, x, y, z) | (coils, x, y, time); ``coil_axis`` moves
        the coil axis if it is not first. Any number of coils.
    layout : '2d' | '3d' | 'dynamic'.
    mask : bool array of the spatial(-temporal) shape (without the coil axis). Default: non-zero samples.
    spacing : pixel/voxel size in mm per spatial axis ((dx, dy) or (dx, dy, dz)). Default 1 mm
        isotropic. Only the aspect ratio matters for 'laplace'; for 'helmholtz' the absolute size sets
        the wavelength, so pass the real voxel size.
    plane : 2-D / dynamic: 'axial' | 'coronal' | 'sagittal'. Tells the model where B0 points relative
        to the image (axial: B0 through the slice; coronal/sagittal: B0 along the first image axis).
        If unsure, run twice and keep the lower 'data_loss' in ``info``.
    axes : 3-D: physical axis of each array axis, e.g. 'xyz' (B0 = z along the last axis), or 'auto'
        (default) to pick the B0 axis from the data (adds ~3 short wire fits).
    calib : calibration (ACS) size per axis, int or tuple. Default: 24 (2-D/dynamic), 12 (3-D) clipped
        to the matrix. Use the size of your fully sampled centre; larger = slower, more detail.
    noise : optional noise-only samples (n_samples, coils) for pre-whitening. Recommended for real data
        with correlated channels. Maps are returned for the original (un-whitened) channels.
    physics : 'laplace' (quasi-static, default) | 'helmholtz' (3-D only; wave-aware, for >= 3 T body
        imaging). ``b0_tesla`` sets the Larmor frequency for 'helmholtz'.
    model : 'pinn' (wires + harmonic correction, default) | 'wires' (wires only, faster, cruder).
    rank : dynamic only. Number of temporal motion modes of the coil array; 0 = one static map set
        shared by all frames (frames are then only used as extra data).
    quality : 'fast' | 'default'. Iteration presets (see PRESETS); override with iters_wires=...,
        iters_pinn=..., iters_auto=... . Dynamic data use 4 random frames per step (slice_batch=...).
    normalise : 'rss' (sum_j |C_j|^2 = 1 everywhere, the ESPIRiT/SENSE convention) | 'none' (raw model
        scale; the maps then still extrapolate smoothly to the whole FOV).
    phase_ref : common-phase convention: 'inati' (virtual body coil, default) | 'model' | 'coil'.
    cfg_overrides : any other FitConfig field (e.g. lambda_phys=0.1, field_mode='complex').

    Returns
    -------
    maps : complex64 array with the shape and axis order of ``ksp``.
    info : dict with 'support' (object mask used for the phase reference), 'data_loss',
        'handedness', 'orientation' (3-D), 'k2'/'tissue' (helmholtz), 'temporal_modes' (dynamic),
        'seconds', 'config'.
    """
    layout = layout.lower()
    if layout not in PRESETS:
        raise ValueError("layout must be '2d', '3d' or 'dynamic'")
    ksp = np.moveaxis(np.asarray(ksp), coil_axis, 0)
    want = {"2d": 3, "3d": 4, "dynamic": 4}[layout]
    if ksp.ndim != want:
        raise ValueError(f"layout '{layout}' expects a {want}-D array (coils first), got shape {ksp.shape}")
    if not np.iscomplexobj(ksp):
        raise ValueError("k-space must be complex")
    if physics == "helmholtz" and layout != "3d":
        raise ValueError("physics='helmholtz' needs 3-D data (the out-of-plane curvature is unobserved on one slice)")
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != ksp.shape[1:]:
            raise ValueError(f"mask must have shape {ksp.shape[1:]}, got {mask.shape}")
    nsp = 3 if layout == "3d" else 2
    spacing = tuple(float(s) for s in (spacing or (1.0,) * nsp))
    if len(spacing) != nsp:
        raise ValueError(f"spacing needs {nsp} values")
    if noise is not None:
        noise = np.asarray(noise).reshape(-1, ksp.shape[0])

    ia, iw, ip = PRESETS[layout][quality]
    cfg = FitConfig(model=model, physics=physics, b0_tesla=b0_tesla, phase_ref="model", device=device,
                    threads=threads, seed=seed, iters_auto=ia, iters_wires=iw, iters_pinn=ip,
                    log_every=max(100, ip // 5))
    if physics == "helmholtz":
        from .models import tissue_k2  # noqa: F401  (generic tissue prior eps_r 50, sigma 0.6 by default)
    if layout == "3d":
        cfg.calib = _calib(calib, 12, 3)
    else:
        cfg.calib = int(max(_calib(calib, 24, 1)))
    if layout == "dynamic":
        cfg.slice_batch = min(4, ksp.shape[-1])        # frames per optimisation step (cost ~ frames)
    for k, v in cfg_overrides.items():
        if not hasattr(cfg, k):
            raise TypeError(f"unknown option {k!r} (not a FitConfig field)")
        setattr(cfg, k, v)
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    if threads:
        torch.set_num_threads(threads)

    if layout == "2d":
        C, info = _run_2d(ksp, mask, spacing, plane, noise, cfg, log)
    elif layout == "3d":
        C, info = _run_3d(ksp, mask, spacing, axes, noise, cfg, log)
    else:
        C, info = _run_dynamic(ksp, mask, spacing, plane, noise, cfg, rank, iters_dynamic, log)

    # common phase + normalisation; C is (N, *spatial) or (T, N, H, W) for dynamic
    sup = _support(ksp, mask, layout, cfg.calib)
    C = _finish(C, sup, layout, phase_ref, normalise)
    if layout == "dynamic":
        C = np.moveaxis(C, 0, -1)                       # (T,N,X,Y) -> (N,X,Y,T)
    info.update(support=sup, seconds=time.time() - t0, config=_cfg_dict(cfg))
    log(f"maps {C.shape} in {info['seconds']:.0f}s")
    return C.astype(np.complex64), info


# ---------------------------------------------------------------------------------- per layout
def _run_2d(ksp, mask, spacing, plane, noise, cfg, log):
    from .data import MultiCoilData
    from .fit import fit
    from .geometry import SliceGeometry
    N, X, Y = ksp.shape
    geom = SliceGeometry((X, Y), spacing, np.zeros(1), plane)
    data = MultiCoilData(ksp[None], geom, mask=None if mask is None else mask[None, None], noise=noise)
    model, prob, hist, extra = fit(data, cfg, log=log)
    pts = geom.grid().reshape(-1, 3)
    with torch.enable_grad():
        C = prob.mixC(model.sensitivity_batched(pts), -2).detach().numpy().reshape(N, X, Y)
    return C, {"handedness": extra["handedness"], "data_loss": _last(hist), "model": model, "problem": prob}


def _run_3d(ksp, mask, spacing, axes, noise, cfg, log):
    from .volume import VolumeData, VolumeGeometry, fit3d, volume_maps
    orient = "auto" if axes == "auto" else "given"
    geom = VolumeGeometry(ksp.shape[1:], spacing, "zyx" if axes == "auto" else axes)
    data = VolumeData(ksp, geom, mask=mask, noise=noise)
    model, prob, hist, extra = fit3d(data, cfg, log=log, orientation=orient)
    C = volume_maps(model, prob).numpy()
    info = {"handedness": extra["handedness"], "orientation": prob.data.geom.axes,
            "orientation_note": "physical axis (x,y,z; B0 = z) of array axes 1..3", "data_loss": _last(hist),
            "model": model, "problem": prob}
    if hasattr(model, "k2_re"):
        from .models import k2_to_tissue
        k2 = complex(float(model.k2_re), float(model.k2_im))
        info["k2"] = k2
        try:
            info["tissue"] = k2_to_tissue(k2, cfg.b0_tesla, prob.data.geom.half_fov / 1000.0)
        except Exception:                                                  # pragma: no cover
            pass
    return C, info


def _run_dynamic(ksp, mask, spacing, plane, noise, cfg, rank, iters_dynamic, log):
    from .data import MultiCoilData
    from .geometry import SliceGeometry
    N, X, Y, T = ksp.shape
    kt = np.ascontiguousarray(np.moveaxis(ksp, -1, 0))                      # (T,N,X,Y)
    mt = None if mask is None else np.moveaxis(mask, -1, 0)[:, None]        # (T,1,X,Y)
    geom = SliceGeometry((X, Y), spacing, np.zeros(T), plane)               # frames = co-located slices
    data = MultiCoilData(kt, geom, mask=mt, noise=noise, meta={"dynamic": True})
    if mt is None:
        data.mask = (np.abs(kt).sum(1, keepdims=True) > 0)
    pts = geom.grid((X, Y))[0].reshape(-1, 3)
    if rank == 0:
        from .fit import fit
        model, prob, hist, extra = fit(data, cfg, log=log)
        with torch.enable_grad():
            C = prob.mixC(model.sensitivity_batched(pts), -2).detach().numpy().reshape(N, X, Y)
        return np.broadcast_to(C[None], (T, N, X, Y)).copy(), {
            "handedness": extra["handedness"], "data_loss": _last(hist), "rank": 0, "model": model, "problem": prob}
    from .dynamic import DynConfig, fit_dynamic
    dcfg = DynConfig(rank=rank, iters_dyn=iters_dynamic or max(300, cfg.iters_pinn // 2))
    dyn, prob, hist, extra = fit_dynamic(data, cfg, dcfg, log=log)
    out = []
    with torch.enable_grad():
        for i in range(0, T, 8):
            fr = list(range(i, min(T, i + 8)))
            c = dyn.sensitivity_frames(pts, fr, create_graph=False).detach()  # (B,N_model,P)
            out.append(prob.mixC(c, -2).numpy())
    C = np.concatenate(out).reshape(T, N, X, Y)
    return C, {"handedness": extra["handedness"], "rank": rank, "temporal_modes": dyn.alpha.detach().numpy(),
               "data_loss": float(hist["dynamic"][-1]["data"]) if hist.get("dynamic") else None,
               "model": dyn, "problem": prob}


# ---------------------------------------------------------------------------------- helpers
def _calib(calib, default, n):
    if calib is None:
        calib = default
    c = tuple(int(v) for v in (calib if isinstance(calib, (tuple, list)) else (calib,) * n))
    return c if n > 1 else c[:1]


def _last(hist):
    for key in ("pinn", "wires"):
        h = hist.get(key) if isinstance(hist, dict) else None
        if h:
            rec = h[-1]
            return float(rec.get("data", rec.get("loss", np.nan)))
    return None


def _cfg_dict(cfg):
    import dataclasses
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


def _support(ksp, mask, layout, calib):
    """Object mask from a Hann-windowed low-resolution root-sum-of-squares image (used only to choose
    the phase reference)."""
    from scipy.ndimage import binary_dilation, binary_fill_holes, uniform_filter
    k = ksp if layout != "dynamic" else ksp.sum(-1)                         # dynamic: time-averaged
    if mask is not None:
        m = mask if layout != "dynamic" else mask.any(-1)
        k = k * m
    sp = k.shape[1:]
    cal = calib if isinstance(calib, (tuple, list)) else (calib,) * len(sp)
    win = np.ones(sp)
    for a, (n, c) in enumerate(zip(sp, cal)):
        c = min(int(c), n)
        w = np.zeros(n)
        w[n // 2 - c // 2: n // 2 - c // 2 + c] = np.hanning(c + 2)[1:-1]
        win = win * w.reshape([-1 if i == a else 1 for i in range(len(sp))])
    ax = tuple(range(1, k.ndim))
    img = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k * win, axes=ax), axes=ax), axes=ax)
    rss = np.sqrt((np.abs(img) ** 2).sum(0))
    rss = uniform_filter(rss, 5)
    s = rss > 0.1 * np.quantile(rss, 0.99)
    s = binary_fill_holes(s)
    return binary_dilation(s, iterations=2)


def _finish(C, sup, layout, phase_ref, normalise):
    from .recon import phase_reference, virtual_body_coil
    Ct = torch.from_numpy(np.ascontiguousarray(C)).to(torch.complex64)
    if layout == "2d":
        Cs, S = Ct[None], torch.from_numpy(sup)[None]                    # (1,N,X,Y)
    elif layout == "3d":
        Cs, S = Ct.permute(1, 0, 2, 3), torch.from_numpy(sup)             # (X,N,Y,Z): first axis as slices
    else:
        Cs, S = Ct, torch.from_numpy(sup)[None].expand(Ct.shape[0], -1, -1)  # (T,N,X,Y)
    if phase_ref not in ("model", "none"):
        d = None
        if phase_ref in ("inati", "inati-eig"):
            d = virtual_body_coil(Cs, S.float(), robust=(phase_ref == "inati"))
        Cs, _, _, _ = phase_reference(Cs, phase_ref, S, None, d)
    if normalise == "rss":
        Cs = Cs / (Cs.abs() ** 2).sum(1, keepdim=True).sqrt().clamp_min(1e-12)
    elif normalise != "none":
        raise ValueError("normalise must be 'rss' or 'none'")
    if layout == "2d":
        return Cs[0].numpy()
    if layout == "3d":
        return Cs.permute(1, 0, 2, 3).numpy()
    return Cs.numpy()


def coil_combine(ksp, maps, layout="2d", coil_axis=0):
    """Fully sampled data -> SENSE-optimal combined image sum_j C_j^* x_j / sum_j |C_j|^2 (same spatial
    shape as one coil). For undersampled data use bspinn.evaluation.cg_sense or BART pics with the maps."""
    ksp = np.moveaxis(np.asarray(ksp), coil_axis, 0)
    maps = np.moveaxis(np.asarray(maps), coil_axis, 0)
    ax = (1, 2) if layout in ("2d", "dynamic") else (1, 2, 3)
    x = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ksp, axes=ax), axes=ax, norm="ortho"), axes=ax)
    return (maps.conj() * x).sum(0) / np.maximum((np.abs(maps) ** 2).sum(0), 1e-12)
