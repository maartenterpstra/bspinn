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
import warnings

import numpy as np
import torch

from . import accel
from .fit import FitConfig

PRESETS = {      # iterations: (handedness/orientation probe, wire stage, PINN stage)
    "2d": {"fast": (100, 500, 800), "default": (300, 1500, 2500)},
    "3d": {"fast": (80, 300, 300), "default": (150, 600, 600)},
    "dynamic": {"fast": (100, 500, 800), "default": (300, 1500, 2500)},
}


def estimate_maps(ksp, layout="2d", *, coil_axis=0, mask=None, spacing=None, plane="auto", axes="auto",
                  calib=None, noise=None, physics="laplace", b0_tesla=3.0, model="pinn", rank=2,
                  iters_dynamic=None, quality="default", normalise="rss", phase_ref="inati", device="cpu",
                  threads=0, seed=0, verbose=True, return_image=False, sense_lambda="auto", sense_lambda_scale=1.0, sense_lambda_options=None, sense_iters=60, mask_maps=False, support_threshold=0.1,
                  support_dilate=2, undersample=None, solve_image=False, fold="auto", readout_axis="auto", plane_select="heldout",
                  image_prior=None, image_prior_weight="auto", image_prior_options=None, dynamic_static="average",
                  dynamic_options=None, dynamic_base="2d", dynamic_validate="auto",
                  **cfg_overrides):
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
    plane : 2-D / dynamic: where B0 points relative to the image. 'auto' (default): short wire fits
        with B0 through the slice ('axial'), along the image rows ('coronal'; 'sagittal' is the same up
        to a rotation about B0) and along the image columns ('coronal_t'); the lowest calibration loss
        wins (``info['plane_scores']``). Or give it: 'axial' | 'coronal' | 'sagittal' | 'coronal_t'.
        Through-plane vs in-plane B0 is well identified; rows vs columns for in-plane B0 often is not
        (a warning is issued when the two are within 15 %): pass ``plane`` if you know it.
        A wrong plane fits much worse and can give false zeros inside the object.
    axes : 3-D: physical axis of each array axis, e.g. 'xyz' (B0 = z along the last axis), or 'auto'
        (default) to pick the B0 axis from the data (adds ~3 short wire fits).
    calib : calibration (ACS) size per axis, int or tuple. Default: 24 (2-D/dynamic), 12 (3-D) clipped
        to the matrix. Use the size of your fully sampled centre; larger = slower, more detail.
    noise : optional noise-only samples (n_samples, coils) for pre-whitening. Recommended for real data
        with correlated channels. Maps are returned for the original (un-whitened) channels.
    physics : 'laplace' (quasi-static, default) | 'helmholtz' (3-D only; wave-aware, for >= 3 T body
        imaging). ``b0_tesla`` sets the Larmor frequency for 'helmholtz'.
    model : 'pinn' (wires + harmonic correction, default) | 'wires' (wires only, faster, cruder).
    dynamic_static : dynamic only. 'average' (default, v50): the static part of the coil model is fitted to the
        time-averaged k-space and the temporal modes (rank > 0) to the frames; 'frames': the static part is
        fitted jointly to all frames (ill-posed when the frames are undersampled without a calibration region).
        With 'average' the time-averaged model is the FIXED base and only low-rank per-frame corrections are
        learned (v55), started at zero and shrunk towards the base.
    dynamic_options : dynamic only. dict of bspinn.dynamic.DynConfig fields, e.g. {'lambda_corr': 1e-2,
        'freeze_static': True, 'alpha_init': 'auto' | 'dct' | 'data', 'parts': 'both' | 'wires' | 'fields'}.
    dynamic_base : dynamic with dynamic_static='average' only. '2d' (default, v57): the time-averaged k-space is
        fitted by the full 2-D path (held-out B0-plane decision, handedness from the plane search, fold-over
        correction, joint stage with ``solve_image``), so rank 0 returns exactly the 2-D maps of the time average
        for every frame; rank > 0 learns per-frame corrections on top of that FIXED model, and the returned
        frame maps are the 2-D maps plus the corrections (same phase convention and normalisation). 'legacy':
        the v50-v56 path (B0 plane from the probe alone, no fold correction).
    dynamic_validate : dynamic_base='2d', rank > 0. 'auto' (default) | True | False. Withholds
        ``dynamic_options['holdout']`` (default 10 %, at least one) of the acquired lines of every frame from the
        correction fit, then scores the series with the base maps and with the corrections scaled by
        ``shrink_grid`` (default 0.25, 0.5, 1) on them (spatio-temporal TV image from the other lines, fixed
        weight); the corrections are kept, scaled, only if they beat the base by ``min_gain`` (default 2 %),
        otherwise the base maps are returned. ``info['dynamic_validation']``. 'auto' = True.
    rank : dynamic only. Number of temporal motion modes of the coil array; 0 = one static map set
        shared by all frames (frames are then only used as extra data).
    quality : 'fast' | 'default'. Iteration presets (see PRESETS); override with iters_wires=...,
        iters_pinn=..., iters_auto=... . Dynamic data use 4 random frames per step (fewer above 20 coils; slice_batch=...).
    normalise : 'rss' (sum_j |C_j|^2 = 1 everywhere, the ESPIRiT/SENSE convention) | 'none' (raw model
        scale; the maps then still extrapolate smoothly to the whole FOV).
    phase_ref : common-phase convention: 'inati' (virtual body coil, default) | 'model' | 'coil'.
    return_image : also return the full-resolution image reconstructed with the estimated maps
        (SENSE: closed form when everything inside the acquired k-space extent is sampled, CG-SENSE with
        Tikhonov ``sense_lambda`` otherwise; zero-padded or partial-Fourier k-space edges are not
        extrapolated; per frame for dynamic data). Returns (maps, image, info). ``sense_lambda`` is the
        main noise / aliasing trade-off: a number (relative to the mean of sum |C|^2; 1e-2: close to least
        squares, the most residual noise; 3e-2 to 1e-1: smoother, NLINV-like), or chosen from the data:
        'auto' (default: k-space cross-validation on withheld acquired phase-encode lines, which tracks the
        image error, floored by the noise-based Wiener value; = 'auto-cv+map'), 'auto-cv', 'auto-map'
        (Wiener: noise variance / mean signal power, closed form), 'auto-gcv', 'auto-discrepancy'
        (data-space criteria; less reliable); see
        bspinn/senselam.py. ``sense_lambda_scale`` multiplies the automatic value. The chosen value and
        every rule's value are in ``info['sense_lambda']``. ``sense_lambda_options``: dict passed to
        senselam.choose_sense_lambda (e.g. folds=4 for a lower-variance cross-validation, ceiling=10 to cap
        the weight at 10x the Wiener value). ``sense_iters`` caps the CG iterations (fewer =
        stronger implicit regularisation, as in NLINV). With ``solve_image`` the weight (numeric, or chosen
        automatically at the start of the joint stage from the calibration maps) is the image weight of the
        joint fit itself, and the returned image is the joint solution (dynamic data with an automatic
        weight: the image is recomputed with it after the fit). Joint-fit options (FitConfig fields):
        ``joint_start='wires'`` skips the calibration-only PINN stage, so maps and image are fitted
        together from the first PINN iteration on all acquired k-space (the joint stage then runs
        iters_pinn + iters_joint iterations); ``joint_image='gradient'`` makes the image a parameter of the
        same optimiser as the maps (simultaneous steps, Adam or L-BFGS) instead of an exact CG-SENSE solve
        after every map step (default 'cg').
    image_prior, image_prior_weight, image_prior_options : the returned image with a sparsity prior instead
        of Tikhonov (2-D): 'tv' | 'wavelet' | 'tv+wavelet' (weight 'auto' = k-space cross-validation, or a
        relative number), or 'net:<weights.pt>' (a network from scripts/train_recon.py / zs_ssl.py); see
        sense_image. With ``solve_image`` and a built-in prior the joint fit uses the same prior for its image
        updates (warm-started ADMM after every map step, weight by cross-validation with the calibration maps;
        ``joint_prior=None`` keeps the Tikhonov image there); the joint image is kept in
        ``info['image_prior_joint']`` (or ``info['image_tikhonov_joint']``) and the returned image is recomputed
        with the prior for the final maps; ``info['joint_prior']`` has the joint weight.
        The weight and the cross-validation curve are in ``info['image_prior']``.
    mask_maps : False (default; maps cover the whole FOV) | True (zero the maps, and the images, outside
        the estimated object support ``info['support']``, as ESPIRiT's cropped maps; CG-SENSE is then
        support-constrained) | a bool array of the spatial shape (your own support; also used for the
        phase reference).
    support_threshold, support_dilate : the estimated support is a low-resolution root-sum-of-squares
        image (calibration region, Hann window, 5-pixel smoothing) above ``support_threshold`` x its
        99th percentile, holes filled, dilated by ``support_dilate`` pixels. Lower the threshold or
        dilate more if the mask clips the object edge (masked maps make SENSE zero there).
    solve_image : solve for the full-resolution image jointly with the maps (NLINV-style): after the
        calibration fit, maps and image are refined together on ALL acquired k-space (the maps stay
        physics-constrained; the image is solved by CG-SENSE inside the acquired k-space extent at every
        step). Implies ``return_image``; the returned image is that joint solution. 2-D (and dynamic
        with rank=0) only. Iterations: ``iters_joint`` (default 300), image regularisation
        ``lambda_joint_img`` (relative, default 1e-2).
    fold : 'auto' (default) | False. Objects larger than the field of view fold over: tissue beyond one
        edge appears at the opposite edge, seen by the coils on the far side. Physical maps are not periodic,
        so they would give that band the wrong coils (stripes in undersampled SENSE). 'auto' (2-D) compares,
        on the calibration images, the maps with the maps shifted by one FOV along each image axis; where
        the shifted maps explain the coil images much better, the returned maps are the effective
        (folded) sensitivities there. ``info['fold']`` reports the affected fraction per axis/direction.
        The readout axis is never corrected (oversampled by the scanner; an object cut there is truncated,
        not folded). Shifted maps that pass close to a conductor (its near field: much rougher than the
        maps themselves) are never used.
    plane_select : 2-D, plane='auto': 'heldout' (default) | 'probe'. When B0 is in-plane but rows vs
        columns is ambiguous (probe losses within 15 %), 'heldout' fits both orientations and keeps the one
        that better predicts acquired samples outside the calibration box (needs such samples, i.e. a
        mask with data outside the calibration region; doubles the fit time in that case only). The
        calibration loss cannot decide this: the wrong orientation can fit the calibration data as well.
    readout_axis : 2-D: image axis (0 or 1) of the readout, excluded from the fold-over search. 'auto'
        (default): the axis along which every acquired line is fully sampled, from ``mask``. Pass it
        when the mask cannot tell (e.g. only a fully sampled calibration box is given).
    undersample : retrospective Cartesian undersampling of (fully sampled) ``ksp`` before the fit: an
        acceleration R (float, or (R1, R2) in 3-D), a dict of ``bspinn.masks.cartesian_mask`` options
        (e.g. ``dict(R=4, pattern='vd', partial_fourier=0.75, seed=1)``) or a ready bool mask. The
        mask always contains a fully sampled calibration block, by default of the fit's ``calib`` size
        (``acs=`` to change it), and never samples outside the acquired data. The fit then uses every
        acquired sample in its calibration grid; ``return_image`` unfolds the rest by CG-SENSE.
        ``info['mask']``, ``info['R_effective']`` report what was used.
    cfg_overrides : any other FitConfig field (e.g. lambda_phys=0.1, field_mode='complex',
        optimizer='lbfgs' for full-batch L-BFGS in every stage, or 'adam+lbfgs' for Adam followed by an
        L-BFGS polish of all parameters (the 2-D PINN default; optimizer='adam' for Adam only); with
        'lbfgs' iters_* count L-BFGS iterations, each costing ~1-3 Adam
        iterations). GPU:
        compile=True | 'reduce-overhead' (torch.compile of each stage's iteration, + CUDA graphs),
        tf32=True (README §8.2).

    Returns
    -------
    maps : complex64 array with the shape and axis order of ``ksp``.
    image : (only with return_image=True) complex64, the spatial(-temporal) shape of ``ksp`` without
        the coil axis, in the same phase/scale convention as ``maps`` (ksp ~ F(maps * image)).
    info : dict with 'image_joint' (the image of the joint fit: it only lives on the calibration grid,
        i.e. it has the resolution of the calibration region; returned interpolated to the full matrix,
        in the convention of ``maps``), 'support' (object mask used for the phase reference), 'data_loss',
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
    _internal = cfg_overrides.pop("_internal", None)
    if layout == "dynamic" and dynamic_static == "average" and dynamic_base == "2d":
        return _dynamic_via_2d(
            ksp, mask, rank=rank, iters_dynamic=iters_dynamic, dynamic_options=dynamic_options,
            validate=dynamic_validate, undersample=undersample, return_image=return_image or solve_image,
            image_kw=dict(noise=noise, lam=sense_lambda, iters=sense_iters, device=device, lam_scale=sense_lambda_scale,
                          lam_options=sense_lambda_options, prior=image_prior, prior_weight=image_prior_weight,
                          prior_options=image_prior_options),
            kw2d=dict(spacing=spacing, plane=plane, calib=calib, noise=noise, physics=physics, b0_tesla=b0_tesla,
                      model=model, quality=quality, normalise=normalise, phase_ref=phase_ref, device=device,
                      threads=threads, seed=seed, verbose=verbose, mask_maps=mask_maps,
                      support_threshold=support_threshold, support_dilate=support_dilate, solve_image=solve_image,
                      fold=fold, readout_axis=readout_axis, plane_select=plane_select, image_prior=image_prior,
                      image_prior_weight=image_prior_weight, image_prior_options=image_prior_options,
                      sense_lambda=sense_lambda, sense_lambda_scale=sense_lambda_scale,
                      sense_lambda_options=sense_lambda_options, sense_iters=sense_iters, **cfg_overrides))
    if dynamic_base not in ("2d", "legacy"):
        raise ValueError("dynamic_base must be '2d' or 'legacy'")

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
        # frames per optimisation step: memory/cost ~ frames x coils; 4 up to 20 coils, 2 at 38 (OCMR fs_0020
        # OOM'd a 6 GB container with 4); frames are resampled every step
        cfg.slice_batch = min(4, max(1, 80 // max(1, ksp.shape[0])), ksp.shape[-1])
    if layout in ("2d", "dynamic") and model == "pinn":     # v50: the dynamic layout too (same static model)
        # v33 defaults from scripts/method_lab.py (real knees/brain + synthetic): learnable complex channel
        # coupling and a smooth per-coil complex factor (Fourier order 3)
        cfg.learn_mixing, cfg.resid_order = True, 3
        # v48: continuation of the smooth factor to order 8 after the PINN stage (README 8.10; k19870:
        # held-out leave-coil-out excess 0.047 -> 0.026, ACS residual 29.6 -> 17.4 x floor); resid_order_final=0
        # for the v47 behaviour (better on exact Biot-Savart phantoms: ground-truth error 3.5e-5 vs 1.0e-4)
        cfg.resid_order_final = 8
        # v42: Adam followed by a full-batch L-BFGS polish (runs/eval_joint, 20 real slices, joint protocol R=4:
        # lower leave-coil-out residual than Adam alone on 19/20, lower SENSE R4 error on 16/20, ~+4 s)
        cfg.optimizer = "adam+lbfgs"
    for k, v in cfg_overrides.items():
        if not hasattr(cfg, k):
            raise TypeError(f"unknown option {k!r} (not a FitConfig field)")
        setattr(cfg, k, v)
    if cfg.compress:
        raise ValueError("compress is not supported by estimate_maps (maps are returned for the input channels)")
    cfg._fold = fold if layout == "2d" else False
    cfg._plane_select = plane_select
    if solve_image:
        if layout == "3d":
            raise ValueError("solve_image is implemented for 2-D (and dynamic with rank=0) data")
        if layout == "dynamic" and rank != 0:
            raise ValueError("solve_image with dynamic data needs rank=0 (one static map set)")
        cfg.joint_full = True
        return_image = True
        from .prior import PRIORS
        if image_prior in PRIORS and layout == "2d" and "joint_prior" not in cfg_overrides:
            # the prior inside the joint fit as well: the maps are refined against the prior image (v48)
            cfg.joint_prior, cfg.joint_prior_weight = image_prior, image_prior_weight
            cfg.joint_prior_options = dict(image_prior_options or {})
        if "lambda_joint_img" not in cfg_overrides and (not isinstance(sense_lambda, str) or layout == "2d"):
            cfg.lambda_joint_img = sense_lambda              # one image-regularisation knob for both image paths
            cfg._lam_options = sense_lambda_options          # 2-D 'auto': chosen inside the joint stage
    under_info = {}
    if undersample is not None:
        ksp, mask, under_info = _apply_undersampling(ksp, mask, undersample, layout, cfg.calib)
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    accel.TIMES.clear()
    if threads:
        torch.set_num_threads(threads)

    if layout == "2d":
        C, info = _run_2d(ksp, mask, spacing, plane, noise, cfg, log)
    elif layout == "3d":
        C, info = _run_3d(ksp, mask, spacing, axes, noise, cfg, log)
    else:
        C, info = _run_dynamic(ksp, mask, spacing, plane, noise, cfg, rank, iters_dynamic, log, dynamic_static,
                               dynamic_options)

    # C: data-channel maps, (N, *spatial) or (T, N, X, Y) for dynamic. Pre-whitening made the data
    # channels M @ (original channels): map back to the original channels.
    prob = info["problem"]
    M = prob.data.meta.get("mix")
    fold_maps = info.pop("_fold_maps", None) or {}
    if M is not None:
        Minv = np.linalg.inv(np.asarray(M))
        if layout == "dynamic":                  # frame by frame in complex64: bounded memory for long series
            Minv64 = Minv.astype(np.complex64)
            C = np.stack([np.einsum("ij,j...->i...", Minv64, C[t]) for t in range(C.shape[0])])
        else:
            C = np.einsum("ij,j...->i...", Minv, C)
        fold_maps = {kk: np.einsum("ij,j...->i...", Minv, v) for kk, v in fold_maps.items()}
    C_raw = C
    if fold_maps and layout == "2d":
        C, info["fold"] = _fold_correct(C, fold_maps, ksp, mask, cfg.calib, log, readout_axis=readout_axis)
        if info["fold"]["corrected"] and solve_image:
            log("fold-over corrected maps: the full-resolution image is recomputed by SENSE with them "
                "(the joint solution used the uncorrected maps)")
            info.pop("_xfull", None)
    if _internal is not None:                  # the dynamic layout's base (v57): model, raw maps, its config
        _internal.update(cfg=cfg, C_pre_finish=C, model=info.get("model"), problem=prob, m=info.get("_m"))
    # common phase + normalisation (per-pixel complex scalars, identical for all coils)
    if isinstance(mask_maps, (bool, np.bool_)) or mask_maps is None:
        sup = _support(ksp, mask, layout, cfg.calib, support_threshold, support_dilate)
    else:
        sup = np.asarray(mask_maps, bool)
        if sup.shape != ksp.shape[1:3 if layout == "dynamic" else None]:
            raise ValueError(f"mask_maps array must have the spatial shape {ksp.shape[1:3 if layout == 'dynamic' else None]}")
    with accel.phase("finish"):
        C = _finish(C, sup, layout, phase_ref, normalise, device)
    x_solved = None
    xf = info.pop("_xfull", None)
    if xf is not None:                         # joint full-resolution image, in the convention of the maps
        f = np.asarray(xf) * prob.scale
        cax = 1 if layout == "dynamic" else 0
        Cd = C_raw if M is None else (np.einsum("ij,tj...->ti...", np.asarray(M), C_raw) if layout == "dynamic"
                                      else np.einsum("ij,j...->i...", np.asarray(M), C_raw))   # data channels
        Cn = C_raw / np.maximum(np.sqrt((np.abs(Cd) ** 2).sum(cax, keepdims=True)), 1e-30)   # RSS as in the joint stage
        g = (Cn.conj() * C).sum(cax) / np.maximum((np.abs(Cn) ** 2).sum(cax), 1e-30)
        f = f[0] if layout == "2d" else f
        x_solved = (f / np.where(np.abs(g) > 0, g, 1)).astype(np.complex64)
    # joint-fit image, interpolated to the full matrix, in the convention of the returned maps
    info["image_joint"] = _joint_image(info.pop("_m", None), prob, C_raw, C, layout)
    if layout == "dynamic":
        C = np.moveaxis(C, 0, -1)                       # (T,N,X,Y) -> (N,X,Y,T)
        if info["image_joint"] is not None:
            info["image_joint"] = np.moveaxis(info["image_joint"], 0, -1)
    if x_solved is not None and layout == "dynamic":
        x_solved = np.moveaxis(x_solved, 0, -1)
    if mask_maps is not False and mask_maps is not None:
        sx = sup[..., None] if layout == "dynamic" else sup         # dynamic: same support for all frames
        C = C * sx[None]
        if x_solved is not None:
            x_solved = x_solved * sx
        if info["image_joint"] is not None:
            info["image_joint"] = info["image_joint"] * sx
    info.update(under_info)
    info.update(support=sup, maps_masked=mask_maps is not False and mask_maps is not None,
                seconds=time.time() - t0, config=_cfg_dict(cfg), timings=dict(accel.TIMES))
    log(f"maps {C.shape} in {info['seconds']:.0f}s")
    C = C.astype(np.complex64)
    if not return_image:
        return C, info
    if x_solved is not None and image_prior is not None:      # maps jointly; the returned image with the prior
        info["image_solved_jointly"] = False
        info["image_prior_joint" if info.get("joint_prior") else "image_tikhonov_joint"] = x_solved
    elif x_solved is not None and (not isinstance(sense_lambda, str) or layout == "2d"):
        info["image_solved_jointly"] = True
        if info.get("joint_lambda") is not None:
            info["sense_lambda"] = info["joint_lambda"]
        return C, x_solved, info
    if x_solved is not None:                                   # dynamic + automatic weight
        info["image_solved_jointly"] = False                   # maps jointly; image recomputed below
    lam_rep = {}
    with accel.phase("sense_image", device):
        img = sense_image(ksp, C, layout, mask=mask, noise=noise, lam=sense_lambda, iters=sense_iters, device=device,
                          lam_scale=sense_lambda_scale, report=lam_rep, lam_options=sense_lambda_options,
                          prior=image_prior, prior_weight=image_prior_weight, prior_options=image_prior_options)
    info["image_prior" if image_prior is not None else "sense_lambda"] = lam_rep
    info["timings"] = dict(accel.TIMES)
    info["seconds"] = time.time() - t0
    return C, img, info


# ---------------------------------------------------------------------------------- per layout
def _select_plane(kst, mst, spacing, noise, cfg, log, meta=None):
    """kst (S,N,X,Y), mst (S,1,X,Y) or None. Returns (plane, scores)."""
    from .data import MultiCoilData
    from .fit import Problem, fit_wires_many
    from .geometry import SliceGeometry
    S, N, X, Y = kst.shape
    scores, hscores, items = {}, {}, []
    planes = ("axial", "coronal", "coronal_t")
    for pl in planes:
        d = MultiCoilData(kst.copy(), SliceGeometry((X, Y), spacing, np.zeros(S), pl), mask=mst, noise=noise,
                          meta=dict(meta or {}))
        d.prewhiten()
        prob = Problem(d, cfg).to(cfg.device)
        items += [(prob, 1), (prob, -1)]
    res = fit_wires_many(items, cfg, cfg.iters_auto)               # 6 probes, concurrently on a GPU
    for i, pl in enumerate(planes):
        hscores[pl] = {1: res[2 * i][3], -1: res[2 * i + 1][3]}
        scores[pl] = min(hscores[pl].values())
    best = min(scores, key=scores.get)
    meta_out = {"handedness_scores": hscores[best], "all_handedness": hscores, "ambiguous": False}
    log("B0 orientation (plane) scores: " + ", ".join(f"{k} {v:.3e}" for k, v in scores.items()) + f" -> {best}")
    if best != "axial" and min(scores["coronal"], scores["coronal_t"]) > 0.85 * max(scores["coronal"], scores["coronal_t"]):
        meta_out["ambiguous"] = True
        if (meta or {}).get("resolvable"):
            log(f"B0 in-plane orientation ambiguous ({scores['coronal']:.3e} vs {scores['coronal_t']:.3e}): "
                "fitting both, deciding by held-out k-space prediction")
            return best, scores, meta_out
        import warnings
        msg = ("B0 lies in the image plane, but whether it runs along the rows ('coronal'/'sagittal') or the "
               f"columns ('coronal_t') is hardly identifiable from these data ({scores['coronal']:.3e} vs "
               f"{scores['coronal_t']:.3e}); picked {best!r}. Pass plane= if you know the acquisition.")
        warnings.warn(msg)
        log("warning: " + msg)
    return best, scores, meta_out


def _calib_box_outside(k, mask, calib):
    N, X, Y = k.shape
    m = (np.abs(k).sum(0) > 0) if mask is None else np.asarray(mask, bool)
    cal = calib if isinstance(calib, (tuple, list)) else (calib, calib)
    box = np.zeros((X, Y), bool)
    box[X // 2 - cal[0] // 2: X // 2 - cal[0] // 2 + cal[0], Y // 2 - cal[1] // 2: Y // 2 - cal[1] // 2 + cal[1]] = True
    return m, m & ~box


def _n_heldout(k, mask, calib):
    """Acquired samples outside the calibration box (0 if they are fewer than 1 % of the acquired ones)."""
    m, out = _calib_box_outside(k, mask, calib)
    n = int(out.sum())
    return n if n >= 0.01 * m.sum() else 0


def _heldout_score(C, k, mask, calib, device, frac=0.2, folds=2, iters=30, lam=1e-3, seed=0):
    """Relative error of predicting acquired k-space samples outside the calibration box that the
    reconstruction did not use: CG-SENSE (RSS-normalised maps C) from the other acquired samples, then
    C x at the held-out samples; mean over ``folds`` disjoint random folds. None if fewer than 1 % of
    the samples lie outside the box."""
    N, X, Y = k.shape
    m, out = _calib_box_outside(k, mask, calib)
    idx = np.flatnonzero(out)
    if idx.size < 0.01 * m.sum():
        return None
    dev = torch.device(device)
    Cn = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)
    Ct = torch.from_numpy(Cn.astype(np.complex64)).to(dev)
    kt = torch.from_numpy(np.ascontiguousarray(k).astype(np.complex64)).to(dev)
    F = lambda x: torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))
    Fi = lambda x: torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))
    rng = np.random.default_rng(seed); rng.shuffle(idx)
    errs = []
    for f in range(folds):
        ho = np.zeros(X * Y, bool); ho[idx[int(f * frac * idx.size): int((f + 1) * frac * idx.size)]] = True
        ho = ho.reshape(X, Y)
        tr = torch.from_numpy((m & ~ho).astype(np.float32)).to(dev)
        hot = torch.from_numpy(ho).to(dev)
        A = lambda x: (Ct.conj() * Fi(tr * F(Ct * x))).sum(0) + lam * x
        b = (Ct.conj() * Fi(tr * kt)).sum(0)
        x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rs = (r.conj() * r).real.sum()
        for _ in range(iters):
            Ap = A(p); al = rs / (p.conj() * Ap).real.sum().clamp_min(1e-30)
            x = x + al * p; r = r - al * Ap; rn = (r.conj() * r).real.sum()
            p = r + rn / rs.clamp_min(1e-30) * p; rs = rn
        pred = F(Ct * x)
        errs.append(float((pred[:, hot] - kt[:, hot]).abs().pow(2).sum() / kt[:, hot].abs().pow(2).sum().clamp_min(1e-30)))
    return float(np.sqrt(np.mean(errs)))


def _fit_2d_plane(ksp, mask, spacing, plane, noise, cfg, log):
    from .data import MultiCoilData
    from .fit import fit
    from .geometry import SliceGeometry
    N, X, Y = ksp.shape
    geom = SliceGeometry((X, Y), spacing, np.zeros(1), plane)
    data = MultiCoilData(ksp[None], geom, mask=None if mask is None else mask[None, None], noise=noise)
    model, prob, hist, extra = fit(data, cfg, log=log)
    with accel.phase("maps_eval", cfg.device):
        pts = geom.grid().reshape(-1, 3)
        model.to(cfg.device)                                   # evaluate the full-resolution maps on the device
        C = prob.mixC(model.sensitivity_batched(pts), -2).detach().numpy().reshape(N, X, Y)
        fold_maps = {}
        if getattr(cfg, "_fold", "auto"):
            g = geom.grid().reshape(X, Y, 3)
            fov = {0: (g[1, 0] - g[0, 0]) * X, 1: (g[0, 1] - g[0, 0]) * Y}     # one FOV along each image axis
            for a in (0, 1):
                for sgn in (1, -1):
                    ps = pts + sgn * fov[a]
                    fold_maps[(a, sgn)] = prob.mixC(model.sensitivity_batched(ps), -2).detach().numpy().reshape(N, X, Y)
        model.cpu()
    return C, {"handedness": extra["handedness"], "data_loss": _last(hist), "model": model, "problem": prob,
               "_m": extra["image_lowres"], "_xfull": extra.get("image_full"), "plane": plane,
               "_fold_maps": fold_maps, "joint_lambda": extra.get("joint_lambda"),
               "joint_prior": extra.get("joint_prior")}


def _run_2d(ksp, mask, spacing, plane, noise, cfg, log):
    """Fit one 2-D slice. plane='auto': short wire probes for every orientation. When B0 is in-plane
    but rows vs columns is ambiguous (within 15 %) and there are acquired samples outside the
    calibration box, both in-plane orientations are fitted in full and the one that predicts held-out
    acquired k-space better wins: on real knees the wrong orientation fits the calibration data as well
    or better (the calibration loss cannot tell) but extrapolates much worse."""
    scores, cands = None, None
    if plane == "auto":
        resolvable = getattr(cfg, "_plane_select", "heldout") == "heldout" and _n_heldout(ksp, mask, cfg.calib) > 0
        with accel.phase("plane_search", cfg.device):
            plane, scores, ps = _select_plane(ksp[None], None if mask is None else mask[None, None], spacing, noise,
                                              cfg, log, meta={"resolvable": resolvable})
        if ps.get("ambiguous") and resolvable:
            cands = [("coronal", ps["all_handedness"]["coronal"]), ("coronal_t", ps["all_handedness"]["coronal_t"])]
        else:
            cands = [(plane, ps["handedness_scores"])]
    else:
        cands = [(plane, None)]
    results = []
    for pl, hs in cands:
        c = cfg
        if hs is not None and cfg.handedness == 0:   # the plane search fitted both handednesses on this problem
            c = copy.copy(cfg)
            c.handedness = min(hs, key=hs.get)
            log(f"[{pl}] handedness from the plane search: {c.handedness:+d} (data loss +1: {hs[1]:.3e}, -1: {hs[-1]:.3e})")
        C, info = _fit_2d_plane(ksp, mask, spacing, pl, noise, c, log)
        results.append((C, info))
    if len(results) > 1:
        M = results[0][1]["problem"].data.meta.get("mix")
        kw = ksp if M is None else np.einsum("ij,j...->i...", np.asarray(M), ksp)   # the data channels of C
        ho = {info["plane"]: _heldout_score(C, kw, mask, cfg.calib, cfg.device) for C, info in results}
        pick = min(range(len(results)), key=lambda i: ho[results[i][1]["plane"]])
        log("in-plane B0 orientation by held-out k-space prediction: " +
            ", ".join(f"{k} {v:.4f}" for k, v in ho.items()) + f" -> {results[pick][1]['plane']}")
        C, info = results[pick]
        info["plane_heldout"] = ho
    else:
        C, info = results[0]
    info["plane_scores"] = scores
    return C, info


def _run_3d(ksp, mask, spacing, axes, noise, cfg, log):
    from .volume import VolumeData, VolumeGeometry, fit3d, volume_maps
    orient = "auto" if axes == "auto" else "given"
    geom = VolumeGeometry(ksp.shape[1:], spacing, "zyx" if axes == "auto" else axes)
    data = VolumeData(ksp, geom, mask=mask, noise=noise)
    model, prob, hist, extra = fit3d(data, cfg, log=log, orientation=orient)
    from .models import HelmholtzPINN, WaveWireLoopCoils
    with accel.phase("maps_eval", cfg.device):
        if not isinstance(model, (HelmholtzPINN, WaveWireLoopCoils)):
            model.to(cfg.device)                               # wave models: evaluated as trained (CPU grids)
        C = volume_maps(model, prob).numpy()
        model.cpu()
    info = {"handedness": extra["handedness"], "orientation": prob.data.geom.axes,
            "orientation_note": "physical axis (x,y,z; B0 = z) of array axes 1..3", "data_loss": _last(hist),
            "model": model, "problem": prob, "_m": extra["image_lowres"]}
    if hasattr(model, "k2_re"):
        from .models import k2_to_tissue
        k2 = complex(float(model.k2_re), float(model.k2_im))
        info["k2"] = k2
        try:
            info["tissue"] = k2_to_tissue(k2, cfg.b0_tesla, prob.data.geom.half_fov / 1000.0)
        except Exception:                                                  # pragma: no cover
            pass
    return C, info


def _run_dynamic(ksp, mask, spacing, plane, noise, cfg, rank, iters_dynamic, log, static="average", options=None):
    from .data import MultiCoilData
    from .geometry import SliceGeometry
    N, X, Y, T = ksp.shape
    kt = np.ascontiguousarray(np.moveaxis(ksp, -1, 0))                      # (T,N,X,Y)
    mt = None if mask is None else np.moveaxis(mask, -1, 0)[:, None]        # (T,1,X,Y)
    scores = None
    m_all = mt if mt is not None else (np.abs(kt).sum(1, keepdims=True) > 0)  # (T,1,X,Y)
    kav = cnt = None
    if static == "average":
        # the static part from the time-averaged k-space (sum over frames / frames sampling each point): with
        # per-frame undersampling (GRO: no per-frame calibration region) a joint fit to the frames is
        # ill-posed; the average is densely sampled (v50, README 8.11). Frame by frame: bounded memory.
        cnt = m_all[:, 0].sum(0).astype(np.float32)
        kav = np.zeros((N, X, Y), np.complex64)
        for t in range(T):
            kav += kt[t] * m_all[t]
        kav /= np.maximum(cnt, 1)[None]
    elif static != "frames":
        raise ValueError("dynamic_static must be 'average' or 'frames'")
    if plane == "auto":
        # with the static part from the average, select the B0 plane on it too (the data of the static fit;
        # on all frames the wire fits scale with the number of frames: us_0001, 48 frames, OOM'd 6 GB)
        if kav is not None:
            plane, scores, _ = _select_plane(kav[None], (cnt > 0)[None, None], spacing, noise, cfg, log,
                                             meta={"dynamic": True})
        else:
            plane, scores, _ = _select_plane(kt, m_all, spacing, noise, cfg, log, meta={"dynamic": True})
    geom = SliceGeometry((X, Y), spacing, np.zeros(T), plane)               # frames = co-located slices
    data = MultiCoilData(kt, geom, mask=mt, noise=noise, meta={"dynamic": True})
    if mt is None:
        data.mask = m_all
    pts = geom.grid((X, Y))[0].reshape(-1, 3)
    avg_data = None
    if kav is not None:
        avg_data = MultiCoilData(kav[None], SliceGeometry((X, Y), spacing, np.zeros(1), plane),
                                 mask=(cnt > 0)[None, None], noise=noise)
    if rank == 0:
        from .fit import fit
        model, prob, hist, extra = fit(avg_data if avg_data is not None else data, cfg, log=log)
        model.to(cfg.device)
        C = prob.mixC(model.sensitivity_batched(pts), -2).detach().numpy().reshape(N, X, Y)
        model.cpu()
        return np.broadcast_to(C[None], (T, N, X, Y)).copy(), {
            "handedness": extra["handedness"], "data_loss": _last(hist), "rank": 0, "model": model, "problem": prob,
            "_m": extra["image_lowres"], "_xfull": extra.get("image_full"), "plane": plane, "plane_scores": scores}
    from .dynamic import DynConfig, fit_dynamic
    dcfg = DynConfig(rank=rank, iters_dyn=iters_dynamic or max(300, cfg.iters_pinn // 2), **(options or {}))
    dyn, prob, hist, extra = fit_dynamic(data, cfg, dcfg, log=log, static_data=avg_data)
    out = []
    dyn.to(cfg.device); pts_d = pts.to(cfg.device)
    # frames x pixels in pieces: bounded memory at full resolution (the field intermediates scale with
    # pixels x wire segments x coils; 16384 pixels OOM'd at 38 coils in a 6 GB container, OCMR fs_0020)
    chunk = max(1024, 4096 * 16 // max(16, prob.N_model))
    with torch.no_grad():
        for t in range(T):
            c = torch.cat([dyn.sensitivity_frames(pts_d[j:j + chunk], [t], create_graph=False).detach().cpu()
                           for j in range(0, pts_d.shape[0], chunk)], -1)               # (1,N_model,P)
            out.append(prob.mixC(c, -2).numpy())
    dyn.cpu()
    C = np.concatenate(out).reshape(T, N, X, Y)
    return C, {"handedness": extra["handedness"], "rank": rank, "temporal_modes": dyn.alpha.detach().numpy(),
               "data_loss": float(hist["dynamic"][-1]["data"]) if hist.get("dynamic") else None,
               "model": dyn, "problem": prob, "_m": extra["m"], "plane": plane, "plane_scores": scores}


# ---------------------------------------------------------------------------------- dynamic from the 2-D fit (v57)
def _time_average(ksp, m_all):
    N, X, Y, T = ksp.shape
    cnt = m_all.sum(-1)
    kav = np.zeros((N, X, Y), np.complex64)
    for t in range(T):
        kav += ksp[..., t] * m_all[..., t]
    kav /= np.maximum(cnt, 1)[None]
    return kav, cnt > 0


def _dynamic_via_2d(ksp, mask, *, rank, iters_dynamic, dynamic_options, validate, undersample, return_image,
                    image_kw, kw2d):
    """layout='dynamic', dynamic_static='average', dynamic_base='2d': the 2-D path on the time-averaged k-space
    gives the static maps (identical to estimate_maps(time average, '2d')); rank > 0 adds per-frame corrections
    of that fixed model, optionally validated on withheld lines (see estimate_maps)."""
    log = print if kw2d.get("verbose", True) else (lambda *a, **k: None)
    if kw2d.get("solve_image") and rank != 0:
        raise ValueError("solve_image with dynamic data needs rank=0 (one static map set)")
    t0 = time.time()
    under_info = {}
    if undersample is not None:
        ksp, mask, under_info = _apply_undersampling(ksp, mask, undersample, "dynamic", int(max(_calib(kw2d.get("calib"), 24, 1))))
    N, X, Y, T = ksp.shape
    m_all = (np.abs(ksp).sum(0) > 0) if mask is None else np.asarray(mask, bool)
    kav, mavg = _time_average(ksp, m_all)
    internal = {}
    log(f"dynamic: maps of the time-averaged k-space by the 2-D path ({T} frames, {int(mavg.any(0).sum())} lines)")
    out = estimate_maps(kav, "2d", mask=mavg, return_image=False, _internal=internal, **kw2d)
    C2, info = out[0], dict(out[-1])
    ij = info.pop("image_joint", None)
    info.pop("image_tikhonov_joint", None); info.pop("image_prior_joint", None)
    info.update(rank=rank, dynamic_base="2d", time_average_lines=int(mavg.any(0).sum()))
    if rank == 0:
        C = np.repeat(C2[..., None], T, -1)
    else:
        C = _dynamic_corrections(ksp, m_all, C2, internal, info, rank, iters_dynamic, dynamic_options, validate, kw2d, log)
    info.update(under_info)
    info["image_joint"] = None if ij is None else np.repeat(np.asarray(ij)[..., None], T, -1)   # the time average's
    info["seconds"] = time.time() - t0
    C = C.astype(np.complex64)
    if not return_image:
        return C, info
    rep = {}
    img = sense_image(ksp, C, "dynamic", mask=m_all, report=rep, **image_kw)
    info["image_solved_jointly"] = False
    info["image_prior" if image_kw.get("prior") is not None else "sense_lambda"] = rep
    info["seconds"] = time.time() - t0
    return C, img, info


def _holdout_lines(m_all, frac, seed=0):
    """(X,Y,T) mask -> held-out mask: round(frac x acquired lines) (>= 1) whole phase-encode lines per frame, never
    a frame's only line."""
    rng = np.random.default_rng(seed + 4242)
    lines = m_all.any(0)                                   # (Y,T)
    ho = np.zeros_like(lines)
    for t in range(lines.shape[1]):
        acq = np.flatnonzero(lines[:, t])
        n = min(max(1, int(round(frac * acq.size))), acq.size - 1)
        if n > 0:
            ho[rng.choice(acq, n, replace=False), t] = True
    return m_all & ho[None]


def _dynamic_corrections(ksp, m_all, C2, internal, info, rank, iters_dynamic, dynamic_options, validate, kw2d, log):
    from .data import MultiCoilData
    from .dynamic import DynConfig, fit_dynamic
    from .geometry import SliceGeometry
    N, X, Y, T = ksp.shape
    opts = dict(dynamic_options or {})
    vopt = {k: opts.pop(k) for k in ("holdout", "shrink_grid", "min_gain", "val_weight", "val_iters") if k in opts}
    validate = True if validate == "auto" else bool(validate)
    cfg = copy.deepcopy(internal["cfg"])
    cfg.joint_full = False
    cfg.joint_prior = None
    cfg.handedness = int(info["handedness"])
    if "slice_batch" not in kw2d:
        cfg.slice_batch = min(4, max(1, 80 // max(1, N)), T)
    ho = _holdout_lines(m_all, vopt.get("holdout", 0.1), cfg.seed) if validate else np.zeros_like(m_all)
    tr = m_all & ~ho
    kt = np.ascontiguousarray(np.moveaxis(ksp * tr[None], -1, 0))          # (T,N,X,Y) training samples
    mt = np.moveaxis(tr, -1, 0)[:, None]
    geom = SliceGeometry((X, Y), tuple(kw2d["spacing"]), np.zeros(T), info["plane"])
    data = MultiCoilData(kt, geom, mask=mt, noise=kw2d.get("noise"), meta={"dynamic": True})
    dcfg = DynConfig(rank=rank, iters_dyn=iters_dynamic or max(300, cfg.iters_pinn // 2), **opts)
    dyn, prob, hist, extra = fit_dynamic(data, cfg, dcfg, log=log, static_fit=(
        internal["model"], internal["problem"], internal["m"], info["handedness"]))
    del kt, data
    # per-frame corrections at full resolution, data channels -> original channels (as the 2-D maps)
    pts = geom.grid((X, Y))[0].reshape(-1, 3).to(cfg.device)
    dyn.to(cfg.device)
    chunk = max(1024, 4096 * 16 // max(16, prob.N_model))
    M = prob.data.meta.get("mix")
    Minv = None if M is None else np.linalg.inv(np.asarray(M)).astype(np.complex64)

    def frame(t):
        c = torch.cat([dyn.sensitivity_frames(pts[j:j + chunk], [t], create_graph=False).detach().cpu()
                       for j in range(0, pts.shape[0], chunk)], -1)
        return prob.mixC(c, -2).numpy()[0].reshape(-1, X, Y)
    with torch.no_grad():
        a = dyn.alpha.data.clone()
        dyn.alpha.data.zero_()
        base = frame(0)
        dyn.alpha.data.copy_(a)
        D = np.stack([frame(t) - base for t in range(T)], -1)                    # (N,X,Y,T)
    dyn.cpu()
    if Minv is not None:
        D = np.einsum("ij,j...->i...", Minv, D)
    Craw = np.asarray(internal["C_pre_finish"])
    g = (Craw.conj() * C2).sum(0) / np.maximum((np.abs(Craw) ** 2).sum(0), 1e-30)   # per-pixel gauge/normalisation
    rel = float(np.sqrt((np.abs(D) ** 2).mean() / max((np.abs(Craw) ** 2).mean(), 1e-30)))
    log(f"dynamic corrections: relative size {rel:.4f} (RMS over frames, vs the base maps)")

    def maps(s):
        Cs = g[..., None] * (Craw[..., None] + s * D)
        if kw2d.get("normalise", "rss") == "rss":
            Cs = Cs / np.maximum(np.sqrt((np.abs(Cs) ** 2).sum(0, keepdims=True)), 1e-12)
        return np.where(np.abs(C2[..., None]).sum(0, keepdims=True) > 0, Cs, 0).astype(np.complex64)
    info["model_static"] = info.get("model")
    info.update(temporal_modes=dyn.alpha.detach().numpy(), correction_rel=rel, model=dyn, dynamic_maps_fn=maps,
                data_loss_dynamic=float(hist["dynamic"][-1]["data"]) if hist.get("dynamic") else None)
    if not validate:
        return maps(1.0)
    shrink = list(vopt.get("shrink_grid", (0.25, 0.5, 1.0)))
    scores = {0.0: _heldout_series(ksp, np.repeat(C2[..., None], T, -1), tr, ho, vopt, kw2d.get("device", "cpu"))}
    for s in shrink:
        scores[float(s)] = _heldout_series(ksp, maps(s), tr, ho, vopt, kw2d.get("device", "cpu"))
    best = min(scores, key=scores.get)
    keep = best > 0 and scores[best] < (1 - vopt.get("min_gain", 0.02)) * scores[0.0]
    s_use = best if keep else 0.0
    log("dynamic corrections, held-out error by scale: " + ", ".join(f"{k:g} {v:.4f}" for k, v in scores.items())
        + f" -> {s_use:g}")
    info["dynamic_validation"] = {"scores": {str(k): v for k, v in scores.items()}, "scale": s_use,
                                  "heldout_samples": int(ho.sum())}
    return maps(s_use) if s_use > 0 else np.repeat(C2[..., None], T, -1)


def _heldout_series(ksp, C, tr, ho, vopt, device):
    """Relative error predicting the held-out samples ho (X,Y,T) from a spatio-temporal TV image of the training
    samples tr with maps C (N,X,Y,T); fixed weight (relative, bspinn.prior scaling)."""
    from .prior import _dyn_setup, _fft2c, dynamic_admm
    k, m, Cn, Ct, mt, yt, x0, s = _dyn_setup(ksp * tr[None], C, tr, device)
    w = float(vopt.get("val_weight", 3e-3))
    with torch.no_grad():
        x = dynamic_admm(yt, Ct, mt, w, 0.25 * w, int(vopt.get("val_iters", 40)), x0=x0)
        P = _fft2c(Ct * x[:, None]) * s                                          # (T,N,X,Y)
    hot = torch.from_numpy(np.moveaxis(ho, -1, 0)).to(P.device)                  # (T,X,Y)
    y = torch.from_numpy(np.moveaxis(np.asarray(ksp), -1, 0).astype(np.complex64)).to(P.device)
    num = (P - y).abs().pow(2).permute(0, 2, 3, 1)[hot].sum()
    den = y.abs().pow(2).permute(0, 2, 3, 1)[hot].sum().clamp_min(1e-30)
    return float(torch.sqrt(num / den))


# ---------------------------------------------------------------------------------- helpers
def _apply_undersampling(ksp, mask, undersample, layout, calib):
    """Retrospective undersampling: build (or take) a Cartesian mask with the calibration block of
    the fit, restricted to the acquired samples; zero the rest of ``ksp``."""
    from .masks import cartesian_mask, effective_acceleration
    acquired = (np.abs(ksp).sum(0) > 0) if mask is None else mask
    if isinstance(undersample, np.ndarray):
        m = np.asarray(undersample, bool)
        if m.shape != ksp.shape[1:]:
            raise ValueError(f"undersample mask must have shape {ksp.shape[1:]}, got {m.shape}")
        opts = {}
    else:
        opts = dict(undersample) if isinstance(undersample, dict) else {"R": undersample}
        if "acs" not in opts:                                  # the calibration region of the fit
            c = calib if isinstance(calib, (tuple, list)) else (calib,)
            opts["acs"] = c[0] if layout != "3d" else (c[1], c[2]) if len(c) == 3 else c[0]
        m = cartesian_mask(ksp.shape[1:], layout=layout, extent=acquired, **opts)
    m = m & acquired
    _check_calibration(m, acquired, calib, layout)
    info = {"mask": m, "R_effective": effective_acceleration(m, acquired), "undersample": {k: v for k, v in opts.items()}}
    return ksp * m[None], m, info


def _check_calibration(m, acquired, calib, layout):
    """Warn if the centre ``calib`` block of k-space (what the fit calibrates on) is not fully sampled
    wherever data were acquired. The fit still runs (k-space data term), but with fewer samples."""
    import warnings
    sp = m.shape if layout != "dynamic" else m.shape[:-1]
    c = calib if isinstance(calib, (tuple, list)) else (calib,) * len(sp)
    sl = tuple(slice(max(0, n // 2 - min(int(a), n) // 2), max(0, n // 2 - min(int(a), n) // 2) + min(int(a), n))
               for n, a in zip(sp, c))
    if layout == "dynamic":
        sl = sl + (slice(None),)
    frac = float(m[sl].sum() / max(1, acquired[sl].sum()))
    if frac < 0.999:
        warnings.warn(f"the mask samples {100 * frac:.0f}% of the central calibration block "
                      f"{tuple(int(min(a, n)) for a, n in zip(c, sp))} the fit uses (calib={calib}); "
                      "add an ACS of that size to the mask (cartesian_mask(..., acs=...)) or lower calib")


def _calib(calib, default, n):
    if calib is None:
        calib = default
    c = tuple(int(v) for v in (calib if isinstance(calib, (tuple, list)) else (calib,) * n))
    return c if n > 1 else c[:1]


def _last(hist):
    for key in ("polish", "pinn", "wires"):
        h = hist.get(key) if isinstance(hist, dict) else None
        if h:
            rec = h[-1]
            return float(rec.get("data", rec.get("loss", np.nan)))
    return None


def _cfg_dict(cfg):
    import dataclasses
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


def _support(ksp, mask, layout, calib, thresh=0.1, dilate=2):
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
    s = rss > thresh * np.quantile(rss, 0.99)
    from .keepout import clean_support
    s = clean_support(s[None], True)[0] if s.ndim == 2 else clean_support(s, False)
    return binary_dilation(s, iterations=dilate) if dilate > 0 else s


def _proj_rough(C):
    """Per-pixel gauge-free roughness: max over the image axes of ||P(r) - P(r+1)|| for the projector
    P = c c^H of the RSS-normalised maps (sqrt(2 - 2 |c(r)^H c(r+1)|^2))."""
    Cn = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-20)
    out = np.zeros(C.shape[1:])
    for ax in (1, 2):
        ov = np.abs((Cn.conj() * np.roll(Cn, -1, axis=ax)).sum(0)) ** 2
        r = np.sqrt(np.clip(2 - 2 * ov, 0, None))
        sl = [slice(None)] * 2; sl[ax - 1] = slice(-1, None); r[tuple(sl)] = 0     # no wrap-around step
        out = np.maximum(out, np.maximum(r, np.roll(r, 1, axis=ax - 1)))
    return out


def _fold_correct(C, fold_maps, ksp, mask, calib, log, ridge=1e-3, min_gain=0.3, min_frac=0.25, min_misfit=0.2,
                  readout_axis="auto", rough_factor=3.0):
    """Effective maps where the object folds over the FOV edge (see estimate_maps(fold=...)).

    On the calibration coil images I (central calib x calib block, Hann window), per pixel: residual of the
    single map set C, and of [C, C_shift] (C_shift = maps one FOV away along an image axis) with fitted
    coefficients (m1, m2). A shift is accepted where it explains I clearly better (residual < min_gain x
    single-set residual), carries > min_frac of the fitted energy, the single set misfits there (relative
    residual > min_misfit) and the pixel has signal; only regions connected to the FOV edge where folded
    tissue appears are kept (fold-over is a band at the edge), then smoothed. There the effective sensitivity is m1 C + m2 C_shift (divided by the larger of m1,
    m2: no object phase), which is what the coils really see. Returns (maps, report)."""
    from scipy.ndimage import binary_dilation, gaussian_filter
    N, X, Y = C.shape
    m = (np.abs(ksp).sum(0) > 0) if mask is None else np.asarray(mask, bool)
    win = np.zeros((X, Y))
    c = int(calib)
    r0, c0 = X // 2 - c // 2, Y // 2 - c // 2
    win[r0:r0 + c, c0:c0 + c] = np.outer(np.hanning(c + 2)[1:-1], np.hanning(c + 2)[1:-1])
    I = _cfft(ksp * (m * win)[None], (1, 2), True)                       # low-resolution coil images
    rssI = np.sqrt((np.abs(I) ** 2).sum(0))
    sig = rssI > 0.05 * np.percentile(rssI, 99)

    def fit(sets):                                                        # per-pixel LS: I ~ sum_k m_k S_k
        S = np.stack(sets, 1)                                             # (N,K,X,Y)
        G = np.einsum("nkxy,nlxy->xykl", S.conj(), S)
        b = np.einsum("nkxy,nxy->xyk", S.conj(), I)
        K = len(sets)
        G = G + ridge * np.trace(G, axis1=2, axis2=3).real[..., None, None] / K * np.eye(K)
        mm = np.linalg.solve(G, b[..., None])[..., 0]                     # (X,Y,K)
        r = I - np.einsum("nkxy,xyk->nxy", S, mm)
        return mm, (np.abs(r) ** 2).sum(0)
    # the readout axis (every acquired line fully sampled along it) is oversampled by the scanner: no fold
    if readout_axis in (0, 1):
        ro = [int(readout_axis)]
    else:
        ro = [a for a in (0, 1) if mask is not None and m.any() and bool((m == m.any(axis=a, keepdims=True)).all())
              and not m.all()]
        if not ro and not m.all():
            log("fold-over search on both image axes: the mask does not show the readout axis "
                "(pass readout_axis= to exclude it)")
    fold_maps = {kk: v for kk, v in fold_maps.items() if kk[0] not in ro}
    # maps one FOV away can pass right next to a conductor (allowed there: outside the object). Its near
    # field is far rougher than any physical map inside the object: never use those pixels.
    rough0 = _proj_rough(C)
    rough_lim = rough_factor * max(float(np.percentile(rough0[sig], 99)) if sig.any() else 0.0, 1e-3)
    _, r1 = fit([C])
    misfit = r1 / np.maximum(rssI ** 2, 1e-30) > min_misfit
    out, rep = C.copy(), {"corrected": False, "regions": {}, "readout_axis_excluded": ro}
    best_gain = np.ones((X, Y)); choice, coef = {}, {}
    for key, Cs in fold_maps.items():
        mm, r2 = fit([C, Cs])
        frac = np.abs(mm[..., 1]) ** 2 / np.maximum((np.abs(mm) ** 2).sum(-1), 1e-30)
        gain = r2 / np.maximum(r1, 1e-30)
        smooth_ok = _proj_rough(Cs) <= rough_lim
        smooth_ok = ~binary_dilation(~smooth_ok, iterations=3)
        acc = sig & misfit & (gain < min_gain) & (frac > min_frac) & smooth_ok
        # tissue from beyond one edge appears at the opposite edge: C(r - FOV) shows up near the + edge
        from scipy.ndimage import label
        lab, n = label(acc)
        a, sgn = key
        edge = (lab[-3:, :] if sgn < 0 else lab[:3, :]) if a == 0 else (lab[:, -3:] if sgn < 0 else lab[:, :3])
        keep = np.setdiff1d(np.unique(edge), [0])
        acc = np.isin(lab, keep)
        upd = acc & (gain < best_gain)
        best_gain = np.where(upd, gain, best_gain)
        for kk in list(choice):
            choice[kk] = choice[kk] & ~upd
        choice[key], coef[key] = upd, mm
    for key in fold_maps:
        sel = choice[key]
        frac_px = float(sel.mean())
        if frac_px < 5e-3:                                               # a few stray pixels: not a fold
            continue
        w = np.clip(gaussian_filter(sel.astype(float), 2.0) * 1.5, 0, 1) * sig
        mm = coef[key]
        m1, m2 = mm[..., 0], mm[..., 1]
        big1 = np.abs(m1) >= np.abs(m2)
        den = np.where(big1, m1, m2)
        den = np.where(np.abs(den) > 0, den, 1)
        Ceff = (m1[None] * C + m2[None] * fold_maps[key]) / den[None]
        out = np.where(w[None] > 0, (1 - w)[None] * out + w[None] * Ceff, out)
        rep["regions"][f"axis{key[0]}{'+' if key[1] > 0 else '-'}FOV"] = frac_px
        rep["corrected"] = True
    if rep["corrected"]:
        log("fold-over detected (object larger than the FOV): effective maps in " +
            ", ".join(f"{k} {100 * v:.1f}% of the pixels" for k, v in rep["regions"].items()))
    return out.astype(np.complex64), rep


def _unwind_series(C, sup):
    """(T,N,X,Y) maps: remove the gauge poles of the time-mean maps with ONE common factor for every frame."""
    from .recon import remove_gauge_poles
    _, n, fac = remove_gauge_poles(C.mean(0), sup)
    return C * fac[None, None].astype(C.dtype) if n else C


FINISH_CHUNK_ELEMENTS = 20_000_000          # dynamic maps above this size: phase reference in chunks of frames


def _finish(C, sup, layout, phase_ref, normalise, device=None):
    from .recon import phase_reference, virtual_body_coil
    if layout == "dynamic" and C.size > FINISH_CHUNK_ELEMENTS and C.shape[0] > 8:
        # long series (OCMR us_0001: 48 frames x 34 coils OOM'd 6 GB): the virtual body coil weights from <= 8
        # evenly spaced frames (one weight vector for the series, as before), then frames in chunks of 8
        T = C.shape[0]
        d = None
        if phase_ref in ("inati", "inati-eig"):
            idx = np.unique(np.linspace(0, T - 1, 8).round().astype(int))
            Cs = torch.from_numpy(np.ascontiguousarray(C[idx])).to(torch.complex64)
            d = virtual_body_coil(Cs, torch.from_numpy(sup)[None].expand(len(idx), -1, -1).float(),
                                  robust=(phase_ref == "inati"), device=device).cpu()
            del Cs
        out = np.empty(C.shape, np.complex64)
        modes = set()
        for t0 in range(0, T, 8):
            Ct = torch.from_numpy(np.ascontiguousarray(C[t0:t0 + 8])).to(torch.complex64)
            S = torch.from_numpy(sup)[None].expand(Ct.shape[0], -1, -1)
            if phase_ref not in ("model", "none"):
                Ct, _, inf, _ = phase_reference(Ct, phase_ref, S, None, d, unwind=False)
                modes.add(str(inf.get("phase_ref")))
            if normalise == "rss":
                Ct = Ct / (Ct.abs() ** 2).sum(1, keepdim=True).sqrt().clamp_min(1e-12)
            elif normalise != "none":
                raise ValueError("normalise must be 'rss' or 'none'")
            out[t0:t0 + 8] = Ct.numpy()
        if len(modes) > 1:
            warnings.warn(f"phase reference differed between frame chunks ({sorted(modes)})")
        if "model" in modes and phase_ref == "inati":
            out = _unwind_series(out, sup)
        return out
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
            d = virtual_body_coil(Cs, S.float(), robust=(phase_ref == "inati"), device=device)
        # 2-D: gauge poles of a model-phase fallback removed in place; dynamic: one factor for all frames (from
        # the time-mean maps, so the gauge does not jump in time); 3-D: per-slice removal would be
        # inconsistent between slices, so not done
        Cs, _, inf, _ = phase_reference(Cs, phase_ref, S, None, d, unwind=(layout == "2d"))
        if layout == "dynamic" and inf.get("phase_ref") == "model" and phase_ref == "inati":
            Cs = torch.from_numpy(_unwind_series(Cs.numpy(), sup))
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


def _cfft(x, axes, inverse=False):
    f = np.fft.ifftn if inverse else np.fft.fftn
    return np.fft.fftshift(f(np.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)


def _joint_image(m, prob, C_raw, C_out, layout):
    """The image of the joint fit lives on the calibration grid (prob.calib points spanning the FOV)
    and is paired with the raw model maps on that grid. Interpolate it to the full matrix by k-space
    zero-padding (it is band-limited to the calibration region), restore the data scale, and move it
    to the convention of the returned maps: C_out = C_raw * g per pixel  =>  image_out = image / g."""
    if m is None:
        return None
    m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
    if layout == "3d":
        m = m.reshape(m.shape[-3:])[None]                       # (1, gd, gh, gw)
        full = C_out.shape[1:]
    else:
        m = m[:, 0] if m.ndim == 4 else m                       # (S|T, h, w): primary set
        full = C_out.shape[-2:]
    ax = tuple(range(1, m.ndim))
    k = _cfft(m, ax)
    # centre index n//2 of the small grid must land on F//2 of the large one
    pad = [(0, 0)] + [(F // 2 - g // 2, F - g - (F // 2 - g // 2)) for g, F in zip(m.shape[1:], full)]
    f = _cfft(np.pad(k, pad), ax, inverse=True) * prob.scale
    if layout == "2d":
        f, Cr, Co = f[0], C_raw, C_out
        g = (Cr.conj() * Co).sum(0) / np.maximum((np.abs(Cr) ** 2).sum(0), 1e-30)
    elif layout == "3d":
        f = f[0]
        g = (C_raw.conj() * C_out).sum(0) / np.maximum((np.abs(C_raw) ** 2).sum(0), 1e-30)
    else:                                        # frame by frame: bounded memory for long series
        g = np.stack([(C_raw[t].conj() * C_out[t]).sum(0) / np.maximum((np.abs(C_raw[t]) ** 2).sum(0), 1e-30)
                      for t in range(C_out.shape[0])])
    return (f / np.where(np.abs(g) > 0, g, 1)).astype(np.complex64)


def acquired_extent(m):
    """Box spanned by the acquired k-space along each axis (first..last index with any sample): the
    region a reconstruction can estimate. Zero padding, partial Fourier and a cropped readout lie outside."""
    box = np.ones(m.shape, bool)
    for a in range(m.ndim):
        prof = m.any(axis=tuple(i for i in range(m.ndim) if i != a))
        idx = np.nonzero(prof)[0]
        keep = np.zeros(m.shape[a], bool)
        if idx.size:
            keep[idx[0]:idx[-1] + 1] = True
        box &= keep.reshape([-1 if i == a else 1 for i in range(m.ndim)])
    return box


def _sense_torch(ksp, C, m, box, lam, iters, device):
    """CG-SENSE on ``device`` (torch FFTs), same system and stopping rule as the NumPy path (residual
    checked every 10 iterations: one host sync each). Runs in complex64; the NumPy path in complex128
    (np.fft upcasts); the two agree to ~1e-5."""
    dev = torch.device(device)
    y = torch.from_numpy(np.ascontiguousarray(ksp)).to(dev)
    Ct = torch.from_numpy(np.ascontiguousarray(C)).to(dev)
    M = torch.from_numpy(np.ascontiguousarray(m)).to(dev)
    B = None if box.all() else torch.from_numpy(np.ascontiguousarray(box)).to(dev)
    ax = tuple(range(1, y.ndim)); axi = tuple(range(y.ndim - 1))
    F = lambda x, a: torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=a), dim=a, norm="ortho"), dim=a)
    Fi = lambda x, a: torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=a), dim=a, norm="ortho"), dim=a)
    band = (lambda x: x) if B is None else (lambda x: Fi(F(x, axi) * B, axi))
    def AhA(x):
        return band((Ct.conj() * Fi(M * F(Ct * band(x)[None], ax), ax)).sum(0)) + lam * x
    b = band((Ct.conj() * Fi(M * y, ax)).sum(0))
    x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rs = (r.conj() * r).real.sum(); b0 = rs
    for it in range(iters):
        Ap = AhA(p)
        al = rs / (p.conj() * Ap).real.sum()
        x = x + al * p; r = r - al * Ap
        rn = (r.conj() * r).real.sum()
        if it % 10 == 9 and float(rn) < 1e-10 * float(b0):
            break
        p = r + (rn / rs) * p; rs = rn
    return band(x).cpu().numpy().astype(np.complex64)


def sense_image(ksp, maps, layout="2d", mask=None, noise=None, lam=1e-3, iters=60, coil_axis=0, extrapolate=False,
                device="cpu", lam_scale=1.0, report=None, lam_options=None, prior=None, prior_weight="auto",
                prior_options=None):
    """Full-resolution image from k-space and maps: argmin ||M F (C x) - y||^2 + lam' ||x||^2 with
    noise pre-whitening (if ``noise`` is given) applied to data and maps alike (``lam`` relative to the
    mean of sum_j |C_j|^2). Dynamic: frame by frame.

    Only k-space inside the acquired extent (``acquired_extent``) is estimated: missing samples inside
    it (undersampling) are unfolded by CG-SENSE; zero-padded edges, partial-Fourier gaps and cropped
    readout are left empty, as in a zero-filled reconstruction. Recovering those from coil
    sensitivities alone is ill-conditioned and only amplifies noise into high spatial frequencies.
    If everything inside the extent is acquired, this is the closed-form SENSE-1 combination of the
    zero-filled data. ``extrapolate=True`` restores CG over the whole k-space grid.
    ``lam`` may be 'auto' / 'auto-gcv' / 'auto-map' / 'auto-discrepancy' (bspinn.senselam, times
    ``lam_scale``); the chosen value and the rules' diagnostics go into the dict ``report``.

    ``prior`` replaces the Tikhonov term (2-D): 'tv' (total variation), 'wavelet' (orthonormal db4,
    coarsest band free), 'tv+wavelet' (bspinn.prior; ADMM), with ``prior_weight`` a relative number or
    'auto' (k-space cross-validation on withheld phase-encode lines, as for lam; the choice and the CV curve
    go into ``report``); or 'net:<weights.pt>' for a network trained by scripts/train_recon.py or
    scripts/zs_ssl.py (bspinn.unrolled). ``prior_options``: ratio (wavelet / TV weight), iters, cg_iters,
    rho, levels, wave, grid, cv_options (holdout, folds, calib, seed); for 'net:' architecture overrides."""
    ksp = np.moveaxis(np.asarray(ksp), coil_axis, 0).astype(np.complex64)
    C = np.moveaxis(np.asarray(maps), coil_axis, 0).astype(np.complex64)
    if noise is not None:
        n = np.asarray(noise).reshape(-1, ksp.shape[0])
        L = np.linalg.inv(np.linalg.cholesky((n.conj().T @ n) / n.shape[0]))
        ksp = np.einsum("ij,j...->i...", L, ksp)
        C = np.einsum("ij,j...->i...", L, C)
    if layout == "dynamic":
        ms = [None if mask is None else mask[..., t] for t in range(ksp.shape[-1])]
        reps = [{} for _ in range(ksp.shape[-1])]
        out = np.stack([sense_image(ksp[..., t], C[..., t], "2d", ms[t], None, lam, iters, extrapolate=extrapolate,
                                    device=device, lam_scale=lam_scale, report=reps[t], lam_options=lam_options,
                                    prior=prior, prior_weight=prior_weight, prior_options=prior_options)
                        for t in range(ksp.shape[-1])], -1)
        if report is not None:
            report["frames"] = reps
        return out
    m = (np.abs(ksp).sum(0) > 0) if mask is None else np.asarray(mask, bool)
    ax = tuple(range(1, ksp.ndim))
    den = (np.abs(C) ** 2).sum(0)
    box = np.ones(m.shape, bool) if extrapolate else acquired_extent(m)
    if (m | ~box).all():
        return ((C.conj() * _cfft(ksp * m[None], ax, True)).sum(0) / np.maximum(den, 1e-12)).astype(np.complex64)
    if prior is not None:
        if ksp.ndim != 3:
            raise ValueError("prior= is implemented for 2-D images")
        po = dict(prior_options or {})
        if str(prior).startswith("net:"):
            from .unrolled import load_network, reconstruct
            net = load_network(str(prior)[4:], device, **po)
            x = reconstruct(net, ksp, C, m, device)
            if report is not None:
                report.update(method="net", weights=str(prior)[4:])
        else:
            from .prior import prior_image
            x = prior_image(ksp, C, m, prior, prior_weight, device=device, report=report, **po)
        if not box.all():                    # as the CG path: nothing estimated outside the acquired extent
            x = _cfft(_cfft(x, (0, 1)) * box, (0, 1), True)
        return x.astype(np.complex64)
    if isinstance(lam, str):
        from .senselam import choose_sense_lambda
        method = {"auto": "cv+map"}.get(lam, lam.replace("auto-", ""))
        lam, rep = choose_sense_lambda(ksp * m[None], C, m, noise_var=1.0 if noise is not None else None,
                                       method=method, scale=lam_scale, device=device, iters=iters, **(lam_options or {}))
        if report is not None:
            report.update(rep, value=lam)
        if lam is None:
            lam = 0.0
    elif report is not None:
        report.update(method="manual", value=float(lam))
    lam = lam * float(den.mean())
    if str(device) != "cpu":
        return _sense_torch(ksp, C, m, box, lam, iters, device)
    band = (lambda x: x) if box.all() else (lambda x: _cfft(_cfft(x, tuple(range(x.ndim))) * box, tuple(range(x.ndim)), True))

    def AhA(x):
        return band((C.conj() * _cfft(m[None] * _cfft(C * band(x)[None], ax), ax, True)).sum(0)) + lam * x
    b = band((C.conj() * _cfft(m[None] * ksp, ax, True)).sum(0))
    x = np.zeros_like(b); r = b.copy(); p = r.copy(); rs = np.vdot(r, r).real; b0 = rs
    for it in range(iters):                               # same stopping rule as the device path
        Ap = AhA(p)
        al = rs / np.vdot(p, Ap).real
        x += al * p; r -= al * Ap
        rn = np.vdot(r, r).real
        if it % 10 == 9 and rn < 1e-10 * b0:
            break
        p = r + (rn / rs) * p; rs = rn
    return band(x).astype(np.complex64)
