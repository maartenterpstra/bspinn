"""Command line interface.

Examples
--------
Real data (BART .cfl, fastMRI .h5 or .npy/.npz), any number of coils:
    python -m bspinn.cli fit ksp.cfl --out runs/brain --plane axial --pixel-spacing 1 1

Synthetic validation with known Biot–Savart ground truth:
    python -m bspinn.cli synthetic --coils 12 --out runs/synth
"""
from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np
import torch

from .data import load, load_maps, load_noncart, make_folded_synthetic, make_synthetic
from .fit import FitConfig, fit
from .report import evaluate


def _add_cfg_args(p):
    for f in dataclasses.fields(FitConfig):
        flag = "--" + f.name.replace("_", "-")
        p.add_argument(flag, type=type(f.default), default=f.default)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bspinn", description="Biot–Savart constrained coil sensitivity estimation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit", help="fit on a multi-coil k-space file")
    pf.add_argument("path")
    pf.add_argument("--out", required=True)
    pf.add_argument("--plane", default="axial", choices=["axial", "coronal", "sagittal", "coronal_t"])
    pf.add_argument("--pixel-spacing", type=float, nargs=2, default=(1.0, 1.0), metavar=("ROW_MM", "COL_MM"))
    pf.add_argument("--slice-spacing", type=float, default=5.0)
    pf.add_argument("--slices", type=int, nargs="*", default=None, help="slice indices to fit jointly")
    pf.add_argument("--noise", default=None, help="noise-only samples (.cfl/.npy) for pre-whitening")
    pf.add_argument("--traj", default=None, help="non-Cartesian trajectory (BART .cfl or .npy); k-space in BART layout")
    pf.add_argument("--fov", type=float, default=256.0, help="FOV in mm (non-Cartesian)")
    pf.add_argument("--drop-first", type=int, default=0, help="discard first N spokes (non-steady state)")
    pf.add_argument("--recon-size", type=int, default=128, help="matrix of the non-Cartesian evaluation recon")
    pf.add_argument("--ref-maps", nargs="*", default=[], metavar="NAME=PATH",
                    help="extra sensitivity maps (BART .cfl or .npy) to compare against, e.g. espirit=maps")
    _add_cfg_args(pf)

    ps = sub.add_parser("synthetic", help="validate on a synthetic Biot–Savart phantom")
    ps.add_argument("--out", required=True)
    ps.add_argument("--coils", type=int, default=8)
    ps.add_argument("--size", type=int, default=128)
    ps.add_argument("--n-slices", type=int, default=1)
    ps.add_argument("--noise-level", type=float, default=0.01)
    ps.add_argument("--nonharmonic", type=float, default=0.0)
    ps.add_argument("--accel", type=float, default=0.0)
    ps.add_argument("--scenario", default="standard", choices=["standard", "folded"])
    ps.add_argument("--noise-corr", type=float, default=0.0, help="inter-channel noise correlation (adds a noise prescan)")
    _add_cfg_args(ps)

    pd = sub.add_parser("fit-dataset", help="fit many fastMRI-style volumes (independent or shared trunk)")
    pd.add_argument("files", nargs="+", help="fastMRI .h5 files or glob patterns")
    pd.add_argument("--out", required=True)
    pd.add_argument("--mode", default="independent", choices=["independent", "shared"])
    pd.add_argument("--slices-per-volume", type=int, default=0, help="evenly spaced slices per volume (0 = all)")
    pd.add_argument("--steps", type=int, default=3000, help="shared mode: training steps")
    pd.add_argument("--holdout", type=int, default=0, help="shared mode: volumes held out for the prior test")
    pd.add_argument("--shard", default="0/1", help="independent mode: i/n to split files over workers")
    _add_cfg_args(pd)

    py = sub.add_parser("dynamic", help="time-resolved, low-rank Biot–Savart coil maps")
    py.add_argument("path", nargs="?", default=None, help="BART k-space (1,samples,spokes,coils); omit with --synthetic")
    py.add_argument("--out", required=True)
    py.add_argument("--traj", help="BART trajectory .cfl (3,samples,spokes)")
    py.add_argument("--spokes-per-frame", type=int, default=21)
    py.add_argument("--fov", type=float, default=256.0)
    py.add_argument("--frames", type=int, nargs="*", default=None)
    py.add_argument("--noise", default=None)
    py.add_argument("--plane", default="axial")
    py.add_argument("--synthetic", action="store_true", help="moving-coil synthetic with ground truth")
    py.add_argument("--rank", type=int, default=2)
    py.add_argument("--iters-dyn", type=int, default=1500)
    py.add_argument("--lambda-smooth", type=float, default=1e-3)
    py.add_argument("--map-size", type=int, default=0, help="matrix of the exported maps (0 = data matrix)")
    py.add_argument("--dynamic-part", default="both", choices=["both", "wires", "fields"],
                    help="what may change over time: conductor motion, harmonic fields, or both")
    py.add_argument("--max-disp-mm", type=float, default=0.0, help="bound on coil conductor motion (0 = none)")
    _add_cfg_args(py)

    p3 = sub.add_parser("fit3d", help="3-D encoded data (or --synthetic): volumetric Biot–Savart fit")
    p3.add_argument("path", nargs="?", default=None, help="BART .cfl (X,Y,Z,COIL) or .npz with 'ksp' (N,D,H,W)")
    p3.add_argument("--out", required=True)
    p3.add_argument("--voxel", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("D_MM", "H_MM", "W_MM"))
    p3.add_argument("--axes", default="zyx", help="physical axis (B0 = z) of array axes D,H,W, or 'auto'")
    p3.add_argument("--noise", default=None, help="noise-only samples for pre-whitening")
    p3.add_argument("--synthetic", action="store_true", help="Biot–Savart ground-truth volume")
    p3.add_argument("--size", type=int, default=48)
    p3.add_argument("--coils", type=int, default=12)
    p3.add_argument("--noise-level", type=float, default=0.01)
    p3.add_argument("--mismatch", type=float, default=0.0)
    _add_cfg_args(p3)

    pm = sub.add_parser("maps", help="simplest path: k-space array file -> maps file (2-D, 3-D or dynamic)")
    pm.add_argument("path", help=".npy (complex array), .npz (key 'ksp', optional 'mask', 'noise') or BART .cfl")
    pm.add_argument("--layout", required=True, choices=["2d", "3d", "dynamic"],
                    help="2d: (coils,x,y); 3d: (coils,x,y,z); dynamic: (coils,x,y,time)")
    pm.add_argument("--out", required=True, help="output .npy (maps, same shape/axis order as the input)")
    pm.add_argument("--coil-axis", type=int, default=0, help="axis of the coils in the input (BART .cfl: last)")
    pm.add_argument("--spacing", type=float, nargs="+", default=None, help="pixel/voxel size in mm per spatial axis")
    pm.add_argument("--plane", default="auto", choices=["auto", "axial", "coronal", "sagittal", "coronal_t"])
    pm.add_argument("--axes", default="auto", help="3-D: physical axes of x,y,z array axes (e.g. xyz) or auto")
    pm.add_argument("--calib", type=int, nargs="+", default=None, help="ACS size (one value or one per axis)")
    pm.add_argument("--noise", default=None, help="noise-only samples (.npy, (n, coils)) for pre-whitening")
    pm.add_argument("--physics", default="laplace", choices=["laplace", "helmholtz"])
    pm.add_argument("--b0", type=float, default=3.0, help="field strength in T (helmholtz)")
    pm.add_argument("--model", default="pinn", choices=["pinn", "wires"])
    pm.add_argument("--rank", type=int, default=2, help="dynamic: temporal modes (0 = static maps)")
    pm.add_argument("--quality", default="default", choices=["fast", "default"])
    pm.add_argument("--normalise", default="rss", choices=["rss", "none"])
    pm.add_argument("--phase-ref", default="inati", choices=["inati", "model", "coil"])
    pm.add_argument("--device", default="cpu", help="cpu | cuda | cuda:1 ...")
    pm.add_argument("--compile", nargs="?", const=True, default=False, metavar="MODE",
                    help="torch.compile each stage's iteration; MODE: default | reduce-overhead (CUDA graphs) | max-autotune")
    pm.add_argument("--no-cuda-graph", dest="cuda_graph", action="store_false",
                    help="GPU: run the training iterations eagerly instead of replaying a captured CUDA graph")
    pm.add_argument("--tf32", action="store_true", help="TF32 matmuls on Ampere+ GPUs")
    pm.add_argument("--optimizer", default="adam", choices=["adam", "lbfgs", "adam+lbfgs"],
                    help="adam (default) | lbfgs: full-batch L-BFGS in every stage | adam+lbfgs: Adam, then an L-BFGS polish")
    pm.add_argument("--threads", type=int, default=0)
    pm.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="any FitConfig field, e.g. iters_pinn=1000")
    pm.add_argument("--solve-image", action="store_true",
                    help="solve maps and the full-resolution image jointly on all acquired k-space (2-D); writes --image")
    pm.add_argument("--undersample", type=float, nargs="+", default=None, metavar="R",
                    help="retrospective Cartesian undersampling: R (or R1 R2 in 3-D); calibration block = --calib")
    pm.add_argument("--undersample-mask", default=None, help="use this bool .npy mask instead of generating one")
    _add_mask_args(pm, with_R=False)
    pm.add_argument("--mask-maps", nargs="?", const=True, default=False, metavar="SUPPORT.npy",
                    help="zero the maps (and images) outside the object support: estimated, or your own bool .npy")
    pm.add_argument("--support-threshold", type=float, default=0.1, help="estimated support: fraction of the 99th pct")
    pm.add_argument("--support-dilate", type=int, default=2, help="estimated support: dilation in pixels")
    pm.add_argument("--image", "--combined", dest="image", default=None,
                    help="also write the full-resolution image (SENSE with the estimated maps; CG if undersampled) "
                         "here, and the joint-fit (calibration-resolution) image next to it as *_joint.npy")

    pk = sub.add_parser("mask", help="write a Cartesian undersampling mask (.npy) with a calibration block")
    pk.add_argument("--shape", type=int, nargs="+", required=True, help="spatial shape (2-D/3-D), or (x, y, t) for dynamic")
    pk.add_argument("--out", required=True)
    pk.add_argument("--layout", default=None, choices=["2d", "3d", "dynamic"])
    _add_mask_args(pk)

    a = ap.parse_args(argv)
    if a.cmd == "mask":
        from .masks import cartesian_mask, effective_acceleration
        m = cartesian_mask(tuple(a.shape), layout=a.layout, **_mask_opts(a))
        np.save(a.out, m)
        print(f"wrote {a.out} {m.shape}, R_effective {effective_acceleration(m):.2f}")
        return
    if a.cmd == "maps":
        _run_maps(a)
        return
    if a.cmd == "fit3d":
        from .volume import evaluate3d, fit3d, load_volume, make_synthetic3d
        cfg = FitConfig(**{f.name: getattr(a, f.name) for f in dataclasses.fields(FitConfig)})
        gt = obj = None
        if a.synthetic:
            data, gt, img = make_synthetic3d(a.coils, (a.size,) * 3, noise=a.noise_level, mismatch=a.mismatch, seed=cfg.seed)
            obj = img > 0
        else:
            data = load_volume(a.path, tuple(a.voxel), "zyx" if a.axes == "auto" else a.axes, noise_path=a.noise)
        model, prob, hist, extra = fit3d(data, cfg, orientation="auto" if a.axes == "auto" else "given")
        evaluate3d(model, prob, extra, a.out, gt=gt, obj=obj)
        return
    if a.cmd == "dynamic":
        from .dynamic import DynConfig, evaluate_dynamic, fit_dynamic, load_dynamic_radial, make_dynamic_synthetic
        cfg = FitConfig(**{f.name: getattr(a, f.name) for f in dataclasses.fields(FitConfig)})
        dcfg = DynConfig(rank=a.rank, iters_dyn=a.iters_dyn, lambda_smooth=a.lambda_smooth, parts=a.dynamic_part,
                         max_disp_mm=a.max_disp_mm)
        gt, info = None, {}
        if a.synthetic:
            data, gt, motion, sup = make_dynamic_synthetic(seed=cfg.seed)
            info = {"true_motion": motion, "object_support": sup}
        else:
            data = load_dynamic_radial(a.path, a.traj, a.spokes_per_frame, a.fov, a.frames, a.noise, a.plane)
        dyn, prob, hist, extra = fit_dynamic(data, cfg, dcfg)
        extra.update(info)
        shape = (a.map_size, a.map_size) if a.map_size else None
        evaluate_dynamic(dyn, prob, extra, a.out, gt=gt, shape=shape)
        torch.save({"state_dict": dyn.state_dict(), "rank": dyn.K}, f"{a.out}/model.pt")
        return
    if a.cmd == "fit-dataset":
        from .dataset import expand, fit_independent, fit_shared
        cfg = FitConfig(**{f.name: getattr(a, f.name) for f in dataclasses.fields(FitConfig)})
        files = expand(a.files)
        if a.mode == "independent":
            i, n = (int(x) for x in a.shard.split("/"))
            fit_independent(files, cfg, a.out, a.slices_per_volume, (i, n))
        else:
            fit_shared(files, cfg, a.out, a.slices_per_volume, a.steps, a.holdout)
        return
    cfg = FitConfig(**{f.name: getattr(a, f.name) for f in dataclasses.fields(FitConfig)})
    gt = None
    if a.cmd == "fit":
        if a.traj:
            data = load_noncart(a.path, a.traj, a.fov, plane=a.plane, noise_path=a.noise, drop_first=a.drop_first)
        else:
            data = load(a.path, plane=a.plane, pixel_spacing=a.pixel_spacing, slice_spacing=a.slice_spacing,
                        noise_path=a.noise, slices=a.slices)
    else:
        if a.scenario == "folded":
            data, gt, _ = make_folded_synthetic(a.coils, (a.size, a.size), noise=a.noise_level, seed=cfg.seed)
        else:
            data, gt, _ = make_synthetic(a.coils, (a.size, a.size), a.n_slices, noise=a.noise_level,
                                         nonharmonic=a.nonharmonic, accel=a.accel or None, seed=cfg.seed,
                                         noise_corr=a.noise_corr)
    refs = {}
    for item in getattr(a, "ref_maps", []):
        name, path = item.split("=", 1)
        refs[name] = load_maps(path)
        print(f"reference maps '{name}': {refs[name].shape}")
    model, prob, hist, extra = fit(data, cfg)
    evaluate(model, prob, extra, out_dir=a.out, cfg=cfg, history=hist, gt=gt,
             recon_size=getattr(a, "recon_size", 128), seed=cfg.seed, ref_maps=refs)


def _add_mask_args(p, with_R=True):
    if with_R:
        p.add_argument("--R", type=float, nargs="+", default=[4.0], help="acceleration (R1 R2 for 3-D lattices)")
    p.add_argument("--pattern", default="uniform", choices=["uniform", "random", "vd", "caipi", "poisson"])
    p.add_argument("--acs", type=int, nargs="+", default=None, help="calibration block per PE axis (maps: default --calib)")
    p.add_argument("--partial-fourier", type=float, default=1.0)
    p.add_argument("--mask-seed", type=int, default=0)


def _mask_opts(a, R=None):
    o = {"pattern": a.pattern, "partial_fourier": a.partial_fourier, "seed": a.mask_seed}
    R = R if R is not None else getattr(a, "R", None)
    if R is not None:
        o["R"] = R[0] if len(R) == 1 else tuple(R)
    if a.acs is not None:
        o["acs"] = a.acs[0] if len(a.acs) == 1 else tuple(a.acs)
    if getattr(a, "acs", None) is None and getattr(a, "R", None) is not None and "acs" not in o:
        o["acs"] = 24
    return o


def _run_maps(a):
    import os
    from .api import estimate_maps
    mask = noise = None
    if a.path.endswith(".npy"):
        k = np.load(a.path)
    elif a.path.endswith(".npz"):
        z = np.load(a.path)
        k = z["ksp"]
        mask = z["mask"] if "mask" in z else None
        noise = z["noise"] if "noise" in z else None
    else:
        from .data import _read_cfl
        k = np.squeeze(_read_cfl(a.path))
    if a.noise:
        noise = np.load(a.noise)
    if mask is not None:
        mask = np.moveaxis(np.asarray(mask), a.coil_axis, 0)[0] if mask.ndim == k.ndim else mask
    over = {}
    defaults = {f.name: f.default for f in dataclasses.fields(FitConfig)}
    for kv in a.set:
        key, val = kv.split("=", 1)
        if key not in defaults:
            raise SystemExit(f"unknown option {key}")
        t = type(defaults[key])
        if key == "compile" and val.lower() not in ("0", "1", "true", "false", "yes", "no"):
            over[key] = val                                       # a torch.compile mode
            continue
        over[key] = (val.lower() in ("1", "true", "yes")) if t is bool else t(val)
    over.setdefault("compile", a.compile)
    over.setdefault("tf32", a.tf32)
    over.setdefault("cuda_graph", a.cuda_graph)
    over.setdefault("optimizer", a.optimizer)
    calib = None if a.calib is None else (a.calib[0] if len(a.calib) == 1 else tuple(a.calib))
    under = None
    if a.undersample_mask:
        under = np.load(a.undersample_mask).astype(bool)
    elif a.undersample:
        under = _mask_opts(a, R=a.undersample)
    res = estimate_maps(k, a.layout, return_image=bool(a.image) or a.solve_image, solve_image=a.solve_image, undersample=under, coil_axis=a.coil_axis, mask=mask, spacing=a.spacing, plane=a.plane,
                               axes=a.axes, calib=calib, noise=noise, physics=a.physics, b0_tesla=a.b0,
                               model=a.model, rank=a.rank, quality=a.quality, normalise=a.normalise,
                               phase_ref=a.phase_ref, device=a.device, threads=a.threads,
                               mask_maps=np.load(a.mask_maps) if isinstance(a.mask_maps, str) else a.mask_maps,
                               support_threshold=a.support_threshold, support_dilate=a.support_dilate, **over)
    maps, info = (res[0], res[2]) if (a.image or a.solve_image) else res
    if a.solve_image and not a.image:
        a.image = a.out.replace(".npy", "") + "_image.npy"
    maps = np.moveaxis(maps, 0, a.coil_axis)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, maps)
    meta = {k_: v for k_, v in info.items()
            if k_ not in ("support", "model", "problem", "temporal_modes", "config", "image_joint", "mask")}
    if "k2" in meta:
        meta["k2"] = [meta["k2"].real, meta["k2"].imag]
    if "temporal_modes" in info:
        np.save(a.out.replace(".npy", "") + "_modes.npy", info["temporal_modes"])
    np.save(a.out.replace(".npy", "") + "_support.npy", info["support"])
    if "mask" in info:
        np.save(a.out.replace(".npy", "") + "_mask.npy", info["mask"])
    with open(a.out.replace(".npy", "") + "_info.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
    if a.image:
        np.save(a.image, res[1])
        if info.get("image_joint") is not None:
            np.save(a.image.replace(".npy", "") + "_joint.npy", info["image_joint"])
    print(f"wrote {a.out} {maps.shape}")


if __name__ == "__main__":
    main()
