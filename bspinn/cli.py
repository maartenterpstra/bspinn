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
    pf.add_argument("--plane", default="axial", choices=["axial", "coronal", "sagittal"])
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

    a = ap.parse_args(argv)
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


if __name__ == "__main__":
    main()
