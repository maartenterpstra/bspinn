"""Training over many volumes (fastMRI-scale).

mode 'independent'
    Fit every volume separately (all selected slices jointly = one 3-D field per coil, slice
    minibatches), write full-resolution maps + report per volume. Embarrassingly parallel:
    shard the file list over processes/GPUs with --shard i/n. Resumable (skips finished volumes).

mode 'shared'  (amortised)
    One PINN trunk shared by all volumes; each volume keeps its own wire model, per-coil latent
    codes and receive phases (auto-decoder). Coil counts may differ between volumes. The trunk
    is saved (``trunk.pt``) and can initialise new fits with ``--pretrained``. With ``--holdout k``
    the last k volumes are excluded from training and fitted twice with the same small budget
    (pretrained vs from scratch) to measure whether the learned prior helps.
"""
from __future__ import annotations

import copy
import glob
import json
import os
import time

import numpy as np
import torch

from .fastmri import load_fastmri
from .fit import FitConfig, Problem, fit, fit_pinn, fit_wires, make_pinn
from .models import HarmonicPINN, make_trunk
from .report import evaluate


def _load(path, cfg, slices_per_volume, slice_spacing=None):
    import h5py
    with h5py.File(path, "r") as f:
        S = f["kspace"].shape[0]
    sl = None
    if slices_per_volume and S > slices_per_volume:      # evenly spread slices through the volume
        sl = np.unique(np.linspace(0, S - 1, slices_per_volume).round().astype(int)).tolist()
    return load_fastmri(path, calib=cfg.calib, slices=sl, slice_spacing=slice_spacing)


def _gt(path, data):
    p = path + ".gt.npy"
    if not os.path.exists(p):
        return None
    gt = np.load(p)
    return gt[data.meta["slices"]]


def fit_independent(files, cfg: FitConfig, out_dir, slices_per_volume=0, shard=(0, 1), log=print):
    cfg.validate("fit")
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for i, path in enumerate(files):
        if i % shard[1] != shard[0]:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        vdir = os.path.join(out_dir, name)
        if os.path.exists(os.path.join(vdir, "report.json")):
            log(f"[{name}] done already, skipping"); continue
        t0 = time.time()
        data = _load(path, cfg, slices_per_volume)
        log(f"[{name}] {data.kspace.shape[0]} slices, {data.n_coils} coils, plane {data.geom.plane}")
        model, prob, hist, extra = fit(data, copy.deepcopy(cfg), log=lambda *a: None)
        rep, _, _ = evaluate(model, prob, extra, out_dir=vdir, cfg=cfg, history=hist, gt=_gt(path, data), log=lambda *a: None)
        rep["seconds"] = time.time() - t0
        summary[name] = rep
        log(f"[{name}] calib loss {rep['calib_data_loss']:.4e}  div {rep['physics_rel_rms']['div']:.3f}  "
            f"{rep['seconds']:.0f}s" + (f"  GT abs err {rep['vs_ground_truth']['biot_savart_model']['absolute_err']:.3f}"
                                        if "vs_ground_truth" in rep else ""))
    with open(os.path.join(out_dir, f"summary_shard{shard[0]}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def _volume(path, cfg, slices_per_volume, log):
    data = _load(path, cfg, slices_per_volume)
    data.prewhiten()
    prob = Problem(data, cfg).to(cfg.device)
    h = cfg.handedness
    if h == 0:
        sc = {hh: fit_wires(prob, cfg, hh, cfg.iters_auto, log=lambda *a: None)[3] for hh in (1, -1)}
        h = min(sc, key=sc.get)
    wires, _, _, loss = fit_wires(prob, cfg, h, cfg.iters_wires, log=lambda *a: None)
    log(f"  wires {os.path.basename(path)}: {prob.N} coils, {prob.S} slices, h={h:+d}, data {loss:.3e}")
    return data, prob, wires, h


def fit_shared(files, cfg: FitConfig, out_dir, slices_per_volume=0, steps=3000, holdout=0, log=print):
    cfg.validate("shared")
    os.makedirs(out_dir, exist_ok=True)
    train_files = files[:len(files) - holdout] if holdout else files
    test_files = files[len(files) - holdout:] if holdout else []
    torch.manual_seed(cfg.seed)
    K = 1 if cfg.field_mode == "real" else 2
    trunk = make_trunk(3 + cfg.latent, cfg.width, cfg.depth, K).to(cfg.device)
    torch.nn.init.zeros_(trunk[-1].weight)
    vols = []
    log(f"fitting wire models for {len(train_files)} training volumes")
    for p in train_files:
        data, prob, wires, h = _volume(p, cfg, slices_per_volume, log)
        pinn = HarmonicPINN(prob.N_model, cfg.field_mode, cfg.latent, cfg.width, cfg.depth, h, base=wires, net=trunk).to(cfg.device)
        vols.append({"path": p, "data": data, "prob": prob, "pinn": pinn, "h": h})
    own = [q for v in vols for n, q in v["pinn"].named_parameters() if not n.startswith("net.")]
    from . import accel
    opt = accel.adam([{"params": trunk.parameters(), "lr": cfg.lr_pinn},
                      {"params": own, "lr": cfg.lr_pinn}], cfg.device)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.1 * cfg.lr_pinn)
    gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed + 7)       # collocation sampled on-device
    slice_rng = np.random.default_rng(cfg.seed + 8)                            # slice choice on the host
    hist = []
    for it in range(steps):
        v = vols[it % len(vols)]
        prob, pinn = v["prob"], v["pinn"]
        sl = None
        if cfg.slice_batch and prob.S > cfg.slice_batch:
            sl = sorted(slice_rng.permutation(prob.S)[:cfg.slice_batch].tolist())
        opt.zero_grad()
        C = pinn.sensitivity(prob.slice_coords(sl))
        d = prob.data_loss(C, None, sl)
        phys = pinn.physics_loss(prob.collocation(cfg.n_colloc, gen))
        loss = d + cfg.lambda_phys * phys + cfg.lambda_norm * prob.norm_loss(C, sl)
        loss.backward(); opt.step(); sched.step()
        if it % cfg.log_every == 0 or it == steps - 1:
            hist.append({"it": it, "vol": os.path.basename(v["path"]), "data": d.item(), "phys": phys.item()})
            log(f"[shared] it {it:5d}  {os.path.basename(v['path'])[:24]:24s} data {d.item():.4e}  laplace {phys.item():.3e}")
    codes = torch.cat([v["pinn"].codes.detach() for v in vols])
    ck = {"net": trunk.state_dict(), "latent": cfg.latent, "width": cfg.width, "depth": cfg.depth,
          "field_mode": cfg.field_mode, "codes_mean": codes.mean(0).cpu(), "n_volumes": len(vols)}
    torch.save(ck, os.path.join(out_dir, "trunk.pt"))
    summary = {"train": {}, "history": hist}
    for v in vols:
        name = os.path.splitext(os.path.basename(v["path"]))[0]
        v["pinn"].cpu(); v["prob"].to("cpu")
        rep, _, _ = evaluate(v["pinn"], v["prob"], {"wires": v["pinn"].base, "handedness": v["h"], "image_lowres": None},
                             out_dir=os.path.join(out_dir, name), cfg=cfg, gt=_gt(v["path"], v["data"]), log=lambda *a: None)
        summary["train"][name] = rep
    if test_files:
        summary["holdout"] = holdout_comparison(test_files, cfg, os.path.join(out_dir, "trunk.pt"),
                                                slices_per_volume, log)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def holdout_comparison(files, cfg, trunk_path, slices_per_volume, log=print, budget=None):
    """Same wire init and the same (small) PINN budget, with and without the pretrained trunk."""
    budget = budget or max(100, cfg.iters_pinn // 5)
    out = {}
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        data, prob, wires, h = _volume(p, cfg, slices_per_volume, log)
        res = {}
        for tag, pre in (("scratch", ""), ("pretrained", trunk_path)):
            c = copy.deepcopy(cfg); c.pretrained = pre
            w = copy.deepcopy(wires)
            pinn = make_pinn(prob, c, w, h)
            _, hist = fit_pinn(prob, c, pinn, None, log=lambda *a: None, iters=budget)
            pinn.cpu(); prob.to("cpu")
            rep, _, _ = evaluate(pinn, prob, {"wires": w.cpu(), "handedness": h, "image_lowres": None},
                                 gt=_gt(p, data), log=lambda *a: None)
            prob.to(cfg.device)
            res[tag] = {"calib_data_loss": rep["calib_data_loss"], "physics_div": rep["physics_rel_rms"]["div"],
                        "loss_curve": [e["data"] for e in hist]}
            if "vs_ground_truth" in rep:
                res[tag]["gt_absolute_err"] = rep["vs_ground_truth"]["biot_savart_model"]["absolute_err"]
                res[tag]["gt_subspace_err"] = rep["vs_ground_truth"]["biot_savart_model"]["subspace_err"]
        log(f"[holdout {name}] budget {budget} it: scratch {res['scratch']['calib_data_loss']:.4e} vs "
            f"pretrained {res['pretrained']['calib_data_loss']:.4e}")
        out[name] = res
    return out


def expand(patterns):
    files = []
    for p in patterns:
        files += sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
    return files
