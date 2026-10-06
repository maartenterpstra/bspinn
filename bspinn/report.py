"""Full-resolution evaluation, outputs and figures."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .benchmark import HeldoutBenchmark, fourier_resample
from .components import MultiSetCoils
from .fit import sens_sets
from .geometry import fft2c, ifft2c
from .physics import maxwell_static_residual
from .recon import (absolute_error, cg_sense, cg_sense_sets, coil_combine, phase_reference, lowres_division_maps, normalize_maps,
                    projection_residual, subspace_error, upsample_lowres_division)


def _heldout_split(n_cand, frac, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_cand, generator=g) < frac


def evaluate(model, prob, extra, out_dir=None, cfg=None, history=None, gt=None, log=print,
             recon_size=128, heldout_frac=0.2, seed=0, ref_maps=None):
    """Full-resolution maps + reconstruction + metrics.

    * fully sampled Cartesian: projection residual of the *full-resolution* coil images onto
      span(C) (high spatial frequencies were never used for calibration -> generalisation test).
    * undersampled (Cartesian or radial): a random fraction of acquired samples outside the
      calibration region is held out; CG-SENSE with each map set predicts them (NRMSE).
    """
    geom = prob.data.geom
    S, N = prob.S, prob.N
    H, W = geom.shape
    if prob.noncart:
        H = W = min(recon_size, H)
    pts = geom.grid((H, W)).reshape(-1, 3)
    Nm = prob.N_model
    Cphys = model.sensitivity_batched(pts)                          # model (physical) channels
    C = prob.mixC(Cphys, -2).reshape(N, S, H, W).permute(1, 0, 2, 3).contiguous()   # data channels
    Cphys = Cphys.reshape(Nm, S, H, W).permute(1, 0, 2, 3).contiguous()
    Cn = normalize_maps(C)
    rep = {"n_coils": N, "n_model_channels": Nm, "channel_mixing": prob.mix is not None,
           "compression_energy": prob.data.meta.get("compression_energy"), "n_slices": S, "shape": [H, W], "data_term": prob.data_term,
           "handedness": extra["handedness"], "encoding": "radial (NUDFT)" if prob.noncart else "cartesian"}
    img_base = None
    fully = (not prob.noncart) and bool(prob.mask_full.all()) and not prob.calib_only
    if prob.calib_only:
        # only the calibration region was loaded (large datasets): low-res metrics only
        I_up = fourier_resample(prob.I, (H, W))
        base = normalize_maps(I_up)
        rss = (I_up.abs() ** 2).sum(1).sqrt()
        img = coil_combine(C, I_up)
    elif fully:
        base = lowres_division_maps(prob.ks_full, prob.kcalib)
        I_full = ifft2c(prob.ks_full)
        rss = (I_full.abs() ** 2).sum(1).sqrt()
        img = coil_combine(C, I_full)
    else:
        bench = HeldoutBenchmark(prob, (H, W), frac=heldout_frac, seed=seed)
        base = upsample_lowres_division(prob.I, (H, W)) if prob.noncart else \
            lowres_division_maps(prob.ks_full * prob.mask_full, prob.kcalib)
        rss = bench.support.clone()
    # object support from the smooth low-res calibration image (noise-robust; a full-resolution RSS
    # threshold lets background noise in when many coils / low SNR)
    rss_lr = (fourier_resample(prob.I, (H, W)).abs() ** 2).sum(1).sqrt()
    support = prob.object_support(rss_lr)

    refs = {name: None for name in (ref_maps or {})}
    sets = {"biot_savart_model": Cn, "lowres_division_baseline": base}
    if fully:
        rep["projection_residual_fullres"] = {k: projection_residual(M, I_full, support) for k, M in sets.items()}
    for name, M in (ref_maps or {}).items():
        Mt = torch.from_numpy(np.asarray(M)).to(torch.complex64)
        if Mt.shape[-2:] != (H, W):
            Mt = fourier_resample(Mt, (H, W))
        refs[name] = normalize_maps(Mt) * ((Mt.abs() ** 2).sum(1, keepdim=True) > 0)
        sets[name] = refs[name]
        if fully:
            rep["projection_residual_fullres"][name] = projection_residual(refs[name], I_full, support)
    wires = extra.get("wires")
    Cw = None
    if wires is not None and wires is not model:
        with torch.enable_grad():
            Cw = prob.mixC(wires.sensitivity_batched(pts), -2).reshape(N, S, H, W).permute(1, 0, 2, 3)
        sets["wire_init_only"] = normalize_maps(Cw)
        if fully:
            rep["projection_residual_fullres"]["wire_init_only"] = projection_residual(Cw, I_full, support)
    if not fully and not prob.calib_only:
        hm = {"region": "random 20% of acquired samples between the calibration box and 2x its size",
              "n_samples_per_coil": bench.n_heldout_per_coil, "unmasked": {}, "common_support_mask": {}}
        for k, M in sets.items():
            hm["unmasked"][k], x = bench.score(M, masked=False)
            hm["common_support_mask"][k], xm = bench.score(M, masked=True)
            if k == "biot_savart_model":
                img = xm
            if k == "lowres_division_baseline":
                img_base = xm
        rep["heldout_kspace_nrmse"] = hm
        rss = (bench.op.adjoint(bench.y_tr * (1 if bench.weights is None else bench.weights)).abs() ** 2).sum(1).sqrt()
    with torch.enable_grad():
        Cl = sens_sets(model, prob.coords, create_graph=False).detach()
        rep["calib_data_loss"] = float(prob.data_loss(Cl, extra.get("image_lowres")))
    set_imgs = None
    if isinstance(model, MultiSetCoils):
        Ks = prob.mixC(model.sensitivity_sets_batched(pts), -2).reshape(model.n_sets, N, S, H, W).permute(2, 0, 1, 3, 4)
        rep["sets"] = {"kinds": model.set_kinds(),
                       "relative_energy": [float((Ks[:, k].abs() ** 2).mean() / (Ks[:, 0].abs() ** 2).mean()) for k in range(model.n_sets)],
                       "learned_shifts": model.shifts.detach().tolist()}
        if fully or prob.calib_only:
            Iref = I_full if fully else I_up
            set_imgs = multiset_combine(Ks, Iref, prob.cfg.set_ridge)
        else:
            set_imgs = cg_sense_sets(bench.op, bench.y_tr, normalize_set_maps(Ks), lam=bench.lam,
                                     iters=bench.iters, weights=bench.weights, ridge=prob.cfg.set_ridge)
        rep["sets"]["image_energy"] = [float((set_imgs[:, k].abs() ** 2).sum() / (set_imgs.abs() ** 2).sum()) for k in range(model.n_sets)]
    sp = prob.support_pts[torch.randperm(prob.support_pts.shape[0])[:1500]]
    try:
        with torch.enable_grad():
            div, curl = maxwell_static_residual(lambda x: model.field(x, create_graph=True), sp, model.n_coils)
        rep["physics_rel_rms"] = {"div": float(div.mean()), "curl": float(curl.mean())}
    except NotImplementedError:
        rep["physics_rel_rms"] = None                      # --model free: no magnetic field
    if gt is not None:
        gt_phys = torch.from_numpy(gt)
        mixing = prob.data.meta.get("mix")                           # ground truth is in physical channels
        gtC = gt_phys if mixing is None else (gt_phys.movedim(1, -1) @ torch.as_tensor(mixing).T).movedim(-1, 1)
        rep["vs_ground_truth"] = {
            "biot_savart_model": {"subspace_err": subspace_error(C, gtC, support), "absolute_err": absolute_error(C, gtC, support)},
            "lowres_division_baseline": {"subspace_err": subspace_error(base, gtC, support), "absolute_err": absolute_error(base, gtC, support)}}
        if prob.mix is not None:
            rep["vs_ground_truth"]["biot_savart_model_physical_channels"] = {
                "subspace_err": subspace_error(Cphys, gt_phys, support), "absolute_err": absolute_error(Cphys, gt_phys, support)}
        if Cw is not None:
            rep["vs_ground_truth"]["wire_init_only"] = {"subspace_err": subspace_error(Cw, gtC, support), "absolute_err": absolute_error(Cw, gtC, support)}
    # common-phase convention of the exported maps (metrics above are gauge-invariant or use the
    # model gauge); the image is counter-rotated so that C * m is unchanged
    pref = prob.cfg.phase_ref
    C, img_r, pinfo, Phi = phase_reference(C, pref, support, img if img is not None and img.dim() == 3 else None)
    if img_r is not None:
        img = img_r
    if set_imgs is not None and Phi is not None:
        set_imgs = set_imgs * Phi[:, None]
    Cn = normalize_maps(C)
    rep["phase_reference"] = pinfo
    # phase singularities of the exported maps (physical zeros of B1- show up in every gauge;
    # a vanishing phase reference would add one shared pole to every coil)
    from .recon import phase_vortices
    sup2 = support.reshape(S, H, W) if support.dim() != 3 else support
    rep["phase_vortices"] = {f"biot_savart_model[{pinfo.get('phase_ref', pref)}]": phase_vortices(C, sup2),
                             "lowres_division_baseline": phase_vortices(base, sup2)}
    for name, M in refs.items():
        rep["phase_vortices"][name] = phase_vortices(M, sup2)
    log(json.dumps(rep, indent=1))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "maps.npy"), C.numpy())            # channels of the (processed) data
        if prob.mix is not None:                                            # the physical coils, same phase convention
            Cp = Cphys if Phi is None else Cphys * Phi.conj()[:, None]
            np.save(os.path.join(out_dir, "maps_physical.npy"), Cp.numpy())
            np.save(os.path.join(out_dir, "channel_mixing.npy"), prob.mix.numpy())
        np.save(os.path.join(out_dir, "maps_normalized.npy"), Cn.numpy())
        np.save(os.path.join(out_dir, "maps_baseline_lowres_division.npy"), base.numpy())
        np.save(os.path.join(out_dir, "image.npy"), img.numpy())
        if set_imgs is not None:
            np.save(os.path.join(out_dir, "set_images.npy"), set_imgs.numpy())
            _set_figure(out_dir, set_imgs, model.set_kinds())
        torch.save({"state_dict": model.state_dict(), "class": type(model).__name__,
                    "config": cfg.__dict__ if cfg else None, "n_coils": N,
                    "handedness": extra["handedness"]}, os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump({"report": rep, "history": history, "config": cfg.__dict__ if cfg else None}, f, indent=1)
        _figures(out_dir, C, Cn, base, img, rss, support, history, img_base, pref)
    return rep, C, img


def multiset_combine(Ks, I, ridge):
    """Per-pixel least squares I(r) = sum_k C_k(r) m_k(r) -> set images (S,K,H,W)."""
    G = torch.einsum("sknhw,slnhw->shwkl", Ks.conj(), Ks)
    b = torch.einsum("sknhw,snhw->shwk", Ks.conj(), I)
    K = Ks.shape[1]
    diag = G.diagonal(dim1=-2, dim2=-1).real
    r0 = 1e-6 * diag[..., 0].mean()
    regs = [r0.expand_as(diag[..., 0])] + [ridge * diag[..., k] + r0 for k in range(1, K)]
    m = torch.linalg.solve(G + torch.diag_embed(torch.stack(regs, -1).to(G.dtype)), b.unsqueeze(-1)).squeeze(-1)
    return m.permute(0, 3, 1, 2)


def normalize_set_maps(Ks):
    """Normalise all sets by the primary set's RSS (keeps their relative weights)."""
    return Ks / (Ks[:, 0].abs() ** 2).sum(1, keepdim=True).sqrt().clamp_min(1e-12)[:, None]


def _set_figure(out_dir, set_imgs, kinds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = set_imgs.shape[1]
    fig, ax = plt.subplots(1, K, figsize=(3.2 * K, 3.4), squeeze=False)
    vmax = float(set_imgs[0].abs().max())
    for k in range(K):
        ax[0, k].imshow(set_imgs[0, k].abs().numpy(), cmap="gray", vmin=0, vmax=vmax)
        ax[0, k].set_title(f"set {k + 1}: {kinds[k]}", fontsize=9); ax[0, k].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "set_images.png"), dpi=110); plt.close(fig)


def _figures(out_dir, C, Cn, base, img, rss, support, history, img_base=None, pref="model"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = 0
    N = C.shape[1]
    sup = support[s].numpy() > 0
    fig, ax = plt.subplots(4, N, figsize=(1.6 * N, 7), squeeze=False)
    # Biot–Savart row: the exported phase (convention --phase-ref); baseline: relative to coil 1
    for row, (M, name) in enumerate([(Cn, "Biot–Savart"), (base, "low-res division")]):
        ref = torch.exp(-1j * torch.angle(M[s, :1])) if row == 1 else 1.0
        Mr = (M[s] * ref).numpy()
        for j in range(N):
            ax[2 * row, j].imshow(np.abs(Mr[j]), cmap="gray", vmin=0, vmax=1)
            ax[2 * row + 1, j].imshow(np.where(sup, np.angle(Mr[j]), np.nan), cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax[2 * row, 0].set_ylabel(f"{name}\n|C| (norm.)", fontsize=8)
        ax[2 * row + 1, 0].set_ylabel(f"{name}\n" + (f"phase (ref: {pref})" if row == 0 else "phase rel. coil 1"), fontsize=8)
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    for j in range(N):
        ax[0, j].set_title(f"coil {j + 1}", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "maps_normalized.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(2, max(N // 2, 1) if N > 1 else 1, figsize=(1.8 * max(N // 2, 1), 4), squeeze=False)
    absC = C[s].abs().numpy(); vmax = np.percentile(absC, 99.5)
    for j, a in enumerate(ax.flat):
        if j < N:
            a.imshow(absC[j], cmap="magma", vmin=0, vmax=vmax); a.set_title(f"|C| coil {j + 1}", fontsize=8)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Absolute (un-normalised) Biot–Savart sensitivities", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "maps_absolute.png"), dpi=110); plt.close(fig)

    n = 2 if img_base is None else 3
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 4))
    ax[0].imshow(rss[s].numpy(), cmap="gray"); ax[0].set_title("RSS (zero-filled / gridded)")
    ax[1].imshow(np.abs(img[s].numpy()), cmap="gray"); ax[1].set_title("SENSE / combine with Biot–Savart maps")
    if img_base is not None:
        ax[2].imshow(np.abs(img_base[s].numpy()), cmap="gray"); ax[2].set_title("SENSE with low-res division maps")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "image.png"), dpi=110); plt.close(fig)

    if history:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
        off = 0
        for stage, key in [("wires", "data"), ("pinn", "data"), ("pinn", "phys")]:
            if stage in history and history[stage]:
                h = history[stage]
                its = [e["it"] + (0 if stage == "wires" else off) for e in h]
                ax.semilogy(its, [e[key] for e in h], label=f"{stage}: {key}")
            if stage == "wires" and "wires" in history:
                off = history["wires"][-1]["it"] + 1
        ax.set_xlabel("iteration (stages concatenated)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss.png"), dpi=110); plt.close(fig)
