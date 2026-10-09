"""Dynamic MRI: time-varying coil sensitivities, low rank in time, every frame Biot–Savart.

Model (frame t, coil j):

    B_j(r, t) = B_wires(r; v0 + sum_k a_tk V_k)  -  grad Phi_j^static(r)  -  sum_k a_tk grad Phi_jk(r)
    C_j(r, t) = exp(i theta_j) (B_x + i h B_y)

* the wire part is an exact Biot–Savart field for *every* frame, with low-rank coil motion
  (vertices move along K shared temporal modes a_t -- e.g. a flexible array breathing with the chest);
* the PINN part is a real, rank-K combination of harmonic potentials, hence harmonic for every frame
  (Laplace is linear), so the physics constraint holds frame by frame by construction;
* the receive-chain phases theta_j do not change over time.

This is the COMPACT-maps idea (low-rank temporal kernels) with the per-frame smoothness prior
replaced by the Biot–Savart constraint. Temporal modes a_t are initialised from the data
(principal components of the per-frame coil energy distribution, a self-gating signal), kept
smooth (||D_t a||^2) and at unit RMS (fixes the scale shared with V / Phi_k).

Frames are handled as "slices" of one Problem (same geometry, own sampling per frame), so any
Cartesian mask or per-frame non-Cartesian trajectory works.
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import MultiCoilData, _read_cfl
from .fit import FitConfig, Problem, _extra_penalty, _image_param, _log, fit, sens_sets
from .geometry import SliceGeometry, fft2c_np
from .models import HarmonicPINN
from .physics import grad


class DynamicCoils(nn.Module):
    def __init__(self, static: HarmonicPINN, T: int, K: int, alpha_init=None, latent=16, width=64, depth=4,
                 parts: str = "both"):
        """parts: 'both' (moving wires + rank-K harmonic potentials), 'wires' (only conductor
        motion: the most constrained, physically interpretable option) or 'fields'."""
        super().__init__()
        assert static.base is not None and static.field_mode == "real"
        self.static, self.T, self.K = static, T, K
        N = static.n_coils
        self.n_coils, self.handedness = N, static.handedness
        L, M = static.base.verts.shape[1:3]
        self.parts = parts
        self.V = nn.Parameter(torch.zeros(K, N, L, M, 3), requires_grad=parts in ("both", "wires"))
        self.dyn = (HarmonicPINN(N * K, "real", latent, width, depth, static.handedness, last_init="zero")
                    if K and parts in ("both", "fields") else None)
        a = torch.as_tensor(alpha_init, dtype=torch.float32) if alpha_init is not None else 0.1 * torch.randn(T, K)
        self.alpha = nn.Parameter(a.clone())

    @property
    def device(self):
        return self.V.device

    def frame_field(self, points, frames, create_graph=True):
        """(B, N, P, 3) real field for frames ``frames`` at shared points (P,3)."""
        st = self.static
        a = self.alpha[frames]                                                # (B,K)
        v = st.base.verts[None] + torch.einsum("bk,knlmd->bnlmd", a, self.V)  # (B,N,L,M,3)
        B = st.base.field(points, verts=v)[..., 0]                            # (B,N,P,3)
        B = B + st.correction_field(points, create_graph)[None, ..., 0]
        if self.dyn is not None:
            Bd = self.dyn.correction_field(points, create_graph)[..., 0]       # (N*K,P,3)
            Bd = Bd.reshape(self.K, self.n_coils, *Bd.shape[1:])
            B = B + torch.einsum("bk,knpd->bnpd", a, Bd)
        return B

    def sensitivity_frames(self, points, frames, create_graph=True):
        B = self.frame_field(points, frames, create_graph)
        C = (B[..., 0] + 1j * self.handedness * B[..., 1]) * self.static.gain()[None, :, None]
        return C                                                               # (B,N,P)

    def physics_loss(self, points):
        loss = self.static.physics_loss(points)
        if self.dyn is not None:
            loss = loss + self.dyn.physics_loss(points)
        return loss

    def displacement(self, frames):
        return torch.einsum("bk,knlmd->bnlmd", self.alpha[frames], self.V)       # (B,N,L,M,3)

    def regularisers(self, frames, support, margin, max_disp=None):
        a = self.alpha
        smooth = (a[1:] - a[:-1]).pow(2).mean() if a.shape[0] > 1 else a.sum() * 0
        unit = ((a.pow(2).mean(0) - 1) ** 2).sum()
        v = self.static.base.verts[None] + torch.einsum("bk,knlmd->bnlmd", a[frames], self.V)
        from .keepout import KeepOut, keepout_penalty
        if isinstance(support, KeepOut):
            outside = keepout_penalty(support, v, margin)
        else:
            d = torch.cdist(v.reshape(-1, 3), support).min(dim=1).values
            outside = torch.relu(margin - d).pow(2).mean()
        if max_disp is not None:
            # physical bound on conductor motion (e.g. a flexible array on a breathing torso)
            dn = self.displacement(frames).norm(dim=-1)
            outside = outside + 100 * torch.relu(dn - max_disp).pow(2).mean()
        return smooth, unit, outside


def alpha_from_data(prob: Problem, K: int, smooth: int = 3):
    """Self-gating initialisation: leading temporal principal components of the per-frame
    distribution of energy over coils (insensitive to global contrast changes)."""
    e = (prob.I.abs() ** 2).sum((2, 3))                     # (T,N)
    e = e / e.sum(1, keepdim=True)
    e = e - e.mean(0, keepdim=True)
    if smooth > 1:
        ker = torch.ones(1, 1, smooth) / smooth
        e = torch.nn.functional.conv1d(e.T[:, None], ker, padding=smooth // 2)[:, 0].T[: prob.S]
    U, Sv, _ = torch.linalg.svd(e, full_matrices=False)
    a = U[:, :K] * np.sqrt(prob.S)
    return a, (Sv[:K] ** 2 / (Sv ** 2).sum()).tolist()


@dataclass
class DynConfig:
    rank: int = 2
    iters_dyn: int = 1500
    lambda_smooth: float = 1e-3
    lambda_unit: float = 1e-2
    lr_alpha: float = 1e-2
    lr_verts: float = 3e-3
    parts: str = "both"
    max_disp_mm: float = 0.0          # bound on conductor displacement (0 = none)


def fit_dynamic(data: MultiCoilData, cfg: FitConfig, dcfg: DynConfig, log=print):
    """Stage 1: static Biot–Savart fit over all frames (per-frame images).
    Stage 2: low-rank dynamic coils, initialised at the static solution."""
    cfg.validate("dynamic")
    t0 = time.time()
    scfg = copy.deepcopy(cfg)
    static_model, prob, hist, extra = fit(data, scfg, log=log)
    snapshot = copy.deepcopy(static_model)
    prob.to(cfg.device); static_model.to(cfg.device)
    m_static = extra["image_lowres"].to(cfg.device) if extra["image_lowres"] is not None else None
    a0, ev = alpha_from_data(prob, dcfg.rank)
    log(f"temporal modes initialised from data (explained variance of coil-energy PCA: {np.round(ev, 3).tolist()})")
    dyn = DynamicCoils(static_model, prob.S, dcfg.rank, a0, cfg.latent, cfg.width, cfg.depth, dcfg.parts).to(cfg.device)
    m = torch.nn.Parameter(m_static.clone()) if m_static is not None else None
    params = [{"params": dyn.static.parameters(), "lr": cfg.lr_pinn},
              {"params": dyn.dyn.parameters() if dyn.dyn is not None else [], "lr": cfg.lr_pinn},
              {"params": [dyn.V], "lr": dcfg.lr_verts},
              {"params": [dyn.alpha], "lr": dcfg.lr_alpha}]
    if m is not None:
        params.append({"params": [m], "lr": cfg.lr_image_pinn})
    from .fit import _colloc, _physics_at, _run_stage
    gen = torch.Generator(device=prob.device).manual_seed(cfg.seed + 11)
    frame_rng = np.random.default_rng(cfg.seed + 12)
    h, w = prob.calib
    pts = prob.coords.reshape(prob.S, h * w, 3)[0]                 # all frames share the grid
    sup = prob.support_pts[torch.randperm(prob.support_pts.shape[0], generator=gen, device=prob.device)[:1024]]
    dhist = []
    max_disp = dcfg.max_disp_mm / prob.data.geom.half_fov if dcfg.max_disp_mm else None
    B = cfg.slice_batch or prob.S
    wires = [dyn.static.base] if dyn.static.base is not None else []

    def step_loss(fr, pin, pair):
        C = dyn.sensitivity_frames(pts, fr)                        # (B,N,P)
        Cflat = C.permute(1, 0, 2).reshape(prob.N_model, -1)
        d = prob.data_loss(Cflat, m, fr)
        phys = _physics_at(dyn, cfg, pin, pair)
        smooth, unit, outside = dyn.regularisers(fr, prob.keep, cfg.outside_margin, max_disp)
        loss = (d + cfg.lambda_phys * phys + cfg.lambda_norm * prob.norm_loss(Cflat, fr)
                + dcfg.lambda_smooth * smooth + dcfg.lambda_unit * unit + cfg.lambda_outside * outside
                + _extra_penalty(dyn, C, m, cfg))
        return loss, d, phys, smooth

    def after(it, o):
        if it % cfg.log_every == 0 or it == dcfg.iters_dyn - 1:
            _, d, phys, smooth = o
            _log(dhist, it, data=d.item(), phys=phys.item(), smooth=smooth.item())
            log(f"[dynamic K={dcfg.rank}] it {it:5d}  data {d.item():.4e}  laplace {phys.item():.3e}  "
                f"|V| {dyn.V.abs().max().item():.3f}")

    frames = [sorted(frame_rng.permutation(prob.S)[:B].tolist()) for _ in range(dcfg.iters_dyn)]
    _run_stage(prob, cfg, dcfg.iters_dyn, params, cfg.lr_min_pinn * cfg.lr_pinn, step_loss,
               lambda: _colloc(prob, cfg, gen, wires), frames, [gen], after)
    log(f"dynamic fit done in {time.time() - t0:.1f}s")
    hist["dynamic"] = dhist
    dyn.cpu(); prob.to("cpu")
    return dyn, prob, hist, {"static_snapshot": snapshot, "m_static": None if m_static is None else m_static.cpu(),
                             "m": None if m is None else m.detach().cpu(), "alpha_init": a0, "explained_var": ev,
                             "handedness": extra["handedness"]}


@torch.no_grad()
def _frame_loss(prob, C_all, m):
    """Per-frame calibration data loss for maps C_all (T,N,P)."""
    out = []
    for t in range(prob.S):
        out.append(float(prob.data_loss(C_all[t], m, [t])))
    return np.array(out)


def evaluate_dynamic(dyn, prob, extra, out_dir, gt=None, shape=None, log=print):
    os.makedirs(out_dir, exist_ok=True)
    geom = prob.data.geom
    H, W = shape or geom.shape
    T, N = prob.S, prob.N_model
    h, w = prob.calib
    pts_full = geom.grid((H, W))[0].reshape(-1, 3)
    pts_cal = prob.coords.reshape(T, h * w, 3)[0]
    frames = list(range(T))
    with torch.enable_grad():
        Cd = torch.cat([dyn.sensitivity_frames(pts_full, frames[i:i + 8], create_graph=False).detach()
                        for i in range(0, T, 8)])                                   # (T,N,P)
        Cs = dyn.static.sensitivity_batched(pts_full)                              # static stage maps
        Cd_cal = torch.cat([dyn.sensitivity_frames(pts_cal, frames[i:i + 8], create_graph=False).detach()
                            for i in range(0, T, 8)])
    # re-fit images with the static maps for a fair static-vs-dynamic data-consistency comparison
    static_snapshot = extra["static_snapshot"]
    with torch.enable_grad():
        Cs_cal = static_snapshot.sensitivity(pts_cal, create_graph=False).detach()
    loss_dyn = _frame_loss(prob, Cd_cal, extra["m"])
    loss_sta = _frame_loss(prob, Cs_cal[None].expand(T, -1, -1), extra["m_static"])
    with torch.no_grad():
        disp_mm = float(dyn.displacement(list(range(T))).norm(dim=-1).max()) * prob.data.geom.half_fov
    rep = {"frames": T, "coils": N, "rank": dyn.K, "max_conductor_displacement_mm": disp_mm,
           "calib_data_loss": {"static": float(loss_sta.mean()), "dynamic": float(loss_dyn.mean())},
           "map_change_over_time_rel": float((Cd - Cd.mean(0, keepdim=True)).abs().pow(2).mean().sqrt()
                                             / Cd.abs().pow(2).mean().sqrt())}
    Cd_g = Cd.reshape(T, N, H, W)
    Cs_g = static_snapshot.sensitivity_batched(pts_full).reshape(N, H, W)
    if gt is not None:
        from .recon import absolute_error, subspace_error
        gtT = torch.from_numpy(gt)                                                  # (T,N,H,W)
        sup = (gtT.abs().pow(2).sum(1) > 0).float() * torch.from_numpy(extra.get("object_support", np.ones((H, W)))).float()
        rep["vs_ground_truth"] = {
            "dynamic": {"absolute_err": float(np.mean([absolute_error(Cd_g[t:t + 1], gtT[t:t + 1], sup[t:t + 1]) for t in range(T)])),
                        "subspace_err": float(np.mean([subspace_error(Cd_g[t:t + 1], gtT[t:t + 1], sup[t:t + 1]) for t in range(T)]))},
            "static": {"absolute_err": float(np.mean([absolute_error(Cs_g[None], gtT[t:t + 1], sup[t:t + 1]) for t in range(T)])),
                       "subspace_err": float(np.mean([subspace_error(Cs_g[None], gtT[t:t + 1], sup[t:t + 1]) for t in range(T)]))}}
        if "true_motion" in extra:
            mot = np.asarray(extra["true_motion"])
            a = dyn.alpha.detach().numpy()
            rep["alpha_corr_with_true_motion"] = [float(abs(np.corrcoef(a[:, k], mot)[0, 1])) for k in range(dyn.K)]
    # common-phase convention (--phase-ref): one virtual body coil for all frames, so the phase
    # convention does not change over time; frame images are counter-rotated (C * m unchanged)
    from .recon import phase_reference, virtual_body_coil
    pref = prob.cfg.phase_ref
    d = virtual_body_coil(Cd_g, robust=(pref == "inati")) if pref in ("inati", "inati-eig") else None
    Cd_g, _, pinfo, _ = phase_reference(Cd_g, pref, None, None, d)
    Cs_g, _, _, _ = phase_reference(Cs_g[None], pref, None, None, d)
    Cs_g = Cs_g[0]
    imgs = extra["m"][:, 0] if extra["m"] is not None else None
    if imgs is not None:
        Cl = Cd_cal.reshape(T, N, h, w)
        _, imgs, _, _ = phase_reference(Cl, pref, None, imgs, d)
    rep["phase_reference"] = pinfo
    np.save(os.path.join(out_dir, "maps_dynamic.npy"), Cd_g.numpy())                 # physical / model channels
    if prob.mix is not None:                                                            # + channels of the processed data
        np.save(os.path.join(out_dir, "maps_dynamic_data_channels.npy"), prob.mixC(Cd_g, 1).numpy())
    np.save(os.path.join(out_dir, "maps_static.npy"), Cs_g.numpy())
    np.save(os.path.join(out_dir, "alpha.npy"), dyn.alpha.detach().numpy())
    if imgs is not None:
        np.save(os.path.join(out_dir, "frame_images_lowres.npy"), imgs.numpy())
    rep["calib_data_loss_per_frame"] = {"static": loss_sta.tolist(), "dynamic": loss_dyn.tolist()}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    _dyn_figures(out_dir, dyn, Cd_g, loss_sta, loss_dyn, extra)
    log(json.dumps({k: v for k, v in rep.items() if k != "calib_data_loss_per_frame"}, indent=1))
    return rep


def _dyn_figures(out_dir, dyn, Cd, loss_sta, loss_dyn, extra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = dyn.alpha.detach().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for k in range(a.shape[1]):
        ax[0].plot(a[:, k], label=f"mode {k + 1}")
    if "true_motion" in extra:
        mot = np.asarray(extra["true_motion"]); ax[0].plot(mot / (mot.std() + 1e-9), "k--", lw=1, label="true motion (scaled)")
    ax[0].set_xlabel("frame"); ax[0].set_title("learned temporal modes of the coil fields"); ax[0].legend(fontsize=7)
    ax[1].plot(loss_sta, label="static Biot–Savart maps"); ax[1].plot(loss_dyn, label=f"dynamic (rank {dyn.K})")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("calibration data loss"); ax[1].legend(fontsize=7); ax[1].set_yscale("log")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "dynamic_modes.png"), dpi=110); plt.close(fig)
    # map change: frame with max |alpha| minus mean, magnitude, coils
    T, N = Cd.shape[:2]
    t1 = int(np.argmax(np.abs(a[:, 0]))) if a.shape[1] else 0
    D = (Cd[t1] - Cd.mean(0)).abs().numpy()
    ref = Cd.abs().mean(0).numpy().max()
    fig, ax = plt.subplots(2, N, figsize=(1.6 * N, 3.6), squeeze=False)
    for j in range(N):
        ax[0, j].imshow(Cd[t1, j].abs().numpy(), cmap="magma"); ax[0, j].set_title(f"coil {j + 1}", fontsize=8)
        ax[1, j].imshow(D[j] / ref, cmap="viridis", vmin=0, vmax=max(D.max() / ref, 1e-6))
    ax[0, 0].set_ylabel(f"|C| frame {t1}", fontsize=8); ax[1, 0].set_ylabel("|C_t - mean_t C|", fontsize=8)
    for x in ax.flat:
        x.set_xticks([]); x.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "dynamic_map_change.png"), dpi=110); plt.close(fig)


# ----------------------------------------------------------------------------- data
def make_dynamic_synthetic(n_coils=8, shape=(64, 64), T=40, motion=0.12, period=10.0, accel=4, calib=12,
                           noise=0.01, seed=0):
    """Static phantom with contrast uptake; the anterior half of the receive array moves with
    'respiration' (displacement up to ``motion`` normalised units). Per-frame Cartesian random
    sampling (centre ``calib`` fully sampled). Returns (data, gt_maps (T,N,H,W), true_motion (T,), support)."""
    from .data import shepp_logan_like, synthetic_array
    from .physics import biot_savart_polyline
    H, Wd = shape
    rng = np.random.default_rng(seed)
    geom = SliceGeometry(shape, (1.0, 1.0), np.zeros(T), "axial")
    pts = geom.grid()[0].reshape(-1, 3)
    verts, ph = synthetic_array(n_coils, seed=seed)
    t = np.arange(T)
    resp = np.sin(2 * np.pi * t / period) + 0.3 * np.sin(4 * np.pi * t / period + 0.7)
    anterior = np.clip(verts[:, 0, :, 1].mean(-1), 0, None)            # coils with y>0 move, prop. to y
    obj = shepp_logan_like(H, Wd, seed)
    uptake = 1 + 0.6 * (1 - np.exp(-t / (T / 4)))                       # contrast change over time
    lesion = np.zeros((H, Wd)); yy, xx = np.ogrid[:H, :Wd]
    lesion[((yy - H * 0.4) ** 2 + (xx - Wd * 0.6) ** 2) < (H * 0.08) ** 2] = 1
    gts, ks, masks = [], [], []
    for i in range(T):
        v = verts.copy()
        v[..., 1] += (motion * resp[i] * anterior)[:, None, None]
        B = biot_savart_polyline(torch.from_numpy(v), pts).sum(1).numpy()
        C = (np.exp(1j * ph)[:, None] * (B[..., 0] + 1j * B[..., 1])).reshape(n_coils, H, Wd)
        gts.append(C)
    gts = np.stack(gts)
    gts = gts / np.sqrt((np.abs(gts) ** 2).sum(1).mean())
    for i in range(T):
        img = obj * (1 + (uptake[i] - 1) * lesion)
        k = fft2c_np(gts[i] * img)
        k = k + noise * (rng.normal(size=k.shape) + 1j * rng.normal(size=k.shape)) / np.sqrt(2)
        m = rng.random((H, Wd)) < 1.0 / accel
        r0, c0 = H // 2 - calib // 2, Wd // 2 - calib // 2
        m[r0:r0 + calib, c0:c0 + calib] = True
        ks.append(k * m); masks.append(m[None])
    data = MultiCoilData(np.stack(ks).astype(np.complex64), geom, mask=np.stack(masks),
                         meta={"source": "dynamic synthetic", "dynamic": True})
    return data, gts.astype(np.complex64), resp, (np.abs(obj) > 0).astype(float)


def load_dynamic_radial(ksp_path, traj_path, spokes_per_frame, fov_mm, frames=None, noise_path=None,
                        plane="axial") -> MultiCoilData:
    """Golden-angle radial dynamic data in BART layout: k-space (1, samples, spokes, coils) and a
    BART trajectory (3, samples, spokes); spokes are binned into frames of ``spokes_per_frame``."""
    k = _read_cfl(ksp_path)
    k = k.reshape([d for d in k.shape if d > 1])                              # (samples, spokes, coils)
    ns, nsp, N = k.shape
    tr = _read_cfl(traj_path).real.reshape(3, ns, nsp)
    T = nsp // spokes_per_frame
    k = k[:, :T * spokes_per_frame].reshape(ns, T, spokes_per_frame, N)
    tr = tr[:, :, :T * spokes_per_frame].reshape(3, ns, T, spokes_per_frame)
    ks = np.transpose(k, (1, 3, 2, 0)).reshape(T, N, -1)                     # (T,N,spf*ns)
    traj = np.transpose(tr[:2], (2, 3, 1, 0)).reshape(T, -1, 2)               # (T,M,2)
    if frames is not None:
        ks, traj = ks[frames], traj[frames]
    matrix = int(2 * np.ceil(np.abs(traj).max()))
    geom = SliceGeometry((matrix, matrix), (fov_mm / matrix,) * 2, np.zeros(ks.shape[0]), plane)
    noise = _read_cfl(noise_path).reshape(-1, N) if noise_path else None
    return MultiCoilData(ks.astype(np.complex64), geom, noise=noise, traj=traj.astype(np.float32),
                         meta={"source": ksp_path, "dynamic": True, "spokes_per_frame": spokes_per_frame})
