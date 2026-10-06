"""Training: joint estimation of coil sensitivities (Biot–Savart constrained) and image.

Objective (NLINV-style, but the Sobolev smoothness penalty is replaced by physics):

    L = D(C, y) + lambda_phys * R_phys(C) + lambda_norm * (log mean_r sum_j |C_j|^2)^2 [+ wire / set regularisers]

Data term D, chosen automatically:
  * 'varpro'  -- calibration region fully sampled: the image(s) are eliminated in closed form
                 (variable projection). With K sensitivity sets A(r) = [C^(1)(r) .. C^(K)(r)] (N x K):
                 D = sum_r ||I||^2 - b^H (A^H A + R)^-1 b,  b = A^H I,  R = ridge on sets >= 2.
                 For K = 1 this is the projection residual of the coil images onto span C(r).
  * 'kspace'  -- undersampled / non-Cartesian: explicit low-res images m_s and
                 D = ||E(sum_s m_s C^(s)) - y||^2 / ||y||^2.

Sensitivities are *absolute* (not RSS-normalised); the gauge m -> m*phi, C -> C/phi is broken by
the physics prior: phi must keep every coil's map a Biot–Savart field.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .components import MultiSetCoils
from .data import MultiCoilData
from .encoding import CartesianOp, NUDFTOp, NUDFTOpBatched, radial_dcf
from .geometry import PLANES, crop_center, ifft2c
from .models import HarmonicPINN, WireLoopCoils


@dataclass
class FitConfig:
    model: str = "pinn"              # 'pinn' (wires + harmonic PINN) | 'wires' (hard only) | 'free' (no physics, ablation)
    field_mode: str = "real"         # 'real' (quasi-static) | 'complex'
    handedness: int = 0              # +1, -1, or 0 = auto
    calib: int = 48                  # calibration matrix (centre of k-space), clipped to data
    calib_oversample: int = 2        # >1 (Cartesian): model maps and image on an f-times finer grid and
                                     #   predict the calibration k-space as the centre of FFT(C*m), as NLINV
                                     #   does; avoids the truncation (Gibbs) mismatch of the pixel-wise
                                     #   low-res model. Forces the 'kspace' data term.
    data_term: str = "auto"          # 'auto' | 'varpro' | 'kspace'
    # wires
    loops_per_coil: int = 1
    verts_per_loop: int = 12
    iters_wires: int = 1500
    iters_auto: int = 300
    lr_wires: float = 1e-2
    lambda_outside: float = 10.0
    lambda_curv: float = 1e-1
    correction_grid: int = 12        # 3-D only: the PINN correction is evaluated on this many points per
                                     #   axis and upsampled (trilinear) to the calibration grid in training
    init_z_offset: float = 0.15      # +/- out-of-plane offset of initial loops (breaks z-mirror symmetry; 0 = off)
    outside_margin: float = 0.1      # min. conductor distance from object (normalised; 0.1 ~ 5% of FOV)
    # pinn
    width: int = 64
    depth: int = 4
    latent: int = 16
    init: str = "wires"              # 'wires': PINN = wire field + harmonic NN correction | 'none'
    pretrained: str = ""             # trunk checkpoint from `fit-dataset --mode shared`
    iters_pinn: int = 2500
    lr_pinn: float = 1e-3
    lambda_phys: float = 1e-2
    n_colloc: int = 768
    slab: float = 0.15               # half-thickness (normalised) of collocation slab around slices
    # extra (non-resolvable) components, cf. ENLIVE
    fold_sets: int = 0               # 1: add C(r +- FOV) along --fold-axis (object larger than FOV)
    fold_axis: str = "col"           # image axis of the folding (phase encode): 'row' | 'col'
    shift_sets: int = 0              # sets C(r + t_s) with learnable t_s (motion, fat shift)
    free_sets: int = 0               # independent extra Biot–Savart map sets (ENLIVE analogue)
    set_ridge: float = 1e-2          # ridge on images of sets >= 2 (relative)
    lambda_sets: float = 1e-3        # ||C||^2 penalty on free sets
    # no-physics ablation (--model free)
    lambda_sobolev: float = 1e-3     # first-order smoothness weight for --model free
    iters_free_init: int = 300       # distillation of the free model to low-res division maps
    # channels: whitening / compression produce *virtual* channels = complex mixtures of physical ones
    compress: int = 0                # number of virtual channels after SVD coil compression (0 = off)
    model_channels: str = "physical" # 'physical': model the physical coils, apply the known mixing to the
                                     #   predictions (exact) | 'data': model the virtual channels directly
    # outputs
    phase_ref: str = "inati"         # common-phase convention of exported maps: inati | inati-eig | coil | model
    # scaling
    slice_batch: int = 0             # slices per step for large volumes (0 = all)
    device: str = "cpu"
    # shared
    lambda_norm: float = 1e-2
    lambda_img: float = 1e-4
    lambda_temporal: float = 0.0     # ||m_t - m_t-1||^2 on frame images (dynamic data)
    support_thresh: float = 0.08
    seed: int = 0
    log_every: int = 100
    threads: int = 0

    @property
    def multiset(self):
        return bool(self.fold_sets or self.shift_sets or self.free_sets)

    def validate(self, context: str = "fit"):
        """Every extension is an independent option; combinations that are not implemented
        (or not meaningful) are rejected here with an explicit message."""
        err = []
        if self.model not in ("pinn", "wires", "free"):
            err.append(f"--model must be pinn|wires|free, got {self.model}")
        if self.field_mode not in ("real", "complex"):
            err.append("--field-mode must be real|complex")
        if self.model_channels not in ("physical", "data"):
            err.append("--model-channels must be physical|data")
        if self.compress < 0:
            err.append("--compress must be >= 0")
        if self.phase_ref not in ("inati", "inati-eig", "coil", "model", "none"):
            err.append("--phase-ref must be inati|inati-eig|coil|model")
        if (self.fold_sets or self.shift_sets) and self.model == "free":
            err.append("--fold-sets/--shift-sets evaluate the coil field at displaced positions, which "
                       "needs a physical model (--model wires|pinn)")
        if self.free_sets and self.model != "pinn":
            err.append("--free-sets needs --model pinn")
        if self.pretrained and self.model != "pinn":
            err.append("--pretrained needs --model pinn")
        if self.model == "pinn" and self.init == "none" and self.pretrained == "":
            pass  # allowed (pure PINN), but see README
        if context == "dynamic":
            if self.model != "pinn" or self.init != "wires" or self.field_mode != "real":
                err.append("dynamic: needs --model pinn --init wires --field-mode real")
            if self.multiset:
                err.append("dynamic: multiple map sets are not implemented together with dynamics")
        if context == "fit3d":
            if self.model not in ("pinn", "wires"):
                err.append("3-D: --model pinn|wires (the no-physics ablation is 2-D only)")
            if self.multiset:
                err.append("3-D: multiple map sets (fold/shift/free sets) are not implemented")
            if self.compress:
                err.append("3-D: --compress is not implemented (pre-whitening is)")
            if self.correction_grid < 4:
                err.append("--correction-grid must be >= 4")
        if context == "shared":
            if self.model != "pinn" or self.init != "wires":
                err.append("fit-dataset --mode shared: needs --model pinn --init wires")
            if self.multiset:
                err.append("fit-dataset --mode shared: multiple map sets not implemented (use --mode independent)")
            if self.data_term not in ("auto", "varpro"):
                err.append("fit-dataset --mode shared uses the VarPro data term")
        if err:
            raise ValueError("invalid option combination:\n  " + "\n  ".join(err))
        return self


class Problem:
    """Pre-processed calibration problem (low-resolution centre of k-space)."""

    def __init__(self, data: MultiCoilData, cfg: FitConfig):
        self.data, self.cfg = data, cfg
        geom = data.geom
        ks = torch.from_numpy(data.kspace)
        S, N = ks.shape[:2]
        H, W = geom.shape
        h, w = min(cfg.calib, H), min(cfg.calib, W)
        self.calib = (h, w)        # model grid (= k-space calibration box unless calib_oversample > 1)
        self.kcalib = (h, w)       # k-space calibration box
        self.noncart = data.traj is not None
        self.calib_only = bool(data.meta.get("calib_only", False))
        self.per_slice_traj = self.noncart and data.traj.ndim == 3
        if self.per_slice_traj:
            # one trajectory per slice / frame (dynamic radial): pad calibration samples
            traj = torch.from_numpy(data.traj).float()                       # (S,M,2)
            self.traj_full = traj
            sel = (traj[..., 0].abs() < h / 2) & (traj[..., 1].abs() < w / 2)
            Mc = int(sel.sum(1).max())
            tc = torch.zeros(S, Mc, 2); valid = torch.zeros(S, Mc, dtype=torch.bool)
            yk = torch.zeros(S, N, Mc, dtype=torch.complex64)
            for s_ in range(S):
                idx = torch.nonzero(sel[s_]).squeeze(1)
                tc[s_, :len(idx)] = traj[s_, idx]; valid[s_, :len(idx)] = True
                yk[s_, :, :len(idx)] = ks[s_][:, idx]
            self.calib_sel = sel
            self.op = NUDFTOpBatched(tc, valid, (h, w))
            mk = torch.ones(1)
            dcf = radial_dcf(tc) * valid
            dcf = dcf * (h * w) / dcf.sum(1, keepdim=True)
            I0 = self.op.adjoint(yk * dcf[:, None])
            self.mask_full = None
        elif self.noncart:
            traj = torch.from_numpy(data.traj).float()                       # (M,2)
            self.traj_full = traj
            sel = (traj[:, 0].abs() < h / 2) & (traj[:, 1].abs() < w / 2)
            self.calib_sel = sel
            tc = traj[sel]
            self.op = NUDFTOp(tc, (h, w))
            yk = ks[..., sel]
            mk = torch.ones(1)
            dcf = radial_dcf(tc)
            dcf = dcf * (h * w) / dcf.sum()
            I0 = self.op.adjoint(yk * dcf)
            self.mask_full = None
        else:
            mask = torch.from_numpy(data.mask)
            yk = crop_center(ks, (h, w))
            mk = crop_center(mask, (h, w)).to(torch.float32)
            f = int(cfg.calib_oversample)
            if f > 1:
                fh, fw = min(f * h, H), min(f * w, W)
                r0, c0 = fh // 2 - h // 2, fw // 2 - w // 2
                ykp = torch.zeros(*yk.shape[:-2], fh, fw, dtype=yk.dtype)
                ykp[..., r0:r0 + h, c0:c0 + w] = yk
                mkp = torch.zeros(*mk.shape[:-2], fh, fw)
                mkp[..., r0:r0 + h, c0:c0 + w] = mk
                yk, mk = ykp, mkp
                self.calib = (fh, fw)
                h, w = fh, fw
            self.op = CartesianOp(mk)
            I0 = ifft2c(yk * mk)
            self.mask_full = mask
        rss = (I0.abs() ** 2).sum(1).sqrt()
        self.scale = float(torch.quantile(rss.flatten(), 0.99))
        self.ks_full = ks / self.scale
        self.yk, self.mk = (yk if self.noncart else yk * mk) / self.scale, mk
        self.I = I0 / self.scale                                   # (S,N,h,w) zero-filled/gridded low-res coil images
        rss = rss / self.scale
        self.noise_sigma = self._estimate_noise(data) / self.scale
        self.support = self.object_support(rss)
        self.S, self.N = S, N
        # channel mixing (whitening / compression): data channel v = sum_n mix[v,n] * physical coil n
        mix = data.meta.get("mix")
        self.mix = (torch.as_tensor(mix, dtype=torch.complex64)
                    if (mix is not None and cfg.model_channels == "physical") else None)
        self.N_model = self.mix.shape[1] if self.mix is not None else N
        self.coords = geom.grid((h, w)).reshape(-1, 3)              # (S*h*w,3)
        sup = self.support.flatten() > 0
        self.support_pts = self.coords[sup]
        self.weights = (self.I.abs() ** 2).permute(1, 0, 2, 3).reshape(N, -1)
        self.coil_images = self.I.permute(1, 0, 2, 3).reshape(N, -1)
        if self.mix is not None:                                    # approximate physical-coil images (init only)
            self.coil_images_model = torch.linalg.pinv(self.mix) @ self.coil_images
        else:
            self.coil_images_model = self.coil_images
        self.weights_model = self.coil_images_model.abs() ** 2
        fully = (not self.noncart) and bool(mk.min() > 0)
        if cfg.data_term == "auto":
            self.data_term = "varpro" if fully else "kspace"
        else:
            self.data_term = cfg.data_term
            if self.data_term == "varpro" and not fully:
                raise ValueError("varpro needs a fully sampled Cartesian calibration region")
        self.normal_axis = geom.normal_axis()
        self.Inorm_slice = (self.I.abs() ** 2).sum((1, 2, 3))           # (S,)
        self.ynorm_slice = self.yk.abs().pow(2).reshape(S, -1).sum(1)
        # displacement of one FOV along the folding axis (normalised physical coordinates)
        ax_img = 0 if cfg.fold_axis == "row" else 1
        fov = geom.shape[ax_img] * geom.pixel_spacing[ax_img] / geom.half_fov
        self.fold_vec = torch.zeros(3); self.fold_vec[PLANES[geom.plane][ax_img]] = fov
        self.device = torch.device("cpu")

    # ---------------------------------------------------------------- support / noise
    @staticmethod
    def _estimate_noise(data):
        """Per-sample complex noise std sqrt(E|n|^2). Exactly 1 after pre-whitening with a noise
        prescan (compression is unitary on the whitened noise); otherwise estimated from the outermost
        acquired k-space with a low quantile (robust to residual edge signal there)."""
        if "noise_cov" in data.meta:
            return 1.0
        ks = data.kspace
        if data.traj is not None:
            t = np.asarray(data.traj).reshape(-1, 2) if np.asarray(data.traj).ndim == 2 else np.asarray(data.traj).reshape(ks.shape[0], -1, 2)
            r = np.linalg.norm(t, axis=-1)
            sel = r > np.quantile(r, 0.97)
            vals = ks[..., sel] if r.ndim == 1 else ks.transpose(1, 0, 2)[:, sel]
        else:
            S, N, H, W = ks.shape
            yy, xx = np.meshgrid(np.abs(np.arange(H) - H // 2) / (H / 2), np.abs(np.arange(W) - W // 2) / (W / 2), indexing="ij")
            corner = np.maximum(yy, xx) > 0.85
            acq = data.mask[:, 0] & corner[None]
            vals = ks.transpose(1, 0, 2, 3)[:, acq]
        v = np.abs(vals).ravel()
        # |n| is Rayleigh with E|n|^2 = sigma^2: P(|n| < q) = 1 - exp(-q^2/sigma^2)
        return float(np.quantile(v, 0.25) / np.sqrt(-np.log(0.75))) if v.size else 0.0

    def object_support(self, rss):
        """Object mask: RSS above both a fraction of its 99th percentile and ~3x the expected noise RSS."""
        thr = self.cfg.support_thresh * float(torch.quantile(rss.flatten(), 0.99))
        thr = max(thr, 3.0 * self.noise_sigma * np.sqrt(self.I.shape[1]))
        return (rss > thr).float()

    # ---------------------------------------------------------------- device
    _tensors = ("ks_full", "yk", "mk", "I", "support", "coords", "support_pts", "weights", "coil_images",
                "mix", "coil_images_model", "weights_model",
                "Inorm_slice", "ynorm_slice", "traj_full", "calib_sel", "mask_full", "fold_vec")

    def to(self, device):
        device = torch.device(device)
        for name in self._tensors:
            v = getattr(self, name, None)
            if torch.is_tensor(v):
                setattr(self, name, v.to(device))
        self.op = self.op.to(device)
        self.device = device
        return self

    # ---------------------------------------------------------------- helpers
    def mixC(self, C, dim=-2):
        """Model (physical) channels -> data (virtual) channels along ``dim``."""
        if self.mix is None:
            return C
        return (C.movedim(dim, -1) @ self.mix.T).movedim(-1, dim)

    def slice_coords(self, sl=None):
        if sl is None:
            return self.coords
        h, w = self.calib
        return self.coords.reshape(self.S, h * w, 3)[sl].reshape(-1, 3)

    def to_grid(self, C, S=None):
        """(..., N, S*h*w) -> (S, ..., N, h, w)"""
        h, w = self.calib
        S = S or self.S
        lead = C.shape[:-2]
        Cg = C.reshape(*lead, self.N, S, h, w)
        return Cg.movedim(-3, 0)

    def collocation(self, n, gen):
        """Points inside the imaged object, jittered in-plane and spread through a slab
        along the slice normal (gives the PDE a 3-D domain even for one 2-D slice)."""
        idx = torch.randint(0, self.support_pts.shape[0], (n,), generator=gen)
        p = self.support_pts[idx.to(self.support_pts.device)].clone()
        jit = 0.02 * torch.randn(p.shape, generator=gen)
        jit[:, self.normal_axis] = (2 * torch.rand(n, generator=gen) - 1) * self.cfg.slab
        return p + jit.to(p.device)

    def data_loss(self, Cs, m=None, sl=None, return_images=False):
        """Cs: (K,N,P) or (N,P) sensitivities on the calibration grid of slices ``sl``."""
        if Cs.dim() == 2:
            Cs = Cs[None]
        Cs = self.mixC(Cs, -2)
        K = Cs.shape[0]
        idx = slice(None) if sl is None else sl
        S = self.S if sl is None else len(sl)
        Cg = self.to_grid(Cs, S)                                       # (S,K,N,h,w)
        if self.data_term == "varpro" and K == 1:
            I = self.I[idx]
            C1 = Cg[:, 0]
            b = (C1.conj() * I).sum(1)
            den = (C1.abs() ** 2).sum(1)
            den = den + 1e-6 * den.mean().detach()
            mm = b / den
            res = (I.abs() ** 2).sum(1) - b.abs() ** 2 / den
            loss = res.sum() / self.Inorm_slice[idx].sum()
            return (loss, mm[:, None]) if return_images else loss
        if self.data_term == "varpro":
            I = self.I[idx]
            G = torch.einsum("sknhw,slnhw->shwkl", Cg.conj(), Cg)
            b = torch.einsum("sknhw,snhw->shwk", Cg.conj(), I)
            # ridge: tiny & global for the primary set; for extra sets relative to their own
            # per-pixel energy, i.e. a penalty on the contribution ||C_k m_k||^2 (scale-invariant)
            diag = G.diagonal(dim1=-2, dim2=-1).real
            r0 = 1e-6 * diag[..., 0].mean().detach()
            regs = [r0.expand_as(diag[..., 0])] + [self.cfg.set_ridge * diag[..., k] + r0 for k in range(1, K)]
            Gr = G + torch.diag_embed(torch.stack(regs, -1).to(G.dtype))
            mm = torch.linalg.solve(Gr, b.unsqueeze(-1)).squeeze(-1)   # (S,h,w,K)
            res = (I.abs() ** 2).sum(1) - (b.conj() * mm).real.sum(-1)
            loss = res.sum() / self.Inorm_slice[idx].sum()
            return (loss, mm.permute(0, 3, 1, 2)) if return_images else loss
        x = (m[idx][:, :, None] * Cg).sum(1)                            # (S,N,h,w)
        if sl is None or (self.noncart and not self.per_slice_traj):
            op = self.op
        elif self.per_slice_traj:
            op = self.op.subset(sl)
        else:
            op = CartesianOp(self.mk[idx])
        pred = op.forward(x)
        loss = (pred - self.yk[idx]).abs().pow(2).sum() / self.ynorm_slice[idx].sum()
        return (loss, m[idx]) if return_images else loss

    def norm_loss(self, Cs, sl=None):
        C = self.mixC(Cs[0] if Cs.dim() == 3 else Cs, -2)
        S = self.S if sl is None else len(sl)
        sup = self.support if sl is None else self.support[sl]
        n = ((self.to_grid(C, S).abs() ** 2).sum(1) * sup).sum() / sup.sum().clamp_min(1)
        return torch.log(n.clamp_min(1e-12)) ** 2

    def init_image(self, C):
        Cg = self.to_grid(self.mixC(C.detach(), -2))
        return ((Cg.conj() * self.I).sum(1) / (Cg.abs() ** 2).sum(1).clamp_min(1e-8)).clone()


def sens_sets(model, pts, create_graph=True):
    if isinstance(model, MultiSetCoils):
        return model.sensitivity_sets(pts, create_graph)
    return model.sensitivity(pts, create_graph)[None]


def _log(hist, it, **kw):
    hist.append({"it": it, **{k: float(v) for k, v in kw.items()}})


def _batch(prob, cfg, gen):
    if cfg.slice_batch and prob.S > cfg.slice_batch and (not prob.noncart or prob.per_slice_traj):
        return sorted(torch.randperm(prob.S, generator=gen)[:cfg.slice_batch].tolist())
    return None


def _wrap(primary, prob, cfg, prev=None, free=0):
    if not cfg.multiset:
        return primary
    folds = [prob.fold_vec.tolist(), (-prob.fold_vec).tolist()] if cfg.fold_sets else []
    w = MultiSetCoils(primary, folds, cfg.shift_sets, free, cfg.latent, cfg.width, cfg.depth, cfg.seed)
    if prev is not None and isinstance(prev, MultiSetCoils) and prev.shifts.numel():
        with torch.no_grad():
            w.shifts.copy_(prev.shifts)
    return w


def _extra_penalty(model, Cs, m, cfg):
    pen = 0.0
    if isinstance(model, MultiSetCoils) and model.n_free:
        nf = model.n_free
        pen = pen + cfg.lambda_sets * (Cs[-nf:].abs() ** 2).mean() / (Cs[0].abs() ** 2).mean().detach()
    if m is not None:
        pen = pen + cfg.lambda_img * (m[:, 0].abs() ** 2).mean()
        if cfg.lambda_temporal and m.shape[0] > 1:
            pen = pen + cfg.lambda_temporal * (m[1:, 0] - m[:-1, 0]).abs().pow(2).mean() / (m[:, 0].abs() ** 2).mean().detach()
        if m.shape[1] > 1:
            pen = pen + cfg.set_ridge * (m[:, 1:].abs() ** 2).mean() / (m[:, 0].abs() ** 2).mean().detach()
    return pen


def _image_param(prob, C0, K, m_prev=None):
    if m_prev is not None and m_prev.shape[1] == K:
        return torch.nn.Parameter(m_prev.detach().clone())
    m0 = prob.init_image(C0) if m_prev is None else m_prev[:, 0].detach()
    m = torch.zeros(prob.S, K, *prob.calib, dtype=torch.complex64, device=prob.device)
    m[:, 0] = m0
    if m_prev is not None:
        k = min(K, m_prev.shape[1]); m[:, :k] = m_prev[:, :k].detach()
    return torch.nn.Parameter(m)


# ===================================================================================== wires
def fit_wires(prob: Problem, cfg: FitConfig, handedness: int, iters: int, log=print, tag="wires"):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed + 5)
    wires = WireLoopCoils(prob.N_model, cfg.loops_per_coil, cfg.verts_per_loop, handedness)
    wires.init_from_data(prob.coords, prob.weights_model, prob.support_pts,
                         standoff=max(0.15, cfg.outside_margin + 0.05), seed=cfg.seed,
                         normal_axis=prob.normal_axis, z_offset=cfg.init_z_offset)
    wires.init_gains(prob.coords.cpu(), prob.coil_images_model.cpu(),
                     (prob.weights_model * prob.support.flatten()[None]).cpu())
    wires.to(prob.device)
    model = _wrap(wires, prob, cfg)
    params = [{"params": model.parameters(), "lr": cfg.lr_wires}]
    m = None
    if prob.data_term == "kspace":
        with torch.no_grad():
            C = wires.sensitivity(prob.coords, create_graph=False)
            n = ((prob.to_grid(prob.mixC(C)).abs() ** 2).sum(1) * prob.support).sum() / prob.support.sum()
            wires.amp.div_(n.sqrt())
            K = getattr(model, "n_sets", 1)
            m = _image_param(prob, wires.sensitivity(prob.coords, create_graph=False), K)
        params.append({"params": [m], "lr": 2e-2})
    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.05 * cfg.lr_wires)
    hist = []
    sup_sub = prob.support_pts[torch.randperm(prob.support_pts.shape[0], generator=gen)[:2048].to(prob.device)]
    for it in range(iters):
        sl = _batch(prob, cfg, gen)
        opt.zero_grad()
        Cs = sens_sets(model, prob.slice_coords(sl), create_graph=False)
        d = prob.data_loss(Cs, m, sl)
        out, curv = wires.regularisers(sup_sub, cfg.outside_margin)
        loss = d + cfg.lambda_outside * out + cfg.lambda_curv * curv + cfg.lambda_norm * prob.norm_loss(Cs, sl)
        loss = loss + _extra_penalty(model, Cs, m, cfg)
        loss.backward()
        opt.step(); sched.step()
        if it % cfg.log_every == 0 or it == iters - 1:
            _log(hist, it, data=d.item(), outside=out.item(), curv=curv.item())
            log(f"[{tag} h={handedness:+d}] it {it:5d}  data {d.item():.4e}  outside {out.item():.2e}")
    return model, m, hist, float(d.detach())


# ===================================================================================== pinn
def load_trunk(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    return ck


def make_pinn(prob, cfg, wires, h):
    net = None
    codes_init = None
    if cfg.pretrained:
        from .models import make_trunk
        ck = load_trunk(cfg.pretrained, prob.device)
        net = make_trunk(3 + ck["latent"], ck["width"], ck["depth"], 1 if ck["field_mode"] == "real" else 2)
        net.load_state_dict(ck["net"])
        codes_init = ck["codes_mean"]
        cfg.latent, cfg.width, cfg.depth, cfg.field_mode = ck["latent"], ck["width"], ck["depth"], ck["field_mode"]
    pinn = HarmonicPINN(prob.N_model, cfg.field_mode, cfg.latent, cfg.width, cfg.depth, h, base=wires, net=net)
    if codes_init is not None:
        with torch.no_grad():
            pinn.codes.copy_(codes_init[None].expand_as(pinn.codes) + 0.01 * torch.randn_like(pinn.codes))
    return pinn.to(prob.device)


def fit_pinn(prob: Problem, cfg: FitConfig, model, m=None, log=print, iters=None):
    iters = iters or cfg.iters_pinn
    gen = torch.Generator().manual_seed(cfg.seed + 2)
    params = [{"params": model.parameters(), "lr": cfg.lr_pinn}]
    K = getattr(model, "n_sets", 1)
    if prob.data_term == "kspace":
        C0 = model.sensitivity(prob.coords, create_graph=False).detach() if m is None else None
        m = _image_param(prob, C0, K, m)
        params.append({"params": [m], "lr": 1e-2})
    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.1 * cfg.lr_pinn)
    hist = []
    for it in range(iters):
        sl = _batch(prob, cfg, gen)
        opt.zero_grad()
        Cs = sens_sets(model, prob.slice_coords(sl))
        d = prob.data_loss(Cs, m, sl)
        phys = model.physics_loss(prob.collocation(cfg.n_colloc, gen))
        loss = d + cfg.lambda_phys * phys + cfg.lambda_norm * prob.norm_loss(Cs, sl) + _extra_penalty(model, Cs, m, cfg)
        loss.backward(); opt.step(); sched.step()
        if it % cfg.log_every == 0 or it == iters - 1:
            _log(hist, it, data=d.item(), phys=phys.item())
            log(f"[pinn] it {it:5d}  data {d.item():.4e}  laplace {phys.item():.3e}")
    return m, hist


# ===================================================================================== driver
def _fit_free(prob, cfg, history, t0, log):
    """No-physics ablation: smooth neural maps, distilled to low-res division maps, then the same
    data term with a Sobolev smoothness penalty in place of the Biot–Savart constraint."""
    from .models import FreeMaps
    gen = torch.Generator().manual_seed(cfg.seed + 3)
    model = FreeMaps(prob.N_model, cfg.latent, cfg.width, cfg.depth).to(prob.device)
    rss = (prob.I.abs() ** 2).sum(1, keepdim=True).sqrt().clamp_min(1e-6)
    target = (prob.I / rss).permute(1, 0, 2, 3).reshape(prob.N, -1)
    w = prob.support.flatten()[None]
    opt = torch.optim.Adam(model.parameters(), lr=3 * cfg.lr_pinn)
    hist = []
    for it in range(cfg.iters_free_init):
        opt.zero_grad()
        C = prob.mixC(model.sensitivity(prob.coords))
        loss = ((C - target).abs() ** 2 * w).sum() / (w.sum() * prob.N)
        loss.backward(); opt.step()
    log(f"[free] distilled to low-res division maps: {loss.item():.3e}")
    c2 = FitConfig(**{**cfg.__dict__, "lambda_phys": cfg.lambda_sobolev})
    m, history["pinn"] = fit_pinn(prob, c2, model, None, log=log)
    log(f"fit done in {time.time() - t0:.1f}s")
    model.cpu(); prob.to("cpu")
    return model, prob, history, {"wires": None, "handedness": 1, "image_lowres": None if m is None else m.detach().cpu()}


def fit(data: MultiCoilData, cfg: FitConfig, log=print):
    cfg.validate("fit")
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    t0 = time.time()
    data.prewhiten()
    if cfg.compress:
        data.compress(cfg.compress, cfg.calib)
    prob = Problem(data, cfg).to(cfg.device)
    if prob.mix is not None or cfg.compress:
        log(f"channels: {prob.N} data (virtual) channels, modelling {prob.N_model} "
            f"{'physical coils + known mixing' if prob.mix is not None else 'virtual channels directly'}")
    log(f"data: {prob.S} slice(s), {prob.N} coils, calib {prob.calib}, data term '{prob.data_term}', "
        f"{prob.support_pts.shape[0]} support points, device {cfg.device}")
    history = {}

    if cfg.model == "free":
        return _fit_free(prob, cfg, history, t0, log)

    h = cfg.handedness
    if h == 0:
        scores = {}
        for hh in (+1, -1):
            _, _, _, scores[hh] = fit_wires(prob, cfg, hh, cfg.iters_auto, log=lambda *a: None)
        h = min(scores, key=scores.get)
        log(f"handedness auto-selected: {h:+d}  (data loss +1: {scores[1]:.3e}, -1: {scores[-1]:.3e})")
        history["handedness_scores"] = {str(k): v for k, v in scores.items()}

    wires_model = m = None
    if cfg.model == "wires" or cfg.init == "wires":
        wires_model, m, history["wires"], _ = fit_wires(prob, cfg, h, cfg.iters_wires, log=log)
    model = wires_model
    if cfg.model == "wires" and cfg.free_sets:
        log("note: free sets need the PINN (--model pinn); ignored for --model wires")
    if cfg.model == "pinn":
        wires = wires_model.primary if isinstance(wires_model, MultiSetCoils) else wires_model
        pinn = make_pinn(prob, cfg, wires, h)
        model = _wrap(pinn, prob, cfg, prev=wires_model, free=cfg.free_sets)
        m, history["pinn"] = fit_pinn(prob, cfg, model, m, log=log)
    log(f"fit done in {time.time() - t0:.1f}s")
    wires = wires_model.primary if isinstance(wires_model, MultiSetCoils) else wires_model
    # evaluation runs on CPU
    model.cpu(); prob.to("cpu")
    if wires is not None:
        wires.cpu()
    if m is not None:
        m = m.detach().cpu()
    return model, prob, history, {"wires": wires, "handedness": h, "image_lowres": m}
