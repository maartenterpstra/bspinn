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

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import accel
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
    physics: str = "laplace"         # 'laplace' (quasi-static, default) | 'helmholtz' (3-D: wave-aware correction)
    b0_tesla: float = 3.0            # helmholtz: field strength -> Larmor frequency
    eps_r: float = 50.0              # helmholtz: initial relative permittivity of the object
    sigma: float = 0.6               # helmholtz: initial conductivity (S/m)
    learn_k2: bool = True            # helmholtz: fit the complex k^2 (else keep the tissue value)
    lambda_div: float = 1.0          # helmholtz: weight of div B_c = 0 relative to the Helmholtz residual
    correction_grid: int = 12        # 3-D only: the PINN correction is evaluated on this many points per
                                     #   axis and upsampled (trilinear) to the calibration grid in training
    init_z_offset: float = 0.15      # +/- out-of-plane offset of initial loops (breaks z-mirror symmetry; 0 = off)
    outside_margin: float = 0.1      # min. conductor distance from object (normalised; 0.1 ~ 5% of FOV),
                                     #   enforced along every wire segment in every stage
    wire_radius: float = 0.05        # conductor radius (normalised; Rankine core): the field is bounded near
                                     #   the wires instead of ~1/d; exact beyond this distance, hence exact
                                     #   inside the object while wire_radius <= outside_margin. 0 = thin wires
    fov_extend: float = 0.0          # objects cut by the FOV edge are assumed to continue this far beyond it
                                     #   (normalised) for the conductor keep-out
    lambda_fov: float = 0.0          # optional prior: push conductors out of the imaged region (FOV box,
                                     #   2-D: +/- slab around the slice) where the data allow; the field in
                                     #   the air of the image is unobserved, so this only changes the
                                     #   extrapolation outside the object
    joint_full: bool = False         # after calibration: refine maps jointly with a full-resolution image on
                                     #   ALL acquired k-space (NLINV-style; 2-D Cartesian). Maps stay on the
                                     #   calibration grid and are interpolated to full resolution in k-space
    iters_joint: int = 300           # joint full-resolution iterations (map steps)
    joint_maps: str = "hybrid"       # 'hybrid': exact wire field at every pixel + smooth network correction from
                                     #   the calibration grid (k-space upsampled) | 'exact': everything at every
                                     #   pixel | 'interp': whole maps from the calibration grid (biased near wires)
    lambda_joint_prox: float = 0.0   # proximal weight: ||C - C_calib||^2 / ||C_calib||^2 on the calibration grid
                                     #   (keeps the joint maps near the calibration solution)
    lr_joint: float = 0.0            # learning rate of the joint stage (0 = lr_pinn)
    joint_chunk: int = 0             # points per chunk in the exact joint stage (0 = auto: from free GPU memory;
                                     #   one single forward/backward pass when everything fits)
    joint_cg: int = 3                # CG steps on the image per map step (warm-started)
    lambda_joint_img: float = 1e-2   # Tikhonov weight on the image, relative to mean(sum_j |C_j|^2)
    compile: bool | str = False      # torch.compile each stage's whole iteration (see accel.py): False | True |
                                     #   'reduce-overhead' (+ CUDA graphs) | 'max-autotune'; ~10-60 s per stage
    compile_min_iters: int = 200     # stages with fewer iterations run eagerly (not worth compiling)
    cuda_graph: bool = True          # CUDA only: replay each stage's iteration (forward, backward, Adam) from one
                                     #   captured CUDA graph after 3 warm-up iterations (accel.train_graphed);
                                     #   4-7x faster per iteration on an RTX PRO 6000. No effect on CPU
    tf32: bool = False               # allow TF32 matmuls on Ampere+ GPUs (faster, ~1e-3 relative precision)
    air_colloc: float = 0.25         # fraction of Laplace collocation points in the air around the object
                                     #   (>= 2 wire radii from any wire): keeps the network correction harmonic,
                                     #   hence smooth, outside the object too. 0 = object only
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
    # optimiser (all stages; see scripts/hyper_sweep.py)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_schedule: str = "cosine"      # 'cosine' | 'linear' | 'constant' (after the optional warm-up)
    lr_warmup: float = 0.0           # fraction of each stage spent on a linear warm-up
    lr_min_wires: float = 0.05       # final learning rate / initial, wire stage
    lr_min_pinn: float = 0.1         # final learning rate / initial, PINN (and dynamic) stage
    lr_image_wires: float = 2e-2     # learning rate of the calibration-grid image (k-space data term)
    lr_image_pinn: float = 1e-2
    lambda_temporal: float = 0.0     # ||m_t - m_t-1||^2 on frame images (dynamic data)
    support_thresh: float = 0.08
    iters_polish: int = 0            # >0: full-batch L-BFGS refinement of every parameter after the PINN stage
                                     #   (fixed collocation set of polish_colloc points); Adam alone stalls well
                                     #   above the noise floor on high-SNR calibration data
    polish_colloc: int = 4096
    resid_order: int = 0             # >0: per-coil smooth complex factor exp(u_j) of this Fourier order
                                     #   multiplying the physics maps (PINN stage on; see models.SmoothFactor)
    lambda_resid: float = 1e-3       # Sobolev ridge on that factor's coefficients
    learn_mixing: bool | str = False # learnable complex channel mixing C -> (I + E) C (coupling): True = PINN
                                     #   stage, 'all' = from the wire stage on
    lambda_mix: float = 1e-3         # ridge on E
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
        if self.joint_full:
            if context != "fit":
                err.append("--joint-full is implemented for 2-D Cartesian data (fit); not for 3-D, dynamic or shared")
            if self.model not in ("pinn", "wires") or self.multiset:
                err.append("--joint-full needs --model pinn|wires with a single map set")
            if self.joint_maps not in ("exact", "interp", "hybrid"):
                err.append("--joint-maps must be hybrid|exact|interp")
        if context == "dynamic":
            if self.model != "pinn" or self.init != "wires" or self.field_mode != "real":
                err.append("dynamic: needs --model pinn --init wires --field-mode real")
            if self.multiset:
                err.append("dynamic: multiple map sets are not implemented together with dynamics")
        if self.physics not in ("laplace", "helmholtz"):
            err.append("--physics must be laplace|helmholtz")
        if self.physics == "helmholtz":
            if context != "fit3d":
                err.append("--physics helmholtz needs 3-D data (fit3d): on one slice the d^2/dz^2 term is unobserved")
            if self.model not in ("pinn", "wires") or self.init != "wires" or self.pretrained:
                err.append("--physics helmholtz needs --model pinn|wires --init wires")
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
        from .keepout import KeepOut
        self.keep = KeepOut(self.support, self.coords.reshape(S, h, w, 3), volumetric=False, slab=cfg.slab,
                            extend=cfg.fov_extend)
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
        from .keepout import clean_support
        return torch.from_numpy(clean_support((rss > thr).cpu().numpy(), per_slice=True)).float()

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
        self.keep.to(device)
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
        dev = gen.device
        idx = torch.randint(0, self.support_pts.shape[0], (n,), generator=gen, device=dev)
        p = self.support_pts[idx].clone()
        jit = 0.02 * torch.randn(p.shape, generator=gen, device=dev)
        jit[:, self.normal_axis] = (2 * torch.rand(n, generator=gen, device=dev) - 1) * self.cfg.slab
        return p + jit

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
            mm = torch.linalg.solve_ex(Gr, b.unsqueeze(-1), check_errors=False)[0].squeeze(-1)   # (S,h,w,K); no sync
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
        rng = prob.__dict__.setdefault("_slice_rng", np.random.default_rng(cfg.seed + 101))   # host-side: no GPU sync
        return sorted(rng.permutation(prob.S)[:cfg.slice_batch].tolist())
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
    res = model._modules.get("resid") if isinstance(model, torch.nn.Module) else None
    if res is not None:
        pen = pen + cfg.lambda_resid * res.penalty()
    mx = model._modules.get("mixer") if isinstance(model, torch.nn.Module) else None
    if mx is not None:
        pen = pen + cfg.lambda_mix * mx.penalty()
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
    """Wire-loop stage: (model, image parameter or None, history, final data loss)."""
    spec = _wires_setup(prob, cfg, handedness, iters, log, tag)
    return spec["finish"](_run_stage(*spec["stage"]))


def fit_wires_many(items, cfg: FitConfig, iters: int, log=None):
    """Independent wire stages, e.g. the plane / orientation / handedness probes: items = [(prob, h), ...].
    Same results as calling fit_wires for each in turn; on CUDA with graphs the stages are captured once
    each and their replays interleaved on separate streams (small problems: they run concurrently)."""
    import copy
    quiet = log or (lambda *a, **k: None)
    cfg = copy.copy(cfg)
    cfg.compile = False        # short probes on different geometries: compiling each would cost far more
    specs = [_wires_setup(p, cfg, h, iters, quiet, "wires") for p, h in items]
    if len(specs) > 1 and all(_graph_ok(sp["stage"]) for sp in specs):
        outs = accel.train_graphed_many([_graph_job(*sp["stage"]) for sp in specs], log=quiet)
    else:
        outs = [_run_stage(*sp["stage"]) for sp in specs]
    return [sp["finish"](o) for sp, o in zip(specs, outs)]


def _wires_setup(prob, cfg, handedness, iters, log, tag):
    torch.manual_seed(cfg.seed)
    gen = _device_gen(prob, cfg.seed + 5)
    if getattr(cfg, "physics", "laplace") == "helmholtz":
        from .models import WaveWireLoopCoils, tissue_k2
        k2 = tissue_k2(cfg.b0_tesla, cfg.eps_r, cfg.sigma, prob.data.geom.half_fov / 1000.0)
        wires = WaveWireLoopCoils(prob.N_model, cfg.loops_per_coil, cfg.verts_per_loop, handedness,
                                  k2_init=k2, learn_k2=cfg.learn_k2)
        if getattr(prob, "coarse_coords", None) is not None and prob.coarse_shape != prob.calib:
            wires.coarse = (prob.coords, prob.coarse_coords, prob.calib, prob.coarse_shape)
    else:
        wires = WireLoopCoils(prob.N_model, cfg.loops_per_coil, cfg.verts_per_loop, handedness)
    wires.core = float(cfg.wire_radius)
    wires.init_from_data(prob.coords, prob.weights_model, prob.support_pts,
                         standoff=max(0.15, cfg.outside_margin + 0.05), seed=cfg.seed,
                         normal_axis=prob.normal_axis, z_offset=cfg.init_z_offset)
    wires.init_gains(prob.coords.cpu(), prob.coil_images_model.cpu(),
                     (prob.weights_model * prob.support.flatten()[None]).cpu())
    wires.to(prob.device)
    if cfg.learn_mixing == "all":                     # coupling learnt from the wire stage on
        from .models import ChannelMixer
        wires.mixer = ChannelMixer(wires.n_coils).to(prob.device)
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
        params.append({"params": [m], "lr": cfg.lr_image_wires})
    def step_loss(sl):
        Cs = sens_sets(model, prob.slice_coords(sl), create_graph=False)
        d = prob.data_loss(Cs, m, sl)
        out, curv = wires.regularisers(prob.keep, cfg.outside_margin)
        if cfg.lambda_fov:
            from .keepout import fov_penalty
            out = out + cfg.lambda_fov / cfg.lambda_outside * fov_penalty(prob.keep, wires.verts, cfg.outside_margin)
        loss = d + cfg.lambda_outside * out + cfg.lambda_curv * curv + cfg.lambda_norm * prob.norm_loss(Cs, sl)
        loss = loss + _extra_penalty(model, Cs, m, cfg)
        return loss, d, out, curv

    hist = []

    def after(it, o):
        if it % cfg.log_every == 0 or it == iters - 1:
            _, d, out, curv = o
            _log(hist, it, data=d.item(), outside=out.item(), curv=curv.item())
            log(f"[{tag} h={handedness:+d}] it {it:5d}  data {d.item():.4e}  outside {out.item():.2e}")

    stage = (prob, cfg, iters, params, cfg.lr_min_wires * cfg.lr_wires, step_loss, lambda: (),
             [_batch(prob, cfg, gen) for _ in range(iters)], [gen], after)
    return {"stage": stage, "finish": lambda o: (model, m, hist, float(o[1].detach()))}


# ===================================================================================== pinn
def load_trunk(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    return ck


def make_pinn(prob, cfg, wires, h):
    if getattr(cfg, "physics", "laplace") == "helmholtz":
        from .models import HelmholtzPINN, tissue_k2
        L = prob.data.geom.half_fov / 1000.0                                 # mm -> m
        k2 = tissue_k2(cfg.b0_tesla, cfg.eps_r, cfg.sigma, L)
        return HelmholtzPINN(prob.N_model, cfg.latent, cfg.width, cfg.depth, h, base=wires, k2_init=k2,
                             learn_k2=cfg.learn_k2, lambda_div=cfg.lambda_div).to(prob.device)
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
    gen = _device_gen(prob, cfg.seed + 2)
    named = list(model.named_parameters())
    k2p = [q for n, q in named if n.endswith(("k2_re", "k2_im")) and q.requires_grad]
    params = [{"params": [q for n, q in named if not n.endswith(("k2_re", "k2_im"))], "lr": cfg.lr_pinn}]
    if k2p:                                    # helmholtz: k^2 is O(1-10) in normalised units -> its own step size
        params.append({"params": k2p, "lr": 2e-2})
    K = getattr(model, "n_sets", 1)
    wires = wire_modules(model)
    if prob.data_term == "kspace":
        C0 = model.sensitivity(prob.coords, create_graph=False).detach() if m is None else None
        m = _image_param(prob, C0, K, m)
        params.append({"params": [m], "lr": cfg.lr_image_pinn})
    # compiled on a GPU: data grid + collocation in one trunk pass (fewer GEMMs; inductor fuses the rest).
    # Otherwise separate passes: uncompiled, the Laplacian on the data grid costs more than it saves
    # (CUDA graphs: 2.5 -> 3.1 ms per iteration), and on CPU everything is compute-bound.
    joint_pass = getattr(model, "sensitivity_and_physics", None) \
        if (prob.device.type == "cuda" and accel.compile_mode(cfg)) else None

    def step_loss(sl, pin, pair):
        r = joint_pass(prob.slice_coords(sl), pin, pair, cfg.air_colloc) if joint_pass else None
        if r is not None:                    # data grid + collocation in one trunk pass
            Cs, phys = r[0][None], r[1]
        else:
            Cs = sens_sets(model, prob.slice_coords(sl))
            phys = _physics_at(model, cfg, pin, pair)
        d = prob.data_loss(Cs, m, sl)
        loss = d + cfg.lambda_phys * phys + cfg.lambda_norm * prob.norm_loss(Cs, sl) + _extra_penalty(model, Cs, m, cfg)
        for w in wires:                     # the wires keep training here: keep them outside the object too
            out, _ = w.regularisers(prob.keep, cfg.outside_margin)   # (no curvature term: it shrinks loops)
            loss = loss + cfg.lambda_outside * out
            if cfg.lambda_fov:
                from .keepout import fov_penalty
                loss = loss + cfg.lambda_fov * fov_penalty(prob.keep, w.verts, cfg.outside_margin)
        return loss, d, phys

    hist = []

    def after(it, o):
        if it % cfg.log_every == 0 or it == iters - 1:
            _, d, phys = o
            _log(hist, it, data=d.item(), phys=phys.item())
            log(f"[pinn] it {it:5d}  data {d.item():.4e}  laplace {phys.item():.3e}")

    _run_stage(prob, cfg, iters, params, cfg.lr_min_pinn * cfg.lr_pinn, step_loss, lambda: _colloc(prob, cfg, gen, wires),
               [_batch(prob, cfg, gen) for _ in range(iters)], [gen], after)
    return m, hist


def polish(prob, cfg, model, m, log=print):
    """Full-batch L-BFGS on the PINN-stage objective (data + physics on a FIXED collocation set + norm +
    keep-out + extra penalties) over all parameters, image included. Deterministic objective, so a
    quasi-Newton method can converge far below where Adam's stochastic steps stall."""
    gen = _device_gen(prob, cfg.seed + 11)
    wires = wire_modules(model)
    c2 = copy.copy(cfg); c2.n_colloc = cfg.polish_colloc
    pin, pair = _colloc(prob, c2, gen, wires)
    sl = _batch(prob, cfg, gen)
    params = [q for q in model.parameters() if q.requires_grad] + ([m] if m is not None else [])
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg.iters_polish, history_size=50,
                            tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")
    state = {}

    def closure():
        opt.zero_grad(set_to_none=True)
        Cs = sens_sets(model, prob.slice_coords(sl))
        d = prob.data_loss(Cs, m, sl)
        phys = _physics_at(model, cfg, pin, pair)
        loss = d + cfg.lambda_phys * phys + cfg.lambda_norm * prob.norm_loss(Cs, sl) + _extra_penalty(model, Cs, m, cfg)
        for w in wires:
            out, _ = w.regularisers(prob.keep, cfg.outside_margin)
            loss = loss + cfg.lambda_outside * out
        loss.backward()
        state["d"], state["phys"] = float(d.detach()), float(phys.detach())
        return loss
    d0 = None
    with torch.enable_grad():
        closure(); d0 = state["d"]
        opt.step(closure)
    log(f"[polish] L-BFGS {cfg.iters_polish} it: data {d0:.4e} -> {state['d']:.4e}  laplace {state['phys']:.3e}")
    return [{"data0": d0, "data": state["d"], "phys": state["phys"]}]


def _fft2c_t(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def _ifft2c_t(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def upsample_maps(Cg, full):
    """Band-limited (k-space zero-padding) interpolation of maps on the calibration grid (..., h, w)
    to the full matrix, preserving point values."""
    h, w = Cg.shape[-2:]
    H, W = full
    K = _fft2c_t(Cg)
    pr, pc = H // 2 - h // 2, W // 2 - w // 2
    Kp = torch.nn.functional.pad(K, (pc, W - w - pc, pr, H - h - pr))
    return _ifft2c_t(Kp) * float(np.sqrt(H * W / (h * w)))


def _cg_image(C, y, M, box, lam, x0, iters, check=False):
    """Few CG steps of (C^H F^H M F C + lam) x = C^H F^H M y per slice, x restricted to the acquired
    k-space extent ``box`` (no extrapolation). C (S,N,H,W), y (S,N,H,W), M (S,1,H,W), box (H,W)."""
    band = (lambda x: x) if box is None else (lambda x: _ifft2c_t(_fft2c_t(x) * box))
    def A(x):
        return band((C.conj() * _ifft2c_t(M * _fft2c_t(C * x[:, None]))).sum(1)) + lam * x
    b = band((C.conj() * _ifft2c_t(M * y)).sum(1))
    x = band(x0)
    r = b - A(x); p = r.clone(); rs = (r.conj() * r).real.sum()
    for _ in range(iters):
        Ap = A(p)
        al = rs / (p.conj() * Ap).real.sum().clamp_min(1e-30)
        x = x + al * p; r = r - al * Ap
        rn = (r.conj() * r).real.sum()
        if check and rn < 1e-12 * (b.conj() * b).real.sum():                 # host sync: only when asked
            break
        p = r + (rn / rs) * p; rs = rn
    return x


def fit_joint_full(prob: Problem, cfg: FitConfig, model, log=print):
    """Joint refinement of maps and a full-resolution image on all acquired k-space.

    Maps are RSS-normalised per pixel inside this stage (as returned by the API). The image is the
    CG-SENSE solution for the current maps (a few warm-started steps per iteration,
    only inside the acquired k-space extent); the maps take Adam steps on the full-k-space data term,
    with the same physics penalty, norm and conductor keep-out as the PINN stage. Returns the image
    (S, H, W), in the units of prob.ks_full, paired with the RSS-normalised data-channel maps."""
    from .api import acquired_extent
    if prob.noncart:
        raise NotImplementedError("joint_full: Cartesian data only")
    dev = prob.device
    y = prob.ks_full.to(dev)                                               # (S,N,H,W)
    S, N, H, W = y.shape
    M = torch.from_numpy(np.asarray(prob.data.mask, np.float32)).to(dev)   # (S,1,H,W)
    box_np = acquired_extent(np.asarray(prob.data.mask).any(axis=(0, 1)))
    box = None if box_np.all() else torch.from_numpy(box_np).to(dev)
    pts = prob.coords
    gen = _device_gen(prob, cfg.seed + 9)
    wires = wire_modules(model)

    hybrid = cfg.joint_maps == "hybrid" and isinstance(model, HarmonicPINN) and model.base is not None
    exact = cfg.joint_maps in ("exact", "hybrid") and not hybrid           # (wires-only model: exact is cheap)
    pts_full = prob.data.geom.grid().reshape(-1, 3).to(dev)                # (S*H*W, 3)
    n_model = prob.N_model
    chunk = _joint_chunk(cfg, model, n_model, dev, pts_full.shape[0])
    single = (exact or hybrid) and chunk >= pts_full.shape[0]              # whole image in one graph
    if exact:
        log(f"joint full-res: exact maps, {pts_full.shape[0]} points x {n_model} coils, "
            + ("single pass" if single else f"chunks of {chunk} points"))

    def to_full(Cflat):                                                    # (N, S*H*W) -> (S, N, H, W)
        return Cflat.reshape(Cflat.shape[0], S, H, W).permute(1, 0, 2, 3)

    def rssn(C, dim):
        """Per-pixel RSS normalisation over coils: the image variable then lives in the convention of the
        returned (RSS-normalised) maps, so one Tikhonov weight acts uniformly across the FOV (raw maps
        vary ~1000x between deep tissue and the conductors)."""
        return C / (C.abs().pow(2).sum(dim, keepdim=True).sqrt() + 1e-12)

    h_, w_ = prob.calib

    def sens_from_field(B):                                                # CoilModel.sensitivity on a field
        if B.shape[-1] == 1:
            Bx, By = B[..., 0, 0], B[..., 1, 0]
        else:
            Bx = torch.complex(B[..., 0, 0], B[..., 0, 1]); By = torch.complex(B[..., 1, 0], B[..., 1, 1])
        return (Bx + 1j * model.handedness * By) * model.gain()[:, None]

    def wire_field(points):                                               # (N,P,3,K), closed form
        Bw = model.base.field(points)
        if model.K == 2:
            th = model.base.theta
            Bw = torch.cat([Bw * torch.cos(th)[:, None, None, None], Bw * torch.sin(th)[:, None, None, None]], -1)
        return Bw

    def correction_up():                                                   # (N,S*H*W,3,K) from the grid
        Bc = model.correction_field(pts)                                   # (N,S*h*w,3,K), with graph
        Nn, _, _, K = Bc.shape
        g = Bc.reshape(Nn, S, h_, w_, 3 * K).permute(1, 0, 4, 2, 3).reshape(S, Nn * 3 * K, h_, w_)
        up = upsample_maps(g.to(torch.complex64), (H, W)).real              # band-limited interpolation
        return up.reshape(S, Nn, 3 * K, H, W).permute(1, 0, 3, 4, 2).reshape(Nn, S * H * W, 3, K)

    def hybrid_field_nograd():
        with torch.no_grad():
            Bw = torch.cat([wire_field(pts_full[i:i + chunk]) for i in range(0, pts_full.shape[0], chunk)], 1)
            return Bw + correction_up()

    def maps_from_field(F):
        return rssn(to_full(prob.mixC(sens_from_field(F), -2)), 1)

    def maps(create_graph):
        if hybrid and create_graph:
            return None, None                                              # computed in the loop
        if hybrid:
            return maps_from_field(hybrid_field_nograd()), None
        if single and create_graph:
            return None, None                                              # computed with the graph in the loop
        if exact:
            with torch.no_grad():
                Cf = torch.cat([model.sensitivity(pts_full[i:i + chunk], create_graph=False)
                                for i in range(0, pts_full.shape[0], chunk)], -1)
            return rssn(to_full(prob.mixC(Cf, -2)), 1), None
        Cg = model.sensitivity(pts, create_graph=create_graph)            # (N_model, S*h*w)
        return rssn(upsample_maps(prob.to_grid(prob.mixC(Cg, -2)), (H, W)), 1), Cg  # (S,N,H,W)

    def backprop_exact(gC):
        """d(data)/d(theta) = Re <dL/dC, dC/dtheta>, accumulated over pixel chunks (no full graph)."""
        g = gC.permute(1, 0, 2, 3).reshape(gC.shape[1], -1)                # (N_data, S*H*W)
        for i in range(0, pts_full.shape[0], chunk):
            Cc = rssn(prob.mixC(model.sensitivity(pts_full[i:i + chunk], create_graph=True), -2), 0)
            (Cc * g[:, i:i + chunk].conj()).real.sum().backward()

    with torch.no_grad():
        Cg0 = model.sensitivity(pts, create_graph=False).detach()          # calibration solution (prox anchor)
        C0, _ = maps(False)
        lam = cfg.lambda_joint_img * float((C0.abs() ** 2).sum(1).mean())
        x = _cg_image(C0, y, M, box, lam, torch.zeros(S, H, W, dtype=torch.complex64, device=dev), 30, check=True)
    ynorm = float((y.abs() ** 2).sum())
    params = [q for q in model.parameters() if q.requires_grad]
    lr = cfg.lr_joint or cfg.lr_pinn
    hist = []
    state = {}

    def step():
        """One map step for the current image x (partial backward passes happen inside; returns the rest)."""
        C, Cg = maps(True)
        if hybrid and single:                                              # one graph, one wire-field pass
            C = maps_from_field(wire_field(pts_full) + correction_up())
            d = (M * _fft2c_t(C * x[:, None]) - y).abs().pow(2).sum() / ynorm
            Cg = model.sensitivity(pts, create_graph=True)
            rest = d
        elif hybrid:
            # dL/dF on the full-resolution field, then back through (a) the exact wire field in chunks and
            # (b) the grid network + k-space upsampling (one small graph)
            Fv = hybrid_field_nograd().requires_grad_(True)
            Cv = maps_from_field(Fv)
            d = (M * _fft2c_t(Cv * x[:, None]) - y).abs().pow(2).sum() / ynorm
            d.backward()
            gF = Fv.grad
            for i in range(0, pts_full.shape[0], chunk):
                (wire_field(pts_full[i:i + chunk]) * gF[:, i:i + chunk]).sum().backward()
            (correction_up() * gF).sum().backward()
            C = Cv.detach()
            Cg = model.sensitivity(pts, create_graph=True)
            rest = 0.0
        elif single:                                                       # one graph: no recomputation
            C = rssn(to_full(prob.mixC(model.sensitivity(pts_full, create_graph=True), -2)), 1)
            d = (M * _fft2c_t(C * x[:, None]) - y).abs().pow(2).sum() / ynorm
            Cg = model.sensitivity(pts, create_graph=True)
            rest = d
        elif exact:
            Cv = C.detach().requires_grad_(True)
            d = (M * _fft2c_t(Cv * x[:, None]) - y).abs().pow(2).sum() / ynorm
            d.backward()
            backprop_exact(Cv.grad)
            Cg = model.sensitivity(pts, create_graph=True)                 # calibration grid: norm term
            C = Cv.detach()
            rest = 0.0
        else:
            d = (M * _fft2c_t(C * x[:, None]) - y).abs().pow(2).sum() / ynorm
            rest = d
        rest = rest + cfg.lambda_phys * _physics(model, prob, cfg, gen, wires) + cfg.lambda_norm * prob.norm_loss(Cg)
        if cfg.lambda_joint_prox:
            rest = rest + cfg.lambda_joint_prox * (Cg - Cg0).abs().pow(2).sum() / Cg0.abs().pow(2).sum()
        for w in wires:
            out, _ = w.regularisers(prob.keep, cfg.outside_margin)
            rest = rest + cfg.lambda_outside * out
        state["C"] = C
        return rest, d

    def post():                                                            # image update for the new maps
        with torch.no_grad():
            x.copy_(_cg_image(state["C"].detach(), y, M, box, lam, x, cfg.joint_cg))

    def after(it, o):
        if it % cfg.log_every == 0 or it == cfg.iters_joint - 1:
            _log(hist, it, data=o[1].item())
            log(f"[joint full-res] it {it:5d}  data {o[1].item():.4e}")

    if accel.graph_mode(cfg, dev, cfg.iters_joint):                        # the whole iteration, CG included
        opt, base = accel.capturable_adam(params, dev, default_lr=lr)
        accel.train_graphed(step, opt, base, 0.1 * lr, cfg.iters_joint, [gen], None, after,
                            log=log, post=post)
    else:
        opt = accel.adam(params, dev, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.iters_joint, eta_min=0.1 * lr)
        for it in range(cfg.iters_joint):
            opt.zero_grad()
            o = step()
            o[0].backward(); opt.step(); sched.step()
            post()
            after(it, o)
    with torch.no_grad():
        C, _ = maps(False)
        x = _cg_image(C, y, M, box, lam, x, 20, check=True)
    return x.detach(), hist


def _joint_chunk(cfg, model, n_coils, dev, n_points):
    """Points per chunk for exact full-resolution map evaluation with a graph. CPU: the historical
    65536 point-coils per call. CUDA: from the free memory (half of it), possibly all points at once."""
    if cfg.joint_chunk:
        return int(cfg.joint_chunk)
    if dev.type != "cuda":
        return max(256, 65536 // max(1, n_coils))
    free, _ = torch.cuda.mem_get_info(dev)
    width, depth = getattr(cfg, "width", 64), getattr(cfg, "depth", 4)
    # saved activations per point-coil: value + 3 tangents per layer (fp32), x3 for backward temporaries,
    # plus the wire terms (vertices x ~12 floats)
    verts = sum(w.verts[0].numel() // 3 for w in wire_modules(model)) or 12
    bytes_pp = 4 * (4 * width * (depth + 1) * 3 + 12 * verts)
    per_point = bytes_pp * n_coils
    return max(1024, int(0.5 * free // per_point))


def wire_modules(model):
    """All trainable wire-loop modules inside a (wrapped) model."""
    out, seen = [], set()
    for mod in model.modules():
        if isinstance(mod, WireLoopCoils) and id(mod) not in seen and mod.verts.requires_grad:
            seen.add(id(mod)); out.append(mod)
    return out


def air_points(prob, n, gen, wires, min_dist):
    """Collocation points in the air around the object: uniform over the calibration grid (2-D:
    jittered through the slab), outside the object, and at least ``min_dist`` from every wire."""
    if n <= 0:
        return None
    dev = prob.coords.device
    idx = torch.randint(0, prob.coords.shape[0], (4 * n,), generator=gen, device=dev)
    p = prob.coords[idx].clone()
    if "_grid_step" not in prob.__dict__:                     # (setdefault would evaluate -> sync every call)
        prob._grid_step = float((prob.coords[1] - prob.coords[0]).abs().max())
    step = prob._grid_step
    p = p + 0.5 * step * torch.randn(p.shape, generator=gen, device=dev)
    na = getattr(prob, "normal_axis", None)
    if na is not None and prob.__class__.__name__ == "Problem":
        p[:, na] = (2 * torch.rand(p.shape[0], generator=gen, device=dev) - 1) * prob.cfg.slab
    with torch.no_grad():
        keep = prob.keep.distance(p) > 0
        if wires:
            from .keepout import segment_samples
            ws = torch.cat([segment_samples(w.verts.detach(), 4).reshape(-1, 3) for w in wires])
            keep &= torch.cdist(p, ws).min(1).values > min_dist
        # fixed-size draw: admissible candidates first, in random order (static shapes, no host sync;
        # topk is ~50x faster than multinomial without replacement)
        score = keep.float() + torch.rand(keep.shape, generator=gen, device=dev)
        pick = torch.topk(score, n, sorted=False).indices
    return p[pick]


def _device_gen(prob, seed):
    """Random generator on the problem's device: sampling happens where the data are (no host
    round-trip per iteration). On CPU identical to torch.Generator().manual_seed(seed)."""
    return torch.Generator(device=prob.device).manual_seed(seed)


def _colloc(prob, cfg, gen, wires):
    """This iteration's collocation points: (inside the object, in the air or None)."""
    n_air = int(round(cfg.air_colloc * cfg.n_colloc))
    pin = prob.collocation(cfg.n_colloc - n_air, gen)
    pair = air_points(prob, n_air, gen, wires, 2 * max(cfg.wire_radius, 0.02)) if n_air > 0 else None
    return pin, pair


def _physics_at(model, cfg, pin, pair):
    if pair is None:
        return model.physics_loss(pin)
    if hasattr(model, "physics_loss_pair"):                  # both point sets in one network pass
        return model.physics_loss_pair(pin, pair, cfg.air_colloc)
    return model.physics_loss(pin) + cfg.air_colloc * model.physics_loss(pair)


def _physics(model, prob, cfg, gen, wires):
    """Laplace (or Helmholtz) residual of the network correction: inside the object and, with
    weight air_colloc, in the surrounding air (source-free too, as long as we stay off the wires)."""
    return _physics_at(model, cfg, *_colloc(prob, cfg, gen, wires))


def _run_stage(prob, cfg, iters, groups, eta_min, step_loss, draw, sls, gens, after):
    """One training stage: for each iteration,  out = step_loss(sl, *draw()); out[0].backward(); Adam step;
    cosine learning-rate decay to ``eta_min``; after(it, out).  ``sls``: the host-chosen slice batch per
    iteration (lists or None), ``draw``: the iteration's random inputs (device generators ``gens``).
    Runs eagerly or replayed from a CUDA graph (cfg.cuda_graph, CUDA only), with the step optionally compiled
    (cfg.compile; inside the graph: fused kernels replayed with one launch).
    Returns the outputs of the last iteration."""
    dev = prob.device
    if _graph_ok((prob, cfg, iters, groups, eta_min, step_loss, draw, sls, gens, after)):
        return accel.train_graphed(*_graph_job(prob, cfg, iters, groups, eta_min, step_loss, draw, sls, gens, after))
    betas = (cfg.adam_beta1, cfg.adam_beta2)
    opt = accel.adam(groups, dev, betas=betas)
    base = [g["lr"] for g in opt.param_groups]
    step = accel.compile_step(step_loss, cfg, iters)
    compiled = step is not step_loss
    out = None
    for it in range(iters):
        for g, b in zip(opt.param_groups, base):           # same closed-form schedule as the graph path
            g["lr"] = accel.cosine_lr(b, eta_min, it, iters, accel.schedule_of(cfg))
        opt.zero_grad()
        out = step(_sl_arg(prob, sls[it], compiled), *draw())
        out[0].backward(); opt.step()
        after(it, out)
    return out


def _graph_ok(stage):
    prob, cfg, iters, sls = stage[0], stage[1], stage[2], stage[7]
    return accel.graph_mode(cfg, prob.device, iters) and not (sls[0] is not None and getattr(prob, "per_slice_traj", False))


def _graph_job(prob, cfg, iters, groups, eta_min, step_loss, draw, sls, gens, after):
    """Arguments of accel.train_graphed for one stage: capturable Adam, static slice-batch tensor."""
    dev = prob.device
    opt, base = accel.capturable_adam(groups, dev, betas=(cfg.adam_beta1, cfg.adam_beta2))
    opt.schedule = accel.schedule_of(cfg)
    tab = None if sls[0] is None else torch.tensor(np.asarray(sls), device=dev)
    static_sl = None if tab is None else tab[0].clone()

    def before(it):
        if tab is not None:
            static_sl.copy_(tab[it])
    fn = accel.compile_step(step_loss, cfg, iters)               # cfg.compile: fused kernels inside the graph
    return (lambda: fn(static_sl, *draw()), opt, base, eta_min, iters, gens, before, after)


def _sl_arg(prob, sl, compiled):
    """Slice batch for a compiled step: an index tensor (a list would be a compile-time constant and
    recompile for every new batch). Eager: unchanged."""
    if sl is None or not compiled or getattr(prob, "per_slice_traj", False):
        return sl
    return torch.tensor(sl, device=prob.device)


# ===================================================================================== driver
def _fit_free(prob, cfg, history, t0, log):
    """No-physics ablation: smooth neural maps, distilled to low-res division maps, then the same
    data term with a Sobolev smoothness penalty in place of the Biot–Savart constraint."""
    from .models import FreeMaps
    gen = _device_gen(prob, cfg.seed + 3)
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
    # the ablation's smoothness term acts on the object only (as defined), not on the air collocation
    c2 = FitConfig(**{**cfg.__dict__, "lambda_phys": cfg.lambda_sobolev, "air_colloc": 0.0})
    m, history["pinn"] = fit_pinn(prob, c2, model, None, log=log)
    log(f"fit done in {time.time() - t0:.1f}s")
    model.cpu(); prob.to("cpu")
    return model, prob, history, {"wires": None, "handedness": 1, "image_lowres": None if m is None else m.detach().cpu()}


def fit(data: MultiCoilData, cfg: FitConfig, log=print):
    cfg.validate("fit")
    from .accel import configure
    configure(cfg)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    t0 = time.time()
    with accel.phase("problem_setup", cfg.device):
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
        with accel.phase("handedness_search", cfg.device):
            r = fit_wires_many([(prob, +1), (prob, -1)], cfg, cfg.iters_auto)
            scores = {+1: r[0][3], -1: r[1][3]}
        h = min(scores, key=scores.get)
        log(f"handedness auto-selected: {h:+d}  (data loss +1: {scores[1]:.3e}, -1: {scores[-1]:.3e})")
        history["handedness_scores"] = {str(k): v for k, v in scores.items()}

    wires_model = m = None
    if cfg.model == "wires" or cfg.init == "wires":
        with accel.phase("wires", cfg.device):
            wires_model, m, history["wires"], _ = fit_wires(prob, cfg, h, cfg.iters_wires, log=log)
    model = wires_model
    if cfg.model == "wires" and cfg.free_sets:
        log("note: free sets need the PINN (--model pinn); ignored for --model wires")
    if cfg.model == "pinn":
        wires = wires_model.primary if isinstance(wires_model, MultiSetCoils) else wires_model
        with accel.phase("pinn", cfg.device):
            pinn = make_pinn(prob, cfg, wires, h)
            if cfg.resid_order > 0:
                from .models import SmoothFactor
                pinn.resid = SmoothFactor(pinn.n_coils, prob.keep.comp, cfg.resid_order).to(prob.device)
            if wires is not None and wires._modules.get("mixer") is not None:
                pinn.mixer = wires._modules["mixer"]               # continue from the wire stage's coupling
            elif cfg.learn_mixing:
                from .models import ChannelMixer
                pinn.mixer = ChannelMixer(pinn.n_coils).to(prob.device)
            model = _wrap(pinn, prob, cfg, prev=wires_model, free=cfg.free_sets)
            m, history["pinn"] = fit_pinn(prob, cfg, model, m, log=log)
        if cfg.iters_polish > 0:
            with accel.phase("polish", cfg.device):
                history["polish"] = polish(prob, cfg, model, m, log=log)
    image_full = None
    if cfg.joint_full:
        with accel.phase("joint_image", cfg.device):
            image_full, history["joint"] = fit_joint_full(prob, cfg, model, log=log)
    log(f"fit done in {time.time() - t0:.1f}s")
    wires = wires_model.primary if isinstance(wires_model, MultiSetCoils) else wires_model
    # evaluation runs on CPU
    model.cpu(); prob.to("cpu")
    if wires is not None:
        wires.cpu()
    if m is not None:
        m = m.detach().cpu()
    return model, prob, history, {"wires": wires, "handedness": h, "image_lowres": m,
                                  "image_full": None if image_full is None else image_full.cpu()}
