"""Volumetric (3-D encoded) Biot–Savart coil sensitivity estimation.

Same model and objective as the 2-D code, with a genuinely 3-D problem:

* data: 3-D Cartesian k-space (N, D, H, W), 3-D FFT encoding, a 3-D calibration block;
* the field (wires + harmonic PINN) is fitted on a 3-D grid, and the Laplace penalty is applied
  at collocation points throughout the imaged volume, not in a thin slab around a slice;
* the PINN correction is evaluated on a coarse grid (``FitConfig.correction_grid`` points per
  axis) and upsampled trilinearly to the calibration grid during training. The correction is a
  smooth harmonic field, so this costs little accuracy but makes CPU training feasible; final
  maps evaluate the network exactly at every voxel.

Everything else (handedness selection, wire fit, PINN fit, the oversampled calibration grid,
noise pre-whitening with physical-coil modelling) reuses ``bspinn.fit``.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .fit import FitConfig, fit_pinn, fit_wires, make_pinn, _image_param  # noqa: F401
from .models import CoilModel, HarmonicPINN

AXES = {"x": 0, "y": 1, "z": 2}


def fftnc(x, dims=(-3, -2, -1)):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)


def ifftnc(x, dims=(-3, -2, -1)):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=dims), dim=dims, norm="ortho"), dim=dims)


def fftnc_np(x, axes=(-3, -2, -1)):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)


def ifftnc_np(x, axes=(-3, -2, -1)):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)


def crop3(x, size):
    D, H, W = x.shape[-3:]
    d, h, w = size
    return x[..., D // 2 - d // 2: D // 2 - d // 2 + d, H // 2 - h // 2: H // 2 - h // 2 + h, W // 2 - w // 2: W // 2 - w // 2 + w]


def pad3(x, size):
    out = torch.zeros(*x.shape[:-3], *size, dtype=x.dtype, device=x.device)
    d, h, w = x.shape[-3:]
    D, H, W = size
    out[..., D // 2 - d // 2: D // 2 - d // 2 + d, H // 2 - h // 2: H // 2 - h // 2 + h, W // 2 - w // 2: W // 2 - w // 2 + w] = x
    return out


# ===================================================================================== geometry / data
@dataclass
class VolumeGeometry:
    """shape: (D, H, W) array axes; voxel: mm along each array axis; axes: the physical axis
    (x, y or z; B0 = z) of each array axis, e.g. "zyx" = slices along B0, rows along y, cols along x."""
    shape: tuple
    voxel: tuple = (1.0, 1.0, 1.0)
    axes: str = "zyx"

    def __post_init__(self):
        assert sorted(self.axes) == ["x", "y", "z"], "axes must be a permutation of 'xyz'"
        self.shape = tuple(int(v) for v in self.shape)
        self.voxel = tuple(float(v) for v in self.voxel)

    @property
    def half_fov(self):
        return max(n * v for n, v in zip(self.shape, self.voxel)) / 2

    def grid(self, shape=None) -> torch.Tensor:
        """Normalised physical coordinates (x, y, z) of voxel centres of a grid spanning the same
        FOV at resolution ``shape``: (d, h, w, 3)."""
        shape = shape or self.shape
        cs = []
        for n_full, n, v in zip(self.shape, shape, self.voxel):
            dv = v * n_full / n
            cs.append((np.arange(n) - n // 2) * dv / self.half_fov)
        mesh = np.meshgrid(*cs, indexing="ij")
        out = np.zeros(tuple(shape) + (3,), np.float32)
        for a, ax in enumerate(self.axes):
            out[..., AXES[ax]] = mesh[a]
        return torch.from_numpy(out)


@dataclass
class VolumeData:
    """kspace: (N, D, H, W) complex; mask: (D, H, W) bool (None: non-zero samples)."""
    kspace: np.ndarray
    geom: VolumeGeometry
    mask: np.ndarray | None = None
    noise: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.kspace = np.asarray(self.kspace, np.complex64)
        if self.mask is None:
            self.mask = np.abs(self.kspace).sum(0) > 0
        self.mask = np.asarray(self.mask, bool)

    @property
    def n_coils(self):
        return self.kspace.shape[0]

    def prewhiten(self):
        if self.noise is None:
            return self
        n = self.noise.reshape(-1, self.n_coils)
        psi = (n.conj().T @ n) / n.shape[0]
        Linv = np.linalg.inv(np.linalg.cholesky(psi))
        self.kspace = np.einsum("ij,j...->i...", Linv, self.kspace).astype(np.complex64)
        self.meta["noise_cov"] = psi
        prev = self.meta.get("mix")
        self.meta["mix"] = (Linv if prev is None else Linv @ prev).astype(np.complex64)
        self.noise = None
        return self


def load_volume(path, voxel=(1.0, 1.0, 1.0), axes="zyx", noise_path=None):
    """BART .cfl (X, Y, Z, COIL) or .npz with 'ksp' (N, D, H, W) [+ 'mask', 'noise' (Ns, N)].
    BART X, Y, Z become array axes D, H, W."""
    from .data import _read_cfl
    noise = mask = None
    if path.endswith(".npz"):
        z = np.load(path)
        k = z["ksp"]; mask = z["mask"] if "mask" in z else None; noise = z["noise"] if "noise" in z else None
    else:
        k = _read_cfl(path[:-4] if path.endswith((".cfl", ".hdr")) else path)
        k = k.reshape(k.shape[:4])
        k = np.transpose(k, (3, 0, 1, 2))
    if noise_path:
        nz = _read_cfl(noise_path) if not noise_path.endswith(".npy") else np.load(noise_path)
        noise = np.asarray(nz).reshape(-1, k.shape[0])
    return VolumeData(k, VolumeGeometry(k.shape[1:], voxel, axes), mask=mask, noise=noise, meta={"source": path})


# ===================================================================================== problem
class Problem3D:
    """Interface-compatible with ``bspinn.fit.Problem`` (S = 1 'slice' = the whole volume) so the
    wire and PINN training loops are shared with the 2-D code."""

    noncart = False
    per_slice_traj = False
    calib_only = False

    def __init__(self, data: VolumeData, cfg: FitConfig):
        self.data, self.cfg = data, cfg
        geom = data.geom
        D, H, W = geom.shape
        c = cfg.calib if isinstance(cfg.calib, (tuple, list)) else (cfg.calib,) * 3
        kc = tuple(min(int(a), b) for a, b in zip(c, (D, H, W)))
        f = max(1, int(cfg.calib_oversample))
        g = tuple(min(f * a, b) for a, b in zip(kc, (D, H, W)))
        self.kcalib, self.calib = kc, g
        ks = torch.from_numpy(data.kspace)
        mask = torch.from_numpy(data.mask)
        yk = pad3(crop3(ks, kc), g)
        mk = pad3(crop3(mask.to(torch.float32), kc), g)
        I0 = ifftnc(yk * mk)
        rss = (I0.abs() ** 2).sum(0).sqrt()
        self.scale = float(torch.quantile(rss.flatten()[torch.randperm(rss.numel())[:200000]], 0.99))
        self.S, self.N = 1, data.n_coils
        self.yk = (yk * mk)[None] / self.scale                                   # (1,N,gd,gh,gw)
        self.mk = mk[None]                                                        # (1,gd,gh,gw)
        self.I = I0[None] / self.scale
        self.noise_sigma = self._estimate_noise(data) / self.scale
        self.support = self.object_support(rss[None] / self.scale)
        mix = data.meta.get("mix")
        self.mix = (torch.as_tensor(mix, dtype=torch.complex64)
                    if (mix is not None and cfg.model_channels == "physical") else None)
        self.N_model = self.mix.shape[1] if self.mix is not None else self.N
        self.coords = geom.grid(g).reshape(-1, 3)
        self.support_pts = self.coords[self.support.flatten() > 0]
        self.coil_images = self.I[0].reshape(self.N, -1)
        self.weights = self.coil_images.abs() ** 2
        self.coil_images_model = (torch.linalg.pinv(self.mix) @ self.coil_images) if self.mix is not None else self.coil_images
        self.weights_model = self.coil_images_model.abs() ** 2
        fully = bool(crop3(mask, kc).all()) and g == kc
        self.data_term = cfg.data_term if cfg.data_term != "auto" else ("varpro" if fully else "kspace")
        if self.data_term == "varpro" and not fully:
            raise ValueError("varpro needs a fully sampled calibration block and calib_oversample=1")
        self.normal_axis = AXES[geom.axes[0]]
        self.Inorm_slice = (self.I.abs() ** 2).reshape(1, -1).sum(1)
        self.ynorm_slice = (self.yk.abs() ** 2).reshape(1, -1).sum(1)
        self.fold_vec = torch.zeros(3)
        lo, hi = self.coords.min(0).values, self.coords.max(0).values
        self.coarse_shape = tuple(min(cfg.correction_grid, n) for n in g)
        self.coarse_coords = self._coarse_grid(lo, hi)
        self.device = torch.device("cpu")

    def _coarse_grid(self, lo, hi):
        """Coarse grid with the same corner points as the calibration grid (array-axis order)."""
        geom = self.data.geom
        cs = []
        for a, n in enumerate(self.coarse_shape):
            p = AXES[geom.axes[a]]
            cs.append(torch.linspace(float(lo[p]), float(hi[p]), n))
        mesh = torch.meshgrid(*cs, indexing="ij")
        out = torch.zeros(*self.coarse_shape, 3)
        for a in range(3):
            out[..., AXES[geom.axes[a]]] = mesh[a]
        return out.reshape(-1, 3)

    @staticmethod
    def _estimate_noise(data):
        if "noise_cov" in data.meta:
            return 1.0
        D, H, W = data.mask.shape
        r = np.sqrt(sum(np.meshgrid(*[((np.arange(n) - n // 2) / (n / 2)) ** 2 for n in (D, H, W)], indexing="ij")))
        sel = data.mask & (r > 0.95)
        v = np.abs(data.kspace[:, sel]).ravel()
        return float(np.quantile(v, 0.25) / np.sqrt(-np.log(0.75))) if v.size else 0.0

    def object_support(self, rss):
        thr = self.cfg.support_thresh * float(torch.quantile(rss.flatten()[torch.randperm(rss.numel())[:200000]], 0.99))
        thr = max(thr, 3.0 * self.noise_sigma * math.sqrt(self.N))
        return (rss > thr).float()

    _tensors = ("yk", "mk", "I", "support", "coords", "support_pts", "weights", "coil_images", "mix",
                "coil_images_model", "weights_model", "Inorm_slice", "ynorm_slice", "fold_vec", "coarse_coords")

    def to(self, device):
        device = torch.device(device)
        for name in self._tensors:
            v = getattr(self, name, None)
            if torch.is_tensor(v):
                setattr(self, name, v.to(device))
        self.device = device
        return self

    def mixC(self, C, dim=-2):
        if self.mix is None:
            return C
        return (C.movedim(dim, -1) @ self.mix.T).movedim(-1, dim)

    def slice_coords(self, sl=None):
        return self.coords

    def to_grid(self, C, S=None):
        """(..., N, P) -> (1, ..., N, gd, gh, gw)"""
        lead = C.shape[:-2]
        return C.reshape(*lead, self.N, *self.calib)[None]

    def collocation(self, n, gen):
        """Points throughout the imaged volume (the PDE domain is genuinely 3-D here)."""
        idx = torch.randint(0, self.support_pts.shape[0], (n,), generator=gen)
        p = self.support_pts[idx.to(self.support_pts.device)].clone()
        return p + (0.5 * (self.coords[1] - self.coords[0]).abs().max() * torch.randn(p.shape, generator=gen)).to(p.device)

    def data_loss(self, Cs, m=None, sl=None, return_images=False):
        if Cs.dim() == 2:
            Cs = Cs[None]
        Cs = self.mixC(Cs, -2)
        if Cs.shape[0] != 1:
            raise NotImplementedError("3-D: one map set")
        Cg = self.to_grid(Cs)                                            # (1,1,N,gd,gh,gw)
        if self.data_term == "varpro":
            C1 = Cg[:, 0]
            b = (C1.conj() * self.I).sum(1)
            den = (C1.abs() ** 2).sum(1)
            den = den + 1e-6 * den.mean().detach()
            res = (self.I.abs() ** 2).sum(1) - b.abs() ** 2 / den
            loss = res.sum() / self.Inorm_slice.sum()
            return (loss, (b / den)[:, None]) if return_images else loss
        x = (m[:, :, None] * Cg).sum(1)                                  # (1,N,gd,gh,gw)
        pred = fftnc(x) * self.mk[:, None]
        loss = (pred - self.yk).abs().pow(2).sum() / self.ynorm_slice.sum()
        return (loss, m) if return_images else loss

    def norm_loss(self, Cs, sl=None):
        C = self.mixC(Cs[0] if Cs.dim() == 3 else Cs, -2)
        n = ((self.to_grid(C).abs() ** 2).sum(1) * self.support).sum() / self.support.sum().clamp_min(1)
        return torch.log(n.clamp_min(1e-12)) ** 2

    def init_image(self, C):
        Cg = self.to_grid(self.mixC(C.detach(), -2))
        return ((Cg.conj() * self.I).sum(1) / (Cg.abs() ** 2).sum(1).clamp_min(1e-8)).clone()


# ===================================================================================== coarse correction
class CoarseCorrection(CoilModel):
    """Training-time wrapper: B = B_wires (exact, calibration grid) - grad Phi_NN evaluated on a
    coarse grid and trilinearly upsampled. Any other point set is evaluated exactly."""

    def __init__(self, pinn: HarmonicPINN, prob: Problem3D):
        super().__init__()
        self.pinn = pinn
        self.n_coils, self.handedness = pinn.n_coils, pinn.handedness
        self._fine = prob.coords
        self._coarse = prob.coarse_coords
        self._fine_shape, self._coarse_shape = prob.calib, prob.coarse_shape

    @property
    def base(self):
        return self.pinn.base

    def gain(self):
        return self.pinn.gain()

    def field(self, points, create_graph=True):
        if points is not self._fine or self._coarse_shape == self._fine_shape:
            return self.pinn.field(points, create_graph)
        Bw = self.pinn._base_field(points)                                       # (N,P,3,K)
        Bc = self.pinn.correction_field(self._coarse, create_graph)              # (N,Pc,3,K)
        N, K = Bc.shape[0], Bc.shape[-1]
        g = Bc.reshape(N, *self._coarse_shape, 3 * K).permute(0, 4, 1, 2, 3)
        g = F.interpolate(g, size=self._fine_shape, mode="trilinear", align_corners=True)
        Bc = g.permute(0, 2, 3, 4, 1).reshape(N, -1, 3, K)
        return Bc if Bw is None else Bw + Bc

    def physics_loss(self, points):
        return self.pinn.physics_loss(points)


# ===================================================================================== driver
B0_ORIENTATIONS = {0: "zyx", 1: "yzx", 2: "yxz"}      # B0 (z) along array axis 0 / 1 / 2


def select_orientation(data: VolumeData, cfg: FitConfig, log=print):
    """Unknown B0 direction: short wire fits with B0 along each array axis (both handednesses);
    the orientation with the lowest calibration data loss wins. Returns the axes string."""
    import copy
    scores = {}
    for ax, axes in B0_ORIENTATIONS.items():
        d = copy.copy(data)
        d.geom = VolumeGeometry(data.geom.shape, data.geom.voxel, axes)
        prob = Problem3D(d, cfg).to(cfg.device)
        scores[axes] = min(fit_wires(prob, cfg, hh, cfg.iters_auto, log=lambda *a: None)[3] for hh in (1, -1))
    best = min(scores, key=scores.get)
    log("B0 orientation scores (data loss): " + ", ".join(f"{a} {v:.3e}" for a, v in scores.items()) + f" -> {best}")
    return best, scores


def fit3d(data: VolumeData, cfg: FitConfig, log=print, orientation: str = "given"):
    """Fit the Biot–Savart model to 3-D data. Returns (model, prob, history, extra).
    ``orientation='auto'`` picks the B0 axis from the data (see ``select_orientation``)."""
    cfg.validate("fit3d")
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    t0 = time.time()
    data.prewhiten()
    orient_scores = None
    if orientation == "auto":
        axes, orient_scores = select_orientation(data, cfg, log)
        data.geom = VolumeGeometry(data.geom.shape, data.geom.voxel, axes)
    prob = Problem3D(data, cfg).to(cfg.device)
    log(f"3-D data: {prob.N} coils, volume {data.geom.shape}, calib block {prob.kcalib} on a {prob.calib} grid, "
        f"data term '{prob.data_term}', {prob.support_pts.shape[0]} support points, correction grid {prob.coarse_shape}")
    hist = {"orientation": data.geom.axes, "orientation_scores": orient_scores}
    h = cfg.handedness
    if h == 0:
        sc = {hh: fit_wires(prob, cfg, hh, cfg.iters_auto, log=lambda *a: None)[3] for hh in (1, -1)}
        h = min(sc, key=sc.get)
        hist["handedness_scores"] = {str(k): v for k, v in sc.items()}
        log(f"handedness {h:+d}  (data loss +1: {sc[1]:.3e}, -1: {sc[-1]:.3e})")
    wires, m, hist["wires"], _ = fit_wires(prob, cfg, h, cfg.iters_wires, log=log)
    t_wires = time.time() - t0
    model = wires
    snapshot = None
    if cfg.model == "pinn":
        import copy
        snapshot = copy.deepcopy(wires).cpu()          # the wire stage alone (the PINN co-adapts its wires)
        pinn = make_pinn(prob, cfg, wires, h)
        wrapped = CoarseCorrection(pinn, prob)
        m, hist["pinn"] = fit_pinn(prob, cfg, wrapped, m, log=log)
        model = pinn
    log(f"3-D fit done in {time.time() - t0:.0f}s")
    model.cpu(); prob.to("cpu")
    return model, prob, hist, {"wires": wires.cpu(), "wires_stage": snapshot or wires, "t_wires": t_wires,
                               "handedness": h, "image_lowres": None if m is None else m.detach().cpu()}


def volume_maps(model, prob: Problem3D, shape=None, chunk=16384):
    """Sensitivities of the data channels on the full (or any) grid: (N, D, H, W) complex.
    Wave models (Helmholtz) are evaluated exactly as they were trained: closed-form parts at every
    voxel, the smooth parts (wave-wire remainder, network correction) from the coarse grid, upsampled."""
    from .models import HelmholtzPINN, WaveWireLoopCoils
    geom = prob.data.geom
    shape = tuple(shape or geom.shape)
    pts = geom.grid(shape).reshape(-1, 3)
    if isinstance(model, (HelmholtzPINN, WaveWireLoopCoils)):
        C = _wave_maps(model, prob, pts, shape, chunk)
    else:
        C = model.sensitivity_batched(pts, chunk)
    C = prob.mixC(C, -2)
    return C.reshape(prob.N, *shape)


def _upsample(F, cs, fs):
    """(N, Pc, 3, 2) on the coarse grid -> (N, P, 3, 2) on grid ``fs`` (trilinear, same corners)."""
    N = F.shape[0]
    g = F.reshape(N, *cs, 6).permute(0, 4, 1, 2, 3)
    g = F_interp(g, fs)
    return g.permute(0, 2, 3, 4, 1).reshape(N, -1, 3, 2)


def F_interp(g, fs):
    return F.interpolate(g, size=fs, mode="trilinear", align_corners=True)


@torch.no_grad()
def _wave_maps(model, prob, pts, shape, chunk):
    from .models import HelmholtzPINN
    from .physics import transverse_to_sensitivity
    wires = model.base if isinstance(model, HelmholtzPINN) else model
    geom = prob.data.geom
    cs = prob.coarse_shape
    pc = prob.coarse_coords                                                        # training's coarse grid
    # smooth parts on that grid
    rem = torch.view_as_real(wires._wave(pc, False, None))                         # (N,Pc,3,2)
    smooth = rem
    if isinstance(model, HelmholtzPINN):
        th = wires.theta
        c, s_ = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
        re, im = rem[..., 0], rem[..., 1]
        smooth = torch.stack([re * c - im * s_, re * s_ + im * c], -1) + model.correction_field(pc)
    # trilinear lookup at every voxel (clamped at the coarse box edge), identical to training's upsampling
    N = smooth.shape[0]
    vol = smooth.reshape(N, *cs, 6).permute(0, 4, 1, 2, 3)                         # (N,6,cd,ch,cw)
    pcg = pc.reshape(*cs, 3)
    lohi = []
    for a in range(3):
        ph_ = AXES[geom.axes[a]]
        lohi.append((float(pcg[..., ph_].min()), float(pcg[..., ph_].max()), ph_))
    u = torch.stack([2 * (pts[:, ph_] - lo_) / (hi_ - lo_) - 1 for lo_, hi_, ph_ in lohi], -1)   # array order d,h,w
    grid = u.flip(-1).reshape(1, -1, 1, 1, 3).expand(N, -1, -1, -1, -1)            # grid_sample wants (w,h,d)
    up = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)   # (N,6,P,1,1)
    up = up[..., 0, 0].permute(0, 2, 1).reshape(N, -1, 3, 2)
    out = []
    for i in range(0, pts.shape[0], chunk):
        p = pts[i:i + chunk]
        st = WireLoopCoils_field(wires, p) + wires._wave_1r(p, None)                # complex (N,p,3)
        stc = torch.view_as_real(st)
        if isinstance(model, HelmholtzPINN):
            th = wires.theta
            c, s_ = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
            stc = torch.stack([stc[..., 0] * c - stc[..., 1] * s_, stc[..., 0] * s_ + stc[..., 1] * c], -1)
        B = stc + up[:, i:i + chunk]
        Bx = torch.complex(B[:, :, 0, 0], B[:, :, 0, 1]); By = torch.complex(B[:, :, 1, 0], B[:, :, 1, 1])
        out.append(transverse_to_sensitivity(Bx, By, model.handedness) * model.gain()[:, None])
    return torch.cat(out, 1)


def WireLoopCoils_field(wires, p):
    """Closed-form quasi-static (Biot-Savart) part of a wave-wire model, as complex (N,P,3)."""
    from .models import WireLoopCoils
    return WireLoopCoils.field(wires, p, False)[..., 0].to(torch.complex64)


# ===================================================================================== synthetic data
def ellipsoid_phantom(shape, seed=0):
    """3-D head-like phantom of ellipsoids (Shepp–Logan in spirit) on a (D,H,W) grid in [-1,1]^3."""
    D, H, W = shape
    z, y, x = np.meshgrid(np.linspace(-1, 1, D), np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    E = [  # (cx, cy, cz, ax, ay, az, value)
        (0, 0, 0, .69, .82, .72, 1.0), (0, -.02, 0, .65, .78, .68, -.6), (.22, 0, -.1, .11, .31, .22, -.2),
        (-.22, 0, -.1, .16, .41, .28, -.2), (0, .35, .2, .21, .25, .35, .15), (0, .1, -.3, .046, .046, .05, .15),
        (-.08, -.6, .15, .046, .023, .02, .15), (.06, -.6, .15, .023, .046, .02, .15), (0, -.3, .4, .1, .05, .1, .25)]
    img = np.zeros(shape, np.float32)
    for cx, cy, cz, ax, ay, az, v in E:
        img += v * ((((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2 + ((z - cz) / az) ** 2) <= 1)
    return img


def wave_loop_fields(verts, points, k2, n_r=8, n_phi=32, chunk=2048):
    """Full-wave field of planar circular current loops in a homogeneous lossy medium.

    Each loop is replaced by the equivalent uniform magnetic-dipole sheet over its disc; every dipole
    field is an exact time-harmonic Maxwell solution, B = 1/(4 pi) e^{-ikR} [k^2 (R^ x m) x R^ / R
    + (3 R^ (R^.m) - m)(1/R^3 + i k / R^2)], so the sum satisfies div B = 0 and lap B + k^2 B = 0 (units mu0 I / 4 pi)
    everywhere away from the coils. k^2 = 0 gives the quasi-static Biot-Savart field of the loops.
    verts: (N, 1, M, 3) loop polygons (circles); points: (P, 3); k2 in normalised units.
    Returns complex (N, P, 3)."""
    k = complex(np.sqrt(complex(k2)))
    if k.imag > 0:
        k = -k                                                     # outgoing, decaying for e^{-ikR}
    V = torch.as_tensor(verts, dtype=torch.float64)[:, 0]           # (N, M, 3)
    c = V.mean(1)                                                   # loop centres
    e1 = V[:, 0] - c
    a = e1.norm(dim=-1, keepdim=True)
    e1 = e1 / a
    e2 = V[:, V.shape[1] // 4] - c
    e2 = e2 - (e2 * e1).sum(-1, keepdim=True) * e1
    e2 = e2 / e2.norm(dim=-1, keepdim=True)
    n = torch.linalg.cross(e1, e2)
    # orientation: same circulation as the polygon (right-hand rule)
    circ = torch.linalg.cross(V[:, 0] - c, V[:, 1] - c)
    n = n * torch.sign((circ * n).sum(-1, keepdim=True))
    rho = (torch.arange(n_r, dtype=torch.float64) + 0.5) / n_r
    phi = torch.arange(n_phi, dtype=torch.float64) * 2 * math.pi / n_phi
    R_, P_ = torch.meshgrid(rho, phi, indexing="ij")
    dA = (math.pi / n_phi) * ((R_ + 0.5 / n_r) ** 2 - (R_ - 0.5 / n_r) ** 2)          # annulus sector areas (unit radius)
    pos = c[:, None, :] + a[:, None, :] * (R_.reshape(-1, 1) * torch.cos(P_).reshape(-1, 1) * e1[:, None, :]
                                           + R_.reshape(-1, 1) * torch.sin(P_).reshape(-1, 1) * e2[:, None, :])  # (N, D, 3)
    m = n[:, None, :] * (dA.reshape(1, -1, 1) * a[:, None, :] ** 2)                      # (N, D, 3), unit current
    kk = torch.tensor(k, dtype=torch.complex128)
    pts = torch.as_tensor(points, dtype=torch.float64)
    res = np.zeros((pos.shape[0], pts.shape[0], 3), np.complex128)
    for j in range(pos.shape[0]):                       # one coil at a time, points in chunks (memory)
        for i in range(0, pts.shape[0], chunk):
            R = pts[None, i:i + chunk, :] - pos[j][:, None, :]                          # (D, P, 3)
            r = R.norm(dim=-1, keepdim=True)
            u = R / r
            mm = m[j][:, None, :].expand_as(u)
            um = (u * mm).sum(-1, keepdim=True)
            far = torch.linalg.cross(torch.linalg.cross(u, mm), u)
            near = 3 * u * um - mm
            ph = torch.exp(-1j * kk * r)                # units of mu0 I / 4 pi, as biot_savart_polyline
            B = ph * (kk ** 2 * far / r + near * (1 / r ** 3 + 1j * kk / r ** 2))
            res[j, i:i + chunk] = B.sum(0).numpy()
    return res


def make_synthetic3d(n_coils=12, shape=(64, 64, 64), noise=0.01, mismatch=0.0, seed=0, n_rings=2,
                     wave_k2=None, voxel=(1.0, 1.0, 1.0), normalise="volume"):
    """3-D data whose sensitivities are Biot–Savart fields of loops on a cylinder around B0 (z),
    in ``n_rings`` rings along z. ``mismatch`` > 0 multiplies each coil by its own smooth complex
    non-harmonic factor (model-mismatch test). ``wave_k2`` (normalised units, e.g. from
    ``models.tissue_k2``) gives the full-wave fields of the same loops in homogeneous tissue instead
    (``wave_loop_fields``). Returns (VolumeData, gt_maps (N,D,H,W), object)."""
    from .data import synthetic_array
    from .physics import biot_savart_polyline
    geom = VolumeGeometry(shape, voxel, "zyx")
    pts = geom.grid().reshape(-1, 3)
    verts, ph = synthetic_array(n_coils, n_rings=n_rings, seed=seed)
    if wave_k2 is not None:
        B = wave_loop_fields(verts, pts, wave_k2)                                       # complex (N,P,3)
    else:
        B = np.concatenate([biot_savart_polyline(torch.from_numpy(verts), pts[i:i + 65536]).sum(1).numpy()
                            for i in range(0, pts.shape[0], 65536)], axis=1)            # (N,P,3)
    C = np.exp(1j * ph)[:, None] * (B[..., 0] + 1j * B[..., 1])
    if mismatch:
        rp = np.random.default_rng(seed + 99)
        x = pts.numpy()
        basis = np.stack([x[:, 0], x[:, 1], x[:, 2], x[:, 0] ** 2, x[:, 1] ** 2, x[:, 2] ** 2,
                          x[:, 0] * x[:, 1], x[:, 1] * x[:, 2], x[:, 0] * x[:, 2]], 1)
        basis = basis - basis.mean(0)
        for j in range(n_coils):
            a, b = basis @ rp.normal(size=9), basis @ rp.normal(size=9)
            C[j] *= np.exp(mismatch * (a / a.std() + 1j * b / b.std()))
    C = C.reshape(n_coils, *shape)
    obj = ellipsoid_phantom(shape, seed)
    # 'volume': mean sum|C|^2 = 1 over the whole grid (original 3-D cases); 'object': over the object,
    # so the SNR is set by ``noise`` even when the field near the coils is much stronger than inside
    ref = (np.abs(C) ** 2).sum(0)
    C = C / np.sqrt(ref[obj > 0].mean() if normalise == "object" else ref.mean())
    rng = np.random.default_rng(seed + 1)
    k = fftnc_np(C * obj[None])
    k = k + noise * (rng.normal(size=k.shape) + 1j * rng.normal(size=k.shape)) / math.sqrt(2)
    d = VolumeData(k.astype(np.complex64), geom, mask=np.ones(shape, bool),
                   meta={"source": "synthetic3d", "wave_k2": None if wave_k2 is None else [complex(wave_k2).real, complex(wave_k2).imag]})
    return d, C.astype(np.complex64), obj


# ===================================================================================== outputs
def evaluate3d(model, prob: Problem3D, extra: dict, out_dir=None, gt=None, obj=None, log=print):
    """Full-resolution maps, physics residuals, optional ground-truth errors, central-slice figure."""
    import json
    from .physics import maxwell_static_residual
    C = volume_maps(model, prob).numpy()                                     # (N,D,H,W) data channels
    Cn = C / np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)).clip(1e-12)
    rep = {"n_coils": prob.N, "shape": list(prob.data.geom.shape), "voxel_mm": list(prob.data.geom.voxel),
           "orientation": prob.data.geom.axes, "handedness": extra["handedness"], "calib_block": list(prob.kcalib),
           "calib_grid": list(prob.calib), "data_term": prob.data_term}
    with torch.enable_grad():
        Cl = model.sensitivity(prob.coords, create_graph=False).detach()
        rep["calib_data_loss"] = float(prob.data_loss(Cl, extra.get("image_lowres")))
        sp = prob.support_pts[torch.randperm(prob.support_pts.shape[0])[:1500]]
        div, curl = maxwell_static_residual(lambda x: model.field(x, create_graph=True), sp, model.n_coils)
    rep["physics_rel_rms"] = {"div": float(div.mean()), "curl": float(curl.mean())}
    if hasattr(model, "k2_re"):          # Helmholtz: curl B != 0 by design; report the fitted medium
        from .models import k2_to_tissue
        k2 = complex(float(model.k2_re), float(model.k2_im))
        rep["k2_normalised"] = [k2.real, k2.imag]
        b0 = getattr(prob.cfg, "b0_tesla", 3.0)
        er, sg = k2_to_tissue(k2, b0, prob.data.geom.half_fov / 1000.0)
        rep["homogeneous_medium_estimate"] = {"b0_tesla": b0, "eps_r": er, "sigma_S_per_m": sg}
    if gt is not None:
        o = obj if obj is not None else np.ones(C.shape[1:], bool)
        G = gt / np.sqrt((np.abs(gt) ** 2).sum(0, keepdims=True)).clip(1e-12)
        rep["vs_ground_truth"] = {"subspace_err": float((1 - np.abs((Cn.conj() * G).sum(0)) ** 2)[o].mean())}
    log(json.dumps(rep, indent=1))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "maps.npy"), C.astype(np.complex64))
        np.save(os.path.join(out_dir, "maps_normalized.npy"), Cn.astype(np.complex64))
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(rep, f, indent=1)
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        N = C.shape[0]
        D, H, W = C.shape[1:]
        fig, ax = plt.subplots(3, N, figsize=(1.6 * N, 5), squeeze=False)
        for j in range(N):
            for r, sl in enumerate((Cn[j, D // 2], Cn[j, :, H // 2], Cn[j, :, :, W // 2])):
                ax[r, j].imshow(np.abs(sl), cmap="viridis", vmin=0, vmax=1); ax[r, j].axis("off")
        fig.suptitle("normalised |C|: central slices along each array axis (rows)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "maps_central.png"), dpi=90); plt.close(fig)
    return rep, C
