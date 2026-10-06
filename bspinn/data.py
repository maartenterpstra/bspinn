"""Multi-coil data containers, loaders (BART .cfl, fastMRI .h5, .npy/.npz) and a
synthetic Biot–Savart phantom generator for validation."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from .geometry import SliceGeometry, crop_center, fft2c_np, ifft2c_np


@dataclass
class MultiCoilData:
    """Cartesian multi-coil k-space.

    kspace: complex array (S, N, H, W) -- S slices, N coils (any N).
    mask:   bool array broadcastable to kspace, True where sampled.
    geom:   SliceGeometry.
    noise:  optional (Ns, N) noise-only samples for pre-whitening.
    """
    kspace: np.ndarray
    geom: SliceGeometry
    mask: np.ndarray | None = None
    noise: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    traj: np.ndarray | None = None      # (M,2) [k_row, k_col] cycles/FOV -> kspace is (S,N,M)

    def __post_init__(self):
        self.kspace = np.asarray(self.kspace, dtype=np.complex64)
        if self.traj is not None:
            if self.kspace.ndim == 2:
                self.kspace = self.kspace[None]
            self.mask = None
            return
        if self.kspace.ndim == 3:
            self.kspace = self.kspace[None]
        if self.mask is None:
            self.mask = (np.abs(self.kspace).sum(axis=1, keepdims=True) > 0)
        self.mask = np.broadcast_to(np.asarray(self.mask, bool), (self.kspace.shape[0], 1) + self.kspace.shape[2:]).copy()

    @property
    def n_coils(self):
        return self.kspace.shape[1]

    def prewhiten(self):
        """Noise decorrelation with the Cholesky factor of the channel noise covariance.
        Whitened channels are linear combinations of physical channels, so their
        sensitivities are still Biot–Savart (harmonic) fields."""
        if self.noise is None:
            return self
        n = self.noise.reshape(-1, self.n_coils)
        psi = (n.conj().T @ n) / n.shape[0]
        Linv = np.linalg.inv(np.linalg.cholesky(psi))
        self.kspace = np.einsum("ij,sj...->si...", Linv, self.kspace).astype(np.complex64)
        self.meta["noise_cov"] = psi
        self._add_mix(Linv)
        self.noise = None
        return self

    def _add_mix(self, M):
        """Record that data channels are now M @ (previous channels)."""
        prev = self.meta.get("mix")
        self.meta["mix"] = (M if prev is None else M @ prev).astype(np.complex64)

    def compress(self, n_virtual: int, calib: int = 48):
        """SVD coil compression to ``n_virtual`` virtual channels (Buehrer/Huang), computed from the
        (whitened) calibration data. The mixing matrix is recorded so a physical-coil model can
        still be fitted exactly (--model-channels physical)."""
        N = self.n_coils
        if n_virtual >= N:
            return self
        ks = self.kspace
        X = (crop_center(ks, (min(calib, ks.shape[-2]), min(calib, ks.shape[-1]))) if self.traj is None else ks)
        X = np.moveaxis(X, 1, 0).reshape(N, -1)
        U, sv, _ = np.linalg.svd(X, full_matrices=False)
        A = U[:, :n_virtual].conj().T                               # (Nv, N)
        self.kspace = np.einsum("vn,sn...->sv...", A, ks).astype(np.complex64)
        self.meta["compression_energy"] = float((sv[:n_virtual] ** 2).sum() / (sv ** 2).sum())
        self._add_mix(A)
        return self

    def select_slices(self, idx):
        idx = list(idx)
        self.kspace = self.kspace[idx]
        if self.mask is not None:
            self.mask = self.mask[idx]
        self.geom.slice_pos = np.asarray(self.geom.slice_pos)[idx]
        return self


# --------------------------------------------------------------------------- loaders
def _read_cfl(base):
    base = base[:-4] if base.endswith((".cfl", ".hdr")) else base
    with open(base + ".hdr") as f:
        f.readline()
        dims = [int(x) for x in f.readline().split()]
    data = np.fromfile(base + ".cfl", dtype=np.complex64)
    return data.reshape(dims, order="F")


def load(path: str, plane: str = "axial", pixel_spacing=(1.0, 1.0), slice_spacing: float = 5.0,
         noise_path: str | None = None, slices=None, kspace_key: str = "kspace") -> MultiCoilData:
    """Load multi-coil Cartesian k-space.

    * BART .cfl/.hdr: dims (RO, PE1, PE2, COIL, ...). A singleton RO or PE2 gives one
      2-D slice; a full 3-D array is inverse-FFT'd along RO and treated as a slice stack.
    * fastMRI .h5:    dataset 'kspace' (S, N, H, W).
    * .npy/.npz:      (S, N, H, W) or (N, H, W); .npz may hold 'mask' and 'noise'.
    """
    mask = noise = None
    p = path.lower()
    if p.endswith((".cfl", ".hdr")) or os.path.exists(path + ".cfl"):
        k = _read_cfl(path)
        k = k.reshape(k.shape[:4]) if k.ndim >= 4 else k
        d0, d1, d2, N = k.shape
        if d0 == 1:
            ks = np.transpose(k[0], (2, 0, 1))[None]                     # (1,N,d1,d2)
        elif d2 == 1:
            ks = np.transpose(k[:, :, 0], (2, 0, 1))[None]
        else:
            img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k, axes=0), axis=0, norm="ortho"), axes=0)
            ks = np.transpose(img, (0, 3, 1, 2))                         # (d0,N,d1,d2)
    elif p.endswith((".h5", ".hdf5")):
        import h5py
        with h5py.File(path, "r") as f:
            ks = f[kspace_key][()]
            if "mask" in f:
                mask = f["mask"][()]
    elif p.endswith(".npz"):
        z = np.load(path)
        ks = z[kspace_key]; mask = z["mask"] if "mask" in z else None
        noise = z["noise"] if "noise" in z else None
    else:
        ks = np.load(path)
    ks = np.asarray(ks)
    if ks.ndim == 3:
        ks = ks[None]
    if noise_path:
        noise = _read_cfl(noise_path) if not noise_path.endswith(".npy") else np.load(noise_path)
        noise = noise.reshape(-1, noise.shape[-1]) if noise.ndim > 1 else noise
        noise = noise.reshape(-1, ks.shape[1])
    S = ks.shape[0]
    pos = (np.arange(S) - (S - 1) / 2) * slice_spacing
    geom = SliceGeometry(ks.shape[2:], tuple(pixel_spacing), pos, plane)
    if mask is not None and mask.ndim == 1:      # fastMRI column mask
        mask = mask[None, None, None, :]
    d = MultiCoilData(ks, geom, mask=mask, noise=noise, meta={"source": path})
    if slices is not None:
        d.select_slices(slices)
    return d


def load_noncart(path: str, traj_path: str, fov_mm: float, matrix: int | None = None, plane: str = "axial",
                 noise_path: str | None = None, drop_first: int = 0) -> MultiCoilData:
    """Non-Cartesian single-slice data in BART layout.

    path:      k-space .cfl, dims (1, samples, spokes, coils)
    traj_path: BART trajectory .cfl, dims (3, samples, spokes), in cycles/FOV (BART units),
               or .npy (samples*spokes, 2). Use the scanner/BART trajectory (incl. gradient-delay
               correction, e.g. ``bart estdelay`` + ``bart trajcor``) -- see README.
    drop_first: discard the first spokes (e.g. non-steady-state excitations).
    """
    k = _read_cfl(path)
    k = k.reshape(k.shape[1:4]) if k.shape[0] == 1 else k.reshape(k.shape[:3])
    ns, nsp, N = k.shape
    if traj_path.endswith(".npy"):
        t = np.load(traj_path).reshape(nsp, ns, 2)
    else:
        tr = _read_cfl(traj_path).real.reshape(3, ns, nsp)
        t = np.stack([tr[0], tr[1]], -1).transpose(1, 0, 2)           # (spokes, samples, [k0,k1])
    ks = np.transpose(k, (2, 1, 0))                                       # (N, spokes, samples)
    ks, t = ks[:, drop_first:], t[drop_first:]
    if matrix is None:
        matrix = int(2 * np.ceil(np.abs(t).max()))
    geom = SliceGeometry((matrix, matrix), (fov_mm / matrix, fov_mm / matrix), np.zeros(1), plane)
    noise = _read_cfl(noise_path).reshape(-1, N) if noise_path else None
    return MultiCoilData(ks.reshape(N, -1)[None], geom, noise=noise, traj=t.reshape(-1, 2).astype(np.float32),
                         meta={"source": path, "trajectory": traj_path})


# --------------------------------------------------------------------------- synthetic
def synthetic_array(n_coils: int, radius=1.25, loop_radius: float = 0.45, n_rings: int | None = None,
                    seed: int = 0, rotation: float = 0.0):
    """Ground-truth receive array: circular loops on a cylinder around B0 (z)."""
    rng = np.random.default_rng(seed)
    n_rings = n_rings or (1 if n_coils <= 8 else 2)
    per = math.ceil(n_coils / n_rings)
    verts, M = [], 48
    rx, ry = (radius, radius) if np.isscalar(radius) else radius
    t = np.linspace(0, 2 * math.pi, M, endpoint=False)
    for j in range(n_coils):
        ring, k = divmod(j, per)
        phi = 2 * math.pi * (k + 0.5 * ring) / per + rotation
        zc = 0.0 if n_rings == 1 else (ring - (n_rings - 1) / 2) * 0.7
        u = np.array([math.cos(phi), math.sin(phi), 0.0]); e2 = np.array([0, 0, 1.0]); e1 = np.cross(e2, u)
        c = np.array([rx * u[0], ry * u[1], zc])
        verts.append(c + loop_radius * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2))
    verts = np.stack(verts)[:, None]                                      # (N,1,M,3)
    phase = rng.uniform(-math.pi, math.pi, n_coils)
    return verts.astype(np.float32), phase.astype(np.float32)


def shepp_logan_like(H, W, seed=0):
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    ell = [(0, 0, .72, .92, 0, 1.0), (0, -.02, .66, .86, 0, -.6), (.22, 0, .11, .31, -18, -.2),
           (-.22, 0, .16, .41, 18, -.2), (0, .35, .21, .25, 0, .15), (0, .1, .046, .046, 0, .15),
           (-.08, -.6, .046, .023, 0, .15), (.06, -.6, .023, .046, 0, .15), (0, -.3, .1, .05, 30, .25)]
    img = np.zeros((H, W))
    for x0, y0, a, b, ang, v in ell:
        th = np.deg2rad(ang)
        xr = (xx - x0) * np.cos(th) + (yy - y0) * np.sin(th)
        yr = -(xx - x0) * np.sin(th) + (yy - y0) * np.cos(th)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += v
    rng = np.random.default_rng(seed)
    c = rng.normal(size=4)
    phase = 0.6 * (c[0] * xx + c[1] * yy + c[2] * xx * yy + c[3] * (xx ** 2 - yy ** 2))
    return img * np.exp(1j * phase)


def make_folded_synthetic(n_coils=8, shape=(96, 96), width_factor=1.35, noise=0.01, seed=0):
    """Object wider than the FOV along the column (phase-encode) axis: the parts outside the FOV
    fold back in. Coil i's contribution from the copy at r + n*FOV is C_i(r + n*FOV) * obj_n(r),
    i.e. the *same* Biot–Savart field at displaced coordinates (what --fold-sets models).
    Returns (data, gt_maps (1,N,H,W), central object)."""
    import torch
    from .physics import biot_savart_polyline
    H, W = shape
    geom = SliceGeometry(shape, (1.0, 1.0), np.zeros(1), "axial")
    fov = W * 1.0 / geom.half_fov                                      # normalised FOV along x (cols)
    # receive loops on an ellipse that encloses the wide object
    verts, ph = synthetic_array(n_coils, radius=(0.5 * fov * width_factor + 0.35, 1.0), seed=seed)
    Vt = torch.from_numpy(verts)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    rng = np.random.default_rng(seed)
    I = np.zeros((n_coils, H, W), complex)
    for n in (-1, 0, 1):
        pts = geom.grid().reshape(-1, 3).clone()
        pts[:, 0] += n * fov
        B = biot_savart_polyline(Vt, pts).sum(1).numpy()
        C = np.exp(1j * ph)[:, None] * (B[..., 0] + 1j * B[..., 1])
        x = (xx + 2 * n) / width_factor                                 # object coords (wide ellipse)
        obj = ((x / 0.95) ** 2 + (yy / 0.75) ** 2 <= 1).astype(float)
        obj *= 1 + 0.3 * np.cos(4 * x) * np.cos(3 * yy)
        obj = obj * np.exp(1j * 0.4 * (x + yy))
        if n == 0:
            C0, obj0 = C.reshape(n_coils, H, W), obj
        I += C.reshape(n_coils, H, W) * obj
    scale = np.sqrt((np.abs(C0) ** 2).sum(0).mean())
    I, C0 = I / scale, C0 / scale
    k = fft2c_np(I)
    k = k + noise * (rng.normal(size=k.shape) + 1j * rng.normal(size=k.shape)) / math.sqrt(2)
    return MultiCoilData(k[None].astype(np.complex64), geom, meta={"source": "folded synthetic"}), \
        C0[None].astype(np.complex64), obj0


def make_synthetic(n_coils=8, shape=(128, 128), n_slices=1, slice_spacing=8.0, plane="axial",
                   noise=0.01, nonharmonic=0.0, accel=None, calib=24, seed=0, rotation=0.0, noise_corr=0.0,
                   mismatch=0.0):
    """Synthetic dataset whose ground-truth sensitivities obey Biot–Savart exactly
    (``nonharmonic=0``) or with a smooth non-harmonic phase perturbation of strength
    ``nonharmonic`` (rad) emulating conductive/wave effects. That phase is common to all coils
    (unobservable). ``mismatch`` > 0 instead multiplies every coil by its *own* smooth complex
    factor exp(mismatch * (a_j(r) + i b_j(r))), a_j, b_j random quadratic polynomials of unit RMS
    over the FOV: maps that are no longer Biot–Savart fields (model-mismatch test).

    Returns (MultiCoilData, ground_truth_maps (S,N,H,W))."""
    import torch
    from .physics import biot_savart_polyline
    H, W = shape
    geom = SliceGeometry(shape, (1.0, 1.0), (np.arange(n_slices) - (n_slices - 1) / 2) * slice_spacing, plane)
    pts = geom.grid().reshape(-1, 3)
    verts, ph = synthetic_array(n_coils, seed=seed, rotation=rotation)
    B = biot_savart_polyline(torch.from_numpy(verts), pts).sum(1).numpy()           # (N,P,3)
    C = np.exp(1j * ph)[:, None] * (B[..., 0] + 1j * B[..., 1])
    if nonharmonic:
        r2 = (pts.numpy() ** 2).sum(-1)
        C = C * np.exp(1j * nonharmonic * r2)[None]
    if mismatch:
        rp = np.random.default_rng(seed + 99)
        x = pts.numpy()
        basis = np.stack([x[:, 0], x[:, 1], x[:, 0] ** 2, x[:, 1] ** 2, x[:, 0] * x[:, 1], x[:, 2]], 1)
        basis = basis - basis.mean(0)
        for j in range(n_coils):
            a, b = basis @ rp.normal(size=6), basis @ rp.normal(size=6)
            a, b = a / a.std(), b / b.std()
            C[j] = C[j] * np.exp(mismatch * (a + 1j * b))
    S = n_slices
    C = C.reshape(n_coils, S, H, W).transpose(1, 0, 2, 3)
    C = C / np.sqrt((np.abs(C) ** 2).sum(1).mean())
    obj = np.stack([shepp_logan_like(H, W, seed + s) for s in range(S)])[:, None]
    img = C * obj
    rng = np.random.default_rng(seed + 1)
    k = fft2c_np(img)
    # receiver noise, optionally correlated between channels (complex Hermitian covariance L L^H)
    Lc = np.eye(n_coils, dtype=complex)
    if noise_corr:
        R = rng.normal(size=(n_coils, n_coils)) + 1j * rng.normal(size=(n_coils, n_coils))
        Lc = Lc + noise_corr * np.tril(R, -1) / np.sqrt(2)
        Lc = Lc / np.sqrt((np.abs(Lc) ** 2).sum(1, keepdims=True))
    wn = (rng.normal(size=k.shape) + 1j * rng.normal(size=k.shape)) / math.sqrt(2)
    k = k + noise * np.einsum("ij,sj...->si...", Lc, wn)
    prescan = None
    if noise_corr:
        wp = (rng.normal(size=(4000, n_coils)) + 1j * rng.normal(size=(4000, n_coils))) / math.sqrt(2)
        prescan = noise * wp @ Lc.T
    mask = None
    if accel:
        m = rng.random((H, W)) < 1.0 / accel
        r0, c0 = H // 2 - calib // 2, W // 2 - calib // 2
        m[r0:r0 + calib, c0:c0 + calib] = True
        mask = m[None, None]
        k = k * mask
    d = MultiCoilData(k.astype(np.complex64), geom, mask=mask, noise=prescan, meta={"source": "synthetic"})
    if noise_corr:
        psi = Lc @ Lc.conj().T
        off = np.abs(psi / np.sqrt(np.outer(np.diag(psi).real, np.diag(psi).real)))[~np.eye(n_coils, dtype=bool)]
        d.meta["true_noise_corr_mean_abs"] = float(off.mean())
    return d, C.astype(np.complex64), obj[:, 0]


def load_maps(path: str) -> np.ndarray:
    """Reference sensitivity maps -> (S, N, H, W). .npy is taken as (S,N,H,W) or (N,H,W).
    BART maps have dims (X, Y, Z, COIL[, MAPS]); the first map set is used, and the two
    non-singleton spatial dims become (H, W) -- same convention as ``load``."""
    if path.endswith(".npy"):
        m = np.load(path)
        return m[None] if m.ndim == 3 else m
    m = _read_cfl(path)
    m = m.reshape(m.shape[:5] + (1,) * max(0, 5 - m.ndim)) if m.ndim >= 5 else m.reshape(m.shape + (1,) * (5 - m.ndim))
    m = m.reshape(m.shape[:4] + (-1,))[..., 0]                                   # (X,Y,Z,N), first ESPIRiT map set
    X, Y, Z, N = m.shape[:4]
    if X == 1:
        img = m[0]                                         # (Y,Z,N)
    else:
        img = m[:, :, 0]                                   # (X,Y,N)
    return np.transpose(img, (2, 0, 1))[None]
