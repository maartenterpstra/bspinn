"""Coil-sensitivity estimators behind one interface, for the method comparison (``evaluation.py``).

Every estimator gets the same input, an :class:`EvalCase` with calibration k-space, and returns
complex maps (N, H, W) at full resolution. Scaling is method specific (Biot–Savart maps are
absolute; the others are RSS-normalised); all metrics state whether they depend on it.

==================  =====================================================================
name                method
==================  =====================================================================
``lowres``          Hann-windowed calibration region, zero-padded, divided by RSS (mSENSE/McKenzie)
``walsh``           Walsh et al. 2000: dominant eigenvector of the locally averaged coil covariance
``inati``           Inati et al. 2014 iterative fast adaptive combine (ISMRMRD reference algorithm)
``espirit``         BART ``ecalib -m1`` (Uecker et al. 2014), default eigenvalue crop
``nlinv``           BART ``ncalib`` (NLINV-based calibration, Uecker et al. 2008)
``bs_pinn``         Biot–Savart wires + harmonic PINN (this package, default model)
``bs_wires``        Biot–Savart wire loops only (the hard-constrained stage of ``bs_pinn``)
``free``            same pipeline without physics (Sobolev smoothness; ablation)
``bs_pinn_grid1``   ablation: ``bs_pinn`` with ``calib_oversample=1`` (maps and image on the ACS grid)
``bs_pinn_complex`` ``bs_pinn`` with ``field_mode='complex'`` (meaningful with 3-D data)
``bs_helmholtz``    3-D: wires as incident field + vector Helmholtz correction, k^2 learned from a tissue prior
``bs_helmholtz_fixed`` same with k^2 fixed at the tissue prior (case.meta['tissue'])
``oracle``          ground-truth maps (synthetic data only)
==================  =====================================================================
"""
from __future__ import annotations

import copy
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np
import torch


def _sp_axes(x):
    """Spatial axes of a channel-first array (N, H, W) or (N, D, H, W); a single image (H, W) is 2-D."""
    return tuple(range(1, x.ndim)) if x.ndim in (3, 4) else tuple(range(x.ndim))


def ifft2c(k):
    """Centred orthonormal inverse FFT over the spatial axes (2-D or 3-D; name kept for history)."""
    ax = _sp_axes(k)
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k, axes=ax), axes=ax, norm="ortho"), axes=ax)


def fft2c(x):
    ax = _sp_axes(x)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


@dataclass
class EvalCase:
    """One 2-D slice.

    ksp:      (N, H, W) complex k-space, pre-whitened so that every sample has noise variance ``sigma**2``.
    sampled:  (H, W) bool, positions actually acquired (fully sampled data may still have partial-Fourier gaps).
    acs:      (h, w) size of the central calibration box handed to every method.
    calib_kspace: what the estimators see: by default the ACS box of ``ksp`` only (all other samples zeroed),
              so every method gets *identical* information. ``joint=True`` cases (prospectively undersampled
              data) give the estimators all acquired samples instead, which only NLINV/Biot–Savart use.
    """
    name: str
    ksp: np.ndarray
    sampled: np.ndarray
    acs: tuple
    pixel_spacing: tuple = (1.0, 1.0)
    plane: str = "axial"
    sigma: float = 1.0
    fully_sampled: bool = True
    pe_axis: int = 1                     # 2-D: image axis (0 rows, 1 cols) of phase encoding, for undersampling
    pe_axes: tuple = (1, 2)              # 3-D: the two phase-encoding axes (readout = the remaining one)
    voxel: tuple = (1.0, 1.0, 1.0)       # 3-D: mm per array axis
    axes: str = "zyx"                    # 3-D: physical axis of each array axis (B0 = z)
    gt_maps: np.ndarray | None = None    # (N, H, W) ground truth (synthetic)
    gt_object: np.ndarray | None = None  # (H, W) ground-truth object (synthetic)
    joint: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def N(self):
        return self.ksp.shape[0]

    @property
    def shape(self):
        return self.ksp.shape[1:]

    @property
    def ndim(self):
        return self.ksp.ndim - 1

    def acs_mask(self):
        m = np.zeros(self.shape, bool)
        m[tuple(slice(n // 2 - a // 2, n // 2 - a // 2 + a) for n, a in zip(self.shape, self.acs))] = True
        return m & self.sampled

    def calib_kspace(self):
        return self.ksp * (self.sampled if self.joint else self.acs_mask())[None]


# ------------------------------------------------------------------------------- helpers
def _rss_normalise(C, eps=1e-12):
    return C / np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)).clip(eps)


def _lowres_images(case: EvalCase, window=True):
    k = case.ksp * case.acs_mask()[None]
    if window:
        win = np.ones(case.shape)
        for ax, (n, a) in enumerate(zip(case.shape, case.acs)):
            w1 = np.zeros(n); w1[n // 2 - a // 2: n // 2 - a // 2 + a] = np.hanning(a + 2)[1:-1]
            win = win * w1.reshape([-1 if i == ax else 1 for i in range(case.ndim)])
        k = k * win[None]
    return ifft2c(k)


def _box(x, size):
    from scipy.ndimage import uniform_filter
    if np.iscomplexobj(x):
        return uniform_filter(x.real, size) + 1j * uniform_filter(x.imag, size)
    return uniform_filter(x, size)


# ------------------------------------------------------------------------------- estimators
def lowres(case: EvalCase, **kw):
    return _rss_normalise(_lowres_images(case))


def walsh(case: EvalCase, window: int = 0, **kw):
    """Walsh 2000. Covariance R(r) = sum over a window of i(r') i(r')^H from the (windowed) low-res
    calibration images; map = dominant eigenvector, phase referenced to the strongest coil."""
    im = _lowres_images(case)
    N, sp = im.shape[0], im.shape[1:]
    if not window:   # about one low-resolution pixel
        window = int(max(3, round(max(n / a for n, a in zip(sp, case.acs)))) | 1)
    R = np.empty(sp + (N, N), np.complex64)
    for a in range(N):
        for b in range(a, N):
            R[..., a, b] = _box(im[a] * im[b].conj(), (window,) * len(sp))
            R[..., b, a] = R[..., a, b].conj()
    _, V = np.linalg.eigh(R)
    v = V[..., -1]                                             # (..., N), unit norm
    ref = int(np.argmax((np.abs(im) ** 2).sum(tuple(range(1, im.ndim)))))
    v = v * np.exp(-1j * np.angle(v[..., ref]))[..., None]
    return np.moveaxis(v, -1, 0)


def inati(case: EvalCase, smoothing: int = 0, niter: int = 5, thresh=1e-3, **kw):
    """Inati, Hansen, Kellman 2014 iterative fast adaptive combine (as in ismrmrd-python-tools)."""
    im = _lowres_images(case)
    N, sp = im.shape[0], im.shape[1:]
    spx = tuple(range(1, im.ndim))
    bc = (slice(None),) + (None,) * len(sp)
    if not smoothing:
        smoothing = int(max(3, round(max(n / a for n, a in zip(sp, case.acs)))) | 1)
    D_sum = im.sum(spx); D_sum = D_sum / np.linalg.norm(D_sum)
    R = (D_sum.conj()[bc] * im).sum(0)
    cmap = im
    for _ in range(niter):
        prevR = R
        cmap = _box(im * R.conj()[None], (1,) + (smoothing,) * len(sp))
        cmap = cmap / np.sqrt((np.abs(cmap) ** 2).sum(0)).clip(1e-12)[None]
        R = (cmap.conj() * im).sum(0)
        D_sum = cmap.sum(spx); D_sum = D_sum / np.linalg.norm(D_sum)
        imT = (D_sum.conj()[bc] * cmap).sum(0)
        imT = imT / np.abs(imT).clip(1e-12)
        R = R * imT
        cmap = cmap * imT.conj()[None]
        if np.linalg.norm(R - prevR) / max(np.linalg.norm(R), 1e-12) < thresh:
            break
    return cmap


def _bart():
    for p in (os.environ.get("BART"), shutil.which("bart")):
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("BART not found: set $BART to the bart executable")


def _bart_run(cmd, ksp_nhw, out_name="maps"):
    """ksp (N,H,W) -> BART layout (H,W,1,N), or (N,D,H,W) -> (D,H,W,N); returns the first map
    set as (N,H,W) / (N,D,H,W)."""
    from .data import _read_cfl
    tmp = tempfile.mkdtemp()
    try:
        three = ksp_nhw.ndim == 4
        k = np.moveaxis(ksp_nhw, 0, -1)
        if not three:
            k = k[:, :, None, :]
        _write_cfl(os.path.join(tmp, "k"), k.astype(np.complex64))
        subprocess.run([_bart()] + cmd + [os.path.join(tmp, "k"), os.path.join(tmp, out_name)], check=True,
                       capture_output=True)
        m = _read_cfl(os.path.join(tmp, out_name))
        m = m.reshape(m.shape[:4] + (-1,))[..., 0]               # (X,Y,Z,N)
        return np.moveaxis(m, -1, 0) if three else np.transpose(m[:, :, 0, :], (2, 0, 1))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_cfl(name, arr):
    arr = np.asarray(arr, np.complex64)
    with open(name + ".hdr", "w") as f:
        f.write("# Dimensions\n" + " ".join(str(d) for d in arr.shape) + "\n")
    arr.T.tofile(name + ".cfl")       # column-major


def espirit(case: EvalCase, **kw):
    return _bart_run(["ecalib", "-m1", "-r", ":".join(str(a) for a in case.acs)], case.calib_kspace())


def nlinv(case: EvalCase, **kw):
    return _bart_run(["ncalib"], case.calib_kspace())


def _bs(case: EvalCase, model="pinn", cfg_overrides=None, return_wires=False, log=None, **kw):
    from .data import MultiCoilData
    from .fit import FitConfig, fit
    from .geometry import SliceGeometry
    cfg = FitConfig(model=model, calib=int(max(case.acs)), iters_wires=1000, iters_pinn=1500, lambda_phys=0.1,
                    log_every=500, threads=kw.get("threads", 1), phase_ref="model")
    if kw.get("device"):
        cfg.device = kw["device"]
    if kw.get("seed") is not None:
        cfg.seed = kw["seed"]
    for k, v in (cfg_overrides or {}).items():
        setattr(cfg, k, v)
    if case.joint:
        cfg.data_term = "kspace"
    if case.ndim == 3:
        from .volume import VolumeData, VolumeGeometry, fit3d, volume_maps
        cfg.calib = tuple(int(a) for a in case.acs)
        vd = VolumeData(case.calib_kspace(), VolumeGeometry(case.shape, case.voxel, case.axes),
                        mask=case.sampled if case.joint else case.acs_mask())
        mdl, prob, hist, extra = fit3d(vd, cfg, log=log or (lambda *a: None), orientation=kw.get("orientation", "given"))
        out = {model: volume_maps(mdl, prob).numpy(), "_history": hist, "_orientation": prob.data.geom.axes,
               "_n_params": int(sum(p.numel() for p in mdl.parameters() if p.requires_grad))}
        if hasattr(mdl, "k2_re"):
            out["_k2"] = [float(mdl.k2_re), float(mdl.k2_im)]
        if model == "pinn":     # the wire stage of the same run = a stand-alone wire fit (same seed/iters)
            out["wires"] = volume_maps(extra["wires_stage"], prob).numpy()
            out["_t_wires"] = extra["t_wires"]
        return out
    geom = SliceGeometry(case.shape, case.pixel_spacing, [0.0], case.plane)
    data = MultiCoilData(case.calib_kspace()[None], geom, mask=(case.sampled if case.joint else case.acs_mask())[None, None])
    mdl, prob, hist, extra = fit(data, cfg, log=log or (lambda *a: None))
    pts = geom.grid().reshape(-1, 3)
    out = {}
    with torch.enable_grad():
        C = prob.mixC(mdl.sensitivity_batched(pts), -2).detach().numpy().reshape(case.N, *case.shape)
        out[model] = C
        if return_wires and extra.get("wires") is not None and extra["wires"] is not mdl:
            out["wires"] = prob.mixC(extra["wires"].sensitivity_batched(pts), -2).detach().numpy().reshape(case.N, *case.shape)
    out["_history"] = hist
    out["_n_params"] = int(sum(p.numel() for p in mdl.parameters() if p.requires_grad))
    return out


def pinn_api(case: EvalCase, device="cpu", seed=0, quality="default", pinn_options=None, **kw):
    """``bspinn.estimate_maps`` exactly as a user calls it (presets, plane search, fold-over
    handling). Calibration-only cases: the ACS box the other methods see (that mask does not reveal the
    readout axis, so it is passed explicitly, 1 - pe_axis). Joint cases (``case.joint``, e.g. the
    retrospectively undersampled protocol of evaluate_methods --joint): every acquired sample, with its
    mask. ``pinn_options``: extra estimate_maps / FitConfig options (e.g. solve_image=True,
    optimizer='adam+lbfgs'); evaluate_methods --pinn-variant registers named variants with them."""
    from .api import estimate_maps
    opts = dict(pinn_options or {})
    seen = case.sampled if case.joint else case.acs_mask()
    data = case.ksp * case.sampled[None] if case.joint else case.calib_kspace()
    plane = opts.pop("plane", case.meta.get("api_plane", case.plane))
    out = estimate_maps(data, "2d" if case.ndim == 2 else "3d", mask=seen,
                        calib=int(max(case.acs)) if case.ndim == 2 else tuple(int(a) for a in case.acs),
                        spacing=tuple(case.pixel_spacing) if case.ndim == 2 else tuple(case.voxel),
                        plane=plane if case.ndim == 2 else "auto", device=device, seed=seed,
                        quality=opts.pop("quality", quality), verbose=False,
                        **({"readout_axis": 1 - case.pe_axis} if case.ndim == 2 else {}), **opts)
    return np.asarray(out[0])


ESTIMATORS = {"pinn_api": pinn_api, "lowres": lowres, "walsh": walsh, "inati": inati, "espirit": espirit, "nlinv": nlinv}
ALL_METHODS = ["lowres", "walsh", "inati", "espirit", "nlinv", "bs_pinn", "bs_wires", "free", "oracle"]
# lowres/walsh/inati/espirit/nlinv/pinn_api take these keyword arguments; others ignore them
_PASS = {"pinn_api": ("device", "seed", "quality")}


def estimate(case: EvalCase, methods, cache_dir=None, log=print, **kw):
    """Run the requested estimators; returns {name: maps (N,H,W)}, {name: seconds}.
    Results are cached as .npy in ``cache_dir``."""
    maps, secs = {}, {}

    def cached(name):
        if cache_dir and os.path.exists(os.path.join(cache_dir, f"{name}.npy")):
            maps[name] = np.load(os.path.join(cache_dir, f"{name}.npy"))
            t = os.path.join(cache_dir, f"{name}.time")
            secs[name] = float(open(t).read()) if os.path.exists(t) else float("nan")
            return True
        return False

    def store(name, C, t):
        maps[name] = C.astype(np.complex64); secs[name] = t
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(os.path.join(cache_dir, f"{name}.npy"), maps[name])
            open(os.path.join(cache_dir, f"{name}.time"), "w").write(str(t))

    for name in methods:
        if name in maps or cached(name):
            continue
        t0 = time.time()
        if name == "oracle":
            if case.gt_maps is None:
                continue
            store(name, case.gt_maps, 0.0)
        elif name in ESTIMATORS:
            kk = {k: v for k, v in kw.items() if k in _PASS.get(name, ()) or k not in ("device", "seed", "quality")}
            store(name, ESTIMATORS[name](case, **kk), time.time() - t0)
        elif name == "bs_pinn":
            out = _bs(case, "pinn", **kw); store("bs_pinn", out["pinn"], time.time() - t0)
            if "wires" in out and "bs_wires" in methods and not cached("bs_wires"):
                store("bs_wires", out["wires"], out["_t_wires"])
        elif name in ("bs_helmholtz", "bs_helmholtz_fixed"):
            kw2 = dict(kw)
            ov = {**(kw2.get("cfg_overrides") or {}), "physics": "helmholtz", "learn_k2": name == "bs_helmholtz"}
            # learned: start from a generic tissue prior; fixed: the true tissue values when known
            ov.update(case.meta.get("tissue", {}) if name == "bs_helmholtz" else case.meta.get("tissue_true", case.meta.get("tissue", {})))
            kw2["cfg_overrides"] = ov
            out = _bs(case, "pinn", **kw2); store(name, out["pinn"], time.time() - t0)
            if "_k2" in out and cache_dir:
                import json
                with open(os.path.join(cache_dir, f"{name}.k2.json"), "w") as fh:
                    json.dump(out["_k2"], fh)
        elif name == "bs_pinn_complex":
            kw2 = dict(kw)
            kw2["cfg_overrides"] = {**(kw2.get("cfg_overrides") or {}), "field_mode": "complex"}
            out = _bs(case, "pinn", **kw2); store(name, out["pinn"], time.time() - t0)
        elif name == "bs_pinn_grid1":
            # ablation: model on the calibration grid itself (pixel-wise low-res model, v4 default)
            kw2 = dict(kw)
            kw2["cfg_overrides"] = {**(kw2.get("cfg_overrides") or {}), "calib_oversample": 1}
            out = _bs(case, "pinn", **kw2)
            store(name, out["pinn"], time.time() - t0)
        elif name == "bs_wires":
            # a separate wires-only fit: the wire stage inside bs_pinn is co-adapted with the NN
            # correction and is not a stand-alone estimate
            out = _bs(case, "wires", **kw); store("bs_wires", out["wires"], time.time() - t0)
        elif name == "free":
            out = _bs(case, "free", **kw); store("free", out["free"], time.time() - t0)
        else:
            raise ValueError(f"unknown method {name}")
        log(f"  [{case.name}] {name:9s} {secs.get(name, float('nan')):7.1f}s")
    return maps, secs
