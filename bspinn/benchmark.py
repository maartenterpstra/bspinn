"""Held-out k-space benchmark for sensitivity maps (model-agnostic).

A random fraction of the *acquired* samples in the ring between the calibration box and twice
its size is withheld; CG-SENSE with a given map set reconstructs from the rest and predicts the
withheld samples. NRMSE of that prediction is a gauge-invariant, ground-truth-free score that
works for undersampled Cartesian and non-Cartesian data.

Map sets differ in where they are non-zero (ESPIRiT thresholds to the object, Biot–Savart maps
are physically non-zero everywhere), which acts as an extra support constraint in SENSE.
``masked=True`` therefore applies one *common* data-derived support mask to every map set.
"""
from __future__ import annotations

import numpy as np
import torch

from .encoding import CartesianOp, NUDFTOp, radial_dcf
from .geometry import fft2c
from .recon import cg_sense, normalize_maps


def fourier_resample(M, shape):
    """Resample smooth maps/images to another matrix over the same FOV (k-space crop / pad)."""
    from .geometry import ifft2c
    S, N, h, w = M.shape
    H, W = shape
    k = fft2c(M)
    out = torch.zeros(S, N, H, W, dtype=k.dtype)
    hh, ww = min(h, H), min(w, W)
    out[..., H // 2 - hh // 2:H // 2 - hh // 2 + hh, W // 2 - ww // 2:W // 2 - ww // 2 + ww] = \
        k[..., h // 2 - hh // 2:h // 2 - hh // 2 + hh, w // 2 - ww // 2:w // 2 - ww // 2 + ww]
    return ifft2c(out) * ((H * W) / (h * w)) ** 0.5


class HeldoutBenchmark:
    def __init__(self, prob, shape, frac=0.2, seed=0, lam=1e-2, dilate=3):
        H, W = shape
        self.shape, self.N = shape, prob.N
        g = torch.Generator().manual_seed(seed)
        h, w = prob.kcalib
        if prob.noncart:
            traj = prob.traj_full
            band = (traj[:, 0].abs() < H / 2) & (traj[:, 1].abs() < W / 2)
            ring = (traj[:, 0].abs() < h) & (traj[:, 1].abs() < w)
            cand = torch.nonzero(band & ring & ~prob.calib_sel).squeeze(1)
            ho = torch.zeros_like(band); ho[cand[torch.rand(len(cand), generator=g) < frac]] = True
            train = band & ~ho
            self.op = NUDFTOp(traj[train], (H, W))
            self._op_ho = NUDFTOp(traj[ho], (H, W))
            self.weights = radial_dcf(traj[train])
            self.y_tr, self.y_ho = prob.ks_full[..., train], prob.ks_full[..., ho]
            self.lam, self.iters = lam * float(self.weights.mean()), 15
            self._predict = lambda M, x: self._op_ho.forward(M * x[:, None])
        else:
            mask = prob.mask_full
            calib_box = torch.zeros_like(mask)
            calib_box[..., H // 2 - h // 2:H // 2 - h // 2 + h, W // 2 - w // 2:W // 2 - w // 2 + w] = True
            ring = torch.zeros_like(mask)
            ring[..., max(H // 2 - h, 0):H // 2 + h, max(W // 2 - w, 0):W // 2 + w] = True
            cand = torch.nonzero((mask & ring & ~calib_box).flatten()).squeeze(1)
            hof = torch.zeros(mask.numel(), dtype=torch.bool)
            hof[cand[torch.rand(len(cand), generator=g) < frac]] = True
            ho = hof.reshape(mask.shape)
            self.op = CartesianOp(mask & ~ho)
            self.weights = None
            self.y_tr = prob.ks_full * (mask & ~ho)
            hoN = ho.expand(-1, prob.N, -1, -1)
            self.y_ho = prob.ks_full[hoN]
            self.lam, self.iters = lam, 40
            self._predict = lambda M, x: fft2c(M * x[:, None])[hoN]
        # common support: low-res calibration RSS, upsampled, thresholded, dilated
        rss = (fourier_resample(prob.I, (H, W)).abs() ** 2).sum(1).sqrt()
        sup = (rss > prob.cfg.support_thresh * torch.quantile(rss.flatten(), 0.99)).float()
        sup = torch.nn.functional.max_pool2d(sup[:, None], 2 * dilate + 1, 1, dilate)[:, 0]
        self.support = sup

    @property
    def n_heldout_per_coil(self):
        return int(self.y_ho.numel() // self.N)

    def prepare(self, M):
        M = torch.as_tensor(np.asarray(M) if not torch.is_tensor(M) else M).to(torch.complex64)
        if M.shape[-2:] != tuple(self.shape):
            M = fourier_resample(M, self.shape)
        nz = ((M.abs() ** 2).sum(1, keepdim=True) > 0)
        return normalize_maps(M) * nz

    def recon(self, M, masked=False):
        M = M * self.support[:, None] if masked else M
        return cg_sense(self.op, self.y_tr, M, lam=self.lam, iters=self.iters, weights=self.weights)

    def score(self, M, masked=False):
        Mm = M * self.support[:, None] if masked else M
        x = self.recon(M, masked)
        return float((self._predict(Mm, x) - self.y_ho).norm() / self.y_ho.norm()), x
