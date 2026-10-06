"""Prepare the real datasets used by scripts/evaluate_methods.py (writes data/eval_*.npz).

knee15  : BART workshop 'kspace_fs' (fastMRI-style coronal knee, 15 channels, fully sampled except
          the zero-filled partial-Fourier PE edges). Readout 2x oversampling removed, noise covariance
          estimated from signal-free background pixels, data pre-whitened (sigma = 1 per sample).
brain8  : BART workshop 'pics-brain' (8 channels, prospective 2-D Poisson disc). No noise scan:
          sigma estimated from the k-space periphery (signal ~ 0 there).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bspinn.data import _read_cfl  # noqa: E402


def ifft2c(k):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def background_noise_cov(img, frac=0.12, fill=1.0):
    """Covariance of the lowest-RSS pixels (after a 7x7 smoothing of RSS, to pick regions not pixels).
    ``fill``: fraction of k-space actually sampled (image-domain noise variance scales with it)."""
    from scipy.ndimage import uniform_filter
    rss = np.sqrt((np.abs(img) ** 2).sum(0))
    sm = uniform_filter(rss, 7)
    sel = sm <= np.quantile(sm, frac)
    x = img[:, sel]                                         # (N, P)
    cov = x @ x.conj().T / x.shape[1] / fill
    return cov, sel


def whiten(k, cov):
    L = np.linalg.cholesky(cov)
    W = np.linalg.inv(L)
    return np.einsum("ij,j...->i...", W, k), W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knee", required=True, help="path to kspace_fs (BART cfl, no extension)")
    ap.add_argument("--brain", required=True, help="path to pics-brain ksp")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---------------------------------------------------------------- knee
    k = _read_cfl(a.knee).squeeze()                       # (640, 368, 15)
    k = np.transpose(k, (2, 0, 1))                         # (N, RO, PE)
    mask = np.abs(k).sum(0) > 0
    img = ifft2c(k)
    RO = img.shape[1]
    img = img[:, RO // 4: RO // 4 + RO // 2]               # remove 2x readout oversampling
    mask = mask[RO // 4: RO // 4 + RO // 2] | mask.any(0)[None]   # PE lines sampled over full RO
    lines = mask.any(0)
    fill = lines.mean()
    cov, sel = background_noise_cov(img, fill=fill)
    k = fft2c(img) * lines[None, None]
    kw, W = whiten(k, cov)
    # check: whitened background pixel variance should be ~1 * fill
    iw = ifft2c(kw)
    s2 = (np.abs(iw[:, sel]) ** 2).mean() / fill
    print(f"knee: {kw.shape}, sampled PE lines {lines.sum()}/{lines.size}, whitened noise var {s2:.3f}, "
          f"cond(cov) {np.linalg.cond(cov):.1f}")
    np.savez_compressed(os.path.join(a.out, "eval_knee15.npz"), ksp=kw.astype(np.complex64), mask=np.broadcast_to(lines[None], kw.shape[1:]),
                        noise_cov=cov, whitening=W, sigma=1.0, pixel_spacing=(0.4375, 0.4375), plane="coronal",
                        source="bart-workshop ismrm2021/neural_networks kspace_fs (knee, 15ch)")

    # ---------------------------------------------------------------- brain
    kb = _read_cfl(a.brain).squeeze()                      # (RO, PE, N) for pics-brain: (1,230,180,8)?
    if kb.ndim == 3:
        kb = np.transpose(kb, (2, 0, 1))
    mb = np.abs(kb).sum(0) > 0
    # sigma from the sampled k-space periphery (outer 10 % radius): signal there is ~ noise
    H, W2 = mb.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W2), indexing="ij")
    per = mb & (np.hypot(yy, xx) > 0.9)
    v = (np.abs(kb[:, per]) ** 2).mean(1)
    cov_b = np.diag(v)
    kbw = kb / np.sqrt(v)[:, None, None]
    print(f"brain: {kbw.shape}, sampled {mb.mean():.3f}, per-coil sigma {np.sqrt(v).round(6)}")
    np.savez_compressed(os.path.join(a.out, "eval_brain8.npz"), ksp=kbw.astype(np.complex64), mask=mb, noise_cov=cov_b, sigma=1.0,
                        pixel_spacing=(1.0, 1.0), plane="axial", source="bart-workshop ismrm2016/pics-brain (8ch, Poisson disc)")


if __name__ == "__main__":
    main()
