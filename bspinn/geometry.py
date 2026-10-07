"""Image-grid <-> physical-coordinate mapping and centred FFT helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# image axes (row, col, slice-normal) -> physical axes (x=0, y=1, z=2); B0 along z.
PLANES = {
    "axial":    (1, 0, 2),   # rows -> y (AP), cols -> x (LR), normal -> z
    "coronal":  (2, 0, 1),   # rows -> z (SI), cols -> x (LR), normal -> y
    "sagittal": (2, 1, 0),   # rows -> z (SI), cols -> y (AP), normal -> x
    "coronal_t": (0, 2, 1),  # transposed coronal: rows -> x, cols -> z (B0 along the image columns)
}


def fft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def fft2c_np(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def ifft2c_np(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def crop_center(x, shape):
    """Crop the last two dims of x around the centred-FFT origin (index n//2)."""
    H, W = x.shape[-2:]
    h, w = shape
    r0, c0 = H // 2 - h // 2, W // 2 - w // 2
    return x[..., r0:r0 + h, c0:c0 + w]


@dataclass
class SliceGeometry:
    """Physical geometry of a (multi-)slice dataset.

    Attributes:
        shape:          full-resolution image matrix (H, W)
        pixel_spacing:  (row, col) spacing in mm
        slice_pos:      (S,) slice positions along the slice normal in mm
        plane:          'axial' | 'coronal' | 'sagittal' (defines where B0 points)
    """
    shape: tuple
    pixel_spacing: tuple
    slice_pos: np.ndarray
    plane: str = "axial"

    @property
    def half_fov(self) -> float:
        """Normalisation length L (mm): half of the larger in-plane FOV."""
        H, W = self.shape
        return 0.5 * max(H * self.pixel_spacing[0], W * self.pixel_spacing[1])

    def grid(self, shape=None) -> torch.Tensor:
        """Normalised physical coordinates of pixel centres for a grid that spans the
        same FOV at a (possibly lower) resolution ``shape``. Returns (S, h, w, 3)."""
        H, W = self.shape
        h, w = shape or self.shape
        dr = self.pixel_spacing[0] * H / h
        dc = self.pixel_spacing[1] * W / w
        r = (np.arange(h) - h // 2) * dr
        c = (np.arange(w) - w // 2) * dc
        S = len(self.slice_pos)
        R, Cc = np.meshgrid(r, c, indexing="ij")
        out = np.zeros((S, h, w, 3), dtype=np.float32)
        ax = PLANES[self.plane]
        for s, zs in enumerate(self.slice_pos):
            out[s, ..., ax[0]] = R
            out[s, ..., ax[1]] = Cc
            out[s, ..., ax[2]] = zs
        return torch.from_numpy((out / float(self.half_fov)).astype(np.float32))

    def normal_axis(self) -> int:
        return PLANES[self.plane][2]

    def normal_range(self, margin: float) -> tuple:
        z = np.asarray(self.slice_pos) / self.half_fov
        return float(z.min() - margin), float(z.max() + margin)
