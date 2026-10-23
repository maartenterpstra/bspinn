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



def b0_frame(b0_dir):
    """B0 direction in image coordinates (row, column, normal; any length/sign) -> F (3,3) with
    model coordinates = F @ image coordinates and model z along B0 (F[2] = unit B0). The in-plane completion
    (model x, y) is arbitrary: the Biot-Savart model is rotation-invariant about B0."""
    b = np.asarray(b0_dir, np.float64).reshape(3)
    b = b / np.linalg.norm(b)
    a = np.array([0.0, 0.0, 1.0]) if abs(b[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = a - a.dot(b) * b; e1 /= np.linalg.norm(e1)
    e2 = np.cross(b, e1)
    return np.stack([e1, e2, b])


@dataclass
class SliceGeometry:
    """Physical geometry of a (multi-)slice dataset.

    Attributes:
        shape:          full-resolution image matrix (H, W)
        pixel_spacing:  (row, col) spacing in mm
        slice_pos:      (S,) slice positions along the slice normal in mm
        plane:          'axial' | 'coronal' | 'sagittal' | 'coronal_t' (defines where B0 points), or an oblique
                        slice: the B0 direction in image coordinates (row, column, normal), a 3-sequence
                        (e.g. from the scanner header: the B0 (z) components of read, phase and slice direction)
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
        if self.oblique:
            F = b0_frame(self.plane)
            for s, zs in enumerate(self.slice_pos):
                img = np.stack([R, Cc, np.full_like(R, zs)], -1)
                out[s] = img @ F.T
            return torch.from_numpy((out / float(self.half_fov)).astype(np.float32))
        ax = PLANES[self.plane]
        for s, zs in enumerate(self.slice_pos):
            out[s, ..., ax[0]] = R
            out[s, ..., ax[1]] = Cc
            out[s, ..., ax[2]] = zs
        return torch.from_numpy((out / float(self.half_fov)).astype(np.float32))

    @property
    def oblique(self) -> bool:
        return not isinstance(self.plane, str)

    def axis_vec(self, i) -> np.ndarray:
        """Model-coordinate unit vector of image axis i (0 rows, 1 columns, 2 normal)."""
        if self.oblique:
            return b0_frame(self.plane)[:, i].copy()
        v = np.zeros(3); v[PLANES[self.plane][i]] = 1.0
        return v

    def normal_axis(self) -> int:
        """Model axis closest to the slice normal (exact for the named planes)."""
        if self.oblique:
            return int(np.argmax(np.abs(self.axis_vec(2))))
        return PLANES[self.plane][2]

    def normal_range(self, margin: float) -> tuple:
        z = np.asarray(self.slice_pos) / self.half_fov
        return float(z.min() - margin), float(z.max() + margin)
