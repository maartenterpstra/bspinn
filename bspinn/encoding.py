"""Encoding operators: Cartesian (masked centred FFT) and non-Cartesian (exact NUDFT).

The NUDFT is evaluated separably, E f = sum_ij a_si f_ij b_sj with a_si = exp(-2 pi i kx_s x_i/h),
which needs O(M*w) memory instead of O(M*h*w); fine for calibration-size grids and moderate
reconstruction matrices on CPU. k-space coordinates are in cycles/FOV (Nyquist grid = integers).
"""
from __future__ import annotations

import math

import torch

from .geometry import fft2c, ifft2c


class CartesianOp:
    def __init__(self, mask: torch.Tensor):
        self.mask = mask.to(torch.complex64)        # broadcastable to (S,N,H,W)

    def to(self, device):
        self.mask = self.mask.to(device)
        return self

    def forward(self, x):
        return self.mask * fft2c(x)

    def adjoint(self, y):
        return ifft2c(self.mask * y)


class NUDFTOp:
    """traj: (M, 2) [k_row, k_col] in cycles/FOV; image grid (h, w) spanning the FOV."""

    def __init__(self, traj: torch.Tensor, shape, chunk: int = 4):
        h, w = shape
        r = torch.arange(h, dtype=torch.float64) - h // 2
        c = torch.arange(w, dtype=torch.float64) - w // 2
        kr, kc = traj[:, 0].double(), traj[:, 1].double()
        self.A = torch.exp(-2j * math.pi * kr[:, None] * r[None] / h).to(torch.complex64)   # (M,h)
        self.B = torch.exp(-2j * math.pi * kc[:, None] * c[None] / w).to(torch.complex64)   # (M,w)
        self.scale = 1.0 / math.sqrt(h * w)
        self.chunk = chunk

    def to(self, device):
        self.A, self.B = self.A.to(device), self.B.to(device)
        return self

    def forward(self, x):                      # (..., h, w) -> (..., M)
        lead = x.shape[:-2]
        xf = x.reshape(-1, *x.shape[-2:])
        out = [((self.A @ xf[i:i + self.chunk]) * self.B).sum(-1) for i in range(0, xf.shape[0], self.chunk)]
        return (torch.cat(out) * self.scale).reshape(*lead, -1)

    def adjoint(self, y):                      # (..., M) -> (..., h, w)
        lead = y.shape[:-1]
        yf = y.reshape(-1, y.shape[-1])
        Ah = self.A.conj().T
        out = [Ah @ (self.B.conj() * yf[i:i + self.chunk, :, None]) for i in range(0, yf.shape[0], self.chunk)]
        return (torch.cat(out) * self.scale).reshape(*lead, *self.A.shape[1:], self.B.shape[1])


class NUDFTOpBatched:
    """One trajectory per slice/frame: traj (S, M, 2) with validity mask (S, M) for padding.
    forward: (S, N, h, w) -> (S, N, M);  adjoint: (S, N, M) -> (S, N, h, w)."""

    def __init__(self, traj: torch.Tensor, valid: torch.Tensor, shape):
        h, w = shape
        r = torch.arange(h, dtype=torch.float64) - h // 2
        c = torch.arange(w, dtype=torch.float64) - w // 2
        kr, kc = traj[..., 0].double(), traj[..., 1].double()
        self.A = torch.exp(-2j * math.pi * kr[..., None] * r / h).to(torch.complex64)      # (S,M,h)
        self.B = torch.exp(-2j * math.pi * kc[..., None] * c / w).to(torch.complex64)      # (S,M,w)
        self.valid = valid.to(torch.complex64)                                             # (S,M)
        self.scale = 1.0 / math.sqrt(h * w)

    def to(self, device):
        self.A, self.B, self.valid = self.A.to(device), self.B.to(device), self.valid.to(device)
        return self

    def subset(self, sl):
        op = NUDFTOpBatched.__new__(NUDFTOpBatched)
        op.A, op.B, op.valid, op.scale = self.A[sl], self.B[sl], self.valid[sl], self.scale
        return op

    def forward(self, x):
        out = ((self.A[:, None] @ x) * self.B[:, None]).sum(-1)                           # (S,N,M)
        return out * self.valid[:, None] * self.scale

    def adjoint(self, y):
        y = y * self.valid[:, None]
        return (self.A.conj().transpose(-1, -2)[:, None] @ (self.B.conj()[:, None] * y[..., None])) * self.scale


def radial_dcf(traj: torch.Tensor) -> torch.Tensor:
    """Ramp density compensation for radial sampling (|k|, with a floor at the centre)."""
    k = traj.norm(dim=-1)
    return k.clamp_min(0.25)
