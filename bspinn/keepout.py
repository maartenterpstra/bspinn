"""Keep-out region for the conductors: signed distance to the imaged object.

The Biot–Savart model is only source-free (harmonic) where there are no conductors, so every
wire must stay at least ``margin`` away from the object. ``KeepOut`` holds the signed distance
transform of the (hole-filled) object support on the calibration grid and evaluates it, with
gradients w.r.t. the query points, anywhere in space:

* 2-D / multi-slice / dynamic: one in-plane distance map per slice; the anatomy continues through
  the plane, so each slice's object is its outline extruded by +/- ``slab`` along the slice normal
  (the same slab the Laplace penalty uses). Signed distance of that prism; min over slices.
* 3-D: one volumetric map.

Points outside the grid get the distance at the clamped grid position plus the Euclidean excess.
Distances are in the normalised units of the fit (half FOV = 1).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fill_holes(mask: np.ndarray, per_slice: bool) -> np.ndarray:
    from scipy.ndimage import binary_fill_holes
    if per_slice:
        return np.stack([binary_fill_holes(m) for m in mask])
    return binary_fill_holes(mask)


class KeepOut:
    def __init__(self, support: torch.Tensor, coords: torch.Tensor, volumetric: bool, slab: float = 0.15):
        """support: (S,h,w) [2-D] or (1,d,h,w) / (d,h,w) [3-D] 0/1; coords: matching grid (..., 3) flattened OK."""
        from scipy.ndimage import distance_transform_edt
        sup = support.detach().cpu().numpy() > 0
        self.volumetric, self.slab = volumetric, float(slab)
        if volumetric:
            sup = sup.reshape(sup.shape[-3:])[None]
        G = sup.shape[1:]                                                  # grid dims per slice
        c = coords.detach().cpu().reshape(sup.shape[0], *G, 3).numpy()
        # which coordinate component varies along each grid axis, and its step
        self.comp, self.step, self.origin = [], [], []
        for a in range(len(G)):
            idx0 = (0,) + (0,) * len(G)
            idx1 = list(idx0); idx1[1 + a] = 1 if G[a] > 1 else 0
            dv = c[tuple(idx1)] - c[idx0]
            k = int(np.argmax(np.abs(dv)))
            self.comp.append(k)
            self.step.append(float(dv[k]) if G[a] > 1 else 1.0)
            self.origin.append(float(c[idx0][k]))
        normal = [k for k in range(3) if k not in self.comp]
        self.normal = normal[0] if normal else None
        self.slice_z = torch.tensor(c[(slice(None),) + (0,) * len(G)][:, self.normal], dtype=torch.float32) \
            if self.normal is not None else None
        sdf = []
        sampling = [abs(s) for s in self.step]
        for m in sup:
            m = fill_holes(m[None], True)[0] if not volumetric else fill_holes(m, False)
            if m.any():
                d_out = distance_transform_edt(~m, sampling=sampling)
                d_in = distance_transform_edt(m, sampling=sampling)
                h = 0.5 * min(sampling)                                       # boundary between pixel centres
                sdf.append(np.where(m, -(d_in - h), d_out - h))
            else:
                sdf.append(np.full(m.shape, 10.0))
        self.sdf = torch.tensor(np.stack(sdf), dtype=torch.float32)        # (S, *G)
        self.G = G

    def to(self, device):
        self.sdf = self.sdf.to(device)
        if self.slice_z is not None:
            self.slice_z = self.slice_z.to(device)
        return self

    def distance(self, p: torch.Tensor) -> torch.Tensor:
        """Signed distance (negative inside the object) of points p (..., 3)."""
        shp = p.shape[:-1]
        p = p.reshape(-1, 3)
        idx, excess = [], 0.0
        for a, (k, st, o) in enumerate(zip(self.comp, self.step, self.origin)):
            i = (p[:, k] - o) / st                                         # fractional grid index
            ic = i.clamp(0, self.G[a] - 1)
            excess = excess + ((i - ic) * abs(st)) ** 2
            idx.append(ic)
        # grid_sample wants (x, y[, z]) = (last axis, ..., first axis), normalised to [-1, 1]
        g = torch.stack([2 * ic / max(n - 1, 1) - 1 for ic, n in zip(idx, self.G)][::-1], -1)
        S = self.sdf.shape[0]
        if len(self.G) == 2:
            grid = g[None, :, None, :].expand(S, -1, -1, -1)                # (S,Q,1,2)
            d = F.grid_sample(self.sdf[:, None], grid, mode="bilinear", align_corners=True)[:, 0, :, 0]  # (S,Q)
        else:
            grid = g[None, :, None, None, :].expand(S, -1, -1, -1, -1)
            d = F.grid_sample(self.sdf[:, None], grid, mode="bilinear", align_corners=True)[:, 0, :, 0, 0]
        if self.normal is not None:                                         # extruded outline (prism)
            d_xy = torch.where(d > 0, torch.sqrt(d.clamp_min(0) ** 2 + excess + 1e-12), d)
            d_z = (p[None, :, self.normal] - self.slice_z[:, None]).abs() - self.slab   # (S,Q)
            outside = torch.sqrt(d_xy.clamp_min(0) ** 2 + d_z.clamp_min(0) ** 2 + 1e-12)
            inside = torch.maximum(d_xy, d_z).clamp_max(0)
            d = (torch.where((d_xy > 0) | (d_z > 0), outside, inside)).min(0).values
        else:
            d = d[0]
            d = torch.where(d > 0, torch.sqrt(d.clamp_min(0) ** 2 + excess + 1e-12), d)
        return d.reshape(shp)


def segment_samples(V: torch.Tensor, n: int = 4) -> torch.Tensor:
    """Points along every segment of closed loops V (..., M, 3): (..., M*n, 3)."""
    W = torch.roll(V, -1, dims=-2)
    t = torch.arange(n, dtype=V.dtype, device=V.device) / n
    P = V[..., None, :] * (1 - t)[:, None] + W[..., None, :] * t[:, None]   # (..., M, n, 3)
    return P.reshape(*V.shape[:-2], -1, 3)


def keepout_penalty(keep: KeepOut, V: torch.Tensor, margin: float, n: int = 4) -> torch.Tensor:
    """sum over points along the wires of (relu(margin - d) / margin)^2, averaged over coils
    (V: (..., N, L, M, 3)). Normalised so that one point half a margin too close costs ~0.25/N:
    comparable to or larger than the data term (unlike a mean over all vertices)."""
    P = segment_samples(V, n)
    d = keep.distance(P)                                                   # (..., N, L, M*n)
    pen = (torch.relu(margin - d) / margin).pow(2)
    n_coils = V.shape[-4]
    return pen.sum() / (n_coils * max(1, int(np.prod(V.shape[:-4]))))
