"""Keep-out region for the conductors: signed distance to the imaged object.

The Biot–Savart model is only source-free (harmonic) where there are no conductors, so every
wire must stay at least ``margin`` away from the object. ``KeepOut`` holds the signed distance
transform of the (hole-filled) object support on the calibration grid and evaluates it, with
gradients w.r.t. the query points, anywhere in space:

* 2-D / multi-slice / dynamic: one in-plane distance map per slice; the anatomy continues through
  the plane, so each slice's object is its outline extruded by +/- ``slab`` along the slice normal
  (the same slab the Laplace penalty uses). Signed distance of that prism; min over slices.
* 3-D: one volumetric map.

Outside the grid (the field of view) the object is taken as truncated at the FOV edge: a point beyond
the edge is at least its distance to the edge away (objects cut by the FOV, e.g. a knee touching
the image border, have coils just beyond it).
Distances are in the normalised units of the fit (half FOV = 1).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def clean_support(mask: np.ndarray, per_slice: bool, min_frac: float = 0.2) -> np.ndarray:
    """Drop small disconnected blobs (< min_frac of the largest component: e.g. periodic wrap-around of
    the low-resolution image when the object touches the FOV edge), then fill holes."""
    from scipy.ndimage import label
    def one(m):
        lab, n = label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel())[1:]
            m = np.isin(lab, 1 + np.nonzero(sizes >= min_frac * sizes.max())[0])
        return m
    m = np.stack([one(x) for x in mask]) if per_slice else one(mask)
    return fill_holes(m, per_slice)


def fill_holes(mask: np.ndarray, per_slice: bool) -> np.ndarray:
    from scipy.ndimage import binary_fill_holes
    if per_slice:
        return np.stack([binary_fill_holes(m) for m in mask])
    return binary_fill_holes(mask)


class KeepOut:
    def __init__(self, support: torch.Tensor, coords: torch.Tensor, volumetric: bool, slab: float = 0.15,
                 extend: float = 0.0):
        """support: (S,h,w) [2-D] or (1,d,h,w) / (d,h,w) [3-D] 0/1; coords: matching grid (..., 3) flattened OK."""
        from scipy.ndimage import distance_transform_edt
        sup = support.detach().cpu().numpy() > 0
        self.volumetric, self.slab, self.extend = volumetric, float(slab), float(extend)
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
            m = clean_support(m[None], True)[0] if not volumetric else clean_support(m, False)
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
            d_xy = self._truncate(d, excess)
            d_z = (p[None, :, self.normal] - self.slice_z[:, None]).abs() - self.slab   # (S,Q)
            outside = torch.sqrt(d_xy.clamp_min(0) ** 2 + d_z.clamp_min(0) ** 2 + 1e-12)
            inside = torch.maximum(d_xy, d_z).clamp_max(0)
            d = (torch.where((d_xy > 0) | (d_z > 0), outside, inside)).min(0).values
        else:
            d = self._truncate(d[0], excess)
        return d.reshape(shp)


    def _truncate(self, d, excess):
        """Beyond the grid: distance to the object as seen in the FOV, assumed to continue ``extend``
        beyond the FOV edge where it touches it (lower bound)."""
        if not torch.is_tensor(excess):
            return d
        e = torch.sqrt(excess + 1e-12)
        beyond = torch.where(d > 0, torch.sqrt(d.clamp_min(0) ** 2 + excess + 1e-12), e - self.extend)
        return torch.where(excess > 0, beyond, d)

    def fov_depth(self, p: torch.Tensor) -> torch.Tensor:
        """How deep points lie inside the imaged region (the FOV box; 2-D: within +/- slab of a slice),
        0 outside it. Used by the optional prior that keeps conductors out of the image."""
        shp = p.shape[:-1]
        p = p.reshape(-1, 3)
        depth = None
        for a, (k, st, o) in enumerate(zip(self.comp, self.step, self.origin)):
            e = o + st * (self.G[a] - 1)
            lo, hi = min(o, e) - 0.5 * abs(st), max(o, e) + 0.5 * abs(st)
            da = torch.minimum(p[:, k] - lo, hi - p[:, k])
            depth = da if depth is None else torch.minimum(depth, da)
        depth = depth.clamp_min(0)
        if self.normal is not None:
            dz = (p[None, :, self.normal] - self.slice_z[:, None]).abs().min(0).values
            depth = depth * torch.relu(1 - dz / self.slab)
        return depth.reshape(shp)


def fov_penalty(keep: KeepOut, V: torch.Tensor, margin: float, n: int = 4) -> torch.Tensor:
    """Optional prior: conductors inside the imaged region (in air: the keep-out already excludes the
    object) cost (depth / margin)^2 per sample point, averaged over coils."""
    dep = keep.fov_depth(segment_samples(V, n))
    n_coils = V.shape[-4]
    return (dep / margin).pow(2).sum() / (n_coils * max(1, int(np.prod(V.shape[:-4]))))


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
