"""Physics core: Biot–Savart fields, B1 receive mapping, and autograd differential operators.

Conventions
-----------
* Physical axes (x, y, z) with the static field B0 along +z.
* Units: lengths are in normalised coordinates (see ``geometry.py``); the constant
  mu0 / (4 pi) and the coil current are absorbed into per-coil amplitudes, so fields
  are only defined up to a per-coil complex gain.
* Receive sensitivity (principle of reciprocity, quasi-static regime):
      C(r) = g * (Bx(r) + i * h * By(r))
  where h = +1 or -1 encodes the handedness convention (sign of gamma, direction of
  B0, conjugation convention of the reconstruction). For a real, quasi-static field
  this is B1^- up to a constant; the per-coil complex gain g = a * exp(i*theta)
  carries the coil current amplitude and the receive-chain (cable/preamp) phase.
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------------------
# Biot–Savart for closed polygonal wire loops (exact, closed form per straight segment)
# --------------------------------------------------------------------------------------
def biot_savart_polyline(verts: torch.Tensor, points: torch.Tensor, eps: float = 1e-9, core: float = 0.0) -> torch.Tensor:
    """Magnetic field of closed polygonal loops carrying unit current.

    Uses the exact closed-form expression for a finite straight segment
    (current from A to B, observation point P, a = A - P, b = B - P):

        B = (a x b) (|a| + |b|) / ( |a| |b| (|a||b| + a.b) )      [units of mu0 I / 4 pi]

    Args:
        verts:  (..., M, 3) vertices of closed loops (segment k joins vertex k and k+1 mod M).
        points: (P, 3) observation points.
        core:   conductor radius (Rankine core). 0: ideal thin wire (field ~ 1/d near the wire).
                > 0: within distance d < core of a segment the field is scaled by d^2/core^2 (a
                conductor of finite cross-section: bounded, linear in d inside), exact outside.
    Returns:
        (..., P, 3) magnetic field at each point, summed over the loop's segments.
    """
    A = verts.unsqueeze(-2)                                  # (..., M, 1, 3)
    Bv = torch.roll(verts, shifts=-1, dims=-2).unsqueeze(-2)  # (..., M, 1, 3)
    a = A - points                                           # (..., M, P, 3)
    b = Bv - points
    na = torch.linalg.norm(a, dim=-1)
    nb = torch.linalg.norm(b, dim=-1)
    axb = torch.cross(a, b, dim=-1)
    denom = na * nb * (na * nb + (a * b).sum(-1))
    factor = (na + nb) / (denom + eps)
    if core > 0:
        ab = Bv - A                                                       # segment vector
        d2 = (axb * axb).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)     # squared distance to the line
        factor = factor * (d2 / core ** 2).clamp(max=1.0)
    return (axb * factor.unsqueeze(-1)).sum(dim=-3)          # sum over segments


def transverse_to_sensitivity(Bx: torch.Tensor, By: torch.Tensor, handedness: int) -> torch.Tensor:
    """C = Bx + i*h*By (inputs may be real or complex tensors)."""
    return Bx + 1j * handedness * By


# --------------------------------------------------------------------------------------
# Autograd differential operators
# --------------------------------------------------------------------------------------
def grad(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Gradient of sum(y) w.r.t. x. Valid as a per-point gradient when each output row
    depends only on its own input row (true for point-wise networks and field evaluators)."""
    (g,) = torch.autograd.grad(y.sum(), x, create_graph=create_graph, retain_graph=True)
    return g


def divergence_and_curl(field: torch.Tensor, x: torch.Tensor):
    """Divergence and curl of a vector field F(x) evaluated point-wise.

    Args:
        field: (..., 3) field values computed from x with requires_grad.
        x:     (..., 3) coordinates.
    Returns:
        div (...,), curl (..., 3)
    """
    J = torch.stack([grad(field[..., i], x) for i in range(3)], dim=-2)  # J[..., i, j] = dF_i/dx_j
    div = J[..., 0, 0] + J[..., 1, 1] + J[..., 2, 2]
    curl = torch.stack([J[..., 2, 1] - J[..., 1, 2],
                        J[..., 0, 2] - J[..., 2, 0],
                        J[..., 1, 0] - J[..., 0, 1]], dim=-1)
    return div, curl


def maxwell_static_residual(field_fn, points: torch.Tensor, n_coils: int):
    """Relative magnetostatic (source-free Biot–Savart) residual of a field model.

    For every field component k of shape (N, P, 3, K) returned by ``field_fn``,
    computes RMS(div B) and RMS(|curl B|) normalised by RMS(|B|) (units 1/length,
    in normalised coordinates). A true Biot–Savart field with sources outside the
    region gives zero for both. ``field_fn`` receives per-coil points of shape (N, P, 3)
    so that point-wise autograd derivatives are not mixed across coils.
    """
    x = points.detach().unsqueeze(0).repeat(n_coils, 1, 1).requires_grad_(True)
    B = field_fn(x)                                   # (N, P, 3, K)
    divs, curls, mags = [], [], []
    for k in range(B.shape[-1]):
        d, c = divergence_and_curl(B[..., k], x)
        divs.append(d.pow(2).mean(dim=-1))            # (N,)
        curls.append(c.pow(2).sum(-1).mean(dim=-1))
        mags.append(B[..., k].pow(2).sum(-1).mean(dim=-1))
    mag = torch.stack(mags).sum(0).clamp_min(1e-30)
    div_rel = (torch.stack(divs).sum(0) / mag).sqrt()
    curl_rel = (torch.stack(curls).sum(0) / mag).sqrt()
    return div_rel.detach(), curl_rel.detach()
