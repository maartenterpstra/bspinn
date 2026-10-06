"""Coil-sensitivity models constrained to Biot–Savart solutions.

Two complementary ways of enforcing the constraint:

``WireLoopCoils`` (hard constraint)
    Each receive channel is one or more closed polygonal current loops with learnable
    vertices. The sensitivity is computed with the exact Biot–Savart integral, so every
    model output is a Biot–Savart field *by construction*. Low-dimensional and robust,
    but restricted to the chosen wire topology.

``HarmonicPINN`` (soft constraint, physics-informed neural field)
    A coordinate network outputs a magnetic scalar potential Phi_j(r) per coil. The field
    is B = -grad Phi (curl-free exactly, via autograd), and the PDE residual laplace(Phi)
    (= -div B) is penalised at collocation points. In a source-free region, the set of
    fields with div B = 0 and curl B = 0 is *exactly* the set of fields produced by
    currents outside that region (Biot–Savart with external sources), so this is the
    general, geometry-free version of the constraint.

Both models handle an arbitrary number of coils N: the wire model owns N loop sets; the
PINN shares one network across coils and conditions it on a learnable per-coil latent
code (auto-decoder style), so a trained trunk can be re-used for a different array.

Field modes
    'real'    : quasi-static. Field is real; sensitivity = exp(i theta_j) (Bx + i h By).
                Coil phase is a single constant per channel (receive chain).
    'complex' : potential is complex (both parts harmonic). Allows spatially varying
                phase from conductive loading, but on a single 2-D slice it constrains
                much less (see README).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .physics import biot_savart_polyline, grad, transverse_to_sensitivity


def make_trunk(d_in: int, width: int, depth: int, d_out: int) -> nn.Sequential:
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(d_in, width), nn.Tanh()]
        d_in = width
    layers.append(nn.Linear(d_in, d_out))
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    return net


class CoilModel(nn.Module):
    n_coils: int
    handedness: int

    def field(self, points: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """Return B with shape (N, P, 3, K); K=1 real field, K=2 (re, im)."""
        raise NotImplementedError

    def gain(self) -> torch.Tensor:
        return torch.ones(self.n_coils, dtype=torch.complex64, device=self.device)

    @property
    def device(self):
        return next(self.parameters()).device

    def sensitivity(self, points: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        """Complex sensitivities, shape (N, P)."""
        B = self.field(points, create_graph=create_graph)
        if B.shape[-1] == 1:
            Bx, By = B[:, :, 0, 0], B[:, :, 1, 0]
        else:
            Bx = torch.complex(B[:, :, 0, 0], B[:, :, 0, 1])
            By = torch.complex(B[:, :, 1, 0], B[:, :, 1, 1])
        C = transverse_to_sensitivity(Bx, By, self.handedness)
        return C * self.gain()[:, None]

    def sensitivity_batched(self, points: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
        """Inference-time evaluation on large grids (no graph kept)."""
        out = []
        for i in range(0, points.shape[0], chunk):
            p = points[i:i + chunk]
            with torch.enable_grad():
                out.append(self.sensitivity(p, create_graph=False).detach())
        return torch.cat(out, dim=1)

    def physics_loss(self, points: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=points.device)


# ======================================================================================
class WireLoopCoils(CoilModel):
    """Hard Biot–Savart model: N channels x L loops x M vertices."""

    def __init__(self, n_coils: int, loops_per_coil: int = 1, verts_per_loop: int = 12,
                 handedness: int = 1):
        super().__init__()
        self.n_coils, self.handedness = n_coils, handedness
        self.verts = nn.Parameter(torch.zeros(n_coils, loops_per_coil, verts_per_loop, 3))
        self.amp = nn.Parameter(torch.zeros(n_coils, loops_per_coil))   # real loop currents
        self.theta = nn.Parameter(torch.zeros(n_coils))                  # receive-chain phase

    def gain(self):
        return torch.exp(1j * self.theta)

    def field(self, points, create_graph=True, verts=None):
        """``verts`` optionally overrides the loop geometry (e.g. moving coils), shape
        (..., N, L, M, 3) with arbitrary leading batch dims -> field (..., N, P, 3, 1)."""
        V = self.verts if verts is None else verts
        if points.dim() == 2:      # shared points (P,3)
            B = biot_savart_polyline(V, points)                               # (...,N,L,P,3)
        else:                      # per-coil points (N,P,3)
            B = biot_savart_polyline(V, points[:, None, None])                # (...,N,L,P,3)
        B = (B * self.amp[..., None, None]).sum(-3)                           # (...,N,P,3)
        return B.unsqueeze(-1)

    # ---------------------------------------------------------------- initialisation
    @torch.no_grad()
    def init_from_data(self, coords: torch.Tensor, weights: torch.Tensor, support: torch.Tensor,
                       loop_radius: float = 0.4, standoff: float = 0.15, seed: int = 0,
                       normal_axis: int = 2, z_offset: float = 0.0):
        """Place one loop per channel facing the object, at the channel's centroid direction.

        coords:  (P,3) calibration grid points; weights: (N,P) |coil image|^2;
        support: (Q,3) points inside the object.
        normal_axis, z_offset: loops are displaced by +/-z_offset (alternating per channel)
        along the slice normal. For thin slabs this breaks the mirror symmetry z -> -z: a loop
        centred in the slice plane has zero gradient w.r.t. its out-of-plane position, so it
        could otherwise never move to the ring it really sits on. Skipped when the support
        already spans more than 0.3 along the normal (volumes: the centroid carries that info).
        """
        g = torch.Generator().manual_seed(seed)
        coords, weights, support = coords.cpu(), weights.cpu(), support.cpu()
        N, L, M, _ = self.verts.shape
        R0 = support.norm(dim=-1).max().item() + standoff
        w = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
        cent = w @ coords                                               # (N,3)
        ang = torch.linspace(0, 2 * math.pi, M + 1)[:-1]
        zs = support[:, normal_axis]
        thin = (zs.max() - zs.min()).item() < 0.3
        for j in range(N):
            u = cent[j].clone()
            if u.norm() < 0.05:
                phi = torch.rand(1, generator=g).item() * 2 * math.pi
                u = torch.tensor([math.cos(phi), math.sin(phi), 0.0])
            u = u / u.norm()
            ref = torch.tensor([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else torch.tensor([1.0, 0.0, 0.0])
            e1 = torch.linalg.cross(u, ref); e1 = e1 / e1.norm()
            e2 = torch.linalg.cross(u, e1)
            for l in range(L):
                jit = 0.0 if l == 0 else 0.1 * torch.randn(3, generator=g)
                c = R0 * u + jit
                if thin and z_offset:
                    c = c.clone(); c[normal_axis] += z_offset * (1 if j % 2 == 0 else -1)
                self.verts[j, l] = c + loop_radius * (torch.cos(ang)[:, None] * e1 + torch.sin(ang)[:, None] * e2)
            self.amp[j] = 0.0
            self.amp[j, 0] = 1.0

    @torch.no_grad()
    def init_gains(self, coords, coil_images, weights):
        """Least-squares per-coil amplitude and phase relative to a reference coil,
        using object-phase-free ratios I_j * conj(I_ref)."""
        C = self.sensitivity(coords, create_graph=False)                 # (N,P)
        ref = weights.sum(1).argmax()
        Iref = coil_images[ref]
        wsum = weights.sum(0)
        num = (coil_images * Iref.conj() * (C * C[ref].conj()).conj() * (wsum > 0)).sum(1)
        self.theta.add_(torch.angle(num))
        # amplitude: match |C_j| / |C_ref| to |I_j| / |I_ref| in a least-squares sense
        rI = coil_images.abs() / Iref.abs().clamp_min(1e-12)
        rC = C.abs() / C[ref].abs().clamp_min(1e-12)
        ww = wsum / wsum.sum()
        scale = (ww * rI * rC).sum(1) / (ww * rC * rC).sum(1).clamp_min(1e-12)
        self.amp.mul_(scale[:, None])

    def regularisers(self, support: torch.Tensor, margin: float = 0.05):
        """(i) keep conductors outside the imaged object (sources must be external for the
        field to be source-free there); (ii) keep loop shapes smooth."""
        v = self.verts.reshape(-1, 3)
        d = torch.cdist(v, support).min(dim=1).values
        outside = torch.relu(margin - d).pow(2).mean()
        v2 = self.verts
        curv = (torch.roll(v2, 1, -2) - 2 * v2 + torch.roll(v2, -1, -2)).pow(2).sum(-1).mean()
        return outside, curv


def trunk_value_and_grad(net: nn.Sequential, x: torch.Tensor, z: torch.Tensor):
    """Value and spatial gradient of a Linear/Tanh MLP at inputs cat[x, z] by forward-mode
    propagation of the three tangents d/dx_i. Same result as autograd.grad, but a single
    differentiable pass (no double backward), ~3-5x faster for training on large 3-D grids.
    x: (..., 3), z: (..., L)  ->  phi (..., K), dphi/dx (..., 3, K)."""
    layers = list(net)
    W0 = layers[0].weight
    h = torch.nn.functional.linear(torch.cat([x, z], -1), W0, layers[0].bias)
    T = W0[:, :3].T.expand(*h.shape[:-1], 3, W0.shape[0])                    # (..., 3, width)
    for layer in layers[1:]:
        if isinstance(layer, nn.Tanh):
            h = torch.tanh(h)
            T = T * (1 - h * h).unsqueeze(-2)
        elif isinstance(layer, nn.Linear):
            h = torch.nn.functional.linear(h, layer.weight, layer.bias)
            T = T @ layer.weight.T
        else:
            raise TypeError("forward-mode gradient supports Linear/Tanh trunks only")
    return h, T


# ======================================================================================
class HarmonicPINN(CoilModel):
    """Physics-informed neural field for the magnetic scalar potential of each coil."""

    def __init__(self, n_coils: int, field_mode: str = "real", latent_dim: int = 16,
                 width: int = 64, depth: int = 4, handedness: int = 1, base: WireLoopCoils | None = None,
                 net: nn.Module | None = None, last_init: str = "auto"):
        """If ``base`` is given, the network learns a harmonic *correction* to the exact
        Biot–Savart field of the wire model: B = B_wires - grad Phi_NN. The output layer
        starts at zero, so training starts exactly at the wire solution."""
        super().__init__()
        assert field_mode in ("real", "complex")
        self.n_coils, self.handedness, self.field_mode = n_coils, handedness, field_mode
        self.K = 1 if field_mode == "real" else 2
        self.base = base
        self.codes = nn.Parameter(0.5 * torch.randn(n_coils, latent_dim))
        if net is None:                      # ``net`` may be shared between datasets (amortised training)
            net = make_trunk(3 + latent_dim, width, depth, self.K)
            last = "zero" if (last_init == "auto" and base is not None) else last_init
            if last == "zero":
                nn.init.zeros_(net[-1].weight)
            elif last == "small":
                net[-1].weight.data.mul_(1e-2)
        self.net = net
        th = base.theta.detach().clone() if base is not None else torch.zeros(n_coils)
        self.theta = nn.Parameter(th, requires_grad=(field_mode == "real"))

    def gain(self):
        if self.field_mode == "complex":
            return torch.ones(self.n_coils, dtype=torch.complex64, device=self.theta.device)
        return torch.exp(1j * self.theta)

    def correction_field(self, points, create_graph=True):
        """-grad Phi_NN only (no wire part); (N,P,3,K)."""
        x = self._xin(points)
        phi = self.potential(x)
        return torch.stack([-grad(phi[..., k], x, create_graph) for k in range(self.K)], dim=-1)

    def _base_field(self, x):
        if self.base is None:
            return None
        Bw = self.base.field(x)                                          # (N,P,3,1) real
        if self.K == 2:                                                  # carry the coil phase in the field
            th = self.base.theta
            Bw = torch.cat([Bw * torch.cos(th)[:, None, None, None], Bw * torch.sin(th)[:, None, None, None]], -1)
        return Bw


    def _xin(self, points):
        if points.dim() == 2:
            x = points.detach().unsqueeze(0).repeat(self.n_coils, 1, 1).requires_grad_(True)
        else:
            x = points if points.requires_grad else points.detach().clone().requires_grad_(True)
        return x

    def potential(self, x):
        z = self.codes[:, None, :].expand(-1, x.shape[1], -1)
        return self.net(torch.cat([x, z], dim=-1))                     # (N,P,K)

    forward_mode = True        # small batches: spatial gradient by forward-mode tangents (trunk_value_and_grad)

    def field(self, points, create_graph=True):
        x = self._xin(points)
        if (self.forward_mode and x.shape[0] * x.shape[1] <= 65536
                and all(isinstance(l, (nn.Linear, nn.Tanh)) for l in self.net)):
            z = self.codes[:, None, :].expand(-1, x.shape[1], -1)
            _, g = trunk_value_and_grad(self.net, x, z)                       # (N,P,3,K)
            B = -g
        else:
            phi = self.potential(x)
            B = torch.stack([-grad(phi[..., k], x, create_graph) for k in range(self.K)], dim=-1)
        Bw = self._base_field(x)
        return B if Bw is None else B + Bw

    def physics_loss(self, points):
        """Relative Laplace residual mean(lap Phi^2) / mean(|B_transverse|^2), per coil.
        (The wire part of the field is harmonic analytically and contributes no residual.)"""
        x = self._xin(points)
        phi = self.potential(x)
        num = 0.0
        den = 0.0
        Bw = self._base_field(x)
        for k in range(self.K):
            g = grad(phi[..., k], x)
            lap = sum(grad(g[..., d], x)[..., d] for d in range(3))
            num = num + lap.pow(2).mean(1)
            Bt = -g if Bw is None else Bw[..., k] - g
            den = den + (Bt[..., 0].pow(2) + Bt[..., 1].pow(2)).mean(1).detach()
        return (num / den.clamp_min(1e-12)).mean()


# ======================================================================================
class FreeMaps(CoilModel):
    """Ablation / no-physics option (``--model free``): a coordinate MLP that outputs the complex
    maps directly, regularised only by a first-order Sobolev smoothness term
    mean|grad C|^2 / mean|C|^2 (cf. NLINV/ENLIVE). No Biot–Savart constraint at all; use it to
    measure what the physics adds. Has no magnetic field, so physics metrics are not reported."""

    def __init__(self, n_coils: int, latent_dim: int = 16, width: int = 64, depth: int = 4):
        super().__init__()
        self.n_coils, self.handedness = n_coils, 1
        self.codes = nn.Parameter(0.5 * torch.randn(n_coils, latent_dim))
        self.net = make_trunk(3 + latent_dim, width, depth, 2)

    def _eval(self, x):
        z = self.codes[:, None, :].expand(-1, x.shape[1], -1)
        out = self.net(torch.cat([x, z], dim=-1))
        return torch.complex(out[..., 0], out[..., 1])

    def sensitivity(self, points, create_graph=True):
        x = points.unsqueeze(0).expand(self.n_coils, -1, -1) if points.dim() == 2 else points
        return self._eval(x)

    def field(self, points, create_graph=True):
        raise NotImplementedError("--model free has no magnetic field (no physics)")

    def physics_loss(self, points):
        x = points.detach().unsqueeze(0).repeat(self.n_coils, 1, 1).requires_grad_(True)
        C = self._eval(x)
        g2 = grad(C.real, x).pow(2).sum(-1) + grad(C.imag, x).pow(2).sum(-1)
        return g2.mean() / (C.abs() ** 2).mean().detach().clamp_min(1e-12)
