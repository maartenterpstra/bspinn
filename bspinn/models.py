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

import numpy as np

import torch
import torch.utils.checkpoint
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


class SmoothFactor(nn.Module):
    """Per-coil complex multiplicative factor exp(u_j(r)) on the image plane, u_j = sum_b a_jb phi_b(r)
    with a tensor-product cosine/sine basis of order K (1, cos(pi k x / 2), sin(pi k x / 2), k <= K) in
    the two in-plane coordinates (normalised, |x| <= 1 over the FOV; period twice the FOV, so the factor is
    not forced periodic across the FOV). It absorbs smooth per-coil deviations from the physics field
    (wave/conductivity effects, coupling, transmit-receive phase) while the physics part keeps carrying
    the extrapolation. Starts at exp(0) = 1; ``penalty`` is a Sobolev-type ridge (weight
    (1 + |k|^2 / k0^2)^sobolev; default 1 + |k|^2)."""

    def __init__(self, n_coils, comps, order=2, k0=1.0, sobolev=1.0):
        super().__init__()
        self.comps, self.order = tuple(int(c) for c in comps), int(order)
        self.k0, self.sobolev = float(k0), float(sobolev)
        ks = []
        for kx in range(-order, order + 1):
            for ky in range(-order, order + 1):
                ks.append((kx, ky))
        self.register_buffer("kk", torch.tensor(ks, dtype=torch.float32))   # signed k: <0 sin, >0 cos
        self.a = nn.Parameter(torch.zeros(n_coils, len(ks), 2))

    def grown(self, order, k0=None, sobolev=None):
        """The same factor at a higher order (new coefficients 0: identical values), optionally with a new
        penalty shape: continuation from a low order (physics-dominated PINN stage) to a high one."""
        new = SmoothFactor(self.a.shape[0], self.comps, order, self.k0 if k0 is None else k0,
                           self.sobolev if sobolev is None else sobolev).to(self.a.device)
        n_old, n_new = 2 * self.order + 1, 2 * order + 1
        with torch.no_grad():
            for i, (kx, ky) in enumerate(self.kk.long().tolist()):
                new.a[:, (kx + order) * n_new + (ky + order)] = self.a[:, i]
        assert n_old <= n_new
        return new

    def _phi1(self, x, k):
        k_ = k.abs()
        return torch.where(k > 0, torch.cos(np.pi * k_ * x / 2), torch.where(k < 0, torch.sin(np.pi * k_ * x / 2),
                                                                              torch.ones_like(x)))

    def forward(self, points):
        p = points if points.dim() == 2 else points[0]
        # constant continuation beyond the FOV (|x| > 1): the factor is only informed inside it, and the
        # periodic basis would otherwise mirror it there (fold-over maps are evaluated one FOV away)
        x = p[:, self.comps[0]][:, None].clamp(-1, 1)
        y = p[:, self.comps[1]][:, None].clamp(-1, 1)
        phi = self._phi1(x, self.kk[None, :, 0]) * self._phi1(y, self.kk[None, :, 1])   # (P, B)
        u = torch.complex(phi @ self.a[..., 0].T, phi @ self.a[..., 1].T).T              # (N, P)
        return torch.exp(u)

    def penalty(self):
        w = (1 + (self.kk ** 2).sum(1) / self.k0 ** 2) ** self.sobolev                # Sobolev weight
        w = torch.where((self.kk == 0).all(1), torch.zeros_like(w), w)                 # constant: free (gain)
        return (w[None, :, None] * self.a ** 2).sum() / self.a.shape[0]


class ChannelMixer(nn.Module):
    """C -> (I + E) C with a learnable complex off-diagonal E (N x N): residual inductive coupling and
    preamplifier decoupling make every receive channel a complex mixture of the elements' fields, which a
    single real (quasi-static) field per channel with one global phase cannot represent. Starts at E = 0;
    ``penalty`` = mean |E_ij|^2 over coils."""

    def __init__(self, n_coils):
        super().__init__()
        self.e = nn.Parameter(torch.zeros(n_coils, n_coils, 2))
        self.register_buffer("off", 1.0 - torch.eye(n_coils))

    def forward(self, C):
        E = torch.complex(self.e[..., 0], self.e[..., 1]) * self.off
        return C + E.to(C.dtype) @ C if C.dim() == 2 else C + torch.einsum("ij,j...->i...", E.to(C.dtype), C)

    def penalty(self):
        return (self.e ** 2).sum(-1).mul(self.off).sum() / self.e.shape[0]


class CoilModel(nn.Module):
    n_coils: int
    handedness: int
    # optional SmoothFactor multiplying the sensitivities: registered as the submodule 'resid'
    # (looked up in _modules: a class attribute would shadow the registered module)

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
        C = self._sens_from_field(self.field(points, create_graph=create_graph))
        r = self._modules.get("resid")
        C = C if r is None else C * r(points)
        x = self._modules.get("mixer")
        return C if x is None else x(C)

    def _sens_from_field(self, B):
        if B.shape[-1] == 1:
            Bx, By = B[:, :, 0, 0], B[:, :, 1, 0]
        else:
            Bx = torch.complex(B[:, :, 0, 0], B[:, :, 0, 1])
            By = torch.complex(B[:, :, 1, 0], B[:, :, 1, 1])
        C = transverse_to_sensitivity(Bx, By, self.handedness)
        return C * self.gain()[:, None]

    def field_needs_autograd(self) -> bool:
        """True if evaluating the field requires autograd (spatial derivatives by backprop)."""
        return True

    def sensitivity_batched(self, points: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        """Inference-time evaluation on large grids, on the model's device, returned on the CPU.
        No autograd graph when the field is computed in closed form / by Taylor mode."""
        params = list(self.parameters())
        dev = params[0].device if params else points.device
        chunk = chunk or (262144 if dev.type == "cuda" else 16384)
        out = []
        ctx = torch.enable_grad if self.field_needs_autograd() else torch.no_grad
        for i in range(0, points.shape[0], chunk):
            p = points[i:i + chunk].to(dev)
            with ctx():
                out.append(self.sensitivity(p, create_graph=False).detach().cpu())
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
        self.core = 0.0          # conductor radius (normalised); set from FitConfig.wire_radius

    def field_needs_autograd(self):
        return False

    def gain(self):
        return torch.exp(1j * self.theta)

    def field(self, points, create_graph=True, verts=None):
        """``verts`` optionally overrides the loop geometry (e.g. moving coils), shape
        (..., N, L, M, 3) with arbitrary leading batch dims -> field (..., N, P, 3, 1)."""
        V = self.verts if verts is None else verts
        if points.dim() == 2:      # shared points (P,3)
            B = biot_savart_polyline(V, points, core=self.core)               # (...,N,L,P,3)
        else:                      # per-coil points (N,P,3)
            B = biot_savart_polyline(V, points[:, None, None], core=self.core)  # (...,N,L,P,3)
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

    def regularisers(self, support, margin: float = 0.05):
        """(i) keep conductors (every point along every segment, not just the vertices) at least
        ``margin`` outside the imaged object: the field is only source-free where there are no
        conductors; (ii) keep loop shapes smooth. ``support`` is a KeepOut (signed distance to the
        hole-filled object) or, for backwards compatibility, a point set."""
        from .keepout import KeepOut, keepout_penalty, segment_samples
        if isinstance(support, KeepOut):
            outside = keepout_penalty(support, self.verts, margin)
        else:
            v = segment_samples(self.verts).reshape(-1, 3)
            d = torch.cdist(v, support).min(dim=1).values
            outside = (torch.relu(margin - d) / margin).pow(2).sum() / self.n_coils
        v2 = self.verts
        curv = (torch.roll(v2, 1, -2) - 2 * v2 + torch.roll(v2, -1, -2)).pow(2).sum(-1).mean()
        return outside, curv


def trunk_taylor(net: nn.Sequential, x: torch.Tensor, z: torch.Tensor, laplacian: bool = True):
    """Value, spatial gradient and (optionally) spatial Laplacian of a Linear/Tanh MLP at cat[x, z],
    by forward ("Taylor-mode") propagation in ONE differentiable pass: no autograd.grad, no double
    backward. Value, tangents and Laplacian are kept as separate contiguous tensors (one matmul each per
    layer; stacking them into one matmul was 2x slower on CPU because of the per-layer concat). tanh: t' = 1 - t^2, t'' = -2 t t'; for y = tanh(a):
    grad y = t' grad a, lap y = t' lap a + t'' |grad a|^2.

    x: (..., 3); z: latent codes broadcastable to (..., L), e.g. (N, 1, L) for per-coil codes (their
    first-layer contribution is computed once per code, not per point).
    Returns phi (..., K), dphi/dx (..., 3, K), lap phi (..., K) (None if laplacian=False).
    Under torch.compile on CUDA the stacked variant (trunk_taylor_stacked: one matmul per layer) is used:
    inductor fuses its row slicing and concatenation away. Uncompiled, the separate tensors are faster (the
    strided row slices and copies cost more than the saved matmuls; measured on an RTX PRO 6000)."""
    if x.is_cuda and torch.compiler.is_compiling():
        return trunk_taylor_stacked(net, x, z, laplacian)
    layers = list(net)
    W0, b0 = layers[0].weight, layers[0].bias
    Wx, Wz = W0[:, :3], W0[:, 3:]
    zc = torch.nn.functional.linear(z, Wz, b0)                               # (..., width), per code
    h = torch.nn.functional.linear(x, Wx) + zc
    T = Wx.T.expand(*h.shape[:-1], 3, W0.shape[0])                           # (..., 3, width)
    L = None
    for layer in layers[1:]:
        if isinstance(layer, nn.Tanh):
            t = torch.tanh(h)
            s = 1 - t * t
            if laplacian:
                q = (T * T).sum(-2)
                L = (-2 * t * s) * q if L is None else L * s - 2 * t * s * q
            T = T * s.unsqueeze(-2)
            h = t
        elif isinstance(layer, nn.Linear):
            h = torch.nn.functional.linear(h, layer.weight, layer.bias)
            T = T @ layer.weight.T                                                # contiguous, no concat copies
            if L is not None:
                L = torch.nn.functional.linear(L, layer.weight)
        else:
            raise TypeError("Taylor-mode propagation supports Linear/Tanh trunks only")
    if laplacian and L is None:
        L = torch.zeros_like(h)
    return h, T, L


def trunk_taylor_stacked(net: nn.Sequential, x: torch.Tensor, z: torch.Tensor, laplacian: bool = True):
    """trunk_taylor with value, the 3 tangents and the Laplacian stacked as rows of one tensor
    (..., R, width), R = 5 (4 without Laplacian): one matmul per Linear layer instead of three. On a GPU
    the matmuls of these small problems are latency-bound kernels (~30 us each), the extra stack copies
    are cheap elementwise kernels, so this is ~3x fewer GEMMs per pass. Same values as trunk_taylor."""
    layers = list(net)
    W0, b0 = layers[0].weight, layers[0].bias
    Wx, Wz = W0[:, :3], W0[:, 3:]
    zc = torch.nn.functional.linear(z, Wz, b0)
    h = torch.nn.functional.linear(x, Wx) + zc                               # (..., width)
    T = Wx.T.expand(*h.shape[:-1], 3, W0.shape[0])
    rows = [h.unsqueeze(-2), T]
    S = None
    have_L = False
    for layer in layers[1:]:
        if isinstance(layer, nn.Tanh):
            if S is not None:
                h, T = S[..., 0, :], S[..., 1:4, :]
                L = S[..., 4, :] if have_L else None
            else:
                L = None
            t = torch.tanh(h)
            sd = 1 - t * t
            Tn = T * sd.unsqueeze(-2)
            if laplacian:
                q = (T * T).sum(-2)
                Ln = (-2 * t * sd) * q if L is None else L * sd - 2 * t * sd * q
                S = torch.cat([t.unsqueeze(-2), Tn, Ln.unsqueeze(-2)], -2)
                have_L = True
            else:
                S = torch.cat([t.unsqueeze(-2), Tn], -2)
        elif isinstance(layer, nn.Linear):
            if S is None:
                S = torch.cat(rows, -2)
            R = S.shape[-2]
            bias = torch.cat([layer.bias[None], layer.bias.new_zeros(R - 1, layer.bias.shape[0])], 0)
            S = S @ layer.weight.T + bias                                          # (..., R, out)
        else:
            raise TypeError("Taylor-mode propagation supports Linear/Tanh trunks only")
    h, T = S[..., 0, :], S[..., 1:4, :]
    L = S[..., 4, :] if have_L else (torch.zeros_like(h) if laplacian else None)
    return h, T, L


def trunk_value_and_grad(net: nn.Sequential, x: torch.Tensor, z: torch.Tensor):
    """Value and spatial gradient (see trunk_taylor): phi (..., K), dphi/dx (..., 3, K)."""
    h, T, _ = trunk_taylor(net, x, z, laplacian=False)
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

    def field_needs_autograd(self):
        return not self._taylor_ok()

    def _taylor_ok(self):
        """Taylor-mode applies to the standard potential (a Linear/Tanh trunk); subclasses that override
        ``potential`` (e.g. analytic test potentials) use autograd."""
        return (type(self).potential is HarmonicPINN.potential
                and all(isinstance(l, (nn.Linear, nn.Tanh)) for l in self.net))

    def correction_field(self, points, create_graph=True):
        """-grad Phi_NN only (no wire part); (N,P,3,K)."""
        if self._taylor_ok():
            x = points.unsqueeze(0).expand(self.n_coils, -1, -1) if points.dim() == 2 else points
            _, g, _ = trunk_taylor(self.net, x, self.codes[:, None, :], laplacian=False)
            return -g
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

    forward_mode = True        # spatial derivatives by forward (Taylor-mode) propagation, see trunk_taylor

    def field(self, points, create_graph=True):
        x = points.unsqueeze(0).expand(self.n_coils, -1, -1) if points.dim() == 2 else points
        if self._taylor_ok():
            _, g, _ = trunk_taylor(self.net, x, self.codes[:, None, :], laplacian=False)
            B = -g
        else:                                                                 # generic trunk: autograd
            x = self._xin(points)
            phi = self.potential(x)
            B = torch.stack([-grad(phi[..., k], x, create_graph) for k in range(self.K)], dim=-1)
        Bw = self._base_field(x)
        return B if Bw is None else B + Bw

    def physics_loss(self, points):
        """Relative Laplace residual mean(lap Phi^2) / mean(|B_transverse|^2), per coil.
        (The wire part of the field is harmonic analytically and contributes no residual.)
        Value, gradient and Laplacian come from one forward Taylor-mode pass (no double backward)."""
        if not self._taylor_ok():                                             # generic potential: autograd
            x = self._xin(points)
            phi = self.potential(x)
            Bw = self._base_field(x)
            num = den = 0.0
            for k in range(self.K):
                g = grad(phi[..., k], x)
                lap = sum(grad(g[..., d], x)[..., d] for d in range(3))
                num = num + lap.pow(2).mean(1)
                Bt = -g if Bw is None else Bw[..., k] - g
                den = den + (Bt[..., 0].pow(2) + Bt[..., 1].pow(2)).mean(1).detach()
            return (num / den.clamp_min(1e-12)).mean()
        return self._ratio(*self._lap_terms(points), slice(None))

    def _lap_terms(self, points):
        """Per point: squared Laplacian (N,P) and |B_transverse|^2 (N,P), one Taylor-mode pass."""
        x = points.unsqueeze(0).expand(self.n_coils, -1, -1) if points.dim() == 2 else points
        _, g, lap = trunk_taylor(self.net, x.detach(), self.codes[:, None, :], laplacian=True)   # (N,P,3,K), (N,P,K)
        with torch.no_grad():                                                  # enters only the detached norm
            Bw = self._base_field(x.detach())
        Bt = -g if Bw is None else Bw - g
        return lap.pow(2).sum(-1), (Bt[..., 0, :].pow(2) + Bt[..., 1, :].pow(2)).sum(-1)

    @staticmethod
    def _ratio(num, den, sl):
        return (num[:, sl].mean(1) / den[:, sl].mean(1).detach().clamp_min(1e-12)).mean()

    def sensitivity_and_physics(self, pts, pin, pair=None, w2=0.0):
        """sensitivity(pts) and physics_loss(pin) [+ w2 * physics_loss(pair)] from ONE Taylor-mode pass of
        the trunk over all points (data grid and collocation concatenated): fewer, larger matmuls.
        Same values as the separate calls. None if not applicable (generic potential, per-coil points)."""
        if not self._taylor_ok() or pts.dim() != 2:
            return None
        P0, n1 = pts.shape[0], pin.shape[0]
        allp = torch.cat([pts, pin] + ([pair] if pair is not None else []))
        x = allp.unsqueeze(0).expand(self.n_coils, -1, -1)
        _, g, lap = trunk_taylor(self.net, x, self.codes[:, None, :], laplacian=True)
        B = -g[:, :P0]
        Bw = self._base_field(x[:, :P0])                                        # wires train too: with graph
        C = self._sens_from_field(B if Bw is None else B + Bw)
        if self._modules.get("resid") is not None:
            C = C * self._modules["resid"](pts)
        if self._modules.get("mixer") is not None:
            C = self._modules["mixer"](C)
        gc = g[:, P0:]
        with torch.no_grad():
            Bwc = self._base_field(x[:, P0:])
        Bt = -gc if Bwc is None else Bwc - gc
        num = lap[:, P0:].pow(2).sum(-1)
        den = (Bt[..., 0, :].pow(2) + Bt[..., 1, :].pow(2)).sum(-1)
        phys = self._ratio(num, den, slice(0, n1))
        if pair is not None:
            phys = phys + w2 * self._ratio(num, den, slice(n1, None))
        return C, phys

    def physics_loss_pair(self, p1, p2, w2):
        """physics_loss(p1) + w2 * physics_loss(p2) in one trunk pass (object and air collocation)."""
        if not self._taylor_ok() or p1.dim() != 2:
            return self.physics_loss(p1) + w2 * self.physics_loss(p2)
        n1 = p1.shape[0]
        num, den = self._lap_terms(torch.cat([p1, p2]))
        return self._ratio(num, den, slice(0, n1)) + w2 * self._ratio(num, den, slice(n1, None))


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


# ======================================================================================
class WaveWireLoopCoils(WireLoopCoils):
    """Wire loops radiating in a homogeneous lossy medium (time-harmonic, e^{+i omega t}).

    Each current element I dl contributes dB = (1 + i k R) e^{-ikR} (dl x R) / R^3 (units mu0 I / 4 pi):
    the exact field of a current element in a medium of complex wavenumber k, so the total satisfies
    div B = 0 and lap B + k^2 B = 0 everywhere away from the wires. Computed as the closed-form
    Biot-Savart field plus a Gauss-Legendre quadrature of the (non-singular) difference; k^2 = 0 gives
    exactly the quasi-static model. One complex k^2 for all coils, learnable. The k^2/(2r) term is also
    closed form; only the bounded remainder is integrated numerically (and, in 3-D training, on a coarse grid)."""

    def __init__(self, n_coils, loops_per_coil=1, verts_per_loop=12, handedness=1, k2_init: complex = 0j,
                 learn_k2: bool = True, n_quad: int = 3, chunk: int = 2048):
        super().__init__(n_coils, loops_per_coil, verts_per_loop, handedness)
        self.k2_re = nn.Parameter(torch.tensor(float(complex(k2_init).real)), requires_grad=learn_k2)
        self.k2_im = nn.Parameter(torch.tensor(float(complex(k2_init).imag)), requires_grad=learn_k2)
        t, w = np.polynomial.legendre.leggauss(n_quad)
        self.register_buffer("qt", torch.tensor((t + 1) / 2, dtype=torch.float32))
        self.register_buffer("qw", torch.tensor(w / 2, dtype=torch.float32))
        self.chunk = chunk
        self.coarse = None      # (fine_points, coarse_points, fine_shape, coarse_shape): set by the 3-D problem

    @property
    def k2(self) -> complex:
        return complex(float(self.k2_re.detach()), float(self.k2_im.detach()))

    def _k(self):
        k = torch.sqrt(torch.complex(self.k2_re, self.k2_im))
        return torch.where(k.imag > 0, -k, k)                         # outgoing, decaying

    def field(self, points, create_graph=True, verts=None):
        """(N,P,3,2): real and imaginary parts of the complex field."""
        static = self._static_and_1r(points, verts)                   # Biot-Savart + closed-form k^2/(2r), one pass
        if self.coarse is not None and points is self.coarse[0] and verts is None:
            # the wave part is smooth (non-singular): evaluate it on the coarse grid, upsample trilinearly
            _, pc, fs, cs = self.coarse
            w = self._wave(pc, create_graph, None)                    # (N,Pc,3) complex
            g = torch.view_as_real(w).reshape(w.shape[0], *cs, 6).permute(0, 4, 1, 2, 3)
            g = torch.nn.functional.interpolate(g, size=fs, mode="trilinear", align_corners=True)
            w = torch.view_as_complex(g.permute(0, 2, 3, 4, 1).reshape(w.shape[0], -1, 3, 2).contiguous())
        else:
            w = self._wave(points, create_graph, verts)
        B = static + w
        return torch.stack([B.real, B.imag], -1)

    def _static_and_1r(self, points, verts):
        """Static Biot-Savart field + 0.5 k^2 * (closed-form 1/r integral), sharing the segment geometry."""
        from . import physics
        V = self.verts if verts is None else verts
        pts = points[:, None, None] if points.dim() == 3 else points
        st, vl = physics.segment_static_and_log(V, pts, core=self.core)    # (N,L,P,3) each
        st = (st * self.amp[..., None, None]).sum(-3)
        vl = (vl * self.amp[..., None, None]).sum(-3)
        return st + 0.5 * torch.complex(self.k2_re, self.k2_im) * vl

    def _wave_1r(self, points, verts):
        """Leading term of the wave kernel, k^2/(2r): along a straight segment dl x R = d x (P - A) is
        constant, and int_0^1 dt / |P - A - t d| has a closed form (as Biot-Savart's segment formula)."""
        V = self.verts if verts is None else verts
        A = V.unsqueeze(-2)
        d = (torch.roll(V, -1, dims=-2) - V).unsqueeze(-2)              # (N,L,M,1,3)
        pts = points[:, None, None] if points.dim() == 3 else points
        a = pts - A                                                    # (N,L,M,P,3) = P - A
        b = a - d                                                      # P - B
        nd = d.norm(dim=-1)
        na, nb = a.norm(dim=-1), b.norm(dim=-1)
        ad = (a * d).sum(-1)
        I = torch.log((nd * nb + nd * nd - ad).clamp_min(1e-12) / (nd * na - ad).clamp_min(1e-12)) / nd
        v = (torch.linalg.cross(d.expand_as(a), a) * I[..., None]).sum(-3)   # (N,L,P,3)
        v = (v * self.amp[..., None, None]).sum(-3)                       # (N,P,3)
        k2 = torch.complex(self.k2_re, self.k2_im)
        return 0.5 * k2 * v

    def _wave(self, points, create_graph, verts):
        """Smooth remainder of the wave kernel, ((1 + ikr) e^{-ikr} - 1 - (kr)^2/2) / r^3 (bounded),
        by Gauss-Legendre quadrature along each segment."""
        V = self.verts if verts is None else verts                    # (N,L,M,3)
        d = torch.roll(V, -1, dims=-2) - V
        pos = V[..., None, :] + self.qt[:, None] * d[..., None, :]    # (N,L,M,Q,3)
        dl = d[..., None, :] * self.qw[:, None]
        k = self._k()
        per_coil = points.dim() == 3
        P = points.shape[-2]

        def part(pts, pos, dl, k, amp):
            R = pts - pos[..., None, :]                                            # (N,L,M,Q,p,3)
            r = R.norm(dim=-1)
            kr = k * r
            f = ((1 + 1j * kr) * torch.exp(-1j * kr) - 1 - 0.5 * kr * kr) / r ** 3  # bounded remainder
            cr = torch.linalg.cross(dl[..., None, :].expand_as(R), R)
            dB = (f[..., None] * cr).sum((-4, -3))                                 # (N,L,p,3)
            return (dB * amp[..., None, None]).sum(-3)                             # (N,p,3)

        out = []
        grad_on = torch.is_grad_enabled()
        for i in range(0, P, self.chunk):
            pts = points[:, i:i + self.chunk][:, None, None, None] if per_coil else points[i:i + self.chunk]
            if grad_on and not create_graph:
                out.append(torch.utils.checkpoint.checkpoint(part, pts, pos, dl, k, self.amp, use_reentrant=False,
                                                             preserve_rng_state=False))   # no RNG inside; graph-capturable
            else:   # recompute-in-backward keeps memory flat; double backward (physics) needs the plain graph
                out.append(part(pts, pos, dl, k, self.amp))
        return torch.cat(out, -2)


# ======================================================================================
def tissue_k2(b0_tesla: float, eps_r: float, sigma: float, half_fov_m: float) -> complex:
    """Complex wavenumber squared of homogeneous tissue at the proton Larmor frequency, in the
    normalised coordinates of the fit (lengths in units of the half field of view):
    k^2 = (omega^2 mu0 eps0 eps_r - i omega mu0 sigma) * L^2   (time dependence e^{+i omega t})."""
    mu0, eps0 = 4e-7 * math.pi, 8.8541878128e-12
    w = 2 * math.pi * 42.577478e6 * b0_tesla
    return complex(w * w * mu0 * eps0 * eps_r, -w * mu0 * sigma) * half_fov_m ** 2


def k2_to_tissue(k2: complex, b0_tesla: float, half_fov_m: float):
    """Inverse of ``tissue_k2``: (relative permittivity, conductivity S/m) of the homogeneous medium."""
    mu0, eps0 = 4e-7 * math.pi, 8.8541878128e-12
    w = 2 * math.pi * 42.577478e6 * b0_tesla
    k2 = complex(k2) / half_fov_m ** 2
    return k2.real / (w * w * mu0 * eps0), -k2.imag / (w * mu0)


class HelmholtzPINN(CoilModel):
    """Wave-aware coil field for heterogeneous-free (homogeneous) tissue.

    B = B_wires + B_c. The wire field is the quasi-static (free-space) incident field of the fitted
    loops, carrying the receive phase e^{i theta}; the network outputs the complex vector correction
    B_c (the field scattered by the tissue's displacement and conduction currents) directly, so no
    derivative is needed to evaluate the maps. Physics, at collocation points inside the object:

        lap B_c + k^2 (B_c + B_wires) = 0     (vector Helmholtz for the total field; lap B_wires = 0)
        div B_c = 0

    with one complex k^2 for the object (learned from ``k2_init`` or fixed). Reduces to the complex-
    field Laplace model for k^2 = 0."""

    def __init__(self, n_coils: int, latent_dim: int = 16, width: int = 64, depth: int = 4, handedness: int = 1,
                 base: WireLoopCoils | None = None, k2_init: complex = 0j, learn_k2: bool = True, lambda_div: float = 1.0):
        super().__init__()
        self.n_coils, self.handedness, self.field_mode = n_coils, handedness, "complex"
        self.K = 2
        self.base = base
        self.codes = nn.Parameter(0.5 * torch.randn(n_coils, latent_dim))
        self.net = make_trunk(3 + latent_dim, width, depth, 6)
        nn.init.zeros_(self.net[-1].weight)
        self.wave_base = base is not None and hasattr(base, "k2_re")
        if self.wave_base:                      # one k^2 shared by the wave wires and the correction
            self.k2_re, self.k2_im = base.k2_re, base.k2_im
        else:
            self.k2_re = nn.Parameter(torch.tensor(float(complex(k2_init).real)), requires_grad=learn_k2)
            self.k2_im = nn.Parameter(torch.tensor(float(complex(k2_init).imag)), requires_grad=learn_k2)
        self.lambda_div = lambda_div

    @property
    def k2(self) -> complex:
        return complex(float(self.k2_re.detach()), float(self.k2_im.detach()))

    def gain(self):
        return torch.ones(self.n_coils, dtype=torch.complex64, device=self.codes.device)

    def _base_field(self, x):
        if self.base is None:
            return None
        Bw = self.base.field(x, create_graph=False)                      # (N,P,3,1) real or (N,P,3,2)
        th = self.base.theta
        c, s_ = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
        if Bw.shape[-1] == 1:
            return torch.cat([Bw * c[..., None], Bw * s_[..., None]], -1)
        re, im = Bw[..., 0], Bw[..., 1]
        return torch.stack([re * c - im * s_, re * s_ + im * c], -1)

    def field_needs_autograd(self):
        return False                                                         # direct network output + wires

    def _net(self, x):
        z = self.codes[:, None, :].expand(-1, x.shape[1], -1)
        out = self.net(torch.cat([x, z], dim=-1))                         # (N,P,6)
        return out.reshape(*out.shape[:-1], 3, 2)

    def correction_field(self, points, create_graph=True):
        x = points.unsqueeze(0).expand(self.n_coils, -1, -1) if points.dim() == 2 else points
        return self._net(x)

    def field(self, points, create_graph=True):
        x = points.unsqueeze(0).expand(self.n_coils, -1, -1) if points.dim() == 2 else points
        Bc = self._net(x)
        Bw = self._base_field(x)
        return Bc if Bw is None else Bc + Bw

    def physics_loss(self, points):
        """mean|lap B_c + k^2 B_total|^2 / (mean|B_total|^2 max(|k^2|^2, 1)) + lambda_div mean|div B_c|^2 / mean|B_total|^2.
        Jacobian (for div) and Laplacian of all six outputs from one forward Taylor-mode pass."""
        return self._combine(*self._helm_terms(points), slice(None))

    def _helm_terms(self, points):
        """Per point (N,P): Helmholtz residual, |B_total|^2, |div B_c|^2; one Taylor-mode pass."""
        x = points.detach().unsqueeze(0).expand(self.n_coils, -1, -1)
        out, J, L = trunk_taylor(self.net, x, self.codes[:, None, :], laplacian=True)   # (N,P,6), (N,P,3,6), (N,P,6)
        Bc = out.reshape(*out.shape[:-1], 3, 2)
        lap = L.reshape(*L.shape[:-1], 3, 2)
        Jr = J.reshape(*J.shape[:-1], 3, 2)                                  # [..., d_e, component d, part p]
        div = Jr[..., 0, 0, :] + Jr[..., 1, 1, :] + Jr[..., 2, 2, :]                 # (N,P,2): d_e B_e
        with torch.no_grad():                                                  # the wire field is a constant here
            Bw = self._base_field(x)
        Bt = Bc if Bw is None else Bc + Bw
        # wave wires already solve the Helmholtz equation: only the correction enters the residual;
        # quasi-static wires are harmonic, so their k^2 B_w term remains as a source
        Br = Bc if (Bw is None or self.wave_base) else Bt
        a, b = self.k2_re, self.k2_im
        rr = lap[..., 0] + a * Br[..., 0] - b * Br[..., 1]
        ri = lap[..., 1] + a * Br[..., 1] + b * Br[..., 0]
        return (rr.pow(2) + ri.pow(2)).sum(-1), Bt.pow(2).sum((-1, -2)), div[..., 0] ** 2 + div[..., 1] ** 2

    def _combine(self, res, bt2, div2, sl):
        den = bt2[:, sl].mean(1).detach().clamp_min(1e-12)                    # per coil
        kk = (self.k2_re.detach() ** 2 + self.k2_im.detach() ** 2).clamp_min(1.0)   # |k^2|^2, no host sync
        helm = (res[:, sl].mean(1) / (den * kk)).mean()
        dv = (div2[:, sl].mean(1) / den).mean()
        return helm + self.lambda_div * dv

    def physics_loss_pair(self, p1, p2, w2):
        """physics_loss(p1) + w2 * physics_loss(p2) in one trunk pass."""
        n1 = p1.shape[0]
        t = self._helm_terms(torch.cat([p1, p2]))
        return self._combine(*t, slice(0, n1)) + w2 * self._combine(*t, slice(n1, None))
