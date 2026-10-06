"""Multiple sensitivity sets ("non-resolvable components", cf. ENLIVE / soft-SENSE), with
every set still a Biot–Savart field.

ENLIVE/ESPIRiT add free map sets m_i * c_j^(i) whenever the single-SENSE model is violated.
For the physical causes of such violations the extra set is *the same coil field evaluated at
displaced coordinates*:

  * FOV folding (object larger than the FOV):  C_j(r + n * FOV_pe), n = +-1  ('fold' sets, fixed)
  * rigid motion between calibration samples, or a chemically shifted species (fat):
                                               C_j(r + t_s), t_s learnable  ('shift' sets)

which is a far stronger constraint than free maps. For anything else, 'free' sets are
independent harmonic (Biot–Savart) fields, the direct ENLIVE analogue.

Data model per pixel:  I(r) = sum_s C^(s)(r) m_s(r).  Unneeded sets are suppressed with
ENLIVE's factor-norm penalty (ridge on m_s for s >= 1 plus ||C^(s)||^2 for free sets), the
surrogate of a nuclear-norm (low-rank) penalty on the lifted matrix.
"""
from __future__ import annotations

import torch
from torch import nn

from .models import CoilModel, HarmonicPINN


class MultiSetCoils(nn.Module):
    def __init__(self, primary: CoilModel, fold_shifts=(), n_shift: int = 0, n_free: int = 0,
                 latent: int = 16, width: int = 64, depth: int = 4, seed: int = 0):
        super().__init__()
        self.primary = primary
        dev = primary.device
        self.register_buffer("fold", torch.as_tensor(list(fold_shifts), dtype=torch.float32, device=dev).reshape(-1, 3))
        g = torch.Generator().manual_seed(seed)
        self.shifts = nn.Parameter((0.05 * torch.randn(n_shift, 3, generator=g)).to(dev))
        self.n_free = n_free
        self.free = None
        if n_free:
            N = primary.n_coils
            self.free = HarmonicPINN(N * n_free, "real", latent, width, depth, primary.handedness,
                                     last_init="small").to(dev)
            with torch.no_grad():
                self.free.theta.copy_(primary.gain().angle().detach().repeat(n_free))

    @property
    def n_coils(self):
        return self.primary.n_coils

    @property
    def n_sets(self):
        return 1 + self.fold.shape[0] + self.shifts.shape[0] + self.n_free

    @property
    def handedness(self):
        return self.primary.handedness

    def set_kinds(self):
        return ["primary"] + ["fold"] * self.fold.shape[0] + ["shift"] * self.shifts.shape[0] + ["free"] * self.n_free

    def _wires(self):
        p = self.primary
        return p if hasattr(p, "verts") else getattr(p, "base", None)

    def conductor_mask(self, points, margin=0.1, width=0.03):
        """Smooth 0..1 mask: 0 where ``points`` lie within ``margin`` of a coil conductor. The
        imaged object cannot overlap the coils, so displaced copies are only allowed elsewhere."""
        w = self._wires()
        if w is None:
            return torch.ones(points.shape[0], device=points.device)
        d = torch.cdist(points, w.verts.detach().reshape(-1, 3)).min(dim=1).values
        return torch.sigmoid((d - margin) / width)

    def sensitivity_sets(self, points, create_graph=True):
        """(K, N, P) complex: one map set per component."""
        out = [self.primary.sensitivity(points, create_graph)]
        for t in list(self.fold) + list(self.shifts):
            q = points + t
            out.append(self.primary.sensitivity(q, create_graph) * self.conductor_mask(q)[None])
        if self.free is not None:
            Cf = self.free.sensitivity(points, create_graph)
            out += list(Cf.reshape(self.n_free, self.n_coils, -1))
        return torch.stack(out)

    def sensitivity_sets_batched(self, points, chunk=16384):
        res = []
        for i in range(0, points.shape[0], chunk):
            with torch.enable_grad():
                res.append(self.sensitivity_sets(points[i:i + chunk], create_graph=False).detach())
        return torch.cat(res, dim=-1)

    def physics_loss(self, points):
        # the primary field must be harmonic wherever it is sampled, including displaced copies
        pts = [points]
        for t in list(self.fold) + list(self.shifts):
            q = points + t.detach()
            pts.append(q[self.conductor_mask(q) > 0.5])          # only where an object could be
        loss = self.primary.physics_loss(torch.cat(pts))
        if self.free is not None:
            loss = loss + self.free.physics_loss(points)
        return loss

    # passthroughs so evaluation code can treat this like the primary model
    def field(self, points, create_graph=True):
        return self.primary.field(points, create_graph)

    def sensitivity(self, points, create_graph=True):
        return self.primary.sensitivity(points, create_graph)

    def sensitivity_batched(self, points, chunk=16384):
        return self.primary.sensitivity_batched(points, chunk)
