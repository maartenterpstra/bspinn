"""Physics-guided unrolled reconstruction network (ResNet denoiser + CG data consistency, shared weights):
the model of scripts/zs_ssl.py (zero-shot, per scan) and scripts/train_recon.py (whole dataset), usable as an
image prior in api.sense_image / estimate_maps (prior='net:<weights.pt>'). Complex images (H, W); maps
(N, H, W), RSS-normalised; masks (H, W) float."""
import json
import math
import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt


# ----------------------------------------------------------------------------------- operators
def fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


class Sense:
    """A = M F C (maps C: (N,H,W) complex, RSS-normalised; mask M: (H,W) float)."""

    def __init__(self, C, mask):
        self.C, self.m = C, mask

    def A(self, x):
        return self.m * fft2c(self.C * x)

    def AH(self, y):
        return (self.C.conj() * ifft2c(self.m * y)).sum(-3)

    def AHA(self, x):
        return self.AH(self.A(x))


def cg(op, rhs, mu, x0, iters):
    """(A^H A + mu I) x = rhs by CG from x0 (differentiable, fixed number of steps)."""
    x = x0
    r = rhs - (op.AHA(x) + mu * x)
    p = r
    rs = (r.conj() * r).real.sum()
    for _ in range(iters):
        Ap = op.AHA(p) + mu * p
        a = rs / ((p.conj() * Ap).real.sum() + 1e-30)
        x = x + a * p
        r = r - a * Ap
        rn = (r.conj() * r).real.sum()
        p = r + (rn / (rs + 1e-30)) * p
        rs = rn
    return x


# ----------------------------------------------------------------------------------- network
class ResBlock(nn.Module):
    def __init__(self, ch, scale):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.scale = scale

    def forward(self, x):
        return x + self.scale * self.c2(torch.relu(self.c1(x)))


class ResNet(nn.Module):
    """ZS-SSL / SSDU denoiser: conv in, B residual blocks (scaled by 0.1), conv, skip, conv out (+ input)."""

    def __init__(self, ch=64, blocks=15, scale=0.1):
        super().__init__()
        self.head = nn.Conv2d(2, ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(ch, scale) for _ in range(blocks)])
        self.mid = nn.Conv2d(ch, ch, 3, padding=1)
        self.tail = nn.Conv2d(ch, 2, 3, padding=1)

    def forward(self, x):
        h = self.head(x)
        return x + self.tail(self.mid(self.body(h)) + h)


class Unrolled(nn.Module):
    def __init__(self, unrolls=10, cg_iters=10, ch=64, blocks=15, mu0=0.05, amp=False, ckpt=True):
        super().__init__()
        self.den = ResNet(ch, blocks)
        self.mu_raw = nn.Parameter(torch.tensor(math.log(math.expm1(mu0))))
        self.T, self.cg_iters, self.amp, self.ckpt = unrolls, cg_iters, amp, ckpt

    @property
    def mu(self):
        return nn.functional.softplus(self.mu_raw)

    def denoise(self, x):
        z = torch.stack([x.real, x.imag], 0)[None]
        if self.amp and z.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = self.den(z)
            z = z.float()
        else:
            z = self.den(z)
        return torch.complex(z[0, 0], z[0, 1])

    def forward(self, op, y):
        rhs = op.AH(y)
        mu = self.mu
        x = cg(op, rhs, mu, torch.zeros_like(rhs), self.cg_iters)          # regularised SENSE start

        def step(xr, xi):
            x_ = torch.complex(xr, xi)
            z = self.denoise(x_)
            out = cg(op, rhs + mu * z, mu, z, self.cg_iters)
            return out.real, out.imag

        for _ in range(self.T):
            if self.ckpt and torch.is_grad_enabled():
                xr, xi = _ckpt(step, x.real.contiguous(), x.imag.contiguous(), use_reentrant=False)
            else:
                xr, xi = step(x.real, x.imag)
            x = torch.complex(xr, xi)
        return x


def kspace_loss(op_full, x, y, lam_mask):
    """Normalised l1-l2 (ZS-SSL / SSDU) between predicted and acquired k-space at Lambda."""
    pred = lam_mask * fft2c(op_full.C * x)
    t = lam_mask * y
    d = pred - t
    return torch.linalg.vector_norm(d) / torch.linalg.vector_norm(t).clamp_min(1e-12) + \
        d.abs().sum() / t.abs().sum().clamp_min(1e-12)


NET_KEYS = ("unrolls", "cg_iters", "ch", "blocks", "mu0")


def load_network(path, device="cpu", amp=False, **overrides):
    """Weights from train_recon.py (best.pt / last.pt) or zs_ssl.py; the architecture is read from the
    run's split.json / config.json next to the weights (else the defaults: 10 unrolls, 15 x 64 ResNet)."""
    cfg = {}
    for name in ("config.json", "split.json"):
        p = os.path.join(os.path.dirname(os.path.abspath(path)), name)
        if os.path.exists(p):
            d = json.load(open(p))
            d = d.get("args", d)
            cfg = {k: d[k] for k in NET_KEYS if k in d}
            break
    cfg.update({k: v for k, v in overrides.items() if k in NET_KEYS})
    net = Unrolled(cfg.get("unrolls", 10), cfg.get("cg_iters", 10), cfg.get("ch", 64), cfg.get("blocks", 15),
                   cfg.get("mu0", 0.05), amp=amp, ckpt=False).to(device)
    st = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(st["net"] if isinstance(st, dict) and "net" in st else st)
    net.eval()
    return net


def reconstruct(net, ksp, C, mask, device="cpu"):
    """Image from acquired k-space (N, H, W) with maps C (RSS-normalised inside) and mask (H, W)."""
    import numpy as np
    C = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)
    Ct = torch.from_numpy(C.astype(np.complex64)).to(device)
    m = torch.from_numpy(np.asarray(mask, np.float32)).to(device)
    y = torch.from_numpy((ksp * mask[None]).astype(np.complex64)).to(device)
    op = Sense(Ct, m)
    with torch.no_grad():
        x0 = cg(op, op.AH(y), torch.tensor(0.05, device=device), torch.zeros_like(op.AH(y)), 10)
        s = torch.quantile(x0.abs().flatten(), 0.99).clamp_min(1e-12)
        x = net(op, y / s) * s
    return x.cpu().numpy()
