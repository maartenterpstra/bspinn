"""Noise amplification of different coil maps under the SAME reconstruction: g-factor maps by the
pseudo-multiple-replica method, plus the reconstruction error, for each set of maps.

    python scripts/gfactor_compare.py --kspace slice.npz --maps espirit=esp.npy pinn=pinn.npy nlinv=nl.npy \\
        --R 4 --acs 24 --device cuda --out gfactor.json --png gfactor.png
    python scripts/gfactor_compare.py --kspace slice.npz --maps espirit=esp.npy nlinv=nl.npy --run-pinn ...

k-space: (coils, x, y) under --key (fully sampled; the script undersamples it: uniform lines along
--pe-axis, central --acs lines kept). Maps: (coils, x, y) .npy, same channel order (any per-pixel phase /
normalisation: the recon is invariant to it after RSS normalisation, which is applied to every set).
--run-pinn adds the maps of estimate_maps on these data. --mask-maps zeroes every set outside the same
object support (isolates the effect of map values in the background).

For every set: CG-SENSE (identical lambda, iterations, mask) of the undersampled data -> NRMSE vs the
fully sampled SENSE-1 combination with the same maps, and the g-factor
    g = std_replicas(SENSE_R(noise)) / (std_replicas(SENSE_1(noise)) * sqrt(R_eff))
from --replicas noise realisations with the coil noise covariance (periphery estimate or --noise-npy).
Reported: mean / median / 95th percentile / max of g inside the object.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

import accuracy_sweep as acc


def tfft(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def tifft(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def cg_sense(y, C, M, lam, iters):
    """Batched CG-SENSE: y (B,N,H,W) masked k-space, C (N,H,W), M (H,W) -> x (B,H,W)."""
    def A(x):
        return (C.conj() * tifft(M * tfft(C * x[:, None]))).sum(1) + lam * x
    b = (C.conj() * tifft(M * y)).sum(1)
    x = torch.zeros_like(b); r = b.clone(); p = r.clone(); rs = (r.conj() * r).real.sum((1, 2))
    for _ in range(iters):
        Ap = A(p)
        al = rs / (p.conj() * Ap).real.sum((1, 2)).clamp_min(1e-30)
        x = x + al[:, None, None] * p; r = r - al[:, None, None] * Ap
        rn = (r.conj() * r).real.sum((1, 2))
        p = r + (rn / rs.clamp_min(1e-30))[:, None, None] * p; rs = rn
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kspace", required=True)
    ap.add_argument("--key", default="kspace")
    ap.add_argument("--maps", nargs="*", default=[], metavar="NAME=FILE.npy")
    ap.add_argument("--run-pinn", action="store_true", help="also estimate_maps on these data")
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--acs", type=int, default=24)
    ap.add_argument("--pe-axis", type=int, default=1, choices=[0, 1], help="undersampled image axis")
    ap.add_argument("--lam", type=float, default=1e-2, help="Tikhonov weight relative to mean sum|C|^2")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--replicas", type=int, default=64)
    ap.add_argument("--noise-npy", default=None)
    ap.add_argument("--mask-maps", action="store_true")
    ap.add_argument("--mask-thresh", type=float, default=0.05, help="--mask-maps support: RSS > this x 99th pct")
    ap.add_argument("--mask-dilate", type=int, default=4, help="--mask-maps support: dilation in pixels")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="gfactor.json")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    f = np.load(a.kspace)
    k = np.asarray(f[a.key] if hasattr(f, "files") and a.key in f.files else f[f.files[0]] if hasattr(f, "files") else f)
    k = k.astype(np.complex64)
    N, H, W = k.shape
    maps = {}
    for s in a.maps:
        name, path = s.split("=", 1)
        C = np.load(path).astype(np.complex64)
        if C.shape != k.shape:
            raise SystemExit(f"{name}: maps {C.shape} vs k-space {k.shape}")
        maps[name] = C
    if a.run_pinn:
        from bspinn.api import estimate_maps
        maps["pinn (estimate_maps)"] = estimate_maps(k, "2d", calib=a.acs, device=a.device, verbose=False)[0]
    if not maps:
        raise SystemExit("no maps given")
    mag = np.abs(k).sum(0)
    sampled = mag > 1e-6 * mag.max()
    if a.noise_npy:
        nz = np.load(a.noise_npy).reshape(-1, N); psi = (nz.T @ nz.conj()) / nz.shape[0]
    else:
        psi = acc.noise_cov(k, sampled)
    L = np.linalg.cholesky(psi + 1e-9 * np.trace(psi).real / N * np.eye(N))
    rss = np.sqrt((np.abs(acc.ifft(k, (1, 2))) ** 2).sum(0))
    obj = rss > max(0.1 * np.percentile(rss, 99), 4 * np.sqrt(np.trace(psi).real))
    if obj.mean() < 0.02:                    # noise level over-estimated (e.g. signal at the k-space edge)
        obj = rss > 0.1 * np.percentile(rss, 99)
    from scipy.ndimage import binary_dilation, binary_fill_holes
    support = binary_dilation(binary_fill_holes(rss > a.mask_thresh * np.percentile(rss, 99)), iterations=a.mask_dilate)
    res_support = float(support.mean())
    from bspinn.masks import cartesian_mask
    lines = cartesian_mask((H, W) if a.pe_axis == 1 else (W, H), R=a.R, acs=a.acs)
    M = lines if a.pe_axis == 1 else lines.T
    M = M & sampled
    R_eff = sampled.sum() / M.sum()
    dev = torch.device(a.device)
    Mt = torch.from_numpy(M.astype(np.float32)).to(dev)
    Ft = torch.from_numpy(sampled.astype(np.float32)).to(dev)
    kt = torch.from_numpy(k).to(dev)
    g_ = torch.Generator(device=dev).manual_seed(0)
    Lt = torch.from_numpy(L.astype(np.complex64)).to(dev)
    res, gmaps = {"R_eff": float(R_eff), "lam": a.lam, "iters": a.iters, "replicas": a.replicas,
                  "object_fraction": float(obj.mean()), "object_pixels": int(obj.sum()),
                  "mask_maps": a.mask_maps, "support_fraction": res_support, "maps": {}}, {}
    for name, C in maps.items():
        C = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)
        if a.mask_maps:
            C = C * support[None]
        Ct = torch.from_numpy(C.astype(np.complex64)).to(dev)
        den = (Ct.abs() ** 2).sum(0).clamp_min(1e-12)
        lam = a.lam * float(den.mean())
        full = (Ct.conj() * tifft(kt * Ft)).sum(0) / den                         # SENSE-1, fully sampled
        x = cg_sense((kt * Mt)[None], Ct, Mt, lam, a.iters)[0]
        objt = torch.from_numpy(obj).to(dev)
        nrmse = float((x - full)[objt].norm() / full[objt].norm())
        s1, sR = [], []
        for i in range(0, a.replicas, 16):
            b = min(16, a.replicas - i)
            w = (torch.randn((b, N, H, W), generator=g_, device=dev) + 1j * torch.randn((b, N, H, W), generator=g_, device=dev)) / np.sqrt(2)
            n = torch.einsum("ij,bjxy->bixy", Lt, w.to(torch.complex64))
            s1.append((Ct.conj() * tifft(n * Ft)).sum(1) / den)
            sR.append(cg_sense(n * Mt, Ct, Mt, lam, a.iters))
        s1, sR = torch.cat(s1), torch.cat(sR)
        g = (sR.std(0) / (s1.std(0) * np.sqrt(R_eff)).clamp_min(1e-30)).cpu().numpy()
        gin = g[obj]
        res["maps"][name] = {"nrmse_vs_full": nrmse, "g_mean": float(gin.mean()), "g_median": float(np.median(gin)),
                             "g_p95": float(np.percentile(gin, 95)), "g_max": float(gin.max()),
                             "sense1_noise_std_mean": float(s1.std(0).cpu().numpy()[obj].mean())}
        gmaps[name] = g
        r = res["maps"][name]
        print(f"{name:22s} NRMSE {nrmse:.4f}  g mean {r['g_mean']:.3f}  median {r['g_median']:.3f}  "
              f"p95 {r['g_p95']:.3f}  max {r['g_max']:.2f}  (SENSE-1 noise {r['sense1_noise_std_mean']:.3g})", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(gmaps), figsize=(4 * len(gmaps), 4), squeeze=False)
        for axx, (name, g) in zip(axs[0], gmaps.items()):
            im = axx.imshow(np.where(obj, g, np.nan), vmin=1, vmax=max(2.0, np.percentile(np.concatenate([gm[obj] for gm in gmaps.values()]), 99)), cmap="magma")
            axx.set_title(f"{name}\nmean g {res['maps'][name]['g_mean']:.2f}"); axx.axis("off")
        fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8)
        fig.savefig(a.png, dpi=120, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
