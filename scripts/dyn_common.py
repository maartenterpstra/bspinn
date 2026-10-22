"""Shared helpers for the r2-dyn experiments: cached cines (scripts/dyn_prep.py), GRO masks as in
scripts/ocmr_eval.py, map metrics (leave-coil-out on fully sampled frames) and image metrics."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

DATA = os.environ.get("DYN_DATA", "/home/claude/agents/r2/dyn/results/data")


def setup_threads():
    import torch
    torch.set_num_threads(1)


def compress(c, C, nv):
    """Prototype speed-up: SVD coil compression of data (k, ku, kavg) and maps C to nv virtual coils (same matrix)."""
    if not nv or nv >= c["N"]:
        return c, C
    X = c["kavg"].reshape(c["N"], -1)
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    A = U[:, :nv].conj().T.astype(np.complex64)
    c = dict(c)
    for key in ("k", "ku", "kavg"):
        c[key] = np.einsum("vn,n...->v...", A, c[key]).astype(np.complex64)
    c["N"] = nv
    return c, np.einsum("vn,n...->v...", A, C).astype(np.complex64)


def load(name, R=1, max_frames=0, calib=24, crop_x=0):
    """Fully sampled cine: dict k (N,X,Y,T) whitened, m (X,Y,T) GRO mask, ku, kavg, mavg, cal, sp, sampled, ref, obj."""
    from ocmr_eval import central_lines, ifft2c
    from bspinn import evaluation as ev
    from bspinn.masks import gro_mask
    from bspinn.methods import EvalCase
    d = np.load(os.path.join(DATA, f"{name}.npz"))
    k = d["k"]
    if max_frames:
        k = k[..., :max_frames]
    if crop_x and crop_x < k.shape[1]:
        # readout (x) is fully sampled: cropping x in the image domain keeps the per-line sampling exact
        X0 = k.shape[1]
        img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k, axes=1), axis=1, norm="ortho"), axes=1)
        img = img[:, X0 // 2 - crop_x // 2: X0 // 2 - crop_x // 2 + crop_x]
        k = (np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=1), axis=1, norm="ortho"), axes=1)
             * (np.abs(k).sum((0, 1, 3)) > 0)[None, None, :, None]).astype(np.complex64)
        name = f"{name}_cx{crop_x}"
    N, X, Y, T = k.shape
    sampled = np.abs(k).sum((0, 3)) > 0
    sp = tuple(float(v) for v in d["spacing"])
    m = gro_mask((X, Y), T, R, pe_axis=1, extent=sampled, s=2.2, alpha=3.0)
    ku = (k * m[None]).astype(np.complex64)
    cnt = m.sum(-1)
    mavg = cnt > 0
    kavg = (ku.sum(-1) / np.maximum(cnt, 1)[None]).astype(np.complex64)
    cal = central_lines(mavg, calib)
    ref = np.sqrt((np.abs(ifft2c(k, (1, 2))) ** 2).sum(0))
    avg_case = EvalCase(name, k.mean(-1), sampled, (calib, calib), sigma=1.0 / np.sqrt(T), pe_axis=1)
    obj = ev.object_mask(avg_case)
    return dict(name=name, k=k, m=m, ku=ku, kavg=kavg, mavg=mavg, cal=cal, sp=sp, sampled=sampled, ref=ref, obj=obj,
                N=N, X=X, Y=Y, T=T, R=R)


def lco(C, c, every=4, calib=24):
    """Leave-coil-out / floor outside the ACS on fully sampled frames (as ocmr_eval R = 1), mean over frames.
    C (N,X,Y) or (N,X,Y,T)."""
    from bspinn import refree
    from bspinn.methods import EvalCase
    Cn = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)
    out = []
    for t in range(0, c["T"], every):
        fc = EvalCase(f"t{t}", c["k"][..., t], c["sampled"], (calib, calib), sigma=1.0, pe_axis=1)
        Ct = Cn[..., t] if Cn.ndim == 4 else Cn
        out.append(refree.leave_coil_out(Ct, fc, c["obj"])["outside_acs"]["residual_over_noise_floor"])
    return float(np.mean(out))


def nrmse(x, c):
    """Frame-mean NRMSE (sense_lambda_eval.metrics) and dyn_nrmse vs the fully sampled RSS, band-limited as ocmr_eval."""
    from ocmr_eval import series_metrics
    from zs_ssl import band_limit
    from bspinn.api import acquired_extent
    ext = acquired_extent(c["sampled"])
    xb = np.stack([band_limit(x[..., t], ext) for t in range(x.shape[-1])], -1)
    q = series_metrics(xb, c["ref"], c["obj"])
    return q["nrmse"], q["dyn_nrmse"]


def subspace_dist(A, B, obj):
    """1 - mean |<a,b>| / (|a||b|) over the object pixels (map agreement, gauge-invariant)."""
    num = np.abs((A.conj() * B).sum(0))
    den = np.sqrt((np.abs(A) ** 2).sum(0) * (np.abs(B) ** 2).sum(0)) + 1e-30
    return float(1 - (num / den)[obj].mean())
