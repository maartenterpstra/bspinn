"""Non-Cartesian (MRI-NUFFT) check: synthetic radial / spiral data with Biot-Savart ground truth, and the radial
head of data/radial16_* (BART NLINV maps as the baseline, held-out spokes as the score).

    python scripts/noncart_eval.py --synth --spokes 64 --size 96 --device cpu --out runs/noncart.jsonl
    python scripts/noncart_eval.py --radial16 data/radial16 --fov 220 --plane axial --device cuda --out runs/noncart.jsonl

Synthetic: map error (1 - |<C_est, C_true>| / ..., per pixel in the object, gauge-invariant) and image NRMSE in the
object for CG-SENSE and TV with the estimated and the true maps. Radial16: relative error on withheld spokes
(maps and images from the other spokes) for the PINN and BART NLINV maps, CG-SENSE and TV.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bspinn.noncart import (estimate_maps_noncart, nufft_operator, sense_image_noncart,  # noqa: E402
                            to_mrinufft)


def map_error(C, G, obj):
    a = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)
    b = G / np.maximum(np.sqrt((np.abs(G) ** 2).sum(0, keepdims=True)), 1e-12)
    return float(np.mean(1 - np.abs((a.conj() * b).sum(0))[obj]))


def nrmse(x, ref, obj):
    x, ref = np.abs(x)[obj], np.abs(ref)[obj]
    a = (x * ref).sum() / max((x * x).sum(), 1e-30)
    return float(np.linalg.norm(a * x - ref) / np.linalg.norm(ref))


def write(out, row):
    print("##", json.dumps(row), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")


def synth(a):
    from mrinufft import initialize_2D_radial, initialize_2D_spiral
    from bspinn.data import make_synthetic
    n = a.size
    d, Cg, obj = make_synthetic(a.coils, (n, n), noise=0.0, seed=a.seed)
    G, x_true = Cg[0].astype(np.complex64), obj[0].astype(np.complex64)
    sup = np.abs(x_true) > 0.05 * np.abs(x_true).max()
    rss = np.sqrt((np.abs(G) ** 2).sum(0))
    Gn, x_ref = (G / rss).astype(np.complex64), (x_true * rss).astype(np.complex64)   # RSS-normalised convention
    for kind in a.kinds:
        if kind == "radial":
            tr = initialize_2D_radial(a.spokes, 2 * n).astype(np.float32)          # (spokes, samples, 2)
        else:
            tr = initialize_2D_spiral(a.spokes // 4, 4 * n, nb_revolutions=8).astype(np.float32)
        t = to_mrinufft(tr, (n, n))
        op = nufft_operator(t, (n, n), a.coils, smaps=G, density=False, backend="finufft")
        y = np.asarray(op.op(x_true))
        rng = np.random.default_rng(a.seed)
        sig = a.noise * np.abs(y).max()
        y = (y + sig * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)) / np.sqrt(2)).astype(np.complex64)
        y = y.reshape(a.coils, tr.shape[0], tr.shape[1])
        t0 = time.time()
        C, info = estimate_maps_noncart(y, tr, (n, n), spacing=d.geom.pixel_spacing, plane=d.geom.plane,
                                        calib=a.calib, quality=a.quality, device=a.device, verbose=False)
        row = {"case": f"synth_{kind}", "shots": int(tr.shape[0]), "samples": int(t.shape[0]), "coils": a.coils,
               "maps_seconds": time.time() - t0, "map_error_pinn": map_error(C, G, sup)}
        for name, maps in (("pinn", C), ("true", Gn)):
            for prior in (None, "tv"):
                rep = {}
                x = sense_image_noncart(y, maps, tr, (n, n), prior=prior, device=a.device, report=rep)
                row[f"nrmse_{name}_{prior or 'cg'}"] = nrmse(x, x_ref, sup)
                if prior:
                    row[f"tv_weight_{name}"] = rep.get("weight")
        write(a.out, row)


def radial16(a):
    from bspinn.data import _read_cfl
    k = _read_cfl(a.radial16 + "_ksp.cfl")
    k = k.reshape(k.shape[1], k.shape[2], k.shape[3])                               # (samples, spokes, coils)
    tr = _read_cfl(a.radial16 + "_traj.cfl").real.reshape(3, k.shape[0], k.shape[1])
    noise = _read_cfl(a.radial16 + "_noise.cfl").reshape(-1, k.shape[2])
    n = a.size or 256
    ks = np.ascontiguousarray(np.transpose(k, (2, 1, 0))).astype(np.complex64)      # (coils, spokes, samples)
    t = np.stack([tr[0], tr[1]], -1).transpose(1, 0, 2)                           # (spokes, samples, 2) cycles/FOV
    S = ks.shape[1]
    rng = np.random.default_rng(a.seed)
    ho = np.zeros(S, bool); ho[rng.choice(S, max(1, S // 10), replace=False)] = True
    ktr, ttr = ks[:, ~ho], t[~ho]
    L = np.linalg.inv(np.linalg.cholesky((noise.conj().T @ noise) / noise.shape[0]))
    th = to_mrinufft(t[ho], (n, n), "cycles")
    yho = np.einsum("ij,jm->im", L, ks[:, ho].reshape(ks.shape[0], -1))
    maps = {}
    t0 = time.time()
    maps["pinn"], _ = estimate_maps_noncart(ktr, ttr, (n, n), traj_units="cycles", fov_mm=a.fov, plane=a.plane,
                                            noise=noise, calib=a.calib, quality=a.quality, device=a.device,
                                            verbose=False)
    tp = time.time() - t0
    nl = _read_cfl(a.radial16 + "_nlinv.cfl")
    if nl.shape[0] == n:
        maps["nlinv (all spokes)"] = np.ascontiguousarray(nl.reshape(n, n, -1).transpose(2, 0, 1)).astype(np.complex64)
    for name, C in maps.items():
        row = {"case": "radial16", "maps": name, "maps_seconds": tp if name == "pinn" else None,
               "heldout_spokes": int(ho.sum())}
        for prior in (None, "tv"):
            rep = {}
            x = sense_image_noncart(ktr, C, ttr, (n, n), traj_units="cycles", noise=noise, prior=prior,
                                    device=a.device, report=rep)
            Cw = np.einsum("ij,j...->i...", L, C).astype(np.complex64)
            op = nufft_operator(th, (n, n), C.shape[0], smaps=Cw, density=False, backend="finufft")
            pred = np.asarray(op.op(x))
            g = np.vdot(pred, yho) / max(np.vdot(pred, pred).real, 1e-30)               # scale-free
            row[f"heldout_{prior or 'cg'}"] = float(np.linalg.norm(g * pred - yho) / np.linalg.norm(yho))
            if a.save:
                np.save(f"{os.path.splitext(a.out)[0]}_{name.split()[0]}_{prior or 'cg'}.npy", x)
        write(a.out, row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--kinds", nargs="+", default=["radial", "spiral"])
    ap.add_argument("--radial16", default=None, help="BART base name, e.g. data/radial16")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--coils", type=int, default=8)
    ap.add_argument("--spokes", type=int, default=48)
    ap.add_argument("--noise", type=float, default=0.002)
    ap.add_argument("--calib", type=int, default=24)
    ap.add_argument("--fov", type=float, default=220.0)
    ap.add_argument("--plane", default="axial")
    ap.add_argument("--quality", default="fast", choices=["fast", "default"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.synth:
        a.size = a.size or 96
        synth(a)
    if a.radial16:
        radial16(a)


if __name__ == "__main__":
    main()
