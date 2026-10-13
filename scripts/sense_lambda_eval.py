"""Calibrate the automatic SENSE Tikhonov weight (bspinn.senselam) against a fully sampled reference.

    python scripts/sense_lambda_eval.py --loader dataloader_v2:get_dataloader --base-folder /path \\
        --n 20 --R 4 --acs 24 --device cuda --out sense_lambda.json
    python scripts/sense_lambda_eval.py slice1.npz slice2.npz --key kspace --device cuda

For every slice (fully sampled (coils, x, y); a volume (slices, coils, x, y) gives its middle slice):
  1. retrospective undersampling (variable density by default, R, ACS kept),
  2. maps from the undersampled data with estimate_maps (your settings: --plane, --solve-image, --quality,
     --set key=value ...; or --maps-from nlinv_csm to read maps stored in the .npz), cached in --cache,
  3. the reference: the fully sampled data combined with the SAME maps (SENSE-1, closed form), so that only
     the regularisation differs (also reported: vs root-sum-of-squares),
  4. CG-SENSE images for a log grid of lambda (relative, as estimate_maps' sense_lambda) -> NRMSE / SSIM /
     PSNR inside the object; the oracle lambda (lowest NRMSE vs the reference),
  5. every automatic rule (cv = the default 'auto', map, gcv, discrepancy) and fixed 1e-2 / 3e-2: their
     lambda and image metrics.
Summary: per rule the median NRMSE relative to the oracle, how often it is within 5 %, and the median
log10(lambda_rule / lambda_oracle) -> the --sense-lambda-scale that would centre it (printed as
'suggested scale'). One JSON line per slice in --out (resume-safe: slices already in it are skipped).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def items(a):
    from accuracy_sweep import _to_numpy
    for p in a.files:
        z = np.load(p)
        k = _to_numpy(z[a.key] if hasattr(z, "files") else z)
        yield os.path.basename(p), k, (z if hasattr(z, "files") else None)
    if a.loader:
        import importlib
        sys.path.insert(0, os.getcwd())
        mod, fn = a.loader.split(":")
        f = getattr(importlib.import_module(mod), fn)
        ds = f(a.base_folder) if a.base_folder else f()
        idx = a.indices or sorted(np.random.default_rng(a.pick_seed).choice(len(ds), size=min(a.n, len(ds)), replace=False).tolist())
        print(f"dataset {a.loader}: {len(ds)} items, using {idx}", flush=True)
        for i in idx:
            it = ds[i]
            yield f"item{i}", _to_numpy(it[a.key] if isinstance(it, dict) else it), None


def metrics(x, ref, obj):
    from skimage.metrics import structural_similarity
    a, b = np.abs(x), np.abs(ref)
    s = (a[obj] * b[obj]).sum() / max((a[obj] ** 2).sum(), 1e-30)
    a = a * s
    e = a[obj] - b[obj]
    nrmse = float(np.linalg.norm(e) / np.linalg.norm(b[obj]))
    psnr = float(20 * np.log10(b[obj].max() / max(np.sqrt((e ** 2).mean()), 1e-30)))
    _, sm = structural_similarity(a, b, data_range=float(b[obj].max()), full=True)
    return {"nrmse": nrmse, "ssim": float(sm[obj].mean()), "psnr": psnr}


def object_mask(rss):
    from scipy.ndimage import binary_fill_holes, binary_opening, uniform_filter
    r = uniform_filter(rss, 7)
    return binary_fill_holes(binary_opening(r > 0.1 * np.quantile(r, 0.99), iterations=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--key", default="kspace")
    ap.add_argument("--loader", default=None, metavar="MODULE:FUNCTION")
    ap.add_argument("--base-folder", default=None)
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--pick-seed", type=int, default=0)
    ap.add_argument("--coil-axis", type=int, default=0)
    ap.add_argument("--R", type=float, default=4.0)
    ap.add_argument("--acs", type=int, default=24)
    ap.add_argument("--pattern", default="vd", help="cartesian_mask pattern (vd | uniform | ...)")
    ap.add_argument("--mask-seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plane", default="auto")
    ap.add_argument("--quality", default="default")
    ap.add_argument("--solve-image", action="store_true", help="maps by the joint fit (solve_image=True)")
    ap.add_argument("--set", nargs="*", default=[], help="extra estimate_maps / FitConfig options key=value")
    ap.add_argument("--maps-from", default=None, help="use maps stored in the .npz under this key instead of fitting")
    ap.add_argument("--grid", type=float, nargs=3, default=[-5, 0, 21], metavar=("LOG10_MIN", "LOG10_MAX", "N"))
    ap.add_argument("--sense-iters", type=int, default=60)
    ap.add_argument("--probes", type=int, default=2, help="Hutchinson probes for gcv / discrepancy")
    ap.add_argument("--cache", default="sense_lambda_maps")
    ap.add_argument("--out", default="sense_lambda.json")
    a = ap.parse_args()

    import torch
    from bspinn.api import estimate_maps
    from bspinn.masks import cartesian_mask
    from bspinn.senselam import choose_sense_lambda, grid_solutions
    from method_lab import parse_set
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available: using the CPU (slow)"); a.device = "cpu"
    over = parse_set(a.set)
    os.makedirs(a.cache, exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["name"] for l in open(a.out) if l.strip()}
    grid = np.logspace(a.grid[0], a.grid[1], int(a.grid[2]))
    ax = (1, 2)
    Fi = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)
    for name, k, z in items(a):
        if name in done:
            print(f"{name}: already in {a.out}, skipped"); continue
        if k.ndim == 4:
            k = k[k.shape[0] // 2]
        k = np.moveaxis(k, a.coil_axis, 0).astype(np.complex64)
        sampled = np.abs(k).sum(0) > 0
        m = cartesian_mask(k.shape[1:], R=a.R, acs=a.acs, pattern=a.pattern, seed=a.mask_seed) & sampled
        ku = k * m[None]
        t0 = time.time()
        cp = os.path.join(a.cache, f"{name}.npy")
        if a.maps_from:
            C = np.asarray(z[a.maps_from]).astype(np.complex64)
        elif os.path.exists(cp):
            C = np.load(cp)
        else:
            C, _ = estimate_maps(ku, "2d", mask=m, calib=a.acs, plane=a.plane, device=a.device, quality=a.quality,
                                 verbose=False, solve_image=False, **({"joint_full": True} if a.solve_image else {}), **over)
            np.save(cp, C)
        t_maps = time.time() - t0
        Cn = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)
        I = Fi(k * sampled[None])
        ref = (Cn.conj() * I).sum(0)                                     # SENSE-1, fully sampled, same maps
        rss = np.sqrt((np.abs(I) ** 2).sum(0))
        obj = object_mask(rss)
        t1 = time.time()
        X = grid_solutions(ku, Cn, m, grid, device=a.device, iters=a.sense_iters)
        mg = [metrics(x, ref, obj) for x in X]
        io = int(np.argmin([q["nrmse"] for q in mg]))
        rec = {"name": name, "shape": list(k.shape), "R_eff": float(sampled.sum() / m.sum()), "seconds_maps": t_maps,
               "grid": grid.tolist(), "grid_nrmse": [q["nrmse"] for q in mg], "grid_ssim": [q["ssim"] for q in mg],
               "oracle": {"lam": float(grid[io]), **mg[io]}, "rules": {}}
        lam_auto, info = choose_sense_lambda(ku, Cn, m, method="gcv", device=a.device, iters=a.sense_iters, probes=a.probes)
        lam_cv, info_cv = choose_sense_lambda(ku, Cn, m, method="cv", device=a.device, iters=a.sense_iters, calib=a.acs)
        cand = {"cv": lam_cv, "map": info.get("lam_map"), "gcv": info.get("lam_gcv"), "discrepancy": info.get("lam_discrepancy"),
                "fixed_1e-2": 1e-2, "fixed_3e-2": 3e-2}
        cand = {kk: v for kk, v in cand.items() if v is not None}
        Xc = grid_solutions(ku, Cn, m, list(cand.values()), device=a.device, iters=a.sense_iters)
        for (kk, lam), x in zip(cand.items(), Xc):
            q = metrics(x, ref, obj)
            rec["rules"][kk] = {"lam": float(lam), **q, "nrmse_over_oracle": q["nrmse"] / rec["oracle"]["nrmse"],
                                "log10_lam_over_oracle": float(np.log10(lam / rec["oracle"]["lam"]))}
        rec["vs_rss_oracle"] = metrics(X[io], rss, obj)
        rec["sigma2"] = info.get("sigma2"); rec["signal_power"] = info.get("signal_power")
        rec["seconds_lambda"] = time.time() - t1
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{name}: oracle lam {rec['oracle']['lam']:.2e} (NRMSE {rec['oracle']['nrmse']:.4f}) | " +
              "  ".join(f"{kk} {v['lam']:.1e}:{v['nrmse_over_oracle']:.3f}" for kk, v in rec["rules"].items()), flush=True)
    rows = [json.loads(l) for l in open(a.out) if l.strip()]
    if not rows:
        return
    print(f"\n{len(rows)} slices. NRMSE / oracle NRMSE (median, 90th pct, fraction within 5 %), "
          "median log10(lam / lam_oracle), suggested --sense-lambda-scale:")
    for kk in rows[0]["rules"]:
        r = np.array([x["rules"][kk]["nrmse_over_oracle"] for x in rows if kk in x["rules"]])
        l = np.array([x["rules"][kk]["log10_lam_over_oracle"] for x in rows if kk in x["rules"]])
        sug = f"{10 ** -np.median(l):.3g}" if kk in ("cv", "map", "gcv", "discrepancy") else "-"
        print(f"  {kk:12s} {np.median(r):.3f}  {np.percentile(r, 90):.3f}  {np.mean(r < 1.05):.2f}   {np.median(l):+.2f}   {sug}")
    lo = np.array([x["oracle"]["lam"] for x in rows])
    print(f"  oracle lam: median {np.median(lo):.2e}, range {lo.min():.1e} .. {lo.max():.1e}")


if __name__ == "__main__":
    main()
