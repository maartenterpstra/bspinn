"""Image priors with fixed coil maps on the joint protocol: Tikhonov (CG-SENSE) vs total variation vs wavelets
vs TV + wavelets (bspinn.prior, weights by k-space cross-validation), optionally a trained network
(train_recon.py / zs_ssl.py weights), against the fully sampled RSS image.

    python scripts/prior_eval.py --loader dataloader_v2:get_dataloader --base-folder /path --n 20 --R 4 \\
        --maps pinn_api nlinv --maps-cache runs/eval_joint --device cuda --oracle --out prior_eval.jsonl
    python scripts/prior_eval.py ... --net runs/recon/best.pt          # + the trained network as the prior
    LAB_DATA=runs/labdata python scripts/prior_eval.py --lab-cases k19870 val_cc --maps-dir runs/lab/j_nlinv --device cpu

Per (slice, maps, method): NRMSE / SSIM / PSNR inside the object (one least-squares scale; k-space band-limited
to the acquired extent), the chosen weight, seconds. ``--oracle`` also scans the weight grid against the
reference (the best possible weight, to see how close the cross-validation gets). One JSON line per record
in --out (resume-safe), a summary at the end: medians and wins over Tikhonov with the same maps.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

PRIOR_METHODS = {"tv": "tv", "wavelet": "wavelet", "tvw": "tv+wavelet"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loader", default=None, metavar="MODULE:FUNCTION")
    ap.add_argument("--base-folder", default=None)
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pick-seed", type=int, default=0)
    ap.add_argument("--key", default="kspace")
    ap.add_argument("--coil-axis", type=int, default=0)
    ap.add_argument("--lab-cases", nargs="*", default=[], help="scripts/method_lab.py cases (LAB_DATA)")
    ap.add_argument("--R", type=float, nargs="+", default=[4.0])
    ap.add_argument("--maps", nargs="+", default=["pinn_api"])
    ap.add_argument("--pinn-variant", nargs="+", action="append", default=[], metavar=("NAME", "KEY=VALUE"),
                    help="register maps 'pinn_NAME' = estimate_maps with these options (as evaluate_methods.py), "
                         "e.g. --pinn-variant jtv solve_image=true image_prior=tv")
    ap.add_argument("--maps-cache", default=None, help="evaluate_methods.py --joint results folder (<case>_jR/maps)")
    ap.add_argument("--maps-dir", default=None, help="folder with <case>.npy maps (overrides --maps; e.g. runs/lab/j_nlinv)")
    ap.add_argument("--methods", nargs="+", default=["tikhonov", "tv", "wavelet", "tvw"])
    ap.add_argument("--net", default=None, help="trained network weights (adds method 'net')")
    ap.add_argument("--oracle", action="store_true", help="also scan the weight grid against the reference")
    ap.add_argument("--grid", type=float, nargs=3, default=[-4, -1, 10], metavar=("LOG10_MIN", "LOG10_MAX", "N"))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--cg-iters", type=int, default=5)
    ap.add_argument("--ratio", type=float, default=1.0, help="tvw: wavelet weight / TV weight")
    ap.add_argument("--scale", type=float, default=None, help="multiplies the cross-validated weight (default bspinn.prior.AUTO_SCALE)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pinn-quality", default="default")
    ap.add_argument("--out", default="prior_eval.jsonl")
    a = ap.parse_args()

    import torch
    import evaluate_methods as evm
    from method_lab import CASES as LAB, undersampled
    from sense_lambda_eval import metrics
    from zs_ssl import band_limit
    from bspinn import evaluation as ev
    from bspinn import methods as bm
    from bspinn.api import acquired_extent, sense_image
    from bspinn.prior import AUTO_SCALE, admm, prior_image, _setup
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available: using the CPU (slow)"); a.device = "cpu"
    grid = np.logspace(a.grid[0], a.grid[1], int(a.grid[2]))
    kw = dict(iters=a.iters, cg_iters=a.cg_iters)
    net = None
    if a.net:
        from bspinn.unrolled import load_network, reconstruct
        net = load_network(a.net, a.device)
    done = set()
    if os.path.exists(a.out):
        done = {(r["case"], r["maps"], r["method"]) for r in map(json.loads, (l for l in open(a.out) if l.strip()))}

    def makers():
        for c in a.lab_cases:
            yield LAB[c]
        if a.loader:
            yield from evm.loader_cases(a.loader, a.base_folder, a.indices, a.n, a.pick_seed, a.key, a.coil_axis, 24, None, None)

    from functools import partial
    from method_lab import parse_set
    for spec in a.pinn_variant:
        bm.ESTIMATORS[f"pinn_{spec[0]}"] = partial(bm.pinn_api, pinn_options=parse_set(spec[1:]))
        bm._PASS[f"pinn_{spec[0]}"] = bm._PASS["pinn_api"]
    est_kw = {"threads": 1, "cfg_overrides": {}, "device": a.device, "quality": a.pinn_quality}
    for make in makers():
        case0 = make()
        obj = ev.object_mask(case0)
        if obj.sum() < 0.005 * obj.size:
            continue
        ext = acquired_extent(case0.sampled)
        rss = np.sqrt((np.abs(ev.ifft2c(case0.ksp * case0.sampled[None])) ** 2).sum(0))
        for R in a.R:
            fit = undersampled(case0, R)
            fit.name = f"{case0.name}_j{R:g}"
            for mname in ([os.path.basename(a.maps_dir.rstrip("/"))] if a.maps_dir else a.maps):
                meths = list(a.methods) + (["net"] if net is not None else [])
                if all((fit.name, mname, m) in done for m in meths):
                    continue
                if a.maps_dir:
                    C = np.load(os.path.join(a.maps_dir, f"{case0.name}.npy"))
                else:
                    cdir = os.path.join(a.maps_cache, fit.name, "maps") if a.maps_cache else os.path.join(os.path.dirname(os.path.abspath(a.out)), "prior_maps", fit.name)
                    try:
                        C = bm.estimate(fit, [mname], cache_dir=cdir, log=print, **est_kw)[0][mname]
                    except Exception as e:          # one bad slice must not stop a long run
                        print(f"{fit.name} {mname}: maps failed ({type(e).__name__}: {e}); skipped", flush=True)
                        continue
                Cn = ev.normalise(C)
                recs = []

                def record(method, x, extra=None):
                    q = metrics(band_limit(x, ext), rss, obj)
                    recs.append({"case": fit.name, "R": R, "maps": mname, "method": method, **q, **(extra or {})})
                    print(f"{fit.name} {mname:10s} {method:9s} NRMSE {q['nrmse']:.4f}  SSIM {q['ssim']:.4f}  PSNR {q['psnr']:.2f}"
                          + (f"  weight {extra['weight']:.2e}" if extra and extra.get("weight") is not None else ""), flush=True)

                for meth in meths:
                    if (fit.name, mname, meth) in done:
                        continue
                    t0 = time.time()
                    if meth == "tikhonov":
                        rep = {}
                        x = sense_image(fit.ksp, Cn, "2d", mask=fit.sampled, lam="auto", device=a.device, report=rep)
                        record(meth, x, {"weight": rep.get("value"), "seconds": time.time() - t0})
                    elif meth == "net":
                        record(meth, reconstruct(net, fit.ksp, Cn, fit.sampled, a.device), {"seconds": time.time() - t0})
                    else:
                        pr = PRIOR_METHODS[meth]
                        rep = {}
                        x = prior_image(fit.ksp, Cn, fit.sampled, pr, "auto", device=a.device, report=rep, grid=grid,
                                        ratio=a.ratio, scale=AUTO_SCALE if a.scale is None else a.scale, **kw)
                        extra = {"weight": rep["weight"], "weight_grid": rep["weight_grid"], "cv_error": rep["cv_error"], "rule": rep.get("rule"),
                                 "at_grid_edge": rep["at_grid_edge"], "seconds": time.time() - t0}
                        if a.oracle:          # every grid weight against the reference
                            _, _, Ct, mt, yt, x0, s = _setup(fit.ksp, Cn, fit.sampled, a.device)
                            with torch.no_grad():
                                X = admm(yt, Ct, mt, torch.tensor(grid, dtype=torch.float32, device=a.device), pr,
                                         ratio=a.ratio, x0=x0, **kw).cpu().numpy() * s
                            curve = [metrics(band_limit(xx, ext), rss, obj)["nrmse"] for xx in X]
                            io = int(np.argmin(curve))
                            extra.update(oracle_weight=float(grid[io]), oracle_nrmse=float(curve[io]), grid_nrmse=curve)
                        record(meth, x, extra)
                with open(a.out, "a") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
    rows = [json.loads(l) for l in open(a.out) if l.strip()] if os.path.exists(a.out) else []
    if not rows:
        return
    print("\nmedian NRMSE / SSIM / PSNR, wins over Tikhonov (same maps and R), and NRMSE / oracle-weight NRMSE:")
    for R in dict.fromkeys(r["R"] for r in rows):
        for mname in dict.fromkeys(r["maps"] for r in rows):
            base = {r["case"]: r for r in rows if r["R"] == R and r["maps"] == mname and r["method"] == "tikhonov"}
            for meth in dict.fromkeys(r["method"] for r in rows):
                rs = [r for r in rows if r["R"] == R and r["maps"] == mname and r["method"] == meth]
                if not rs:
                    continue
                win = [r["nrmse"] < base[r["case"]]["nrmse"] for r in rs if r["case"] in base]
                orc = [r["nrmse"] / r["oracle_nrmse"] for r in rs if r.get("oracle_nrmse")]
                print(f"  R={R:<4g} {mname:10s} {meth:9s} n={len(rs):3d}  {np.median([r['nrmse'] for r in rs]):.4f}  "
                      f"{np.median([r['ssim'] for r in rs]):.4f}  {np.median([r['psnr'] for r in rs]):6.2f}"
                      + (f"  wins {np.mean(win):.2f}" if meth != "tikhonov" and win else "")
                      + (f"  vs oracle {np.median(orc):.3f}" if orc else ""))


if __name__ == "__main__":
    main()
