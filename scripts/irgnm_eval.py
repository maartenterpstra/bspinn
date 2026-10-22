"""IRGNM map refinement (estimate_maps(..., refine='irgnm'), bspinn/irgnm.py) on the joint protocol: PINN vs
PINN + IRGNM vs NLINV, maps and final images, scored on the fully sampled data.

    # your data (GPU): reuses the maps of an evaluate_methods.py --joint run if --maps-cache points at it
    BART=bart/bart python scripts/irgnm_eval.py --loader dataloader_v2:get_dataloader --base-folder /data --n 20 \\
        --maps pinn_api pinn_api+irgnm nlinv --maps-cache runs/eval_joint --device cuda --out runs/irgnm_eval.jsonl
    # lab cases with cached maps (CPU)
    LAB_DATA=runs/labdata python scripts/irgnm_eval.py --lab-cases k19870 val_outlier \\
        --maps v33 v33+irgnm nlinv --maps-dir v33=runs/lab/j_v33 nlinv=runs/lab/j_nlinv --device cpu --out x.jsonl

Map names: any bspinn.methods estimator (pinn_api, nlinv, espirit, pinn_<variant> via --pinn-variant), a name
given a folder of <case>.npy with --maps-dir, or <name>+irgnm[@key=value...] = irgnm_refine of <name>'s maps on
all acquired samples (held-out step choice unless @holdout=0@steps=N). Scores per (slice, maps): leave-coil-out
/ floor outside the ACS (lco_out) and over the object (lco_obj), combine->expand / floor inside the ACS (ce_acs),
CG-SENSE R4 NRMSE (method_lab.score), and the final images at the protocol's R: Tikhonov (sense_image 'auto')
and TV (bspinn.prior, CV weight) NRMSE vs the RSS, with --oracle also the best TV weight of a grid.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loader", default=None)
    ap.add_argument("--base-folder", default=None)
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pick-seed", type=int, default=0)
    ap.add_argument("--key", default="kspace")
    ap.add_argument("--coil-axis", type=int, default=0)
    ap.add_argument("--lab-cases", nargs="*", default=[])
    ap.add_argument("--R", type=float, default=4.0)
    ap.add_argument("--maps", nargs="+", default=["pinn_api", "pinn_api+irgnm", "nlinv"])
    ap.add_argument("--maps-dir", nargs="*", default=[], metavar="NAME=DIR")
    ap.add_argument("--maps-cache", default=None, help="evaluate_methods.py --joint folder (<case>_jR/maps)")
    ap.add_argument("--pinn-variant", nargs="+", action="append", default=[], metavar=("NAME", "KEY=VALUE"))
    ap.add_argument("--pinn-quality", default="default")
    ap.add_argument("--methods", nargs="*", default=["tikhonov", "tv"])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available: using the CPU (slow)"); a.device = "cpu"
    torch.set_num_threads(a.threads)
    from functools import partial
    import evaluate_methods as evm
    import method_lab
    from method_lab import CASES as LAB, parse_set, undersampled
    from sense_lambda_eval import metrics
    from zs_ssl import band_limit
    from bspinn import evaluation as ev
    from bspinn import methods as bm
    from bspinn.api import acquired_extent, sense_image
    from bspinn.irgnm import irgnm_refine
    from bspinn.prior import AUTO_SCALE, prior_image
    for spec in a.pinn_variant:
        bm.ESTIMATORS[f"pinn_{spec[0]}"] = partial(bm.pinn_api, pinn_options=parse_set(spec[1:]))
        bm._PASS[f"pinn_{spec[0]}"] = bm._PASS["pinn_api"]
    dirs = dict(s.split("=", 1) for s in a.maps_dir)
    est_kw = {"threads": a.threads, "cfg_overrides": {}, "device": a.device, "quality": a.pinn_quality}
    done = set()
    if os.path.exists(a.out):
        done = {(r["case"], r["maps"]) for r in map(json.loads, (l for l in open(a.out) if l.strip()))}
    grid = np.logspace(-4, -1, 10)

    def makers():
        for c in a.lab_cases:
            yield LAB[c]
        if a.loader:
            yield from evm.loader_cases(a.loader, a.base_folder, a.indices, a.n, a.pick_seed, a.key, a.coil_axis, 24,
                                        None, None)

    for make in makers():
        case0 = make()
        obj = ev.object_mask(case0)
        if obj.sum() < 0.005 * obj.size:
            continue
        fit = undersampled(case0, a.R)
        fit.name = f"{case0.name}_j{a.R:g}"
        cdir = (os.path.join(a.maps_cache, fit.name, "maps") if a.maps_cache else
                os.path.join(os.path.dirname(os.path.abspath(a.out)), "irgnm_maps", fit.name))
        os.makedirs(cdir, exist_ok=True)
        ext = acquired_extent(fit.sampled)
        rss = np.sqrt((np.abs(ev.ifft2c(case0.ksp * case0.sampled[None])) ** 2).sum(0))
        memo = {}

        def get(name):
            if name in memo:
                return memo[name]
            t0, info = time.time(), {}
            if "+irgnm" in name:
                base, rest = name.split("+irgnm", 1)
                cp = os.path.join(cdir, name.replace("@", "_") + ".npy")
                if os.path.exists(cp):
                    C = np.load(cp)
                else:
                    C0 = get(base)[0]
                    t0 = time.time()
                    ro = {"holdout": 0.1, **parse_set([o for o in rest.split("@") if o])}
                    C, _, info = irgnm_refine(fit.ksp, C0, fit.sampled, device=a.device, **ro)
                    np.save(cp, C.astype(np.complex64))
            elif name in dirs:
                C = np.load(os.path.join(dirs[name], f"{case0.name}.npy"))
            else:
                C = bm.estimate(fit, [name], cache_dir=cdir, log=print, **est_kw)[0][name]
            memo[name] = (np.asarray(C), time.time() - t0, info)
            return memo[name]

        for mname in a.maps:
            if (fit.name, mname) in done:
                continue
            try:
                C, secs, info = get(mname)
            except Exception as e:                      # one bad slice must not stop a long run
                print(f"{fit.name} {mname}: maps failed ({type(e).__name__}: {e}); skipped", flush=True)
                continue
            Cn = ev.normalise(C)
            r = {"case": fit.name, "maps": mname, "R": a.R, "seconds": round(secs, 1),
                 **{k: v for k, v in info.items() if k in ("steps", "heldout_err")},
                 **method_lab.score(Cn, case0)}
            if "tikhonov" in a.methods:
                x = sense_image(fit.ksp, Cn, "2d", mask=fit.sampled, lam="auto", device=a.device)
                r["tik_nrmse"] = metrics(band_limit(x, ext), rss, obj)["nrmse"]
            if "tv" in a.methods:
                rep = {}
                x = prior_image(fit.ksp, Cn, fit.sampled, "tv", "auto", device=a.device, report=rep, grid=grid,
                                scale=AUTO_SCALE)
                r["tv_nrmse"] = metrics(band_limit(x, ext), rss, obj)["nrmse"]; r["tv_weight"] = rep.get("weight")
                if a.oracle:
                    r["tv_oracle"] = min(metrics(band_limit(prior_image(fit.ksp, Cn, fit.sampled, "tv", float(wt),
                                                                       device=a.device), ext), rss, obj)["nrmse"]
                                         for wt in grid)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()
                           if k in ("case", "maps", "steps", "ce_acs", "lco_out", "lco_obj", "sense_R4", "tik_nrmse",
                                    "tv_nrmse", "tv_oracle", "seconds")), flush=True)


if __name__ == "__main__":
    main()
