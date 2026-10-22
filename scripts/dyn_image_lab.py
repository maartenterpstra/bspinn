"""Part 2 lab: cine image priors with fixed (time-averaged) maps, every candidate on a weight grid, scored by
NRMSE against the fully sampled frames (oracle = best grid weight) and by k-space CV (the select_weight rule).

    python3 scripts/dyn_image_lab.py --case fs_0020_3T --R 4 --maps nlinv --methods ttv:0.25 ttv:0 tfft ls llr ref
Methods (bspinn.prior_dyn): see METHODS below. One JSON line per (case, R, maps, method) in --out."""
import argparse
import json
import os
import time

import numpy as np
import torch

from dyn_common import load, nrmse, setup_threads

OUT = "/home/claude/agents/r2/dyn/results"


def get_maps(c, name):
    p = f"{OUT}/maps/{c['name']}_R{c['R']:g}_{name}.npy"
    if os.path.exists(p):
        return np.load(p)
    if name == "nlinv":
        from bspinn import methods as bm
        from bspinn.methods import EvalCase
        case = EvalCase(c["name"], c["kavg"], c["mavg"], (c["cal"], c["cal"]), pixel_spacing=c["sp"], pe_axis=1, joint=True)
        C = bm.nlinv(case)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(p, C.astype(np.complex64))
        return C
    raise SystemExit(f"no cached maps {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--R", type=float, default=4)
    ap.add_argument("--maps", default="nlinv")
    ap.add_argument("--methods", nargs="+", default=["ttv:0.25"])
    ap.add_argument("--grid", type=float, nargs=3, default=[-4, 0, 9])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--cv", action="store_true", help="also the k-space CV curve (2 folds)")
    ap.add_argument("--out", default=f"{OUT}/image_lab.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("--crop-x", type=int, default=0, help="prototype: central readout crop (exact sub-problem)")
    ap.add_argument("--vc", type=int, default=0, help="prototype: compress to this many virtual coils")
    a = ap.parse_args()
    setup_threads()
    from bspinn import prior_dyn as pd
    c = load(a.case, a.R, a.max_frames, crop_x=a.crop_x)
    C = get_maps(c, a.maps)
    if a.vc:
        from dyn_common import compress
        c, C = compress(c, C, a.vc)
    prob = pd.DynProblem(c["ku"], C, c["m"])
    grid = np.logspace(a.grid[0], a.grid[1], int(a.grid[2]))
    for spec in a.methods:
        t0 = time.time()
        solver = pd.make_solver(spec, prob, iters=a.iters)
        errs, xs_best, best = [], None, (np.inf, None)
        dyn = []
        for w in grid:
            X = solver(prob.y, prob.m, float(w))
            x = np.moveaxis(prob.out(X).cpu().numpy() * prob.s, 0, -1)
            e, de = nrmse(x, c)
            errs.append(e); dyn.append(de)
            if e < best[0]:
                best = (e, float(w))
        rec = {"vc": a.vc, "crop_x": a.crop_x, "case": a.case, "R": a.R, "maps": a.maps, "method": spec, "tag": a.tag, "iters": a.iters,
               "grid": grid.tolist(), "nrmse": errs, "dyn_nrmse": dyn, "oracle_nrmse": best[0], "oracle_weight": best[1],
               "oracle_dyn": dyn[int(np.argmin(errs))]}
        if a.cv:
            cv = pd.cv_curve(prob, solver, grid)
            from bspinn.prior import select_weight
            w, rule = select_weight(grid, cv, 1.0)
            X = solver(prob.y, prob.m, w)
            e, de = nrmse(np.moveaxis(prob.out(X).cpu().numpy() * prob.s, 0, -1), c)
            rec.update(cv_error=cv, cv_weight=w, cv_rule=rule, cv_nrmse=e, cv_dyn=de)
        rec["seconds"] = time.time() - t0
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{a.case} R={a.R:g} {a.maps} {spec:24s} oracle {best[0]:.4f} @ {best[1]:.1e}"
              + (f"  cv {rec['cv_nrmse']:.4f} @ {rec['cv_weight']:.1e} ({rec['cv_rule']})" if a.cv else "")
              + f"  [{rec['seconds']:.0f}s]  curve " + " ".join(f"{e:.4f}" for e in errs), flush=True)


if __name__ == "__main__":
    main()
