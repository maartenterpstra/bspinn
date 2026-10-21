"""r2-image: automatic weight (k-space CV + select_weight) vs the oracle weight for prior variants, on the
cached joint-protocol maps (scripts/image_lab.py cases). One JSON line per (case, R, maps, variant)."""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from image_lab import Case, solve_grid
from bspinn import prior as P

VARIANTS = {
    "tv": ("tv", {}),
    "tv_phase": ("tv", {"phase": "lowres", "phase_weight": 0.3}),
    "tgv": ("tgv", {}),
    "tgv_phase": ("tgv", {"phase": "lowres", "phase_weight": 0.3}),
    "tvw05": ("tv+wavelet", {"ratio": 0.5}),
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["k19870", "knee15", "val_cc", "val_outlier"])
    ap.add_argument("--R", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--maps", nargs="+", default=["j_v33"])
    ap.add_argument("--variants", nargs="+", default=["tv", "tv_phase", "tgv_phase"])
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--oracle-grid", type=float, nargs=3, default=[-3.6, -1.4, 12])
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--out", default="/home/claude/agents/r2/image/results/cv_eval.jsonl")
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        done = {(r["case"], r["R"], r["maps"], r["variant"], r["crop"]) for r in map(json.loads, open(a.out))}
    og = np.logspace(*a.oracle_grid[:2], int(a.oracle_grid[2]))
    for cn in a.cases:
        for R in a.R:
            for mp in a.maps:
                c = Case(cn, R, mp, a.crop or None)
                for v in a.variants:
                    if (cn, R, mp, v, a.crop) in done:
                        continue
                    prior, kw = VARIANTS[v]
                    t = time.time(); rep = {}
                    x = P.prior_image(c.ksp, c.C, c.m, prior, "auto", report=rep, cv_options={"calib": c.acs}, **kw)
                    q = c.score(x)
                    rec = {"case": cn, "R": R, "maps": mp, "variant": v, "crop": a.crop, "auto": q,
                           "weight": rep["weight"], "rule": rep["rule"], "cv_error": rep["cv_error"], "secs_auto": time.time() - t}
                    if not a.no_oracle:
                        X = solve_grid(c, prior, og, **kw)
                        cur = [c.score(xx) for xx in X]
                        i = int(np.argmin([z["nrmse"] for z in cur]))
                        rec.update(oracle_grid=og.tolist(), grid_nrmse=[z["nrmse"] for z in cur], grid_ssim=[z["ssim"] for z in cur],
                                   oracle={"weight": float(og[i]), **cur[i]}, auto_over_oracle=q["nrmse"] / cur[i]["nrmse"])
                    rec["secs"] = time.time() - t
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"{cn} R{R} {mp} {v}: auto {q['nrmse']:.4f} (w {rep['weight']:.2e} {rep['rule']})" +
                          (f" oracle {rec['oracle']['nrmse']:.4f} (w {rec['oracle']['weight']:.2e}) ratio {rec['auto_over_oracle']:.3f}" if not a.no_oracle else "")
                          + f" {rec['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
