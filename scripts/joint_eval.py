"""Compare ways of fitting maps and image (sequential vs true joint) on N fully sampled slices.

    python scripts/joint_eval.py --loader dataloader_v2:get_dataloader --base-folder /path --n 20 \\
        --R 4 --acs 24 --device cuda --plane auto --out joint_eval.json
    python scripts/joint_eval.py slice.npz --variants calib_only joint_cg joint_start_grad --device cuda

Every slice is pre-whitened (noise covariance from the k-space periphery / image background), then
retrospectively undersampled (variable density, R, ACS kept). Each variant runs estimate_maps on the
undersampled data and returns maps + image; scored against the fully sampled data:
  image  NRMSE / SSIM / PSNR of |image| vs the fully sampled root-sum-of-squares, inside the object,
         after one least-squares scale (method-independent reference)
  maps   leave-coil-out prediction outside the ACS / noise floor (bspinn.refree), CG-SENSE-free
Variants (add your own with --set, applied to all):
  calib_only        maps from the calibration data, image by CG-SENSE afterwards (return_image=True)
  joint_cg          solve_image=True: calibration fit, then maps + image jointly (image by CG after every
                    map step; v34 behaviour)
  joint_grad        same, image a parameter of the same optimiser (simultaneous steps)
  joint_start_cg    joint from the first PINN step (joint_start='wires'), image by CG
  joint_start_grad  joint from the first PINN step, simultaneous steps
  *_lbfgs           any of the above with optimizer='lbfgs' (e.g. joint_start_grad_lbfgs)
The image weight is sense_lambda='auto' unless --sense-lambda is given. One JSON line per (slice, variant)
in --out (resume-safe); a summary table at the end (median over slices, and wins vs calib_only).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

VARIANTS = {
    "calib_only": dict(return_image=True),
    "joint_cg": dict(solve_image=True),
    "joint_grad": dict(solve_image=True, joint_image="gradient"),
    "joint_start_cg": dict(solve_image=True, joint_start="wires"),
    "joint_start_grad": dict(solve_image=True, joint_start="wires", joint_image="gradient"),
}


def variant(name):
    if name in VARIANTS:
        return dict(VARIANTS[name])
    if name.endswith("_lbfgs") and name[:-6] in VARIANTS:
        return {**VARIANTS[name[:-6]], "optimizer": "lbfgs"}
    raise SystemExit(f"unknown variant {name}; known: {list(VARIANTS)} (+ _lbfgs)")


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
    ap.add_argument("--pattern", default="vd")
    ap.add_argument("--mask-seed", type=int, default=1)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plane", default="auto")
    ap.add_argument("--quality", default="default")
    ap.add_argument("--sense-lambda", default="auto")
    ap.add_argument("--set", nargs="*", default=[], help="extra estimate_maps / FitConfig options key=value (all variants)")
    ap.add_argument("--out", default="joint_eval.json")
    a = ap.parse_args()

    import dataclasses
    import torch
    import evaluate_methods as evm
    from sense_lambda_eval import items, metrics
    from method_lab import parse_set
    from bspinn import evaluation as ev, refree
    from bspinn.api import estimate_maps
    from bspinn.masks import cartesian_mask
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available: using the CPU (slow)"); a.device = "cpu"
    extra = parse_set(a.set)
    lam = a.sense_lambda if str(a.sense_lambda).startswith("auto") else float(a.sense_lambda)
    done = set()
    if os.path.exists(a.out):
        done = {(r["name"], r["variant"]) for r in map(json.loads, (l for l in open(a.out) if l.strip()))}
    for name, k, _ in items(a):
        if k.ndim == 4:
            k = k[k.shape[0] // 2]
        k = np.moveaxis(k, a.coil_axis, 0)
        case = evm.from_kspace(name, k, a.acs)                       # whitened, sigma = 1, fully sampled
        m = cartesian_mask(case.shape, R=a.R, acs=a.acs, pattern=a.pattern, seed=a.mask_seed) & case.sampled
        ku = (case.ksp * m[None]).astype(np.complex64)
        obj = ev.object_mask(case)
        rss = np.sqrt((np.abs(ev.ifft2c(case.ksp * case.sampled[None])) ** 2).sum(0))
        for v in a.variants:
            if (name, v) in done:
                continue
            kw = {**variant(v), **extra}
            t0 = time.time()
            C, img, info = estimate_maps(ku, "2d", mask=m, calib=a.acs, plane=a.plane, device=a.device,
                                         quality=a.quality, verbose=False, sense_lambda=lam,
                                         readout_axis=1 - case.pe_axis, **kw)
            secs = time.time() - t0
            lco = refree.leave_coil_out(np.asarray(C), case, obj)
            rec = {"name": name, "variant": v, "options": {kk: str(vv) for kk, vv in kw.items()}, "seconds": secs,
                   "plane": info.get("plane"), "lambda": (info.get("sense_lambda") or {}).get("value"),
                   "image": metrics(img, rss, obj),
                   "lco_out": lco["outside_acs"]["residual_over_noise_floor"],
                   "lco_worst": lco["outside_acs"]["worst_coil_over_floor"]}
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{name} {v:22s} NRMSE {rec['image']['nrmse']:.4f} SSIM {rec['image']['ssim']:.4f} "
                  f"LCO {rec['lco_out']:.3f} lam {rec['lambda']} {secs:.0f}s", flush=True)
    rows = [json.loads(l) for l in open(a.out) if l.strip()]
    names = sorted({r["name"] for r in rows})
    by = {(r["name"], r["variant"]): r for r in rows}
    print(f"\n{len(names)} slices: median image NRMSE, SSIM, PSNR, map LCO/floor, seconds; "
          "and the fraction of slices where the variant's NRMSE beats calib_only")
    for v in dict.fromkeys(r["variant"] for r in rows):
        rs = [by[(n, v)] for n in names if (n, v) in by]
        win = [by[(n, v)]["image"]["nrmse"] < by[(n, "calib_only")]["image"]["nrmse"]
               for n in names if (n, v) in by and (n, "calib_only") in by]
        print(f"  {v:22s} {np.median([r['image']['nrmse'] for r in rs]):.4f}  {np.median([r['image']['ssim'] for r in rs]):.4f}  "
              f"{np.median([r['image']['psnr'] for r in rs]):6.2f}  {np.median([r['lco_out'] for r in rs]):.3f}  "
              f"{np.median([r['seconds'] for r in rs]):6.1f}  " + (f"{np.mean(win):.2f}" if win else "-"))


if __name__ == "__main__":
    main()
