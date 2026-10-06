"""Comparative evaluation of coil-sensitivity estimators.

    python scripts/evaluate_methods.py --cases synth8 knee15 --out runs/eval
    python scripts/evaluate_methods.py --list

Cases (all single 2-D slices, pre-whitened):
  synth8            8 coils, Biot–Savart ground truth, noise sigma 0.01, 128^2, ACS 24
  synth8_noise{lo,hi}  same, sigma 0.003 / 0.03
  synth8_mismatch   same, every coil x its own smooth complex non-Biot–Savart factor (strength 0.3)
  knee15[_acsK]     real fully sampled 15-channel coronal knee, ACS 24 (or K = 16/32/48)
  brain8            real prospective Poisson-disc brain (8 ch), joint calibration, held-out samples

Every estimator sees the same ACS box (calibration-only protocol) except in ``brain8``, where
NLINV and the Biot–Savart fit may use all acquired samples (their natural mode).
Writes <out>/<case>/results.json, maps/*.npy (cache) and figures.npz (images for the report).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bspinn import evaluation as ev  # noqa: E402
from bspinn.data import make_synthetic  # noqa: E402
from bspinn.methods import ALL_METHODS, EvalCase, estimate, ifft2c  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def synth(name, noise=0.01, mismatch=0.0, n_coils=8, size=128, acs=24):
    d, C, obj = make_synthetic(n_coils, (size, size), noise=noise, seed=0, mismatch=mismatch)
    return EvalCase(name, d.kspace[0], np.ones((size, size), bool), (acs, acs), sigma=noise,
                    gt_maps=C[0], gt_object=obj[0], meta={"noise": noise, "mismatch": mismatch})


def real(name, path, acs, fully=True, joint=False, pe_axis=1, bs_calib=None):
    z = np.load(os.path.join(ROOT, path))
    c = EvalCase(name, z["ksp"], z["mask"], (acs, acs), tuple(float(v) for v in z["pixel_spacing"]), str(z["plane"]),
                 sigma=float(z["sigma"]), fully_sampled=fully, joint=joint, pe_axis=pe_axis,
                 meta={"source": str(z["source"])})
    if bs_calib:
        c.meta["bs_calib"] = bs_calib
    return c


CASES = {
    "synth8": lambda: synth("synth8", 0.01),
    "synth8_noiselo": lambda: synth("synth8_noiselo", 0.003),
    "synth8_noisehi": lambda: synth("synth8_noisehi", 0.03),
    "synth8_mismatch": lambda: synth("synth8_mismatch", 0.01, mismatch=0.3),
    "knee15": lambda: real("knee15", "data/eval_knee15.npz", 24),
    "knee15_acs16": lambda: real("knee15_acs16", "data/eval_knee15.npz", 16),
    "knee15_acs32": lambda: real("knee15_acs32", "data/eval_knee15.npz", 32),
    "knee15_acs48": lambda: real("knee15_acs48", "data/eval_knee15.npz", 48),
    "brain8": lambda: real("brain8", "data/eval_brain8.npz", 20, fully=False, joint=True, bs_calib=48),
}


def _json(o):
    if isinstance(o, dict):
        return {k: _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def run_case(case: EvalCase, methods, out, Rs=(2, 3, 4, 5, 6), threads=1, log=print):
    cdir = os.path.join(out, case.name)
    os.makedirs(cdir, exist_ok=True)
    over = {}
    if case.meta.get("bs_calib"):
        over["calib"] = case.meta["bs_calib"]
    maps, secs = estimate(case, methods, cache_dir=os.path.join(cdir, "maps"), log=log, threads=threads,
                          cfg_overrides=over)
    obj = ev.object_mask(case)
    from scipy.ndimage import binary_dilation
    sup = binary_dilation(obj, iterations=3)          # common SENSE support (same for every method)
    res = {"case": {"name": case.name, "n_coils": case.N, "shape": list(case.shape), "acs": list(case.acs),
                    "fully_sampled": case.fully_sampled, "joint_calibration": case.joint, "sigma": case.sigma,
                    "object_pixels": int(obj.sum()), "sampled_fraction": float(case.sampled.mean()), **case.meta},
           "runtime_s": secs, "methods": {}}
    figs = {"object": obj, "support": sup}
    I = ifft2c(case.ksp * case.sampled[None])
    rss = np.sqrt((np.abs(I) ** 2).sum(0))
    ref = rss if case.gt_maps is None else np.abs(case.gt_object) * np.sqrt((np.abs(case.gt_maps) ** 2).sum(0))
    figs["reference"] = ref
    for name, C in maps.items():
        t0 = time.time()
        r = {"maps": ev.map_properties(C, case, obj)}
        if case.fully_sampled:
            r["kspace_consistency"] = ev.kspace_consistency(C, case)
            r["reference_image"], m = ev.reference_image(C, case, obj)
            figs[f"{name}/combined"] = m
            r["sense"], r["sense_support"] = {}, {}
            for R in Rs:
                r["sense"][str(R)], x = ev.sense_at_R(C, case, obj, R, ref)
                r["sense_support"][str(R)], xs = ev.sense_at_R(C, case, obj, R, ref, support=sup)
                if R == 4:
                    figs[f"{name}/sense_R4"] = x
                    figs[f"{name}/sense_support_R4"] = xs
                    g = ev.gfactor(ev.normalise(C) * sup[None], 4, case.pe_axis, obj)
                    if g:
                        figs[f"{name}/gfactor_R4"] = g[1]
        else:
            r["prospective"], x = ev.heldout_prospective(C, case, obj)
            figs[f"{name}/sense_all"] = x
        Cn = ev.normalise(C)
        figs[f"{name}/maps"] = Cn[: min(case.N, 8)].astype(np.complex64)
        res["methods"][name] = r
        log(f"  [{case.name}] metrics {name:9s} ({time.time() - t0:.0f}s)")
    with open(os.path.join(cdir, "results.json"), "w") as f:
        json.dump(_json(res), f, indent=1)
    np.savez_compressed(os.path.join(cdir, "figures.npz"), **{k: np.asarray(v) for k, v in figs.items()})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=["synth8", "knee15"])
    ap.add_argument("--methods", nargs="*", default=ALL_METHODS)
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(CASES)); return
    for c in a.cases:
        case = CASES[c]()
        print(f"== {c}: {case.N} coils, {case.shape}, ACS {case.acs}", flush=True)
        run_case(case, a.methods, a.out, threads=a.threads, log=lambda *x: print(*x, flush=True))


if __name__ == "__main__":
    main()
