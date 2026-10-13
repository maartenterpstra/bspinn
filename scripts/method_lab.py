"""Method lab: run one variant of estimate_maps on a fixed panel of real and synthetic slices and score it.

    python scripts/method_lab.py --variant base --cases k19870 knee15 synth8 synth8_mismatch synth8_noisehi
    python scripts/method_lab.py --variant cplx --set field_mode=complex
    python scripts/method_lab.py --report runs/lab/results.jsonl

Protocol (as in evaluate_methods.py): pre-whitened data, every fit sees only the central 24x24 ACS box,
the B0 orientation is given (coronal for the knees, axial for the phantoms) so that orientation errors do
not mix with method changes. Quality preset 'fast' unless --quality default.

Scores per case
  real (fully sampled):  leave-coil-out / floor outside the ACS (and worst coil), combine->expand / floor
                         inside and outside the ACS, residual slope vs signal, whiteness, CG-SENSE R4 NRMSE
                         on a common support vs RSS.
  synthetic:             the same, plus the ground-truth subspace error.
Maps are cached in runs/lab/<variant>/<case>.npy; one JSON line per (variant, case) in results.jsonl.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

import evaluate_methods as evm  # noqa: E402
from bspinn import evaluation as ev, refree  # noqa: E402

ROOT = os.path.dirname(HERE)
EXTRA = os.environ.get("LAB_DATA", "")      # folder with item*.npz (fully sampled 'kspace') from your loader


def k19870():
    p = os.path.join(EXTRA, "item19870_full.npz")
    k = np.load(p)["kspace"]
    c = evm.from_kspace("k19870", k, 24)
    c.meta["api_plane"] = "coronal"
    return c


def knee15():
    c = evm.real("knee15", "data/eval_knee15.npz", 24)
    c.meta["api_plane"] = "coronal"
    return c


def selfcons():
    """Self-consistency: the real k19870 anatomy (SENSE-1 image with NLINV maps) times the baseline PINN maps of
    k19870 (exactly representable by the model), same sampling, white noise sigma=1. Can the pipeline
    reach the noise floor when the physics model is exactly right?"""
    from bspinn.methods import EvalCase, fft2c
    real = k19870()
    Cn = ev.normalise(np.load(os.path.join(EXTRA, "k19870_nlinv_csm.npy")))
    img = (Cn.conj() * ev.ifft2c(real.ksp * real.sampled[None])).sum(0) * ev.object_mask(real)
    Ct = ev.normalise(np.load(os.path.join(ROOT, "runs", "lab", "base", "k19870.npy")))
    rng = np.random.default_rng(7)
    k = fft2c(Ct * img[None]) + (rng.normal(size=Ct.shape) + 1j * rng.normal(size=Ct.shape)) / np.sqrt(2)
    c = EvalCase("selfcons", (k * real.sampled[None]).astype(np.complex64), real.sampled, real.acs, sigma=1.0,
                 pe_axis=real.pe_axis, gt_maps=Ct, gt_object=img, meta={"api_plane": "coronal"})
    return c


def _val(name, plane):
    def make():
        c = evm.from_kspace(name, np.load(os.path.join(EXTRA, f"{name}.npz"))["kspace"], 24)
        c.meta["api_plane"] = os.environ.get(f"PLANE_{name}", plane)
        return c
    return make


def fold_e6dc():
    """Undersampled (vd R=4, ACS 24) 15-ch knee whose tissue folds over the phase-encode FOV edge (the stripe
    case): prospective, scored by held-out k-space prediction; fitted with fold='auto' on all its samples
    inside the calibration box... i.e. the ACS, as everywhere else."""
    import dataclasses
    k = np.load(os.path.join(EXTRA, "fold_e6dc.npz"))["kspace"]
    c = evm.from_kspace("fold_e6dc", k, 24)
    c = dataclasses.replace(c, fully_sampled=False)
    c.meta.update(api_plane="coronal", fold="auto")
    return c


CASES = {"k19870": k19870, "selfcons": selfcons, "fold_e6dc": fold_e6dc,
         "val_outlier": _val("val_outlier", "axial"), "val_noise": _val("val_noise", "axial"),
         "val_cc": _val("val_cc", "auto"), "knee15": knee15,
         "synth8": evm.CASES["synth8"], "synth8_mismatch": evm.CASES["synth8_mismatch"],
         "synth8_noisehi": evm.CASES["synth8_noisehi"], "synth8_noiselo": evm.CASES["synth8_noiselo"]}


def parse_set(items):
    out = {}
    for s in items or []:
        k, v = s.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v); break
            except ValueError:
                pass
        else:
            v = {"true": True, "false": False}.get(v.lower(), v)
        out[k] = v
    return out


def fit(case, overrides, quality, device, threads):
    from bspinn.api import estimate_maps
    t = time.time()
    C, info = estimate_maps(case.calib_kspace(), "2d", mask=case.acs_mask(), calib=int(max(case.acs)),
                            spacing=tuple(case.pixel_spacing), plane=case.meta.get("api_plane", case.plane),
                            device=device, seed=0, quality=quality, verbose=False, threads=threads,
                            readout_axis=1 - case.pe_axis, fold=case.meta.get("fold", False), **overrides)
    return np.asarray(C), time.time() - t, info


def score(C, case):
    obj = ev.object_mask(case)
    if not case.fully_sampled:
        r, _ = ev.heldout_prospective(C, case, obj, frac=0.2, folds=2, iters=40)
        return {"heldout": r["heldout_nrmse"], "heldout_floor": r["heldout_noise_floor_nrmse"], "acquired": r["acquired_nrmse"],
                "ce_acs": float("nan"), "ce_out": float("nan"), "lco_out": float("nan"), "lco_worst": float("nan"),
                "lco_obj": float("nan"), "slope": float("nan"), "white": float("nan"), "sense_R4": float("nan")}
    from scipy.ndimage import binary_dilation
    sup = binary_dilation(obj, iterations=3)
    I = ev.ifft2c(case.ksp * case.sampled[None])
    ref = np.sqrt((np.abs(I) ** 2).sum(0)) if case.gt_maps is None else \
        np.abs(case.gt_object) * np.sqrt((np.abs(case.gt_maps) ** 2).sum(0))
    kc = ev.kspace_consistency(C, case)
    lco = refree.leave_coil_out(C, case, obj)
    rs = refree.residual_structure(C, case, obj)
    se, _ = ev.sense_at_R(C, case, obj, 4, ref, support=sup)
    out = {"ce_acs": kc["acs"]["residual_over_noise_floor"], "ce_out": kc["outside_acs"]["residual_over_noise_floor"],
           "lco_out": lco["outside_acs"]["residual_over_noise_floor"], "lco_worst": lco["outside_acs"]["worst_coil_over_floor"],
           "lco_obj": lco["object"]["residual_over_noise_floor"], "slope": rs["log_slope_vs_signal_power"],
           "white": rs["whiteness_ratio"], "sense_R4": se["vs_reference"]["nrmse"]}
    if case.gt_maps is not None:
        out["gt_sub"] = ev.map_properties(C, case, obj)["gt_subspace_err"]
    return out


def report(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    keys = ["lco_out", "lco_worst", "ce_acs", "ce_out", "slope", "white", "sense_R4", "gt_sub", "secs"]
    cases = list(dict.fromkeys(r["case"] for r in rows))
    for c in cases:
        print(f"\n== {c}")
        print(f"{'variant':24s}" + "".join(f"{k:>10s}" for k in keys))
        for r in rows:
            if r["case"] == c:
                print(f"{r['variant'][:24]:24s}" + "".join(f"{r.get(k, float('nan')):10.4g}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--set", nargs="*", default=[], help="estimate_maps / FitConfig overrides key=value")
    ap.add_argument("--cases", nargs="*", default=["k19870", "knee15", "synth8", "synth8_mismatch"])
    ap.add_argument("--quality", default="fast")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "lab"))
    ap.add_argument("--maps", default=None, help="score a ready .npy (whitened channels) instead of fitting")
    ap.add_argument("--estimator", default=None, help="score a bspinn.methods estimator (nlinv, espirit, ...) instead")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    if a.report:
        report(a.report); return
    over = parse_set(a.set)
    os.makedirs(os.path.join(a.out, a.variant), exist_ok=True)
    for cn in a.cases:
        case = CASES[cn]()
        cache = os.path.join(a.out, a.variant, f"{cn}.npy")
        secs, plane = float("nan"), None
        if a.maps:
            C = np.load(a.maps)
        elif a.estimator and not os.path.exists(cache):
            from bspinn.methods import ESTIMATORS
            t0 = time.time(); C = np.asarray(ESTIMATORS[a.estimator](case)); secs = time.time() - t0
            np.save(cache, C.astype(np.complex64))
        elif os.path.exists(cache):
            C = np.load(cache)
        else:
            C, secs, info = fit(case, over, a.quality, a.device, a.threads)
            plane = info.get("plane")
            np.save(cache, C.astype(np.complex64))
        r = {"variant": a.variant, "case": cn, "set": over, "quality": a.quality, "secs": secs, "plane": plane, **score(C, case)}
        with open(os.path.join(a.out, "results.jsonl"), "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)


if __name__ == "__main__":
    main()
