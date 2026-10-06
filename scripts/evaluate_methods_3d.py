"""Comparative evaluation of coil-sensitivity estimators on 3-D encoded data.

    python scripts/evaluate_methods_3d.py --cases synth3d bart3d --out runs/eval3d

Cases (48^3, pre-whitened, ACS 12^3 for every method):
  synth3d           12 coils in two rings along B0, Biot–Savart ground truth, sigma 0.01
  synth3d_mismatch  same, each coil x its own smooth complex non-Biot–Savart factor (strength 0.3)
  bart3d            BART's 3-D phantom with BART's simulated 8-channel sensitivities (not
                    Biot–Savart fields; B0 direction unknown -> picked from the data)

Metrics as in the 2-D comparison (bspinn.evaluation), with 3-D FFTs: combine->expand residual over
noise floor inside/outside the ACS block, reference image (SNR efficiency, NRMSE/SSIM vs truth),
CG-SENSE with 2-D undersampling (R1 x R2 over the two phase-encoding axes + ACS block), analytic
3-D g-factor, ground-truth map errors, roughness, vortices (all slices), coverage, runtime.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bspinn import evaluation as ev  # noqa: E402
from bspinn.methods import EvalCase, estimate, ifft2c  # noqa: E402
from bspinn.volume import fftnc_np, make_synthetic3d  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS3D = {"iters_auto": 150, "iters_wires": 600, "iters_pinn": 600}


def synth(name, mismatch=0.0, noise=0.01, size=48, acs=12):
    d, C, obj = make_synthetic3d(12, (size,) * 3, noise=noise, mismatch=mismatch)
    return EvalCase(name, d.kspace, np.ones((size,) * 3, bool), (acs,) * 3, sigma=noise, gt_maps=C, gt_object=obj,
                    pe_axes=(1, 2), meta={"noise": noise, "mismatch": mismatch, "orientation": "given"})


def bart3d(name="bart3d", noise=0.01, acs=12, seed=0):
    z = np.load(os.path.join(ROOT, "data/eval_bart3d.npz"))
    sens, img = z["sens"], np.abs(z["img"]).astype(np.float32)
    img = img / img.max()
    obj = img > 1e-3
    sens = sens / np.sqrt((np.abs(sens[:, obj]) ** 2).sum(0).mean())
    rng = np.random.default_rng(seed)
    k = fftnc_np(sens * img[None])
    k = k + noise * (rng.normal(size=k.shape) + 1j * rng.normal(size=k.shape)) / np.sqrt(2)
    return EvalCase(name, k.astype(np.complex64), np.ones(img.shape, bool), (acs,) * 3, sigma=noise,
                    gt_maps=sens.astype(np.complex64), gt_object=img, pe_axes=(1, 2),
                    meta={"noise": noise, "source": "BART phantom -3 -s 8 (simulated coils)", "orientation": "auto"})


CASES = {"synth3d": lambda: synth("synth3d"), "synth3d_mismatch": lambda: synth("synth3d_mismatch", mismatch=0.3),
         "bart3d": lambda: bart3d()}
RS = [(2, 1), (2, 2), (3, 2), (4, 2), (3, 3)]


def _json(o):
    if isinstance(o, dict):
        return {k: _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def run_case(case, methods, out, log=print):
    from scipy.ndimage import binary_dilation
    cdir = os.path.join(out, case.name)
    os.makedirs(cdir, exist_ok=True)
    maps, secs = estimate(case, methods, cache_dir=os.path.join(cdir, "maps"), log=log, threads=1,
                          cfg_overrides=dict(BS3D), orientation=case.meta.get("orientation", "given"))
    obj = ev.object_mask(case)
    sup = binary_dilation(obj, iterations=2)
    ref = np.abs(case.gt_object) * np.sqrt((np.abs(case.gt_maps) ** 2).sum(0))
    res = {"case": {"name": case.name, "n_coils": case.N, "shape": list(case.shape), "acs": list(case.acs), "sigma": case.sigma,
                    "object_voxels": int(obj.sum()), **case.meta}, "runtime_s": secs, "methods": {}}
    D = case.shape[0]
    figs = {"object": obj[D // 2], "reference": ref[D // 2], "reference_cor": ref[:, case.shape[1] // 2]}
    for name, C in maps.items():
        t0 = time.time()
        r = {"maps": ev.map_properties(C, case, obj), "kspace_consistency": ev.kspace_consistency(C, case)}
        r["reference_image"], _ = ev.reference_image(C, case, obj)
        r["sense"], r["sense_support"] = {}, {}
        for R in RS:
            key = f"{R[0]}x{R[1]}"
            r["sense"][key], _ = ev.sense_at_R(C, case, obj, R, ref)
            r["sense_support"][key], xs = ev.sense_at_R(C, case, obj, R, ref, support=sup)
            if R == (2, 2):
                figs[f"{name}/sense_support_R4"] = xs[D // 2]
                figs[f"{name}/sense_support_R4_cor"] = xs[:, case.shape[1] // 2]
        Cn = ev.normalise(C)
        figs[f"{name}/maps"] = Cn[:, D // 2].astype(np.complex64)
        figs[f"{name}/maps_cor"] = Cn[:, :, case.shape[1] // 2].astype(np.complex64)
        res["methods"][name] = r
        log(f"  [{case.name}] metrics {name:15s} ({time.time() - t0:.0f}s)  "
            f"k-out {r['kspace_consistency']['outside_acs']['residual_over_noise_floor']:.4f}  "
            f"sub {r['maps'].get('gt_subspace_err', float('nan')):.2e}  "
            f"R4x2 {r['sense_support']['4x2']['vs_reference']['nrmse']:.4f}/{r['sense']['4x2']['vs_reference']['nrmse']:.4f}")
    with open(os.path.join(cdir, "results.json"), "w") as f:
        json.dump(_json(res), f, indent=1)
    np.savez_compressed(os.path.join(cdir, "figures.npz"), **{k: np.asarray(v) for k, v in figs.items()})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--methods", nargs="*", default=["oracle", "lowres", "walsh", "inati", "espirit", "nlinv", "bs_wires", "bs_pinn"])
    ap.add_argument("--out", default="runs/eval3d")
    a = ap.parse_args()
    for c in a.cases:
        case = CASES[c]()
        print(f"== {c}: {case.N} coils, {case.shape}, ACS {case.acs}", flush=True)
        run_case(case, a.methods, a.out, log=lambda *x: print(*x, flush=True))


if __name__ == "__main__":
    main()
