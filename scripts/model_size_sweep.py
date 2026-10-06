"""Influence of model size on the Biot–Savart estimators.

    python scripts/model_size_sweep.py --case synth8_mismatch --group pinn --out runs/size
Groups: 'wires' (loops per coil x vertices per loop, wire-only model) and 'pinn'
(network width x depth on the default wire init). Writes one JSON per (case, config)."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bspinn import evaluation as ev  # noqa: E402
from bspinn.methods import _bs  # noqa: E402
from evaluate_methods import CASES  # noqa: E402

GROUPS = {
    "wires": [("wires", {"loops_per_coil": L, "verts_per_loop": M}) for L in (1, 2, 3) for M in (6, 12, 24)],
    "pinn": [("pinn", {"width": w, "depth": 4}) for w in (16, 32, 64, 128)] + [("pinn", {"width": 64, "depth": d}) for d in (2, 6)],
    # not size, but the other knob of the correction: how hard the harmonic (Laplace) penalty is
    "lambda": [("pinn", {"lambda_phys": l}) for l in (1.0, 0.01, 0.001)],
    "fieldmode": [("pinn", {"field_mode": "complex"})],
    # no wire model: one harmonic network for the whole field, same total iterations (1000 + 1500)
    "pure": [("pinn", {"init": "none", "iters_pinn": 2500}), ("pinn", {"init": "none", "iters_pinn": 2500, "width": 128})],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--group", required=True, choices=list(GROUPS))
    ap.add_argument("--out", default="runs/size")
    ap.add_argument("--only", nargs="*", default=None, help="subset of config tags")
    a = ap.parse_args()
    case = CASES[a.case]()
    obj = ev.object_mask(case)
    from scipy.ndimage import binary_dilation
    sup = binary_dilation(obj, iterations=3)
    ref = np.abs(case.gt_object) * np.sqrt((np.abs(case.gt_maps) ** 2).sum(0)) if case.gt_maps is not None else \
        np.sqrt((np.abs(ev.ifft2c(case.ksp * case.sampled[None])) ** 2).sum(0))
    os.makedirs(a.out, exist_ok=True)
    configs = GROUPS[a.group]
    if a.case.startswith("knee") and a.group == "wires":     # real data: a cheaper cross through the grid
        configs = [c for c in configs if c[1]["verts_per_loop"] == 12 or c[1]["loops_per_coil"] == 1]
    for model, over in configs:
        tag = f"{model}_" + "_".join(f"{k}{v}" for k, v in over.items())
        if a.only and tag not in a.only:
            continue
        path = os.path.join(a.out, f"{a.case}__{tag}.json")
        if os.path.exists(path):
            continue
        t0 = time.time()
        out = _bs(case, model, cfg_overrides=over)
        C = out[model]
        r = {"case": a.case, "model": model, "config": over, "n_params": out["_n_params"], "seconds": time.time() - t0,
             "maps": ev.map_properties(C, case, obj),
             "kspace_outside_acs": ev.kspace_consistency(C, case)["outside_acs"]["residual_over_noise_floor"],
             "kspace_inside_acs": ev.kspace_consistency(C, case)["acs"]["residual_over_noise_floor"]}
        s, _ = ev.sense_at_R(C, case, obj, 4, ref)
        ss, _ = ev.sense_at_R(C, case, obj, 4, ref, support=sup)
        r["sense4_delivered"] = s["vs_reference"]["nrmse"]; r["sense4_support"] = ss["vs_reference"]["nrmse"]
        json.dump(r, open(path, "w"), indent=1)
        print(f"{a.case} {tag}: params {r['n_params']}, {r['seconds']:.0f}s, k-out {r['kspace_outside_acs']:.4f}, "
              f"sub {r['maps'].get('gt_subspace_err', float('nan')):.2e}, R4 {r['sense4_delivered']:.4f}/{r['sense4_support']:.4f}", flush=True)


if __name__ == "__main__":
    main()
