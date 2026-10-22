"""Lab: IRGNM refinement (bspinn.irgnm) of cached joint-protocol maps (runs/lab/<anchor>/<case>.npy), scored like
method_lab (ce_acs, lco_out, sense_R4 on the fully sampled data). One JSON line per (case, anchor, spec).

    LAB_DATA=... python scripts/irgnm_lab.py --cases k19870 val_outlier --anchors j_v33 --specs steps=8 holdout=0.1
    spec: comma-separated key=value for irgnm_refine, e.g. steps=6,q=0.5   holdout=0.1,folds=2   none
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")


def main():
    import torch
    torch.set_num_threads(1)
    import method_lab
    from bspinn.irgnm import irgnm_refine
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["k19870", "val_outlier"])
    ap.add_argument("--anchors", nargs="+", default=["j_v33"])
    ap.add_argument("--maps-root", default="/home/claude/bspinn/runs/lab")
    ap.add_argument("--specs", nargs="+", default=["none", "steps=8"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", default=None, help="folder to save refined maps")
    a = ap.parse_args()
    for cn in a.cases:
        case = method_lab.CASES[cn]()
        u = method_lab.undersampled(case, 4)
        for an in a.anchors:
            p = an if an.endswith(".npy") else os.path.join(a.maps_root, an, f"{cn}.npy")
            C0 = np.load(p)
            for spec in a.specs:
                kw = method_lab.parse_set([s for s in spec.split(",") if s and s != "none"])
                t = time.time()
                if spec == "none":
                    C, info = C0, {}
                else:
                    C, _, info = irgnm_refine(u.ksp, C0, u.sampled, **kw)
                secs = time.time() - t
                if a.save:
                    os.makedirs(a.save, exist_ok=True)
                    np.save(os.path.join(a.save, f"{cn}__{os.path.basename(an)}__{spec}.npy"), C.astype(np.complex64))
                r = {"case": cn, "anchor": an, "spec": spec, "secs": round(secs, 1),
                     **{k: v for k, v in info.items() if k in ("steps", "heldout_err", "data_residual")},
                     **method_lab.score(C, case)}
                with open(a.out, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                  if k in ("case", "anchor", "spec", "secs", "steps", "heldout_err", "ce_acs",
                                           "lco_out", "lco_worst", "lco_obj", "sense_R4")}), flush=True)


if __name__ == "__main__":
    main()
