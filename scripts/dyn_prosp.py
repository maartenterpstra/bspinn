"""Part 1 on prospective OCMR (us_*): the dynamic layout (2-D base + rank-K corrections) vs its base (= pinn_avg),
scored as scripts/ocmr_prospective.py: 2 acquired lines per frame withheld from maps and images, spatio-temporal TV
image from the rest, held-out error per sample over the noise level (1 = noise only).

    flock heavy.lock python3 scripts/dyn_prosp.py --case us_0001_3T --rank 2
Writes maps to results/maps/<case>_pros_{base,full,val}.npy and one JSON line per map set to results/prosp.jsonl."""
import argparse
import contextlib
import io
import json
import os
import time

import numpy as np

from dyn_common import DATA, setup_threads

OUT = "/home/claude/agents/r2/dyn/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="us_0001_3T")
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--quality", default="fast")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--dyn-set", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--weights", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
    ap.add_argument("--recon-only", action="store_true")
    a = ap.parse_args()
    setup_threads()
    import torch
    from method_lab import parse_set
    from ocmr_eval import central_lines
    from ocmr_prospective import heldout_split, prediction_error
    from bspinn.api import estimate_maps
    from bspinn.prior import _dyn_setup, dynamic_admm
    d = np.load(os.path.join(DATA, f"{a.case}.npz"))
    k, acq, sp = d["k"], d["acq"], tuple(float(v) for v in d["spacing"])
    if a.max_frames:
        k, acq = k[..., :a.max_frames], acq[..., :a.max_frames]
    N, X, Y, T = k.shape
    tr, ho = heldout_split(acq, 2, 0)
    ktr = (k * tr[None]).astype(np.complex64)
    cal = central_lines(tr.sum(-1) > 0, 24)
    tag = f"{a.case}_pros_K{a.rank}{a.tag}"
    paths = {s: f"{OUT}/maps/{tag}_{s}.npy" for s in ("base", "full", "val")}
    if not a.recon_only:
        t0 = time.time()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            C, info = estimate_maps(ktr, "dynamic", mask=tr, rank=a.rank, calib=cal, spacing=sp, quality=a.quality,
                                    device="cpu", verbose=True, dynamic_options=parse_set(a.dyn_set) or None)
        open(f"{OUT}/logs/{tag}_fit.txt", "w").write(buf.getvalue())
        np.save(paths["val"], C)
        np.save(paths["base"], info["dynamic_maps_fn"](0.0)); np.save(paths["full"], info["dynamic_maps_fn"](1.0))
        meta = {"case": a.case, "rank": a.rank, "plane": info["plane"], "validation": info.get("dynamic_validation"),
                "correction_rel": info.get("correction_rel"), "fit_seconds": time.time() - t0, "dyn_set": a.dyn_set}
        print(json.dumps(meta), flush=True)
        with open(f"{OUT}/prosp.jsonl", "a") as f:
            f.write(json.dumps({"meta": meta, "tag": tag}) + "\n")
    for s, p in paths.items():
        C = np.load(p)
        for w in a.weights:
            k_, m_, Cn, Ct, mt, yt, x0, sc = _dyn_setup(ktr, C, tr, "cpu")
            with torch.no_grad():
                x = dynamic_admm(yt, Ct, mt, w, 0.25 * w, 50, x0=x0).numpy() * sc
            x = np.moveaxis(x, 0, -1)
            Cnn = np.moveaxis(Cn, 0, -1) if Cn.ndim == 4 else Cn
            e_ho = prediction_error(k, Cnn, x, ho, 12)
            e_tr = prediction_error(k, Cnn, x, tr, 12)
            rec = {"tag": tag, "maps": s, "weight": w, "heldout": e_ho["all"], "heldout_centre": e_ho["centre"],
                   "heldout_outer": e_ho["outer"], "train": e_tr["all"]}
            print(json.dumps(rec), flush=True)
            with open(f"{OUT}/prosp.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
