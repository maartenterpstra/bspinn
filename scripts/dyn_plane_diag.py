"""Part 1 diagnosis: pinn_avg (2-D on the time average) vs pinn_dyn0 (layout 'dynamic', rank 0), with the
B0 plane chosen by each path and with the 2-D plane forced into the dynamic one.

    flock heavy.lock python3 scripts/dyn_plane_diag.py --case fs_0020_3T --R 1 --what avg dyn0 dyn0_forced
"""
import argparse
import contextlib
import io
import json
import os
import time

import numpy as np

from dyn_common import load, lco, setup_threads, subspace_dist

OUT = "/home/claude/agents/r2/dyn/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--R", type=float, default=1)
    ap.add_argument("--quality", default="fast")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--what", nargs="+", default=["avg", "dyn0", "dyn0_forced"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", nargs="*", default=[], help="estimate_maps key=value")
    ap.add_argument("--dyn-set", nargs="*", default=[], help="dynamic_options key=value")
    a = ap.parse_args()
    setup_threads()
    from method_lab import parse_set
    from bspinn.api import estimate_maps
    c = load(a.case, a.R, a.max_frames)
    common = dict(calib=c["cal"], spacing=c["sp"], device="cpu", quality=a.quality, verbose=True, **parse_set(a.set))
    os.makedirs(f"{OUT}/maps", exist_ok=True)
    res = {}
    plane2d = None
    for w in a.what:
        cp = f"{OUT}/maps/{a.case}_R{a.R:g}_{w}{a.tag}.npy"
        t0 = time.time()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if w == "avg":
                C, info = estimate_maps(c["kavg"] * c["mavg"][None], "2d", mask=c["mavg"], readout_axis=0, **common)
                plane2d = info["plane"]
                extra = {"plane": info["plane"], "plane_scores": info.get("plane_scores"),
                         "plane_heldout": info.get("plane_heldout"), "fold": (info.get("fold") or {}).get("corrected"),
                         "handedness": info.get("handedness")}
            else:
                kw = dict(common)
                if w == "dyn0_forced":
                    kw["plane"] = plane2d or json.load(open(f"{OUT}/diag_{a.case}_R{a.R:g}{a.tag}.json"))["avg"]["plane"]
                rank = 0 if w.startswith("dyn0") else int(w[3])
                C, info = estimate_maps(c["ku"], "dynamic", mask=c["m"], rank=rank,
                                        dynamic_options=parse_set(a.dyn_set) or None, **kw)
                if rank == 0:
                    C = C[..., 0]
                extra = {"plane": info["plane"], "plane_scores": info.get("plane_scores"),
                         "plane_heldout": info.get("plane_heldout"), "handedness": info.get("handedness"),
                         "fold": (info.get("fold") or {}).get("corrected"),
                         "validation": info.get("dynamic_validation"), "correction_rel": info.get("correction_rel")}
        log = buf.getvalue()
        open(f"{OUT}/logs_{a.case}_R{a.R:g}_{w}{a.tag}.txt", "w").write(log)
        np.save(cp, C.astype(np.complex64))
        extra = json.loads(json.dumps(extra, default=lambda o: o if not isinstance(o, (np.floating, np.integer)) else o.item()))
        extra["seconds"] = time.time() - t0
        extra["lco"] = lco(C, c) if a.R == 1 else None
        res[w] = extra
        print(w, json.dumps(extra), flush=True)
    ks = list(res)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            A = np.load(f"{OUT}/maps/{a.case}_R{a.R:g}_{ks[i]}{a.tag}.npy"); B = np.load(f"{OUT}/maps/{a.case}_R{a.R:g}_{ks[j]}{a.tag}.npy")
            if A.ndim != B.ndim:
                continue
            res[f"dist_{ks[i]}_{ks[j]}"] = subspace_dist(A, B, c["obj"])
            print(f"subspace distance {ks[i]} vs {ks[j]}: {res[f'dist_{ks[i]}_{ks[j]}']:.4f}", flush=True)
    p = f"{OUT}/diag_{a.case}_R{a.R:g}{a.tag}.json"
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(res)
    json.dump(old, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
