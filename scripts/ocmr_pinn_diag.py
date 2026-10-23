"""Why the time-averaged PINN is unstable on some cines (fs_0016_3T, fs_0021_3T: NRMSE varies 0.043-0.068 with
the seed, NLINV 0.023): several seeds of estimate_maps on the time average of the GRO-undersampled cine, with the
decisions the 2-D path takes (plane, handedness, fold-over correction, held-out scores) and a map-quality score
against the fully sampled data. The log of every fit (verbose) goes to the console.

    BART=bart/bart python bspinn/scripts/ocmr_pinn_diag.py --files data/ocmr/fs_0016_3T.h5 data/ocmr/fs_0021_3T.h5 \
        data/ocmr/fs_0020_3T.h5 --R 4 --seeds 0 1 2 --device cuda:0 --out runs/r3_diag.jsonl 2>&1 | tee runs/r3_diag.log

Score 'resid': combine->expand residual of the fully sampled time-averaged multi-coil image inside the object,
||I - C (C^H I)|| / ||I|| per pixel (gauge-invariant; maps that span the coil images give ~ the noise level),
and 'lco': leave-one-coil-out prediction of the same images (bspinn.refree), both also for NLINV.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocmr_eval import central_lines, ifft2c, whiten  # noqa: E402
from bspinn import evaluation as ev  # noqa: E402
from bspinn import methods as bm  # noqa: E402
from bspinn.api import estimate_maps  # noqa: E402
from bspinn.masks import gro_mask  # noqa: E402
from bspinn.methods import EvalCase  # noqa: E402
from bspinn.ocmr import cine_slices  # noqa: E402


def scores(C, I, obj):
    Cn = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)
    r = I - Cn * (Cn.conj() * I).sum(0, keepdims=True)
    resid = float(np.sqrt((np.abs(r[:, obj]) ** 2).sum() / (np.abs(I[:, obj]) ** 2).sum()))
    # leave-one-coil-out: coil n from the others through the maps (least squares image of the rest)
    num = den = 0.0
    for n in range(C.shape[0]):
        keep = np.arange(C.shape[0]) != n
        Ck, Ik = Cn[keep], I[keep]
        x = (Ck.conj() * Ik).sum(0) / np.maximum((np.abs(Ck) ** 2).sum(0), 1e-12)
        num += float((np.abs(I[n] - Cn[n] * x)[obj] ** 2).sum()); den += float((np.abs(I[n])[obj] ** 2).sum())
    return {"resid": resid, "lco": float(np.sqrt(num / den))}


def pick(info):
    out = {}
    for k, v in info.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict) and k in ("plane_select", "fold", "plane", "handedness", "plane_scores"):
            out[k] = json.loads(json.dumps(v, default=lambda o: str(o)[:200]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--R", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--calib", type=int, default=24)
    ap.add_argument("--plane", default="header", help="'header' (B0 from the slice orientation), 'auto' or a named plane")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for f in a.files:
        name = os.path.basename(f).replace(".h5", "")
        for s, k_raw, info in cine_slices(f, "mid"):
            k, _ = whiten(k_raw, info["noise"], "scan")
            sampled = np.abs(k).sum((0, 3)) > 0
            X, Y, T = k.shape[1:]
            sp = tuple(float(v) for v in info["pixel_spacing_mm"])
            m = gro_mask((X, Y), T, a.R, pe_axis=1, extent=sampled, s=2.2, alpha=3)
            ku = (k * m[None]).astype(np.complex64); cnt = m.sum(-1); mavg = cnt > 0
            kavg = (ku.sum(-1) / np.maximum(cnt, 1)[None]).astype(np.complex64)
            cal = central_lines(mavg, a.calib)
            I = ifft2c(k.mean(-1), (1, 2))                                   # fully sampled time average
            obj = ev.object_mask(EvalCase(name, k.mean(-1), sampled, (cal, cal), sigma=1.0 / np.sqrt(T), pe_axis=1))
            case = EvalCase(f"{name}_R{a.R:g}", kavg, mavg, (cal, cal), pixel_spacing=sp, pe_axis=1, joint=True)
            row = {"file": name, "R": a.R, "maps": "nlinv", **scores(bm.nlinv(case), I, obj)}
            print("##", json.dumps(row), flush=True)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            for seed in a.seeds:
                for ro in (0,):
                    t0 = time.time()
                    print(f"\n######## {name} R={a.R:g} seed {seed}", flush=True)
                    C, inf = estimate_maps(kavg * mavg[None], "2d", mask=mavg, readout_axis=ro, calib=cal, spacing=sp,
                                           device=a.device, seed=seed, verbose=True,
                                           plane=(info.get("b0_dir") or "auto") if a.plane == "header" else a.plane)
                    row = {"file": name, "R": a.R, "maps": f"pinn_{a.plane}_seed{seed}", "seconds": time.time() - t0,
                           **scores(C, I, obj), "info": pick(inf)}
                    print("##", json.dumps({k_: v for k_, v in row.items() if k_ != "info"}), flush=True)
                    with open(a.out, "a") as fh:
                        fh.write(json.dumps(row, default=str) + "\n")
            break


if __name__ == "__main__":
    main()
