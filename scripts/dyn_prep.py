"""Cache whitened mid-slice OCMR cines as .npz (k (N,X,Y,T) complex64, spacing, sampled) for fast experiments.

    python scripts/dyn_prep.py --files /home/claude/data/ocmr/fs_0020_3T.h5 --out-dir ../results/data
Prospective files (us_*): the acquisition mask and corner-level whitening of ocmr_prospective are used."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    from bspinn.ocmr import cine_slices
    os.makedirs(a.out_dir, exist_ok=True)
    for f in a.files:
        name = os.path.basename(f).replace(".h5", "")
        (s, k_raw, info), = list(cine_slices(f, "mid"))
        if a.max_frames:
            k_raw = k_raw[..., :a.max_frames]
        sp = np.array(info["pixel_spacing_mm"], float)
        if name.startswith("us_"):
            from ocmr_prospective import acquisition_mask, whiten_scan
            acq = acquisition_mask(k_raw, info["kx_range"])
            k, w = whiten_scan(k_raw, acq, info["noise"])
            np.savez(os.path.join(a.out_dir, f"{name}.npz"), k=k, spacing=sp, acq=acq)
        else:
            from ocmr_eval import whiten
            k, w = whiten(k_raw, info["noise"], "scan")
            np.savez(os.path.join(a.out_dir, f"{name}.npz"), k=k, spacing=sp)
        print(name, k.shape, sp, w, flush=True)
        del k_raw, k


if __name__ == "__main__":
    main()
