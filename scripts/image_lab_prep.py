"""Prepare compact cases for the image-quality lab: whitened k-space, joint-protocol masks (R=4,6),
cached maps (j_v33, j_nlinv, j_espirit), RSS reference, object mask, acquired extent."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import evaluate_methods as evm
evm.ROOT = "/home/claude/bspinn"
import method_lab as ml
from bspinn import evaluation as ev
from bspinn.api import acquired_extent
OUT = sys.argv[1]
for cn in sys.argv[2:]:
    c = ml.CASES[cn]()
    obj = ev.object_mask(c)
    rss = np.sqrt((np.abs(ev.ifft2c(c.ksp * c.sampled[None])) ** 2).sum(0))
    d = dict(ksp=c.ksp.astype(np.complex64), sampled=c.sampled, obj=obj, rss=rss.astype(np.float32),
             ext=acquired_extent(c.sampled), pe_axis=c.pe_axis, acs=int(max(c.acs)))
    for R in (4, 6, 8):
        d[f"m{R}"] = ml.undersampled(c, R).sampled
    for v in ("j_v33", "j_nlinv", "j_espirit"):
        d[v] = ev.normalise(np.load(f"/home/claude/bspinn/runs/lab/{v}/{cn}.npy")).astype(np.complex64)
    np.savez(os.path.join(OUT, f"{cn}.npz"), **d)
    print(cn, c.ksp.shape, "R_eff", [float(c.sampled.sum() / d[f"m{R}"].sum()) for R in (4, 6, 8)], "pe", c.pe_axis, flush=True)
