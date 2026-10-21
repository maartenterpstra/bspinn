"""Image-quality lab (r2-image): prior variants on cached joint-protocol maps, scored against the fully
sampled RSS (scripts/prior_eval.py metric: magnitude NRMSE inside the object, band-limited)."""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from sense_lambda_eval import metrics
from zs_ssl import band_limit
from bspinn import prior as P

CACHE = "/home/claude/agents/r2/image/results/cache"
CASES = ["k19870", "knee15", "val_cc", "val_outlier"]


class Case:
    def __init__(self, name, R=4, maps="j_v33", crop=None):
        """crop: number of image rows (readout axis 0, fully sampled -> exactly separable) kept around the
        object centre, for cheap prototypes; None = full slice."""
        z = np.load(os.path.join(CACHE, f"{name}.npz"))
        self.name, self.R, self.maps = name, R, maps
        full, C, rss, obj, ext = z["ksp"], z[maps], z["rss"], z["obj"], z["ext"]
        if crop:
            rows = np.where(obj.any(1))[0]
            c0 = int(np.clip((rows[0] + rows[-1]) // 2 - crop // 2, 0, obj.shape[0] - crop))
            sl = slice(c0, c0 + crop)
            I = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(full, axes=1), axis=1, norm="ortho"), axes=1)[:, sl]
            full = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(I, axes=1), axis=1, norm="ortho"), axes=1).astype(np.complex64)
            C, rss, obj = C[:, sl], rss[sl], obj[sl]
            ext = np.broadcast_to(ext.any(0)[None], (crop, ext.shape[1])).copy()
            self.mfull = z[f"m{R}"]
            m = np.broadcast_to(z[f"m{R}"][0][None], (crop, ext.shape[1])).copy()
        else:
            m = z[f"m{R}"]
        self.m = m; self.full = full; self.ksp = full * m[None]
        self.C, self.rss, self.obj, self.ext, self.acs = C, rss, obj, ext, int(z["acs"])

    def score(self, x):
        return metrics(band_limit(x, self.ext), self.rss, self.obj)

    def nrmse(self, x):
        return self.score(x)["nrmse"]


def oracle_curve(case, solve, grid):
    """solve(ksp, C, mask, weights) -> images (L,H,W) in the units of ksp."""
    X = solve(case.ksp, case.C, case.m, np.asarray(grid))
    return [case.nrmse(x) for x in X], X


def solve_grid(case, prior, grid, device="cpu", **kw):
    """Images (L,H,W) for every weight of ``grid`` (bspinn.prior.admm, _setup scaling) + the setup."""
    _, _, Ct, mt, yt, x0, s = P._setup(case.ksp, case.C, case.m, device)
    kw = P.resolve_options(kw, Ct, mt, x0, case.acs)
    with torch.no_grad():
        X = P.admm(yt, Ct, mt, torch.tensor(np.asarray(grid), dtype=torch.float32), prior, x0=x0, **kw)
    return X.cpu().numpy() * s


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["val_cc"])
    ap.add_argument("--R", type=int, nargs="+", default=[4])
    ap.add_argument("--maps", default="j_v33")
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--prior", default="tv")
    ap.add_argument("--grid", type=float, nargs="+", default=[1e-3, 2e-3, 4e-3])
    ap.add_argument("--kw", default="{}", help="JSON options for bspinn.prior.admm")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="/home/claude/agents/r2/image/results/lab.jsonl")
    a = ap.parse_args()
    kw = json.loads(a.kw)
    for cn in a.cases:
        for R in a.R:
            c = Case(cn, R, a.maps, a.crop or None)
            t = time.time()
            X = solve_grid(c, a.prior, a.grid, **kw)
            sc = [c.score(x) for x in X]
            i = int(np.argmin([q["nrmse"] for q in sc]))
            rec = {"case": cn, "R": R, "maps": a.maps, "crop": a.crop, "prior": a.prior, "kw": kw, "tag": a.tag,
                   "grid": list(a.grid), "nrmse": [q["nrmse"] for q in sc], "ssim": [q["ssim"] for q in sc],
                   "best": {"weight": a.grid[i], **sc[i]}, "secs": time.time() - t}
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{cn} R{R} {a.maps} crop{a.crop} {a.prior} {a.tag} {kw}: " +
                  " ".join(f"{g:.1e}:{q['nrmse']:.4f}" for g, q in zip(a.grid, sc)) + f"  best {sc[i]['nrmse']:.4f} ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
