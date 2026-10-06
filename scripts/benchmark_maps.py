"""Score sensitivity-map sets on held-out k-space (no ground truth needed).

Example:
  python scripts/benchmark_maps.py data/brain8_ksp --calib 48 \
      --maps biot_savart=runs/brain8/maps_normalized.npy espirit=data/brain8_espirit nlinv=data/brain8_nlinv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from bspinn.benchmark import HeldoutBenchmark  # noqa: E402
from bspinn.data import load, load_maps, load_noncart  # noqa: E402
from bspinn.fit import FitConfig, Problem  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--maps", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--calib", type=int, default=48)
    ap.add_argument("--traj"); ap.add_argument("--fov", type=float, default=256.0)
    ap.add_argument("--drop-first", type=int, default=0); ap.add_argument("--noise")
    ap.add_argument("--recon-size", type=int, default=96)
    ap.add_argument("--lam", type=float, nargs="+", default=[1e-2])
    ap.add_argument("--frac", type=float, default=0.2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(2)
    data = load_noncart(a.path, a.traj, a.fov, noise_path=a.noise, drop_first=a.drop_first) if a.traj \
        else load(a.path, noise_path=a.noise)
    data.prewhiten()
    prob = Problem(data, FitConfig(calib=a.calib))
    shape = (a.recon_size,) * 2 if prob.noncart else tuple(data.geom.shape)
    res = {}
    for lam in a.lam:
        bench = HeldoutBenchmark(prob, shape, frac=a.frac, seed=a.seed, lam=lam)
        for item in a.maps:
            name, path = item.split("=", 1)
            M = bench.prepare(load_maps(path))
            for masked in (False, True):
                key = f"lam={lam:g} {'common-support' if masked else 'unmasked':14s} {name}"
                res[key] = bench.score(M, masked)[0]
                print(f"{res[key]:.4f}  {key}", flush=True)
    print(f"held-out samples per coil: {bench.n_heldout_per_coil}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
