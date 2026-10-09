"""How many iterations does estimate_maps need on YOUR data? Accuracy vs iteration budget, no ground truth.

    python scripts/accuracy_sweep.py data/*.npz --device cuda --out sweep.json
    python scripts/accuracy_sweep.py knee.npz --key kspace --coil-axis 0 --calib 24 --seeds 0 1 2
    # from your own dataset object: dataloader_v2.get_dataloader(base_folder)[i]['kspace']
    python scripts/accuracy_sweep.py --loader dataloader_v2:get_dataloader --base-folder /path/to/data \
        --random 8 --device cuda --out sweep.json          # or --indices 0 17 523 ...

Each input: a .npz (array under --key, default 'kspace') or .npy with 2-D multi-coil k-space (coils, x, y)
(--layout 3d: (coils, x, y, z)). Fully sampled data are best: the metrics retrospectively undersample.

For every file, every iteration budget (--budgets, default: half-fast, fast, mid, default, long) and every seed:

  residual_over_floor  combine -> expand k-space residual outside the calibration block, divided by
                       the same residual for pure noise (noise level from the k-space periphery, or
                       --noise-npy). 1.0 = consistent with the data down to the noise; lower = better.
                       (In the benchmark this metric ranked methods like the ground-truth map error.)
  sense_nrmse_R{R}     CG-SENSE of retrospectively undersampled data (uniform, R, calibration block
                       kept) vs the fully sampled image combined with the same maps; lower = better.
  change_vs_longest    1 - |<c, c_ref>|^2 / (|c|^2 |c_ref|^2) per pixel, mean over the object: how far the
                       maps still are from those of the longest budget with the same seed (convergence).
  seed_spread          the same quantity between seeds of one budget: differences between budgets
                       smaller than this are noise, not accuracy.
  seconds, phase times, data_loss (calibration residual of the fit).

Send back the JSON (--out); it contains numbers only, no images or k-space.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

BUDGETS = {                         # (probe iterations, wire iterations, PINN iterations)
    "half-fast": (100, 250, 400),
    "fast": (100, 500, 800),
    "mid": (200, 1000, 1600),
    "default": (300, 1500, 2500),
    "long": (300, 3000, 5000),
}


def fft(x, axes):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)


def ifft(x, axes):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)


def calib_mask(shape, calib):
    m = np.zeros(shape, bool)
    sl = tuple(slice(n // 2 - min(c, n) // 2, n // 2 - min(c, n) // 2 + min(c, n)) for n, c in zip(shape, calib))
    m[sl] = True
    return m


def noise_sigma(k, sampled):
    """Per-coil noise std from the k-space periphery (outer shell beyond 90 % of the radius), robust
    (median absolute value of real/imag parts / 0.6745)."""
    grids = np.meshgrid(*[np.linspace(-1, 1, n) for n in k.shape[1:]], indexing="ij")
    r = np.sqrt(sum(g ** 2 for g in grids)) / np.sqrt(len(grids))
    shell = (r > 0.9 * r[sampled].max()) & sampled
    if shell.sum() < 50:
        shell = (r > 0.8 * r[sampled].max()) & sampled
    v = k[:, shell]
    return np.median(np.abs(np.concatenate([v.real, v.imag], 1)), axis=1) / 0.6745


def residual_over_floor(k, C, sampled, acs, sigma, ax, rng):
    y = k * sampled[None]
    den = np.maximum((np.abs(C) ** 2).sum(0), 1e-12)
    def resid(data):
        x = (C.conj() * ifft(data, ax)).sum(0) / den
        return (data - fft(C * x[None], ax)) * sampled[None]
    out = sampled & ~acs
    er = (np.abs(resid(y)[:, out]) ** 2).sum()
    fl = 0.0
    for _ in range(2):
        n = sigma[(slice(None),) + (None,) * (k.ndim - 1)] * (rng.normal(size=k.shape) + 1j * rng.normal(size=k.shape)) / np.sqrt(2)
        fl += (np.abs(resid(n * sampled[None])[:, out]) ** 2).sum() / 2
    return float(er / max(fl, 1e-30))


def subspace_change(C, R, obj):
    num = np.abs((C.conj() * R).sum(0)) ** 2
    den = (np.abs(C) ** 2).sum(0) * (np.abs(R) ** 2).sum(0)
    e = 1 - num / np.maximum(den, 1e-30)
    return float(e[obj].mean())


def _to_numpy(x):
    if hasattr(x, "detach"):                       # torch tensor
        x = x.detach().cpu()
        if not x.is_complex() and x.shape[-1] == 2:  # real view (..., 2) of complex data
            import torch
            x = torch.view_as_complex(x.contiguous())
        x = x.numpy()
    return np.asarray(x)


def iter_inputs(a, report):
    """(name, k-space array) for the files and/or the dataset items."""
    for path in a.files:
        arr = np.load(path)
        yield os.path.basename(path), _to_numpy(arr[a.key] if hasattr(arr, "files") else arr)
    if a.loader:
        import importlib
        sys.path.insert(0, os.getcwd())             # your module, importable from where you run this
        mod, fn = a.loader.split(":")
        ds = getattr(importlib.import_module(mod), fn)(a.base_folder) if a.base_folder else \
            getattr(importlib.import_module(mod), fn)()
        n = len(ds)
        idx = a.indices if a.indices else \
            sorted(np.random.default_rng(a.pick_seed).choice(n, size=min(a.random or 6, n), replace=False).tolist())
        report["dataset"] = {"loader": a.loader, "length": n, "indices": idx}
        print(f"dataset {a.loader}: {n} items, using {idx}", flush=True)
        for i in idx:
            item = ds[i]
            k = _to_numpy(item[a.key] if isinstance(item, dict) else item)
            meta = {key: str(item[key])[:120] for key in ("source", "dataset", "filename", "fname", "slice", "contrast")
                    if isinstance(item, dict) and key in item}
            report.setdefault("item_meta", {})[f"item{i}"] = meta
            yield f"item{i}", k


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--loader", default=None, metavar="MODULE:FUNCTION",
                    help="dataset factory, called as FUNCTION(base_folder); items are dicts with --key")
    ap.add_argument("--base-folder", default=None)
    ap.add_argument("--indices", type=int, nargs="*", default=None, help="dataset items to use")
    ap.add_argument("--random", type=int, default=0, help="or: this many random items (seeded, --pick-seed)")
    ap.add_argument("--pick-seed", type=int, default=0)
    ap.add_argument("--key", default="kspace")
    ap.add_argument("--coil-axis", type=int, default=0)
    ap.add_argument("--layout", default="2d", choices=["2d", "3d"])
    ap.add_argument("--calib", type=int, default=None, help="calibration size (default: 24 in 2-D, 12 in 3-D)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budgets", nargs="+", default=list(BUDGETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--R", type=float, default=4.0, help="retrospective acceleration for the SENSE test")
    ap.add_argument("--noise-npy", default=None, help="noise-only samples (n, coils) instead of the periphery estimate")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default="accuracy_sweep.json")
    a = ap.parse_args()

    import torch
    from bspinn.api import estimate_maps, sense_image
    from bspinn.masks import cartesian_mask
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, using the CPU (slow)"); a.device = "cpu"
    calib = a.calib or (24 if a.layout == "2d" else 12)
    report = {"device": a.device, "budgets": {b: BUDGETS[b] for b in a.budgets}, "seeds": a.seeds,
              "R": a.R, "calib": calib, "files": {}}
    rng = np.random.default_rng(0)
    for name, k in iter_inputs(a, report):
        if a.layout == "2d" and k.ndim == 4:        # (slices, coils, x, y) volume: its middle slice
            name, k = f"{name}_slice{k.shape[0] // 2}", k[k.shape[0] // 2]
        if k.ndim != (3 if a.layout == "2d" else 4):
            print(f"skipping {name}: expected (coils, x, y{', z' if a.layout == '3d' else ''}), got {k.shape}")
            continue
        k = np.moveaxis(k, a.coil_axis, 0).astype(np.complex64)
        ax = tuple(range(1, k.ndim))
        sampled = np.abs(k).sum(0) > 0
        acs = calib_mask(k.shape[1:], (calib,) * (k.ndim - 1)) & sampled
        sigma = (np.std(np.load(a.noise_npy).reshape(-1, k.shape[0]), axis=0) if a.noise_npy
                 else noise_sigma(k, sampled))
        rss = np.sqrt((np.abs(ifft(k, ax)) ** 2).sum(0))
        obj = rss > 0.1 * np.percentile(rss, 99)
        R = a.R if a.layout == "2d" else (2, a.R / 2)
        und = cartesian_mask(k.shape[1:], R=R, acs=calib, extent=sampled) & sampled
        print(f"== {name}: {k.shape}, sampled {sampled.mean():.2f}, object {obj.mean():.2f}, "
              f"undersampled R_eff {sampled.sum() / und.sum():.2f}", flush=True)
        runs, maps = [], {}
        for b in a.budgets:
            ia, iw, ip = BUDGETS[b]
            for seed in a.seeds:
                t = time.time()
                C, info = estimate_maps(k, a.layout, calib=calib, device=a.device, verbose=False, seed=seed,
                                        iters_auto=ia, iters_wires=iw, iters_pinn=ip, compile=a.compile)
                sec = time.time() - t
                ref_img = (C.conj() * ifft(k, ax)).sum(0) / np.maximum((np.abs(C) ** 2).sum(0), 1e-12)
                x = sense_image(k * und[None], C, a.layout, mask=und, device=a.device)
                nr = float(np.linalg.norm((x - ref_img)[obj]) / np.linalg.norm(ref_img[obj]))
                r = {"budget": b, "seed": seed, "seconds": sec, "phase_seconds": info.get("timings"),
                     "data_loss": float(info["data_loss"]) if info.get("data_loss") is not None else None,
                     "residual_over_floor": residual_over_floor(k, C, sampled, acs, sigma, ax, rng),
                     f"sense_nrmse_R{a.R:g}": nr, "handedness": info.get("handedness"),
                     "plane": info.get("plane") or info.get("orientation")}
                maps[(b, seed)] = C
                runs.append(r)
                print(f"  {b:9s} seed {seed}: {sec:6.1f} s  residual/floor {r['residual_over_floor']:.4f}  "
                      f"SENSE nrmse {nr:.4f}  data_loss {r['data_loss']}", flush=True)
        longest = a.budgets[-1]
        for r in runs:
            r["change_vs_longest"] = subspace_change(maps[(r["budget"], r["seed"])], maps[(longest, r["seed"])], obj)
        spread = {}
        for b in a.budgets:
            v = [subspace_change(maps[(b, s1)], maps[(b, s2)], obj) for i, s1 in enumerate(a.seeds) for s2 in a.seeds[i + 1:]]
            spread[b] = float(np.mean(v)) if v else None
        summary = {}
        for b in a.budgets:
            rr = [r for r in runs if r["budget"] == b]
            summary[b] = {key: float(np.mean([r[key] for r in rr])) for key in
                          ("seconds", "residual_over_floor", f"sense_nrmse_R{a.R:g}", "change_vs_longest")}
            summary[b]["seed_spread"] = spread[b]
        print("  budget      s/run  residual/floor  SENSE nrmse  change vs longest  seed spread")
        for b, v in summary.items():
            sp = "-" if v["seed_spread"] is None else f"{v['seed_spread']:.2e}"
            print(f"  {b:9s} {v['seconds']:7.1f}  {v['residual_over_floor']:14.4f}  {v[f'sense_nrmse_R{a.R:g}']:11.4f}"
                  f"  {v['change_vs_longest']:17.2e}  {sp}", flush=True)
        report["files"][name] = {"shape": list(k.shape), "object_fraction": float(obj.mean()),
                                 "summary": summary, "runs": runs}
        json.dump(report, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
