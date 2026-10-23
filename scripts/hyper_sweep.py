"""Hyper-parameter sweep for estimate_maps: the fastest settings that keep the quality.

    python scripts/hyper_sweep.py --loader dataloader_v2:get_dataloader --base-folder /path \\
        --random 8 --synthetic 2 --device cuda --out hyper.json
    python scripts/hyper_sweep.py a.npz b.npz --synthetic 2 --device cuda --out hyper.json   # files
    python scripts/hyper_sweep.py ... --resume          # continue an interrupted sweep (same --out)

Inputs as for accuracy_sweep.py (2-D, fully sampled preferred); low-signal slices are skipped.
``--synthetic N`` adds N synthetic Biot–Savart slices with known maps (one of them with non-Biot–Savart
coil factors), so that a ground-truth error is checked next to the data metrics.

Three phases (all on by default; each needs the previous one, results are cached in --out):

 1. ofat     one knob at a time around the current defaults, at the 'fast' and 'mid' budgets:
               optimiser   adam_beta1, adam_beta2, lr_schedule, lr_warmup
               step sizes  lr_wires, lr_pinn, lr_min_wires, lr_min_pinn, lr_image_wires, lr_image_pinn
               batch       n_colloc (collocation points per step), air_colloc
               regularis.  lambda_phys, lambda_norm, lambda_outside, lambda_curv
               model       network width/depth, latent size, vertices per loop
               budget      iters_auto (plane/handedness probes)
 2. combine  greedy: knob values that improved the score at every phase-1 budget by more than
             --min-gain (score: ground-truth error ratio on the synthetic inputs; the residual if there are
             none) with all guards met (residual, SENSE, worst synthetic input), or that save > --speed-gain
             time at neutral quality; added in order of gain, each kept only if the combination improves.
 3. final    defaults vs the combined settings at every budget: which budget of the new settings matches
             the quality of the current 'default' preset, and how much faster it is.

Cost: phase 1 is ~45 settings x 2 budgets x inputs runs of 3-6 s on a GPU (10 inputs: ~75 min), phases
2 and 3 another ~20-30 min. Interrupt any time and continue with --resume; restrict with --knobs.

Quality metrics per run (see accuracy_sweep.py): k-space residual over the noise floor outside the
calibration block (primary), CG-SENSE error at R = 4, and for synthetic inputs the ground-truth subspace
error. Everything is compared with the default settings at the same budget, input and seed: pooled
medians of the ratios (< 1 = better). Wall time per run is recorded too.
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import accuracy_sweep as acc

BUDGETS = dict(acc.BUDGETS)

KNOBS = {                      # knob -> alternative values (the default is the baseline)
    "adam_beta1": [{"adam_beta1": 0.8}, {"adam_beta1": 0.95}],
    "adam_beta2": [{"adam_beta2": 0.99}, {"adam_beta2": 0.9999}],
    "lr_schedule": [{"lr_schedule": "linear"}, {"lr_schedule": "constant"}, {"lr_warmup": 0.05}],
    "lr_wires": [{"lr_wires": 5e-3}, {"lr_wires": 2e-2}, {"lr_wires": 4e-2}],
    "lr_pinn": [{"lr_pinn": 5e-4}, {"lr_pinn": 2e-3}, {"lr_pinn": 4e-3}, {"lr_pinn": 8e-3}, {"lr_pinn": 1.6e-2}],
    "lr_min_wires": [{"lr_min_wires": 0.01}, {"lr_min_wires": 0.2}],
    "lr_min_pinn": [{"lr_min_pinn": 0.01}, {"lr_min_pinn": 0.3}],
    "lr_image_wires": [{"lr_image_wires": 5e-3}, {"lr_image_wires": 5e-2}],
    "lr_image_pinn": [{"lr_image_pinn": 3e-3}, {"lr_image_pinn": 3e-2}],
    "n_colloc": [{"n_colloc": 256}, {"n_colloc": 2048}, {"n_colloc": 4096}],
    "air_colloc": [{"air_colloc": 0.0}, {"air_colloc": 0.5}],
    "lambda_phys": [{"lambda_phys": 3e-3}, {"lambda_phys": 3e-2}, {"lambda_phys": 1e-1}],
    "lambda_norm": [{"lambda_norm": 1e-3}, {"lambda_norm": 1e-1}],
    "lambda_outside": [{"lambda_outside": 3.0}, {"lambda_outside": 30.0}],
    "lambda_curv": [{"lambda_curv": 1e-2}, {"lambda_curv": 1.0}],
    "network": [{"width": 32}, {"width": 128}, {"depth": 3}, {"depth": 5}],
    "latent": [{"latent": 8}, {"latent": 32}],
    "verts_per_loop": [{"verts_per_loop": 8}, {"verts_per_loop": 16}, {"verts_per_loop": 20}],
    "iters_auto": [{"iters_auto": 30}, {"iters_auto": 60}],
}


def key_of(over, budget, item, seed):
    return json.dumps([sorted(over.items()), budget, item, seed])


class Runner:
    def __init__(self, a, report):
        self.a, self.report = a, report
        self.cache = report.setdefault("runs", {})
        self.items = {}

    def add_item(self, name, k, gt=None, gt_sup=None):
        a = self.a
        k = k.astype(np.complex64)
        ax = (1, 2)
        mag = np.abs(k).sum(0)
        sampled = mag > 1e-6 * mag.max()
        acs = acc.calib_mask(k.shape[1:], (a.calib, a.calib)) & sampled
        psi = acc.noise_cov(k, sampled)
        rss = np.sqrt((np.abs(acc.ifft(k, ax)) ** 2).sum(0))
        noise_rss = float(np.sqrt(np.trace(psi).real))
        snr = float(np.percentile(rss, 99) / max(noise_rss, 1e-30))
        obj = rss > max(0.1 * np.percentile(rss, 99), 4 * noise_rss)
        if gt is None and (snr < 10 or obj.mean() > 0.9 or obj.sum() < 10):
            print(f"  skipping {name}: low signal (SNR {snr:.0f}, object fraction {obj.mean():.2f})")
            return
        from bspinn.masks import cartesian_mask
        und = cartesian_mask(k.shape[1:], R=a.R, acs=a.calib, extent=sampled) & sampled
        self.items[name] = dict(k=k, sampled=sampled, acs=acs, psi=psi, obj=obj, und=und, gt=gt,
                                gt_sup=gt_sup if gt_sup is not None else obj)
        self.report.setdefault("items", {})[name] = {"shape": list(k.shape), "snr": snr,
                                                     "object_fraction": float(obj.mean()), "synthetic": gt is not None}

    def run(self, over, budget, item, seed):
        kk = key_of(over, budget, item, seed)
        if kk in self.cache:
            return self.cache[kk]
        from bspinn.api import estimate_maps, sense_image
        a, it = self.a, self.items[item]
        ia, iw, ip = BUDGETS[budget]
        o = dict(over)
        ia = o.pop("iters_auto", ia)
        t = time.time()
        C, info = estimate_maps(it["k"], "2d", calib=a.calib, device=a.device, verbose=False, seed=seed,
                                iters_auto=ia, iters_wires=iw, iters_pinn=ip, compile=a.compile, **o)
        sec = time.time() - t
        ax = (1, 2)
        ref = (C.conj() * acc.ifft(it["k"], ax)).sum(0) / np.maximum((np.abs(C) ** 2).sum(0), 1e-12)
        x = sense_image(it["k"] * it["und"][None], C, "2d", mask=it["und"], device=a.device, prior=None)
        obj = it["obj"]
        r = {"seconds": sec, "phase_seconds": info.get("timings"),
             "residual": acc.residual_over_floor(it["k"], C, it["sampled"], it["acs"], it["psi"], ax,
                                                 np.random.default_rng(0)),
             "sense": float(np.linalg.norm((x - ref)[obj]) / np.linalg.norm(ref[obj]))}
        if it["gt"] is not None:
            r["gt_err"] = acc.subspace_change(C, it["gt"], it["gt_sup"])
        self.cache[kk] = r
        return r

    def pooled(self, over, budget, base_over=None, base_budget=None):
        """Median over inputs and seeds of metric(over, budget) / metric(baseline, base_budget)."""
        base_over = {} if base_over is None else base_over
        base_budget = base_budget or budget
        rat = {"residual": [], "sense": [], "gt_err": [], "seconds": []}
        for item in self.items:
            for seed in self.a.seeds:
                r = self.run(over, budget, item, seed)
                b = self.run(base_over, base_budget, item, seed)
                for m in rat:
                    if m in r and m in b and b[m] > 0:
                        rat[m].append(r[m] / b[m])
                self.save()
        return {m: (float(np.median(v)) if v else None) for m, v in rat.items()} | \
               {m + "_worst": (float(np.max(v)) if v else None) for m, v in rat.items() if m != "seconds"}

    def save(self):
        json.dump(self.report, open(self.a.out, "w"), indent=1)


def score(p):
    """Single number for ranking (lower = better): the ground-truth error ratio on the synthetic inputs if
    there are any (the real-data metrics move by < 0.5 % between settings: too flat to rank by, see the
    first sweep), else the residual ratio."""
    return p["gt_err"] if p.get("gt_err") is not None else p["residual"]


def acceptable(p, tol, res_tol=0.002):
    """Guards: residual (median) not worse than 1 + res_tol, SENSE not worse than 1 + tol, ground-truth
    error not worse than 1 + tol on ANY synthetic input."""
    if p["sense"] is not None and p["sense"] > 1 + tol:
        return False
    if p.get("gt_err_worst") is not None and p["gt_err_worst"] > 1 + tol:
        return False
    return p["residual"] <= 1 + res_tol


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--key", default="kspace")
    ap.add_argument("--coil-axis", type=int, default=0)
    ap.add_argument("--loader", default=None)
    ap.add_argument("--base-folder", default=None)
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--pick-seed", type=int, default=0)
    ap.add_argument("--synthetic", type=int, default=4, help="synthetic inputs with ground truth")
    ap.add_argument("--calib", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--R", type=float, default=4.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--phases", nargs="+", default=["ofat", "combine", "final"])
    ap.add_argument("--knobs", nargs="*", default=None, help=f"subset of: {' '.join(KNOBS)}")
    ap.add_argument("--ofat-budgets", nargs="+", default=["fast", "mid"])
    ap.add_argument("--min-gain", type=float, default=0.02,
                    help="relative gain in the score (ground-truth error; residual without synthetic inputs)")
    ap.add_argument("--tol", type=float, default=0.01, help="allowed relative loss in SENSE / ground truth")
    ap.add_argument("--speed-gain", type=float, default=0.05, help="time saving that counts for a quality-neutral knob")
    ap.add_argument("--try", dest="try_", nargs="*", default=None, metavar="JSON",
                    help='extra settings to compare at every budget, e.g. \'{"lr_pinn": 0.004}\'')
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="hyper_sweep.json")
    a = ap.parse_args()
    a.layout = "2d"
    import torch
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, using the CPU (slow)"); a.device = "cpu"
    report = json.load(open(a.out)) if (a.resume and os.path.exists(a.out)) else {}
    report.update(device=a.device, budgets=BUDGETS, seeds=a.seeds, calib=a.calib)
    R = Runner(a, report)

    print("inputs:")
    for name, k in acc.iter_inputs(a, report):
        if k.ndim == 4:
            name, k = f"{name}_slice{k.shape[0] // 2}", k[k.shape[0] // 2]
        if k.ndim != 3:
            print(f"  skipping {name}: shape {k.shape}"); continue
        R.add_item(name, np.moveaxis(k, a.coil_axis, 0))
    from bspinn.data import make_synthetic
    for s in range(a.synthetic):
        d, gt, obj = make_synthetic(8, (160, 160), noise=0.01, seed=100 + s, mismatch=0.3 if s % 2 else 0.0)
        R.add_item(f"synthetic{s}{'_mismatch' if s % 2 else ''}", d.kspace[0], gt=gt[0],
                   gt_sup=np.abs(obj[0]) > 0)
    print(f"  {len(R.items)} inputs: {list(R.items)}")
    if not R.items:
        raise SystemExit("no usable inputs")

    knobs = a.knobs or list(KNOBS)
    res = report.setdefault("ofat", {})
    if "ofat" in a.phases:
        print("phase 1: one knob at a time (ratios vs defaults at the same budget; < 1 = better)")
        for b in a.ofat_budgets:
            for kn in knobs:
                for over in KNOBS[kn]:
                    p = R.pooled(over, b)
                    res.setdefault(b, {})[json.dumps(over)] = p
                    print(f"  {b:5s} {json.dumps(over):32s} residual {p['residual']:.4f} (worst {p['residual_worst']:.4f})"
                          f"  SENSE {p['sense']:.4f}  GT {p['gt_err'] if p['gt_err'] is None else round(p['gt_err'], 4)}"
                          f"  time {p['seconds']:.2f}", flush=True)
                    R.save()

    if "combine" in a.phases:
        bs = [b for b in a.ofat_budgets if b in res]
        cand = []
        for kn in knobs:
            best = None
            for over in KNOBS[kn]:
                ps = [res[b].get(json.dumps(over)) for b in bs]
                if any(p is None for p in ps) or not all(acceptable(p, a.tol) for p in ps):
                    continue
                better = all(score(p) < 1 - a.min_gain for p in ps)          # consistent at every budget
                faster = all(p["seconds"] < 1 - a.speed_gain and score(p) <= 1 + a.tol / 2 for p in ps)
                if better or faster:
                    g = max(score(p) for p in ps) - (a.speed_gain if faster and not better else 0)
                    if best is None or g < best[2]:
                        best = (over, ps, g)
            if best:
                cand.append(best)
        cand.sort(key=lambda c: c[2])
        fb = bs[0]
        print(f"phase 2: combining {len(cand)} knob values (better at {bs}, or faster and neutral): "
              f"{[c[0] for c in cand]}")
        combo, cur = {}, None
        for over, _, _ in cand:
            trial = {**combo, **over}
            ps = [R.pooled(trial, b) for b in bs]
            ok = all(acceptable(p, a.tol) for p in ps)
            gain = cur is None or max(score(p) for p in ps) < max(score(p) for p in cur) - a.min_gain / 2 \
                or (max(p["seconds"] for p in ps) < max(p["seconds"] for p in cur) - a.speed_gain / 2
                    and max(score(p) for p in ps) <= max(score(p) for p in cur) + a.tol / 2)
            keep = ok and gain
            print(f"  + {json.dumps(over):32s} -> score {[round(score(p), 4) for p in ps]}  residual "
                  f"{[round(p['residual'], 4) for p in ps]}  SENSE {[round(p['sense'], 4) for p in ps]}  time "
                  f"{[round(p['seconds'], 2) for p in ps]}  {'kept' if keep else 'dropped'}", flush=True)
            if keep:
                combo, cur = trial, ps
        report["combined"] = {"settings": combo, "pooled": dict(zip(bs, cur or []))}
        R.save()

    if "final" in a.phases:
        combo = report.get("combined", {}).get("settings", {})
        print(f"phase 3: defaults vs {combo} at every budget, relative to the defaults at 'default'")
        fin = {}
        extra = {f"try{i}": json.loads(t) for i, t in enumerate(a.try_ or [])}
        sets = {"defaults": {}, "combined": combo, **extra}
        for b in BUDGETS:
            fin[b] = {name: R.pooled(over, b, {}, "default") for name, over in sets.items()}
            for name in sets:
                p = fin[b][name]
                print(f"  {b:9s} {name:8s} residual {p['residual']:.4f}  SENSE {p['sense']:.4f}  GT {p['gt_err']}"
                      f"  time {p['seconds']:.2f} x", flush=True)
            R.save()
        report["final"] = {"table": fin, "settings": sets, "matches": {}}
        for name, over in sets.items():
            match = next((b for b in BUDGETS if acceptable(fin[b][name], a.tol / 2) and score(fin[b][name]) <= 1.0), None)
            tm = "n/a" if match is None else "%.2fx the time" % fin[match][name]["seconds"]
            report["final"]["matches"][name] = {"budget": match, "time_vs_default":
                                                None if match is None else fin[match][name]["seconds"]}
            print(f"=> {name} {over}: smallest budget at the current 'default' quality: {match} ({tm})")
        R.save()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
