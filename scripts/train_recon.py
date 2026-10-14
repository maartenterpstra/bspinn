"""Train the unrolled reconstruction network of scripts/zs_ssl.py on a whole dataset, generalising over
undersampling masks; evaluate it with bspinn (or any) coil maps on held-out slices.

Every training sample is one dataset item (pre-whitened as in evaluate_methods.py) with a NEW random mask:
acceleration R ~ U[--R-min, --R-max], pattern from --patterns (vd | random | uniform lines), calibration
lines from --acs-choices, phase encoding along the item's PE axis. Coil maps for training come from that
mask's calibration region (fast estimators computed in the data workers: inati | walsh | lowres), or from a
cache precomputed once per item with any bspinn.methods estimator (e.g. the PINN: --precompute-maps
pinn_api, shardable over GPUs; then the calibration size is fixed to --maps-acs).

Losses: 'supervised' (the fully sampled SENSE-1 image with the same maps; normalised l1-l2 in the image
domain, inside the acquired k-space extent) or 'ssdu' (self-supervised on the acquired samples only: random
Theta / Lambda split per step, normalised l1-l2 on the Lambda k-space; usable on prospectively undersampled
data). Held-out evaluation slices (the same picks as evaluate_methods.py --n / --pick-seed) and every index
within --exclude-window of them (neighbouring slices of the same volume) are never trained on; a validation
set (--val-n) is drawn from the rest and chooses the best checkpoint.

    # 1. (optional) PINN maps for every training item, split over 4 GPUs / jobs
    python scripts/train_recon.py --loader dataloader_v2:get_dataloader --base-folder /path \\
        --precompute-maps pinn_api --shard 0/4 --device cuda:0 --out runs/recon
    # 2. train (resumable: rerun the same command)
    python scripts/train_recon.py --loader dataloader_v2:get_dataloader --base-folder /path \\
        --train-maps inati --loss supervised --steps 100000 --device cuda --amp --workers 8 --out runs/recon
    # 3. evaluate on the 20 slices of runs/eval_joint at several R, with PINN and NLINV maps (+ optional
    #    zero-shot fine-tuning per slice from the trained weights)
    python scripts/train_recon.py --loader ... --eval --eval-R 4 6 8 --eval-maps pinn_api nlinv \\
        --maps-cache runs/eval_joint --zs-epochs 0 10 --device cuda --amp --out runs/recon
"""
import argparse
import contextlib
import dataclasses
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

import torch                                    # noqa: E402
import zs_ssl as Z                              # noqa: E402

FAST_MAPS = ("inati", "walsh", "lowres")


# ----------------------------------------------------------------------------------- data
def _load_dataset(spec, base):
    import importlib
    sys.path.insert(0, os.getcwd())
    mod, fn = spec.split(":")
    f = getattr(importlib.import_module(mod), fn)
    return f(base) if base else f()


def _item_kspace(ds, i, key, coil_axis, rng=None):
    from accuracy_sweep import _to_numpy
    item = ds[i]
    k = _to_numpy(item[key] if isinstance(item, dict) else item)
    if k.ndim == 4:                                   # a volume: random slice for training, middle otherwise
        k = k[rng.integers(k.shape[0]) if rng is not None else k.shape[0] // 2]
    k = np.moveaxis(k, coil_axis, 0)
    return k if k.ndim == 3 else None


def _case(i, k, acs):
    import evaluate_methods as evm
    with contextlib.redirect_stdout(io.StringIO()):
        return evm.from_kspace(f"item{i}", k, acs)


def _maps(case, name, acs, cache_dir=None, est_kw=None):
    """Calibration-only maps from the central acs x acs box (inside every mask with >= acs calibration lines)."""
    from bspinn import methods as bm
    c = dataclasses.replace(case, acs=(acs, acs), joint=False)
    if name in FAST_MAPS:
        with contextlib.redirect_stdout(io.StringIO()):
            C = bm.ESTIMATORS[name](c).astype(np.complex64)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(os.path.join(cache_dir, f"{name}.npy"), C)
        return C
    m_, _ = bm.estimate(c, [name], cache_dir=cache_dir, log=lambda *x, **k: None, **(est_kw or {}))
    return m_[name]


def _cache_path(root, name, i):
    return os.path.join(root, "maps", name, f"item{i}")


class TrainItems(torch.utils.data.Dataset):
    """Item index -> dict(k, sampled, mask, maps, acs, R, pattern). Random mask per call."""

    def __init__(self, a, indices, fixed_mask=False):
        self.a, self.idx, self.fixed = a, list(indices), fixed_mask
        self.ds = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        from bspinn.masks import cartesian_mask
        a = self.a
        if self.ds is None:
            self.ds = _load_dataset(a.loader, a.base_folder)
        i = self.idx[j]
        rng = np.random.default_rng(1000003 * i + 1) if self.fixed else np.random.default_rng(int(torch.randint(2 ** 62, (1,))))
        try:
            k = _item_kspace(self.ds, i, a.key, a.coil_axis, None if self.fixed else rng)
            if k is None:
                return None
            cached = a.train_maps.startswith("cache:")
            if self.fixed:
                R, pattern, acs = 4.0, "vd", (a.maps_acs if cached else 24)
            else:
                R = float(rng.uniform(a.R_min, a.R_max))
                pattern = str(rng.choice(a.patterns))
                acs = a.maps_acs if cached else int(rng.choice(a.acs_choices))
            case = _case(i, k, acs)
            m = cartesian_mask(case.shape, R=R, acs=acs, pattern=pattern, pe_axes=(case.pe_axis,),
                               seed=1 if self.fixed else int(rng.integers(2 ** 31))) & case.sampled
            if cached:
                p = os.path.join(_cache_path(a.out, a.train_maps[6:], i), f"{a.train_maps[6:]}.npy")
                if not os.path.exists(p):
                    return None                        # not precomputed (yet): skipped
                C = np.load(p)
            else:
                C = _maps(case, a.train_maps, acs)
            return {"index": i, "k": case.ksp.astype(np.complex64), "sampled": case.sampled, "mask": m,
                    "maps": C.astype(np.complex64), "R": R, "pattern": pattern, "acs": acs}
        except Exception as e:                           # a broken item must not stop a long run
            print(f"item {i}: {type(e).__name__}: {e}", flush=True)
            return None


def split_indices(a, n_total):
    """Evaluation picks (as evaluate_methods.py), an exclusion window around them, validation, training."""
    ev = sorted(a.indices) if a.indices else sorted(np.random.default_rng(a.pick_seed).choice(
        n_total, size=min(a.n, n_total), replace=False).tolist())
    banned = np.zeros(n_total, bool)
    for i in ev:
        banned[max(0, i - a.exclude_window): i + a.exclude_window + 1] = True
    pool = np.flatnonzero(~banned)
    pool = np.random.default_rng(a.split_seed).permutation(pool)
    val = sorted(int(i) for i in pool[: a.val_n])
    banned_v = banned.copy()
    for i in val:
        banned_v[max(0, i - a.exclude_window): i + a.exclude_window + 1] = True
    train = [int(i) for i in pool[a.val_n:] if not banned_v[i]]
    if a.train_n:
        train = train[: a.train_n]
    return ev, val, train


# ----------------------------------------------------------------------------------- training step
def _band(x, ext):
    return Z.ifft2c(Z.fft2c(x) * ext)


def prepare(s, device):
    """Batch dict -> device tensors, data scaled so that the regularised SENSE start has p99 ~ 1."""
    k = torch.as_tensor(s["k"]).to(device)
    C = torch.as_tensor(s["maps"]).to(device)
    C = C / torch.sqrt((C.abs() ** 2).sum(0, keepdim=True)).clamp_min(1e-12)
    m = torch.as_tensor(np.asarray(s["mask"], np.float32)).to(device)
    full = torch.as_tensor(np.asarray(s["sampled"], np.float32)).to(device)
    from bspinn.api import acquired_extent
    ext = torch.as_tensor(acquired_extent(np.asarray(s["sampled"], bool)).astype(np.float32)).to(device)
    op = Z.Sense(C, m)
    y = m * k
    with torch.no_grad():
        x0 = Z.cg(op, op.AH(y), torch.tensor(0.05, device=device), torch.zeros_like(op.AH(y)), 10)
        sc = torch.quantile(x0.abs().flatten(), 0.99).clamp_min(1e-12)
    return dict(C=C, m=m, full=full, ext=ext, y=y / sc, kfull=(full * k) / sc, scale=sc)


def loss_step(net, t, a, rng):
    if a.loss == "supervised":
        target = (t["C"].conj() * Z.ifft2c(t["kfull"])).sum(0)          # SENSE-1 of the full data, same maps
        out = _band(net(Z.Sense(t["C"], t["m"]), t["y"]), t["ext"])
        d = out - target
        return torch.linalg.vector_norm(d) / torch.linalg.vector_norm(target).clamp_min(1e-12) + \
            d.abs().sum() / target.abs().sum().clamp_min(1e-12)
    om = t["m"].cpu().numpy() > 0
    lam = Z.select(om, a.rho, rng, mode=a.select)
    th = torch.as_tensor((om & ~lam).astype(np.float32), device=t["m"].device)
    out = net(Z.Sense(t["C"], th), t["y"] * th)
    return Z.kspace_loss(Z.Sense(t["C"], torch.ones_like(t["m"])), out, t["y"],
                         torch.as_tensor(lam.astype(np.float32), device=t["m"].device))


def validate(net, val_items, a):
    from sense_lambda_eval import metrics
    from bspinn import evaluation as ev
    net.eval()
    out = []
    for s in val_items:
        t = prepare(s, a.device)
        with torch.no_grad():
            x = _band(net(Z.Sense(t["C"], t["m"]), t["y"]), t["ext"]) * t["scale"]
        k = np.asarray(s["k"]); sm = np.asarray(s["sampled"], bool)
        rss = np.sqrt((np.abs(ev.ifft2c(k * sm[None])) ** 2).sum(0))
        from scipy.ndimage import binary_fill_holes, binary_opening, uniform_filter
        r = uniform_filter(rss, 7)
        obj = binary_fill_holes(binary_opening(r > 0.1 * np.quantile(r, 0.99), iterations=2))
        out.append(metrics(x.cpu().numpy(), rss, obj))
    net.train()
    return {kk: float(np.median([o[kk] for o in out])) for kk in out[0]} if out else {}


def train(a):
    ds = _load_dataset(a.loader, a.base_folder)
    ev_idx, val_idx, tr_idx = split_indices(a, len(ds))
    del ds
    os.makedirs(a.out, exist_ok=True)
    json.dump({"evaluation": ev_idx, "validation": val_idx, "train_count": len(tr_idx), "args": vars(a)},
              open(os.path.join(a.out, "split.json"), "w"), indent=1, default=str)
    print(f"{len(tr_idx)} training items, {len(val_idx)} validation, {len(ev_idx)} held-out evaluation "
          f"(+/- {a.exclude_window} indices excluded around evaluation and validation items)", flush=True)
    val_ds = TrainItems(a, val_idx, fixed_mask=True)
    val_items = [s for s in (val_ds[j] for j in range(len(val_ds))) if s is not None]
    loader = torch.utils.data.DataLoader(TrainItems(a, tr_idx), batch_size=None, shuffle=True, num_workers=a.workers,
                                         persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    net = Z.make_net(a, a.device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    warm = max(1, int(0.02 * a.steps))

    def lr_at(st):          # linear warm-up, cosine decay to lr/100 (stateless: --steps may change on resume)
        if st < warm:
            return a.lr * (0.1 + 0.9 * st / warm)
        q = min(1.0, (st - warm) / max(1, a.steps - warm))
        return a.lr * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * q)))
    step, best = 0, float("inf")
    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last):
        st = torch.load(last, map_location=a.device, weights_only=False)   # our own checkpoint
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"])
        step, best = st["step"], st["best"]
        print(f"resumed at step {step} (best validation NRMSE {best:.4f})", flush=True)
    rng = np.random.default_rng(a.seed + step)
    logf = open(os.path.join(a.out, "train_log.jsonl"), "a")
    t0, run, n_acc = time.time(), [], 0
    net.train()
    while step < a.steps:
        for s in loader:
            if s is None:
                continue
            t = prepare(s, a.device)
            loss = loss_step(net, t, a, rng) / a.accum
            loss.backward()
            run.append(float(loss.detach()) * a.accum)
            n_acc += 1
            if n_acc < a.accum:
                continue
            n_acc = 0
            torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0:
                print(f"step {step}/{a.steps}  loss {np.mean(run[-a.log_every * a.accum:]):.4f}  mu {float(net.mu.detach()):.3f}  "
                      f"lr {lr_at(step):.2e}  {(time.time() - t0) / a.log_every:.2f}s/step", flush=True)
                t0 = time.time()
            if step % a.val_every == 0 or step == a.steps:
                v = validate(net, val_items, a)
                rec = {"step": step, "train_loss": float(np.mean(run[-a.val_every * a.accum:])), **{f"val_{k}": x for k, x in v.items()}}
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                print(f"  validation (R=4 vd, {len(val_items)} slices): " + "  ".join(f"{k} {x:.4f}" for k, x in v.items()), flush=True)
                if v and v["nrmse"] < best:
                    best = v["nrmse"]
                    torch.save(net.state_dict(), os.path.join(a.out, "best.pt"))
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                            "step": step, "best": best}, last)
            if step >= a.steps:
                break
    print(f"done: best validation NRMSE {best:.4f}; weights {os.path.join(a.out, 'best.pt')}")


# ----------------------------------------------------------------------------------- precompute / evaluate
def precompute(a):
    ds = _load_dataset(a.loader, a.base_folder)
    ev_idx, val_idx, tr_idx = split_indices(a, len(ds))
    k_, n_ = (int(x) for x in a.shard.split("/"))
    todo = sorted(val_idx + tr_idx)[k_::n_]
    est_kw = {"threads": 1, "cfg_overrides": {}, "device": a.device, "quality": a.pinn_quality}
    t0 = time.time()
    for c, i in enumerate(todo):
        d = _cache_path(a.out, a.precompute_maps, i)
        if os.path.exists(os.path.join(d, f"{a.precompute_maps}.npy")):
            continue
        try:
            k = _item_kspace(ds, i, a.key, a.coil_axis)
            if k is None:
                continue
            _maps(_case(i, k, a.maps_acs), a.precompute_maps, a.maps_acs, cache_dir=d, est_kw=est_kw)
        except Exception as e:
            print(f"item {i}: {type(e).__name__}: {e}", flush=True); continue
        if (c + 1) % 20 == 0:
            print(f"shard {a.shard}: {c + 1}/{len(todo)}  {(time.time() - t0) / (c + 1):.1f}s/item", flush=True)


def evaluate(a):
    """Held-out slices, joint protocol at each --eval-R (method_lab.undersampled: vd + ACS, seed 1), maps from
    --eval-maps (cached as evaluate_methods.py --joint does): CG-SENSE (auto lambda) vs the trained network,
    optionally fine-tuned per slice by ZS-SSL for each of --zs-epochs."""
    import evaluate_methods as evm
    from method_lab import undersampled
    from sense_lambda_eval import metrics
    from bspinn import evaluation as ev
    from bspinn import methods as bm
    from bspinn.api import acquired_extent, sense_image
    w = a.weights or os.path.join(a.out, "best.pt")
    state = torch.load(w, map_location=a.device)
    res_path = os.path.join(a.out, "eval.jsonl")
    done = set()
    if os.path.exists(res_path):
        done = {(r["case"], r["maps"], r["method"]) for r in map(json.loads, (l for l in open(res_path) if l.strip()))}
    est_kw = {"threads": 1, "cfg_overrides": {}, "device": a.device, "quality": a.pinn_quality}
    for make in evm.loader_cases(a.loader, a.base_folder, a.indices, a.n, a.pick_seed, a.key, a.coil_axis, 24, None, None):
        case0 = make()
        obj = ev.object_mask(case0)
        if obj.sum() < 0.005 * obj.size:
            continue
        ext = acquired_extent(case0.sampled)
        rss = np.sqrt((np.abs(ev.ifft2c(case0.ksp * case0.sampled[None])) ** 2).sum(0))
        for R in a.eval_R:
            fit = undersampled(case0, R)
            fit.name = f"{case0.name}_j{R:g}"
            for mname in a.eval_maps:
                methods = ["cg_auto", "net"] + [f"net_zs{e}" for e in a.zs_epochs if e > 0]
                if all((fit.name, mname, m) in done for m in methods):
                    continue
                cdir = os.path.join(a.maps_cache, fit.name, "maps") if a.maps_cache else os.path.join(a.out, "eval_maps", fit.name)
                m_, _ = bm.estimate(fit, [mname], cache_dir=cdir, log=print, **est_kw)
                Cn = ev.normalise(m_[mname])
                recs = []

                def record(method, x, extra=None):
                    q = metrics(Z.band_limit(x, ext), rss, obj)
                    recs.append({"case": fit.name, "R": R, "maps": mname, "method": method, **q, **(extra or {})})
                    print(f"{fit.name} {mname:10s} {method:9s} NRMSE {q['nrmse']:.4f}  SSIM {q['ssim']:.4f}  PSNR {q['psnr']:.2f}", flush=True)

                record("cg_auto", sense_image(fit.ksp, Cn, "2d", mask=fit.sampled, lam="auto", device=a.device))
                sl = Z.Slice(fit.ksp, Cn, fit.sampled, a.device)
                net = Z.make_net(a, a.device); net.load_state_dict(state)
                t0 = time.time()
                record("net", Z.reconstruct(net, sl, sl.omega), {"seconds": time.time() - t0})
                for e in a.zs_epochs:
                    if e <= 0:
                        continue
                    net = Z.make_net(a, a.device); net.load_state_dict(state)
                    a_ = argparse.Namespace(**{**vars(a), "epochs": e})
                    t0 = time.time()
                    info = Z.train_zero_shot(net, sl, a_, np.random.default_rng(a.seed + 1), print, tag=f"{fit.name}/{mname}")
                    record(f"net_zs{e}", Z.reconstruct(net, sl, sl.omega), {"seconds": time.time() - t0, "epochs": info["epochs"]})
                with open(res_path, "a") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
    rows = [json.loads(l) for l in open(res_path) if l.strip()] if os.path.exists(res_path) else []
    print("\nmedian NRMSE / SSIM / PSNR over held-out slices, and the fraction of slices beating CG-SENSE (same maps, same R):")
    for R in dict.fromkeys(r["R"] for r in rows):
        for mname in dict.fromkeys(r["maps"] for r in rows):
            base = {r["case"]: r for r in rows if r["R"] == R and r["maps"] == mname and r["method"] == "cg_auto"}
            for meth in dict.fromkeys(r["method"] for r in rows):
                rs = [r for r in rows if r["R"] == R and r["maps"] == mname and r["method"] == meth]
                if not rs:
                    continue
                win = [r["nrmse"] < base[r["case"]]["nrmse"] for r in rs if r["case"] in base]
                print(f"  R={R:<4g} {mname:10s} {meth:9s} n={len(rs):3d}  {np.median([r['nrmse'] for r in rs]):.4f}  "
                      f"{np.median([r['ssim'] for r in rs]):.4f}  {np.median([r['psnr'] for r in rs]):6.2f}"
                      + (f"  wins {np.mean(win):.2f}" if meth != "cg_auto" else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("data")
    g.add_argument("--loader", required=True, metavar="MODULE:FUNCTION")
    g.add_argument("--base-folder", default=None)
    g.add_argument("--key", default="kspace")
    g.add_argument("--coil-axis", type=int, default=0)
    g.add_argument("--indices", type=int, nargs="*", default=None, help="held-out evaluation items (default: --n picked by --pick-seed)")
    g.add_argument("--n", type=int, default=20)
    g.add_argument("--pick-seed", type=int, default=0)
    g.add_argument("--exclude-window", type=int, default=300, help="indices this close to an evaluation/validation item are not trained on")
    g.add_argument("--val-n", type=int, default=20)
    g.add_argument("--train-n", type=int, default=0, help="cap on training items (0 = all)")
    g.add_argument("--split-seed", type=int, default=1)
    g = ap.add_argument_group("masks (training: a new one per sample)")
    g.add_argument("--R-min", type=float, default=2.0)
    g.add_argument("--R-max", type=float, default=8.0)
    g.add_argument("--patterns", nargs="+", default=["vd", "random", "uniform"])
    g.add_argument("--acs-choices", type=int, nargs="+", default=[16, 24, 32])
    g = ap.add_argument_group("coil maps")
    g.add_argument("--train-maps", default="inati", help="inati | walsh | lowres (from each mask's calibration region, "
                                                          "computed in the workers) or cache:NAME (from --precompute-maps NAME)")
    g.add_argument("--maps-acs", type=int, default=24, help="calibration size of cached maps (fixes the training ACS)")
    g.add_argument("--precompute-maps", default=None, metavar="NAME", help="compute and cache NAME maps for all training/validation items")
    g.add_argument("--shard", default="0/1", help="k/n: this job's share of --precompute-maps")
    g.add_argument("--pinn-quality", default="default", choices=["fast", "default"])
    g = ap.add_argument_group("network / optimisation")
    g.add_argument("--unrolls", type=int, default=10)
    g.add_argument("--cg-iters", type=int, default=10)
    g.add_argument("--blocks", type=int, default=15)
    g.add_argument("--ch", type=int, default=64)
    g.add_argument("--mu0", type=float, default=0.05)
    g.add_argument("--loss", default="supervised", choices=["supervised", "ssdu"])
    g.add_argument("--rho", type=float, default=0.4, help="ssdu: |Lambda| / |Omega|")
    g.add_argument("--select", default="gaussian", choices=["gaussian", "uniform"])
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--steps", type=int, default=100000)
    g.add_argument("--accum", type=int, default=1, help="gradient accumulation (items per optimiser step)")
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--val-every", type=int, default=2000)
    g.add_argument("--amp", action="store_true")
    g.add_argument("--no-checkpoint", action="store_true")
    g = ap.add_argument_group("evaluation")
    g.add_argument("--eval", action="store_true", help="evaluate instead of training")
    g.add_argument("--weights", default=None, help="default <out>/best.pt")
    g.add_argument("--eval-R", type=float, nargs="+", default=[4.0])
    g.add_argument("--eval-maps", nargs="+", default=["pinn_api", "nlinv"])
    g.add_argument("--maps-cache", default=None, help="an evaluate_methods.py --joint results folder (reuses <case>_jR/maps)")
    g.add_argument("--zs-epochs", type=int, nargs="+", default=[0], help="also fine-tune per slice by ZS-SSL for this many epochs")
    g.add_argument("--splits", type=int, default=50)
    g.add_argument("--val-frac", type=float, default=0.2)
    g.add_argument("--patience", type=int, default=25)
    g.add_argument("--device", default="cuda")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="runs/recon")
    a = ap.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available: using the CPU (slow)"); a.device = "cpu"
    torch.manual_seed(a.seed)
    if a.precompute_maps:
        precompute(a)
    elif a.eval:
        evaluate(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
