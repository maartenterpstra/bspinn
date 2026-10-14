"""Zero-shot self-supervised reconstruction (ZS-SSL, Yaman et al., ICLR 2022) with bspinn coil maps.

The image prior the joint fit is missing: a physics-guided unrolled network (ResNet denoiser + CG data
consistency, weights shared over the unrolls) trained on the acquired samples of ONE scan only. The acquired
mask Omega is split once into a validation set Gamma (early stopping) and the rest; the rest is split K times
into Theta (data consistency) and Lambda (loss). No fully sampled data is used for training: the fully
sampled k-space only scores the result.

Same protocol as `evaluate_methods.py --joint R`: every slice from your dataloader is pre-whitened,
retrospectively undersampled (variable density R + ACS, method_lab.undersampled), the coil maps come from the
same estimators on the same samples (and are read from that run's cache when --maps-cache points to it).

    # zero-shot on 20 slices (the paper's method), PINN maps from the joint comparison's cache
    python scripts/zs_ssl.py --loader dataloader_v2:get_dataloader --base-folder /path --n 20 --R 4 \\
        --maps pinn_api nlinv --maps-cache runs/eval_joint --device cuda --amp --out runs/zs_ssl

    # optional: self-supervised pretraining (SSDU) on other slices of the same dataset, then zero-shot fine-tuning
    python scripts/zs_ssl.py --loader dataloader_v2:get_dataloader --base-folder /path --n 20 --R 4 \\
        --maps pinn_api --pretrain-n 200 --pretrain-steps 20000 --device cuda --amp --out runs/zs_ssl_tl

Per (slice, map set) the script reports, against the fully sampled RSS image inside the object (one
least-squares scale, k-space band-limited to the acquired extent so partial-Fourier gaps count for nobody):
  cg_auto     CG-SENSE with the automatic Tikhonov weight (bspinn sense_image, lambda='auto')
  zs          ZS-SSL (zero-shot, or fine-tuned from --init / the pretrained weights)
  tl0         (with pretraining) the pretrained network applied without fine-tuning
NRMSE / SSIM / PSNR; one JSON line per (slice, maps, method) in <out>/results.jsonl (resume-safe), images in
<out>/<case>/<maps>.npz, a summary at the end. Training: Adam 5e-4, normalised l1-l2 k-space loss, early stop
on the Gamma loss (patience), best-validation weights. Memory: --checkpoint (default on) recomputes each unroll
in the backward pass; --amp runs the denoiser in bfloat16 (the CG data consistency stays float32).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

import torch                       # noqa: E402


from bspinn.unrolled import (Sense, ResBlock, ResNet, Unrolled, cg, fft2c, ifft2c,  # noqa: E402,F401
                             kspace_loss)


# ----------------------------------------------------------------------------------- masks
def select(mask, frac, rng, keep=(4, 4), mode="gaussian", std_scale=4.0):
    """Choose ``frac`` of the samples in ``mask`` (point-wise), never in the central ``keep`` box;
    'gaussian': probability ~ a Gaussian around the k-space centre (std = size / std_scale), as SSDU."""
    H, W = mask.shape
    cand = mask.copy()
    cand[H // 2 - keep[0] // 2: H // 2 + (keep[0] + 1) // 2, W // 2 - keep[1] // 2: W // 2 + (keep[1] + 1) // 2] = False
    idx = np.flatnonzero(cand)
    n = int(round(frac * mask.sum()))
    n = min(n, idx.size)
    if mode == "gaussian":
        r, c = np.unravel_index(idx, mask.shape)
        p = np.exp(-0.5 * (((r - H / 2) / (H / std_scale)) ** 2 + ((c - W / 2) / (W / std_scale)) ** 2))
        p = p / p.sum()
    else:
        p = None
    pick = rng.choice(idx, size=n, replace=False, p=p)
    out = np.zeros_like(mask)
    out.flat[pick] = True
    return out


# ----------------------------------------------------------------------------------- data
class Slice:
    """One undersampled slice on the device: y (N,H,W) acquired k-space (scaled), C maps, Omega mask."""

    def __init__(self, ku, C, m, device):
        C = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)
        self.C = torch.from_numpy(C.astype(np.complex64)).to(device)
        self.omega = m.astype(bool)
        y = torch.from_numpy((ku * m[None]).astype(np.complex64)).to(device)
        # scale: the regularised SENSE start has a 99th-percentile magnitude of ~1
        op = Sense(self.C, torch.from_numpy(m.astype(np.float32)).to(device))
        with torch.no_grad():
            x0 = cg(op, op.AH(y), torch.tensor(0.05, device=device), torch.zeros_like(op.AH(y)), 10)
            s = float(torch.quantile(x0.abs().flatten(), 0.99).clamp_min(1e-12))
        self.scale = s
        self.y = y / s
        self.device = device

    def op(self, mask_np):
        return Sense(self.C, torch.from_numpy(mask_np.astype(np.float32)).to(self.device))

    def mask_t(self, mask_np):
        return torch.from_numpy(mask_np.astype(np.float32)).to(self.device)


def train_zero_shot(net, sl, a, rng, log, tag=""):
    """ZS-SSL on one slice: Gamma (validation) from Omega, K Theta/Lambda splits of the rest."""
    gamma = select(sl.omega, a.val_frac, rng, mode=a.select)
    train = sl.omega & ~gamma
    lams = [select(train, a.rho * train.sum() / sl.omega.sum(), rng, mode=a.select) for _ in range(a.splits)]
    op_full = sl.op(np.ones_like(sl.omega))
    op_train = sl.op(train)
    g_t = sl.mask_t(gamma)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(a.epochs):
        t0 = time.time()
        net.train()
        order = rng.permutation(len(lams))
        tl = 0.0
        for k in order:
            theta = train & ~lams[k]
            out = net(sl.op(theta), sl.y * sl.mask_t(theta))
            loss = kspace_loss(op_full, out, sl.y, sl.mask_t(lams[k]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += float(loss.detach())
        net.eval()
        with torch.no_grad():
            out = net(op_train, sl.y * sl.mask_t(train))
            vl = float(kspace_loss(op_full, out, sl.y, g_t))
        hist.append({"epoch": ep, "train": tl / len(lams), "val": vl, "seconds": time.time() - t0})
        log(f"  {tag} epoch {ep:3d}  train {tl / len(lams):.4f}  val {vl:.4f}  mu {float(net.mu.detach()):.3f}  "
            f"{time.time() - t0:.1f}s", flush=True)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return {"train_mask": train, "best_val": best, "epochs": len(hist), "best_epoch": int(np.argmin([h["val"] for h in hist])) if hist else None,
            "history": hist}


def reconstruct(net, sl, mask):
    net.eval()
    with torch.no_grad():
        x = net(sl.op(mask), sl.y * sl.mask_t(mask))
    return (x * sl.scale).cpu().numpy()


# ----------------------------------------------------------------------------------- scoring
def band_limit(x, extent):
    if extent.all():
        return x
    ax = (-2, -1)
    k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=ax), norm="ortho"), axes=ax) * extent
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=ax), norm="ortho"), axes=ax)


def make_net(a, device):
    return Unrolled(a.unrolls, a.cg_iters, a.ch, a.blocks, a.mu0, amp=a.amp, ckpt=not a.no_checkpoint).to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("data (as evaluate_methods.py)")
    g.add_argument("--cases", nargs="*", default=[], help="named cases of evaluate_methods.py (e.g. synth8) for a quick test")
    g.add_argument("--loader", default=None, metavar="MODULE:FUNCTION")
    g.add_argument("--base-folder", default=None)
    g.add_argument("--indices", type=int, nargs="*", default=None)
    g.add_argument("--n", type=int, default=6)
    g.add_argument("--pick-seed", type=int, default=0)
    g.add_argument("--key", default="kspace")
    g.add_argument("--coil-axis", type=int, default=0)
    g.add_argument("--acs", type=int, default=24)
    g.add_argument("--R", type=float, default=4.0, help="joint-protocol acceleration (variable density + ACS)")
    g = ap.add_argument_group("coil maps")
    g.add_argument("--maps", nargs="+", default=["pinn_api"], help="estimators (bspinn.methods names, or pinn_NAME of --pinn-variant)")
    g.add_argument("--pinn-variant", nargs="+", action="append", default=[], metavar=("NAME", "KEY=VALUE"),
                   help="as evaluate_methods.py, e.g. --pinn-variant best solve_image=true")
    g.add_argument("--maps-cache", default=None, help="results folder of an evaluate_methods.py --joint run to reuse its maps "
                                                     "(<cache>/<case>_jR/maps/<name>.npy); default <out>/<case>/maps")
    g.add_argument("--pinn-quality", default="default", choices=["fast", "default"])
    g = ap.add_argument_group("network / training (defaults follow the ZS-SSL paper)")
    g.add_argument("--unrolls", type=int, default=10)
    g.add_argument("--cg-iters", type=int, default=10)
    g.add_argument("--blocks", type=int, default=15)
    g.add_argument("--ch", type=int, default=64)
    g.add_argument("--mu0", type=float, default=0.05)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--splits", type=int, default=50, help="K Theta/Lambda splits per slice (one epoch = K steps)")
    g.add_argument("--rho", type=float, default=0.4, help="|Lambda| / |Omega minus Gamma|")
    g.add_argument("--val-frac", type=float, default=0.2, help="|Gamma| / |Omega|")
    g.add_argument("--select", default="gaussian", choices=["gaussian", "uniform"])
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--patience", type=int, default=25)
    g.add_argument("--infer-mask", default="all", choices=["all", "train"],
                   help="data consistency at inference: all acquired samples (SSDU) or Omega minus Gamma (as trained)")
    g.add_argument("--init", default=None, help="start every slice from these weights (.pt), e.g. a pretrained network")
    g.add_argument("--amp", action="store_true", help="bfloat16 autocast for the denoiser (CUDA)")
    g.add_argument("--no-checkpoint", action="store_true", help="keep all unroll activations (faster, much more memory)")
    g = ap.add_argument_group("optional self-supervised pretraining on other slices (SSDU), then zero-shot fine-tuning")
    g.add_argument("--pretrain-n", type=int, default=0, help="number of OTHER dataset items for pretraining (0: pure zero-shot)")
    g.add_argument("--pretrain-seed", type=int, default=12345)
    g.add_argument("--pretrain-steps", type=int, default=20000)
    g.add_argument("--finetune-epochs", type=int, default=None, help="zero-shot epochs after pretraining (default --epochs; 0 = none)")
    g.add_argument("--device", default="cuda")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="runs/zs_ssl")
    a = ap.parse_args()

    import evaluate_methods as evm
    from functools import partial
    from method_lab import parse_set, undersampled
    from sense_lambda_eval import metrics
    from bspinn import evaluation as ev
    from bspinn import methods as bm
    from bspinn.api import acquired_extent, sense_image
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available: using the CPU (slow)"); a.device = "cpu"
    torch.manual_seed(a.seed)
    log = print
    for spec in a.pinn_variant:
        nm = f"pinn_{spec[0]}"
        bm.ESTIMATORS[nm] = partial(bm.pinn_api, pinn_options=parse_set(spec[1:]))
        bm._PASS[nm] = bm._PASS["pinn_api"]
    est_kw = {"threads": 1, "cfg_overrides": {}, "device": a.device, "quality": a.pinn_quality}
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        done = {(r["case"], r["maps"], r["method"]) for r in map(json.loads, (l for l in open(res_path) if l.strip()))}

    def cases(indices, n, seed):
        for c in a.cases:
            yield evm.CASES[c]
        if a.loader:
            yield from evm.loader_cases(a.loader, a.base_folder, indices, n, seed, a.key, a.coil_axis, a.acs, None, None)

    def prepare(make):
        case = make()
        if ev.object_mask(case).sum() < 0.005 * np.prod(case.shape):
            log(f"{case.name}: no object, skipped"); return None
        fit = undersampled(case, a.R)
        tag = f"{case.name}_j{a.R:g}"
        case.name = fit.name = tag
        return case, fit

    def get_maps(fit, name):
        cdir = os.path.join(a.maps_cache, fit.name, "maps") if a.maps_cache else os.path.join(a.out, fit.name, "maps")
        m_, _ = bm.estimate(fit, [name], cache_dir=cdir, log=log, **est_kw)
        if name not in m_:
            raise SystemExit(f"{fit.name}: estimator {name} returned no maps")
        return m_[name]

    # ---- optional pretraining (self-supervised on other slices; no fully sampled data used)
    init_state = torch.load(a.init, map_location=a.device) if a.init else None
    if a.pretrain_n:
        ckpt = os.path.join(a.out, f"pretrained_{a.maps[0]}.pt")
        if os.path.exists(ckpt):
            init_state = torch.load(ckpt, map_location=a.device); log(f"pretrained weights from {ckpt}")
        else:
            if not a.loader:
                raise SystemExit("--pretrain-n needs --loader")
            import importlib
            sys.path.insert(0, os.getcwd())
            mod, fn = a.loader.split(":")
            f_ = getattr(importlib.import_module(mod), fn)
            ds = f_(a.base_folder) if a.base_folder else f_()
            # the evaluation slices exactly as loader_cases picks them, never used for pretraining
            eval_idx = set(a.indices) if a.indices else set(
                np.random.default_rng(a.pick_seed).choice(len(ds), size=min(a.n, len(ds)), replace=False).tolist())
            pool = np.random.default_rng(a.pretrain_seed).permutation(len(ds))
            tr_idx = [int(i) for i in pool if int(i) not in eval_idx][: a.pretrain_n]
            del ds
            log(f"pretraining on {len(tr_idx)} slices (disjoint from the evaluation slices), maps {a.maps[0]}")
            train_set = []
            for make in evm.loader_cases(a.loader, a.base_folder, tr_idx, 0, 0, a.key, a.coil_axis, a.acs, None, None):
                p = prepare(make)
                if p is None:
                    continue
                case, fit = p
                C = get_maps(fit, a.maps[0])
                train_set.append(Slice(fit.ksp, C, fit.sampled, "cpu"))        # kept on the CPU, moved per step
            net = make_net(a, a.device)
            opt = torch.optim.Adam(net.parameters(), lr=a.lr)
            rng = np.random.default_rng(a.seed)
            t0, run = time.time(), []
            for step in range(a.pretrain_steps):
                sl = train_set[rng.integers(len(train_set))]
                sl_d = Slice.__new__(Slice)
                sl_d.C, sl_d.y, sl_d.omega, sl_d.scale, sl_d.device = sl.C.to(a.device), sl.y.to(a.device), sl.omega, sl.scale, a.device
                lam = select(sl.omega, a.rho, rng, mode=a.select)
                theta = sl.omega & ~lam
                net.train()
                out = net(sl_d.op(theta), sl_d.y * sl_d.mask_t(theta))
                loss = kspace_loss(sl_d.op(np.ones_like(theta)), out, sl_d.y, sl_d.mask_t(lam))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                run.append(float(loss.detach()))
                if (step + 1) % 200 == 0:
                    log(f"  pretrain step {step + 1}/{a.pretrain_steps}  loss {np.mean(run[-200:]):.4f}  "
                        f"{(time.time() - t0) / (step + 1):.2f}s/step", flush=True)
                if (step + 1) % 2000 == 0 or step + 1 == a.pretrain_steps:
                    torch.save(net.state_dict(), ckpt)
            init_state = net.state_dict()
            log(f"pretrained weights saved to {ckpt}")

    # ---- evaluation slices
    for make in cases(a.indices, a.n, a.pick_seed):
        p = prepare(make)
        if p is None:
            continue
        case, fit = p
        obj = ev.object_mask(case)
        ext = acquired_extent(case.sampled)
        rss = np.sqrt((np.abs(ev.ifft2c(case.ksp * case.sampled[None])) ** 2).sum(0))
        cdir = os.path.join(a.out, case.name)
        os.makedirs(cdir, exist_ok=True)
        for mname in a.maps:
            if all((case.name, mname, m) in done for m in (["cg_auto", "zs"] + (["tl0"] if init_state is not None else []))):
                log(f"{case.name} / {mname}: done"); continue
            C = get_maps(fit, mname)
            Cn = ev.normalise(C)
            imgs, recs = {}, []

            def record(method, x, extra=None):
                xb = band_limit(x, ext)
                imgs[method] = xb.astype(np.complex64)
                q = metrics(xb, rss, obj)
                rec = {"case": case.name, "maps": mname, "method": method, **q, **(extra or {})}
                recs.append(rec)
                log(f"{case.name} {mname:10s} {method:8s} NRMSE {q['nrmse']:.4f}  SSIM {q['ssim']:.4f}  PSNR {q['psnr']:.2f}", flush=True)

            t0 = time.time()
            rep = {}
            x_cg = sense_image(fit.ksp, Cn, "2d", mask=fit.sampled, lam="auto", device=a.device, report=rep)
            record("cg_auto", x_cg, {"lambda": rep.get("value"), "seconds": time.time() - t0})
            sl = Slice(fit.ksp, Cn, fit.sampled, a.device)
            net = make_net(a, a.device)
            if init_state is not None:
                net.load_state_dict(init_state)
                record("tl0", reconstruct(net, sl, sl.omega))
            rng = np.random.default_rng(a.seed + 1)
            epochs = a.epochs if (init_state is None or a.finetune_epochs is None) else a.finetune_epochs
            t0 = time.time()
            info = {"epochs": 0, "train_mask": sl.omega}
            if epochs > 0:
                a_ = argparse.Namespace(**{**vars(a), "epochs": epochs})
                info = train_zero_shot(net, sl, a_, rng, log, tag=f"{case.name}/{mname}")
            secs = time.time() - t0
            infer = sl.omega if a.infer_mask == "all" else info["train_mask"]
            record("zs", reconstruct(net, sl, infer),
                   {"seconds": secs, "epochs": info.get("epochs"), "best_epoch": info.get("best_epoch"),
                    "best_val": info.get("best_val"), "mu": float(net.mu), "init": "pretrained" if init_state is not None else "random"})
            np.savez_compressed(os.path.join(cdir, f"{mname}.npz"), rss=rss.astype(np.float32), obj=obj, **imgs)
            if info.get("history"):
                json.dump(info["history"], open(os.path.join(cdir, f"{mname}_history.json"), "w"))
            with open(res_path, "a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
    # ---- summary
    rows = [json.loads(l) for l in open(res_path) if l.strip()] if os.path.exists(res_path) else []
    if not rows:
        return
    print(f"\nmedian over slices (NRMSE / SSIM / PSNR vs fully sampled RSS) and the fraction of slices where "
          f"ZS-SSL beats CG-SENSE with the same maps:")
    for mname in dict.fromkeys(r["maps"] for r in rows):
        for meth in dict.fromkeys(r["method"] for r in rows):
            rs = [r for r in rows if r["maps"] == mname and r["method"] == meth]
            if not rs:
                continue
            by = {r["case"]: r for r in rows if r["maps"] == mname and r["method"] == "cg_auto"}
            win = [r["nrmse"] < by[r["case"]]["nrmse"] for r in rs if r["case"] in by]
            print(f"  {mname:12s} {meth:8s} n={len(rs):3d}  {np.median([r['nrmse'] for r in rs]):.4f}  "
                  f"{np.median([r['ssim'] for r in rs]):.4f}  {np.median([r['psnr'] for r in rs]):6.2f}  "
                  + (f"wins vs cg_auto {np.mean(win):.2f}" if meth != "cg_auto" else ""))


if __name__ == "__main__":
    main()
