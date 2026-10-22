"""Prospectively undersampled OCMR cine (us_*.h5): no fully sampled reference, so methods are compared by
held-out k-space prediction.

    BART=bart/bart python scripts/ocmr_prospective.py --files /data/ocmr/us_0001_3T.h5 \\
        --maps espirit nlinv pinn_avg pinn_dyn --device cuda --out runs/ocmr_us/results.jsonl

Per file (middle slice):
  * per-sample acquisition mask: the phase-encode lines acquired in each frame x the acquired readout range
    (asymmetric echo: the empty start of the readout only holds leakage after the oversampling removal, so it is
    zeroed and masked);
  * pre-whitening with the noise-scan covariance; the scalar level re-estimated from the outer k-space corners
    of the acquired samples (an upper bound: residual signal there only raises it);
  * held-out set: --holdout lines per frame (random, fixed seed), withheld from EVERYTHING (maps and images);
  * maps from the training samples: espirit / nlinv (BART, time-averaged k-space), pinn_avg (2-D on the time
    average), pinn_dyn / pinn_dyn0 (layout 'dynamic', per-frame maps with --rank temporal modes / rank 0);
  * images: 'sense' per-frame CG-SENSE (Tikhonov weight automatic on the middle frame) and 'tv' spatio-temporal
    TV (bspinn.prior.dynamic_prior_image, weight by k-space cross-validation within the training samples);
  * score: the held-out samples predicted as F(C_t x_t): mean |residual|^2 per sample over the noise level
    (whitened: 1 = noise only) -- overall, central lines (|ky| < --centre) and outer lines -- and the same on the
    training samples (data fit). The held-out error rewards maps + image that explain unseen data; it is
    the same test for every map set.
One JSON line per (file, maps, recon) in --out (resume-safe); maps cached in <out dir>/maps; --save-images
writes the magnitude series."""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

from ocmr_eval import central_lines, fft2c, ifft2c  # noqa: E402


def acquisition_mask(k, kx_range):
    """k (N,X,Y,T) -> per-sample mask (X,Y,T): lines acquired per frame x the acquired readout range."""
    lines = np.abs(k).sum((0, 1)) > 0                                     # (Y,T)
    kx = np.zeros(k.shape[1], bool)
    kx[kx_range[0]:kx_range[1] + 1] = True
    return kx[:, None, None] & lines[None]


def whiten_scan(k, mask, noise, corner=0.15):
    """Noise-scan covariance (shape), level from the outer k-space corners of the acquired samples."""
    N = k.shape[0]
    if noise is not None:
        n = noise.reshape(-1, N)
        psi = (n.T @ n.conj()) / n.shape[0]
        src = "noise scan covariance, level from k-space corners"
    else:
        psi = np.eye(N, dtype=complex)
        src = "identity covariance, level from k-space corners"
    L = np.linalg.cholesky(psi + 1e-9 * np.trace(psi).real / N * np.eye(N))
    kw = np.linalg.solve(L, k.reshape(N, -1)).reshape(k.shape)
    X, Y = k.shape[1:3]
    fx = np.abs(np.arange(X) - X // 2) / (X / 2); fy = np.abs(np.arange(Y) - Y // 2) / (Y / 2)
    corn = (fx[:, None] > 1 - 2 * corner) & (fy[None, :] > 1 - 2 * corner)  # (X,Y)
    sel = mask & corn[..., None]
    lev = float(np.mean(np.abs(kw[:, sel]) ** 2))
    kw = kw / np.sqrt(lev)
    return (kw * mask[None]).astype(np.complex64), {"noise_source": src, "corner_samples": int(sel.sum())}


def heldout_split(mask, n_hold, seed=0):
    """Withhold n_hold of the acquired lines per frame: (train mask, held-out mask), (X,Y,T)."""
    rng = np.random.default_rng(seed)
    lines = mask.any(0)                                                   # (Y,T)
    hold = np.zeros_like(lines)
    for t in range(lines.shape[1]):
        acq = np.where(lines[:, t])[0]
        hold[rng.choice(acq, min(n_hold, max(acq.size - 1, 0)), replace=False), t] = True
    ho = mask & hold[None]
    return mask & ~ho, ho


def prediction_error(k, Cn, x, sel, centre):
    """Mean |F(C_t x_t) - k|^2 over the samples in sel (X,Y,T): overall, central, outer lines."""
    T = x.shape[-1]
    Y = k.shape[2]
    cen = (np.abs(np.arange(Y) - Y // 2) < centre)[None, :]
    num = {"all": 0.0, "centre": 0.0, "outer": 0.0}
    cnt = {"all": 0, "centre": 0, "outer": 0}
    for t in range(T):
        C = Cn[..., t] if Cn.ndim == 4 else Cn
        r = np.abs(fft2c(C * x[None, ..., t]) - k[..., t]) ** 2           # (N,X,Y)
        for key, reg in (("all", sel[..., t]), ("centre", sel[..., t] & cen), ("outer", sel[..., t] & ~cen)):
            num[key] += float(r[:, reg].sum()); cnt[key] += int(reg.sum()) * k.shape[0]
    return {key: num[key] / max(cnt[key], 1) for key in num}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--maps", nargs="+", default=["espirit", "nlinv", "pinn_avg", "pinn_dyn"])
    ap.add_argument("--recon", nargs="+", default=["sense", "tv"])
    ap.add_argument("--holdout", type=int, default=2, help="held-out lines per frame")
    ap.add_argument("--centre", type=int, default=12, help="|ky| below this counts as central")
    ap.add_argument("--calib", type=int, default=24)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--quality", default="default", choices=["fast", "default"])
    ap.add_argument("--dyn-set", nargs="*", default=[], help="pinn_dyn: bspinn.dynamic.DynConfig fields key=value "
                    "(e.g. lambda_corr=0.01 freeze_static=False alpha_init=data)")
    ap.add_argument("--pinn-set", nargs="*", default=[])
    ap.add_argument("--plane", default="auto")
    ap.add_argument("--spatial-on", default="image", choices=["image", "difference"],
                    help="tv: spatial TV of the frames or of their difference from the time average (v57)")
    ap.add_argument("--spatial-ratio", type=float, default=0.25)
    ap.add_argument("--tv-iters", type=int, default=50)
    ap.add_argument("--tv-grid", type=float, nargs=3, default=None, metavar=("LOG10_MIN", "LOG10_MAX", "N"))
    ap.add_argument("--tv-folds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--out", default="runs/ocmr_us/results.jsonl")
    a = ap.parse_args()

    import torch
    from method_lab import parse_set
    from bspinn import methods as bm
    from bspinn.api import estimate_maps, sense_image
    from bspinn.methods import EvalCase
    from bspinn.ocmr import cine_slices
    from bspinn.prior import dynamic_prior_image
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available: using the CPU (slow)"); a.device = "cpu"
    pset = parse_set(a.pinn_set)
    dset = parse_set(a.dyn_set)
    root = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(root, exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {(r["file"], r["maps"], r["recon"]) for r in map(json.loads, (l for l in open(a.out) if l.strip()))}
    for f in a.files:
        fname = os.path.basename(f).replace(".h5", "")
        (s, k_raw, info), = list(cine_slices(f, "mid"))                  # list(): releases the raw array
        if a.max_frames:
            k_raw = k_raw[..., :a.max_frames]
        N, X, Y, T = k_raw.shape
        acq = acquisition_mask(k_raw, info["kx_range"])
        k, winfo = whiten_scan(k_raw, acq, info["noise"])
        del k_raw
        tr, ho = heldout_split(acq, a.holdout, a.seed)
        ktr = (k * tr[None]).astype(np.complex64)
        # memory-mapped copies (long series on small machines): pages are read when used, so the map fits
        # below get the RAM; deleted at the end
        cache = os.path.join(root, f"_{fname}_cache")
        os.makedirs(cache, exist_ok=True)
        np.save(os.path.join(cache, "k.npy"), k); np.save(os.path.join(cache, "ktr.npy"), ktr)
        del k, ktr
        k = np.load(os.path.join(cache, "k.npy"), mmap_mode="r")
        ktr = np.load(os.path.join(cache, "ktr.npy"), mmap_mode="r")
        sp = tuple(float(v) for v in info["pixel_spacing_mm"])
        cnt = tr.sum(-1); mavg = cnt > 0
        kavg = (ktr.sum(-1) / np.maximum(cnt, 1)[None]).astype(np.complex64)
        cal = central_lines(mavg, a.calib)
        lpf = acq.any(0).sum(0)
        print(f"== {fname} slice {s}: {N} coils, {X}x{Y}, {T} frames, {lpf.min()}-{lpf.max()} of {Y} lines per frame "
              f"(R {Y / lpf.mean():.1f}), readout {info['kx_range'][0]}..{info['kx_range'][1]} of {X}, "
              f"{a.holdout} held out per frame, time average covers {mavg.any(0).sum()} lines, calibration {cal}; "
              f"{winfo['noise_source']}", flush=True)
        imgs = {}
        for mname in a.maps:
            if all((fname, mname, rc) in done for rc in a.recon):
                continue
            cp = os.path.join(root, "maps", f"{fname}_{mname}.npy")
            t0 = time.time()
            if os.path.exists(cp):
                C = np.load(cp)
            else:
                case = EvalCase(fname, kavg, mavg, (cal, cal), pixel_spacing=sp, pe_axis=1, joint=True)
                common = dict(calib=cal, spacing=sp, device=a.device, quality=a.quality, verbose=False,
                              plane=a.plane, **pset)
                if mname == "espirit":
                    import dataclasses
                    C = bm.espirit(dataclasses.replace(case, joint=False))
                elif mname == "nlinv":
                    C = bm.nlinv(case)
                elif mname == "pinn_avg":
                    C = estimate_maps(kavg * mavg[None], "2d", mask=mavg, readout_axis=0, **common)[0]
                elif mname in ("pinn_dyn", "pinn_dyn0"):
                    C = estimate_maps(ktr, "dynamic", mask=tr, rank=0 if mname == "pinn_dyn0" else a.rank,
                                      dynamic_options=dset or None, **common)[0]
                    if mname == "pinn_dyn0":
                        C = C[..., 0]
                else:
                    raise SystemExit(f"unknown maps {mname}")
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                np.save(cp, np.asarray(C).astype(np.complex64))
            t_maps = time.time() - t0
            C = np.asarray(C)
            Cn = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)
            Ct = (lambda t: Cn[..., t]) if Cn.ndim == 4 else (lambda t: Cn)
            for rc in a.recon:
                if (fname, mname, rc) in done:
                    continue
                t0 = time.time(); extra = {}
                if rc == "sense":
                    rep = {}
                    tm = T // 2
                    sense_image(ktr[..., tm], Ct(tm), "2d", mask=tr[..., tm], lam="auto", device=a.device, report=rep)
                    lam = rep.get("value") or 1e-2
                    x = np.stack([sense_image(ktr[..., t], Ct(t), "2d", mask=tr[..., t], lam=lam, device=a.device)
                                  for t in range(T)], -1)
                    extra["lambda"] = float(lam)
                elif rc == "tv":
                    rep = {}
                    grid = None if a.tv_grid is None else np.logspace(a.tv_grid[0], a.tv_grid[1], int(a.tv_grid[2]))
                    x = dynamic_prior_image(ktr, Cn, tr, "auto", spatial_ratio=a.spatial_ratio, spatial_on=a.spatial_on, device=a.device,
                                            report=rep, iters=a.tv_iters, grid=grid, folds=a.tv_folds)
                    extra.update({kk: rep.get(kk) for kk in ("weight", "rule", "grid", "cv_error")})
                else:
                    raise SystemExit(f"unknown recon {rc}")
                e_ho = prediction_error(k, Cn, x, ho, a.centre)
                e_tr = prediction_error(k, Cn, x, tr, a.centre)
                rec = {"file": fname, "slice": s, "maps": mname, "recon": rc, "frames": T, "coils": N,
                       "maps_seconds": t_maps, "recon_seconds": time.time() - t0, "calib": cal,
                       "heldout": e_ho["all"], "heldout_centre": e_ho["centre"], "heldout_outer": e_ho["outer"],
                       "train": e_tr["all"], **extra}
                with open(a.out, "a") as fh:                              # per row: survives interruptions
                    fh.write(json.dumps(rec) + "\n")
                imgs[f"{mname}/{rc}"] = np.abs(x).astype(np.float16)
                print(f"   {mname:10s} {rc:6s} held-out/noise {e_ho['all']:8.3f}  centre {e_ho['centre']:8.3f}  "
                      f"outer {e_ho['outer']:7.3f}   train/noise {e_tr['all']:.3f}"
                      + (f"  weight {extra['weight']:.2e} ({extra['rule']})" if extra.get("weight") else ""), flush=True)
            if a.save_images and imgs:
                p = os.path.join(root, f"{fname}_images.npz")
                old = dict(np.load(p)) if os.path.exists(p) else {}
                np.savez_compressed(p, **{**old, **imgs})
                imgs = {}
        del k, ktr
        import shutil
        shutil.rmtree(cache, ignore_errors=True)
    rows = [json.loads(l) for l in open(a.out) if l.strip()] if os.path.exists(a.out) else []
    if rows:
        print("\nheld-out k-space prediction error / noise level (lower is better; 1 = noise only):")
        for r in rows:
            print(f"  {r['file']}  {r['maps']:10s} {r['recon']:6s} {r['heldout']:8.3f}  centre {r['heldout_centre']:8.3f}  "
                  f"outer {r['heldout_outer']:7.3f}  train {r['train']:.3f}")


if __name__ == "__main__":
    main()
