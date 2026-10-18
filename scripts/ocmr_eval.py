"""Dynamic (cine) evaluation on OCMR fully sampled 2-D cardiac data: coil maps from the time-averaged
k-space (ESPIRiT, NLINV, the PINN) vs per-dynamic PINN maps, without and with GRO undersampling.

    BART=bart/bart python scripts/ocmr_eval.py --files /data/ocmr/fs_*.h5 --slices mid --R 1 4 8 \\
        --maps espirit nlinv pinn_avg pinn_dyn --device cuda --oracle --out runs/ocmr/results.jsonl

Per file and slice (bspinn.ocmr.cine_slices: readout oversampling removed, averages summed):
  * pre-whitening with the noise scans (coil covariance; the scalar level re-estimated from the data, since
    noise scans can have a different bandwidth), or from the data alone (--noise data);
  * for every R: a GRO mask per frame (bspinn.masks.gro_mask, R = 1: fully sampled; inside the acquired
    phase-encode extent) and the time-averaged k-space (sum over frames / number of frames sampling it);
  * maps:  espirit / nlinv   BART ecalib (ACS box of --calib lines, the time average covers it) / ncalib on
                             the time-averaged k-space (one map set for all frames)
           pinn_avg          estimate_maps 2-D on the time-averaged k-space (all its samples)
           pinn_dyn          estimate_maps layout='dynamic' (per-frame maps, --rank temporal modes of the coil
                             array) on the undersampled frames
           pinn_dyn0         the same with rank 0: one static map set fitted to all frames jointly
           pinn_frame        independent 2-D fits per frame (per-dynamic maps without temporal sharing; slow)
  * reconstruction: R = 1: SENSE-1 per frame (closed form); R > 1: 'sense' per-frame CG-SENSE (automatic
    Tikhonov weight from the middle frame, used for all frames) and 'tv' spatio-temporal TV
    (bspinn.prior.dynamic_prior_image, weight by k-space cross-validation; --oracle also scores every grid
    weight against the reference);
  * metrics against the fully sampled per-frame RSS inside the object (time-averaged RSS mask): NRMSE / SSIM /
    PSNR per frame (one LS scale per frame), their mean; dyn_nrmse = NRMSE of the dynamic part (frame minus
    the time average, one scale for the series): temporal fidelity, what temporal blurring damages. R = 1 also
    reports reference-free map metrics per frame (combine->expand residual / noise floor inside and outside
    the ACS, leave-coil-out / floor; every --rf-every-th frame).
One JSON line per (file, slice, R, maps, recon) in --out (resume-safe); maps cached in <out dir>/maps;
--save-images writes magnitude series per (file, slice, R). Summary at the end.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

STATIC = ("espirit", "nlinv", "pinn_avg", "pinn_dyn0")


def fft2c(x, ax=(-2, -1)):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


def ifft2c(x, ax=(-2, -1)):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho"), axes=ax)


def whiten(k, noise, mode):
    """k (N,X,Y,T) -> whitened (per-sample noise variance ~1), info."""
    import evaluate_methods as evm
    N = k.shape[0]
    sampled = np.abs(k).sum((0, 3)) > 0
    mid = k[..., k.shape[-1] // 2]
    if noise is not None and mode == "scan":
        n = noise.reshape(-1, N)
        psi = (n.T @ n.conj()) / n.shape[0]
        src = "noise scan (level re-estimated from the data)"
    else:
        psi, _ = evm.estimate_noise(mid, sampled)
        src = "data"
    L = np.linalg.cholesky(psi + 1e-9 * np.trace(psi).real / N * np.eye(N))
    kw = np.linalg.solve(L, k.reshape(N, -1)).reshape(k.shape)
    if src.startswith("noise scan"):
        p2, _ = evm.estimate_noise(kw[..., k.shape[-1] // 2], sampled)
        kw = kw / np.sqrt(np.trace(p2).real / N)
    return (kw * sampled[None, :, :, None]).astype(np.complex64), {"noise_source": src}


def central_lines(mavg, calib, pe_axis=1):
    """Largest centred block <= calib of phase-encode lines all sampled at least once (time average)."""
    prof = mavg.any(axis=1 - pe_axis)
    n = prof.size
    c = calib
    while c > 4 and not prof[n // 2 - c // 2: n // 2 - c // 2 + c].all():
        c -= 2
    return c


def series_metrics(x, ref, obj):
    """x, ref (X,Y,T): per-frame NRMSE/SSIM/PSNR (sense_lambda_eval.metrics), means, and dyn_nrmse."""
    from sense_lambda_eval import metrics
    per = [metrics(x[..., t], ref[..., t], obj) for t in range(x.shape[-1])]
    a, b = np.abs(x), np.abs(ref)
    s = (a[obj] * b[obj]).sum() / max((a[obj] ** 2).sum(), 1e-30)
    da = a * s - (a * s).mean(-1, keepdims=True); db = b - b.mean(-1, keepdims=True)
    dyn = float(np.linalg.norm(da[obj] - db[obj]) / max(np.linalg.norm(db[obj]), 1e-30))
    out = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    out.update(dyn_nrmse=dyn, per_frame_nrmse=[p["nrmse"] for p in per], per_frame_ssim=[p["ssim"] for p in per])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True, help="OCMR fully sampled .h5 files (globs ok)")
    ap.add_argument("--slices", default="mid", help="mid | all | comma-separated indices")
    ap.add_argument("--max-frames", type=int, default=0, help="use only the first n frames (0: all)")
    ap.add_argument("--R", type=float, nargs="+", default=[1, 4, 8])
    ap.add_argument("--maps", nargs="+", default=["espirit", "nlinv", "pinn_avg", "pinn_dyn"])
    ap.add_argument("--recon", nargs="+", default=["sense", "tv"], help="R > 1: sense | tv")
    ap.add_argument("--calib", type=int, default=24)
    ap.add_argument("--rank", type=int, default=2, help="pinn_dyn temporal modes")
    ap.add_argument("--noise", default="scan", choices=["scan", "data"])
    ap.add_argument("--quality", default="default", choices=["fast", "default"])
    ap.add_argument("--pinn-set", nargs="*", default=[], help="extra estimate_maps / FitConfig options key=value")
    ap.add_argument("--plane", default="auto")
    ap.add_argument("--spatial-ratio", type=float, default=0.25, help="tv: spatial / temporal TV weight")
    ap.add_argument("--tv-iters", type=int, default=50)
    ap.add_argument("--oracle", action="store_true", help="tv: also score every grid weight against the reference")
    ap.add_argument("--rf-every", type=int, default=4, help="R = 1: reference-free map metrics every n-th frame")
    ap.add_argument("--gro-s", type=float, default=2.2)
    ap.add_argument("--gro-alpha", type=float, default=3.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--out", default="runs/ocmr/results.jsonl")
    a = ap.parse_args()

    import torch
    from method_lab import parse_set
    from zs_ssl import band_limit
    from bspinn import evaluation as ev, methods as bm, refree
    from bspinn.api import acquired_extent, estimate_maps, sense_image
    from bspinn.masks import gro_mask
    from bspinn.methods import EvalCase
    from bspinn.ocmr import cine_slices
    from bspinn.prior import dynamic_prior_image
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available: using the CPU (slow)"); a.device = "cpu"
    files = sorted({f for p in a.files for f in (glob.glob(p) or [p])})
    pset = parse_set(a.pinn_set)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    root = os.path.dirname(os.path.abspath(a.out))
    done = set()
    if os.path.exists(a.out):
        done = {(r["file"], r["slice"], r["R"], r["maps"], r["recon"]) for r in map(json.loads, (l for l in open(a.out) if l.strip()))}
    sl_arg = a.slices if a.slices in ("mid", "all") else [int(s) for s in a.slices.split(",")]
    for f in files:
        fname = os.path.basename(f).replace(".h5", "")
        print(f"== {fname}", flush=True)
        for s, k_raw, info in cine_slices(f, sl_arg):
            if a.max_frames:
                k_raw = k_raw[..., :a.max_frames]
            N, X, Y, T = k_raw.shape
            k, winfo = whiten(k_raw, info["noise"], a.noise)
            sampled = np.abs(k).sum((0, 3)) > 0
            ext = acquired_extent(sampled)
            sp = tuple(float(v) for v in info["pixel_spacing_mm"])
            ref = np.sqrt((np.abs(ifft2c(k, (1, 2))) ** 2).sum(0))                         # (X,Y,T) RSS per frame
            avg_case = EvalCase(f"{fname}_s{s}", k.mean(-1), sampled, (a.calib, a.calib), sigma=1.0 / np.sqrt(T),
                                pe_axis=1)
            obj = ev.object_mask(avg_case)
            print(f"   slice {s}: {N} coils, {X}x{Y}, {T} frames, spacing {sp[0]:.2f}x{sp[1]:.2f} mm, "
                  f"noise from {winfo['noise_source']}, object {obj.mean():.2f} of the FOV", flush=True)
            imgs = {"ref": np.abs(ref).astype(np.float16)}
            for R in a.R:
                m = gro_mask((X, Y), T, R, pe_axis=1, extent=sampled, s=a.gro_s, alpha=a.gro_alpha)  # (X,Y,T)
                ku = (k * m[None]).astype(np.complex64)
                cnt = m.sum(-1)
                mavg = cnt > 0
                kavg = (ku.sum(-1) / np.maximum(cnt, 1)[None]).astype(np.complex64)
                cal = central_lines(mavg, a.calib)
                tag = f"{fname}_s{s}_R{R:g}"
                print(f"   R={R:g}: {m[..., 0].any(0).sum()} of {sampled.any(0).sum()} lines per frame, "
                      f"time average covers {mavg.any(0).sum()} lines, calibration {cal}", flush=True)
                for mname in a.maps:
                    recons = ["sense1"] if R <= 1 else list(a.recon)
                    if all((fname, s, R, mname, rc) in done for rc in recons):
                        continue
                    cp = os.path.join(root, "maps", f"{tag}_{mname}.npy")
                    t0 = time.time()
                    try:
                        if os.path.exists(cp):
                            C = np.load(cp)
                        else:
                            case = EvalCase(tag, kavg, mavg, (cal, cal), pixel_spacing=sp, pe_axis=1, joint=True)
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
                                C = estimate_maps(ku, "dynamic", mask=m, rank=0 if mname == "pinn_dyn0" else a.rank,
                                                  **common)[0]                                     # (N,X,Y,T)
                                if mname == "pinn_dyn0":
                                    C = C[..., 0]
                            elif mname == "pinn_frame":
                                C = np.stack([estimate_maps(ku[..., t], "2d", mask=m[..., t], readout_axis=0, **common)[0]
                                              for t in range(T)], -1)
                            else:
                                raise SystemExit(f"unknown maps {mname}")
                            os.makedirs(os.path.dirname(cp), exist_ok=True)
                            np.save(cp, np.asarray(C).astype(np.complex64))
                    except (FileNotFoundError, OSError) as e:
                        print(f"   {mname}: skipped ({type(e).__name__}: {e})", flush=True); continue
                    t_maps = time.time() - t0
                    C = np.asarray(C)
                    Cn = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)
                    Ct = (lambda t: Cn[..., t]) if Cn.ndim == 4 else (lambda t: Cn)
                    recs = []
                    for rc in recons:
                        if (fname, s, R, mname, rc) in done:
                            continue
                        t0 = time.time(); extra = {}
                        if rc == "sense1":
                            x = np.stack([(Ct(t).conj() * ifft2c(k[..., t])).sum(0) for t in range(T)], -1)
                            rf = []
                            for t in range(0, T, max(1, a.rf_every)):
                                fc = EvalCase(f"{tag}_t{t}", k[..., t], sampled, (a.calib, a.calib), sigma=1.0, pe_axis=1)
                                kc = ev.kspace_consistency(Ct(t), fc)
                                lco = refree.leave_coil_out(Ct(t), fc, obj)
                                rf.append((kc["acs"]["residual_over_noise_floor"], kc["outside_acs"]["residual_over_noise_floor"],
                                           lco["outside_acs"]["residual_over_noise_floor"]))
                            rf = np.array(rf)
                            extra.update(ce_acs_over_floor=float(rf[:, 0].mean()), ce_outside_over_floor=float(rf[:, 1].mean()),
                                         lco_outside_over_floor=float(rf[:, 2].mean()))
                        elif rc == "sense":
                            rep = {}
                            tm = T // 2
                            sense_image(ku[..., tm], Ct(tm), "2d", mask=m[..., tm], lam="auto", device=a.device, report=rep)
                            lam = rep.get("value") or 1e-2
                            x = np.stack([sense_image(ku[..., t], Ct(t), "2d", mask=m[..., t], lam=lam, device=a.device)
                                          for t in range(T)], -1)
                            extra["lambda"] = float(lam)
                        elif rc == "tv":
                            rep = {}
                            ofn = None
                            if a.oracle:
                                ofn = lambda xx: series_metrics(np.stack([band_limit(xx[..., t], ext) for t in range(T)], -1), ref, obj)["nrmse"]  # noqa: E731
                            x = dynamic_prior_image(ku, Cn, m, "auto", spatial_ratio=a.spatial_ratio, device=a.device,
                                                    report=rep, iters=a.tv_iters, oracle_fn=ofn)
                            extra.update({kk: rep.get(kk) for kk in ("weight", "rule", "grid", "cv_error", "grid_error")})
                        else:
                            raise SystemExit(f"unknown recon {rc}")
                        x = np.stack([band_limit(x[..., t], ext) for t in range(T)], -1)
                        q = series_metrics(x, ref, obj)
                        rec = {"file": fname, "slice": s, "R": R, "maps": mname, "recon": rc, "frames": T, "coils": N,
                               "maps_seconds": t_maps, "recon_seconds": time.time() - t0, "calib": cal, **q, **extra}
                        recs.append(rec)
                        imgs[f"R{R:g}/{mname}/{rc}"] = np.abs(x).astype(np.float16)
                        print(f"   R={R:<3g} {mname:10s} {rc:6s} NRMSE {q['nrmse']:.4f}  SSIM {q['ssim']:.4f}  "
                              f"PSNR {q['psnr']:.2f}  dyn {q['dyn_nrmse']:.4f}"
                              + (f"  lco {extra['lco_outside_over_floor']:.3f}" if "lco_outside_over_floor" in extra else "")
                              + (f"  weight {extra['weight']:.2e} ({extra['rule']})" if extra.get("weight") else ""), flush=True)
                    with open(a.out, "a") as fh:
                        for r in recs:
                            fh.write(json.dumps(r) + "\n")
            if a.save_images:
                np.savez_compressed(os.path.join(root, f"{fname}_s{s}_images.npz"), **imgs)
    rows = [json.loads(l) for l in open(a.out) if l.strip()] if os.path.exists(a.out) else []
    if not rows:
        return
    print("\nmedian over slices of the frame-mean NRMSE / SSIM / PSNR, dyn_nrmse; R = 1 also leave-coil-out / floor; "
          "wins = fraction of slices with lower NRMSE than ESPIRiT maps (same R and recon):")
    for R in sorted({r["R"] for r in rows}):
        for rc in dict.fromkeys(r["recon"] for r in rows if r["R"] == R):
            base = {(r["file"], r["slice"]): r for r in rows if r["R"] == R and r["recon"] == rc and r["maps"] == "espirit"}
            for mname in dict.fromkeys(r["maps"] for r in rows):
                rs = [r for r in rows if r["R"] == R and r["recon"] == rc and r["maps"] == mname]
                if not rs:
                    continue
                win = [r["nrmse"] < base[(r["file"], r["slice"])]["nrmse"] for r in rs if (r["file"], r["slice"]) in base]
                line = (f"  R={R:<3g} {rc:6s} {mname:10s} n={len(rs):3d}  {np.median([r['nrmse'] for r in rs]):.4f}  "
                        f"{np.median([r['ssim'] for r in rs]):.4f}  {np.median([r['psnr'] for r in rs]):6.2f}  "
                        f"dyn {np.median([r['dyn_nrmse'] for r in rs]):.4f}")
                if "lco_outside_over_floor" in rs[0]:
                    line += f"  lco {np.median([r['lco_outside_over_floor'] for r in rs]):.3f}"
                if win and mname != "espirit":
                    line += f"  wins {np.mean(win):.2f}"
                print(line)


if __name__ == "__main__":
    main()
