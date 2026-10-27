"""Coil maps of an OCMR cine with every method, time-averaged and per frame, saved as .npy.

    uv run python bspinn/scripts/ocmr_maps.py data/ocmr/fs_0021_3T.h5 --R 4 --device cuda:0 --out runs/ocmr_maps

Per (file, slice) it writes to --out:

  <file>_s<slice>_R<R>_<method>.npy   maps, complex64
      espirit_avg, nlinv_avg, pinn_avg   (coils, x, y)          from the time-averaged k-space
      espirit_frame, nlinv_frame         (coils, x, y, frames)  each frame on its own
      pinn_dyn                           (coils, x, y, frames)  low-rank dynamic Biot-Savart model (--rank)
  <file>_s<slice>_R<R>_<method>_<recon>.npy   coil-combined cine with those maps, complex64 (x, y, frames);
      --recon tv (default: the benchmark's temporal TV) and/or sense (CG-SENSE per frame)
  <file>_s<slice>_R<R>_data.npz       k (whitened, undersampled), mask, the fully sampled reference RSS per frame

The steps are those of ocmr_eval.py (the benchmark): noise-scan prewhitening, GRO undersampling (R = 1: fully
sampled), a calibration block of up to 24 central lines that are acquired in the time average, B0 from the header.
All maps are for the whitened channels and RSS-normalised.

Per-frame ESPIRiT needs a fully sampled calibration block in every frame: with GRO undersampling the frames do not
have one (only the time average does), so espirit_frame is only computed for R = 1 or when the frame has the lines
(else it is skipped with a message). Per-frame NLINV works on the undersampled frames (all acquired samples).
"""
import argparse
import dataclasses
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def central_lines(mavg, calib, pe_axis=1):
    """Largest centred block <= calib of phase-encode lines that are all acquired."""
    prof = mavg.any(axis=1 - pe_axis)
    n, c = prof.size, calib
    while c > 4 and not prof[n // 2 - c // 2: n // 2 - c // 2 + c].all():
        c -= 2
    return c if prof[n // 2 - c // 2: n // 2 - c // 2 + c].all() else 0


def rss_normalise(C):
    return (C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--slices", default="mid", help="'mid', 'all' or e.g. 0,2")
    ap.add_argument("--R", type=float, default=4, help="GRO acceleration; 1 = fully sampled")
    ap.add_argument("--calib", type=int, default=24)
    ap.add_argument("--rank", type=int, default=2, help="pinn_dyn temporal modes")
    ap.add_argument("--methods", nargs="+",
                    default=["espirit_avg", "nlinv_avg", "pinn_avg", "espirit_frame", "nlinv_frame", "pinn_dyn"])
    ap.add_argument("--recon", nargs="*", default=["tv"], choices=["tv", "sense"],
                    help="coil-combined cine per map set (none: maps only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/ocmr_maps")
    a = ap.parse_args()
    if not os.environ.get("BART") and os.path.exists("bart/bart"):
        os.environ["BART"] = os.path.abspath("bart/bart")

    from bspinn import methods as bm
    from bspinn.api import acquired_extent, estimate_maps, sense_image
    from bspinn.prior import dynamic_prior_image
    from zs_ssl import band_limit
    from bspinn.masks import gro_mask
    from bspinn.methods import EvalCase
    from bspinn.ocmr import cine_slices
    from ocmr_eval import whiten
    os.makedirs(a.out, exist_ok=True)
    sl = a.slices if a.slices in ("mid", "all") else [int(s) for s in a.slices.split(",")]
    for f in a.files:
        fname = os.path.basename(f).replace(".h5", "")
        for s, k_raw, info in cine_slices(f, sl):
            k, _ = whiten(k_raw, info["noise"], "scan")        # the benchmark's prewhitening (ocmr_eval.py)
            N, X, Y, T = k.shape
            sampled = np.abs(k).sum((0, 3)) > 0
            ref = np.sqrt((np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(1, 2)), axes=(1, 2),
                                                                norm="ortho"), axes=(1, 2))) ** 2).sum(0))
            if a.R > 1:
                m = gro_mask((X, Y), T, a.R, pe_axis=1, extent=sampled, s=2.2, alpha=3.0)
            else:
                m = np.repeat(sampled[..., None], T, -1)
            ku = (k * m[None]).astype(np.complex64)
            cnt = m.sum(-1)
            mavg = cnt > 0
            kavg = (ku.sum(-1) / np.maximum(cnt, 1)[None]).astype(np.complex64)
            cal = central_lines(mavg, a.calib)
            sp = tuple(float(v) for v in info["pixel_spacing_mm"])
            plane = info.get("b0_dir") or "auto"
            tag = f"{fname}_s{s}_R{a.R:g}"
            print(f"== {tag}: {N} coils, {X}x{Y}, {T} frames, calibration {cal} lines, B0 {plane}", flush=True)
            np.savez_compressed(os.path.join(a.out, f"{tag}_data.npz"), k=ku, mask=m, ref=ref.astype(np.float32))
            avg_case = EvalCase(tag, kavg, mavg, (cal, cal), pixel_spacing=sp, pe_axis=1, joint=True)
            ext = acquired_extent(sampled)
            obj = ref.mean(-1) > 0.08 * ref.mean(-1).max()
            obj3 = np.repeat(obj[..., None], T, -1)
            common = dict(calib=cal, spacing=sp, plane=plane, device=a.device, verbose=False)

            def fit_maps(meth):
                try:
                    if meth == "espirit_avg":                 # ACS block of the time average, as in the benchmark
                        C = bm.espirit(dataclasses.replace(avg_case, joint=False))
                    elif meth == "nlinv_avg":                 # all acquired samples of the time average
                        C = bm.nlinv(avg_case)
                    elif meth == "pinn_avg":
                        C = estimate_maps(kavg, "2d", mask=mavg, readout_axis=0, **common)[0]
                    elif meth in ("espirit_frame", "nlinv_frame"):
                        Cs = []
                        for t in range(T):
                            ct = central_lines(m[..., t], a.calib)
                            case = EvalCase(f"{tag}_t{t}", ku[..., t], m[..., t], (max(ct, 4),) * 2,
                                            pixel_spacing=sp, pe_axis=1, joint=meth == "nlinv_frame")
                            if meth == "espirit_frame":
                                if ct < 8:
                                    raise ValueError(f"frame {t} has only {ct} central lines: no calibration block "
                                                     "(per-frame ESPIRiT needs R = 1 or ACS in every frame)")
                                Cs.append(bm.espirit(case))
                            else:
                                Cs.append(bm.nlinv(case))
                        C = np.stack(Cs, -1)
                    elif meth == "pinn_dyn":                  # low-rank dynamic model: every frame Biot-Savart
                        C, dinfo = estimate_maps(ku, "dynamic", mask=m, rank=a.rank, **common)
                        np.save(os.path.join(a.out, f"{tag}_pinn_dyn_modes.npy"), dinfo["temporal_modes"])
                    else:
                        raise ValueError(f"unknown method {meth}")
                except Exception as e:      # noqa: BLE001  one failing method must not stop the others
                    print(f"   {meth}: failed ({type(e).__name__}: {e})", flush=True)
                    return None
                return rss_normalise(np.asarray(C))

            def reconstruct(C, rc):
                """Coil-combined cine (X, Y, T): 'tv' = the benchmark's temporal TV + spatial TV of the difference
                from the time average (bspinn.prior.dynamic_prior_image, weight by cross-validation); 'sense' =
                CG-SENSE per frame with a cross-validated Tikhonov weight. Band-limited to the acquired extent."""
                Ct = (lambda t: C[..., t]) if C.ndim == 4 else (lambda t: C)
                if rc == "tv":
                    x = dynamic_prior_image(ku, C, m, "auto", spatial_on="difference", device=a.device)
                elif rc == "sense":
                    rep = {}
                    sense_image(ku[..., T // 2], Ct(T // 2), "2d", mask=m[..., T // 2], lam="auto", device=a.device,
                                report=rep, prior=None)
                    lam = rep.get("value") or 1e-2
                    x = np.stack([sense_image(ku[..., t], Ct(t), "2d", mask=m[..., t], lam=lam, device=a.device,
                                              prior=None) for t in range(T)], -1)
                else:
                    raise ValueError(f"unknown recon {rc}")
                return np.stack([band_limit(x[..., t], ext) for t in range(T)], -1)

            for meth in a.methods:
                out = os.path.join(a.out, f"{tag}_{meth}.npy")
                t0 = time.time()
                if os.path.exists(out):
                    C = np.load(out)
                    print(f"   {meth}: maps exist", flush=True)
                else:
                    C = fit_maps(meth)
                    if C is None:
                        continue
                    np.save(out, C)
                    print(f"   {meth:14s} {str(C.shape):22s} {time.time() - t0:6.1f} s", flush=True)
                for rc in a.recon:                            # the coil-combined cine with these maps
                    ro = os.path.join(a.out, f"{tag}_{meth}_{rc}.npy")
                    if os.path.exists(ro):
                        continue
                    t0 = time.time()
                    try:
                        x = reconstruct(C, rc)
                    except Exception as e:  # noqa: BLE001
                        print(f"   {meth} {rc}: failed ({type(e).__name__}: {e})", flush=True); continue
                    np.save(ro, x.astype(np.complex64))
                    xs = np.abs(x); r_ = ref
                    sc = (xs[obj3] * r_[obj3]).sum() / max((xs[obj3] ** 2).sum(), 1e-30)
                    e = np.linalg.norm((sc * xs - r_)[obj3]) / np.linalg.norm(r_[obj3])
                    print(f"   {meth:14s} {rc:6s} image {str(x.shape):16s} NRMSE {e:.4f}  {time.time() - t0:6.1f} s",
                          flush=True)



if __name__ == "__main__":
    main()
