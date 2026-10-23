"""Phase poles and small calibration regions: ESPIRiT, NLINV (with / without the phase-pole correction of
Blumenthal & Uecker, arXiv 2508.04685) and the BS-PINN on fully sampled 2-D k-space, retrospectively undersampled.

    BART=/path/to/bart python scripts/phase_pole_eval.py --files data/ksp_2D data/ksp_2D2 data/ksp_2D3 \\
        --acs 3 7 15 --device cuda --out runs/poles/results.jsonl

Input: BART .cfl/.hdr (base name without extension) of ONE fully sampled 2-D plane, coils on the last non-singleton
axis (e.g. the paper's brain slices ksp_2D*: 1 x 256 x 208 x 20, a ky-kz plane), or .npy (coils, x, y).
Per file, as in the paper: pre-whitened (noise covariance from the data: k-space periphery or image background;
--noise-npy for a noise scan), compressed to --coils virtual coils, --R1 x --R2 undersampling with a centred ACS
box of --acs lines (each value one case).
Maps from the undersampled data (--maps):
  espirit   BART ecalib -m1 on the ACS (kernel 5 for ACS < 10; not run for ACS < 5)
  nlinv     BART nlinv (joint image + maps, all samples); also scores its own image (the paper's 'black hole')
  nlinv_pp  the same with --phase-pole 0 (the paper's correction; needs BART >= 1.0)
  pinn      estimate_maps 2-D, maps only (the --box calibration box), v54 options
  pinn_v54  estimate_maps with the joint full-resolution stage (solve_image=True, all samples), v54 options
  pinn_v55  the same with the v55 defaults (init_from_acs, resid_zero_mean, joint_lbfgs)
  <maps>+irgnm[@key=value...]  IRGNM refinement (bspinn.irgnm, held-out step choice) of the cached <maps>
            (list <maps> first), e.g. pinn_v54+irgnm  pinn_v54+irgnm@steps=8@holdout=0
  <pinn*>@key=value[@key=value...]   a variant with extra estimate_maps / FitConfig options, its own cache and
            rows, for ablations, e.g. pinn_v55@init_from_acs=False  pinn_v55@joint_lbfgs=0  pinn_v54@seed=1
Scored against the fully sampled plane: gauge poles inside the object (bspinn.recon.gauge_poles); leave-coil-out
residual / noise floor (gauge-invariant) over the object and within 8 px of any method's pole; CG-SENSE NRMSE at
the same undersampling (object, near poles). One JSON line per (file, acs, maps) in --out (resume-safe); maps cached
in <out dir>/maps. NB the 'near poles' region is the union over the map sets of that run."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)

V54 = dict(init_from_acs=False, resid_zero_mean=False, joint_lbfgs=0)


def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1), norm="ortho"), axes=(-2, -1))


def load_plane(path):
    """BART cfl base or .npy -> (coils, x, y) complex128."""
    from bspinn.data import _read_cfl
    if path.endswith(".npy"):
        return np.load(path).astype(np.complex128)
    k = np.squeeze(_read_cfl(path))
    if k.ndim != 3:
        raise SystemExit(f"{path}: expected one 2-D plane with coils, got shape {k.shape}")
    return np.moveaxis(k, -1, 0).astype(np.complex128)


def nlinv_full(k, extra):
    """BART nlinv -> (maps (N,H,W), image (H,W))."""
    from bspinn import methods as bm
    from bspinn.data import _read_cfl
    from bspinn.methods import _write_cfl
    tmp = tempfile.mkdtemp()
    try:
        _write_cfl(os.path.join(tmp, "k"), np.moveaxis(k, 0, -1)[:, :, None, :])
        subprocess.run([bm._bart(), "nlinv"] + extra + [os.path.join(tmp, "k"), os.path.join(tmp, "i"),
                                                        os.path.join(tmp, "s")], check=True, capture_output=True)
        s = _read_cfl(os.path.join(tmp, "s")); i = _read_cfl(os.path.join(tmp, "i"))
        s = s.reshape(s.shape[:4] + (-1,))[..., 0]
        return np.transpose(s[:, :, 0, :], (2, 0, 1)), np.squeeze(i.reshape(i.shape[:4] + (-1,))[..., 0])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--acs", type=int, nargs="+", default=[3, 7, 15])
    ap.add_argument("--R1", type=int, default=2)
    ap.add_argument("--R2", type=int, default=2)
    ap.add_argument("--coils", type=int, default=12, help="virtual coils after compression (0: all)")
    ap.add_argument("--maps", nargs="+", default=["espirit", "nlinv", "nlinv_pp", "pinn_v54", "pinn_v55"])
    ap.add_argument("--box", type=int, default=24, help="PINN calibration box (need not be fully sampled)")
    ap.add_argument("--spacing", type=float, nargs=2, default=[1.0, 0.9])
    ap.add_argument("--quality", default="fast", choices=["fast", "default"])
    ap.add_argument("--pinn-set", nargs="*", default=[], help="extra estimate_maps options key=value")
    ap.add_argument("--noise-npy", default=None, help="noise samples (n, coils) for pre-whitening")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/poles/results.jsonl")
    a = ap.parse_args()

    import torch
    import evaluate_methods as evm
    from method_lab import parse_set
    from ocmr_eval import series_metrics
    from bspinn import evaluation as ev, refree
    from bspinn import methods as bm
    from bspinn.api import estimate_maps, sense_image
    from bspinn.methods import EvalCase
    from bspinn.recon import gauge_poles
    from scipy.ndimage import binary_dilation
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available: using the CPU (slow)"); a.device = "cpu"
    pset = parse_set(a.pinn_set)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    root = os.path.dirname(os.path.abspath(a.out))
    done = set()
    if os.path.exists(a.out):
        done = {(r["file"], r["cal"], r["maps"]) for r in map(json.loads, (l for l in open(a.out) if l.strip()))}
    nrm = lambda C: C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)  # noqa: E731
    sp = tuple(a.spacing)
    for path in a.files:
        fname = os.path.basename(path).replace(".npy", "")
        k = load_plane(path)
        sampled = np.ones(k.shape[1:], bool)
        if a.noise_npy:
            n = np.load(a.noise_npy).reshape(-1, k.shape[0]); psi = (n.T @ n.conj()) / n.shape[0]
            ninfo = {"noise_source": "noise scan"}
        else:
            psi, ninfo = evm.estimate_noise(k, sampled)
        L = np.linalg.cholesky(psi + 1e-9 * np.trace(psi).real / k.shape[0] * np.eye(k.shape[0]))
        k = np.linalg.solve(L, k.reshape(k.shape[0], -1)).reshape(k.shape)
        if a.coils and a.coils < k.shape[0]:
            U, _, _ = np.linalg.svd(k.reshape(k.shape[0], -1), full_matrices=False)
            k = np.tensordot(U[:, :a.coils].conj().T, k, 1)                   # unitary: still white
        N, X, Y = k.shape
        full = EvalCase(fname, k, sampled, (24, 24), sigma=1.0)
        obj = ev.object_mask(full)
        ref = np.sqrt((np.abs(ifft2c(k)) ** 2).sum(0))
        print(f"== {fname}: {N} coils, {X}x{Y}, object {obj.mean():.2f} of the FOV, noise from "
              f"{ninfo['noise_source']}", flush=True)
        for cal in a.acs:
            m = np.zeros((X, Y), bool); m[::a.R1, ::a.R2] = True
            acs_only = np.zeros((X, Y), bool)
            acs_only[X // 2 - cal // 2: X // 2 - cal // 2 + cal, Y // 2 - cal // 2: Y // 2 - cal // 2 + cal] = True
            m |= acs_only
            ku = (k * m[None]).astype(np.complex64)
            maps, own = {}, {}
            for name in a.maps:
                cp = os.path.join(root, "maps", f"{fname}_acs{cal}_{name}.npy")
                t0 = time.time()
                try:
                    if os.path.exists(cp):
                        C = np.load(cp)
                        if name.startswith("nlinv") and os.path.exists(cp.replace(".npy", "_img.npy")):
                            own[name] = np.load(cp.replace(".npy", "_img.npy"))
                    elif name == "espirit":
                        if cal < 5:
                            raise RuntimeError("ACS smaller than the smallest usable ESPIRiT kernel")
                        C = bm._bart_run(["ecalib", "-m1"] + (["-k", "5"] if cal < 10 else []) + ["-r", f"{cal}:{cal}"],
                                         ku * acs_only[None])
                    elif name in ("nlinv", "nlinv_pp"):
                        C, im = nlinv_full(ku, [] if name == "nlinv" else ["--phase-pole", "0"])
                        own[name] = im
                        os.makedirs(os.path.dirname(cp), exist_ok=True)
                        np.save(cp.replace(".npy", "_img.npy"), im.astype(np.complex64))
                    elif "+irgnm" in name:                         # IRGNM refinement of another map set's cache
                        from bspinn.irgnm import irgnm_refine
                        base, rest = name.split("+irgnm", 1)
                        C0 = np.load(os.path.join(root, "maps", f"{fname}_acs{cal}_{base}.npy"))
                        ro = {"holdout": 0.1, **parse_set([o for o in rest.split("@") if o])}
                        C, _, rinfo = irgnm_refine(ku, C0, m, device=a.device, **ro)
                        print(f"   ACS {cal:2d} {name}: {rinfo['steps']} steps, held-out {rinfo.get('heldout_err')}",
                              flush=True)
                    elif name.split("@")[0] in ("pinn", "pinn_v54", "pinn_v55"):
                        base, *opts = name.split("@")             # e.g. pinn_v55@resid_zero_mean=False@seed=1
                        kw = dict(mask=m, calib=a.box, spacing=sp, quality=a.quality, verbose=False, device=a.device,
                                  readout_axis=0)
                        if base != "pinn":
                            kw.update(solve_image=True, optimizer="adam+lbfgs")
                        if base != "pinn_v55":
                            kw.update(V54)
                        kw.update(pset)
                        kw.update(parse_set(opts))
                        C = estimate_maps(ku, "2d", **kw)[0]
                    else:
                        raise SystemExit(f"unknown maps {name}")
                    os.makedirs(os.path.dirname(cp), exist_ok=True)
                    np.save(cp, np.asarray(C, np.complex64))
                    maps[name] = (nrm(np.asarray(C)), time.time() - t0)
                except subprocess.CalledProcessError as e:
                    print(f"   ACS {cal:2d} {name:9s} failed: BART {' '.join(e.cmd[1:3])} exit {e.returncode}", flush=True)
                except RuntimeError as e:
                    print(f"   ACS {cal:2d} {name:9s} failed: {e}", flush=True)
            near = np.zeros((X, Y), bool)
            for C, _ in maps.values():
                for r0, c0, _ in gauge_poles(C, obj):
                    near[int(r0), int(c0)] = True
            near = binary_dilation(near, iterations=8) & obj
            for name, (C, secs) in maps.items():
                if (fname, cal, name) in done:
                    continue
                nps = len(gauge_poles(C, obj))
                lo = refree.leave_coil_out(C, full, obj)["object"]["residual_over_noise_floor"]
                ln = refree.leave_coil_out(C, full, near)["object"]["residual_over_noise_floor"] if near.any() else float("nan")
                x = sense_image(ku, C, "2d", mask=m, lam="auto", device=a.device, prior=None)
                q = series_metrics(x[..., None], ref[..., None], obj)["nrmse"]
                qn = series_metrics(x[..., None], ref[..., None], near)["nrmse"] if near.any() else float("nan")
                r = {"file": fname, "cal": cal, "maps": name, "gauge_poles": nps, "lco_object": lo, "lco_near": ln,
                     "sense_nrmse": q, "sense_nrmse_near": qn, "near_pixels": int(near.sum()), "seconds": secs}
                line = (f"   ACS {cal:2d} {name:9s} gauge poles {nps:2d}  lco object {lo:6.3f}  near {ln:6.3f}   "
                        f"SENSE NRMSE {q:.4f}  near {qn:.4f}")
                if name in own and own[name].shape == ref.shape:
                    im = own[name]
                    r["own_image_nrmse"] = series_metrics(im[..., None], ref[..., None], obj)["nrmse"]
                    r["own_image_nrmse_near"] = (series_metrics(im[..., None], ref[..., None], near)["nrmse"]
                                                 if near.any() else float("nan"))
                    line += f"   own image NRMSE {r['own_image_nrmse']:.4f}  near {r['own_image_nrmse_near']:.4f}"
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
                print(line + f"  ({secs:.0f} s)", flush=True)
    rows = [json.loads(l) for l in open(a.out) if l.strip()] if os.path.exists(a.out) else []
    if rows:
        print("\nmedian over files: gauge poles / lco object / lco near poles / SENSE NRMSE / near poles")
        for cal in sorted({r["cal"] for r in rows}):
            for mname in dict.fromkeys(r["maps"] for r in rows):
                rs = [r for r in rows if r["cal"] == cal and r["maps"] == mname]
                if rs:
                    md = lambda kk: np.nanmedian([r[kk] for r in rs])  # noqa: E731
                    print(f"  ACS {cal:2d} {mname:9s} n={len(rs)}  poles {sum(r['gauge_poles'] for r in rs):2d}  "
                          f"lco {md('lco_object'):.3f} / {md('lco_near'):.3f}  SENSE {md('sense_nrmse'):.4f} / "
                          f"{md('sense_nrmse_near'):.4f}")


if __name__ == "__main__":
    main()
