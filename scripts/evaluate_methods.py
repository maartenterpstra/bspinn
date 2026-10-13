"""Comparative evaluation of coil-sensitivity estimators.

    python scripts/evaluate_methods.py --cases synth8 knee15 --out runs/eval
    python scripts/evaluate_methods.py --list
    # all synthetic cases + N random slices from your own dataset, on the GPU:
    python scripts/evaluate_methods.py --cases synth --loader dataloader_v2:get_dataloader \
        --base-folder /path/to/data --n 10 --device cuda --Rs 4 --out runs/eval_local

Cases (all single 2-D slices, pre-whitened):
  synth8            8 coils, Biot–Savart ground truth, noise sigma 0.01, 128^2, ACS 24
  synth8_noise{lo,hi}  same, sigma 0.003 / 0.03
  synth8_mismatch   same, every coil x its own smooth complex non-Biot–Savart factor (strength 0.3)
  knee15[_acsK]     real fully sampled 15-channel coronal knee, ACS 24 (or K = 16/32/48)
  brain8            real prospective Poisson-disc brain (8 ch), joint calibration, held-out samples

Every estimator sees the same ACS box (calibration-only protocol) except in ``brain8``, where
NLINV and the Biot–Savart fit may use all acquired samples (their natural mode).
Writes <out>/<case>/results.json, maps/*.npy (cache) and figures.npz (images for the report).

Every method also gets a ``reference_free`` block (bspinn/refree.py): metrics from the measured k-space
only, built so that fitting noise does not pay off (leave-coil-out prediction, residual structure,
projector bandwidth; with --resample r2r / sure also recorrupted-to-recorrupted and Monte-Carlo SURE,
which re-run each estimator on perturbed data). A table of the headline numbers is printed per case and
written to <out>/<case>/reference_free.tsv.

Dataset items (--loader MODULE:FUNCTION, called as FUNCTION(base_folder); item[--key] is the
k-space, (coils, x, y) or (slices, coils, x, y) -> middle slice): treated as fully sampled single 2-D
slices, pre-whitened with the coil noise covariance from the k-space periphery (or --noise-npy),
calibration = central --acs box, phase-encoding axis from the partial-Fourier zeros (or --pe-axis).
``pinn_api`` = bspinn.estimate_maps as you call it (plane search, fold-over handling); bs_pinn /
bs_wires / free use the fixed benchmark config and assume the case's plane (axial for dataset items),
so they are not in the default method list for dataset items.
At the end: <out>/summary.tsv (one row per case x method) and <out>/ranks.tsv (mean rank per method
and metric over the synthetic and the real cases; direction-aware: |x - 1| for ratios to a floor,
|x| for slopes / optimism, higher-is-better for SSIM).
"""
import argparse
import subprocess
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bspinn import evaluation as ev  # noqa: E402
from bspinn import refree  # noqa: E402
from bspinn.data import make_synthetic  # noqa: E402
from bspinn.methods import ALL_METHODS, EvalCase, estimate, ifft2c  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def synth(name, noise=0.01, mismatch=0.0, n_coils=8, size=128, acs=24):
    d, C, obj = make_synthetic(n_coils, (size, size), noise=noise, seed=0, mismatch=mismatch)
    return EvalCase(name, d.kspace[0], np.ones((size, size), bool), (acs, acs), sigma=noise,
                    gt_maps=C[0], gt_object=obj[0], meta={"noise": noise, "mismatch": mismatch})


def real(name, path, acs, fully=True, joint=False, pe_axis=1, bs_calib=None):
    z = np.load(os.path.join(ROOT, path))
    c = EvalCase(name, z["ksp"], z["mask"], (acs, acs), tuple(float(v) for v in z["pixel_spacing"]), str(z["plane"]),
                 sigma=float(z["sigma"]), fully_sampled=fully, joint=joint, pe_axis=pe_axis,
                 meta={"source": str(z["source"])})
    if bs_calib:
        c.meta["bs_calib"] = bs_calib
    return c


CASES = {
    "synth8": lambda: synth("synth8", 0.01),
    "synth8_noiselo": lambda: synth("synth8_noiselo", 0.003),
    "synth8_noisehi": lambda: synth("synth8_noisehi", 0.03),
    "synth8_mismatch": lambda: synth("synth8_mismatch", 0.01, mismatch=0.3),
    "knee15": lambda: real("knee15", "data/eval_knee15.npz", 24),
    "knee15_acs16": lambda: real("knee15_acs16", "data/eval_knee15.npz", 16),
    "knee15_acs32": lambda: real("knee15_acs32", "data/eval_knee15.npz", 32),
    "knee15_acs48": lambda: real("knee15_acs48", "data/eval_knee15.npz", 48),
    "brain8": lambda: real("brain8", "data/eval_brain8.npz", 20, fully=False, joint=True, bs_calib=48),
}


def estimate_noise(k, sampled, bg_frac=0.15, smooth=9):
    """Coil noise covariance without a noise scan. Two estimates, both biased UP by signal only:
    the k-space periphery (accuracy_sweep.noise_cov) and the image background (the bg_frac of pixels
    with the lowest locally averaged RSS power; covariance / sampled fraction, orthonormal FFT).
    The one with the smaller trace is used. Every '..._over_floor' metric on real data is only as
    good as this: pass --noise-npy when you have noise-only samples."""
    from scipy.ndimage import uniform_filter
    from accuracy_sweep import noise_cov
    p_k = noise_cov(k, sampled)
    I = ifft2c(k * sampled[None])
    loc = uniform_filter((np.abs(I) ** 2).sum(0), smooth)
    bg = loc <= np.quantile(loc, bg_frac)
    v = I[:, bg]
    p_i = (v @ v.conj().T) / v.shape[1] / sampled.mean()
    tk, ti = float(np.trace(p_k).real), float(np.trace(p_i).real)
    info = {"noise_trace_kspace_periphery": tk, "noise_trace_image_background": ti,
            "noise_source": "kspace_periphery" if tk <= ti else "image_background",
            "noise_estimates_ratio": max(tk, ti) / max(min(tk, ti), 1e-30)}
    return (p_k if tk <= ti else p_i), info


def from_kspace(name, k, acs=24, pe_axis=None, noise=None, meta=None):
    """Fully sampled (coils, x, y) k-space -> pre-whitened EvalCase (sigma = 1)."""
    k = np.asarray(k).astype(np.complex128)
    mag = np.abs(k).sum(0)
    sampled = mag > 1e-6 * mag.max()
    if noise is None:
        psi, ninfo = estimate_noise(k, sampled)
        if ninfo["noise_estimates_ratio"] > 1.5:
            print(f"  [{name}] noise estimates disagree x{ninfo['noise_estimates_ratio']:.2f} "
                  f"(using {ninfo['noise_source']}): floors are approximate", flush=True)
    else:
        psi, ninfo = (noise.T @ noise.conj()) / noise.shape[0], {"noise_source": "noise_npy"}
    L = np.linalg.cholesky(psi + 1e-9 * np.trace(psi).real / len(psi) * np.eye(len(psi)))
    kw = np.linalg.solve(L, k.reshape(k.shape[0], -1)).reshape(k.shape) * sampled[None]
    if pe_axis is None:                 # partial-Fourier zeros are whole phase-encoding lines
        zr, zc = int((~sampled.any(1)).sum()), int((~sampled.any(0)).sum())
        pe_axis = 0 if zr > zc else 1
    m = {"api_plane": "auto", "pe_axis": pe_axis, "noise_trace": float(np.trace(psi).real), **ninfo, **(meta or {})}
    return EvalCase(name, kw.astype(np.complex64), sampled, (acs, acs), sigma=1.0, pe_axis=pe_axis, meta=m)


def loader_cases(spec, base_folder, indices, n, pick_seed, key, coil_axis, acs, pe_axis, noise):
    import importlib
    from accuracy_sweep import _to_numpy
    sys.path.insert(0, os.getcwd())
    mod, fn = spec.split(":")
    f = getattr(importlib.import_module(mod), fn)
    ds = f(base_folder) if base_folder else f()
    idx = indices if indices else sorted(np.random.default_rng(pick_seed).choice(len(ds), size=min(n, len(ds)), replace=False).tolist())
    print(f"dataset {spec}: {len(ds)} items, using {idx}", flush=True)
    for i in idx:
        item = ds[i]
        k = _to_numpy(item[key] if isinstance(item, dict) else item)
        meta = {kk: str(item[kk])[:120] for kk in ("source", "dataset", "filename", "fname", "slice", "contrast")
                if isinstance(item, dict) and kk in item}
        meta["index"] = int(i)
        if k.ndim == 4:
            k = k[k.shape[0] // 2]; meta["slice_of_volume"] = "middle"
        k = np.moveaxis(k, coil_axis, 0)
        if k.ndim != 3:
            print(f"skipping item {i}: expected (coils, x, y), got {k.shape}"); continue
        yield (lambda i=i, k=k, meta=meta: from_kspace(f"item{i}", k, acs, pe_axis, noise, meta))


def _json(o):
    if isinstance(o, dict):
        return {k: _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


MIN_OBJECT = 0.005          # fraction of the image the object mask must cover for a case to be scored


def run_case(case: EvalCase, methods, out, Rs=(2, 3, 4, 5, 6), threads=1, log=print, resample=(), r2r_alpha=0.7,
             sure_eps=0.1, sure_probes=1, device=None, quality=None, fit_case=None):
    """``fit_case``: what the estimators see (default: ``case``); every metric is computed on ``case``
    (--joint: the undersampled copy is fitted, the fully sampled original scores)."""
    fit_case = fit_case or case
    obj = ev.object_mask(case)
    if obj.sum() < MIN_OBJECT * obj.size:      # e.g. an edge slice of a volume: noise only, nothing to score
        log(f"  [{case.name}] skipped: no object found ({int(obj.sum())} pixels above the noise threshold)")
        return None
    cdir = os.path.join(out, case.name)
    os.makedirs(cdir, exist_ok=True)
    over = {}
    if case.meta.get("bs_calib"):
        over["calib"] = case.meta["bs_calib"]
    est_kw = {"threads": threads, "cfg_overrides": over}
    if device:
        est_kw["device"] = device
    if quality:
        est_kw["quality"] = quality
    maps, secs = {}, {}
    for mth in methods:                 # one at a time: a missing tool (BART) skips that method only
        try:
            m_, s_ = estimate(fit_case, [mth], cache_dir=os.path.join(cdir, "maps"), log=log, **est_kw)
            maps.update(m_); secs.update(s_)
        except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as e:
            log(f"  [{case.name}] skipping {mth}: {e}")
    from scipy.ndimage import binary_dilation
    sup = binary_dilation(obj, iterations=3)          # common SENSE support (same for every method)
    res = {"case": {"name": case.name, "n_coils": case.N, "shape": list(case.shape), "acs": list(case.acs),
                    "fully_sampled": case.fully_sampled, "joint_calibration": case.joint, "sigma": case.sigma,
                    "object_pixels": int(obj.sum()), "sampled_fraction": float(case.sampled.mean()),
                    "synthetic": case.gt_maps is not None,
                    "protocol": "joint" if fit_case is not case else "calibration",
                    "fit_sampled_fraction": float(fit_case.sampled.mean()), **case.meta},
           "runtime_s": secs, "methods": {}}
    figs = {"object": obj, "support": sup}
    I = ifft2c(case.ksp * case.sampled[None])
    rss = np.sqrt((np.abs(I) ** 2).sum(0))
    ref = rss if case.gt_maps is None else np.abs(case.gt_object) * np.sqrt((np.abs(case.gt_maps) ** 2).sum(0))
    figs["reference"] = ref
    for name, C in maps.items():
        t0 = time.time()
        r = {"maps": ev.map_properties(C, case, obj)}
        if case.fully_sampled:
            r["kspace_consistency"] = ev.kspace_consistency(C, case)
            r["reference_image"], m = ev.reference_image(C, case, obj)
            figs[f"{name}/combined"] = m
            r["sense"], r["sense_support"] = {}, {}
            for R in Rs:
                r["sense"][str(R)], x = ev.sense_at_R(C, case, obj, R, ref)
                r["sense_support"][str(R)], xs = ev.sense_at_R(C, case, obj, R, ref, support=sup)
                if R == 4:
                    figs[f"{name}/sense_R4"] = x
                    figs[f"{name}/sense_support_R4"] = xs
                    g = ev.gfactor(ev.normalise(C) * sup[None], 4, case.pe_axis, obj)
                    if g:
                        figs[f"{name}/gfactor_R4"] = g[1]
        else:
            r["prospective"], x = ev.heldout_prospective(C, case, obj)
            figs[f"{name}/sense_all"] = x
        t1 = time.time()
        r["reference_free"] = refree.evaluate(C, case, obj, name=name, est_kw=est_kw,
                                              resample=resample, r2r_alpha=r2r_alpha, sure_eps=sure_eps,
                                              sure_probes=sure_probes)
        log(f"  [{case.name}] reference-free {name:9s} ({time.time() - t1:.0f}s)")
        Cn = ev.normalise(C)
        figs[f"{name}/maps"] = Cn[: min(case.N, 8)].astype(np.complex64)
        res["methods"][name] = r
        log(f"  [{case.name}] metrics {name:9s} ({time.time() - t0:.0f}s)")
    with open(os.path.join(cdir, "results.json"), "w") as f:
        json.dump(_json(res), f, indent=1)
    np.savez_compressed(os.path.join(cdir, "figures.npz"), **{k: np.asarray(v) for k, v in figs.items()})
    table = refree_table(res)
    open(os.path.join(cdir, "reference_free.tsv"), "w").write(table)
    log(table.replace("\t", "  "))
    return res


# metric -> how to turn it into "lower is better" for ranking
def _badness(key, v):
    if v is None or not np.isfinite(v):
        return None
    if key.startswith("ssim"):
        return -v
    if key.endswith("over_floor") or key.endswith("over_oracle"):
        return abs(v - 1)
    if key in ("residual_signal_slope", "r2r_optimism"):
        return abs(v)
    return v


def flat_row(res, method):
    r = res["methods"][method]
    row = {}
    kc = r.get("kspace_consistency")
    if kc:
        row["ce_acs_over_floor"] = kc["acs"]["residual_over_noise_floor"]
        row["ce_outside_acs_over_floor"] = kc["outside_acs"]["residual_over_noise_floor"]
    row.update(r["reference_free"]["summary"])
    ss = r.get("sense_support", {})
    for R, v in ss.items():                 # every --Rs value
        row[f"sense_R{R}_support_nrmse_vs_rss_or_truth"] = v["vs_reference"]["nrmse"]
        if v.get("gfactor") and R != "1":
            row[f"gfactor_R{R}_mean"] = v["gfactor"]["mean"]
    if "prospective" in r:
        row["heldout_nrmse"] = r["prospective"]["heldout_nrmse"]
    if "gt_subspace_err" in r["maps"]:
        row["gt_subspace_err"] = r["maps"]["gt_subspace_err"]
    row["runtime_s"] = res["runtime_s"].get(method, float("nan"))
    return row


def write_summary(results, out):
    """summary.tsv (case x method) and ranks.tsv (mean rank per method, synthetic / real)."""
    rows = [(res["case"]["name"], "synthetic" if res["case"].get("synthetic") else "real", m, flat_row(res, m))
            for res in results for m in res["methods"]]
    keys = list(dict.fromkeys(k for *_, r in rows for k in r))
    with open(os.path.join(out, "summary.tsv"), "w") as f:
        f.write("case\tgroup\tmethod\t" + "\t".join(keys) + "\n")
        for c, g, m, r in rows:
            f.write(f"{c}\t{g}\t{m}\t" + "\t".join(f"{r[k]:.5g}" if k in r and r[k] is not None else "-" for k in keys) + "\n")
    rank_keys = [k for k in keys if k != "runtime_s"]
    lines = ["group\tmethod\tn_cases\t" + "\t".join(rank_keys)]
    for g in ("synthetic", "real"):
        ranks = {}
        for c in dict.fromkeys(c for c, gg, *_ in rows if gg == g):
            sub = [(m, r) for cc, gg, m, r in rows if cc == c]
            for k in rank_keys:
                vals = [(m, _badness(k, r.get(k))) for m, r in sub]
                vals = [(m, v) for m, v in vals if v is not None]
                order = sorted(vals, key=lambda t: t[1])
                for i, (m, _) in enumerate(order):
                    ranks.setdefault(m, {}).setdefault(k, []).append(i + 1)
        for m, d in ranks.items():
            n = max(len(v) for v in d.values())
            lines.append(f"{g}\t{m}\t{n}\t" + "\t".join(f"{np.mean(d[k]):.2f}" if k in d else "-" for k in rank_keys))
    open(os.path.join(out, "ranks.tsv"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines).replace("\t", "  "), flush=True)


def refree_table(res):
    """Headline reference-free numbers, one row per method (tab separated)."""
    rows = {m: r["reference_free"]["summary"] for m, r in res["methods"].items()}
    keys = list(dict.fromkeys(k for s in rows.values() for k in s))
    lines = ["method\t" + "\t".join(keys)]
    for m, s in rows.items():
        lines.append(m + "\t" + "\t".join(f"{s[k]:.4g}" if k in s else "-" for k in keys))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=[], help="case names, or 'synth' for all synthetic cases")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="default: all for named cases; lowres walsh inati espirit nlinv pinn_api for dataset items")
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default=None, help="for the Biot-Savart methods, e.g. cuda")
    ap.add_argument("--pinn-variant", nargs="+", action="append", default=[], metavar=("NAME", "KEY=VALUE"),
                    help="add a PINN method 'pinn_NAME' = estimate_maps with these options (repeatable), e.g. "
                         "--pinn-variant best solve_image=true optimizer=adam+lbfgs")
    ap.add_argument("--joint", type=float, default=0,
                    help="joint protocol: every case retrospectively undersampled (variable density, R, ACS kept), "
                         "every method sees all acquired samples (NLINV, the PINN with its mask), metrics on the "
                         "fully sampled data; results in <out>/<case>_jR")
    ap.add_argument("--pinn-quality", default=None, choices=["fast", "default"], help="estimate_maps preset for pinn_api")
    ap.add_argument("--Rs", type=int, nargs="+", default=[2, 3, 4, 5, 6], help="SENSE accelerations")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--resample", nargs="*", default=[], choices=["r2r", "sure"],
                    help="reference-free metrics that re-run every estimator on perturbed data (r2r: 1 extra run, sure: 2 x probes)")
    ap.add_argument("--r2r-alpha", type=float, default=0.7)
    ap.add_argument("--sure-eps", type=float, default=0.1, help="SURE finite-difference step, x sigma (raise to ~1 for stochastic estimators)")
    ap.add_argument("--sure-probes", type=int, default=1)
    g = ap.add_argument_group("dataset items")
    g.add_argument("--loader", default=None, metavar="MODULE:FUNCTION")
    g.add_argument("--base-folder", default=None)
    g.add_argument("--indices", type=int, nargs="*", default=None)
    g.add_argument("--n", type=int, default=6, help="number of random items (seeded by --pick-seed) if no --indices")
    g.add_argument("--pick-seed", type=int, default=0)
    g.add_argument("--key", default="kspace")
    g.add_argument("--coil-axis", type=int, default=0)
    g.add_argument("--acs", type=int, default=24)
    g.add_argument("--pe-axis", type=int, default=None, choices=[0, 1], help="default: from partial-Fourier zeros, else 1")
    g.add_argument("--pinn-plane", default="auto", choices=["auto", "axial", "coronal", "coronal_t", "sagittal"],
                   help="B0 orientation handed to pinn_api for dataset items (e.g. coronal for fastMRI coronal knees: "
                        "with only the calibration box the rows-vs-columns choice can be ambiguous)")
    g.add_argument("--noise-npy", default=None, help="noise-only samples (n, coils) for whitening")
    a = ap.parse_args()
    if a.list:
        print("\n".join(CASES)); return
    names = [n for c in a.cases for n in ([k for k in CASES if k.startswith("synth")] if c == "synth" else [c])]
    if not names and not a.loader:
        names = ["synth8", "knee15"]
    if a.device and a.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available, using the CPU"); a.device = "cpu"
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    from functools import partial
    from bspinn import methods as bm
    from method_lab import parse_set, undersampled
    pv = []
    for spec in a.pinn_variant:
        nm = f"pinn_{spec[0]}"
        bm.ESTIMATORS[nm] = partial(bm.pinn_api, pinn_options=parse_set(spec[1:]))
        bm._PASS[nm] = bm._PASS["pinn_api"]
        pv.append(nm)
        print(f"method {nm}: estimate_maps options {parse_set(spec[1:])}")
    if a.joint and a.resample:
        print("--resample is not available with --joint (it re-fits on the calibration protocol); ignored")
        a.resample = []
    jobs = [(CASES[c], (a.methods or ALL_METHODS) + [m for m in pv if not a.methods or m not in a.methods]) for c in names]
    if a.loader:
        noise = np.load(a.noise_npy).reshape(-1, np.load(a.noise_npy).shape[-1]) if a.noise_npy else None
        dm = (a.methods or ["lowres", "walsh", "inati", "espirit", "nlinv", "pinn_api"]) + [m for m in pv if not a.methods or m not in a.methods]
        jobs += [(mk, dm) for mk in loader_cases(a.loader, a.base_folder, a.indices, a.n, a.pick_seed, a.key,
                                                 a.coil_axis, a.acs, a.pe_axis, noise)]
    os.makedirs(a.out, exist_ok=True)
    results, skipped = [], []
    for make, methods in jobs:
        case = make()
        if case.meta.get("api_plane") == "auto" and a.pinn_plane != "auto":
            case.meta["api_plane"] = a.pinn_plane
        print(f"== {case.name}: {case.N} coils, {case.shape}, ACS {case.acs}, PE axis {case.pe_axis}", flush=True)
        fit_case = None
        if a.joint:
            fit_case = undersampled(case, a.joint)
            case.name = fit_case.name = f"{case.name}_j{a.joint:g}"
            print(f"   joint protocol: fitted on {fit_case.sampled.mean():.3f} of k-space (R={a.joint:g} vd + ACS)", flush=True)
        res = run_case(case, methods, a.out, Rs=tuple(a.Rs), threads=a.threads, log=log, resample=a.resample,
                       fit_case=fit_case,
                       r2r_alpha=a.r2r_alpha, sure_eps=a.sure_eps, sure_probes=a.sure_probes, device=a.device,
                       quality=a.pinn_quality)
        if res is None:
            skipped.append(case.name); continue
        results.append(res)
        write_summary(results, a.out)          # updated after every case
    if skipped:
        print(f"skipped (no object): {' '.join(skipped)}")


if __name__ == "__main__":
    main()
