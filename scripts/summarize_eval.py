"""Aggregate runs/eval/*/results.json into a flat CSV, a compact summary.json for the report,
metric-validity statistics, and figure PNGs.

    python scripts/summarize_eval.py --eval runs/eval
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

ORDER = ["oracle", "lowres", "walsh", "inati", "espirit", "nlinv", "free", "bs_wires", "bs_pinn", "bs_pinn_grid1"]
PANEL = ["oracle", "lowres", "walsh", "inati", "espirit", "nlinv", "bs_wires", "bs_pinn"]
LABEL = {"oracle": "Ground truth", "lowres": "Low-res division", "walsh": "Walsh", "inati": "Inati",
         "espirit": "ESPIRiT", "nlinv": "NLINV", "free": "Free (no physics)", "bs_wires": "Biot–Savart wires",
         "bs_pinn": "Biot–Savart PINN", "bs_pinn_grid1": "Biot–Savart PINN, ACS grid (v4)"}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)) or v is None:
            out[key] = v
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return None
    return float(spearmanr(a[ok], b[ok]).statistic)


# GT-free metrics a user could compute on real data, with the direction "lower is better"
GT_FREE = {
    "k-space residual / noise floor, outside ACS": ("kspace_consistency.outside_acs.residual_over_noise_floor", 1),
    "k-space residual / noise floor, inside ACS": ("kspace_consistency.acs.residual_over_noise_floor", 1),
    "SNR efficiency (reference image)": ("reference_image.snr_efficiency_mean", -1),
    "SENSE R4 missing-k-space NRMSE (support)": ("sense_support.4.missing_kspace_nrmse", 1),
    "SENSE R4 NRMSE vs own full combination (support)": ("sense_support.4.vs_own_full_combination.nrmse", 1),
    "g-factor R4 mean (support)": ("sense_support.4.gfactor.mean", 1),
    "projector roughness": ("maps.projector_roughness", 1),
}
GT_TARGETS = {"subspace error": "maps.gt_subspace_err",
              "SENSE R4 NRMSE vs truth (support)": "sense_support.4.vs_reference.nrmse"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="runs/eval")
    ap.add_argument("--eval3d", default="runs/eval3d", help="3-D results (optional)")
    a = ap.parse_args()
    if a.eval3d and os.path.isdir(a.eval3d):
        summarize3d(a.eval3d)
    results = {}
    for p in sorted(glob.glob(os.path.join(a.eval, "*", "results.json"))):
        r = json.load(open(p))
        results[r["case"]["name"]] = r
    rows = []
    for c, r in results.items():
        for m, v in r["methods"].items():
            f = flatten(v)
            f["runtime_s"] = r["runtime_s"].get(m)
            for k, val in f.items():
                rows.append({"case": c, "method": m, "metric": k, "value": val})
    with open(os.path.join(a.eval, "all_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, ["case", "method", "metric", "value"]); w.writeheader(); w.writerows(rows)

    # metric validity on synthetic cases (ground truth known)
    validity = {}
    for c, r in results.items():
        if not c.startswith("synth"):
            continue
        ms = [m for m in ORDER if m in r["methods"]]
        flat = {m: flatten(r["methods"][m]) for m in ms}
        validity[c] = {}
        for tname, tkey in GT_TARGETS.items():
            t = [flat[m].get(tkey) for m in ms]
            validity[c][tname] = {}
            for name, (key, sign) in GT_FREE.items():
                if key == tkey:
                    continue
                x = [None if flat[m].get(key) is None else sign * flat[m][key] for m in ms]
                validity[c][tname][name] = spearman(x, t)
    summary = {"cases": {c: r["case"] for c, r in results.items()}, "order": ORDER, "labels": LABEL,
               "metrics": {c: {m: flatten(v) | {"runtime_s": r["runtime_s"].get(m)} for m, v in r["methods"].items()}
                           for c, r in results.items()},
               "radial_profiles": {c: {m: v["kspace_consistency"]["radial_profile"] for m, v in r["methods"].items()
                                       if "kspace_consistency" in v} for c, r in results.items()},
               "validity_spearman": validity}
    with open(os.path.join(a.eval, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    figures(results, a.eval)
    print(f"{len(results)} cases, {len(rows)} metric values -> {a.eval}/all_metrics.csv, summary.json, fig_*.png")


LABEL3D = dict(LABEL, bs_pinn_complex="Biot–Savart PINN, complex field", bs_helmholtz="Helmholtz (k² learned)",
               bs_helmholtz_fixed="Helmholtz (true k²)")
ORDER3D = ["oracle", "lowres", "walsh", "inati", "espirit", "nlinv", "bs_wires", "bs_pinn", "bs_pinn_complex", "bs_helmholtz",
           "bs_helmholtz_fixed"]


def summarize3d(out):
    """runs/eval3d/*/results.json -> summary3d.json, all_metrics3d.csv, fig3d_<case>.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = {}
    for p in sorted(glob.glob(os.path.join(out, "*", "results.json"))):
        r = json.load(open(p)); results[r["case"]["name"]] = r
    rows = [{"case": c, "method": m, "metric": k, "value": v} for c, r in results.items() for m, mv in r["methods"].items()
            for k, v in (flatten(mv) | {"runtime_s": r["runtime_s"].get(m)}).items()]
    with open(os.path.join(out, "all_metrics3d.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, ["case", "method", "metric", "value"]); w.writeheader(); w.writerows(rows)
    summ = {"cases": {c: r["case"] for c, r in results.items()}, "order": ORDER3D, "labels": LABEL3D,
            "metrics": {c: {m: flatten(v) | {"runtime_s": r["runtime_s"].get(m)} for m, v in r["methods"].items()} for c, r in results.items()}}
    with open(os.path.join(out, "summary3d.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    for c in results:
        z = np.load(os.path.join(out, c, "figures.npz"))
        ms = [m for m in ORDER3D if f"{m}/maps" in z.files]
        obj, ref = z["object"], z["reference"]
        n = len(ms)
        fig, ax = plt.subplots(4, n + 1, figsize=(1.9 * (n + 1), 8), squeeze=False)
        vmax = np.quantile(ref[obj], 0.995) if obj.any() else ref.max()
        ax[0, 0].imshow(ref, cmap="gray", vmin=0, vmax=vmax); ax[0, 0].set_title("reference", fontsize=8)
        rc = z["reference_cor"]; ax[1, 0].imshow(rc, cmap="gray", vmin=0, vmax=vmax)
        ax[2, 0].axis("off"); ax[3, 0].axis("off")
        e = [(np.abs(z[f"{ms[0]}/maps"][k]) ** 2)[obj].mean() for k in range(z[f"{ms[0]}/maps"].shape[0])]
        ref_coils = {k for m in ms for k in range(z[f"{m}/maps"].shape[0]) if np.angle(z[f"{m}/maps"][k])[obj].std() < 1e-3}
        cand = [k for k in np.argsort(e) if k not in ref_coils] or list(np.argsort(e))
        j = int(cand[len(cand) // 2])
        for i, m in enumerate(ms):
            for row, key, r_ in ((0, "sense_support_R4", ref), (1, "sense_support_R4_cor", rc)):
                x = np.abs(z[f"{m}/{key}"]); s_ = (x * r_).sum() / max((x * x).sum(), 1e-30)
                ax[row, i + 1].imshow(s_ * x, cmap="gray", vmin=0, vmax=vmax)
            ax[0, i + 1].set_title(LABEL3D[m], fontsize=7)
            C = z[f"{m}/maps"][j]
            ax[2, i + 1].imshow(np.abs(C), cmap="viridis", vmin=0, vmax=1)
            ax[3, i + 1].imshow(np.angle(C), cmap="twilight", vmin=-np.pi, vmax=np.pi)
            ax[3, i + 1].contour(obj, [0.5], colors="w", linewidths=0.5)
        for a_ in ax.flat:
            a_.set_xticks([]); a_.set_yticks([])
        fig.tight_layout(); fig.savefig(os.path.join(out, f"fig3d_{c}.png"), dpi=100); plt.close(fig)
    print(f"3-D: {len(results)} cases -> {out}/summary3d.json")


def figures(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for c in results:
        p = os.path.join(out, c, "figures.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        ms = [m for m in PANEL if f"{m}/maps" in z.files]
        obj = z["object"]
        ref = z["reference"]
        key = "sense_support_R4" if f"{ms[0]}/sense_support_R4" in z.files else "sense_all"
        n = len(ms)
        # recon + error
        fig, ax = plt.subplots(2 if key != "sense_all" else 1, n + 1, figsize=(2.1 * (n + 1), 4.4 if key != "sense_all" else 2.4),
                               squeeze=False)
        vmax = np.quantile(ref[obj], 0.995)
        ax[0, 0].imshow(ref, cmap="gray", vmin=0, vmax=vmax); ax[0, 0].set_title("reference", fontsize=8)
        if key != "sense_all":
            ax[1, 0].axis("off")
        for i, m in enumerate(ms):
            x = np.abs(z[f"{m}/{key}"])
            s = (x[obj] * ref[obj]).sum() / max((x[obj] ** 2).sum(), 1e-30)
            ax[0, i + 1].imshow(s * x, cmap="gray", vmin=0, vmax=vmax); ax[0, i + 1].set_title(LABEL[m], fontsize=8)
            if key != "sense_all":
                ax[1, i + 1].imshow(np.where(obj, np.abs(s * x - ref), 0), cmap="magma", vmin=0, vmax=0.25 * vmax)
        for a_ in ax.flat:
            a_.set_xticks([]); a_.set_yticks([])
        fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_{c}_recon.png"), dpi=110); plt.close(fig)
        # map phase / magnitude of one coil + g-factor
        rows = 3 if any(f"{m}/gfactor_R4" in z.files for m in ms) else 2
        fig, ax = plt.subplots(rows, n, figsize=(2.1 * n, 2.2 * rows), squeeze=False)
        # show the coil of median energy in the object (not the phase-reference coil of any method)
        e = [(np.abs(z[f"{ms[0]}/maps"][k]) ** 2)[obj].mean() for k in range(z[f"{ms[0]}/maps"].shape[0])]
        ref_coils = {k for m in ms for k in range(z[f"{m}/maps"].shape[0])
                     if np.angle(z[f"{m}/maps"][k])[obj].std() < 1e-3}          # phase-reference coils (Walsh, ESPIRiT...)
        cand = [k for k in np.argsort(e) if k not in ref_coils] or list(np.argsort(e))
        j = int(cand[len(cand) // 2])
        for i, m in enumerate(ms):
            C = z[f"{m}/maps"]
            ax[0, i].imshow(np.abs(C[j]), cmap="viridis", vmin=0, vmax=1); ax[0, i].set_title(LABEL[m], fontsize=8)
            ax[1, i].imshow(np.angle(C[j]), cmap="twilight", vmin=-np.pi, vmax=np.pi)
            ax[1, i].contour(obj, [0.5], colors="w", linewidths=0.5)
            if rows == 3 and f"{m}/gfactor_R4" in z.files:
                ax[2, i].imshow(np.where(obj, z[f"{m}/gfactor_R4"], np.nan), cmap="inferno", vmin=1, vmax=4)
        for a_ in ax.flat:
            a_.set_xticks([]); a_.set_yticks([])
        fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_{c}_maps.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
