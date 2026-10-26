"""Figures of the ISMRM 2027 method abstract (Fig. 1, the method overview, is made separately).

  fig2_static   coil-combined images and coil sensitivity maps of static slices (ESPIRiT, NLINV, proposed)
  fig3_cine     animated GIF of a cardiac cine: fully sampled, proposed reconstruction (R = 4, TV) and one
                coil's per-frame sensitivity with its change from the time average; plus a PNG of one frame
  fig4_summary  one graph: median image NRMSE of all experiments per method (from dashboard_data.json)

Run from the folder that contains bspinn/ and runs/:

    # once: save the cine frames of one cine with the per-frame maps (cached maps are reused; TV only)
    uv run python bspinn/scripts/ocmr_eval.py --files data/ocmr/fs_0021*.h5 --slices mid --R 4 --recon tv \
        --maps pinn_dyn --save-images --device cuda:0 --out runs/ocmr_v59/fig.jsonl

    uv run python bspinn/scripts/ismrm_figures.py --data dashboard_data.json --eval runs/eval_v59 \
        --ocmr runs/ocmr_v59 --cine fs_0021_3T --out figs

Static slices come from runs/eval_v59/<case>/figures.npz (keys <method>/combined, <method>/sense_R4,
<method>/maps); by default the two slices nearest to the median NRMSE, or --cases item1066_j4 item2644_j4.
Cine: runs/ocmr_v59/<cine>_s0_images.npz (keys ref, R4/<maps>/tv) and runs/ocmr_v59/maps/<cine>_s0_R4_pinn_dyn.npy.
Output: PNG (300 dpi) and PDF; the cine also as GIF.
"""
import argparse
import colorsys
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

COL = {"espirit": "#8c8c8c", "nlinv": "#d9822b", "pinn": "#1f5fa8"}
LAB = {"espirit": "ESPIRiT", "nlinv": "NLINV", "pinn": "Proposed"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.dpi": 300})


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return statistics.median(v) if v else float("nan")


def save(fig, out, name, exts=("png", "pdf")):
    os.makedirs(out, exist_ok=True)
    for ext in exts:
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.join(out, name)}.{exts[0]}")


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def gray(ax, im, vmax, title=None):
    ax.imshow(im, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest"); bare(ax)
    if title:
        ax.set_title(title, pad=2)


def fit_scale(x, ref, obj):
    """Least-squares magnitude scale of |x| onto |ref| inside obj (the benchmark's NRMSE convention)."""
    x, r = np.abs(x), np.abs(ref)
    return x * (x[obj] * r[obj]).sum() / max((x[obj] ** 2).sum(), 1e-30)


def crop_box(mask, pad=6):
    r, c = np.nonzero(mask)
    if not r.size:
        return slice(None), slice(None)
    return slice(max(r.min() - pad, 0), r.max() + pad + 1), slice(max(c.min() - pad, 0), c.max() + pad + 1)


def complex_rgb(z, vmax=None, mask=None):
    """Hue = phase, brightness = magnitude (black = 0): smooth maps look smooth, phase poles show as colour wheels."""
    z = np.asarray(z)
    vmax = vmax or np.percentile(np.abs(z), 99.5) or 1.0
    h = (np.angle(z) + np.pi) / (2 * np.pi)
    v = np.clip(np.abs(z) / vmax, 0, 1)
    rgb = np.stack(np.vectorize(colorsys.hsv_to_rgb)(h, 0.85 * np.ones_like(h), v), -1)
    if mask is not None:
        rgb = rgb * mask[..., None]
    return rgb


def gauge_fix(C, mask):
    """Remove the arbitrary common phase of a map set (phase of the coil sum inside the mask = 0 on average), so
    different methods can be compared by eye. The relative coil phases are unchanged."""
    s = C.sum(0)
    return C * np.exp(-1j * np.angle(s[mask].sum()))


def montage(maps, coils, rows=2):
    """(N,X,Y) -> one image of the chosen coils on a rows x cols grid."""
    tiles = [maps[c] for c in coils]
    rows = min(rows, len(tiles))
    cols = int(np.ceil(len(tiles) / rows))
    X, Y = tiles[0].shape
    out = np.zeros((rows * X, cols * Y), complex)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * X:(r + 1) * X, c * Y:(c + 1) * Y] = t
    return out


def phase_wheel(ax):
    n = 101
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    z = (xx + 1j * yy) * (np.abs(xx + 1j * yy) <= 1)
    ax.imshow(complex_rgb(z, 1.0)); bare(ax)
    ax.set_title("hue: phase\nbrightness: magnitude", fontsize=6, pad=2)


# ------------------------------------------------------------------------------------- Fig. 2: static slices
def fig2(d, a):
    if not a.eval:
        print("fig2: --eval not given: skipped"); return
    prior = [r for r in d.get("prior", []) if r["method"] == a.static_prior and r["maps"] == a.pinn_key]
    by = {r["case"]: r["nrmse"] for r in prior}
    cases = a.cases
    if not cases:
        m = med(list(by.values()))
        cases = [c for c in sorted(by, key=lambda c: abs(by[c] - m))
                 if os.path.exists(os.path.join(a.eval, c, "figures.npz"))][:2]
    cases = [c for c in cases if os.path.exists(os.path.join(a.eval, c, "figures.npz"))]
    if not cases:
        print(f"fig2: no <case>/figures.npz in {a.eval}: skipped"); return
    methods = (("espirit", "espirit"), ("nlinv", "nlinv"), (a.pinn_key, "pinn"))
    fig = plt.figure(figsize=(7.2, 3.7 * len(cases)))
    outer = GridSpec(len(cases), 1, figure=fig, hspace=0.1)
    for i, case in enumerate(cases):
        f = np.load(os.path.join(a.eval, case, "figures.npz"))
        obj = f["object"].astype(bool)
        sup = f["support"].astype(bool) if "support" in f.files else obj
        ref = np.abs(f["reference"])
        box = crop_box(sup)
        vmax = np.percentile(ref[obj], 99.5)
        g = GridSpecFromSubplotSpec(2, 4, outer[i], hspace=0.04, wspace=0.03)
        ax = fig.add_subplot(g[0, 0])
        gray(ax, ref[box], vmax, "Fully sampled (RSS)" if i == 0 else None)
        ax.text(0.03, 0.97, "ABCDEFGH"[i], transform=ax.transAxes, color="white", fontweight="bold", va="top")
        phase_wheel(fig.add_subplot(GridSpecFromSubplotSpec(3, 3, g[1, 0])[1, 1]))
        title = {"combined": "coil-combined", "sense_R4": "SENSE, R = 4"}[a.static_image]
        for j, (key, m) in enumerate(methods):
            img_key = f"{key}/{a.static_image}"
            if img_key not in f.files:
                print(f"fig2: {case}: {img_key} missing"); continue
            x = fit_scale(f[img_key], ref, obj)
            ax = fig.add_subplot(g[0, j + 1])
            gray(ax, x[box], vmax, f"{LAB[m]}, {title}" if i == 0 else None)
            if a.static_image == "sense_R4" or a.show_nrmse:
                e = np.linalg.norm((x - ref)[obj]) / np.linalg.norm(ref[obj])
                ax.text(0.03, 0.03, f"NRMSE {e:.3f}", transform=ax.transAxes, color="white", fontsize=6)
            if f"{key}/maps" in f.files:
                C = gauge_fix(f[f"{key}/maps"][:, box[0], box[1]], sup[box])
                coils = a.coils if a.coils else list(range(min(4, C.shape[0])))
                mk = montage(np.repeat(sup[box][None].astype(complex), C.shape[0], 0), coils).real > 0
                ax = fig.add_subplot(g[1, j + 1])
                ax.imshow(complex_rgb(montage(C, coils), 1.0, mk), interpolation="nearest"); bare(ax)
                if j == 0:
                    ax.set_ylabel(f"coil maps {', '.join(str(c + 1) for c in coils)}", fontsize=6.5)
    save(fig, a.out, "fig2_static")


# ------------------------------------------------------------------------------------- Fig. 3: cine (GIF)
def fig3(d, a):
    if not (a.ocmr and a.cine):
        print("fig3: --ocmr and --cine not given: skipped"); return
    p_img = os.path.join(a.ocmr, f"{a.cine}_s0_images.npz")
    p_map = os.path.join(a.ocmr, "maps", f"{a.cine}_s0_R{a.cine_R:g}_{a.cine_maps}.npy")
    if not os.path.exists(p_img):
        print(f"fig3: {p_img} not found (see the ocmr_eval.py --save-images command at the top): skipped"); return
    f = np.load(p_img)
    ref = np.abs(f["ref"].astype(np.float32))                                  # (X,Y,T)
    key = f"R{a.cine_R:g}/{a.cine_maps}/tv"
    if key not in f.files:
        key = next((k for k in f.files if k.startswith(f"R{a.cine_R:g}/pinn") and k.endswith("/tv")), None)
    if key is None:
        print(f"fig3: no R{a.cine_R:g} proposed TV series in {p_img}: skipped"); return
    rec = f[key].astype(np.float32)
    C = np.load(p_map) if os.path.exists(p_map) else None
    if C is None:
        print(f"fig3: {p_map} not found: map columns skipped")
    X, Y, T = ref.shape
    obj = ref.mean(-1) > a.cine_mask * ref.mean(-1).max()
    box = crop_box(obj, 4) if a.cine_crop else (slice(None), slice(None))
    vmax = np.percentile(ref[obj], 99.5)
    recs = np.stack([fit_scale(rec[..., t], ref[..., t], obj) for t in range(T)], -1)
    nrmse = np.linalg.norm((recs - ref)[obj]) / np.linalg.norm(ref[obj])
    j = a.cine_coil
    if C is not None:
        if C.ndim == 3:                                                          # time-averaged maps: constant
            C = np.repeat(C[..., None], T, -1)
        C = C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)
        if j is None:                                                            # the coil that changes most
            j = int(np.argmax(np.abs(C - C.mean(-1, keepdims=True))[:, obj].mean((1, 2))))
        Cj = C[j]
        dmap = Cj - Cj.mean(-1, keepdims=True)
        dmax = max(np.percentile(np.abs(dmap[obj]), 99.5), 1e-6)
        cmax = np.percentile(np.abs(Cj[obj]), 99.5)
        rel = np.linalg.norm(dmap[obj]) / np.linalg.norm(Cj[obj])
        print(f"fig3: coil {j + 1}: change over the cycle {rel:.1%} of the map (RMS inside the object)")

    ncol = 4 if C is not None else 2
    fig, axs = plt.subplots(1, ncol, figsize=(1.8 * ncol, 2.15))
    titles = ["Fully sampled", f"Proposed, R = {a.cine_R:g}, TV\nNRMSE {nrmse:.3f}"]
    if C is not None:
        titles += [f"Sensitivity, coil {j + 1}", f"Change from time average\n(×{cmax / dmax:.0f})"]
    ims = [axs[0].imshow(ref[..., 0][box], cmap="gray", vmin=0, vmax=vmax),
           axs[1].imshow(recs[..., 0][box], cmap="gray", vmin=0, vmax=vmax)]
    if C is not None:
        ims.append(axs[2].imshow(complex_rgb(Cj[..., 0][box], cmax, obj[box])))
        ims.append(axs[3].imshow(complex_rgb(dmap[..., 0][box], dmax, obj[box])))
    for ax, t in zip(axs, titles):
        bare(ax); ax.set_title(t, fontsize=7, pad=2)
    lab = fig.text(0.01, 0.01, "", fontsize=6.5, color="0.3")
    fig.tight_layout(pad=0.3)

    def draw(t):
        ims[0].set_data(ref[..., t][box]); ims[1].set_data(recs[..., t][box])
        if C is not None:
            ims[2].set_data(complex_rgb(Cj[..., t][box], cmax, obj[box]))
            ims[3].set_data(complex_rgb(dmap[..., t][box], dmax, obj[box]))
        lab.set_text(f"frame {t + 1}/{T}")
        return ims + [lab]

    from matplotlib.animation import FuncAnimation, PillowWriter
    os.makedirs(a.out, exist_ok=True)
    gif = os.path.join(a.out, "fig3_cine.gif")
    FuncAnimation(fig, draw, frames=T, blit=False).save(gif, writer=PillowWriter(fps=a.fps), dpi=a.gif_dpi)
    print(f"wrote {gif} ({os.path.getsize(gif) / 1e6:.1f} MB)")
    draw(a.cine_frame if a.cine_frame is not None else T // 3)
    save(fig, a.out, "fig3_cine_frame")


# ------------------------------------------------------------------------------------- Fig. 4: one graph
def fig4(d, a):
    groups = []                                     # (label, {method: median NRMSE})
    pr = [r for r in d.get("prior", []) if r["method"] == a.static_prior]
    groups.append((f"Static\nR = 4, n = {len({r['case'] for r in pr})}",
                   {m: med([r["nrmse"] for r in pr if r["maps"] == k])
                    for k, m in (("espirit", "espirit"), ("nlinv", "nlinv"), (a.pinn_key, "pinn"))}))
    oc = [r for r in d.get("ocmr", []) if r.get("run") == a.cine_run and r["recon"] == "tv"]
    for R in (4.0, 8.0):
        n = len({r["file"] for r in oc if r["R"] == R})
        groups.append((f"Cine\nR = {R:g}, n = {n}",
                       {m: med([r["nrmse"] for r in oc if r["maps"] == k and r["R"] == R])
                        for k, m in (("espirit", "espirit"), ("nlinv", "nlinv"), ("pinn_avg", "pinn"))}))
    po = d.get("poles", [])
    for cal in sorted({r["cal"] for r in po}):
        groups.append((f"Brain\n{cal}×{cal} calib.",
                       {m: med([r["sense_nrmse"] for r in po if r["maps"] == k and r["cal"] == cal])
                        for k, m in (("espirit", "espirit"), ("nlinv_pp", "nlinv"), ("pinn_v59", "pinn"))}))
    nc = {r["case"]: r for r in d.get("extra", {}).get("r4_noncart", [])}
    for k in ("radial", "spiral"):
        r = nc.get(f"synth_{k}")
        if r:
            groups.append((f"{k.capitalize()}\n(simulated)", {"true": r["nrmse_true_tv"], "pinn": r["nrmse_pinn_tv"]}))

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    w = 0.27
    for gi, (lab, v) in enumerate(groups):
        ks = [k for k in ("espirit", "nlinv", "true", "pinn") if k in v and np.isfinite(v[k])]
        for i, k in enumerate(ks):
            x, y = gi + (i - (len(ks) - 1) / 2) * w, v[k]
            if k == "true":
                ax.bar(x, y, w, facecolor="white", edgecolor="0.3", lw=0.6, hatch="////")
            else:
                ax.bar(x, y, w, color=COL[k])
            ax.text(x, y + 0.004, f"{y:.3f}", ha="center", va="bottom", fontsize=5.5, rotation=90)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=6.3)
    ax.set_ylabel("Median image NRMSE")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("espirit", "nlinv", "pinn")]
    labels = ["ESPIRiT", "NLINV (brain: with pole correction)", "Proposed"]
    if any("true" in g[1] for g in groups):
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.3", hatch="////"))
        labels.append("True sensitivities")
    ax.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    ax.set_ylim(0, 1.2 * max(y for g in groups for y in g[1].values() if np.isfinite(y)))
    save(fig, a.out, "fig4_summary")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dashboard_data.json")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--figs", nargs="+", default=["2", "3", "4"])
    ap.add_argument("--pinn-key", default="pinn_v59", help="name of the proposed maps in the static results")
    ap.add_argument("--eval", default=None, help="evaluate_methods.py output folder (runs/eval_v59)")
    ap.add_argument("--cases", nargs="*", default=None, help="static slices to show (default: 2 near the median)")
    ap.add_argument("--static-image", default="combined", choices=["combined", "sense_R4"],
                    help="combined: fully sampled data combined with each method's maps; sense_R4: SENSE at R = 4")
    ap.add_argument("--show-nrmse", action="store_true", help="NRMSE on the coil-combined images too")
    ap.add_argument("--static-prior", default="tgv_phase_auto")
    ap.add_argument("--coils", type=int, nargs="*", default=None, help="coils to show, 0-based (default 0 1 2 3)")
    ap.add_argument("--ocmr", default=None, help="ocmr_eval.py output folder (runs/ocmr_v59)")
    ap.add_argument("--cine", default=None, help="cine file name without .h5, e.g. fs_0021_3T")
    ap.add_argument("--cine-R", type=float, default=4)
    ap.add_argument("--cine-maps", default="pinn_dyn", help="pinn_dyn (per-frame maps) or pinn_avg")
    ap.add_argument("--cine-coil", type=int, default=None, help="0-based; default: the coil whose map changes most")
    ap.add_argument("--cine-run", default="ocmr_v59")
    ap.add_argument("--cine-mask", type=float, default=0.08)
    ap.add_argument("--cine-crop", action="store_true")
    ap.add_argument("--cine-frame", type=int, default=None)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--gif-dpi", type=int, default=150)
    a = ap.parse_args()
    d = json.load(open(a.data))
    for n in a.figs:
        {"2": fig2, "3": fig3, "4": fig4}[n](d, a)


if __name__ == "__main__":
    main()
