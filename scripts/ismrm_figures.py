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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def normalise(C):
    return C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)


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
STATIC_METHODS = ("espirit", "nlinv", "pinn")


def static_from_npz(path, image_key, pinn_key):
    """One slice from evaluate_methods.py's figures.npz: the stored R = 4 SENSE or fully sampled combination."""
    f = np.load(path)
    obj = f["object"].astype(bool)
    out = {"obj": obj, "sup": f["support"].astype(bool) if "support" in f.files else obj,
           "ref": np.abs(f["reference"]), "x": {}, "C": {}}
    for key, m in (("espirit", "espirit"), ("nlinv", "nlinv"), (pinn_key, "pinn")):
        if f"{key}/{image_key}" in f.files:
            out["x"][m] = np.abs(f[f"{key}/{image_key}"])
        if f"{key}/maps" in f.files:
            out["C"][m] = f[f"{key}/maps"]
    return out


def recon_title(rc, m):
    if rc == "bart":
        return "BART nlinv" if m == "nlinv" else f"BART pics ℓ2"
    return {"tikhonov": "SENSE", "tv": "TV", "tgv": "TGV", "tgv_phase_auto": "TGV + phase",
            "tv_phase_auto": "TV + phase"}.get(rc, rc)


def bart_pics(k, C, lam):
    """BART pics -l2 (Tikhonov-regularised CG-SENSE), k (N,X,Y) zero where not sampled, C (N,X,Y) -> (X,Y)."""
    import shutil
    import subprocess
    import tempfile
    from bspinn import methods as bm
    from bspinn.data import _read_cfl
    from bspinn.methods import _write_cfl
    tmp = tempfile.mkdtemp()
    try:
        _write_cfl(os.path.join(tmp, "k"), np.moveaxis(k, 0, -1)[:, :, None, :].astype(np.complex64))
        _write_cfl(os.path.join(tmp, "s"), np.moveaxis(C, 0, -1)[:, :, None, :].astype(np.complex64))
        subprocess.run([bm._bart(), "pics", "-S", "-l2", "-r", f"{lam:g}", os.path.join(tmp, "k"),
                        os.path.join(tmp, "s"), os.path.join(tmp, "x")], check=True, capture_output=True)
        return np.squeeze(_read_cfl(os.path.join(tmp, "x")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def static_live(a, index, R):
    """One dataset item undersampled at R (the benchmark's variable-density mask, 24 ACS lines); maps estimated
    on it (or read from the maps cache of the benchmark, <maps-cache>/item<i>_j<R>/maps) and the image
    reconstructed with --recon. Needs the dataloader and, for new R, a GPU."""
    from functools import partial
    import evaluate_methods as evm
    from method_lab import parse_set, undersampled
    from prior_eval import PRIOR_METHODS
    from scipy.ndimage import binary_dilation
    from zs_ssl import band_limit
    from bspinn import evaluation as ev, methods as bm
    from bspinn.api import acquired_extent, sense_image
    from bspinn.prior import AUTO_SCALE, prior_image
    bm.ESTIMATORS[a.pinn_key] = partial(bm.pinn_api, pinn_options=parse_set(a.pinn_set))
    bm._PASS[a.pinn_key] = bm._PASS["pinn_api"]
    case0 = next(evm.loader_cases(a.loader, a.base_folder, [index], 1, 0, a.key, a.coil_axis, 24, None, None))()
    obj = ev.object_mask(case0)
    sup = binary_dilation(obj, iterations=3)
    ext = acquired_extent(case0.sampled)
    ref = np.sqrt((np.abs(ev.ifft2c(case0.ksp * case0.sampled[None])) ** 2).sum(0))
    fit = undersampled(case0, R)
    fit.name = f"{case0.name}_j{R:g}"
    cdir = os.path.join(a.maps_cache or a.eval or "runs/fig2_maps", fit.name, "maps")
    est_kw = {"threads": 1, "cfg_overrides": {}, "device": a.device, "quality": "default"}
    out = {"obj": obj, "sup": sup, "ref": ref, "x": {}, "C": {}, "R_eff": float(case0.sampled.sum() / fit.sampled.sum())}
    for key, m in (("espirit", "espirit"), ("nlinv", "nlinv"), (a.pinn_key, "pinn")):
        if m in a.refit or "all" in a.refit:            # drop the cached maps: estimated again below
            for suffix in (".npy", ".time"):
                if os.path.exists(os.path.join(cdir, key + suffix)):
                    os.remove(os.path.join(cdir, key + suffix))
            print(f"fig2: item{index} R = {R:g} {key}: re-estimating", flush=True)
        try:
            C = bm.estimate(fit, [key], cache_dir=cdir, log=print, **est_kw)[0][key]
        except Exception as e:          # noqa: BLE001  (e.g. BART missing: show the other methods)
            print(f"fig2: item{index} R = {R:g} {key} failed ({type(e).__name__}: {e}); left out", flush=True)
            continue
        Cn = ev.normalise(C)
        rc = getattr(a, f"recon_{m}") or a.recon
        if rc == "bart" and m == "nlinv":        # BART nlinv's own joint image (and its own maps)
            from phase_pole_eval import nlinv_full
            Cb, x = nlinv_full(fit.ksp, [])
            Cn = ev.normalise(Cb)
        elif rc == "bart":                       # BART pics, l2 (Tikhonov) SENSE with these maps
            x = bart_pics(fit.ksp, Cn, a.bart_l2)
        elif rc == "tikhonov":
            x = sense_image(fit.ksp, Cn, "2d", mask=fit.sampled, lam="auto", device=a.device, prior=None)
        else:
            pr = PRIOR_METHODS[rc]
            pr, pkw = pr if isinstance(pr, tuple) else (pr, {})
            x = prior_image(fit.ksp, Cn, fit.sampled, pr, "auto", device=a.device, scale=AUTO_SCALE,
                            grid=np.logspace(-4, -1, 10), iters=50, cg_iters=5, **pkw)
        out["x"][m] = np.abs(band_limit(x, ext))
        out["C"][m] = Cn.astype(np.complex64)
        out.setdefault("titles", {})[m] = f"R = {R:g}, {recon_title(rc, m)}"
        print(f"fig2: item{index} R = {R:g} {m}: done", flush=True)
    return out


def fig2(d, a):
    """Rows per slice: images, error x gain, difference to the proposed image x gain, coil maps (gauge-aligned to
    the proposed maps), map difference to the proposed maps."""
    from cine_gif import align
    slices = []                       # (name, data, image title)
    if a.loader:
        for idx in a.index:
            for R in a.R_static:
                rt = {"tikhonov": "SENSE", "tv": "TV", "tgv_phase_auto": "TGV + phase"}.get(a.recon, a.recon)
                slices.append((f"item{idx}, R = {R:g}", static_live(a, idx, R), f"R = {R:g}, {rt}"))
    else:
        if not a.eval:
            print("fig2: give --eval (stored R = 4 results) or --loader with --index and --R-static: skipped"); return
        prior = [r for r in d.get("prior", []) if r["method"] == a.static_prior and r["maps"] == a.pinn_key]
        by = {r["case"]: r["nrmse"] for r in prior}
        cases = a.cases
        if not cases:
            m = med(list(by.values()))
            cases = [c for c in sorted(by, key=lambda c: abs(by[c] - m))
                     if os.path.exists(os.path.join(a.eval, c, "figures.npz"))][:1]
        for c in cases:
            p = os.path.join(a.eval, c, "figures.npz")
            if os.path.exists(p):
                slices.append((c, static_from_npz(p, a.static_image, a.pinn_key),
                               {"combined": "coil-combined", "sense_R4": "R = 4, SENSE"}[a.static_image]))
            else:
                print(f"fig2: {p} not found")
    if not slices:
        print("fig2: nothing to show"); return
    row_lab = ["", f"Error ×{a.err_gain:g}", f"Difference to\nproposed ×{a.err_gain:g}", None,
               f"Map difference\nto proposed\n(0–{a.map_vmax:g})"]
    fig = plt.figure(figsize=(7.2, 8.6 * len(slices)))
    outer = GridSpec(len(slices), 1, figure=fig, hspace=0.06)
    for i, (name, s_, title) in enumerate(slices):
        obj, sup, ref = s_["obj"], s_["sup"], s_["ref"]
        box = crop_box(sup)
        mk = sup[box]
        vmax = np.percentile(ref[obj], 99.5)
        g = GridSpecFromSubplotSpec(5, 4, outer[i], hspace=0.04, wspace=0.03)
        ax = fig.add_subplot(g[0, 0])
        gray(ax, ref[box], vmax, "Fully sampled (RSS)")
        ax.set_ylabel(name, fontsize=7)
        for r in (1, 2, 4):
            ax = fig.add_subplot(g[r, 0]); ax.axis("off")
            ax.text(0.5, 0.5, row_lab[r], ha="center", va="center", fontsize=7, color="0.3", transform=ax.transAxes)
        phase_wheel(fig.add_subplot(GridSpecFromSubplotSpec(3, 3, g[3, 0])[1, 1]))
        xs = {m: fit_scale(x, ref, obj) for m, x in s_["x"].items()}
        Cs = {m: normalise(C[:, box[0], box[1]]) for m, C in s_["C"].items()}
        P = gauge_fix(Cs["pinn"], mk) if "pinn" in Cs else None
        coils = a.coils if a.coils else list(range(min(4, next(iter(Cs.values())).shape[0]))) if Cs else []
        for j, m in enumerate(STATIC_METHODS):
            col = j + 1
            if m in xs:
                x = xs[m]
                e = np.linalg.norm((x - ref)[obj]) / np.linalg.norm(ref[obj])
                ax = fig.add_subplot(g[0, col])
                gray(ax, x[box], vmax, f"{LAB[m]}\n{s_.get('titles', {}).get(m, title)}")
                ax = fig.add_subplot(g[1, col]); gray(ax, (np.abs(x - ref) * obj)[box] * a.err_gain, vmax)
                ax.text(0.03, 0.03, f"NRMSE {e:.3f}", transform=ax.transAxes, color="white", fontsize=6)
                ax = fig.add_subplot(g[2, col])
                if m == "pinn" or "pinn" not in xs:
                    ax.axis("off")
                else:
                    dd = np.linalg.norm((x - xs["pinn"])[obj]) / np.linalg.norm(ref[obj])
                    gray(ax, (np.abs(x - xs["pinn"]) * obj)[box] * a.err_gain, vmax)
                    ax.text(0.03, 0.03, f"{dd:.3f}", transform=ax.transAxes, color="white", fontsize=6)
            if m in Cs:
                C, dmap = align(Cs[m], P) if P is not None else (gauge_fix(Cs[m], mk), None)
                tm = montage(np.repeat(mk[None].astype(complex), C.shape[0], 0), coils).real > 0
                ax = fig.add_subplot(g[3, col])
                ax.imshow(complex_rgb(montage(C, coils), 1.0, tm), interpolation="nearest"); bare(ax)
                if j == 0:
                    ax.set_ylabel(f"coil maps {', '.join(str(c + 1) for c in coils)}", fontsize=6.5)
                ax = fig.add_subplot(g[4, col])
                if dmap is None or m == "pinn":
                    ax.axis("off")
                else:
                    ax.imshow(dmap * mk, cmap="magma", vmin=0, vmax=a.map_vmax, interpolation="nearest"); bare(ax)
                    ax.text(0.03, 0.03, f"mean {dmap[obj[box]].mean():.3f}", transform=ax.transAxes, color="white",
                            fontsize=6)
    tag = "_".join(n.replace(", R = ", "_R").replace(" ", "") for n, _, _ in slices) if a.loader else "static"
    save(fig, a.out, f"fig2_{tag}" if a.loader else "fig2_static")


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


# ------------------------------------------------------------------------------------- Fig. 4: quantitative results
def fig4(d, a):
    """The numbers of the Results section: (A) image NRMSE, static (TGV + phase) and cine (TV); (B) LOCO excess,
    static; (C) mean g-factor, static, R = 2/4/8; (D) prospective cines, error on withheld lines / noise."""
    M3 = (("espirit", "espirit"), ("nlinv", "nlinv"), (a.pinn_key, "pinn"))
    MC = (("espirit", "espirit"), ("nlinv", "nlinv"), ("pinn_avg", "pinn"))
    pr = [r for r in d.get("prior", []) if r["method"] == a.static_prior]
    oc = [r for r in d.get("ocmr", []) if r.get("run") == a.cine_run and r["recon"] == "tv"]
    ev = [r for r in d.get("eval", []) if r.get("group") != "synthetic"]
    us = [r for r in d.get("ocmr_us", []) if r["recon"] == "tv"]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
    axs = axs.ravel()
    g = [(f"Static\nR = 4 (n = {len({r['case'] for r in pr})})",
          {m: med([r["nrmse"] for r in pr if r["maps"] == k]) for k, m in M3})]
    for R in (4.0, 8.0):
        g.append((f"Cine\nR = {R:g} (n = {len({r['file'] for r in oc if r['R'] == R})})",
                  {m: med([r["nrmse"] for r in oc if r["maps"] == k and r["R"] == R]) for k, m in MC}))
    bars(axs[0], g, "Image NRMSE ↓", "median NRMSE")
    cases = sorted({r["case"] for r in ev})
    lco = {m: {r["case"]: r["lco_outside_acs_over_floor"] - 1 for r in ev if r["method"] == k} for k, m in M3}
    win = sum(lco["pinn"][c] < lco["nlinv"][c] for c in cases if c in lco["pinn"] and c in lco["nlinv"])
    bars(axs[1], [(f"Static, R = 4\nproposed better than\nNLINV in {win}/{len(cases)}",
                   {m: med(list(v.values())) for m, v in lco.items()})],
         "Leave-one-coil-out excess ↓", "median (error / noise floor − 1)", "{:.4f}")
    bars(axs[2], [(f"R = {R}", {m: med([r[f"gfactor_R{R}_mean"] for r in ev if r["method"] == k]) for k, m in M3})
                  for R in (2, 4, 8)], "Static: mean g-factor ↓", "median over slices", "{:.1f}")
    files = sorted({r["file"] for r in us})
    bars(axs[3], [(f"cine {i + 1}", {m: next((r["heldout"] for r in us if r["file"] == f and r["maps"] == k), None)
                                     for k, m in MC}) for i, f in enumerate(files)],
         "Prospective cines (R ≈ 9) ↓", "error on withheld lines / noise", "{:.2f}", log=True, ref_line=1)
    for ax, l in zip(axs, "ABCD"):
        ax.text(-0.02, 1.04, l, transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right", va="bottom")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("espirit", "nlinv", "pinn")]
    fig.legend(handles, ["ESPIRiT", "NLINV", "Proposed"], loc="upper center", ncol=3, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.5, w_pad=1.2)
    save(fig, a.out, "fig4_quantitative")


# ------------------------------------------------------------------------------------- Fig. 5: every metric
def bars(ax, groups, title, ylabel, fmt="{:.3f}", log=False, ref_line=None):
    """groups: [(label, {method: value})]; methods espirit / nlinv / pinn (and 'true', hatched)."""
    w = 0.27
    for gi, (lab, v) in enumerate(groups):
        ks = [k for k in ("espirit", "nlinv", "true", "pinn") if k in v and v[k] is not None and np.isfinite(v[k])]
        for i, k in enumerate(ks):
            x, y = gi + (i - (len(ks) - 1) / 2) * w, v[k]
            if k == "true":
                ax.bar(x, y, w, facecolor="white", edgecolor="0.3", lw=0.6, hatch="////")
            else:
                ax.bar(x, y, w, color=COL[k])
            ax.text(x, y * (1.06 if log else 1) + (0 if log else 0.01 * ax.get_ylim()[1]), fmt.format(y),
                    ha="center", va="bottom", fontsize=5, rotation=90)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=6)
    ax.set_title(title, fontsize=7.5); ax.set_ylabel(ylabel, fontsize=6.5)
    if log:
        ax.set_yscale("log")
    if ref_line is not None:
        ax.axhline(ref_line, color="0.4", lw=0.6, ls="--")
    vals = [y for _, v in groups for y in v.values() if y is not None and np.isfinite(y)]
    if vals and not log:
        lo = min(vals)
        ax.set_ylim(min(0, lo * 1.2), max(vals) * 1.3)
    elif vals:
        ax.set_ylim(min(vals) / 1.5, max(vals) * 4)


def fig5(d, a):
    M3 = (("espirit", "espirit"), ("nlinv", "nlinv"), (a.pinn_key, "pinn"))
    MC = (("espirit", "espirit"), ("nlinv", "nlinv"), ("pinn_avg", "pinn"))
    MB = (("espirit", "espirit"), ("nlinv_pp", "nlinv"), ("pinn_v59", "pinn"))
    pr = [r for r in d.get("prior", []) if r["method"] == a.static_prior]
    oc = [r for r in d.get("ocmr", []) if r.get("run") == a.cine_run]
    ev = [r for r in d.get("eval", []) if r.get("group") != "synthetic"]
    po = d.get("poles", [])
    us = [r for r in d.get("ocmr_us", []) if r["recon"] == "tv"]
    cals = sorted({r["cal"] for r in po})

    def static(key):
        return ("Static\nR = 4", {m: med([r[key] for r in pr if r["maps"] == k]) for k, m in M3})

    def cine(key, R, recon="tv", f=lambda x: x):
        return (f"Cine\nR = {R:g}", {m: med([f(r[key]) for r in oc if r["maps"] == k and r["R"] == R
                                              and r["recon"] == recon]) for k, m in MC})

    fig = plt.figure(figsize=(7.2, 9.4))
    gsm = GridSpec(4, 3, figure=fig, height_ratios=[1.1, 1, 1, 1])
    axs = [fig.add_subplot(gsm[0, :])] + [fig.add_subplot(gsm[1 + i // 3, i % 3]) for i in range(8)]
    nc = {r["case"]: r for r in d.get("extra", {}).get("r4_noncart", [])}
    g = [static("nrmse"), cine("nrmse", 4.0), cine("nrmse", 8.0)]
    g += [(f"Brain\n{c}×{c}", {m: med([r["sense_nrmse"] for r in po if r["maps"] == k and r["cal"] == c])
                               for k, m in MB}) for c in cals]
    g += [(f"{k[:6].capitalize()}\n(sim.)", {"true": nc[f"synth_{k}"]["nrmse_true_tv"],
                                             "pinn": nc[f"synth_{k}"]["nrmse_pinn_tv"]})
          for k in ("radial", "spiral") if f"synth_{k}" in nc]
    bars(axs[0], g, "Image NRMSE ↓ (static: TGV + phase; cine: TV; brain: SENSE; simulations: TV)", "median NRMSE")
    bars(axs[1], [static("ssim"), cine("ssim", 4.0), cine("ssim", 8.0)], "SSIM ↑", "median SSIM")
    axs[1].set_ylim(0.8, 1.0)
    bars(axs[2], [static("psnr"), cine("psnr", 4.0), cine("psnr", 8.0)], "PSNR ↑", "median PSNR (dB)", "{:.1f}")
    bars(axs[3], [cine("dyn_nrmse", 4.0), cine("dyn_nrmse", 8.0)], "Temporal NRMSE (frame − mean) ↓",
         "median NRMSE")
    ex = lambda x: x - 1  # noqa: E731
    lco = [("Static\nR = 4", {m: med([r["lco_outside_acs_over_floor"] - 1 for r in ev if r["method"] == k])
                                for k, m in M3}),
           ("Cine, fully\nsampled", {m: med([r["lco_outside_over_floor"] - 1 for r in oc if r["maps"] == k
                                              and r["R"] == 1.0 and r.get("lco_outside_over_floor") is not None])
                                      for k, m in MC})]
    lco += [(f"Brain\n{c}×{c}", {m: med([r["lco_object"] - 1 for r in po if r["maps"] == k and r["cal"] == c])
                                 for k, m in MB}) for c in cals]
    bars(axs[4], lco, "Leave-one-coil-out error above noise ↓", "median (error / noise − 1)", log=True)
    sg = [(f"R = {R}", {m: med([r[f"sense_R{R}_support_nrmse_vs_rss_or_truth"] for r in ev if r["method"] == k])
                        for k, m in M3}) for R in (2, 4, 8)]
    bars(axs[5], sg, "Static: SENSE (Tikhonov) NRMSE ↓", "median NRMSE")
    gf = [(f"R = {R}", {m: med([r[f"gfactor_R{R}_mean"] for r in ev if r["method"] == k]) for k, m in M3})
          for R in (2, 4, 8)]
    bars(axs[6], gf, "Static: mean g-factor ↓", "median of slice means", "{:.2f}")
    files = sorted({r["file"] for r in us})
    bars(axs[7], [(f"cine {i + 1}", {m: next((r["heldout"] for r in us if r["file"] == f and r["maps"] == k), None)
                                     for k, m in MC}) for i, f in enumerate(files)],
         "Prospective (R ≈ 9): withheld lines ↓", "error / noise", "{:.2f}", log=True, ref_line=1)
    def count(k, c):
        rr = [r["gauge_poles"] for r in po if r["maps"] == k and r["cal"] == c]
        return float(sum(rr)) if rr else None

    pg = [(f"{c}×{c}", {m: count(k, c) for k, m in (("espirit", "espirit"), ("nlinv", "nlinv"), ("pinn_v59", "pinn"))})
          for c in cals]
    bars(axs[8], pg, "Brain: phase singularities ↓\n(NLINV without correction)", "count over 3 planes", "{:.0f}")
    axs[8].set_xlabel("calibration region", fontsize=6.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("espirit", "nlinv", "pinn")]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.3", hatch="////"))
    fig.legend(handles, ["ESPIRiT", "NLINV (brain: with pole correction, except singularities)", "Proposed",
                         "True sensitivities"], loc="upper center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=1.0, w_pad=0.8)
    save(fig, a.out, "fig5_metrics")


def main():
    if not os.environ.get("BART") and os.path.exists("bart/bart"):      # as run_v59.sh: BART=${BART:-bart/bart}
        os.environ["BART"] = os.path.abspath("bart/bart")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dashboard_data.json")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--figs", nargs="+", default=["2", "3", "4"])
    ap.add_argument("--pinn-key", default="pinn_v59", help="name of the proposed maps in the static results")
    ap.add_argument("--eval", default=None, help="evaluate_methods.py output folder (runs/eval_v59)")
    g2 = ap.add_argument_group("fig 2 at any case and R (reconstructs; needs the dataloader, a GPU for new R)")
    g2.add_argument("--loader", default=None, metavar="MODULE:FUNCTION", help="e.g. dataloader_v2:get_dataloader")
    g2.add_argument("--base-folder", default=None)
    g2.add_argument("--index", type=int, nargs="+", default=[], help="dataset item(s), e.g. 1066 for item1066")
    g2.add_argument("--R-static", type=float, nargs="+", default=[4.0], help="undersampling factor(s)")
    g2.add_argument("--recon", default="tgv_phase_auto", choices=["tikhonov", "tv", "tgv", "tgv_phase_auto", "tv_phase_auto"])
    for m_ in ("espirit", "nlinv", "pinn"):
        g2.add_argument(f"--recon-{m_}", default=None,
                        choices=["tikhonov", "tv", "tgv", "tgv_phase_auto", "tv_phase_auto", "bart"],
                        help=f"reconstruction for the {m_} maps (default: --recon); 'bart': " +
                             ("BART nlinv's own joint image and maps" if m_ == "nlinv" else "BART pics -l2 with these maps"))
    g2.add_argument("--bart-l2", type=float, default=0.01, help="pics -r for --recon-... bart")
    g2.add_argument("--maps-cache", default=None, help="maps cache root (default: --eval, so the benchmark's R = 4 "
                    "maps are reused; other R are fitted and stored there as item<i>_j<R>/maps)")
    g2.add_argument("--pinn-set", nargs="*", default=["solve_image=true"], help="options of the proposed maps (run_v59)")
    g2.add_argument("--refit", nargs="*", default=[], choices=["pinn", "espirit", "nlinv", "all"],
                    help="estimate these maps again instead of reading the cache (the new maps replace the cached ones)")
    g2.add_argument("--key", default="kspace")
    g2.add_argument("--coil-axis", type=int, default=0)
    g2.add_argument("--device", default="cuda")
    ap.add_argument("--cases", nargs="*", default=None, help="static slices to show (default: 2 near the median)")
    ap.add_argument("--static-image", default="combined", choices=["combined", "sense_R4"],
                    help="combined: fully sampled data combined with each method's maps; sense_R4: SENSE at R = 4")
    ap.add_argument("--err-gain", type=float, default=5.0, help="image errors and differences shown this much brighter")
    ap.add_argument("--map-vmax", type=float, default=0.3, help="colour range of the map-difference panels")
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
        {"2": fig2, "3": fig3, "4": fig4, "5": fig5}[n](d, a)


if __name__ == "__main__":
    main()
