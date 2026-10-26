"""Figures 2-5 of the ISMRM 2027 method abstract, from the run_v59 outputs.

The numbers come from dashboard_data.json (run_v59.sh, step bundle); the images from the map and image files the
runs left behind. A panel whose files are missing is skipped with a message; the quantitative panels need only the
JSON. Run from the folder that contains bspinn/ and runs/:

    uv run python bspinn/scripts/ismrm_figures.py --data dashboard_data.json --out figs \
        --eval runs/eval_v59 --poles runs/poles_v59 --brain /localscratch/mterpst5 \
        --ocmr-images runs/ocmr_v59/fs_0021_3T_s0_images.npz --ocmr-images-axis runs/r3_ocmr/fs_0021_3T_s0_images.npz \
        --noncart runs/fig_noncart

Image files that the benchmark did not save (cine frames, non-Cartesian images) are made once with:

    # cine frames of fs_0021 (3 T) at R = 4, reusing the cached maps; header B0 (v59) and axis-aligned B0 (r3)
    uv run python bspinn/scripts/ocmr_eval.py --files data/ocmr/fs_0021*.h5 --slices mid --R 4 --recon tv \
        --maps nlinv pinn_avg --save-images --device cuda:0 --out runs/ocmr_v59/fig.jsonl
    uv run python bspinn/scripts/ocmr_eval.py --files data/ocmr/fs_0021*.h5 --slices mid --R 4 --recon tv \
        --maps pinn_avg --plane auto --save-images --device cuda:0 --out runs/r3_ocmr/fig.jsonl
    # non-Cartesian simulations with images (about 2 min on a GPU)
    uv run python bspinn/scripts/noncart_eval.py --synth --kinds radial spiral --size 128 --coils 8 --spokes 96 \
        --quality default --device cuda:0 --save --out runs/fig_noncart.jsonl

Output: fig2_static.png, fig3_cine.png, fig4_poles.png, fig5_noncart.png (300 dpi, also .pdf).
"""
import argparse
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

COL = {"espirit": "#8c8c8c", "nlinv": "#d9822b", "nlinv_pp": "#a85d12", "pinn": "#1f5fa8", "pinn_axis": "#8fb3dc"}
LAB = {"espirit": "ESPIRiT", "nlinv": "NLINV", "nlinv_pp": "NLINV + pole corr.", "pinn": "Proposed",
       "pinn_axis": "Proposed, axis-aligned B0"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.dpi": 300})


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return statistics.median(v) if v else float("nan")


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.join(out, name)}.png")


def letter(ax, s, dx=0.0):
    """Panel letter left of the axes' top-left corner; dx moves it further left (axes fraction)."""
    ax.text(-0.02 - dx, 1.0, s, transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right", va="bottom")


def imshow(ax, im, title=None, vmax=None, cmap="gray", **kw):
    ax.imshow(im, cmap=cmap, vmin=kw.pop("vmin", 0), vmax=vmax, interpolation="nearest", **kw)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, pad=2)


def fit_scale(x, ref, obj):
    """Least-squares magnitude scale of x onto ref inside obj (the NRMSE convention of the benchmark)."""
    x, r = np.abs(x), np.abs(ref)
    return x * (x[obj] * r[obj]).sum() / max((x[obj] ** 2).sum(), 1e-30)


def crop_box(mask, pad=6):
    r, c = np.nonzero(mask)
    if not r.size:
        return slice(None), slice(None)
    return (slice(max(r.min() - pad, 0), r.max() + pad + 1), slice(max(c.min() - pad, 0), c.max() + pad + 1))


def scatter_pair(ax, x, y, color, label, lo, hi):
    """Comparison (x) against proposed (y); points beyond the axis range sit on the edge as triangles."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ax.plot([lo, hi], [lo, hi], color="0.75", lw=0.8, zorder=0)
    inside = (x <= hi) & (y <= hi)
    ax.scatter(x[inside], y[inside], s=14, color=color, edgecolor="white", linewidth=0.4, label=label, zorder=2)
    if (~inside).any():
        ax.scatter(np.minimum(x[~inside], hi), np.minimum(y[~inside], hi), s=22, color=color, marker=">",
                   edgecolor="white", linewidth=0.4, zorder=2, clip_on=False)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")


# ------------------------------------------------------------------------------------------------ Fig. 2: static
def fig2(d, a):
    prior = [r for r in d["prior"] if r["method"] == a.static_prior]
    by = {(r["case"], r["maps"]): r["nrmse"] for r in prior}
    cases = sorted({r["case"] for r in prior})
    ev = {(r["case"], r["method"]): r for r in d["eval"]}
    real = sorted({r["case"] for r in d["eval"] if r.get("group") != "synthetic" and (r["case"], "pinn_v59") in ev})
    lco = lambda c, m: ev[(c, m)]["lco_outside_acs_over_floor"] - 1  # noqa: E731   excess over the noise floor

    example = None
    if a.eval:
        pick = a.static_case or min(cases, key=lambda c: abs(by.get((c, "pinn_v59"), 9) -
                                                           med([by.get((k, "pinn_v59")) for k in cases])))
        p = os.path.join(a.eval, pick, "figures.npz")
        if os.path.exists(p):
            example = (pick, np.load(p))
        else:
            print(f"fig2: {p} not found: example images skipped")

    fig = plt.figure(figsize=(7.2, 5.6 if example else 3.0))
    gs = GridSpec(2 if example else 1, 1, figure=fig, height_ratios=[1.15, 1] if example else [1], hspace=0.32)
    if example:
        name, f = example
        obj, ref = f["object"].astype(bool), np.abs(f["reference"])
        box = crop_box(obj)
        top = GridSpecFromSubplotSpec(2, 4, gs[0], hspace=0.04, wspace=0.03)
        vmax = np.percentile(ref[obj], 99.5)
        imshow(fig.add_subplot(top[0, 0]), ref[box], "Fully sampled", vmax)
        ax0 = fig.add_subplot(top[1, 0]); ax0.axis("off")
        ax0.text(0.5, 0.5, f"SENSE, R = 4\nerror ×{a.err_gain:g}\n({name.replace('_j4', '')})", ha="center",
                 va="center", transform=ax0.transAxes, fontsize=7, color="0.3")
        for j, (key, m) in enumerate((("espirit", "espirit"), ("nlinv", "nlinv"), ("pinn_v59", "pinn"))):
            if f"{key}/sense_R4" not in f.files:
                continue
            x = fit_scale(f[f"{key}/sense_R4"], ref, obj)
            e = np.linalg.norm((x - ref)[obj]) / np.linalg.norm(ref[obj])
            ax = fig.add_subplot(top[0, j + 1]); imshow(ax, x[box], LAB[m], vmax)
            ax.text(0.03, 0.96, f"{e:.3f}", transform=ax.transAxes, color="white", fontsize=6.5, va="top")
            imshow(fig.add_subplot(top[1, j + 1]), (np.abs(x - ref) * obj)[box] * a.err_gain, None, vmax)
        letter(fig.axes[0], "A")
    bot = GridSpecFromSubplotSpec(1, 2, gs[-1], wspace=0.35)

    ax = fig.add_subplot(bot[0])
    xs = [by[(c, "pinn_v59")] for c in cases if (c, "nlinv") in by and (c, "pinn_v59") in by]
    lo, hi = 0.9 * min(min(by.values()), 0.03), 1.05 * max(by.values())
    for m in ("espirit", "nlinv"):
        cc = [c for c in cases if (c, m) in by and (c, "pinn_v59") in by]
        win = sum(by[(c, "pinn_v59")] < by[(c, m)] for c in cc)
        scatter_pair(ax, [by[(c, m)] for c in cc], [by[(c, "pinn_v59")] for c in cc], COL[m],
                     f"{LAB[m]} ({med([by[(c, m)] for c in cc]):.3f}): {win}/{len(cc)}", lo, hi)
    ax.set_xlabel("ESPIRiT or NLINV sensitivities"); ax.set_ylabel("Proposed sensitivities")
    ax.set_title(f"Image NRMSE, TGV + phase, R = 4\n(proposed median {med(xs):.3f})")
    ax.legend(loc="upper left", handletextpad=0.1, title="median; proposed lower in", title_fontsize=6.5,
              alignment="left", fontsize=6.5)
    letter(ax, "B" if example else "A", 0.22)

    ax = fig.add_subplot(bot[1])
    vals = [lco(c, m) for c in real for m in ("nlinv", "pinn_v59") if (c, m) in ev]   # ESPIRiT outliers clipped
    lo, hi = min(0, min(vals)) - 0.01, 1.15 * max(vals)
    for m in ("espirit", "nlinv"):
        cc = [c for c in real if (c, m) in ev]
        win = sum(lco(c, "pinn_v59") < lco(c, m) for c in cc)
        scatter_pair(ax, [lco(c, m) for c in cc], [lco(c, "pinn_v59") for c in cc], COL[m],
                     f"{LAB[m]} ({med([lco(c, m) for c in cc]):.3f}): {win}/{len(cc)}", lo, hi)
    ax.set_xlabel("ESPIRiT or NLINV sensitivities"); ax.set_ylabel("Proposed sensitivities")
    ax.set_title("Leave-one-coil-out error above noise\n"
                 f"(proposed median {med([lco(c, 'pinn_v59') for c in real]):.3f}; ▶ off scale)")
    ax.legend(loc="upper left", handletextpad=0.1, title="median; proposed lower in", title_fontsize=6.5,
              alignment="left", fontsize=6.5)
    letter(ax, "C" if example else "B", 0.22)
    save(fig, a.out, "fig2_static")


# ------------------------------------------------------------------------------------------------ Fig. 3: cine
def fig3(d, a):
    rows = d["ocmr"] + [r for k in ("r3_ocmr",) for r in d.get("extra", {}).get(k, [])]
    run_hdr, run_ax = a.cine_run, a.cine_axis_run

    def get(run, maps, R):
        return {r["file"]: r["nrmse"] for r in rows if r.get("run") == run and r["maps"] == maps and r["R"] == R
                and r["recon"] == "tv"}

    us = [r for r in d.get("ocmr_us", []) if r["recon"] == "tv" and r["maps"] in ("espirit", "nlinv", "pinn_avg")]
    imgs = None
    if a.ocmr_images and os.path.exists(a.ocmr_images):
        imgs = np.load(a.ocmr_images)
        ax_imgs = np.load(a.ocmr_images_axis) if a.ocmr_images_axis and os.path.exists(a.ocmr_images_axis) else None
    elif a.ocmr_images:
        print(f"fig3: {a.ocmr_images} not found: example frames skipped")

    fig = plt.figure(figsize=(7.2, 5.4 if imgs is not None else 2.9))
    gs = GridSpec(2 if imgs is not None else 1, 1, figure=fig, height_ratios=[1, 1.05] if imgs is not None else [1],
                  hspace=0.35)
    top = GridSpecFromSubplotSpec(1, 3, gs[0], wspace=0.42, width_ratios=[1, 1, 0.9])
    series = (("nlinv", run_hdr, "nlinv"), ("pinn_axis", run_ax, "pinn_avg"), ("pinn", run_hdr, "pinn_avg"))
    for j, R in enumerate((4.0, 8.0)):
        ax = fig.add_subplot(top[j])
        data = [get(run, maps, R) for _, run, maps in series]
        files = sorted(set.intersection(*(set(v) for v in data)))
        for f in files:
            ax.plot(range(3), [v[f] for v in data], color="0.8", lw=0.7, zorder=1)
        for i, ((key, _, _), v) in enumerate(zip(series, data)):
            y = [v[f] for f in files]
            ax.scatter([i] * len(y), y, s=14, color=COL[key], edgecolor="white", linewidth=0.4, zorder=2)
            ax.plot([i - 0.25, i + 0.25], [med(y)] * 2, color="black", lw=1.2, zorder=3)
        meds = [med(list(v[f] for f in files)) for v in data]
        ax.set_xticks(range(3), [f"NLINV\n{meds[0]:.3f}", f"Proposed\naxis B0\n{meds[1]:.3f}",
                                 f"Proposed\nheader B0\n{meds[2]:.3f}"], fontsize=6.5)
        ax.set_xlim(-0.5, 2.5); ax.set_ylabel("Frame NRMSE (TV)")
        ax.set_title(f"Retrospective, R = {R:g}")
        if "fs_0021_3T" in files:
            y0 = data[1]["fs_0021_3T"]
            ax.annotate("3 T cine", (1, y0), (1.25, y0 + 0.012), fontsize=6, color="0.35", va="center",
                        arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5))
        letter(ax, "AB"[j], 0.3)
    ax = fig.add_subplot(top[2])
    files = sorted({r["file"] for r in us})
    w = 0.26
    for i, m in enumerate(("espirit", "nlinv", "pinn_avg")):
        key = "pinn" if m == "pinn_avg" else m
        v = {r["file"]: r["heldout"] for r in us if r["maps"] == m}
        ax.bar(np.arange(len(files)) + (i - 1) * w, [v.get(f, np.nan) for f in files], w, color=COL[key],
               label=LAB[key])
    ax.axhline(1, color="0.4", lw=0.7, ls="--"); ax.set_yscale("log")
    ax.set_xticks(range(len(files)), [f"cine {i + 1}" for i in range(len(files))])
    ax.set_ylabel("Error on withheld lines / noise"); ax.set_title("Prospective, R ≈ 9")
    ax.legend(loc="upper right", fontsize=6.5)
    letter(ax, "C", 0.25)

    if imgs is not None:
        ref = np.abs(imgs["ref"].astype(np.float32))
        T = ref.shape[-1]
        t = a.cine_frame if a.cine_frame is not None else T // 3
        obj = ref.mean(-1) > 0.08 * ref.mean(-1).max()
        box = crop_box(obj, 4) if a.cine_crop else (slice(None), slice(None))
        cols = [("Fully sampled", ref[..., t])]
        if "R4/nlinv/tv" in imgs.files:
            cols.append((LAB["nlinv"], imgs["R4/nlinv/tv"][..., t].astype(np.float32)))
        if ax_imgs is not None and "R4/pinn_avg/tv" in ax_imgs.files:
            cols.append(("Proposed, axis B0", ax_imgs["R4/pinn_avg/tv"][..., t].astype(np.float32)))
        if "R4/pinn_avg/tv" in imgs.files:
            cols.append(("Proposed, header B0", imgs["R4/pinn_avg/tv"][..., t].astype(np.float32)))
        bot = GridSpecFromSubplotSpec(2, len(cols), gs[1], hspace=0.04, wspace=0.03)
        vmax = np.percentile(cols[0][1][obj], 99.5)
        for j, (title, x) in enumerate(cols):
            x = fit_scale(x, cols[0][1], obj) if j else x
            ax = fig.add_subplot(bot[0, j]); imshow(ax, x[box], title, vmax)
            if j == 0:
                d_ax = ax
            if j:
                e = np.linalg.norm((x - cols[0][1])[obj]) / np.linalg.norm(cols[0][1][obj])
                ax.text(0.03, 0.96, f"NRMSE {e:.3f}", transform=ax.transAxes, color="white", fontsize=6.5, va="top")
                imshow(fig.add_subplot(bot[1, j]), (np.abs(x - cols[0][1]) * obj)[box] * a.err_gain, None, vmax)
            else:
                a0 = fig.add_subplot(bot[1, 0]); a0.axis("off")
                a0.text(0.5, 0.5, f"3 T cine, frame {t + 1}/{T}\nR = 4, TV\nerror ×{a.err_gain:g}", ha="center",
                        va="center", transform=a0.transAxes, fontsize=7, color="0.3")
        letter(d_ax, "D")
    save(fig, a.out, "fig3_cine")


# ------------------------------------------------------------------------------------------------ Fig. 4: poles
def fig4(d, a):
    rows = [r for r in d["poles"] if r["maps"] in ("nlinv", "nlinv_pp", "pinn_v59")]
    cals = sorted({r["cal"] for r in rows})
    methods = (("nlinv", "nlinv"), ("nlinv_pp", "nlinv_pp"), ("pinn_v59", "pinn"))
    have_maps = a.poles and os.path.isdir(os.path.join(a.poles, "maps"))
    if a.poles and not have_maps:
        print(f"fig4: {a.poles}/maps not found: phase maps skipped")
    fig = plt.figure(figsize=(7.2, 4.6) if have_maps else (3.6, 2.8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.75, 1], wspace=0.22) if have_maps else GridSpec(1, 1, figure=fig)
    gs_bar = gs[1] if have_maps else gs[0]
    if have_maps:
        from bspinn.recon import gauge_poles
        mask = None
        if a.brain:
            try:
                from phase_pole_eval import ifft2c, load_plane
                k = load_plane(os.path.join(a.brain, a.pole_file))
                rss = np.sqrt((np.abs(ifft2c(k)) ** 2).sum(0))
                mask = rss > a.pole_mask * rss.max()
            except Exception as e:          # noqa: BLE001  (the figure without a mask is still useful)
                print(f"fig4: could not read {a.pole_file} from {a.brain} ({e}); using the NLINV image as mask")
        sub = GridSpecFromSubplotSpec(len(methods), len(cals), gs[0], hspace=0.06, wspace=0.04)
        for i, (name, key) in enumerate(methods):
            for j, cal in enumerate(cals):
                ax = fig.add_subplot(sub[i, j])
                p = os.path.join(a.poles, "maps", f"{a.pole_file}_acs{cal}_{name}.npy")
                if not os.path.exists(p):
                    ax.axis("off"); continue
                C = np.load(p)
                m = mask
                if m is None:
                    q = os.path.join(a.poles, "maps", f"{a.pole_file}_acs{cal}_nlinv_img.npy")
                    if os.path.exists(q):
                        im = np.abs(np.load(q)); m = im > a.pole_mask * im.max()
                    else:
                        m = np.ones(C.shape[1:], bool)
                ph = np.angle(C.sum(0))                         # common (gauge) phase: where the poles show
                show = np.where(m, ph, np.nan)
                ax.imshow(show, cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="nearest")
                ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                for (r, c, _) in gauge_poles(C, support=m):
                    ax.add_patch(plt.Circle((c, r), 9, fill=False, color="white", lw=1.1))
                if i == 0:
                    ax.set_title(f"{cal}×{cal} calibration")
                if j == 0:
                    ax.set_ylabel(LAB[key], fontsize=7.5)
        letter(fig.axes[0], "A")
    ax = fig.add_subplot(gs_bar)
    w = 0.26
    for i, (name, key) in enumerate(methods):
        y = [med([r["sense_nrmse"] for r in rows if r["maps"] == name and r["cal"] == c]) for c in cals]
        poles = [sum(r["gauge_poles"] for r in rows if r["maps"] == name and r["cal"] == c) for c in cals]
        x = np.arange(len(cals)) + (i - 1) * w
        ax.bar(x, y, w, color=COL[key], label=LAB[key])
        for xx, yy, pp in zip(x, y, poles):
            if pp:
                ax.text(xx, yy + 0.003, f"{pp}●", ha="center", va="bottom", fontsize=6.5, color=COL[key])
    ax.set_xticks(range(len(cals)), [f"{c}×{c}" for c in cals]); ax.set_xlabel("Calibration region")
    ax.set_ylabel("SENSE NRMSE (median of 3 planes)")
    ax.set_title("SENSE error; ● phase singularities\n(number, over 3 planes)")
    ax.legend(loc="upper right", fontsize=6.5)
    letter(ax, "B" if have_maps else "A", 0.25)
    save(fig, a.out, "fig4_poles")


# ------------------------------------------------------------------------------------------------ Fig. 5: noncart
def fig5(d, a):
    rows = {r["case"]: r for r in d.get("extra", {}).get("r4_noncart", [])}
    kinds = [k for k in ("radial", "spiral") if a.noncart and os.path.exists(f"{a.noncart}_{k}.npz")]
    if not kinds:
        print(f"fig5: {a.noncart}_radial.npz / _spiral.npz not found (see the noncart_eval.py --save command "
              "at the top); numbers only")
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        for i, k in enumerate(("radial", "spiral")):
            r = rows.get(f"synth_{k}")
            if r:
                ax.bar([i - 0.18, i + 0.18], [r["nrmse_pinn_tv"], r["nrmse_true_tv"]], 0.34,
                       color=[COL["pinn"], "0.3"], label=None if i else None)
        ax.set_xticks([0, 1], ["Radial", "Spiral"]); ax.set_ylabel("TV image NRMSE")
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL["pinn"]), plt.Rectangle((0, 0), 1, 1, color="0.3")],
                  labels=["Estimated sensitivities", "True sensitivities"])
        save(fig, a.out, "fig5_noncart")
        return
    fig = plt.figure(figsize=(7.2, 1.3 * len(kinds) + 0.45))
    gs = GridSpec(len(kinds), 6, figure=fig, wspace=0.08, hspace=0.12, width_ratios=[1, 1, 1, 1, 1, 1])
    for i, k in enumerate(kinds):
        f = np.load(f"{a.noncart}_{k}.npz")
        sup, ref = f["support"].astype(bool), np.abs(f["x_ref"])
        vmax = np.percentile(ref[sup], 99.5)
        tr = f["traj"]
        ax = fig.add_subplot(gs[i, 0])
        step = max(1, tr.shape[0] // (16 if k == "radial" else 6))
        for s in range(0, tr.shape[0], step):
            ax.plot(tr[s, :, 0], tr[s, :, 1], lw=0.5, color=COL["pinn"])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_ylabel(f"{k.capitalize()}\n({tr.shape[0]} {'spokes' if k == 'radial' else 'interleaves'})",
                      fontsize=7.5)
        if i == 0:
            ax.set_title("k-space\n(every n-th shot)")
        Cp, Ct = f["maps_pinn"], f["maps_true"]
        nrm = lambda C: C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)  # noqa: E731
        Cp, Ct = nrm(Cp), nrm(Ct)
        g = np.vdot(Cp[0][sup], Ct[0][sup]); g = g / max(abs(g), 1e-30)       # common phase for display
        imshow(fig.add_subplot(gs[i, 1]), np.abs(Ct[0]) * sup, "True map\n(coil 1)" if i == 0 else None,
               cmap="magma")
        imshow(fig.add_subplot(gs[i, 2]), np.abs(Cp[0] * g) * sup, "Estimated map\n(coil 1)" if i == 0 else None, cmap="magma")
        xp = fit_scale(f["pinn_tv"], ref, sup); xt = fit_scale(f["true_tv"], ref, sup)
        e = lambda x: np.linalg.norm((x - ref)[sup]) / np.linalg.norm(ref[sup])  # noqa: E731
        ax = fig.add_subplot(gs[i, 3]); imshow(ax, xt * sup, "TV,\ntrue maps" if i == 0 else None, vmax)
        ax.text(0.03, 0.96, f"NRMSE {e(xt):.3f}", transform=ax.transAxes, color="white", fontsize=6.5, va="top")
        ax = fig.add_subplot(gs[i, 4])
        imshow(ax, xp * sup, "TV,\nestimated maps" if i == 0 else None, vmax)
        ax.text(0.03, 0.96, f"NRMSE {e(xp):.3f}", transform=ax.transAxes, color="white", fontsize=6.5, va="top")
        imshow(fig.add_subplot(gs[i, 5]), np.abs(xp - xt) * sup * a.err_gain_noncart,
               f"Difference\n×{a.err_gain_noncart:g}" if i == 0 else None, vmax)
    save(fig, a.out, "fig5_noncart")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dashboard_data.json")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--figs", nargs="+", default=["2", "3", "4", "5"])
    ap.add_argument("--err-gain", type=float, default=5.0, help="error maps are shown this many times brighter")
    ap.add_argument("--eval", default=None, help="evaluate_methods.py output folder (runs/eval_v59): example slice")
    ap.add_argument("--static-case", default=None, help="case for the example images (default: median NRMSE)")
    ap.add_argument("--static-prior", default="tgv_phase_auto")
    ap.add_argument("--cine-run", default="ocmr_v59", help="run label with B0 from the header")
    ap.add_argument("--cine-axis-run", default="r3_ocmr", help="run label with axis-aligned B0 (plane search)")
    ap.add_argument("--ocmr-images", default=None)
    ap.add_argument("--ocmr-images-axis", default=None)
    ap.add_argument("--cine-frame", type=int, default=None)
    ap.add_argument("--cine-crop", action="store_true", help="crop the cine frames to the object")
    ap.add_argument("--poles", default=None, help="phase_pole_eval.py output folder (runs/poles_v59)")
    ap.add_argument("--brain", default=None, help="folder with ksp_2D* (object mask for the phase maps)")
    ap.add_argument("--pole-file", default="ksp_2D")
    ap.add_argument("--pole-mask", type=float, default=0.06)
    ap.add_argument("--err-gain-noncart", type=float, default=20.0)
    ap.add_argument("--noncart", default=None, help="noncart_eval.py --save output base (runs/fig_noncart)")
    a = ap.parse_args()
    d = json.load(open(a.data))
    for n in a.figs:
        {"2": fig2, "3": fig3, "4": fig4, "5": fig5}[n](d, a)


if __name__ == "__main__":
    main()
