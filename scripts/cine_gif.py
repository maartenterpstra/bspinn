"""Animated GIF comparing the cine reconstructions and coil maps of several methods.

One row per method (ESPIRiT, NLINV, proposed time-averaged, proposed per-frame, as available), and
the fully sampled cine on top. The columns are:

  reconstruction                   magnitude, scaled to the fully sampled frame (least squares in the object)
  |reconstruction - fully sampled| x err_gain
  |reconstruction - proposed|      x err_gain: where the methods' images differ
  coil map j                       hue = phase, brightness = magnitude, gauge-aligned to the proposed maps
  map difference to proposed       sqrt(sum_j |c_j - e^{i phi} p_j|^2) per pixel; phi is the per-pixel common
                                   phase (the gauge freedom), so only real differences remain; x map_gain

Used by ocmr_eval.py --save-gif, or stand-alone on the files ocmr_eval.py --save-images left behind:

    uv run python bspinn/scripts/cine_gif.py --images runs/ocmr_v59/fs_0021_3T_s0_images.npz \
        --maps-dir runs/ocmr_v59/maps --R 4 --out runs/ocmr_v59/gifs
"""
import argparse
import colorsys
import os

import numpy as np

LABELS = {"espirit": "ESPIRiT", "nlinv": "NLINV", "pinn_avg": "Proposed", "pinn_dyn": "Proposed, per frame",
          "pinn_dyn0": "Proposed (dyn, rank 0)", "pinn_frame": "Proposed, per frame (2-D)"}
ORDER = ["espirit", "nlinv", "pinn_avg", "pinn_dyn", "pinn_dyn0", "pinn_frame"]


def complex_rgb(z, vmax, mask=None):
    h = (np.angle(z) + np.pi) / (2 * np.pi)
    v = np.clip(np.abs(z) / max(vmax, 1e-12), 0, 1)
    rgb = np.stack(np.vectorize(colorsys.hsv_to_rgb, otypes=[float, float, float])(h, 0.85 * np.ones_like(h), v), -1)
    return rgb * mask[..., None] if mask is not None else rgb


def normalise(C):
    return C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), 1e-12)


def align(C, P):
    """Remove the per-pixel common phase of C relative to P (both (N,X,Y), RSS-normalised); return the aligned C
    and the gauge-free difference sqrt(sum_j |c_j - p_j|^2) after alignment."""
    ph = np.exp(-1j * np.angle((C * P.conj()).sum(0)))
    Ca = C * ph[None]
    return Ca, np.sqrt((np.abs(Ca - P) ** 2).sum(0))


def crop_box(mask, pad=4):
    r, c = np.nonzero(mask)
    if not r.size:
        return slice(None), slice(None)
    return slice(max(r.min() - pad, 0), r.max() + pad + 1), slice(max(c.min() - pad, 0), c.max() + pad + 1)


def fit_scale(x, ref, obj):
    return x * (x[obj] * ref[obj]).sum() / max((x[obj] ** 2).sum(), 1e-30)


def make_cine_gif(ref, recons, maps, out, base="pinn_avg", coil=None, err_gain=5.0, map_gain=None, fps=10,
                  dpi=90, crop=True, mask_level=0.08, title="", frame_png=True):
    """ref (X,Y,T) magnitude; recons {method: (X,Y,T)}; maps {method: (N,X,Y) or (N,X,Y,T)}. Writes out (.gif) and,
    with frame_png, a PNG of the frame at T // 3 next to it. Returns a dict with per-method numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ref = np.abs(np.asarray(ref, np.float32))
    X, Y, T = ref.shape
    obj = ref.mean(-1) > mask_level * ref.mean(-1).max()
    box = crop_box(obj) if crop else (slice(None), slice(None))
    methods = [m for m in ORDER if m in recons] + [m for m in recons if m not in ORDER]
    if base not in recons:
        base = methods[-1]
    rec = {m: np.stack([fit_scale(np.abs(recons[m][..., t]).astype(np.float32), ref[..., t], obj) for t in range(T)], -1)
           for m in methods}
    vmax = np.percentile(ref[obj], 99.5)
    M = {}
    for m in methods:
        if m in maps:
            C = normalise(np.asarray(maps[m]))
            M[m] = C if C.ndim == 4 else C[..., None]                 # (N,X,Y,T or 1)
    stats = {m: {"nrmse": float(np.linalg.norm((rec[m] - ref)[obj]) / np.linalg.norm(ref[obj])),
                 "diff_to_base": float(np.linalg.norm((rec[m] - rec[base])[obj]) / np.linalg.norm(ref[obj]))}
             for m in methods}
    P = M.get(base)
    if P is not None:
        if coil is None:                                              # the coil with most signal in the object
            coil = int(np.argmax((np.abs(P[..., 0]) ** 2)[:, obj].sum(1)))
        for m in M:
            d = [align(M[m][..., min(t, M[m].shape[-1] - 1)], P[..., min(t, P.shape[-1] - 1)])[1][obj].mean()
                 for t in range(0, T, max(1, T // 8))]
            stats[m]["map_diff_to_base"] = float(np.mean(d))
        md = [stats[m]["map_diff_to_base"] for m in M if m != base]
        map_gain = map_gain or (0.5 / max(float(np.median(md)) if md else 0.0, 1e-3))
    cmax = float(np.percentile(np.abs(P[coil][obj]), 99.5)) if P is not None else 1.0

    cols = ["Reconstruction", f"Error ×{err_gain:g}", f"Difference to {LABELS.get(base, base)} ×{err_gain:g}"]
    if M:
        cols += [f"Coil {coil + 1} map", f"Map difference ×{map_gain:.0f}"]
    nr, nc = len(methods) + 1, len(cols)
    h, w = (box[0].stop or X) - (box[0].start or 0), (box[1].stop or Y) - (box[1].start or 0)
    fig, axs = plt.subplots(nr, nc, figsize=(1.45 * nc, 1.45 * nr * h / w + 0.5), squeeze=False)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
    for j, c in enumerate(cols):
        axs[0, j].set_title(c, fontsize=6.5, pad=2)
    axs[0, 0].set_ylabel("Fully sampled", fontsize=6.5)
    upd = []
    upd.append((axs[0, 0].imshow(ref[..., 0][box], cmap="gray", vmin=0, vmax=vmax), lambda t: ref[..., t][box]))
    for j in range(1, nc):
        axs[0, j].axis("off")
    for i, m in enumerate(methods, start=1):
        st = stats[m]
        axs[i, 0].set_ylabel(f"{LABELS.get(m, m)}\nNRMSE {st['nrmse']:.3f}", fontsize=6)
        r = rec[m]
        upd.append((axs[i, 0].imshow(r[..., 0][box], cmap="gray", vmin=0, vmax=vmax), (lambda r: lambda t: r[..., t][box])(r)))
        upd.append((axs[i, 1].imshow((np.abs(r[..., 0] - ref[..., 0]) * obj)[box], cmap="gray", vmin=0, vmax=vmax / err_gain),
                    (lambda r: lambda t: (np.abs(r[..., t] - ref[..., t]) * obj)[box])(r)))
        if m == base:
            axs[i, 2].text(0.5, 0.5, "reference\nfor differences", ha="center", va="center", fontsize=6, color="0.4",
                           transform=axs[i, 2].transAxes)
        else:
            upd.append((axs[i, 2].imshow((np.abs(r[..., 0] - rec[base][..., 0]) * obj)[box], cmap="gray", vmin=0,
                                         vmax=vmax / err_gain),
                        (lambda r: lambda t: (np.abs(r[..., t] - rec[base][..., t]) * obj)[box])(r)))
            axs[i, 2].text(0.03, 0.03, f"{st['diff_to_base']:.3f}", color="white", fontsize=5.5,
                           transform=axs[i, 2].transAxes)
        if m in M and P is not None:
            def cm(t, m=m):
                C = M[m][..., min(t, M[m].shape[-1] - 1)]
                Pt = P[..., min(t, P.shape[-1] - 1)]
                Ca, d = align(C, Pt)
                return Ca[coil], d
            Ca0, d0 = cm(0)
            upd.append((axs[i, 3].imshow(complex_rgb(Ca0[box], cmax, obj[box])),
                        (lambda cm: lambda t: complex_rgb(cm(t)[0][box], cmax, obj[box]))(cm)))
            if m == base:
                axs[i, 4].text(0.5, 0.5, "reference\nfor differences", ha="center", va="center", fontsize=6,
                               color="0.4", transform=axs[i, 4].transAxes)
            else:
                upd.append((axs[i, 4].imshow((d0 * obj)[box], cmap="magma", vmin=0, vmax=1.0 / map_gain),
                            (lambda cm: lambda t: (cm(t)[1] * obj)[box])(cm)))
                axs[i, 4].text(0.03, 0.03, f"{st['map_diff_to_base']:.3f}", color="white", fontsize=5.5,
                               transform=axs[i, 4].transAxes)
        elif M:
            axs[i, 3].axis("off"); axs[i, 4].axis("off")
    lab = fig.text(0.5, 0.005, "", ha="center", fontsize=6.5, color="0.3")
    fig.suptitle(title, fontsize=7.5, y=0.995)
    fig.tight_layout(pad=0.25, h_pad=0.15, w_pad=0.1, rect=(0, 0.025, 1, 1))

    def draw(t):
        for im, f in upd:
            im.set_data(f(t))
        lab.set_text(f"frame {t + 1}/{T}   (difference numbers: mean over the object; map difference 0 = identical, "
                     f"√2 = orthogonal)")
        return [im for im, _ in upd] + [lab]

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    FuncAnimation(fig, draw, frames=T, blit=False).save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    if frame_png:
        draw(T // 3)
        fig.savefig(os.path.splitext(out)[0] + "_frame.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)", flush=True)
    return stats


def load_maps(maps_dir, tag, methods):
    out = {}
    for m in methods:
        p = os.path.join(maps_dir, f"{tag}_{m}.npy")
        if os.path.exists(p):
            out[m] = np.load(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, help="<file>_s<slice>_images.npz from ocmr_eval.py --save-images")
    ap.add_argument("--maps-dir", default=None, help="the maps cache of the same run (default: <images dir>/maps)")
    ap.add_argument("--R", type=float, nargs="+", default=None, help="default: every R in the file")
    ap.add_argument("--recon", default="tv")
    ap.add_argument("--base", default="pinn_avg")
    ap.add_argument("--coil", type=int, default=None, help="0-based (default: strongest coil)")
    ap.add_argument("--err-gain", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--dpi", type=int, default=90)
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--out", default=None, help="folder (default: <images dir>/gifs)")
    a = ap.parse_args()
    for p in a.images:
        f = np.load(p)
        root = os.path.dirname(os.path.abspath(p))
        stem = os.path.basename(p).replace("_images.npz", "")
        Rs = a.R or sorted({float(k.split("/")[0][1:]) for k in f.files if k.startswith("R") and "/" in k})
        for R in Rs:
            recons = {k.split("/")[1]: f[k] for k in f.files if k.startswith(f"R{R:g}/") and k.endswith(f"/{a.recon}")}
            if not recons:
                print(f"{p}: no R = {R:g} {a.recon} images"); continue
            maps = load_maps(a.maps_dir or os.path.join(root, "maps"), f"{stem}_R{R:g}", recons)
            make_cine_gif(f["ref"], recons, maps, os.path.join(a.out or os.path.join(root, "gifs"), f"{stem}_R{R:g}.gif"),
                          base=a.base, coil=a.coil, err_gain=a.err_gain, fps=a.fps, dpi=a.dpi, crop=not a.no_crop,
                          title=f"{stem}, R = {R:g}, {a.recon.upper()}")


if __name__ == "__main__":
    main()
