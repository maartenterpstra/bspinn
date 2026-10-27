"""Animated comparison of cine reconstructions and coil maps: methods along the columns, the coil-combined cine in
the first row, then one row per chosen coil channel (hue = phase, brightness = magnitude), all over time.

    from show_cine import show_cine
    data_dic = {"ESPIRiT": {"recon": x_esp, "csm": C_esp}, "NLINV": {...}, "Proposed": {...}}
    show_cine(data_dic, channels=[0, 5, 12], out="cine.gif", ref=ref)

recon: (x, y, frames), complex or magnitude.  csm: (coils, x, y) time-averaged or (coils, x, y, frames).
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import hsv_to_rgb  # noqa: E402


def complex_to_rgb(z, vmax=1.0, mask=None, saturation=0.85):
    """Hue = phase, brightness = magnitude / vmax (black = 0)."""
    hsv = np.stack([(np.angle(z) + np.pi) / (2 * np.pi), np.full(z.shape, saturation),
                    np.clip(np.abs(z) / vmax, 0, 1)], -1)
    rgb = hsv_to_rgb(hsv)
    return rgb * mask[..., None] if mask is not None else rgb


def rss_normalise(C, eps=1e-12):
    return C / np.maximum(np.sqrt((np.abs(C) ** 2).sum(0, keepdims=True)), eps)


def _frames(C, T):
    """(coils, x, y) or (coils, x, y, T) -> (coils, x, y, T)."""
    C = np.asarray(C)
    return np.repeat(C[..., None], T, -1) if C.ndim == 3 else C


def show_cine(data_dic, channels=(0,), out="cine.gif", ref=None, mask=None, phase="global", align_to=None,
              fps=10, dpi=100, crop=None, panel=1.6, frame_png=True):
    """data_dic[method] = {'recon': (x, y, T), 'csm': (coils, x, y[, T])}; returns the figure.

    channels : coil indices shown in rows 2.. (same channel for every method: the same physical coil).
    ref      : optional fully sampled cine (x, y, T); every recon is then scaled to it (least squares in the mask)
               and it sets the common grey scale. Without it, every recon is scaled to the first method's.
    mask     : optional (x, y) bool; coil maps black outside (default: where the time-averaged image > 8 % of max).
    phase    : 'global' = one phase rotation per method (from the coil sum over all pixels and frames; keeps
               every relative phase, phase singularities included); 'pixel' = the per-pixel common phase
               removed relative to ``align_to`` (default: the last method), so identical maps look identical;
               None = as given.
    crop     : (slice, slice) to zoom, e.g. (slice(40, 150), slice(20, 130)).
    Brightness: maps are RSS-normalised, so vmax = 1 for every panel; recon panels share one grey scale.
    """
    methods = list(data_dic)
    X, Y, T = np.asarray(data_dic[methods[0]]["recon"]).shape
    box = crop or (slice(None), slice(None))

    # reconstructions: one scale per method, one grey level for all
    base = np.abs(np.asarray(ref if ref is not None else data_dic[methods[0]]["recon"], dtype=np.complex64))
    if mask is None:
        mask = base.mean(-1) > 0.08 * base.mean(-1).max()
    m3 = np.repeat(mask[..., None], T, -1)
    rec = {}
    for k in methods:
        x = np.abs(np.asarray(data_dic[k]["recon"]))
        rec[k] = x * (x[m3] * base[m3]).sum() / max((x[m3] ** 2).sum(), 1e-30)
    gmax = np.percentile(base[m3], 99.5)

    # coil maps: RSS = 1, then the phase convention
    csm = {k: rss_normalise(_frames(data_dic[k]["csm"], T)) for k in methods}
    if phase == "global":
        for k in methods:
            csm[k] = csm[k] * np.exp(-1j * np.angle(csm[k].sum(0)[m3].sum()))
    elif phase == "pixel":
        P = csm[align_to or methods[-1]]
        for k in methods:
            csm[k] = csm[k] * np.exp(-1j * np.angle((csm[k] * P.conj()).sum(0)))[None]

    nr, nc = 1 + len(channels), len(methods) + (1 if ref is not None else 0)
    h = (box[0].stop or X) - (box[0].start or 0)
    w = (box[1].stop or Y) - (box[1].start or 0)
    fig, axs = plt.subplots(nr, nc, figsize=(panel * nc, panel * nr * h / w + 0.4), squeeze=False)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    upd = []
    c0 = 0
    if ref is not None:                                   # first column: the fully sampled cine
        axs[0, 0].set_title("Fully sampled", fontsize=8)
        upd.append((axs[0, 0].imshow(base[..., 0][box], cmap="gray", vmin=0, vmax=gmax),
                    lambda t: base[..., t][box]))
        for r in range(1, nr):
            axs[r, 0].axis("off")
        c0 = 1
    for j, k in enumerate(methods):
        c = c0 + j
        axs[0, c].set_title(k, fontsize=8)
        upd.append((axs[0, c].imshow(rec[k][..., 0][box], cmap="gray", vmin=0, vmax=gmax),
                    (lambda k: lambda t: rec[k][..., t][box])(k)))
        for i, ch in enumerate(channels, start=1):
            upd.append((axs[i, c].imshow(complex_to_rgb(csm[k][ch, ..., 0][box], 1.0, mask[box])),
                        (lambda k, ch: lambda t: complex_to_rgb(csm[k][ch, ..., t][box], 1.0, mask[box]))(k, ch)))
    axs[0, c0].set_ylabel("Reconstruction", fontsize=8)
    for i, ch in enumerate(channels, start=1):
        axs[i, c0].set_ylabel(f"Coil {ch + 1}", fontsize=8)
    lab = fig.text(0.5, 0.005, "", ha="center", fontsize=7, color="0.3")
    fig.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.1, rect=(0, 0.03, 1, 1))

    def draw(t):
        for im, f in upd:
            im.set_data(f(t))
        lab.set_text(f"frame {t + 1}/{T}   coil maps: hue = phase, brightness = magnitude (RSS-normalised)")
        return [im for im, _ in upd] + [lab]

    if out:
        FuncAnimation(fig, draw, frames=T, blit=False).save(out, writer=PillowWriter(fps=fps), dpi=dpi)
        if frame_png:
            draw(T // 3)
            fig.savefig(out.rsplit(".", 1)[0] + "_frame.png", dpi=300, bbox_inches="tight", facecolor="white")
    return fig
