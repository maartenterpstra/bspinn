"""Cartesian undersampling masks with a fully sampled calibration (ACS) region.

    from bspinn.masks import cartesian_mask
    m = cartesian_mask((320, 368), R=4, acs=24)                              # 2-D: lines along axis 1
    m = cartesian_mask((320, 368), R=4, acs=24, pattern="vd", seed=1)        # variable-density random lines
    m = cartesian_mask((64, 96, 96), R=(2, 3), acs=24, pattern="caipi")     # 3-D: R1 x R2 over axes 1, 2
    m = cartesian_mask((64, 96, 96), R=6, acs=24, pattern="poisson")        # 3-D: variable-density Poisson disc
    m = cartesian_mask((128, 128, 20), R=4, acs=16, layout="dynamic")       # (x, y, t): a pattern per frame

The mask has the spatial (and temporal) shape of one coil and is True where a sample is acquired.
The readout direction is always fully sampled. ``acs`` is the size of the central fully sampled
block per phase-encoding axis; with ``estimate_maps(..., undersample=...)`` it defaults to the
calibration size of the fit, so the calibration region is always acquired.

Patterns (phase-encoding lines in 2-D / dynamic, phase-encoding plane in 3-D):
  'uniform'  every R-th line (R1 x R2 grid in 3-D), centred on k = 0           (GRAPPA/SENSE style)
  'random'   lines drawn uniformly at random outside the ACS                     (exact line count)
  'vd'       variable-density random: line probability ~ (1 - |k|/kmax)^power   (compressed-sensing style)
  'caipi'    3-D only: R1 x R2 lattice with a shift of ``caipi_shift`` per row     (CAIPIRINHA)
  'poisson'  3-D only: variable-density Poisson disc over the phase-encoding plane
Dynamic data: 'uniform' shifts the lattice every frame (interleaved, t mod R), the random patterns
draw a new realisation per frame.

``R`` is the outer acceleration for 'uniform' / 'caipi' (spacing of the lattice) and the target net
acceleration over the acquired extent, ACS included, for the random patterns. ``partial_fourier``
(< 1) drops the far end of the first phase-encoding axis. ``extent`` restricts sampling to an existing
acquisition (e.g. the non-zero entries of zero-padded k-space).
"""
from __future__ import annotations

import numpy as np


def cartesian_mask(shape, R=4, acs=24, pattern="uniform", layout=None, pe_axes=None, partial_fourier=1.0,
                   power=2.0, caipi_shift=1, seed=0, extent=None):
    """Returns a bool mask of ``shape``. ``layout`` '2d' | '3d' | 'dynamic' (default: 2-D for 2 dims,
    3-D for 3 dims). ``pe_axes``: phase-encoding axes (default 2-D/dynamic (1,), 3-D (1, 2)); the
    remaining spatial axis is the readout (fully sampled). Dynamic: the last axis is time."""
    shape = tuple(int(s) for s in shape)
    layout = layout or ("2d" if len(shape) == 2 else "3d")
    rng = np.random.default_rng(seed)
    if layout == "dynamic":
        sp, T = shape[:-1], shape[-1]
        ext = None if extent is None else np.asarray(extent, bool)
        frames = []
        for t in range(T):
            e_t = None if ext is None else (ext[..., t] if ext.ndim == len(shape) else ext)
            frames.append(_mask_2d(sp, R, acs, pattern, pe_axes or (1,), partial_fourier, power,
                                   np.random.default_rng(seed + 7919 * t), e_t, shift=t))
        return np.stack(frames, -1)
    if layout == "2d":
        return _mask_2d(shape, R, acs, pattern, pe_axes or (1,), partial_fourier, power, rng, extent, shift=0)
    if layout == "3d":
        return _mask_3d(shape, R, acs, pattern, pe_axes or (1, 2), partial_fourier, power, caipi_shift, rng, extent)
    raise ValueError("layout must be '2d', '3d' or 'dynamic'")


def effective_acceleration(mask, extent=None):
    """Acquired-extent samples / acquired samples (1 = fully sampled)."""
    m = np.asarray(mask, bool)
    e = np.ones_like(m) if extent is None else np.asarray(extent, bool)
    return float(e.sum() / max(1, (m & e).sum()))


# ------------------------------------------------------------------------------------------ helpers
def _centred_block(n, a):
    sel = np.zeros(n, bool)
    a = min(int(a), n)
    sel[n // 2 - a // 2: n // 2 - a // 2 + a] = True
    return sel


def _line_extent(n, extent_prof, partial_fourier):
    """Lines that may be acquired: the existing acquisition (if any), minus the partial-Fourier end."""
    ok = np.ones(n, bool) if extent_prof is None else extent_prof.copy()
    if partial_fourier < 1.0:
        keep = int(np.ceil(partial_fourier * n))
        ok[keep:] = False                                   # drop the far (high +k) end
        if keep <= n // 2:
            raise ValueError("partial_fourier must keep more than half of k-space")
    return ok


def _choose_lines(n, R, acs, pattern, power, rng, allowed, shift=0):
    lines = _centred_block(n, acs) & allowed
    if pattern == "uniform":
        R = int(round(R))
        lat = np.zeros(n, bool)
        lat[(n // 2 + shift) % R::R] = True
        return lines | (lat & allowed)
    if pattern in ("random", "vd"):
        target = int(round(allowed.sum() / float(R)))
        need = max(0, target - int(lines.sum()))
        cand = np.nonzero(allowed & ~lines)[0]
        if need >= cand.size:
            return lines | allowed
        if pattern == "random":
            p = np.ones(cand.size)
        else:
            k = np.abs(cand - n // 2) / (n / 2)
            p = np.clip(1 - k, 1e-3, None) ** power
        pick = rng.choice(cand, size=need, replace=False, p=p / p.sum())
        lines[pick] = True
        return lines
    raise ValueError(f"pattern {pattern!r} is not available here (2-D/dynamic: uniform | random | vd)")


def _mask_2d(shape, R, acs, pattern, pe_axes, partial_fourier, power, rng, extent, shift=0):
    if len(shape) != 2:
        raise ValueError("2-D masks need a 2-D shape")
    a = pe_axes[0]
    n = shape[a]
    ext = None if extent is None else np.asarray(extent, bool)
    prof = None if ext is None else ext.any(axis=1 - a)
    acs_a = acs[0] if isinstance(acs, (tuple, list)) else acs
    lines = _choose_lines(n, R, acs_a, pattern, power, rng, _line_extent(n, prof, partial_fourier), shift)
    m = np.broadcast_to(lines[None, :] if a == 1 else lines[:, None], shape).copy()
    return m if ext is None else m & ext


def _mask_3d(shape, R, acs, pattern, pe_axes, partial_fourier, power, caipi_shift, rng, extent):
    if len(shape) != 3:
        raise ValueError("3-D masks need a 3-D shape")
    a1, a2 = pe_axes
    ro = ({0, 1, 2} - {a1, a2}).pop()
    n1, n2 = shape[a1], shape[a2]
    ext = None if extent is None else np.asarray(extent, bool)
    plane_ext = None if ext is None else ext.any(axis=ro)                         # (n1, n2) in axis order
    if plane_ext is not None and a1 > a2:
        plane_ext = plane_ext.T
    allowed = np.ones((n1, n2), bool) if plane_ext is None else plane_ext.copy()
    if partial_fourier < 1.0:
        allowed &= _line_extent(n1, None, partial_fourier)[:, None]
    c1, c2 = (acs, acs) if not isinstance(acs, (tuple, list)) else acs
    centre = np.outer(_centred_block(n1, c1), _centred_block(n2, c2))
    R1, R2 = (R if isinstance(R, (tuple, list)) else _split_R(R))
    p = np.zeros((n1, n2), bool)
    if pattern in ("uniform", "caipi"):
        R1, R2 = int(round(R1)), int(round(R2))
        i = np.arange(n1)[:, None]; j = np.arange(n2)[None, :]
        off = (caipi_shift * ((i - n1 // 2) // R1)) % R2 if pattern == "caipi" else 0
        p = (((i - n1 // 2) % R1) == 0) & (((j - n2 // 2 - off) % R2) == 0)
    elif pattern in ("random", "vd"):
        target = int(round(allowed.sum() / float(R1 * R2)))
        cand = np.flatnonzero(allowed & ~centre)
        need = max(0, target - int((centre & allowed).sum()))
        if pattern == "random":
            w = np.ones(cand.size)
        else:
            k1, k2 = np.unravel_index(cand, (n1, n2))
            r = np.sqrt(((k1 - n1 // 2) / (n1 / 2)) ** 2 + ((k2 - n2 // 2) / (n2 / 2)) ** 2) / np.sqrt(2)
            w = np.clip(1 - r, 1e-3, None) ** power
        if need < cand.size:
            p.ravel()[rng.choice(cand, size=need, replace=False, p=w / w.sum())] = True
        else:
            p = allowed.copy()
    elif pattern == "poisson":
        p = _poisson_disc((n1, n2), R1 * R2, power, rng, allowed & ~centre, centre_count=int((centre & allowed).sum()))
    else:
        raise ValueError(f"pattern {pattern!r}: 3-D uses uniform | caipi | random | vd | poisson")
    plane = (p | centre) & allowed
    if a1 > a2:
        plane = plane.T
    m = np.expand_dims(plane, ro)
    m = np.broadcast_to(m, shape).copy()
    return m if ext is None else m & ext


def _split_R(R):
    """Scalar R for a 2-D phase-encoding plane -> (R1, R2) with R1 * R2 ~ R (R1 >= R2)."""
    R = float(R)
    r2 = max(1, int(np.floor(np.sqrt(R))))
    while r2 > 1 and abs(round(R / r2) * r2 - R) > 1e-6:
        r2 -= 1
    return (R / r2, r2)


def _poisson_disc(shape, R, power, rng, allowed, centre_count=0, iters=12):
    """Variable-density Poisson disc on a grid: random order, accept a point if no accepted point lies
    within r(k) = r0 * (1 + power * |k|); r0 bisected so that the net acceleration is ~R."""
    n1, n2 = shape
    k1, k2 = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    kr = np.sqrt(((k1 - n1 // 2) / (n1 / 2)) ** 2 + ((k2 - n2 // 2) / (n2 / 2)) ** 2) / np.sqrt(2)
    order = rng.permutation(np.flatnonzero(allowed))
    target = int(round((allowed.sum() + centre_count) / float(R))) - centre_count

    def draw(r0):
        acc = np.zeros(shape, bool)
        rad = r0 * (1 + power * kr)
        for f in order:
            i, j = divmod(int(f), n2)
            r = rad[i, j]
            ri = int(np.ceil(r))
            i0, i1, j0, j1 = max(0, i - ri), min(n1, i + ri + 1), max(0, j - ri), min(n2, j + ri + 1)
            win = acc[i0:i1, j0:j1]
            if win.any():
                ii, jj = np.nonzero(win)
                if np.min((ii + i0 - i) ** 2 + (jj + j0 - j) ** 2) < r * r:
                    continue
            acc[i, j] = True
        return acc

    lo, hi = 0.3, 0.5 * np.sqrt(float(R)) * 3
    best = draw(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = draw(mid)
        best = m
        if m.sum() > target:
            lo = mid
        else:
            hi = mid
        if abs(int(m.sum()) - target) <= max(1, 0.01 * target):
            break
    return best


# ----------------------------------------------------------------------------------- GRO (dynamic, Cartesian)
def gro_lines(n, frames, lines, s=2.2, alpha=3.0, tau=1, n_enc=1, enc=0):
    """Golden-ratio offset (GRO) Cartesian sampling for dynamic MRI (Joshi, Ahmad et al.; technical report
    arXiv:2206.03630, MATLAB https://github.com/OSU-CMR/GRO-CAVA). ``lines`` uniformly spaced samples on a
    shrunk grid of Ns = ceil(n / s) points, offset from frame to frame by the golden ratio times the
    sample spacing (tau = 1: golden ratio, tau > 1: tiny golden ratios), then stretched to the n-point
    grid by v - kappa sign(c - v) |c - v|^alpha (variable density, denser in the centre). Returns a
    (frames, n) bool array (duplicate indices after rounding merge, so a frame can have < ``lines``)."""
    g = (1 + np.sqrt(5)) / 2
    g = 1.0 / (g + tau - 1)
    Ns = int(np.ceil(n / s))
    kap = (n / 2 - Ns / 2) / ((Ns / 2) ** alpha)
    v0 = np.arange(lines) * (Ns / lines) + 0.5 + 1e-10                    # (1/2 : Ns/M : Ns+1/2) in MATLAB
    v0 = v0 + (enc + 1) / n_enc * Ns / lines
    out = np.zeros((frames, n), bool)
    for j in range(frames):
        v = np.remainder(v0 + j * Ns / lines * g - 1, Ns) + 1
        v = v - Ns * (v >= Ns + 0.5)
        c = Ns / 2 + 0.5
        vc = v - kap * np.sign(c - v) * np.abs(c - v) ** alpha + (n - Ns) / 2 + (0.5 if n % 2 == 0 else 0.0)
        vc = vc - n * (vc >= n + 0.5)
        idx = np.clip(np.round(np.sort(vc)).astype(int), 1, n) - 1            # 1-based -> 0-based
        out[j, idx] = True
    return out


def gro_mask(shape, frames, R, pe_axis=1, extent=None, **kw):
    """(X, Y, T) bool mask: GRO lines along ``pe_axis`` with round(n / R) lines per frame, n = the phase-
    encode lines inside the acquired extent (partial-Fourier gaps excluded; ``extent`` an (X, Y) bool array
    of the acquired samples), readout fully sampled. R = 1: every acquired line in every frame."""
    shape = tuple(int(a) for a in shape)
    ext = np.ones(shape, bool) if extent is None else np.asarray(extent, bool)
    prof = ext.any(axis=1 - pe_axis)
    pos = np.flatnonzero(prof)
    if R <= 1:
        lines = np.zeros((frames, shape[pe_axis]), bool); lines[:, pos] = True
    else:
        g = gro_lines(pos.size, frames, max(1, int(round(pos.size / R))), **kw)
        lines = np.zeros((frames, shape[pe_axis]), bool); lines[:, pos] = g
    m = lines[:, None, :] if pe_axis == 1 else lines[:, :, None]           # (T, X, Y)
    return np.moveaxis(np.broadcast_to(m, (frames,) + shape) & ext[None], 0, -1).copy()
