"""Reference-free quality metrics for coil-sensitivity maps.

Everything here is computed from the measured k-space alone: no ground-truth image, no ground-truth
maps. Data are pre-whitened (every sample has independent complex noise of variance sigma^2), so noise
floors are exact expectations (Monte Carlo through the same linear operator), not estimates.

Why more than the combine -> expand residual (``evaluation.kspace_consistency``)? That residual is
evaluated on the same noise the maps were estimated from, so a method that adapts its maps to the
noise (inside the calibration region in particular) is rewarded: it can fall *below* the noise floor.
The metrics below are built so that fitting noise does not pay off:

1. ``leave_coil_out`` (cheap, fully sampled): predict coil j from the other N-1 coils with the maps,
   x_{-j} = sum_{i!=j} C_i^* I_i / sum_{i!=j} |C_i|^2,   r_j = I_j - C_j x_{-j}.
   The noise of coil j is independent of the prediction, so the residual cannot shrink below the
   noise floor sigma^2 (1 + |C_j|^2 / sum_{i!=j}|C_i|^2) (computed by Monte Carlo). It tests the
   *relative* magnitudes and phases between coils, i.e. exactly what parallel imaging uses, and is
   invariant to the unidentifiable per-pixel common factor. It is noise-proof only on data the maps
   did not see: the headline is the k-space residual outside the ACS (in the calibration-only
   protocol); with joint calibration (estimators see every sample) only 4./5. are noise-proof.
   ``leave_coil_out_cg`` is the variant for undersampled (prospective) data: x_{-j} by CG-SENSE from
   the other coils' acquired samples.
2. ``residual_structure`` (cheap, fully sampled): is the combine -> expand residual noise-like?
   (a) residual / noise floor in bins of signal level: noise is signal-independent, model error grows
       with signal (slope > 0), noise fitting shows as ratio < 1 where the signal is high;
   (b) whiteness: low-pass energy fraction of the residual relative to that of pure noise pushed
       through the same operator (1 = white; > 1 = spatially coherent = map error).
3. ``projector_bandwidth`` (cheap, any data): gauge-invariant smoothness prior. Spectrum of the
   per-pixel projector P = c c^H (c = RSS-normalised maps) on a tapered object mask; fraction of its
   energy beyond 1x and 2x the calibration bandwidth. Physical sensitivities are smooth; energy
   outside the band was either extrapolated or fitted to noise.
4. ``recorrupted`` (expensive: one re-estimation per method; Recorrupted-to-Recorrupted, Pang et al.
   2021): split the data into two copies with *independent* noise,
       y1 = y + a z,   y2 = y - z / a,    z ~ CN(0, sigma^2)   (Cov(n1, n2) = 0),
   estimate maps from y1, evaluate the combine -> expand and leave-coil-out residuals on y2. The
   "optimism" (residual on the estimation copy y1 vs on the independent copy y2, both over their own
   floors) directly measures how much a method fits noise. Optimism = unexplained signal fraction
   (residual - noise floor) / signal on y2 minus on y1 (>= 0; 0 = no noise fitting).
5. ``sure`` (expensive: 2 x probes re-estimations per method; Stein's unbiased risk estimate of the
   combine -> expand prediction f(y) = M F P_C(y_calib) F^H M y):
       SURE = ||y - f(y)||^2 - M sigma^2 + 2 sigma^2 (tr(M F P F^H M) + d_maps),
   tr(...) = #sampled x fraction of pixels with non-zero maps (exact), and d_maps, the extra degrees of
   freedom the maps spend on the calibration noise, by a Monte-Carlo (antithetic finite difference)
   divergence over the calibration samples. Reported over the risk of perfect fixed maps
   (sigma^2 x #sampled; the true maps are non-zero everywhere): 1 = as good as the true maps, and d_maps itself (0 for data-independent maps).
   eps (x sigma, default 0.1) trades bias for variance: a data-copying estimator gives 5.98 at 0.1
   (exact 6) but 3.2 at 1.0. Stochastic estimators (NN training on GPU) add jitter / (2 eps sigma) to
   d_maps: use a fixed seed and deterministic kernels, or a larger eps and more probes.

6. ``coil_ssim`` (cheap, fully sampled): SSIM of the magnitude coil images after expansion against
   the measured coil images, per coil (object mask, weighted by coil energy; also the worst coil):
   (a) combine -> expand C_j m, (b) the same restricted to k-space outside the ACS (high-pass coil
   images: data the maps did not see), (c) the leave-coil-out prediction C_j x_{-j}. HIGHER is better.
   The reference is noisy, so even the true maps stay below 1 (the ceiling falls with SNR): compare
   methods within a case, not across cases.

``summary`` collects one headline number per metric.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .evaluation import cg_sense, combine, expand, kspace_consistency, normalise
from .methods import EvalCase, fft2c, ifft2c


# ----------------------------------------------------------------------------------- helpers
def _noise(case: EvalCase, rng, sigma=None):
    s = case.sigma if sigma is None else sigma
    sh = case.ksp.shape
    return s * (rng.normal(size=sh) + 1j * rng.normal(size=sh)) / math.sqrt(2) * case.sampled[None]


def _box(shape, size):
    m = np.zeros(shape, bool)
    m[tuple(slice(max(n // 2 - a // 2, 0), min(n // 2 - a // 2 + a, n)) for n, a in zip(shape, size))] = True
    return m


def _ce_residual(C, I):
    """combine -> expand residual in the image domain."""
    return I - expand(C, combine(C, I))


def _lco_residual(C, I):
    """Leave-coil-out residual: coil j predicted from the other coils (closed form, per pixel)."""
    a2 = np.abs(C) ** 2
    S = (C.conj() * I).sum(0)
    D = a2.sum(0)
    Sj = S[None] - C.conj() * I
    Dj = D[None] - a2
    pos = D[D > 0]
    tiny = 1e-8 * (np.median(pos) if pos.size else 1.0)
    x = np.where(Dj > tiny, Sj / np.maximum(Dj, tiny), 0)
    return I - C * x


def _seen(case: EvalCase):
    """Samples the estimators saw."""
    return case.sampled if case.joint else case.acs_mask()


# ----------------------------------------------------------------------------------- 1. leave-coil-out
def leave_coil_out(C, case: EvalCase, obj, replicas=2, seed=1234):
    """Fully sampled data: closed-form leave-coil-out prediction; k-space residual (all acquired
    samples / outside the ACS) and image-domain residual inside the object, each over the exact
    Monte-Carlo noise floor of the same operator. 1 = perfect relative maps; > 1 = error."""
    s = case.sampled[None]
    y = case.ksp * s
    ri = _lco_residual(C, ifft2c(y))
    rk = fft2c(ri) * s
    rng = np.random.default_rng(seed)
    fk = np.zeros(case.ksp.shape); fi = np.zeros(case.ksp.shape)
    for _ in range(replicas):
        rn = _lco_residual(C, ifft2c(_noise(case, rng)))
        fi += np.abs(rn) ** 2 / replicas
        fk += np.abs(fft2c(rn) * s) ** 2 / replicas
    acs = case.acs_mask()
    out = {}
    for name, reg in (("all", case.sampled), ("outside_acs", case.sampled & ~acs)):
        e = (np.abs(rk[:, reg]) ** 2).sum(1); f = fk[:, reg].sum(1); yy = (np.abs(y[:, reg]) ** 2).sum(1)
        pc = e / np.maximum(f, 1e-30)
        out[name] = {"residual_over_noise_floor": float(e.sum() / f.sum()),
                     "excess_rel": float((e.sum() - f.sum()) / yy.sum()),
                     "nrmse": float(math.sqrt(e.sum() / yy.sum())),
                     "noise_floor_nrmse": float(math.sqrt(f.sum() / yy.sum())),
                     "worst_coil_over_floor": float(pc.max()), "worst_coil": int(pc.argmax()),
                     "per_coil_over_floor": [float(v) for v in pc]}
    e = float((np.abs(ri[:, obj]) ** 2).sum()); f = float(fi[:, obj].sum())
    out["object"] = {"residual_over_noise_floor": e / max(f, 1e-30)}
    return out


def leave_coil_out_cg(C, case: EvalCase, lam=1e-3, iters=100, coils=None, replicas=1, seed=99):
    """Undersampled data: for each coil j, CG-SENSE from the other coils' acquired samples, predict
    coil j at its acquired samples outside the calibration region (data no estimator saw, unless joint).
    Two normalisations: over the sample noise sigma^2 (prediction error incl. the noise each map set
    propagates: the overall figure of merit) and over the Monte-Carlo floor of the same reconstruction
    applied to pure noise (the error beyond noise propagation: map bias). CG is run to convergence:
    a truncated Krylov solve is not linear in the data and its noise floor would not be comparable."""
    Cn = normalise(C)
    y = case.ksp * case.sampled[None]
    reg = case.sampled & ~_seen(case)
    if not reg.any():
        reg = case.sampled & ~case.acs_mask()
    coils = list(range(case.N) if coils is None else coils)
    rng = np.random.default_rng(seed)
    noise = [_noise(case, rng) for _ in range(replicas)]

    def pred_err(data, j):
        keep = [i for i in range(case.N) if i != j]
        x = cg_sense(data[keep], case.sampled, Cn[keep], lam, iters, tol=1e-12)   # converged = linear in the data
        pred = fft2c((Cn[j] * x)[None])[0]
        return float((np.abs(pred[reg] - data[j][reg]) ** 2).sum())
    per, per_f, E, F, Y = [], [], 0.0, 0.0, 0.0
    s0 = case.sigma ** 2 * reg.sum()
    for j in coils:
        e = pred_err(y, j)
        f = float(np.mean([pred_err(n, j) for n in noise])) if replicas else float("nan")
        per.append(e / s0); per_f.append(e / f); E += e; F += f
        Y += float((np.abs(y[j][reg]) ** 2).sum())
    n = len(per)
    return {"outside_calib": {"residual_over_sample_noise": E / (s0 * n), "residual_over_noise_floor": E / F,
                              "noise_floor_over_sample_noise": F / (s0 * n), "nrmse": math.sqrt(E / Y),
                              "worst_coil_over_sample_noise": float(max(per)),
                              "per_coil_over_sample_noise": per, "per_coil_over_noise_floor": per_f}}


# ----------------------------------------------------------------------------------- 2. residual structure
def residual_structure(C, case: EvalCase, obj, n_bins=5, lowpass_px=2.0, replicas=2, seed=4321):
    from scipy.ndimage import gaussian_filter
    s = case.sampled[None]
    I = ifft2c(case.ksp * s)
    r = _ce_residual(C, I)
    rng = np.random.default_rng(seed)
    rn = [_ce_residual(C, ifft2c(_noise(case, rng))) for _ in range(replicas)]
    e = (np.abs(r) ** 2).sum(0)
    f = sum((np.abs(x) ** 2).sum(0) for x in rn) / replicas
    rss = np.sqrt((np.abs(I) ** 2).sum(0))
    prof = []
    bg = ~obj
    if bg.any():
        prof.append({"region": "background", "pixels": int(bg.sum()),
                     "residual_over_noise_floor": float(e[bg].sum() / max(f[bg].sum(), 1e-30))})
    q = np.quantile(rss[obj], np.linspace(0, 1, n_bins + 1)) if obj.any() else []
    xs, ys = [], []
    for i in range(len(q) - 1):
        reg = obj & (rss >= q[i]) & ((rss < q[i + 1]) if i < len(q) - 2 else (rss <= q[i + 1]))
        if not reg.any():
            continue
        ratio = float(e[reg].sum() / max(f[reg].sum(), 1e-30))
        sig = float((rss[reg] ** 2).mean())
        prof.append({"region": f"object_q{i}", "signal_lo": float(q[i]), "signal_hi": float(q[i + 1]),
                     "pixels": int(reg.sum()), "residual_over_noise_floor": ratio})
        xs.append(math.log(max(sig, 1e-30))); ys.append(math.log(max(ratio, 1e-30)))
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else float("nan")
    sig_ = (0,) + (lowpass_px,) * case.ndim

    def lp_frac(x):
        lp = gaussian_filter(x.real, sig_) + 1j * gaussian_filter(x.imag, sig_)
        return float((np.abs(lp[:, obj]) ** 2).sum() / max((np.abs(x[:, obj]) ** 2).sum(), 1e-30))
    white = float(lp_frac(r) / max(float(np.mean([lp_frac(x) for x in rn])), 1e-30))
    hi = [p["residual_over_noise_floor"] for p in prof if p["region"].startswith("object")]
    return {"by_signal": prof, "log_slope_vs_signal_power": slope,
            "top_signal_bin_over_floor": hi[-1] if hi else float("nan"),
            "whiteness_ratio": white, "lowpass_px": lowpass_px}


# ----------------------------------------------------------------------------------- SSIM
def _ssim_per_coil(est, ref, obj, w):
    from skimage.metrics import structural_similarity
    vals = []
    for j in range(ref.shape[0]):
        a, b = np.abs(est[j]), np.abs(ref[j])
        dr = float(b.max())
        if dr <= 0:
            vals.append(float("nan")); continue
        _, m = structural_similarity(a, b, data_range=dr, full=True)
        vals.append(float(m[obj].mean()))
    v = np.asarray(vals); ok = np.isfinite(v)
    return {"weighted_mean": float((v[ok] * w[ok]).sum() / max(w[ok].sum(), 1e-30)), "worst_coil": float(v[ok].min()),
            "per_coil": [float(x) for x in v]}


def coil_ssim(C, case: EvalCase, obj):
    s = case.sampled[None]
    y = case.ksp * s
    I = ifft2c(y)
    E = expand(C, combine(C, I))
    L = I - _lco_residual(C, I)
    ho = (case.sampled & ~case.acs_mask())[None]
    w = (np.abs(I[:, obj]) ** 2).sum(1)
    return {"expand": _ssim_per_coil(E, I, obj, w),
            "expand_outside_acs": _ssim_per_coil(ifft2c(fft2c(E) * ho), ifft2c(y * ho), obj, w),
            "leave_coil_out": _ssim_per_coil(L, I, obj, w)}


# ----------------------------------------------------------------------------------- 3. bandwidth
def projector_bandwidth(C, case: EvalCase, obj, taper_px=3.0):
    from scipy.ndimage import gaussian_filter
    Cn = normalise(C)
    w = gaussian_filter(obj.astype(float), taper_px)
    b1 = _box(case.shape, case.acs)
    b2 = _box(case.shape, tuple(2 * a for a in case.acs))
    tot = o1 = o2 = 0.0
    for a in range(case.N):
        Pa = Cn[a][None] * Cn.conj() * w[None]          # row a of the projector, tapered
        k = np.abs(fft2c(Pa)) ** 2
        tot += float(k.sum()); o1 += float(k[:, ~b1].sum()); o2 += float(k[:, ~b2].sum())
    return {"frac_outside_calib_band": o1 / max(tot, 1e-30), "frac_outside_2x_calib_band": o2 / max(tot, 1e-30)}


# ----------------------------------------------------------------------------------- 4./5. re-estimation
def _reestimate(case: EvalCase, name, est_kw):
    from .methods import estimate
    maps, _ = estimate(case, [name], cache_dir=None, log=lambda *a: None, **(est_kw or {}))
    if name not in maps:
        raise ValueError(f"{name}: estimator returned nothing")
    return maps[name]


def recorrupted(case: EvalCase, obj, name, est_kw=None, alpha=0.7, seed=0, C1=None):
    """Recorrupted-to-recorrupted split. Returns the residuals of the maps estimated from y1 on y1
    itself ("self") and on the independent copy y2 ("independent"), over their own noise floors."""
    rng = np.random.default_rng(seed)
    z = _noise(case, rng)
    c1 = replace(case, ksp=case.ksp + alpha * z, sigma=case.sigma * math.sqrt(1 + alpha ** 2))
    c2 = replace(case, ksp=case.ksp - z / alpha, sigma=case.sigma * math.sqrt(1 + 1 / alpha ** 2))
    if C1 is None:
        C1 = _reestimate(c1, name, est_kw)
    out = {"alpha": alpha}
    if case.fully_sampled:
        own, ind = kspace_consistency(C1, c1), kspace_consistency(C1, c2)
        for reg in ("all", "acs", "outside_acs"):
            out[reg] = {"self_over_floor": own[reg]["residual_over_noise_floor"],
                        "independent_over_floor": ind[reg]["residual_over_noise_floor"],
                        "self_excess_rel": own[reg]["excess_rel"],
                        "independent_excess_rel": ind[reg]["excess_rel"]}
            # unexplained signal (residual minus its noise floor, / signal energy) on the independent copy
            # minus on the estimation copy. Not the difference of the over-floor ratios: the two copies have
            # different noise levels, so the same bias gives different ratios (v29 did that: negative values).
            out[reg]["optimism"] = out[reg]["independent_excess_rel"] - out[reg]["self_excess_rel"]
        l1, l2 = leave_coil_out(C1, c1, obj), leave_coil_out(C1, c2, obj)
        out["leave_coil_out"] = {"self_over_floor": l1["all"]["residual_over_noise_floor"],
                                 "independent_over_floor": l2["all"]["residual_over_noise_floor"],
                                 "independent_excess_rel": l2["all"]["excess_rel"]}
    else:
        out["leave_coil_out_independent"] = leave_coil_out_cg(C1, c2)["outside_calib"]
    return out


def _ce_kspace(C, y, s):
    """f(y) = M F C combine(C, F^H y): the prediction whose risk SURE estimates."""
    return fft2c(expand(C, combine(C, ifft2c(y)))) * s


def sure(C, case: EvalCase, name, est_kw=None, eps=0.1, probes=1, seed=0):
    """Monte-Carlo SURE of the combine -> expand prediction (fully sampled data)."""
    s = case.sampled[None]
    y = case.ksp * s
    sig2 = case.sigma ** 2
    fy = _ce_kspace(C, y, s)
    res = float((np.abs(y - fy) ** 2).sum())
    M = case.N * int(case.sampled.sum())
    nz = (np.abs(C) ** 2).sum(0) > 0
    tr = float(case.sampled.sum() * nz.mean())
    seen = _seen(case)[None]
    rng = np.random.default_rng(seed)
    d = []
    h = eps * case.sigma
    for _ in range(probes):
        b = _noise(case, rng, sigma=1.0) * seen
        Cp = _reestimate(replace(case, ksp=case.ksp + h * b), name, est_kw)
        Cm = _reestimate(replace(case, ksp=case.ksp - h * b), name, est_kw)
        d.append(float(np.vdot(b, _ce_kspace(Cp, y, s) - _ce_kspace(Cm, y, s)).real / (2 * h)))
    dm = float(np.mean(d))
    risk = res - M * sig2 + 2 * sig2 * (tr + dm)
    # perfect maps are non-zero everywhere: their risk is sigma^2 x #sampled. (v29 divided by the
    # method's own trace, which inflated the ratio for maps that are zero outside a support, e.g. ESPIRiT.)
    oracle = sig2 * float(case.sampled.sum())
    return {"sure_over_oracle": risk / oracle, "naive_over_oracle": (res - M * sig2 + 2 * sig2 * tr) / oracle,
            "map_dof": dm, "map_dof_probes": d, "projection_dof": tr, "nonzero_fraction": float(nz.mean()), "eps_sigma": eps, "probes": probes}


# ----------------------------------------------------------------------------------- run / summary
def evaluate(C, case: EvalCase, obj, name=None, est_kw=None, resample=(), r2r_alpha=0.7, sure_eps=0.1,
             sure_probes=1, lco_coils=None):
    """All reference-free metrics for one map set. ``resample``: subset of {"r2r", "sure"} (these
    re-run the estimator ``name`` with ``est_kw``)."""
    out = {}
    if case.fully_sampled:
        out["leave_coil_out"] = leave_coil_out(C, case, obj)
        out["residual_structure"] = residual_structure(C, case, obj)
        out["coil_ssim"] = coil_ssim(C, case, obj)
    else:
        out["leave_coil_out"] = leave_coil_out_cg(C, case, coils=lco_coils)
    out["projector_bandwidth"] = projector_bandwidth(C, case, obj)
    if name is not None and "r2r" in resample:
        out["recorrupted"] = recorrupted(case, obj, name, est_kw, alpha=r2r_alpha)
    if name is not None and "sure" in resample and case.fully_sampled:
        out["sure"] = sure(C, case, name, est_kw, eps=sure_eps, probes=sure_probes)
    out["summary"] = summary(out)
    return out


def summary(rf):
    """One headline number per metric (lower is better except ssim_*; ideal values in the README)."""
    s = {}
    l = rf.get("leave_coil_out", {})
    if "outside_acs" in l:
        s["lco_outside_acs_over_floor"] = l["outside_acs"]["residual_over_noise_floor"]
        s["lco_worst_coil_over_floor"] = l["outside_acs"]["worst_coil_over_floor"]
        s["lco_object_over_floor"] = l["object"]["residual_over_noise_floor"]
    elif "outside_calib" in l:
        s["lco_outside_calib_over_sample_noise"] = l["outside_calib"]["residual_over_sample_noise"]
        s["lco_outside_calib_over_floor"] = l["outside_calib"]["residual_over_noise_floor"]
    if "residual_structure" in rf:
        s["residual_signal_slope"] = rf["residual_structure"]["log_slope_vs_signal_power"]
        s["residual_whiteness"] = rf["residual_structure"]["whiteness_ratio"]
    if "coil_ssim" in rf:
        cs = rf["coil_ssim"]
        s["ssim_expand"] = cs["expand"]["weighted_mean"]
        s["ssim_expand_outside_acs"] = cs["expand_outside_acs"]["weighted_mean"]
        s["ssim_leave_coil_out"] = cs["leave_coil_out"]["weighted_mean"]
        s["ssim_leave_coil_out_worst_coil"] = cs["leave_coil_out"]["worst_coil"]
    s["projector_hf_fraction"] = rf["projector_bandwidth"]["frac_outside_calib_band"]
    if "recorrupted" in rf:
        r = rf["recorrupted"]
        if "all" in r:
            s["r2r_independent_excess_rel"] = r["all"]["independent_excess_rel"]
            s["r2r_optimism"] = r["all"]["optimism"]
            s["r2r_lco_independent_over_floor"] = r["leave_coil_out"]["independent_over_floor"]
        else:
            s["r2r_lco_independent_over_sample_noise"] = r["leave_coil_out_independent"]["residual_over_sample_noise"]
    if "sure" in rf:
        s["sure_over_oracle"] = rf["sure"]["sure_over_oracle"]
        s["sure_map_dof"] = rf["sure"]["map_dof"]
    return s
