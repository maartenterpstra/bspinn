"""Round 2-4 (v57-v59) blocks of the benchmark page, from dashboard_data.json['extra'] (dashboard_bundle.py --extra).
render(d, base_ocmr) -> dict(eyebrow, findings, reco, r2, caveats) or None when there is no 'extra' data."""
import collections
import html
import re
import statistics as st

esc = html.escape


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def f4(x, n=4):
    return "–" if x is None else f"{x:.{n}f}"


def short(f):
    p = f.split("_")
    return f"{p[0]}_{p[1]}" + (f' <span class="muted">{"3 T" if "3T" in f else "1.5 T"}</span>' if len(p) > 2 else "")


def tbl(groups, heads, rows, first="Case"):
    """rows: list of (label_html, [cell_html...]); cells already formatted (class handled by caller)."""
    o = ["<table><thead>"]
    if groups:
        o.append('<tr class="group"><th></th>' + "".join(
            f'<th colspan="{n}" class="{"gstart" if i else ""}">{esc(t)}</th>' for i, (t, n) in enumerate(groups)) + "</tr>")
    o.append(f"<tr><th>{first}</th>" + "".join(f"<th>{h}</th>" for h in heads) + "</tr></thead><tbody>")
    for lab, cells in rows:
        o.append(f"<tr><td>{lab}</td>" + "".join(cells) + "</tr>")
    o.append("</tbody></table>")
    return "".join(o)


def num(x, best=False, n=4, gstart=False, flag=False):
    c = "num" + (" best" if best else "") + (" gstart" if gstart else "") + (" flag" if flag else "")
    return f'<td class="{c}">{f4(x, n)}</td>'


def txt(s, gstart=False):
    return f'<td class="num{" gstart" if gstart else ""}">{s}</td>'


def render(d, base_ocmr):
    X = d.get("extra") or {}
    if not X:
        return None
    out = {}
    nums = {}

    # ---------------------------------------------------------------- static image priors
    img = collections.defaultdict(dict)
    for r in X.get("r2_image", []):
        img[(r["R"], r["maps"], r["method"])][re.sub(r"_j\d+$", "", r["case"])] = r
    IM = [("tv", "TV"), ("tgv", "TGV"), ("tgv_phase", "TGV + phase"), ("tgv_phase_auto", "TGV + phase, auto")]
    rows = []
    for R in (4.0, 6.0):
        for maps, ml in (("pinn_v54", "PINN v54 maps"), ("nlinv", "NLINV maps")):
            base = img.get((R, maps, "tv"), {})
            cells = []
            meds = []
            for i, (m, _) in enumerate(IM):
                x = img.get((R, maps, m), {})
                cs = sorted(set(x) & set(base))
                if not cs:
                    cells += [txt("–", i > 0), txt("–")]; meds.append(None); continue
                a = [x[c]["nrmse"] for c in cs]
                meds.append(med(a))
                if m == "tv":
                    cells += [None, txt("–")]
                else:
                    w = sum(x[c]["nrmse"] < base[c]["nrmse"] for c in cs)
                    worst = max(x[c]["nrmse"] / base[c]["nrmse"] for c in cs)
                    cells += [None, txt(f"{w}/{len(cs)} · ×{worst:.2f}")]
            b = min(v for v in meds if v is not None)
            k = 0
            for i, v in enumerate(meds):
                cells[2 * i] = num(v, v is not None and abs(v - b) < 1e-12, gstart=i > 0)
            rows.append((f"R = {R:g} · {ml}", cells))
            if maps == "pinn_v54":
                nums[f"img_R{R:g}"] = (meds[0], meds[3])
    ch = collections.Counter(r.get("phase_choice") for r in X.get("r2_image", []) if r["method"] == "tgv_phase_auto")
    img_tbl = tbl([(t, 2) for _, t in IM], ["Median NRMSE", "Better than TV · worst slice"] * 4, rows, first="R, maps")

    # ---------------------------------------------------------------- IRGNM on the 20 slices
    ir = collections.defaultdict(dict)
    for r in X.get("r2_irgnm", []):
        ir[r["maps"]][r["case"]] = r
    IRM = [("pinn_v54", "PINN v54"), ("pinn_v54+irgnm@holdout=0@steps=8", "+ IRGNM, 8 steps"),
           ("pinn_v54+irgnm@holdout=0@steps=12@x_ref=zero", "+ IRGNM, 12 steps, image → 0 (v59 default)"),
           ("nlinv", "NLINV"), ("nlinv+irgnm@holdout=0@steps=8", "NLINV + IRGNM, 8 steps")]
    rows = []
    for m, lab in IRM:
        x = ir.get(m)
        if not x:
            continue
        base = ir["pinn_v54" if m.startswith("pinn") else "nlinv"]
        cs = sorted(x)
        g = lambda k: med([x[c][k] for c in cs])  # noqa: E731
        w = lambda k: sum(x[c][k] < base[c][k] - 1e-9 for c in cs if c in base)  # noqa: E731
        base_row = m in ("pinn_v54", "nlinv")
        rows.append((esc(lab), [num(g("ce_acs"), n=3), num(g("lco_out")), num(g("sense_R4")), num(g("tv_nrmse")),
                                txt("–" if base_row else f"{w('sense_R4')}/{len(cs)}", True),
                                txt("–" if base_row else f"{w('tv_nrmse')}/{len(cs)}")]))
    irgnm_tbl = tbl([("20 slices, R = 4 (medians)", 4), ("Slices better than the unrefined maps", 2)],
                    ["ACS combine→expand / floor", "Leave-coil-out / floor", "SENSE NRMSE", "TV NRMSE", "SENSE", "TV"],
                    rows, first="Maps")

    # ---------------------------------------------------------------- brain planes with IRGNM
    P = collections.defaultdict(list)
    for r in X.get("r2_poles", []):
        P[(r["cal"], r["maps"])].append(r)
    PMs = [("nlinv_pp", "NLINV + pole correction"), ("pinn_v54", "PINN v54"),
           ("pinn_v54+irgnm@holdout=0@steps=8", "+ IRGNM, 8 steps"),
           ("pinn_v54+irgnm@holdout=0@steps=12@x_ref=zero", "+ IRGNM, 12 steps, image → 0")]
    bests = {c: min(st.mean(r["sense_nrmse"] for r in P[(c, m)]) for m, _ in PMs if P.get((c, m))) for c in (3, 7, 15)}
    rows = []
    for m, lab in PMs:
        cells = []
        for i, c in enumerate((3, 7, 15)):
            rs = P.get((c, m), [])
            if not rs:
                cells += [txt("–", i > 0), txt("–")]; continue
            s4 = st.mean(r["sense_nrmse"] for r in rs)
            cells += [num(st.mean(r["lco_object"] for r in rs), n=2, gstart=i > 0), num(s4, abs(s4 - bests[c]) < 1e-12)]
        rows.append((esc(lab), cells))
    poles_tbl = tbl([(f"ACS {c}×{c}", 2) for c in (3, 7, 15)], ["Leave-coil-out", "SENSE NRMSE"] * 3, rows, first="Maps")
    zp = {c: st.mean(r["sense_nrmse"] for r in P.get((c, "pinn_v54+irgnm@holdout=0@steps=12@x_ref=zero"), [{"sense_nrmse": 0}]))
          for c in (3, 7, 15)}
    nl = {c: st.mean(r["sense_nrmse"] for r in P.get((c, "nlinv_pp"), [{"sense_nrmse": 0}])) for c in (3, 7, 15)}

    # ---------------------------------------------------------------- cine: header B0
    o2 = {(r["file"], r["R"], r["maps"]): r for r in X.get("r2_ocmr", [])}
    o3 = {(r["file"], r["R"], r["maps"]): r for r in X.get("r3_ocmr", [])}
    o4 = {(r["file"], r["R"]): r for r in X.get("r4_ocmr", [])}
    files = sorted({k[0] for k in o4})
    rows, agg = [], {4.0: ([], [], []), 8.0: ([], [], [])}
    for f in files:
        cells = []
        for i, R in enumerate((4.0, 8.0)):
            a = o2.get((f, R, "nlinv"), {}).get("nrmse"); b = o3.get((f, R, "pinn_avg"), {}).get("nrmse")
            c = o4.get((f, R), {}).get("nrmse")
            vals = [v for v in (a, b, c) if v is not None]
            bb = min(vals) if vals else None
            cells += [num(a, a == bb, gstart=i > 0), num(b, b == bb, flag=b is not None and a is not None and b > 1.1 * a),
                      num(c, c == bb)]
            for lst, v in zip(agg[R], (a, b, c)):
                lst.append(v)
        rows.append((short(f), cells))
    medrow = []
    for i, R in enumerate((4.0, 8.0)):
        ms = [med(l) for l in agg[R]]
        bb = min(m for m in ms if m is not None)
        medrow += [num(m, m == bb, gstart=(i > 0 and j == 0)) for j, m in enumerate(ms)]
        a, b, c = agg[R]
        nums[f"cine_R{R:g}"] = (ms[0], ms[1], ms[2], sum(1 for x, z in zip(a, c) if x is not None and z is not None and z < x), len(files))
    rows.append(("<b>Median</b>", medrow))
    cine_tbl = tbl([("R = 4, frame NRMSE", 3), ("R = 8, frame NRMSE", 3)],
                   ["NLINV", "PINN, plane search", "PINN, header B0"] * 2, rows)

    # map quality (diag)
    d3, d4 = collections.defaultdict(list), collections.defaultdict(list)
    nld = {}
    for r in X.get("r3_diag", []):
        (nld.__setitem__(r["file"], r) if r["maps"] == "nlinv" else d3[r["file"]].append(r))
    for r in X.get("r4_diag", []):
        if r["maps"] != "nlinv":
            d4[r["file"]].append(r)
    rows = []

    def rng(rs, k):
        v = [r[k] for r in rs]
        return "–" if not v else (f"{min(v):.3f}" if max(v) - min(v) < 5e-4 else f"{min(v):.3f}–{max(v):.3f}")
    for f in sorted(nld):
        n = nld[f]
        rows.append((short(f), [num(n["resid"], n=3), num(n["lco"], n=3), txt(rng(d3[f], "resid"), True), txt(rng(d3[f], "lco")),
                                txt(rng(d4[f], "resid"), True), txt(rng(d4[f], "lco"))]))
    diag_tbl = tbl([("NLINV", 2), ("PINN, plane search (seeds)", 2), ("PINN, header B0 (seeds)", 2)],
                   ["Combine→expand", "Leave-coil-out"] * 3, rows)

    # cine prior: difference vs image (r2 vs the v56 run, NLINV maps: same maps)
    old = {(r["file"], r["R"], r["maps"]): r for r in base_ocmr if r.get("recon") == "tv"}
    ratios = {R: [o2[k]["nrmse"] / old[k]["nrmse"] for k in o2 if k[1] == R and k[2] == "nlinv" and k in old]
              for R in (4.0, 8.0)}
    dyn = {R: [o2[k]["dyn_nrmse"] / old[k]["dyn_nrmse"] for k in o2 if k[1] == R and k[2] == "nlinv" and k in old
               and old[k].get("dyn_nrmse")] for R in (4.0, 8.0)}

    # prospective
    u2 = {(r["file"], r["maps"]): r for r in X.get("r2_ocmr_us", []) if r["recon"] == "tv"}
    u3 = {(r["file"], r["maps"]): r for r in X.get("r3_ocmr_us", []) if r["recon"] == "tv"}
    u4 = {r["file"]: r for r in X.get("r4_ocmr_us", [])}
    rows = []
    for f in sorted(u4):
        a = u2.get((f, "nlinv"), {}).get("heldout"); e = u2.get((f, "espirit"), {}).get("heldout")
        b = u3.get((f, "pinn_avg"), {}).get("heldout"); c = u4[f]["heldout"]
        dz = u3.get((f, "pinn_dyn0"), {}).get("heldout")
        bb = min(v for v in (a, b, c) if v is not None)
        rows.append((esc(f), [num(e, n=2), num(a, a == bb, n=2), num(b, b == bb, n=2), num(dz, n=2), num(c, c == bb, n=2)]))
    us_tbl = tbl([("Held-out k-space error / noise, spatio-temporal TV", 5)],
                 ["ESPIRiT", "NLINV", "PINN, plane search", "PINN dynamic, rank 0 (v58)", "PINN, header B0 (v59)"], rows)

    # non-Cartesian
    rows = []
    for r in X.get("r4_noncart", []):
        lab = {"synth_radial": "Radial", "synth_spiral": "Spiral"}.get(r["case"], r["case"])
        rows.append((f'{lab} <span class="muted">{r["shots"]} shots, {r["coils"]} coils</span>',
                     [txt(f'{r["map_error_pinn"]:.1e}'), num(r["nrmse_pinn_cg"], n=3, gstart=True), num(r["nrmse_true_cg"], n=3),
                      num(r["nrmse_pinn_tv"], n=3, gstart=True), num(r["nrmse_true_tv"], n=3)]))
    nc_tbl = tbl([("Maps", 1), ("CG-SENSE NRMSE", 2), ("TV NRMSE", 2)],
                 ["Error vs truth", "PINN maps", "True maps", "PINN maps", "True maps"], rows, first="128² phantom")

    # ---------------------------------------------------------------- blocks
    c4, c8 = nums["cine_R4"], nums["cine_R8"]
    i4, i6 = nums["img_R4"], nums["img_R6"]
    out["eyebrow"] = ('<div class="eyebrow">bspinn v59 · coil sensitivity estimation · baseline run of 20–21 Oct, '
                      'rounds 2–4 of 21–24 Oct 2026</div>')
    diff_find = "" if not ratios[4.0] else f'''      <div class="finding"><span class="tag good">new default</span><h3>Cine images: TV of the difference from the time average</h3>
        <p>Same maps, frame NRMSE lower on {sum(x < 1 for x in ratios[4.0]) + sum(x < 1 for x in ratios[8.0])} of
        {len(ratios[4.0]) + len(ratios[8.0])} cine/R pairs (median ×{med(ratios[4.0]):.3f} at R = 4, ×{med(ratios[8.0]):.3f} at
        R = 8); the frame-to-frame error is mixed (median ×{med(dyn[4.0]):.2f}, up to ×{max(dyn[4.0] + dyn[8.0]):.2f}).</p></div>
'''
    out["findings"] = f'''    <div class="findings">
      <div class="finding"><span class="tag good">fixed</span><h3>Cine: with the scanner's B0 direction the PINN ties NLINV</h3>
        <p>Cardiac views are double-oblique (B0 51–65° from the slice normal); the PINN only knew axis-aligned B0. With the
        direction from the ISMRMRD header the median R = 4 frame NRMSE is {c4[2]:.4f} against {c4[0]:.4f} for NLINV
        (was {c4[1]:.4f}), better on {c4[3]} of {c4[4]} cines; R = 8 {c8[2]:.4f} vs {c8[0]:.4f}. fs_0021: 0.0683 → 0.0209.</p></div>
      <div class="finding"><span class="tag good">PINN</span><h3>Better maps than NLINV on the 3 T cines</h3>
        <p>Leave-coil-out on the fully sampled time average: fs_0021 0.058 vs 0.084, fs_0020 0.059 vs 0.074; fs_0016 0.125 vs
        0.118. The seeds now agree to 0.001.</p></div>
      <div class="finding"><span class="tag good">new default</span><h3>Static images: TGV + phase, switched per slice</h3>
        <p>On the 20 slices (PINN maps) the median NRMSE falls from {i4[0]:.4f} (TV) to {i4[1]:.4f} at R = 4 and from
        {i6[0]:.4f} to {i6[1]:.4f} at R = 6, better than TV on 18 of 20 slices, worst slice +3 %. Cross-validation drops the
        phase term where the phase is not smooth (on {ch.get("off", 0)} of {sum(ch.values())} cases).</p></div>
{diff_find}      <div class="finding"><span class="tag even">small</span><h3>IRGNM map refinement: better maps, same images</h3>
        <p>On the fastMRI slices it lowers the calibration residual and leave-coil-out slightly, the TV image does not change.
        On the brain planes it is the best map set at ACS 7 and 15 (SENSE {zp[7]:.3f} vs {nl[7]:.3f} for corrected NLINV) and
        narrows ACS 3 ({zp[3]:.3f} vs {nl[3]:.3f}), which stays the PINN's weak spot.</p></div>
      <div class="finding"><span class="tag even">safe</span><h3>Per-dynamic maps: rank 0 = time average, corrections rejected</h3>
        <p>The dynamic layout now starts from the 2-D time-average fit and reads the readout axis from the frames (the time
        average hid it). A held-out check keeps per-frame corrections only if they predict withheld lines better: it
        rejected them on every cine (they fit aliasing with 16–52 lines per frame).</p></div>
      <div class="finding"><span class="tag good">new</span><h3>Non-Cartesian via MRI-NUFFT</h3>
        <p>Radial and spiral phantoms: maps from the calibration samples are exact (error 1.5e-5), and CG-SENSE / TV
        images with them equal those with the true maps (TV 0.037 radial, 0.057 spiral).</p></div>
      <div class="finding"><span class="tag bad">open</span><h3>Still behind NLINV</h3>
        <p>3×3 calibration on the brain planes (SENSE {zp[3]:.3f} vs {nl[3]:.3f}); fs_0016 (×1.08 at R = 4); prospective us_0001
        (held-out 1.37 vs 1.27 noise, worse than with the plane search).</p></div>
    </div>'''
    out["reco"] = '''    <div class="reco">
      <pre>maps, image, info = estimate_maps(kspace, mask=mask, device="cuda:0", solve_image=True)
# v59 defaults: IRGNM refinement, image prior TGV + phase 'auto' (cine: TV of the difference),
# readout axis from the sampling; oblique slices: plane=info_from_header['b0_dir']
image = sense_image(kspace, maps, "2d", mask=mask)              # same prior for maps you already have</pre>
      <ul>
        <li>Pass the B0 direction for oblique slices: <code>plane=(b_row, b_col, b_normal)</code>, the B0 (z) components of the read, phase and slice directions; <code>bspinn.ocmr</code> returns it as <code>info['b0_dir']</code> and the OCMR scripts use it by default.</li>
        <li>Cine: time-averaged maps (or <code>layout="dynamic"</code>, rank 0 — now identical), reconstructed with <code>sense_image(..., "dynamic")</code>.</li>
        <li>Non-Cartesian: <code>bspinn.noncart.estimate_maps_noncart(kspace, traj, shape, plane=..., return_image=True)</code> (needs <code>mri-nufft</code> + <code>finufft</code>).</li>
        <li><code>image_prior=None</code> / <code>prior=None</code>: the old Tikhonov image; <code>refine=None</code>: no refinement.</li>
      </ul>
    </div>'''
    out["caveats"] = '''    <ul class="plain">
      <li>The sections below the round 2–4 results are the v56b baseline run; its cine and static numbers predate the header B0 direction, the new priors and the refinement.</li>
      <li>The header-B0 cine run used all v59 defaults, so it also includes the IRGNM refinement; on the static slices the refinement alone changed the images by &lt; 0.2 %, so most of the cine gain is the B0 direction.</li>
      <li>The fully sampled references are noisy: image NRMSE differences of ~1 % are within the slice-to-slice spread; paired wins are more reliable.</li>
      <li>Next: a full v59 rerun of the static benchmark and the brain planes (<code>run_v59.sh</code>); the 3×3 calibration case; us_0001.</li>
    </ul>'''
    out["r2"] = f'''  <section aria-labelledby="h-r2">
    <div class="sec-head"><div class="eyebrow">Rounds 2–4 · v57–v59 · 21–24 Oct 2026</div><h2 id="h-r2">What changed since the baseline run</h2>
      <p class="note">Each change was found by an investigation on this machine, then confirmed on your GPU on the same data as the baseline. The tables below are those GPU runs.</p></div>
    <h3>Cardiac cine with the B0 direction from the scanner header</h3>
    <div class="table-wrap">{cine_tbl}</div>
    <p class="note">9 fully sampled OCMR cines, GRO undersampling, spatio-temporal TV with spatial TV of the difference from the time average (the same prior for all three). Red: PINN more than 10 % behind NLINV.</p>
    <div class="table-wrap">{diag_tbl}</div>
    <p class="note">Map quality on the fully sampled time average (R = 4 maps, gauge-invariant; lower is better).</p>
    <div class="table-wrap">{us_tbl}</div>
    <p class="note">Prospective cines (R ≈ 9): error on withheld acquired lines, 1 = noise level.</p>
    <h3>Static images: priors on 20 slices</h3>
    <div class="table-wrap">{img_tbl}</div>
    <p class="note">Weight by k-space cross-validation for every prior; maps of the baseline run (joint protocol). "auto" switches the low-resolution phase constraint per slice by cross-validation.</p>
    <h3>IRGNM map refinement</h3>
    <div class="table-wrap">{irgnm_tbl}</div>
    <div class="table-wrap">{poles_tbl}</div>
    <p class="note">Brain MPRAGE planes of the phase-pole paper, mean over 3 planes, 2×2 undersampling plus the ACS block.</p>
    <h3>Non-Cartesian (MRI-NUFFT)</h3>
    <div class="table-wrap">{nc_tbl}</div>
    <p class="note">Biot–Savart phantoms, 128², NRMSE against the object times the coils' root-sum-of-squares (the convention of RSS-normalised maps); TV weight by cross-validation on withheld shots.</p>
  </section>
'''
    return out
