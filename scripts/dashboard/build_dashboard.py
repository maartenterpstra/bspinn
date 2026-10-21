"""Build the Reference-Free Coil Benchmark page from dashboard_data.json: python scripts/dashboard/build_dashboard.py dashboard_data.json out.html scripts/dashboard/template.html"""
import collections
import html
import json
import statistics as st
import sys

SRC, OUT = sys.argv[1], sys.argv[2]
d = json.load(open(SRC))
E = d["eval"]
REAL = [r for r in E if r["group"] == "real"]
SYN = [r for r in E if r["group"] == "synthetic"]

LABEL = {"nlinv": "NLINV", "espirit": "ESPIRiT", "walsh": "Walsh", "inati": "Inati", "lowres": "Low-res division",
         "pinn_api": "PINN, maps only", "pinn_v53": "PINN v53 (joint)", "pinn_v54": "PINN v54 (joint, default)",
         "pinn_v56": "PINN v56 (v55 options on)", "pinn_v56_vp0": "v56, no var. projection",
         "pinn_v56_nolbfgs": "v56, no L-BFGS finish", "pinn_v56_nozm": "v56, no zero-mean factor",
         "pinn_v56_grad": "v56, gradient image", "bs_pinn": "Biot–Savart PINN (bench)", "bs_wires": "Biot–Savart wires",
         "free": "Free (no physics)", "oracle": "Ground truth",
         "pinn_avg": "PINN, time-averaged", "pinn_dyn0": "PINN dynamic, rank 0", "pinn_dyn": "PINN per-dynamic (K=2)",
         "nlinv_pp": "NLINV + pole correction", "pinn_v55": "PINN v55"}
COLOR = {"nlinv": "--m-nlinv", "espirit": "--m-espirit", "walsh": "--m-walsh", "inati": "--m-inati",
         "lowres": "--m-lowres", "pinn_api": "--m-pinn-old", "pinn_v53": "--m-pinn", "pinn_v54": "--m-pinn-joint",
         "pinn_v56": "--m-pinn-js", "pinn_avg": "--m-pinn", "pinn_dyn0": "--m-pinn-lr", "pinn_dyn": "--m-pinn-js",
         "nlinv_pp": "--m-nlinv", "pinn_v55": "--m-pinn-js"}
esc = html.escape


def med(vals):
    v = [x for x in vals if x is not None]
    return st.median(v) if v else None


def fmt(x, n=4):
    if x is None:
        return "–"
    return f"{x:.2e}" if 0 < abs(x) < 1e-3 else f"{x:.{n}f}"


def swatch(m):
    c = COLOR.get(m.split("@")[0])
    s = f'<span class="swatch" style="background:var({c})"></span>' if c else '<span class="swatch ghost"></span>'
    return f'<span class="mname">{s}{esc(LABEL.get(m, m))}</span>'


def table(head_groups, heads, rows, best_cols=(), lower=True, ref_rows=(), abl_rows=()):
    """rows: list of (key, [cells]) with numeric cells (float|None|str)."""
    out = ['<table>']
    if head_groups:
        out.append('<thead><tr class="group"><th></th>' + "".join(
            f'<th colspan="{n}" class="gstart">{esc(t)}</th>' for i, (t, n) in enumerate(head_groups)) + "</tr>")
    else:
        out.append("<thead>")
    out.append("<tr><th>Method</th>" + "".join(f"<th>{h}</th>" for h in heads) + "</tr></thead><tbody>")
    best = {}
    for c in best_cols:
        vals = [r[1][c] for r in rows if isinstance(r[1][c], (int, float)) and r[0] not in ref_rows and r[0] not in abl_rows]
        if vals:
            best[c] = min(vals) if lower else max(vals)
    for key, cells in rows:
        cls = "ref" if key in ref_rows else ("abl" if key in abl_rows else "")
        tds = [f"<td>{swatch(key)}</td>"]
        for i, v in enumerate(cells):
            if isinstance(v, (int, float)):
                b = " best" if i in best and abs(v - best[i]) < 1e-12 else ""
                tds.append(f'<td class="num{b}">{fmt(v, 4 if abs(v) < 10 else 2)}</td>')
            else:
                tds.append(f'<td class="num">{esc(str(v)) if v is not None else "–"}</td>')
        out.append(f'<tr class="{cls}">' + "".join(tds) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


# ------------------------------------------------------------------ static benchmark
by_case = collections.defaultdict(dict)
for r in REAL:
    by_case[r["case"]][r["method"]] = r
prior = collections.defaultdict(dict)
for r in d["prior"]:
    prior[(r["case"], r["method"])][r["maps"]] = r


def wins(m, key, ref="nlinv"):
    cs = [c for c in by_case if m in by_case[c] and ref in by_case[c]]
    return f"{sum(by_case[c][m][key] < by_case[c][ref][key] for c in cs)}/{len(cs)}"


def prior_med(m, meth):
    return med([v[m]["nrmse"] for (c, mm), v in prior.items() if mm == meth and m in v])


def prior_wins(m, meth, ref="nlinv"):
    ks = [k for k in prior if k[1] == meth and m in prior[k] and ref in prior[k]]
    return f"{sum(prior[k][m]['nrmse'] < prior[k][ref]['nrmse'] for k in ks)}/{len(ks)}"


ORDER = ["nlinv", "espirit", "pinn_v53", "pinn_v54", "pinn_v56", "pinn_api", "lowres", "walsh", "inati",
         "pinn_v56_vp0", "pinn_v56_nolbfgs", "pinn_v56_nozm", "pinn_v56_grad"]
ABL = {"pinn_v56_vp0", "pinn_v56_nolbfgs", "pinn_v56_nozm", "pinn_v56_grad"}
srows = []
for m in ORDER:
    rs = [r for r in REAL if r["method"] == m]
    if not rs:
        continue
    g = lambda k: med([r.get(k) for r in rs])  # noqa: E731
    lco, wc = g("lco_outside_acs_over_floor"), g("lco_worst_coil_over_floor")
    srows.append((m, [lco - 1, wc - 1, g("ce_acs_over_floor"), g("sense_R4_support_nrmse_vs_rss_or_truth"),
                      g("sense_R8_support_nrmse_vs_rss_or_truth"),
                      "–" if m == "nlinv" else wins(m, "lco_outside_acs_over_floor"),
                      "–" if m == "nlinv" else wins(m, "sense_R4_support_nrmse_vs_rss_or_truth"),
                      med([r.get("gt_subspace_err") for r in SYN if r["method"] == m]),
                      g("runtime_s")]))
static_tbl = table([("Maps on 20 real slices (medians)", 5), ("Slices better than NLINV", 2), ("Phantoms", 1), ("", 1)],
                   ["Leave-coil-out excess", "Worst-coil excess", "ACS combine→expand / floor", "SENSE R4 NRMSE",
                    "SENSE R8 NRMSE", "on leave-coil-out", "on SENSE R4", "True map error (subspace)", "Seconds"],
                   srows, best_cols=(0, 1, 2, 3, 4, 7), abl_rows=ABL)

prows = []
for m in ["nlinv", "espirit", "pinn_v53", "pinn_v54", "pinn_v56"]:
    prows.append((m, [prior_med(m, "tikhonov"), prior_med(m, "tv"), prior_med(m, "tvw"),
                      med([v[m]["psnr"] for (c, mm), v in prior.items() if mm == "tv" and m in v]),
                      "–" if m == "nlinv" else prior_wins(m, "tv"), "–" if m == "nlinv" else prior_wins(m, "tikhonov")]))
prior_tbl = table([("Image NRMSE on 20 real slices (medians, R = 4)", 4), ("Slices better than NLINV", 2)],
                  ["Tikhonov (CG-SENSE)", "Total variation", "TV + wavelets", "TV PSNR (dB)", "with TV", "with Tikhonov"],
                  prows, best_cols=(0, 1, 2))


# paired scatter: TV image NRMSE, PINN v53 vs NLINV per slice
def scatter(pairs, xlab, ylab, w=520, h=380):
    pad_l, pad_b, pad_t, pad_r = 62, 48, 16, 16
    vals = [v for p in pairs for v in p[1:]]
    lo, hi = min(vals) * 0.92, max(vals) * 1.05
    sx = lambda v: pad_l + (v - lo) / (hi - lo) * (w - pad_l - pad_r)  # noqa: E731
    sy = lambda v: h - pad_b - (v - lo) / (hi - lo) * (h - pad_b - pad_t)  # noqa: E731
    s = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(ylab)} against {esc(xlab)} per slice">']
    ticks = [lo + i * (hi - lo) / 4 for i in range(5)]
    for t in ticks:
        s.append(f'<line class="grid" x1="{pad_l}" x2="{w - pad_r}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        s.append(f'<text x="{pad_l - 8}" y="{sy(t) + 3.5:.1f}" text-anchor="end">{t:.3f}</text>')
        s.append(f'<text x="{sx(t):.1f}" y="{h - pad_b + 16}" text-anchor="middle">{t:.3f}</text>')
    s.append(f'<line class="floor" x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}"/>')
    s.append(f'<text class="note-t" x="{sx(hi) - 4:.1f}" y="{sy(hi) + 14:.1f}" text-anchor="end">equal</text>')
    for name, x, y in pairs:
        better = y < x
        s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" class="{"pt-good" if better else "pt-bad"}">'
                 f'<title>{esc(name)}: NLINV {x:.4f}, PINN {y:.4f}</title></circle>')
    s.append(f'<text class="axis-title" x="{(pad_l + w - pad_r) / 2:.0f}" y="{h - 8}" text-anchor="middle">{esc(xlab)}</text>')
    s.append(f'<text class="axis-title" transform="translate(16 {(pad_t + h - pad_b) / 2:.0f}) rotate(-90)" text-anchor="middle">{esc(ylab)}</text>')
    s.append("</svg>")
    return "".join(s)


pairs = [(k[0].replace("_j4", ""), v["nlinv"]["nrmse"], v["pinn_v53"]["nrmse"])
         for k, v in prior.items() if k[1] == "tv" and "nlinv" in v and "pinn_v53" in v]
scatter_tv = scatter(pairs, "NLINV maps: TV image NRMSE", "PINN v53 maps: TV image NRMSE")
n_pinn_better = sum(p[2] < p[1] for p in pairs)

# ------------------------------------------------------------------ OCMR
O = [r for r in d["ocmr"] if r["run"] == "ocmr_v56"]
O54 = [r for r in d["ocmr"] if r["run"] == "ocmr_v54"]
ofiles = sorted({r["file"] for r in O})
g = collections.defaultdict(list)
for r in O:
    g[(r["R"], r["recon"], r["maps"])].append(r)
OM = ["nlinv", "espirit", "pinn_avg", "pinn_dyn0", "pinn_dyn"]
orows = []
for m in OM:
    def mm(R, rc, k="nrmse"):
        return med([r.get(k) for r in g.get((R, rc, m), [])])
    orows.append((m, [mm(1, "sense1", "lco_outside_over_floor"), mm(4, "sense"), mm(4, "tv"), mm(8, "sense"), mm(8, "tv"),
                      mm(4, "tv", "psnr"), mm(8, "tv", "psnr")]))
ocmr_tbl = table([("R = 1", 1), ("R = 4", 2), ("R = 8", 2), ("TV PSNR (dB)", 2)],
                 ["Leave-coil-out / floor", "SENSE NRMSE", "TV NRMSE", "SENSE NRMSE", "TV NRMSE", "R = 4", "R = 8"],
                 orows, best_cols=(0, 1, 2, 3, 4))
# per-file R=4 TV
pf = collections.defaultdict(dict)
for r in O:
    if r["R"] == 4 and r["recon"] == "tv":
        pf[r["file"]][r["maps"]] = r["nrmse"]
lco1 = collections.defaultdict(dict)
for r in O:
    if r["R"] == 1:
        lco1[r["file"]][r["maps"]] = r.get("lco_outside_over_floor")
pfh = ['<table><thead><tr class="group"><th></th><th colspan="5">R = 4, TV image NRMSE</th>'
       '<th colspan="2" class="gstart">R = 1, leave-coil-out / floor</th></tr><tr><th>Case</th>'
       + "".join(f"<th>{esc(LABEL[m])}</th>" for m in OM) + "<th>PINN, time-averaged</th><th>PINN dynamic, rank 0</th></tr></thead><tbody>"]
for f in ofiles:
    row = pf[f]
    vals = [row.get(m) for m in OM]
    best = min(v for v in vals if v is not None)
    bad = row.get("pinn_dyn0") is not None and row.get("pinn_avg") is not None and row["pinn_dyn0"] > 1.1 * row["pinn_avg"]
    cells = "".join(f'<td class="num{" best" if v is not None and abs(v - best) < 1e-12 else ""}{" flag" if (m == "pinn_dyn0" and bad) else ""}">{fmt(v)}</td>' for m, v in zip(OM, vals))
    l1 = lco1[f]
    cells += f'<td class="num gstart">{fmt(l1.get("pinn_avg"), 3)}</td><td class="num{" flag" if bad else ""}">{fmt(l1.get("pinn_dyn0"), 3)}</td>'
    field = "3 T" if "3T" in f else "1.5 T"
    pfh.append(f'<tr><td>{esc(f.split("_")[0] + "_" + f.split("_")[1])} <span class="muted">{field}</span></td>{cells}</tr>')
pfh.append("</tbody></table>")
ocmr_pf = "".join(pfh)
n_dyn_bad = sum(1 for f in ofiles if pf[f].get("pinn_dyn0") and pf[f].get("pinn_avg") and pf[f]["pinn_dyn0"] > 1.1 * pf[f]["pinn_avg"])

# ------------------------------------------------------------------ prospective
U = d["ocmr_us"]
uf = sorted({r["file"] for r in U})
ut = collections.defaultdict(dict)
for r in U:
    ut[(r["file"], r["recon"])][r["maps"]] = r
UM = ["nlinv", "espirit", "pinn_avg", "pinn_dyn"]
uh = ['<table><thead><tr class="group"><th></th><th colspan="4">Held-out k-space error / noise, TV</th>'
      '<th colspan="4" class="gstart">Same, central lines only</th></tr><tr><th>Case</th>'
      + "".join(f"<th>{esc(LABEL[m])}</th>" for m in UM) + "".join(f"<th>{esc(LABEL[m])}</th>" for m in UM) + "</tr></thead><tbody>"]
for f in uf:
    row = ut[(f, "tv")]
    v1 = [row[m]["heldout"] if m in row else None for m in UM]
    v2 = [row[m]["heldout_centre"] if m in row else None for m in UM]
    b1, b2 = min(x for x in v1 if x is not None), min(x for x in v2 if x is not None)
    c = "".join(f'<td class="num{" best" if x == b1 else ""}">{fmt(x, 2)}</td>' for x in v1)
    c += "".join(f'<td class="num{" gstart" if i == 0 else ""}{" best" if x == b2 else ""}">{fmt(x, 2)}</td>' for i, x in enumerate(v2))
    uh.append(f"<tr><td>{esc(f)}</td>{c}</tr>")
uh.append("</tbody></table>")
us_tbl = "".join(uh)

# ------------------------------------------------------------------ brain phase poles
P = d["poles"]
pg = collections.defaultdict(list)
for r in P:
    pg[(r["cal"], r["maps"])].append(r)
PM = ["nlinv", "nlinv_pp", "espirit", "pinn_v54", "pinn_v54@seed=1", "pinn_v55", "pinn_v55@seed=1",
      "pinn_v55@init_from_acs=False", "pinn_v55@joint_lbfgs=0", "pinn_v55@resid_zero_mean=False", "pinn_v55@joint_lbfgs_cg=0"]
PLAB = {"pinn_v54@seed=1": "PINN v54, seed 1", "pinn_v55@seed=1": "PINN v55, seed 1",
        "pinn_v55@init_from_acs=False": "v55 without ACS support fix", "pinn_v55@joint_lbfgs=0": "v55 without L-BFGS finish",
        "pinn_v55@resid_zero_mean=False": "v55 without zero-mean factor", "pinn_v55@joint_lbfgs_cg=0": "v55 without var. projection",
        "pinn_v54": "PINN v54 (joint, default)"}
LABEL.update(PLAB)
bh = ['<table><thead><tr class="group"><th></th>' + "".join(
    f'<th colspan="3" class="{"gstart" if i else ""}">ACS {c}×{c}</th>' for i, c in enumerate((3, 7, 15)))
      + '</tr><tr><th>Maps</th>' + "<th>Poles</th><th>lco</th><th>SENSE</th>" * 3 + "</tr></thead><tbody>"]
bests = {}
for c in (3, 7, 15):
    vals = [med([r["sense_nrmse"] for r in pg[(c, m)]]) for m in PM if pg.get((c, m))]
    bests[c] = min(v for v in vals if v is not None and v == v)
for m in PM:
    abl = "@" in m and "seed" not in m
    cells = []
    for i, c in enumerate((3, 7, 15)):
        rs = pg.get((c, m), [])
        if not rs:
            cells.append(f'<td class="num{" gstart" if i else ""}" colspan="3">fails</td>' if m == "espirit" else
                         f'<td class="num" colspan="3">–</td>')
            continue
        poles = sum(r["gauge_poles"] for r in rs)
        lco = med([r["lco_object"] for r in rs]); s4 = med([r["sense_nrmse"] for r in rs])
        if m == "espirit" and c == 7:
            cells.append(f'<td class="num gstart" colspan="3">fails (lco {lco:.0f})</td>'); continue
        pc = f'<span class="pill bad">{poles}</span>' if poles else '<span class="pill ok">0</span>'
        b = " best" if s4 is not None and abs(s4 - bests[c]) < 1e-12 else ""
        cells.append(f'<td class="num{" gstart" if i else ""}">{pc}</td><td class="num">{lco:.3f}</td><td class="num{b}">{s4:.4f}</td>')
    bh.append(f'<tr class="{"abl" if abl else ""}"><td>{swatch(m)}</td>{"".join(cells)}</tr>')
bh.append("</tbody></table>")
brain_tbl = "".join(bh)

# explorer data (static real + synthetic)
KEYS = ["lco_outside_acs_over_floor", "lco_worst_coil_over_floor", "ce_acs_over_floor", "ssim_leave_coil_out",
        "sense_R4_support_nrmse_vs_rss_or_truth", "sense_R8_support_nrmse_vs_rss_or_truth", "gt_subspace_err", "runtime_s"]
EX = collections.defaultdict(list)
for r in E:
    EX[r["case"]].append({"m": r["method"], "l": LABEL.get(r["method"], r["method"]), **{k: r.get(k) for k in KEYS}})
explorer_json = json.dumps(EX, separators=(",", ":"))

# headline numbers
nl_tv, pv_tv = prior_med("nlinv", "tv"), prior_med("pinn_v53", "tv")
o_nl, o_pa = med([r["nrmse"] for r in g[(4, "tv", "nlinv")]]), med([r["nrmse"] for r in g[(4, "tv", "pinn_avg")]])
o_es = med([r["nrmse"] for r in g[(4, "tv", "espirit")]])

page = open(sys.argv[3]).read()
rep = {
    "{{STATIC}}": static_tbl, "{{PRIOR}}": prior_tbl, "{{SCATTER}}": scatter_tv, "{{OCMR}}": ocmr_tbl,
    "{{OCMR_PF}}": ocmr_pf, "{{US}}": us_tbl, "{{BRAIN}}": brain_tbl, "{{EXPLORER}}": explorer_json,
    "{{N_BETTER}}": str(n_pinn_better), "{{N_PAIRS}}": str(len(pairs)), "{{NL_TV}}": f"{nl_tv:.4f}", "{{PV_TV}}": f"{pv_tv:.4f}",
    "{{O_NL}}": f"{o_nl:.4f}", "{{O_PA}}": f"{o_pa:.4f}", "{{O_ES}}": f"{o_es:.4f}", "{{N_OCMR}}": str(len(ofiles)),
    "{{N_DYN_BAD}}": str(n_dyn_bad), "{{N_US}}": str(len(uf)),
}
for k, v in rep.items():
    page = page.replace(k, v)
open(OUT, "w").write(page)
print("written", OUT, len(page))
