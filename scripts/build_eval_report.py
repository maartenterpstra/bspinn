"""Build the interactive HTML report from runs/eval/summary.json (+ fig_*.png).

    python scripts/summarize_eval.py --eval runs/eval
    python scripts/build_eval_report.py --eval runs/eval --out runs/eval/report/index.html

Images are copied next to the HTML (report/img/). Findings text lives in FINDINGS below and is
written from the numbers in summary.json.
"""
import argparse
import json
import os
import shutil

CASE_INFO = {
    "synth8": ("Synthetic, 8 coils", "synthetic", "Biot–Savart ground truth, one ring of 8 loops, 128², σ = 0.01, ACS 24×24."),
    "synth8_noiselo": ("Synthetic, low noise", "synthetic", "As synth8 with σ = 0.003."),
    "synth8_noisehi": ("Synthetic, high noise", "synthetic", "As synth8 with σ = 0.03."),
    "synth8_mismatch": ("Synthetic, model mismatch", "synthetic",
                        "As synth8, but every coil is multiplied by its own smooth complex factor exp(0.3·(a+ib)) (random quadratics): the true maps are not Biot–Savart fields."),
    "knee15": ("Knee, 15 ch (real)", "real", "Fully sampled coronal knee (BART workshop, fastMRI-style), readout oversampling removed, pre-whitened, 320×368, 332 of 368 PE lines acquired (partial Fourier), ACS 24×24."),
    "knee15_acs16": ("Knee, ACS 16", "real", "Knee with a 16×16 ACS block."),
    "knee15_acs32": ("Knee, ACS 32", "real", "Knee with a 32×32 ACS block."),
    "knee15_acs48": ("Knee, ACS 48", "real", "Knee with a 48×48 ACS block."),
    "brain8": ("Brain, 8 ch (prospective)", "prospective", "Axial brain (BART workshop pics-brain), prospective 2-D Poisson disc, 12.7 % of k-space, fully sampled centre 20×20. ESPIRiT, Walsh, Inati and low-res division use the 20×20 centre; NLINV calibrates on its centre; the Biot–Savart fit uses all samples in the central 48×48 (k-space data term)."),
}
CASE_ORDER = ["knee15", "brain8", "synth8", "synth8_noiselo", "synth8_noisehi", "synth8_mismatch",
              "knee15_acs16", "knee15_acs32", "knee15_acs48"]

CASE3D = {
    "wave15": ("Full-wave 1.5 T", "48³, 240 mm FOV, 12 loops: exact time-harmonic fields in homogeneous tissue at 64 MHz (ε_r 70, σ 0.5 S/m; coils and object in one medium, which exaggerates wave effects), σ = 0.01 relative to the object signal, ACS 12³. The Helmholtz model starts from a generic prior (ε_r 60, σ 0.5)."),
    "wave3": ("Full-wave 3 T", "As full-wave 1.5 T at 128 MHz (ε_r 50, σ 0.6 S/m). 'Helmholtz (true k²)' fixes k² at the true tissue value; 'k² learned' starts from the generic prior."),
    "synth3d": ("Synthetic 3-D", "48³ volume, 12 loops in two rings along B0, Biot–Savart ground truth, σ = 0.01, ACS 12³ for every method."),
    "synth3d_mismatch": ("Synthetic 3-D, mismatch", "As synthetic 3-D, each coil multiplied by its own smooth complex non-Biot–Savart factor (strength 0.3)."),
    "bart3d": ("BART 3-D phantom", "BART's 3-D phantom with its simulated 8-channel head-coil sensitivities (not Biot–Savart fields), σ = 0.01, ACS 12³. The B0 direction is selected from the data."),
}

FINDINGS = []   # filled in by --findings JSON file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="runs/eval")
    ap.add_argument("--out", default="runs/eval/report/index.html")
    ap.add_argument("--findings", default=None, help="JSON list of {title, text} written from the results")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_report_template.html"))
    a = ap.parse_args()
    S = json.load(open(os.path.join(a.eval, "summary.json")))
    out_dir = os.path.dirname(a.out)
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    imgs = {}
    for c in S["cases"]:
        for kind in ("recon", "maps"):
            src = os.path.join(a.eval, f"fig_{c}_{kind}.png")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out_dir, "img", f"{c}_{kind}.png"))
                imgs.setdefault(c, {})[kind] = f"img/{c}_{kind}.png"
    cases = [c for c in CASE_ORDER if c in S["cases"]] + [c for c in S["cases"] if c not in CASE_ORDER]
    data = {"cases": cases, "caseInfo": {c: {"label": CASE_INFO.get(c, (c, "", ""))[0], "kind": CASE_INFO.get(c, ("", "real", ""))[1],
                                             "desc": CASE_INFO.get(c, ("", "", ""))[2], **S["cases"][c]} for c in cases},
            "order": S["order"], "labels": S["labels"], "metrics": S["metrics"], "profiles": S["radial_profiles"],
            "validity": S["validity_spearman"], "images": imgs,
            "findings": json.load(open(a.findings)) if a.findings else FINDINGS}
    p3 = os.path.join(os.path.dirname(a.eval.rstrip("/")), "eval3d")
    if os.path.exists(os.path.join(p3, "summary3d.json")):
        S3 = json.load(open(os.path.join(p3, "summary3d.json")))
        im3 = {}
        for c in S3["cases"]:
            src = os.path.join(p3, f"fig3d_{c}.png")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out_dir, "img", f"3d_{c}.png")); im3[c] = f"img/3d_{c}.png"
        order3 = [c for c in ("wave15", "wave3", "synth3d", "synth3d_mismatch", "bart3d") if c in S3["cases"]]
        data["three"] = {"cases": order3, "caseInfo": {c: {"label": CASE3D.get(c, (c, ""))[0], "desc": CASE3D.get(c, ("", ""))[1], **S3["cases"][c]} for c in order3},
                         "order": S3["order"], "labels": S3["labels"], "metrics": S3["metrics"], "images": im3}
    payload = json.dumps(_clean(data), separators=(",", ":")).replace("</", "<\\/")
    html = open(a.template).read().replace("/*__DATA__*/null", payload)
    with open(a.out, "w") as f:
        f.write(html)
    print(f"wrote {a.out} ({len(html) / 1e3:.0f} kB) and {sum(len(v) for v in imgs.values())} images")


def _clean(o):
    import math
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


if __name__ == "__main__":
    main()
