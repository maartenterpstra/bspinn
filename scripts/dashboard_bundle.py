"""Collect evaluation results into one JSON for the benchmark dashboard (send this file back to rebuild it).

    python scripts/dashboard_bundle.py --eval runs/eval_v56 --prior runs/prior_v56.jsonl \\
        --ocmr runs/ocmr_v54/results.jsonl runs/ocmr_v56/results.jsonl --ocmr-us runs/ocmr_us/results.jsonl \\
        --poles runs/poles/results.jsonl --out dashboard_data.json

--eval: an evaluate_methods.py output folder (summary.tsv: one row per case x method; ranks.tsv). The others are
the JSON-lines files of prior_eval.py, ocmr_eval.py, ocmr_prospective.py and phase_pole_eval.py; several files per
option are tagged by their folder name (e.g. runs/ocmr_v54/results.jsonl -> 'ocmr_v54'). Missing files are skipped."""
import argparse
import json
import os


def tsv(path):
    if not os.path.exists(path):
        return []
    lines = [l.rstrip("\n").split("\t") for l in open(path) if l.strip()]
    head, out = lines[0], []
    for parts in lines[1:]:
        r = {}
        for k, v in zip(head, parts):
            try:
                r[k] = None if v == "-" else float(v)
            except ValueError:
                r[k] = v
        out.append(r)
    return out


def jsonl(paths):
    rows = []
    for p in paths or []:
        if not os.path.exists(p):
            print(f"skipped (missing): {p}")
            continue
        tag = os.path.basename(os.path.dirname(os.path.abspath(p))) if os.path.basename(p) == "results.jsonl" \
            else os.path.splitext(os.path.basename(p))[0]
        bad = 0
        for n_, l in enumerate(open(p), 1):
            if l.strip():
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:              # a line cut off by a crash / two writers
                    bad += 1
                    print(f"{p}: line {n_} is not valid JSON, skipped")
                    continue
                r.pop("per_frame_nrmse", None); r.pop("per_frame_ssim", None)          # keep the bundle small
                r.pop("cv_error", None); r.pop("grid_error", None)
                r["run"] = tag
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", nargs="*", default=[], help="evaluate_methods.py output folders")
    ap.add_argument("--prior", nargs="*", default=[])
    ap.add_argument("--ocmr", nargs="*", default=[])
    ap.add_argument("--ocmr-us", nargs="*", default=[])
    ap.add_argument("--poles", nargs="*", default=[])
    ap.add_argument("--note", default="", help="free text shown with the run (code version, settings)")
    ap.add_argument("--out", default="dashboard_data.json")
    a = ap.parse_args()
    out = {"note": a.note, "eval": [], "ranks": [], "prior": jsonl(a.prior), "ocmr": jsonl(a.ocmr),
           "ocmr_us": jsonl(a.ocmr_us), "poles": jsonl(a.poles)}
    for d in a.eval:
        tag = os.path.basename(os.path.abspath(d))
        for r in tsv(os.path.join(d, "summary.tsv")):
            r["run"] = tag; out["eval"].append(r)
        for r in tsv(os.path.join(d, "ranks.tsv")):
            r["run"] = tag; out["ranks"].append(r)
    json.dump(out, open(a.out, "w"))
    print(f"{a.out}: " + ", ".join(f"{k} {len(v)} rows" for k, v in out.items() if isinstance(v, list)))


if __name__ == "__main__":
    main()
