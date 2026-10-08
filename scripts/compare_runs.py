"""Before/after comparison of two benchmark runs (results.json per case).

    python scripts/compare_runs.py --old runs/prev_v7 --new runs --out runs/compare_v8.json

For every case and every method present in both runs: the headline metrics, old -> new, and the
rank of each Biot–Savart variant among all methods of the case (old -> new).
"""
import argparse
import json
import os

FIT = ("bs_pinn", "bs_wires", "free", "bs_pinn_grid1", "bs_pinn_complex", "bs_helmholtz", "bs_helmholtz_fixed")


def metrics(m, three_d):
    out = {}
    if "prospective" in m:
        out["heldout_nrmse"] = m["prospective"]["heldout_nrmse"]
    mp = m.get("maps", {})
    for k in ("gt_subspace_err", "gt_absolute_err", "coverage", "phase_vortices"):
        if k in mp:
            out[k] = mp[k]
    if "kspace_consistency" in m:
        out["k_out_over_floor"] = m["kspace_consistency"]["outside_acs"]["residual_over_noise_floor"]
    if "reference_image" in m:
        out["snr_eff"] = m["reference_image"]["snr_efficiency_mean"]
    R = "4x2" if three_d else "4"
    if "sense" in m and R in m["sense"]:
        out[f"sense_R{R}_delivered"] = m["sense"][R]["vs_reference"]["nrmse"]
        out[f"sense_R{R}_support"] = m["sense_support"][R]["vs_reference"]["nrmse"]
    return out


LOWER = {"heldout_nrmse", "gt_subspace_err", "gt_absolute_err", "phase_vortices", "k_out_over_floor",
         "sense_R4_delivered", "sense_R4_support", "sense_R4x2_delivered", "sense_R4x2_support"}


def rank(allm, name, key):
    vals = {n: v[key] for n, v in allm.items() if key in v and v[key] is not None and n != "oracle"}
    if name not in vals:
        return None
    order = sorted(vals, key=lambda n: vals[n] if key in LOWER else -vals[n])
    return f"{order.index(name) + 1}/{len(order)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="runs/prev_v7")
    ap.add_argument("--new", default="runs")
    ap.add_argument("--out", default="runs/compare_v8.json")
    a = ap.parse_args()
    report = {}
    for sub, three in (("eval", False), ("eval3d", True)):
        d_new = os.path.join(a.new, sub)
        for case in sorted(os.listdir(d_new)):
            pn, po = os.path.join(d_new, case, "results.json"), os.path.join(a.old, sub, case, "results.json")
            if not (os.path.exists(pn) and os.path.exists(po)):
                continue
            rn, ro = json.load(open(pn)), json.load(open(po))
            mn = {k: metrics(v, three) for k, v in rn["methods"].items()}
            mo = {k: metrics(v, three) for k, v in ro["methods"].items()}
            rep = {}
            for meth in FIT:
                if meth in mn and meth in mo:
                    rep[meth] = {k: {"old": mo[meth].get(k), "new": mn[meth][k],
                                     "rank_old": rank(mo, meth, k), "rank_new": rank(mn, meth, k)}
                                 for k in mn[meth]}
                    rep[meth]["runtime_s"] = {"old": ro.get("runtime_s", {}).get(meth), "new": rn.get("runtime_s", {}).get(meth)}
            best = {}
            for k in next(iter(mn.values())).keys():
                vals = {n: v.get(k) for n, v in mn.items() if v.get(k) is not None and n != "oracle"}
                if vals:
                    b = min(vals, key=vals.get) if k in LOWER else max(vals, key=vals.get)
                    best[k] = [b, vals[b]]
            rep["_best_new"] = best
            report[case] = rep
            print(f"== {case}")
            for meth in FIT:
                if meth in rep:
                    print("  " + meth + ": " + "; ".join(
                        f"{k} {v['old']:.4g}->{v['new']:.4g} ({v['rank_old']}->{v['rank_new']})"
                        for k, v in rep[meth].items() if k != "runtime_s" and v["old"] is not None and v["new"] is not None))
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
