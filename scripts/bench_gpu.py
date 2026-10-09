"""GPU benchmark and correctness check for bspinn.

    python scripts/bench_gpu.py                      # all checks and timings on cuda:0
    python scripts/bench_gpu.py --device cuda:1 --quick
    python scripts/bench_gpu.py --only timings --cases 2d,joint

1. Correctness (device vs CPU)
   * kernels: the same model parameters evaluated on CPU and on the device (wire field, PINN maps,
     wave-wire field, physics losses) must agree to float32 precision;
   * end to end: estimate_maps on a synthetic 2-D case on CPU and on the device; both subspace
     errors against the ground truth are printed (the random collocation streams differ between
     CPU and CUDA generators, so the maps are not bitwise equal, only equally accurate).
2. Timings per optimisation iteration (ms/it, from the difference of a short and a long run, so that
   per-stage setup cancels; the one-off cost of the first call, i.e. compilation, is *_first_call_s), for
   eager | graph (cfg.cuda_graph: each iteration replayed from one captured CUDA graph) | compile
   [+tf32], each mode in a fresh subprocess (torch.compile and the TF32 switches are process-global),
   plus a torch.profiler summary of 12 PINN iterations per mode (top operators, kernel launches):
     2d     8 coils, 256x256, calib 24: wire stage, PINN stage, full-res maps evaluation
     joint  the solve_image stage on 256x256: hybrid vs exact maps
     3d     8 coils, 48^3, calib 12^3: wire stage, PINN stage (Laplace)
     helm   8 coils, 32^3, 3 T Helmholtz: wave-wire stage, Helmholtz PINN stage
Results go to stdout and --out (JSON).
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

QUIET = dict(log=lambda *a, **k: None)


def sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)


def timed(dev, f, n):
    sync(dev); t = time.perf_counter()
    r = f()
    sync(dev)
    return r, (time.perf_counter() - t) / n * 1e3


# --------------------------------------------------------------------------- correctness
def check_kernels(dev):
    from bspinn.models import HarmonicPINN, HelmholtzPINN, WaveWireLoopCoils, WireLoopCoils, tissue_k2
    torch.manual_seed(0)
    w = WireLoopCoils(8, 1, 12, 1)
    with torch.no_grad():
        w.verts.normal_(0, 0.3); w.verts.add_(torch.tensor([1.5, 0, 0])); w.amp.fill_(1)
    p = HarmonicPINN(8, "complex", 16, 64, 4, 1, base=w)
    torch.nn.init.normal_(p.net[-1].weight, std=0.1)
    ww = WaveWireLoopCoils(8, 1, 12, 1, k2_init=tissue_k2(3.0, 50, 0.6, 0.12))
    with torch.no_grad():
        ww.verts.copy_(w.verts); ww.amp.fill_(1)
    hp = HelmholtzPINN(8, 16, 64, 4, 1, base=ww)
    torch.nn.init.normal_(hp.net[-1].weight, std=0.1)
    pts = torch.rand(20000, 3) * 2 - 1
    out = {}
    tests = {
        "wire field": (w, lambda m, x: m.field(x)),
        "pinn maps": (p, lambda m, x: m.sensitivity(x)),
        "pinn physics loss": (p, lambda m, x: m.physics_loss(x[:2048])),
        "wave-wire field": (ww, lambda m, x: m.field(x[:4096], create_graph=False)),
        "helmholtz physics loss": (hp, lambda m, x: hp_loss(m, x[:1024])),
    }
    for name, (m, f) in tests.items():
        a = f(m.cpu(), pts).detach()
        b = f(m.to(dev), pts.to(dev)).detach().cpu()
        m.cpu()
        rel = float((a - b).abs().max() / a.abs().max().clamp_min(1e-30))
        out[name] = rel
        print(f"  {name:24s} max |cpu - {dev}| / max|cpu| = {rel:.2e}  {'ok' if rel < 1e-3 else 'CHECK'}", flush=True)
    return out


def hp_loss(m, x):
    return m.physics_loss(x)


def check_end_to_end(dev, quick):
    from bspinn.api import estimate_maps
    from bspinn.data import make_synthetic
    from bspinn.recon import subspace_error
    d, gt, obj = make_synthetic(8, (128, 128), noise=0.01)
    k = d.kspace[0]
    sup = torch.from_numpy((np.abs(obj[:1]) > 0).astype(np.float32))           # object support
    res = {}
    runs = [("cpu", "cpu", {}), (dev, dev, {})]
    if torch.device(dev).type == "cuda":
        runs.append((dev + " cuda_graph", dev, {"cuda_graph": True}))
    for name, device, extra in runs:
        t = time.time()
        C, info = estimate_maps(k, "2d", calib=24, quality="fast", device=device, verbose=False,
                                **({"iters_pinn": 150, "iters_wires": 100} if quick else {}), **extra)
        e = subspace_error(torch.from_numpy(C)[None], torch.from_numpy(gt[:1]), sup)
        from bspinn import accel
        res[name] = {"subspace_err": e, "seconds": time.time() - t, "cuda_graph_stats": dict(accel.STATS)}
        accel.STATS.update(captured=0, fallback=0, last_error=None)
        print(f"  estimate_maps 128^2 on {name:18s}: subspace error {e:.2e}, {time.time() - t:.1f} s", flush=True)
    return res


# --------------------------------------------------------------------------- timings (child process)
def on_device(prob, dev):
    """Problems are built on the CPU; fit() / fit3d() move them to cfg.device (v10/v11 of this script did
    not, so their per-stage timings were CPU timings)."""
    prob = prob.to(dev)
    assert prob.coords.device.type == torch.device(dev).type, "problem not on the benchmark device"
    return prob


def per_iter(dev, run, n, k0=2):
    """(result, ms per iteration, first-call overhead in s) from runs of k0 and k0 + n iterations, after a
    first call that pays the one-off costs (with cfg.compile: tracing and compiling the stage's graph,
    reused by later stages with the same shapes)."""
    _, t0 = timed(dev, lambda: run(k0), 1)                                  # warm caches / compile
    _, t1 = timed(dev, lambda: run(k0), 1)
    r, t2 = timed(dev, lambda: run(k0 + n), 1)
    ms = (t2 - t1) / n
    return r, ms, max(0.0, (t0 - t1) / 1e3)


def _cfg(**kw):
    from bspinn.fit import FitConfig
    return FitConfig(log_every=10 ** 6, compile=MODE["compile"], tf32=MODE["tf32"], compile_min_iters=0,
                     cuda_graph=MODE["graph"], **kw)


def bench_2d(dev, n):
    from bspinn.data import make_synthetic
    from bspinn.fit import Problem, _wrap, fit_pinn, fit_wires, make_pinn
    d, _, _ = make_synthetic(8, (256, 256), noise=0.01)
    cfg = _cfg(calib=24, device=dev)
    from bspinn import accel; accel.configure(cfg)
    d.prewhiten(); prob = on_device(Problem(d, cfg), dev)
    (w, m, _, _), tw, cw = per_iter(dev, lambda k: fit_wires(prob, cfg, 1, k, **QUIET), n)
    pinn = make_pinn(prob, cfg, w, 1); wr = _wrap(pinn, prob, cfg, prev=w)
    _, tp, cp = per_iter(dev, lambda k: fit_pinn(prob, cfg, wr, m, iters=k, **QUIET), n)
    prof = profile_stage(dev, lambda: fit_pinn(prob, cfg, wr, m, iters=12, **QUIET))
    pts = torch.from_numpy(np.asarray(d.geom.grid(), np.float32).reshape(-1, 3))      # full-res 256^2 grid
    pinn.to(dev)
    pinn.sensitivity_batched(pts.to(dev))
    _, te = timed(dev, lambda: pinn.sensitivity_batched(pts.to(dev)), 1)
    return {"wire_ms_per_it": tw, "pinn_ms_per_it": tp, "maps_eval_ms": te,
            "wire_first_call_s": cw, "pinn_first_call_s": cp, "pinn_profile_12_it": prof}


def profile_stage(dev, run):
    """torch.profiler over one short stage: the top operators by host time and by device time, and the
    number of kernel launches (tells launch-bound from compute-bound, and what a compiled graph runs)."""
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.device(dev).type == "cuda" else [])
    try:
        with profile(activities=acts) as pr:
            run()
            sync(dev)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    ev = pr.key_averages()
    dk = "self_device_time_total" if hasattr(ev[0], "self_device_time_total") else "self_cuda_time_total"
    rows = lambda key: [{"op": e.key[:80], "calls": e.count, "self_host_ms": e.self_cpu_time_total / 1e3,
                         "self_device_ms": getattr(e, dk, 0) / 1e3} for e in sorted(ev, key=lambda e: -getattr(e, key, 0))[:15]]
    launches = sum(e.count for e in ev if e.key.startswith(("cudaLaunchKernel", "cuLaunchKernel", "cudaGraphLaunch")))
    return {"by_host_time": rows("self_cpu_time_total"), "by_device_time": rows(dk),
            "total_host_ms": sum(e.self_cpu_time_total for e in ev) / 1e3,
            "total_device_ms": sum(getattr(e, dk, 0) for e in ev) / 1e3, "kernel_launch_calls": launches}


def bench_joint(dev, n):
    import copy
    from bspinn.data import make_synthetic
    from bspinn.fit import FitConfig, Problem, _wrap, fit_joint_full, fit_wires, make_pinn
    d, _, _ = make_synthetic(8, (256, 256), noise=0.01)
    cfg = FitConfig(calib=24, log_every=10 ** 6, device=dev, joint_full=True, compile=MODE["compile"], tf32=MODE["tf32"])
    from bspinn import accel; accel.configure(cfg)
    d.prewhiten(); prob = on_device(Problem(d, cfg), dev)
    w, m, _, _ = fit_wires(prob, cfg, 1, 5, **QUIET)
    wr = _wrap(make_pinn(prob, cfg, w, 1), prob, cfg, prev=w)
    out = {}
    for mode in ("hybrid", "exact"):
        c = copy.copy(cfg); c.joint_maps = mode
        c.iters_joint = 2; fit_joint_full(prob, c, copy.deepcopy(wr), **QUIET)
        c.iters_joint = 2 + n
        _, t_long = timed(dev, lambda: fit_joint_full(prob, c, copy.deepcopy(wr), **QUIET), 1)
        c.iters_joint = 2
        _, t_short = timed(dev, lambda: fit_joint_full(prob, c, copy.deepcopy(wr), **QUIET), 1)
        out[f"joint_{mode}_ms_per_it"] = (t_long - t_short) / n                # setup/final CG cancel
    return out


def bench_3d(dev, n, helmholtz):
    from bspinn.fit import fit_pinn, fit_wires, make_pinn
    from bspinn.models import tissue_k2
    from bspinn.volume import CoarseCorrection, Problem3D, make_synthetic3d
    if helmholtz:
        k2 = tissue_k2(3.0, 50, 0.6, 0.12)
        d, _, _ = make_synthetic3d(8, (32, 32, 32), noise=0.01, wave_k2=k2, voxel=(7.5,) * 3, normalise="object")
        cfg = _cfg(calib=(12, 12, 12), physics="helmholtz", device=dev)
    else:
        d, _, _ = make_synthetic3d(8, (48, 48, 48), noise=0.01)
        cfg = _cfg(calib=(12, 12, 12), device=dev)
    from bspinn import accel; accel.configure(cfg)
    d.prewhiten(); prob = on_device(Problem3D(d, cfg), dev)
    (w, m, _, _), tw, cw = per_iter(dev, lambda k: fit_wires(prob, cfg, 1, k, **QUIET), n)
    p = make_pinn(prob, cfg, w, 1); cc = CoarseCorrection(p, prob)
    _, tp, cp = per_iter(dev, lambda k: fit_pinn(prob, cfg, cc, m, iters=k, **QUIET), n)
    return {"wire_ms_per_it": tw, "pinn_ms_per_it": tp, "wire_first_call_s": cw, "pinn_first_call_s": cp}


MODE = {"compile": False, "tf32": False, "graph": False}


def check_syncs(dev, iters=6):
    """Host-device synchronisations inside the training loops (torch.cuda.set_sync_debug_mode): where
    they happen, per stage. Logging iterations (.item()) are excluded by a large log_every."""
    import traceback
    import warnings
    from bspinn.data import make_synthetic
    from bspinn.fit import Problem, fit_pinn, fit_wires, make_pinn
    d, _, _ = make_synthetic(8, (128, 128), noise=0.01)
    cfg = _cfg(calib=24, device=dev)
    d.prewhiten(); prob = on_device(Problem(d, cfg), dev)
    w, m, _, _ = fit_wires(prob, cfg, 1, 2, **QUIET)
    p = make_pinn(prob, cfg, w, 1)
    sites = {}

    def show(message, category, filename, lineno, file=None, line=None):
        if "synchroniz" in str(message):
            st = [f"{os.path.basename(f.filename)}:{f.lineno}" for f in traceback.extract_stack()
                  if "warnings" not in f.filename and "bench_gpu" not in f.filename]
            key = " < ".join(st[::-1][:5])
            sites[key] = sites.get(key, 0) + 1
    old = warnings.showwarning
    warnings.showwarning = show
    torch.cuda.set_sync_debug_mode(1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = show
            fit_wires(prob, cfg, 1, iters + 1, **QUIET)
            fit_pinn(prob, cfg, p, m, iters=iters + 1, **QUIET)
    finally:
        torch.cuda.set_sync_debug_mode(0)
        warnings.showwarning = old
    # the last iteration of each stage logs (2 .item() calls): expected
    return {"syncs_per_2_stages_of_%d_it" % (iters + 1): sites}


def child(a):
    base = a.mode.split("+")[0]
    MODE.update(compile={"eager": False, "graph": False, "compile": True}.get(base, base), tf32="tf32" in a.mode,
                graph=base == "graph")
    n = 20 if a.quick else 100
    res = {}
    cases = a.cases.split(",")
    if base == "eager" and torch.device(a.device).type == "cuda":
        cases = ["syncs"] + cases
    for case in cases:
        if torch.device(a.device).type == "cuda":
            torch.cuda.synchronize(a.device); torch.cuda.reset_peak_memory_stats(a.device)
        try:
            if case == "syncs":
                res[case] = check_syncs(a.device)
            elif case == "2d":
                res[case] = bench_2d(a.device, n)
            elif case == "joint":
                res[case] = bench_joint(a.device, max(2, n // 20))
            elif case == "3d":
                res[case] = bench_3d(a.device, n // 2, False)
            elif case == "helm":
                res[case] = bench_3d(a.device, n // 2, True)
        except Exception as e:                                     # report and carry on
            res[case] = {"error": f"{type(e).__name__}: {e}"}
        if torch.device(a.device).type == "cuda":
            res[case]["peak_mem_GB"] = torch.cuda.max_memory_allocated(a.device) / 1e9
        from bspinn import accel
        res[case]["cuda_graph_stats"] = dict(accel.STATS)
        accel.STATS.update(captured=0, fallback=0, last_error=None)
    print("RESULT " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default="all", choices=["all", "correctness", "timings"])
    ap.add_argument("--cases", default="2d,joint,3d,helm")
    ap.add_argument("--modes", default="eager,graph,graph+tf32,compile",
                    help="comma list of eager | graph (CUDA-graph replay) | compile | reduce-overhead | max-autotune, "
                         "each optionally +tf32")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="bench_gpu.json")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--mode", default="eager", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._child:
        return child(a)
    report = {"device": a.device, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(a.device) if torch.device(a.device).type == "cuda" else None}
    print(f"bspinn GPU benchmark on {a.device} ({report['gpu'] or 'no GPU'}), torch {torch.__version__}", flush=True)
    if a.only in ("all", "correctness"):
        print("correctness: kernels", flush=True)
        report["kernels"] = check_kernels(a.device)
        print("correctness: end to end", flush=True)
        report["end_to_end"] = check_end_to_end(a.device, a.quick)
    if a.only in ("all", "timings"):
        report["timings"] = {}
        for mode in a.modes.split(","):
            print(f"timings: {mode}", flush=True)
            cmd = [sys.executable, os.path.abspath(__file__), "--_child", "--mode", mode, "--device", a.device,
                   "--cases", a.cases] + (["--quick"] if a.quick else [])
            env = dict(os.environ)
            if mode.split("+")[0] != "eager":                    # report recompiles / skipped CUDA graphs
                env["TORCH_LOGS"] = "recompiles,perf_hints"
            out = subprocess.run(cmd, capture_output=True, text=True, env=env)
            diag = [l.strip()[:300] for l in out.stderr.splitlines()
                    if "Recompiling function" in l or "skipping cudagraphs" in l.lower() or "graph break" in l.lower()]
            line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
            if not line:
                report["timings"][mode] = {"error": out.stderr[-2000:]}
                print("  failed:\n" + out.stderr[-2000:], flush=True)
                continue
            r = json.loads(line[-1][7:])
            if diag:
                r["_compile_diagnostics"] = {"count": len(diag), "first": diag[:20]}
            report["timings"][mode] = r
            for case, v in r.items():
                if isinstance(v, dict) and "first" in v:
                    print(f"  compile diagnostics: {v['count']} lines (see {a.out})", flush=True)
                    continue
                print(f"  {case:6s} " + ", ".join(f"{k} {x:.1f}" if isinstance(x, float) else f"{k} {x}"
                                                for k, x in v.items() if not isinstance(x, dict)), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
