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
2. Timings per optimisation iteration (ms/it, after warm-up, torch.cuda.synchronize around each
   stage), for eager | compile | compile+tf32, each mode in a fresh subprocess (torch.compile and the
   TF32 switches are process-global):
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
    for device in ("cpu", dev):
        t = time.time()
        C, info = estimate_maps(k, "2d", calib=24, quality="fast", device=device, verbose=False,
                                **({"iters_pinn": 150, "iters_wires": 100} if quick else {}))
        e = subspace_error(torch.from_numpy(C)[None], torch.from_numpy(gt[:1]), sup)
        res[device] = {"subspace_err": e, "seconds": time.time() - t}
        print(f"  estimate_maps 128^2 on {device:7s}: subspace error {e:.2e}, {time.time() - t:.1f} s", flush=True)
    return res


# --------------------------------------------------------------------------- timings (child process)
def bench_2d(dev, n):
    from bspinn.data import make_synthetic
    from bspinn.fit import FitConfig, Problem, _wrap, fit_pinn, fit_wires, make_pinn
    d, _, _ = make_synthetic(8, (256, 256), noise=0.01)
    cfg = FitConfig(calib=24, log_every=10 ** 6, device=dev, compile=MODE["compile"], tf32=MODE["tf32"])
    from bspinn import accel; accel.configure(cfg)
    d.prewhiten(); prob = Problem(d, cfg)
    fit_wires(prob, cfg, 1, 3, **QUIET)                                       # warm-up / compile
    (w, m, _, _), tw = timed(dev, lambda: fit_wires(prob, cfg, 1, n, **QUIET), n)
    pinn = make_pinn(prob, cfg, w, 1); wr = _wrap(pinn, prob, cfg, prev=w)
    fit_pinn(prob, cfg, wr, m, iters=3, **QUIET)
    _, tp = timed(dev, lambda: fit_pinn(prob, cfg, wr, m, iters=n, **QUIET), n)
    pts = torch.from_numpy(np.asarray(d.geom.grid(), np.float32).reshape(-1, 3))      # full-res 256^2 grid
    pinn.to(dev)
    pinn.sensitivity_batched(pts.to(dev))
    _, te = timed(dev, lambda: pinn.sensitivity_batched(pts.to(dev)), 1)
    return {"wire_ms_per_it": tw, "pinn_ms_per_it": tp, "maps_eval_ms": te}


def bench_joint(dev, n):
    import copy
    from bspinn.data import make_synthetic
    from bspinn.fit import FitConfig, Problem, _wrap, fit_joint_full, fit_wires, make_pinn
    d, _, _ = make_synthetic(8, (256, 256), noise=0.01)
    cfg = FitConfig(calib=24, log_every=10 ** 6, device=dev, joint_full=True, compile=MODE["compile"], tf32=MODE["tf32"])
    from bspinn import accel; accel.configure(cfg)
    d.prewhiten(); prob = Problem(d, cfg)
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
    from bspinn.fit import FitConfig, fit_pinn, fit_wires, make_pinn
    from bspinn.models import tissue_k2
    from bspinn.volume import CoarseCorrection, Problem3D, make_synthetic3d
    if helmholtz:
        k2 = tissue_k2(3.0, 50, 0.6, 0.12)
        d, _, _ = make_synthetic3d(8, (32, 32, 32), noise=0.01, wave_k2=k2, voxel=(7.5,) * 3, normalise="object")
        cfg = FitConfig(calib=(12, 12, 12), physics="helmholtz", log_every=10 ** 6, device=dev)
    else:
        d, _, _ = make_synthetic3d(8, (48, 48, 48), noise=0.01)
        cfg = FitConfig(calib=(12, 12, 12), log_every=10 ** 6, device=dev)
    cfg.compile, cfg.tf32 = MODE["compile"], MODE["tf32"]
    from bspinn import accel; accel.configure(cfg)
    d.prewhiten(); prob = Problem3D(d, cfg)
    fit_wires(prob, cfg, 1, 2, **QUIET)
    (w, m, _, _), tw = timed(dev, lambda: fit_wires(prob, cfg, 1, n, **QUIET), n)
    p = make_pinn(prob, cfg, w, 1); cc = CoarseCorrection(p, prob)
    fit_pinn(prob, cfg, cc, m, iters=2, **QUIET)
    _, tp = timed(dev, lambda: fit_pinn(prob, cfg, cc, m, iters=n, **QUIET), n)
    return {"wire_ms_per_it": tw, "pinn_ms_per_it": tp}


MODE = {"compile": False, "tf32": False}


def child(a):
    MODE.update(compile="compile" in a.mode, tf32="tf32" in a.mode)
    n = 5 if a.quick else 20
    res = {}
    for case in a.cases.split(","):
        try:
            if case == "2d":
                res[case] = bench_2d(a.device, n)
            elif case == "joint":
                res[case] = bench_joint(a.device, max(2, n // 4))
            elif case == "3d":
                res[case] = bench_3d(a.device, max(2, n // 4), False)
            elif case == "helm":
                res[case] = bench_3d(a.device, max(2, n // 4), True)
        except Exception as e:                                     # report and carry on
            res[case] = {"error": f"{type(e).__name__}: {e}"}
        if torch.device(a.device).type == "cuda":
            res[case]["peak_mem_GB"] = torch.cuda.max_memory_allocated(a.device) / 1e9
            torch.cuda.reset_peak_memory_stats(a.device)
    print("RESULT " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default="all", choices=["all", "correctness", "timings"])
    ap.add_argument("--cases", default="2d,joint,3d,helm")
    ap.add_argument("--modes", default="eager,compile,compile+tf32")
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
            out = subprocess.run(cmd, capture_output=True, text=True)
            line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
            if not line:
                report["timings"][mode] = {"error": out.stderr[-2000:]}
                print("  failed:\n" + out.stderr[-2000:], flush=True)
                continue
            r = json.loads(line[-1][7:])
            report["timings"][mode] = r
            for case, v in r.items():
                print(f"  {case:6s} " + ", ".join(f"{k} {x:.1f}" if isinstance(x, float) else f"{k} {x}"
                                                for k, x in v.items()), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
