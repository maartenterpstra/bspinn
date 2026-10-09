"""Performance switches shared by all fitting stages.

The calibration problems are small (a few thousand grid points, ~1000 collocation points), so on a GPU
an iteration is bound by kernel-launch and Python overhead (~1800 small ops per PINN iteration), not by
arithmetic. The remedies here therefore act on whole iterations:

  * ``compile_step(fn, cfg, iters)``: torch.compile of the *entire* differentiable loss of one iteration
    (forward and, through AOTAutograd, backward) with static shapes: one graph, compiled once per stage,
    a few dozen fused kernels instead of ~1800 launches. Randomness (collocation points, slice batches)
    is drawn outside and passed in as tensors, so the graph never changes and never recompiles.
      cfg.compile = False | True ("default") | "reduce-overhead" (also CUDA graphs: the whole forward
      and backward replayed with one launch) | "max-autotune" | "max-autotune-no-cudagraphs".
    Stages shorter than ``cfg.compile_min_iters`` run eagerly (compiling costs ~10-60 s per stage).
  * ``adam(params, device)``: fused Adam on CUDA (one kernel for all parameters).
  * cfg.tf32 -> TF32 matmuls on Ampere+ GPUs. Off by default: the maps are accurate to ~1e-5 in the
    subspace metric, and TF32 has ~1e-3 relative precision.

(An earlier version compiled the individual kernels with dynamic shapes; on a GPU that was much slower
than eager because of recompilation and per-call overhead, and it was removed.)
"""
import torch


def configure(cfg):
    if getattr(cfg, "tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if getattr(cfg, "compile", False):
        from torch import _dynamo                              # (not `import torch._dynamo`: rebinds `torch`)
        cfgd = _dynamo.config
        for name in ("recompile_limit", "cache_size_limit"):        # one graph per stage, several stages
            if hasattr(cfgd, name):
                setattr(cfgd, name, max(getattr(cfgd, name), 64))


def compile_mode(cfg):
    c = getattr(cfg, "compile", False)
    if not c:
        return None
    return "default" if c is True or str(c).lower() in ("1", "true", "yes") else str(c)


def compile_step(fn, cfg, iters=None):
    """``fn`` compiled for the iterations of one stage (see module doc), or ``fn`` itself."""
    mode = compile_mode(cfg)
    if mode is None or (iters is not None and iters < getattr(cfg, "compile_min_iters", 200)):
        return fn
    return torch.compile(fn, dynamic=False, mode=None if mode == "default" else mode)


def adam(params, device=None, **kw):
    """torch.optim.Adam, fused into one kernel on CUDA when every parameter is a real floating tensor."""
    groups = list(params)
    if groups and isinstance(groups[0], dict):                    # materialise generators (model.parameters())
        groups = [{**g, "params": list(g["params"])} for g in groups]
        flat = [p for g in groups for p in g["params"]]
    else:
        flat = groups
    dev = torch.device(device) if device is not None else (flat[0].device if flat else torch.device("cpu"))
    if dev.type == "cuda" and flat and all(p.is_floating_point() and p.device == dev for p in flat):
        try:
            return torch.optim.Adam(groups, fused=True, **kw)
        except (RuntimeError, TypeError, ValueError):
            pass
    return torch.optim.Adam(groups, **kw)


# ------------------------------------------------------------------------------------ CUDA graphs
def graph_mode(cfg, device, iters, warmup=3):
    """cfg.cuda_graph on a CUDA device, and enough iterations to warm up, capture and replay."""
    on_dev = torch.device(device).type == "cuda" or getattr(cfg, "_graph_emulate", False)   # (test hook: CPU)
    return bool(getattr(cfg, "cuda_graph", False)) and on_dev and iters >= warmup + 2 and not compile_mode(cfg)


def capturable_adam(groups, device, default_lr=1e-3):
    """Adam whose step can be captured in a CUDA graph: device tensors for the step counts and the
    learning rates (changed between replays with fill_, no host sync). Returns (optimizer, base lrs)."""
    groups = list(groups)
    if not groups or not isinstance(groups[0], dict):
        groups = [{"params": groups}]
    base, out = [], []
    for g in groups:
        lr = float(g.get("lr", default_lr))
        base.append(lr)
        out.append({**g, "params": list(g["params"]), "lr": torch.tensor(lr, device=device)})
    if torch.device(device).type != "cuda":           # CPU emulation (tests): same algorithm, no capture
        return torch.optim.Adam(out, foreach=False), base
    return torch.optim.Adam(out, capturable=True, foreach=True), base


def cosine_lr(base, eta_min, it, T):
    """CosineAnnealingLR's learning rate before step ``it`` (closed form; T = number of iterations)."""
    import math
    return eta_min + (base - eta_min) * (1 + math.cos(math.pi * it / T)) / 2


def train_graphed(step, opt, base_lrs, eta_min, iters, gens=(), before=None, after=None, warmup=3, log=print):
    """Run ``iters`` iterations of  out = step(); out[0].backward(); opt.step()  with the iterations after
    ``warmup`` replayed from one CUDA graph (forward, backward and optimizer: one launch per iteration).

    step():  computes the loss tuple; draws its random numbers with the generators ``gens`` (registered
             with the graph, so every replay draws new numbers) and reads per-iteration inputs from static
             tensors that ``before(it)`` fills (device-to-device copies) before each iteration.
    after(it, out): called after every iteration with the (static) outputs, e.g. for logging.
    If capture fails (an operation that is not capturable), the remaining iterations run eagerly.
    Returns the outputs of the last iteration."""
    def set_lr(it):
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"].fill_(cosine_lr(b, eta_min, it, iters))

    def eager(it):
        set_lr(it)
        if before:
            before(it)
        opt.zero_grad(set_to_none=True)
        out = step()
        out[0].backward()
        opt.step()
        if after:
            after(it, out)
        return out

    if not opt.param_groups[0]["lr"].is_cuda:           # CPU emulation (tests): everything eager
        out = None
        for it in range(iters):
            out = eager(it)
        return out
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                       # warm-up (real iterations) on a side stream
        for it in range(warmup):
            out = eager(it)
    torch.cuda.current_stream().wait_stream(s)
    try:
        g = torch.cuda.CUDAGraph()
        for gen in gens:
            g.register_generator_state(gen)
        set_lr(warmup)
        if before:
            before(warmup)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            static = step()
            static[0].backward()
            opt.step()
    except Exception as e:                           # not capturable: carry on eagerly
        log(f"CUDA graph capture failed ({type(e).__name__}: {str(e)[:200]}); continuing eagerly")
        torch.cuda.synchronize()
        for it in range(warmup, iters):
            out = eager(it)
        return out
    for it in range(warmup, iters):
        if it > warmup:
            set_lr(it)
            if before:
                before(it)
        g.replay()
        if after:
            after(it, static)
    return static
