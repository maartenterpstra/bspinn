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
        import torch._dynamo
        cfgd = torch._dynamo.config
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
