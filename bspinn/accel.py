"""Performance switches shared by all fitting stages.

``configure(cfg)`` is called at the start of every fit:
  * cfg.compile -> torch.compile the two kernels that dominate run time: the Biot-Savart segment field
    (long chains of elementwise ops on (coils x vertices x points x 3) tensors) and the Taylor-mode MLP
    pass (value, gradient, Laplacian). On a GPU this produces fused Triton kernels; on CPU fused C++.
    Each new tensor shape is compiled once (dynamic shapes are enabled to limit recompilation).
  * cfg.tf32 -> TF32 matmuls on Ampere+ GPUs (and cuDNN). Off by default: the maps are accurate to
    ~1e-5 in the subspace metric, and TF32 has ~1e-3 relative precision.
"""
import torch

_state = {"compiled": False}


def configure(cfg):
    if getattr(cfg, "tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if getattr(cfg, "compile", False) and not _state["compiled"]:
        enable_compile()


def enable_compile():
    from . import models, physics
    if _state["compiled"]:
        return
    _state["eager"] = (physics.biot_savart_polyline, models.trunk_taylor, physics.segment_static_and_log)
    bs = torch.compile(physics.biot_savart_polyline, dynamic=True)
    physics.biot_savart_polyline = bs
    models.biot_savart_polyline = bs
    models.trunk_taylor = torch.compile(models.trunk_taylor, dynamic=True)
    physics.segment_static_and_log = torch.compile(physics.segment_static_and_log, dynamic=True)
    _state["compiled"] = True


def disable_compile():
    from . import models, physics
    if not _state["compiled"]:
        return
    bs, tt, sl = _state["eager"]
    physics.biot_savart_polyline = bs
    models.biot_savart_polyline = bs
    models.trunk_taylor = tt
    physics.segment_static_and_log = sl
    _state["compiled"] = False
