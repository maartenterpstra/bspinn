"""Which of the v8 conductor changes moves the brain8 held-out error? One bs_pinn fit per variant."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bspinn import evaluation as ev  # noqa: E402
from bspinn.methods import _bs  # noqa: E402
from evaluate_methods import CASES  # noqa: E402

VARIANTS = {"default": {}, "air0": {"air_colloc": 0.0}, "core0": {"wire_radius": 0.0},
            "outside1": {"lambda_outside": 1.0}}
out = sys.argv[1]
case = CASES["brain8"]()
obj = ev.object_mask(case)
for name in sys.argv[2:]:
    path = os.path.join(out, f"brain8__{name}.json")
    if os.path.exists(path):
        continue
    t = time.time()
    over = ({"calib": case.meta["bs_calib"]} if case.meta.get("bs_calib") else {}) | VARIANTS[name]
    C = _bs(case, "pinn", cfg_overrides=over)["pinn"]
    r, _ = ev.heldout_prospective(C, case, obj)
    r["seconds"] = time.time() - t; r["overrides"] = over
    json.dump(r, open(path, "w"), indent=1)
    print(name, r["heldout_nrmse"], f"{r['seconds']:.0f}s", flush=True)
