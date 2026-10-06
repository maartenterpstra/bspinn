#!/bin/bash
# Wave-effect comparison (3-D full-wave synthetic, 1.5 T and 3 T tissue). Results: runs/eval3d/wave*/
cd "$(dirname "$0")"
export BART=${BART:-bart}
E="python scripts/evaluate_methods_3d.py --out runs/eval3d"
M="oracle lowres walsh inati espirit nlinv bs_pinn bs_wires bs_pinn_complex bs_helmholtz"
( $E --cases wave3 --methods $M bs_helmholtz_fixed ) > runs/eval_wave3.log 2>&1 &
( $E --cases wave15 --methods $M ) > runs/eval_wave15.log 2>&1 &
wait
