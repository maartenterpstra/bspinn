#!/bin/bash
# 3-D method comparison (2 CPU cores). Results: runs/eval3d/<case>/results.json
cd "$(dirname "$0")"
export BART=${BART:-bart}
E="python scripts/evaluate_methods_3d.py --out runs/eval3d"
M="oracle lowres walsh inati espirit nlinv bs_wires bs_pinn"
( $E --cases synth3d --methods $M; $E --cases synth3d_mismatch --methods $M bs_pinn_complex ) > runs/eval3d_A.log 2>&1 &
( $E --cases bart3d --methods $M bs_pinn_complex ) > runs/eval3d_B.log 2>&1 &
wait
