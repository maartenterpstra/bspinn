#!/bin/bash
cd "$(dirname "$0")"
S="python scripts/model_size_sweep.py --out runs/size"
( $S --case synth8 --group pure; $S --case knee15 --group pure --only pinn_init_none_iters_pinn2500 ) > runs/pure_A.log 2>&1 &
( $S --case synth8_mismatch --group pure --only pinn_init_none_iters_pinn2500 ) > runs/pure_B.log 2>&1 &
wait
