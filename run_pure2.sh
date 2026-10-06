#!/bin/bash
cd "$(dirname "$0")"
S="python scripts/model_size_sweep.py --out runs/size"
$S --case synth8_mismatch --group pure --only pinn_initnone_iters_pinn2500 > runs/pure_B.log 2>&1
while pgrep -f "case synth8 --group pure" > /dev/null; do sleep 20; done
$S --case knee15 --group pure --only pinn_initnone_iters_pinn2500 > runs/pure_C.log 2>&1
