#!/bin/bash
# Model-size sweep (2 chains). Results: runs/size/*.json
cd "$(dirname "$0")"
S="python scripts/model_size_sweep.py --out runs/size"
( $S --case synth8_mismatch --group wires; $S --case knee15 --group wires; $S --case synth8_mismatch --group pinn ) > runs/size_A.log 2>&1 &
( $S --case synth8 --group wires; $S --case synth8 --group pinn; $S --case knee15 --group pinn --only pinn_width16_depth4 pinn_width128_depth4 ) > runs/size_B.log 2>&1 &
wait
