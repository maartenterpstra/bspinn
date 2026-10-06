#!/bin/bash
# Full method comparison (2 CPU cores: two chains in parallel). Results: runs/eval/<case>/results.json
cd "$(dirname "$0")"
export BART=${BART:-bart}
E="python scripts/evaluate_methods.py --out runs/eval"
ALL="oracle lowres walsh inati espirit nlinv bs_pinn bs_wires free bs_pinn_grid1"
CORE="oracle lowres walsh inati espirit nlinv bs_pinn bs_wires bs_pinn_grid1"
( $E --cases synth8 synth8_mismatch --methods $ALL; $E --cases synth8_noiselo synth8_noisehi --methods $CORE ) > runs/eval_chainA.log 2>&1 &
( $E --cases knee15 --methods $ALL; $E --cases brain8 knee15_acs16 knee15_acs32 knee15_acs48 --methods $CORE ) > runs/eval_chainB.log 2>&1 &
wait
