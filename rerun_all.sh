#!/bin/bash
# Full benchmark rerun with the current code (2 queues for 2 CPU cores). Baseline maps are cached.
cd "$(dirname "$0")"
export BART=${BART:-bart}
E2="python scripts/evaluate_methods.py --out runs/eval"
E3="python scripts/evaluate_methods_3d.py --out runs/eval3d"
S="python scripts/model_size_sweep.py --out runs/size"
ALL="oracle lowres walsh inati espirit nlinv bs_pinn bs_wires free bs_pinn_grid1"
CORE="oracle lowres walsh inati espirit nlinv bs_pinn bs_wires bs_pinn_grid1"
M3="oracle lowres walsh inati espirit nlinv bs_wires bs_pinn"
MW="oracle lowres walsh inati espirit nlinv bs_pinn bs_wires bs_pinn_complex bs_helmholtz"
(
  echo "== 2d A $(date)"; $E2 --cases synth8 synth8_mismatch --methods $ALL; $E2 --cases synth8_noiselo synth8_noisehi --methods $CORE
  echo "== 3d A $(date)"; $E3 --cases synth3d --methods $M3; $E3 --cases synth3d_mismatch --methods $M3 bs_pinn_complex
  echo "== wave A $(date)"; $E3 --cases wave15 --methods $MW
  echo "== size A $(date)"; $S --case synth8 --group wires; $S --case synth8 --group pinn; $S --case synth8 --group pure
  $S --case synth8_mismatch --group lambda; $S --case synth8_mismatch --group fieldmode
  $S --case synth8_mismatch --group pure --only pinn_initnone_iters_pinn2500
  echo "== done A $(date)"
) > runs/rerun_A.log 2>&1 &
(
  echo "== 2d B $(date)"; $E2 --cases knee15 --methods $ALL; $E2 --cases brain8 knee15_acs16 knee15_acs32 knee15_acs48 --methods $CORE
  echo "== 3d B $(date)"; $E3 --cases bart3d --methods $M3 bs_pinn_complex
  echo "== wave B $(date)"; $E3 --cases wave3 --methods $MW bs_helmholtz_fixed
  echo "== size B $(date)"; $S --case synth8_mismatch --group wires; $S --case synth8_mismatch --group pinn
  $S --case knee15 --group wires; $S --case knee15 --group pinn --only pinn_width16_depth4 pinn_width128_depth4
  $S --case knee15 --group pure --only pinn_initnone_iters_pinn2500
  echo "== done B $(date)"
) > runs/rerun_B.log 2>&1 &
wait
# free ablation refit (air collocation must not apply to it) on the cases that ran before the fix
for c in synth8 synth8_mismatch knee15; do rm -f runs/eval/$c/maps/free.*; done
python scripts/evaluate_methods.py --out runs/eval --cases synth8 synth8_mismatch knee15 --methods $ALL > runs/rerun_free.log 2>&1
