#!/bin/bash
# Resume the benchmark rerun after an interruption (finished results are skipped/cached).
cd "$(dirname "$0")"
export BART=${BART:-bart}
E2="python scripts/evaluate_methods.py --out runs/eval"
E3="python scripts/evaluate_methods_3d.py --out runs/eval3d"
S="python scripts/model_size_sweep.py --out runs/size"
ALL="oracle lowres walsh inati espirit nlinv bs_pinn bs_wires free bs_pinn_grid1"
MW="oracle lowres walsh inati espirit nlinv bs_pinn bs_wires bs_pinn_complex bs_helmholtz"
(
  echo "== size A $(date)"; $S --case synth8 --group pinn; $S --case synth8 --group pure
  $S --case synth8_mismatch --group lambda; $S --case synth8_mismatch --group fieldmode
  $S --case synth8_mismatch --group pure --only pinn_initnone_iters_pinn2500
  $S --case synth8_mismatch --group pinn
  echo "== free refit $(date)"
  for c in synth8 synth8_mismatch knee15; do rm -f runs/eval/$c/maps/free.*; done
  $E2 --cases synth8 synth8_mismatch knee15 --methods $ALL
  echo "== done A $(date)"
) >> runs/rerun_A.log 2>&1 &
(
  echo "== wave B (resumed) $(date)"; $E3 --cases wave3 --methods $MW bs_helmholtz_fixed
  echo "== size B $(date)"; $S --case synth8_mismatch --group wires
  $S --case knee15 --group wires; $S --case knee15 --group pinn --only pinn_width16_depth4 pinn_width128_depth4
  $S --case knee15 --group pure --only pinn_initnone_iters_pinn2500
  echo "== done B $(date)"
) >> runs/rerun_B.log 2>&1 &
wait
python scripts/brain_ablation.py runs/ablation_brain8 default air0 core0 outside1 > runs/ablation_brain8.log 2>&1
