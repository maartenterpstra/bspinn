#!/usr/bin/env bash
# Round-4 (v59): oblique B0 from the ISMRMRD header for the cines, and the non-Cartesian (MRI-NUFFT) check.
# Resume-safe. Run from the folder that contains bspinn/ and runs/:
#
#   uv add mri-nufft finufft            # once (GPU NUFFT optional: uv add cufinufft)
#   bash bspinn/scripts/run_r4.sh 2>&1 | tee runs/run_r4.log
#
#   diag     fs_0016 / fs_0021 / fs_0020, R 4: the PINN with the header B0 direction (2 seeds) vs NLINV
#            (compare with runs/r3_diag.jsonl: the named-plane search)
#   ocmr     all fully sampled cines, R 4 / 8: pinn_avg with the header B0 direction (v59 defaults: IRGNM
#            refinement, cine prior on the difference from the time average); compare with runs/r2_ocmr (NLINV)
#            and runs/r3_ocmr (pinn_avg, named planes)
#   ocmr_us  the prospective cines, pinn_avg with the header B0 direction (compare with runs/r3_ocmr_us)
#   noncart  synthetic radial / spiral (Biot-Savart ground truth): maps, CG-SENSE and TV with MRI-NUFFT
set -u
BART=${BART:-bart/bart}; DEVICE=${DEVICE:-cuda:0}; PY=${PY:-uv run python}; OCMR=${OCMR:-data/ocmr}
STEPS=${STEPS:-diag ocmr ocmr_us noncart}
S=bspinn/scripts
export BART
FAILED=()
step() {
    local name=$1; shift
    [[ " $STEPS " == *" $name "* ]] || { echo "== skip $name"; return; }
    echo "== $(date '+%F %T') start $name"
    if "$@"; then echo "== $(date '+%F %T') done $name"; else echo "== $(date '+%F %T') FAILED $name (exit $?)"; FAILED+=("$name"); fi
}

diag() {
    [ -s runs/r4_diag.jsonl ] && { echo "runs/r4_diag.jsonl exists: skipped"; return 0; }
    $PY $S/ocmr_pinn_diag.py --files "$OCMR"/fs_0016_3T.h5 "$OCMR"/fs_0021_3T.h5 "$OCMR"/fs_0020_3T.h5 \
        --R 4 --seeds 0 1 --plane header --device "$DEVICE" --out runs/r4_diag.jsonl
}

ocmr() {
    $PY $S/ocmr_eval.py --files "$OCMR"/fs_*.h5 --slices mid --R 4 8 --maps pinn_avg --recon tv \
        --plane header --device "$DEVICE" --out runs/r4_ocmr/results.jsonl
}

ocmr_us() {
    $PY $S/ocmr_prospective.py --files "$OCMR"/us_*.h5 --maps pinn_avg --recon tv --plane header \
        --device "$DEVICE" --out runs/r4_ocmr_us/results.jsonl
}

noncart() {
    $PY -c "import mrinufft, finufft" || { echo "needs: uv add mri-nufft finufft"; return 1; }
    [ -s runs/r4_noncart.jsonl ] && { echo "runs/r4_noncart.jsonl exists: skipped"; return 0; }
    $PY $S/noncart_eval.py --synth --kinds radial spiral --size 128 --coils 8 --spokes 96 --quality default \
        --device "$DEVICE" --out runs/r4_noncart.jsonl
}

step diag diag
step ocmr ocmr
step ocmr_us ocmr_us
step noncart noncart
echo "== $(date '+%F %T') finished; failed steps: ${FAILED[*]:-none}"
echo "send me: runs/r4_diag.jsonl runs/r4_ocmr/results.jsonl runs/r4_ocmr_us/results.jsonl runs/r4_noncart.jsonl runs/run_r4.log"
[ ${#FAILED[@]} -eq 0 ]
