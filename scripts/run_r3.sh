#!/usr/bin/env bash
# Round-3 follow-up (v58), after run_r2.sh: adds only what is new, resume-safe. Run from the same folder:
#
#   BASE=/smb/user/mterpst5/DS-Data/Radiotherapie/Research/Project/fastMRI_NYU \
#       bash bspinn/scripts/run_r3.sh 2>&1 | tee runs/run_r3.log
#
#   image    the phase constraint switched per slice by CV (tv_phase_auto, tgv_phase_auto); appended to
#            runs/r2_image.jsonl (the methods already there are skipped)
#   irgnm    IRGNM with the image pulled towards zero (x_ref=zero, 12 steps; best on the brain planes); appended
#            to runs/r2_irgnm.jsonl
#   ocmr     cine maps after the readout-axis fix: pinn_avg and pinn_dyn0 must now agree; pinn_avg with a
#            second seed measures the run-to-run spread on the 3 T cines
#   ocmr_us  the same on the prospective cines
set -u
BART=${BART:-bart/bart}; DEVICE=${DEVICE:-cuda:0}; PY=${PY:-uv run python}; OCMR=${OCMR:-data/ocmr}
N=${N:-20}; LOADER=${LOADER:-dataloader_v2:get_dataloader}; BASE=${BASE:-}
STEPS=${STEPS:-image irgnm ocmr ocmr_us}
S=bspinn/scripts
export BART
FAILED=()
step() {
    local name=$1; shift
    [[ " $STEPS " == *" $name "* ]] || { echo "== skip $name"; return; }
    echo "== $(date '+%F %T') start $name"
    if "$@"; then echo "== $(date '+%F %T') done $name"; else echo "== $(date '+%F %T') FAILED $name (exit $?)"; FAILED+=("$name"); fi
}
V54="solve_image=true optimizer=adam+lbfgs resid_zero_mean=false joint_lbfgs=0 init_from_acs=false"

image() {
    [ -n "$BASE" ] || { echo "BASE not set"; return 1; }
    $PY $S/prior_eval.py --loader "$LOADER" --base-folder "$BASE" --n "$N" --R 4 6 --pinn-variant v54 $V54 \
        --maps pinn_v54 nlinv --maps-cache runs/eval_v56 --methods tv tgv tv_phase tgv_phase tv_phase_auto tgv_phase_auto \
        --oracle --device "$DEVICE" --out runs/r2_image.jsonl
}

irgnm() {
    [ -n "$BASE" ] || { echo "BASE not set"; return 1; }
    $PY $S/irgnm_eval.py --loader "$LOADER" --base-folder "$BASE" --n "$N" --R 4 --pinn-variant v54 $V54 \
        --maps pinn_v54 pinn_v54+irgnm@holdout=0@steps=8 pinn_v54+irgnm@holdout=0@steps=12@x_ref=zero \
               nlinv nlinv+irgnm@holdout=0@steps=8 nlinv+irgnm@holdout=0@steps=12@x_ref=zero \
        --maps-cache runs/eval_v56 --methods tikhonov tv --oracle --device "$DEVICE" --out runs/r2_irgnm.jsonl
}

ocmr() {
    $PY $S/ocmr_eval.py --files "$OCMR"/fs_*.h5 --slices mid --R 4 8 \
        --maps pinn_avg pinn_dyn0 --recon tv --spatial-on difference --device "$DEVICE" \
        --out runs/r3_ocmr/results.jsonl &&
    $PY $S/ocmr_eval.py --files "$OCMR"/fs_*.h5 --slices mid --R 4 8 \
        --maps pinn_avg --recon tv --spatial-on difference --device "$DEVICE" --pinn-set seed=1 \
        --out runs/r3_ocmr_seed1/results.jsonl
}

ocmr_us() {
    $PY $S/ocmr_prospective.py --files "$OCMR"/us_*.h5 \
        --maps pinn_avg pinn_dyn0 --recon tv --spatial-on difference --device "$DEVICE" \
        --out runs/r3_ocmr_us/results.jsonl
}

step image image
step irgnm irgnm
step ocmr ocmr
step ocmr_us ocmr_us
echo "== $(date '+%F %T') finished; failed steps: ${FAILED[*]:-none}"
echo "send me: runs/r2_image.jsonl runs/r2_irgnm.jsonl runs/r3_ocmr/results.jsonl runs/r3_ocmr_seed1/results.jsonl runs/r3_ocmr_us/results.jsonl runs/run_r3.log"
[ ${#FAILED[@]} -eq 0 ]
