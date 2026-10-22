#!/usr/bin/env bash
# Round-2 confirmations on your data (README §8.13-8.15), one after another, resume-safe like run_all.sh.
# Run from the folder that contains bspinn/ and runs/ (the same place as run_all.sh; it reuses the cached maps of
# runs/eval_v56):
#
#   BASE=/home/ds.umcutrecht.nl/mterpst5/netwerkschijven BRAIN=/localscratch/mterpst5 \
#       nohup bash bspinn/scripts/run_r2.sh > runs/run_r2.log 2>&1 &
#
# Steps (STEPS, default all):
#   image   static prior: tv vs tgv vs tv_phase vs tgv_phase, R 4 and 6, PINN v54 and NLINV maps, oracle grid
#   irgnm   IRGNM map refinement on the PINN v54 and NLINV maps (R 4): lco, ACS residual, SENSE, Tikhonov, TV
#   ocmr    cine: spatio-temporal TV on the difference from the time average; dynamic layout on the 2-D base
#   ocmr_us prospective cine: the same, scored by held-out k-space (pinn_dyn has the held-out correction check)
#   poles   brain phase-pole planes: IRGNM on the PINN maps at ACS 3 / 7 / 15
set -u
BART=${BART:-bart/bart}; DEVICE=${DEVICE:-cuda:0}; PY=${PY:-uv run python}; OCMR=${OCMR:-data/ocmr}
N=${N:-20}; LOADER=${LOADER:-dataloader_v2:get_dataloader}; BASE=${BASE:-}; BRAIN=${BRAIN:-}
STEPS=${STEPS:-image irgnm ocmr ocmr_us poles}
S=bspinn/scripts
export BART
mkdir -p runs
FAILED=()
step() {
    local name=$1; shift
    [[ " $STEPS " == *" $name "* ]] || { echo "== skip $name"; return; }
    echo "== $(date '+%F %T') start $name"
    if "$@"; then echo "== $(date '+%F %T') done $name"; else echo "== $(date '+%F %T') FAILED $name (exit $?)"; FAILED+=("$name"); fi
}
# same definition as in run_all.sh, so the cached maps of runs/eval_v56 are reused
V54="solve_image=true optimizer=adam+lbfgs resid_zero_mean=false joint_lbfgs=0 init_from_acs=false"

image() {
    [ -n "$BASE" ] || { echo "BASE not set"; return 1; }
    $PY $S/prior_eval.py --loader "$LOADER" --base-folder "$BASE" --n "$N" --R 4 6 --pinn-variant v54 $V54 \
        --maps pinn_v54 nlinv --maps-cache runs/eval_v56 --methods tv tgv tv_phase tgv_phase \
        --oracle --device "$DEVICE" --out runs/r2_image.jsonl
}

irgnm() {
    [ -n "$BASE" ] || { echo "BASE not set"; return 1; }
    $PY $S/irgnm_eval.py --loader "$LOADER" --base-folder "$BASE" --n "$N" --R 4 --pinn-variant v54 $V54 \
        --maps pinn_v54 pinn_v54+irgnm@holdout=0@steps=8 pinn_v54+irgnm nlinv nlinv+irgnm@holdout=0@steps=8 \
        --maps-cache runs/eval_v56 --methods tikhonov tv --oracle --device "$DEVICE" --out runs/r2_irgnm.jsonl
}

ocmr() {
    # baseline (current prior, same maps for pinn_avg / nlinv): runs/ocmr_v56/results.jsonl
    $PY $S/ocmr_eval.py --files "$OCMR"/fs_*.h5 --slices mid --R 4 8 \
        --maps nlinv pinn_avg pinn_dyn0 pinn_dyn --recon tv --spatial-on difference --oracle --device "$DEVICE" \
        --out runs/r2_ocmr/results.jsonl
}

ocmr_us() {
    # baseline: runs/ocmr_us/results.jsonl
    $PY $S/ocmr_prospective.py --files "$OCMR"/us_*.h5 \
        --maps nlinv pinn_avg pinn_dyn0 pinn_dyn --spatial-on difference --device "$DEVICE" \
        --out runs/r2_ocmr_us/results.jsonl
}

poles() {
    [ -n "$BRAIN" ] || { echo "BRAIN not set: skipped"; return 0; }
    $PY $S/phase_pole_eval.py --files "$BRAIN"/ksp_2D "$BRAIN"/ksp_2D2 "$BRAIN"/ksp_2D3 \
        --acs 3 7 15 --device "$DEVICE" --out runs/r2_poles/results.jsonl \
        --maps nlinv_pp pinn_v54 pinn_v54+irgnm pinn_v54+irgnm@holdout=0@steps=8 \
               pinn_v54+irgnm@holdout=0@steps=12@x_ref=zero
}

step image image
step irgnm irgnm
step ocmr ocmr
step ocmr_us ocmr_us
step poles poles
echo "== $(date '+%F %T') finished; failed steps: ${FAILED[*]:-none}"
echo "send me: runs/r2_image.jsonl runs/r2_irgnm.jsonl runs/r2_ocmr/results.jsonl runs/r2_ocmr_us/results.jsonl runs/r2_poles/results.jsonl"
[ ${#FAILED[@]} -eq 0 ]
