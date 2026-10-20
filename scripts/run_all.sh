#!/usr/bin/env bash
# All evaluations for the benchmark dashboard, one after another, then the bundle (dashboard_data.json).
# Every step is resume-safe: re-running the script skips what is done. A failing step is logged and the
# next one runs. Run from the folder that contains bspinn/ (e.g. /localscratch/mterpst5/VibeCoiling):
#
#   BASE=/home/ds.umcutrecht.nl/mterpst5/netwerkschijven BRAIN=/localscratch/mterpst5 \
#       nohup bash bspinn/scripts/run_all.sh > runs/run_all.log 2>&1 &
#
# Settings (environment variables, defaults in brackets):
#   BASE    dataloader base folder (required for the static cases)
#   BRAIN   folder with the phase-pole brain planes ksp_2D, ksp_2D2, ksp_2D3 (.cfl/.hdr) [skip if unset]
#   BART    BART binary [bart/bart]          DEVICE   torch device [cuda:0]
#   PY      python launcher [uv run python]  OCMR     OCMR data folder [data/ocmr]
#   STEPS   which steps, space separated [static prior ocmr ocmr54 ocmr_us poles bundle]
#   N       random dataloader slices [20]    LOADER   [dataloader_v2:get_dataloader]
set -u
BART=${BART:-bart/bart}; DEVICE=${DEVICE:-cuda:0}; PY=${PY:-uv run python}; OCMR=${OCMR:-data/ocmr}
N=${N:-20}; LOADER=${LOADER:-dataloader_v2:get_dataloader}; BASE=${BASE:-}; BRAIN=${BRAIN:-}
STEPS=${STEPS:-static prior ocmr ocmr54 ocmr_us poles bundle}
S=bspinn/scripts
export BART
mkdir -p runs "$OCMR"
FAILED=()
step() {                                   # step NAME command...
    local name=$1; shift
    [[ " $STEPS " == *" $name "* ]] || { echo "== skip $name"; return; }
    echo "== $(date '+%F %T') start $name"
    if "$@"; then echo "== $(date '+%F %T') done $name"; else echo "== $(date '+%F %T') FAILED $name (exit $?)"; FAILED+=("$name"); fi
}

V53="solve_image=true optimizer=adam+lbfgs joint_extra_penalty=false resid_zero_mean=false joint_lbfgs=0 init_from_acs=false"
V54="solve_image=true optimizer=adam+lbfgs resid_zero_mean=false joint_lbfgs=0 init_from_acs=false"
V56="solve_image=true optimizer=adam+lbfgs"

static() {
    [ -n "$BASE" ] || { echo "BASE (dataloader base folder) not set"; return 1; }
    $PY $S/evaluate_methods.py --cases synth --loader "$LOADER" --base-folder "$BASE" \
        --n "$N" --joint 4 --device "$DEVICE" --Rs 2 4 8 --pinn-quality default \
        --pinn-variant v53 $V53 \
        --pinn-variant v54 $V54 \
        --pinn-variant v56 $V56 \
        --pinn-variant v56_vp0 $V56 joint_lbfgs_cg=0 \
        --pinn-variant v56_nolbfgs $V56 joint_lbfgs=0 \
        --pinn-variant v56_nozm $V56 resid_zero_mean=false \
        --pinn-variant v56_grad $V56 joint_image=gradient \
        --out runs/eval_v56
}

prior() {
    [ -n "$BASE" ] || { echo "BASE not set"; return 1; }
    $PY $S/prior_eval.py --loader "$LOADER" --base-folder "$BASE" --n "$N" --R 4 \
        --maps nlinv espirit pinn_v53 pinn_v54 pinn_v56 --maps-cache runs/eval_v56 \
        --methods tikhonov tv tvw --oracle --device "$DEVICE" --out runs/prior_v56.jsonl
}

ocmr_download() {
    local f
    for f in fs_0004_1_5T fs_0005_1_5T fs_0006_1_5T fs_0009_1_5T fs_0010_1_5T \
             fs_0016_3T fs_0019_3T fs_0020_3T fs_0021_3T us_0001_3T us_0002_3T us_0003_3T; do
        [ -s "$OCMR/$f.h5" ] && continue
        echo "downloading $f"
        if curl -f -sS -o "$OCMR/$f.h5.part" "https://ocmr.s3.amazonaws.com/data/$f.h5"; then
            mv "$OCMR/$f.h5.part" "$OCMR/$f.h5"
        else
            echo "download failed: $f"; rm -f "$OCMR/$f.h5.part"; return 1
        fi
    done
}

ocmr() {
    ocmr_download || return 1
    $PY $S/ocmr_eval.py --files "$OCMR"/fs_*.h5 --slices mid --R 1 4 8 \
        --maps espirit nlinv pinn_avg pinn_dyn0 pinn_dyn --device "$DEVICE" --oracle --save-images \
        --out runs/ocmr_v56/results.jsonl
}

ocmr54() {
    ocmr_download || return 1
    $PY $S/ocmr_eval.py --files "$OCMR"/fs_*.h5 --slices mid --R 1 4 8 \
        --maps pinn_avg pinn_dyn --device "$DEVICE" --save-images \
        --pinn-set resid_zero_mean=false --dyn-set freeze_static=false alpha_init=data \
        --out runs/ocmr_v54/results.jsonl
}

ocmr_us() {
    ocmr_download || return 1
    $PY $S/ocmr_prospective.py --files "$OCMR"/us_*.h5 \
        --maps espirit nlinv pinn_avg pinn_dyn --device "$DEVICE" --save-images --out runs/ocmr_us/results.jsonl
}

poles() {
    [ -n "$BRAIN" ] || { echo "BRAIN not set: phase-pole step skipped"; return 0; }
    $PY $S/phase_pole_eval.py --files "$BRAIN"/ksp_2D "$BRAIN"/ksp_2D2 "$BRAIN"/ksp_2D3 \
        --acs 3 7 15 --device "$DEVICE" --out runs/poles/results.jsonl \
        --maps espirit nlinv nlinv_pp pinn_v54 pinn_v55 pinn_v55@joint_lbfgs_cg=0 \
               pinn_v55@init_from_acs=False pinn_v55@resid_zero_mean=False pinn_v55@joint_lbfgs=0 \
               pinn_v54@seed=1 pinn_v55@seed=1
}

bundle() {
    $PY $S/dashboard_bundle.py --eval runs/eval_v56 --prior runs/prior_v56.jsonl \
        --ocmr runs/ocmr_v56/results.jsonl runs/ocmr_v54/results.jsonl --ocmr-us runs/ocmr_us/results.jsonl \
        --poles runs/poles/results.jsonl --note "v56 run_all, $(date +%F)" --out dashboard_data.json
}

step static static
step prior prior
step ocmr ocmr
step ocmr54 ocmr54
step ocmr_us ocmr_us
step poles poles
step bundle bundle
echo "== $(date '+%F %T') finished; failed steps: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
