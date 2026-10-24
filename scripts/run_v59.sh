#!/usr/bin/env bash
# The whole benchmark again with the v59 defaults, into fresh folders (*_v59), then the dashboard bundle with the
# round 2-4 results included. Resume-safe: re-running skips what is done; a failing step is logged and the next
# one runs. Run from the folder that contains bspinn/ and runs/ (the same place as run_all.sh):
#
#   BASE=/smb/user/mterpst5/DS-Data/Radiotherapie/Research/Project/fastMRI_NYU BRAIN=/localscratch/mterpst5 \
#       bash bspinn/scripts/run_v59.sh 2>&1 | tee runs/run_v59.log
#
# Steps (STEPS, default all): static prior ocmr ocmr_us poles bundle. Output for me: dashboard_data.json.
set -u
BART=${BART:-bart/bart}; DEVICE=${DEVICE:-cuda:0}; PY=${PY:-uv run python}; OCMR=${OCMR:-data/ocmr}
N=${N:-20}; LOADER=${LOADER:-dataloader_v2:get_dataloader}; BASE=${BASE:-}; BRAIN=${BRAIN:-}
STEPS=${STEPS:-static prior ocmr ocmr_us poles bundle}
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
V59="solve_image=true"          # everything else: the v59 defaults (IRGNM refinement, TGV + phase 'auto' image)

static() {
    [ -n "$BASE" ] || { echo "BASE (dataloader base folder) not set"; return 1; }
    $PY $S/evaluate_methods.py --cases synth --loader "$LOADER" --base-folder "$BASE" \
        --n "$N" --joint 4 --device "$DEVICE" --Rs 2 4 8 --pinn-quality default \
        --pinn-variant v59 $V59 --out runs/eval_v59
}

prior() {
    [ -n "$BASE" ] || { echo "BASE not set"; return 1; }
    $PY $S/prior_eval.py --loader "$LOADER" --base-folder "$BASE" --n "$N" --R 4 --pinn-variant v59 $V59 \
        --maps nlinv espirit pinn_v59 --maps-cache runs/eval_v59 \
        --methods tikhonov tv tgv_phase_auto --oracle --device "$DEVICE" --out runs/prior_v59.jsonl
}

ocmr() {
    $PY $S/ocmr_eval.py --files "$OCMR"/fs_*.h5 --slices mid --R 1 4 8 \
        --maps espirit nlinv pinn_avg pinn_dyn0 pinn_dyn --device "$DEVICE" --oracle \
        --out runs/ocmr_v59/results.jsonl
}

ocmr_us() {
    $PY $S/ocmr_prospective.py --files "$OCMR"/us_*.h5 \
        --maps espirit nlinv pinn_avg pinn_dyn --device "$DEVICE" --out runs/ocmr_us_v59/results.jsonl
}

poles() {
    [ -n "$BRAIN" ] || { echo "BRAIN not set: phase-pole step skipped"; return 0; }
    $PY $S/phase_pole_eval.py --files "$BRAIN"/ksp_2D "$BRAIN"/ksp_2D2 "$BRAIN"/ksp_2D3 \
        --acs 3 7 15 --device "$DEVICE" --out runs/poles_v59/results.jsonl \
        --maps espirit nlinv nlinv_pp pinn_v59 pinn_v59@seed=1
}

bundle() {
    local ex=() f
    for f in r2_image=runs/r2_image.jsonl r2_irgnm=runs/r2_irgnm.jsonl r2_ocmr=runs/r2_ocmr/results.jsonl \
             r3_ocmr=runs/r3_ocmr/results.jsonl r3_ocmr_seed1=runs/r3_ocmr_seed1/results.jsonl \
             r4_ocmr=runs/r4_ocmr/results.jsonl r2_ocmr_us=runs/r2_ocmr_us/results.jsonl \
             r3_ocmr_us=runs/r3_ocmr_us/results.jsonl r4_ocmr_us=runs/r4_ocmr_us/results.jsonl \
             r2_poles=runs/r2_poles/results.jsonl r3_diag=runs/r3_diag.jsonl r4_diag=runs/r4_diag.jsonl \
             r4_noncart=runs/r4_noncart.jsonl; do
        [ -s "${f#*=}" ] && ex+=("$f")
    done
    $PY $S/dashboard_bundle.py --eval runs/eval_v59 --prior runs/prior_v59.jsonl \
        --ocmr runs/ocmr_v56/results.jsonl runs/ocmr_v59/results.jsonl --ocmr-us runs/ocmr_us_v59/results.jsonl \
        --poles runs/poles_v59/results.jsonl --extra "${ex[@]}" --note "v59 run_v59, $(date +%F)" --out dashboard_data.json
}

step static static
step prior prior
step ocmr ocmr
step ocmr_us ocmr_us
step poles poles
step bundle bundle
echo "== $(date '+%F %T') finished; failed steps: ${FAILED[*]:-none}"
echo "send me: dashboard_data.json (and runs/run_v59.log)"
[ ${#FAILED[@]} -eq 0 ]
