#!/bin/bash
# Commands behind the README §6 results (CPU, ~30 min each). Real data: see run_real.sh.
cd "$(dirname "$0")"
F="synthetic --scenario folded --size 96 --calib 32 --iters-wires 700 --iters-pinn 800 --lambda-phys 0.1 --threads 1"
python -m bspinn.cli $F --out runs/fold_single
python -m bspinn.cli $F --fold-sets 1 --out runs/fold_sets
python -m bspinn.cli dynamic --synthetic --out runs/dyn_synth --iters-wires 800 --iters-pinn 800 --iters-dyn 1500 \
    --slice-batch 8 --calib 32 --lambda-phys 0.1 --threads 1
python -c "from bspinn.fastmri import write_fake_fastmri as w; [w(f'data/fakefastmri/file_brain_AXT2_{i:03d}.h5', n_coils=n, n_slices=6, shape=(80,80), seed=100+i) for i,n in enumerate([8,12,16,10,8,12,16,10])]"
python -m bspinn.cli fit-dataset "data/fakefastmri/*.h5" --mode shared --holdout 2 --out runs/dataset_shared --calib 32 \
    --iters-auto 100 --iters-wires 600 --steps 1500 --iters-pinn 1000 --slice-batch 3 --lambda-phys 0.1 --threads 1
# GRASP (bart-workshop ismrm2016/grasp): bart traj -G -x512 -y2100 t; bart scale 0.6 t data/grasp_traj
G="data/grasp_ksp --traj data/grasp_traj --spokes-per-frame 21 --fov 380 --calib 32 --loops-per-coil 2 --slice-batch 10 \
   --iters-auto 200 --iters-wires 800 --iters-pinn 600 --iters-dyn 1200 --lambda-phys 0.1 --rank 2 --map-size 128 --threads 1"
python -m bspinn.cli dynamic $G --lambda-temporal 0.1 --out runs/grasp_dyn
python -m bspinn.cli dynamic $G --lambda-temporal 0 --dynamic-part wires --out runs/grasp_dyn_wiresonly
python -m bspinn.cli dynamic $G --lambda-temporal 0 --dynamic-part wires --max-disp-mm 15 --out runs/grasp_dyn_bounded
