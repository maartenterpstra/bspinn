#!/bin/bash
# Reproduce the real-data runs. Data: https://github.com/mrirecon/bart-workshop
#   ismrm2016/pics-brain/ksp           -> data/brain8_ksp
#   mri_together_2025/data/{ksp,noise} -> radial (see below)
# Radial trajectory + whitening + delay correction with BART (as in the workshop notebook):
#   bart seq --FOV 0.256 --BR 256 -r 377 -f 1 --dwell 10e-6 --FA 50 --raga --spoiled --TR 250e-3 --TE 3e-3 samples grad mom pulseq.seq
#   bart vec 1 1 0 a; bart extract 0 0 3 samples - | bart fmac - a seq_traj
#   bart whiten -n ksp noise radial16_ksp_white; bart estdelay -R seq_traj radial16_ksp_white gd
#   bart trajcor -V gd -O seq_traj radial16_traj
# Reference maps: bart ecalib brain8_ksp brain8_espirit; bart ncalib brain8_ksp brain8_nlinv
#                 bart ncalib -t radial16_traj radial16_ksp_white radial16_nlinv
cd "$(dirname "$0")"
C="--iters-wires 1000 --log-every 250 --lambda-phys 0.1 --threads 1"
python -m bspinn.cli fit data/radial16_ksp_white --traj data/radial16_traj --drop-first 1 --fov 256 --out runs/radial16 \
    --calib 32 --iters-pinn 1200 --recon-size 96 $C --ref-maps nlinv_bart=data/radial16_nlinv > runs/radial16.log 2>&1 &
python -m bspinn.cli fit data/brain8_ksp --out runs/brain8_zoff --calib 48 --iters-pinn 1500 $C \
    --ref-maps espirit_bart=data/brain8_espirit nlinv_bart=data/brain8_nlinv > runs/brain8_zoff.log 2>&1 &
wait
