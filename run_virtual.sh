#!/bin/bash
cd /home/claude/bspinn
W="synthetic --coils 12 --size 96 --noise-level 0.004 --noise-corr 1.0 --iters-wires 800 --iters-pinn 800 --lambda-phys 0.1 --log-every 200 --threads 1"
python -m bspinn.cli $W --model-channels physical --out runs/vc_whiten_physical > runs/vc_whiten_physical.log 2>&1 &
python -m bspinn.cli $W --model-channels data --out runs/vc_whiten_data > runs/vc_whiten_data.log 2>&1 &
wait
C="synthetic --coils 16 --size 96 --noise-level 0.004 --iters-wires 800 --iters-pinn 800 --lambda-phys 0.1 --log-every 200 --threads 1"
python -m bspinn.cli $C --compress 6 --model-channels physical --out runs/vc_compress6_physical > runs/vc_compress6_physical.log 2>&1 &
python -m bspinn.cli $C --compress 6 --model-channels data --field-mode complex --out runs/vc_compress6_data_complex > runs/vc_compress6_data_complex.log 2>&1 &
wait
python -m bspinn.cli $C --compress 6 --model-channels data --out runs/vc_compress6_data_real > runs/vc_compress6_data_real.log 2>&1 &
python -m bspinn.cli $C --out runs/vc_full16 > runs/vc_full16.log 2>&1 &
wait
