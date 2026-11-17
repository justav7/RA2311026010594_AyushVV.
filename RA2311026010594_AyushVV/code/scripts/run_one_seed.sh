#!/usr/bin/env bash
# Train + evaluate every model for ONE seed (used to run seeds 1 and 2 in parallel,
# single-threaded, during the reported experiments). Usage: bash scripts/run_one_seed.sh 1
set -e
export PYTHONPATH=src OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
SEED=${1:?seed}
for M in supervised siamese protonet osapnet osap_no_se osap_no_adaptive osap_no_openloss; do python -m osapnet.train --model $M --seed $SEED; done
for M in transfer siamese protonet osapnet osap_no_se osap_no_adaptive osap_no_openloss; do python -m osapnet.evaluate --model $M --seed $SEED --protocol all; done
for P in same cross_load cross_domain noise; do python -m osapnet.evaluate --model cnn_scratch --seed $SEED --protocol $P; done
echo "SEED $SEED DONE"
