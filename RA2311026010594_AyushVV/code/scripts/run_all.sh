#!/usr/bin/env bash
# Full experimental pipeline (NOT executed in the FJ3 submission build - see README "Status").
# Approximate CPU time on a 2-core laptop: 6-10 h; on a Colab T4 GPU: ~1-2 h.
set -euo pipefail
export PYTHONPATH=src
python scripts/check_leakage.py
for SEED in 0 1 2; do
  for M in supervised siamese protonet osapnet osap_no_se osap_no_adaptive osap_no_openloss; do
    python -m osapnet.train --model $M --seed $SEED
  done
  for M in transfer siamese protonet osapnet osap_no_se osap_no_adaptive osap_no_openloss; do
    python -m osapnet.evaluate --model $M --seed $SEED --protocol all
  done
  python -m osapnet.evaluate --model cnn_scratch --seed $SEED --protocol same
  python -m osapnet.evaluate --model cnn_scratch --seed $SEED --protocol cross_load
  python -m osapnet.evaluate --model cnn_scratch --seed $SEED --protocol cross_domain
  python -m osapnet.evaluate --model cnn_scratch --seed $SEED --protocol noise
done
python -m osapnet.report
python -m osapnet.infer --model osapnet --seed 0 --known IR014 B014 OR014@6 --query-class IR028
