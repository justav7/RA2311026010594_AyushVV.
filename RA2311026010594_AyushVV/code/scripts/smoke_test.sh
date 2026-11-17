#!/usr/bin/env bash
# Seconds-long check that every code path executes. Results are meaningless
# (models are trained for a handful of steps) and are written to outputs_smoke/.
set -eo pipefail
export PYTHONPATH=src
sed 's#outputs: outputs#outputs: outputs_smoke#' configs/default.yaml > configs/smoke.yaml
python scripts/check_leakage.py > /dev/null && echo "leakage audit ok"
for M in supervised siamese protonet osapnet; do
  python -m osapnet.train --config configs/smoke.yaml --model $M --seed 0 --episodes 3 | tail -1
done
for M in transfer siamese protonet osapnet; do
  python -m osapnet.evaluate --config configs/smoke.yaml --model $M --seed 0 --protocol all --episodes 4 > /dev/null && echo "evaluate $M ok"
done
python -m osapnet.evaluate --config configs/smoke.yaml --model cnn_scratch --seed 0 --protocol same --episodes 2 > /dev/null && echo "evaluate cnn_scratch ok"
python -m osapnet.infer --config configs/smoke.yaml --model osapnet --seed 0 --n-queries 2 > /dev/null && echo "infer ok"
python -m pytest -q tests
echo "SMOKE TEST PASSED (numbers produced here are NOT results)"
