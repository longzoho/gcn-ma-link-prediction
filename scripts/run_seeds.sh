#!/usr/bin/env bash
# Run a GCN_MA experiment across 3 seeds.
#
# Usage: scripts/run_seeds.sh <dataset>
# Example: scripts/run_seeds.sh collegemsg
#
# Reads the base experiment config at configs/experiments/gcn_ma_<dataset>.yaml,
# overrides the seed for each run, and writes to results/metrics.jsonl.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset>" >&2; exit 1
fi
DATASET="$1"
SEEDS=(42 123 2024)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_CFG="configs/experiments/gcn_ma_${DATASET}.yaml"
if [ ! -f "$BASE_CFG" ]; then
    echo "Missing experiment config: $BASE_CFG" >&2; exit 1
fi

mkdir -p results/logs results/configs_runtime
for SEED in "${SEEDS[@]}"; do
    RUN_CFG="results/configs_runtime/gcn_ma_${DATASET}_seed${SEED}.yaml"
    # Replace `seed: NN` and `experiment_name: ...` with seed-specific values
    sed -e "s/^seed:.*/seed: ${SEED}/" \
        -e "s/^experiment_name:.*/experiment_name: gcn_ma_${DATASET}_seed${SEED}/" \
        "$BASE_CFG" > "$RUN_CFG"
    LOG="results/logs/gcn_ma_${DATASET}_seed${SEED}_$(date +%Y%m%d-%H%M%S).log"
    echo "=== Running ${DATASET} seed=${SEED} → $LOG ==="
    .venv/bin/python scripts/train.py --config "$RUN_CFG" 2>&1 | tee "$LOG"
done
echo "Done: ${DATASET} × 3 seeds. metrics.jsonl appended."
