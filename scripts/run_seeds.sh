#!/usr/bin/env bash
# Run a model × dataset experiment across 3 seeds.
#
# Usage: scripts/run_seeds.sh <dataset> [<model>]
# Default model: gcn_ma (backward compatible).
# Example: scripts/run_seeds.sh collegemsg evolvegcn_o
#
# Reads configs/experiments/<model>_<dataset>.yaml, overrides seed for
# each run, appends to results/metrics.jsonl.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset> [<model>]" >&2; exit 1
fi
DATASET="$1"
MODEL="${2:-gcn_ma}"
SEEDS=(42 123 2024)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_CFG="configs/experiments/${MODEL}_${DATASET}.yaml"
if [ ! -f "$BASE_CFG" ]; then
    echo "Missing experiment config: $BASE_CFG" >&2; exit 1
fi

mkdir -p results/logs results/configs_runtime
for SEED in "${SEEDS[@]}"; do
    RUN_CFG="results/configs_runtime/${MODEL}_${DATASET}_seed${SEED}.yaml"
    sed -e "s/^seed:.*/seed: ${SEED}/" \
        -e "s/^experiment_name:.*/experiment_name: ${MODEL}_${DATASET}_seed${SEED}/" \
        "$BASE_CFG" > "$RUN_CFG"
    LOG="results/logs/${MODEL}_${DATASET}_seed${SEED}_$(date +%Y%m%d-%H%M%S).log"
    echo "=== Running ${MODEL}/${DATASET} seed=${SEED} → $LOG ==="
    .venv/bin/python scripts/train.py --config "$RUN_CFG" 2>&1 | tee "$LOG"
done
echo "Done: ${MODEL}/${DATASET} × 3 seeds. metrics.jsonl appended."
