#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-22201}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiments/exp222_individual_continual_learning"
BIN="$EXP/bin/ngu_continual"
ID="pretrain_s${SEED}"
ARTIFACT_ROOT="${NGU_ARTIFACT_ROOT:-$EXP}"
PRETRAIN_DATA="${NGU_PRETRAIN_DATA:-$EXP/data/pretrain.bin}"
LR="${NGU_LR:-0.025}"

mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/models" \
  "$ARTIFACT_ROOT/results/raw"
/usr/bin/time -v -o "$ARTIFACT_ROOT/logs/${ID}.time" \
  env NGU_ACTIVE_SUCCESSORS=4 NGU_ADMISSION_QUOTA=512 \
  NGU_NEGATIVE_FEEDBACK=1 NGU_LAYOUT_MODE=1 NGU_TRAIN_WORKERS=8 \
  NGU_SPEC_WORKERS=8 NGU_SPEC_WINDOW=1024 NGU_SPEC_ADAPTIVE=0 \
  NGU_SPEC_COMPUTE=1 NGU_POOL_MODE=0 NGU_PHASE_MODE=3 \
  "$BIN" train --condition exp152 \
  --train "$PRETRAIN_DATA" --train-limit 4999998 \
  --eval /dev/null --model-out "$ARTIFACT_ROOT/models/${ID}.model" \
  --output "$ARTIFACT_ROOT/results/raw/${ID}.json" \
  --vocab 4096 --max-order 4 --gcp-max 50000 --gm1-max 200000 \
  --glp-max 500000 --gm1-mode 0 --medial-mode 1 --shard-mode 14 \
  --max-successors 512 --capacity-policy fixed --successor-index on \
  --light-interval 25000 --deep-interval 100000 --wake-interval 50000 \
  --lr "$LR" --sleep-policy uniform --train-order-seed "$SEED" \
  --preserve-order --generation-samples 0 \
  >"$ARTIFACT_ROOT/logs/${ID}.stdout" \
  2>"$ARTIFACT_ROOT/logs/${ID}.stderr"
