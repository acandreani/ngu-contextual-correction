#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 LABEL MODEL DATA OUTPUT" >&2
  exit 2
fi

LABEL="$1"
MODEL="$2"
DATA="$3"
OUTPUT="$4"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiments/exp222_individual_continual_learning"
BIN="$EXP/bin/ngu_continual"
ARTIFACT_ROOT="${NGU_ARTIFACT_ROOT:-$EXP}"

mkdir -p "$ARTIFACT_ROOT/logs" "$(dirname "$OUTPUT")"
/usr/bin/time -v -o "$ARTIFACT_ROOT/logs/eval_${LABEL}.time" \
  env NGU_ACTIVE_SUCCESSORS=4 NGU_LAYOUT_MODE=1 \
  "$BIN" eval --condition exp152 --model "$MODEL" \
  --eval "$DATA" --output "$OUTPUT" --score-tail 1 \
  --sleep-policy uniform --gm1-mode 0 --medial-mode 1 --shard-mode 14 \
  --generation-samples 0 \
  >"$ARTIFACT_ROOT/logs/eval_${LABEL}.stdout" \
  2>"$ARTIFACT_ROOT/logs/eval_${LABEL}.stderr"
