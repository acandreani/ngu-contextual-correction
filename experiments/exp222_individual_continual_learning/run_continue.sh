#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 full|pos|q0 BLOCK INPUT_MODEL OUTPUT_MODEL" >&2
  exit 2
fi

CONDITION="$1"
BLOCK="$2"
INPUT="$3"
OUTPUT="$4"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiments/exp222_individual_continual_learning"
BIN="$EXP/bin/ngu_continual"
ARTIFACT_ROOT="${NGU_ARTIFACT_ROOT:-$EXP}"
STREAM_DIR="${NGU_STREAM_DIR:-$EXP/data/stream}"
LR="${NGU_LR:-0.025}"

case "$CONDITION" in
  full) EXTRA=(NGU_NEGATIVE_FEEDBACK=1 NGU_ADMISSION_QUOTA=512) ;;
  pos) EXTRA=(NGU_NEGATIVE_FEEDBACK=0 NGU_ADMISSION_QUOTA=512) ;;
  q0)
    EXTRA=(
      NGU_NEGATIVE_FEEDBACK=1
      NGU_ADMISSION_QUOTA=0
      NGU_FREEZE_IDENTITIES=1
    )
    ;;
  *) echo "unknown condition: $CONDITION" >&2; exit 2 ;;
esac

BLOCK_PADDED="$(printf '%02d' "$BLOCK")"
ID="${CONDITION}_b${BLOCK_PADDED}"
mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/models" \
  "$ARTIFACT_ROOT/results/raw"

/usr/bin/time -v -o "$ARTIFACT_ROOT/logs/${ID}.time" \
  env NGU_ACTIVE_SUCCESSORS=4 "${EXTRA[@]}" NGU_LAYOUT_MODE=1 \
  NGU_TRAIN_WORKERS=8 NGU_SPEC_WORKERS=8 NGU_SPEC_WINDOW=1024 \
  NGU_SPEC_ADAPTIVE=0 NGU_SPEC_COMPUTE=1 NGU_POOL_MODE=0 \
  NGU_PHASE_MODE=3 \
  "$BIN" continue --condition exp152 --model "$INPUT" \
  --train "$STREAM_DIR/block_${BLOCK_PADDED}.bin" \
  --train-limit 99996 --eval /dev/null --model-out "$OUTPUT" \
  --output "$ARTIFACT_ROOT/results/raw/${ID}.json" \
  --vocab 4096 --max-order 4 --gcp-max 50000 --gm1-max 200000 \
  --glp-max 500000 --gm1-mode 0 --medial-mode 1 --shard-mode 14 \
  --max-successors 512 --capacity-policy fixed --successor-index on \
  --light-interval 25000 --deep-interval 100000 --wake-interval 50000 \
  --lr "$LR" --sleep-policy uniform --train-order-seed "$((22300 + BLOCK))" \
  --preserve-order --generation-samples 0 \
  >"$ARTIFACT_ROOT/logs/${ID}.stdout" \
  2>"$ARTIFACT_ROOT/logs/${ID}.stderr"
