#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiments/exp225_continual_learning_confirmation"
SRC="$ROOT/experiments/exp222_individual_continual_learning"
E224="$ROOT/experiments/exp224_contextual_negative_feedback"
BASE="$E224/models/pretrain_s22401.model"

mkdir -p "$EXP/results/eval" "$EXP/logs"
if [[ ! -f "$EXP/data/manifest.json" ]]; then
  "$EXP/generate_streams.py" >"$EXP/data_generation.log"
fi

for split in base temporal_old temporal_current novel hard hard_stale; do
  NGU_ARTIFACT_ROOT="$EXP" \
    "$SRC/run_eval.sh" "initial_${split}" "$BASE" \
    "$E224/data/holdouts/${split}.bin" \
    "$EXP/results/eval/initial_${split}.json"
done

for seed in 22511 22522 22533; do
  stream="$EXP/data/seed_${seed}/stream"
  for condition in full pos; do
    run="$EXP/runs/seed_${seed}/${condition}"
    mkdir -p "$run/models" "$run/results/eval" "$run/results/raw" "$run/logs"
    current="$BASE"
    for block in $(seq 0 9); do
      padded="$(printf '%02d' "$block")"
      NGU_ARTIFACT_ROOT="$run" \
        "$SRC/run_eval.sh" "preq_${condition}_${seed}_${padded}" \
        "$current" "$stream/block_${padded}.bin" \
        "$run/results/eval/preq_${padded}.json"
      next="$run/models/${condition}_b${padded}.model"
      NGU_ARTIFACT_ROOT="$run" NGU_STREAM_DIR="$stream" NGU_LR=0.2 \
        "$SRC/run_continue.sh" "$condition" "$block" "$current" "$next"
      current="$next"
    done
    for split in base temporal_old temporal_current novel hard hard_stale; do
      NGU_ARTIFACT_ROOT="$run" \
        "$SRC/run_eval.sh" "${condition}_${seed}_${split}" "$current" \
        "$E224/data/holdouts/${split}.bin" \
        "$run/results/eval/${split}.json"
    done
  done
done

"$EXP/analyze.py"
