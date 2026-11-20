#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
core=experiments/exp222_individual_continual_learning
base=experiments/exp224_contextual_negative_feedback
trial=experiments/exp225_continual_learning_confirmation
for name in ${!NGU_@}; do unset "$name"; done
export LC_ALL=C
test -x /usr/bin/time || { echo "GNU time is required at /usr/bin/time" >&2; exit 1; }
printf '%s\n' "1/4 Building the frozen Rust source"
mkdir -p "$core/bin"
rustc --edition=2021 -O -C target-cpu=native "$core/generated/ngu_continual.rs" -o "$core/bin/ngu_continual"
printf '%s\n' "2/4 Generating synthetic data and verifying archived SHA-256 manifests"
python3 "$base/generate_corpus.py" > "$base/data_generation.log"
python3 "$trial/generate_streams.py" > "$trial/data_generation.log"
cmp reference/corpus.json "$base/data/manifest.json"
cmp reference/streams.json "$trial/data/manifest.json"
printf '%s\n' "3/4 Pretraining a new common checkpoint"
NGU_ARTIFACT_ROOT="$PWD/$base" NGU_PRETRAIN_DATA="$PWD/$base/data/pretrain.bin" "$core/run_pretrain.sh" 22401
printf '%s\n' "4/4 Scoring and learning ten blocks per arm and seed"
"$trial/run_campaign.sh"
python3 papers/contextual_correction/check_revision.py
