# Correcting obsolete associations in a non-neural graph memory

Standalone reproduction of the EXP225 results in the paper of the same name.
This repository contains the frozen Rust source used for the experiment, the
synthetic data generators, the historical training and evaluation commands,
and a small checker for the published table. No existing checkpoints, external
datasets, API keys, GPUs, Python packages or Cargo downloads are needed.

```sh
git clone https://github.com/acandreani/ngu-contextual-correction.git
cd ngu-contextual-correction
./reproduce.sh
```

Requirements: Linux, Bash, GNU time at `/usr/bin/time`, standard shell utilities
(including `cmp`), CPython 3 and a recent stable `rustc`. The runner uses
CPU threads. Run in a fresh clone or a disposable checkout: rerunning replaces
generated data, binaries, checkpoints, logs and results there. No sudo needed.

## What is reproduced

1. Compile the archived, single-file Rust runtime; generate all records.
   Match both generated SHA-256 manifests to the historical manifests.
2. Pretrain on 4,999,998 synthetic tokens. Copy the starting state logically
   by loading the same checkpoint for every arm and stream.
3. For each of three seeds, score each of ten blocks before updating, then
   score corrected, obsolete, unchanged and other terminal probes. POS updates
   target credit; FULL additionally updates non-target features and records
   conflicts. Both receive identical labels.
4. Write `table.tsv` from the evaluator JSON and fail on a mismatch with
   `reference/table.tsv` at the precision printed in the paper. Print paired
   effects and run the audit of distinct-cue counts and of the exact-lookup
   deduction. The 90% lookup figure is derived from coverage, not measured
   by a separate timed training run. Timing and memory are not parity gates.

The generated files stay under `experiments/`: EXP222 holds the compiled
runtime; EXP224 the shared pretraining data and checkpoint; EXP225 the three
confirmation streams and per-seed/arm outputs in `runs/`. Directory numbers
preserve the archival layout. Internal runtime experiment labels are legacy
labels; the controller, seed, arm and input manifest identify this study.
Only the table and data fingerprints in `reference/` are reference inputs
to checks, not substitutes for training or inference.
