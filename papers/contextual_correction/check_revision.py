"""Check the manuscript's distinct-cue counts and exact-lookup deduction.

Read archived EXP224/225 artifacts; do not run a learner or change artifacts.
"""
import json
from collections import Counter
from pathlib import Path
from struct import iter_unpack

ROOT = Path(__file__).resolve().parents[2] / "experiments"
DATA = ROOT / "exp224_contextual_negative_feedback" / "data"
CONF = ROOT / "exp225_continual_learning_confirmation"


def records(path):
    for *cue, label, delimiter in iter_unpack("<6I", path.read_bytes()):
        assert delimiter == 0
        yield tuple(cue), label


def inventory(path):
    mapping = {}
    for cue, label in records(path):
        assert mapping.setdefault(cue, label) == label
    return mapping


if __name__ == "__main__":
    pretrain = inventory(DATA / "pretrain.bin")
    for seed in (22511, 22522, 22533):
        blocks = sorted((CONF / "data" / f"seed_{seed}" / "stream").glob("block_*.bin"))
        first = inventory(blocks[0])
        assert len(blocks) == 10 and len(first) == 320
        assert all(pretrain.get(cue) != label for cue, label in first.items())
        for block in blocks:
            assert block.stat().st_size == 16666 * 24
            assert inventory(block).items() <= first.items()
        assert inventory(DATA / "holdouts/hard.bin").items() <= first.items()
        print(f"{seed}: exact lookup (derived) 0/100% first/later blocks; 90% pooled")
        for family in ("hard", "hard_stale"):
            counts = Counter(cue for cue, _ in records(DATA / f"holdouts/{family}.bin"))
            assert Counter(counts.values()) == {78: 113, 79: 15}
            for arm in ("full", "pos"):
                row = json.loads((CONF / "runs" / f"seed_{seed}" / arm / "results" /
                                  "eval" / f"{family}.json").read_text())
                n = row["eval_predictions"]
                hits = round(n * row["top1_accuracy"])
                assert n == 9999 and row["top5_accuracy"] == 1
                k, r = divmod(hits, 78)
                assert 0 <= k <= 128 and 0 <= r <= 15
                print(f"  {arm:4s} {family:10s} hits={hits:4d} cues={k:3d}/128")
