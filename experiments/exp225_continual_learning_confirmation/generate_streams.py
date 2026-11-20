#!/usr/bin/env python3
"""Generate three unseen chronological orders for EXP225 confirmation."""

import hashlib
import importlib.util
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent
SOURCE = EXP.parent / "exp224_contextual_negative_feedback" / "generate_corpus.py"
spec = importlib.util.spec_from_file_location("exp224_data", str(SOURCE))
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)
data.OUT = EXP / "data"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    novel = data.novel_facts()
    current = data.temporal_facts(True)
    hard = data.hard_facts(False)
    manifest = {"experiment": "EXP225", "seeds": {}}
    for seed in (22511, 22522, 22533):
        root = EXP / "data" / "seed_{}".format(seed) / "stream"
        root.mkdir(parents=True, exist_ok=True)
        rows = []
        for block in range(10):
            path = root / "block_{:02d}.bin".format(block)
            row = data.write_docs(
                path,
                data.weighted_docs(
                    99_996,
                    seed * 100 + block,
                    (
                        ("novel", 0.40, novel),
                        ("temporal_current", 0.40, current),
                        ("hard_corrected", 0.20, hard),
                    ),
                ),
            )
            # write_docs reports relative to the EXP224 output root.
            row["path"] = str(path.relative_to(EXP))
            row["sha256"] = sha256(path)
            row["block"] = block
            rows.append(row)
        manifest["seeds"][str(seed)] = rows
    (EXP / "data" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
