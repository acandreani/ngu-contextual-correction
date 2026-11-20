#!/usr/bin/env python3
"""EXP224 corpus: difficult contexts start confidently associated to a wrong target."""

import hashlib
import json
import random
import struct
from pathlib import Path

EXP = Path(__file__).resolve().parent
OUT = EXP / "data"
VOCAB = 4096
DOC_LEN = 6


def stable_facts():
    return [
        (100, 200 + i, 20 + i % 8, 40 + (i // 8) % 8, 1000 + i)
        for i in range(256)
    ]


def temporal_facts(current):
    base = 1500 if current else 1400
    return [
        (101, 500 + i, 30 + i % 8, 50 + (i // 8) % 8, base + i)
        for i in range(64)
    ]


def novel_facts():
    return [
        (102, 600 + i, 60 + i % 8, 70 + (i // 8) % 8, 1700 + i)
        for i in range(128)
    ]


def hard_facts(stale=False):
    rows = []
    for i in range(64):
        entity = 800 + i
        relation = 80 + i % 8
        qualifier = 90 + (i // 8) % 8
        targets = (2000 + i, 1900 + i) if stale else (1900 + i, 2000 + i)
        rows.append((103, entity, relation, qualifier, targets[0]))
        rows.append((104, entity, relation, qualifier, targets[1]))
    return rows


def weighted_docs(total_tokens, seed, groups):
    rng = random.Random(seed)
    names = [row[0] for row in groups]
    weights = [row[1] for row in groups]
    facts = {row[0]: row[2] for row in groups}
    for _ in range(total_tokens // DOC_LEN):
        name = rng.choices(names, weights=weights, k=1)[0]
        yield (*facts[name][rng.randrange(len(facts[name]))], 0)


def balanced_docs(total_tokens, seed, facts):
    rng = random.Random(seed)
    order = list(range(len(facts)))
    for index in range(total_tokens // DOC_LEN):
        if index % len(order) == 0:
            rng.shuffle(order)
        yield (*facts[order[index % len(order)]], 0)


def write_docs(path, docs):
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as output:
        block = []
        for doc in docs:
            block.extend(doc)
            count += len(doc)
            if len(block) >= 16384:
                payload = struct.pack("<{}I".format(len(block)), *block)
                output.write(payload)
                digest.update(payload)
                block.clear()
        if block:
            payload = struct.pack("<{}I".format(len(block)), *block)
            output.write(payload)
            digest.update(payload)
    return {
        "path": str(path.relative_to(OUT)),
        "tokens": count,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main():
    (OUT / "stream").mkdir(parents=True, exist_ok=True)
    (OUT / "holdouts").mkdir(exist_ok=True)
    stable = stable_facts()
    old = temporal_facts(False)
    current = temporal_facts(True)
    novel = novel_facts()
    hard = hard_facts(False)
    stale = hard_facts(True)
    manifest = {
        "experiment": "EXP224",
        "format": "uint32-little-endian",
        "vocab_size": VOCAB,
        "document_tokens": DOC_LEN,
        "hard_pretraining": "marker-conditioned targets swapped",
        "splits": {},
    }
    manifest["splits"]["pretrain"] = write_docs(
        OUT / "pretrain.bin",
        weighted_docs(
            4_999_998,
            22401,
            (
                ("stable", 0.90, stable),
                ("temporal_old", 0.05, old),
                ("hard_stale", 0.05, stale),
            ),
        ),
    )
    stream = []
    for block in range(10):
        row = write_docs(
            OUT / "stream" / "block_{:02d}.bin".format(block),
            weighted_docs(
                99_996,
                22500 + block,
                (
                    ("novel", 0.40, novel),
                    ("temporal_current", 0.40, current),
                    ("hard_corrected", 0.20, hard),
                ),
            ),
        )
        row["block"] = block
        stream.append(row)
    manifest["splits"]["stream"] = stream
    for name, facts, seed in (
        ("base", stable, 22601),
        ("temporal_old", old, 22602),
        ("temporal_current", current, 22603),
        ("novel", novel, 22604),
        ("hard", hard, 22605),
        ("hard_stale", stale, 22606),
    ):
        manifest["splits"][name] = write_docs(
            OUT / "holdouts" / "{}.bin".format(name),
            balanced_docs(59_994, seed, facts),
        )
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
