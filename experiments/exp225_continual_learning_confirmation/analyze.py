#!/usr/bin/env python3
"""Rebuild the paper table from newly generated evaluator outputs."""
import json
from pathlib import Path
from statistics import mean, stdev

TRIAL = Path(__file__).resolve().parent
ROOT = TRIAL.parents[1]
lines = ["seed\tPOS_top1_pct\tFULL_top1_pct\tPOS_NLL\tFULL_NLL\tPOS_online_pct\tFULL_online_pct"]
effects = []
for seed in (22511, 22522, 22533):
    scores = []
    for arm in ("pos", "full"):
        folder = TRIAL / "runs" / ("seed_%d" % seed) / arm / "results/eval"
        def read(name):
            return json.loads((folder / (name + ".json")).read_text())
        hard = read("hard")
        preq = [read("preq_%02d" % i) for i in range(10)]
        n = sum(p["eval_predictions"] for p in preq)
        assert hard["eval_predictions"] == 9999 and n == 166660
        online = sum(p["eval_predictions"] * p["top1_accuracy"] for p in preq) / n
        scores.append((100 * hard["top1_accuracy"], hard["nll_per_token"], 100 * online))
    (p, pn, po), (f, fn, fo) = scores
    lines.append("%d\t%.2f\t%.2f\t%.3f\t%.3f\t%.2f\t%.2f" % (seed,p,f,pn,fn,po,fo))
    effects.append((f-p, 100*(1-fn/pn), fo-po))
table = "\n".join(lines) + "\n"
(ROOT / "table.tsv").write_text(table)
print(table, end="")
for label, values in zip(("hard top-1 (pp)", "relative hard NLL reduction (%)", "online top-1 (pp)"), zip(*effects)):
    print("%s: %.2f +/- %.2f" % (label, mean(values), stdev(values)))
if table.splitlines() != (ROOT / "reference/table.tsv").read_text().splitlines():
    raise SystemExit("FAIL: table differs from paper at published precision; inspect table.tsv")
print("PASS: table matches all published rows at displayed precision")
