"""Runs every stage of the study in order; each stage resumes where it stopped.

    stages: prepare | pilot | search | grid | selected | defects | all
Needs llama-server on LLAMA_URL (default http://127.0.0.1:8080) serving Qwen3-1.7B, and the two
parquet files in DATA_DIR (default ./data).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hr import data as D  # noqa: E402
from hr.prompts import GRID, OFF_1500, THINK_64, THINK_512, Harness  # noqa: E402
from hr.runner import run_budget, run_harness  # noqa: E402
from hr.search import search  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS = os.path.join(ROOT, "runs")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
POOLS = ["gsm8k", "mmlupro"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fallback_gold(raw_answer: str) -> str | None:
    """Defect B1 for GSM8K: a 'historical fallback' that reads the gold as the result of the last
    <<calc>> annotation in the rationale instead of the '####' line."""
    import re
    calcs = re.findall(r"<<[^>]*=\s*(-?[\d.,]+)\s*>>", raw_answer.split("####")[0])
    return D.norm_number(calcs[-1]) if calcs else None


def prepare():
    os.makedirs(RUNS, exist_ok=True)
    out = []
    for pool, loader, fn in [("gsm8k", D.load_gsm8k, "gsm8k_test.parquet"),
                             ("mmlupro", D.load_mmlupro, "mmlupro_test.parquet")]:
        parts = D.partitions(loader(os.path.join(DATA_DIR, fn)))
        for name, items in parts.items():
            for it in items:
                rec = {**asdict(it), "partition": name}
                rec["options"] = list(it.options)
                rec["fallback_gold"] = fallback_gold(it.raw_answer) if pool == "gsm8k" else None
                out.append(rec)
    with open(os.path.join(RUNS, "partitions.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log("prepared", len(out), "rows")


def load_parts() -> dict[tuple[str, str], list[D.Item]]:
    parts: dict = {}
    with open(os.path.join(RUNS, "partitions.jsonl")) as f:
        for r in map(json.loads, f):
            it = D.Item(r["pool"], r["id"], r["question"], r["gold"], tuple(r["options"]),
                        r["raw_answer"], r["category"])
            parts.setdefault((r["pool"], r["partition"]), []).append(it)
    return parts


def path(pool, partition, name):
    return os.path.join(RUNS, pool, partition, f"{name}.jsonl")


def pilot(parts):
    for pool in POOLS:
        items = parts[(pool, "search")][:20]
        for h in (THINK_512, OFF_1500):
            run_harness(items, h, "pilot", os.path.join(RUNS, "pilot", pool, f"{h.name}.jsonl"))
        log("pilot done", pool)


def run_search(parts):
    for pool in POOLS:
        for tag, start in (("CLEAN", OFF_1500), ("BROKEN", THINK_64)):
            t = search(parts[(pool, "search")], start, f"{tag}-{pool}", os.path.join(RUNS, "search"))
            log("search", tag, pool, "selected", t["selected"]["system"][:40], "|",
                t["selected"]["instruction"][:60], "score", t["selected_search_score"])


def grid(parts):
    for partition in ("replication", "test"):
        for pool in POOLS:
            items = parts[(pool, partition)]
            for h in GRID:
                run_harness(items, h, partition, path(pool, partition, h.name))
                log("grid", partition, pool, h.name)
            run_budget(items, THINK_512, path(pool, partition, THINK_512.name), partition,
                       path(pool, partition, "BUDGET-512-1500"))
            log("grid", partition, pool, "BUDGET-512-1500")


def selected(parts):
    for pool in POOLS:
        for tag in ("CLEAN", "BROKEN"):
            with open(os.path.join(RUNS, "search", f"{tag}-{pool}.trace.json")) as f:
                h = Harness(**json.load(f)["selected"])
            for partition in ("replication", "test"):
                run_harness(parts[(pool, partition)], h, partition, path(pool, partition, h.name))
                log("selected", h.name, partition)


def defects(parts):
    for partition in ("replication", "test"):
        for pool in POOLS:
            run_harness(parts[(pool, partition)], OFF_1500, partition,
                        path(pool, partition, "OFF-1500.defectA"), leak=True)
            log("defect A", partition, pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "pilot", "search", "grid", "selected", "defects", "all"])
    a = ap.parse_args()
    if a.stage == "prepare":
        return prepare()
    parts = load_parts()
    stages = {"pilot": [pilot], "search": [run_search], "grid": [grid], "selected": [selected],
              "defects": [defects], "all": [run_search, grid, selected, defects]}[a.stage]
    for fn in stages:
        fn(parts)
    log("stage done", a.stage)


if __name__ == "__main__":
    main()
