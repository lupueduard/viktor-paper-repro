"""Benchmark loading and the search / replication / test partitions.

GSM8K (openai/gsm8k, test split) and MMLU-Pro (TIGER-Lab/MMLU-Pro, test split, stratified by
category) are read from their parquet files. Partitions are drawn once with a fixed seed and
are disjoint: search (used only by the harness-search arm), replication, test.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field

SEED = 4242
SIZES = {"search": 100, "replication": 150, "test": 300}
LETTERS = "ABCDEFGHIJ"


@dataclass(frozen=True)
class Item:
    pool: str          # "gsm8k" or "mmlupro"
    id: str
    question: str
    gold: str          # normalised number (gsm8k) or letter (mmlupro)
    options: tuple = field(default_factory=tuple)
    raw_answer: str = ""   # gsm8k rationale, used by the gold-fallback defect
    category: str = ""


def norm_number(s: str) -> str | None:
    """'1,234.50' -> '1234.5', '$7' -> '7', '-3.0' -> '-3'. None if not a number."""
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        return None
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def load_gsm8k(path: str) -> list[Item]:
    import pandas as pd
    df = pd.read_parquet(path)
    items = []
    for i, r in df.iterrows():
        gold = norm_number(r["answer"].split("####")[-1])
        assert gold is not None, r["answer"]
        items.append(Item("gsm8k", f"gsm8k-{i}", r["question"], gold, (), r["answer"]))
    return items


def load_mmlupro(path: str) -> list[Item]:
    import pandas as pd
    df = pd.read_parquet(path)
    items = []
    for _, r in df.iterrows():
        opts = tuple(str(o) for o in r["options"])
        items.append(Item("mmlupro", f"mmlupro-{r['question_id']}", r["question"],
                          LETTERS[int(r["answer_index"])], opts, "", r["category"]))
    return items


def stratified_sample(items: list[Item], n: int, rng: random.Random) -> list[Item]:
    """Proportional allocation by category (largest remainder), then shuffled."""
    by = defaultdict(list)
    for it in items:
        by[it.category].append(it)
    cats = sorted(by)
    total = len(items)
    quotas = {c: n * len(by[c]) / total for c in cats}
    alloc = {c: int(quotas[c]) for c in cats}
    for c in sorted(cats, key=lambda c: -(quotas[c] - alloc[c]))[: n - sum(alloc.values())]:
        alloc[c] += 1
    out = []
    for c in cats:
        pool = sorted(by[c], key=lambda it: it.id)
        out += rng.sample(pool, alloc[c])
    rng.shuffle(out)
    return out


def partitions(items: list[Item], seed: int = SEED, sizes: dict = SIZES) -> dict[str, list[Item]]:
    rng = random.Random(seed)
    n = sum(sizes.values())
    if items and items[0].pool == "mmlupro":
        drawn = stratified_sample(items, n, rng)
    else:
        drawn = rng.sample(sorted(items, key=lambda it: it.id), n)
    out, k = {}, 0
    for name, size in sizes.items():
        out[name] = drawn[k:k + size]
        k += size
    return out
