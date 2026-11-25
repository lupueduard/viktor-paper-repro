"""TREC-50 data, pool construction and splits.

Follows Appendix B.1 of arXiv:2610.09239 where it can on a small budget:
the pool is the public train split followed by the test split (published order),
item i of split s gets id ``trec-{s}-{i}``. The pool is shuffled with Random(1234);
the first ``gold_size`` items are the held-out (gold) set, never used for selection
or feedback. For a loop with seed r, Random(r) first draws a 256-item feedback pool
from the remainder and then a fixed ordering of what is left, whose first n items are
the selection set, so selection sets are nested across n for a given seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

TREC_FILES = {"train": "train_5500.label", "test": "TREC_10.label"}
TREC_URLS = {
    "train": "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
    "test": "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
}

TASK_DESCRIPTION = (
    "Classify a question by the type of answer it seeks into one of 50 fine-grained TREC "
    "labels of the form COARSE:fine (the label list is given in the input). The first line "
    "of the response is matched against the label set; otherwise the last label mentioned is used."
)
SEED_INSTRUCTION = "Classify the question."


@dataclass(frozen=True)
class Item:
    id: str
    text: str
    label: str


def parse_trec_line(line: str) -> tuple[str, str]:
    label, text = line.strip().split(" ", 1)
    return label, text.strip()


def load_trec(data_dir: str | Path) -> list[Item]:
    data_dir = Path(data_dir)
    pool: list[Item] = []
    for split in ("train", "test"):
        raw = (data_dir / TREC_FILES[split]).read_bytes().decode("latin-1")
        lines = [l for l in raw.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            label, text = parse_trec_line(line)
            pool.append(Item(id=f"trec-{split}-{i}", text=text, label=label))
    return pool


def label_set(pool: list[Item]) -> list[str]:
    return sorted({it.label for it in pool})


@dataclass
class Splits:
    gold: list[Item]
    feedback: list[Item]
    selection_order: list[Item]

    def selection(self, n: int) -> list[Item]:
        return self.selection_order[:n]


def make_gold(pool: list[Item], gold_size: int) -> tuple[list[Item], list[Item]]:
    shuffled = list(pool)
    random.Random(1234).shuffle(shuffled)
    return shuffled[:gold_size], shuffled[gold_size:]


def make_splits(pool: list[Item], seed: int, gold_size: int = 600, feedback_size: int = 256) -> Splits:
    gold, rest = make_gold(pool, gold_size)
    rng = random.Random(seed)
    feedback = rng.sample(rest, feedback_size)
    fb_ids = {it.id for it in feedback}
    remaining = [it for it in rest if it.id not in fb_ids]
    rng.shuffle(remaining)
    return Splits(gold=gold, feedback=feedback, selection_order=remaining)
