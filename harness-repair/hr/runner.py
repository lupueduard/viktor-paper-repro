"""Runs a harness on a partition and appends one JSON line per row (resumable)."""
from __future__ import annotations

import hashlib
import json
import os
import threading

from . import llm
from .data import Item
from .prompts import BUDGET_ANSWER, Harness, reference_text, render

FORCED_END = "\n</think>\n\n"


def _done_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(l)["id"] for l in f if l.strip()}


def read_runs(path: str) -> dict[str, dict]:
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, f))}


def _writer(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = threading.Lock()

    def write(rec: dict):
        with lock, open(path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return write


def run_harness(items: list[Item], h: Harness, partition: str, path: str, leak: bool = False) -> None:
    done = _done_ids(path)
    todo = [it for it in items if it.id not in done]
    write = _writer(path)
    prompts = [render(it, h, reference_text(it) if leak else None) for it in todo]

    def on_done(i, r):
        it = todo[i]
        write({"id": it.id, "pool": it.pool, "partition": partition, "harness": h.name,
               "thinking": h.thinking, "max_tokens": h.max_tokens, "leak": leak,
               "render_sha256": hashlib.sha256(prompts[i].encode()).hexdigest()[:16], **r})

    llm.complete_many([(p, h.max_tokens) for p in prompts], on_done)


def run_budget(items: list[Item], base_h: Harness, base_path: str, partition: str, path: str) -> None:
    """BUDGET-512-1500: thinking is ended after the base call's 512 tokens (a '</think>' is forced
    when the model had not closed it) and up to 1500 answer tokens follow. Greedy decoding makes
    the first 512 tokens identical to THINK-512's, so the call continues THINK-512's own output."""
    base = read_runs(base_path)
    done = _done_ids(path)
    write = _writer(path)
    jobs, meta = [], []
    for it in items:
        if it.id in done:
            continue
        b = base[it.id]
        prefix = render(it, base_h)
        if "</think>" not in b["text"]:
            cont = b["text"] + FORCED_END
            forced = True
        elif b["stop"] == "limit":
            cont = b["text"]          # thinking ended inside the cap, answer was cut: let it finish
            forced = False
        else:
            write({**b, "harness": "BUDGET-512-1500", "forced_end": False, "base_tokens": b["n_tokens"]})
            continue
        jobs.append((prefix + cont, BUDGET_ANSWER))
        meta.append((it, b, cont, forced))

    def on_done(i, r):
        it, b, cont, forced = meta[i]
        write({"id": it.id, "pool": it.pool, "partition": partition, "harness": "BUDGET-512-1500",
               "thinking": True, "max_tokens": b["max_tokens"] + BUDGET_ANSWER, "leak": False,
               "forced_end": forced, "base_tokens": b["n_tokens"], "text": cont + r["text"],
               "stop": r["stop"], "n_tokens": b["n_tokens"] + r["n_tokens"],
               "prompt_tokens": b.get("prompt_tokens", 0)})

    llm.complete_many(jobs, on_done)
