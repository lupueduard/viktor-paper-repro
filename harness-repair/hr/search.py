"""A small harness search: greedy keep-if-better over a fixed pool of prompt-text edits.

Cheap stand-in for the paper's GEPA arm (Appendix J). Like GEPA there, it edits prompt text only
(a system prompt and an instruction appended to the user turn); the thinking switch and token
limit belong to the start and are not searched. Unlike GEPA, the candidates come from a fixed
pool written before any output was seen, not from a reflective proposer, and the search makes
one pass in a fixed order, accepting a candidate only on strict improvement on the search rows.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, replace

from .data import Item
from .prompts import Harness
from .runner import read_runs, run_harness
from .scoring import score

# Fixed before any call; no soft switches (/think, /no_think), no item text.
CANDIDATES = [
    {"system": "You are a careful expert problem solver.", "instruction": ""},
    {"system": "", "instruction": "Be concise."},
    {"system": "", "instruction": "Do not deliberate at length. Give the final answer right away in the required format."},
    {"system": "", "instruction": "Reason step by step, then give the final answer in the required format."},
    {"system": "", "instruction": "Keep any reasoning to two or three short sentences, then give the final answer in the required format."},
    {"system": "You are a precise assistant. Always end your reply with the final answer in the required format.", "instruction": ""},
    {"system": "", "instruction": "Double-check your answer before giving it."},
    {"system": "", "instruction": "Start your reply with the final answer in the required format, then explain briefly."},
]


def accuracy(items: list[Item], path: str, thinking: bool) -> float:
    runs = read_runs(path)
    return sum(score(it.pool, it.gold, runs[it.id]["text"], thinking, runs[it.id]["stop"])["correct"]
               for it in items) / len(items)


def search(items: list[Item], start: Harness, tag: str, out_dir: str) -> dict:
    """Returns the trace; the selected harness is trace['selected']."""
    os.makedirs(out_dir, exist_ok=True)
    trace_path = os.path.join(out_dir, f"{tag}.trace.json")
    seed_path = os.path.join(out_dir, f"{tag}.c0.jsonl")
    run_harness(items, start, "search", seed_path)
    inc, inc_score = start, accuracy(items, seed_path, start.thinking)
    steps = [{"candidate": 0, "harness": asdict(start), "score": inc_score, "accepted": True}]
    for k, cand in enumerate(CANDIDATES, start=1):
        h = replace(start, name=f"{tag}-c{k}", **cand)
        path = os.path.join(out_dir, f"{tag}.c{k}.jsonl")
        run_harness(items, h, "search", path)
        s = accuracy(items, path, h.thinking)
        accepted = s > inc_score
        steps.append({"candidate": k, "harness": asdict(h), "score": s, "accepted": accepted})
        if accepted:
            inc, inc_score = h, s
    selected = replace(inc, name=f"SEARCH-{tag}")
    trace = {"tag": tag, "start": asdict(start), "n_rows": len(items), "steps": steps,
             "selected": asdict(selected), "selected_search_score": inc_score}
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=1)
    return trace
