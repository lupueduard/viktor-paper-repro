"""Greedy keep-if-better self-improvement loop with oracle (held-out) instrumentation.

At each generation the proposer (the solver model itself) sees the incumbent instruction,
the task description and four failures from a separate feedback pool, and writes K rewrites.
Each candidate is scored on the reused n-item selection set, paired with the incumbent; the
best-measured candidate is committed iff its measured gain is > 0 (greedy acceptance).
The seed and the final incumbent are also scored on the held-out gold set (never shown to the loop).
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from . import prompts
from .data import Item


class Chat(Protocol):
    def chat(self, system: str, user: str, **kw) -> str: ...
    def chat_many(self, system: str, users: list[str], **kw) -> list[str]: ...


class OutputCache:
    """Solver outputs keyed by (instruction, item id). Greedy decoding makes them reusable
    across loops, e.g. the seed instruction is scored once per item for all runs."""

    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS out (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    @staticmethod
    def key(instr: str, item_id: str) -> str:
        return hashlib.sha1(instr.encode()).hexdigest() + "|" + item_id

    def get_many(self, instr: str, ids: list[str]) -> dict[str, str]:
        keys = {self.key(instr, i): i for i in ids}
        res: dict[str, str] = {}
        with self.lock:
            klist = list(keys)
            for s in range(0, len(klist), 500):
                chunk = klist[s:s + 500]
                q = "SELECT k, v FROM out WHERE k IN (%s)" % ",".join("?" * len(chunk))
                for k, v in self.db.execute(q, chunk):
                    res[keys[k]] = v
        return res

    def put_many(self, instr: str, pairs: dict[str, str]) -> None:
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO out VALUES (?, ?)",
                                [(self.key(instr, i), v) for i, v in pairs.items()])
            self.db.commit()


@dataclass
class Scorer:
    llm: Chat
    labels: list[str]
    cache: OutputCache | None = None
    max_tokens: int = 48
    evals: int = 0  # solver calls actually made (cache misses)

    def outputs(self, instr: str, items: list[Item]) -> dict[str, str]:
        have = self.cache.get_many(instr, [it.id for it in items]) if self.cache else {}
        todo = [it for it in items if it.id not in have]
        if todo:
            outs = self.llm.chat_many(instr, [prompts.solver_user_message(self.labels, it.text) for it in todo],
                                      temperature=0.0, max_tokens=self.max_tokens)
            new = {it.id: o for it, o in zip(todo, outs)}
            self.evals += len(todo)
            if self.cache:
                self.cache.put_many(instr, new)
            have.update(new)
        return have

    def vector(self, instr: str, items: list[Item]) -> tuple[list[int], dict[str, str]]:
        outs = self.outputs(instr, items)
        vec = [int(prompts.parse_classification(outs[it.id], self.labels) == it.label) for it in items]
        return vec, outs


def mean(v: list[int]) -> float:
    return sum(v) / len(v) if v else 0.0


@dataclass
class LoopConfig:
    n: int
    seed: int
    generations: int = 10
    k: int = 4
    proposal_rounds: int = 5
    n_failures: int = 4
    proposer_temperature: float = 0.9
    proposer_top_p: float = 0.95
    proposer_max_tokens: int = 400


@dataclass
class LoopResult:
    config: dict
    generations: list[dict] = field(default_factory=list)
    incumbents: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def propose(llm: Chat, task: str, incumbent: str, failures: list[tuple[str, str, str]], cfg: LoopConfig,
            rng: random.Random, existing: set[str]) -> tuple[list[str], int]:
    """Sample K valid, distinct rewrites; invalid ones and duplicates are resampled."""
    user = prompts.proposer_user_message(task, incumbent, failures)
    got: list[str] = []
    attempts = 0
    for _ in range(cfg.proposal_rounds):
        need = cfg.k - len(got)
        if need <= 0:
            break
        seeds = [rng.randrange(2**31) for _ in range(need)]
        with ThreadPoolExecutor(max_workers=need) as ex:
            outs = list(ex.map(lambda s: llm.chat(prompts.PROPOSER_SYSTEM, user,
                                                  temperature=cfg.proposer_temperature,
                                                  top_p=cfg.proposer_top_p,
                                                  max_tokens=cfg.proposer_max_tokens, seed=s), seeds))
        attempts += need
        for o in outs:
            p = prompts.parse_proposal(o)
            if p and p not in existing and p not in got:
                got.append(p)
    return got[: cfg.k], attempts


def run_loop(llm: Chat, scorer: Scorer, task: str, seed_instruction: str, selection: list[Item],
             feedback: list[Item], gold: list[Item], cfg: LoopConfig, log=print) -> LoopResult:
    rng = random.Random(cfg.seed * 1000 + cfg.n)
    res = LoopResult(config=cfg.__dict__.copy())

    def score_incumbent(instr: str, gen: int) -> tuple[list[int], list[int], dict[str, str]]:
        sel_vec, _ = scorer.vector(instr, selection)
        fb_vec, fb_out = scorer.vector(instr, feedback)
        res.incumbents.append({"generation": gen, "instruction": instr, "proxy": mean(sel_vec),
                               "feedback": mean(fb_vec)})
        return sel_vec, fb_vec, fb_out

    incumbent = seed_instruction
    inc_sel, inc_fb, inc_fb_out = score_incumbent(incumbent, 0)
    seen = {incumbent}
    log(f"[n={cfg.n} seed={cfg.seed}] gen 0 proxy={mean(inc_sel):.3f}")
    for gen in range(1, cfg.generations + 1):
        wrong = [it for it, ok in zip(feedback, inc_fb) if not ok]
        fails = [(it.text, inc_fb_out[it.id], it.label) for it in rng.sample(wrong, min(cfg.n_failures, len(wrong)))]
        cands, attempts = propose(llm, task, incumbent, fails, cfg, rng, seen)
        seen.update(cands)
        cand_records = []
        for c in cands:
            vec, _ = scorer.vector(c, selection)
            cand_records.append({"instruction": c, "proxy": mean(vec), "gain": mean(vec) - mean(inc_sel),
                                 "vector": vec})
        best = max(cand_records, key=lambda r: r["gain"]) if cand_records else None
        committed = bool(best and best["gain"] > 0)
        res.generations.append({"generation": gen, "incumbent_proxy": mean(inc_sel), "n_candidates": len(cands),
                                "proposal_attempts": attempts, "candidates": cand_records,
                                "committed": committed})
        if committed:
            incumbent = best["instruction"]
            inc_sel, inc_fb, inc_fb_out = score_incumbent(incumbent, gen)
        log(f"[n={cfg.n} seed={cfg.seed}] gen {gen} cands={len(cands)} best_gain="
            f"{(best['gain'] if best else float('nan')):+.3f} commit={committed} proxy={mean(inc_sel):.3f}")

    # Oracle instrumentation: the seed and the final incumbent on the held-out gold set.
    seed_gold, _ = scorer.vector(seed_instruction, gold)
    final_gold, _ = scorer.vector(incumbent, gold)
    res.summary = {
        "seed_proxy": res.incumbents[0]["proxy"], "final_proxy": mean(inc_sel),
        "seed_heldout": mean(seed_gold), "final_heldout": mean(final_gold),
        "commits": sum(g["committed"] for g in res.generations),
        "final_instruction": incumbent,
        "seed_gold_vector": seed_gold, "final_gold_vector": final_gold,
    }
    log(f"[n={cfg.n} seed={cfg.seed}] FINAL proxy={mean(inc_sel):.3f} heldout={mean(final_gold):.3f} "
        f"(seed proxy={res.incumbents[0]['proxy']:.3f} heldout={mean(seed_gold):.3f})")
    return res


def save_result(res: LoopResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"config": res.config, "generations": res.generations,
                                      "incumbents": res.incumbents,
                                      "summary": res.summary}))
