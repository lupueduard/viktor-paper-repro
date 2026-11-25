import random
from pathlib import Path

from wcurse import data, prompts
from wcurse.loop import LoopConfig, OutputCache, Scorer, run_loop

LABELS = ["ABBR:abb", "ENTY:animal", "HUM:ind", "NUM:dist", "NUM:date"]


def test_parse_classification_first_line_and_fallback():
    assert prompts.parse_classification("HUM:ind\nbecause...", LABELS) == "HUM:ind"
    assert prompts.parse_classification("`num:date`", LABELS) == "NUM:date"
    assert prompts.parse_classification("It could be ENTY:animal or maybe HUM:ind.", LABELS) == "HUM:ind"
    assert prompts.parse_classification("no idea", LABELS) is None
    assert prompts.parse_classification("", LABELS) is None


def test_parse_proposal():
    assert prompts.parse_proposal("Sure! <<< Be precise. >>>") == "Be precise."
    assert prompts.parse_proposal("<<<first<<<second>>>") == "first"
    assert prompts.parse_proposal("no block") is None
    assert prompts.parse_proposal("<<<" + "w " * 200 + ">>>") is None


def _pool(n=2000):
    rng = random.Random(0)
    return [data.Item(f"trec-train-{i}", f"q{i}", rng.choice(LABELS)) for i in range(n)]


def test_splits_disjoint_and_nested():
    pool = _pool()
    sp = data.make_splits(pool, seed=11, gold_size=600)
    gold = {i.id for i in sp.gold}
    fb = {i.id for i in sp.feedback}
    sel = {i.id for i in sp.selection(256)}
    assert not gold & fb and not gold & sel and not fb & sel
    assert len(fb) == 256 and len(sel) == 256
    assert [i.id for i in sp.selection(16)] == [i.id for i in sp.selection(256)][:16]
    # gold set is the same for every seed
    assert {i.id for i in data.make_splits(pool, seed=12, gold_size=600).gold} == gold


class FakeLLM:
    """Solver answers correctly with a probability that depends on a hash of (instruction, item);
    proposals are random strings, so all true effects are pure noise around zero."""

    def __init__(self, labels):
        self.labels = labels
        self.calls = 0

    def chat(self, system, user, **kw):
        self.calls += 1
        if "Write an improved" in user:
            return f"<<<instr {random.Random(kw.get('seed')).random():.6f}>>>"
        q = user.split("Question: ")[1].split("\n")[0]
        h = random.Random(hash((system, q))).random()
        return self.labels[0] if h < 0.5 else "none"

    def chat_many(self, system, users, **kw):
        return [self.chat(system, u, **kw) for u in users]


def test_loop_runs_and_shows_winners_curse(tmp_path: Path):
    labels = ["ABBR:abb", "HUM:ind"]
    pool = [data.Item(f"trec-train-{i}", f"q{i}", "ABBR:abb") for i in range(1500)]
    sp = data.make_splits(pool, seed=1, gold_size=600)
    llm = FakeLLM(labels)
    scorer = Scorer(llm, labels, OutputCache(tmp_path / "c.sqlite"))
    res = run_loop(llm, scorer, "task", "seed instr", sp.selection(16), sp.feedback, sp.gold,
                   LoopConfig(n=16, seed=1, generations=5), log=lambda m: None)
    s = res.summary
    assert len(res.generations) == 5
    assert s["final_proxy"] >= s["seed_proxy"]  # greedy never lowers the proxy
    # held-out accuracy of every instruction is ~0.5, so any reported gain is selection noise
    assert abs(s["final_heldout"] - 0.5) < 0.1
    # cache: re-scoring the seed costs no new calls
    before = llm.calls
    scorer.vector("seed instr", sp.gold)
    assert llm.calls == before
