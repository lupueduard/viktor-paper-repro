"""Run greedy keep-if-better loops for a grid of (n, seed) against a running llama-server.

Example:
    python scripts/run_loops.py --data data --out runs --ns 16 64 256 --seeds 11 12 13 14 \
        --generations 10 --gold-size 600 --parallel 3
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wcurse import data  # noqa: E402
from wcurse.llm import LlamaServer  # noqa: E402
from wcurse.loop import LoopConfig, OutputCache, Scorer, run_loop, save_result  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--ns", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14])
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--gold-size", type=int, default=600)
    ap.add_argument("--parallel", type=int, default=3, help="loops run concurrently")
    ap.add_argument("--workers", type=int, default=8, help="concurrent solver requests per loop")
    args = ap.parse_args()

    pool = data.load_trec(args.data)
    labels = data.label_set(pool)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = OutputCache(out / "solver_cache.sqlite")

    jobs = [(n, s) for s in args.seeds for n in args.ns]

    def one(job):
        n, seed = job
        path = out / f"trec_n{n}_seed{seed}.json"
        if path.exists():
            print(f"skip {path}", flush=True)
            return
        sp = data.make_splits(pool, seed, gold_size=args.gold_size)
        llm = LlamaServer(args.server, workers=args.workers)
        scorer = Scorer(llm, labels, cache)
        t0 = time.time()
        res = run_loop(llm, scorer, data.TASK_DESCRIPTION, data.SEED_INSTRUCTION, sp.selection(n),
                       sp.feedback, sp.gold, LoopConfig(n=n, seed=seed, generations=args.generations, k=args.k),
                       log=lambda m: print(m, flush=True))
        res.config.update({"task": "trec", "gold_size": args.gold_size, "solver_calls": scorer.evals,
                           "wall_seconds": round(time.time() - t0, 1)})
        save_result(res, path)
        print(f"done n={n} seed={seed} in {time.time() - t0:.0f}s ({scorer.evals} solver calls)", flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for f in [ex.submit(one, j) for j in jobs]:
            f.result()


if __name__ == "__main__":
    main()
