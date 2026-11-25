"""Summarize loop runs into results tables (markdown + CSV).

    python scripts/analyze.py --runs runs --out results

Quantities (points = percentage points of accuracy), following arXiv:2610.09239:
  gap (inflation)  = final selection-set score (proxy) - final held-out accuracy
  reported gain    = final proxy - seed proxy
  held-out gain    = final held-out - seed held-out
  overstatement    = reported gain - held-out gain  (= gap(final) - gap(seed))
  seed gap         = seed proxy - seed held-out (luck of the selection-set draw, before any selection)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

from scipy import stats


def load(runs: Path) -> list[dict]:
    rows = []
    for p in sorted(runs.glob("trec_n*_seed*.json")):
        d = json.loads(p.read_text())
        s, c = d["summary"], d["config"]
        rows.append({
            "n": c["n"], "seed": c["seed"], "generations": c["generations"], "commits": s["commits"],
            "seed_proxy": 100 * s["seed_proxy"], "final_proxy": 100 * s["final_proxy"],
            "seed_heldout": 100 * s["seed_heldout"], "final_heldout": 100 * s["final_heldout"],
            "solver_calls": c.get("solver_calls"), "wall_seconds": c.get("wall_seconds"),
            "final_instruction": s["final_instruction"],
            "underfilled_generations": sum(g["n_candidates"] < c["k"] for g in d["generations"]),
        })
    for r in rows:
        r["gap"] = r["final_proxy"] - r["final_heldout"]
        r["seed_gap"] = r["seed_proxy"] - r["seed_heldout"]
        r["reported_gain"] = r["final_proxy"] - r["seed_proxy"]
        r["heldout_gain"] = r["final_heldout"] - r["seed_heldout"]
        r["overstatement"] = r["reported_gain"] - r["heldout_gain"]
    return rows


def ms(xs: list[float]) -> str:
    if not xs:
        return "–"
    m = st.mean(xs)
    se = st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")
    return f"{m:+.1f} ± {se:.1f}" if not math.isnan(se) else f"{m:+.1f}"


def paired(rows, metric, a, b):
    by = {(r["n"], r["seed"]): r[metric] for r in rows}
    seeds = sorted({s for (n, s) in by if (a, s) in by and (b, s) in by})
    d = [by[(a, s)] - by[(b, s)] for s in seeds]
    if len(d) < 2:
        return None
    t = stats.ttest_1samp(d, 0.0)
    return {"metric": metric, "a": a, "b": b, "pairs": len(d), "mean": st.mean(d),
            "se": st.stdev(d) / math.sqrt(len(d)), "wins": sum(x > 0 for x in d), "p": float(t.pvalue)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    rows = load(Path(args.runs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["n", "seed", "commits", "seed_proxy", "final_proxy", "seed_heldout", "final_heldout", "gap", "seed_gap",
            "reported_gain", "heldout_gain", "overstatement", "underfilled_generations", "solver_calls",
            "wall_seconds"]
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["final_instruction"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["n"], r["seed"])):
            w.writerow({k: (round(r[k], 2) if isinstance(r[k], float) else r[k]) for k in cols + ["final_instruction"]})

    ns = sorted({r["n"] for r in rows})
    lines = ["| n (selection items) | seeds | final proxy − held-out (gap) | seed gap | reported gain | held-out gain | "
             "overstatement | commits |", "|---|---|---|---|---|---|---|---|"]
    for n in ns:
        g = [r for r in rows if r["n"] == n]
        lines.append(f"| {n} | {len(g)} | {ms([r['gap'] for r in g])} | {ms([r['seed_gap'] for r in g])} | {ms([r['reported_gain'] for r in g])} | "
                     f"{ms([r['heldout_gain'] for r in g])} | {ms([r['overstatement'] for r in g])} | "
                     f"{st.mean([r['commits'] for r in g]):.1f} |")
    tests = []
    if 16 in ns and 256 in ns:
        for m in ("gap", "overstatement", "seed_gap"):
            tests.append(paired(rows, m, 16, 256))
        t = paired(rows, "heldout_gain", 256, 16)
        tests.append(t)
    lines += ["", "Paired by seed (two-sided one-sample t-test on the per-seed differences):", "",
              "| test | pairs | mean ± s.e. (pts) | seeds in direction | p |", "|---|---|---|---|---|"]
    for t in filter(None, tests):
        lines.append(f"| {t['metric']}(n={t['a']}) − {t['metric']}(n={t['b']}) | {t['pairs']} | "
                     f"{t['mean']:+.1f} ± {t['se']:.1f} | {t['wins']} / {t['pairs']} | {t['p']:.3f} |")
    lines += ["", "Per run:", "", "| n | seed | commits | seed proxy | final proxy | seed held-out | final held-out | gap | "
              "overstatement |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["n"], r["seed"])):
        lines.append(f"| {r['n']} | {r['seed']} | {r['commits']} | {r['seed_proxy']:.1f} | {r['final_proxy']:.1f} | "
                     f"{r['seed_heldout']:.1f} | {r['final_heldout']:.1f} | {r['gap']:+.1f} | {r['overstatement']:+.1f} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    (out / "tests.json").write_text(json.dumps([t for t in tests if t], indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
