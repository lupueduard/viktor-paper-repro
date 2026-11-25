# Winner's curse in greedy self-improvement loops: a small reproduction

This folder checks one claim from *The Winner's Curse in LLM Self-Improvement Loops*
(Hu & Tang, [arXiv:2610.09239](https://arxiv.org/abs/2610.09239)): a greedy keep-if-better
prompt-optimization loop overfits the selection set it reuses. The paper reports that the final
selection-set score is higher than held-out accuracy by **13–20 points with 16 selection items**
and by **1–5 points with 256**.

**Verdict: the claim holds here in direction and roughly in size, with one caveat.**

- At n=16 the gap is +24.8 ± 2.2 points, which is larger than the paper's 13–20.
- At n=256 the gap is +4.0 ± 2.1 points, inside the paper's 1–5.
- Gap(16) − gap(256) is +20.7 ± 3.8 points. All 4 seeds go that way (p = 0.012). The paper
  reports +15.9 ± 2.7 on 8 seeds.

The caveat is that about 14 of the 25 points at n=16 came from luck in drawing the selection
set: the four 16-item sets happened to be easier for the seed instruction than the held-out set,
before any selection ran. The part the loop itself added (*overstatement*: reported gain minus
held-out gain) is +10.2 ± 3.8 points at n=16 against +1.9 ± 2.2 at n=256. It points the same
way on 4/4 seeds, but with only 4 seeds it is not significant (p = 0.06).

## Setup (mirrors the paper's confirmatory TREC setting, scaled down)

| | Paper (confirmatory) | Here |
|---|---|---|
| Solver and proposer | Qwen2.5-1.5B-Instruct (the paper also has a llama.cpp GGUF stack) | Qwen2.5-1.5B-Instruct, Q8_0 GGUF on llama.cpp b12722, CPU only |
| Task | TREC-50, pool = train + test, gold set of 600 drawn with `Random(1234)` | same |
| Prompts, seed instruction, task description | Appendix B | verbatim from Appendix B (`wcurse/prompts.py`, `wcurse/data.py`) |
| Proposer | K=4, T=0.9, top_p=0.95, 400 max tokens, up to 5 resample rounds | same |
| Solver decoding | greedy, 48 max tokens | same |
| Failure examples | separate 256-item feedback pool | same |
| Selection set | reused, nested across n for each seed | same |
| Acceptance | greedy keep-if-better (strict improvement on the selection set) | same |
| Selection sizes | 16, 64, 256 | 16, 64, 256 |
| Generations | 20 | **10** |
| Seeds | 8 | **4** (11–14) |
| Gold scoring | every generation | **seed and final incumbent only** |

Cost: 72.6k solver calls and about 9 h of loop time, run 3 loops at a time on one 16-CPU box
(about 3.3 h wall clock). Outputs are cached by (instruction, item), so a repeated candidate is
never re-scored.

## Results (4 seeds, mean ± s.e., accuracy points)

| n | gap: final proxy − held-out | seed gap (before selection) | reported gain | held-out gain | overstatement | commits |
|---|---|---|---|---|---|---|
| 16 | **+24.8 ± 2.2** | +14.5 ± 4.4 | +15.6 ± 4.0 | +5.4 ± 2.3 | +10.2 ± 3.8 | 1.5 |
| 64 | +8.4 ± 1.1 | +4.0 ± 2.3 | +14.1 ± 3.7 | +9.6 ± 1.9 | +4.4 ± 1.8 | 4.0 |
| 256 | **+4.0 ± 2.1** | +2.1 ± 1.2 | +13.8 ± 2.6 | +11.8 ± 2.0 | +1.9 ± 2.2 | 4.2 |

Paired by seed (two-sided t-test on per-seed differences, 4 pairs):

| test | here | seeds in direction | p | paper |
|---|---|---|---|---|
| gap(16) − gap(256) | +20.7 ± 3.8 | 4/4 | 0.012 | +15.9 ± 2.7, 8/8, p = 0.002 |
| overstatement(16) − overstatement(256) | +8.3 ± 2.8 | 4/4 | 0.060 | – |
| seed gap(16) − seed gap(256) | +12.4 ± 3.7 | 4/4 | 0.044 | – (zero in expectation) |
| held-out gain(256) − held-out gain(16) | +6.5 ± 2.5 | 4/4 | 0.083 | 24.2 ± 0.7 vs 13.7 ± 2.9 |

Inflation (the final gap) against the paper's Section 5.2, which is measured after 11–20
generations: 13.6 / 7.2 / 1.7 there, and 24.8 / 8.4 / 4.0 here, for n = 16 / 64 / 256.

Per-run numbers are in [`results/results.md`](results/results.md) and
[`results/runs.csv`](results/runs.csv), and the full loop traces are in `runs/`.

## Where it held and where it didn't

- **Held:** the final proxy − held-out gap is large at n=16 and small at n=256. The n=256 value
  (+4.0) falls inside the paper's 1–5 range, and the paired difference is significant even with
  4 seeds.
- **Held, larger than reported:** the n=16 gap (+24.8) is above the paper's 13–20. Most of the
  excess is selection-set draw luck, not loop overfitting. The four 16-item sets scored 12.5–31.2%
  with the seed instruction, while held-out was 10.5%, so they started +14.5 points ahead on
  average. The gap metric doesn't separate the two effects. Overstatement does, and it shows the
  loop-induced part: about 10 points at n=16 and about 2 at n=256.
- **Held in direction only:** with more selection items the held-out gain is larger (11.8 vs 5.4
  points, 4/4 seeds), but neither the gains nor the difference reach the paper's sizes or
  significance. Likely reasons: half the generations, and n=16 loops stall after 1–2 commits
  because a 16-item set can only move in steps of 6.25 points.
- **Not tested:** the paper's fixes (Bayes gate, McNemar acceptance, select-then-confirm),
  Banking77 and GSM8K, GEPA and MIPROv2, other model sizes, and 20 generations. These were left
  out to keep each run small.

## Caveats

- 4 seeds, so p-values are coarse and the s.e. are rough.
- T=10 instead of 20. The paper's Section 5.2 figures are measured after 11–20 generations.
- llama.cpp parallel batching is not bitwise deterministic. The same (instruction, item) pair can
  occasionally decode differently, for example one item out of 256 in the feedback pool between
  two loops that started together. The cache makes repeats within a run consistent.
- In 3 of 120 generations fewer than 4 valid candidates came back after 5 resample rounds; those
  generations ran with 3 candidates.

## Layout and how to run

```
wcurse/data.py       TREC loading, gold / feedback / nested selection splits, seed instruction
wcurse/prompts.py    solver and proposer prompts (Appendix B), output parsing
wcurse/llm.py        OpenAI-compatible client for llama-server
wcurse/loop.py       greedy loop, output cache, scorer
scripts/run_loops.py grid runner (skips finished runs)
scripts/analyze.py   results.md / runs.csv / tests.json
tests/test_core.py   fake-LLM unit tests
runs/                one JSON per loop (config, every generation's candidates and scores, gold vectors)
results/             analysis output
```

```bash
# data (TREC from CogComp)
mkdir -p data && cd data
curl -LO https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label
curl -LO https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label
cd ..
# model server (llama.cpp, Qwen2.5-1.5B-Instruct Q8_0 GGUF)
llama-server -m qwen2.5-1.5b-instruct-q8_0.gguf -t 16 -np 32 -c 131072 --port 8080 &
# grid, then analysis
python scripts/run_loops.py --out runs --ns 16 64 256 --seeds 11 12 13 14 --generations 10 --parallel 3 --workers 16
python scripts/analyze.py --runs runs --out results   # needs scipy
python -m pytest -q tests
```
