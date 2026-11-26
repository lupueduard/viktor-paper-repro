# Plan (frozen before any replication or test call)

Small CPU replication of the four hypotheses of *What Does a Harness Repair? A Preregistered
Study of Visibility, Baseline Adequacy and Evaluation Defects* (Xu & Chen,
[arXiv:2610.05533v1](https://arxiv.org/abs/2610.05533v1)). This file is committed and pushed
before any replication or test row is called; its commit time is the freeze. Before the freeze,
only the 40-row pilot (20 search rows per pool, THINK-512 and OFF-1500) has been called. The
pilot showed parse and truncation rates (THINK-512 truncated 40/40; OFF-1500 parsed 36/40) but
no other accuracy was computed.

## Setting

- Model: Qwen3-1.7B (base, not the longevity tune), Q8_0 GGUF, llama.cpp `llama-server` b12774,
  CPU, 16 parallel slots, greedy (temperature 0, top-k 1). Raw ChatML prompts (`hr/prompts.py`).
- Pools: GSM8K test (random) and MMLU-Pro test (stratified by category), each split into
  search 100 / replication 150 / test 300 with seed 4242 (`hr/data.py`). The search
  partition is used only by the harness-search arm. A cell is one pool (one model, so two cells).
- Scoring (`hr/scoring.py`): only the text after the last `</think>` is read (thinking-off calls
  read everything). GSM8K: last `#### n`, else `\boxed{n}`, else "answer is n". MMLU-Pro: last
  "answer is (X)", else `\boxed{X}`, else "Answer: X". Unparsed scores as wrong.

## Configurations

| name | request |
|---|---|
| THINK-64, THINK-512 | thinking on, 64 / 512 tokens for thinking and answer together |
| OFF-1500 (paper: AUTHORS-1500) | thinking off, 1,500 tokens |
| BUDGET-512-1500 | THINK-512's output; if thinking was not closed, `</think>` is forced, then up to 1,500 answer tokens |
| RESCUE-1500 | THINK-512, except where that call hit the token limit: OFF-1500's call on the row |
| SEARCH-CLEAN / SEARCH-BROKEN | harness search from OFF-1500 / THINK-64 (below) |

Harness search (stand-in for the paper's GEPA arm): a fixed pool of 8 prompt-text edits (system
prompt or appended instruction; `hr/search.py`, no soft switches), evaluated in a fixed order on
the 100 search rows, greedy keep-if-better on strict improvement. The final incumbent is
selected, then run on replication and test. One run per start and pool (4 runs).

## Hypotheses and verdict rules (m = δ = 0.05)

Percentile bootstrap on items, B = 20,000, paired on rows. Intervals are Bonferroni-adjusted
within each family (q = 0.025 / family size). Index rule as in paper §3.5.

- **H1** (2 endpoints, one per pool). The gain of OFF-1500 over THINK-512 comes mostly from rows
  THINK-512 leaves unparsed. Split Δ = ΔV + ΔB + ΔL; T = ΔV − 0.8 (ΔV + ΔB). If Δ's lower bound
  ≤ 0: *no gain: not tested*. Otherwise *visibility dominates* if T's lower bound > 0,
  *quality matters* if T's upper bound < 0, else *inconclusive*.
- **H2** (6 endpoints). θ = acc(OFF-1500) − acc(comparator) for RESCUE-1500, SEARCH-CLEAN and
  SEARCH-BROKEN in each pool. *Holds* if θ's lower bound > −δ, *fails* if its upper bound < −δ,
  else *inconclusive*.
- **H3** (2 endpoints). BUDGET-512-1500 against THINK-512: D_T = drop in truncation and D_P =
  rise in parse rate. *No room* if THINK-512's truncation upper bound < m (applied first);
  *holds* if both lower bounds > m; *fails* if either upper bound < m; else *inconclusive*.
- **H4** (6 pairs). The clean start is OFF-1500. S = replication partition and R = test
  partition. D = (reported quantity with the defect) − (clean), and Γ = D_R − D_S. *Holds* if
  D_S and D_R lie wholly outside ±m with the same sign and Γ lies wholly inside ±m. *Fails* if
  any of the three lies wholly on the other side of its boundary. Else *inconclusive*. A pair is
  *not evaluable* if the share of S rows whose (prediction, gold) can differ from the clean
  instance is ≤ m. The defects:
  - A, both pools: the reference answer is sent as an earlier assistant turn (GSM8K: the
    dataset rationale with `#### n`; MMLU-Pro: "The answer is (X)").
  - B1, GSM8K: gold read from the last `<<calc>>` result of the rationale instead of the `####`
    line.
  - B2, MMLU-Pro: an unparsed answer is scored as the letter A.
  - D, both pools: broken start. The reported gain is measured over THINK-64 instead of over the
    clean start, so D = acc(OFF-1500) − acc(THINK-64).
- Replication label: a test verdict is *replicated* if the replication partition gives the same
  verdict, *not replicated* if it gives the opposite one, else *inconclusive replication*.
- Descriptive only: accuracy, parse, truncation and non-commitment per configuration; tokens;
  trivial baselines (oracle constant letter / constant number); search score versus test
  accuracy of each selected harness.

## What is not replicated

LongevityBench and the longevity-tuned model (CC-BY-NC data, GPU), the other two models, GEPA
itself (replaced by the fixed-pool search), defects C (skewed split) and E (tied output layer),
audit-check false alarms, the seeded-sampling arm, and THINK-1536.
