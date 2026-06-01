# Verifier-guided ReAct on GAIA: what helped, what regressed, and why

A short, empirical write-up of the policy experiments behind this repo. The
headline is a negative result, stated plainly: on a capacity-limited model,
every elaboration we added on top of bare ReAct **regressed** accuracy, by an
amount proportional to how much model capacity the elaboration consumed. The
useful gains came from fixing structural failure modes in the loop itself, not
from wrapping it in more reasoning.

All numbers are on the **GAIA validation set (165 tasks: 53 L1 / 86 L2 /
26 L3)**, real API, no mocks. Reproduction commands are at the end.

## The baseline that won

| Policy | Model | Accuracy | Cost |
|---|---|---|---|
| **`react`** (bare) | gpt-5-nano | **42.4 %** (70 / 165) | $0.87 |

Per level: **L1 49.1 %**, **L2 46.5 %**, **L3 15.4 %**. 92 % of tasks finish
through a clean commit path; the remaining 8 % trip a budget axis. Median task
finishes in 4 productive steps in under a minute.

## The ablation

Same model, same tools, same task set. The "friendly names" describe what was
layered on top of bare ReAct.

| Row | Policy | Accuracy | Δ vs bare ReAct |
|---|---|---|---|
| A | `react` (bare) | **42.4 %** (70 / 165) | — |
| B | `react` + intrinsic verifiers (format / arithmetic / citation / coverage, with retry) | 37.6 % (62 / 165) | **−4.8 pp** |
| G | `react` + extrinsic verifier (reflexion-style closure check) | 40.0 % (66 / 165) | **−2.4 pp** |
| C | `planner_react` (plan-first decomposition) | ~18–25 % (killed early) | **≈ −10 to −17 pp** |

Every wrapping strategy lost ground. The severity ordering is consistent:
**planning hurt most, intrinsic verification next, extrinsic verification
least.**

### Why each one regressed

- **Planning (worst).** The planner decomposes the question into
  sub-questions, but each sub-question executor sees only a slice of the
  original prompt. On L1 single-fact queries, decomposition is pure overhead and
  the executor commits prematurely on a fragment. L1 cratered to near zero in
  the partial run, which is what killed it.
- **Intrinsic verifiers.** Verifiers checking the *trace* fire false positives
  on answers that were already correct. The retry tick then burns steps/evidence
  and the re-derived answer is often worse than the original. The dominant cost
  was a spike in repair-step-budget exits.
- **Extrinsic verifier (reflexion).** Closest to neutral, because it grades the
  answer against the *user's stated constraints* and skips silently when there
  are none to extract — so it doesn't always-find-something the way a
  trace-grading verifier does. It produced **more** clean-commit exits than bare
  ReAct (it genuinely culls some malformed answers) but the extra critic call +
  occasional retry still cost more capacity than the culling saved.

### The unifying mechanism

`gpt-5-nano` has limited working-memory headroom. Every added component — a
planner call, a verifier-driven retry, a critic call — consumes capacity the
core reasoning loop needs. The Reflexion / Tree-of-Thoughts / planner-first
techniques that these elaborations are modeled on were validated on
GPT-4-tier models, where the headroom to absorb that overhead exists. On a
small model there is no slack: every point of capacity diverted to scaffolding
is a point taken from the answer.

This yields a **falsifiable prediction**: on a higher-capacity model the
regression order should flip — extrinsic verification recovers its cost first,
then intrinsic, then planning starts to pay off.

### A first data point for the capacity hypothesis

The same `react` policy and harness, run on **claude-sonnet-4-5** over the 26
L3 tasks, scores **26.9 % (7 / 26)** versus **15.4 % (4 / 26)** for gpt-5-nano
on the identical set. A capacity jump nearly doubles L3 accuracy with no change
to the policy or tooling — consistent with the bottleneck being model capacity,
not the scaffolding. (The cost is the trade-off: the Sonnet L3-only run cost
~$20.75, roughly 24× the entire 165-task nano run. This is a single-set probe,
not a full cross-model benchmark.)

## What actually helped: the structural fixes (C1–C6)

The gains came from fixing failure modes *inside* the loop. Each was diagnosed
from a specific failure pattern in early runs and is pinned by unit tests.

| Fix | Failure mode it removed | Effect |
|---|---|---|
| **C1 — repair-step axis** | empty-reply / nudge ticks ate the productive-step budget | repair attempts get their own capped axis; a stuck loop no longer starves the task |
| **C2 — empty-reply detection** | model emits no content *and* no tool call, loop hangs until budget trips | after 2 empties with no evidence, force any-tool choice; after 2 commit nudges, bail to `final_answer` |
| **C3 — budget-exhaustion synthesis** | budget trips → `pred_answer = null` (hard zero) | a time-bounded forced-`final_answer` call on exhaustion; null-on-exit dropped ~87 % (22 → 3) |
| **C4 / C4b — default budgets** | L1 dominated by step-cap exhaustion; then by wall-cap | step caps 8/14/20 → 12/18/24; wall 120 s → 240 s |
| **C5 — attachment hints** | rich PDF/XLSX tools used ~0 times; model defaulted to `read_file` on binaries | extension-routed tool hints + a cheap per-attachment summary injected into the question |
| **C6 — per-verifier nudges** | verifier rejection fed back a generic "address the issues" | each verifier emits an actionable nudge (the recomputed value, the missing citation id, …) |

These moved bare ReAct from **33.9 %** (pre-fix, on the verifier-wrapped
policy where the bugs first surfaced) up to the **42.4 %** baseline — a larger
swing than any policy elaboration produced, in the *opposite* direction.

## Honest limitations

- **One model, one provider** for the full set (gpt-5-nano on OpenAI). The
  Sonnet number is L3-only.
- **No held-out split.** GAIA's official test split is private; we report on
  the public validation set, which was also used during engineering iteration.
  This is engineering polish on a known eval, not a generalization claim. A run
  on a separate benchmark would close that loop.
- **Single-agent loop.** No multi-agent delegation.
- The regression results are specific to a capacity-limited model; the
  prediction that they flip on larger models is stated but not yet fully
  measured.

## Reproduce

```bash
pip install -e ".[dev]"
export OPENAI_API_KEY=...

# the 42.4% baseline, full validation set
python experiments/02_gaia_full/run.py \
    --policy react --provider openai --model gpt-5-nano \
    --all-levels --budget-cost 2.0 --budget-wall-s 240 \
    --out-dir runs/$(date +%Y%m%dT%H%M%S)_react_full
```

Per-task event logs land in `<out-dir>/logs/<task_id>.jsonl`; aggregates in
`<out-dir>/results.jsonl` and `<out-dir>/summary.json`.

> The research policies behind rows B / C / G (verifier-retry, planner, and the
> verifier suite) live in the private research repo and are reintroduced into
> the public CLI as each one's validation lands. The public tree ships the bare
> `react` engine and its interactive `react+` subclass.
