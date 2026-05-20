# GAIA validation: C1–C6 reliability pass

This is the pre-/post-fix evaluation behind the README's "Failure modes & fixes" claim. It documents how the seven structural failure modes were diagnosed, what each fix changed, and what the changes did to GAIA accuracy, cost, and failure distribution.

> Status: post-fix numbers are populated from the `runs/20260519T135650_postC1C6b_verifier_retry_full/` run. Pre-fix numbers come from `runs/20260516_postC0_verifier_retry_full/` (the run that surfaced the bugs).

## Setup

| Item | Value |
|---|---|
| Benchmark | GAIA validation, all three levels |
| Tasks | 165 (53 L1 / 86 L2 / 26 L3) |
| Model | `gpt-5-nano` (OpenAI) |
| Policies tested | `react + intrinsic` (pre & post-C1–C6b), `react` (bare — post-C1–C6b ablation row A); further ablation rows queued |
| Provider | `openai` (real API, no mocks) |
| Default budget — pre | steps 8 / 14 / 20 (L1/L2/L3), wall 120 s, cost unlimited |
| Default budget — post | steps 12 / 18 / 24 (L1/L2/L3), wall **240 s**, cost cap $2.00/task |
| Repair-step budget — post | `max_repair_steps = 6` (separate axis from main steps) |

### Reproduce

From a clean clone of the public repo:

```bash
pip install -e ".[dev]"
export OPENAI_API_KEY=sk-...

python experiments/02_gaia_full/run.py \
    --policy verifier_retry --provider openai --model gpt-5-nano \
    --all-levels \
    --budget-cost 2.0 --budget-wall-s 240 \
    --out-dir runs/$(date +%Y%m%dT%H%M%S)_verifier_retry_full
```

A run takes ~4–5 hours and costs ~$1.20 in OpenAI API credits. The per-task event log is written to `<out-dir>/logs/<task_id>.jsonl`; the resume table is written to `<out-dir>/results.jsonl`.

## Headline

| Run | Set | Policy (friendly name) | Accuracy | Cost | Notes |
|---|---|---|---|---|---|
| Pre-C1–C6 baseline | GAIA val (165) | `react + intrinsic` | **33.9 %** (56 / 165) | $0.94 | Surfaced 7 structural failure modes |
| Post-C1–C6b | GAIA val (165) | `react + intrinsic` | **37.6 %** (62 / 165) | $1.01 | +3.7 pp from substrate fixes alone |
| Post-C1–C6b | GAIA val (165) | **`react`** (bare — ablation row A) | **42.4 %** (70 / 165) | $0.87 | **+4.8 pp over `react + intrinsic`** — *intrinsic verification regresses accuracy on this substrate* |

The bare-`react` row is the surprise. On the post-fix substrate, the intrinsic verifier set (Format / Arithmetic / Citation / Coverage) is a **net negative** of ~5 pp. The mechanism appears to be false-positive retries: verifiers reject correct answers, the retry tick burns evidence and steps, the retried answer is worse than the original.

This makes the **next ablation question the headline question**: does extrinsic (reflexion-style) verification — checking the answer against the user's prompt rather than against the trace — flip the sign? See *"Ablation table"* below.

### Per-level (post-C1–C6b)

| Level | n | `react + intrinsic` | **`react`** (row A) | Δ |
|---|---|---|---|---|
| L1 | 53 | 47.2 % | **49.1 %** | +1.9 pp |
| L2 | 86 | 38.4 % | **46.5 %** | **+8.1 pp** |
| L3 | 26 | 15.4 % | 15.4 % | 0 |

L2 is where the regression concentrates — exactly the regime where verifier retries fire most often (multi-hop questions with cited evidence). The 8.1 pp gap at L2 is the smoking gun: verifier retries kill more correct answers than they save.

### Exit distribution (post-C1–C6b)

| budget_reason | `react + intrinsic` | **`react`** (row A) |
|---|---|---|
| `ok` (clean commit) | 138 / 165 (84 %) | **152 / 165 (92 %)** |
| `budget_wall` | 13 | 9 |
| `budget_repair_steps` | 12 | 4 |
| `budget_steps` | 2 | 0 |
| `pred_answer = null` | 3 | (still being scored) |

Repair-step exits dropped by 3× without verifiers — bare ReAct doesn't enter the repair loop in the first place.

## Ablation table — the path forward

Two rows done; the next four are the test of whether verification *as a class* helps:

| Row | Friendly name | Implementation | Status |
|---|---|---|---|
| A | `react` | `ReActPolicy` | **done — 42.4 %** |
| B | `react + intrinsic` | `verifier_retry(react, [format, arithmetic, citation, coverage])` | **done — 37.6 % (regression)** |
| C | `planner_react` | `PlannerReActPolicy` | queued (next) |
| G | `react + extrinsic` | `verifier_retry(react, [count, reflexion])` | queued |
| E | `react + intrinsic + extrinsic` | `verifier_retry(react, [...all 6...])` | queued |
| F | `planner_react + intrinsic + extrinsic` | `verifier_retry(planner_react, [...all 6...])` | queued |

The architecturally interesting deltas:
- **A → G**: does extrinsic-only verification recover or improve on bare ReAct? (Tests whether the wrapper-with-good-verifiers is positive.)
- **A → C**: does planning alone help over bare ReAct?
- **C → F**: does adding verifiers to planner_react regress the same way it regressed react?
- **G vs. E**: does keeping intrinsic *alongside* extrinsic still hurt?

## Failure-mode comparison

For each ID, count of tasks that exited the loop with that `budget_reason`:

All counts are for `react + intrinsic` (the policy on which the C1–C6 fixes were diagnosed). See the *Exit distribution* table above for the post-fix bare-`react` counts.

| Code | budget_reason | Pre-fix | Post-fix (`react + intrinsic`) | Diagnosis |
|---|---|---|---|---|
| — | `ok` (finished cleanly) | ~110 / 165 | **138 / 165** | The dominant healthy path |
| — | `budget_steps` | 27 (L1: 19) | **2** | Step cap; near-eliminated by C4 |
| C1 | `budget_repair_steps` (new axis) | n/a | **12** | New tripwire catches `[empty_reply]` loops + verifier-retry storms before they eat productive steps |
| — | `budget_wall` | 25 / 49 (partial C4-only run) | **13** | Wall cap; C4b raised 120 → 240 s |
| — | `budget_tokens` | 0 | 0 | Never binding on this set |
| C3 | `pred_answer = null` on exit | 22 | **3** | Budget-exhaustion synthesis hook firing (~87 % reduction) |

## The seven fixes

Each entry: one-line failure mode → root cause → fix → expected measurable effect.

### C1 — Repair-step axis split

**Symptom (pre-fix):** 75 `[empty_reply]` events across 29 tasks. The 29 affected tasks scored 21 % accuracy vs. 37 % on the clean set. A model that produced empty content on tick 2 burned that tick against the productive step cap, leaving 5–6 productive steps for the actual task.

**Root cause:** `Budget.steps_used` counted every tick, including repair-style THINKs (empty-reply fallbacks, commit-required nudges, verifier-retry feedback). One axis conflated "trying to make progress" with "trying to recover from a stuck loop."

**Fix (`src/banna_agent/core/types.py`, `src/banna_agent/core/state.py`):**
- New axis `Budget.repair_steps_used`, capped at `max_repair_steps = 6`.
- `state.append_step` checks `action.meta.get("repair")`. Repair-flagged actions tick the new axis; productive actions tick the main axis and reset the repair counter.
- New `BudgetReason.REPAIR_STEPS` enum entry so the failure taxonomy can distinguish.

**Pinned by tests:** `tests/core/test_state.py::test_max_repair_steps_trips_separate_axis`, `test_repair_step_does_not_tick_main_budget`, `test_productive_step_resets_repair_streak`.

### C2 — Empty-reply detection upgrades

**Symptom (pre-fix):** One representative trace (`dc28cf18`) had 8 consecutive empty replies, then `budget_steps` tripped with zero tool calls and zero evidence. The model emitted no content *and* no tool call — the LLM equivalent of a hung process.

**Root cause:** No detection. The policy treated empty content as a noop and waited for the next tick.

**Fix (`src/banna_agent/policies/react.py`):**
- `_count_consecutive_empty_replies(state)` and `_trace_has_evidence(state)` helpers.
- `_force_tool_extra(provider, tool_name=None)` extended to support an "any tool" mode: OpenAI `"required"`, Anthropic `{"type": "any"}`, Gemini `ANY` mode *without* `allowed_function_names`.
- After 2 consecutive empties with no evidence in state, force `tool_choice` to any tool (escape the loop).
- After 2 commit-required nudges + preceding text, bail to `final_answer`.

**Pinned by tests:** `tests/policies/test_react.py::test_two_empties_with_no_evidence_forces_required_tool_choice_openai`, `test_two_empties_with_evidence_forces_final_answer`.

### C3 — Budget-exhaustion synthesis

**Symptom (pre-fix):** 22 tasks returned `pred_answer = null` when budget tripped. Hard zero — no chance to score.

**Root cause:** Budget trip exited the loop without giving the model a final chance to commit.

**Fix (`src/banna_agent/policies/react.py`, `src/banna_agent/policies/verifier_retry.py`, `src/banna_agent/core/agent.py`):**
- New `policy.synthesize_on_exhaustion(state, llm, tools, timeout_s=15.0)` hook.
- On any non-OK budget trip with no final answer in trace, the driver calls the hook.
- `ReActPolicy.synthesize_on_exhaustion`: one threaded LLM call with `final_answer` forced; threaded so a hung HTTP can't extend the run by another N seconds.
- Cheap fallback chain when no LLM is available or it fails: last claim → last short preceding text → `None`.

**Pinned by tests:** `tests/policies/test_react.py::test_synthesize_uses_last_claim_when_no_llm`, `test_synthesize_uses_short_preceding_text_when_no_claims`, `test_synthesize_calls_llm_with_forced_final_answer_when_provided`, `test_synthesize_falls_back_to_cheap_on_llm_failure`.

### C4 — Default step caps

**Symptom (pre-fix):** L1 had 19 / 27 `budget_steps` failures. Step cap = 8 included nudge-eaten steps.

**Fix (`src/banna_agent/benchmarks/gaia/runner.py`):**
- `_default_budget`: step caps 8 / 14 / 20 → **12 / 18 / 24** for L1 / L2 / L3.

**Pinned by tests:** `tests/benchmarks/test_gaia_runner.py::test_default_budget_step_caps_match_post_c4`.

### C4b — Default wall cap

**Symptom (post-C4 first run):** With step caps raised, 25 / 49 tasks at the running snapshot tripped `budget_wall` at 120 s (median timeout 127 s — i.e. tasks ran to the wire). Bigger step budget without bigger wall budget shifted the dominant failure mode rather than fixing it.

**Fix:** `_default_budget` wall cap **120 s → 240 s**. Regression test extended.

**Pinned by tests:** same test as C4, with an added wall-cap assertion.

### C5 — Attachment hint

**Symptom (pre-fix):** Across 38 GAIA tasks with file attachments, the model used the rich PDF tools `0` times and the XLSX tools `1` time. It defaulted to `read_file` on every attachment, including binaries.

**Root cause:** The system prompt's tool affordance hint listed `read_file` as the canonical attachment tool. PDF / XLSX tool families existed but weren't surfaced when the task actually had a PDF / XLSX.

**Fix (`src/banna_agent/benchmarks/gaia/runner.py`):**
- `_RICH_TOOL_HINTS`: extension → tool family.
- `_file_hint(path)`: routes by extension to `pdf_open / pdf_read_page / pdf_find / pdf_read_tables` for `.pdf`, `xlsx_list_sheets / xlsx_describe / xlsx_read_range / xlsx_find` for `.xlsx`, etc.
- `_file_summary(path)`: cheap, silent-on-failure pre-introspection — pypdf page count, openpyxl sheet names, CSV header line — injected into the question text.

**Pinned by tests:** `tests/benchmarks/test_gaia_runner.py::test_format_question_pdf_hint_mentions_pdf_tools`, `test_format_question_xlsx_hint_mentions_xlsx_tools`, `test_format_question_csv_keeps_read_file`, `test_format_question_omits_summary_when_introspection_fails`.

### C6 — Per-verifier actionable nudges

**Symptom (pre-fix):** When a verifier rejected the proposed answer, the retry-THINK feedback was a generic "address the issue(s) above." Verifier-retry repair rate was low — the model was told *that* it failed but not *what to do*.

**Root cause:** Verifier `ClaimCheck.meta` carried only the structural reason, no actionable string.

**Fix (`src/banna_agent/verifiers/{arithmetic,citation,coverage,format}.py`, `src/banna_agent/policies/verifier_retry.py`):**
- `ArithmeticVerifier`: on fail, `meta["nudge"]` names the recomputed value and the asserted value. "Your reasoning asserts `47 * 83 = 3801` but recomputed `3901`. Recompute the step, then re-emit `final_answer`."
- `CitationVerifier`: nudge names the missing numeric or broken `evidence_id`; instructs to use real IDs from prior tool calls.
- `CoverageVerifier`: nudge instructs to run `search`/`read_url` before re-emitting.
- `FormatVerifier`: nudge names the empty `answer` field and tells the model to put the literal answer string there, not in `reasoning`.
- `VerifierRetryPolicy._format_feedback` rewritten: group failures by verifier, emit one nudge per verifier (not per claim). Compact, scan-able.

**Pinned by tests:** `tests/policies/test_verifier_retry.py::test_format_feedback_includes_per_verifier_nudge`, `test_format_feedback_groups_by_verifier_one_per_kind`, `test_format_feedback_canonical_emits_required_action`, and per-verifier nudge content tests.

## Representative traces

When the post-fix run completes, three task IDs will be selected to illustrate the fix story end-to-end:

1. **A clean ReAct success on L1** — shows the happy path the public repo gets right out of the box. _Trace path TBD._
2. **A verifier-retry rescue on L2** — task where the first commit was rejected by `CitationVerifier`, the nudge fired, the next tick re-searched and re-committed correctly. _Trace path TBD._
3. **A hard attachment task using PDF or XLSX tools** — shows the C5 fix exercising the rich tool family that the pre-fix run never reached. _Trace path TBD._

Traces will be linked from this report as `runs/<id>/logs/<task_id>.jsonl` for full reproducibility.

## Limitations of this evaluation

- **One model, one provider.** Numbers are specific to `gpt-5-nano` on OpenAI. Behavior on Claude / Gemini / Ollama models can differ — for example, the empty-reply failure mode is provider-specific (it was more common on `gpt-5-nano` than the Anthropic / Gemini runs).
- **No ablation runs in this report.** A clean ablation would re-run each Cx in isolation against the C0 baseline to attribute deltas to specific fixes. Adding that would multiply runtime by 6; deferred unless a reviewer asks for it.
- **GAIA validation set was used both to surface the failure modes and to score the fixes.** Strictly this is data leakage — fixes were tuned to bugs visible in the same data they're scored on. The honest read: this is engineering polish on a known eval, not a held-out generalization claim. A true held-out run on a separate benchmark (BrowseComp, HLE) would close that loop.
