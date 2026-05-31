# GAIA validation — `react` on `gpt-5-nano`

This is the validation run behind the public README's accuracy claim. It documents the policy, the setup, the headline numbers, the failure-mode breakdown, and the limitations of the evaluation.

## Setup

| Item | Value |
|---|---|
| Benchmark | GAIA validation, all three levels |
| Tasks | 165 (53 L1 / 86 L2 / 26 L3) |
| Policy | `react` (one LLM call per tick; model picks `THINK` / `TOOL_CALL` / `FINAL_ANSWER`) |
| Model | `gpt-5-nano` |
| Provider | OpenAI (real API, no mocks) |
| Per-task step caps | L1 = 12, L2 = 18, L3 = 24 |
| Per-task wall cap | 240 s |
| Per-task cost cap | $2.00 |
| Separate repair-step axis | `max_repair_steps = 6` (caps recovery attempts independently of the productive-step budget) |
| Run date | 2026-05-20 |

## Reproduce

From a clean clone of this repository:

```bash
pip install -e ".[dev]"
export OPENAI_API_KEY=sk-...

python experiments/02_gaia_full/run.py \
    --policy react --provider openai --model gpt-5-nano \
    --all-levels \
    --budget-cost 2.0 --budget-wall-s 240 \
    --out-dir runs/$(date +%Y%m%dT%H%M%S)_react_full
```

End-to-end wall time on a typical home connection is ~3–4 hours; total cost ~$0.90 in OpenAI API credits. The per-task event log lands in `<out-dir>/logs/<task_id>.jsonl`; aggregate results in `<out-dir>/results.jsonl`.

## Headline

**42.4 %** (70 / 165) on GAIA validation. Total cost: **$0.87**.

### Per-level

| Level | n | Accuracy |
|---|---|---|
| L1 | 53 | **49.1 %** (26 / 53) |
| L2 | 86 | **46.5 %** (40 / 86) |
| L3 | 26 | 15.4 % (4 / 26) |

The L3 gap reflects the chained-reasoning regime: tasks requiring 5+ steps of dependent search/read/reason against the model's effective context window. This is the regime where compositional policies (planning, best-of-N, etc.) are expected to help; their validation runs are in progress.

### Exit distribution

| Reason loop exited | Count | Share |
|---|---|---|
| Clean commit (`ok`) | 152 | 92 % |
| Wall budget exhausted | 9 | 5 % |
| Repair-step budget exhausted | 4 | 2 % |
| Step budget exhausted | 0 | 0 % |

92 % of tasks finish through the normal commit path; the remaining 8 % trip a budget axis. No step-budget exhaustions on this set, which is the post-engineering target for the default step caps.

### Operational stats

| Metric | Value |
|---|---|
| Total tool calls across the run | 656 |
| Tool-call mix (top) | `search` 467, `read_file` 41, `read_url` 36, `run_python` 27, `xlsx_list_sheets` 22, `browser_open` 15 |
| Median `steps_used` per task | 4 |
| p90 `steps_used` | 10 |
| Median wall per task | 63.7 s |
| p90 wall | 188.4 s |

Median task finishes in 4 productive steps under a minute. The long tail (p90 ≈ 188 s) is approaching the 240 s wall cap; the 9 wall-cap exits are concentrated here.

## Reliability engineering decisions in this build

The `react` loop ships with five structural choices that meaningfully change behavior versus a stock ReAct implementation. Each was added in response to a specific failure mode observed in early validation runs and is now pinned by unit tests.

| Decision | What changed | Why it matters |
|---|---|---|
| Multi-axis budget tracker | The `Budget` separates productive steps from repair steps. Actions tagged `meta["repair"]=True` (empty-reply fallbacks, commit-required nudges, retry feedback) tick a dedicated `repair_steps_used` axis with its own cap. | A model stuck in an empty-reply loop no longer drains its productive-step budget. The two failure modes get distinct exit codes. |
| Empty-reply detection + provider-agnostic forced tool use | After two consecutive empty replies with no evidence accumulated, the policy forces the provider's "any tool" mode (OpenAI `"required"`, Anthropic `{"type":"any"}`, Gemini `ANY`). After two commit-required nudges, the policy bails to `final_answer`. | Empty-reply loops were a top cause of zero-tool / zero-evidence finishes on small models. |
| Budget-exhaustion synthesis hook | When any budget axis trips without a committed answer, the policy calls a 15-second-bounded LLM with `tool_choice` forced to `final_answer`, falling through to a cheap chain (last claim → last short preceding text → none). | Eliminates the "ran out of budget, returned `null`" failure class; surfaces a best-effort answer instead of a hard zero. |
| Default budget profile | L1 / L2 / L3 step caps at 12 / 18 / 24, wall cap at 240 s. Repair-step cap independent at 6. | Earlier defaults were too tight on L1 (step exhaustion was the dominant failure) and too loose on wall (long tails tripped wall mid-task). |
| Attachment-aware tool hints | The system prompt is extended with extension-routed tool affordances (PDF / XLSX / CSV / image), and a cheap per-attachment summary (page count, sheet names, header line) is injected into the question. | Without this, the model defaulted to `read_file` on every attachment, including binaries; the rich PDF / XLSX tools were rarely picked. |

## Limitations of this evaluation

- **One model, one provider.** Numbers are specific to `gpt-5-nano` on OpenAI. Behavior on Claude, Gemini, or Ollama models can differ, and is not measured here.
- **No held-out test split.** GAIA's official test split is private; we report on the public validation set only. The same set was used during engineering iteration, so this is engineering polish on a known eval, not a generalization claim against a held-out distribution. A run against a separate benchmark (e.g. BrowseComp) is queued.
- **Single-agent loop.** This is not a multi-agent system. Tasks requiring delegation across specialist agents are out of scope.
- **No video / image-reasoning tools.** A small number of GAIA tasks require video frame extraction or vision-grade image reasoning; those are structurally unreachable without additional tools.
- **Tool isolation depends on the backend.** This run used the default `process` backend (host subprocess), gated by a confirmation prompt and a denylist. A `docker` backend (network-less, read-only container) is available via `--sandbox=docker` for untrusted input; it was not used for this evaluation, so the numbers reflect host execution.

## Why not wrap `react` with a verifier-retry loop?

A natural question — and one we measured. Two alternative policies were validated against bare `react` on the same model, same substrate, same task set:

| Policy | Accuracy | Δ vs. bare `react` |
|---|---|---|
| `react` (bare) | **42.4 %** (70 / 165) | — |
| `react` + verifier-retry with self-consistency checks (Format / Arithmetic / Citation / Coverage on the trace) | 37.6 % (62 / 165) | **−4.8 pp** |
| `react` + verifier-retry with reflexion-style closure check (LLM-as-critic grading the answer against the user's stated constraints) | 40.0 % (66 / 165) | **−2.4 pp** |

Both wrapping strategies regressed on this model. Mechanism: each elaboration consumes a slice of the model's reasoning capacity (extra LLM call, retry tick has to reconstruct context, false-positive retries discard correct answers). On `gpt-5-nano`, there isn't enough headroom for that overhead to pay off — the retry tick degrades the answer more often than it improves it. The reflexion variant is closer to neutral because it skips silently on questions with no extractable constraints, but still regresses on net.

The plausible interpretation: each elaboration only pays off when the model has enough spare reasoning capacity to absorb its overhead (an extra LLM call, a retry tick that must reconstruct context, the risk of a false-positive retry discarding a correct answer). On `gpt-5-nano`, that headroom isn't there, so the overhead dominates and the wrapping regresses.

## Ongoing work

Further policies (planner-based decomposition, best-of-N selection, verifier-retry) are developed and validated in the private research repo; they are not part of this public tree, which ships only the bare `react` engine and its interactive `react+` subclass. Each validated row graduates into the public CLI as it lands, so the public surface stays to what has a benchmark run behind it.
