# banna

A from-scratch, provider-agnostic reasoning agent with a **typed state substrate** and a **verifier-guided** loop. Built to study where ReAct-style agents fail on the **GAIA** benchmark and to fix those failures structurally — not with prompt patches.

No LangChain, no LlamaIndex, no smolagents in the core. The reasoning loop is a typed transition function over `(state, action, observation) → state'`; ReAct, verifier-retry, planner-ReAct, BFS/DFS/best-first-over-plans, and best-of-N are each ~200 LOC `Policy` implementations over that same substrate.

## What's interesting about this repo

1. **Forensic GAIA debugging.** A full-validation run on `gpt-5-nano` was instrumented end-to-end, traces were dumped per task, and seven distinct structural failure modes were diagnosed and fixed — not by prompt-tweaking, but by changing the loop. See [Failure modes & fixes](#failure-modes--fixes-the-c1c6-pass) below.
2. **Multi-axis budget tracker** that separates *productive* steps from *repair* steps. A model stuck in an `[empty_reply]` loop no longer burns its productive-step budget; instead it trips a separate `max_repair_steps` axis with a forced tool-choice escape.
3. **Per-verifier actionable nudges.** Each verifier (Arithmetic, Citation, Coverage, Format) attaches a `meta["nudge"]` to its fail verdicts that names the missing thing (the recomputed value, the missing evidence_id, the unsupported number, the empty field). The retry policy groups these by verifier and emits one line per kind — short enough that the model actually reads them.
4. **Budget-exhaustion synthesis.** When the agent runs out of steps mid-task, instead of returning `null`, a final forced-`final_answer` call gives it one last shot with a cheap fallback chain (last claim → last short text → none).
5. **Provider-agnostic tool forcing.** A single helper translates "force any tool" into OpenAI's `tool_choice: "required"`, Anthropic's `{type: "any"}`, and Gemini's `ANY` mode — used to break out of empty-reply loops.

## Architecture

```mermaid
flowchart TB
    Q[User question<br/>+ optional attachment] --> AG[Agent.run_policy]

    subgraph SUB[banna_agent runtime]
      AG --> ST[AgentState<br/>typed: Trace · Evidence · Claim · Budget]
      ST -->|propose| POL{Policy}

      POL --> REACT[ReActPolicy]
      POL --> VR[VerifierRetryPolicy<br/>wraps inner]
      POL --> PL[PlannerReActPolicy]
      POL --> BFS[BFS / DFS / Best-First<br/>over plans]
      POL --> BON[BestOfNPolicy<br/>K trajectories + selector]

      REACT -->|Action| EX[Execute]
      VR -->|FINAL_ANSWER| VERS{Verifiers}
      VERS -- pass --> COMMIT[commit]
      VERS -- fail --> NUDGE[per-verifier nudge<br/>→ THINK feedback]
      NUDGE --> POL

      EX --> TOOLS[ToolRegistry]
      TOOLS --> T1[search]
      TOOLS --> T2[read_url]
      TOOLS --> T3[read_file<br/>+ pdf / xlsx tools]
      TOOLS --> T4[python_sandbox]
      TOOLS --> T5[calculator]
      TOOLS --> T6[run_shell · grep · list_files]

      EX -->|observation| ST
      ST --> BUD[BudgetTracker<br/>steps · repair_steps · wall · tokens · cost]
      BUD -. trip .-> SYN[synthesize_on_exhaustion<br/>forced final_answer]
      SYN --> COMMIT
    end

    subgraph VERIFIERS[Verifiers]
      VF[FormatVerifier<br/>shape / empty answer]
      VA[ArithmeticVerifier<br/>safe-AST recompute]
      VC[CitationVerifier<br/>numeric + Jaccard support]
      VG[CoverageVerifier<br/>claim ↔ evidence]
    end
    VERS --- VF
    VERS --- VA
    VERS --- VC
    VERS --- VG

    COMMIT --> ANS[answer]
```

## Install

```bash
# 1. From PyPI (once published)
pip install banna

# 2. From GitHub directly (no clone, no PyPI required)
pip install git+https://github.com/siavashmonfared/banna.git

# 3. Isolated install with pipx (recommended for CLI use)
pipx install git+https://github.com/siavashmonfared/banna.git

# 4. From a local clone (for development)
git clone https://github.com/siavashmonfared/banna.git
cd banna
pip install -e ".[dev]"
```

Any install path drops a `banna` (and `banna-agent`) executable on your `$PATH`.

## Quickstart

On first run `banna` walks you through a one-time setup — pick a provider, paste an API key (or use a local Ollama model), and save the choice to `~/.config/banna/`. After that, just type `banna`.

```bash
# first run — interactive wizard auto-launches if no config is found
banna
# ● banna — first-run setup
# No LLM provider configured. Let's pick one.
#   1. Ollama       (local, 2 models installed)
#   2. OpenAI       (cloud, paid)
#   3. Anthropic    (cloud, paid)
#   4. Gemini       (cloud, free tier)
# Provider: [1]

# subsequent runs use saved defaults; override any time with flags:
banna --policy verifier_retry --provider openai --model gpt-5-nano

# or run a single GAIA Level-1 question (no REPL)
python -m banna_agent.benchmarks.gaia.runner \
    --policy verifier_retry --provider openai --model gpt-5-nano \
    --level 1 --n 1
```

### Example REPL session

```
$ banna --policy verifier_retry --provider openai --model gpt-5-nano

● banna · v0.1.0   provider=openai   model=gpt-5-nano   policy=verifier_retry

> How many studio albums did Mercedes Sosa release between 2000 and 2009?

  thinking…
  ▸ search(query="Mercedes Sosa discography studio albums 2000-2009")
    ↳ 8 results · evidence_id ev_a3f
  ▸ read_url(url="https://en.wikipedia.org/wiki/Mercedes_Sosa")
    ↳ 12.4 kB · evidence_id ev_91c
  thinking…
  ▸ final_answer(answer="3", evidence_ids=["ev_a3f", "ev_91c"])
  verifiers: format ✓  citation ✓  coverage ✓  arithmetic skip

● banna
  3

  3 steps · 4.7s · 1840→210 tok · $0.0021

> /show trace
  …step-by-step dump of action + observation + meta…

> /exit
```

### Subcommands

```bash
banna init                       # re-run the setup wizard
banna config get                 # show saved defaults
banna config set model gpt-4o    # change a single default
banna providers                  # list configured providers + status
banna providers --validate       # also make a 1-token test call against each
```

The full GAIA validation runner (165 questions across L1/L2/L3) is in `experiments/02_gaia_full/run.py`.

## Failure modes & fixes (the C1–C6 pass)

Diagnosed from a full GAIA validation run on `gpt-5-nano`. Each fix lands as a structural change to the loop, with unit tests pinning the new behavior.

| ID | Failure mode | Root cause | Fix |
|----|--------------|------------|-----|
| C1 | `[empty_reply]` loops eat the step budget | Repair-style THINKs counted as productive steps | New `Budget.repair_steps_used` axis + `max_repair_steps=6`; `meta["repair"]=True` routes off the main counter |
| C2 | Model returns empty content + no tool call | No detection / no escape | After 2 consecutive empties with no evidence, force `tool_choice` to any tool (provider-agnostic) |
| C3 | `pred_answer=null` on budget exhaustion | Loop exits with no commit | `policy.synthesize_on_exhaustion(state)`: one threaded LLM call with forced `final_answer` + cheap fallback chain |
| C4 | L1 step cap too tight (8 steps) | Default budget profile | L1/L2/L3 caps bumped to 12/18/24 |
| C5 | Rich file tools never used on attachments | Hint steered model toward `read_file` even for PDF/XLSX | Extension-routed `_file_hint()` + cheap `_file_summary()` (pypdf page count, openpyxl sheet names, CSV header) |
| C6 | Verifier retries low repair rate | Feedback was generic | Each verifier populates `meta["nudge"]` with a verifier-specific actionable instruction; retry feedback groups by verifier |

## GAIA validation results

> The fixes are landed and tested (562 tests passing in this public repo, 568 in the private superset). The post-fix re-run on GAIA validation is the next thing on the queue.

| Run | Provider · model | Policy | Set | Accuracy | Notes |
|-----|------------------|--------|-----|----------|-------|
| Pre-C1–C6 | OpenAI · `gpt-5-nano` | verifier_retry | GAIA val (165 Q) | **33.9 %** (56 / 165) | $0.94, 7 structural bugs surfaced |
| Post-C1–C6 | OpenAI · `gpt-5-nano` | verifier_retry | GAIA val (165 Q) | _re-run pending_ | target ≥ 40 % |
| Post-C1–C6 | OpenAI · `gpt-5-nano` | best_of_n (K=3) | GAIA val (165 Q) | _re-run pending_ | stretch ≥ 45 % |

(Numbers populate once the re-run completes; CI is wired to gate on a held-out smoke subset.)

## Repo layout

```
src/banna_agent/
├── core/          AgentState, Trace, Action, Budget, EventLog, run_policy
├── llm/           provider-agnostic LLMClient + adapters (anthropic, openai, gemini, ollama, bedrock)
├── tools/         search, read_url, read_file, pdf/xlsx tools, python_sandbox,
│                  calculator, run_shell, grep, list_files, plan, memory, final_answer
├── policies/      react, planner_react, verifier_retry, bfs/dfs/best_first_over_plans, best_of_n
├── verifiers/     arithmetic, citation, coverage, format, command (+ base protocol)
├── benchmarks/    gaia/ (loader, runner, scorer, report)
├── memory/        in_memory_store, jsonl_store, skill_library, embeddings
└── cli/           Rich-based REPL: /policy /budget /show /skills /compact /save /load …
```

Tests live in `tests/` and are organized to mirror `src/`. Run them with:

```bash
pytest -q
```

Current status: **562 passed, 0 failed** on this public branch (no external substrate dependencies).

## License

MIT
