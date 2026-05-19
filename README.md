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

The agent is a **typed transition function** over an `AgentState`. A `Policy` proposes the next `Action`; the driver executes it (LLM call, tool invocation, or terminal commit); the resulting `Observation` is folded back into state; `Verifiers` score any proposed answer; a multi-axis `Budget` decides when to stop.

```
Action = THINK | TOOL_CALL(name, args) | FINAL_ANSWER(answer, evidence_ids)

run_policy : AgentState × Policy × ToolRegistry × LLMClient → AgentState
                ↑                                                ↓
                └────── Policy.propose → execute → observe ──────┘
```

### State

`AgentState` is the single immutable-ish object that every component reads and writes through:

| Field | Type | What it holds |
|---|---|---|
| `trace` | `list[Step]` | Append-only log: `Step(idx, action, observation, wall_s, tokens, meta)`. The replay/audit primitive. |
| `evidence` | `list[Evidence]` | Tool-fetched material with `evidence_id`. Search hits, URL bodies, PDF pages, file reads. Citations point here. |
| `claims` | `list[Claim]` | Propositions the model has asserted, each with `supports: list[evidence_id]` and per-verifier verdicts. |
| `budget` | `Budget` | Multi-axis tracker: `steps`, `repair_steps`, `wall_s`, `tokens`, `cost_usd`. Each axis can trip independently. |
| `metadata` | `dict` | Policy-private state (planner plans, retry counters, frontier candidates, etc.). |

### Tools

Tools are `Callable[[dict], dict]` with a `ToolSpec` schema. Each one writes evidence into `state.evidence` and returns a deterministic dict the policy sees as its next observation.

| Tool | Purpose | When the model picks it |
|---|---|---|
| `search` | Web search (DuckDuckGo / Bing / SerpAPI / YaCy backends) | Open-ended factoid questions, finding sources |
| `read_url` | Fetch + clean HTML to text; HTTP cache aware | After `search` returns a promising link |
| `read_file` | Generic local file read (text, with magic-byte sniffing) | GAIA attachment is a `.txt` / `.csv` / unknown blob |
| `pdf_reader` | pypdf-based text extraction + optional pdfplumber tables | GAIA attachment is a PDF |
| `xlsx_reader` | openpyxl-based sheet/cell access | GAIA attachment is an XLSX |
| `python_sandbox` | Run user-supplied Python in a restricted exec | Multi-step arithmetic, parsing, data manipulation |
| `calculator` | Single-expression safe-AST evaluator | Quick arithmetic where `python_sandbox` is overkill |
| `grep`, `list_files` | Code-task primitives | Repo-shaped questions |
| `run_shell` | Whitelisted shell with risky-command confirm | When file ops or process control is unavoidable |
| `plan` | Records a structured plan into state | Used by `planner_react` and the `*_over_plans` policies |
| `memory` | Reads/writes a persistent skill / fact store | When `--skills` enables the SkillLibrary |
| `final_answer` | The terminal commit; takes `answer`, `reasoning`, `evidence_ids` | Always — plain-text replies are rejected |

### Policies

A `Policy` implements one method: `propose(state, llm, tools) → Action`. The driver doesn't care which strategy is running.

| Policy | Mechanism | Best at | Cost vs. ReAct |
|---|---|---|---|
| `react` | One LLM call per tick; model picks `THINK` / `TOOL_CALL` / `FINAL_ANSWER` | Cheap baseline, latency-sensitive runs | 1× |
| `planner_react` | Planner produces an ordered subtask list once; ReAct executes step-by-step against it | Multi-hop questions where wandering is expensive | ~1.1× (one extra planning call) |
| `verifier_retry` | Wraps any inner policy. On `FINAL_ANSWER`, runs verifiers; on fail, converts to a THINK with per-verifier nudges so the inner policy retries. Up to `max_retries` (default 3). | Reducing the "looked right, was wrong" failure class | ~1.2–1.5× when retries fire |
| `best_of_n` | K independent trajectories of `verifier_retry(react)`, then a selector (`majority_vote` for free or `llm_judge` for one extra call) picks the answer | Hardest tasks; trades $ for accuracy | ~K× |
| `bfs_over_plans` | Propose K candidate plans; expand all by one step; score; keep the best frontier | Search-shaped problems with clear scoring | ~K× |
| `dfs_over_plans` | Propose K plans; fully expand one before moving to the next | When a good plan exists and we want depth, not breadth | ~K× worst-case, often less |
| `best_first_over_plans` | K plans; at each step, expand the highest-scored frontier node | Best-of-both: depth + pruning | ~K× worst-case |

The wrapping is compositional — `best_of_n(verifier_retry(react))` is one line in the constructor. Adding a new policy is ~200 LOC because all of the substrate, verifiers, budgets, and tools come for free.

### Verifiers

Verifiers grade the model's output against ground truth that doesn't require an LLM — that's the point. Each returns a list of `ClaimCheck(claim_id, verdict ∈ {ok, fail, warn, skip}, detail, meta)`. On `fail`, `meta["nudge"]` is a verifier-specific actionable instruction that gets surfaced to the model on the retry tick.

| Verifier | What it catches | How |
|---|---|---|
| `FormatVerifier` | Empty `answer` field, wrong shape (e.g. prose where a number was asked), surrounding quotes/markdown | Schema check + canonical-answer suggestion when the expected shape is known |
| `ArithmeticVerifier` | Wrong math in claims or reasoning ("47 × 83 = 3801" when it's 3901) | Regex out every equality, safe-AST re-evaluate the LHS, compare to asserted RHS within tolerance |
| `CitationVerifier` | Claims whose cited evidence doesn't actually contain the claimed numbers; broken `evidence_id` references | Jaccard token overlap + per-number substring/tolerance check; refetches empty URL evidence through the HTTP cache |
| `CoverageVerifier` | Factual claims with no supporting evidence at all | Structural: every Claim that reads factual must have `supports: [evidence_id, …]` non-empty |
| `CommandVerifier` (optional) | Code-task failures: failing tests, type errors, lint errors, build errors | Shells out to `pytest` / `mypy` / `ruff` in the sandbox; off by default for QA runs |

The verifier suite is the research signal — the failures it catches are exactly the silent-but-wrong answers that pure ReAct accepts.

### Budget

`Budget` has five independently-tripping axes. The motivation: stuck-loop behavior shouldn't burn budget meant for productive work.

| Axis | What it bounds | Tripped when |
|---|---|---|
| `steps_used` / `max_steps` | Productive ticks (`meta["repair"]` is not set) | Hard cap per task (12 / 18 / 24 on GAIA L1 / L2 / L3) |
| `repair_steps_used` / `max_repair_steps` | Empty-reply, verifier-retry, and forced-tool-choice escape ticks | Stuck-loop protection; 6 by default |
| `wall_s` | Wall-clock seconds since `t0` | Latency cap |
| `tokens_in + tokens_out` | Cumulative LLM tokens | Cost proxy when explicit pricing is unknown |
| `cost_usd` | Provider-priced cost | Hard $ cap per task |

When any axis trips, the driver calls `policy.synthesize_on_exhaustion(state)` — one last forced-`final_answer` LLM call with a 15s timeout and a cheap fallback chain (last claim → last short text → none) — so the run commits something instead of returning `null`.

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
