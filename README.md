# banna

A from-scratch, provider-agnostic reasoning agent with a **typed state substrate** and a **verifier-guided** loop. Built to study where ReAct-style agents fail on the **GAIA** benchmark and to fix those failures structurally — not with prompt patches.

No LangChain, no LlamaIndex, no smolagents in the core. The reasoning loop is a typed transition function over `(state, action, observation) → state'`; each control strategy (ReAct, verifier-retry, planner-ReAct, BFS/DFS/best-first-over-plans, best-of-N) is a ~200 LOC `Policy` implementation over that same substrate. The public CLI currently exposes only the policies that have a benchmark run behind them — **`react` today** (42.4 % on GAIA val, the best validated public result so far); the wrapper + extra inner policies are gated until their ablation rows land.

## What's interesting about this repo

1. **Instrumented GAIA validation.** A full-set run on `gpt-5-nano` is logged end-to-end (per-task event traces, aggregate `results.jsonl`), with per-level accuracy, exit-reason distribution, and operational stats. See [`docs/evals/gaia_validation_report.md`](docs/evals/gaia_validation_report.md).
2. **Multi-axis budget tracker** that separates *productive* steps from *repair* steps. A model stuck in an empty-reply loop no longer burns its productive-step budget; instead it trips a separate `max_repair_steps` axis with a forced tool-choice escape.
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

Policies are introduced to the public build as their benchmark validation lands in [`docs/evals/`](docs/evals/). The repository ships the implementations for the rest of the policy family, but only validated policies are selectable via `--policy` / `/policy`. This is deliberate: the public build only advertises what's been measured.

| Policy | Status | Mechanism | Cost vs. ReAct |
|---|---|---|---|
| `react` | **Validated — 42.4 % (70 / 165) on GAIA validation with `gpt-5-nano`. See [`docs/evals/gaia_validation_report.md`](docs/evals/gaia_validation_report.md).** | One LLM call per tick; model picks `THINK` / `TOOL_CALL` / `FINAL_ANSWER` | 1× |
| `verifier_retry` (wraps any inner policy) | Code present; not exposed in the CLI until its validation run lands. | On `FINAL_ANSWER`, runs verifiers; on fail, converts to a THINK with per-verifier nudges so the inner policy retries. | ~1.2–1.5× when retries fire |
| `planner_react` | Code present; not exposed | Planner produces an ordered subtask list once; ReAct executes step-by-step against it | ~1.1× |
| `best_of_n` | Code present; not exposed | K independent trajectories, selector (`majority_vote` / `llm_judge`) picks the answer | ~K× |
| `bfs_over_plans` / `dfs_over_plans` / `best_first_over_plans` | Code present; not exposed | Plan-frontier search variants over K candidate plans | ~K× worst-case |

The wrapping is compositional — `best_of_n(verifier_retry(react))` is one line in the constructor. Adding a new policy is ~200 LOC because the substrate, verifiers, budgets, and tools come for free. The CLI gate (only `react` selectable today) is independent of the code: every policy ships, only the validated one is advertised.

### Verifiers

Verifiers grade the model's output against ground truth that doesn't require an LLM — that's the point. Each returns a list of `ClaimCheck(claim_id, verdict ∈ {ok, fail, warn, skip}, detail, meta)`. On `fail`, `meta["nudge"]` is a verifier-specific actionable instruction that gets surfaced to the model on the retry tick.

| Verifier | What it catches | How |
|---|---|---|
| `FormatVerifier` | Empty `answer` field (returns an actionable `nudge` so the retry tick re-emits the answer in the right field) | Direct check on the proposed answer; a programmatic, GAIA-metadata-driven shape/canonical check is staged for a follow-up phase but not in the public build today |
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
banna --policy react --provider openai --model gpt-5-nano

# or run a single GAIA Level-1 question (no REPL)
python -m banna_agent.benchmarks.gaia.runner \
    --policy react --provider openai --model gpt-5-nano \
    --level 1 --n 1
```

### Example REPL session

```
$ banna --policy react --provider openai --model gpt-5-nano

● banna · v0.1.1   provider=openai   model=gpt-5-nano   policy=react

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

## Reliability engineering

Five structural decisions in the loop, each in response to a specific failure mode observed during validation, each pinned by unit tests.

| Decision | What changed | Why |
|---|---|---|
| Multi-axis budget tracker | `Budget` separates productive steps from repair steps. Actions tagged `meta["repair"]=True` (empty-reply fallbacks, commit-required nudges, retry feedback) tick a dedicated `repair_steps_used` axis with its own cap. | A model stuck in an empty-reply loop no longer drains its productive-step budget; the two failure modes get distinct exit codes. |
| Empty-reply detection + forced tool use | After two consecutive empty replies with no evidence accumulated, the policy forces the provider's "any tool" mode (OpenAI `"required"`, Anthropic `{"type":"any"}`, Gemini `ANY`). After two commit-required nudges, the policy bails to `final_answer`. | Empty-reply loops were a top cause of zero-tool / zero-evidence finishes on small models. |
| Budget-exhaustion synthesis | When any budget axis trips without a committed answer, the policy calls a 15-second-bounded LLM with `tool_choice=final_answer` forced, falling through to a cheap chain (last claim → last short preceding text → none). | Eliminates the "ran out of budget, returned `null`" failure class. |
| Calibrated default budget profile | L1 / L2 / L3 step caps at 12 / 18 / 24; wall cap at 240 s; repair-step cap independent at 6. | Earlier defaults exhausted steps on L1 and tripped wall on long tails. |
| Attachment-aware tool hints | System prompt extended with extension-routed tool affordances (PDF / XLSX / CSV / image), plus a cheap pre-introspection summary (page count, sheet names, header line) injected into the question. | Without this, the model defaulted to `read_file` on every attachment, including binaries; rich PDF / XLSX tools were rarely picked. |

## GAIA validation results

**`react` on `gpt-5-nano` scores 42.4 %** (70 / 165) on GAIA validation. Total cost: **$0.87**. See [`docs/evals/gaia_validation_report.md`](docs/evals/gaia_validation_report.md) for per-level numbers, exit distributions, operational stats, reproduction instructions, and an honest *"limitations of this evaluation"* section.

| Level | n | Accuracy |
|---|---|---|
| L1 | 53 | 49.1 % |
| L2 | 86 | 46.5 % |
| L3 | 26 | 15.4 % |

92 % of tasks finish through the normal commit path; the remaining 8 % trip a budget axis (wall or repair-step). Median task finishes in 4 productive steps under a minute.

Further policies (planner-based decomposition, verifier-retry variants, best-of-N selection) ship in `src/banna_agent/policies/` but are not exposed in the CLI until each has a validation run behind it. Adding a new ablation is a YAML config, not a code change.

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

Current status: **778 passed, 3 skipped, 0 failed** on this public branch (skips are optional `chromadb` backend + tool-conformance tests that need real API keys).

## Limitations

Honest self-assessment — what this codebase is *not*:

- **No OS-level isolation for code execution.** `python_sandbox` runs untrusted model-emitted Python via `exec()` against a restricted namespace; `run_shell` uses a regex allowlist on the command string. Both run in the same OS process as the agent and inherit the user's filesystem, network, and credentials. If the namespace filter has a hole, a malicious or buggy script can read `~/.ssh/`, write any file the user can write, or open outbound network connections. **Acceptable for a research harness on the developer's own machine; not safe for executing untrusted user input or running unattended on shared infrastructure.** Tracked follow-up: Docker-backed `python_sandbox` with `--sandbox=docker|process|none` and a curated base image. Expected in a future minor release.
- **The verifier suite catches structural failures, not factual ones.** `CitationVerifier` checks whether the model's claim is *defensible against the evidence it cited*, not whether the cited evidence is *actually true*. A coherent answer grounded in a wrong Wikipedia revision passes the verifier and fails GAIA.
- **No multi-agent coordination.** This is a single-agent loop. Tasks that need delegation across specialized agents (e.g. a planner + a critic + an executor as separate processes) are out of scope.
- **No streaming / interactive tool use.** Tools are synchronous `dict → dict`. Long-running tools (e.g. headless-browser sessions, multi-turn shells) would need a redesign.
- **GAIA-flavored.** The verifiers, tool registry, and budget defaults are tuned for the GAIA benchmark's distribution (open-domain factoids + attachments). Adapting to other benchmarks (e.g. SWE-Bench, MMLU-Pro) would mean reworking the verifier set and adding domain tools.

## License

MIT — see [LICENSE](LICENSE).
