# banna

A provider-agnostic reasoning agent built from scratch around a typed state substrate and a verifier-guided control loop. It is designed to study where ReAct-style agents fail on the [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) benchmark and to address those failures structurally rather than through prompt patches.

The core has no agent-framework dependencies (no LangChain, LlamaIndex, or smolagents). The reasoning loop is a typed transition function over `(state, action, observation) → state'`, and each control strategy is a small `Policy` implementation over that same substrate.

## Installation

Requires Python 3.10+.

```bash
# From PyPI
pip install banna

# Or directly from GitHub
pip install git+https://github.com/siavashmonfared/banna.git

# Isolated CLI install
pipx install git+https://github.com/siavashmonfared/banna.git

# From a local clone (development)
git clone https://github.com/siavashmonfared/banna.git
cd banna
pip install -e ".[dev]"
```

Every install path provides a `banna` (and `banna-agent`) executable on your `PATH`.

## Quickstart

Every launch of `banna` opens a session-setup dashboard (Textual TUI), pre-filled from your saved defaults in `~/.config/banna/config.toml` — press `s` to start with what's shown, or arrow around to change things first:

```
  Provider      anthropic          ✓ ANTHROPIC_API_KEY (shell env)
  Model         claude-sonnet-5
  Policy        react+             react + evidence recall & coverage pressure
  Budget        15 steps · 300s · $5.00 · ∞ tokens
  Theme         solarized-light
  Sandbox       process
  Temperature   0.7
  Skills        off
  n_candidates  3

  ↑↓ move · ←/→ toggle · ↵ edit · s start · d save as default & start · q quit
```

API keys are searched in your shell env, `./.env`, and `~/.config/banna/.env`; if a provider has no key you can paste one right there (validated with a 1-token call, saved mode 0600) or switch to a local Ollama model. Model pickers list correct, current model ids per provider; Ollama models are listed live and probed for tool-calling support. Numeric fields (budget, temperature) step with ↑/↓ or take typed values.

```bash
banna                 # launch — dashboard first, then the REPL
banna --no-setup      # skip the dashboard, start straight into the REPL

# Flags still override any saved default
banna --policy react --provider openai --model gpt-5-nano
```

### Example session

```
$ banna --policy react --provider openai --model gpt-5-nano

● banna · v0.4.0   provider=openai   model=gpt-5-nano   policy=react

> How many studio albums did Mercedes Sosa release between 2000 and 2009?

  thinking…
  ▸ search(query="Mercedes Sosa discography studio albums 2000-2009")
    ↳ 8 results · evidence_id ev_a3f
  ▸ read_url(url="https://en.wikipedia.org/wiki/Mercedes_Sosa")
    ↳ 12.4 kB · evidence_id ev_91c
  ▸ final_answer(answer="3", evidence_ids=["ev_a3f", "ev_91c"])
  verifiers: format ✓  citation ✓  coverage ✓  arithmetic skip

● banna
  3

  3 steps · 4.7s · 1840→210 tok · $0.0021
```

### Subcommands

```bash
banna init                       # re-run the plain-text setup wizard (non-TTY fallback)
banna config get                 # show saved defaults
banna config set model gpt-4o    # change a single default
banna providers                  # list configured providers and status
banna providers --validate       # make a 1-token test call against each
```

## MCP servers

Use tools served by an external [MCP](https://modelcontextprotocol.io) server as if they were native tools. Both stdio (local subprocess) and HTTP/SSE (remote) transports are supported.

```bash
# register a local stdio server (its tools appear namespaced, e.g. collab.collab_start)
banna config mcp add collab -- python3 /path/to/server.py
# register a remote HTTP server
banna config mcp add remote --http https://example.com/mcp
banna config mcp list            # show configured servers
banna config mcp remove collab   # drop one
```

Inside the REPL, `/mcp` shows every configured server with a live connection status (`● connected` / `✗ failed` + the error) and the namespaced tools it contributed. banna also ships a standard-server catalog: `/mcp install collab` fetches [agentic-tools](https://github.com/siavashmonfared/agentic-tools) (a shared brainstorm thread for multi-agent workflows), provisions its dependencies in a dedicated venv under `~/.config/banna/mcp-servers/`, registers it, and connects — no manual setup. `/mcp reload` reconnects after config changes.

Servers connect when the REPL starts and shut down on exit; a server that fails to start is reported and skipped rather than crashing the session. MCP tools run external code, so they go through the same per-call permission prompt as `run_shell`.

## Sessions & resume

Every conversation is auto-saved to `~/.config/banna/sessions/` as it happens, so you can pick up where you left off.

```bash
banna --resume          # pick from a list of recent sessions
banna --resume last     # resume the most recent
banna --resume <id>     # resume a specific session
```

Inside the REPL, `/sessions` lists them and `/resume [id|last]` switches. The explicit `/save <path>` and `/load <path>` still work for hand-managed transcripts.

## Memory

A persistent memory store (`~/.config/myagent/memory.jsonl`) survives across sessions. The agent can write and search it via the `memory` tool, and relevant entries are **auto-recalled** into context on each turn (gated by topical overlap so unrelated facts don't leak in).

## Trace viewer

Turn any run's JSONL event log into a self-contained HTML report — every step's reasoning, tool calls and results, parallel batches, and the final answer, in one file with no external assets.

```bash
banna trace view runs/<id>/logs/<task>.jsonl        # writes <task>.html
banna trace view <log.jsonl> -o report.html         # custom output path
```

## Policies

A `Policy` implements a single method, `propose(state, llm, tools) → Action`; the driver is agnostic to which strategy is running. Three policies are available from the CLI via `--policy` / `/policy` (`react+` is the default):

| Policy | Description |
|---|---|
| `react` | The core ReAct loop. One LLM call per tick; the model chooses `THINK`, `TOOL_CALL`, or `FINAL_ANSWER`. Fully autonomous, with no human in the loop. This is the benchmarked baseline. |
| `react+` *(default)* | ReAct extended for interactive, human-in-the-loop use. Adds an `ask_user` clarifying-question affordance, a per-tool permission gate for shell commands, and error-scoping prompt guardrails. `react+` subclasses `react`, so it inherits the entire engine unchanged. |
| `verifier_retry` | A self-checking wrapper. It delegates to an inner policy (`react` by default) and, when that policy proposes a `FINAL_ANSWER`, re-checks it against the four intrinsic verifiers (`arithmetic`, `citation`, `format`, `coverage`); on any failure it discards the answer and feeds the verifier critique back as a `THINK`, forcing a revision (up to `max_retries`). Orthogonal to `react+` — it composes with any inner policy. See the [ablation](docs/evals/ablation.md) for where this helps and where it hurts. |

`react+` is the default because it is built for interactive sessions, where a person is present to answer clarifying questions and approve tool calls. `react` and `verifier_retry` are both fully autonomous (no human in the loop), which is what the GAIA benchmark requires — so the published numbers below are for those two.

## Architecture

The agent is a typed transition function over an `AgentState`. A `Policy` proposes the next `Action`; the driver executes it (LLM call, tool invocation, or terminal commit); the resulting `Observation` is folded back into state; `Verifiers` score any proposed answer; a multi-axis `Budget` decides when to stop.

```
Action = THINK | TOOL_CALL(name, args) | ASK_USER(question) | FINAL_ANSWER(answer, evidence_ids)

run_policy : AgentState × Policy × ToolRegistry × LLMClient → AgentState
                ↑                                                ↓
                └────── Policy.propose → execute → observe ──────┘
```

### State

`AgentState` is the single object every component reads and writes through:

| Field | Type | Contents |
|---|---|---|
| `trace` | `list[Step]` | Append-only log of `Step(idx, action, observation, wall_s, tokens, meta)`. The replay/audit primitive. |
| `evidence` | `list[Evidence]` | Tool-fetched material with an `evidence_id`: search hits, URL bodies, PDF pages, file reads. Citations point here. |
| `claims` | `list[Claim]` | Propositions the model has asserted, each with `supports: list[evidence_id]` and per-verifier verdicts. |
| `budget` | `Budget` | Multi-axis tracker: `steps`, `repair_steps`, `wall_s`, `tokens`, `cost_usd`. Each axis trips independently. |
| `metadata` | `dict` | Policy-private state (plans, retry counters, user replies, etc.). |

### Tools

Tools are `Callable[[dict], dict]` with a `ToolSpec` schema. Each writes evidence into `state.evidence` and returns a deterministic dict that the policy reads as its next observation.

| Tool | Purpose |
|---|---|
| `search` | Web search (DuckDuckGo / Bing / SerpAPI / YaCy backends) |
| `read_url` | Fetch and clean HTML to text; HTTP-cache aware |
| `read_file` | Generic local file read with magic-byte sniffing |
| `pdf_reader` | pypdf text extraction with optional pdfplumber tables |
| `xlsx_reader` | openpyxl sheet/cell access |
| `python_sandbox` | Run model-emitted Python in a restricted namespace |
| `calculator` | Single-expression safe-AST evaluator |
| `grep`, `list_files` | Code- and repo-task primitives |
| `run_shell` | Allowlisted shell; gated by a permission prompt under `react+` |
| `plan` | Records a structured plan into state |
| `memory` | Reads/writes a persistent skill and fact store |
| `final_answer` | Terminal commit; takes `answer`, `reasoning`, `evidence_ids` |

### Verifiers

Verifiers grade output against checks that do not require an LLM. Each returns a list of `ClaimCheck(claim_id, verdict ∈ {ok, fail, warn, skip}, detail, meta)`. On a `fail`, `meta["nudge"]` provides an actionable instruction surfaced to the model on a retry tick.

| Verifier | Catches |
|---|---|
| `FormatVerifier` | Empty or malformed `answer` field |
| `ArithmeticVerifier` | Wrong math in claims or reasoning (re-evaluates each equality with a safe AST) |
| `CitationVerifier` | Claims whose cited evidence does not contain the claimed values; broken `evidence_id` references |
| `CoverageVerifier` | Factual claims with no supporting evidence |
| `CommandVerifier` (optional) | Code-task failures via `pytest` / `mypy` / `ruff`; off by default |

`CitationVerifier` checks whether a claim is defensible against the evidence it cited, not whether that evidence is factually correct.

### Budget

`Budget` has five independently-tripping axes so that stuck-loop behavior does not consume budget meant for productive work:

| Axis | Bounds |
|---|---|
| `steps_used` / `max_steps` | Productive ticks |
| `repair_steps_used` / `max_repair_steps` | Empty-reply, retry, and forced-tool-choice escape ticks |
| `wall_s` | Wall-clock time (excludes time paused on an interactive prompt) |
| `tokens_in + tokens_out` | Cumulative LLM tokens |
| `cost_usd` | Provider-priced cost |

When any axis trips without a committed answer, the driver calls `policy.synthesize_on_exhaustion(state)` — a time-bounded forced-`final_answer` call with a cheap fallback chain (last claim → last short text → none) — so the run commits something rather than returning `null`.

## GAIA validation results

Measured with the bare `react` engine: a full run on `gpt-5-nano` (165 questions across Levels 1–3), plus a cross-model L3-only probe on `claude-sonnet-4-5`.

| Run | Overall | L1 | L2 | L3 | Cost |
|---|---|---|---|---|---|
| `react` · `gpt-5-nano` (full, 165 Q) | **42.4%** (70/165) | 49.1% | 46.5% | 15.4% | ~$0.87 |
| `react` · `claude-sonnet-4-5` (L3 only, 26 Q) | — | — | — | **26.9%** (7/26) | ~$20.75 |

On `gpt-5-nano`, `react` finishes 92% of tasks through the normal commit path; the remaining 8% trip a budget axis. Median task finishes in 4 productive steps in under a minute. The Level-3 gap is the chained-reasoning regime: swapping in `claude-sonnet-4-5` (same policy, same tools) nearly doubles L3 accuracy (26.9% vs 15.4%) at ~24× the cost — a single-set probe consistent with model capacity, not the scaffolding, being the L3 bottleneck.

Full per-level numbers, exit-reason distributions, operational statistics, reproduction instructions, and an evaluation-limitations section are in [`docs/evals/gaia_validation_report.md`](docs/evals/gaia_validation_report.md). The full validation runner is in `experiments/02_gaia_full/run.py`.

### Capacity × verification (2×2 ablation)

Does intrinsic self-verification help? It depends on the model. This 2×2 crosses **model capacity** (`gpt-5-nano` vs `gpt-5-mini`) with **verification** (bare `react` vs `verifier_retry(react)` over the four intrinsic verifiers), all on the full 165-task GAIA validation set:

| Model | `react` | `verifier_retry(react)` | Δ (verification) |
|---|---|---|---|
| `gpt-5-nano` | 42.4% (70/165) | 37.6% (62/165) | **−4.8pp** |
| `gpt-5-mini` | 51.5% (85/165) | 53.9% (89/165) | **+2.4pp** |

**The sign flips:** intrinsic verification is net-negative on the capacity-limited model and net-positive on the stronger one (capacity × verification interaction = +7.3pp). The mechanism is the verifier's false-positive rejection rate — it breaks already-correct answers 30% of the time on `gpt-5-nano` but only 14% on `gpt-5-mini`. Both single-model effects are directional, not significant at n=165; the false-positive rate is the directly-measured number that carries the result. Full per-level numbers, the McNemar tests, cost, and reproduction are in [`docs/evals/ablation.md`](docs/evals/ablation.md).

> **Why the table reports `react` / `verifier_retry`, not the shipped `react+` defaults.** The benchmark runs the autonomous policies, because `react+`'s interactive affordances (`ask_user`, the permission gate) have no human to engage in batch and are therefore inert. This is measured, not assumed: the shipped defaults reproduce the `gpt-5-nano` row to within one task — `react+` scores **43.0%** (71/165) vs `react`'s 42.4%, and `react+verify` (= `verifier_retry(react+)`) scores **39.4%** (65/165) vs `verifier_retry(react)`'s 37.6%, same full 165-task set. The `+` policies are what you run interactively; the bare policies are what the numbers above measure, and on GAIA they are the same thing.

## Repository layout

```
src/banna_agent/
├── core/          AgentState, Trace, Action, Budget, EventLog, run_policy
├── llm/           provider-agnostic LLMClient + adapters (anthropic, openai, gemini, ollama, bedrock)
├── tools/         search, read_url, read_file, pdf/xlsx, python_sandbox,
│                  calculator, run_shell, grep, list_files, plan, memory, final_answer
│   └── mcp/       MCP client (stdio + HTTP/SSE) + JsonTool bridge
├── policies/      react (engine, benchmarked) + react+ (interactive, default) + verifier_retry (self-checking wrapper)
├── verifiers/     arithmetic, citation, coverage, format, command (+ base protocol)
├── benchmarks/    gaia/ (loader, runner, scorer, report)
├── memory/        in_memory_store, jsonl_store, skill_library, embeddings
├── trace/         render a run's JSONL event log to static HTML
└── cli/           Rich-based REPL: /policy /budget /show /sessions /resume /save /load …
```

Tests mirror `src/` under `tests/`. Run them with:

```bash
pytest -q
```

Current status on this branch: **891 passed, 3 skipped** (skips require the optional `chromadb` backend or real API keys).

## Limitations

- **Execution isolation is opt-in.** Code-running tools dispatch through a `SandboxBackend`. The default `process` backend runs each call as a host subprocess (real timeout and memory separation, but it inherits the user's filesystem, network, and credentials) — fine for a research harness on your own machine, not for untrusted input. For untrusted input or shared infrastructure, start the agent with `--sandbox=docker` (or `BANNA_SANDBOX=docker`): every `run_python` / `run_shell` call then executes in a throwaway container with no network, a read-only root filesystem, dropped capabilities, and cpu/memory/pid limits. Because the container has no network, a missing third-party package can't be `pip install`-ed at runtime; instead the sandbox builds a derived image in a separate, network-enabled build step (which never runs model code) and re-runs the code against it. Packages on a trusted allowlist install with no prompt; anything else prompts for approval in interactive runs. The allowlist ships with a curated, version-pinned default set (numpy, pandas, scipy, sympy, matplotlib, scikit-learn, pillow, opencv, requests, lxml, openpyxl, …), and you can extend or override it with `banna config packages add <import> <dist==version>` (`banna config packages list` shows both). Override the base image with `--sandbox-image` (or `BANNA_SANDBOX_IMAGE`). Note that the `docker` backend is container-level isolation, not a security boundary against a determined adversary: containers share the host kernel, so a kernel-level exploit can still escape. For genuinely hostile code, run under a stronger runtime — a syscall-filtering sandbox such as gVisor (`runsc`) or a microVM such as Firecracker / Kata — which `--sandbox-image` and the `SandboxBackend` interface are designed to accommodate.
- **Verifiers catch structural failures, not factual ones.** A coherent answer grounded in an incorrect source passes the verifiers and still fails GAIA.
- **Single-agent.** There is no multi-agent delegation or coordination.
- **Synchronous tools.** Tools are `dict → dict`; long-running or streaming tools (headless-browser sessions, multi-turn shells) would require a redesign.
- **GAIA-tuned.** The verifiers, tool registry, and budget defaults target GAIA's distribution. Adapting to other benchmarks would require reworking the verifier set and adding domain tools.

## License

MIT — see [LICENSE](LICENSE).
</content>
