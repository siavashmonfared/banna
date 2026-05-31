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

On first run, `banna` launches a one-time setup wizard: choose a provider, supply an API key (or select a local Ollama model), and the choice is saved to `~/.config/banna/`. Subsequent runs use the saved defaults.

```bash
# First run — the setup wizard launches automatically if no config exists
banna

# Override saved defaults with flags at any time
banna --policy react --provider openai --model gpt-5-nano
```

### Example session

```
$ banna --policy react --provider openai --model gpt-5-nano

● banna · v0.1.2   provider=openai   model=gpt-5-nano   policy=react

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
banna init                       # re-run the setup wizard
banna config get                 # show saved defaults
banna config set model gpt-4o    # change a single default
banna providers                  # list configured providers and status
banna providers --validate       # make a 1-token test call against each
```

## Policies

A `Policy` implements a single method, `propose(state, llm, tools) → Action`; the driver is agnostic to which strategy is running. Two policies are available from the CLI via `--policy` / `/policy`:

| Policy | Description |
|---|---|
| `react` | The core ReAct loop. One LLM call per tick; the model chooses `THINK`, `TOOL_CALL`, or `FINAL_ANSWER`. Fully autonomous, with no human in the loop. This is the benchmarked baseline. |
| `react+` | ReAct extended for interactive, human-in-the-loop use. Adds an `ask_user` clarifying-question affordance, a per-tool permission gate for shell commands, and error-scoping prompt guardrails. `react+` subclasses `react`, so it inherits the entire engine unchanged. |

`react+` is intended for interactive sessions where a person is present to answer clarifying questions and approve tool calls — conditions the GAIA benchmark does not test. On a capacity-constrained model with no human in the loop, its additional machinery measures lower than bare `react` (see results below); the two are kept separate for this reason.

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

Measured on the GAIA validation set (165 questions across Levels 1–3) with `gpt-5-nano`.

| Policy | Overall | L1 | L2 | L3 | Cost |
|---|---|---|---|---|---|
| `react` | **42.4%** (70/165) | 49.1% | 46.5% | 15.4% | ~$0.87 |
| `react+` | 35.8% (59/165) | 45.3% | 38.4% | 7.7% | ~$0.87 |

`react` finishes 92% of tasks through the normal commit path; the remaining 8% trip a budget axis. Median task finishes in 4 productive steps in under a minute. `react+` measures lower here because its interactive affordances (`ask_user`, permission gating) have no human to engage on the benchmark; it is built for interactive use, not autonomous scoring.

Full per-level numbers, exit-reason distributions, operational statistics, reproduction instructions, and an evaluation-limitations section are in [`docs/evals/gaia_validation_report.md`](docs/evals/gaia_validation_report.md). The full validation runner is in `experiments/02_gaia_full/run.py`.

## Repository layout

```
src/banna_agent/
├── core/          AgentState, Trace, Action, Budget, EventLog, run_policy
├── llm/           provider-agnostic LLMClient + adapters (anthropic, openai, gemini, ollama, bedrock)
├── tools/         search, read_url, read_file, pdf/xlsx, python_sandbox,
│                  calculator, run_shell, grep, list_files, plan, memory, final_answer
├── policies/      react, react+ (and supporting research policies)
├── verifiers/     arithmetic, citation, coverage, format, command (+ base protocol)
├── benchmarks/    gaia/ (loader, runner, scorer, report)
├── memory/        in_memory_store, jsonl_store, skill_library, embeddings
└── cli/           Rich-based REPL: /policy /budget /show /skills /compact /save /load …
```

Tests mirror `src/` under `tests/`. Run them with:

```bash
pytest -q
```

Current status on this branch: **818 passed, 3 skipped** (skips require the optional `chromadb` backend or real API keys).

## Limitations

- **Execution isolation is opt-in.** Code-running tools dispatch through a `SandboxBackend`. The default `process` backend runs each call as a host subprocess (real timeout and memory separation, but it inherits the user's filesystem, network, and credentials) — fine for a research harness on your own machine, not for untrusted input. For untrusted input or shared infrastructure, start the agent with `--sandbox=docker` (or `BANNA_SANDBOX=docker`): every `run_python` / `run_shell` call then executes in a throwaway container with no network, a read-only root filesystem, dropped capabilities, and cpu/memory/pid limits. Because the container has no network, a missing third-party package can't be `pip install`-ed at runtime; instead the sandbox builds a derived image in a separate, network-enabled build step (which never runs model code) and re-runs the code against it. Packages on a trusted allowlist install with no prompt; anything else prompts for approval in interactive runs. The allowlist ships with a curated, version-pinned default set (numpy, pandas, scipy, sympy, matplotlib, scikit-learn, pillow, opencv, requests, lxml, openpyxl, …), and you can extend or override it with `banna config packages add <import> <dist==version>` (`banna config packages list` shows both). Override the base image with `--sandbox-image` (or `BANNA_SANDBOX_IMAGE`). Note that the `docker` backend is container-level isolation, not a security boundary against a determined adversary: containers share the host kernel, so a kernel-level exploit can still escape. For genuinely hostile code, run under a stronger runtime — a syscall-filtering sandbox such as gVisor (`runsc`) or a microVM such as Firecracker / Kata — which `--sandbox-image` and the `SandboxBackend` interface are designed to accommodate.
- **Verifiers catch structural failures, not factual ones.** A coherent answer grounded in an incorrect source passes the verifiers and still fails GAIA.
- **Single-agent.** There is no multi-agent delegation or coordination.
- **Synchronous tools.** Tools are `dict → dict`; long-running or streaming tools (headless-browser sessions, multi-turn shells) would require a redesign.
- **GAIA-tuned.** The verifiers, tool registry, and budget defaults target GAIA's distribution. Adapting to other benchmarks would require reworking the verifier set and adding domain tools.

## License

MIT — see [LICENSE](LICENSE).
</content>
