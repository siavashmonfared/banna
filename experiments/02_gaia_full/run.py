"""Experiment 02 — full Phase-2/3 toolset on GAIA.

The same skeleton as experiment 01, but registers the Phase-2 browser
suite, Phase-3 PDF/XLSX/image/audio readers, and honours the HTTP
cache so record/replay ablation is one flag away. Drops the `read_url`
+ Phase-0 shell + python sandbox tools when running GAIA-style QA, on
the principle that fewer redundant tools = less LLM confusion.

Usage (from repo root, env preloaded):

    PYTHONPATH=src python experiments/02_gaia_full/run.py \\
        --policy react --level 1 --n 20 \\
        --cache-mode replay_or_record --cache-path .cache/http

After the run, the script prints the resume table by calling
``benchmarks.gaia.report.aggregate`` over the produced results.jsonl.

Flags:
    --provider / --model        — same as experiment 01.
    --policy                    — react | planner_react | bfs_over_plans |
                                  dfs_over_plans | best_first_over_plans |
                                  verifier_retry | best_of_n
    --level / --n               — GAIA level + max tasks.
    --dataset / --jsonl         — gaia (HF) or jsonl with --jsonl path.
    --out-dir                   — defaults to runs/<ts>_<policy>_L<level>.
    --cache-mode                — off | record | replay | replay_or_record.
                                  Sets BANNA_HTTP_CACHE before any tool
                                  is built. Default off.
    --cache-path                — directory for the on-disk HTTP cache
                                  (default .cache/http under repo root).
    --enable-vision             — register the extract_image tool (LLM-cost).
    --enable-transcribe         — register transcribe_audio (uses an OpenAI
                                  client built from OPENAI_API_KEY).
    --enable-browser            — register the six browser_* actions
                                  (default on; --no-browser disables).
    --enable-pdf-tools          — register pdf_open/read_page/find/read_tables
                                  (default on).
    --enable-xlsx-tools         — register xlsx_list_sheets/describe/
                                  read_range/find (default on).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_dotenv() -> None:
    """Populate os.environ from REPO_ROOT/.env if present. Lines look like
    KEY=VALUE; surrounding quotes and "export " prefix are tolerated. Does
    not overwrite values already in the environment."""
    p = REPO_ROOT / ".env"
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-2/3 GAIA runner.")
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--policy",
        choices=["react", "planner_react", "bfs_over_plans",
                 "dfs_over_plans", "best_first_over_plans",
                 "verifier_retry", "best_of_n"],
        default="react",
    )
    ap.add_argument("--n-candidates", type=int, default=3)
    ap.add_argument("--n-trajectories", type=int, default=3,
                    help="Number of independent trajectories for --policy best_of_n.")
    ap.add_argument("--selector", default="majority_vote",
                    choices=["majority_vote", "llm_judge"],
                    help="Best-of-N selector. majority_vote is free; "
                         "llm_judge adds one short LLM call per task.")
    ap.add_argument("--level", type=int, default=1,
                    help="GAIA level (1/2/3). Ignored when --all-levels is set.")
    ap.add_argument("--all-levels", action="store_true",
                    help="Load all three levels (full validation set).")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--dataset", choices=["gaia", "jsonl"], default="gaia")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--out-dir", default=None)
    # HTTP cache wiring.
    ap.add_argument(
        "--cache-mode",
        choices=["off", "record", "replay", "replay_or_record"],
        default="off",
    )
    ap.add_argument("--cache-path", default=str(REPO_ROOT / ".cache" / "http"))
    # Tool selection.
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-pdf-tools", action="store_true")
    ap.add_argument("--no-xlsx-tools", action="store_true")
    ap.add_argument("--enable-vision", action="store_true",
                    help="Register extract_image tool (LLM-cost per call).")
    ap.add_argument("--enable-transcribe", action="store_true",
                    help="Register transcribe_audio (requires OPENAI_API_KEY).")
    ap.add_argument("--budget-steps", type=int, default=None,
                    help="Override per-task max steps (default level-based: "
                         "L1=8, L2=14, L3=20).")
    ap.add_argument("--budget-wall-s", type=float, default=None,
                    help="Override per-task wall-time cap in seconds "
                         "(default 120).")
    ap.add_argument("--budget-tokens", type=int, default=None,
                    help="Per-task max total tokens (default: unlimited).")
    ap.add_argument("--budget-cost", type=float, default=None,
                    help="Per-task max USD cost (default: unlimited).")
    ap.add_argument("--max-total-cost", type=float, default=None,
                    help="Hard cap on cumulative USD spend across ALL tasks. "
                         "The run stops early once the running total reaches "
                         "this (whole-run kill switch, separate from the "
                         "per-task --budget-cost axis).")
    args = ap.parse_args()

    # ---- HTTP cache: set BEFORE building any tools that fetch ----
    if args.cache_mode != "off":
        Path(args.cache_path).mkdir(parents=True, exist_ok=True)
        os.environ["BANNA_HTTP_CACHE"] = f"{args.cache_mode}:{args.cache_path}"

    # Imports must happen after the env var is set so the http_cache
    # singleton lazy-initialises in the right mode.
    from banna_agent.benchmarks.gaia.loader import (
        GAIATask, load_gaia, load_gaia_from_jsonl,
    )
    from banna_agent.benchmarks.gaia.report import aggregate
    from banna_agent.benchmarks.gaia.runner import run_gaia
    from banna_agent.llm.registry import make_client
    from banna_agent.tools.base import ToolRegistry
    from banna_agent.tools.calculator import make_calculator_tool
    from banna_agent.tools.file_reader import make_file_reader_tool
    from banna_agent.tools.final_answer import make_final_answer_tool
    from banna_agent.tools.grep import make_grep_tool
    from banna_agent.tools.list_files import make_list_files_tool
    from banna_agent.tools.plan import make_plan_tool
    from banna_agent.tools.python_sandbox import make_python_sandbox_tool
    from banna_agent.tools.search import make_search_tool
    from banna_agent.tools.url_reader import make_url_reader_tool

    # ---- tasks ----
    if args.dataset == "jsonl":
        if not args.jsonl:
            ap.error("--dataset=jsonl requires --jsonl")
        tasks = load_gaia_from_jsonl(args.jsonl)
        if args.level:
            tasks = [t for t in tasks if t.level == args.level]
        if args.n is not None:
            tasks = tasks[: args.n]
    else:
        level_filter = {1, 2, 3} if args.all_levels else {args.level}
        tasks = load_gaia(levels=level_filter, limit=args.n)

    if not tasks:
        print("no tasks matched", file=sys.stderr)
        return 2

    # ---- out dir ----
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).parent / "runs"
        / f"{ts}_{args.policy}_L{args.level}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- llm ----
    llm = make_client(args.provider, model=args.model)

    # ---- tool registry ----
    tool_list = [
        make_search_tool(),
        make_url_reader_tool(),
        make_file_reader_tool(),
        make_list_files_tool(),
        make_grep_tool(),
        make_calculator_tool(),
        make_python_sandbox_tool(),
        make_plan_tool(),
        make_final_answer_tool(),
    ]
    if not args.no_browser:
        from banna_agent.tools.browser import make_browser_tools
        tool_list.extend(make_browser_tools())
    if not args.no_pdf_tools:
        from banna_agent.tools.pdf_reader import make_pdf_tools
        # Pass the LLM so scanned/image PDF pages get a vision-OCR fallback
        # and the pdf_read_page_visual tool is registered.
        tool_list.extend(make_pdf_tools(llm))
    if not args.no_xlsx_tools:
        from banna_agent.tools.xlsx_reader import make_xlsx_tools
        tool_list.extend(make_xlsx_tools())
    if args.enable_vision:
        from banna_agent.tools.image_extract import make_image_extract_tool
        tool_list.append(make_image_extract_tool(llm))
    if args.enable_transcribe:
        # Build an OpenAI SDK client from env. Fails clean if no key.
        try:
            import openai
            oai = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except Exception as exc:
            print(f"WARN: --enable-transcribe but couldn't build OpenAI client: {exc}",
                  file=sys.stderr)
            oai = None
        from banna_agent.tools.audio_transcribe import make_transcribe_tool
        tool_list.append(make_transcribe_tool(openai_client=oai))

    tools = ToolRegistry(tool_list)
    policy = _build_policy(args)

    # Optional budget override (per-task caps). The defaults in
    # gaia/runner.py are L1=8 / L2=14 / L3=20 steps, 120 s wall, no
    # token/cost cap. Each axis is opt-in independently — passing
    # --budget-cost without --budget-steps still picks up the level-
    # appropriate step cap.
    budget_factory = None
    if any(
        getattr(args, f) is not None
        for f in ("budget_steps", "budget_wall_s", "budget_tokens", "budget_cost")
    ):
        from banna_agent.core.types import Budget
        default_steps = {1: 8, 2: 14, 3: 20}
        def budget_factory(task):
            steps = args.budget_steps if args.budget_steps is not None else default_steps.get(task.level, 12)
            wall = args.budget_wall_s if args.budget_wall_s is not None else 120.0
            return Budget(
                max_steps=steps,
                max_wall_s=wall,
                max_tokens_total=args.budget_tokens,
                max_cost_usd=args.budget_cost,
            )

    print("\n== GAIA run ==")
    print(f"  policy     : {args.policy}")
    print(f"  provider   : {args.provider}")
    print(f"  model      : {args.model or '(provider default)'}")
    print(f"  level      : {'all (1,2,3)' if args.all_levels else args.level}")
    print(f"  n_tasks    : {len(tasks)}")
    print(f"  cache_mode : {args.cache_mode} ({args.cache_path})")
    if budget_factory is not None:
        print(
            f"  budget     : steps={args.budget_steps or '(default)'} "
            f"wall_s={args.budget_wall_s or '(default)'} "
            f"tokens={args.budget_tokens if args.budget_tokens is not None else 'unlimited'} "
            f"cost_usd={'$' + format(args.budget_cost, '.2f') if args.budget_cost is not None else 'unlimited'}"
        )
    print(f"  tools      : {[t.name for t in tool_list]}")
    print(f"  out_dir    : {out_dir}\n")
    if args.max_total_cost is not None:
        print(f"  max_total  : ${args.max_total_cost:.2f} (whole-run cap)\n")

    run_gaia(tasks, llm=llm, tools=tools, policy=policy,
             out_dir=out_dir, verbose=True,
             budget_factory=budget_factory,
             max_total_cost_usd=args.max_total_cost)

    # ---- Phase 1 report on the just-written results.jsonl ----
    report = aggregate([out_dir / "results.jsonl"])
    print("\n== Resume table ==")
    print(report.render_table())
    print()
    print(report.render_failure_taxonomy())
    print(f"\n  logs : {out_dir}/logs/")
    print(f"  data : {out_dir}/results.jsonl")
    return 0


GAIA_SYSTEM_PROMPT = (
    "You are a careful research agent answering GAIA benchmark questions. "
    "Use the available tools to gather evidence, then commit via the "
    "`final_answer` tool.\n\n"
    "HOW TO COMMIT: call `final_answer` with these fields:\n"
    "  • `answer` — ONLY the literal answer string (the scorer reads this).\n"
    "  • `reasoning` — optional brief summary of how you got there. Put ALL "
    "    preamble, framing, working-out, and 'looking at the data...' "
    "    sentences HERE, not in `answer`. The scorer ignores `reasoning`.\n"
    "  • `evidence_ids` — list of evidence_id strings (from your tool results) "
    "    that ground the answer. Cite the pieces of evidence whose content "
    "    actually supports your answer. Tool results include `evidence_id` "
    "    fields on search hits and fetched pages — copy those IDs verbatim. "
    "    The citation verifier rejects answers whose cited evidence doesn't "
    "    support them, so fabricated answers with no grounding get caught.\n"
    "Plain-text replies are NOT accepted as answers. You must call the tool.\n\n"
    "CRITICAL: the `answer` field must contain ONLY the literal answer. The "
    "GAIA scorer does exact-match after normalization. Any preamble in "
    "`answer` will cause a wrong-answer mark.\n\n"
    "Format rules:\n"
    "  • If the question asks for a NUMBER: pass only the number. No units "
    "    unless the question explicitly asks for them. No thousands separators. "
    "    No words around it.\n"
    "  • UNIT AWARENESS — read the question's unit carefully. If the question "
    "    asks for the answer in 'thousand X', 'million X', '%', 'minutes' (vs. "
    "    seconds), etc., your answer must be in THAT unit. Divide / multiply / "
    "    convert your computed raw value into the requested unit before "
    "    committing. Example: 'how many thousand hours' with raw=17054.888 → "
    "    answer is '17' (= 17054.888 / 1000, rounded), NOT '17000' (raw hours) "
    "    and NOT '17054' (unrounded). When in doubt, the answer is whatever "
    "    number would naturally fit in front of the unit word in the question.\n"
    "  • If the question asks for a NAME or single word: pass only that.\n"
    "  • If the question asks for a LIST: pass a comma-separated list with "
    "    one space after each comma. No trailing 'and', no enumeration.\n"
    "  • Do NOT write 'The answer is …', 'Based on …', 'Looking at …', "
    "    'I found that …'. Just the answer.\n"
    "  • Do NOT wrap the answer in quotes or markdown.\n\n"
    "Examples of correct final_answer arguments:\n"
    "  question: 'How many studio albums…'  →  answer: '3'\n"
    "  question: 'What country…'           →  answer: 'Guatemala'\n"
    "  question: 'List the ingredients…'   →  answer: 'flour, sugar, eggs'\n"
    "  question: 'What chess move…'        →  answer: 'Rd5'\n\n"
    "Process:\n"
    "  • Gather evidence efficiently — for a simple lookup, a few tool calls "
    "    are usually enough, so don't over-search once you have the fact.\n"
    "  • But READ THE SOURCES YOU FIND. When the relevant source is a "
    "    document, open it and read the parts that matter before concluding: "
    "    for a PDF (local path OR a URL), use pdf_open then pdf_find to locate "
    "    the right page, pdf_read_page to read it, and pdf_read_page_visual to "
    "    read a figure/chart/plot or a scanned page where the text isn't "
    "    selectable. Do not answer 'unable to determine' about a document you "
    "    located but did not actually open and read.\n"
    "  • When the question needs a value you must compute or look up in data, "
    "    use the calculator or run_python rather than estimating by hand.\n"
    "  • Only after a genuine effort to read the located sources, if you still "
    "    cannot determine the answer, pass your best single-token guess to "
    "    final_answer rather than refusing or explaining."
)


def _build_policy(args):
    from banna_agent.policies.best_first_over_plans import BestFirstOverPlansPolicy
    from banna_agent.policies.bfs_over_plans import BFSOverPlansPolicy
    from banna_agent.policies.dfs_over_plans import DFSOverPlansPolicy
    from banna_agent.policies.planner_react import PlannerReActPolicy
    from banna_agent.policies.react import ReActPolicy
    from banna_agent.policies.verifier_retry import VerifierRetryPolicy

    if args.policy == "react":
        return ReActPolicy(model=args.model, system_prompt=GAIA_SYSTEM_PROMPT)
    if args.policy == "planner_react":
        return PlannerReActPolicy(model=args.model, executor_system=GAIA_SYSTEM_PROMPT)
    if args.policy == "bfs_over_plans":
        return BFSOverPlansPolicy(model=args.model, n_candidates=args.n_candidates)
    if args.policy == "dfs_over_plans":
        return DFSOverPlansPolicy(model=args.model, n_candidates=args.n_candidates)
    if args.policy == "best_first_over_plans":
        return BestFirstOverPlansPolicy(model=args.model, n_candidates=args.n_candidates)
    if args.policy == "verifier_retry":
        return VerifierRetryPolicy(inner=ReActPolicy(model=args.model, system_prompt=GAIA_SYSTEM_PROMPT))
    if args.policy == "best_of_n":
        from banna_agent.policies.best_of_n import BestOfNPolicy
        return BestOfNPolicy(
            n=args.n_trajectories,
            selector=args.selector,
            inner=VerifierRetryPolicy(
                inner=ReActPolicy(model=args.model, system_prompt=GAIA_SYSTEM_PROMPT),
            ),
            judge_model=args.model,
        )
    raise ValueError(f"unknown policy: {args.policy}")


if __name__ == "__main__":
    sys.exit(main())
