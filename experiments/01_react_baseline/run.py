"""Experiment 01 — ReAct baseline on GAIA Level 1.

Usage (from repo root):

    cd banna_agent
    set -a && source <(grep -v '^#' .env) && set +a
    PYTHONPATH=src python experiments/01_react_baseline/run.py --n 5

Flags:
    --provider  anthropic | bedrock | openai | gemini | ollama  (default anthropic)
    --model     override the provider default model
    --level     1 | 2 | 3  (default 1)
    --n         how many tasks (default: all at that level)
    --dataset   gaia | jsonl   (default gaia — HF download)
    --jsonl     path to JSONL file when --dataset=jsonl
    --out-dir   where to write logs/summary (default experiments/01_react_baseline/runs/<ts>)
    --no-shell  drop run_shell from the tool registry
    --no-plan   drop plan from the tool registry
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

# Ensure `src/` is on the path when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from banna_agent.benchmarks.gaia.loader import (  # noqa: E402
    GAIATask,
    load_gaia,
    load_gaia_from_jsonl,
)
from banna_agent.benchmarks.gaia.runner import run_gaia  # noqa: E402
from banna_agent.llm.registry import make_client  # noqa: E402
from banna_agent.memory.embeddings import HashEmbedder  # noqa: E402
from banna_agent.memory.in_memory_store import InMemoryStore  # noqa: E402
from banna_agent.memory.jsonl_store import JSONLStore  # noqa: E402
from banna_agent.policies.best_first_over_plans import BestFirstOverPlansPolicy  # noqa: E402
from banna_agent.policies.bfs_over_plans import BFSOverPlansPolicy  # noqa: E402
from banna_agent.policies.dfs_over_plans import DFSOverPlansPolicy  # noqa: E402
from banna_agent.policies.planner_react import PlannerReActPolicy  # noqa: E402
from banna_agent.policies.react import ReActPolicy  # noqa: E402
from banna_agent.policies.verifier_retry import VerifierRetryPolicy  # noqa: E402
from banna_agent.tools.base import ToolRegistry  # noqa: E402
from banna_agent.tools.calculator import make_calculator_tool  # noqa: E402
from banna_agent.tools.file_reader import make_file_reader_tool  # noqa: E402
from banna_agent.tools.grep import make_grep_tool  # noqa: E402
from banna_agent.tools.list_files import make_list_files_tool  # noqa: E402
from banna_agent.tools.memory import make_memory_tool  # noqa: E402
from banna_agent.tools.plan import make_plan_tool  # noqa: E402
from banna_agent.tools.python_sandbox import make_python_sandbox_tool  # noqa: E402
from banna_agent.tools.run_shell import make_run_shell_tool  # noqa: E402
from banna_agent.tools.search import make_search_tool  # noqa: E402
from banna_agent.tools.url_reader import make_url_reader_tool  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="ReAct baseline on GAIA.")
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--policy",
        choices=["react", "planner_react", "bfs_over_plans",
                 "dfs_over_plans", "best_first_over_plans",
                 "verifier_retry"],
        default="react",
        help="Which Policy to run.",
    )
    ap.add_argument("--n-candidates", type=int, default=3,
                    help="N candidate plans for BFS/DFS/best-first.")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--dataset", choices=["gaia", "jsonl"], default="gaia")
    ap.add_argument("--jsonl", default=None, help="Path to a GAIA-shape JSONL")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-shell", action="store_true")
    ap.add_argument("--no-plan", action="store_true")
    # memory flags
    ap.add_argument("--memory", choices=["off", "inmemory", "jsonl"], default="off",
                    help="Memory backend for the ablation. Default off.")
    ap.add_argument("--memory-path", default=None,
                    help="Path to memory.jsonl when --memory=jsonl.")
    ap.add_argument("--embedder", choices=["hash", "none"], default="hash",
                    help="Embedder for cosine search. Default hash (zero-dep).")
    # compaction + skills — wired for future use; flags are parsed today
    # so week-2 experiments don't need a CLI change.
    ap.add_argument("--compact", action="store_true",
                    help="Enable trace compaction when the trace exceeds threshold.")
    ap.add_argument("--skills", action="store_true",
                    help="Enable skill-library injection (requires --memory).")
    ap.add_argument("--skills-path", default=None)
    args = ap.parse_args()

    # --- tasks ---------------------------------------------------------
    tasks: list[GAIATask]
    if args.dataset == "jsonl":
        if not args.jsonl:
            ap.error("--dataset=jsonl requires --jsonl")
        tasks = load_gaia_from_jsonl(args.jsonl)
        if args.level:
            tasks = [t for t in tasks if t.level == args.level]
        if args.n is not None:
            tasks = tasks[: args.n]
    else:
        tasks = load_gaia(levels={args.level}, limit=args.n)

    if not tasks:
        print("no tasks matched", file=sys.stderr)
        return 2

    # --- out dir -------------------------------------------------------
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).parent / "runs" / f"{ts}_{args.provider}_L{args.level}"
    )

    # --- llm + tools ---------------------------------------------------
    llm = make_client(args.provider, model=args.model)
    tool_list = [
        make_search_tool(),
        make_url_reader_tool(),
        make_file_reader_tool(),
        make_calculator_tool(),
        make_python_sandbox_tool(),
        make_list_files_tool(),
        make_grep_tool(),
    ]
    if not args.no_plan:
        tool_list.append(make_plan_tool())
    if not args.no_shell:
        tool_list.append(make_run_shell_tool())

    # --- memory --------------------------------------------------------
    memory_store = None
    if args.memory != "off":
        embedder = HashEmbedder(dim=256) if args.embedder == "hash" else None
        if args.memory == "inmemory":
            memory_store = InMemoryStore(embedder=embedder)
        else:  # jsonl
            mem_path = args.memory_path or str(out_dir / "memory.jsonl")
            memory_store = JSONLStore(mem_path, embedder=embedder)
        tool_list.append(make_memory_tool(memory_store))

    tools = ToolRegistry(tool_list)

    policy = _build_policy(args)

    print(f"\n== GAIA run ==")
    print(f"  policy    : {args.policy}")
    print(f"  provider  : {args.provider}")
    print(f"  model     : {args.model or '(provider default)'}")
    print(f"  level     : {args.level}")
    print(f"  n_tasks   : {len(tasks)}")
    print(f"  tools     : {[t.name for t in tool_list]}")
    print(f"  memory    : {args.memory} ({args.embedder} embedder)")
    print(f"  compact   : {args.compact}  skills={args.skills}")
    print(f"  out_dir   : {out_dir}\n")

    compactor = None
    if args.compact:
        from banna_agent.memory.compactor import (  # noqa: E402
            CompactionConfig, TraceCompactor,
        )
        compactor = TraceCompactor(
            llm=llm,
            config=CompactionConfig(enabled=True),
        )

    agg = run_gaia(tasks, llm=llm, tools=tools, policy=policy,
                   out_dir=out_dir, compactor=compactor)

    print("\n== Summary ==")
    print(f"  accuracy      : {agg.accuracy:.3f}  ({agg.n_correct}/{agg.n_total})")
    print(f"  total tokens  : in={agg.total_tokens_in}  out={agg.total_tokens_out}")
    print(f"  total wall_s  : {agg.total_wall_s:.1f}")
    print(f"  by level      : {agg.by_level}")
    print(f"  by match_kind : {agg.by_match_kind}")
    print(f"  logs          : {out_dir}/logs/")
    return 0


def _build_policy(args):
    if args.policy == "react":
        return ReActPolicy(model=args.model)
    if args.policy == "planner_react":
        return PlannerReActPolicy(model=args.model)
    if args.policy == "bfs_over_plans":
        return BFSOverPlansPolicy(model=args.model, n_candidates=args.n_candidates)
    if args.policy == "dfs_over_plans":
        return DFSOverPlansPolicy(model=args.model, n_candidates=args.n_candidates)
    if args.policy == "best_first_over_plans":
        return BestFirstOverPlansPolicy(model=args.model, n_candidates=args.n_candidates)
    if args.policy == "verifier_retry":
        return VerifierRetryPolicy(inner=ReActPolicy(model=args.model))
    raise ValueError(f"unknown policy: {args.policy}")


if __name__ == "__main__":
    sys.exit(main())
