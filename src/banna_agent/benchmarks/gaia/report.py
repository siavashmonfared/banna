"""Aggregate GAIA `results.jsonl` files into the ablation/resume table.

Reads one or more `results.jsonl` files (each is the output of a single
`run_gaia` invocation), groups them by policy + model + level, and
emits:

  (a) The resume table:
        policy  model  level  N   acc%   cost/task  steps/task   tokens/task  timeout%
  (b) A failure taxonomy:
        policy x finished_reason → count

Designed to be invoked as:

    python -m banna_agent.benchmarks.gaia.report runs/*.jsonl

…or programmatically via `aggregate(paths) -> AggregatedReport`.

A "cell" of the resume table is (policy_name, model_name, level).
Policy/model are read from the JSONL records; we deliberately don't
trust the run directory name. If your JSONL doesn't carry those fields
(e.g. an old run), the cell key falls back to ("?", "?", level) and the
table still renders.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class CellStats:
    policy: str
    model: str
    level: int
    n: int = 0
    n_correct: int = 0
    sum_cost: float = 0.0
    sum_steps: int = 0
    sum_tokens_in: int = 0
    sum_tokens_out: int = 0
    sum_wall_s: float = 0.0
    n_timeout: int = 0
    finished_reasons: Counter = field(default_factory=Counter)
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def cost_per_task(self) -> float:
        return self.sum_cost / self.n if self.n else 0.0

    @property
    def steps_per_task(self) -> float:
        return self.sum_steps / self.n if self.n else 0.0

    @property
    def tokens_per_task(self) -> float:
        return (self.sum_tokens_in + self.sum_tokens_out) / self.n if self.n else 0.0

    @property
    def timeout_pct(self) -> float:
        return 100.0 * self.n_timeout / self.n if self.n else 0.0


@dataclass
class AggregatedReport:
    cells: list[CellStats] = field(default_factory=list)

    def to_table_rows(self) -> list[list[str]]:
        header = ["policy", "model", "level", "N", "acc%", "cost/task", "steps/task", "tok/task", "timeout%"]
        rows: list[list[str]] = [header]
        # Sort: policy, model, level
        for c in sorted(self.cells, key=lambda x: (x.policy, x.model, x.level)):
            rows.append([
                c.policy,
                c.model,
                str(c.level),
                str(c.n),
                f"{c.accuracy * 100:.1f}",
                f"{c.cost_per_task:.4f}",
                f"{c.steps_per_task:.1f}",
                f"{c.tokens_per_task:.0f}",
                f"{c.timeout_pct:.1f}",
            ])
        return rows

    def render_table(self) -> str:
        rows = self.to_table_rows()
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        lines = []
        for i, r in enumerate(rows):
            lines.append("  ".join(c.ljust(widths[j]) for j, c in enumerate(r)))
            if i == 0:
                lines.append("  ".join("-" * w for w in widths))
        return "\n".join(lines)

    def render_failure_taxonomy(self) -> str:
        # policy → counter
        by_policy: dict[str, Counter] = defaultdict(Counter)
        for c in self.cells:
            by_policy[c.policy].update(c.finished_reasons)
        if not by_policy:
            return "(no finished_reason data — older run format?)"
        out_lines = ["Failure taxonomy (finished_reason counts, all levels):"]
        all_reasons = sorted({k for c in by_policy.values() for k in c})
        header = ["policy"] + all_reasons
        rows = [header]
        for policy, counter in sorted(by_policy.items()):
            rows.append([policy] + [str(counter.get(r, 0)) for r in all_reasons])
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        for i, r in enumerate(rows):
            out_lines.append("  ".join(c.ljust(widths[j]) for j, c in enumerate(r)))
            if i == 0:
                out_lines.append("  ".join("-" * w for w in widths))
        return "\n".join(out_lines)


def aggregate(paths: Iterable[str | Path]) -> AggregatedReport:
    """Load every JSONL line from `paths` and group into cells."""
    cells: dict[tuple[str, str, int], CellStats] = {}
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            policy = str(rec.get("policy_name") or "?")
            model = str(rec.get("model_name") or "?")
            level = int(rec.get("level") or 0)
            key = (policy, model, level)
            c = cells.get(key)
            if c is None:
                c = CellStats(policy=policy, model=model, level=level)
                cells[key] = c
            c.n += 1
            if rec.get("is_correct"):
                c.n_correct += 1
            c.sum_cost += float(rec.get("cost_usd") or 0.0)
            c.sum_steps += int(rec.get("steps_used") or 0)
            c.sum_tokens_in += int(rec.get("tokens_in") or 0)
            c.sum_tokens_out += int(rec.get("tokens_out") or 0)
            c.sum_wall_s += float(rec.get("wall_s") or 0.0)
            if rec.get("timeout"):
                c.n_timeout += 1
            fr = rec.get("finished_reason") or rec.get("budget_reason") or "ok"
            c.finished_reasons[str(fr)] += 1
            c.cache_hits += int(rec.get("cache_hits") or 0)
            c.cache_misses += int(rec.get("cache_misses") or 0)
    return AggregatedReport(cells=list(cells.values()))


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate GAIA results.jsonl files.")
    ap.add_argument("paths", nargs="+", help="One or more results.jsonl paths.")
    ap.add_argument(
        "--no-taxonomy",
        action="store_true",
        help="Skip the failure-taxonomy section.",
    )
    args = ap.parse_args(argv)
    report = aggregate(args.paths)
    print(report.render_table())
    if not args.no_taxonomy:
        print()
        print(report.render_failure_taxonomy())
    return 0


if __name__ == "__main__":
    sys.exit(_main())
