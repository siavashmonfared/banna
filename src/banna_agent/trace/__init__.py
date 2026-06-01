"""Trace rendering: turn a run's JSONL event log into a static HTML report.

`render_html(events) -> str` produces a single self-contained HTML
document (inline CSS, no JS, no external assets) so a reviewer can open
one file and *see* the agent think: each step's reasoning, the tool calls
and their results, parallel batches, evidence, running cost, and the final
answer vs. gold. `render_file(jsonl_path) -> html_path` is the CLI entry.
"""
from __future__ import annotations

from .html import render_file, render_html

__all__ = ["render_html", "render_file"]
