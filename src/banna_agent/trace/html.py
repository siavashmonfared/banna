"""Render a JSONL event log to a self-contained static HTML trace.

No JS, no external assets — one file you can open or commit. The input is
the per-run JSONL written by `EventLog` (one event dict per line); the
event schema is documented in `core/events.py`.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _esc(s: Any) -> str:
    return html.escape(str(s if s is not None else ""))


def _fmt_json(obj: Any, limit: int = 4000) -> str:
    try:
        s = json.dumps(obj, indent=2, default=str)
    except (TypeError, ValueError):
        s = str(obj)
    if len(s) > limit:
        s = s[:limit] + f"\n… (+{len(s) - limit} chars truncated)"
    return _esc(s)


_CSS = """
:root{--bg:#0f1115;--card:#181b22;--muted:#8b94a7;--fg:#e6e9ef;--accent:#6ea8fe;
--ok:#3fb950;--err:#f85149;--think:#d2a8ff;--tool:#79c0ff;--batch:#ffa657;--line:#262b36;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
.wrap{max-width:980px;margin:0 auto;padding:28px 20px 80px;}
h1{font-size:19px;margin:0 0 4px;}
.sub{color:var(--muted);font-size:12px;margin-bottom:20px;}
.q{background:var(--card);border-left:3px solid var(--accent);padding:12px 14px;
border-radius:6px;margin-bottom:8px;white-space:pre-wrap;}
.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em;}
.step{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px 14px;margin:10px 0;}
.step .hd{display:flex;gap:8px;align-items:center;margin-bottom:6px;}
.badge{font-size:11px;padding:1px 8px;border-radius:10px;font-weight:600;}
.b-think{background:rgba(210,168,255,.15);color:var(--think);}
.b-tool{background:rgba(121,192,255,.15);color:var(--tool);}
.b-batch{background:rgba(255,166,87,.15);color:var(--batch);}
.b-final{background:rgba(63,185,80,.15);color:var(--ok);}
.b-ask{background:rgba(255,166,87,.12);color:var(--batch);}
.stepno{color:var(--muted);font-size:12px;margin-left:auto;}
.text{white-space:pre-wrap;margin:6px 0;}
.tool{border-top:1px dashed var(--line);margin-top:8px;padding-top:8px;}
.tname{color:var(--tool);font-weight:600;}
.ok{color:var(--ok);}.err{color:var(--err);}
pre{background:#0b0d11;border:1px solid var(--line);border-radius:6px;
padding:8px 10px;overflow:auto;margin:6px 0;font-size:12.5px;}
.batchwrap{border-left:3px solid var(--batch);padding-left:10px;margin:6px 0;}
.final{background:var(--card);border:1px solid var(--ok);border-radius:8px;
padding:14px;margin-top:14px;}
.final.wrong{border-color:var(--err);}
.kv{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:12px;margin-top:6px;}
.footer{color:var(--muted);font-size:11px;margin-top:24px;text-align:center;}
"""


def _load(jsonl: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def render_html(events: list[dict[str, Any]]) -> str:
    """Render parsed events (list of dicts) into a full HTML document."""
    by_kind: dict[str, list[dict]] = {}
    for e in events:
        by_kind.setdefault(e.get("kind", ""), []).append(e)

    run_start = (by_kind.get("run_start") or [{}])[0]
    run_end = (by_kind.get("run_end") or [{}])[0]
    rs_p = run_start.get("payload", {})
    re_p = run_end.get("payload", {})
    question = rs_p.get("question", "(question not in log)")
    policy = rs_p.get("policy", "?")
    run_id = run_start.get("run_id") or (events[0].get("run_id") if events else "?")

    body: list[str] = []
    body.append('<div class="wrap">')
    body.append("<h1>banna trace</h1>")
    body.append(f'<div class="sub">run {_esc(run_id)} · policy '
                f'<b>{_esc(policy)}</b> · {len(events)} events</div>')
    body.append('<div class="label">question</div>')
    body.append(f'<div class="q">{_esc(question)}</div>')

    # Walk events in order, rendering each propose/tool/batch as a step card.
    last_obs: dict[str, Any] = {}
    step_seen = 0
    i = 0
    n = len(events)
    while i < n:
        e = events[i]
        kind = e.get("kind")
        p = e.get("payload", {})

        if kind == "propose":
            step_seen += 1
            ka = p.get("kind_of_action", "")
            card = ['<div class="step">']
            badge, cls = _badge_for(ka, p)
            card.append(f'<div class="hd"><span class="badge {cls}">{_esc(badge)}</span>'
                        f'<span class="stepno">step {step_seen}</span></div>')
            atext = (p.get("action_text") or "").strip()
            if atext:
                card.append(f'<div class="text">{_esc(atext)}</div>')
            # Attach immediately-following tool_call/tool_result/tool_batch.
            j = i + 1
            while j < n and events[j].get("kind") in ("tool_call", "tool_result", "tool_batch"):
                card.append(_render_tool_event(events[j]))
                j += 1
            card.append("</div>")
            body.append("".join(card))
            i = j
            continue

        if kind == "observation":
            last_obs = p
            i += 1
            continue

        if kind == "ask_user":
            body.append(
                f'<div class="step"><div class="hd">'
                f'<span class="badge b-ask">ASK_USER</span></div>'
                f'<div class="text">{_esc(p.get("question",""))}</div></div>')
            i += 1
            continue

        i += 1

    # Final answer card.
    final = re_p.get("final_answer")
    if final is not None:
        reason = re_p.get("budget_reason", "ok")
        body.append('<div class="final">')
        body.append('<div class="label">final answer</div>')
        body.append(f'<div class="text">{_esc(final)}</div>')
        body.append(f'<div class="kv"><span>exit: {_esc(reason)}</span>'
                    f'<span>steps: {_esc(re_p.get("steps_used","?"))}</span></div>')
        body.append("</div>")

    # Run totals from the last observation.
    if last_obs:
        body.append('<div class="kv" style="margin-top:14px">'
                    f'<span>tokens in: {_esc(last_obs.get("cumulative_tokens_in","?"))}</span>'
                    f'<span>tokens out: {_esc(last_obs.get("cumulative_tokens_out","?"))}</span>'
                    f'<span>wall: {_esc(_round(last_obs.get("cumulative_wall_s")))}s</span>'
                    f'<span>evidence: {_esc(last_obs.get("evidence_count","?"))}</span>'
                    "</div>")

    body.append('<div class="footer">generated by banna trace view</div>')
    body.append("</div>")

    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>banna trace {_esc(run_id)}</title><style>{_CSS}</style></head>"
        f"<body>{''.join(body)}</body></html>"
    )


def _badge_for(kind_of_action: str, p: dict) -> tuple[str, str]:
    k = (kind_of_action or "").lower()
    if "final" in k or p.get("has_answer"):
        return "FINAL_ANSWER", "b-final"
    if "batch" in k:
        return "TOOL_BATCH", "b-batch"
    if "tool" in k:
        return f"TOOL · {p.get('tool_name','')}", "b-tool"
    if "ask" in k:
        return "ASK_USER", "b-ask"
    return "THINK", "b-think"


def _render_tool_event(e: dict) -> str:
    kind = e.get("kind")
    p = e.get("payload", {})
    if kind == "tool_batch":
        names = ", ".join(p.get("tool_names", []) or [])
        return (f'<div class="tool"><span class="badge b-batch">PARALLEL BATCH</span> '
                f'<span class="tname">{_esc(names)}</span> '
                f'<span class="label">({_esc(p.get("n","?"))} concurrent)</span></div>')
    if kind == "tool_call":
        args = p.get("arguments", {})
        inb = ' <span class="label">[in batch]</span>' if p.get("in_batch") else ""
        return (f'<div class="tool"><span class="label">call</span> '
                f'<span class="tname">{_esc(p.get("tool_name",""))}</span>{inb}'
                f'<pre>{_fmt_json(args, 1500)}</pre></div>')
    if kind == "tool_result":
        ok = p.get("ok")
        status = '<span class="ok">✓ ok</span>' if ok else '<span class="err">✗ error</span>'
        preview = p.get("preview") or p.get("error") or ""
        wall = _round(p.get("wall_s"))
        return (f'<div class="tool"><span class="label">result</span> {status} '
                f'<span class="label">{_esc(wall)}s</span>'
                f'<pre>{_esc(preview)}</pre></div>')
    return ""


def _round(v: Any) -> str:
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "?"


def render_file(jsonl_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Render a JSONL log file to an HTML file. Default output is the input
    path with a `.html` suffix. Returns the written path."""
    src = Path(jsonl_path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"no such trace log: {src}")
    events = _load(src.read_text(encoding="utf-8"))
    doc = render_html(events)
    dst = Path(out_path).expanduser() if out_path else src.with_suffix(".html")
    dst.write_text(doc, encoding="utf-8")
    return dst
