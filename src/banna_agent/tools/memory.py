"""Memory as a tool the model can invoke.

Exposes `op ∈ {write, read, search, delete, list}` as a single JsonTool.
The model decides what to remember and when to recall — matches the
Claude / OpenAI Agent SDK convention and gives a clean ablation lever
(tool-on vs. tool-off).

The tool is *stateful* — it closes over a `Memory` instance. Build one
per run and register the resulting `JsonTool` in your ToolRegistry.
"""
from __future__ import annotations

from typing import Any

from ..memory.base import Memory, MemoryEntry, MemoryKind, MemoryQuery
from .base import JsonTool


_VALID_KINDS: tuple[str, ...] = ("fact", "lesson", "skill", "example", "scratch")


def _handler_factory(store: Memory):
    def handle(args: dict[str, Any]) -> dict[str, Any]:
        op = args.get("op", "")

        if op == "write":
            content = str(args.get("content", ""))
            if not content.strip():
                raise ValueError("'content' must be a non-empty string")
            kind = args.get("kind", "fact")
            if kind not in _VALID_KINDS:
                raise ValueError(f"kind must be one of {_VALID_KINDS}; got {kind!r}")
            entry = MemoryEntry(
                content=content,
                kind=kind,
                tags=list(args.get("tags") or []),
                confidence=float(args.get("confidence", 1.0)),
                metadata=dict(args.get("metadata") or {}),
            )
            eid = store.write(entry)
            return {"op": op, "id": eid, "kind": entry.kind}

        if op == "read":
            eid = str(args.get("id", ""))
            if not eid:
                raise ValueError("'id' is required for op=read")
            entry = store.read_by_id(eid)
            return {"op": op, "entry": entry.to_dict() if entry else None}

        if op == "search":
            query = str(args.get("query", ""))
            if not query.strip():
                raise ValueError("'query' must be a non-empty string")
            kind_filter = args.get("kind")
            if kind_filter is not None and kind_filter not in _VALID_KINDS:
                raise ValueError(f"kind must be one of {_VALID_KINDS}")
            q = MemoryQuery(
                query=query,
                k=int(args.get("k", 5)),
                kind_filter=kind_filter,
                tags_filter=list(args.get("tags") or []),
                min_confidence=float(args.get("min_confidence", 0.0)),
            )
            hits = store.search(q)
            return {
                "op": op,
                "query": query,
                "count": len(hits),
                "hits": [
                    {"id": e.id, "kind": e.kind, "content": e.content,
                     "score": score, "tags": list(e.tags),
                     "confidence": e.confidence}
                    for e, score in hits
                ],
            }

        if op == "delete":
            eid = str(args.get("id", ""))
            if not eid:
                raise ValueError("'id' is required for op=delete")
            ok = store.delete(eid)
            return {"op": op, "ok": ok, "id": eid}

        if op == "list":
            kind = args.get("kind")
            if kind is not None and kind not in _VALID_KINDS:
                raise ValueError(f"kind must be one of {_VALID_KINDS}")
            entries = store.all(kind=kind)
            return {
                "op": op,
                "count": len(entries),
                "entries": [e.to_dict() for e in entries],
            }

        raise ValueError(f"unknown op {op!r}; use write|read|search|delete|list")

    return handle


MEMORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {
            "type": "string",
            "enum": ["write", "read", "search", "delete", "list"],
            "description": "Memory operation.",
        },
        "content": {"type": "string", "description": "Text content to store (op=write)."},
        "kind": {
            "type": "string",
            "enum": list(_VALID_KINDS),
            "description": (
                "Category of memory: fact (external truth), lesson "
                "(agent-authored reflection), skill (callable code), "
                "example (q+a pair), scratch (ephemeral)."
            ),
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Free-form tags for retrieval filtering.",
        },
        "confidence": {
            "type": "number",
            "description": "0.0–1.0 confidence in this entry (op=write).",
            "default": 1.0,
        },
        "id": {"type": "string", "description": "Entry id (op=read/delete)."},
        "query": {"type": "string", "description": "Search text (op=search)."},
        "k": {"type": "integer", "description": "Max hits (op=search)", "default": 5},
        "min_confidence": {
            "type": "number",
            "description": "Filter out entries below this confidence (op=search).",
            "default": 0.0,
        },
        "metadata": {
            "type": "object",
            "description": "Free-form structured data attached to an entry (op=write).",
            "additionalProperties": True,
        },
    },
    "required": ["op"],
    "additionalProperties": False,
}


def make_memory_tool(store: Memory) -> JsonTool:
    """Build a JsonTool over the given Memory instance."""
    return JsonTool(
        name="memory",
        description=(
            "Persistent memory you can write to and search. Use op=write to "
            "save a fact or lesson, op=search to recall relevant entries "
            "for the current question, op=list to see all entries, "
            "op=read/delete by id. Prefer writing short, factual entries."
        ),
        input_schema=MEMORY_SCHEMA,
        handler=_handler_factory(store),
        capabilities=frozenset({"state", "memory"}),
    )
