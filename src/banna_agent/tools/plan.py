"""Stateful plan / todo scratchpad tool.

Why this exists:
    Models are meaningfully better at multi-step tasks when they can
    externalize plan state instead of holding it in context. This tool
    gives the model a structured place to write "what I'm going to do,
    what I've done, and what I'm stuck on."

    The plan state is process-local (a module-level dict keyed by
    plan_id). The default plan_id is 'default' — the common case of one
    plan per run. For parallel agents / branching search (week 2), each
    AgentState can pass its own `plan_id` to keep plans isolated.

Schema of one item:
    { "id": str, "step": str, "status": "todo"|"doing"|"done"|"blocked",
      "notes": str }

Operations:
    op=add     fields: step, notes?, id?     -> returns the new item
    op=update  fields: id, status?, notes?, step? -> returns item
    op=list    no fields                     -> returns all items
    op=clear   no fields                     -> wipes the plan

The driver never reads plan state directly; it's observable by the model
through op=list. This keeps the plan a deliberate externalization, not a
hidden channel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from .base import JsonTool


_VALID_STATUSES = {"todo", "doing", "done", "blocked"}


@dataclass
class PlanItem:
    id: str
    step: str
    status: str = "todo"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Module-level store: plan_id -> list[PlanItem].
# Process-scoped; reset with op=clear or by calling `reset_all()` in tests.
_PLANS: dict[str, list[PlanItem]] = {}


def reset_all() -> None:
    """Test helper: wipe every plan."""
    _PLANS.clear()


def _get(plan_id: str) -> list[PlanItem]:
    return _PLANS.setdefault(plan_id, [])


def _add(plan_id: str, step: str, notes: str = "", item_id: str | None = None) -> PlanItem:
    if not step.strip():
        raise ValueError("'step' must be non-empty")
    iid = item_id or f"p{len(_get(plan_id)) + 1}_{uuid4().hex[:4]}"
    item = PlanItem(id=iid, step=step.strip(), notes=notes)
    _get(plan_id).append(item)
    return item


def _update(
    plan_id: str,
    item_id: str,
    *,
    status: str | None = None,
    notes: str | None = None,
    step: str | None = None,
) -> PlanItem:
    items = _get(plan_id)
    for it in items:
        if it.id == item_id:
            if status is not None:
                if status not in _VALID_STATUSES:
                    raise ValueError(
                        f"invalid status {status!r}; must be one of {sorted(_VALID_STATUSES)}"
                    )
                it.status = status
            if notes is not None:
                it.notes = notes
            if step is not None and step.strip():
                it.step = step.strip()
            return it
    raise KeyError(f"no plan item with id {item_id!r} in plan {plan_id!r}")


def _handler(args: dict[str, Any]) -> dict[str, Any]:
    op = args.get("op", "")
    plan_id = args.get("plan_id", "default") or "default"

    if op == "add":
        item = _add(
            plan_id,
            step=str(args.get("step", "")),
            notes=str(args.get("notes", "")),
            item_id=args.get("id") or None,
        )
        return {"op": op, "plan_id": plan_id, "item": item.to_dict(),
                "plan": [i.to_dict() for i in _get(plan_id)]}

    if op == "update":
        item_id = str(args.get("id", ""))
        if not item_id:
            raise ValueError("'id' is required for op=update")
        item = _update(
            plan_id,
            item_id=item_id,
            status=args.get("status"),
            notes=args.get("notes"),
            step=args.get("step"),
        )
        return {"op": op, "plan_id": plan_id, "item": item.to_dict(),
                "plan": [i.to_dict() for i in _get(plan_id)]}

    if op == "list":
        return {"op": op, "plan_id": plan_id,
                "plan": [i.to_dict() for i in _get(plan_id)]}

    if op == "clear":
        _PLANS.pop(plan_id, None)
        return {"op": op, "plan_id": plan_id, "plan": []}

    raise ValueError(f"unknown op {op!r}; use add|update|list|clear")


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {
            "type": "string",
            "enum": ["add", "update", "list", "clear"],
            "description": "Operation on the plan.",
        },
        "plan_id": {
            "type": "string",
            "description": "Optional plan scope id (default: 'default').",
            "default": "default",
        },
        "step": {"type": "string", "description": "Short step description (op=add/update)."},
        "notes": {"type": "string", "description": "Optional freeform notes."},
        "id": {"type": "string", "description": "Plan item id (required for update)."},
        "status": {
            "type": "string",
            "enum": ["todo", "doing", "done", "blocked"],
            "description": "New status for op=update.",
        },
    },
    "required": ["op"],
    "additionalProperties": False,
}


def make_plan_tool() -> JsonTool:
    return JsonTool(
        name="plan",
        description=(
            "Maintain a structured to-do / plan as you work. "
            "Use op=add to write a step, op=update to mark it done/blocked, "
            "op=list to see the plan, op=clear to reset."
        ),
        input_schema=PLAN_SCHEMA,
        handler=_handler,
        capabilities=frozenset({"state", "scratchpad"}),
    )
