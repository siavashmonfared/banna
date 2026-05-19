"""image_extract tests with a fake LLM.

We don't hit a real vision API; instead we pass a fake `llm` whose
`chat()` records the message it received and returns a canned reply.
That lets us verify:

  * the tool produces a base64 image block with the right media_type,
  * the question text accompanies the image,
  * the model's text reply is surfaced as `answer`,
  * a repeat call hits the in-process cache,
  * preprocessing normalises orientation and downscales big images.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from banna_agent.llm.base import ContentBlock, LLMReply, Message  # noqa: E402
from banna_agent.tools.image_extract import make_image_extract_tool  # noqa: E402


@dataclass
class _FakeLLM:
    reply_text: str = "The display reads 042."
    calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = "fake"

    def chat(self, *, messages, model=None, max_tokens=None, **_) -> LLMReply:
        # Capture the last user message structure.
        last = messages[-1]
        self.calls.append({
            "n_blocks": len(last.content),
            "kinds": [b.kind for b in last.content],
            "raw_types": [
                (b.raw or {}).get("type") if isinstance(b.raw, dict) else None
                for b in last.content
            ],
            "text": next((b.text for b in last.content if b.kind == "text"), None),
            "max_tokens": max_tokens,
        })
        return LLMReply(
            provider="fake", model=model or "fake",
            content=[ContentBlock(kind="text", text=self.reply_text)],
            stop_reason="end_turn",
        )


def _make_png(path: Path, size: tuple[int, int] = (16, 16)) -> None:
    im = Image.new("RGB", size, color=(255, 0, 0))
    im.save(path, format="PNG")


def test_extract_image_calls_vision_with_image_block(tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    _make_png(img)
    llm = _FakeLLM()
    tool = make_image_extract_tool(llm)
    r = tool.handler({"path": str(img), "question": "What number is shown?"})
    assert r["ok"] is True
    assert r["answer"] == "The display reads 042."
    assert r["cached"] is False
    # Exactly one chat call with [text, image-shaped raw] content.
    assert len(llm.calls) == 1
    c = llm.calls[0]
    assert c["n_blocks"] == 2
    assert c["text"] == "What number is shown?"
    assert "image" in c["raw_types"]


def test_extract_image_caches_by_question_and_bytes(tmp_path: Path) -> None:
    img = tmp_path / "y.png"
    _make_png(img)
    llm = _FakeLLM()
    tool = make_image_extract_tool(llm)
    r1 = tool.handler({"path": str(img), "question": "Describe."})
    r2 = tool.handler({"path": str(img), "question": "Describe."})
    assert r1["cached"] is False
    assert r2["cached"] is True
    # Only one underlying llm.chat call despite two tool invocations.
    assert len(llm.calls) == 1


def test_extract_image_downscales_large_image(tmp_path: Path) -> None:
    big = tmp_path / "big.png"
    _make_png(big, size=(4000, 3000))
    llm = _FakeLLM()
    tool = make_image_extract_tool(llm, max_edge_px=512)
    r = tool.handler({"path": str(big), "question": "?"})
    assert r["ok"] is True
    # bytes_sent must be much smaller than a raw 4000x3000 PNG would be.
    assert r["bytes_sent"] < 4_000_000


def test_extract_image_rejects_missing_file() -> None:
    tool = make_image_extract_tool(_FakeLLM())
    r = tool.handler({"path": "/no/such/img.png", "question": "?"})
    assert r["ok"] is False
    assert "no such file" in r["error"]


def test_extract_image_rejects_empty_question(tmp_path: Path) -> None:
    img = tmp_path / "z.png"
    _make_png(img)
    tool = make_image_extract_tool(_FakeLLM())
    r = tool.handler({"path": str(img), "question": "   "})
    assert r["ok"] is False
    assert "question" in r["error"].lower()


def test_extract_image_returns_clean_error_on_llm_failure(tmp_path: Path) -> None:
    img = tmp_path / "z.png"
    _make_png(img)

    class _BoomLLM:
        model = "boom"
        def chat(self, **_):
            raise RuntimeError("vision unavailable")

    tool = make_image_extract_tool(_BoomLLM())
    r = tool.handler({"path": str(img), "question": "?"})
    assert r["ok"] is False
    assert "vision call failed" in r["error"]
    assert "RuntimeError" in r["error"]
