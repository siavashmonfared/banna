"""Audio transcription.

GAIA L2 attaches the occasional MP3 / WAV / M4A and asks the agent to
read what's spoken. We don't ship a local whisper install (heavy dep);
instead the tool calls OpenAI's audio.transcriptions endpoint when an
OpenAI client + key are available, and returns a clean structured
error otherwise so the policy can fall back.

The factory accepts the OpenAI client explicitly — same pattern as
`image_extract` — so tests inject a fake. In production, callers pass
in the same `OpenAIClient.sdk` they've already built for the run.

Per-content-hash cache so repeated transcription of the same audio
file is free across a run.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import JsonTool


_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".aac"}


@dataclass
class _Transcriber:
    """Holds the OpenAI SDK client (or any duck-compatible stand-in)."""

    openai_client: Any | None
    model: str = "whisper-1"
    cache: dict[str, str] = field(default_factory=dict)

    def transcribe(self, path: str | Path) -> dict[str, Any]:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            return {"ok": False, "error": f"no such file: {p}"}
        if p.suffix.lower() not in _AUDIO_EXTS:
            return {"ok": False, "error": f"unsupported audio extension: {p.suffix!r}"}

        # Content hash → cache key. We read the bytes once.
        data = p.read_bytes()
        key = hashlib.sha256(self.model.encode() + b"\0" + data).hexdigest()
        if key in self.cache:
            return {"ok": True, "transcript": self.cache[key], "cached": True,
                    "path": str(p), "model": self.model}

        if self.openai_client is None:
            return {
                "ok": False,
                "error": "audio transcription not configured (no OpenAI client wired)",
                "hint": "pass openai_client= to make_transcribe_tool, or set OPENAI_API_KEY and use OpenAIClient.",
            }

        try:
            # The OpenAI SDK accepts a file-like object via the `file` arg.
            with p.open("rb") as f:
                resp = self.openai_client.audio.transcriptions.create(
                    model=self.model,
                    file=f,
                )
            text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else "")
        except Exception as exc:
            return {"ok": False, "error": f"transcription failed: {type(exc).__name__}: {exc}"}

        if not text:
            return {"ok": False, "error": "transcription returned no text"}

        self.cache[key] = text
        return {"ok": True, "transcript": text, "cached": False,
                "path": str(p), "model": self.model, "bytes_sent": len(data)}


def make_transcribe_tool(
    openai_client: Any | None = None,
    *,
    model: str = "whisper-1",
) -> JsonTool:
    """Build the audio-transcription JsonTool.

    `openai_client` is the OpenAI SDK client (`openai.OpenAI(...)`),
    typically the `.sdk` attribute of an already-built `OpenAIClient`.
    Pass None to leave the tool registered but inert; calls will return
    a clear "not configured" error so the policy can fall back.
    """
    transcriber = _Transcriber(openai_client=openai_client, model=model)
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to an audio file (mp3/wav/m4a/etc.)."},
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    return JsonTool(
        name="transcribe_audio",
        description=(
            "Transcribe a local audio file (mp3/wav/m4a/ogg/flac/webm/aac) using "
            "OpenAI's audio.transcriptions endpoint. Returns the transcript text. "
            "Caches by content hash. Returns a structured error if no OpenAI client "
            "was wired at tool construction time."
        ),
        input_schema=schema,
        handler=lambda a: transcriber.transcribe(a["path"]),
        capabilities=frozenset({"read", "filesystem", "network"}),
    )
