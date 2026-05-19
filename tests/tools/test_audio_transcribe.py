"""audio_transcribe tests with a fake OpenAI client."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from banna_agent.tools.audio_transcribe import make_transcribe_tool


@dataclass
class _FakeTranscriptions:
    text: str = "hello world this is audio"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, *, model, file):
        self.calls.append({"model": model, "fname": getattr(file, "name", "?")})
        class _R: pass
        r = _R()
        r.text = self.text
        return r


@dataclass
class _FakeAudio:
    transcriptions: _FakeTranscriptions = field(default_factory=_FakeTranscriptions)


@dataclass
class _FakeClient:
    audio: _FakeAudio = field(default_factory=_FakeAudio)


def _make_wav(path: Path) -> None:
    # 1 frame of silence is enough; we don't decode it.
    path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
                     b"\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")


def test_transcribe_uses_openai_client(tmp_path: Path) -> None:
    f = tmp_path / "a.wav"
    _make_wav(f)
    client = _FakeClient()
    tool = make_transcribe_tool(openai_client=client)
    r = tool.handler({"path": str(f)})
    assert r["ok"] is True
    assert r["transcript"] == "hello world this is audio"
    assert r["cached"] is False
    assert len(client.audio.transcriptions.calls) == 1


def test_transcribe_caches_by_content(tmp_path: Path) -> None:
    f = tmp_path / "a.wav"
    _make_wav(f)
    client = _FakeClient()
    tool = make_transcribe_tool(openai_client=client)
    r1 = tool.handler({"path": str(f)})
    r2 = tool.handler({"path": str(f)})
    assert r1["cached"] is False
    assert r2["cached"] is True
    assert len(client.audio.transcriptions.calls) == 1  # one network call total


def test_transcribe_without_client_returns_structured_error(tmp_path: Path) -> None:
    f = tmp_path / "a.wav"
    _make_wav(f)
    tool = make_transcribe_tool(openai_client=None)
    r = tool.handler({"path": str(f)})
    assert r["ok"] is False
    assert "not configured" in r["error"]
    assert "hint" in r


def test_transcribe_rejects_unsupported_extension(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("not audio")
    tool = make_transcribe_tool(openai_client=_FakeClient())
    r = tool.handler({"path": str(f)})
    assert r["ok"] is False
    assert "unsupported audio extension" in r["error"]


def test_transcribe_handles_api_failure(tmp_path: Path) -> None:
    f = tmp_path / "a.wav"
    _make_wav(f)

    class _BoomTrans:
        def create(self, **_): raise RuntimeError("rate limit")
    class _BoomAudio:
        transcriptions = _BoomTrans()
    class _BoomClient:
        audio = _BoomAudio()

    tool = make_transcribe_tool(openai_client=_BoomClient())
    r = tool.handler({"path": str(f)})
    assert r["ok"] is False
    assert "transcription failed" in r["error"]
    assert "RuntimeError" in r["error"]
