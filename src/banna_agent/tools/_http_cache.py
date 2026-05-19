"""On-disk HTTP cache for reproducible benchmark runs.

The web changes between runs; "same task, different policy" ablation
results are noisy unless every fetch returns the same bytes. This
module wraps `requests` calls with a content-addressed cache so that:

  * a `record` run hits the live web and writes responses to disk;
  * a `replay` run reads only from disk and raises if a key is missing;
  * `replay_or_record` falls through to the live web for unknown keys
    and stores the new response (good for incremental capture).

A cache key is `sha256(method, url, sorted(params), body)`. Headers and
the `User-Agent` are deliberately *excluded* from the key — small
header tweaks shouldn't invalidate captured pages.

Storage layout (under cache_root):

    <key[:2]>/<key>.json    # {url, method, status, headers, body_b64, ts}

Activation is process-wide via either `set_cache(...)` or the env var
`BANNA_HTTP_CACHE`. The env spec is `<mode>:<path>`, e.g.
`replay:./.cache/http`. With no env and no explicit set, mode is `off`
and `cached_request` simply calls `requests` directly.

Stats — hits, misses, bytes — are tracked on the singleton and surfaced
to the GAIA runner so per-task JSONL records cache_hits/misses.

Not thread-safe; benchmark runs are single-process single-thread.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


Mode = str  # "off" | "record" | "replay" | "replay_or_record"


class HttpCacheMiss(Exception):
    """Raised in replay mode when a key isn't on disk."""


@dataclass
class CachedResponse:
    """Minimal stand-in for `requests.Response`. Holds what callers actually use."""

    status_code: int
    url: str
    headers: dict[str, str]
    content: bytes
    from_cache: bool = False

    @property
    def text(self) -> str:
        # requests would honor Content-Type charset; we keep it utf-8-best-effort
        # since that's what every site we hit serves and what BeautifulSoup wants.
        return self.content.decode("utf-8", errors="replace")

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            raise RuntimeError(f"HTTP {self.status_code} for {self.url}")

    def json(self) -> Any:
        return json.loads(self.text)


@dataclass
class HttpCache:
    """Content-addressed disk cache. Singleton-friendly but not enforced."""

    root: Path
    mode: Mode = "off"
    hits: int = 0
    misses: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    _live_request: Any = None  # injectable for tests

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        if self.mode != "off":
            self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        data: Any = None,
        json_body: Any = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 20.0,
    ) -> CachedResponse:
        """The one entry point. Behavior depends on `self.mode`."""
        method = method.upper()
        key = self._key(method, url, params, data, json_body)

        if self.mode in ("replay", "replay_or_record"):
            hit = self._load(key)
            if hit is not None:
                self.hits += 1
                self.bytes_read += len(hit.content)
                return hit
            if self.mode == "replay":
                raise HttpCacheMiss(f"no cached response for {method} {url}")
            # replay_or_record: fall through to live fetch + write
        elif self.mode == "off":
            # No cache layer at all — just go live and don't track stats.
            return self._do_live(method, url, params, data, json_body, headers, timeout)

        # record / replay_or_record path: live fetch + store
        resp = self._do_live(method, url, params, data, json_body, headers, timeout)
        self.misses += 1
        self._store(key, resp)
        return resp

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _key(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None,
        data: Any,
        json_body: Any,
    ) -> str:
        h = hashlib.sha256()
        h.update(method.encode())
        h.update(b"\0")
        h.update(url.encode())
        h.update(b"\0")
        if params:
            items = sorted((str(k), str(v)) for k, v in dict(params).items())
            h.update(json.dumps(items, sort_keys=True).encode())
        h.update(b"\0")
        if data is not None:
            if isinstance(data, (dict, list)):
                h.update(json.dumps(_canon(data), sort_keys=True).encode())
            elif isinstance(data, bytes):
                h.update(data)
            else:
                h.update(str(data).encode())
        h.update(b"\0")
        if json_body is not None:
            h.update(json.dumps(_canon(json_body), sort_keys=True).encode())
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def _load(self, key: str) -> CachedResponse | None:
        p = self._path(key)
        if not p.is_file():
            return None
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        body = base64.b64decode(doc.get("body_b64", "").encode())
        return CachedResponse(
            status_code=int(doc.get("status", 0)),
            url=str(doc.get("url", "")),
            headers={str(k): str(v) for k, v in (doc.get("headers") or {}).items()},
            content=body,
            from_cache=True,
        )

    def _store(self, key: str, resp: CachedResponse) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "url": resp.url,
            "status": resp.status_code,
            "headers": resp.headers,
            "body_b64": base64.b64encode(resp.content).decode(),
            "ts": time.time(),
        }
        blob = json.dumps(doc).encode()
        p.write_bytes(blob)
        self.bytes_written += len(blob)

    def _do_live(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None,
        data: Any,
        json_body: Any,
        headers: Mapping[str, str] | None,
        timeout: float,
    ) -> CachedResponse:
        if self._live_request is not None:
            return self._live_request(method, url, params, data, json_body, headers, timeout)
        import requests
        resp = requests.request(
            method, url,
            params=params, data=data, json=json_body,
            headers=headers, timeout=timeout, allow_redirects=True,
        )
        return CachedResponse(
            status_code=resp.status_code,
            url=resp.url,
            headers={k: v for k, v in resp.headers.items()},
            content=resp.content or b"",
            from_cache=False,
        )


def _canon(v: Any) -> Any:
    """Sort dicts recursively so JSON-encoded keys are stable."""
    if isinstance(v, dict):
        return {k: _canon(v[k]) for k in sorted(v)}
    if isinstance(v, list):
        return [_canon(x) for x in v]
    return v


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------


_CACHE: HttpCache | None = None


def get_cache() -> HttpCache:
    """Return the active cache, lazy-initialising from `BANNA_HTTP_CACHE`."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    spec = os.environ.get("BANNA_HTTP_CACHE", "").strip()
    if not spec:
        _CACHE = HttpCache(root=Path("."), mode="off")
        return _CACHE
    mode, _, path = spec.partition(":")
    mode = (mode or "off").strip()
    path = (path or "./.cache/http").strip()
    if mode not in ("off", "record", "replay", "replay_or_record"):
        raise ValueError(f"bad BANNA_HTTP_CACHE mode: {mode!r}")
    _CACHE = HttpCache(root=Path(path), mode=mode)
    return _CACHE


def set_cache(cache: HttpCache | None) -> None:
    """Override the singleton. Pass None to reset for the next get_cache() call."""
    global _CACHE
    _CACHE = cache


def cached_request(
    method: str,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    data: Any = None,
    json: Any = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 20.0,
) -> CachedResponse:
    """Convenience: route a request through the active cache.

    When the cache is in mode `off` (the default), this still works —
    it just bypasses storage and goes live each time, returning a
    CachedResponse so callers have one consistent type to handle.
    """
    return get_cache().fetch(
        method, url,
        params=params, data=data, json_body=json,
        headers=headers, timeout=timeout,
    )
