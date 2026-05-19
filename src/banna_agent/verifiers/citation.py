"""CitationVerifier — claims must overlap their cited evidence.

For every Claim with at least one supporting Evidence id, score the
content overlap between the claim text and each cited Evidence's
content. The verdict reflects whether the claim is *defensible* given
the evidence currently in state.

Phase 5 upgrade — three things the old version couldn't do:

  1. **Numeric support check.** A claim like "The cube root of 1331 is
     11" can hit a topical Jaccard threshold against a page that
     mentions cubes and 11 of *something else*. The verifier now
     extracts numeric literals (≥ 2 digits) from the claim and requires
     each to appear in the evidence (either as a substring or within a
     small relative tolerance for floats). A claim with no numbers
     ignores this check.

  2. **Refetch via the project HTTP cache.** Empty evidence with a URL
     source is refetched through `tools._http_cache.cached_request`
     when `use_http_cache=True`. Record/replay modes flow through
     naturally.

  3. **Failure taxonomy.** ``meta.error_class`` distinguishes:
       - "broken_citation"    — evidence id not in state
       - "tool_error"         — refetch raised / returned non-OK
       - "evidence_not_found" — fetched OK but claim isn't supported
     Plus existing "format/coverage" classes are unchanged. The
     rejection-deposit table keys on these.

v1's Jaccard threshold (``min_overlap``) is still here, additive with
the numeric check. Both must pass for "ok"; either failing yields a
"fail" with ``evidence_not_found``. Unreachable evidence yields "warn"
with ``tool_error`` so verifier-retry can downweight without giving
up.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..core.state import AgentState
from .base import ClaimCheck


# Common stopwords + numerics we don't want dominating Jaccard scores.
_STOP = frozenset(
    "a an the of in on at to for from by with as is are was were be been being "
    "and or but if while because so than then this that these those it its their "
    "his her our your my we you they them us i me him she he do does did done has "
    "have had not no yes which what who whom where when how why".split()
)


# Numbers worth checking: 2+ digits so we don't trip on stray "1"s.
_NUMBER_RE = re.compile(r"\b-?\d+(?:\.\d+)?\b")


def _tokenize(text: str, *, min_len: int = 3) -> set[str]:
    """Lowercase, alpha-numeric tokens, length ≥ min_len, with stopword strip."""
    if not text:
        return set()
    raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", text.lower())
    return {t for t in raw if len(t) >= min_len and t not in _STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _extract_numbers(text: str) -> list[str]:
    """Return numeric literals worth checking (≥2 digits in the integer part).

    Single-digit numbers are too noisy as substrings ("5" appears in
    almost any longish page); only check numbers with at least two
    digits in the integer portion.
    """
    if not text:
        return []
    out: list[str] = []
    for m in _NUMBER_RE.finditer(text):
        tok = m.group()
        # Strip leading minus for the digit-count threshold.
        intpart = tok.split(".", 1)[0].lstrip("-")
        if len(intpart) >= 2:
            out.append(tok)
    return out


def _number_supported(claim_num: str, evidence: str, *, rel_tol: float = 1e-3) -> bool:
    """Is `claim_num` supported by `evidence`?

    Exact substring is the cheap path. Numeric tolerance handles
    "12.345" supported by "12.3450001" or "1234" supported by "1,234"
    (after we strip the comma).
    """
    if claim_num in evidence:
        return True
    ev_compact = evidence.replace(",", "")
    if claim_num in ev_compact:
        return True
    try:
        v = float(claim_num)
    except ValueError:
        return False
    for tok in _NUMBER_RE.findall(ev_compact):
        try:
            ev = float(tok)
        except ValueError:
            continue
        if v == 0:
            if abs(ev) < 1e-9:
                return True
        elif abs(ev - v) <= max(1e-9, abs(v) * rel_tol):
            return True
    return False


def _default_cache_fetcher(url: str) -> str:
    """Default URL→text fetcher: use the project HTTP cache.

    Raises on non-OK status so the verifier can tag tool_error. Used
    only when the caller opts in via ``use_http_cache=True``.
    """
    from ..tools._http_cache import cached_request
    resp = cached_request(
        "GET", url,
        headers={"User-Agent": "banna_agent/0.1", "Accept": "*/*"},
        timeout=20.0,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}")
    return resp.text


@dataclass
class CitationVerifier:
    """Score claim ↔ evidence overlap.

    Args:
      min_overlap        — Jaccard threshold for "ok".
      require_number_match — when True (default), every numeric literal
                           in the claim must appear in some piece of
                           evidence; otherwise the claim fails with
                           ``evidence_not_found``. Set False for v1
                           behavior (overlap-only).
      fetch_url          — optional callable to refetch empty URL evidence.
                           Should return content as a string and raise
                           on network/HTTP failure (so the verifier can
                           tag ``tool_error``). When None, empty
                           evidence yields a warn ("can't grade").
    """

    name: str = "citation"
    min_overlap: float = 0.10
    require_number_match: bool = True
    fetch_url: Callable[[str], str] | None = None

    def check(
        self,
        state: AgentState,
        proposed_answer: str | None = None,
    ) -> list[ClaimCheck]:
        out: list[ClaimCheck] = []
        ev_by_id = {ev.evidence_id: ev for ev in state.evidence}

        for cl in state.claims:
            if not cl.supports:
                out.append(ClaimCheck(
                    claim_id=cl.claim_id,
                    verifier_name=self.name,
                    verdict="skip",
                    detail="claim has no supporting evidence (coverage check)",
                ))
                continue

            claim_tokens = _tokenize(cl.text)
            if not claim_tokens:
                out.append(ClaimCheck(
                    claim_id=cl.claim_id,
                    verifier_name=self.name,
                    verdict="skip",
                    detail="claim text has no scoreable tokens",
                ))
                continue

            claim_numbers = _extract_numbers(cl.text) if self.require_number_match else []

            best_overlap = 0.0
            best_source = ""
            missing_refs: list[str] = []
            tool_errors: list[str] = []
            empty_present = False
            graded_any = False
            # Aggregates everything we actually graded against this claim
            # (originals + refetched). Used for the numeric-support
            # check so we don't have to re-read state.evidence (which
            # never sees refetched bodies).
            graded_contents: list[str] = []

            for ev_id in cl.supports:
                ev = ev_by_id.get(ev_id)
                if ev is None:
                    missing_refs.append(ev_id)
                    continue
                content = ev.content or ""
                if not content.strip() and self.fetch_url and ev.source.startswith(("http://", "https://")):
                    try:
                        content = self.fetch_url(ev.source) or ""
                    except Exception as exc:
                        tool_errors.append(f"{ev.source}: {type(exc).__name__}: {exc}")
                        continue
                if not content.strip():
                    empty_present = True
                    continue

                graded_any = True
                graded_contents.append(content)
                ov = _jaccard(claim_tokens, _tokenize(content))
                if ov > best_overlap:
                    best_overlap = ov
                    best_source = ev.source

            # Verdicts, in precedence order.

            if missing_refs:
                out.append(ClaimCheck(
                    claim_id=cl.claim_id,
                    verifier_name=self.name,
                    verdict="fail",
                    detail=f"broken citation(s): {', '.join(missing_refs[:3])}",
                    meta={
                        "missing": missing_refs,
                        "error_class": "broken_citation",
                        "nudge": (
                            f"You cited evidence IDs that don't exist: "
                            f"{missing_refs[:3]}. Only use evidence_ids "
                            f"returned by your prior `search` / `read_url` "
                            f"/ `read_file` tool calls — copy them from "
                            f"the tool's JSON response. If you don't have "
                            f"a real citation, omit `evidence_ids`."
                        ),
                    },
                ))
                continue

            if not graded_any and tool_errors:
                out.append(ClaimCheck(
                    claim_id=cl.claim_id,
                    verifier_name=self.name,
                    verdict="warn",
                    detail=f"all cited URLs unreachable; first: {tool_errors[0][:160]}",
                    meta={"error_class": "tool_error", "errors": tool_errors[:3]},
                ))
                continue

            if not graded_any and empty_present:
                out.append(ClaimCheck(
                    claim_id=cl.claim_id,
                    verifier_name=self.name,
                    verdict="warn",
                    detail="all cited evidence has empty content (couldn't grade)",
                    meta={"error_class": "evidence_not_found"},
                ))
                continue

            # We have at least one piece of content to grade against.
            # Check claim numbers against the union of everything we
            # actually graded (so a number can be supported by a
            # different citation than the best Jaccard one).
            missing_nums: list[str] = []
            if claim_numbers and graded_contents:
                all_content = " \n ".join(graded_contents)
                missing_nums = [n for n in claim_numbers if not _number_supported(n, all_content)]

            overlap_ok = best_overlap >= self.min_overlap
            numbers_ok = not missing_nums

            if overlap_ok and numbers_ok:
                out.append(ClaimCheck(
                    claim_id=cl.claim_id,
                    verifier_name=self.name,
                    verdict="ok",
                    detail=f"overlap={best_overlap:.3f} (>= {self.min_overlap})"
                           + (f", numbers={claim_numbers} all supported" if claim_numbers else ""),
                    score=best_overlap,
                    meta={"best_source": best_source, "overlap": best_overlap,
                          "claim_numbers": claim_numbers},
                ))
            else:
                reasons = []
                if not overlap_ok:
                    reasons.append(f"overlap={best_overlap:.3f} < {self.min_overlap}")
                if missing_nums:
                    reasons.append(f"numbers not in evidence: {missing_nums}")
                nudge_parts: list[str] = []
                if missing_nums:
                    nudge_parts.append(
                        f"Your answer/claim references {missing_nums} but "
                        f"those values don't appear in the cited evidence."
                    )
                if not overlap_ok:
                    nudge_parts.append(
                        f"Cited evidence has only {best_overlap:.2f} token "
                        f"overlap with your claim — the source likely doesn't "
                        f"actually support it."
                    )
                nudge_parts.append(
                    "Re-run a `search` or `read_url` to find a source that "
                    "explicitly contains the value, then cite the new "
                    "evidence_id from that call. Or revise the answer to "
                    "match what your evidence actually says."
                )
                out.append(ClaimCheck(
                    claim_id=cl.claim_id,
                    verifier_name=self.name,
                    verdict="fail",
                    detail="; ".join(reasons),
                    score=best_overlap,
                    meta={
                        "best_source": best_source,
                        "overlap": best_overlap,
                        "missing_numbers": missing_nums,
                        "error_class": "evidence_not_found",
                        "nudge": " ".join(nudge_parts),
                    },
                ))

        return out


def default_citation_verifier(
    *,
    use_http_cache: bool = True,
    min_overlap: float = 0.10,
    require_number_match: bool = True,
) -> CitationVerifier:
    """Build a CitationVerifier wired to refetch URL evidence via the cache.

    `use_http_cache=False` reverts to v1 behavior: empty URL evidence
    yields a warn, never refetched.
    """
    return CitationVerifier(
        min_overlap=min_overlap,
        require_number_match=require_number_match,
        fetch_url=_default_cache_fetcher if use_http_cache else None,
    )
