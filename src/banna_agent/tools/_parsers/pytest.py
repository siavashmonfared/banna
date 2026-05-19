"""Pytest output parser.

Expects pytest invoked with `--tb=short -q` (or default). Walks the
short-traceback section and the summary line.

Best-effort: if we can't parse cleanly we still emit one generic
test_failure ClaimCheck per `FAILED ` summary line so the agent at
least learns *something* failed.
"""
from __future__ import annotations

import re

from . import Failure


_SUMMARY_FAILED = re.compile(r"^FAILED\s+(\S+)(?:\s+-\s+(.*))?$", re.MULTILINE)
_SUMMARY_ERROR = re.compile(r"^ERROR\s+(\S+)(?:\s+-\s+(.*))?$", re.MULTILINE)
# Short-tb format: `path:line: in name` then `>   stmt` then `E   Msg`
_E_LINE = re.compile(r"^E\s{3}(.+)$", re.MULTILINE)


def parse(stdout: str, stderr: str, returncode: int) -> list[Failure]:
    text = (stdout or "") + "\n" + (stderr or "")
    if returncode == 0:
        return []

    fails: dict[str, Failure] = {}
    # Summary section: `FAILED tests/test_foo.py::test_bar - AssertionError: ...`
    for m in _SUMMARY_FAILED.finditer(text):
        nodeid = m.group(1)
        msg = (m.group(2) or "").strip() or "test failed"
        fails[nodeid] = Failure(
            kind="test_failure",
            name=nodeid,
            detail=msg[:240],
            location=_location_from_nodeid(nodeid),
        )
    for m in _SUMMARY_ERROR.finditer(text):
        nodeid = m.group(1)
        msg = (m.group(2) or "").strip() or "test errored during collection/setup"
        fails.setdefault(nodeid, Failure(
            kind="test_failure",
            name=nodeid,
            detail=msg[:240],
            location=_location_from_nodeid(nodeid),
        ))

    if fails:
        return list(fails.values())

    # No FAILED/ERROR summary lines parsed but rc != 0.
    # Try to glean an `E   ...` assertion line; otherwise emit a generic.
    e_lines = _E_LINE.findall(text)
    if e_lines:
        return [Failure(
            kind="test_failure",
            name="pytest:unparsed",
            detail=e_lines[-1].strip()[:240],
        )]
    return [Failure(
        kind="test_failure",
        name="pytest:nonzero_exit",
        detail=f"pytest exited {returncode}; no failure summary detected",
    )]


def _location_from_nodeid(nodeid: str) -> str:
    # nodeid is like "tests/test_x.py::TestY::test_z" — path is the prefix.
    return nodeid.split("::", 1)[0] if "::" in nodeid else nodeid
