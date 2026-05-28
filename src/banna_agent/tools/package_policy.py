"""Trusted-package allowlist for on-demand installs in the Docker sandbox.

When model-emitted Python under the Docker sandbox raises
``ModuleNotFoundError``, the backend consults a :class:`PackagePolicy` to
decide whether the missing module may be installed:

  * an **allowlisted** import (``import_name -> "dist==version"`` pin) is
    installed with no human in the loop — works for every policy, including
    headless GAIA/batch runs (which simply never construct a policy);
  * anything else is routed to an approval callback and, if approved, kept
    for the rest of the session via :meth:`approve_session`.

The allowlist is a plain ``import_name -> spec`` map. ``spec`` is normally a
string (``"pandas==2.2.2"``); a dict ``{"spec": ..., "sha256": ...}`` is also
accepted so hash-pinning can be layered on later without a schema break.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Curated, version-pinned default allowlist: the common data/scientific-Python
# stack, trusted enough to install with no prompt under the docker sandbox.
# Keyed by *import* name (the thing that appears in ``ModuleNotFoundError``);
# the value is the *pip* spec, which handles import-vs-distribution name
# mismatches (``cv2`` -> ``opencv-python-headless``, ``sklearn`` ->
# ``scikit-learn``, ``PIL`` -> ``pillow``, ``bs4`` -> ``beautifulsoup4`` …).
# Pins are chosen to coexist in a single ``pip install``. A user's
# ``[packages]`` config is merged on top of this and overrides any entry.
DEFAULT_ALLOWLIST: dict[str, str] = {
    "numpy": "numpy==2.1.3",
    "pandas": "pandas==2.2.3",
    "scipy": "scipy==1.14.1",
    "sympy": "sympy==1.13.3",
    "matplotlib": "matplotlib==3.9.2",
    "sklearn": "scikit-learn==1.5.2",
    "statsmodels": "statsmodels==0.14.4",
    "networkx": "networkx==3.4.2",
    "PIL": "pillow==11.0.0",
    "cv2": "opencv-python-headless==4.10.0.84",
    "bs4": "beautifulsoup4==4.12.3",
    "lxml": "lxml==5.3.0",
    "requests": "requests==2.32.3",
    "yaml": "pyyaml==6.0.2",
    "openpyxl": "openpyxl==3.1.5",
    "tabulate": "tabulate==0.9.0",
    "dateutil": "python-dateutil==2.9.0.post0",
    "pytz": "pytz==2024.2",
    "CoolProp": "CoolProp==6.6.0",  # fluid/thermophysical properties library
}


def default_allowlist() -> dict[str, str]:
    """A fresh copy of the built-in trusted allowlist (safe to mutate)."""
    return dict(DEFAULT_ALLOWLIST)


def _spec_of(value: Any) -> str | None:
    """Extract the pip spec from an allowlist value (str or {spec,...} dict)."""
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        spec = value.get("spec")
        return spec if isinstance(spec, str) and spec else None
    return None


@dataclass
class PackagePolicy:
    """Decides whether a missing import may be installed, and to what pin.

    ``allowlist`` is the persisted, user-maintained map. ``session_pins``
    holds packages a human approved during this session (not persisted unless
    the CLI also writes them back to the allowlist).
    """

    allowlist: dict[str, Any] = field(default_factory=dict)
    session_pins: dict[str, str] = field(default_factory=dict)

    def resolve(self, import_name: str) -> str | None:
        """Return the pip spec for an allowed import, else ``None``.

        Session approvals take precedence over the static allowlist.
        """
        if import_name in self.session_pins:
            return self.session_pins[import_name]
        return _spec_of(self.allowlist.get(import_name))

    def is_allowed(self, import_name: str) -> bool:
        return self.resolve(import_name) is not None

    def approve_session(self, import_name: str, pip_spec: str) -> None:
        """Record an approval valid for the remainder of this session."""
        self.session_pins[import_name] = pip_spec
