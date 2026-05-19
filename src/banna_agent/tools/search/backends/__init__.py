"""Concrete web-search backends -- each implements WebSearchBackend."""

from .yacy import YaCyBackend


__all__ = [
    "YaCyBackend",
]
