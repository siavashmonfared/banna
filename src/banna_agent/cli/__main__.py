"""Allows `python3 -m banna_agent.cli` without installing the package."""

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
