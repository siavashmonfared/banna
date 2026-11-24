"""Interactive CLI for the agent.

Run as `myagent` / `myAgent` (after `pip install -e .`) or directly via
`python3 -m banna_agent.cli`.

Importing this package is cheap and has no terminal side effects: the
interactive app (and readline, which reconfigures the tty when it loads)
is only imported when `main` runs. Other programs can therefore use pieces
such as `banna_agent.cli.theme.brand_header` inside their own TUI.
"""


def main(argv: list[str] | None = None) -> int:
    from .app import main as _main

    return _main(argv)


__all__ = ["main"]

