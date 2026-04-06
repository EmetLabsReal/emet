"""Emet terminal user interface."""

from __future__ import annotations


def launch_tui() -> None:
    """Launch the Emet TUI application."""
    from emet.tui.app import EmetApp
    app = EmetApp()
    app.run()
