"""Emet TUI application."""

from __future__ import annotations

from textual.app import App

from emet.tui.screens.home import HomeScreen


class EmetApp(App):
    """Model reduction certification."""

    CSS_PATH = "emet.tcss"

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    SCREENS = {}

    def on_mount(self) -> None:
        self.push_screen(HomeScreen())
