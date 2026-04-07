"""Kahan envelope certification display."""

from __future__ import annotations

from typing import Any

from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class KahanPanel(Widget):
    """Shows Kahan IEEE 754 certification status and margin."""

    DEFAULT_CSS = """
    KahanPanel {
        height: auto;
        width: 1fr;
        padding: 1;
        border: solid #333333;
    }
    """

    certified: reactive[bool | None] = reactive(None)
    margin: reactive[float | None] = reactive(None)

    def compose(self):
        yield Static("", id="kahan-content")

    def watch_certified(self, value: bool | None) -> None:
        self._update()

    def watch_margin(self, value: float | None) -> None:
        self._update()

    def _update(self) -> None:
        content = self.query_one("#kahan-content", Static)
        if self.certified is None:
            content.update("")
            return

        if self.certified:
            status = "[bold #00ff41]CERTIFIED[/]"
        else:
            status = "[bold #ff4444]NOT CERTIFIED[/]"

        lines = [f"Kahan envelope: {status}"]
        if self.margin is not None:
            lines.append(f"Margin: {self.margin:.6e}")
        content.update("\n".join(lines))

    def set_from_report(self, report: dict[str, Any]) -> None:
        """Update from an emet decision report dict."""
        adv = report.get("advanced_metrics", {})
        kahan = adv.get("kahan", {})
        self.certified = kahan.get("certified", False)
        self.margin = kahan.get("margin")
