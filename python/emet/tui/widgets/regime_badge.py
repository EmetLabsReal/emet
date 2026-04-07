"""Color-coded regime indicator."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

REGIME_COLORS = {
    "subcritical": "#00ff41",
    "pre_admissible": "#ffaa00",
    "supercritical": "#ff4444",
    "C": "#00ff41",
    "B": "#00ff41",
    "A": "#ff4444",
}


class RegimeBadge(Widget):
    """Displays the current regime as a single colored line."""

    DEFAULT_CSS = """
    RegimeBadge {
        height: 1;
        width: 1fr;
    }
    """

    regime: reactive[str] = reactive("unknown")
    licensed: reactive[bool] = reactive(False)

    def compose(self):
        yield Static("", classes="badge-label")

    def watch_regime(self, value: str) -> None:
        self._update_display()

    def watch_licensed(self, value: bool) -> None:
        self._update_display()

    def _update_display(self) -> None:
        label = self.query_one(".badge-label", Static)
        color = REGIME_COLORS.get(self.regime, "#555555")
        name = self.regime.upper().replace("_", "-")
        if self.regime == "unknown":
            label.update("[dim]--[/]")
        elif self.licensed:
            label.update(f"[{color}]{name} (licensed)[/]")
        else:
            label.update(f"[{color}]{name}[/]")
