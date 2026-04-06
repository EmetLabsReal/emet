"""Live metrics display: chi, gamma, lambda, spectral gap."""

from __future__ import annotations

from typing import Any

from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class MetricsPanel(Widget):
    """Displays chi/gamma/lambda and optional extra metrics."""

    DEFAULT_CSS = """
    MetricsPanel {
        height: auto;
        width: 1fr;
        padding: 1;
    }
    MetricsPanel .metric-row {
        height: 1;
    }
    """

    metrics: reactive[dict[str, Any]] = reactive(dict, always_update=True)

    def compose(self):
        yield Static("", id="metrics-content")

    def watch_metrics(self, value: dict[str, Any]) -> None:
        self._render_metrics(value)

    def _render_metrics(self, m: dict[str, Any]) -> None:
        content = self.query_one("#metrics-content", Static)
        if not m:
            content.update("")
            return

        chi = m.get("chi")
        gamma = m.get("gamma", 0.0)
        lam = m.get("lambda", 0.0)

        chi_color = "#00ff41" if chi is not None and chi < 1.0 else "#ff4444"

        lines = []
        if chi is not None:
            lines.append(f"[bold {chi_color}]χ = {chi:.6f}[/]")
            lines.append("")
        lines.append(f"[dim]γ[/]  {gamma:.8f}")
        lines.append(f"[dim]λ[/]  {lam:.8f}")

        gap = m.get("spectral_gap")
        if gap is not None:
            lines.append(f"[dim]gap[/]  {gap:.6e}")

        for key in ("mass_gap", "transfer_gap", "qber", "key_rate"):
            val = m.get(key)
            if val is not None:
                lines.append(f"[dim]{key}[/]  {val:.6e}")

        content.update("\n".join(lines))

    def set_from_report(self, report: dict[str, Any]) -> None:
        """Update from an emet decision report dict."""
        adv = report.get("advanced_metrics", {})
        self.metrics = {
            "chi": adv.get("chi"),
            "gamma": adv.get("gamma", 0.0),
            "lambda": adv.get("lambda", 0.0),
            "spectral_gap": adv.get("spectral_gap"),
        }
