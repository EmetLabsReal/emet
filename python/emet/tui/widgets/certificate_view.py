"""Certificate display widget — SHA-256 seal + license badge."""

from __future__ import annotations

from typing import Any

from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class CertificateView(Widget):
    """Displays an EmetCertificate's seal and key fields."""

    DEFAULT_CSS = """
    CertificateView {
        height: auto;
        width: 1fr;
        padding: 1;
        border: solid #333333;
    }
    """

    certificate: reactive[dict[str, Any] | None] = reactive(None)

    def compose(self):
        yield Static("", id="cert-content")

    def watch_certificate(self, value: dict[str, Any] | None) -> None:
        self._update_display(value)

    def _update_display(self, cert: dict[str, Any] | None) -> None:
        content = self.query_one("#cert-content", Static)
        if cert is None:
            content.update("")
            return

        seal = cert.get("seal", "")
        self._full_seal = seal
        regime = cert.get("regime", "unknown")
        licensed = cert.get("licensed", False)
        kahan = cert.get("kahan_certified", False)

        from emet.tui.widgets.regime_badge import REGIME_COLORS
        color = REGIME_COLORS.get(regime, "#888888")

        lic_mark = "[bold #00ff41]✓[/]" if licensed else "[bold #ff4444]✗[/]"
        kahan_mark = "[#00ff41]✓[/]" if kahan else "[#ff4444]✗[/]"

        lines = [
            f"[bold]{seal[:8]}[/]  [dim]ctrl+y to copy[/]",
            f"  {lic_mark} [{color}]{regime.upper()}[/]  {kahan_mark} [dim]kahan[/]",
        ]

        content.update("\n".join(lines))

    def set_from_certificate(self, cert: Any) -> None:
        """Set from an EmetCertificate dataclass instance."""
        from emet.certificate import to_json
        import json
        self.certificate = json.loads(to_json(cert))
