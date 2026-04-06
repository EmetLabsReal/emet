"""Certify screen — full certification pipeline with SHA-256 seal."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Static

from emet.tui.registry import DOMAIN_MAP
from emet.tui.widgets.certificate_view import CertificateView
from emet.tui.widgets.kahan_panel import KahanPanel
from emet.tui.widgets.matrix_view import MatrixView
from emet.tui.widgets.metrics_panel import MetricsPanel
from emet.tui.widgets.regime_badge import RegimeBadge


class CertifyScreen(Screen):
    """Run decide + certify pipeline, display sealed certificate."""

    BINDINGS = [
        ("escape", "pop_screen", "Back"),
        ("q", "pop_screen", "Back"),
    ]

    def __init__(
        self,
        domain_key: str,
        params: dict[str, Any],
        last_report: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._domain_key = domain_key
        self._params = params
        self._last_report = last_report

    def compose(self) -> ComposeResult:
        desc = DOMAIN_MAP.get(self._domain_key)
        domain_name = desc.name if desc else self._domain_key
        yield Static(
            f"certify [dim]— {domain_name}[/]",
            id="title-bar",
        )
        with Vertical(id="certify-grid"):
            with Vertical():
                yield RegimeBadge(id="cert-regime")
                yield MetricsPanel(id="cert-metrics")
                yield KahanPanel(id="cert-kahan")
            with Vertical():
                yield Static("[dim]computing...[/]", id="cert-status")
                yield CertificateView(id="cert-view")
                yield MatrixView(id="cert-matrix")
        yield Static("[dim]esc[/] back  [dim]q[/] quit", id="status-bar")

    def on_mount(self) -> None:
        self.run_worker(self._certify_worker(), name="certify", thread=True)

    async def _certify_worker(self) -> None:
        from emet.tui.workers import run_certify

        try:
            result = run_certify(self._domain_key, self._params)
        except Exception as e:
            self.app.call_from_thread(
                self._show_error, str(e),
            )
            return

        self.app.call_from_thread(self._show_result, result)

    def _show_error(self, msg: str) -> None:
        self.query_one("#cert-status", Static).update(f"error: {msg}")

    def _show_result(self, result: dict[str, Any]) -> None:
        report = result["report"]
        cert = result["certificate"]
        h_eff = result.get("H_eff")

        self.query_one("#cert-status", Static).update("sealed")

        regime = report.get("regime", "unknown")
        valid = report.get("valid", False)
        self.query_one("#cert-regime", RegimeBadge).regime = regime
        self.query_one("#cert-regime", RegimeBadge).licensed = valid
        self.query_one("#cert-metrics", MetricsPanel).set_from_report(report)
        self.query_one("#cert-kahan", KahanPanel).set_from_report(report)

        cert_view = self.query_one("#cert-view", CertificateView)
        cert_view.set_from_certificate(cert)

        if h_eff is not None:
            mat_view = self.query_one("#cert-matrix", MatrixView)
            mat_view.title_text = "H_eff"
            mat_view.matrix = h_eff

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
