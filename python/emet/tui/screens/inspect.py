"""Inspect screen — detail view of last decision result."""

from __future__ import annotations

from typing import Any

import numpy as np
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Static

from emet.tui.registry import DOMAIN_MAP
from emet.tui.widgets.matrix_view import MatrixView
from emet.tui.widgets.metrics_panel import MetricsPanel
from emet.tui.widgets.regime_badge import RegimeBadge


class InspectScreen(Screen):
    """Detailed inspection of a decision result."""

    BINDINGS = [
        ("escape", "pop_screen", "Back"),
        ("q", "pop_screen", "Back"),
    ]

    def __init__(
        self,
        domain_key: str,
        params: dict[str, Any],
        report: dict[str, Any],
    ) -> None:
        super().__init__()
        self._domain_key = domain_key
        self._params = params
        self._report = report

    def compose(self) -> ComposeResult:
        desc = DOMAIN_MAP.get(self._domain_key)
        domain_name = desc.name if desc else self._domain_key
        yield Static(
            f"inspect [dim]— {domain_name}[/]",
            id="title-bar",
        )
        with Vertical(id="inspect-grid"):
            with Vertical():
                yield RegimeBadge(id="insp-regime")
                yield MetricsPanel(id="insp-metrics")
                yield Static("", id="insp-partition")
                yield Static("", id="insp-detail")
            with Vertical():
                yield MatrixView(id="insp-heff")
                yield MatrixView(id="insp-full")
        yield Static("[dim]esc[/] back  [dim]q[/] quit", id="status-bar")

    def on_mount(self) -> None:
        report = self._report
        regime = report.get("regime", "unknown")
        valid = report.get("valid", False)

        self.query_one("#insp-regime", RegimeBadge).regime = regime
        self.query_one("#insp-regime", RegimeBadge).licensed = valid
        self.query_one("#insp-metrics", MetricsPanel).set_from_report(report)

        pp = report.get("partition_profile", {})
        self.query_one("#insp-partition", Static).update(
            f"retained: {pp.get('retained_dimension', '?')}  "
            f"omitted: {pp.get('omitted_dimension', '?')}  "
            f"total: {pp.get('total_dimension', '?')}"
        )

        adv = report.get("advanced_metrics", {})
        lines = []
        reason = report.get("reason_code", "")
        if reason:
            lines.append(f"reason: {reason}")
        kahan = adv.get("kahan", {})
        if kahan.get("certified"):
            margin = kahan.get("margin")
            lines.append(f"kahan margin: {margin:.6e}" if margin else "kahan: certified")
        self.query_one("#insp-detail", Static).update("\n".join(lines))

        self.run_worker(self._compute_matrices(), name="matrices", thread=True)

    async def _compute_matrices(self) -> None:
        from emet.tui.registry import get_domain

        desc = get_domain(self._domain_key)
        try:
            H, retained, omitted = desc.build_fn(**self._params)
        except Exception:
            return

        P = sorted(retained)
        Q = sorted(omitted)

        H_eff = None
        try:
            H_PP = H[np.ix_(P, P)]
            H_PQ = H[np.ix_(P, Q)]
            H_QQ = H[np.ix_(Q, Q)]
            H_QP = H[np.ix_(Q, P)]
            H_eff = H_PP - H_PQ @ np.linalg.solve(H_QQ, H_QP)
        except Exception:
            pass

        def update_ui() -> None:
            if H_eff is not None:
                heff_view = self.query_one("#insp-heff", MatrixView)
                heff_view.title_text = "H_eff (Schur complement)"
                heff_view.matrix = H_eff

            full_view = self.query_one("#insp-full", MatrixView)
            full_view.title_text = "H (full)"
            full_view.matrix = H

        self.app.call_from_thread(update_ui)

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
