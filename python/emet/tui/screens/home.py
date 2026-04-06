"""Home screen — domain picker, live params, decide, seal."""

from __future__ import annotations

from typing import Any

import numpy as np
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Static

from emet.tui.registry import DOMAIN_MAP, DOMAINS, domain_keys
from emet.tui.widgets.certificate_view import CertificateView
from emet.tui.widgets.domain_list import DomainList
from emet.tui.widgets.kahan_panel import KahanPanel
from emet.tui.widgets.metrics_panel import MetricsPanel
from emet.tui.widgets.param_panel import ParamPanel
from emet.tui.widgets.regime_badge import RegimeBadge
from emet.tui.widgets.torus_plot import TorusPlot


class HomeScreen(Screen):

    BINDINGS = [
        ("ctrl+s", "seal", "Seal"),
        ("ctrl+y", "copy_seal", "Copy seal"),
        ("tab", "sweep", "Sweep"),
        ("ctrl+c", "certify_screen", "Certify"),
        ("escape", "blur", "Unfocus"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._H: np.ndarray | None = None
        self._retained: list[int] | None = None
        self._omitted: list[int] | None = None
        self._report: dict[str, Any] | None = None
        self._domain_key: str = "torus"

    def compose(self) -> ComposeResult:
        yield Static("emet", id="title-bar")
        with Horizontal(id="main-layout"):
            with Vertical(id="left-rail"):
                yield Static("[dim]domains[/]", id="domain-header")
                yield DomainList(id="domain-list")
            with Vertical(id="center-col"):
                yield ParamPanel(id="param-panel")
                yield Static("", id="run-hint")
                yield RegimeBadge(id="regime-badge")
                yield MetricsPanel(id="metrics-panel")
                yield KahanPanel(id="kahan-panel")
                yield CertificateView(id="cert-view")
            with Vertical(id="right-col"):
                yield TorusPlot(id="torus-plot")
        yield Static(
            "[dim]enter[/] decide  "
            "[dim]ctrl+s[/] seal  "
            "[dim]ctrl+y[/] copy  "
            "[dim]tab[/] sweep  "
            "[dim]esc[/] unfocus  "
            "[dim]q[/] quit",
            id="status-bar",
        )

    def on_mount(self) -> None:
        # Select default domain
        self._select_domain("torus")
        # Highlight first item
        dl = self.query_one("#domain-list", DomainList)
        dl.highlighted = 0

    def on_option_list_option_selected(self, event: DomainList.OptionSelected) -> None:
        """User picked a domain from the list."""
        key = str(event.option.id)
        self._select_domain(key)

    def _select_domain(self, key: str) -> None:
        desc = DOMAIN_MAP.get(key)
        if desc is None:
            return
        self._domain_key = key
        panel = self.query_one("#param-panel", ParamPanel)
        panel.domain = desc
        # Reset result displays
        self._report = None
        self._H = None
        self.query_one("#regime-badge", RegimeBadge).regime = "unknown"
        self.query_one("#metrics-panel", MetricsPanel).metrics = {}
        self.query_one("#kahan-panel", KahanPanel).certified = None
        self.query_one("#cert-view", CertificateView).certificate = None
        self.query_one("#run-hint", Static).update("")
        # Update torus visibility
        torus = self.query_one("#torus-plot", TorusPlot)
        if desc.has_torus_plot and desc.beta_param:
            torus.display = True
            for p in desc.params:
                if p.name == desc.beta_param:
                    torus.beta = float(p.default)
                    break
        else:
            torus.display = False

    def on_param_panel_params_changed(self, event: ParamPanel.ParamsChanged) -> None:
        """Live-update torus when params change."""
        desc = DOMAIN_MAP.get(self._domain_key)
        if desc and desc.has_torus_plot and desc.beta_param:
            beta_val = event.params.get(desc.beta_param)
            if beta_val is not None:
                try:
                    self.query_one("#torus-plot", TorusPlot).beta = float(beta_val)
                except (ValueError, TypeError):
                    pass

    def on_input_submitted(self, event: Any) -> None:
        """Enter in any param field runs the pipeline."""
        self._run()
        self.set_focus(None)

    def action_blur(self) -> None:
        self.set_focus(None)

    def _run(self) -> None:
        desc = DOMAIN_MAP.get(self._domain_key)
        if desc is None:
            return
        panel = self.query_one("#param-panel", ParamPanel)
        params = panel.get_params()
        self.query_one("#run-hint", Static).update("")
        self.run_worker(self._pipeline(desc, params), name="run", thread=True)

    async def _pipeline(self, desc: Any, params: dict[str, Any]) -> None:
        try:
            H, ret, omit = desc.build_fn(**params)
        except Exception as e:
            self.app.call_from_thread(
                self.query_one("#run-hint", Static).update,
                f"[red]{e}[/]",
            )
            return

        self._H = H
        self._retained = ret
        self._omitted = omit

        try:
            import emet
            report = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
        except Exception as e:
            self.app.call_from_thread(
                self.query_one("#run-hint", Static).update,
                f"[red]decide: {e}[/]",
            )
            return

        self._report = report
        self.app.call_from_thread(self._show, report, params)

    def _show(self, report: dict[str, Any], params: dict[str, Any]) -> None:
        regime = report.get("regime", "unknown")
        valid = report.get("valid", False)
        adv = report.get("advanced_metrics") or {}

        # Regime badge
        badge = self.query_one("#regime-badge", RegimeBadge)
        badge.regime = regime
        badge.licensed = valid

        # Metrics
        metrics = self.query_one("#metrics-panel", MetricsPanel)
        metrics.set_from_report(report)

        # Kahan
        kahan = self.query_one("#kahan-panel", KahanPanel)
        kahan.set_from_report(report)

        # Partition info
        pp = report.get("partition_profile", {})
        self.query_one("#run-hint", Static).update(
            f"{pp.get('retained_dimension', '?')} retained  "
            f"{pp.get('omitted_dimension', '?')} omitted  "
            f"{pp.get('total_dimension', '?')} total"
        )

        # Update torus from chi if no beta_param
        desc = DOMAIN_MAP.get(self._domain_key)
        chi = adv.get("chi")
        if desc and desc.has_torus_plot and desc.beta_param:
            beta_val = params.get(desc.beta_param)
            if beta_val is not None:
                self.query_one("#torus-plot", TorusPlot).beta = float(beta_val)
        elif chi is not None:
            torus = self.query_one("#torus-plot", TorusPlot)
            torus.display = True
            if chi < 1.0:
                torus.beta = 2.0 + (1.0 - chi) * 2.0
            else:
                torus.beta = max(0.5, 2.0 - min(chi - 1.0, 1.5))

    def action_seal(self) -> None:
        if self._report is None or self._H is None:
            self.query_one("#run-hint", Static).update("[red]run first[/]")
            return
        self.run_worker(self._seal(), name="seal", thread=True)

    async def _seal(self) -> None:
        from emet.certificate import certify

        domain = self._domain_key
        panel = self.query_one("#param-panel", ParamPanel)
        params = panel.get_params()

        beta = None
        desc = DOMAIN_MAP.get(domain)
        if desc and desc.beta_param and desc.beta_param in params:
            beta = float(params[desc.beta_param])

        try:
            cert = certify(
                self._H, self._retained, self._omitted,
                self._report, domain=domain, beta=beta, params=params,
            )
        except Exception as e:
            self.app.call_from_thread(
                self.query_one("#run-hint", Static).update,
                f"[red]seal: {e}[/]",
            )
            return

        def show() -> None:
            cert_view = self.query_one("#cert-view", CertificateView)
            cert_view.set_from_certificate(cert)

        self.app.call_from_thread(show)

    def action_copy_seal(self) -> None:
        cert_view = self.query_one("#cert-view", CertificateView)
        seal = getattr(cert_view, "_full_seal", None)
        if not seal:
            return
        try:
            import subprocess
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=seal.encode(), check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            try:
                subprocess.run(
                    ["pbcopy"],
                    input=seal.encode(), check=True,
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        self.query_one("#run-hint", Static).update(
            f"[dim]copied {seal[:8]}[/]"
        )

    def action_sweep(self) -> None:
        from emet.tui.screens.sweep import SweepScreen
        panel = self.query_one("#param-panel", ParamPanel)
        params = panel.get_params()
        self.app.push_screen(SweepScreen(self._domain_key, params))

    def action_certify_screen(self) -> None:
        from emet.tui.screens.certify import CertifyScreen
        panel = self.query_one("#param-panel", ParamPanel)
        params = panel.get_params()
        self.app.push_screen(
            CertifyScreen(self._domain_key, params, self._report)
        )
