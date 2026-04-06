"""Sweep screen — parameter sweep with live DataTable."""

from __future__ import annotations

from typing import Any

import numpy as np
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Label, ProgressBar, Static

from emet.tui.registry import DOMAIN_MAP
from emet.tui.widgets.sweep_table import SweepTable
from emet.tui.workers import SweepRow


class SweepScreen(Screen):
    """Parameter sweep with live-updating DataTable."""

    BINDINGS = [
        ("escape", "pop_screen", "Back"),
        ("q", "pop_screen", "Back"),
    ]

    def __init__(
        self,
        domain_key: str,
        params: dict[str, Any],
    ) -> None:
        super().__init__()
        self._domain_key = domain_key
        self._params = params
        self._sweep_param: str | None = None
        self._sweep_values: list[float] = []

    def compose(self) -> ComposeResult:
        desc = DOMAIN_MAP.get(self._domain_key)
        domain_name = desc.name if desc else self._domain_key
        yield Static(
            f"sweep [dim]— {domain_name}[/]",
            id="title-bar",
        )
        with Vertical(id="sweep-grid"):
            yield Static("", id="sweep-info")
            yield SweepTable(id="sweep-table")
            yield ProgressBar(id="sweep-progress", total=100, show_eta=True)
        yield Static("[dim]esc[/] back  [dim]q[/] quit", id="status-bar")

    def on_mount(self) -> None:
        desc = DOMAIN_MAP.get(self._domain_key)
        if desc is None:
            return

        for spec in desc.params:
            if spec.sweepable and spec.type in (float, int):
                self._sweep_param = spec.name
                if spec.min_val is not None and spec.max_val is not None:
                    self._sweep_values = np.linspace(
                        spec.min_val, spec.max_val, 20,
                    ).tolist()
                break

        if self._sweep_param is None:
            self.query_one("#sweep-info", Static).update(
                "No sweepable parameter found"
            )
            return

        info = self.query_one("#sweep-info", Static)
        info.update(
            f"{self._sweep_param}: "
            f"{self._sweep_values[0]:.4f} -> {self._sweep_values[-1]:.4f} "
            f"({len(self._sweep_values)} points)"
        )

        self.query_one("#sweep-progress", ProgressBar).update(
            total=len(self._sweep_values)
        )

        self.run_worker(self._sweep_worker(), name="sweep", thread=True)

    async def _sweep_worker(self) -> None:
        from emet.tui.workers import run_sweep

        def on_row(row: SweepRow) -> None:
            table = self.query_one("#sweep-table", SweepTable)
            self.app.call_from_thread(table.add_sweep_row, row)

        def on_progress(current: int, total: int) -> None:
            bar = self.query_one("#sweep-progress", ProgressBar)
            self.app.call_from_thread(bar.update, progress=current)

        run_sweep(
            domain_key=self._domain_key,
            base_params=self._params,
            sweep_param=self._sweep_param,
            sweep_values=self._sweep_values,
            on_row=on_row,
            on_progress=on_progress,
        )

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
