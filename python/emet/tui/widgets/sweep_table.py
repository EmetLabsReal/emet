"""DataTable for sweep results with regime-colored rows."""

from __future__ import annotations

from textual.widgets import DataTable

from emet.tui.widgets.regime_badge import REGIME_COLORS
from emet.tui.workers import SweepRow


class SweepTable(DataTable):
    """DataTable displaying sweep results with color-coded regime column."""

    DEFAULT_CSS = """
    SweepTable {
        height: 1fr;
    }
    """

    def on_mount(self) -> None:
        self.add_columns(
            "idx", "param", "value", "regime", "licensed",
            "chi", "gamma", "lambda", "kahan",
        )
        self.cursor_type = "row"

    def add_sweep_row(self, row: SweepRow) -> None:
        color = REGIME_COLORS.get(row.regime, "#888888")
        regime_text = f"[{color}]{row.regime.upper()}[/]"
        licensed_text = "[#00ff41]YES[/]" if row.licensed else "[#ff4444]NO[/]"
        chi_text = f"{row.chi:.8f}" if row.chi is not None else "N/A"
        kahan_text = "[#00ff41]OK[/]" if row.kahan_certified else "[#ff4444]--[/]"

        self.add_row(
            str(row.index),
            row.param_name,
            f"{row.param_value:.4f}",
            regime_text,
            licensed_text,
            chi_text,
            f"{row.gamma:.8f}",
            f"{row.lambda_:.8f}",
            kahan_text,
        )
