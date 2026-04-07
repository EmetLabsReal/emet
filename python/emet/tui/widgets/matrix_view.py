"""Reduced matrix display as a formatted table."""

from __future__ import annotations

from typing import Any

import numpy as np
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class MatrixView(Widget):
    """Renders a small matrix (H_eff or block) as a formatted table."""

    DEFAULT_CSS = """
    MatrixView {
        height: auto;
        width: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    """

    matrix: reactive[np.ndarray | None] = reactive(None, always_update=True)
    title_text: reactive[str] = reactive("Matrix")

    def compose(self):
        yield Static("", classes="matrix-content")

    def watch_matrix(self, value: np.ndarray | None) -> None:
        self._update_display(value)

    def watch_title_text(self, value: str) -> None:
        self._update_display(self.matrix)

    def _update_display(self, mat: np.ndarray | None) -> None:
        content = self.query_one(".matrix-content", Static)
        if mat is None:
            content.update(f"[dim]{self.title_text}: not computed[/]")
            return

        n = mat.shape[0]
        max_display = 12

        lines = [f"[bold]{self.title_text}[/] ({n}x{n})"]

        show_n = min(n, max_display)
        for i in range(show_n):
            row_vals = []
            for j in range(show_n):
                v = mat[i, j]
                if abs(v) < 1e-14:
                    row_vals.append("  0       ")
                elif abs(v) < 1e-3 or abs(v) > 1e5:
                    row_vals.append(f"{v:10.3e}")
                else:
                    row_vals.append(f"{v:10.5f}")
            row_str = " ".join(row_vals)
            if n > max_display and j == show_n - 1:
                row_str += " ..."
            lines.append(row_str)

        if n > max_display:
            lines.append(f"  ... ({n - max_display} more rows)")

        content.update("\n".join(lines))
