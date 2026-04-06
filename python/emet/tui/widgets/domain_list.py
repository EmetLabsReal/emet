"""OptionList of available domain adapters."""

from __future__ import annotations

from textual.widgets import OptionList
from textual.widgets.option_list import Option

from emet.tui.registry import DOMAINS


class DomainList(OptionList):
    """Selectable list of the 15 domain adapters."""

    DEFAULT_CSS = """
    DomainList {
        height: 1fr;
        width: 1fr;
        border: solid #333333;
    }
    """

    def on_mount(self) -> None:
        for domain in DOMAINS:
            self.add_option(Option(
                f"{domain.name}",
                id=domain.key,
            ))
