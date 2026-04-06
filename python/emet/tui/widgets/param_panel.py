"""Dynamic parameter input panel generated from ParamSpec."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Label, Select

from emet.tui.registry import DomainDescriptor, ParamSpec


class ParamPanel(Widget):
    """Renders input fields for the current domain's parameters."""

    DEFAULT_CSS = """
    ParamPanel {
        height: auto;
        width: 1fr;
        padding: 1;
    }
    ParamPanel .param-label {
        height: 1;
        margin-top: 1;
        color: #aaaaaa;
    }
    ParamPanel .param-label:first-child {
        margin-top: 0;
    }
    ParamPanel Input {
        height: 3;
    }
    ParamPanel Select {
        height: 3;
    }
    """

    class ParamsChanged(Message):
        """Posted when any parameter value changes."""
        def __init__(self, params: dict[str, Any]) -> None:
            super().__init__()
            self.params = params

    domain: reactive[DomainDescriptor | None] = reactive(None, recompose=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._specs: list[ParamSpec] = []
        self._values: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        desc = self.domain
        self._specs = []
        self._values = {}
        if desc is None:
            return

        self._specs = list(desc.params)
        for spec in self._specs:
            self._values[spec.name] = spec.default
            yield Label(spec.label, classes="param-label")
            if spec.choices is not None:
                options = [(str(c), c) for c in spec.choices]
                yield Select(options, value=spec.default, id=f"param-{spec.name}")
            else:
                yield Input(
                    value=str(spec.default),
                    placeholder=spec.label,
                    id=f"param-{spec.name}",
                )

    def on_input_changed(self, event: Input.Changed) -> None:
        widget_id = event.input.id or ""
        if not widget_id.startswith("param-"):
            return
        name = widget_id[6:]
        spec = self._find_spec(name)
        if spec is None:
            return
        try:
            self._values[name] = spec.type(event.value)
        except (ValueError, TypeError):
            return
        self.post_message(self.ParamsChanged(dict(self._values)))

    def on_select_changed(self, event: Select.Changed) -> None:
        widget_id = event.select.id or ""
        if not widget_id.startswith("param-"):
            return
        name = widget_id[6:]
        if event.value is not Select.BLANK:
            self._values[name] = event.value
            self.post_message(self.ParamsChanged(dict(self._values)))

    def _find_spec(self, name: str) -> ParamSpec | None:
        for s in self._specs:
            if s.name == name:
                return s
        return None

    def get_params(self) -> dict[str, Any]:
        return dict(self._values)
