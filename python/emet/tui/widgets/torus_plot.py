"""3D torus rendered in braille unicode.

The torus cross-section radius is modulated by V_eff(beta, v):
  r(v) = r0 * (1 - alpha * V_eff_normalized(v))

At beta < 2: smooth torus (attractive well, no pinch)
At beta = 2: Feller threshold (critical pinch)
At beta > 2: mexican hat — the torus necks down, centrifugal barrier visible

The pinch IS the barrier. You see the geometry of the reduction.
"""

from __future__ import annotations

import math

from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

BRAILLE_BASE = 0x2800
DOT_MAP = {
    (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
    (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80,
}


def _render_torus_3d(
    beta: float,
    cols: int = 70,
    rows: int = 18,
    angle_x: float = 0.45,
    angle_z: float = 0.3,
) -> str:
    """Render a 3D torus with beta-dependent cross-section pinch."""
    dot_w = cols * 2
    dot_h = rows * 4

    R = 1.0      # major radius
    r0 = 0.45    # base minor radius

    # Pinch strength from beta
    # beta > 2: positive V_eff → pinch the torus (neck narrows)
    # beta < 2: negative V_eff → bulge (no pinch)
    # beta = 2: no modulation
    pinch = (beta - 2.0) * 0.15
    pinch = max(-0.3, min(pinch, 0.35))  # clamp

    # Precompute rotation
    cx, sx = math.cos(angle_x), math.sin(angle_x)
    cz, sz = math.cos(angle_z), math.sin(angle_z)

    # Sample torus surface
    n_u = 80   # around the tube
    n_v = 120  # around the ring

    # Collect projected points with depth
    points: list[tuple[float, float, float]] = []

    for iv in range(n_v):
        v = 2.0 * math.pi * iv / n_v
        for iu in range(n_u):
            u = 2.0 * math.pi * iu / n_u

            # Modulate minor radius: pinch at v=0 (top of cross-section)
            # V_eff ~ 1/s^2, so the pinch is strongest where s is smallest
            # Map v to the cross-section: pinch at v=0 and v=pi
            modulation = math.cos(u) ** 2  # pinch at top and bottom of tube
            r = r0 * (1.0 - pinch * modulation)
            r = max(0.05, r)  # don't collapse to zero

            # Parametric torus
            x = (R + r * math.cos(u)) * math.cos(v)
            y = (R + r * math.cos(u)) * math.sin(v)
            z = r * math.sin(u)

            # Rotate around X axis (tilt)
            y1 = y * cx - z * sx
            z1 = y * sx + z * cx

            # Rotate around Z axis
            x2 = x * cz - y1 * sz
            y2 = x * sz + y1 * cz
            z2 = z1

            points.append((x2, y2, z2))

    if not points:
        return "[dim]no data[/]"

    # Project: orthographic
    xs = [p[0] for p in points]
    ys = [p[2] for p in points]  # z → screen y (we look from the side)
    zs = [p[1] for p in points]  # y → depth

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    # Scale to dot buffer with margin
    margin = 2
    ew = dot_w - 2 * margin
    eh = dot_h - 2 * margin

    # Z-buffer: only render front-facing points
    z_min = min(zs)
    z_max = max(zs)
    z_range = z_max - z_min or 1.0
    z_mid = (z_min + z_max) / 2.0

    dots = [[False] * dot_w for _ in range(dot_h)]
    zbuf = [[float("-inf")] * dot_w for _ in range(dot_h)]

    for x3, y3, z3 in points:
        # Only render front half (depth > midpoint) for cleaner look
        if z3 < z_mid:
            continue

        px = int(margin + (x3 - x_min) / x_range * ew)
        py = int(margin + (1.0 - (y3 - y_min) / y_range) * eh)

        if 0 <= px < dot_w and 0 <= py < dot_h:
            if z3 > zbuf[py][px]:
                zbuf[py][px] = z3
                dots[py][px] = True

    # Also render back half more sparsely for depth
    for i, (x3, y3, z3) in enumerate(points):
        if z3 >= z_mid:
            continue
        if i % 4 != 0:  # sparse sampling for back
            continue

        px = int(margin + (x3 - x_min) / x_range * ew)
        py = int(margin + (1.0 - (y3 - y_min) / y_range) * eh)

        if 0 <= px < dot_w and 0 <= py < dot_h:
            if not dots[py][px]:  # don't overwrite front
                dots[py][px] = True

    # Convert to braille
    lines = []
    for row in range(rows):
        chars = []
        for col in range(cols):
            code = BRAILLE_BASE
            for dy in range(4):
                for dx in range(2):
                    py = row * 4 + dy
                    px = col * 2 + dx
                    if py < dot_h and px < dot_w and dots[py][px]:
                        code |= DOT_MAP[(dx, dy)]
            chars.append(chr(code))
        lines.append("".join(chars))

    # Color and label
    if beta > 2.05:
        color = "#00ff41"
        label = "licensed — barrier confines"
    elif beta > 1.95:
        color = "#ffaa00"
        label = "Feller threshold"
    else:
        color = "#ff4444"
        label = "supercritical — no confinement"

    v_eff_peak = beta * (beta - 2.0) / (4.0 * 0.3 * 0.3) if abs(0.3) > 1e-10 else 0.0

    header = f"[bold]beta={beta:.2f}[/]  [{color}]{label}[/]"
    braille = "\n".join(f"[{color}]{l}[/]" for l in lines)
    info = f"[dim]V_eff(0.3)={v_eff_peak:+.2f}  pinch={pinch:+.3f}[/]"

    return f"{header}\n{braille}\n{info}"


class TorusPlot(Widget):
    """3D torus with beta-dependent pinch at the Feller threshold."""

    DEFAULT_CSS = """
    TorusPlot {
        height: 22;
        width: 1fr;
    }
    """

    beta: reactive[float] = reactive(3.0)

    def compose(self):
        yield Static("", id="torus-canvas")

    def watch_beta(self, value: float) -> None:
        self._render_plot(value)

    def on_mount(self) -> None:
        self._render_plot(self.beta)

    def _render_plot(self, beta: float) -> None:
        canvas = self.query_one("#torus-canvas", Static)
        try:
            cols = max(30, self.size.width - 2)
        except Exception:
            cols = 70
        canvas.update(_render_torus_3d(beta, cols=cols, rows=18))

    @staticmethod
    def render_static(beta: float) -> str:
        return _render_torus_3d(beta, cols=70, rows=18)
