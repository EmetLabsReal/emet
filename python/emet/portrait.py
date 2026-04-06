"""Phase portrait: the (gamma, lambda) plane as diagnostic space.

Every certified reduction maps to a point (gamma, lambda) in the portrait.
The regime is a derived property of the point:

    A: Cap > 0, beta < 1         accessible boundary, no unique extension
    B: Cap = 0, V_eff <= 0       passive confinement (1 <= beta <= 2)
    C: Cap = 0, V_eff > 0        active confinement (beta > 2)

When beta is unknown, the chi < 1 test gives subcritical/supercritical,
which collapses B and C into "licensed" and A into "not licensed".

The phase point is the complete invariant. The regime is a consequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence


class Regime(Enum):
    """Trichotomy by capacity and potential, or chi-based fallback."""

    A = "A"                          # Cap > 0, beta < 1
    B = "B"                          # Cap = 0, V_eff <= 0, 1 <= beta <= 2
    C = "C"                          # Cap = 0, V_eff > 0, beta > 2
    SUBCRITICAL = "subcritical"      # chi < 1, beta unknown
    SUPERCRITICAL = "supercritical"  # chi >= 1, beta unknown
    PRE_ADMISSIBLE = "pre_admissible"  # gamma ~ 0, chi undefined


def _classify(chi: float | None, beta: float | None) -> Regime:
    """Derive regime from chi and beta."""
    if chi is None or (isinstance(chi, float) and not math.isfinite(chi)):
        return Regime.PRE_ADMISSIBLE
    if beta is not None:
        if beta < 1.0:
            return Regime.A
        if beta <= 2.0:
            return Regime.B
        return Regime.C
    if chi < 1.0:
        return Regime.SUBCRITICAL
    return Regime.SUPERCRITICAL


@dataclass(frozen=True)
class PhasePoint:
    """A point in the (gamma, lambda) diagnostic plane."""

    gamma: float
    lambda_: float
    chi: float | None
    beta: float | None
    regime: Regime
    determinacy: bool

    @classmethod
    def from_report(
        cls,
        report: dict[str, Any],
        *,
        beta: float | None = None,
    ) -> PhasePoint:
        """Extract a phase point from a decision report dict."""
        metrics = report.get("advanced_metrics") or {}
        gamma = metrics.get("gamma", 0.0)
        lam = metrics.get("lambda", 0.0)
        chi = metrics.get("chi")
        regime = _classify(chi, beta)
        determinacy = regime in (Regime.B, Regime.C, Regime.SUBCRITICAL)
        return cls(
            gamma=gamma,
            lambda_=lam,
            chi=chi,
            beta=beta,
            regime=regime,
            determinacy=determinacy,
        )

    @classmethod
    def from_values(
        cls,
        gamma: float,
        lambda_: float,
        *,
        beta: float | None = None,
    ) -> PhasePoint:
        """Construct directly from spectral quantities."""
        if gamma > 1e-15:
            chi = (lambda_ / gamma) ** 2
        else:
            chi = None
        regime = _classify(chi, beta)
        determinacy = regime in (Regime.B, Regime.C, Regime.SUBCRITICAL)
        return cls(
            gamma=gamma,
            lambda_=lambda_,
            chi=chi,
            beta=beta,
            regime=regime,
            determinacy=determinacy,
        )

    @property
    def licensed(self) -> bool:
        return self.chi is not None and self.chi < 1.0

    @property
    def cap_zero(self) -> bool | None:
        """Cap = 0 if beta >= 1 or chi < 1. None if undetermined."""
        if self.beta is not None:
            return self.beta >= 1.0
        if self.chi is not None and self.chi < 1.0:
            return True
        return None


@dataclass
class PhasePortrait:
    """Accumulated phase points from certified reductions."""

    points: list[PhasePoint]

    def __init__(self, points: Sequence[PhasePoint] | None = None):
        self.points = list(points) if points else []

    def add(self, point: PhasePoint) -> None:
        self.points.append(point)

    def add_from_report(
        self,
        report: dict[str, Any],
        *,
        beta: float | None = None,
    ) -> PhasePoint:
        """Add a point from a decision report. Returns the point."""
        pt = PhasePoint.from_report(report, beta=beta)
        self.points.append(pt)
        return pt

    def boundary(self) -> list[tuple[float, float]]:
        """The chi = 1 curve as (gamma, lambda) pairs: lambda = gamma."""
        if not self.points:
            return []
        gammas = [p.gamma for p in self.points if p.gamma > 0]
        if not gammas:
            return []
        g_min, g_max = min(gammas), max(gammas)
        n = 200
        step = (g_max - g_min) / max(n - 1, 1)
        return [(g_min + i * step, g_min + i * step) for i in range(n)]

    def by_regime(self) -> dict[Regime, list[PhasePoint]]:
        """Group points by regime."""
        groups: dict[Regime, list[PhasePoint]] = {}
        for pt in self.points:
            groups.setdefault(pt.regime, []).append(pt)
        return groups

    def predict(
        self,
        gamma: float,
        lambda_: float,
        *,
        beta: float | None = None,
    ) -> Regime:
        """Predict regime without computing chi.

        If beta is known, classify directly from the trichotomy.
        Otherwise, use chi = (lambda/gamma)^2.
        """
        if gamma > 1e-15:
            chi = (lambda_ / gamma) ** 2
        else:
            chi = None
        return _classify(chi, beta)

    def query_neighborhood(
        self,
        gamma: float,
        lambda_: float,
        radius: float,
    ) -> list[PhasePoint]:
        """All points within Euclidean radius in (gamma, lambda) space."""
        r2 = radius * radius
        return [
            p for p in self.points
            if (p.gamma - gamma) ** 2 + (p.lambda_ - lambda_) ** 2 <= r2
        ]

    def query_trajectory(
        self,
        domain: str,
        points_with_meta: Sequence[tuple[PhasePoint, dict[str, Any]]],
        param_name: str,
    ) -> list[tuple[float, PhasePoint]]:
        """Order phase points by a named parameter from their metadata.

        Takes (point, metadata_dict) pairs and returns sorted by param_name.
        """
        trajectory = []
        for pt, meta in points_with_meta:
            val = meta.get(param_name)
            if val is not None:
                trajectory.append((float(val), pt))
        trajectory.sort(key=lambda t: t[0])
        return trajectory
