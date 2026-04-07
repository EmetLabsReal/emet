"""Tests for the phase portrait module."""

from __future__ import annotations

import numpy as np
import pytest

import emet
from emet.portrait import PhasePoint, PhasePortrait, Regime, _classify


class TestRegimeClassification:
    """Classification by chi and beta."""

    def test_pre_admissible_none_chi(self):
        assert _classify(None, None) == Regime.PRE_ADMISSIBLE

    def test_pre_admissible_inf_chi(self):
        assert _classify(float("inf"), None) == Regime.PRE_ADMISSIBLE

    def test_subcritical_no_beta(self):
        assert _classify(0.5, None) == Regime.SUBCRITICAL

    def test_supercritical_no_beta(self):
        assert _classify(1.5, None) == Regime.SUPERCRITICAL

    def test_regime_a(self):
        assert _classify(1.5, 0.5) == Regime.A

    def test_regime_b(self):
        assert _classify(0.5, 1.5) == Regime.B

    def test_regime_c(self):
        assert _classify(0.01, 3.0) == Regime.C

    def test_regime_a_at_chi_below_one(self):
        """beta < 1 forces Regime A even if chi < 1."""
        assert _classify(0.1, 0.5) == Regime.A

    def test_regime_b_boundary(self):
        """beta = 1 is the Feller threshold: Regime B."""
        assert _classify(0.5, 1.0) == Regime.B

    def test_regime_b_upper_boundary(self):
        """beta = 2 is the upper boundary of B."""
        assert _classify(0.5, 2.0) == Regime.B

    def test_regime_c_just_above(self):
        """beta > 2 forces Regime C."""
        assert _classify(0.01, 2.01) == Regime.C


class TestPhasePoint:
    """PhasePoint construction and properties."""

    def test_from_values_subcritical(self):
        pt = PhasePoint.from_values(5.0, 0.3)
        assert pt.chi is not None
        assert pt.chi == pytest.approx((0.3 / 5.0) ** 2)
        assert pt.licensed
        assert pt.regime == Regime.SUBCRITICAL
        assert pt.determinacy

    def test_from_values_supercritical(self):
        pt = PhasePoint.from_values(0.3, 5.0)
        assert pt.chi is not None
        assert pt.chi > 1.0
        assert not pt.licensed
        assert pt.regime == Regime.SUPERCRITICAL
        assert not pt.determinacy

    def test_from_values_with_beta_regime_c(self):
        pt = PhasePoint.from_values(5.0, 0.3, beta=3.0)
        assert pt.regime == Regime.C
        assert pt.determinacy
        assert pt.cap_zero is True

    def test_from_values_with_beta_regime_a(self):
        pt = PhasePoint.from_values(0.3, 5.0, beta=0.5)
        assert pt.regime == Regime.A
        assert not pt.determinacy
        assert pt.cap_zero is False

    def test_from_values_pre_admissible(self):
        pt = PhasePoint.from_values(0.0, 1.0)
        assert pt.chi is None
        assert pt.regime == Regime.PRE_ADMISSIBLE
        assert not pt.determinacy

    def test_from_report_subcritical(self):
        """Round-trip through the engine."""
        H = np.array([
            [2.0, 0.25, 0.0],
            [0.25, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ])
        report = emet.decide_dense_matrix(H, retained=[0], omitted=[1, 2])
        pt = PhasePoint.from_report(report)
        assert pt.licensed
        assert pt.regime == Regime.SUBCRITICAL
        assert pt.determinacy

    def test_from_report_with_beta(self):
        H = np.array([
            [2.0, 0.25, 0.0],
            [0.25, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ])
        report = emet.decide_dense_matrix(H, retained=[0], omitted=[1, 2])
        pt = PhasePoint.from_report(report, beta=3.0)
        assert pt.regime == Regime.C
        assert pt.beta == 3.0

    def test_cap_zero_none_when_unknown(self):
        pt = PhasePoint.from_values(0.3, 5.0)
        assert pt.cap_zero is None


class TestPhasePortrait:
    """Portrait accumulation and queries."""

    def test_add_and_length(self):
        portrait = PhasePortrait()
        portrait.add(PhasePoint.from_values(5.0, 0.3))
        portrait.add(PhasePoint.from_values(0.3, 5.0))
        assert len(portrait.points) == 2

    def test_add_from_report(self):
        portrait = PhasePortrait()
        H = np.array([
            [2.0, 0.25, 0.0],
            [0.25, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ])
        report = emet.decide_dense_matrix(H, retained=[0], omitted=[1, 2])
        pt = portrait.add_from_report(report, beta=3.0)
        assert pt.regime == Regime.C
        assert len(portrait.points) == 1

    def test_by_regime(self):
        portrait = PhasePortrait()
        portrait.add(PhasePoint.from_values(5.0, 0.3))
        portrait.add(PhasePoint.from_values(0.3, 5.0))
        portrait.add(PhasePoint.from_values(5.0, 0.1, beta=3.0))
        groups = portrait.by_regime()
        assert Regime.SUBCRITICAL in groups
        assert Regime.SUPERCRITICAL in groups
        assert Regime.C in groups

    def test_predict(self):
        portrait = PhasePortrait()
        assert portrait.predict(5.0, 0.3) == Regime.SUBCRITICAL
        assert portrait.predict(0.3, 5.0) == Regime.SUPERCRITICAL
        assert portrait.predict(5.0, 0.3, beta=3.0) == Regime.C
        assert portrait.predict(5.0, 0.3, beta=0.5) == Regime.A

    def test_query_neighborhood(self):
        portrait = PhasePortrait()
        portrait.add(PhasePoint.from_values(5.0, 0.3))
        portrait.add(PhasePoint.from_values(50.0, 0.01))
        near = portrait.query_neighborhood(5.0, 0.3, radius=1.0)
        assert len(near) == 1

    def test_boundary(self):
        portrait = PhasePortrait()
        portrait.add(PhasePoint.from_values(1.0, 0.5))
        portrait.add(PhasePoint.from_values(10.0, 0.1))
        curve = portrait.boundary()
        assert len(curve) == 200
        for g, l in curve:
            assert g == pytest.approx(l)


class TestPhasePointConvenience:
    """Test the emet.phase_point() entry point."""

    def test_phase_point_yang_mills(self):
        from emet.domains.yang_mills import build_plaquette_blocks
        H, ret, omit, _ = build_plaquette_blocks(g_squared=8.0)
        pt, report = emet.phase_point(H, ret, omit, beta=8.0)
        assert pt.regime == Regime.C
        assert pt.determinacy
        assert pt.beta == 8.0
        assert pt.licensed

    def test_phase_point_no_beta(self):
        H = np.array([
            [2.0, 0.25, 0.0],
            [0.25, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ])
        pt, report = emet.phase_point(H, [0], [1, 2])
        assert pt.beta is None
        assert pt.regime == Regime.SUBCRITICAL
