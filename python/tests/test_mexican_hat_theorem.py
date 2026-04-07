"""Tests for the Mexican hat theorem: licensed reductions force potential structure."""

import unittest

import numpy as np

from emet.domains.torus import build_torus_operator, effective_potential
from emet.domains.mexican_hat import (
    licensed_implies_mexican_hat,
    verify_mexican_hat_centrifugal,
)
from emet.domains.yang_mills import build_plaquette_blocks
from emet.domains.kahan import certified_subcritical

import emet


class TestVEffAlgebra(unittest.TestCase):
    """V_eff sign properties — the algebraic core of the theorem."""

    def test_positive_above_two(self):
        s = np.linspace(0.01, 1.0, 50)
        for beta in [2.5, 3.0, 5.0, 10.0]:
            v = effective_potential(beta, s)
            self.assertTrue(np.all(v > 0), f"V_eff not positive at beta={beta}")

    def test_negative_below_two(self):
        s = np.linspace(0.01, 1.0, 50)
        for beta in [0.5, 1.0, 1.5]:
            v = effective_potential(beta, s)
            self.assertTrue(np.all(v < 0), f"V_eff not negative at beta={beta}")

    def test_zero_at_two(self):
        s = np.linspace(0.01, 1.0, 50)
        v = effective_potential(2.0, s)
        np.testing.assert_allclose(v, 0.0, atol=1e-15)

    def test_monotone_above_two(self):
        s = np.array([0.1, 0.5, 1.0])
        for s_val in s:
            prev = effective_potential(2.5, np.array([s_val]))[0]
            for beta in [3.0, 4.0, 5.0]:
                curr = effective_potential(beta, np.array([s_val]))[0]
                self.assertGreater(curr, prev, f"V_eff not increasing at s={s_val}")
                prev = curr


class TestTorusImplication(unittest.TestCase):
    """For the torus family: licensed AND beta > 2 → Mexican hat."""

    def test_deep_coupling_both_hold(self):
        """beta > 2 should give both licensing and centrifugal repulsion."""
        for beta in [2.5, 3.0, 4.0]:
            H, ret, omit = build_torus_operator(beta)
            result = licensed_implies_mexican_hat(H, ret, omit, beta)
            self.assertTrue(result["licensed"], f"Not licensed at beta={beta}")
            self.assertTrue(result["centrifugal_mexican_hat"],
                            f"No centrifugal at beta={beta}")
            self.assertTrue(result["both"], f"Not both at beta={beta}")

    def test_sweep_consistency(self):
        """Sweep beta: both should align for beta > 2."""
        for beta in np.arange(2.5, 5.0, 0.5):
            H, ret, omit = build_torus_operator(beta)
            result = licensed_implies_mexican_hat(H, ret, omit, beta)
            self.assertTrue(result["both"])


class TestBetaGap(unittest.TestCase):
    """The 1 < beta < 2 window: licensed but NOT centrifugal Mexican hat."""

    def test_gap_regime(self):
        """beta = 1.5: should be licensed but V_eff < 0."""
        beta = 1.5
        H, ret, omit = build_torus_operator(beta)
        result = licensed_implies_mexican_hat(H, ret, omit, beta)
        self.assertTrue(result["licensed"], "Should be licensed at beta=1.5")
        self.assertFalse(result["centrifugal_mexican_hat"],
                         "Should NOT have centrifugal repulsion at beta=1.5")
        self.assertFalse(result["both"])


class TestYangMillsImplication(unittest.TestCase):
    """Yang-Mills at deep coupling: licensed AND Mexican hat."""

    def test_strong_coupling(self):
        for g2 in [3.0, 5.0, 10.0, 20.0]:
            beta = g2  # the identification
            H, ret, omit, _ = build_plaquette_blocks(g2, j_max=4.0, j_cut=1.0)
            report = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
            chi = report["advanced_metrics"]["chi"]
            self.assertIsNotNone(chi)
            self.assertLess(chi, 1.0, f"Not licensed at g²={g2}")
            self.assertTrue(verify_mexican_hat_centrifugal(beta),
                            f"No centrifugal at g²={g2}")


class TestKahanCertifiedMexicanHat(unittest.TestCase):
    """Kahan-certified chi + eps_chi < 1 AND V_eff > 0."""

    def test_certified_and_centrifugal(self):
        for g2 in [3.0, 5.0, 10.0]:
            beta = g2
            H, ret, omit, _ = build_plaquette_blocks(g2, j_max=4.0, j_cut=1.0)
            report = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
            m = report["advanced_metrics"]
            cert = certified_subcritical(m["chi"], m["gamma"], m["lambda"])
            self.assertTrue(cert["certified"], f"Not Kahan-certified at g²={g2}")
            self.assertTrue(verify_mexican_hat_centrifugal(beta),
                            f"No centrifugal at g²={g2}")


if __name__ == "__main__":
    unittest.main()
