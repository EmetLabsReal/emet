"""Tests for the pinched torus domain."""

import unittest
import numpy as np
from emet.domains.torus import (
    build_torus_operator,
    effective_potential,
    ground_state,
    ground_state_variance,
    sweep_beta,
)


class TestEffectivePotential(unittest.TestCase):
    def test_hardy_constant_at_beta_1(self):
        """At β=1, V_eff(s) = -1/(4s²) — the Hardy constant."""
        s = np.array([0.1, 0.5, 1.0])
        V = effective_potential(1.0, s)
        expected = -1.0 / (4.0 * s ** 2)
        np.testing.assert_allclose(V, expected, rtol=1e-12)

    def test_zero_at_beta_2(self):
        """At β=2, V_eff vanishes identically."""
        s = np.array([0.1, 0.5, 1.0])
        V = effective_potential(2.0, s)
        np.testing.assert_allclose(V, 0.0, atol=1e-14)

    def test_repulsive_above_beta_2(self):
        """For β > 2, V_eff > 0 everywhere: repulsive barrier."""
        s = np.linspace(0.01, 1.0, 100)
        V = effective_potential(3.0, s)
        self.assertTrue(np.all(V > 0))

    def test_attractive_below_beta_2(self):
        """For β < 2, V_eff < 0: attractive toward boundary."""
        s = np.linspace(0.01, 1.0, 100)
        V = effective_potential(0.5, s)
        self.assertTrue(np.all(V < 0))


class TestOperatorConstruction(unittest.TestCase):
    def test_symmetric(self):
        H, _, _ = build_torus_operator(2.0)
        np.testing.assert_allclose(H, H.T, atol=1e-14)

    def test_block_dimensions(self):
        H, ret, omit = build_torus_operator(2.0, n_valley=5, n_barrier=3)
        self.assertEqual(H.shape, (8, 8))
        self.assertEqual(ret, [0, 1, 2, 3, 4])
        self.assertEqual(omit, [5, 6, 7])

    def test_positive_definite_past_feller(self):
        """Past the Feller threshold, the operator should be positive definite."""
        H, _, _ = build_torus_operator(2.0)
        eigvals = np.linalg.eigvalsh(H)
        self.assertTrue(np.all(eigvals > 0))


class TestSweep(unittest.TestCase):
    """Integration tests: full β-sweep through the chi engine."""

    def setUp(self):
        self.betas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        self.results = sweep_beta(self.betas)

    def test_supercritical_below_feller(self):
        """β = 0.5: below Feller threshold, barrier is transparent, chi > 1."""
        r = self.results[0]
        self.assertFalse(r['valid'])
        self.assertEqual(r['regime'], 'supercritical')

    def test_subcritical_past_feller(self):
        """β ≥ 1.5: past Feller, barrier activated, chi < 1."""
        for r in self.results:
            if r['beta'] >= 1.5:
                self.assertTrue(r['valid'],
                    f"β={r['beta']} should be licensed, got {r['regime']}, chi={r['chi']}")

    def test_chi_decreases_past_feller(self):
        """χ strictly decreases as β increases past the threshold."""
        licensed = [r for r in self.results if r['valid']]
        chis = [r['chi'] for r in licensed]
        for i in range(len(chis) - 1):
            self.assertGreater(chis[i], chis[i + 1])

    def test_gamma_increases_past_feller(self):
        """γ strictly increases for β > 1: barrier strengthens."""
        past = [r for r in self.results if r['beta'] > 1.0]
        gammas = [r['gamma'] for r in past]
        for i in range(len(gammas) - 1):
            self.assertLess(gammas[i], gammas[i + 1])

    def test_mass_gap_increases_with_beta(self):
        """λ₀ strictly increasing for β ≥ 1."""
        post_feller = [r for r in self.results if r['beta'] >= 1.0]
        gaps = [r['lambda_0'] for r in post_feller]
        for i in range(len(gaps) - 1):
            self.assertLess(gaps[i], gaps[i + 1],
                f"Mass gap not increasing: λ₀(β={post_feller[i]['beta']})={gaps[i]} >= "
                f"λ₀(β={post_feller[i+1]['beta']})={gaps[i+1]}")

    def test_ground_state_variance_decreases(self):
        """Var(φ₀) strictly decreasing for β ≥ 1."""
        post_feller = [r for r in self.results if r['beta'] >= 1.0]
        variances = [r['variance'] for r in post_feller]
        for i in range(len(variances) - 1):
            self.assertGreater(variances[i], variances[i + 1],
                f"Variance not decreasing at β={post_feller[i]['beta']}")


if __name__ == "__main__":
    unittest.main()
