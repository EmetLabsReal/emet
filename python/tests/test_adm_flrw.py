"""Tests for the ADM constraint operator on closed FLRW (S³)."""

import unittest
import numpy as np
from emet.domains.adm_flrw import (
    scalar_eigenvalue,
    scalar_degeneracy,
    tt_eigenvalue,
    gamma_homogeneous,
    chi_homogeneous,
    build_adm_flrw_tower,
    sweep_chi,
    A_CRIT,
)


class TestS3Spectrum(unittest.TestCase):
    def test_scalar_eigenvalue_ell0(self):
        """ℓ=0 scalar eigenvalue is 0."""
        self.assertAlmostEqual(scalar_eigenvalue(0, 1.0), 0.0)

    def test_scalar_eigenvalue_ell1(self):
        """ℓ=1: eigenvalue = 1·3/a² = 3/a²."""
        self.assertAlmostEqual(scalar_eigenvalue(1, 2.0), 3.0 / 4.0)

    def test_scalar_degeneracy(self):
        """Degeneracy of ℓ-th harmonic on S³ is (ℓ+1)²."""
        self.assertEqual(scalar_degeneracy(0), 1)
        self.assertEqual(scalar_degeneracy(1), 4)
        self.assertEqual(scalar_degeneracy(2), 9)
        self.assertEqual(scalar_degeneracy(5), 36)

    def test_tt_eigenvalue_ell2(self):
        """ℓ=2: Lichnerowicz eigenvalue = (2·4−2)/a² = 6/a²."""
        self.assertAlmostEqual(tt_eigenvalue(2, 1.0), 6.0)

    def test_tt_eigenvalue_requires_ell_ge_2(self):
        with self.assertRaises(ValueError):
            tt_eigenvalue(1, 1.0)
        with self.assertRaises(ValueError):
            tt_eigenvalue(0, 1.0)


class TestHomogeneousBlock(unittest.TestCase):
    def test_gamma_at_a_crit(self):
        """γ vanishes at a_crit = 4/3."""
        self.assertAlmostEqual(gamma_homogeneous(A_CRIT), 0.0, places=14)

    def test_gamma_positive_above_a_crit(self):
        """γ > 0 for a > a_crit."""
        self.assertGreater(gamma_homogeneous(2.0), 0.0)

    def test_gamma_negative_below_a_crit(self):
        """γ < 0 for a < a_crit."""
        self.assertLess(gamma_homogeneous(1.0), 0.0)

    def test_chi_diverges_at_a_crit(self):
        """χ → ∞ at a_crit."""
        self.assertEqual(chi_homogeneous(A_CRIT), float("inf"))

    def test_chi_finite_away_from_a_crit(self):
        """χ is finite away from a_crit."""
        self.assertTrue(np.isfinite(chi_homogeneous(2.0)))

    def test_a_crit_value(self):
        """a_crit = 4/3."""
        self.assertAlmostEqual(A_CRIT, 4.0 / 3.0, places=14)


class TestTTDecoupling(unittest.TestCase):
    def test_off_diagonal_zero_for_ell_ge_2(self):
        """H_PQ(ℓ) = 0 for all ℓ ≥ 2 in the full tower."""
        H, retained, omitted = build_adm_flrw_tower(2.0, ell_max=20)
        # Check that all off-diagonal blocks for ℓ ≥ 2 are zero.
        # The ℓ=0 block occupies indices 0,1.
        # Each subsequent ℓ occupies indices 2*(ℓ-2)+2, 2*(ℓ-2)+3
        # i.e. ℓ=2 → [2,3], ℓ=3 → [4,5], ...
        for ell in range(2, 21):
            base = 2 * (ell - 2) + 2
            self.assertAlmostEqual(H[base, base + 1], 0.0)
            self.assertAlmostEqual(H[base + 1, base], 0.0)

    def test_ell0_block_has_coupling(self):
        """The ℓ=0 homogeneous block has nonzero coupling."""
        H, _, _ = build_adm_flrw_tower(2.0)
        self.assertNotAlmostEqual(H[0, 1], 0.0)

    def test_matrix_is_symmetric(self):
        """Full tower matrix is symmetric (Hermitian)."""
        H, _, _ = build_adm_flrw_tower(2.0, ell_max=10)
        np.testing.assert_allclose(H, H.T, atol=1e-15)


class TestSweep(unittest.TestCase):
    def test_sweep_contains_divergence(self):
        """Sweep passes through a_crit and χ diverges."""
        a_vals, chi_vals = sweep_chi()
        self.assertTrue(np.any(np.isinf(chi_vals)) or np.max(chi_vals) > 1e6)

    def test_sweep_length(self):
        a_vals, chi_vals = sweep_chi()
        self.assertEqual(len(a_vals), 500)
        self.assertEqual(len(chi_vals), 500)


if __name__ == "__main__":
    unittest.main()
