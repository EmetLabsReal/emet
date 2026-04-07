"""Tests for transfer matrix / thermodynamic limit: gap, certification, scaling."""

import unittest

import numpy as np

from emet.domains.lattice import (
    build_transfer_matrix,
    transfer_matrix_gap,
    certify_transfer_matrix,
    multi_plaquette_gap,
    scaling_analysis,
)


class TestBuildTransferMatrix(unittest.TestCase):
    """build_transfer_matrix returns a symmetric positive matrix."""

    def test_symmetric(self):
        T = build_transfer_matrix(4.0, j_max=3.0)
        np.testing.assert_allclose(T, T.T, atol=1e-14)

    def test_positive_eigenvalues(self):
        T = build_transfer_matrix(4.0, j_max=3.0)
        eigvals = np.linalg.eigvalsh(T)
        self.assertTrue(np.all(eigvals > 0),
                        f"Transfer matrix has non-positive eigenvalues: {eigvals}")

    def test_correct_shape(self):
        T = build_transfer_matrix(4.0, j_max=3.0)
        # j = 0, 0.5, 1, 1.5, 2, 2.5, 3 => 7 values
        self.assertEqual(T.shape, (7, 7))

    def test_strong_coupling_diagonal_dominant(self):
        """At strong coupling, electric term dominates => T close to diagonal."""
        T = build_transfer_matrix(20.0, j_max=3.0)
        diag = np.diag(T)
        off_diag_norm = np.linalg.norm(T - np.diag(diag))
        diag_norm = np.linalg.norm(diag)
        self.assertGreater(diag_norm, off_diag_norm,
                           "Strong coupling T should be diagonally dominant")


class TestTransferMatrixGap(unittest.TestCase):
    """transfer_matrix_gap returns a positive gap for strong coupling."""

    def test_positive_gap_strong_coupling(self):
        T = build_transfer_matrix(4.0, j_max=3.0)
        gap = transfer_matrix_gap(T)
        self.assertGreater(gap, 0.0, "Gap should be positive at strong coupling")

    def test_gap_increases_with_coupling(self):
        """Stronger coupling => larger gap (more confinement)."""
        gaps = []
        for g2 in [3.0, 5.0, 10.0]:
            T = build_transfer_matrix(g2, j_max=3.0)
            gaps.append(transfer_matrix_gap(T))
        for i in range(len(gaps) - 1):
            self.assertGreater(gaps[i + 1], gaps[i],
                               f"Gap should increase with coupling: {gaps}")

    def test_gap_is_finite(self):
        T = build_transfer_matrix(4.0, j_max=3.0)
        gap = transfer_matrix_gap(T)
        self.assertTrue(np.isfinite(gap))


class TestCertifyTransferMatrix(unittest.TestCase):
    """certify_transfer_matrix: structure, gap, and chi behavior.

    The transfer matrix T = exp(-H) transforms the spectral structure.
    The partition into low-j / high-j sectors yields chi >= 1 for T
    (unlike for H directly) because the exponential compresses eigenvalues.
    The certification still returns a well-formed report with a positive gap.
    """

    def test_positive_transfer_gap(self):
        result = certify_transfer_matrix(4.0)
        self.assertGreater(result["transfer_gap"], 0.0)

    def test_result_contains_required_keys(self):
        result = certify_transfer_matrix(4.0)
        for key in ["g_squared", "transfer_gap", "chi", "gamma", "lambda",
                     "licensed", "regime", "kahan_certified"]:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_chi_is_numeric_or_none(self):
        """Chi should be a number (possibly large) or None."""
        result = certify_transfer_matrix(4.0)
        chi = result["chi"]
        self.assertTrue(chi is None or isinstance(chi, (int, float)))

    def test_gap_increases_with_coupling(self):
        """Transfer gap should increase with coupling strength."""
        gaps = []
        for g2 in [3.0, 5.0, 10.0]:
            result = certify_transfer_matrix(g2)
            gaps.append(result["transfer_gap"])
        for i in range(len(gaps) - 1):
            self.assertGreater(gaps[i + 1], gaps[i])

    def test_partition_sizes(self):
        result = certify_transfer_matrix(4.0, j_max=3.0, j_cut=1.0)
        self.assertEqual(result["n_retained"], 3)  # j=0, 0.5, 1.0
        self.assertEqual(result["n_omitted"], 4)   # j=1.5, 2.0, 2.5, 3.0


class TestMultiPlaquetteGap(unittest.TestCase):
    """multi_plaquette_gap returns the same gap regardless of n_plaquettes."""

    def test_gap_independent_of_n(self):
        """The gap is a property of T, not of N. It should be identical."""
        g2 = 4.0
        gaps = [multi_plaquette_gap(g2, n, j_max=3.0) for n in [1, 2, 5, 10, 100]]
        for gap in gaps:
            np.testing.assert_allclose(gap, gaps[0], atol=1e-14,
                                       err_msg="Gap should not depend on n_plaquettes")

    def test_gap_is_positive(self):
        gap = multi_plaquette_gap(4.0, 1, j_max=3.0)
        self.assertGreater(gap, 0.0)

    def test_matches_single_transfer_gap(self):
        """multi_plaquette_gap should match transfer_matrix_gap directly."""
        g2 = 4.0
        T = build_transfer_matrix(g2, j_max=3.0)
        expected = transfer_matrix_gap(T)
        actual = multi_plaquette_gap(g2, 1, j_max=3.0)
        np.testing.assert_allclose(actual, expected, atol=1e-14)


class TestScalingAnalysis(unittest.TestCase):
    """scaling_analysis returns consistent Hamiltonian vs transfer matrix results."""

    def test_returns_list(self):
        results = scaling_analysis([4.0, 6.0])
        self.assertEqual(len(results), 2)

    def test_both_gaps_positive(self):
        results = scaling_analysis([4.0, 6.0, 10.0])
        for r in results:
            self.assertGreater(r["hamiltonian_gap"], 0.0,
                               f"Hamiltonian gap not positive at g^2={r['g_squared']}")
            self.assertGreater(r["transfer_gap"], 0.0,
                               f"Transfer gap not positive at g^2={r['g_squared']}")

    def test_chi_transfer_present(self):
        """scaling_analysis should report chi_transfer for each coupling."""
        results = scaling_analysis([4.0, 6.0, 10.0])
        for r in results:
            self.assertIn("chi_transfer", r)
            self.assertIn("licensed_transfer", r)

    def test_result_contains_required_keys(self):
        results = scaling_analysis([4.0])
        r = results[0]
        for key in ["g_squared", "hamiltonian_gap", "transfer_gap",
                     "chi_transfer", "licensed_transfer"]:
            self.assertIn(key, r, f"Missing key: {key}")

    def test_gaps_both_increase_with_coupling(self):
        """Both Hamiltonian and transfer gaps should increase with g^2."""
        results = scaling_analysis([3.0, 5.0, 10.0])
        h_gaps = [r["hamiltonian_gap"] for r in results]
        t_gaps = [r["transfer_gap"] for r in results]
        for i in range(len(h_gaps) - 1):
            self.assertGreater(h_gaps[i + 1], h_gaps[i],
                               "Hamiltonian gap should increase with coupling")
            self.assertGreater(t_gaps[i + 1], t_gaps[i],
                               "Transfer gap should increase with coupling")


if __name__ == "__main__":
    unittest.main()
