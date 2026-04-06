"""Core tests: given a split, compute chi and verify against known values."""

import unittest
import numpy as np
import emet


class TestSubcritical(unittest.TestCase):
    """3x3 matrix, gamma=2, lambda=0.25, chi=0.015625."""

    def setUp(self):
        self.H = np.array([
            [1.0,  0.0, 0.25],
            [0.0,  2.0, 0.0 ],
            [0.25, 0.0, 2.0 ],
        ])
        self.report = emet.decide_dense_matrix(self.H, retained=[0, 1], omitted=[2])
        self.m = self.report["advanced_metrics"]

    def test_licensed(self):
        self.assertTrue(self.report["valid"])

    def test_regime(self):
        self.assertEqual(self.report["regime"], "subcritical")

    def test_reason(self):
        self.assertEqual(self.report["reason_code"], "valid_reduction")

    def test_gamma(self):
        self.assertAlmostEqual(self.m["gamma"], 2.0, places=10)

    def test_lambda(self):
        self.assertAlmostEqual(self.m["lambda"], 0.25, places=10)

    def test_chi(self):
        self.assertAlmostEqual(self.m["chi"], 0.015625, places=10)

    def test_effective_matrix(self):
        rm = self.report["reduced_matrix"]["data"]
        self.assertAlmostEqual(rm[0][0], 0.96875, places=10)
        self.assertAlmostEqual(rm[1][1], 2.0, places=10)


class TestSupercritical(unittest.TestCase):
    """3x3 matrix, gamma=1, lambda=2, chi=4."""

    def setUp(self):
        self.H = np.array([
            [1.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
        ])
        self.report = emet.decide_dense_matrix(self.H, retained=[0, 1], omitted=[2])
        self.m = self.report["advanced_metrics"]

    def test_not_licensed(self):
        self.assertFalse(self.report["valid"])

    def test_regime(self):
        self.assertEqual(self.report["regime"], "supercritical")

    def test_reason(self):
        self.assertEqual(self.report["reason_code"], "coupling_too_strong")

    def test_chi(self):
        self.assertAlmostEqual(self.m["chi"], 4.0, places=10)


class TestPreAdmissible(unittest.TestCase):
    """2x2 quarter-turn: gamma=0, chi undefined."""

    def setUp(self):
        self.H = np.array([
            [0.0, -1.0],
            [1.0,  0.0],
        ])
        self.report = emet.decide_dense_matrix(self.H, retained=[0], omitted=[1])
        self.m = self.report["advanced_metrics"]

    def test_not_licensed(self):
        self.assertFalse(self.report["valid"])

    def test_regime(self):
        self.assertEqual(self.report["regime"], "pre_admissible")

    def test_gamma_zero(self):
        self.assertAlmostEqual(self.m["gamma"], 0.0, places=10)

    def test_chi_none(self):
        self.assertIsNone(self.m["chi"])


class TestSearchCanonical(unittest.TestCase):
    """Exhaustive search on a 4x4 diagonal-dominant matrix."""

    def test_finds_canonical(self):
        H = np.diag([10.0, 5.0, 1.0, 0.5])
        H[0, 2] = H[2, 0] = 0.01
        H[1, 3] = H[3, 1] = 0.01
        result = emet.search_canonical_dense_matrix(H, retained_dim=2)
        self.assertIn(result["status"], ["unique_canonical", "symmetry_tied"])
        self.assertEqual(result["canonical_retained"], [0, 1])


class TestProposePCA(unittest.TestCase):
    def test_orders_by_diagonal(self):
        H = np.diag([3.0, 1.0, 5.0, 2.0])
        split = emet.propose_partition_pca(H, retained_dim=2)
        self.assertEqual(split["retained"], [0, 2])
        self.assertEqual(split["omitted"], [1, 3])


if __name__ == "__main__":
    unittest.main()
