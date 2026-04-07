"""Tests for domain adapters."""

import unittest
import numpy as np
import emet
from emet.domains.kuramoto import preset_smoluchowski, preset_hysteresis
from emet.domains.graph_laplacian import preset_decoupled, preset_high_coupling
from emet.domains.transformer import build_gram, partition_from_mask


class TestKuramoto(unittest.TestCase):
    def test_smoluchowski_is_subcritical(self):
        matrix, retained, omitted = preset_smoluchowski()
        report = emet.decide_dense_matrix(matrix, retained=retained, omitted=omitted)
        self.assertTrue(report["valid"])
        self.assertEqual(report["regime"], "subcritical")

    def test_hysteresis_is_pre_admissible(self):
        matrix, retained, omitted = preset_hysteresis()
        report = emet.decide_dense_matrix(matrix, retained=retained, omitted=omitted)
        self.assertFalse(report["valid"])
        self.assertEqual(report["regime"], "pre_admissible")


class TestGraphLaplacian(unittest.TestCase):
    def test_decoupled_is_subcritical(self):
        matrix, retained, omitted = preset_decoupled()
        report = emet.decide_dense_matrix(matrix, retained=retained, omitted=omitted)
        self.assertTrue(report["valid"])
        self.assertEqual(report["regime"], "subcritical")

    def test_high_coupling_is_supercritical(self):
        matrix, retained, omitted = preset_high_coupling()
        report = emet.decide_dense_matrix(matrix, retained=retained, omitted=omitted)
        self.assertFalse(report["valid"])
        self.assertEqual(report["regime"], "supercritical")


class TestTransformer(unittest.TestCase):
    def test_gram_is_spd(self):
        rng = np.random.default_rng(42)
        K = rng.standard_normal((8, 4))
        H = build_gram(K, eta=1.0)
        eigvals = np.linalg.eigvalsh(H)
        self.assertTrue(np.all(eigvals > 0))

    def test_partition_from_mask(self):
        retained, omitted = partition_from_mask([1, 0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(retained, [0, 2, 4, 6])
        self.assertEqual(omitted, [1, 3, 5, 7])

    def test_eviction_decision(self):
        rng = np.random.default_rng(42)
        K = rng.standard_normal((8, 4))
        H = build_gram(K, eta=10.0)  # Strong regularizer -> small cross-coupling
        retained, omitted = partition_from_mask([1, 1, 1, 1, 0, 0, 0, 0])
        report = emet.decide_dense_matrix(H, retained=retained, omitted=omitted)
        # With eta=10 the regularizer dominates, so chi should be small
        self.assertIsNotNone(report["advanced_metrics"]["chi"])


if __name__ == "__main__":
    unittest.main()
