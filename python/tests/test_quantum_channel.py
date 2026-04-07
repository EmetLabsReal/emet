"""Tests for quantum channel domain adapter."""

import math
import unittest

import numpy as np

from emet.domains.quantum_channel import (
    SIGNAL_INDICES,
    ERROR_INDICES,
    binary_entropy,
    certify_channel,
    choi_amplitude_damping,
    choi_bit_flip,
    choi_dephasing,
    choi_depolarizing,
    choi_from_kraus,
    choi_identity,
    choi_misaligned,
    qber_from_choi,
    shor_preskill_rate,
)


class TestChoiProperties(unittest.TestCase):
    """Choi matrices must be Hermitian PSD with trace = d."""

    def _check_choi(self, choi: np.ndarray):
        self.assertEqual(choi.shape, (4, 4))
        np.testing.assert_allclose(choi, choi.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(choi)
        self.assertTrue(np.all(eigvals >= -1e-12), f"Not PSD: {eigvals}")
        np.testing.assert_allclose(np.trace(choi), 2.0, atol=1e-12)

    def test_identity(self):
        self._check_choi(choi_identity())

    def test_depolarizing(self):
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
            self._check_choi(choi_depolarizing(p))

    def test_dephasing(self):
        for p in [0.0, 0.1, 0.25, 0.5]:
            self._check_choi(choi_dephasing(p))

    def test_bit_flip(self):
        for p in [0.0, 0.1, 0.25, 0.5]:
            self._check_choi(choi_bit_flip(p))

    def test_amplitude_damping(self):
        for gamma in [0.0, 0.1, 0.5, 0.9, 1.0]:
            self._check_choi(choi_amplitude_damping(gamma))

    def test_misaligned(self):
        for p in [0.0, 0.1, 0.25, 0.5]:
            self._check_choi(choi_misaligned(p))


class TestPauliChannelsBlockDiagonal(unittest.TestCase):
    """Pauli channels have zero cross-block: lambda = 0, chi = 0."""

    def _check_zero_cross_block(self, choi: np.ndarray):
        for i in SIGNAL_INDICES:
            for j in ERROR_INDICES:
                self.assertAlmostEqual(choi[i, j], 0.0, places=12)
                self.assertAlmostEqual(choi[j, i], 0.0, places=12)

    def test_identity_cross_block(self):
        self._check_zero_cross_block(choi_identity())

    def test_depolarizing_cross_block(self):
        for p in [0.0, 0.1, 0.5, 1.0]:
            self._check_zero_cross_block(choi_depolarizing(p))

    def test_dephasing_cross_block(self):
        for p in [0.0, 0.1, 0.5]:
            self._check_zero_cross_block(choi_dephasing(p))

    def test_bit_flip_cross_block(self):
        for p in [0.0, 0.1, 0.5]:
            self._check_zero_cross_block(choi_bit_flip(p))

    def test_amplitude_damping_cross_block(self):
        for gamma in [0.0, 0.1, 0.5, 1.0]:
            self._check_zero_cross_block(choi_amplitude_damping(gamma))


class TestMisalignedCrossBlock(unittest.TestCase):
    """Misaligned channel has nonzero cross-block for p > 0."""

    def test_misaligned_zero_is_identity(self):
        np.testing.assert_allclose(
            choi_misaligned(0.0), choi_identity(), atol=1e-12,
        )

    def test_misaligned_nonzero_cross_block(self):
        for p in [0.1, 0.25, 0.5]:
            choi = choi_misaligned(p)
            cross = np.zeros((2, 2))
            for ii, i in enumerate(SIGNAL_INDICES):
                for jj, j in enumerate(ERROR_INDICES):
                    cross[ii, jj] = choi[i, j]
            self.assertGreater(np.linalg.norm(cross), 1e-6)


class TestCertification(unittest.TestCase):
    """End-to-end certification via emet."""

    def test_identity_pre_admissible(self):
        # Error block is zero — no bit-flip errors exist
        r = certify_channel(choi_identity())
        self.assertEqual(r["regime"], "pre_admissible")
        self.assertAlmostEqual(r["lambda"], 0.0, places=10)

    def test_depolarizing_subcritical(self):
        # Depolarizing puts weight on all Pauli errors — error block is nonsingular
        for p in [0.01, 0.1, 0.25]:
            r = certify_channel(choi_depolarizing(p))
            self.assertTrue(r["valid"], f"Failed at p={p}")
            self.assertAlmostEqual(r["chi"], 0.0, places=10)

    def test_dephasing_pre_admissible(self):
        # Dephasing = phase-flip only, no bit-flip errors → error block zero
        r = certify_channel(choi_dephasing(0.1))
        self.assertEqual(r["regime"], "pre_admissible")
        self.assertAlmostEqual(r["lambda"], 0.0, places=10)

    def test_bit_flip_pre_admissible(self):
        # Bit-flip error block is rank-1 (singular) → pre-admissible
        r = certify_channel(choi_bit_flip(0.1))
        self.assertEqual(r["regime"], "pre_admissible")
        self.assertAlmostEqual(r["lambda"], 0.0, places=10)

    def test_misaligned_has_nonzero_lambda(self):
        r = certify_channel(choi_misaligned(0.3))
        self.assertGreater(r["lambda"], 1e-6)
        # Error block is singular (gamma = 0), so chi is None
        self.assertIsNone(r["chi"])
        self.assertFalse(r["valid"])


class TestKeyRate(unittest.TestCase):
    """Shor-Preskill key rate."""

    def test_binary_entropy_endpoints(self):
        self.assertAlmostEqual(binary_entropy(0.0), 0.0)
        self.assertAlmostEqual(binary_entropy(1.0), 0.0)
        self.assertAlmostEqual(binary_entropy(0.5), 1.0)

    def test_key_rate_no_noise(self):
        self.assertAlmostEqual(shor_preskill_rate(0.0), 1.0)

    def test_key_rate_max_noise(self):
        self.assertAlmostEqual(shor_preskill_rate(0.5), 0.0)

    def test_key_rate_threshold(self):
        # r = 0 at QBER ~ 11%
        r = shor_preskill_rate(0.11)
        self.assertGreater(r, 0.0)
        r = shor_preskill_rate(0.12)
        self.assertAlmostEqual(r, 0.0, places=2)

    def test_identity_key_rate(self):
        r = certify_channel(choi_identity())
        self.assertAlmostEqual(r["qber"], 0.0, places=10)
        self.assertAlmostEqual(r["key_rate"], 1.0, places=10)

    def test_depolarizing_qber(self):
        # For depolarizing with error rate p, QBER = 2p/3
        for p in [0.0, 0.15, 0.3, 0.75]:
            r = certify_channel(choi_depolarizing(p))
            expected_qber = 2 * p / 3
            self.assertAlmostEqual(r["qber"], expected_qber, places=6)


class TestEdgeCases(unittest.TestCase):

    def test_fully_depolarizing(self):
        # p = 1: maximally mixed output
        r = certify_channel(choi_depolarizing(1.0))
        self.assertTrue(r["valid"])
        self.assertAlmostEqual(r["qber"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(r["key_rate"], 0.0, places=2)

    def test_full_amplitude_damping(self):
        # gamma = 1: channel sends everything to |0>
        choi = choi_amplitude_damping(1.0)
        self.assertAlmostEqual(np.trace(choi), 2.0, places=12)

    def test_choi_from_kraus_custom(self):
        # Single Kraus op = unitary channel
        U = np.array([[0, 1], [1, 0]], dtype=float)  # X gate
        choi = choi_from_kraus([U])
        self.assertEqual(choi.shape, (4, 4))
        np.testing.assert_allclose(np.trace(choi), 2.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
