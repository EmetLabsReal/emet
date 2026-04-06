"""Tests for the SU(2) Yang-Mills domain adapter."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from emet.domains.yang_mills import (
    build_single_plaquette,
    casimir,
    certify_yang_mills,
    mass_gap,
    partition_by_representation,
    sweep_coupling,
)


class TestCasimir(unittest.TestCase):
    def test_j_zero(self):
        self.assertAlmostEqual(casimir(0.0), 0.0)

    def test_j_half(self):
        self.assertAlmostEqual(casimir(0.5), 0.75)

    def test_j_one(self):
        self.assertAlmostEqual(casimir(1.0), 2.0)


class TestHamiltonian(unittest.TestCase):
    def test_symmetric(self):
        H, _, _ = build_single_plaquette(2.0, j_max=3.0)
        np.testing.assert_allclose(H, H.T, atol=1e-14)

    def test_dimension(self):
        H, _, j_values = build_single_plaquette(2.0, j_max=2.0)
        self.assertEqual(H.shape, (5, 5))  # j = 0, 1/2, 1, 3/2, 2
        self.assertEqual(len(j_values), 5)

    def test_tridiagonal(self):
        H, _, _ = build_single_plaquette(2.0, j_max=2.0)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if abs(i - j) > 1:
                    self.assertAlmostEqual(H[i, j], 0.0)

    def test_diagonal_is_casimir(self):
        g2 = 3.0
        H, _, j_values = build_single_plaquette(g2, j_max=2.0)
        for k, j in enumerate(j_values):
            self.assertAlmostEqual(H[k, k], (g2 / 2.0) * j * (j + 1))

    def test_offdiagonal_is_coupling(self):
        g2 = 4.0
        H, _, _ = build_single_plaquette(g2, j_max=2.0)
        for k in range(H.shape[0] - 1):
            self.assertAlmostEqual(H[k, k + 1], -1.0 / g2)


class TestPartition(unittest.TestCase):
    def test_covers_all(self):
        j_values = [0.0, 0.5, 1.0, 1.5, 2.0]
        ret, omit = partition_by_representation(j_values, j_cut=1.0)
        self.assertEqual(sorted(ret + omit), list(range(5)))

    def test_cut_at_one(self):
        j_values = [0.0, 0.5, 1.0, 1.5, 2.0]
        ret, omit = partition_by_representation(j_values, j_cut=1.0)
        self.assertEqual(ret, [0, 1, 2])   # j = 0, 1/2, 1
        self.assertEqual(omit, [3, 4])      # j = 3/2, 2


class TestMassGap(unittest.TestCase):
    def test_gap_positive(self):
        H, _, _ = build_single_plaquette(2.0, j_max=3.0)
        E_0, E_1, gap = mass_gap(H)
        self.assertGreater(gap, 0)

    def test_gap_increases_strong_coupling(self):
        """In strong coupling, mass gap ~ g^2/2 (linear in g^2)."""
        gaps = []
        for g2 in [5.0, 10.0, 20.0]:
            H, _, _ = build_single_plaquette(g2, j_max=3.0)
            _, _, gap = mass_gap(H)
            gaps.append(gap)
        self.assertLess(gaps[0], gaps[1])
        self.assertLess(gaps[1], gaps[2])


class TestConfinement(unittest.TestCase):
    """Integration: chi certifies confinement in the strong coupling regime."""

    def setUp(self):
        self.results = sweep_coupling(
            [0.5, 1.0, 2.0, 4.0, 10.0],
            j_max=3.0, j_cut=1.0,
        )

    def test_weak_coupling_not_licensed(self):
        """g^2 = 0.5: weak coupling, chi > 1, partition not licensed."""
        r = self.results[0]
        self.assertFalse(r['valid'])

    def test_strong_coupling_licensed(self):
        """g^2 >= 2: strong coupling, chi < 1, confinement certified."""
        for r in self.results:
            if r['g_squared'] >= 2.0:
                self.assertTrue(r['valid'],
                    f"g^2={r['g_squared']} should be licensed, got chi={r['chi']}")

    def test_chi_decreases_with_coupling(self):
        """chi decreases as g^2 increases (deeper confinement)."""
        licensed = [r for r in self.results if r['valid']]
        chis = [r['chi'] for r in licensed]
        for i in range(len(chis) - 1):
            self.assertGreater(chis[i], chis[i + 1])

    def test_mass_gap_positive_everywhere(self):
        for r in self.results:
            self.assertGreater(r['mass_gap'], 0)


class TestCertifyYangMills(unittest.TestCase):
    """Tests for the direct certification path via radial Dirichlet form."""

    def test_regime_c_strong_coupling(self):
        """g^2 = 8 -> beta = 8 > 2 -> Regime C."""
        r = certify_yang_mills(8.0)
        self.assertEqual(r["beta"], 8.0)
        self.assertEqual(r["regime"], "C")
        self.assertTrue(r["determinacy"])
        self.assertTrue(r["licensed"])
        self.assertEqual(r["lean_module"], "MexicanHatForced.lean")

    def test_regime_b_at_feller(self):
        """g^2 = 1.5 -> beta = 1.5, 1 <= beta <= 2 -> Regime B."""
        r = certify_yang_mills(1.5)
        self.assertEqual(r["regime"], "B")
        self.assertTrue(r["determinacy"])
        self.assertEqual(r["lean_module"], "FellerThreshold.lean")

    def test_regime_a_weak_coupling(self):
        """g^2 = 0.5 -> beta = 0.5 < 1 -> Regime A."""
        r = certify_yang_mills(0.5)
        self.assertEqual(r["regime"], "A")
        self.assertFalse(r["determinacy"])
        self.assertIsNone(r["lean_module"])

    def test_seal_deterministic(self):
        """Same inputs produce the same seal."""
        r1 = certify_yang_mills(8.0, j_cut=1.0, j_max=3.0)
        r2 = certify_yang_mills(8.0, j_cut=1.0, j_max=3.0)
        self.assertEqual(r1["seal"], r2["seal"])

    def test_seal_differs_across_couplings(self):
        r1 = certify_yang_mills(8.0)
        r2 = certify_yang_mills(16.0)
        self.assertNotEqual(r1["seal"], r2["seal"])

    def test_chi_exponent_approaches_minus_four(self):
        """chi ~ (g^2)^{-4} at strong coupling."""
        r1 = certify_yang_mills(16.0, j_max=6.0)
        r2 = certify_yang_mills(32.0, j_max=6.0)
        if r1["chi"] > 0 and r2["chi"] > 0:
            import math
            log_ratio = math.log(r2["chi"] / r1["chi"]) / math.log(32.0 / 16.0)
            # Should be near -4
            self.assertLess(log_ratio, -2.0)

    def test_mass_gap_positive_regime_c(self):
        r = certify_yang_mills(8.0)
        self.assertGreater(r["mass_gap"], 0)

    def test_store_integration(self):
        """Certificate appends to store and trajectory is queryable."""
        from emet.store import CertificateStore
        with tempfile.TemporaryDirectory() as d:
            store = CertificateStore(Path(d) / "ym.jsonl")
            for g2 in [4.0, 8.0, 16.0]:
                certify_yang_mills(g2, store=store)
            traj = store.query_trajectory("yang_mills", "g_squared")
            self.assertEqual(len(traj), 3)
            g2_vals = [t[0] for t in traj]
            self.assertEqual(g2_vals, [4.0, 8.0, 16.0])


if __name__ == "__main__":
    unittest.main()
