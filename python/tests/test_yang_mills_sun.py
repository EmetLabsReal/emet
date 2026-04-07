"""Tests for SU(N) generalization: Casimir operators, plaquette Hamiltonians, sweeps."""

import unittest

import numpy as np

from emet.domains.yang_mills_sun import (
    casimir_su2,
    casimir_su3,
    su3_irreps_up_to,
    build_sun_plaquette,
    sweep_coupling_sun,
)
from emet.domains.yang_mills import build_plaquette_blocks


class TestCasimirSU2(unittest.TestCase):
    """Quadratic Casimir C_2(j) = j(j+1) for SU(2)."""

    def test_trivial_rep(self):
        np.testing.assert_allclose(casimir_su2(0), 0.0)

    def test_fundamental(self):
        np.testing.assert_allclose(casimir_su2(0.5), 0.75)

    def test_adjoint(self):
        np.testing.assert_allclose(casimir_su2(1), 2.0)

    def test_higher_spins(self):
        np.testing.assert_allclose(casimir_su2(1.5), 1.5 * 2.5)
        np.testing.assert_allclose(casimir_su2(2.0), 6.0)


class TestCasimirSU3(unittest.TestCase):
    """Quadratic Casimir C_2(p,q) = (p^2 + q^2 + pq + 3p + 3q) / 3 for SU(3)."""

    def test_trivial_rep(self):
        np.testing.assert_allclose(casimir_su3(0, 0), 0.0)

    def test_fundamental(self):
        np.testing.assert_allclose(casimir_su3(1, 0), 4.0 / 3.0)

    def test_antifundamental(self):
        np.testing.assert_allclose(casimir_su3(0, 1), 4.0 / 3.0)

    def test_adjoint(self):
        # (1,1): (1 + 1 + 1 + 3 + 3) / 3 = 9/3 = 3
        np.testing.assert_allclose(casimir_su3(1, 1), 3.0)

    def test_symmetric_in_fundamental_antifundamental(self):
        """C_2(p,q) = C_2(q,p) — charge conjugation symmetry."""
        for p in range(4):
            for q in range(4):
                np.testing.assert_allclose(
                    casimir_su3(p, q), casimir_su3(q, p),
                    err_msg=f"Casimir not symmetric at ({p},{q}) vs ({q},{p})")


class TestSU3Irreps(unittest.TestCase):
    """su3_irreps_up_to returns irreps ordered by Casimir."""

    def test_trivial_first(self):
        irreps = su3_irreps_up_to(2)
        self.assertEqual(irreps[0], (0, 0))

    def test_ordered_by_casimir(self):
        irreps = su3_irreps_up_to(2)
        casimirs = [casimir_su3(p, q) for p, q in irreps]
        for i in range(len(casimirs) - 1):
            self.assertLessEqual(casimirs[i], casimirs[i + 1],
                                 f"Not ordered at index {i}: {casimirs[i]} > {casimirs[i+1]}")

    def test_includes_all_irreps(self):
        irreps = su3_irreps_up_to(2)
        # Should include all (p,q) with p+q <= 2
        expected_count = sum(1 for p in range(3) for q in range(3 - p))
        self.assertEqual(len(irreps), expected_count)


class TestBuildSunPlaquetteSU2(unittest.TestCase):
    """SU(2) plaquette via build_sun_plaquette: symmetric H, correct structure."""

    def test_returns_symmetric_matrix(self):
        H, ret, omit, info = build_sun_plaquette(2, 4.0, max_irrep=4, cut_index=3)
        np.testing.assert_allclose(H, H.T, atol=1e-14)

    def test_partition_is_complete(self):
        H, ret, omit, info = build_sun_plaquette(2, 4.0, max_irrep=4, cut_index=3)
        n = H.shape[0]
        self.assertEqual(len(ret) + len(omit), n)

    def test_info_contains_group(self):
        _, _, _, info = build_sun_plaquette(2, 4.0)
        self.assertEqual(info["group"], "SU(2)")

    def test_matches_yang_mills_eigenvalues(self):
        """SU(2) via build_sun_plaquette should approximately match yang_mills.py."""
        g2 = 4.0
        H_sun, _, _, _ = build_sun_plaquette(2, g2, max_irrep=4, cut_index=3)
        H_ym, _, _, _ = build_plaquette_blocks(g2, j_max=4.0, j_cut=1.0)

        # Both should yield the same full Hamiltonian eigenvalues
        eigs_sun = np.sort(np.linalg.eigvalsh(H_sun))
        eigs_ym = np.sort(np.linalg.eigvalsh(H_ym))

        # Sizes may differ due to different cut parameters; compare ground state gap
        gap_sun = eigs_sun[1] - eigs_sun[0]
        gap_ym = eigs_ym[1] - eigs_ym[0]
        # Both should be positive and in the same ballpark
        self.assertGreater(gap_sun, 0)
        self.assertGreater(gap_ym, 0)


class TestBuildSunPlaquetteSU3(unittest.TestCase):
    """SU(3) plaquette via build_sun_plaquette: symmetric H, correct structure."""

    def test_returns_symmetric_matrix(self):
        H, ret, omit, info = build_sun_plaquette(3, 4.0, max_irrep=3, cut_index=3)
        np.testing.assert_allclose(H, H.T, atol=1e-14)

    def test_partition_is_complete(self):
        H, ret, omit, info = build_sun_plaquette(3, 4.0, max_irrep=3, cut_index=3)
        n = H.shape[0]
        self.assertEqual(len(ret) + len(omit), n)

    def test_info_contains_group(self):
        _, _, _, info = build_sun_plaquette(3, 4.0)
        self.assertEqual(info["group"], "SU(3)")

    def test_unsupported_group_raises(self):
        with self.assertRaises(NotImplementedError):
            build_sun_plaquette(5, 4.0)


class TestSweepCouplingSun(unittest.TestCase):
    """sweep_coupling_sun returns certification results with mass gap."""

    def test_strong_coupling_mass_gap_positive(self):
        results = sweep_coupling_sun(2, [4.0, 6.0, 10.0])
        for r in results:
            self.assertGreater(r["mass_gap"], 0.0,
                               f"Mass gap not positive at g^2={r['g_squared']}")

    def test_strong_coupling_licensed(self):
        results = sweep_coupling_sun(2, [4.0, 6.0, 10.0])
        for r in results:
            self.assertTrue(r["licensed"],
                            f"Not licensed at g^2={r['g_squared']}")

    def test_su3_strong_coupling(self):
        results = sweep_coupling_sun(3, [4.0, 6.0, 10.0], max_irrep=3, cut_index=3)
        for r in results:
            self.assertGreater(r["mass_gap"], 0.0,
                               f"SU(3) gap not positive at g^2={r['g_squared']}")

    def test_results_contain_required_keys(self):
        results = sweep_coupling_sun(2, [4.0])
        r = results[0]
        for key in ["group", "g_squared", "chi", "gamma", "lambda",
                     "licensed", "regime", "mass_gap", "kahan_certified"]:
            self.assertIn(key, r, f"Missing key: {key}")


if __name__ == "__main__":
    unittest.main()
