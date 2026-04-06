"""Tests for the 2D Ising block-spin domain adapter."""

import unittest
import numpy as np
from emet.domains.ising_block_spin import (
    ising_energies,
    block_spin_partition,
    sweep_temperature,
    chi_ratio_at_Tc,
    scaling_analysis,
    T_C,
    BETA_C,
    J_DEFAULT,
)


class TestIsingEnergies(unittest.TestCase):
    def test_ground_state_energy_2x2(self):
        """2×2 periodic Ising: ground state energy = -8J.

        PBC on 2×2: site (1,y) wraps to (0,y), site (x,1) wraps to (x,0).
        Each of 4 sites contributes 2 bonds (right, up), but each bond
        is counted once in the sum.  With Lx=Ly=2 and PBC, there are
        2*Lx*Ly = 8 bonds.  Ground state: all aligned, E = -8J.
        """
        configs, energies = ising_energies(2, 2)
        self.assertAlmostEqual(np.min(energies), -8.0 * J_DEFAULT)

    def test_z2_symmetry(self):
        """Flipping all spins doesn't change the energy."""
        configs, energies = ising_energies(2, 2)
        # configs[0] = [-1,-1,-1,-1], configs[-1] = [+1,+1,+1,+1]
        # Both should have the same (ground state) energy.
        self.assertAlmostEqual(energies[0], energies[-1])

    def test_external_field_breaks_symmetry(self):
        """External field h > 0 favors + spins."""
        configs, e0 = ising_energies(2, 2, h=0.0)
        configs, eh = ising_energies(2, 2, h=0.5)
        # All +1 state should have lower energy with h > 0
        all_plus = np.argmax(configs.sum(axis=1))
        all_minus = np.argmin(configs.sum(axis=1))
        self.assertLess(eh[all_plus], eh[all_minus])

    def test_config_count(self):
        """2^N configurations for N = Lx*Ly."""
        configs, energies = ising_energies(3, 2)
        self.assertEqual(len(configs), 2**6)
        self.assertEqual(configs.shape[1], 6)


class TestBlockSpinPartition(unittest.TestCase):
    def test_partition_sizes(self):
        """4×4 with 2×2 blocks: 4 retained, 12 omitted."""
        r = block_spin_partition(4, 4, 2, 2, BETA_C)
        self.assertEqual(r["n_retained"], 4)
        self.assertEqual(r["n_omitted"], 12)

    def test_covariance_symmetric(self):
        """C_PP and C_QQ are symmetric."""
        r = block_spin_partition(4, 4, 2, 2, BETA_C)
        np.testing.assert_allclose(r["C_PP"], r["C_PP"].T, atol=1e-14)
        np.testing.assert_allclose(r["C_QQ"], r["C_QQ"].T, atol=1e-14)

    def test_covariance_psd(self):
        """C_QQ is positive semidefinite."""
        r = block_spin_partition(4, 4, 2, 2, 1.0 / (3.0 * T_C))
        eigvals = np.linalg.eigvalsh(r["C_QQ"])
        self.assertTrue(np.all(eigvals >= -1e-14))

    def test_block_size_must_divide(self):
        with self.assertRaises(ValueError):
            block_spin_partition(4, 4, 3, 2, BETA_C)

    def test_chi_large_at_Tc(self):
        """χ >> 1 at T_c for 4×4 lattice."""
        r = block_spin_partition(4, 4, 2, 2, BETA_C)
        self.assertGreater(r["chi"], 100)

    def test_chi_small_at_high_T(self):
        """χ < 1 at T >> T_c."""
        r = block_spin_partition(4, 4, 2, 2, 1.0 / (3.0 * T_C))
        self.assertLess(r["chi"], 1.0)


class TestCapacityTransition(unittest.TestCase):
    """The core result: Cap = 0 (licensed) at high T, Cap > 0 (unlicensed) at T_c."""

    def test_licensed_at_high_T(self):
        """At T = 3*T_c, χ < 1: block-spin reduction is licensed."""
        r = block_spin_partition(4, 4, 2, 2, 1.0 / (3.0 * T_C))
        self.assertLess(r["chi"], 1.0)

    def test_unlicensed_at_Tc(self):
        """At T_c, χ >> 1: block-spin reduction fails."""
        r = block_spin_partition(4, 4, 2, 2, BETA_C)
        self.assertGreater(r["chi"], 100)

    def test_scaling_grows_at_Tc(self):
        """χ(4×4) / χ(4×2) > 2 at T_c: divergent scaling."""
        r1 = block_spin_partition(4, 2, 2, 2, BETA_C)
        r2 = block_spin_partition(4, 4, 2, 2, BETA_C)
        ratio = r2["chi"] / r1["chi"]
        self.assertGreater(ratio, 2.0)

    def test_scaling_converges_at_high_T(self):
        """χ(4×4) / χ(4×2) < 1.5 at T = 3*T_c: convergent scaling."""
        beta = 1.0 / (3.0 * T_C)
        r1 = block_spin_partition(4, 2, 2, 2, beta)
        r2 = block_spin_partition(4, 4, 2, 2, beta)
        ratio = r2["chi"] / r1["chi"]
        self.assertLess(ratio, 1.5)


class TestSweep(unittest.TestCase):
    def test_sweep_returns_list(self):
        results = sweep_temperature(Lx=4, Ly=2, bx=2, by=2,
                                     T_values=np.array([1.0, 2.0, 3.0]))
        self.assertEqual(len(results), 3)

    def test_sweep_chi_decreases_with_T(self):
        """In the disordered regime, χ decreases with T."""
        results = sweep_temperature(Lx=4, Ly=2, bx=2, by=2,
                                     T_values=np.linspace(3.0, 8.0, 5))
        chis = [r["chi"] for r in results]
        for i in range(len(chis) - 1):
            self.assertGreater(chis[i], chis[i + 1])


class TestExactValues(unittest.TestCase):
    def test_T_c_value(self):
        """Onsager's exact T_c."""
        self.assertAlmostEqual(T_C, 2.0 / np.log(1.0 + np.sqrt(2.0)), places=10)

    def test_chi_ratio(self):
        """chi_ratio_at_Tc returns three regimes."""
        r = chi_ratio_at_Tc(Lx=4, Ly=2, bx=2, by=2)
        self.assertIn("T_c", r)
        self.assertIn("ordered", r)
        self.assertIn("disordered", r)
        # Disordered should have much smaller chi than T_c
        self.assertGreater(r["T_c"]["chi"], 10 * r["disordered"]["chi"])

    def test_scaling_analysis(self):
        """scaling_analysis returns ratios for multiple temperatures."""
        r = scaling_analysis()
        self.assertIn("T_c", r)
        self.assertIn("3_T_c", r)
        # Ratio at T_c should be larger than at 3*T_c
        self.assertGreater(r["T_c"]["ratio"], r["3_T_c"]["ratio"])


if __name__ == "__main__":
    unittest.main()
