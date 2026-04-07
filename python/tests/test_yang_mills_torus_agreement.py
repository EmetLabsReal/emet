"""Tests for Yang-Mills / torus identification.

The Kogut-Susskind Hamiltonian built via torus block assembly must
produce the same operator as the native tridiagonal construction.
"""

import unittest
import numpy as np
from emet.domains.yang_mills import (
    build_plaquette_blocks,
    build_single_plaquette,
    partition_by_representation,
    yang_mills_as_torus_params,
)
import emet


class TestMatrixAgreement(unittest.TestCase):
    """The block-assembled and native matrices must agree."""

    def test_agreement_g2_2(self):
        self._check_agreement(2.0, 3.0, 1.0)

    def test_agreement_g2_5(self):
        self._check_agreement(5.0, 4.0, 1.5)

    def test_agreement_g2_10(self):
        self._check_agreement(10.0, 3.0, 1.0)

    def _check_agreement(self, g2, j_max, j_cut):
        # Native path
        H_native, _, j_values = build_single_plaquette(g2, j_max)
        ret_native, omit_native = partition_by_representation(j_values, j_cut)

        # Block path
        H_block, ret_block, omit_block, _ = build_plaquette_blocks(g2, j_max, j_cut)

        # Partition must agree
        self.assertEqual(ret_native, ret_block)
        self.assertEqual(omit_native, omit_block)

        # Matrices must agree (same entries, same ordering)
        np.testing.assert_allclose(H_native, H_block, atol=1e-14,
            err_msg=f"Matrix mismatch at g^2={g2}, j_max={j_max}, j_cut={j_cut}")


class TestChiAgreement(unittest.TestCase):
    """Chi from both paths must be identical."""

    def test_chi_identical(self):
        for g2 in [1.0, 2.0, 5.0, 10.0]:
            H_native, _, j_values = build_single_plaquette(g2, 4.0)
            ret, omit = partition_by_representation(j_values, 1.0)
            r_native = emet.decide_dense_matrix(H_native, retained=ret, omitted=omit)

            H_block, ret_b, omit_b, _ = build_plaquette_blocks(g2, 4.0, 1.0)
            r_block = emet.decide_dense_matrix(H_block, retained=ret_b, omitted=omit_b)

            chi_native = r_native["advanced_metrics"]["chi"]
            chi_block = r_block["advanced_metrics"]["chi"]

            if chi_native is None:
                self.assertIsNone(chi_block)
            else:
                self.assertAlmostEqual(chi_native, chi_block, places=12,
                    msg=f"Chi mismatch at g^2={g2}")


class TestTorusParams(unittest.TestCase):
    def test_beta_is_g_squared(self):
        params = yang_mills_as_torus_params(5.0, 3.0, 1.0)
        self.assertEqual(params["beta_equivalent"], 5.0)

    def test_partition_sizes(self):
        params = yang_mills_as_torus_params(2.0, 3.0, 1.0)
        # j = 0, 1/2, 1 retained (3); j = 3/2, 2, 5/2, 3 omitted (4)
        self.assertEqual(params["n_valley"], 3)
        self.assertEqual(params["n_barrier"], 4)

    def test_tunneling_amplitude(self):
        params = yang_mills_as_torus_params(4.0, 3.0, 1.0)
        self.assertAlmostEqual(params["tunneling_amplitude"], 0.25)


if __name__ == "__main__":
    unittest.main()
