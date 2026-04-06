"""Tests for 4D dimension-dependent torus: geometry, Feller threshold, certification."""

import unittest

import numpy as np

from emet.domains.torus_4d import (
    pinching_exponent,
    feller_critical_coupling,
    feller_critical_alpha,
    build_4d_torus_operator,
    certify_4d,
    dimension_sweep,
)


class TestPinchingExponent(unittest.TestCase):
    """Alpha = g^2 / d: the pinching exponent in d dimensions."""

    def test_basic_values(self):
        np.testing.assert_allclose(pinching_exponent(4, 4.0), 1.0)
        np.testing.assert_allclose(pinching_exponent(4, 2.0), 0.5)
        np.testing.assert_allclose(pinching_exponent(2, 2.0), 1.0)

    def test_scales_with_g_squared(self):
        d = 4
        for g2 in [1.0, 2.0, 5.0, 10.0]:
            np.testing.assert_allclose(pinching_exponent(d, g2), g2 / d)

    def test_scales_inversely_with_dimension(self):
        g2 = 4.0
        for d in [2, 3, 4, 5, 6]:
            np.testing.assert_allclose(pinching_exponent(d, g2), g2 / d)


class TestFellerCritical(unittest.TestCase):
    """Feller threshold: g^2 = 1 for all d, but alpha = 1/d is dimension-dependent."""

    def test_critical_coupling_is_one(self):
        for d in [2, 3, 4, 5, 6, 10]:
            np.testing.assert_allclose(feller_critical_coupling(d), 1.0)

    def test_critical_alpha_is_one_over_d(self):
        for d in [2, 3, 4, 5, 6]:
            np.testing.assert_allclose(feller_critical_alpha(d), 1.0 / d)

    def test_alpha_one_quarter_at_d4(self):
        """In 4D, Feller threshold gives alpha = 1/4."""
        np.testing.assert_allclose(feller_critical_alpha(4), 0.25)


class TestBuild4dTorusOperator(unittest.TestCase):
    """build_4d_torus_operator returns a valid block operator with metadata."""

    def test_returns_symmetric_matrix(self):
        H, ret, omit, params = build_4d_torus_operator(4.0)
        np.testing.assert_allclose(H, H.T, atol=1e-14)

    def test_retained_omitted_partition(self):
        H, ret, omit, params = build_4d_torus_operator(4.0)
        n = H.shape[0]
        self.assertEqual(len(ret) + len(omit), n)
        self.assertEqual(sorted(ret + omit), list(range(n)))

    def test_params_contain_geometry(self):
        H, ret, omit, params = build_4d_torus_operator(4.0, d=4)
        self.assertEqual(params["d"], 4)
        np.testing.assert_allclose(params["g_squared"], 4.0)
        np.testing.assert_allclose(params["alpha"], 1.0)
        np.testing.assert_allclose(params["feller_alpha"], 0.25)
        self.assertTrue(params["past_feller"])

    def test_default_dimension_is_4(self):
        _, _, _, params = build_4d_torus_operator(4.0)
        self.assertEqual(params["d"], 4)


class TestCertify4d(unittest.TestCase):
    """certify_4d at strong coupling: chi < 1, licensed, positive mass gap."""

    def test_strong_coupling_licensed(self):
        result = certify_4d(4.0)
        self.assertTrue(result["licensed"], "g^2=4.0 should be licensed")
        self.assertLess(result["chi"], 1.0)

    def test_strong_coupling_mass_gap(self):
        result = certify_4d(4.0)
        self.assertGreater(result["mass_gap"], 0.0)

    def test_past_feller(self):
        result = certify_4d(4.0)
        self.assertTrue(result["past_feller"])

    def test_kahan_certified(self):
        result = certify_4d(4.0)
        self.assertTrue(result["kahan_certified"],
                        "Strong coupling should be Kahan-certified")

    def test_sweep_strong_couplings(self):
        for g2 in [3.0, 5.0, 10.0]:
            result = certify_4d(g2)
            self.assertTrue(result["licensed"], f"Not licensed at g^2={g2}")
            self.assertLess(result["chi"], 1.0, f"chi >= 1 at g^2={g2}")


class TestDimensionSweep(unittest.TestCase):
    """dimension_sweep shows geometry changes with spacetime dimension."""

    def test_returns_list_of_dicts(self):
        results = dimension_sweep(4.0, dimensions=[2, 3, 4])
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("chi", r)
            self.assertIn("alpha", r)
            self.assertIn("d", r)

    def test_alpha_decreases_with_dimension(self):
        results = dimension_sweep(4.0, dimensions=[2, 3, 4, 5, 6])
        alphas = [r["alpha"] for r in results]
        for i in range(len(alphas) - 1):
            self.assertGreater(alphas[i], alphas[i + 1],
                               "Alpha should decrease as d increases")

    def test_all_past_feller_at_strong_coupling(self):
        results = dimension_sweep(4.0, dimensions=[2, 3, 4, 5, 6])
        for r in results:
            self.assertTrue(r["past_feller"],
                            f"d={r['d']} should be past Feller at g^2=4.0")

    def test_default_dimensions(self):
        results = dimension_sweep(4.0)
        dims = [r["d"] for r in results]
        self.assertEqual(dims, [2, 3, 4, 5, 6])


class TestFellerAlphaAt4d(unittest.TestCase):
    """Verify alpha = 1/4 at Feller threshold for d=4."""

    def test_alpha_one_quarter(self):
        g2_feller = feller_critical_coupling(4)
        alpha = pinching_exponent(4, g2_feller)
        np.testing.assert_allclose(alpha, 0.25)

    def test_feller_alpha_matches_pinching(self):
        """feller_critical_alpha(d) should equal pinching_exponent(d, feller_critical_coupling(d))."""
        for d in [2, 3, 4, 5, 6]:
            g2_f = feller_critical_coupling(d)
            alpha_from_pinch = pinching_exponent(d, g2_f)
            alpha_from_feller = feller_critical_alpha(d)
            np.testing.assert_allclose(alpha_from_pinch, alpha_from_feller,
                                       err_msg=f"Mismatch at d={d}")


if __name__ == "__main__":
    unittest.main()
