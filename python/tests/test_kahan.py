"""Tests for the Kahan precision envelope."""

import unittest
from emet.domains.kahan import (
    UNIT_ROUNDOFF,
    certified_subcritical,
    pessimistic_chi_envelope,
)


class TestEnvelope(unittest.TestCase):
    def test_zero_perturbation(self):
        eps = pessimistic_chi_envelope(0.5, 0.0, 1.0, 0.0, 1.0)
        self.assertAlmostEqual(eps, 0.0)

    def test_scales_with_chi(self):
        eps1 = pessimistic_chi_envelope(0.1, 0.01, 1.0, 0.01, 1.0)
        eps2 = pessimistic_chi_envelope(0.5, 0.01, 1.0, 0.01, 1.0)
        self.assertLess(eps1, eps2)

    def test_inf_on_zero_gamma(self):
        eps = pessimistic_chi_envelope(0.5, 0.01, 1.0, 0.01, 0.0)
        self.assertEqual(eps, float("inf"))


class TestCertification(unittest.TestCase):
    def test_deep_subcritical_certifies(self):
        result = certified_subcritical(0.01, gamma=10.0, lam=1.0)
        self.assertTrue(result["certified"])
        self.assertGreater(result["security_margin"], 0.9)

    def test_supercritical_fails(self):
        result = certified_subcritical(1.5, gamma=1.0, lam=1.5)
        self.assertFalse(result["certified"])

    def test_near_boundary_still_certifies(self):
        """chi = 0.5 with reasonable gamma/lambda should certify."""
        result = certified_subcritical(0.5, gamma=5.0, lam=3.0)
        self.assertTrue(result["certified"])
        self.assertGreater(result["security_margin"], 0.0)

    def test_eps_chi_is_tiny(self):
        """Machine roundoff produces eps_chi ~ 4u * chi."""
        result = certified_subcritical(0.5, gamma=10.0, lam=5.0)
        self.assertLess(result["eps_chi"], 1e-14)


class TestYangMillsSqueeze(unittest.TestCase):
    """Verify Kahan certification on actual Yang-Mills chi values."""

    def test_strong_coupling_certified(self):
        from emet.domains.yang_mills import build_plaquette_blocks
        import emet

        for g2 in [2.0, 5.0, 10.0]:
            H, ret, omit, _ = build_plaquette_blocks(g2, j_max=4.0, j_cut=1.0)
            report = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
            m = report["advanced_metrics"]
            result = certified_subcritical(m["chi"], m["gamma"], m["lambda"])
            self.assertTrue(result["certified"],
                f"g^2={g2}: chi={m['chi']:.6e} not Kahan-certified")


if __name__ == "__main__":
    unittest.main()
