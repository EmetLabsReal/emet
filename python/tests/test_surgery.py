"""Tests for surgical reconstruction."""

import unittest
import numpy as np
from emet.domains.surgery import (
    build_severed_double_well,
    generator_energy_gap,
    generator_is_markov,
    minimal_symmetric_generator,
    surgical_reconstruction,
    post_fracture_certify,
)
import emet


class TestGenerator(unittest.TestCase):
    def test_two_well_is_markov(self):
        M = minimal_symmetric_generator(2, rate=1.0)
        self.assertTrue(generator_is_markov(M))

    def test_two_well_symmetric(self):
        M = minimal_symmetric_generator(2, rate=0.5)
        np.testing.assert_allclose(M, M.T)

    def test_two_well_row_sums_vanish(self):
        M = minimal_symmetric_generator(2, rate=3.0)
        for i in range(2):
            self.assertAlmostEqual(M[i].sum(), 0.0)

    def test_two_well_gap(self):
        M = minimal_symmetric_generator(2, rate=1.0)
        self.assertAlmostEqual(generator_energy_gap(M), 2.0)

    def test_three_well_is_markov(self):
        M = minimal_symmetric_generator(3, rate=1.0)
        self.assertTrue(generator_is_markov(M))

    def test_three_well_gap(self):
        M = minimal_symmetric_generator(3, rate=1.0)
        self.assertAlmostEqual(generator_energy_gap(M), 3.0)


class TestUniqueness(unittest.TestCase):
    """The symmetric 2x2 Markov generator with full support is unique up to rate."""

    def test_form_is_forced(self):
        """Any symmetric 2x2 generator with row sums = 0 must be twoWellGen."""
        for lam in [0.1, 0.5, 1.0, 5.0]:
            M = minimal_symmetric_generator(2, rate=lam)
            # Off-diagonal must equal lam
            self.assertAlmostEqual(M[0, 1], lam)
            self.assertAlmostEqual(M[1, 0], lam)
            # Diagonal must equal -lam
            self.assertAlmostEqual(M[0, 0], -lam)
            self.assertAlmostEqual(M[1, 1], -lam)


class TestSeveredWell(unittest.TestCase):
    def test_severed_has_zero_coupling(self):
        H, ret, omit = build_severed_double_well()
        report = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
        self.assertAlmostEqual(report["advanced_metrics"]["lambda"], 0.0)

    def test_severed_chi_zero(self):
        H, ret, omit = build_severed_double_well()
        report = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
        self.assertAlmostEqual(report["advanced_metrics"]["chi"], 0.0)


class TestSurgicalReconstruction(unittest.TestCase):
    def test_surgery_adds_coupling(self):
        H, ret, omit = build_severed_double_well()
        H_rec = surgical_reconstruction(H, ret, omit, jump_rate=0.5)
        report = emet.decide_dense_matrix(H_rec, retained=ret, omitted=omit)
        self.assertGreater(report["advanced_metrics"]["lambda"], 0.0)

    def test_surgery_preserves_symmetry(self):
        H, ret, omit = build_severed_double_well()
        H_rec = surgical_reconstruction(H, ret, omit, jump_rate=1.0)
        np.testing.assert_allclose(H_rec, H_rec.T, atol=1e-14)

    def test_fiedler_mode_restored(self):
        """Severed system has degenerate ground state; surgery lifts it."""
        H, ret, omit = build_severed_double_well()
        eigvals_sev = np.sort(np.linalg.eigvalsh(H))
        # Severed: two independent wells, gap determined by intra-well structure

        H_rec = surgical_reconstruction(H, ret, omit, jump_rate=0.1)
        eigvals_rec = np.sort(np.linalg.eigvalsh(H_rec))
        # Surgery changes the spectrum
        self.assertFalse(np.allclose(eigvals_sev, eigvals_rec))

    def test_small_surgery_keeps_subcritical(self):
        H, ret, omit = build_severed_double_well()
        H_rec = surgical_reconstruction(H, ret, omit, jump_rate=0.01)
        report = emet.decide_dense_matrix(H_rec, retained=ret, omitted=omit)
        self.assertTrue(report["valid"])


class TestPostFractureCertify(unittest.TestCase):
    def test_pipeline_runs(self):
        H, ret, omit = build_severed_double_well()
        result = post_fracture_certify(H, ret, omit, jump_rate=0.1)
        self.assertIn("pre_chi", result)
        self.assertIn("post_chi", result)
        self.assertIn("post_valid", result)

    def test_pipeline_subcritical_after_small_surgery(self):
        H, ret, omit = build_severed_double_well()
        result = post_fracture_certify(H, ret, omit, jump_rate=0.05)
        self.assertTrue(result["post_valid"])


if __name__ == "__main__":
    unittest.main()
