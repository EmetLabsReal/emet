"""Tests for Weil explicit formula adapter and A11 decomposition."""

import math

import numpy as np
import pytest

from emet.domains.weil_explicit import (
    compute_zeta_zeros,
    gaussian_test,
    gaussian_fourier,
    weil_spectral_side,
    weil_geometric_side,
    weil_integral_terms,
    weil_explicit_verify,
    cosh_envelope,
    cosh_integral,
    cosh_divergence_witness,
    heat_kernel_markov_test,
    build_weil_partition,
)


class TestZetaZeros:
    """Test computation of zeta zeros."""

    def test_first_zero(self):
        zeros = compute_zeta_zeros(1)
        assert len(zeros) == 1
        # First zero: t ≈ 14.134725
        assert abs(zeros[0] - 14.134725) < 0.001

    def test_zeros_positive(self):
        zeros = compute_zeta_zeros(10)
        assert all(t > 0 for t in zeros)

    def test_zeros_increasing(self):
        zeros = compute_zeta_zeros(10)
        for i in range(len(zeros) - 1):
            assert zeros[i] < zeros[i + 1]


class TestGaussianTestFunction:
    """Test Gaussian test function and its Fourier transform."""

    def test_gaussian_at_zero(self):
        assert gaussian_test(0.0, 0.1) == pytest.approx(1.0)

    def test_gaussian_positive(self):
        r = np.linspace(-10, 10, 100)
        h = gaussian_test(r, 0.1)
        assert all(h > 0)

    def test_fourier_at_zero(self):
        alpha = 0.1
        expected = math.sqrt(math.pi / alpha)
        assert gaussian_fourier(0.0, alpha) == pytest.approx(expected)

    def test_fourier_positive(self):
        x = np.linspace(-5, 5, 100)
        h_hat = gaussian_fourier(x, 0.1)
        assert all(h_hat > 0)


class TestWeilExplicitFormula:
    """Test the Weil explicit formula balance."""

    def test_spectral_side_positive(self):
        zeros = compute_zeta_zeros(10)
        s = weil_spectral_side(zeros, alpha=0.1)
        assert s > 0

    def test_explicit_formula_structure(self):
        result = weil_explicit_verify(N_zeros=20, P_max=1000, alpha=0.5)
        assert "spectral_side" in result
        assert "geometric_side" in result
        assert "residual" in result
        assert result["N_zeros"] == 20


class TestLemmaA_CoshEnvelope:
    """Lemma (a): off-line zero → cosh(δv) envelope."""

    def test_cosh_at_zero(self):
        v = np.array([0.0])
        env = cosh_envelope(0.5, v)
        assert env[0] == pytest.approx(1.0)

    def test_cosh_symmetric(self):
        v = np.array([-5.0, 5.0])
        env = cosh_envelope(0.5, v)
        assert env[0] == pytest.approx(env[1])

    def test_cosh_grows(self):
        v = np.array([1.0, 10.0, 100.0])
        env = cosh_envelope(0.5, v)
        assert env[0] < env[1] < env[2]

    def test_cosh_ge_one(self):
        """cosh(δv) ≥ 1 for all v. This is the Lean-proved bound."""
        v = np.linspace(-100, 100, 10000)
        env = cosh_envelope(0.3, v)
        assert all(env >= 1.0 - 1e-15)


class TestLemmaB_CoshNotL1:
    """Lemma (b): cosh(δv) not L¹ against Haar."""

    def test_cosh_integral_formula(self):
        """∫_{-R}^{R} cosh(δv) dv = 2·sinh(δR)/δ."""
        delta = 0.5
        R = 10.0
        expected = 2.0 * math.sinh(delta * R) / delta
        assert cosh_integral(delta, R) == pytest.approx(expected)

    def test_cosh_integral_diverges(self):
        """Integral grows without bound as R → ∞."""
        delta = 0.1
        integrals = [cosh_integral(delta, R) for R in [10, 100, 1000]]
        assert integrals[0] < integrals[1] < integrals[2]

    def test_divergence_witness(self):
        """Can find R making integral exceed any M."""
        delta = 0.5
        for M in [100, 1e6, 1e12]:
            R = cosh_divergence_witness(delta, M)
            assert R > 0
            assert cosh_integral(delta, R) > M

    def test_delta_zero_bounded(self):
        """For δ = 0, cosh(0) = 1, integral = 2R (finite)."""
        v = np.linspace(-100, 100, 10000)
        dv = v[1] - v[0]
        integral = float(np.sum(np.cosh(0.0 * v)) * dv)
        assert integral == pytest.approx(200.0, rel=0.01)


class TestLemmaC_MarkovBreaks:
    """Lemma (c): non-L¹ heat kernel breaks Markov."""

    def test_markov_breaks_offline(self):
        """δ > 0: heat kernel integral diverges."""
        result = heat_kernel_markov_test(delta=0.5, R_max=100.0)
        assert result["diverges"] is True
        assert not result["markov_holds"]

    def test_markov_holds_online(self):
        """δ = 0: heat kernel integral is bounded."""
        result = heat_kernel_markov_test(delta=0.0, R_max=100.0)
        assert result["diverges"] is False
        assert result["markov_holds"]

    def test_integral_grows_with_R(self):
        """For fixed δ > 0, integral grows as R increases."""
        delta = 0.1
        results = [heat_kernel_markov_test(delta, R_max=R) for R in [10, 100, 1000]]
        integrals = [r["integral_exact"] for r in results]
        assert integrals[0] < integrals[1] < integrals[2]


class TestPartition:
    """Test weil partition construction for emet."""

    def test_partition_shape(self):
        H, ret, omit, meta = build_weil_partition(N_zeros=20, N_retain=10)
        assert H.shape == (20, 20)
        assert len(ret) == 10
        assert len(omit) == 10

    def test_partition_symmetric(self):
        H, _, _, _ = build_weil_partition(N_zeros=20, N_retain=10)
        assert np.allclose(H, H.T)

    def test_eigenvalues_positive(self):
        H, _, _, meta = build_weil_partition(N_zeros=20, N_retain=10)
        eigs = np.linalg.eigvalsh(H)
        assert all(eigs > 0)


class TestInverseSpectral:
    """Test inverse spectral consistency check."""

    def test_consistency_structure(self):
        """Verify the reconstruction pipeline runs and returns expected fields."""
        from emet.domains.inverse_spectral import verify_consistency
        result = verify_consistency(N_zeros=20, t_max=10.0, N_grid=100)
        assert result["expected_V"] == 0.25
        assert result["N_zeros"] == 20
        assert len(result["V"]) == 100
        assert len(result["t_grid"]) == 100
