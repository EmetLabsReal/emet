"""Tests for the Schumann resonance domain adapter."""

import numpy as np
import pytest

import emet
from emet.domains.schumann import (
    build_schumann_blocks,
    schumann_frequencies_exact,
    schumann_from_schur,
    _radial_overlap,
)


class TestRadialOverlap:
    def test_diagonal(self):
        assert _radial_overlap(1, 1) == pytest.approx(0.5)
        assert _radial_overlap(3, 3) == pytest.approx(0.5)

    def test_same_parity_zero(self):
        assert _radial_overlap(1, 3) == 0.0
        assert _radial_overlap(2, 4) == 0.0

    def test_different_parity_nonzero(self):
        assert _radial_overlap(1, 2) != 0.0
        assert _radial_overlap(2, 3) != 0.0

    def test_symmetry(self):
        assert _radial_overlap(1, 2) == pytest.approx(_radial_overlap(2, 1))
        assert _radial_overlap(2, 5) == pytest.approx(_radial_overlap(5, 2))


class TestBuildBlocks:
    def test_hermitian(self):
        H, ret, omit, labels = build_schumann_blocks()
        assert np.allclose(H, H.T)

    def test_partition_covers(self):
        H, ret, omit, labels = build_schumann_blocks(n_angular=6, n_radial=4)
        n = H.shape[0]
        assert sorted(ret + omit) == list(range(n))
        assert len(set(ret) & set(omit)) == 0

    def test_retained_count(self):
        n_ang = 8
        H, ret, omit, labels = build_schumann_blocks(n_angular=n_ang)
        assert len(ret) == n_ang

    def test_labels(self):
        H, ret, omit, labels = build_schumann_blocks(n_angular=3, n_radial=2)
        assert len(labels) == H.shape[0]
        assert labels[0] == "(n=0, k=1)"
        assert labels[1] == "(n=0, k=2)"


class TestSubcritical:
    def test_thin_shell_subcritical(self):
        """Thin shell (δ/R = 0.01) should be subcritical."""
        H, ret, omit, _ = build_schumann_blocks(delta_over_R=0.01)
        r = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
        assert r["regime"] == "subcritical"
        assert r["advanced_metrics"]["chi"] < 1

    def test_thinner_shell_smaller_chi(self):
        """Thinner shell → smaller χ (radial modes more separated)."""
        chis = []
        for eps in [0.05, 0.02, 0.01, 0.005]:
            H, ret, omit, _ = build_schumann_blocks(delta_over_R=eps)
            r = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
            chis.append(r["advanced_metrics"]["chi"])
        # χ should decrease as shell gets thinner
        for i in range(len(chis) - 1):
            assert chis[i] > chis[i + 1]

    def test_earth_parameters(self):
        """Earth-like parameters (δ/R ≈ 0.013) should be subcritical."""
        eps = 80e3 / 6.371e6  # ≈ 0.0126
        H, ret, omit, _ = build_schumann_blocks(delta_over_R=eps)
        r = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
        assert r["regime"] == "subcritical"


class TestSchumannFrequencies:
    def test_exact_first_mode(self):
        """f₁ ≈ 10.6 Hz (ideal cavity; observed 7.83 Hz includes refractive index)."""
        freqs = schumann_frequencies_exact(1)
        assert freqs[0] == pytest.approx(10.6, abs=0.1)

    def test_exact_increasing(self):
        freqs = schumann_frequencies_exact(7)
        for i in range(len(freqs) - 1):
            assert freqs[i] < freqs[i + 1]

    def test_schur_recovers_schumann(self):
        """Schur complement eigenvalues should approximate Schumann frequencies."""
        result = schumann_from_schur(delta_over_R=0.005, n_angular=8, n_radial=6)
        assert result["regime"] == "subcritical"
        assert result["relative_errors"] is not None
        # First few modes should be within 1% of exact
        for err in result["relative_errors"][:3]:
            assert err < 0.01
