"""Ramanujan hierarchy: spectral gap, chi bound, attention mask.

1. Chi bound is subcritical when d_x < gap.
2. Chi bound is supercritical when d_x > gap.
3. Max inter-degree matches floor(d - 2*sqrt(d-1)).
4. Attention mask is symmetric with correct cluster structure.
5. Hierarchy from Rust engine returns valid structure.
"""

import numpy as np
import pytest

from emet.domains.ramanujan import (
    alon_boppana_bound,
    build_hierarchy,
    chi_bound,
    max_inter_degree,
    ramanujan_gap,
)


class TestSpectralGap:
    def test_alon_boppana_values(self):
        # d=7: 2*sqrt(6) ≈ 4.899
        assert abs(alon_boppana_bound(7) - 2 * np.sqrt(6)) < 1e-10

    def test_ramanujan_gap_positive_for_d_ge_3(self):
        for d in range(3, 100):
            assert ramanujan_gap(d) > 0

    def test_ramanujan_gap_d2(self):
        # d=2: gap = 2 - 2*1 = 0
        assert abs(ramanujan_gap(2)) < 1e-10


class TestChiBound:
    def test_subcritical_small_cross(self):
        # cluster_size=8 (d=7), d_x=1: chi = (1/2.10)^2 ≈ 0.227
        assert chi_bound(8, 1) < 1.0

    def test_supercritical_large_cross(self):
        # d_x=5 > 2.10
        assert chi_bound(8, 5) > 1.0

    def test_monotone_in_inter_degree(self):
        bounds = [chi_bound(16, dx) for dx in range(1, 10)]
        for i in range(len(bounds) - 1):
            assert bounds[i] < bounds[i + 1]


class TestMaxInterDegree:
    def test_cluster_8(self):
        # d=7: gap ≈ 2.10, floor = 2
        assert max_inter_degree(8) == 2

    def test_cluster_16(self):
        # d=15: gap ≈ 7.51, floor = 7
        assert max_inter_degree(16) == 7

    def test_cluster_64(self):
        # d=63: gap ≈ 47.26, floor = 47
        assert max_inter_degree(64) == 47

    def test_chi_bound_at_max_is_subcritical(self):
        for cs in [8, 16, 32, 64]:
            dx = max_inter_degree(cs)
            assert chi_bound(cs, dx) < 1.0


class TestHierarchyConstruction:
    def test_basic_structure(self):
        h = build_hierarchy(64, 8, 1, seed=42)
        assert h["n_tokens"] == 64
        assert h["cluster_size"] == 8
        assert h["n_clusters"] == 8
        assert h["inter_degree"] == 1
        assert h["depth"] >= 1

    def test_mask_symmetric(self):
        h = build_hierarchy(32, 8, 2, seed=42)
        mask = h["attention_mask"]
        np.testing.assert_array_equal(mask, mask.T)

    def test_mask_no_self_loops(self):
        h = build_hierarchy(32, 8, 2, seed=42)
        mask = h["attention_mask"]
        assert np.all(np.diag(mask) == 0)

    def test_intra_cluster_complete(self):
        h = build_hierarchy(32, 8, 1, seed=42)
        mask = h["attention_mask"]
        # First cluster: tokens 0-7 should all be connected
        cluster = mask[:8, :8]
        expected = np.ones((8, 8)) - np.eye(8)
        np.testing.assert_array_equal(cluster, expected)

    def test_chi_reported(self):
        h = build_hierarchy(64, 16, 1, seed=42)
        assert "chi" in h
        assert "gamma" in h
        assert "lambda" in h
        assert h["chi"] >= 0

    def test_subcritical_flag(self):
        h = build_hierarchy(64, 16, 1, seed=42)
        assert h["subcritical"] == (h["chi"] < 1.0)
