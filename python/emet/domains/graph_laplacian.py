"""Two-clique graph Laplacian with tunable cross-coupling.

H = D - W for two complete subgraphs joined by uniform bipartite
weight w_cross. chi < 1 certifies cluster independence: the
communities decouple when inter-cluster weight is controlled by
intra-cluster structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def build_two_clique_laplacian(
    n_retained: int,
    n_omitted: int,
    w_ret: float,
    w_omit: float,
    w_cross: float,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Graph Laplacian for two cliques with uniform cross weights.

    Returns (matrix, retained_indices, omitted_indices).
    """
    if n_retained < 2 or n_omitted < 2:
        raise ValueError("require n_retained >= 2 and n_omitted >= 2")
    n = n_retained + n_omitted
    w = np.zeros((n, n))
    w[:n_retained, :n_retained] = w_ret * (np.ones((n_retained, n_retained)) - np.eye(n_retained))
    w[n_retained:, n_retained:] = w_omit * (np.ones((n_omitted, n_omitted)) - np.eye(n_omitted))
    w[:n_retained, n_retained:] = w_cross
    w[n_retained:, :n_retained] = w_cross
    np.fill_diagonal(w, 0.0)
    d = np.diag(w.sum(axis=1))
    h = d - w
    retained = list(range(n_retained))
    omitted = list(range(n_retained, n))
    return h, retained, omitted


def preset_decoupled() -> tuple[np.ndarray, list[int], list[int]]:
    """Weak cross coupling (w_cross=1e-12): subcritical, chi ~ 0."""
    return build_two_clique_laplacian(5, 5, 1.0, 2.0, 1e-12)


def preset_high_coupling() -> tuple[np.ndarray, list[int], list[int]]:
    """Stronger cross coupling (w_cross=1e-10): supercritical, chi > 1."""
    return build_two_clique_laplacian(5, 5, 1.0, 2.0, 1e-10)
