"""Ramanujan hierarchy for attention.

Constructs d-regular attention patterns with spectral gap >= d - 2sqrt(d-1).
Tokens are arranged in clusters of size `cluster_size` with intra-cluster
complete connectivity and inter-cluster degree d_x.

When d_x < d - 2sqrt(d-1), the hierarchy guarantees chi < 1 at every level:
the partition boundary has zero capacity and the reduction is faithful.

The Rust engine provides the graph construction and chi computation.
This adapter wraps it for Python and adds the emet.decide integration.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

import emet


def alon_boppana_bound(cluster_degree: int) -> float:
    """Alon-Boppana bound: 2*sqrt(d-1) for d-regular graphs."""
    return 2.0 * np.sqrt(cluster_degree - 1)


def ramanujan_gap(cluster_degree: int) -> float:
    """Spectral gap for a Ramanujan d-regular graph: d - 2*sqrt(d-1)."""
    return cluster_degree - alon_boppana_bound(cluster_degree)


def chi_bound(cluster_size: int, inter_degree: int) -> float:
    """Graph-theoretic chi bound: (d_x / gap)^2.

    Returns the guaranteed upper bound on chi for a Ramanujan hierarchy
    with complete intra-cluster connectivity (degree = cluster_size - 1)
    and inter-cluster degree d_x.
    """
    d = cluster_size - 1
    gap = ramanujan_gap(d)
    if gap <= 0:
        return float("inf")
    return (inter_degree / gap) ** 2


def max_inter_degree(cluster_size: int) -> int:
    """Maximum inter-cluster degree for chi < 1.

    Returns floor(d - 2*sqrt(d-1)) where d = cluster_size - 1.
    """
    return emet.ramanujan_max_inter_degree(cluster_size - 1)


def build_hierarchy(
    n_tokens: int,
    cluster_size: int,
    inter_degree: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a Ramanujan hierarchy and compute chi.

    Returns a dict with:
        n_tokens, cluster_size, inter_degree, n_clusters, depth,
        chi, gamma, lambda, subcritical, max_allowed_inter_degree,
        attention_mask (as numpy array).
    """
    result_json = emet.build_ramanujan_hierarchy_json(
        n_tokens, cluster_size, inter_degree, seed
    )
    result = json.loads(result_json)
    result["attention_mask"] = np.array(result["attention_mask"])
    return result


def build_and_decide(
    n_tokens: int,
    cluster_size: int,
    inter_degree: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Build hierarchy, then run emet.decide on each cluster pair.

    Returns hierarchy info plus per-pair emet decision reports.
    """
    hierarchy = build_hierarchy(n_tokens, cluster_size, inter_degree, seed)
    mask = hierarchy["attention_mask"]
    n = hierarchy["n_tokens"]
    cs = cluster_size
    n_clusters = hierarchy["n_clusters"]

    # Build the Laplacian
    degree_vec = mask.sum(axis=1)
    laplacian = np.diag(degree_vec) - mask

    pair_reports = []
    for c1 in range(n_clusters):
        for c2 in range(n_clusters):
            if c1 == c2:
                continue
            base1 = c1 * cs
            end1 = min(base1 + cs, n)
            base2 = c2 * cs
            end2 = min(base2 + cs, n)

            p_idx = list(range(base1, end1))
            q_idx = list(range(base2, end2))

            # Check connectivity
            cross = mask[np.ix_(p_idx, q_idx)]
            if cross.sum() == 0:
                continue

            # Build combined block for emet.decide
            combined_idx = p_idx + q_idx
            block = laplacian[np.ix_(combined_idx, combined_idx)]

            report = emet.decide_dense_matrix(
                block.tolist(),
                retained=list(range(len(p_idx))),
                omitted=list(range(len(p_idx), len(combined_idx))),
            )
            pair_reports.append({
                "cluster_p": c1,
                "cluster_q": c2,
                "report": report,
            })

    hierarchy["pair_reports"] = pair_reports
    return hierarchy
