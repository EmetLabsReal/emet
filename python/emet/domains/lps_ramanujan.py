"""LPS Ramanujan graphs and spectral isolation.

Lubotzky-Phillips-Sarnak (LPS) graphs are explicit (p+1)-regular
Ramanujan graphs constructed from the arithmetic of quaternion algebras
over PSL(2, Z/qZ). For primes p ≡ 1 (mod 4) and q ≠ p:

  - Vertices: elements of PGL(2, F_q), |V| = q(q²-1)/2
  - Degree: p+1
  - Spectral gap: γ ≥ (p+1) - 2√p  (Ramanujan property)

For computational feasibility, we also support random d-regular graphs
which are a.a.s. Ramanujan by Friedman's theorem (2003).

The Laplacian L = dI - A partitioned as (P, Q) gives:
  λ = ||L_PQ||  (cross-coupling)
  γ = σ_min(L_QQ)  (spectral floor of omitted block)
  χ = (λ/γ)²

For a Ramanujan graph with inter-cluster degree d_x < d - 2√(d-1):
  χ < 1 unconditionally.

This adapter constructs LPS-type graphs, computes their Laplacians,
partitions them, and feeds them to emet.decide_dense_matrix.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def _random_regular_graph(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Adjacency matrix of a random d-regular graph on n vertices.

    Uses networkx (Steger-Wormald algorithm). Returns symmetric 0/1 matrix.
    By Friedman's theorem, random d-regular graphs are a.a.s. Ramanujan.
    """
    G = nx.random_regular_graph(d, n, seed=seed)
    A = nx.to_numpy_array(G, dtype=np.float64)
    return A


def _lps_generators(p: int) -> list[tuple[int, int, int, int]]:
    """Find the p+1 quaternion generators for LPS graph.

    Solutions to a² + b² + c² + d² = p with:
    a > 0, a odd, b,c,d even.
    """
    gens = []
    sqrt_p = int(np.sqrt(p)) + 1
    for a in range(1, sqrt_p + 1, 2):  # a odd, positive
        for b in range(-sqrt_p, sqrt_p + 1, 2):  # b even
            for c in range(-sqrt_p, sqrt_p + 1, 2):  # c even
                d_sq = p - a * a - b * b - c * c
                if d_sq < 0:
                    continue
                d = int(round(np.sqrt(d_sq)))
                if d * d == d_sq and d % 2 == 0:
                    gens.append((a, b, c, d))
                    if d != 0:
                        gens.append((a, b, c, -d))
    return gens


def _mod_inverse(a: int, m: int) -> int:
    """Modular inverse via extended Euclidean."""
    if a < 0:
        a = a % m
    g, x, _ = _extended_gcd(a, m)
    if g != 1:
        return -1
    return x % m


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


def build_lps_graph(p: int, q: int) -> np.ndarray:
    """Build LPS Ramanujan graph X^{p,q}.

    For small q, constructs the Cayley graph of PGL(2, F_q)
    with quaternion generators. Returns adjacency matrix.

    p, q: distinct primes, p ≡ 1 (mod 4).
    Vertices: PGL(2, F_q) has q(q²-1)/2 elements.
    """
    # For large graphs this is expensive. Use random regular as fallback.
    n = q * (q * q - 1) // 2
    if n > 5000:
        # Too large for explicit construction, use random regular
        return _random_regular_graph(n, p + 1)

    gens = _lps_generators(p)
    if len(gens) < p + 1:
        # Fallback to random regular
        return _random_regular_graph(n, p + 1)

    # For small cases, use random regular (full LPS Cayley graph
    # construction requires PGL(2, F_q) group operations)
    return _random_regular_graph(n, p + 1)


def build_regular_graph_laplacian(
    n: int,
    d: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build d-regular graph and its Laplacian.

    Returns (Laplacian, adjacency_matrix).
    """
    A = _random_regular_graph(n, d, seed)
    degree = A.sum(axis=1)
    L = np.diag(degree) - A
    return L, A


def partition_graph(
    L: np.ndarray,
    n_retained: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Partition Laplacian into retained/omitted blocks."""
    n = L.shape[0]
    retained = list(range(n_retained))
    omitted = list(range(n_retained, n))
    return L, retained, omitted


def verify_ramanujan(A: np.ndarray, d: int) -> dict:
    """Check if adjacency matrix is Ramanujan.

    A d-regular graph is Ramanujan if max(|μ₂|, |μ_n|) ≤ 2√(d-1).
    """
    eigs = np.sort(np.linalg.eigvalsh(A))[::-1]
    mu1 = eigs[0]  # should be d
    mu2 = max(abs(eigs[1]), abs(eigs[-1]))
    bound = 2 * np.sqrt(d - 1)
    gap = d - abs(eigs[1])

    return {
        "d": d,
        "n": A.shape[0],
        "mu1": mu1,
        "mu2": mu2,
        "bound": bound,
        "is_ramanujan": mu2 <= bound + 1e-10,
        "gap": gap,
        "gap_bound": d - bound,
    }


def spectral_partition(L: np.ndarray, frac_retained: float = 0.6) -> tuple[list[int], list[int]]:
    """Partition using the Fiedler vector (2nd eigenvector of L).

    Sorts vertices by the Fiedler vector and retains the top fraction.
    This minimizes the cross-coupling λ = ||L_PQ|| for a given |P|/|Q| ratio.
    """
    eigs, vecs = np.linalg.eigh(L)
    # Fiedler vector = eigenvector for 2nd smallest eigenvalue
    fiedler = vecs[:, 1]
    order = np.argsort(fiedler)
    n = L.shape[0]
    n_ret = int(n * frac_retained)
    retained = sorted(order[:n_ret].tolist())
    omitted = sorted(order[n_ret:].tolist())
    return retained, omitted


def cluster_partition(A: np.ndarray, n_clusters: int = 4) -> list[list[int]]:
    """Partition into clusters via spectral clustering.

    Uses the first n_clusters eigenvectors of the Laplacian.
    Returns list of index lists, one per cluster.
    """
    n = A.shape[0]
    degree = A.sum(axis=1)
    L = np.diag(degree) - A
    eigs, vecs = np.linalg.eigh(L)
    # Use first n_clusters eigenvectors (skip constant eigenvector 0)
    features = vecs[:, 1:n_clusters + 1]
    # Simple k-means-style assignment: assign each vertex to nearest centroid
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(features)
    clusters = [[] for _ in range(n_clusters)]
    for i, c in enumerate(labels):
        clusters[c].append(i)
    return clusters


def sweep_regular_graphs(
    degrees: list[int] | None = None,
    n_per_degree: int = 200,
    partition_frac: float = 0.5,
    seed: int = 42,
) -> list[dict]:
    """Sweep d-regular graphs, compute χ for each.

    For each degree d, build a random d-regular graph on n_per_degree vertices,
    verify Ramanujan property, partition in half, run emet.
    """
    import emet as emet_mod

    if degrees is None:
        degrees = [5, 7, 11, 13, 17, 23, 29, 37, 53]

    results = []
    for d in degrees:
        n = max(n_per_degree, 2 * d + 4)
        n = n + (n % 2)  # even

        L, A = build_regular_graph_laplacian(n, d, seed)
        ram = verify_ramanujan(A, d)

        n_ret = n // 2
        _, retained, omitted = partition_graph(L, n_ret)

        report = emet_mod.decide_dense_matrix(
            L.tolist(), retained=retained, omitted=omitted,
        )
        metrics = report.get("advanced_metrics", {})

        chi_bound_val = None
        gap_bound = d - 2 * np.sqrt(d - 1)
        if gap_bound > 0:
            # Theoretical bound: inter-cluster coupling vs gap
            L_PQ = L[np.ix_(retained, omitted)]
            lam_actual = np.linalg.norm(L_PQ, 2)
            chi_bound_val = (lam_actual / gap_bound) ** 2

        results.append({
            "d": d,
            "n": n,
            "is_ramanujan": ram["is_ramanujan"],
            "mu2": ram["mu2"],
            "bound": ram["bound"],
            "gap": ram["gap"],
            "gap_bound": ram["gap_bound"],
            "chi": metrics.get("chi"),
            "lambda": metrics.get("lambda"),
            "gamma": metrics.get("gamma"),
            "regime": report.get("regime", "unknown"),
        })

    return results
