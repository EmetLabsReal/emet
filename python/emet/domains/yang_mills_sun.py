"""SU(N) Kogut-Susskind Hamiltonian for compact simple gauge groups.

Generalizes from SU(2) to arbitrary SU(N).

The Casimir operator for SU(N) in the representation labeled by
highest weight (p, q, ...) determines the barrier eigenvalues.
The plaquette coupling 1/g^2 determines the tunneling amplitude.
The structure is identical to SU(2) — only the Casimir changes.

SU(2): irreps labeled by j = 0, 1/2, 1, 3/2, ...
       C_2(j) = j(j+1)

SU(3): irreps labeled by (p, q) with p, q = 0, 1, 2, ...
       C_2(p,q) = (p^2 + q^2 + pq + 3p + 3q) / 3

SU(N): irreps labeled by highest weight lambda = (l_1, ..., l_{N-1})
       C_2(lambda) = sum_i l_i(l_i + N + 1 - 2i) / N  (Freudenthal formula)
"""

from __future__ import annotations

import numpy as np

import emet
from emet.domains.torus import build_block_operator
from emet.domains.kahan import certified_subcritical


def casimir_su2(j: float) -> float:
    """Quadratic Casimir for SU(2): C_2(j) = j(j+1)."""
    return j * (j + 1)


def casimir_su3(p: int, q: int) -> float:
    """Quadratic Casimir for SU(3): C_2(p,q) = (p^2 + q^2 + pq + 3p + 3q) / 3."""
    return (p ** 2 + q ** 2 + p * q + 3 * p + 3 * q) / 3.0


def su3_irreps_up_to(max_pq: int) -> list[tuple[int, int]]:
    """SU(3) irreps (p,q) with p+q <= max_pq, ordered by Casimir."""
    irreps = []
    for p in range(max_pq + 1):
        for q in range(max_pq + 1 - p):
            irreps.append((p, q))
    irreps.sort(key=lambda pq: casimir_su3(pq[0], pq[1]))
    return irreps


def su2_irreps_up_to(j_max: float) -> list[float]:
    """SU(2) irreps j = 0, 1/2, 1, ..., j_max."""
    j_values = []
    j = 0.0
    while j <= j_max + 1e-10:
        j_values.append(j)
        j += 0.5
    return j_values


def build_sun_plaquette(
    n: int,
    g_squared: float,
    max_irrep: int = 4,
    cut_index: int = 3,
) -> tuple[np.ndarray, list[int], list[int], dict]:
    """Build the SU(N) Kogut-Susskind Hamiltonian on a single plaquette.

    The diagonal entries are (g^2/2) * C_2(irrep).
    The off-diagonal coupling is -1/g^2 between adjacent irreps.

    For SU(2): irreps are j-values, adjacency is j <-> j +/- 1/2.
    For SU(3): irreps are (p,q) pairs, ordered by Casimir.
               Adjacency connects nearest Casimir neighbors.

    Parameters:
        n: gauge group SU(n)
        g_squared: coupling constant
        max_irrep: maximum irrep label (j_max for SU(2), max p+q for SU(3))
        cut_index: partition boundary (first cut_index irreps retained)

    Returns (H, retained, omitted, info).
    """
    if n == 2:
        return _build_su2(g_squared, max_irrep, cut_index)
    elif n == 3:
        return _build_su3(g_squared, max_irrep, cut_index)
    else:
        raise NotImplementedError(f"SU({n}) not yet implemented; SU(2) and SU(3) available")


def _build_su2(
    g_squared: float, j_max_int: int, cut_index: int,
) -> tuple[np.ndarray, list[int], list[int], dict]:
    """SU(2) plaquette. Matches existing yang_mills.py via block assembly."""
    j_max = j_max_int / 2.0 * 2  # ensure half-integer compatible
    j_values = su2_irreps_up_to(j_max)
    dim = len(j_values)
    n_retained = min(cut_index, dim)

    casimirs = np.array([casimir_su2(j) for j in j_values])

    # Build full tridiagonal Hamiltonian
    H_full = np.diag((g_squared / 2.0) * casimirs)
    for k in range(dim - 1):
        H_full[k, k + 1] = -1.0 / g_squared
        H_full[k + 1, k] = -1.0 / g_squared

    retained = list(range(n_retained))
    omitted = list(range(n_retained, dim))

    # Extract blocks
    PP = H_full[np.ix_(retained, retained)]
    QQ = H_full[np.ix_(omitted, omitted)]
    PQ = H_full[np.ix_(retained, omitted)]

    H, ret, omit = build_block_operator(PP, QQ, PQ)

    return H, ret, omit, {
        "group": "SU(2)",
        "irreps": j_values,
        "casimirs": casimirs.tolist(),
        "n_retained": n_retained,
        "g_squared": g_squared,
    }


def _build_su3(
    g_squared: float, max_pq: int, cut_index: int,
) -> tuple[np.ndarray, list[int], list[int], dict]:
    """SU(3) plaquette. Irreps ordered by Casimir, nearest-neighbor coupling."""
    irreps = su3_irreps_up_to(max_pq)
    dim = len(irreps)
    n_retained = min(cut_index, dim)

    casimirs = np.array([casimir_su3(p, q) for p, q in irreps])

    # Diagonal: electric energy from Casimir
    H_full = np.diag((g_squared / 2.0) * casimirs)

    # Off-diagonal: plaquette coupling between adjacent irreps
    # Adjacent = nearest in Casimir ordering (lattice approximation)
    for k in range(dim - 1):
        H_full[k, k + 1] = -1.0 / g_squared
        H_full[k + 1, k] = -1.0 / g_squared

    retained = list(range(n_retained))
    omitted = list(range(n_retained, dim))

    PP = H_full[np.ix_(retained, retained)]
    QQ = H_full[np.ix_(omitted, omitted)]
    PQ = H_full[np.ix_(retained, omitted)]

    H, ret, omit = build_block_operator(PP, QQ, PQ)

    return H, ret, omit, {
        "group": "SU(3)",
        "irreps": irreps,
        "casimirs": casimirs.tolist(),
        "n_retained": n_retained,
        "g_squared": g_squared,
    }


def sweep_coupling_sun(
    n: int,
    g_squared_values: list[float],
    max_irrep: int = 4,
    cut_index: int = 3,
) -> list[dict]:
    """Sweep coupling for SU(N), certifying confinement at each point."""
    results = []
    for g2 in g_squared_values:
        H, ret, omit, info = build_sun_plaquette(n, g2, max_irrep, cut_index)
        report = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
        m = report["advanced_metrics"]
        chi = m["chi"]
        licensed = chi is not None and chi < 1

        eigvals = np.linalg.eigvalsh(H)
        if eigvals.shape[0] < 2:
            gap = 0.0
        else:
            gap = float(eigvals[1] - eigvals[0])

        result = {
            "group": info["group"],
            "g_squared": g2,
            "chi": chi,
            "gamma": m["gamma"],
            "lambda": m["lambda"],
            "licensed": licensed,
            "regime": report["regime"],
            "mass_gap": gap,
        }

        if licensed and chi is not None:
            kahan = certified_subcritical(chi, m["gamma"], m["lambda"])
            result["kahan_certified"] = kahan["certified"]
        else:
            result["kahan_certified"] = False

        results.append(result)
    return results
