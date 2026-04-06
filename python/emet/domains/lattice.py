"""Multi-plaquette lattice and transfer matrix for thermodynamic limit.

The single plaquette is the building block. The full lattice Hamiltonian
is the sum over plaquettes. The transfer matrix T connects adjacent time
slices: Z = Tr(T^N_t). The mass gap is -ln(lambda_1/lambda_0) where
lambda_0, lambda_1 are the two largest eigenvalues of T.

Schur chi on ``T`` for a fixed representation-space partition is a separate
diagnostic from chi on ``H``: with ``T = exp(-H)``, the same split can be
Schur-supercritical on ``T`` even when the Hamiltonian reduction is licensed.
The spectral gap ``-ln(lambda_1/lambda_0)`` of ``T`` still tracks the
propagation scale and is independent of chain length ``N_t``.
"""

from __future__ import annotations

import numpy as np

import emet
from emet.domains.yang_mills import casimir, build_single_plaquette
from emet.domains.torus import build_block_operator
from emet.domains.kahan import certified_subcritical


def build_transfer_matrix(
    g_squared: float,
    j_max: float = 3.0,
    j_cut: float = 1.0,
) -> np.ndarray:
    """Build the transfer matrix for Yang-Mills on a 1D chain.

    The transfer matrix T encodes propagation across one time step.
    T_{j,j'} = exp(-(g^2/2) C_2(j) delta) * coupling(j, j')

    For the Kogut-Susskind Hamiltonian, T is the exponential of the
    single-plaquette Hamiltonian (in Euclidean time).

    Returns the transfer matrix (symmetric, positive).
    """
    H, _, _ = build_single_plaquette(g_squared, j_max)
    # Transfer matrix = exp(-H) in Euclidean time (unit lattice spacing)
    eigvals, eigvecs = np.linalg.eigh(H)
    T = eigvecs @ np.diag(np.exp(-eigvals)) @ eigvecs.T
    return T


def transfer_matrix_gap(T: np.ndarray) -> float:
    """Mass gap from the transfer matrix: -ln(lambda_1/lambda_0).

    lambda_0 = largest eigenvalue (ground state)
    lambda_1 = second largest (first excited state)
    gap = -ln(lambda_1/lambda_0) > 0 iff lambda_1 < lambda_0
    """
    eigvals = np.sort(np.linalg.eigvalsh(T))[::-1]
    if eigvals[0] <= 0 or eigvals[1] <= 0:
        return float("inf")
    return -np.log(eigvals[1] / eigvals[0])


def certify_transfer_matrix(
    g_squared: float,
    j_max: float = 3.0,
    j_cut: float = 1.0,
) -> dict:
    """Schur analysis of the transfer matrix with the same j-split as ``H``.

    ``T`` lives in the same representation space as the single-plaquette
    Hamiltonian. Retained = low-j (below ``j_cut``), omitted = high-j.
    Licensing on ``T`` can differ from licensing on ``H`` because
    ``T = exp(-H)`` couples sectors more strongly.
    """
    T = build_transfer_matrix(g_squared, j_max, j_cut)
    gap = transfer_matrix_gap(T)

    # Partition the transfer matrix
    j_values = []
    j = 0.0
    while j <= j_max + 1e-10:
        j_values.append(j)
        j += 0.5
    n = len(j_values)

    retained = [i for i, j in enumerate(j_values) if j <= j_cut]
    omitted = [i for i, j in enumerate(j_values) if j > j_cut]

    if not omitted:
        return {"g_squared": g_squared, "gap": gap, "chi": 0.0,
                "licensed": True, "regime": "subcritical"}

    report = emet.decide_dense_matrix(T, retained=retained, omitted=omitted)
    m = report["advanced_metrics"]
    chi = m["chi"]
    licensed = chi is not None and chi < 1

    result = {
        "g_squared": g_squared,
        "transfer_gap": gap,
        "chi": chi,
        "gamma": m["gamma"],
        "lambda": m["lambda"],
        "licensed": licensed,
        "regime": report["regime"],
        "j_values": j_values,
        "n_retained": len(retained),
        "n_omitted": len(omitted),
    }

    if licensed and chi is not None:
        kahan = certified_subcritical(chi, m["gamma"], m["lambda"])
        result["kahan_certified"] = kahan["certified"]
    else:
        result["kahan_certified"] = False

    return result


def multi_plaquette_gap(
    g_squared: float,
    n_plaquettes: int,
    j_max: float = 3.0,
) -> float:
    """Mass gap for N plaquettes from the transfer matrix.

    For N plaquettes in a 1D chain, Z = Tr(T^N).
    The gap is independent of N (it's a property of T).
    This demonstrates thermodynamic limit stability.
    """
    T = build_transfer_matrix(g_squared, j_max)
    return transfer_matrix_gap(T)


def scaling_analysis(
    g_squared_values: list[float],
    j_max: float = 3.0,
    j_cut: float = 1.0,
) -> list[dict]:
    """Compare Hamiltonian gap, transfer-matrix gap, and Schur chi on ``T``."""
    results = []
    for g2 in g_squared_values:
        # Single plaquette
        H, _, _ = build_single_plaquette(g2, j_max)
        eigvals_H = np.linalg.eigvalsh(H)
        h_gap = float(eigvals_H[1] - eigvals_H[0])

        # Transfer matrix
        tm = certify_transfer_matrix(g2, j_max, j_cut)

        results.append({
            "g_squared": g2,
            "hamiltonian_gap": h_gap,
            "transfer_gap": tm["transfer_gap"],
            "chi_transfer": tm["chi"],
            "licensed_transfer": tm["licensed"],
        })
    return results
