"""Quantum channel certification via Choi matrix partition.

Choi matrix Phi of a CPTP map E on a single qubit, partitioned into
signal indices {0,3} (|00>, |11>) and error indices {1,2} (|01>, |10>).

chi = (lambda/gamma)^2 where lambda = ||Phi_PQ||_2, gamma = sigma_min(Phi_QQ).
chi < 1: signal sector faithfully reduced. chi >= 1: partition fails.

Pauli channels (depolarizing, dephasing, bit-flip) have lambda = 0.
Basis-misaligned channels (Hadamard mixing) have lambda > 0.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SIGNAL_INDICES = [0, 3]
ERROR_INDICES = [1, 2]


def choi_from_kraus(kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Build unnormalized Choi matrix from Kraus operators.

    Phi = sum_k (I otimes K_k) |Omega><Omega| (I otimes K_k)^dag
    where |Omega> = |00> + |11>.
    """
    d = kraus_ops[0].shape[0]
    omega = np.zeros(d * d)
    for i in range(d):
        omega[i * d + i] = 1.0
    omega_dm = np.outer(omega, omega)
    choi = np.zeros((d * d, d * d), dtype=complex)
    eye = np.eye(d)
    for k in kraus_ops:
        ik = np.kron(eye, k)
        choi += ik @ omega_dm @ ik.conj().T
    choi = choi.real
    choi = 0.5 * (choi + choi.T)
    return choi


def choi_identity() -> np.ndarray:
    """Choi matrix of the identity channel."""
    return choi_from_kraus([np.eye(2)])


def choi_depolarizing(p: float) -> np.ndarray:
    """Choi matrix of the depolarizing channel with error rate p in [0, 1].

    E(rho) = (1 - p) rho + (p/3)(X rho X + Y rho Y + Z rho Z).
    """
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Y = np.array([[0, -1], [1, 0]], dtype=float)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    I = np.eye(2)
    return choi_from_kraus([
        math.sqrt(1 - p) * I,
        math.sqrt(p / 3) * X,
        math.sqrt(p / 3) * Y,
        math.sqrt(p / 3) * Z,
    ])


def choi_dephasing(p: float) -> np.ndarray:
    """Choi matrix of the dephasing (phase-flip) channel.

    E(rho) = (1 - p) rho + p Z rho Z.
    """
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    I = np.eye(2)
    return choi_from_kraus([
        math.sqrt(1 - p) * I,
        math.sqrt(p) * Z,
    ])


def choi_bit_flip(p: float) -> np.ndarray:
    """Choi matrix of the bit-flip channel.

    E(rho) = (1 - p) rho + p X rho X.
    """
    X = np.array([[0, 1], [1, 0]], dtype=float)
    I = np.eye(2)
    return choi_from_kraus([
        math.sqrt(1 - p) * I,
        math.sqrt(p) * X,
    ])


def choi_amplitude_damping(gamma: float) -> np.ndarray:
    """Choi matrix of the amplitude damping channel.

    K0 = [[1, 0], [0, sqrt(1-gamma)]], K1 = [[0, sqrt(gamma)], [0, 0]].
    """
    K0 = np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=float)
    K1 = np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=float)
    return choi_from_kraus([K0, K1])


def choi_misaligned(p: float) -> np.ndarray:
    """Choi matrix of the Hadamard-misaligned channel.

    E(rho) = (1 - p) rho + p H rho H where H is the Hadamard gate.
    This creates coherences that break the signal/error partition.
    """
    H = np.array([[1, 1], [1, -1]], dtype=float) / math.sqrt(2)
    I = np.eye(2)
    return choi_from_kraus([
        math.sqrt(1 - p) * I,
        math.sqrt(p) * H,
    ])


def binary_entropy(p: float) -> float:
    """h(p) = -p log2(p) - (1-p) log2(1-p)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def shor_preskill_rate(qber: float) -> float:
    """Shor-Preskill key rate r = 1 - 2h(q). Clipped to [0, 1]."""
    r = 1.0 - 2.0 * binary_entropy(qber)
    return max(0.0, min(1.0, r))


def qber_from_choi(choi: np.ndarray) -> float:
    """Extract QBER from the Choi matrix.

    QBER = Phi[1,1] + Phi[2,2] for the error-sector diagonal entries,
    normalized by the trace.
    """
    tr = np.trace(choi)
    if tr <= 0:
        return 0.0
    return float((choi[1, 1] + choi[2, 2]) / tr)


def certify_channel(choi: np.ndarray) -> dict[str, Any]:
    """Certify a quantum channel via its Choi matrix.

    Returns emet decision report augmented with qber and key_rate.
    """
    import emet

    report = emet.decide_dense_matrix(
        choi, retained=SIGNAL_INDICES, omitted=ERROR_INDICES,
    )
    m = report["advanced_metrics"]
    qber = qber_from_choi(choi)
    key_rate = shor_preskill_rate(qber)
    chi = m["chi"]
    epsilon = m.get("epsilon")

    return {
        "valid": report["valid"],
        "regime": report["regime"],
        "lambda": m["lambda"],
        "gamma": m["gamma"],
        "chi": chi,
        "epsilon": epsilon,
        "qber": qber,
        "key_rate": key_rate,
        "reduced_matrix": report.get("reduced_matrix"),
        "report": report,
    }
