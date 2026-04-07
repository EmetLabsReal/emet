"""Pinched torus operator with centrifugal barrier and Feller threshold.

The transverse measure s^beta ds on a torus with cross-section pinching
r(s) ~ s^alpha, where beta = d * alpha, produces the effective potential
V_eff(s) = beta(beta-2)/(4s^2).

Block structure:
  QQ (barrier)   eigenvalues from centrifugal potential at the pinch locus
  PP (valley)    eigenvalues from harmonic confinement at the valley radius
  PQ/QP          WKB tunneling amplitude through the barrier

Below the Feller threshold (beta < 1): boundary accessible, barrier
transparent, chi >= 1. Past the threshold: capacity vanishes, measure
degeneration seals the boundary, tunneling decays, chi < 1.

chi < 1 certifies that the Friedrichs variational attractor has
selected the valley sector and the reduction is licensed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import emet


def effective_potential(beta: float, s: np.ndarray) -> np.ndarray:
    """V_eff(s) = β(β - 2) / (4s²).

    The centrifugal/pinch part of the Mexican hat potential. For β > 2
    it is repulsive at s = 0 — the boundary amplifies deviations. At
    β = 1 (Feller threshold), V_eff = -1/(4s²): the Hardy constant.
    """
    s = np.asarray(s, dtype=float)
    return beta * (beta - 2.0) / (4.0 * s ** 2)


def barrier_eigenvalues(beta: float, n_barrier: int) -> np.ndarray:
    """Eigenvalues of the barrier (omitted) block.

    Below the Feller threshold (β < 1): the barrier is transparent.
    The boundary is accessible, the omitted block has weak control.
    Eigenvalues are small.

    Past the Feller threshold (β ≥ 1): measure degeneration s^β ds
    makes the boundary unreachable. The barrier strengthens. Eigenvalues
    grow as the capacity of the pinch locus vanishes.

    The scaling: barrier strength ~ (β - 1)_+ * β / s_k² where s_k are
    representative barrier positions. The (β-1)_+ factor ensures the
    barrier only activates past the Feller threshold.
    """
    # Representative positions in the barrier region
    s_k = np.linspace(0.1, 0.3, n_barrier)

    # Below Feller: small floor. Past Feller: grows with β.
    feller_factor = max(0.0, beta - 1.0)

    # Barrier strength: combines measure degeneration and centrifugal repulsion
    # Floor keeps QQ positive definite even below Feller
    floor = 0.1
    strength = feller_factor * beta / s_k ** 2

    return floor + strength


def valley_eigenvalues(n_valley: int, valley_curvature: float = 2.0) -> np.ndarray:
    """Eigenvalues of the valley (retained) block from harmonic confinement.

    The valley modes live near s = s₀ (valley radius). Their energy is
    set by the confining potential curvature: E_k = base + curvature * k.
    Independent of β — the valley geometry is fixed.
    """
    return np.array([1.0 + valley_curvature * k for k in range(n_valley)])


def tunneling_coupling(beta: float, n_valley: int, n_barrier: int,
                       coupling_base: float = 1.5) -> np.ndarray:
    """Tunneling amplitude through the centrifugal barrier.

    WKB: tunneling probability decays exponentially with barrier height.
    Below the Feller threshold the barrier is transparent and coupling
    is strong. Past the threshold, tunneling is suppressed as the
    measure degeneration seals the boundary.

    Returns the (n_valley, n_barrier) coupling matrix PQ.
    """
    # Suppression: exponential decay past Feller threshold
    # For β < 1, coupling is essentially at full strength
    # For β > 1, coupling decays as exp(-c·(β-1))
    feller_excess = max(0.0, beta - 1.0)
    suppression = np.exp(-1.5 * feller_excess)

    # Mode-dependent profile: higher modes couple more weakly
    valley_profile = np.exp(-0.3 * np.arange(n_valley))
    barrier_profile = np.exp(-0.3 * np.arange(n_barrier))

    return coupling_base * suppression * np.outer(valley_profile, barrier_profile)


def build_block_operator(
    pp: np.ndarray,
    qq: np.ndarray,
    pq: np.ndarray,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Assemble a block matrix from PP (retained), QQ (omitted), PQ (coupling).

    H = [[PP, PQ], [QP, QQ]]

    Returns (H, retained, omitted) ready for emet.decide_dense_matrix().
    This is the shared block assembly used by all domain adapters that
    partition an operator into retained and omitted sectors.
    """
    n_p, n_q = pp.shape[0], qq.shape[0]
    n = n_p + n_q
    H = np.zeros((n, n))
    H[:n_p, :n_p] = pp
    H[:n_p, n_p:] = pq
    H[n_p:, :n_p] = pq.T
    H[n_p:, n_p:] = qq

    H = 0.5 * (H + H.T)

    retained = list(range(n_p))
    omitted = list(range(n_p, n))
    return H, retained, omitted


def build_torus_operator(
    beta: float,
    n_valley: int = 6,
    n_barrier: int = 4,
    *,
    valley_curvature: float = 2.0,
    coupling_base: float = 1.5,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Assemble the pinched torus operator as a block matrix.

    H = [[PP, PQ], [QP, QQ]]

    PP = valley block (diagonal, harmonic confinement eigenvalues).
    QQ = barrier block (diagonal, centrifugal barrier eigenvalues).
    PQ/QP = tunneling coupling (WKB amplitude through barrier).

    Returns (H, retained, omitted) ready for emet.decide_dense_matrix().
    """
    PP = np.diag(valley_eigenvalues(n_valley, valley_curvature))
    QQ = np.diag(barrier_eigenvalues(beta, n_barrier))
    PQ = tunneling_coupling(beta, n_valley, n_barrier, coupling_base)

    return build_block_operator(PP, QQ, PQ)


def ground_state(H: np.ndarray) -> tuple[float, np.ndarray]:
    """Lowest eigenvalue (mass gap) and eigenvector of H."""
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argmin(eigvals)
    return float(eigvals[idx]), eigvecs[:, idx]


def ground_state_variance(phi_0: np.ndarray, n_valley: int) -> float:
    """Variance of ground state projected onto valley modes.

    Measures how tightly the ground state is confined to the valley.
    As β increases past Feller, the barrier strengthens, the valley
    component dominates, and variance decreases.
    """
    valley_part = phi_0[:n_valley]
    prob = valley_part ** 2
    prob_sum = prob.sum()
    if prob_sum < 1e-15:
        return float("inf")
    prob = prob / prob_sum
    modes = np.arange(n_valley, dtype=float)
    mean = np.dot(prob, modes)
    return float(np.dot(prob, (modes - mean) ** 2))


def sweep_beta(
    betas: Sequence[float],
    n_valley: int = 6,
    n_barrier: int = 4,
    **kwargs,
) -> list[dict]:
    """Full β-sweep: build operator, feed to chi engine, collect metrics."""
    results = []
    for beta in betas:
        H, retained, omitted = build_torus_operator(
            beta, n_valley, n_barrier, **kwargs,
        )
        report = emet.decide_dense_matrix(H, retained=retained, omitted=omitted)
        m = report["advanced_metrics"]

        lam0, phi_0 = ground_state(H)
        var = ground_state_variance(phi_0, n_valley)
        chi = m["chi"] if m["chi"] is not None else float("inf")

        results.append({
            "beta": beta,
            "lambda": m["lambda"],
            "gamma": m["gamma"],
            "chi": chi,
            "lambda_0": lam0,
            "variance": var,
            "valid": report["valid"],
            "regime": report["regime"],
        })

    return results
