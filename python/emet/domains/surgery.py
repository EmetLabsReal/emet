"""Post-fracture surgical reconstruction via minimal Markov generator.

When a domain is severed by a capacity-zero boundary, local diffusion
operators become isolated: H_PQ = 0, the sectors cannot communicate,
and the partition is trivially supercritical or pre-admissible.

Surgery restores coherence by adding a non-local jump measure across
the fracture boundary. The Beurling-Deny decomposition splits:

    E(f,g) = E^(c)(f,g) + integral (f(x)-f(y))^2 J(dx,dy)

After severance, E^(c) decouples. Surgery adds J.

For a double-well system, the unique minimal symmetric Markov generator
is:

    M = [[-lam,  lam],
         [ lam, -lam]]

This is the Mexican Hat surgery. Any symmetric Markov generator on two
states with full off-diagonal support has this form (uniqueness up to
rate). The energy gap is 2*lam. The killing measure is zero.

At the Feller threshold, the torus geometry produces a Mexican hat
potential: V_eff = beta(beta-2)/(4s^2). The boundary conditions
crystallize into a double-well landscape. Surgery is the canonical
mechanism that reconnects the wells via the discrete jump operator.
"""

from __future__ import annotations

import numpy as np

import emet


def minimal_symmetric_generator(n_wells: int, rate: float = 1.0) -> np.ndarray:
    """Unique minimal symmetric Markov generator on n_wells states.

    For n_wells = 2: the two-well tunneling matrix with uniform rate.
    Row sums vanish (Markov property). Off-diagonal entries are nonneg
    (jump rates). Symmetric (detailed balance with uniform measure).
    """
    M = np.full((n_wells, n_wells), rate)
    np.fill_diagonal(M, -rate * (n_wells - 1))
    return M


def generator_is_markov(M: np.ndarray, atol: float = 1e-12) -> bool:
    """Check that M is a valid Markov generator: row sums vanish, off-diag nonneg."""
    n = M.shape[0]
    for i in range(n):
        if abs(M[i].sum()) > atol:
            return False
        for j in range(n):
            if i != j and M[i, j] < -atol:
                return False
    return True


def generator_energy_gap(M: np.ndarray) -> float:
    """Spectral gap of Markov generator M.

    Eigenvalues of a Markov generator are all <= 0 with exactly one at 0.
    The gap is |second eigenvalue|.
    """
    eigvals = np.sort(np.linalg.eigvalsh(M))
    return float(abs(eigvals[-2])) if len(eigvals) > 1 else 0.0


def build_severed_double_well(
    barrier_height: float = 4.0,
    valley_energy: float = 1.0,
    n_left: int = 3,
    n_right: int = 3,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Build a double-well operator with severed cross-coupling.

    Two harmonic wells (left P, right Q) separated by a capacity-zero
    barrier. H_PQ = 0: no local coupling through the barrier.

    This is the Mexican hat potential on a lattice, post-fracture.
    The barrier has killed the tunneling amplitude.
    """
    # Left well: harmonic around left minimum
    PP = np.zeros((n_left, n_left))
    for k in range(n_left):
        PP[k, k] = valley_energy + 0.5 * k
    for k in range(n_left - 1):
        PP[k, k + 1] = -0.1
        PP[k + 1, k] = -0.1

    # Right well: harmonic around right minimum (shifted by barrier)
    QQ = np.zeros((n_right, n_right))
    for k in range(n_right):
        QQ[k, k] = valley_energy + barrier_height + 0.5 * k
    for k in range(n_right - 1):
        QQ[k, k + 1] = -0.1
        QQ[k + 1, k] = -0.1

    # Severed: no cross-coupling
    PQ = np.zeros((n_left, n_right))

    n = n_left + n_right
    H = np.zeros((n, n))
    H[:n_left, :n_left] = PP
    H[:n_left, n_left:] = PQ
    H[n_left:, :n_left] = PQ.T
    H[n_left:, n_left:] = QQ

    retained = list(range(n_left))
    omitted = list(range(n_left, n))
    return H, retained, omitted


def surgical_reconstruction(
    H: np.ndarray,
    retained: list[int],
    omitted: list[int],
    jump_rate: float,
) -> np.ndarray:
    """Surgically reconstruct a severed operator by adding jump coupling.

    Adds cross-sector jump rates to the PQ/QP blocks. The jump connects
    the boundary states of each well (last retained <-> first omitted).
    Also adjusts diagonal entries so row sums are preserved (Markov).

    This is the Beurling-Deny jump measure added to the severed local form.
    """
    H_new = H.copy()
    p_boundary = max(retained)
    q_boundary = min(omitted)

    # Add jump: cross-coupling between boundary states
    H_new[p_boundary, q_boundary] += jump_rate
    H_new[q_boundary, p_boundary] += jump_rate

    # Adjust diagonal to preserve operator structure
    H_new[p_boundary, p_boundary] -= jump_rate
    H_new[q_boundary, q_boundary] -= jump_rate

    return H_new


def post_fracture_certify(
    H: np.ndarray,
    retained: list[int],
    omitted: list[int],
    jump_rate: float,
) -> dict:
    """Full post-fracture surgery pipeline.

    1. Certify original system (should be supercritical or pre-admissible)
    2. Apply surgical reconstruction (add jump coupling)
    3. Re-certify the reconstructed system
    4. Return both pre- and post-surgery metrics
    """
    pre_report = emet.decide_dense_matrix(H, retained=retained, omitted=omitted)
    pre_m = pre_report["advanced_metrics"]
    pre_chi = pre_m["chi"] if pre_m["chi"] is not None else float("inf")

    H_reconstructed = surgical_reconstruction(H, retained, omitted, jump_rate)

    post_report = emet.decide_dense_matrix(
        H_reconstructed, retained=retained, omitted=omitted,
    )
    post_m = post_report["advanced_metrics"]
    post_chi = post_m["chi"] if post_m["chi"] is not None else float("inf")

    return {
        "pre_chi": pre_chi,
        "pre_regime": pre_report["regime"],
        "post_chi": post_chi,
        "post_regime": post_report["regime"],
        "post_valid": post_report["valid"],
        "jump_rate": jump_rate,
        "gamma_pre": pre_m["gamma"],
        "gamma_post": post_m["gamma"],
        "lambda_pre": pre_m["lambda"],
        "lambda_post": post_m["lambda"],
    }
