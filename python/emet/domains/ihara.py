"""Ihara zeta function on regular graphs.

For a d-regular graph X, the Ihara zeta function satisfies:
  Z_X(u)^{-1} = (1-u²)^{r-1} det(I - Au + (d-1)u²I)

Non-trivial zeros correspond to eigenvalues μ_i of A via:
  1 - μ_i u + (d-1)u² = 0

For Ramanujan graphs (|μ_i| ≤ 2√(d-1)):
  |u| = 1/√(d-1) → Re(s) = 1/2 under u = (d-1)^{-s}
"""

from __future__ import annotations

import numpy as np


def ihara_zeros_from_eigenvalues(
    eigenvalues: np.ndarray,
    d: int,
) -> np.ndarray:
    """Solve 1 - μu + (d-1)u² = 0 for each eigenvalue μ.

    Returns array of complex u values (2 per eigenvalue).
    Quadratic formula: u = (μ ± √(μ² - 4(d-1))) / (2(d-1))
    """
    dm1 = d - 1
    discriminant = eigenvalues**2 - 4 * dm1
    sqrt_disc = np.sqrt(discriminant.astype(complex))
    u_plus = (eigenvalues + sqrt_disc) / (2 * dm1)
    u_minus = (eigenvalues - sqrt_disc) / (2 * dm1)
    return np.concatenate([u_plus, u_minus])


def ihara_zeros(A: np.ndarray, d: int) -> np.ndarray:
    """Compute non-trivial Ihara zeros from adjacency matrix.

    Skips the trivial eigenvalue μ = d.
    """
    eigs = np.linalg.eigvalsh(A)
    # Remove eigenvalue closest to d (trivial)
    idx_trivial = np.argmin(np.abs(eigs - d))
    nontrivial = np.delete(eigs, idx_trivial)
    return ihara_zeros_from_eigenvalues(nontrivial, d)


def u_to_s(u: np.ndarray, d: int) -> np.ndarray:
    """Convert u to s-plane: u = (d-1)^{-s} → s = -log(u)/log(d-1)."""
    return -np.log(u.astype(complex)) / np.log(d - 1)


def verify_ihara_rh(
    A: np.ndarray,
    d: int,
    tol: float = 1e-10,
) -> dict:
    """Check all non-trivial Ihara zeros have Re(s) = 1/2.

    Returns dict with zeros, s-values, max deviation, and pass/fail.
    """
    eigs = np.linalg.eigvalsh(A)
    idx_trivial = np.argmin(np.abs(eigs - d))
    nontrivial = np.delete(eigs, idx_trivial)

    zeros_u = ihara_zeros_from_eigenvalues(nontrivial, d)
    zeros_s = u_to_s(zeros_u, d)
    re_s = zeros_s.real
    deviations = np.abs(re_s - 0.5)
    max_dev = np.max(deviations)

    # Modulus check: |u| should equal 1/√(d-1) for Ramanujan
    mod_u = np.abs(zeros_u)
    expected_mod = 1.0 / np.sqrt(d - 1)
    mod_deviations = np.abs(mod_u - expected_mod)
    max_mod_dev = np.max(mod_deviations)

    return {
        "eigenvalues": nontrivial,
        "zeros_u": zeros_u,
        "zeros_s": zeros_s,
        "re_s": re_s,
        "max_deviation": max_dev,
        "max_mod_deviation": max_mod_dev,
        "passes_rh": max_dev < tol,
        "n_zeros": len(zeros_u),
        "expected_mod_u": expected_mod,
    }


def ihara_determinant(A: np.ndarray, u: complex, d: int) -> complex:
    """Compute det(I - Au + (d-1)u²I)."""
    n = A.shape[0]
    M = np.eye(n) - A * u + (d - 1) * u**2 * np.eye(n)
    return np.linalg.det(M)
