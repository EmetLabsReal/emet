"""ADM constraint operator on closed FLRW (S³) with dust.

York-Lichnerowicz partition:
  P = TT tensor modes at each ℓ ≥ 2 (physical gravitons, retained)
  Q = scalar + vector constraint modes (omitted)

On a maximally symmetric background (S³), TT modes decouple from
both the Hamiltonian and momentum constraints: H_PQ(ℓ) = 0 for ℓ ≥ 2.
The full harmonic tower reduces to the homogeneous ℓ = 0 block.

The ℓ = 0 block has coercivity γ = (3a − 4)/(4a³), which vanishes
at a_crit = 4/3.  At this point χ → ∞ and Cap > 0 (β < 1):
the constraint boundary is accessible, with no intrinsic Feller mechanism.
"""

from __future__ import annotations

import numpy as np


# S³ spectrum
def scalar_eigenvalue(ell: int, a: float) -> float:
    """Scalar harmonic eigenvalue on S³ of radius a: ℓ(ℓ+2)/a²."""
    return ell * (ell + 2) / a**2


def scalar_degeneracy(ell: int) -> int:
    """Degeneracy of ℓ-th scalar harmonic on S³: (ℓ+1)²."""
    return (ell + 1) ** 2


def tt_eigenvalue(ell: int, a: float) -> float:
    """Lichnerowicz eigenvalue for TT tensors on S³: [ℓ(ℓ+2)−2]/a².

    Defined for ℓ ≥ 2.
    """
    if ell < 2:
        raise ValueError(f"TT tensors require ℓ ≥ 2, got {ell}")
    return (ell * (ell + 2) - 2) / a**2


# Homogeneous block
def gamma_homogeneous(a: float) -> float:
    """Coercivity of the ℓ = 0 constraint block: γ = (3a − 4)/(4a³)."""
    return (3.0 * a - 4.0) / (4.0 * a**3)


A_CRIT = 4.0 / 3.0


def chi_homogeneous(a: float, lambda_coupling: float = 1.0) -> float:
    """Regime parameter χ = (λ/γ)² for the homogeneous block.

    Returns inf when γ = 0 (at a_crit = 4/3).
    """
    g = gamma_homogeneous(a)
    if g == 0.0:
        return float("inf")
    return (lambda_coupling / g) ** 2


def build_adm_flrw_tower(
    a: float,
    ell_max: int = 10,
    lambda_coupling: float = 1.0,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Block-diagonal operator for the full ADM harmonic tower on S³.

    Returns (H, retained, omitted) where H is a block-diagonal matrix
    with the ℓ = 0 homogeneous block (2×2) and decoupled TT blocks
    for ℓ = 2, ..., ell_max.

    The coupling H_PQ(ℓ) = 0 for all ℓ ≥ 2 (TT decoupling).
    """
    blocks = []
    retained = []
    omitted = []
    idx = 0

    # ℓ = 0 homogeneous block (2×2)
    g = gamma_homogeneous(a)
    block_0 = np.array([
        [lambda_coupling**2, lambda_coupling],
        [lambda_coupling, g],
    ])
    blocks.append(block_0)
    retained.append(idx)      # PP component
    omitted.append(idx + 1)   # QQ component
    idx += 2

    # ℓ ≥ 2: TT modes decouple — diagonal blocks, zero coupling
    for ell in range(2, ell_max + 1):
        e_tt = tt_eigenvalue(ell, a)
        e_sc = scalar_eigenvalue(ell, a)
        block_ell = np.array([
            [e_tt, 0.0],
            [0.0, e_sc],
        ])
        blocks.append(block_ell)
        retained.append(idx)      # TT (retained)
        omitted.append(idx + 1)   # constraint (omitted)
        idx += 2

    H = _block_diag(blocks)
    return H, retained, omitted


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    """Assemble block-diagonal matrix from a list of square blocks."""
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n))
    i = 0
    for b in blocks:
        s = b.shape[0]
        out[i : i + s, i : i + s] = b
        i += s
    return out


def sweep_chi(
    a_values: np.ndarray | None = None,
    lambda_coupling: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep scale factor and return (a_values, chi_values).

    Demonstrates χ → ∞ at a_crit = 4/3.
    """
    if a_values is None:
        a_values = np.linspace(0.5, 3.0, 500)
    chi_values = np.array([chi_homogeneous(a, lambda_coupling) for a in a_values])
    return a_values, chi_values
