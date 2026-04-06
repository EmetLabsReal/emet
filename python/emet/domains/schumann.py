"""Schumann resonance cavity: thin spherical shell confinement.

The Earth-ionosphere cavity is a spherical shell [R, R+δ] × S².
The Laplacian decomposes into angular modes (eigenvalues n(n+1)/R²)
and radial modes (eigenvalues ~ (kπ/δ)²).  When δ/R ≪ 1, the radial
modes are high-frequency and decouple from the angular modes.

Block structure:
  PP (angular)   fundamental radial mode (k=1) for each angular mode n
  QQ (radial)    overtone radial modes (k≥2) for each angular mode n
  PQ/QP          coupling from 1/r² variation across the shell thickness

The regime parameter χ → 0 as δ/R → 0: the thin shell forces the
radial sector past Feller, the reduction is licensed, and the effective
operator on the retained angular sector gives the Schumann spectrum

  f_n = (c / 2πR) √(n(n+1)).
"""

from __future__ import annotations

import numpy as np


# Physical constants (SI)
R_EARTH = 6.371e6       # m
C_LIGHT = 2.998e8       # m/s
DELTA_IONOSPHERE = 80e3  # m (effective cavity height)


def _radial_overlap(k: int, kp: int) -> float:
    """Overlap integral 2∫₀¹ t sin(kπt) sin(k'πt) dt.

    Nonzero only when k and k' have different parity.
    For k = k': returns 1/2 (normalization).
    """
    if k == kp:
        return 0.5
    if (k - kp) % 2 == 0:
        return 0.0
    # k - k' odd, k + k' odd
    km = k - kp
    kp_sum = k + kp
    return -8.0 * k * kp / (np.pi**2 * km**2 * kp_sum**2)


def build_schumann_blocks(
    delta_over_R: float = 0.01,
    n_angular: int = 8,
    n_radial: int = 5,
) -> tuple[np.ndarray, list[int], list[int], list[str]]:
    """Construct the Hermitian operator for the Earth-ionosphere cavity.

    Parameters
    ----------
    delta_over_R : float
        Thickness ratio δ/R of the spherical shell.
        Earth: δ/R ≈ 0.013.  Default 0.01.
    n_angular : int
        Number of angular modes n = 0, 1, ..., n_angular-1.
    n_radial : int
        Number of radial modes k = 1, 2, ..., n_radial.

    Returns
    -------
    H : np.ndarray
        Hermitian matrix (n_angular * n_radial × n_angular * n_radial).
    retained : list[int]
        Indices of fundamental radial modes (k=1), one per angular mode.
    omitted : list[int]
        Indices of overtone radial modes (k≥2).
    labels : list[str]
        Mode labels "(n, k)" for each index.
    """
    eps = delta_over_R
    N = n_angular * n_radial
    H = np.zeros((N, N))
    labels = []

    # Index mapping: angular mode n, radial mode k (1-indexed)
    # matrix index i = n * n_radial + (k - 1)
    for n in range(n_angular):
        for k in range(1, n_radial + 1):
            labels.append(f"(n={n}, k={k})")

    # Fill matrix
    for n in range(n_angular):
        ang = n * (n + 1)  # angular eigenvalue n(n+1) (units of 1/R²)
        for k in range(1, n_radial + 1):
            i = n * n_radial + (k - 1)
            # Diagonal: angular eigenvalue + radial eigenvalue
            # Radial eigenvalue: (kπ/ε)² where ε = δ/R
            H[i, i] = ang + (k * np.pi / eps) ** 2

            # Off-diagonal: coupling between radial modes at same n
            # From the 1/r² variation: H_{PQ} ~ n(n+1) · 2ε · overlap
            for kp in range(k + 1, n_radial + 1):
                j = n * n_radial + (kp - 1)
                overlap = _radial_overlap(k, kp)
                if abs(overlap) > 1e-15:
                    coupling = ang * 2.0 * eps * overlap
                    H[i, j] = coupling
                    H[j, i] = coupling

    # Ensure exact symmetry
    H = 0.5 * (H + H.T)

    # Partition: retain fundamental radial mode (k=1) for each angular mode
    retained = [n * n_radial for n in range(n_angular)]
    omitted = [i for i in range(N) if i not in retained]

    return H, retained, omitted, labels


def schumann_frequencies_exact(n_max: int = 7) -> np.ndarray:
    """Exact Schumann eigenfrequencies f_n = (c/2πR)√(n(n+1)) in Hz.

    Parameters
    ----------
    n_max : int
        Maximum angular mode number (n = 1, ..., n_max).

    Returns
    -------
    freqs : np.ndarray
        Schumann frequencies in Hz.
    """
    ns = np.arange(1, n_max + 1)
    return (C_LIGHT / (2.0 * np.pi * R_EARTH)) * np.sqrt(ns * (ns + 1))


def schumann_from_schur(
    delta_over_R: float = 0.01,
    n_angular: int = 8,
    n_radial: int = 5,
) -> dict:
    """Compute Schumann frequencies via Schur complement and compare to exact.

    Returns
    -------
    dict with keys:
        chi : float
        regime : str
        exact_freqs : np.ndarray (Hz, for n=1..n_angular-1)
        schur_eigenvalues : np.ndarray (eigenvalues of H_eff)
        schur_freqs : np.ndarray (Hz, converted from H_eff eigenvalues)
        relative_errors : np.ndarray
    """
    import emet

    H, ret, omit, labels = build_schumann_blocks(
        delta_over_R, n_angular, n_radial,
    )
    result = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
    m = result["advanced_metrics"]

    # Exact Schumann frequencies for n = 1, ..., n_angular-1
    exact = schumann_frequencies_exact(n_angular - 1)

    # Extract Schur complement from engine result
    schur_eigs = None
    schur_freqs = None
    rel_errors = None

    if result["regime"] == "subcritical":
        rm = result["reduced_matrix"]
        H_eff = np.array(rm["data"], dtype=np.float64)
        schur_eigs = np.sort(np.linalg.eigvalsh(H_eff))
        # Subtract the radial baseline (π/ε)² common to all retained modes
        radial_baseline = (np.pi / delta_over_R) ** 2
        angular_eigs = schur_eigs - radial_baseline
        # angular_eigs ≈ n(n+1) for n = 0, 1, ...
        # Convert to Hz: f = (c / 2πR) √(angular_eigenvalue)
        pos = angular_eigs[angular_eigs > 0]
        schur_freqs = (C_LIGHT / (2.0 * np.pi * R_EARTH)) * np.sqrt(pos)
        if len(schur_freqs) >= len(exact):
            rel_errors = np.abs(schur_freqs[:len(exact)] - exact) / exact

    return {
        "chi": m["chi"],
        "gamma": m["gamma"],
        "lambda": m["lambda"],
        "regime": result["regime"],
        "exact_freqs": exact,
        "schur_eigenvalues": schur_eigs,
        "schur_freqs": schur_freqs,
        "relative_errors": rel_errors,
    }
