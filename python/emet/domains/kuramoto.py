"""Kramers-Moyal generator for inertial Kuramoto oscillators.

Phase-space layout: n_theta phase sites, n_hermite momentum modes per site.
Retained modes: Hermite index 0 at each site (overdamped projection).
Omitted modes: Hermite indices 1+ (fast momentum relaxation).
chi < 1 certifies that overdamped reduction is licensed.
"""

from __future__ import annotations

import numpy as np


def flat_index(n_hermite: int, theta_idx: int, hermite_idx: int) -> int:
    return int(theta_idx * n_hermite + hermite_idx)


def hermite_partition(n_theta: int, n_hermite: int) -> tuple[list[int], list[int]]:
    """Retained: all sites, mode 0. Omitted: all sites, modes 1+."""
    if n_theta < 1 or n_hermite < 2:
        raise ValueError("require n_theta >= 1 and n_hermite >= 2")
    retained = [flat_index(n_hermite, i, 0) for i in range(n_theta)]
    omitted = [
        flat_index(n_hermite, i, j)
        for i in range(n_theta)
        for j in range(1, n_hermite)
    ]
    return retained, omitted


def build_generator(
    n_theta: int,
    n_hermite: int,
    *,
    mass_diag: float,
    gamma_excited: float,
    theta_coupling: float,
    streaming_scale: float,
) -> np.ndarray:
    """Symmetric (n_theta * n_hermite) matrix: toy hypoelliptic template."""
    n = n_theta * n_hermite
    h = np.zeros((n, n), dtype=float)
    for i in range(n_theta):
        i0 = flat_index(n_hermite, i, 0)
        h[i0, i0] += mass_diag
        for j in range(1, n_hermite):
            ij = flat_index(n_hermite, i, j)
            h[ij, ij] += gamma_excited * float(j)
        if n_theta > 1:
            ip = (i + 1) % n_theta
            jp0 = flat_index(n_hermite, ip, 0)
            h[i0, jp0] += theta_coupling
            h[jp0, i0] += theta_coupling
        for j in range(1, n_hermite):
            ij = flat_index(n_hermite, i, j)
            i0 = flat_index(n_hermite, i, 0)
            h[i0, ij] += streaming_scale
            h[ij, i0] += streaming_scale
    return 0.5 * (h + h.T)


def preset_smoluchowski() -> tuple[np.ndarray, list[int], list[int]]:
    """4x4 subcritical preset: chi < 1 on standard 2+2 split."""
    matrix = np.array([
        [2.0, 0.1, 0.05, 0.05],
        [0.1, 2.0, 0.05, 0.05],
        [0.05, 0.05, 5.0, 0.0],
        [0.05, 0.05, 0.0, 5.0],
    ], dtype=float)
    retained, omitted = hermite_partition(2, 2)
    return matrix, retained, omitted


def preset_hysteresis() -> tuple[np.ndarray, list[int], list[int]]:
    """4x4 pre-admissible preset: gamma ~ 0, chi undefined."""
    matrix = np.array([
        [1.0, 0.1, 0.0, 0.1],
        [0.1, 1e-12, 0.1, 0.0],
        [0.0, 0.1, 1.0, 0.1],
        [0.1, 0.0, 0.1, 1e-12],
    ], dtype=float)
    retained, omitted = hermite_partition(2, 2)
    return matrix, retained, omitted
