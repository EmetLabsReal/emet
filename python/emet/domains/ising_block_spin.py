"""2D Ising block-spin: capacity transition at criticality.

Wilson's block-spin RG partitions a lattice into blocks of size b×b.
The block average is the retained (P) sector; intra-block fluctuations
are the omitted (Q) sector.

Away from T_c: exponential clustering forces γ > 0, χ < 1, Cap = 0.
The coarse-grained description is a licensed reduction.

At T_c: power-law correlations keep λ bounded below while γ closes.
χ → ∞, Cap > 0.  The block-spin partition has an accessible boundary.
The universality class is the choice of extension.

The operator partitioned is the spin-spin covariance matrix
C_{ij} = ⟨σ_i σ_j⟩ − ⟨σ_i⟩⟨σ_j⟩, an N×N positive semidefinite matrix.
Sites are split into block representatives (P) and intra-block
fluctuation sites (Q).  χ = (||C_PQ|| / σ_min(C_QQ))².

Implementation: exact enumeration for small lattices (≤ 4×4 = 16 spins).
Memory: O(2^N · N) for configurations, O(N²) for the covariance matrix.
"""

from __future__ import annotations

import numpy as np
from itertools import product


# 2D Ising critical temperature (exact, Onsager)
J_DEFAULT = 1.0
T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ≈ 2.269185...
BETA_C = 1.0 / T_C


def _spin_configs(n_spins: int) -> np.ndarray:
    """All 2^n spin configurations as rows of ±1 values."""
    return np.array(list(product([-1, 1], repeat=n_spins)), dtype=np.float64)


def ising_energies(
    Lx: int,
    Ly: int,
    J: float = J_DEFAULT,
    h: float = 0.0,
    periodic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute energies for all 2^(Lx*Ly) configurations of 2D Ising.

    H = -J Σ_{<ij>} σ_i σ_j  -  h Σ_i σ_i

    Parameters
    ----------
    h : float
        External magnetic field.  A small positive h breaks Z₂ symmetry,
        giving well-defined connected correlators in the ordered phase.

    Returns (configs, energies) where configs is (2^N, N) and
    energies is (2^N,).  Never builds the full 2^N × 2^N matrix.
    """
    N = Lx * Ly
    configs = _spin_configs(N)

    def idx(x: int, y: int) -> int:
        return x * Ly + y

    energies = np.zeros(len(configs))
    for x in range(Lx):
        for y in range(Ly):
            i = idx(x, y)
            if periodic or x + 1 < Lx:
                j = idx((x + 1) % Lx, y)
                energies -= J * configs[:, i] * configs[:, j]
            if periodic or y + 1 < Ly:
                j = idx(x, (y + 1) % Ly)
                energies -= J * configs[:, i] * configs[:, j]

    if h != 0.0:
        energies -= h * configs.sum(axis=1)

    return configs, energies


def block_spin_partition(
    Lx: int,
    Ly: int,
    bx: int,
    by: int,
    beta: float,
    J: float = J_DEFAULT,
    h: float = 0.0,
    periodic: bool = True,
) -> dict:
    """Partition the spin-spin covariance matrix into block/fluctuation sectors.

    The covariance matrix C_{ij} = ⟨σ_i σ_j⟩ − ⟨σ_i⟩⟨σ_j⟩ is an N×N
    PSD matrix.  We partition the N = Lx·Ly sites into:
      P = one representative per block (block-average direction)
      Q = remaining sites (intra-block fluctuations)

    χ = (||C_PQ||₂ / σ_min(C_QQ))² measures the coupling strength.
    """
    if Lx % bx != 0 or Ly % by != 0:
        raise ValueError(
            f"Block size ({bx},{by}) must divide lattice ({Lx},{Ly})"
        )

    N = Lx * Ly
    configs, energies = ising_energies(Lx, Ly, J, h, periodic)

    # Boltzmann probabilities (shifted for numerical stability)
    log_w = -beta * energies
    log_w -= np.max(log_w)
    weights = np.exp(log_w)
    probs = weights / np.sum(weights)

    # Spin expectations and covariance (N×N, not 2^N × 2^N)
    mean_sigma = configs.T @ probs  # (N,)
    weighted = configs * np.sqrt(probs)[:, None]  # (2^N, N)
    corr = weighted.T @ weighted  # (N, N)
    cov = corr - np.outer(mean_sigma, mean_sigma)
    cov = 0.5 * (cov + cov.T)

    # Partition sites: one representative per block (top-left corner)
    n_blocks_x = Lx // bx
    n_blocks_y = Ly // by
    retained = []
    omitted = []

    for bxi in range(n_blocks_x):
        for byi in range(n_blocks_y):
            rep = (bxi * bx) * Ly + (byi * by)
            retained.append(rep)
            for dx in range(bx):
                for dy in range(by):
                    site = (bxi * bx + dx) * Ly + (byi * by + dy)
                    if site != rep:
                        omitted.append(site)

    retained = sorted(retained)
    omitted = sorted(omitted)

    C_PP = cov[np.ix_(retained, retained)]
    C_PQ = cov[np.ix_(retained, omitted)]
    C_QQ = cov[np.ix_(omitted, omitted)]

    # chi = (lambda / gamma)^2
    lambda_coupling = float(np.linalg.norm(C_PQ, ord=2))
    eigvals_QQ = np.linalg.eigvalsh(C_QQ)
    gamma = float(np.min(np.abs(eigvals_QQ)))

    if gamma < 1e-15:
        chi = float("inf")
    else:
        chi = (lambda_coupling / gamma) ** 2

    return {
        "C_PP": C_PP,
        "C_PQ": C_PQ,
        "C_QQ": C_QQ,
        "chi": chi,
        "gamma": gamma,
        "lambda_coupling": lambda_coupling,
        "retained": retained,
        "omitted": omitted,
        "n_retained": len(retained),
        "n_omitted": len(omitted),
        "beta": beta,
        "T": 1.0 / beta if beta > 0 else float("inf"),
        "mean_magnetization": float(np.mean(np.abs(mean_sigma))),
    }


def sweep_temperature(
    Lx: int = 4,
    Ly: int = 4,
    bx: int = 2,
    by: int = 2,
    T_values: np.ndarray | None = None,
    J: float = J_DEFAULT,
) -> list[dict]:
    """Sweep temperature and compute χ at each point.

    Shows the capacity transition: Cap = 0 (χ < 1) away from T_c,
    Cap > 0 (χ → ∞) near T_c.
    """
    if T_values is None:
        T_values = np.linspace(0.5, 4.0, 50)

    results = []
    for T in T_values:
        beta = 1.0 / T
        r = block_spin_partition(Lx, Ly, bx, by, beta, J)
        results.append({
            "T": T,
            "beta": beta,
            "chi": r["chi"],
            "gamma": r["gamma"],
            "lambda": r["lambda_coupling"],
        })
    return results


def chi_ratio_at_Tc(
    Lx: int = 4,
    Ly: int = 4,
    bx: int = 2,
    by: int = 2,
    J: float = J_DEFAULT,
) -> dict:
    """Compare χ at T_c vs away from T_c.

    Returns chi at T_c, at T = 0.5*T_c (ordered), and at T = 2*T_c (disordered).
    """
    results = {}
    for label, T in [("T_c", T_C), ("ordered", 0.5 * T_C), ("disordered", 2.0 * T_C)]:
        beta = 1.0 / T
        r = block_spin_partition(Lx, Ly, bx, by, beta, J)
        results[label] = {
            "T": T,
            "chi": r["chi"],
            "gamma": r["gamma"],
            "lambda": r["lambda_coupling"],
        }
    return results


def scaling_analysis(
    bx: int = 2,
    by: int = 2,
    J: float = J_DEFAULT,
) -> dict:
    """Compare χ scaling at T_c vs T > T_c across lattice sizes.

    Uses lattice sizes (4,2) and (4,4) with block size (bx, by).
    At T_c, χ grows rapidly with L (Cap > 0).
    At T > T_c, χ converges (Cap = 0).

    Returns dict with scaling ratios at each temperature.
    """
    sizes = [(4, 2), (4, 4)]
    temperatures = {
        "T_c": T_C,
        "1.5_T_c": 1.5 * T_C,
        "2_T_c": 2.0 * T_C,
        "3_T_c": 3.0 * T_C,
    }

    results = {}
    for label, T in temperatures.items():
        beta = 1.0 / T
        chis = []
        for Lx, Ly in sizes:
            r = block_spin_partition(Lx, Ly, bx, by, beta, J)
            chis.append(r["chi"])
        ratio = chis[1] / chis[0] if chis[0] > 0 else float("inf")
        results[label] = {
            "T": T,
            "chi_small": chis[0],
            "chi_large": chis[1],
            "ratio": ratio,
        }
    return results
