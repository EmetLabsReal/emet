"""Modular surface Γ\H: hyperbolic Laplacian at the cusp.

The modular surface SL(2,Z)\H has a single cusp at y → ∞.
The hyperbolic Laplacian Δ = -y²(∂²_x + ∂²_y) decomposes
into Fourier modes in x.

Liouville transform: t = log(y), u(t) = y^{1/2} f(y).
The operator becomes -d²/dt² + V(t) on L²(dt), where:
  - Zero mode (n=0): V(t) = 1/4
  - Mode n ≠ 0:      V(t) = 1/4 + (2πn)² e^{2t}

This is a standard 1D Schrödinger operator. The cusp y → ∞
maps to t → ∞. The continuous spectrum starts at 1/4.

Partition: P = bulk (t ≤ t_cut), Q = cusp (t > t_cut).
Coupling H_PQ is a single off-diagonal entry at the boundary.
γ = σ_min(H_QQ) → 1/4 as cusp extends. λ = ||H_PQ|| is bounded.
χ = λ²/γ² should converge (and for β ≥ 1, approach a finite limit).

The Feller exponent at the cusp is β = 2 (parabolic rank 1).
Cap = 0 since β ≥ 1. Unique Friedrichs extension.
Self-adjointness → all eigenvalues real → Re(s) = 1/2.
"""

from __future__ import annotations

import numpy as np


def _schrodinger_1d(t_nodes: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Discretize -d²u/dt² + V(t) u on uniform grid.

    Returns symmetric tridiagonal matrix (dense).
    Dirichlet boundary at both ends.
    """
    N = len(t_nodes)
    dt = t_nodes[1] - t_nodes[0]
    inv_dt2 = 1.0 / (dt * dt)

    H = np.zeros((N, N))

    for i in range(1, N - 1):
        H[i, i] = 2.0 * inv_dt2 + V[i]
        H[i, i - 1] = -inv_dt2
        H[i, i + 1] = -inv_dt2

    # Dirichlet: large diagonal at boundaries
    H[0, 0] = 2.0 * inv_dt2 + V[0]
    H[N - 1, N - 1] = 2.0 * inv_dt2 + V[N - 1]

    return H


def build_modular_cusp(
    T_max: float = 10.0,
    T_cut: float = 3.0,
    n_bulk: int = 60,
    n_cusp: int = 40,
    n_fourier: int = 1,
    T_min: float = 0.0,
) -> tuple[np.ndarray, list[int], list[int], dict]:
    """Construct Liouville-transformed hyperbolic Laplacian.

    Coordinates: t = log(y), uniform grid in t.
    Operator: -d²/dt² + V_n(t) for each Fourier mode n.
    V_0(t) = 1/4. V_n(t) = 1/4 + (2πn)² e^{2t} for n > 0.

    Parameters
    ----------
    T_max : float
        Upper truncation in t = log(y) coordinate. y = e^T_max.
    T_cut : float
        Partition boundary. P: t ∈ [T_min, T_cut). Q: t ∈ [T_cut, T_max].
    n_bulk : int
        Grid points in bulk.
    n_cusp : int
        Grid points in cusp.
    n_fourier : int
        Number of Fourier modes (1 = zero mode only).
    T_min : float
        Lower boundary (t = log(y_min), y_min = 1 → T_min = 0).
    """
    t_bulk = np.linspace(T_min, T_cut, n_bulk, endpoint=False)
    t_cusp = np.linspace(T_cut, T_max, n_cusp + 1)
    t_nodes = np.concatenate([t_bulk, t_cusp])
    Nt = len(t_nodes)

    N_total = Nt * n_fourier
    H = np.zeros((N_total, N_total))

    for mode in range(n_fourier):
        # Potential for this Fourier mode
        if mode == 0:
            V = np.full(Nt, 0.25)
        else:
            V = 0.25 + (2 * np.pi * mode) ** 2 * np.exp(2 * t_nodes)

        H_mode = _schrodinger_1d(t_nodes, V)
        offset = mode * Nt
        H[offset:offset + Nt, offset:offset + Nt] = H_mode

    H = 0.5 * (H + H.T)

    retained = []
    omitted = []
    for mode in range(n_fourier):
        offset = mode * Nt
        for i in range(Nt):
            if i < n_bulk:
                retained.append(offset + i)
            else:
                omitted.append(offset + i)

    meta = {
        "T_max": T_max,
        "T_cut": T_cut,
        "T_min": T_min,
        "Y_max": np.exp(T_max),
        "Y_cut": np.exp(T_cut),
        "n_bulk": n_bulk,
        "n_cusp": n_cusp,
        "n_fourier": n_fourier,
        "N_total": N_total,
        "N_retained": len(retained),
        "N_omitted": len(omitted),
        "t_nodes": t_nodes,
    }

    return H, retained, omitted, meta


def cusp_sweep(
    T_values: list[float] | np.ndarray | None = None,
    T_cut: float = 3.0,
    n_bulk: int = 60,
    n_cusp: int = 40,
    n_fourier: int = 1,
) -> list[dict]:
    """Sweep cusp extent T_max, compute χ at each truncation."""
    import emet

    if T_values is None:
        T_values = [4, 5, 6, 7, 8, 9, 10, 12, 15, 20]

    results = []
    for T in T_values:
        # Scale cusp grid with T
        n_c = max(n_cusp, int(n_cusp * (T - T_cut) / (T_values[0] - T_cut)) if T > T_cut else n_cusp)
        H, ret, omit, meta = build_modular_cusp(
            T_max=T,
            T_cut=T_cut,
            n_bulk=n_bulk,
            n_cusp=n_c,
            n_fourier=n_fourier,
        )

        report = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
        metrics = report.get("advanced_metrics", {})

        results.append({
            "T_max": T,
            "Y_max": np.exp(T),
            "N": meta["N_total"],
            "N_P": meta["N_retained"],
            "N_Q": meta["N_omitted"],
            "chi": metrics.get("chi"),
            "lambda": metrics.get("lambda"),
            "gamma": metrics.get("gamma"),
            "regime": report.get("regime", "unknown"),
            "valid": report.get("valid", False),
        })

    return results
