"""Conservation of omitted return.

H_PP = S + R where S is the Schur complement and R = H_PQ H_QQ^{-1} H_QP.
R is the backreaction of the omitted sector. ||R|| <= gamma * chi.

Verify: ||R|| scales as (g^2)^{-3} for Yang-Mills and Z^{-7} for hydrogen.
"""

import numpy as np
import emet
from emet.domains.yang_mills import build_plaquette_blocks


def radial_hamiltonian(n_grid, r_max, Z=1.0, l=0, mass=1.0):
    dr = r_max / (n_grid + 1)
    r = np.array([(i + 1) * dr for i in range(n_grid)])
    kinetic = np.zeros((n_grid, n_grid))
    coeff = 1.0 / (2.0 * mass * dr**2)
    for i in range(n_grid):
        kinetic[i, i] = 2.0 * coeff
        if i > 0:
            kinetic[i, i - 1] = -coeff
        if i < n_grid - 1:
            kinetic[i, i + 1] = -coeff
    potential = np.diag(l * (l + 1) / (2.0 * mass * r**2) - Z / r)
    H = kinetic + potential
    H = 0.5 * (H + H.T)
    return H, r


def transition_rate_matrix(eigs, vecs, r_vec):
    n = len(eigs)
    R_eig = vecs.T @ np.diag(r_vec) @ vecs
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = R_eig[i, i]
        for j in range(n):
            if i != j:
                gap = abs(eigs[i] - eigs[j])
                if gap > 1e-12:
                    T[i, j] = R_eig[i, j] ** 2 / gap
    return 0.5 * (T + T.T)


def compute_return(H, retained, omitted):
    """Compute R = H_PQ H_QQ^{-1} H_QP and its norm."""
    P = retained
    Q = omitted
    H_PQ = H[np.ix_(P, Q)]
    H_QQ = H[np.ix_(Q, Q)]
    H_QP = H[np.ix_(Q, P)]
    QQ_inv = np.linalg.inv(H_QQ)
    R = H_PQ @ QQ_inv @ H_QP
    return np.linalg.norm(R, ord=2)  # operator norm


# --- Yang-Mills ---
print("Yang-Mills: ||R|| vs (g^2)")
print(f"{'g^2':>6} {'||R||':>14} {'gamma*chi':>14} {'slope':>8}")
print("-" * 48)

g2_vals, R_norms_ym = [], []
for g2 in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
    H, ret, omit, _ = build_plaquette_blocks(g2, j_max=10, j_cut=1)
    R_norm = compute_return(H, ret, omit)
    result = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
    m = result["advanced_metrics"]
    gam_chi = m["gamma"] * m["chi"] if m["chi"] is not None else 0

    slope_s = ""
    if len(g2_vals) > 0:
        s = np.log(R_norm / R_norms_ym[-1]) / np.log(g2 / g2_vals[-1])
        slope_s = f"{s:.2f}"

    g2_vals.append(g2)
    R_norms_ym.append(R_norm)
    print(f"{g2:6.1f} {R_norm:14.6e} {gam_chi:14.6e} {slope_s:>8}")

fit = np.polyfit(np.log(g2_vals[-4:]), np.log(R_norms_ym[-4:]), 1)
print(f"\nAsymptotic: ||R|| ~ (g^2)^{fit[0]:.4f}")


# --- Hydrogen ---
print()
print("Hydrogen: ||R|| vs Z")
print(f"{'Z':>6} {'||R||':>14} {'gamma*chi':>14} {'slope':>8}")
print("-" * 48)

n_grid = 400
Z_vals, R_norms_h = [], []
for Z in [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
    r_max = 80.0 / Z
    H, r_vec = radial_hamiltonian(n_grid, r_max, Z=Z, l=0)
    eigs, vecs = np.linalg.eigh(H)
    n_bound = int(np.sum(eigs < 0))
    if n_bound < 2:
        continue

    T = transition_rate_matrix(eigs, vecs, r_vec)
    ret = [0]
    omit = list(range(1, n_grid))

    R_norm = compute_return(T, ret, omit)
    result = emet.decide_dense_matrix(T, retained=ret, omitted=omit)
    m = result["advanced_metrics"]
    gam_chi = m["gamma"] * m["chi"] if m["chi"] is not None else 0

    slope_s = ""
    if len(Z_vals) > 0:
        s = np.log(R_norm / R_norms_h[-1]) / np.log(Z / Z_vals[-1])
        slope_s = f"{s:.2f}"

    Z_vals.append(Z)
    R_norms_h.append(R_norm)
    print(f"{Z:6.1f} {R_norm:14.6e} {gam_chi:14.6e} {slope_s:>8}")

fit_h = np.polyfit(np.log(Z_vals[-4:]), np.log(R_norms_h[-4:]), 1)
print(f"\nAsymptotic: ||R|| ~ Z^{fit_h[0]:.4f}")

print()
print("=" * 48)
print(f"Yang-Mills return exponent: {fit[0]:.2f}")
print(f"Hydrogen return exponent:   {fit_h[0]:.2f}")
print("=" * 48)
