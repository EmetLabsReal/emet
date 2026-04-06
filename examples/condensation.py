"""Condensation as spectral decoupling.

The transition rate matrix T_{ij} = |<i|r|j>|^2 / |E_i - E_j|
measures how fast eigenstate i can transition to eigenstate j
(Fermi golden rule). The diagonal T_{ii} = <i|r|i> is the
expectation value of position in state i.

Chi applied to T with the partition [ground state | rest]
answers: does the ground state decouple from the excited spectrum?

Three regimes:
  chi > 1 at n_cut=1:  no condensation (weak binding)
  chi < 1 at n_cut=1:  BEC (ground state separates)
  chi < 1 at all cuts:  confinement (entire spectrum collapses)

Confinement slopes (Kahan-certified):
  Hydrogen:   chi ~ Z^{-6}
  Yang-Mills:  chi ~ (g^2)^{-4}
"""

import numpy as np
import emet
from emet.domains.kahan import certified_subcritical
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
    """Fermi golden rule matrix in the energy eigenbasis."""
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


n_grid = 400


# --- Table 1: Hydrogen condensation ---
print("Table 1. Hydrogen condensation (n_cut=1, adaptive grid)")
print()
print(f"{'Z':>6} {'chi':>14} {'kahan':>6} {'slope':>8} {'regime':>15}")
print("-" * 56)

Z_list = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
Z_vals, chi_vals = [], []

for Z in Z_list:
    r_max = 80.0 / Z
    H, r_vec = radial_hamiltonian(n_grid, r_max, Z=Z, l=0)
    eigs, vecs = np.linalg.eigh(H)
    n_bound = int(np.sum(eigs < 0))
    if n_bound < 2:
        continue

    T = transition_rate_matrix(eigs, vecs, r_vec)
    result = emet.decide_dense_matrix(T, retained=[0], omitted=list(range(1, n_grid)))
    m = result["advanced_metrics"]
    chi = m["chi"]

    if chi is not None and chi > 0:
        kahan = certified_subcritical(chi, m["gamma"], m["lambda"])
        cert = "YES" if kahan["certified"] else "no"

        slope_s = ""
        if len(Z_vals) > 0:
            s = np.log(chi / chi_vals[-1]) / np.log(Z / Z_vals[-1])
            slope_s = f"{s:.2f}"

        Z_vals.append(Z)
        chi_vals.append(chi)

        chi_s = f"{chi:.6e}"
        print(f"{Z:6.1f} {chi_s:>14} {cert:>6} {slope_s:>8} {result['regime']:>15}")

fit_h = np.polyfit(np.log(Z_vals[-4:]), np.log(chi_vals[-4:]), 1)
print(f"\nAsymptotic slope (Z >= 15): chi ~ Z^{fit_h[0]:.4f}")


# --- Table 2: Hydrogen n_cut sweep ---
print()
print("Table 2. Hydrogen Z=1, vary n_cut")
print()
print(f"{'n_cut':>6} {'chi':>14} {'kahan':>6} {'regime':>15}")
print("-" * 48)

H, r_vec = radial_hamiltonian(n_grid, 80.0, Z=1.0, l=0)
eigs, vecs = np.linalg.eigh(H)
T = transition_rate_matrix(eigs, vecs, r_vec)

for n_cut in range(1, 8):
    result = emet.decide_dense_matrix(T, retained=list(range(n_cut)), omitted=list(range(n_cut, n_grid)))
    m = result["advanced_metrics"]
    chi = m["chi"]
    if chi is not None:
        kahan = certified_subcritical(chi, m["gamma"], m["lambda"])
        cert = "YES" if kahan["certified"] else "no"
    else:
        cert = "---"
    chi_s = f"{chi:.6e}" if chi is not None else "N/A"
    print(f"{n_cut:6d} {chi_s:>14} {cert:>6} {result['regime']:>15}")


# --- Table 3: Yang-Mills confinement ---
print()
print("Table 3. Yang-Mills confinement (j_cut=1)")
print()
print(f"{'g^2':>6} {'chi':>14} {'kahan':>6} {'slope':>8} {'regime':>15}")
print("-" * 56)

g2_vals, chi_ym = [], []

for g2 in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]:
    H_ym, ret, omit, _ = build_plaquette_blocks(g2, j_max=10, j_cut=1)
    result = emet.decide_dense_matrix(H_ym, retained=ret, omitted=omit)
    m = result["advanced_metrics"]
    chi = m["chi"]

    if chi is not None and chi > 0:
        kahan = certified_subcritical(chi, m["gamma"], m["lambda"])
        cert = "YES" if kahan["certified"] else "no"

        slope_s = ""
        if len(g2_vals) > 0:
            s = np.log(chi / chi_ym[-1]) / np.log(g2 / g2_vals[-1])
            slope_s = f"{s:.2f}"

        g2_vals.append(g2)
        chi_ym.append(chi)

        chi_s = f"{chi:.6e}"
        print(f"{g2:6.1f} {chi_s:>14} {cert:>6} {slope_s:>8} {result['regime']:>15}")

fit_ym = np.polyfit(np.log(g2_vals[-4:]), np.log(chi_ym[-4:]), 1)
print(f"\nAsymptotic slope (g^2 >= 16): chi ~ (g^2)^{fit_ym[0]:.4f}")


# --- Summary ---
print()
print("=" * 56)
print("Confinement slopes (Kahan-certified)")
print(f"  Hydrogen:    chi ~ Z^{fit_h[0]:.1f}")
print(f"  Yang-Mills:  chi ~ (g^2)^{fit_ym[0]:.1f}")
print("=" * 56)
