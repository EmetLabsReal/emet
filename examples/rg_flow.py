"""Chi as anomalous dimension.

If chi ~ (coupling)^{-2*gamma_anomalous}, the confinement slope
gives the anomalous dimension directly:

  Yang-Mills:  chi ~ (g^2)^{-4}  =>  lambda/gamma ~ (g^2)^{-2}  =>  gamma = 2
  Hydrogen:    chi ~ Z^{-6}      =>  lambda/gamma ~ Z^{-3}       =>  gamma = 3

Test 1: Does chi(j_cut) at fixed g^2 follow RG flow?
  j_cut is the partition scale (= energy cutoff).
  If chi(j_cut) ~ j_cut^{-alpha}, alpha is the scaling dimension.

Test 2: Does chi(g^2, j_cut) factorize?
  If chi = f(g^2) * h(j_cut), the coupling dependence and the
  scale dependence separate. That's a fixed point.

Test 3: Anomalous dimension extraction.
  lambda/gamma vs coupling on log-log plot. Slope = gamma.
"""

import numpy as np
import emet
from emet.domains.yang_mills import build_plaquette_blocks
from emet.domains.kahan import certified_subcritical


# --- Test 1: Yang-Mills chi vs j_cut (RG flow) ---
print("Test 1: Yang-Mills chi vs j_cut at fixed g^2")
print("j_cut = partition scale = energy cutoff")
print()

j_cuts = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

for g2 in [4.0, 8.0, 16.0]:
    print(f"g^2 = {g2}")
    print(f"  {'j_cut':>6} {'lambda':>10} {'gamma':>10} {'lam/gam':>10} {'chi':>14} {'regime':>12}")
    print(f"  {'-' * 66}")

    j_vals, chi_vals, ratio_vals = [], [], []
    for j_cut in j_cuts:
        if j_cut >= 10:
            continue
        H, ret, omit, _ = build_plaquette_blocks(g2, j_max=10, j_cut=j_cut)
        if not ret or not omit:
            continue
        result = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
        m = result["advanced_metrics"]
        chi = m["chi"]
        lam = m["lambda"]
        gam = m["gamma"]
        ratio = lam / gam if gam > 0 else float("inf")

        chi_s = f"{chi:.6e}" if chi is not None else "N/A"
        j_vals.append(j_cut)
        chi_vals.append(chi)
        ratio_vals.append(ratio)
        print(f"  {j_cut:6.1f} {lam:10.6f} {gam:10.6f} {ratio:10.6f} {chi_s:>14} {result['regime']:>12}")

    if len(j_vals) >= 2:
        fit = np.polyfit(np.log(j_vals), np.log(chi_vals), 1)
        fit_r = np.polyfit(np.log(j_vals), np.log(ratio_vals), 1)
        print(f"  chi ~ j_cut^{fit[0]:.2f},  lambda/gamma ~ j_cut^{fit_r[0]:.2f}")
    print()


# --- Test 2: Factorization ---
print("=" * 80)
print("Test 2: Does chi(g^2, j_cut) factorize?")
print("chi(g^2, j_cut) = f(g^2) * h(j_cut)")
print()
print("Ratio chi(g^2, j_cut) / chi(g^2, j_cut=1):")
print()

header = f"{'g^2':>6}"
for jc in j_cuts:
    header += f" {'j=' + str(jc):>10}"
print(header)
print("-" * (7 + 11 * len(j_cuts)))

for g2 in [2.0, 4.0, 8.0, 16.0, 32.0]:
    row = f"{g2:6.1f}"
    chi_ref = None
    for j_cut in j_cuts:
        H, ret, omit, _ = build_plaquette_blocks(g2, j_max=10, j_cut=j_cut)
        if not ret or not omit:
            row += f" {'---':>10}"
            continue
        result = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
        chi = result["advanced_metrics"]["chi"]
        if chi is not None:
            if j_cut == 1.0:
                chi_ref = chi
            if chi_ref and chi_ref > 0:
                row += f" {chi / chi_ref:10.4f}"
            else:
                row += f" {'---':>10}"
        else:
            row += f" {'---':>10}"
    print(row)

print()
print("If factorized: each column should be constant across rows.")
print()


# --- Test 3: Anomalous dimension extraction ---
print("=" * 80)
print("Test 3: Anomalous dimension gamma = -slope of log(lambda/gamma) vs log(coupling)")
print()

# Yang-Mills
print("Yang-Mills (j_cut=1):")
g2_vals, ratios_ym = [], []
for g2 in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
    H, ret, omit, _ = build_plaquette_blocks(g2, j_max=10, j_cut=1)
    result = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
    m = result["advanced_metrics"]
    ratio = m["lambda"] / m["gamma"] if m["gamma"] > 0 else 0
    g2_vals.append(g2)
    ratios_ym.append(ratio)

for i in range(1, len(g2_vals)):
    slope = np.log(ratios_ym[i] / ratios_ym[i-1]) / np.log(g2_vals[i] / g2_vals[i-1])
    print(f"  g^2={g2_vals[i-1]:5.1f} -> {g2_vals[i]:5.1f}:  gamma = {-slope:.4f}")

fit_ym = np.polyfit(np.log(g2_vals), np.log(ratios_ym), 1)
print(f"  Overall: lambda/gamma ~ (g^2)^{fit_ym[0]:.4f}  =>  gamma_anomalous = {-fit_ym[0]:.4f}")
print()

# Hydrogen
print("Hydrogen (n_cut=1, adaptive grid):")


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


n_grid = 400
Z_vals, ratios_h = [], []
for Z in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
    r_max = 80.0 / Z
    H, r_vec = radial_hamiltonian(n_grid, r_max, Z=Z, l=0)
    eigs, vecs = np.linalg.eigh(H)
    n_bound = int(np.sum(eigs < 0))
    if n_bound < 2:
        continue
    T = transition_rate_matrix(eigs, vecs, r_vec)
    result = emet.decide_dense_matrix(T, retained=[0], omitted=list(range(1, n_grid)))
    m = result["advanced_metrics"]
    ratio = m["lambda"] / m["gamma"] if m["gamma"] > 0 else 0
    Z_vals.append(Z)
    ratios_h.append(ratio)

for i in range(1, len(Z_vals)):
    slope = np.log(ratios_h[i] / ratios_h[i-1]) / np.log(Z_vals[i] / Z_vals[i-1])
    print(f"  Z={Z_vals[i-1]:5.1f} -> {Z_vals[i]:5.1f}:  gamma = {-slope:.4f}")

fit_h = np.polyfit(np.log(Z_vals), np.log(ratios_h), 1)
print(f"  Overall: lambda/gamma ~ Z^{fit_h[0]:.4f}  =>  gamma_anomalous = {-fit_h[0]:.4f}")

print()
print("=" * 80)
print("Summary")
print(f"  Yang-Mills anomalous dimension:  gamma = {-fit_ym[0]:.2f}")
print(f"  Hydrogen anomalous dimension:    gamma = {-fit_h[0]:.2f}")
print("=" * 80)
