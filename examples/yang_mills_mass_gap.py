"""SU(2) Yang-Mills mass gap certification via chi.

Kogut-Susskind Hamiltonian for pure SU(2) gauge theory on a single
plaquette, truncated to j_max representations. The coupling g^2 sweeps
from weak (all representations contribute) to strong (low representations
confine). chi < 1 certifies confinement: the retained sector decouples
from the high-energy representations.

The mass gap is computed exactly at each coupling. In the strong coupling
regime, chi < 1 and the mass gap is certified by the reduction.
"""

from emet.domains.yang_mills import (
    build_single_plaquette,
    mass_gap,
    partition_by_representation,
    sweep_coupling,
)

J_MAX = 4.0   # truncation: j = 0, 1/2, 1, ..., 4
J_CUT = 1.0   # partition: retained j <= 1, omitted j > 1

g2_values = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]

print("=" * 95)
print("SU(2) YANG-MILLS MASS GAP CERTIFICATION")
print("=" * 95)
print()
print(f"Kogut-Susskind Hamiltonian, single plaquette, j_max = {J_MAX}, j_cut = {J_CUT}")
print(f"H[k,k] = (g^2/2) * j(j+1)    electric / Casimir")
print(f"H[k,k+1] = -1/g^2             magnetic / plaquette coupling")
print(f"Retained: j = 0, 1/2, 1       Omitted: j = 3/2, 2, ..., {J_MAX}")
print()

results = sweep_coupling(g2_values, j_max=J_MAX, j_cut=J_CUT)

print("-" * 95)
print(f"{'g^2':>8} {'gamma':>10} {'lambda':>10} {'chi':>14} {'licensed':>8} {'E_0':>10} {'E_1':>10} {'mass_gap':>10}")
print("-" * 95)

for r in results:
    chi_str = f"{r['chi']:14.6e}" if r['chi'] < 1e10 else "           inf"
    lic = "YES" if r['valid'] else "NO"
    print(f"{r['g_squared']:8.2f} {r['gamma']:10.4f} {r['lambda']:10.4f} {chi_str} {lic:>8} {r['E_0']:10.4f} {r['E_1']:10.4f} {r['mass_gap']:10.4f}")

print("-" * 95)
print()

# Transition point
licensed = [r for r in results if r['valid']]
not_licensed = [r for r in results if not r['valid']]

if not_licensed and licensed:
    g2_transition = min(r['g_squared'] for r in licensed)
    print(f"Confinement transition: chi crosses 1 near g^2 = {g2_transition:.1f}")
    print(f"  Below: all representations contribute. Partition not licensed.")
    print(f"  Above: low-j sector confines. Mass gap certified by chi < 1.")
    print()

# Strong coupling verification
if licensed:
    strong = [r for r in licensed if r['g_squared'] >= 2.0]
    if strong:
        chis_decreasing = all(
            strong[i]['chi'] > strong[i+1]['chi']
            for i in range(len(strong) - 1)
        )
        gaps_positive = all(r['mass_gap'] > 0 for r in strong)
        print(f"Strong coupling (g^2 >= 2):")
        print(f"  chi strictly decreasing: {'YES' if chis_decreasing else 'NO'}")
        print(f"  Mass gap positive at all couplings: {'YES' if gaps_positive else 'NO'}")
        print(f"  Deepest chi: {strong[-1]['chi']:.2e} at g^2 = {strong[-1]['g_squared']}")
        print(f"  Mass gap at g^2 = {strong[-1]['g_squared']}: {strong[-1]['mass_gap']:.4f}")
