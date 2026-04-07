"""Quantum channel certification via Choi matrix partition.

Given: the Choi matrix of a quantum channel, partitioned into signal
indices {0,3} and error indices {1,2}.

Checked: cross-block coupling lambda, suppression floor gamma,
regime parameter chi = (lambda/gamma)^2, Schur error scale epsilon.

Returned: regime classification, effective channel on the signal
sector when subcritical, key rate from Shor-Preskill.

Result: On Pauli channels, lambda = 0, chi = 0, the Schur complement
is exact, and the key rate r = 1 - 2h(q) is recovered. On non-Pauli
channels (misalignment), lambda > 0 and the certification diagnostics
detect reduction failure that the key rate formula alone does not.
"""

import sys
sys.path.insert(0, "python")

import numpy as np
from emet.domains.quantum_channel import (
    certify_channel,
    choi_amplitude_damping,
    choi_bit_flip,
    choi_dephasing,
    choi_depolarizing,
    choi_identity,
    choi_misaligned,
    shor_preskill_rate,
)


def _fmt_chi(chi):
    if chi is None:
        return "N/A"
    return f"{chi:.2e}"


def part1_pauli_channels():
    print("PART 1: Pauli channels")
    print()
    print("  Signal indices {0,3} = |00>, |11>. Error indices {1,2} = |01>, |10>.")
    print("  Depolarizing is the only Pauli channel with a full-rank error block.")
    print("  Others (identity, dephasing, bit-flip, amplitude damping) have")
    print("  gamma = 0: the error block is zero or singular.")
    print()
    print(f"{'channel':>20}  {'param':>8}  {'lambda':>10}  {'gamma':>10}  {'chi':>10}  {'QBER':>8}  {'key rate':>10}  {'regime':>14}")
    print("-" * 110)

    channels = [
        ("identity", choi_identity(), "—"),
        ("depolarizing", choi_depolarizing(0.05), "0.05"),
        ("depolarizing", choi_depolarizing(0.10), "0.10"),
        ("depolarizing", choi_depolarizing(0.25), "0.25"),
        ("depolarizing", choi_depolarizing(0.50), "0.50"),
        ("dephasing", choi_dephasing(0.10), "0.10"),
        ("dephasing", choi_dephasing(0.25), "0.25"),
        ("bit-flip", choi_bit_flip(0.10), "0.10"),
        ("bit-flip", choi_bit_flip(0.25), "0.25"),
        ("amplitude-damp", choi_amplitude_damping(0.10), "0.10"),
        ("amplitude-damp", choi_amplitude_damping(0.50), "0.50"),
        ("amplitude-damp", choi_amplitude_damping(0.90), "0.90"),
    ]

    for name, choi, param in channels:
        r = certify_channel(choi)
        chi_str = _fmt_chi(r["chi"])
        print(f"{name:>20}  {param:>8}  {r['lambda']:10.2e}  {r['gamma']:10.2e}  {chi_str:>10}  "
              f"{r['qber']:8.4f}  {r['key_rate']:10.6f}  {r['regime']:>14}")

    print()
    print("Depolarizing: lambda = 0, chi = 0, subcritical. Exact Schur reduction.")
    print("Others: lambda = 0 but gamma = 0 (error block singular). Pre-admissible.")


def part2_misaligned_channel():
    print()
    print("PART 2: Hadamard-misaligned channel — nonzero cross-coupling")
    print()
    print("  E(rho) = (1-p) rho + p H rho H where H is the Hadamard gate.")
    print("  Mixes computational basis into superpositions.")
    print("  Cross-block becomes nonzero: lambda > 0.")
    print("  Error block remains singular: gamma = 0, chi undefined.")
    print("  Diagnostic: lambda detects the misalignment even when chi cannot be formed.")
    print()
    print(f"{'p':>8}  {'lambda':>10}  {'gamma':>10}  {'chi':>10}  {'QBER':>8}  {'key rate':>10}  {'regime':>14}")
    print("-" * 85)

    for p in [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        choi = choi_misaligned(p)
        r = certify_channel(choi)
        chi_str = _fmt_chi(r["chi"])
        print(f"{p:8.2f}  {r['lambda']:10.2e}  {r['gamma']:10.2e}  {chi_str:>10}  "
              f"{r['qber']:8.4f}  {r['key_rate']:10.6f}  {r['regime']:>14}")

    print()
    print("Lambda grows with p. Emet detects that the reduction cannot be formed.")
    print("The key rate formula alone does not see the cross-coupling.")


def part3_comparison():
    print()
    print("PART 3: Depolarizing vs misaligned at same QBER")
    print()
    print("  Two channels with the same QBER but different lambda.")
    print("  Depolarizing: lambda = 0, certified. Misaligned: lambda > 0, not certified.")
    print()

    print(f"{'QBER':>8}  {'channel':>16}  {'lambda':>10}  {'chi':>10}  {'key rate':>10}  {'certified':>10}")
    print("-" * 75)

    for qber_target in [0.05, 0.10, 0.15]:
        # Depolarizing: QBER = 2p/3, so p = 3*QBER/2
        p_depol = min(1.0, 3 * qber_target / 2)
        r_d = certify_channel(choi_depolarizing(p_depol))

        # Misaligned: QBER ~ p/2, so p ~ 2*QBER
        r_m = certify_channel(choi_misaligned(qber_target * 2))

        chi_d = _fmt_chi(r_d["chi"])
        chi_m = _fmt_chi(r_m["chi"])
        print(f"{r_d['qber']:8.4f}  {'depolarizing':>16}  {r_d['lambda']:10.2e}  {chi_d:>10}  "
              f"{r_d['key_rate']:10.6f}  {'YES' if r_d['valid'] else 'NO':>10}")
        print(f"{r_m['qber']:8.4f}  {'misaligned':>16}  {r_m['lambda']:10.2e}  {chi_m:>10}  "
              f"{r_m['key_rate']:10.6f}  {'YES' if r_m['valid'] else 'NO':>10}")
        print()

    print("Same QBER, different lambda. Emet distinguishes them.")


def main():
    part1_pauli_channels()
    part2_misaligned_channel()
    part3_comparison()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("  1. Depolarizing: lambda = 0, chi = 0, exact Schur reduction.")
    print("  2. Misaligned: lambda > 0, reduction cannot be formed.")
    print("  3. Same QBER, different lambda: emet sees what key rate does not.")


if __name__ == "__main__":
    main()
