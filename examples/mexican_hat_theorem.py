"""The Mexican Hat Theorem: licensed reductions force potential structure.

Demonstrates that for the pinched torus / Yang-Mills family:
  - chi < 1 (licensed) AND beta > 2 → Mexican hat potential is FORCED
  - The potential is not observed; it is a consequence of the reduction rules
  - The theorem boundary is sharp: 1 < beta < 2 gives licensing but no centrifugal repulsion

Certified in Lean 4: lean/Emet/Reduction/MexicanHatForced.lean (zero sorry).
"""

import sys
sys.path.insert(0, "python")

import numpy as np

from emet.domains.torus import build_torus_operator, effective_potential
from emet.domains.mexican_hat import licensed_implies_mexican_hat
from emet.domains.yang_mills import build_plaquette_blocks
from emet.domains.kahan import certified_subcritical

import emet


def main():
    print("=" * 100)
    print("THE MEXICAN HAT THEOREM: licensed reductions force potential structure")
    print("=" * 100)
    print()
    print("V_eff(beta, s) = beta*(beta-2) / (4s^2)")
    print("  beta > 2:  V_eff > 0  (repulsive centrifugal barrier = Mexican hat crown)")
    print("  beta = 2:  V_eff = 0  (transition)")
    print("  beta < 2:  V_eff < 0  (attractive, no centrifugal repulsion)")
    print()

    # --- Part 1: Torus sweep ---
    print("-" * 100)
    print("TORUS SWEEP: implication chain at each beta")
    print("-" * 100)
    print(f"  {'beta':>6}  {'chi':>14}  {'licensed':>8}  {'V_eff>0':>8}  {'BOTH':>6}  {'regime'}")
    print("-" * 100)

    betas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    for beta in betas:
        H, ret, omit = build_torus_operator(beta)
        result = licensed_implies_mexican_hat(H, ret, omit, beta)
        chi_str = f"{result['chi']:.6e}" if result["chi"] is not None else "∞"
        lic = "YES" if result["licensed"] else "NO"
        mh = "YES" if result["centrifugal_mexican_hat"] else "NO"
        both = "YES" if result["both"] else "---"
        print(f"  {beta:6.1f}  {chi_str:>14}  {lic:>8}  {mh:>8}  {both:>6}  {result['regime']}")

    print()
    print("The theorem boundary is sharp:")
    print("  beta = 1.5: licensed (chi < 1) but V_eff < 0 — no centrifugal repulsion")
    print("  beta = 2.5: licensed AND V_eff > 0 — full Mexican hat forced")
    print()

    # --- Part 2: Yang-Mills ---
    print("-" * 100)
    print("YANG-MILLS: g^2 = beta. Deep coupling forces Mexican hat.")
    print("-" * 100)
    print(f"  {'g^2':>6}  {'chi':>14}  {'licensed':>8}  {'V_eff>0':>8}  {'Kahan cert':>10}  {'gap':>10}")
    print("-" * 100)

    for g2 in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]:
        beta = g2
        H, ret, omit, _ = build_plaquette_blocks(g2, j_max=4.0, j_cut=1.0)
        report = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
        m = report["advanced_metrics"]
        chi = m["chi"]

        licensed = chi is not None and chi < 1
        centrifugal = beta > 2

        if licensed:
            cert = certified_subcritical(chi, m["gamma"], m["lambda"])
            kahan = "YES" if cert["certified"] else "NO"
        else:
            kahan = "---"

        eigvals = np.linalg.eigvalsh(H)
        if eigvals.shape[0] < 2:
            gap = 0.0
        else:
            gap = float(eigvals[1] - eigvals[0])

        chi_str = f"{chi:.6e}" if chi is not None else "∞"
        lic = "YES" if licensed else "NO"
        mh = "YES" if centrifugal else "NO"
        print(f"  {g2:6.1f}  {chi_str:>14}  {lic:>8}  {mh:>8}  {kahan:>10}  {gap:10.4f}")

    print()
    print("INTERPRETATION")
    print()
    print("  The Mexican hat theorem (lean/Emet/Reduction/MexicanHatForced.lean):")
    print("    LicensedTorusReduction with beta > 2 → MexicanHatCentrifugal")
    print()
    print("  Licensed reduction on a pinched torus with beta > 2")
    print("  produces the Mexican hat. The potential follows from the estimate.")


if __name__ == "__main__":
    main()
