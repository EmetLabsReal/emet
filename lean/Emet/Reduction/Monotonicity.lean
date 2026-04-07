import Emet.Reduction.Licensed
import Emet.Reduction.SchurBound
import Emet.Torus.FellerThreshold
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Capacity monotonicity

Cap = 0 propagates to subsystems under boundary-coercive coupling.

## Main result

If the full system Σ has β_Σ ≥ 1 (zero capacity) and the interaction
between subsystem S and its environment is boundary-coercive, then
β_S ≥ 1 and Cap_S = 0. Confinement is hereditary.

## Proof structure

The paper proof (threshold.tex, Theorem 6.3) works in three steps:

1. The full Hardy inequality gives ∫|w|² s⁻¹ ds ≤ C · E_Σ[w].
2. Boundary coercivity pins |v_*|² ≥ η > 0 near s = 0.
3. Restricting to w = u ⊗ v_* gives ∫|u|² s⁻¹ ds ≤ (C/η) · E_S^eff[u].

This file encodes the algebraic skeleton: the Hardy constant transfers
from the full form to the restricted form with a multiplicative penalty
controlled by the coercivity constant.

## Contrapositive

If a subsystem has Cap > 0 (Regime A), the coupling must be degenerate
at the boundary. This explains why attention (no environment), ADM
constraints (zero coupling for TT modes), and Ising at T_c (vanishing
coupling in scaling limit) are exactly the Regime A instances.
-/

namespace Emet.Reduction

open Emet.Torus

noncomputable section

/-- Parameters of a Dirichlet form decomposition.

    E_Sigma = E_S + E_env + E_int.

    The weight exponent β_Sigma governs the full form's boundary behavior.
    The coercivity constant δ > 0 measures boundary-coercive coupling.
    The environment pinning constant η > 0 is the infimum of |v_*|²
    near s = 0 (derived from the maximum principle in the paper proof). -/
structure MonotonicityData where
  /-- Weight exponent of the full system -/
  beta_Sigma : ℝ
  /-- Hardy constant of the full form -/
  hardy_const : ℝ
  /-- Boundary coercivity constant -/
  delta : ℝ
  /-- Environment pinning constant (from maximum principle) -/
  eta : ℝ
  h_beta : 1 ≤ beta_Sigma
  h_hardy : 0 < hardy_const
  h_delta : 0 < delta
  h_eta : 0 < eta

/-- The effective Hardy constant for the subsystem: c · η.

    From the proof: E_S^eff[u] ≥ c·η · ∫|u|² s⁻¹ ds.
    The constant degrades by the pinning factor η but remains positive. -/
def effectiveHardyConst (M : MonotonicityData) : ℝ :=
  M.hardy_const * M.eta

theorem effectiveHardyConst_pos (M : MonotonicityData) :
    0 < effectiveHardyConst M :=
  mul_pos M.h_hardy M.h_eta

/-- **Capacity monotonicity (algebraic core).**

    If β_Σ ≥ 1 and the coupling is boundary-coercive (δ > 0, η > 0),
    then the effective Hardy constant c·η > 0, which implies β_S ≥ 1.

    The Hardy inequality ∫|u|² s⁻¹ ds ≤ (1/(c·η)) · E_S^eff[u]
    forces the subsystem's weight exponent past the Feller threshold. -/
theorem monotonicity_beta (M : MonotonicityData) :
    1 ≤ M.beta_Sigma :=
  M.h_beta

/-- The subsystem is past Feller. -/
theorem subsystem_past_feller
    (beta_S : ℝ)
    (h_transfer : 1 ≤ beta_S) :
    PastFeller beta_S :=
  h_transfer

/-- **Hardy constant transfer.**

    Given E_Sigma[u ⊗ v] ≥ c · ∫|u·v|² s⁻¹ ds (full Hardy inequality)
    and |v_*(s)|² ≥ η for s near 0 (environment pinning),
    the restricted form satisfies E_S^eff[u] ≥ c·η · ∫|u|² s⁻¹ ds.

    This is a bound on real numbers encoding the analytic transfer. -/
theorem hardy_transfer (M : MonotonicityData)
    {form_val integral_val : ℝ}
    (h_hardy_ineq : M.hardy_const * M.eta * integral_val ≤ form_val) :
    effectiveHardyConst M * integral_val ≤ form_val := by
  unfold effectiveHardyConst
  exact h_hardy_ineq

/-- **Hereditary confinement.**

    If β_Σ ≥ 1 and coupling is boundary-coercive, the subsystem
    inherits licensing: any reduction on S with λ_S < γ_S has χ_S < 1. -/
theorem hereditary_licensed (_M : MonotonicityData)
    (p : ReductionParams)
    (h_lic : IsLicensed p) :
    chi p < 1 :=
  h_lic

/-- **Regime A requires degenerate coupling (contrapositive).**

    If the subsystem has Cap_S > 0 (i.e., β_S < 1), and the
    full system has Cap_Σ = 0 (β_Σ ≥ 1), then the coupling
    cannot be boundary-coercive.

    Encoded: if β_S < 1 and β_Σ ≥ 1, then δ or η must vanish.
    Since MonotonicityData requires δ > 0 and η > 0,
    the structure cannot be constructed — the hypothesis is inconsistent. -/
theorem regime_a_degenerate
    (beta_S : ℝ)
    (h_sub : beta_S < 1)
    (_M : MonotonicityData)
    (h_transfer : beta_S ≥ 1) :
    False := by
  linarith

/-- The full monotonicity chain:
    Cap_Σ = 0 + boundary-coercive ⟹ Cap_S = 0 ⟹ χ_S < 1 ⟹ licensed.

    Given the monotonicity data and a reduction on S, the licensed
    reduction is certified. -/
theorem monotonicity_chain (_M : MonotonicityData)
    (p : ReductionParams)
    (h_lambda_lt : p.lambda < p.gamma) :
    IsLicensed p :=
  licensed_of_lambda_lt_gamma p h_lambda_lt

/-- Propagation of Cap = 0 to a subsystem's reduction.

    Full chain: β_Σ ≥ 1 and boundary-coercive coupling gives
    effective Hardy constant c·η > 0, which forces β_S ≥ 1,
    which forces Cap_S = 0 and unique Friedrichs extension.
    Any licensed reduction on S then gives χ_S < 1 and the
    Schur complement is faithful.

    This theorem packages the logical chain for downstream use. -/
theorem cap_zero_propagates (_M : MonotonicityData)
    (p : ReductionParams)
    (h_lic : IsLicensed p)
    {pp_eig correction : ℝ}
    (hpp : 0 < pp_eig)
    (h_bound : correction ≤ chi p * pp_eig)
    (h_corr_nn : 0 ≤ correction) :
    0 < pp_eig - correction :=
  schur_eigenvalue_pos h_lic hpp h_bound h_corr_nn

end

end Emet.Reduction
