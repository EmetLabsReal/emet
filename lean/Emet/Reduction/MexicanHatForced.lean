import Emet.Torus.FellerThreshold
import Emet.Reduction.Licensed
import Emet.Reduction.SchurBound

namespace Emet.Reduction

open Emet.Torus

/-- A licensed torus reduction with deep coupling.

    The fields encode three facts:
    1. The reduction is licensed (chi < 1): the Schur complement is
       well-defined and the effective operator preserves valley structure.
    2. We are past the Feller threshold (β ≥ 1): the boundary capacity
       has vanished and the Friedrichs extension confines.
    3. The coupling is deep enough (β > 2): the centrifugal potential
       V_eff = β(β-2)/(4s²) is strictly repulsive.

    Facts 1 and 2 are general infrastructure (any licensed reduction
    on any domain). Fact 3 is specific to the pinched torus family and
    gives the explicit Mexican-hat potential shape.

    The hypothesis h_deep is verified numerically by the Python/Rust
    infrastructure for each concrete operator family (torus, Yang-Mills).
    It is NOT assumed — it is a checkable condition. -/
structure LicensedTorusReduction where
  params : ReductionParams
  beta : ℝ
  h_licensed : IsLicensed params
  h_past_feller : PastFeller beta
  h_deep : 2 < beta

/-- Licensed torus reduction with β > 2 forces Mexican-hat centrifugal potential. -/
theorem licensed_torus_forces_mexican_hat (T : LicensedTorusReduction) :
    MexicanHatCentrifugal T.beta :=
  mexicanHat_of_gt_two T.h_deep

/-- Licensed + deep coupling gives valley confinement and centrifugal repulsion. -/
theorem licensed_torus_full_structure (T : LicensedTorusReduction)
    {pp_eig correction : ℝ} (hpp : 0 < pp_eig)
    (h_bound : correction ≤ chi T.params * pp_eig)
    (h_corr_nn : 0 ≤ correction) :
    MexicanHatCentrifugal T.beta ∧ 0 < pp_eig - correction :=
  ⟨licensed_torus_forces_mexican_hat T,
   schur_eigenvalue_pos T.h_licensed hpp h_bound h_corr_nn⟩

end Emet.Reduction
