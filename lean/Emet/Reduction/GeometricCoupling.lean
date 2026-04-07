import Emet.Reduction.Licensed
import Emet.Torus.EffectivePotential
import Emet.Torus.FellerThreshold
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic

/-!
# Geometric coupling

χ²(β) = 8/(3β(β−2)) for the radial-angular partition at the
WdW transverse section valley. Taylor expansion of 1/r² at r = a₀.

## Main results

- `chiSq`: χ²(β) = 8/(3β(β−2))
- `chiSq_three_lt_one`: χ²(3) = 8/9 < 1
- `chiSq_mono`: χ² decreasing for β > 2
- `chiSq_lt_one_of_ge_three`: χ²(β) < 1 for β ≥ 3
- `casimir_cancellation`: mode number n drops out of χ
-/

namespace Emet.Reduction

open Emet.Torus

noncomputable section

/-- χ²(β) = 8/(3β(β−2)). Square of the regime parameter for the
    radial-angular partition. -/
def chiSq (beta : ℝ) : ℝ := 8 / (3 * beta * (beta - 2))

/-- 3β(β−2) > 0 for β > 2. -/
theorem chiSq_denom_pos {beta : ℝ} (hb : 2 < beta) :
    0 < 3 * beta * (beta - 2) := by
  have h1 : 0 < beta := by linarith
  have h2 : 0 < beta - 2 := by linarith
  positivity

/-- χ²(β) > 0 for β > 2. -/
theorem chiSq_pos {beta : ℝ} (hb : 2 < beta) :
    0 < chiSq beta := by
  unfold chiSq
  exact div_pos (by norm_num) (chiSq_denom_pos hb)

/-- χ²(3) = 8/9. -/
theorem chiSq_three : chiSq 3 = 8 / 9 := by
  unfold chiSq
  norm_num

/-- χ²(3) < 1. -/
theorem chiSq_three_lt_one : chiSq 3 < 1 := by
  rw [chiSq_three]
  norm_num

/-- χ² decreasing for β > 2. -/
theorem chiSq_mono {b1 b2 : ℝ} (h1 : 2 < b1) (h2 : b1 < b2) :
    chiSq b2 < chiSq b1 := by
  unfold chiSq
  have hd1 := chiSq_denom_pos h1
  have hd2 : 0 < 3 * b2 * (b2 - 2) := chiSq_denom_pos (by linarith)
  rw [div_lt_div_iff₀ hd2 hd1]
  nlinarith

/-- χ²(β) ≤ 8/9 for β ≥ 3. -/
theorem chiSq_le_chiSq_three {beta : ℝ} (hb : 3 ≤ beta) :
    chiSq beta ≤ chiSq 3 := by
  rcases eq_or_lt_of_le hb with rfl | hlt
  · exact le_refl _
  · exact le_of_lt (chiSq_mono (by norm_num : (2 : ℝ) < 3) hlt)

/-- χ²(β) < 1 for β ≥ 3. -/
theorem chiSq_lt_one_of_ge_three {beta : ℝ} (hb : 3 ≤ beta) :
    chiSq beta < 1 :=
  lt_of_le_of_lt (chiSq_le_chiSq_three hb) chiSq_three_lt_one

/-- Geometric coupling gives λ < γ for β ≥ 3. -/
theorem geometric_coupling_licensed {beta gamma : ℝ}
    (hb : 3 ≤ beta) (hg : 0 < gamma)
    (lambda : ℝ) (hl : lambda ^ 2 = chiSq beta * gamma ^ 2)
    (hln : 0 ≤ lambda) :
    lambda < gamma := by
  have hchi := chiSq_lt_one_of_ge_three hb
  have hg2 : 0 < gamma ^ 2 := by positivity
  have : lambda ^ 2 < gamma ^ 2 := by
    calc lambda ^ 2 = chiSq beta * gamma ^ 2 := hl
      _ < 1 * gamma ^ 2 := by exact mul_lt_mul_of_pos_right hchi hg2
      _ = gamma ^ 2 := one_mul _
  exact (sq_lt_sq₀ hln (le_of_lt hg)).mp this

/-- Mode number n cancels in χ. -/
theorem casimir_cancellation {c d : ℝ} (hc : 0 < c) (hd : 0 < d)
    {n : ℤ} (hn : n ≠ 0) :
    (2 * ↑n ^ 2 / c) / (↑n ^ 2 / d) = 2 * d / c := by
  have hn2 : (0 : ℝ) < ↑n ^ 2 := by positivity
  field_simp

/-- β = 3: past Feller, Mexican hat, χ² < 1, licensed. -/
theorem geometric_chain_d3
    (p : ReductionParams) (h_lic : IsLicensed p) :
    PastFeller 3
    ∧ MexicanHatCentrifugal 3
    ∧ chiSq 3 < 1
    ∧ chi p < 1 :=
  ⟨pastFeller_of_gt_two (by norm_num),
   mexicanHat_of_gt_two (by norm_num),
   chiSq_three_lt_one,
   h_lic⟩

end

end Emet.Reduction
