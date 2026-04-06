import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace Emet.Torus

noncomputable section

/-- The effective potential on a pinched torus with measure s^β ds.
    V_eff(β, s) = β(β − 2) / (4s²).
    For β > 2: repulsive (Mexican hat centrifugal barrier).
    For β < 2: attractive (boundary accessible).
    At β = 1: V_eff = −1/(4s²), the Hardy constant at the Feller threshold. -/
def vEff (beta s : ℝ) : ℝ := beta * (beta - 2) / (4 * s ^ 2)

theorem vEff_pos_of_gt_two {beta s : ℝ} (hb : 2 < beta) (hs : 0 < s) :
    0 < vEff beta s := by
  unfold vEff
  apply div_pos
  · nlinarith
  · positivity

theorem vEff_neg_of_pos_lt_two {beta s : ℝ} (hb0 : 0 < beta) (hb2 : beta < 2) (hs : 0 < s) :
    vEff beta s < 0 := by
  unfold vEff
  apply div_neg_of_neg_of_pos
  · nlinarith
  · positivity

theorem vEff_zero_of_eq_two (s : ℝ) :
    vEff 2 s = 0 := by
  unfold vEff
  simp [sub_self]

theorem vEff_mono {b1 b2 s : ℝ} (hs : 0 < s) (h1 : 2 < b1) (h2 : b1 < b2) :
    vEff b1 s < vEff b2 s := by
  unfold vEff
  have hd : (0 : ℝ) < 4 * s ^ 2 := by positivity
  exact div_lt_div_of_pos_right (by nlinarith) hd

end

end Emet.Torus
