import Emet.Torus.FellerThreshold
import Emet.Reduction.Licensed

namespace Emet.Torus

noncomputable section

/-- beta = d * alpha. -/

def weightExponentDim (d : ℕ) (alpha : ℝ) : ℝ := d * alpha

def fellerAlpha (d : ℕ) : ℝ := 1 / d

/-- fellerAlpha 4 = 1/4. -/
theorem feller_alpha_4d : fellerAlpha 4 = 1 / 4 := by
  unfold fellerAlpha
  norm_num

/-- PastFeller (d * alpha) ↔ 1/d ≤ alpha. -/
theorem pastFeller_iff_alpha {d : ℕ} (hd : 0 < d) (alpha : ℝ) :
    PastFeller (weightExponentDim d alpha) ↔ 1 / (d : ℝ) ≤ alpha := by
  unfold PastFeller weightExponentDim
  have hd_pos : (0 : ℝ) < d := Nat.cast_pos.mpr hd
  rw [show (↑d : ℝ) * alpha = alpha * ↑d from mul_comm _ _]
  exact (div_le_iff₀ hd_pos).symm

/-- g² ≥ 1 → PastFeller (4 * (g²/4)). -/
theorem pastFeller_4d (g_squared : ℝ) (hg : 1 ≤ g_squared) :
    PastFeller (weightExponentDim 4 (g_squared / 4)) := by
  unfold PastFeller weightExponentDim
  linarith

/-- g² > 2 → MexicanHatCentrifugal (4 * (g²/4)). -/
theorem mexicanHat_4d (g_squared : ℝ) (hg : 2 < g_squared) :
    MexicanHatCentrifugal (weightExponentDim 4 (g_squared / 4)) := by
  unfold weightExponentDim
  have : 2 < (4 : ℝ) * (g_squared / 4) := by linarith
  exact mexicanHat_of_gt_two this

/-- d₁ < d₂ → d₁ * alpha < d₂ * alpha for alpha > 0. -/
theorem higher_dim_larger_beta {d1 d2 : ℕ} (alpha : ℝ) (ha : 0 < alpha)
    (hd : d1 < d2) :
    weightExponentDim d1 alpha < weightExponentDim d2 alpha := by
  unfold weightExponentDim
  have : (d1 : ℝ) < (d2 : ℝ) := Nat.cast_lt.mpr hd
  exact mul_lt_mul_of_pos_right this ha

end

end Emet.Torus
