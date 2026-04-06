import Emet.Surgery.Irreversibility
import Emet.Reduction.Licensed

namespace Emet.Reduction

noncomputable section

-- Two-point function decay from spectral gap.
-- ⟨φ(0,t)φ(0,0)⟩ ~ Σ Aₙ exp(-Δₙ t). Positive Δ₀ forces exponential decay.

/-- Exponential decay: amplitude * exp(-gap * t) < amplitude
    when gap > 0 and t > 0. -/
theorem two_point_decay {mass_gap t amplitude : ℝ}
    (h_gap : 0 < mass_gap) (ht : 0 < t) (h_amp : 0 < amplitude) :
    amplitude * Real.exp (-(mass_gap * t)) < amplitude := by
  have hexp := Emet.Surgery.contraction_factor_lt_one h_gap ht
  calc amplitude * Real.exp (-(mass_gap * t))
      < amplitude * 1 := by exact mul_lt_mul_of_pos_left hexp h_amp
    _ = amplitude := mul_one amplitude

/-- Positive spectral gap implies exp(-gap * t) < 1 for all t > 0. -/
theorem decay_iff_gap {mass_gap : ℝ} (h_gap : 0 < mass_gap) :
    ∀ (t : ℝ), 0 < t → Real.exp (-(mass_gap * t)) < 1 :=
  fun _t ht => Emet.Surgery.contraction_factor_lt_one h_gap ht

/-- exp(-gap * t) is strictly decreasing in t for gap > 0. -/
theorem decay_monotone {mass_gap t1 t2 : ℝ}
    (h_gap : 0 < mass_gap) (_ht1 : 0 < t1) (ht : t1 < t2) :
    Real.exp (-(mass_gap * t2)) < Real.exp (-(mass_gap * t1)) := by
  apply Real.exp_strictMono
  nlinarith

end

end Emet.Reduction
