import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace Emet.Reduction

noncomputable section

/-- Parameters of a block reduction: λ = spectral norm of cross-coupling,
    γ = smallest singular value of the omitted block.
    chi = (λ/γ)² is the regime parameter. chi < 1 licenses the reduction. -/
structure ReductionParams where
  lambda : ℝ
  gamma : ℝ
  lambda_nonneg : 0 ≤ lambda
  gamma_pos : 0 < gamma

def chi (p : ReductionParams) : ℝ := (p.lambda / p.gamma) ^ 2

def IsLicensed (p : ReductionParams) : Prop := chi p < 1

theorem chi_nonneg (p : ReductionParams) : 0 ≤ chi p := by
  unfold chi
  exact sq_nonneg _

theorem chi_zero_of_lambda_zero (p : ReductionParams) (h : p.lambda = 0) :
    chi p = 0 := by
  unfold chi
  rw [h, zero_div, sq, mul_zero]

theorem licensed_iff_lambda_lt_gamma (p : ReductionParams) :
    IsLicensed p ↔ p.lambda < p.gamma := by
  unfold IsLicensed chi
  have hg := p.gamma_pos
  have hln := p.lambda_nonneg
  constructor
  · intro h
    by_contra hle
    simp only [not_lt] at hle
    -- gamma ≤ lambda, so lambda/gamma ≥ 1, so (lambda/gamma)^2 ≥ 1
    have hd : 1 ≤ p.lambda / p.gamma := by
      rw [le_div_iff₀' hg]; linarith
    have : 1 ≤ (p.lambda / p.gamma) ^ 2 := by
      have : 0 ≤ p.lambda / p.gamma := div_nonneg hln (le_of_lt hg)
      nlinarith
    linarith
  · intro h
    -- lambda < gamma, so lambda/gamma < 1, so (lambda/gamma)^2 < 1
    have hd : p.lambda / p.gamma < 1 := by
      rw [div_lt_one₀ hg]; exact h
    have hdn : 0 ≤ p.lambda / p.gamma := div_nonneg hln (le_of_lt hg)
    nlinarith [sq_abs (p.lambda / p.gamma)]

theorem licensed_of_lambda_lt_gamma (p : ReductionParams) (h : p.lambda < p.gamma) :
    IsLicensed p :=
  (licensed_iff_lambda_lt_gamma p).mpr h

end

end Emet.Reduction
