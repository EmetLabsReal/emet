import Emet.Reduction.Licensed

namespace Emet.Reduction

noncomputable section

/-- Pessimistic linear error envelope for chi under IEEE 754 perturbation.
    From Cao-Xie-Li (1994) sharp Kahan theorem:
    eps_chi ≤ 2 · chi_hat · (eps_lam/lam + eps_gam/gamma).
    Inherits linearly from perturbations of λ and γ. -/
def epsChi (chi_hat eps_lam lam eps_gam gamma : ℝ) : ℝ :=
  2 * chi_hat * (eps_lam / lam + eps_gam / gamma)

theorem epsChi_nonneg {chi_hat eps_lam lam eps_gam gamma : ℝ}
    (hchi : 0 ≤ chi_hat) (hel : 0 ≤ eps_lam) (hlam : 0 < lam)
    (heg : 0 ≤ eps_gam) (hgam : 0 < gamma) :
    0 ≤ epsChi chi_hat eps_lam lam eps_gam gamma := by
  unfold epsChi
  apply mul_nonneg
  · apply mul_nonneg
    · linarith
    · exact hchi
  · apply add_nonneg
    · exact div_nonneg hel (le_of_lt hlam)
    · exact div_nonneg heg (le_of_lt hgam)

theorem certified_of_sum_lt_one {chi_hat eps_lam lam eps_gam gamma : ℝ}
    (hchi : 0 ≤ chi_hat) (hel : 0 ≤ eps_lam) (hlam : 0 < lam)
    (heg : 0 ≤ eps_gam) (hgam : 0 < gamma)
    (h : chi_hat + epsChi chi_hat eps_lam lam eps_gam gamma < 1) :
    chi_hat < 1 := by
  have heps := epsChi_nonneg hchi hel hlam heg hgam
  linarith

theorem epsChi_zero_of_zero_perturbation {chi_hat lam gamma : ℝ}
    (_hlam : 0 < lam) (_hgam : 0 < gamma) :
    epsChi chi_hat 0 lam 0 gamma = 0 := by
  unfold epsChi
  simp [zero_div]

end

end Emet.Reduction
