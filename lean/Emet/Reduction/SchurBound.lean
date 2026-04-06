import Emet.Reduction.Licensed

namespace Emet.Reduction

/-- Schur complement correction norm ≤ chi · ‖H_PP‖.
    Chi < 1 makes it a strict contraction. -/

theorem schur_contraction {chi_val pp_norm correction_norm : ℝ}
    (hchi : chi_val < 1) (hpp : 0 < pp_norm)
    (h_bound : correction_norm ≤ chi_val * pp_norm) :
    correction_norm < pp_norm := by
  calc correction_norm ≤ chi_val * pp_norm := h_bound
    _ < 1 * pp_norm := by exact mul_lt_mul_of_pos_right hchi hpp
    _ = pp_norm := one_mul pp_norm

theorem schur_correction_nonneg {chi_val pp_norm : ℝ}
    (hchi : 0 ≤ chi_val) (hpp : 0 ≤ pp_norm) :
    0 ≤ chi_val * pp_norm :=
  mul_nonneg hchi hpp

theorem schur_of_zero_coupling {correction pp_eigenvalue : ℝ}
    (h : correction = 0) :
    pp_eigenvalue - correction = pp_eigenvalue := by
  rw [h, sub_zero]

/-- If chi < 1 and the eigenvalue of PP is positive, the corrected
    eigenvalue (eigenvalue - correction) remains positive. -/
theorem schur_eigenvalue_pos {chi_val pp_eig correction : ℝ}
    (hchi : chi_val < 1) (hpp : 0 < pp_eig)
    (h_bound : correction ≤ chi_val * pp_eig)
    (_h_corr_nn : 0 ≤ correction) :
    0 < pp_eig - correction := by
  have h1 : correction < pp_eig := lt_of_le_of_lt h_bound
    (by calc chi_val * pp_eig < 1 * pp_eig := mul_lt_mul_of_pos_right hchi hpp
        _ = pp_eig := one_mul pp_eig)
  linarith

end Emet.Reduction
