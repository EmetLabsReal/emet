import Mathlib.Analysis.CStarAlgebra.Matrix
import Emet.Reduction.Licensed

/-!
# Schur complement norm bound

The return term R = H_PQ · H_QQ⁻¹ · H_QP satisfies ‖R‖₂ ≤ γχ.

This is the load-bearing connection between the regime parameter
χ = (λ/γ)² and the Schur complement.  Without it, χ is a number
the code computes.  With it, χ controls the reduction error.

## Proof chain

    ‖H_PQ · H_QQ⁻¹ · H_QP‖
      ≤ ‖H_PQ‖ · ‖H_QQ⁻¹‖ · ‖H_QP‖     (l2_opNorm_mul)
      = λ · ‖H_QQ⁻¹‖ · λ                  (l2_opNorm_conjTranspose)
      ≤ λ · (1/γ) · λ                       (σ_min hypothesis)
      = λ²/γ = γ(λ/γ)² = γχ                 (algebra)

The hypothesis `‖H_QQ⁻¹‖ ≤ 1/γ` is the operator-norm encoding of
σ_min(H_QQ) ≥ γ.
-/

namespace Emet.Reduction

noncomputable section

open scoped Matrix.Norms.L2Operator

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- A block reduction grounded in matrices with the L2 operator norm.

    * `H_PQ`: cross-coupling block (m × n).
    * `H_QQ_inv`: inverse of the omitted diagonal block (n × n).
    * `gamma`: σ_min(H_QQ), encoded via `‖H_QQ_inv‖ ≤ 1/gamma`. -/
structure MatrixBlock (m n : Type*) [Fintype m] [Fintype n]
    [DecidableEq m] [DecidableEq n] where
  H_PQ : Matrix m n ℝ
  H_QQ_inv : Matrix n n ℝ
  gamma : ℝ
  gamma_pos : 0 < gamma
  h_inv_norm : ‖H_QQ_inv‖ ≤ 1 / gamma

/-- The return term R = H_PQ · H_QQ⁻¹ · H_PQᴴ. -/
def MatrixBlock.R (b : MatrixBlock m n) : Matrix m m ℝ :=
  b.H_PQ * b.H_QQ_inv * b.H_PQ.conjTranspose

/-- λ = ‖H_PQ‖₂. -/
def MatrixBlock.lam (b : MatrixBlock m n) : ℝ := ‖b.H_PQ‖

/-- Lift to abstract ReductionParams. -/
def MatrixBlock.toParams (b : MatrixBlock m n) : ReductionParams where
  lambda := b.lam
  gamma := b.gamma
  lambda_nonneg := norm_nonneg _
  gamma_pos := b.gamma_pos

/-- The abstract χ from matrix data. -/
def MatrixBlock.chiVal (b : MatrixBlock m n) : ℝ :=
  chi b.toParams

/-- Algebraic identity: λ²/γ = γ · (λ/γ)². -/
private theorem lam_sq_div_eq (lam gamma : ℝ) (hg : gamma ≠ 0) :
    lam ^ 2 / gamma = gamma * (lam / gamma) ^ 2 := by
  field_simp

/-- **The Schur complement norm bound: ‖R‖₂ ≤ λ²/γ.**

    Three applications of submultiplicativity plus ‖Aᴴ‖ = ‖A‖. -/
theorem MatrixBlock.R_norm_le (b : MatrixBlock m n) :
    ‖b.R‖ ≤ b.lam ^ 2 / b.gamma := by
  unfold R lam
  calc ‖b.H_PQ * b.H_QQ_inv * b.H_PQ.conjTranspose‖
      ≤ ‖b.H_PQ * b.H_QQ_inv‖ * ‖b.H_PQ.conjTranspose‖ :=
        Matrix.l2_opNorm_mul _ _
    _ ≤ (‖b.H_PQ‖ * ‖b.H_QQ_inv‖) * ‖b.H_PQ.conjTranspose‖ :=
        mul_le_mul_of_nonneg_right (Matrix.l2_opNorm_mul _ _) (norm_nonneg _)
    _ = (‖b.H_PQ‖ * ‖b.H_QQ_inv‖) * ‖b.H_PQ‖ := by
        rw [Matrix.l2_opNorm_conjTranspose]
    _ ≤ (‖b.H_PQ‖ * (1 / b.gamma)) * ‖b.H_PQ‖ :=
        mul_le_mul_of_nonneg_right
          (mul_le_mul_of_nonneg_left b.h_inv_norm (norm_nonneg _))
          (norm_nonneg _)
    _ = ‖b.H_PQ‖ ^ 2 / b.gamma := by ring

/-- **‖R‖₂ ≤ γχ.** -/
theorem MatrixBlock.R_norm_le_gamma_chi (b : MatrixBlock m n) :
    ‖b.R‖ ≤ b.gamma * b.chiVal := by
  have h := b.R_norm_le
  rw [lam_sq_div_eq b.lam b.gamma (ne_of_gt b.gamma_pos)] at h
  unfold chiVal chi toParams lam at *
  exact h

/-- **Licensed contraction: χ < 1 implies ‖R‖₂ < γ.** -/
theorem MatrixBlock.R_lt_gamma (b : MatrixBlock m n)
    (h_lic : IsLicensed b.toParams) :
    ‖b.R‖ < b.gamma :=
  calc ‖b.R‖
      ≤ b.gamma * b.chiVal := b.R_norm_le_gamma_chi
    _ < b.gamma * 1 := mul_lt_mul_of_pos_left h_lic b.gamma_pos
    _ = b.gamma := mul_one _

/-- **Licensed eigenvalue bound: σ_min(H_PP) - ‖R‖ > 0 when χ < 1.**

    Combined with ‖R‖ ≤ γχ and the existing `schur_eigenvalue_pos`,
    this gives σ_min(H_eff) ≥ σ_min(H_PP) - γχ > 0. -/
theorem MatrixBlock.correction_lt_eigenvalue (b : MatrixBlock m n)
    (h_lic : IsLicensed b.toParams)
    {pp_eig : ℝ} (hpp_ge : b.gamma ≤ pp_eig) :
    ‖b.R‖ < pp_eig :=
  lt_of_lt_of_le (b.R_lt_gamma h_lic) hpp_ge

end

end Emet.Reduction
