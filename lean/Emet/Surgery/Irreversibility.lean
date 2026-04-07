import Emet.Surgery.Discrete
import Emet.Surgery.Uniqueness

namespace Emet.Surgery

noncomputable section

-- Eigenvalues of twoWellGen(lam): 0 and -2*lam.
-- Spectral gap = 2*lam.

/-- The diagonal entry of twoWellGen is -lam. -/
theorem twoWellGen_diag (lam : ℝ) (i : Fin 2) :
    twoWellGen lam i i = -lam := by
  unfold twoWellGen
  simp

/-- The trace of twoWellGen is -2*lam. Since one eigenvalue is 0
    (Markov generator), the other must be -2*lam. -/
theorem twoWellGen_trace (lam : ℝ) :
    twoWellGen lam 0 0 + twoWellGen lam 1 1 = -2 * lam := by
  simp [twoWellGen_diag]
  ring

/-- The constant vector (1,1) is in the kernel: M·1 = 0. -/
theorem twoWellGen_kernel (lam : ℝ) (i : Fin 2) :
    twoWellGen lam i 0 + twoWellGen lam i 1 = 0 := by
  have := twoWellGen_rowSum lam i
  unfold rowSum at this
  simp [Fin.sum_univ_two] at this
  exact this

/-- The difference vector d = (1, -1) is an eigenvector with eigenvalue -2*lam. -/
theorem twoWellGen_fiedler_eigenvalue (lam : ℝ) (i : Fin 2) :
    twoWellGen lam i 0 - twoWellGen lam i 1 = -2 * lam * (if i = 0 then 1 else -1) := by
  unfold twoWellGen
  fin_cases i <;> simp <;> ring

/-- Spectral gap = 2*lam. Positive when lam > 0. -/
def spectralGap (lam : ℝ) : ℝ := 2 * lam

theorem spectralGap_pos {lam : ℝ} (hlam : 0 < lam) :
    0 < spectralGap lam := by
  unfold spectralGap
  linarith

theorem spectralGap_eq_neg_trace (lam : ℝ) :
    spectralGap lam = -(twoWellGen lam 0 0 + twoWellGen lam 1 1) := by
  rw [twoWellGen_trace]
  unfold spectralGap
  ring

/-- exp(-gap·t) < 1 for gap > 0 and t > 0. -/
theorem contraction_factor_lt_one {gap t : ℝ} (hgap : 0 < gap) (ht : 0 < t) :
    Real.exp (-(gap * t)) < 1 := by
  rw [Real.exp_lt_one_iff]
  linarith [mul_pos hgap ht]

/-- A nonzero symmetric 2-state Markov generator has positive spectral gap.
    The only alternative is the zero generator. -/
theorem gap_or_trivial (M : GenMat 2)
    (h_markov : IsMarkovGenerator M)
    (h_sym : ∀ (i j : Fin 2), M i j = M j i)
    (h_off : OffDiagNonneg M) :
    (0 < spectralGap (M 0 1)) ∨ (M = fun _ _ => 0) := by
  by_cases h : M 0 1 = 0
  · right
    have huniq := twoWellGen_unique M h_markov h_sym
    rw [huniq, h]
    funext i j
    unfold twoWellGen
    simp
  · left
    have hnn := twoWellGen_rate_nonneg M h_off
    have hpos : 0 < M 0 1 := lt_of_le_of_ne hnn (Ne.symm h)
    exact spectralGap_pos hpos

end

end Emet.Surgery
