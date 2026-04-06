import Emet.Surgery.Discrete

namespace Emet.Surgery

theorem twoWellGen_unique (M : GenMat 2)
    (h_markov : IsMarkovGenerator M)
    (h_sym : ∀ (i j : Fin 2), M i j = M j i) :
    M = twoWellGen (M 0 1) := by
  funext i j
  have h_row0 := h_markov 0
  unfold rowSum at h_row0
  simp [Fin.sum_univ_two] at h_row0
  have h_row1 := h_markov 1
  unfold rowSum at h_row1
  simp [Fin.sum_univ_two] at h_row1
  have h10 : M 1 0 = M 0 1 := (h_sym 0 1).symm
  fin_cases i <;> fin_cases j <;> simp only [twoWellGen] <;> simp <;> linarith [h10]

theorem twoWellGen_rate_nonneg (M : GenMat 2)
    (h_off : OffDiagNonneg M) :
    0 ≤ M 0 1 :=
  h_off 0 1 (by decide)

end Emet.Surgery
