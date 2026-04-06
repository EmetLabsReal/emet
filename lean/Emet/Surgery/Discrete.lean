import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

open scoped BigOperators
open Finset

namespace Emet.Surgery

abbrev GenMat (n : ℕ) := Fin n → Fin n → ℝ

noncomputable def rowSum {n : ℕ} (M : GenMat n) (i : Fin n) : ℝ :=
  ∑ j, M i j

def IsMarkovGenerator {n : ℕ} (M : GenMat n) : Prop :=
  ∀ i, rowSum M i = 0

def OffDiagNonneg {n : ℕ} (M : GenMat n) : Prop :=
  ∀ (i j : Fin n), i ≠ j → 0 ≤ M i j

structure DiscreteValidSurgery (n : ℕ) where
  M : GenMat n
  h_off : OffDiagNonneg M
  h_row : IsMarkovGenerator M

noncomputable def twoWellGen (lam : ℝ) : GenMat 2 :=
  fun i j => if i = j then -lam else lam

theorem twoWellGen_offDiag_nonneg {lam : ℝ} (hlam : 0 ≤ lam) :
    OffDiagNonneg (twoWellGen lam) := by
  intro i j hij
  simp only [twoWellGen, hij, ite_false]
  exact hlam

theorem twoWellGen_rowSum (lam : ℝ) (i : Fin 2) :
    rowSum (twoWellGen lam) i = 0 := by
  unfold rowSum twoWellGen
  simp [Fin.sum_univ_two]
  fin_cases i <;> simp

theorem twoWellGen_isMarkov (lam : ℝ) :
    IsMarkovGenerator (twoWellGen lam) :=
  fun i => twoWellGen_rowSum lam i

noncomputable def twoWellDiscrete (lam : ℝ) (hlam : 0 ≤ lam) :
    DiscreteValidSurgery 2 where
  M := twoWellGen lam
  h_off := twoWellGen_offDiag_nonneg hlam
  h_row := twoWellGen_isMarkov lam

theorem twoWellGen_symmetric (lam : ℝ) (i j : Fin 2) :
    twoWellGen lam i j = twoWellGen lam j i := by
  unfold twoWellGen
  fin_cases i <;> fin_cases j <;> simp

end Emet.Surgery
