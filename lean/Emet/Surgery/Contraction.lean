import Mathlib.Data.NNReal.Defs
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

open scoped BigOperators NNReal
open Finset

namespace Emet.Surgery

noncomputable def truncUnit {α : Type*} (f : α → ℝ) (x : α) : ℝ :=
  max 0 (min (f x) 1)

theorem truncUnit_bound {α : Type*} (f : α → ℝ) (x : α) :
    0 ≤ truncUnit f x ∧ truncUnit f x ≤ 1 := by
  constructor
  · exact le_max_left 0 _
  · exact max_le (zero_le_one) (min_le_right _ _)

noncomputable def jumpEnergyFin {n : ℕ} (k : Fin n → Fin n → NNReal)
    (psi : Fin n → Fin n → ℝ) : ℝ :=
  ∑ i, ∑ j, (k i j : ℝ) * (psi i j) ^ 2

end Emet.Surgery
