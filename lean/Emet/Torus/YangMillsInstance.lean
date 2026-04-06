import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace Emet.Torus

noncomputable section

structure KogutSusskindParams where
  gSquared : ℝ
  jMax : ℝ
  jCut : ℝ
  gSquared_pos : 0 < gSquared
  jMax_pos : 0 < jMax
  jCut_pos : 0 < jCut
  jCut_le_jMax : jCut ≤ jMax

def casimir (j : ℝ) : ℝ := j * (j + 1)

theorem casimir_nonneg {j : ℝ} (hj : 0 ≤ j) : 0 ≤ casimir j := by
  unfold casimir; nlinarith

theorem casimir_strictly_increasing {j₁ j₂ : ℝ} (_h1 : 0 ≤ j₁) (h2 : j₁ < j₂) :
    casimir j₁ < casimir j₂ := by
  unfold casimir; nlinarith

def barrierAtCutoff (p : KogutSusskindParams) : ℝ :=
  (p.gSquared / 2) * casimir p.jCut

theorem barrierAtCutoff_pos (p : KogutSusskindParams) : 0 < barrierAtCutoff p := by
  unfold barrierAtCutoff casimir
  apply mul_pos
  · exact div_pos p.gSquared_pos two_pos
  · have := p.jCut_pos; nlinarith

def tunnelingAmplitude (p : KogutSusskindParams) : ℝ :=
  1 / p.gSquared

theorem tunnelingAmplitude_pos (p : KogutSusskindParams) : 0 < tunnelingAmplitude p := by
  unfold tunnelingAmplitude
  exact div_pos one_pos p.gSquared_pos

def weightExponent (p : KogutSusskindParams) : ℝ := p.gSquared

theorem pastFeller_of_gSquared_ge_one (p : KogutSusskindParams)
    (h : 1 ≤ p.gSquared) : 1 ≤ weightExponent p := h

end

end Emet.Torus
