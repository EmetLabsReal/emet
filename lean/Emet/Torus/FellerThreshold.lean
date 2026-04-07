import Emet.Torus.EffectivePotential
import Emet.Torus.YangMillsInstance
import Emet.Reduction.Licensed

namespace Emet.Torus

/-- The Feller threshold: at β = 1, the measure s^β ds on the pinched
    torus makes the boundary capacity vanish. Past this threshold the
    Friedrichs extension is the unique self-adjoint extension that confines. -/
def PastFeller (beta : ℝ) : Prop := 1 ≤ beta

/-- Mexican hat centrifugal structure: the effective potential V_eff is
    strictly positive for all s > 0, meaning the boundary repels. -/
def MexicanHatCentrifugal (beta : ℝ) : Prop :=
  ∀ (s : ℝ), 0 < s → 0 < vEff beta s

theorem mexicanHat_of_gt_two {beta : ℝ} (h : 2 < beta) :
    MexicanHatCentrifugal beta :=
  fun _s hs => vEff_pos_of_gt_two h hs

theorem pastFeller_of_gt_two {beta : ℝ} (h : 2 < beta) :
    PastFeller beta := by
  unfold PastFeller
  linarith

/-- The Yang-Mills connection: g² is the weight exponent β.
    Past g² ≥ 1, we are past Feller. -/
theorem ym_pastFeller (p : KogutSusskindParams) (h : 1 ≤ p.gSquared) :
    PastFeller (weightExponent p) := h

/-- Deep coupling in Yang-Mills: g² > 2 gives centrifugal repulsion. -/
theorem ym_mexicanHat (p : KogutSusskindParams) (h : 2 < p.gSquared) :
    MexicanHatCentrifugal (weightExponent p) :=
  mexicanHat_of_gt_two h

end Emet.Torus
