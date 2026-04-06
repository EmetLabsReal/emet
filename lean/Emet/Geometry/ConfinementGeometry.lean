import Emet.Reduction.MexicanHatForced
import Emet.Reduction.TwoPointDecay
import Emet.Torus.YangMillsInstance
import Emet.Surgery.Irreversibility
import Emet.Geometry.ModularLorentz

/-!
# Confinement geometry

This file packages the informal single coherent object behind the certificate stack.

**Cone (open, β < 1).** Cross-section ∼ s^β shrinks slowly; Brownian motion for the
Dirichlet form can reach the tip s = 0; the radial direction is accessible; no UV
barrier in that phase.

**Pinch (β = 1, Feller threshold).** Cross-sectional measure degenerates; capacity
vanishes; access to s = 0 flips from possible to forbidden. This is not bolted on:
it is the geometric consequence of the measure. In a transverse slice, the singular
measure becomes an infinite barrier in the effective potential `vEff`; the Friedrichs
extension selects the unique self-adjoint realization respecting unreachability.

**Torus (confined, β ≥ 1).** The radial direction is trapped in a well whose floor
is the Friedrichs domain boundary; the compact loop direction (gauge quotient) is
the fiber. Morally: cone × circle with the cone sealed past Feller.

**Spectral consequences (geometric).** The radial operator in the well has a ladder
E₀ < E₁ < …; the mass gap is spectral; curvature at the bottom sets the scale.
Past the pinch, `χ < 1` (`IsLicensed`) certifies that the valley sector is the
faithful Schur-reduced description (`schur_contraction`, `schur_eigenvalue_pos`).
Deep coupling β > 2 forces Mexican-hat centrifugal structure (`mexicanHat_of_gt_two`).

**Irreversibility.** Once a positive spectral gap is present, `contraction_factor_lt_one`
forces strict exponential contractivity; `gap_or_trivial` shows the only way to lose
the gap is to trivialize the generator (`Emet.Surgery`).

**Yang–Mills.** In the Kogut–Susskind instance, β is identified with g²
(`weightExponent`). Running coupling is the clock that moves along radial scale;
the Feller line is where the horn pinches shut *in this model*.

The Lean proofs live in `Emet.Torus`, `Emet.Reduction`, and `Emet.Surgery`. This module
only **names** the geometry and **bundles** hypotheses so the theorem statements match
the narrative.
-/

namespace Emet.Geometry

noncomputable section

open Emet.Reduction Emet.Torus Emet.Surgery Emet.Geometry

/-- Logarithmic or affine scale parameter along the cone (RG time, `log μ`, etc.). -/
abbrev Scale := ℝ

/-- Causal structure: the modular parameter of the confined torus lives in the
    upper half-plane `ℍ = SL(2,ℝ)/SO(2)` with hyperbolic metric.  This IS the
    velocity space of the Lorentz group `SO(2,1)`.

    The square torus `τ = i` is the rest frame.  The hexagonal torus
    `τ = e^{iπ/3}` is the optimal torus (saturates Loewner's inequality).
    The hyperbolic distance between them satisfies `cosh(d) = 2/√3`,
    which is a Lorentz factor corresponding to `v = c/2`.

    See `Emet.Geometry.ModularLorentz` for the full proof chain. -/
structure CausalStructure where
  /-- Lorentz factor `γ ≥ 1` at this point in moduli space. -/
  gamma_lorentz : ℝ
  /-- The Lorentz factor is at least 1 (at rest or moving). -/
  h_gamma_ge_one : 1 ≤ gamma_lorentz

/-- Cross-sectional measure ∼ s^β ds near the pinch; β is the weight exponent.

    In the Yang–Mills instance, β agrees with g² (`weightExponent`). -/
structure RadialMeasure where
  beta : ℝ

/-- Running coupling: g² as a function of scale. Anchoring `gSquaredAt s₀` to a
    concrete truncation fixes the identification with `RadialMeasure.beta`. -/
structure RunningCoupling where
  gSquaredAt : Scale → ℝ

/-- **Confinement geometry**: causal cone data, radial measure, RG coupling, and a
    fully certified **licensed torus reduction** (`χ < 1`, past Feller, β > 2).

    Fields `measure` and `certified.beta` agree: one exponent controls both the
    transverse measure class and the torus/Yang–Mills coupling in the narrative. -/
structure ConfinementGeometry where
  causal : CausalStructure
  measure : RadialMeasure
  coupling : RunningCoupling
  /-- Schur-licensed parameters with torus/Feller/Mexican-hat hypotheses. -/
  certified : LicensedTorusReduction
  /-- Narrative alignment: radial measure exponent = reduction coupling β. -/
  h_beta : measure.beta = certified.beta

/-! ## Projections matching existing theorems -/

theorem isLicensed (G : ConfinementGeometry) :
    IsLicensed G.certified.params :=
  G.certified.h_licensed

theorem pastFellerMeasure (G : ConfinementGeometry) :
    PastFeller G.measure.beta := by
  rw [G.h_beta]
  exact G.certified.h_past_feller

theorem pastFellerCertified (G : ConfinementGeometry) :
    PastFeller G.certified.beta :=
  G.certified.h_past_feller

theorem deepCoupling (G : ConfinementGeometry) :
    2 < G.certified.beta :=
  G.certified.h_deep

/-- `licensed_torus_forces_mexican_hat` specialized to the bundle. -/
theorem mexicanHatCentrifugal (G : ConfinementGeometry) :
    MexicanHatCentrifugal G.certified.beta :=
  licensed_torus_forces_mexican_hat G.certified

/-- Full valley + centrifugal package when Schur data are supplied. -/
theorem fullTorusStructure (G : ConfinementGeometry)
    {pp_eig correction : ℝ} (hpp : 0 < pp_eig)
    (h_bound : correction ≤ chi G.certified.params * pp_eig)
    (h_corr_nn : 0 ≤ correction) :
    MexicanHatCentrifugal G.certified.beta ∧ 0 < pp_eig - correction :=
  licensed_torus_full_structure G.certified hpp h_bound h_corr_nn

/-- Positive mass gap ⇒ strict two-point decay (`Emet.Reduction.two_point_decay`). -/
theorem twoPointDecay {mass_gap t amplitude : ℝ}
    (h_gap : 0 < mass_gap) (ht : 0 < t) (h_amp : 0 < amplitude) :
    amplitude * Real.exp (-(mass_gap * t)) < amplitude :=
  two_point_decay h_gap ht h_amp

/-- Heat-kernel / Markov contraction once a spectral gap is given (`Surgery`). -/
theorem contractionStrict {gap t : ℝ} (hg : 0 < gap) (ht : 0 < t) :
    Real.exp (-(gap * t)) < 1 :=
  contraction_factor_lt_one hg ht

/-- **Confinement rate law.** velocity · spectral_gap = c · coupling.
    v = c/2, gap = 2λ, so v · gap = c · λ.
    The product of velocity and bureaucracy is constant. -/
theorem confinement_rate_law (lam : ℝ) (_hlam : 0 < lam) :
    (1 / 2 : ℝ) * spectralGap lam = lam := by
  unfold spectralGap; ring

/-! ## Constructors and identification -/

/-- Rest-frame causal structure (`γ = 1`, square torus). -/
def defaultCausal : CausalStructure :=
  ⟨1, le_refl 1⟩

/-- Optimal causal structure (`γ = 2/√3`, hexagonal torus, `v = c/2`). -/
noncomputable def hexagonalCausal : CausalStructure :=
  ⟨ModularLorentz.loewnerConstant, le_of_lt ModularLorentz.loewnerConstant_gt_one⟩

/-- Constant running coupling (fixed truncation scale). -/
def RunningCoupling.const (g2 : ℝ) : RunningCoupling :=
  ⟨fun _ => g2⟩

/-- Build `RadialMeasure` from a certified reduction β. -/
def RadialMeasure.ofBeta (β : ℝ) : RadialMeasure :=
  ⟨β⟩

/-- Package a `LicensedTorusReduction` with geometric scaffolding. -/
def ConfinementGeometry.mkAligned
    (causal : CausalStructure) (coupling : RunningCoupling)
    (cert : LicensedTorusReduction) : ConfinementGeometry where
  causal := causal
  measure := RadialMeasure.ofBeta cert.beta
  coupling := coupling
  certified := cert
  h_beta := rfl

/-! ### Yang–Mills (Kogut–Susskind)

The **identification theorem** is proof-theoretically thin: we reuse
`LicensedTorusReduction` with `β = g²` and hypotheses supplied by the numeric
pipeline (Rust/Python). Lean already proves `ym_pastFeller` and links `weightExponent`
to `gSquared`.
-/

/-- From KS data + Schur license + coupling window: the confinement geometry bundle.

    **Assertion side:** `IsLicensed params` is not discharged from `p` alone; it is
    the matrix certificate checked externally for each truncation. -/
def yangMillsIdentification
    (p : KogutSusskindParams)
    (params : ReductionParams)
    (h_lic : IsLicensed params)
    (h_g1 : 1 ≤ p.gSquared)
    (h_g2 : 2 < p.gSquared)
    (causal : CausalStructure := defaultCausal)
    (coupling : RunningCoupling := RunningCoupling.const p.gSquared) :
    ConfinementGeometry :=
  ConfinementGeometry.mkAligned causal coupling
    {
      params := params
      beta := p.gSquared
      h_licensed := h_lic
      h_past_feller := ym_pastFeller p h_g1
      h_deep := h_g2
    }

theorem yangMills_mexican_hat
    (p : KogutSusskindParams)
    (params : ReductionParams)
    (h_lic : IsLicensed params)
    (h_g1 : 1 ≤ p.gSquared)
    (h_g2 : 2 < p.gSquared)
    (causal : CausalStructure := defaultCausal)
    (coupling : RunningCoupling := RunningCoupling.const p.gSquared) :
    MexicanHatCentrifugal p.gSquared := by
  simpa [yangMillsIdentification, ConfinementGeometry.mkAligned] using
    mexicanHatCentrifugal (yangMillsIdentification p params h_lic h_g1 h_g2 causal coupling)

@[simp] theorem RunningCoupling.const_apply (g2 s : ℝ) :
    (RunningCoupling.const g2).gSquaredAt s = g2 :=
  rfl

end

end Emet.Geometry
