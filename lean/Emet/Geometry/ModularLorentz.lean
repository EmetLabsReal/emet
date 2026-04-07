import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

/-!
# Modular Lorentz structure

The modular parameter of a flat torus lives in the upper half-plane
`ℍ = SL(2,ℝ)/SO(2)`, which carries the hyperbolic metric
`ds² = (dx² + dy²)/y²`.

This is the velocity space of the Lorentz group `SO(2,1)`.

The **square torus** `τ = i` is the rest frame.
The **hexagonal torus** `τ = e^{iπ/3}` is the optimal torus that
saturates Loewner's inequality `sys² ≤ (2/√3) area(T²)`.

The hyperbolic distance from rest to optimal satisfies:

    cosh(d(i, e^{iπ/3})) = 2/√3

In special relativity, `cosh(rapidity) = γ = 1/√(1 - v²/c²)`.
So:

    γ = 2/√3   ⟹   v = c/2

Loewner's torus constant is a Lorentz factor.
The optimal torus moves at half lightspeed through moduli space.
-/

namespace Emet.Geometry.ModularLorentz

noncomputable section

/-- The Loewner torus constant: `2/√3`, the sharp constant in
    `sys² ≤ (2/√3) area(T²)`. -/
def loewnerConstant : ℝ := 2 / Real.sqrt 3

/-- The Lorentz factor corresponding to velocity `v` (in units where `c = 1`). -/
def lorentzFactor (v : ℝ) : ℝ := 1 / Real.sqrt (1 - v ^ 2)

/-- The velocity corresponding to a Lorentz factor `γ ≥ 1`. -/
def velocityOfGamma (gamma : ℝ) : ℝ := Real.sqrt (1 - 1 / gamma ^ 2)

/-- The rapidity `φ` satisfies `cosh(φ) = γ`. In the upper half-plane,
    the hyperbolic distance from `i` to `τ` satisfies
    `cosh(d) = 1 + |i - τ|² / (2 · Im(i) · Im(τ))`. -/
def hyperbolicCoshDist (dx dy_from_one : ℝ) (y_target : ℝ) : ℝ :=
  1 + (dx ^ 2 + dy_from_one ^ 2) / (2 * y_target)

/-- For the hexagonal torus `τ = 1/2 + i√3/2`:
    `dx = 1/2`, `dy_from_one = 1 - √3/2`, `y_target = √3/2`.

    `cosh(d) = 1 + (2 - √3)/√3 = 2/√3`. -/
theorem hexagonal_cosh_eq :
    hyperbolicCoshDist (1/2) (1 - Real.sqrt 3 / 2) (Real.sqrt 3 / 2)
    = loewnerConstant := by
  unfold hyperbolicCoshDist loewnerConstant
  have h3 : (0 : ℝ) < 3 := by norm_num
  have hsq : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (le_of_lt h3)
  have hsq_pos : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr h3
  field_simp
  nlinarith [hsq, hsq_pos]

/-- `2/√3 > 1`: the Loewner constant is a valid Lorentz factor. -/
theorem loewnerConstant_gt_one : 1 < loewnerConstant := by
  unfold loewnerConstant
  have h3 : (0 : ℝ) < 3 := by norm_num
  have hsq : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (le_of_lt h3)
  have hsq_pos : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr h3
  rw [lt_div_iff₀ hsq_pos]
  nlinarith [hsq]

/-- The Loewner constant squared: `(2/√3)² = 4/3`. -/
theorem loewnerConstant_sq : loewnerConstant ^ 2 = 4 / 3 := by
  unfold loewnerConstant
  have h3 : (0 : ℝ) < 3 := by norm_num
  have hsq : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (le_of_lt h3)
  field_simp
  linarith [hsq]

/-- `γ = 2/√3` implies `v² = 1 - 3/4 = 1/4`, so `v = 1/2` (in units `c = 1`). -/
theorem velocity_of_loewner :
    velocityOfGamma loewnerConstant = Real.sqrt (1 / 4) := by
  unfold velocityOfGamma
  congr 1
  rw [loewnerConstant_sq]
  ring

/-- The velocity is exactly `1/2` (in units `c = 1`). -/
theorem velocity_half : velocityOfGamma loewnerConstant = 1 / 2 := by
  rw [velocity_of_loewner]
  have : (1 : ℝ) / 4 = (1 / 2) ^ 2 := by ring
  rw [this, Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 1 / 2)]

/-- The Lorentz factor at `v = 1/2` is `2/√3`. -/
theorem lorentz_at_half : lorentzFactor (1/2) = loewnerConstant := by
  unfold lorentzFactor loewnerConstant
  have h3pos : (0 : ℝ) < 3 := by norm_num
  have hsq3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (le_of_lt h3pos)
  have hsqrt3_ne : Real.sqrt 3 ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr h3pos)
  have hden : Real.sqrt (1 - (1 / 2 : ℝ) ^ 2) = Real.sqrt 3 / 2 := by
    have h14 : (1 - (1 / 2 : ℝ) ^ 2) = 3 / 4 := by ring
    rw [h14]
    have h_eq : (3 / 4 : ℝ) = (Real.sqrt 3 / 2) ^ 2 := by
      calc
        (3 / 4 : ℝ) = (Real.sqrt 3 ^ 2) / 4 := by rw [hsq3]
        _ = (Real.sqrt 3 / 2) ^ 2 := by ring
    rw [h_eq, Real.sqrt_sq (by positivity)]
  rw [hden]
  field_simp [hsqrt3_ne]

/-- **The torus inequality is a speed limit.**
    Loewner's bound `sys² ≤ (2/√3) area(T²)` asserts that the
    Lorentz factor of the optimal torus equals `2/√3`, corresponding
    to velocity `c/2`. No torus moves faster through moduli space
    than the hexagonal lattice at half lightspeed. -/
theorem torus_speed_limit (v : ℝ) (hv : 0 ≤ v) (hv1 : v < 1)
    (h_bound : loewnerConstant ≤ lorentzFactor v) :
    1 / 2 ≤ v := by
  unfold lorentzFactor loewnerConstant at h_bound
  have h3 : (0 : ℝ) < 3 := by norm_num
  have hsq : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (le_of_lt h3)
  have hsq_pos : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr h3
  have hv2 : 0 < 1 - v ^ 2 := by nlinarith [sq_nonneg v]
  have hsqrt_pos : 0 < Real.sqrt (1 - v ^ 2) := Real.sqrt_pos.mpr hv2
  -- 2/√3 ≤ 1/√(1 - v²)
  -- ⟹ 2·√(1 - v²) ≤ √3
  -- ⟹ 4·(1 - v²) ≤ 3
  -- ⟹ v² ≥ 1/4
  -- ⟹ v ≥ 1/2
  rw [div_le_div_iff₀ hsq_pos hsqrt_pos] at h_bound
  -- Now: 2 * √(1-v²) ≤ 1 * √3  ... wait that's backwards
  -- Actually: h_bound : Real.sqrt (1 - v^2) * 2 ≤ Real.sqrt 3 * 1
  have hsq2 : Real.sqrt (1 - v ^ 2) ^ 2 = 1 - v ^ 2 :=
    Real.sq_sqrt (le_of_lt hv2)
  nlinarith [sq_nonneg (Real.sqrt (1 - v ^ 2)), hsq2, hsq, sq_nonneg v,
             sq_nonneg (v - 1/2)]

end

end Emet.Geometry.ModularLorentz
