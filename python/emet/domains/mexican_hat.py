"""Mexican hat theorem: licensed reductions force potential structure.

The effective potential on a pinched torus is V_eff(beta, s) = beta*(beta-2)/(4s^2).
For beta > 2 this is strictly positive (repulsive centrifugal barrier).
Combined with a licensed reduction (chi < 1), this gives the full Mexican hat:
valley confinement via the Schur complement AND centrifugal repulsion at the boundary.

The theorem is certified in Lean 4 (lean/Emet/Reduction/MexicanHatForced.lean).
"""

from __future__ import annotations

import numpy as np

import emet
from emet.domains.torus import effective_potential


def verify_mexican_hat_centrifugal(beta: float,
                                   s_min: float = 1e-2,
                                   s_max: float = 1.0,
                                   n_samples: int = 100) -> bool:
    """Check V_eff > 0 for all sampled s in (s_min, s_max)."""
    s = np.linspace(s_min, s_max, n_samples)
    v = effective_potential(beta, s)
    return bool(np.all(v > 0))


def licensed_implies_mexican_hat(
    H: np.ndarray,
    retained: list[int],
    omitted: list[int],
    beta: float,
) -> dict:
    """End-to-end verification: chi < 1 AND V_eff > 0.

    Returns a dict with the licensing verdict, Mexican hat check,
    and whether both hold simultaneously.
    """
    report = emet.decide_dense_matrix(H, retained=retained, omitted=omitted)
    m = report["advanced_metrics"]
    chi = m["chi"]
    licensed = chi is not None and chi < 1
    centrifugal = verify_mexican_hat_centrifugal(beta)

    return {
        "licensed": licensed,
        "chi": chi,
        "beta": beta,
        "centrifugal_mexican_hat": centrifugal,
        "both": licensed and centrifugal,
        "regime": report["regime"],
    }
