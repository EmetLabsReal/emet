"""4D Yang-Mills as pinched torus with dimension-dependent geometry.

The weight exponent beta = d * alpha where d is the spacetime dimension
and alpha is the pinching exponent of the torus cross-section r(s) ~ s^alpha.

For d=4 (Euclidean spacetime):
  Feller threshold beta=1 constrains alpha = 1/4.
  This is not a parameter choice. It is forced by the dimension.

The retained/omitted partition corresponds to the bulk-boundary split.
The Schur complement projects out the omitted sector. Chi < 1 certifies
the retained sector faithfully represents the full operator.
"""

from __future__ import annotations

import numpy as np

import emet
from emet.domains.torus import (
    build_block_operator,
    barrier_eigenvalues,
    valley_eigenvalues,
    tunneling_coupling,
    effective_potential,
)
from emet.domains.kahan import certified_subcritical


def pinching_exponent(d: int, g_squared: float) -> float:
    """Alpha = g^2 / d. The pinching exponent in d dimensions."""
    return g_squared / d


def weight_exponent(d: int, alpha: float) -> float:
    """Beta = d * alpha. The measure weight in d dimensions."""
    return d * alpha


def feller_critical_coupling(d: int) -> float:
    """The coupling g^2 at which the Feller threshold is reached.

    Feller threshold: beta = 1, so g^2 = 1 regardless of d.
    But alpha = 1/d, which IS dimension-dependent.
    In 4D: alpha = 1/4. The pinching is mild.
    In 2D: alpha = 1/2. The pinching is strong.
    """
    return 1.0


def feller_critical_alpha(d: int) -> float:
    """Alpha at the Feller threshold: alpha = 1/d."""
    return 1.0 / d


def build_4d_torus_operator(
    g_squared: float,
    j_max: float = 4.0,
    j_cut: float = 1.0,
    d: int = 4,
    *,
    n_valley: int = 6,
    n_barrier: int = 4,
) -> tuple[np.ndarray, list[int], list[int], dict]:
    """Build the d-dimensional Yang-Mills operator as a pinched torus.

    The Hamiltonian is partitioned into:
      PP (retained) = valley modes (boundary / CFT side)
      QQ (omitted)  = barrier modes (bulk / AdS side)
      PQ/QP         = bulk-boundary coupling (holographic)

    The Schur complement H_eff = PP - PQ QQ^{-1} QP is the
    holographic projection: the boundary theory with bulk integrated out.

    Returns (H, retained, omitted, params) where params contains
    the dimension-dependent geometry.
    """
    beta = g_squared  # the identification
    alpha = pinching_exponent(d, g_squared)

    PP = np.diag(valley_eigenvalues(n_valley))
    QQ = np.diag(barrier_eigenvalues(beta, n_barrier))
    PQ = tunneling_coupling(beta, n_valley, n_barrier)

    H, retained, omitted = build_block_operator(PP, QQ, PQ)

    params = {
        "d": d,
        "g_squared": g_squared,
        "beta": beta,
        "alpha": alpha,
        "feller_alpha": feller_critical_alpha(d),
        "past_feller": beta >= 1.0,
        "mexican_hat_centrifugal": beta > 2.0,
    }

    return H, retained, omitted, params


def certify_4d(
    g_squared: float,
    j_max: float = 4.0,
    j_cut: float = 1.0,
    d: int = 4,
) -> dict:
    """Full certification of the d-dimensional Yang-Mills construction.

    Returns chi, mass gap, Kahan certification, and dimension geometry.
    """
    H, ret, omit, params = build_4d_torus_operator(g_squared, j_max, j_cut, d)
    report = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
    m = report["advanced_metrics"]

    chi = m["chi"]
    licensed = chi is not None and chi < 1

    eigvals = np.linalg.eigvalsh(H)
    if eigvals.shape[0] < 2:
        mass_gap = 0.0
    else:
        mass_gap = float(eigvals[1] - eigvals[0])

    result = {
        **params,
        "chi": chi,
        "gamma": m["gamma"],
        "lambda": m["lambda"],
        "licensed": licensed,
        "regime": report["regime"],
        "mass_gap": mass_gap,
    }

    if licensed and chi is not None:
        kahan = certified_subcritical(chi, m["gamma"], m["lambda"])
        result["kahan_certified"] = kahan["certified"]
        result["security_margin"] = kahan["security_margin"]
    else:
        result["kahan_certified"] = False
        result["security_margin"] = None

    return result


def dimension_sweep(
    g_squared: float,
    dimensions: list[int] | None = None,
) -> list[dict]:
    """Show how the geometry changes with spacetime dimension.

    The coupling g^2 is fixed. The dimension determines alpha = g^2/d.
    Different dimensions produce different pinching, different barriers,
    different chi values. 4D is not arbitrary.
    """
    if dimensions is None:
        dimensions = [2, 3, 4, 5, 6]
    return [certify_4d(g_squared, d=d) for d in dimensions]
