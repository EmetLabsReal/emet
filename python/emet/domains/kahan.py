"""Kahan precision envelope for chi certification.

The Cao-Xie-Li (1994) sharp Kahan theorem gives the eigenvalue error
bound with constant c = 1. The chi regime parameter inherits a
pessimistic linear error envelope under floating-point perturbation:

    eps_chi <= 2 * chi_hat * (eps_lam / lam + eps_gam / gamma)

where eps_lam, eps_gam are the perturbation bounds on lambda and gamma
from machine arithmetic (typically u * value for unit roundoff u).

This module provides the "from below" half of the squeeze: given
computed chi, gamma, lambda, it certifies that chi < 1 survives
machine-precision perturbation.
"""

from __future__ import annotations

# IEEE 754 binary64 unit roundoff
UNIT_ROUNDOFF = 2.0 ** -52


def pessimistic_chi_envelope(
    chi_hat: float,
    eps_lam: float,
    lam: float,
    eps_gam: float,
    gamma: float,
) -> float:
    """Pessimistic linear error envelope for chi.

    Returns eps_chi such that |chi_true - chi_hat| <= eps_chi.

    From the Kahan-CXL sharp bound: perturbation of chi inherits
    linearly from perturbations of lambda and gamma.
    """
    if lam <= 0.0 or gamma <= 0.0:
        return float("inf")
    return 2.0 * chi_hat * (eps_lam / lam + eps_gam / gamma)


def certified_subcritical(
    chi: float,
    gamma: float,
    lam: float,
    *,
    u: float = UNIT_ROUNDOFF,
) -> dict:
    """Certify that chi < 1 survives floating-point perturbation.

    Uses machine unit roundoff to bound perturbations on gamma and lambda,
    then computes the pessimistic chi envelope. If chi + eps_chi < 1,
    the subcritical verdict is certified.

    Returns dict with eps_chi, chi_upper, certified (bool), security_margin.
    """
    if chi >= 1.0:
        return {
            "eps_chi": 0.0,
            "chi_upper": chi,
            "certified": False,
            "security_margin": 0.0,
        }

    eps_lam = u * lam
    eps_gam = u * gamma

    eps_chi = pessimistic_chi_envelope(chi, eps_lam, lam, eps_gam, gamma)
    chi_upper = chi + eps_chi
    security_margin = 1.0 - chi_upper

    return {
        "eps_chi": eps_chi,
        "chi_upper": chi_upper,
        "certified": chi_upper < 1.0,
        "security_margin": security_margin,
    }
