"""Threaded workers for sweep and certify operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SweepRow:
    """One row of sweep results."""
    index: int
    param_name: str
    param_value: float
    regime: str
    licensed: bool
    chi: float | None
    gamma: float
    lambda_: float
    kahan_certified: bool


def run_sweep(
    domain_key: str,
    base_params: dict[str, Any],
    sweep_param: str,
    sweep_values: list[float],
    on_row: Any = None,
    on_progress: Any = None,
) -> list[SweepRow]:
    """Execute a parameter sweep, calling on_row for each result.

    Args:
        domain_key: Registry key for the domain adapter.
        base_params: Base parameter dict (sweep_param will be overridden).
        sweep_param: Name of the parameter to sweep.
        sweep_values: Values to sweep over.
        on_row: Callback(SweepRow) called for each completed point.
        on_progress: Callback(int, int) called with (current, total).

    Returns:
        List of SweepRow results.
    """
    import emet
    from emet.tui.registry import get_domain

    desc = get_domain(domain_key)
    results: list[SweepRow] = []
    total = len(sweep_values)

    for i, val in enumerate(sweep_values):
        params = dict(base_params)
        params[sweep_param] = val

        try:
            H, retained, omitted = desc.build_fn(**params)
            report = emet.decide_dense_matrix(H, retained=retained, omitted=omitted)
            adv = report.get("advanced_metrics", {})
            row = SweepRow(
                index=i,
                param_name=sweep_param,
                param_value=val,
                regime=report.get("regime", "unknown"),
                licensed=report.get("valid", False),
                chi=adv.get("chi"),
                gamma=adv.get("gamma", 0.0),
                lambda_=adv.get("lambda", 0.0),
                kahan_certified=adv.get("kahan", {}).get("certified", False),
            )
        except Exception:
            row = SweepRow(
                index=i,
                param_name=sweep_param,
                param_value=val,
                regime="error",
                licensed=False,
                chi=None,
                gamma=0.0,
                lambda_=0.0,
                kahan_certified=False,
            )

        results.append(row)
        if on_row is not None:
            on_row(row)
        if on_progress is not None:
            on_progress(i + 1, total)

    return results


def run_certify(
    domain_key: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build, decide, and certify a domain configuration.

    Returns a dict with 'report', 'certificate' (EmetCertificate), and 'H_eff'.
    """
    import emet
    from emet.certificate import certify
    from emet.tui.registry import get_domain

    desc = get_domain(domain_key)
    H, retained, omitted = desc.build_fn(**params)
    report = emet.decide_dense_matrix(H, retained=retained, omitted=omitted)

    beta = None
    if desc.beta_param and desc.beta_param in params:
        beta = float(params[desc.beta_param])

    cert = certify(
        H, retained, omitted, report,
        domain=domain_key, params=params, beta=beta,
    )

    # Compute H_eff = H_PP - H_PQ @ inv(H_QQ) @ H_QP
    H_eff = None
    try:
        P = sorted(retained)
        Q = sorted(omitted)
        H_PP = H[np.ix_(P, P)]
        H_PQ = H[np.ix_(P, Q)]
        H_QQ = H[np.ix_(Q, Q)]
        H_QP = H[np.ix_(Q, P)]
        H_eff = H_PP - H_PQ @ np.linalg.solve(H_QQ, H_QP)
    except Exception:
        pass

    return {
        "report": report,
        "certificate": cert,
        "H_eff": H_eff,
        "H": H,
        "retained": retained,
        "omitted": omitted,
    }
