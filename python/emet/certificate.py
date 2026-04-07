"""Emet certificate: cryptographic seal for licensed reductions.

A certificate covers the full pipeline: input matrix hash, partition,
chi/gamma/lambda, Kahan margin, and reduced matrix hash. The seal is
a SHA-256 digest of the canonical JSON encoding of all fields. Anyone
with the same emet version can reproduce the seal.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "emet-cert-v2"


@dataclass(frozen=True)
class EmetCertificate:
    schema: str
    timestamp: str
    emet_version: str
    domain: str | None
    params: dict[str, Any] | None
    input_hash: str
    partition_hash: str
    matrix_dimension: int
    retained_dim: int
    omitted_dim: int
    chi: float | None
    gamma: float
    lambda_: float
    beta: float | None
    regime: str
    licensed: bool
    determinacy: bool
    kahan_certified: bool
    kahan_margin: float | None
    reduced_matrix_hash: str | None
    seal: str


def _canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, finite floats."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _hash_matrix(matrix: np.ndarray) -> str:
    """SHA-256 of matrix as canonical list-of-lists JSON."""
    rows = [[round(float(v), 15) for v in row] for row in matrix]
    return _sha256(_canonical_json(rows))


def _hash_partition(retained: Sequence[int], omitted: Sequence[int]) -> str:
    return _sha256(_canonical_json({"retained": sorted(retained), "omitted": sorted(omitted)}))


def certify(
    H: np.ndarray,
    retained: Sequence[int],
    omitted: Sequence[int],
    report: dict[str, Any],
    *,
    domain: str | None = None,
    params: dict[str, Any] | None = None,
    beta: float | None = None,
) -> EmetCertificate:
    """Produce a certificate from a decision report.

    The report must come from emet.decide_dense_matrix(H, retained, omitted).
    When beta (weight exponent) is provided, the certificate records the
    full trichotomy (A/B/C) and determinacy classification.
    """
    import emet
    from emet.domains.kahan import certified_subcritical
    from emet.portrait import PhasePoint

    metrics = report.get("advanced_metrics") or {}
    chi = metrics.get("chi")
    gamma = metrics.get("gamma", 0.0)
    lam = metrics.get("lambda", 0.0)
    regime = report.get("regime", "unknown")
    licensed = report.get("valid", False)

    pt = PhasePoint.from_report(report, beta=beta)
    regime_str = pt.regime.value
    determinacy = pt.determinacy

    kahan_certified = False
    kahan_margin = None
    if licensed and chi is not None and chi < 1.0:
        k = certified_subcritical(chi, gamma, lam)
        kahan_certified = k["certified"]
        kahan_margin = k["security_margin"] if k["certified"] else None

    rm = report.get("reduced_matrix")
    reduced_matrix_hash = None
    if rm is not None and isinstance(rm, dict) and rm.get("data"):
        reduced_matrix_hash = _hash_matrix(np.array(rm["data"]))

    input_hash = _hash_matrix(np.asarray(H, dtype=float))
    partition_hash = _hash_partition(retained, omitted)

    pre_seal = {
        "schema": SCHEMA_VERSION,
        "emet_version": emet.__version__,
        "domain": domain,
        "input_hash": input_hash,
        "partition_hash": partition_hash,
        "matrix_dimension": int(np.asarray(H).shape[0]),
        "retained_dim": len(list(retained)),
        "omitted_dim": len(list(omitted)),
        "chi": chi,
        "gamma": gamma,
        "lambda": lam,
        "beta": beta,
        "regime": regime_str,
        "licensed": licensed,
        "determinacy": determinacy,
        "kahan_certified": kahan_certified,
        "kahan_margin": kahan_margin,
        "reduced_matrix_hash": reduced_matrix_hash,
    }
    seal = _sha256(_canonical_json(pre_seal))

    return EmetCertificate(
        schema=SCHEMA_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        emet_version=emet.__version__,
        domain=domain,
        params=params,
        input_hash=input_hash,
        partition_hash=partition_hash,
        matrix_dimension=int(np.asarray(H).shape[0]),
        retained_dim=len(list(retained)),
        omitted_dim=len(list(omitted)),
        chi=chi,
        gamma=gamma,
        lambda_=lam,
        beta=beta,
        regime=regime_str,
        licensed=licensed,
        determinacy=determinacy,
        kahan_certified=kahan_certified,
        kahan_margin=kahan_margin,
        reduced_matrix_hash=reduced_matrix_hash,
        seal=seal,
    )


def verify(
    cert: EmetCertificate,
    H: np.ndarray,
    retained: Sequence[int],
    omitted: Sequence[int],
) -> bool:
    """Re-derive the seal from a matrix and partition and check it matches."""
    import emet

    report = emet.decide_dense_matrix(H, retained=list(retained), omitted=list(omitted))
    new_cert = certify(
        H, retained, omitted, report,
        domain=cert.domain, params=cert.params, beta=cert.beta,
    )
    return new_cert.seal == cert.seal


def to_json(cert: EmetCertificate, *, pretty: bool = False) -> str:
    """Serialize a certificate to JSON."""
    d = asdict(cert)
    d["lambda"] = d.pop("lambda_")
    if pretty:
        return json.dumps(d, indent=2, sort_keys=True)
    return json.dumps(d, sort_keys=True)


def from_json(text: str) -> EmetCertificate:
    """Deserialize a certificate from JSON."""
    d = json.loads(text)
    d["lambda_"] = d.pop("lambda")
    return EmetCertificate(**d)
