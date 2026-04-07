"""SU(2) Kogut-Susskind Hamiltonian as a pinched torus instance.

The Hamiltonian for pure SU(2) gauge theory on a single plaquette in the
representation basis is:

    H = (g^2/2) * C_2(j) - (1/g^2) * Re Tr U_plaquette

where C_2(j) = j(j+1) is the quadratic Casimir and the plaquette operator
connects adjacent representations via Clebsch-Gordan: j <-> j +/- 1/2.

This is a pinched torus instance. The identification:

    j (representation label)      <->  radial coordinate s
    Casimir j(j+1)                <->  centrifugal barrier (QQ block)
    plaquette coupling 1/g^2      <->  WKB tunneling amplitude (PQ/QP)
    g^2 (coupling constant)       <->  weight exponent beta

In the strong coupling regime (large g^2 = large beta), the Casimir
dominates the omitted block (gamma large), tunneling decays as 1/g^2
(lambda small), and chi < 1: confinement is certified. Past the Feller
threshold, the Friedrichs extension is the unique self-adjoint extension.
The mass gap is certified by the Schur complement.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import emet
from emet.domains.torus import build_block_operator


def casimir(j: float) -> float:
    """Quadratic Casimir C_2(j) = j(j+1) for SU(2) representation j."""
    return j * (j + 1.0)


def yang_mills_as_torus_params(
    g_squared: float,
    j_max: float = 3.0,
    j_cut: float = 1.0,
) -> dict:
    """Map Kogut-Susskind parameters to pinched torus vocabulary.

    Returns a dictionary documenting the identification between the
    Yang-Mills representation-basis Hamiltonian and the pinched torus
    operator with centrifugal barrier and Feller threshold.
    """
    j_retained = [k / 2.0 for k in range(int(2 * j_cut) + 1)]
    j_omitted = [k / 2.0 for k in range(int(2 * j_cut) + 1, int(2 * j_max) + 1)]

    gamma_casimir = min(casimir(j) for j in j_omitted) if j_omitted else float("inf")

    return {
        "beta_equivalent": g_squared,
        "n_valley": len(j_retained),
        "n_barrier": len(j_omitted),
        "barrier_mechanism": "Casimir j(j+1)",
        "tunneling_mechanism": "plaquette coupling 1/g^2",
        "feller_threshold": "g^2 = 1 (beta = 1)",
        "valley_eigenvalues": [(g_squared / 2.0) * casimir(j) for j in j_retained],
        "barrier_eigenvalues": [(g_squared / 2.0) * casimir(j) for j in j_omitted],
        "tunneling_amplitude": 1.0 / g_squared,
        "gamma_floor_casimir": (g_squared / 2.0) * gamma_casimir,
    }


def _build_valley_block(g_squared: float, j_retained: list[float]) -> np.ndarray:
    """PP block: Casimir diagonal + plaquette tridiagonal within retained sector."""
    n = len(j_retained)
    PP = np.zeros((n, n))
    for k, j in enumerate(j_retained):
        PP[k, k] = (g_squared / 2.0) * casimir(j)
    coupling = -1.0 / g_squared
    for k in range(n - 1):
        PP[k, k + 1] = coupling
        PP[k + 1, k] = coupling
    return PP


def _build_barrier_block(g_squared: float, j_omitted: list[float]) -> np.ndarray:
    """QQ block: Casimir diagonal + plaquette tridiagonal within omitted sector."""
    n = len(j_omitted)
    QQ = np.zeros((n, n))
    for k, j in enumerate(j_omitted):
        QQ[k, k] = (g_squared / 2.0) * casimir(j)
    coupling = -1.0 / g_squared
    for k in range(n - 1):
        QQ[k, k + 1] = coupling
        QQ[k + 1, k] = coupling
    return QQ


def _build_tunneling_block(
    g_squared: float, n_retained: int, n_omitted: int,
) -> np.ndarray:
    """PQ block: single plaquette coupling at the partition boundary.

    Only the last retained representation couples to the first omitted
    representation, with amplitude -1/g^2. This is the tunneling through
    the Casimir barrier — the WKB amplitude in torus language.
    """
    PQ = np.zeros((n_retained, n_omitted))
    PQ[n_retained - 1, 0] = -1.0 / g_squared
    return PQ


def build_single_plaquette(
    g_squared: float,
    j_max: float = 3.0,
) -> tuple[np.ndarray, list[int], list[float]]:
    """Kogut-Susskind Hamiltonian for SU(2) on a single plaquette.

    Truncates the representation space at j_max. Returns (H, indices, j_values)
    where H is the (2*j_max + 1) x (2*j_max + 1) tridiagonal matrix
    and j_values lists the representation labels.

    H[k,k]   = (g^2/2) * j_k(j_k + 1)   electric energy
    H[k,k+1] = -1/g^2                     magnetic coupling
    """
    j_values = [k / 2.0 for k in range(int(2 * j_max) + 1)]
    n = len(j_values)

    H = np.zeros((n, n))

    # Electric (diagonal): Casimir
    for k, j in enumerate(j_values):
        H[k, k] = (g_squared / 2.0) * casimir(j)

    # Magnetic (off-diagonal): plaquette coupling
    coupling = -1.0 / g_squared
    for k in range(n - 1):
        H[k, k + 1] = coupling
        H[k + 1, k] = coupling

    return H, list(range(n)), j_values


def build_plaquette_blocks(
    g_squared: float,
    j_max: float = 3.0,
    j_cut: float = 1.0,
) -> tuple[np.ndarray, list[int], list[int], list[float]]:
    """Build the KS Hamiltonian via torus block assembly.

    Constructs PP (valley/retained), QQ (barrier/omitted), PQ (tunneling)
    blocks from Casimir eigenvalues and plaquette coupling, then assembles
    via the shared block operator builder.

    Returns (H, retained, omitted, j_values).
    """
    j_values = [k / 2.0 for k in range(int(2 * j_max) + 1)]
    j_retained = [j for j in j_values if j <= j_cut + 1e-12]
    j_omitted = [j for j in j_values if j > j_cut + 1e-12]

    PP = _build_valley_block(g_squared, j_retained)
    QQ = _build_barrier_block(g_squared, j_omitted)
    PQ = _build_tunneling_block(g_squared, len(j_retained), len(j_omitted))

    H, retained, omitted = build_block_operator(PP, QQ, PQ)
    return H, retained, omitted, j_values


def partition_by_representation(
    j_values: list[float],
    j_cut: float = 1.0,
) -> tuple[list[int], list[int]]:
    """Partition into retained (j <= j_cut) and omitted (j > j_cut).

    The retained sector contains the low-energy representations.
    The omitted sector contains the high-energy representations
    whose Casimir energy provides the barrier.
    """
    retained = [k for k, j in enumerate(j_values) if j <= j_cut + 1e-12]
    omitted = [k for k, j in enumerate(j_values) if j > j_cut + 1e-12]
    return retained, omitted


def mass_gap(H: np.ndarray) -> tuple[float, float, float]:
    """Compute the mass gap of Hamiltonian H.

    Returns (E_0, E_1, gap) where E_0 is the ground state energy,
    E_1 is the first excited state energy, and gap = E_1 - E_0.
    """
    eigvals = np.sort(np.linalg.eigvalsh(H))
    E_0 = float(eigvals[0])
    E_1 = float(eigvals[1]) if len(eigvals) > 1 else E_0
    return E_0, E_1, E_1 - E_0


def certify_yang_mills(
    g_squared: float,
    j_cut: float = 1.0,
    j_max: float = 3.0,
    *,
    store: Any = None,
) -> dict:
    """Certify a Yang-Mills reduction via the radial Dirichlet form.

    The identification beta = g^2 bridges the Kogut-Susskind Hamiltonian
    to the radial Dirichlet form on the pinched torus.  The trichotomy
    applies to the form, not to the tridiagonal matrix directly:

        g^2 < 1   ->  beta < 1  ->  Regime A  (accessible boundary)
        g^2 >= 1  ->  beta >= 1 ->  Regime B/C (unique Friedrichs extension)
        g^2 > 2   ->  beta > 2  ->  Regime C  (Mexican hat, spectral gap)

    The numeric chi is computed from the materialized KS matrix to seal
    the certificate.  But the regime classification comes from beta alone.

    Parameters
    ----------
    g_squared : float
        Coupling constant.  Identified with weight exponent beta.
    j_cut : float
        Partition boundary in representation space.
    j_max : float
        Truncation of the representation ladder.
    store : CertificateStore or None
        If provided, the certificate is appended to the store.

    Returns
    -------
    dict with keys: beta, regime, chi, gamma, lambda_, determinacy,
    mass_gap, j_cut, kahan_certified, certificate (EmetCertificate).
    """
    from emet.certificate import certify
    from emet.portrait import PhasePoint

    beta = g_squared

    H, retained, omitted, j_values = build_plaquette_blocks(g_squared, j_max, j_cut)
    report = emet.decide_dense_matrix(H, retained=retained, omitted=omitted)
    m = report["advanced_metrics"]
    chi = m["chi"]

    pt = PhasePoint.from_report(report, beta=beta)

    E_0, E_1, gap = mass_gap(H)

    cert = certify(
        H, retained, omitted, report,
        domain="yang_mills", beta=beta,
        params={"g_squared": g_squared, "j_cut": j_cut, "j_max": j_max},
    )

    if store is not None:
        store.append(
            cert, domain="yang_mills",
            params={"g_squared": g_squared, "j_cut": j_cut, "j_max": j_max},
            beta=beta,
        )

    result = {
        "beta": beta,
        "regime": pt.regime.value,
        "chi": chi,
        "gamma": m["gamma"],
        "lambda_": m["lambda"],
        "determinacy": pt.determinacy,
        "licensed": pt.licensed,
        "mass_gap": gap,
        "j_cut": j_cut,
        "j_max": j_max,
        "kahan_certified": cert.kahan_certified,
        "seal": cert.seal,
        "certificate": cert,
        "lean_module": "MexicanHatForced.lean" if beta > 2.0 else (
            "FellerThreshold.lean" if beta >= 1.0 else None
        ),
    }
    return result


def sweep_coupling(
    g_squared_values: Sequence[float],
    j_max: float = 3.0,
    j_cut: float = 1.0,
) -> list[dict]:
    """Sweep the coupling constant g^2 and certify each partition.

    At each g^2: build the Hamiltonian via torus block assembly, compute
    chi, and extract the mass gap.

    Strong coupling (large g^2): electric dominates, chi < 1, confined.
    Weak coupling (small g^2): magnetic dominates, chi >= 1, deconfined.
    """
    results = []

    for g2 in g_squared_values:
        H, retained, omitted, j_values = build_plaquette_blocks(g2, j_max, j_cut)

        if not omitted:
            results.append({
                "g_squared": g2,
                "lambda": 0.0,
                "gamma": float("inf"),
                "chi": 0.0,
                "valid": True,
                "regime": "trivial",
                "E_0": 0.0,
                "E_1": 0.0,
                "mass_gap": 0.0,
            })
            continue

        report = emet.decide_dense_matrix(H, retained=retained, omitted=omitted)
        m = report["advanced_metrics"]
        chi = m["chi"] if m["chi"] is not None else float("inf")

        E_0, E_1, gap = mass_gap(H)

        results.append({
            "g_squared": g2,
            "lambda": m["lambda"],
            "gamma": m["gamma"],
            "chi": chi,
            "valid": report["valid"],
            "regime": report["regime"],
            "E_0": E_0,
            "E_1": E_1,
            "mass_gap": gap,
        })

    return results
