"""Universal confinement certification via the regime parameter chi.

chi = (lambda / gamma)^2 where lambda is the spectral norm of the
cross-coupling blocks and gamma is the smallest singular value of the
omitted block. chi < 1 licenses the reduction. chi >= 1 does not.
"""

from __future__ import annotations

import itertools
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ._emet import (
    decide_problem_json,
    version,
    build_ramanujan_hierarchy_json,
    ramanujan_max_inter_degree,
)

from emet.portrait import PhasePoint, PhasePortrait, Regime

__all__ = [
    "__version__",
    "decide",
    "decide_file",
    "decide_dense_matrix",
    "phase_point",
    "PhasePoint",
    "PhasePortrait",
    "Regime",
    "propose_partition_pca",
    "pca_propose_then_decide",
    "search_canonical_dense_matrix",
    "search_spectral_canonical_dense_matrix",
    "search_generalized_pencil_canonical_dense_matrix",
]

__version__ = version()


def decide(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Run the Rust engine on a ProblemInput dict. Returns the full decision report."""
    return json.loads(decide_problem_json(json.dumps(problem)))


def decide_file(path: str) -> dict[str, Any]:
    """Load a JSON ProblemInput from disk and decide."""
    from pathlib import Path
    return decide(json.loads(Path(path).read_text()))


def decide_dense_matrix(
    matrix: Any,
    retained: Sequence[int],
    omitted: Sequence[int],
    *,
    exact_license: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide a dense numpy matrix with explicit retained/omitted partition."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be a square 2D array")

    rows, cols = np.nonzero(array)
    entries = [
        {"row": int(row), "col": int(col), "value": float(array[row, col])}
        for row, col in zip(rows.tolist(), cols.tolist())
    ]
    payload: dict[str, Any] = {
        "dimension": int(array.shape[0]),
        "retained": list(map(int, retained)),
        "omitted": list(map(int, omitted)),
        "entries": entries,
    }
    if exact_license is not None:
        payload["exact_license"] = dict(exact_license)
    return decide(payload)


def phase_point(
    matrix: Any,
    retained: Sequence[int],
    omitted: Sequence[int],
    *,
    beta: float | None = None,
    exact_license: Mapping[str, Any] | None = None,
) -> tuple[PhasePoint, dict[str, Any]]:
    """Decide and return the phase point plus the full report.

    This is the primary diagnostic entry point. The phase point is the
    complete invariant; the report carries the details.
    """
    report = decide_dense_matrix(
        matrix, retained=retained, omitted=omitted,
        exact_license=exact_license,
    )
    pt = PhasePoint.from_report(report, beta=beta)
    return pt, report


def propose_partition_pca(
    matrix: Any,
    retained_dim: int,
) -> dict[str, Any]:
    """Propose a retained/omitted split by descending diagonal magnitude."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be a square 2D array")
    dimension = int(array.shape[0])
    if retained_dim < 1 or retained_dim >= dimension:
        raise ValueError("retained_dim must satisfy 1 <= retained_dim < dimension")

    diag_strength = np.abs(np.diag(array))
    ordered = sorted(range(dimension), key=lambda idx: (-diag_strength[idx], idx))
    retained = sorted(ordered[:retained_dim])
    omitted = sorted(ordered[retained_dim:])
    return {
        "retained": retained,
        "omitted": omitted,
        "ordering": ordered,
        "retained_dim": retained_dim,
        "dimension": dimension,
    }


def pca_propose_then_decide(
    matrix: Any,
    *,
    retained_dim: int,
    exact_license: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Propose a split via PCA ordering, then decide."""
    split = propose_partition_pca(matrix, retained_dim=retained_dim)
    decision = decide_dense_matrix(
        matrix,
        retained=split["retained"],
        omitted=split["omitted"],
        exact_license=exact_license,
    )
    return {"partition_proposal": split, "decision": decision}


def search_canonical_dense_matrix(
    matrix: Any,
    *,
    retained_dim: int,
    exact_license: Mapping[str, Any] | None = None,
    score_tolerance: float = 1.0e-9,
    max_family_size: int = 20000,
) -> dict[str, Any]:
    """Exhaustive search over all C(n, retained_dim) coordinate cuts.

    Returns the canonical (highest-scoring) subcritical cut, or reports
    indeterminacy if no unique winner exists.
    """
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be a square 2D array")
    dimension = int(array.shape[0])
    if retained_dim < 1 or retained_dim >= dimension:
        raise ValueError("retained_dim must satisfy 1 <= retained_dim < dimension")

    family_size = math.comb(dimension, retained_dim)
    if family_size > max_family_size:
        raise ValueError(
            f"candidate family size {family_size} exceeds max_family_size={max_family_size}"
        )

    symmetry_classes = _coordinate_symmetry_classes(array, score_tolerance)
    admissible: list[dict[str, Any]] = []

    for retained_tuple in itertools.combinations(range(dimension), retained_dim):
        retained = list(retained_tuple)
        retained_set = set(retained)
        omitted = [idx for idx in range(dimension) if idx not in retained_set]
        decision = decide_dense_matrix(array, retained=retained, omitted=omitted,
                                       exact_license=exact_license)
        if not decision.get("valid", False):
            continue
        score = float(np.trace(array[np.ix_(retained, retained)])) / retained_dim
        admissible.append({
            "retained": retained,
            "omitted": omitted,
            "score": score,
            "orbit_key": sorted(symmetry_classes[idx] for idx in retained),
            "decision": decision,
        })

    if not admissible:
        return {
            "retained_dim": retained_dim,
            "family_size": family_size,
            "status": "no_admissible_cut",
            "canonical_decision": None,
            "tied_candidates": [],
        }

    admissible.sort(key=lambda x: (-x["score"], x["retained"]))
    best = admissible[0]
    tied = [x for x in admissible if abs(x["score"] - best["score"]) <= score_tolerance]
    top_orbits = {tuple(x["orbit_key"]) for x in tied}

    if len(tied) == 1:
        status = "unique_canonical"
    elif len(top_orbits) == 1:
        status = "symmetry_tied"
    else:
        status = "indeterminate"

    return {
        "retained_dim": retained_dim,
        "family_size": family_size,
        "status": status,
        "canonical_decision": best["decision"],
        "canonical_retained": best["retained"],
        "canonical_omitted": best["omitted"],
        "canonical_score": best["score"],
        "tied_count": len(tied),
        "admissible_count": len(admissible),
        "tied_candidates": [
            {"retained": x["retained"], "omitted": x["omitted"], "score": x["score"]}
            for x in tied
        ],
    }


def search_spectral_canonical_dense_matrix(
    matrix: Any,
    *,
    retained_dim: int,
    hermitian_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Spectral canonical reduction: top-k eigenprojector of a Hermitian matrix."""
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be a square 2D array")
    dimension = int(array.shape[0])
    if retained_dim < 1 or retained_dim >= dimension:
        raise ValueError("retained_dim must satisfy 1 <= retained_dim < dimension")
    if not np.allclose(array, array.conj().T, atol=hermitian_tolerance, rtol=0.0):
        raise ValueError("matrix must be Hermitian")

    eigenvalues, eigenvectors = np.linalg.eigh(array)
    order = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    selected = list(range(dimension - retained_dim, dimension))
    basis = eigenvectors[:, selected]
    projector = basis @ basis.conj().T

    cutoff_idx = dimension - retained_dim
    cutoff = float(eigenvalues[cutoff_idx].real)
    below = float(eigenvalues[cutoff_idx - 1].real) if cutoff_idx > 0 else None
    gap = None if below is None else float(cutoff - below)

    cutoff_mult = sum(
        abs(float(v.real) - cutoff) <= hermitian_tolerance for v in eigenvalues
    )

    return {
        "retained_dim": retained_dim,
        "dimension": dimension,
        "status": "continuous_orbit" if cutoff_mult > 1 else "unique_canonical",
        "eigenvalues_descending": eigenvalues[::-1].real.tolist(),
        "cutoff_eigenvalue": cutoff,
        "cutoff_multiplicity": int(cutoff_mult),
        "spectral_gap": gap,
        "projector_real": projector.real.tolist(),
    }


def search_generalized_pencil_canonical_dense_matrix(
    a_matrix: Any,
    b_matrix: Any,
    *,
    retained_dim: int,
    hermitian_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    """Canonical spectral reduction for generalized pencil Ax = lambda Bx (B SPD)."""
    a = np.asarray(a_matrix, dtype=np.complex128)
    b = np.asarray(b_matrix, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a_matrix must be a square 2D array")
    if b.shape != a.shape:
        raise ValueError("b_matrix must match a_matrix shape")
    dimension = int(a.shape[0])
    if retained_dim < 1 or retained_dim >= dimension:
        raise ValueError("retained_dim must satisfy 1 <= retained_dim < dimension")
    if not np.allclose(a, a.conj().T, atol=hermitian_tolerance, rtol=0.0):
        raise ValueError("a_matrix must be Hermitian")
    if not np.allclose(b, b.conj().T, atol=hermitian_tolerance, rtol=0.0):
        raise ValueError("b_matrix must be Hermitian")

    chol = np.linalg.cholesky(b)
    inv_chol = np.linalg.inv(chol)
    reduced = inv_chol @ a @ inv_chol.conj().T
    eigvals, eigvecs = np.linalg.eigh(reduced)
    order = np.argsort(eigvals.real)
    eigvals = eigvals[order]

    selected = list(range(dimension - retained_dim, dimension))
    basis = eigvecs[:, selected]
    projector = basis @ basis.conj().T

    cutoff_idx = dimension - retained_dim
    cutoff = float(eigvals[cutoff_idx].real)
    below = float(eigvals[cutoff_idx - 1].real) if cutoff_idx > 0 else None
    gap = None if below is None else float(cutoff - below)
    cutoff_mult = sum(abs(float(v.real) - cutoff) <= hermitian_tolerance for v in eigvals)

    return {
        "retained_dim": retained_dim,
        "dimension": dimension,
        "status": "continuous_orbit" if cutoff_mult > 1 else "unique_canonical",
        "eigenvalues_descending": eigvals[::-1].real.tolist(),
        "cutoff_eigenvalue": cutoff,
        "cutoff_multiplicity": int(cutoff_mult),
        "spectral_gap": gap,
        "projector_real": projector.real.tolist(),
    }


# ── internal helpers ──────────────────────────────────────────────────────────

def _coordinate_symmetry_classes(array: np.ndarray, tolerance: float) -> dict[int, int]:
    signatures: dict[tuple[Any, ...], int] = {}
    classes: dict[int, int] = {}
    next_class = 0
    for idx in range(array.shape[0]):
        sig = _coordinate_signature(array, idx, tolerance)
        if sig not in signatures:
            signatures[sig] = next_class
            next_class += 1
        classes[idx] = signatures[sig]
    return classes


def _coordinate_signature(array: np.ndarray, idx: int, tolerance: float) -> tuple[Any, ...]:
    row = np.delete(array[idx, :], idx)
    col = np.delete(array[:, idx], idx)
    q = lambda v: int(round(float(v) / max(tolerance, 1e-9)))
    return (
        q(array[idx, idx]),
        tuple(sorted(q(v) for v in row.tolist())),
        tuple(sorted(q(v) for v in col.tolist())),
    )
