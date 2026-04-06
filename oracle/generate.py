"""Generate golden-vector test suite for cert-contract-1.

Each case is a matrix + partition with expected branch, regime, status,
reason_code, and (when computable) gamma, lambda, chi, and reduced_matrix.

Cases are derived from the Rust test fixtures in engine/tests/fixtures/.
Run from the repository root:

    python3 oracle/generate.py
"""

import hashlib
import json
import math
import sys
from pathlib import Path

ORACLE_DIR = Path(__file__).resolve().parent
CASES_DIR = ORACLE_DIR / "cases"
MANIFEST_PATH = ORACLE_DIR / "manifest.json"

# Contract constants (cert-contract-1 §1.3)
EPSILON_SING = 1e-12
EPSILON_INV = 1e-10
EPSILON_REPORT_SCALAR = 1e-12
EPSILON_REPORT_MATRIX = 1e-10

# ── golden-vector definitions ──────────────────────────────────────────────

CASES = [
    {
        "id": "branch-a-001",
        "description": "Branch A: gamma <= epsilon_sing (quarter-turn, no diagonal on omitted block)",
        "matrix": [
            [0.0, -1.0],
            [1.0,  0.0],
        ],
        "retained": [0],
        "omitted": [1],
        "expected": {
            "branch": "A",
            "status": "invalid",
            "regime": "pre_admissible",
            "reason_code": "insufficient_omitted_block_control",
            "gamma": 0.0,
            "lambda": 1.0,
            "chi": None,
            "reduced_matrix": None,
        },
    },
    {
        "id": "branch-c-sub-001",
        "description": "Branch C subcritical: 3x3, gamma=2, lambda=0.25, chi=0.015625",
        "matrix": [
            [1.0,  0.0, 0.25],
            [0.0,  2.0, 0.0 ],
            [0.25, 0.0, 2.0 ],
        ],
        "retained": [0, 1],
        "omitted": [2],
        "expected": {
            "branch": "C",
            "status": "valid",
            "regime": "subcritical",
            "reason_code": "valid_reduction",
            "gamma": 2.0,
            "lambda": 0.25,
            "chi": 0.015625,
            "reduced_matrix": [
                [0.96875, 0.0],
                [0.0,     2.0],
            ],
        },
    },
    {
        "id": "branch-c-sup-001",
        "description": "Branch C supercritical: 3x3, gamma=1, lambda=2, chi=4.0",
        "matrix": [
            [1.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
        ],
        "retained": [0, 1],
        "omitted": [2],
        "expected": {
            "branch": "C",
            "status": "invalid",
            "regime": "supercritical",
            "reason_code": "coupling_too_strong",
            "gamma": 1.0,
            "lambda": 2.0,
            "chi": 4.0,
            "reduced_matrix": [
                [-3.0, 0.0],
                [ 0.0, 2.0],
            ],
        },
    },
    {
        "id": "screening-001",
        "description": "Screening failure: overlapping partition indices",
        "matrix": [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        "retained": [0],
        "omitted": [0],
        "expected": {
            "branch": "screening",
            "status": "invalid",
            "regime": "screening_failure",
            "reason_code": "invalid_partition",
            "gamma": None,
            "lambda": None,
            "chi": None,
            "reduced_matrix": None,
        },
    },
]


def case_to_json(case):
    """Format a case for the oracle/cases/ directory."""
    return {
        "id": case["id"],
        "description": case["description"],
        "input": {
            "dimension": len(case["matrix"]),
            "retained": case["retained"],
            "omitted": case["omitted"],
            "entries": [
                {"row": i, "col": j, "value": v}
                for i, row in enumerate(case["matrix"])
                for j, v in enumerate(row)
                if v != 0.0
            ],
        },
        "expected": case["expected"],
        "tolerances": {
            "scalars": EPSILON_REPORT_SCALAR,
            "matrix_entries": EPSILON_REPORT_MATRIX,
            "enums": "exact",
        },
    }


def canonical_json(obj):
    """Canonical JSON: sorted keys, no whitespace, no trailing newline."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_of(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    CASES_DIR.mkdir(parents=True, exist_ok=True)

    case_digests = []
    for case in CASES:
        case_json = case_to_json(case)
        text = json.dumps(case_json, indent=2, sort_keys=True) + "\n"
        path = CASES_DIR / f"{case['id']}.json"
        path.write_text(text)
        case_digests.append({
            "id": case["id"],
            "file": f"cases/{case['id']}.json",
            "sha256": sha256_of(canonical_json(case_json)),
        })
        print(f"  wrote {path.name}")

    manifest = {
        "contract": "cert-contract-1",
        "generated_by": "oracle/generate.py",
        "cases": case_digests,
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    MANIFEST_PATH.write_text(manifest_text)
    manifest_digest = sha256_of(canonical_json(manifest))
    print(f"  wrote manifest.json (digest: {manifest_digest})")
    print(f"\n{len(CASES)} golden vectors generated.")


if __name__ == "__main__":
    main()
