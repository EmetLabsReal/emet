"""Verify engine output against golden-vector oracle.

For each case in oracle/manifest.json:
  1. Load the input matrix and partition
  2. Run emet.decide_dense_matrix
  3. Compare status, regime, reason_code (exact match)
  4. Compare gamma, lambda, chi within |a - b| <= 1e-12
  5. Compare reduced_matrix entries within |a - b| <= 1e-10

Exit code 0 if all cases pass, 1 otherwise.

    python3 oracle/check.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ORACLE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = ORACLE_DIR / "manifest.json"

EPS_SCALAR = 1e-12
EPS_MATRIX = 1e-10


def load_case(case_ref):
    path = ORACLE_DIR / case_ref["file"]
    with open(path) as f:
        return json.load(f)


def matrix_from_entries(dimension, entries):
    H = np.zeros((dimension, dimension))
    for e in entries:
        H[e["row"], e["col"]] = e["value"]
    return H


def check_scalar(name, actual, expected, eps):
    if expected is None:
        return True
    if actual is None:
        return False
    return abs(actual - expected) <= eps


def check_matrix(actual, expected, eps):
    if expected is None:
        return actual is None
    if actual is None:
        return False
    actual_arr = np.array(actual)
    expected_arr = np.array(expected)
    if actual_arr.shape != expected_arr.shape:
        return False
    return np.all(np.abs(actual_arr - expected_arr) <= eps)


def run_case(case_data):
    import emet

    inp = case_data["input"]
    exp = case_data["expected"]

    H = matrix_from_entries(inp["dimension"], inp["entries"])
    report = emet.decide_dense_matrix(
        H,
        retained=inp["retained"],
        omitted=inp["omitted"],
    )

    failures = []

    for field in ["status", "regime", "reason_code"]:
        actual = report.get(field)
        expected = exp.get(field)
        if actual != expected:
            failures.append(f"{field}: expected {expected!r}, got {actual!r}")

    metrics = report.get("advanced_metrics") or {}
    for field, key in [("gamma", "gamma"), ("lambda", "lambda"), ("chi", "chi")]:
        actual = metrics.get(key)
        expected = exp.get(field)
        if not check_scalar(field, actual, expected, EPS_SCALAR):
            failures.append(f"{field}: expected {expected}, got {actual} (eps={EPS_SCALAR})")

    actual_rm = report.get("reduced_matrix")
    expected_rm = exp.get("reduced_matrix")
    if actual_rm is not None and isinstance(actual_rm, dict):
        actual_rm = actual_rm.get("data")
    if not check_matrix(actual_rm, expected_rm, EPS_MATRIX):
        failures.append(f"reduced_matrix mismatch (eps={EPS_MATRIX})")

    return failures


def main():
    if not MANIFEST_PATH.exists():
        print("ERROR: oracle/manifest.json not found. Run: python3 oracle/generate.py")
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    all_passed = True
    for case_ref in manifest["cases"]:
        case_data = load_case(case_ref)
        case_id = case_data["id"]
        failures = run_case(case_data)
        if failures:
            print(f"FAIL  {case_id}")
            for f in failures:
                print(f"      {f}")
            all_passed = False
        else:
            print(f"PASS  {case_id}")

    if all_passed:
        print(f"\nAll {len(manifest['cases'])} golden vectors passed.")
        sys.exit(0)
    else:
        print("\nSome golden vectors FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
