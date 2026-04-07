"""Dense attention collapse stress test.

Paper 3 predicts: for H = KK^T + eta*I with K in R^{n x d},
chi >= 1 when n > 2d for ANY partition retaining d coordinates.

This script:
1. Sweeps n/d ratio from 1.0 to 4.0 for multiple d values
2. Uses real-world attention head dimensions (d = 64, 128, 256)
3. Tests both random Gaussian K and structured (power-law) K
4. Reports chi and regime at each (n, d) pair
5. Identifies the empirical transition point vs the 2d prediction

If the theory holds: chi crosses 1 near n = 2d for all d.
If it breaks: chi stays below 1 past 2d, or crosses 1 before 2d.
"""

import sys
import time

import numpy as np

import emet
from emet.domains.transformer import build_gram, partition_from_mask


def run_sweep(
    d_head: int,
    ratios: list[float],
    eta: float = 1.0,
    key_distribution: str = "gaussian",
    seed: int = 42,
) -> list[dict]:
    """Sweep n/d ratio, return chi at each point."""
    rng = np.random.default_rng(seed)
    results = []

    for ratio in ratios:
        n = max(d_head + 1, int(round(ratio * d_head)))
        if n <= d_head:
            continue

        # Build Key matrix
        if key_distribution == "gaussian":
            K = rng.standard_normal((n, d_head))
        elif key_distribution == "power_law":
            # Zipf-like: token importance decays as 1/rank
            weights = 1.0 / np.arange(1, n + 1, dtype=float)
            K = rng.standard_normal((n, d_head)) * weights[:, None]
        else:
            raise ValueError(f"Unknown distribution: {key_distribution}")

        # Build Gram matrix
        H = build_gram(K, eta=eta)

        # Partition: retain first d_head tokens, omit the rest
        # (PCA-ordered would give the theory's best case;
        #  sequential gives a realistic eviction scenario)
        retained = list(range(d_head))
        omitted = list(range(d_head, n))

        t0 = time.perf_counter()
        report = emet.decide_dense_matrix(H, retained=retained, omitted=omitted)
        elapsed = time.perf_counter() - t0

        metrics = report.get("advanced_metrics", {})
        chi = metrics.get("chi")
        lam = metrics.get("lambda")
        gamma = metrics.get("gamma")
        regime = report.get("regime", "unknown")

        results.append({
            "d": d_head,
            "n": n,
            "ratio": n / d_head,
            "chi": chi,
            "lambda": lam,
            "gamma": gamma,
            "regime": regime,
            "valid": report.get("valid", False),
            "time_ms": elapsed * 1000,
            "distribution": key_distribution,
        })

    return results


def run_pca_sweep(
    d_head: int,
    ratios: list[float],
    eta: float = 1.0,
    seed: int = 42,
) -> list[dict]:
    """Same sweep but using PCA-ordered partition (theory's best case)."""
    rng = np.random.default_rng(seed)
    results = []

    for ratio in ratios:
        n = max(d_head + 1, int(round(ratio * d_head)))
        if n <= d_head:
            continue

        K = rng.standard_normal((n, d_head))
        H = build_gram(K, eta=eta)

        # PCA partition: retain the d_head tokens with largest diagonal
        diag = np.diag(H)
        order = np.argsort(diag)[::-1]
        retained = sorted(order[:d_head].tolist())
        omitted = sorted(order[d_head:].tolist())

        t0 = time.perf_counter()
        report = emet.decide_dense_matrix(H, retained=retained, omitted=omitted)
        elapsed = time.perf_counter() - t0

        metrics = report.get("advanced_metrics", {})
        chi = metrics.get("chi")
        regime = report.get("regime", "unknown")

        results.append({
            "d": d_head,
            "n": n,
            "ratio": n / d_head,
            "chi": chi,
            "regime": regime,
            "valid": report.get("valid", False),
            "time_ms": elapsed * 1000,
            "partition": "pca",
        })

    return results


def print_results(results: list[dict], title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()
    print(f"{'d':>5} {'n':>6} {'n/d':>6} {'chi':>14} {'regime':>14} {'time_ms':>8}")
    print("-" * 60)

    transition_found = False
    for r in results:
        chi_str = f"{r['chi']:.6e}" if r['chi'] is not None else "undefined"
        marker = ""
        if r['chi'] is not None and r['chi'] >= 1.0 and not transition_found:
            marker = "  ← TRANSITION"
            transition_found = True
        print(f"{r['d']:5d} {r['n']:6d} {r['ratio']:6.2f} {chi_str:>14} "
              f"{r['regime']:>14} {r['time_ms']:8.1f}{marker}")

    # Summary
    subcrit = [r for r in results if r['chi'] is not None and r['chi'] < 1.0]
    supercrit = [r for r in results if r['chi'] is not None and r['chi'] >= 1.0]
    print()
    if supercrit:
        first_super = supercrit[0]
        print(f"Transition at n/d = {first_super['ratio']:.2f} "
              f"(n={first_super['n']}, d={first_super['d']})")
        print(f"Predicted: n/d = 2.00 (n={2*first_super['d']})")
        error = abs(first_super['ratio'] - 2.0)
        print(f"Error: {error:.2f}")
    else:
        print("NO TRANSITION FOUND — chi < 1 at all ratios tested")
    print()


def main():
    print("DENSE ATTENTION COLLAPSE STRESS TEST")
    print(f"emet version: {emet.__version__}")
    print()

    # Fine-grained ratios around the predicted transition at n/d = 2
    ratios = [1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.95, 2.0,
              2.05, 2.1, 2.2, 2.3, 2.5, 3.0, 4.0]

    # ================================================================
    # TEST 1: Standard attention dimensions, Gaussian K, sequential partition
    # ================================================================
    for d in [32, 64, 128]:
        results = run_sweep(d, ratios, eta=1.0, key_distribution="gaussian")
        print_results(results, f"GAUSSIAN K, d_head={d}, eta=1.0, sequential partition")

    # ================================================================
    # TEST 2: Effect of eta (regularization)
    # ================================================================
    for eta in [0.01, 0.1, 1.0, 10.0]:
        results = run_sweep(64, ratios, eta=eta, key_distribution="gaussian")
        print_results(results, f"GAUSSIAN K, d_head=64, eta={eta}, sequential partition")

    # ================================================================
    # TEST 3: PCA-ordered partition (theory's best case)
    # ================================================================
    for d in [32, 64, 128]:
        results = run_pca_sweep(d, ratios, eta=1.0)
        print_results(results, f"GAUSSIAN K, d_head={d}, eta=1.0, PCA partition (best case)")

    # ================================================================
    # TEST 4: Power-law K (realistic token importance distribution)
    # ================================================================
    for d in [32, 64]:
        results = run_sweep(d, ratios, eta=1.0, key_distribution="power_law")
        print_results(results, f"POWER-LAW K, d_head={d}, eta=1.0, sequential partition")

    # ================================================================
    # TEST 5: Production-scale d_head = 256, fewer ratios (larger matrices)
    # ================================================================
    big_ratios = [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
    results = run_sweep(256, big_ratios, eta=1.0, key_distribution="gaussian")
    print_results(results, "GAUSSIAN K, d_head=256, eta=1.0, sequential partition")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("=" * 80)
    print("THEORY PREDICTION: chi >= 1 when n > 2d")
    print("If chi crosses 1 near n/d = 2.0 across all d, eta, distributions:")
    print("  → Paper 3 is confirmed empirically")
    print("If chi stays below 1 past n/d = 2.0:")
    print("  → Paper 3 is falsified")
    print("=" * 80)


if __name__ == "__main__":
    main()
