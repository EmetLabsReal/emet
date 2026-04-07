"""Ramanujan spectral isolation: χ < 1 from graph structure alone.

For a d-regular Ramanujan graph:
  μ₂ ≤ 2√(d-1)  (Alon-Boppana bound, tight for Ramanujan)
  γ = d - μ₂ ≥ d - 2√(d-1)  (spectral gap)

The Laplacian L = dI - A partitioned as (P, Q):
  χ = (λ/γ)² where λ = ||L_PQ||, γ = σ_min(L_QQ)

If the graph is Ramanujan and the partition respects the
spectral structure, χ < 1: the omitted block cannot corrupt
the retained block. This is the discrete analogue of
Cap = 0 on the modular surface.

Sweeps:
1. Degree sweep: d = 5, 7, 11, ..., 53 (primes ≡ 1 mod 4 where possible)
2. Size sweep: fixed d, increasing n
3. Partition sweep: fixed graph, vary |P|/|Q| ratio
4. Spectral verification: eigenvalue distribution vs Kesten-McKay
"""

import numpy as np

import emet
from emet.domains.lps_ramanujan import (
    build_regular_graph_laplacian,
    partition_graph,
    spectral_partition,
    sweep_regular_graphs,
    verify_ramanujan,
)


def degree_sweep():
    print("=" * 100)
    print("SWEEP 1: Degree sweep (random d-regular, n=200, half partition)")
    print("=" * 100)
    print()
    print(f"{'d':>4} {'n':>5} {'Ramanujan':>10} {'μ₂':>8} {'2√(d-1)':>8} "
          f"{'gap':>8} {'chi':>14} {'λ':>12} {'γ':>12} {'regime':>14}")
    print("-" * 110)

    degrees = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    results = sweep_regular_graphs(degrees, n_per_degree=200, seed=42)

    n_sub = 0
    n_sup = 0
    for r in results:
        chi_s = f"{r['chi']:.6e}" if r['chi'] is not None else "N/A"
        lam_s = f"{r['lambda']:.6e}" if r['lambda'] is not None else "N/A"
        gam_s = f"{r['gamma']:.6e}" if r['gamma'] is not None else "N/A"
        ram_s = "YES" if r['is_ramanujan'] else "no"
        print(f"{r['d']:4d} {r['n']:5d} {ram_s:>10} {r['mu2']:8.3f} {r['bound']:8.3f} "
              f"{r['gap']:8.3f} {chi_s:>14} {lam_s:>12} {gam_s:>12} {r['regime']:>14}")
        if r['regime'] == 'subcritical':
            n_sub += 1
        else:
            n_sup += 1

    print()
    print(f"Subcritical: {n_sub}/{len(results)}  Supercritical: {n_sup}/{len(results)}")
    print()


def size_sweep():
    print("=" * 100)
    print("SWEEP 2: Size sweep (d=13, increasing n)")
    print("=" * 100)
    print()

    d = 13
    sizes = [50, 100, 200, 500, 1000, 2000]

    print(f"{'n':>6} {'Ramanujan':>10} {'gap':>8} {'chi':>14} {'λ':>12} {'γ':>12} {'regime':>14}")
    print("-" * 80)

    for n in sizes:
        n = n + (n % 2)
        L, A = build_regular_graph_laplacian(n, d, seed=42)
        ram = verify_ramanujan(A, d)
        _, retained, omitted = partition_graph(L, n // 2)
        report = emet.decide_dense_matrix(L.tolist(), retained=retained, omitted=omitted)
        m = report.get("advanced_metrics", {})
        chi_s = f"{m.get('chi', 0):.6e}"
        lam_s = f"{m.get('lambda', 0):.6e}"
        gam_s = f"{m.get('gamma', 0):.6e}"
        ram_s = "YES" if ram['is_ramanujan'] else "no"
        print(f"{n:6d} {ram_s:>10} {ram['gap']:8.3f} {chi_s:>14} {lam_s:>12} {gam_s:>12} {report.get('regime', '?'):>14}")

    print()


def partition_sweep():
    print("=" * 100)
    print("SWEEP 3: Partition ratio (d=13, n=200)")
    print("=" * 100)
    print()

    d = 13
    n = 200
    L, A = build_regular_graph_laplacian(n, d, seed=42)

    print(f"{'|P|':>5} {'|Q|':>5} {'|P|/|Q|':>8} {'chi':>14} {'λ':>12} {'γ':>12} {'regime':>14}")
    print("-" * 75)

    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        n_ret = max(2, int(n * frac))
        _, retained, omitted = partition_graph(L, n_ret)
        if len(omitted) < 2:
            continue
        report = emet.decide_dense_matrix(L.tolist(), retained=retained, omitted=omitted)
        m = report.get("advanced_metrics", {})
        chi_s = f"{m.get('chi', 0):.6e}"
        lam_s = f"{m.get('lambda', 0):.6e}"
        gam_s = f"{m.get('gamma', 0):.6e}"
        ratio = n_ret / (n - n_ret)
        print(f"{n_ret:5d} {n - n_ret:5d} {ratio:8.2f} {chi_s:>14} {lam_s:>12} {gam_s:>12} {report.get('regime', '?'):>14}")

    print()


def spectral_check():
    """Verify eigenvalue distribution matches Kesten-McKay."""
    print("=" * 100)
    print("SPECTRAL CHECK: Eigenvalue distribution (d=13, n=1000)")
    print("=" * 100)
    print()

    d = 13
    n = 1000
    L, A = build_regular_graph_laplacian(n, d, seed=42)
    ram = verify_ramanujan(A, d)

    eigs_A = np.sort(np.linalg.eigvalsh(A))[::-1]

    print(f"Degree: {d}")
    print(f"Vertices: {n}")
    print(f"Ramanujan: {'YES' if ram['is_ramanujan'] else 'NO'}")
    print(f"μ₁ = {eigs_A[0]:.4f} (should be {d})")
    print(f"μ₂ = {eigs_A[1]:.4f}")
    print(f"|μ_n| = {abs(eigs_A[-1]):.4f}")
    print(f"max(|μ₂|, |μ_n|) = {ram['mu2']:.4f}")
    print(f"Alon-Boppana bound: 2√(d-1) = {ram['bound']:.4f}")
    print(f"Spectral gap: γ = {ram['gap']:.4f}")
    print(f"Gap bound (Ramanujan): d - 2√(d-1) = {ram['gap_bound']:.4f}")
    print()

    # Eigenvalue histogram
    bins = np.linspace(-d, d, 41)
    hist, edges = np.histogram(eigs_A[1:], bins=bins)
    max_h = max(hist)
    print("Eigenvalue distribution (adjacency):")
    for i in range(len(hist)):
        bar = "#" * int(40 * hist[i] / max_h) if max_h > 0 else ""
        if hist[i] > 0:
            print(f"  [{edges[i]:6.1f}, {edges[i+1]:6.1f}): {bar} ({hist[i]})")
    print()

    # Kesten-McKay: ρ(x) = d√(4(d-1) - x²) / (2π(d² - x²)) for |x| ≤ 2√(d-1)
    print("Kesten-McKay prediction: bulk eigenvalues in [-2√(d-1), 2√(d-1)]")
    print(f"  = [{-ram['bound']:.3f}, {ram['bound']:.3f}]")
    in_bulk = np.sum((np.abs(eigs_A[1:]) <= ram['bound'] + 0.1))
    print(f"  Eigenvalues in bulk: {in_bulk}/{n-1} ({100*in_bulk/(n-1):.1f}%)")
    print()


def spectral_partition_sweep():
    """Use Fiedler vector to find the natural partition that minimizes λ."""
    print("=" * 100)
    print("SWEEP 5: Spectral (Fiedler) partition — the graph chooses its own cut")
    print("=" * 100)
    print()

    print(f"{'d':>4} {'n':>5} {'|P|':>4} {'|Q|':>4} {'chi':>14} {'λ':>12} {'γ':>12} {'regime':>14}")
    print("-" * 80)

    for d in [5, 7, 13, 17, 23, 29, 37, 53]:
        n = max(200, 4 * d)
        n = n + (n % 2)
        L, A = build_regular_graph_laplacian(n, d, seed=42)

        # Spectral partition: Fiedler vector, 60% retained
        retained, omitted = spectral_partition(L, frac_retained=0.6)

        report = emet.decide_dense_matrix(L.tolist(), retained=retained, omitted=omitted)
        m = report.get("advanced_metrics", {})
        chi_s = f"{m.get('chi', 0):.6e}"
        lam_s = f"{m.get('lambda', 0):.6e}"
        gam_s = f"{m.get('gamma', 0):.6e}"
        regime = report.get('regime', '?')
        print(f"{d:4d} {n:5d} {len(retained):4d} {len(omitted):4d} "
              f"{chi_s:>14} {lam_s:>12} {gam_s:>12} {regime:>14}")

    print()
    print("The Fiedler vector finds the minimum-conductance cut.")
    print("This is the partition the graph 'wants' — the natural boundary.")
    print()


def spectral_partition_ratio_sweep():
    """Sweep retained fraction with Fiedler partition."""
    print("=" * 100)
    print("SWEEP 6: Fiedler partition, varying retained fraction (d=13, n=500)")
    print("=" * 100)
    print()

    d = 13
    n = 500
    L, A = build_regular_graph_laplacian(n, d, seed=42)

    print(f"{'frac':>6} {'|P|':>5} {'|Q|':>5} {'chi':>14} {'λ':>12} {'γ':>12} {'regime':>14}")
    print("-" * 75)

    for frac in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        retained, omitted = spectral_partition(L, frac_retained=frac)
        if len(omitted) < 2:
            continue
        report = emet.decide_dense_matrix(L.tolist(), retained=retained, omitted=omitted)
        m = report.get("advanced_metrics", {})
        chi_s = f"{m.get('chi', 0):.6e}"
        lam_s = f"{m.get('lambda', 0):.6e}"
        gam_s = f"{m.get('gamma', 0):.6e}"
        print(f"{frac:6.2f} {len(retained):5d} {len(omitted):5d} "
              f"{chi_s:>14} {lam_s:>12} {gam_s:>12} {report.get('regime', '?'):>14}")

    print()


def cluster_pair_sweep():
    """Partition into spectral clusters, test all cluster pairs."""
    print("=" * 100)
    print("SWEEP 7: Cluster-pair χ (d=13, n=500, 4 spectral clusters)")
    print("=" * 100)
    print()

    d = 13
    n = 500
    L, A = build_regular_graph_laplacian(n, d, seed=42)

    # Spectral clustering: use Fiedler + higher eigenvectors
    eigs, vecs = np.linalg.eigh(L)
    n_clusters = 4
    features = vecs[:, 1:n_clusters + 1]

    # K-means clustering on spectral features
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(features)
    clusters = [[] for _ in range(n_clusters)]
    for i, c in enumerate(labels):
        clusters[c].append(i)

    print(f"Cluster sizes: {[len(c) for c in clusters]}")
    print()

    print(f"{'P':>3} {'Q':>3} {'|P|':>5} {'|Q|':>5} {'chi':>14} {'λ':>12} {'γ':>12} {'regime':>14}")
    print("-" * 75)

    for ci in range(n_clusters):
        for cj in range(n_clusters):
            if ci == cj:
                continue
            retained = clusters[ci]
            omitted = clusters[cj]
            combined = sorted(retained + omitted)

            # Extract sub-Laplacian for this pair
            sub_L = L[np.ix_(combined, combined)]
            # Remap indices
            idx_map = {v: i for i, v in enumerate(combined)}
            ret_mapped = [idx_map[v] for v in retained]
            omit_mapped = [idx_map[v] for v in omitted]

            report = emet.decide_dense_matrix(
                sub_L.tolist(), retained=ret_mapped, omitted=omit_mapped,
            )
            m = report.get("advanced_metrics", {})
            chi_s = f"{m.get('chi', 0):.6e}"
            lam_s = f"{m.get('lambda', 0):.6e}"
            gam_s = f"{m.get('gamma', 0):.6e}"
            print(f"{ci:3d} {cj:3d} {len(retained):5d} {len(omitted):5d} "
                  f"{chi_s:>14} {lam_s:>12} {gam_s:>12} {report.get('regime', '?'):>14}")

    print()


def prime_sweep():
    """Sweep primes p ≡ 1 (mod 4) as degrees: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97."""
    print("=" * 100)
    print("SWEEP 4: Primes p ≡ 1 (mod 4) — LPS-eligible degrees")
    print("=" * 100)
    print()

    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97]

    print(f"{'p':>4} {'d=p+1':>6} {'n':>5} {'gap_bound':>10} {'chi':>14} {'regime':>14}")
    print("-" * 60)

    for p in primes_1mod4:
        d = p + 1
        n = max(200, 4 * d)
        n = n + (n % 2)
        gap_b = d - 2 * np.sqrt(d - 1)

        L, A = build_regular_graph_laplacian(n, d, seed=42)
        _, retained, omitted = partition_graph(L, n // 2)
        report = emet.decide_dense_matrix(L.tolist(), retained=retained, omitted=omitted)
        m = report.get("advanced_metrics", {})
        chi_s = f"{m.get('chi', 0):.6e}"
        print(f"{p:4d} {d:6d} {n:5d} {gap_b:10.3f} {chi_s:>14} {report.get('regime', '?'):>14}")

    print()


def main():
    print()
    print("RAMANUJAN SPECTRAL ISOLATION")
    print(f"emet version: {emet.__version__}")
    print()
    print("d-regular Ramanujan graph: μ₂ ≤ 2√(d-1)")
    print("Spectral gap γ ≥ d - 2√(d-1)")
    print("χ = (λ/γ)² — if χ < 1, partition boundary has Cap = 0")
    print()

    degree_sweep()
    size_sweep()
    partition_sweep()
    spectral_partition_sweep()
    spectral_partition_ratio_sweep()
    cluster_pair_sweep()
    spectral_check()
    prime_sweep()

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print("For d-regular Ramanujan graphs:")
    print("  γ ≥ d - 2√(d-1) > 0 for d ≥ 3")
    print("  If λ < γ then χ < 1: licensed reduction, Cap = 0")
    print("  The Friedrichs extension is unique")
    print("  All spectral quantities are determined by graph structure alone")
    print()
    print("Connection to Riemann Hypothesis:")
    print("  LPS graphs are Cayley graphs of PSL(2, Z/qZ)")
    print("  Their Ramanujan property is equivalent to the")
    print("  Ramanujan-Petersson conjecture for GL(2)")
    print("  χ < 1 on these graphs = spectral isolation")
    print("  = discrete analogue of Cap = 0 on the modular surface")
    print("=" * 100)


if __name__ == "__main__":
    main()
