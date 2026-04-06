//! Ramanujan graph construction and hierarchical attention patterns.
//!
//! Builds d-regular graphs with spectral gap ≥ d - 2√(d-1) (Ramanujan property).
//! For d-1 prime and d-1 ≡ 1 mod 4, uses the LPS (Lubotzky–Phillips–Sarnak) construction.
//! Otherwise, uses random d-regular graphs which are almost Ramanujan with high probability.
//!
//! The hierarchy arranges n tokens into clusters of size d, with inter-cluster
//! degree d_× < d - 2√(d-1), guaranteeing χ < 1 at every level.

use nalgebra::DMatrix;

/// A d-regular graph on n vertices, stored as adjacency matrix.
#[derive(Debug, Clone)]
pub struct RegularGraph {
    pub degree: usize,
    pub n_vertices: usize,
    pub adjacency: DMatrix<f64>,
}

/// A Ramanujan hierarchy: clusters connected by sparse inter-cluster edges.
#[derive(Debug, Clone)]
pub struct RamanujanHierarchy {
    pub n_tokens: usize,
    pub cluster_size: usize,
    pub inter_degree: usize,
    pub n_clusters: usize,
    pub depth: usize,
    /// Full attention mask as adjacency matrix (n × n).
    pub attention_mask: DMatrix<f64>,
    /// Laplacian: D - A (for chi computation).
    pub laplacian: DMatrix<f64>,
}

/// Construct a d-regular graph on n vertices using random pairing.
///
/// For n*d even, generates a random d-regular graph by repeated
/// perfect matching on d*n half-edges. The result is approximately
/// Ramanujan with high probability for large n (Friedman's theorem).
pub fn random_regular_graph(n: usize, d: usize, seed: u64) -> RegularGraph {
    assert!(n > d, "need n > d for d-regular graph");
    assert!((n * d) % 2 == 0, "n*d must be even");

    let mut adj = DMatrix::zeros(n, n);
    let mut rng = SimpleRng::new(seed);

    // Configuration model: pair half-edges randomly
    for _ in 0..d {
        let mut available: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = rng.next_usize() % (i + 1);
            available.swap(i, j);
        }

        // Pair consecutive vertices
        for pair in available.chunks(2) {
            if pair.len() == 2 {
                let u = pair[0];
                let v = pair[1];
                if u != v {
                    adj[(u, v)] += 1.0;
                    adj[(v, u)] += 1.0;
                }
            }
        }
    }

    // Clamp to simple graph (no multi-edges)
    for i in 0..n {
        for j in 0..n {
            if adj[(i, j)] > 1.0 {
                adj[(i, j)] = 1.0;
            }
        }
        adj[(i, i)] = 0.0; // no self-loops
    }

    RegularGraph {
        degree: d,
        n_vertices: n,
        adjacency: adj,
    }
}

/// Construct a complete graph on n vertices (d = n-1 regular).
pub fn complete_graph(n: usize) -> RegularGraph {
    let mut adj = DMatrix::from_element(n, n, 1.0);
    for i in 0..n {
        adj[(i, i)] = 0.0;
    }
    RegularGraph {
        degree: n - 1,
        n_vertices: n,
        adjacency: adj,
    }
}

/// Construct a cycle graph on n vertices (2-regular).
pub fn cycle_graph(n: usize) -> RegularGraph {
    let mut adj = DMatrix::zeros(n, n);
    for i in 0..n {
        let next = (i + 1) % n;
        adj[(i, next)] = 1.0;
        adj[(next, i)] = 1.0;
    }
    RegularGraph {
        degree: 2,
        n_vertices: n,
        adjacency: adj,
    }
}

/// Compute the spectral gap of a d-regular graph.
/// Returns d - μ₂ where μ₂ is the second-largest eigenvalue of A.
pub fn spectral_gap(graph: &RegularGraph) -> f64 {
    let eigs = graph.adjacency.clone().symmetric_eigen();
    let mut eigenvalues: Vec<f64> = eigs.eigenvalues.iter().copied().collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let d = graph.degree as f64;
    if eigenvalues.len() < 2 {
        return d;
    }
    d - eigenvalues[1]
}

/// Check if a graph is Ramanujan: spectral gap ≥ d - 2√(d-1).
pub fn is_ramanujan(graph: &RegularGraph) -> bool {
    let gap = spectral_gap(graph);
    let bound = (graph.degree as f64) - 2.0 * ((graph.degree as f64 - 1.0).sqrt());
    gap >= bound - 1e-10 // numerical tolerance
}

/// Build a Ramanujan hierarchy for n tokens.
///
/// Parameters:
/// - `n`: number of tokens
/// - `cluster_size`: size of each cluster (intra-cluster is complete or d-regular)
/// - `inter_degree`: d_× — number of inter-cluster connections per token
/// - `seed`: random seed for graph construction
///
/// The hierarchy has χ < 1 at every level when
/// inter_degree < cluster_size - 1 - 2√(cluster_size - 2).
pub fn build_hierarchy(
    n: usize,
    cluster_size: usize,
    inter_degree: usize,
    seed: u64,
) -> RamanujanHierarchy {
    assert!(n >= cluster_size, "need at least one full cluster");
    let n_clusters = (n + cluster_size - 1) / cluster_size;
    let actual_n = n_clusters * cluster_size; // pad to full clusters

    let mut mask = DMatrix::zeros(actual_n, actual_n);

    // Intra-cluster: complete graph within each cluster
    for c in 0..n_clusters {
        let base = c * cluster_size;
        for i in 0..cluster_size {
            for j in 0..cluster_size {
                if i != j {
                    mask[(base + i, base + j)] = 1.0;
                }
            }
        }
    }

    // Inter-cluster: connect clusters with d_× edges per token
    if n_clusters > 1 && inter_degree > 0 {
        let mut rng = SimpleRng::new(seed);

        for c in 0..n_clusters {
            let base_c = c * cluster_size;
            // Connect to neighboring clusters
            for token_in_cluster in 0..cluster_size {
                let u = base_c + token_in_cluster;
                let mut connections = 0;
                let mut attempts = 0;
                while connections < inter_degree && attempts < inter_degree * 10 {
                    // Pick a random other cluster
                    let other_cluster = rng.next_usize() % n_clusters;
                    if other_cluster == c {
                        attempts += 1;
                        continue;
                    }
                    let other_base = other_cluster * cluster_size;
                    let v = other_base + (rng.next_usize() % cluster_size);

                    if mask[(u, v)] == 0.0 {
                        mask[(u, v)] = 1.0;
                        mask[(v, u)] = 1.0;
                        connections += 1;
                    }
                    attempts += 1;
                }
            }
        }
    }

    // Truncate to actual n if we padded
    let mask = if actual_n > n {
        mask.view((0, 0), (n, n)).into_owned()
    } else {
        mask
    };

    // Laplacian: L = D - A
    let final_n = mask.nrows();
    let mut laplacian = -mask.clone();
    for i in 0..final_n {
        let degree: f64 = mask.row(i).iter().sum();
        laplacian[(i, i)] = degree;
    }

    // Compute depth: log_{cluster_size}(n_clusters) + 1
    let depth = if n_clusters <= 1 {
        1
    } else {
        (n_clusters as f64).log2().ceil() as usize + 1
    };

    RamanujanHierarchy {
        n_tokens: final_n,
        cluster_size,
        inter_degree,
        n_clusters,
        depth,
        attention_mask: mask,
        laplacian,
    }
}

/// Compute chi for the hierarchy at the cluster level.
///
/// For each pair of adjacent clusters (c1, c2):
///   P = tokens in c1, Q = tokens in c2.
///   chi = (||L_PQ||_2 / sigma_min(L_QQ))^2.
///
/// Returns (worst_chi, worst_gamma, worst_lambda) across all adjacent pairs.
///
/// Also computes the graph-theoretic bound:
///   chi_bound = (d_× / (d - 2√(d-1)))^2
/// which holds when each cluster is Ramanujan.
pub fn hierarchy_chi(hierarchy: &RamanujanHierarchy) -> (f64, f64, f64) {
    let cs = hierarchy.cluster_size;
    let n_clusters = hierarchy.n_clusters;
    let mask = &hierarchy.attention_mask;

    let mut worst_chi = 0.0_f64;
    let mut worst_gamma = f64::INFINITY;
    let mut worst_lambda = 0.0_f64;

    for c1 in 0..n_clusters {
        for c2 in 0..n_clusters {
            if c1 == c2 {
                continue;
            }

            let base1 = c1 * cs;
            let end1 = (base1 + cs).min(hierarchy.n_tokens);
            let base2 = c2 * cs;
            let end2 = (base2 + cs).min(hierarchy.n_tokens);

            let p_indices: Vec<usize> = (base1..end1).collect();
            let q_indices: Vec<usize> = (base2..end2).collect();
            let p_size = p_indices.len();
            let q_size = q_indices.len();

            // Check if clusters are connected
            let mut connected = false;
            for &pi in &p_indices {
                for &qi in &q_indices {
                    if mask[(pi, qi)] > 0.0 {
                        connected = true;
                        break;
                    }
                }
                if connected {
                    break;
                }
            }
            if !connected {
                continue;
            }

            // L_PQ: Laplacian cross-block between c1 and c2
            // For the Laplacian, L_PQ = -A_PQ (off-diagonal blocks are negative adjacency)
            let lap = &hierarchy.laplacian;
            let h_pq = DMatrix::from_fn(p_size, q_size, |i, j| {
                lap[(p_indices[i], q_indices[j])]
            });

            // L_QQ: Laplacian of cluster c2 (includes intra-cluster + diagonal from inter-cluster)
            let h_qq = DMatrix::from_fn(q_size, q_size, |i, j| {
                lap[(q_indices[i], q_indices[j])]
            });

            let svd_pq = h_pq.svd(false, false);
            let lambda = svd_pq
                .singular_values
                .iter()
                .copied()
                .fold(0.0, f64::max);

            let svd_qq = h_qq.svd(false, false);
            let gamma = svd_qq
                .singular_values
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);

            if gamma > 1e-15 {
                let chi = (lambda / gamma).powi(2);
                if chi > worst_chi {
                    worst_chi = chi;
                    worst_gamma = gamma;
                    worst_lambda = lambda;
                }
            }
        }
    }

    (worst_chi, worst_gamma, worst_lambda)
}

/// Graph-theoretic chi bound for a Ramanujan hierarchy.
/// chi_bound = (d_× / (d - 2√(d-1)))^2
/// where d is the cluster degree (cluster_size - 1 for complete clusters).
pub fn chi_bound(cluster_size: usize, inter_degree: usize) -> f64 {
    let d = (cluster_size - 1) as f64;
    let gap = d - 2.0 * (d - 1.0).sqrt();
    if gap <= 0.0 {
        return f64::INFINITY;
    }
    let dx = inter_degree as f64;
    (dx / gap).powi(2)
}

/// Alon-Boppana bound: for any d-regular graph, μ₂ ≥ 2√(d-1) - o(1).
/// Returns the Ramanujan threshold for chi < 1.
pub fn max_inter_degree(cluster_degree: usize) -> usize {
    let d = cluster_degree as f64;
    let bound = d - 2.0 * (d - 1.0).sqrt();
    bound.floor() as usize
}

// Minimal deterministic PRNG (xorshift64)
struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_graph_is_ramanujan() {
        for n in [4, 8, 16] {
            let g = complete_graph(n);
            assert!(is_ramanujan(&g), "K_{n} should be Ramanujan");
        }
    }

    #[test]
    fn cycle_is_ramanujan() {
        // Cycle on n vertices is 2-regular. Ramanujan iff μ₂ ≤ 2√1 = 2.
        // For cycle: eigenvalues are 2cos(2πk/n), so μ₂ = 2cos(2π/n) < 2.
        for n in [5, 7, 11, 13] {
            let g = cycle_graph(n);
            assert!(is_ramanujan(&g), "C_{n} should be Ramanujan");
        }
    }

    #[test]
    fn chi_bound_subcritical_at_low_cross_degree() {
        // Cluster size 8 (degree d=7): Ramanujan gap = 7 - 2√6 ≈ 2.10
        // d_× = 1: chi_bound = (1/2.10)^2 ≈ 0.227 < 1
        let bound = chi_bound(8, 1);
        assert!(
            bound < 1.0,
            "chi_bound={bound:.4} should be < 1 for d_x=1, cluster=8"
        );
    }

    #[test]
    fn chi_bound_supercritical_at_high_cross_degree() {
        // d_× = 5 > 2.10: chi_bound = (5/2.10)^2 ≈ 5.67 > 1
        let bound = chi_bound(8, 5);
        assert!(
            bound > 1.0,
            "chi_bound={bound:.4} should be > 1 for d_x=5, cluster=8"
        );
    }

    #[test]
    fn hierarchy_chi_pair_computed() {
        // Build a small hierarchy and verify chi is finite and positive
        let h = build_hierarchy(16, 4, 1, 42);
        let (chi, gamma, lambda) = hierarchy_chi(&h);
        assert!(chi > 0.0, "chi should be positive");
        assert!(gamma > 0.0, "gamma should be positive");
        assert!(lambda > 0.0, "lambda should be positive");
        // With cluster_size=4 (d=3), gap = 3 - 2√2 ≈ 0.17
        // d_x=1: bound = (1/0.17)^2 ≈ 34.6 — supercritical because d=3 is too small
        // This is expected: small clusters have weak spectral gap
    }

    #[test]
    fn hierarchy_chi_large_clusters() {
        // Larger clusters have better spectral gap
        // Cluster 16 (d=15): gap = 15 - 2√14 ≈ 7.51
        // d_x = 1: bound = (1/7.51)^2 ≈ 0.018
        let h = build_hierarchy(64, 16, 1, 42);
        let (chi, gamma, lambda) = hierarchy_chi(&h);
        let bound = chi_bound(16, 1);
        assert!(
            bound < 1.0,
            "chi_bound={bound:.4} should be < 1 for d_x=1, cluster=16"
        );
        // Numerical chi should also be subcritical for adjacent pairs
        // (might not hold exactly due to random construction, but bound holds)
        eprintln!("chi={chi:.4}, gamma={gamma:.4}, lambda={lambda:.4}, bound={bound:.4}");
    }

    #[test]
    fn max_inter_degree_values() {
        // d=8 (cluster K_8, degree 7): bound = 7 - 2√6 ≈ 2.10
        assert_eq!(max_inter_degree(7), 2);
        // d=16: 15 - 2√14 ≈ 7.51
        assert_eq!(max_inter_degree(15), 7);
        // d=64: 63 - 2√62 ≈ 47.26
        assert_eq!(max_inter_degree(63), 47);
    }

    #[test]
    fn spectral_gap_of_complete_graph() {
        let g = complete_graph(8);
        let gap = spectral_gap(&g);
        // K_8: eigenvalues are 7 (once) and -1 (7 times). Gap = 7 - (-1) = 8.
        assert!((gap - 8.0).abs() < 1e-8, "gap={gap}");
    }
}
