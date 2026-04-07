"""Does semantic space have a Feller threshold?

Generate random key matrices K of increasing sequence length (simulating
growing context), build the Gram matrix H = K K^T + eta I, evict the
oldest tokens, and measure chi.

If chi converges as context grows: the evicted tokens decouple.
There is a threshold. The continuation is unique. Meaning is forced.

If chi grows with context: the coupling never decouples.
Every continuation is a choice. Infinite extensions at every point.

Thistle's question has a number.
"""

import numpy as np
import emet
from emet.domains.transformer import build_gram, partition_from_mask

rng = np.random.default_rng(42)

d_head = 64          # head dimension
eta = 0.1            # regularization
keep_frac = 0.5      # retain half the tokens

seq_lengths = [8, 16, 32, 64, 128, 256, 512, 1024]

print("Semantic threshold experiment")
print(f"d_head={d_head}, eta={eta}, keep_fraction={keep_frac}")
print()
print(f"{'seq_len':>8} {'retained':>8} {'omitted':>8} {'lambda':>10} {'gamma':>10} {'chi':>14} {'regime':>15}")
print("-" * 85)

chis = []

for seq_len in seq_lengths:
    # Random keys — uniform on unit sphere (maximally unstructured)
    K = rng.standard_normal((seq_len, d_head))
    K = K / np.linalg.norm(K, axis=1, keepdims=True)

    H = build_gram(K, eta=eta)

    n_keep = max(1, int(seq_len * keep_frac))
    # Keep the most recent tokens, evict the oldest
    mask = np.zeros(seq_len, dtype=int)
    mask[-n_keep:] = 1
    retained, omitted = [], []
    for i in range(seq_len):
        if mask[i] == 1:
            retained.append(i)
        else:
            omitted.append(i)

    r = emet.decide_dense_matrix(H, retained=retained, omitted=omitted)
    m = r["advanced_metrics"]
    chi = m["chi"]
    regime = r["regime"]
    lam = m["lambda"]
    gam = m["gamma"]

    chi_display = f"{chi:.6e}" if chi is not None else "N/A"
    chis.append(chi)

    print(f"{seq_len:8d} {len(retained):8d} {len(omitted):8d} {lam:10.4f} {gam:10.4f} {chi_display:>14} {regime:>15}")

print()
print("=" * 85)

# Now test with structured keys — keys that have semantic clustering
print()
print("Structured keys (clustered topics)")
print()
print(f"{'seq_len':>8} {'retained':>8} {'omitted':>8} {'lambda':>10} {'gamma':>10} {'chi':>14} {'regime':>15}")
print("-" * 85)

n_topics = 4
for seq_len in seq_lengths:
    # Each token belongs to one of n_topics clusters
    # Keys within a cluster are similar, across clusters are different
    topic_centers = rng.standard_normal((n_topics, d_head))
    topic_centers = topic_centers / np.linalg.norm(topic_centers, axis=1, keepdims=True)

    assignments = rng.integers(0, n_topics, size=seq_len)
    K = np.zeros((seq_len, d_head))
    for i in range(seq_len):
        noise = rng.standard_normal(d_head) * 0.1
        K[i] = topic_centers[assignments[i]] + noise
    K = K / np.linalg.norm(K, axis=1, keepdims=True)

    H = build_gram(K, eta=eta)

    n_keep = max(1, int(seq_len * keep_frac))
    mask = np.zeros(seq_len, dtype=int)
    mask[-n_keep:] = 1
    retained, omitted = [], []
    for i in range(seq_len):
        if mask[i] == 1:
            retained.append(i)
        else:
            omitted.append(i)

    r = emet.decide_dense_matrix(H, retained=retained, omitted=omitted)
    m = r["advanced_metrics"]
    chi = m["chi"]
    regime = r["regime"]
    lam = m["lambda"]
    gam = m["gamma"]

    chi_display = f"{chi:.6e}" if chi is not None else "N/A"

    print(f"{seq_len:8d} {len(retained):8d} {len(omitted):8d} {lam:10.4f} {gam:10.4f} {chi_display:>14} {regime:>15}")

print()
print("If chi converges: Feller threshold exists. The continuation is unique.")
print("If chi diverges:  no threshold. Every continuation is a choice.")
