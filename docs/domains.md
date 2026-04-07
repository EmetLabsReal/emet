# Domain adapters

## What a domain adapter does

A domain adapter translates a physical system into the universal input format:
a Hermitian matrix $H$ and a partition $(P, Q)$. The engine then computes $\chi$.

```python
# Every adapter returns this:
H: np.ndarray          # Hermitian matrix (n x n)
retained: list[int]    # indices in P
omitted: list[int]     # indices in Q
```

The adapter constructs H from domain-specific parameters.
The engine does not know what the matrix represents.

## Existing adapters

### `yang_mills` — SU(2) Kogut-Susskind plaquette

```python
from emet.domains.yang_mills import build_plaquette_blocks
H, ret, omit, labels = build_plaquette_blocks(g2=16.0, j_max=10, j_cut=1)
```

- **H**: tridiagonal, diagonal $= g^2 j(j+1)/2$, off-diagonal $= -1/g^2$
- **P**: representations j <= j_cut
- **Q**: representations j > j_cut
- **$\chi$**: $\sim (g^2)^{-4}$, omission exponent $2$

### `yang_mills_sun` — SU(N) generalization

```python
from emet.domains.yang_mills_sun import build_sun_blocks
H, ret, omit, labels = build_sun_blocks(N=3, g2=16.0, j_max=10, j_cut=1)
```

Same structure as SU(2) with SU(N) Casimir eigenvalues.

### `quantum_channel` — Choi matrix

```python
from emet.domains.quantum_channel import choi_matrix_blocks
H, ret, omit = choi_matrix_blocks("depolarizing", p=0.1)
```

- **H**: 4x4 Choi matrix of a single-qubit CPTP map
- **P**: signal indices {0, 3} (|00>, |11>)
- **Q**: error indices {1, 2} (|01>, |10>)
- **$\chi$**: $0$ for Pauli channels, $> 0$ for misaligned

Channel types: `"depolarizing"`, `"dephasing"`, `"bit_flip"`, `"amplitude_damping"`, `"hadamard_misaligned"`.

### `torus` — Pinched torus

```python
from emet.domains.torus import build_torus_blocks
H, ret, omit = build_torus_blocks(beta=1.5, n_grid=100)
```

- **H**: discretized Sturm-Liouville operator $-s^{-\beta}\,\tfrac{d}{ds}(s^\beta\,\tfrac{d}{ds})$
- **P**: interior grid points
- **Q**: boundary grid points
- **$\chi$**: $< 1$ forced when $\beta \geq 1$

### `torus_4d` — 4D torus with Mexican hat

```python
from emet.domains.torus_4d import build_4d_torus_blocks
H, ret, omit = build_4d_torus_blocks(alpha=0.3, n_grid=100)
```

### `mexican_hat` — Symmetry-breaking potential

```python
from emet.domains.mexican_hat import build_mexican_hat_blocks
H, ret, omit = build_mexican_hat_blocks(beta=2.5, n_grid=100)
```

### `lattice` — Transfer matrix

```python
from emet.domains.lattice import build_lattice_blocks
H, ret, omit = build_lattice_blocks(coupling=1.0, size=10)
```

### `surgery` — Two-state Markov generator

```python
from emet.domains.surgery import build_surgery_blocks
H, ret, omit = build_surgery_blocks(rate=0.5)
```

### `graph_laplacian` — Graph Laplacian

```python
from emet.domains.graph_laplacian import build_graph_laplacian_blocks
H, ret, omit = build_graph_laplacian_blocks(adjacency_matrix, partition)
```

### `kuramoto` — Kuramoto oscillators

```python
from emet.domains.kuramoto import build_kuramoto_blocks
H, ret, omit = build_kuramoto_blocks(frequencies, coupling=1.0)
```

### `transformer` — Attention Gram matrix

```python
from emet.domains.transformer import attention_gram_blocks
H, ret, omit = attention_gram_blocks(n_tokens=256, d_head=64, eta=0.1)
```

- **H**: $KK^T + \eta I$, where $K$ is the key matrix
- **P**: most recent $n/2$ tokens
- **Q**: oldest $n/2$ tokens
- **$\chi$**: diverges for $n > 2d$ (cave theorem)

### `schumann` — Schumann resonance cavity

```python
from emet.domains.schumann import build_schumann_blocks
H, ret, omit, labels = build_schumann_blocks(delta_over_R=0.01, n_angular=8, n_radial=5)
```

- **H**: Laplacian on a thin spherical shell $[R, R+\delta] \times S^2$, discretized in angular modes $n$ and radial modes $k$
- **P**: fundamental radial mode ($k=1$) for each angular mode — the Schumann spectrum
- **Q**: radial overtones ($k \geq 2$) — frozen out by shell thinness
- **$\chi$**: $\to 0$ as $\delta/R \to 0$ (thin shell forces radial sector past Feller)

The Schur complement on $P$ recovers the Schumann eigenfrequencies $f_n = (c/2\pi R)\sqrt{n(n+1)}$.

### `ramanujan` — Ramanujan hierarchy attention

```python
from emet.domains.ramanujan import build_hierarchy, chi_bound, max_inter_degree
h = build_hierarchy(n_tokens=1024, cluster_size=16, inter_degree=2, seed=42)
h["chi"]           # worst-case chi across cluster pairs
h["subcritical"]   # True when chi < 1
h["attention_mask"] # (1024, 1024) numpy array — use as attention mask
```

- **H**: Laplacian of the hierarchical graph. Intra-cluster: complete ($d = \text{cluster\_size} - 1$). Inter-cluster: sparse ($d_\times$).
- **P**: tokens in one cluster
- **Q**: tokens in an adjacent cluster
- **$\chi$**: $\leq (d_\times / (d - 2\sqrt{d-1}))^2$ (Ramanujan bound). Subcritical when $d_\times < d - 2\sqrt{d-1}$.

The Rust engine constructs the graph and computes the spectral gap. `max_inter_degree(cluster_size)` returns the largest $d_\times$ that guarantees $\chi < 1$.

### `kahan` — Kahan envelope certification

```python
from emet.domains.kahan import certified_subcritical
result = certified_subcritical(chi, gamma, lam)
```

Not a domain adapter in the usual sense. Wraps the Kahan arithmetic
to certify that $\chi < 1$ survives IEEE 754 rounding.

## Writing a new adapter

An adapter is a Python module in `python/emet/domains/` that:

1. Takes domain-specific parameters
2. Constructs a Hermitian matrix H as a numpy array
3. Returns (H, retained_indices, omitted_indices)

### Template

```python
"""Domain adapter for [your system]."""
import numpy as np


def build_blocks(param1: float, param2: int) -> tuple[np.ndarray, list[int], list[int]]:
    """Construct the Hermitian operator and partition.

    Parameters
    ----------
    param1 : float
        [description]
    param2 : int
        [description]

    Returns
    -------
    H : np.ndarray
        Hermitian matrix (n x n).
    retained : list[int]
        Indices in P (retained sector).
    omitted : list[int]
        Indices in Q (omitted sector).
    """
    n = ...
    H = np.zeros((n, n))
    # Fill H from domain physics
    # H must be Hermitian: H = H.conj().T

    retained = list(range(k))
    omitted = list(range(k, n))
    return H, retained, omitted
```

### Requirements

- $H$ must be Hermitian (real symmetric or complex Hermitian)
- $H$ must be a numpy array with dtype float64 or complex128
- `retained` and `omitted` must be disjoint and cover $\{0, \ldots, n-1\}$
- The adapter must not call `emet.decide_dense_matrix` itself

### Testing

Add a test in `python/tests/test_[domain].py`:

```python
import emet
from emet.domains.your_domain import build_blocks

def test_subcritical():
    H, ret, omit = build_blocks(param1=..., param2=...)
    r = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
    assert r["regime"] == "subcritical"
    assert r["advanced_metrics"]["chi"] < 1

def test_supercritical():
    H, ret, omit = build_blocks(param1=..., param2=...)
    r = emet.decide_dense_matrix(H, retained=ret, omitted=omit)
    assert r["regime"] == "supercritical"
```

Run with `make test` or `pytest python/tests/test_[domain].py`.
