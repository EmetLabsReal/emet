# API reference

## Python

### `emet.decide_dense_matrix`

```python
emet.decide_dense_matrix(
    matrix: np.ndarray,
    retained: list[int],
    omitted: list[int]
) -> dict
```

Compute the regime parameter $\chi$ and classify the reduction.

**Parameters:**

- `matrix`: Hermitian matrix $H$ as a numpy array ($n \times n$, dtype float64 or complex128).
- `retained`: list of indices in $P$ (retained sector).
- `omitted`: list of indices in $Q$ (omitted sector).

`retained` and `omitted` must be disjoint and cover {0, ..., n-1}.

**Returns:** dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `valid` | bool | True if $H_{QQ}$ is numerically invertible |
| `regime` | str | `"subcritical"`, `"supercritical"`, or `"pre_admissible"` |
| `advanced_metrics` | dict | Detailed spectral quantities (see below) |

**`advanced_metrics` dict:**

| Key | Type | Description |
|-----|------|-------------|
| `chi` | float | Regime parameter $(\lambda/\gamma)^2$ |
| `gamma` | float | $\sigma_{\min}(H_{QQ})$ |
| `lambda` | float | $\lVert H_{PQ}\rVert_2$ |
| `sigma_min_qq` | float | $\sigma_{\min}(H_{QQ})$ (same as `gamma`) |
| `delta` | float | $\lambda^2 / \gamma$ |
| `epsilon` | float | Kahan-style error bound |
| `security_margin` | float | $\gamma - \lambda$ |
| `q_inverse_bound` | float | $\lVert H_{QQ}^{-1}\rVert$ |
| `residual_window` | float | Gap between $\gamma$ and $\lambda$ |

**`reduced_matrix` dict** (the Schur complement $H_\text{eff}$, if subcritical):

| Key | Type | Description |
|-----|------|-------------|
| `rows` | int | Number of rows of $H_\text{eff}$ |
| `cols` | int | Number of columns of $H_\text{eff}$ |
| `data` | list[list[float]] | $H_\text{eff}$ as nested list |

**`partition_profile` dict:**

| Key | Type | Description |
|-----|------|-------------|
| `retained_dimension` | int | $|P|$ |
| `omitted_dimension` | int | $|Q|$ |
| `coupling_strength` | float | $\lVert H_{PQ}\rVert_2$ |
| `control_strength` | float | $\sigma_{\min}(H_{QQ})$ |
| `omitted_condition_number` | float | $\kappa(H_{QQ})$ |

**Regime classification:**

- `"subcritical"` ($\chi < 1$): the Schur complement is a faithful reduction. $H_\text{eff}$ is returned.
- `"supercritical"` ($\chi \geq 1$): the reduction is not faithful. The omitted sector couples too strongly.
- `"pre_admissible"`: $H_{QQ}$ is singular ($\gamma \leq 10^{-12}$) or numerically non-invertible. $\chi$ is undefined.

**Example:**

```python
import numpy as np, emet

H = np.diag([1.0, 2.0, 3.0])
H[0, 2] = H[2, 0] = 0.1

r = emet.decide_dense_matrix(H, retained=[0, 1], omitted=[2])
assert r["regime"] == "subcritical"
assert r["advanced_metrics"]["chi"] < 1
```

## Rust CLI

```bash
emet < input.json > output.json
```

**Input JSON format:**

```json
{
  "matrix": [[1.0, 0.0, 0.1], [0.0, 2.0, 0.0], [0.1, 0.0, 3.0]],
  "retained": [0, 1],
  "omitted": [2]
}
```

**Output JSON format:**

```json
{
  "valid": true,
  "regime": "subcritical",
  "advanced_metrics": {
    "chi": 0.0011,
    "gamma": 3.0,
    "lambda": 0.1,
    "condition_number": 1.0
  }
}
```

## Rust library

```rust
use emet::{decide_dense_matrix, DenseDecision};
use nalgebra::DMatrix;

let h = DMatrix::from_row_slice(3, 3, &[
    1.0, 0.0, 0.1,
    0.0, 2.0, 0.0,
    0.1, 0.0, 3.0,
]);

let result: DenseDecision = decide_dense_matrix(&h, &[0, 1], &[2]);
assert!(result.valid);
assert_eq!(result.regime, "subcritical");
```

## Phase portrait API

### `emet.phase_point`

```python
emet.phase_point(
    matrix: np.ndarray,
    retained: list[int],
    omitted: list[int],
    *,
    beta: float | None = None,
) -> tuple[PhasePoint, dict]
```

Decide and return the phase point plus the full report. This is the primary diagnostic entry point.

**Parameters:**

- `matrix`, `retained`, `omitted`: same as `decide_dense_matrix`.
- `beta`: weight exponent of the transverse measure. When known, enables the full A/B/C trichotomy.

**Returns:** `(PhasePoint, report_dict)`.

### `emet.PhasePoint`

```python
@dataclass(frozen=True)
class PhasePoint:
    gamma: float          # sigma_min(H_QQ)
    lambda_: float        # ||H_PQ||_2
    chi: float | None     # (lambda/gamma)^2
    beta: float | None    # weight exponent, if known
    regime: Regime        # A, B, C, subcritical, supercritical, or pre_admissible
    determinacy: bool     # does this question have a unique answer?
```

**Constructors:**

- `PhasePoint.from_report(report, beta=None)` — extract from a decision report dict.
- `PhasePoint.from_values(gamma, lambda_, beta=None)` — construct directly.

**Properties:**

- `licensed: bool` — chi < 1.
- `cap_zero: bool | None` — True if beta >= 1 or chi < 1. None if undetermined.

### `emet.Regime`

```python
class Regime(Enum):
    A = "A"                          # Cap > 0, beta < 1
    B = "B"                          # Cap = 0, V_eff <= 0, 1 <= beta <= 2
    C = "C"                          # Cap = 0, V_eff > 0, beta > 2
    SUBCRITICAL = "subcritical"      # chi < 1, beta unknown
    SUPERCRITICAL = "supercritical"  # chi >= 1, beta unknown
    PRE_ADMISSIBLE = "pre_admissible"  # gamma ~ 0, chi undefined
```

When beta is known, classification uses the full capacity-potential trichotomy. When beta is unknown, falls back to chi-based subcritical/supercritical.

### `emet.PhasePortrait`

```python
portrait = PhasePortrait()
portrait.add(point)
portrait.add_from_report(report, beta=3.0)
portrait.by_regime()                           # dict[Regime, list[PhasePoint]]
portrait.predict(gamma, lambda_, beta=None)    # Regime
portrait.query_neighborhood(gamma, lambda_, radius)  # list[PhasePoint]
```

### `emet.store.CertificateStore`

Append-only JSONL store. Each line is a sealed JSON certificate. The seal is the primary key. Content-addressed, deterministic, immutable.

```python
from emet.store import CertificateStore

store = CertificateStore("certs.jsonl")
store.append(cert, domain="yang_mills", params={"g_squared": 8.0}, beta=8.0)

# Spatial query: all certified points near (gamma, lambda)
store.query_neighborhood(gamma=15.0, lambda_=0.1, radius=1.0)

# Trajectory query: all points from a domain, ordered by parameter
store.query_trajectory("yang_mills", "g_squared")

# Full portrait from all stored certificates
portrait = store.portrait()

# Dedup by seal — returns False if already stored
store.append(cert)  # False (already exists)
```

No database. No server. No trust. Every line is independently verifiable. The file is a git-friendly log that anyone can fork.

## Certificate API (v2)

```python
from emet.certificate import certify, verify, to_json, from_json

cert = certify(H, retained, omitted, report, beta=3.0, domain="yang_mills")
cert.beta          # 3.0
cert.determinacy   # True
cert.regime        # "C"
cert.seal          # SHA-256 hex digest

verify(cert, H, retained, omitted)  # True
```

The seal covers: schema, emet_version, domain, input_hash, partition_hash, matrix_dimension, retained_dim, omitted_dim, chi, gamma, lambda, beta, regime, licensed, determinacy, kahan_certified, kahan_margin, reduced_matrix_hash.

## Domain adapter API

Each adapter in `python/emet/domains/` exports a `build_*_blocks` function:

```python
def build_blocks(...) -> tuple[np.ndarray, list[int], list[int]]:
    """Returns (H, retained, omitted)."""
```

Some adapters return a 4-tuple with labels:

```python
def build_plaquette_blocks(g2, j_max, j_cut) -> tuple[np.ndarray, list[int], list[int], list[str]]:
    """Returns (H, retained, omitted, labels)."""
```

## Kahan certification API

```python
from emet.domains.kahan import certified_subcritical

result = certified_subcritical(chi: float, gamma: float, lam: float) -> dict
```

**Returns:**

| Key | Type | Description |
|-----|------|-------------|
| `certified` | bool | True if $\chi + \epsilon < 1$ under IEEE 754 |
| `chi_upper` | float | Rigorous upper bound on $\chi$ including rounding |
| `chi_lower` | float | Rigorous lower bound |
| `epsilon` | float | Bound on arithmetic error |
