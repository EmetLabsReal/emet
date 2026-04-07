# Quickstart

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/EmetLabsReal/emet.git && cd emet
git checkout production
uv venv && source .venv/bin/activate
uv pip install maturin numpy
maturin develop --release
make test
```

Requirements: Rust (2024 edition) with LAPACK, Python 3.10+, numpy.

## TUI

```bash
uv pip install textual
emet tui
```

Three input fields:

- **source** — a domain name, partial match, or path to a JSON file
- **retained** — comma-separated indices to keep
- **omitted** — comma-separated indices to discard

Press **Enter** to run. Results appear below. Press **Ctrl+S** to seal with SHA-256.

### Source field

The source field accepts domain names, partial matches, or file paths:

```
torus           → Pinched Torus
yang            → Yang-Mills SU(2)
su(2)           → Yang-Mills SU(2)
channel         → Quantum Channel
ising           → Ising Block-Spin
random          → Random Hermitian
covar           → Sample Covariance
laplacian       → Graph Laplacian
surgery         → Surgery
schumann        → Schumann Cavity
transformer     → Transformer Attention
ramanujan       → Ramanujan Hierarchy
lattice         → Lattice Transfer Matrix
adm             → ADM / FLRW
kuramoto        → Kuramoto Oscillators
hat             → Mexican Hat
4d              → 4D Yang-Mills Torus
```

When a domain is matched, the retained/omitted fields auto-fill with the correct default partition. You can override them before pressing Enter.

File paths work too:
```
tests/fixtures/subcritical.json
/absolute/path/to/problem.json
```

### Keybindings

| Key | Action |
|-----|--------|
| Enter | Run pipeline (decide) |
| Ctrl+S | Seal result (SHA-256 certificate) |
| Esc | Unfocus input (so `q` works) |
| Tab | Move between inputs |
| q | Quit (when no input is focused) |

### Torus visualization

For domains with a torus interpretation (torus, yang_mills, mexican_hat, torus_4d), a braille V_eff plot appears showing the effective potential on the pinched torus. The plot updates after each run, mapping chi to the geometric regime:

- Green filled area above V=0: centrifugal barrier (subcritical, licensed)
- Red area below V=0: attractive well (supercritical, unlicensed)
- Dashed line: V=0 (Feller threshold)

## CLI

```bash
emet decide tests/fixtures/subcritical.json --json
emet decide tests/fixtures/supercritical.json --pretty
emet certify tests/fixtures/subcritical.json
emet certify tests/fixtures/subcritical.json --pretty
```

## Python API

```python
import numpy as np, emet

H = np.array([[4.0, 0.1, 0.2],
              [0.1, 3.0, 0.0],
              [0.2, 0.0, 5.0]])

r = emet.decide_dense_matrix(H, retained=[0, 1], omitted=[2])
```

| Key | Value | Meaning |
|-----|-------|---------|
| `r["regime"]` | `"subcritical"` | chi < 1, safe to reduce |
| `r["advanced_metrics"]["chi"]` | `0.0016` | coupling/stability ratio squared |
| `r["advanced_metrics"]["gamma"]` | `5.0` | min singular value of H_QQ |
| `r["advanced_metrics"]["lambda"]` | `0.2` | max singular value of H_PQ |

## Certification

```python
from emet.certificate import certify, verify, to_json

cert = certify(H, retained=[0,1], omitted=[2], report=r)
cert.seal        # SHA-256
cert.licensed    # True
cert.regime      # "subcritical"

assert verify(cert, H, retained=[0,1], omitted=[2])
print(to_json(cert, pretty=True))
```

The seal covers input hash, partition hash, chi/gamma/lambda, regime, Kahan status, and reduced matrix hash. Timestamp excluded so same inputs always produce same seal.

## Kahan certification

```python
from emet.domains.kahan import certified_subcritical

m = r["advanced_metrics"]
k = certified_subcritical(m["chi"], m["gamma"], m["lambda"])
k["certified"]    # True: chi + rounding < 1
```

## Domain adapters

18 domains. Each constructs H and a partition for a specific problem:

| Key | Name | What it tests |
|-----|------|--------------|
| `torus` | Pinched Torus | Feller threshold, centrifugal barrier |
| `yang_mills` | Yang-Mills SU(2) | Confinement as licensed reduction |
| `yang_mills_sun` | Yang-Mills SU(N) | SU(2)/SU(3) generalization |
| `torus_4d` | 4D Yang-Mills Torus | Dimension-dependent pinching |
| `mexican_hat` | Mexican Hat | Symmetry breaking forces potential |
| `quantum_channel` | Quantum Channel | Signal/error decoupling |
| `transformer` | Transformer Attention | KV cache eviction |
| `graph_laplacian` | Graph Laplacian | Network coarsening |
| `surgery` | Surgery | Post-fracture reconstruction |
| `schumann` | Schumann Cavity | Thin-shell confinement |
| `adm_flrw` | ADM / FLRW | Cosmological constraint decoupling |
| `kuramoto` | Kuramoto Oscillators | Hermite mode truncation |
| `lattice` | Lattice Transfer Matrix | Representation truncation |
| `ising` | Ising Block-Spin | Block-spin RG at T_c |
| `ramanujan` | Ramanujan Hierarchy | Spectral gap tightness |
| `random_hermitian` | Random Hermitian | Synthetic baseline |
| `covariance` | Sample Covariance | Latent variable reduction |
| `from_file` | Load from File | Your matrix, certified |

## Next steps

- [Architecture](architecture.md) — layers, data flow, project structure
- [Domain guide](domains.md) — adapter details
- [API reference](api.md) — Python, Rust CLI, Rust library
