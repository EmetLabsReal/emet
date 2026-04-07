<p align="center">
  <img src="docs/assets/noah.jpg" width="200" alt="emet" />
</p>

<h1 align="center">emet</h1>

<p align="center">
  <a href="https://github.com/EmetLabsReal/emet/actions/workflows/ci.yml"><img src="https://github.com/EmetLabsReal/emet/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /></a>
  <br />
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-2024%20edition-dca735?logo=rust&logoColor=black" alt="Rust" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python" /></a>
</p>

Certified spectral reduction for block-partitioned symmetric operators.

`emet` answers one question fast: **can you safely reduce this model?**

Given a symmetric matrix and a retained/omitted partition, it computes
\[
\chi = \left(\frac{\lambda}{\gamma}\right)^2
\]
then classifies reduction risk and optionally emits an auditable certificate seal.

<p align="center">
  <img src="docs/assets/tui_preview.png" width="900" alt="emet tui" />
</p>

## Start in 60 seconds

```bash
git clone https://github.com/EmetLabsReal/emet.git && cd emet
uv venv && source .venv/bin/activate
uv pip install maturin numpy
maturin develop --release
python -c "import numpy as np, emet; H=np.array([[4.0,0.1,0.2],[0.1,3.0,0.0],[0.2,0.0,5.0]]); r=emet.decide_dense_matrix(H,retained=[0,1],omitted=[2]); print(r['regime'], r['advanced_metrics']['chi'])"
```

If this prints a regime and `chi`, you are live.

## Why emet

Model reduction is often done ad hoc: drop states, hope behavior remains stable.
`emet` makes that step explicit, measurable, and auditable.

- **Measure coupling risk** via \(\chi\), where \(\lambda\) is inter-block coupling and \(\gamma\) is intra-retained spectral floor.
- **Classify regime** (`subcritical`, `critical`, `supercritical`) before reducing.
- **Issue a certificate** with a SHA-256 seal over operator, partition, and IEEE-754 envelope.

## Regime semantics

- **`subcritical`**: reduction is licensed; Schur correction remains controlled.
- **`critical`**: near threshold; reduce only with tighter validation.
- **`supercritical`**: reduction not licensed; omitted sector materially affects retained dynamics.

## Trust model (what gets sealed)

The certificate seal is SHA-256 over:

- operator payload
- retained/omitted partition
- IEEE-754 floating-point envelope

## Install

### Quick install (local dev)

```bash
git clone https://github.com/EmetLabsReal/emet.git && cd emet
uv venv && source .venv/bin/activate
uv pip install maturin numpy
maturin develop --release
```

### Version matrix

- Python: `3.10+`
- Rust: edition `2024`
- Tooling: `uv`, `maturin`

## Python usage

```python
import numpy as np, emet
from emet.certificate import certify

H = np.array([[4.0, 0.1, 0.2],
              [0.1, 3.0, 0.0],
              [0.2, 0.0, 5.0]])

r = emet.decide_dense_matrix(H, retained=[0, 1], omitted=[2])
cert = certify(H, retained=[0, 1], omitted=[2], report=r)

r["regime"]                    # "subcritical"
r["advanced_metrics"]["chi"]   # 0.0016
cert.seal                      # "a3f8..."
```

Licensed means Schur-complement reduction is expected to preserve spectral structure within bounded error under the certified envelope.

## CLI

```bash
emet decide my_model.json --json
emet certify my_model.json --pretty
```

Example `emet certify --pretty` output shape:

```json
{
  "regime": "subcritical",
  "advanced_metrics": {
    "chi": 0.0016
  },
  "licensed": true,
  "seal": "a3f8..."
}
```

## TUI

```bash
uv pip install "emet[tui]"
emet tui
```

18 built-in domains. Bring your own matrix as JSON. See [quickstart](docs/quickstart.md).

## When not to reduce

Do not reduce by default when:

- regime is `supercritical`
- retained block has a weak or poorly estimated spectral floor
- partition is unstable under perturbation of the operator
- downstream decisions require full-state traceability

In these cases, keep full dimensionality or redesign the partition.

## Test

```bash
make test        # 274 tests (Rust + Python)
make examples    # runnable examples
```

## Docs

- [Quickstart](docs/quickstart.md) — install, first reduction, domain list
- [API reference](docs/api.md) — Python, Rust CLI, Rust library
- [Domain guide](docs/domains.md) — 18 adapters, writing a new adapter
- [Architecture](docs/architecture.md) — engine layers, phase portrait pipeline, data flow

## License

MIT
