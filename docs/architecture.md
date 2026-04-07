# Architecture

## Overview

Emet answers one question: given a symmetric matrix and a partition into "keep" and "discard", is the reduction structurally stable? The answer is a point in the phase portrait — the (gamma, lambda) plane — certified, sealed, and stored.

Five layers, each usable independently:

```
┌─────────────────────────────────────────────────┐
│  TUI / CLI                                      │
│  Textual workstation + batch CLI + certificates  │
├─────────────────────────────────────────────────┤
│  Phase portrait + Certificate store              │
│  PhasePoint, PhasePortrait, JSONL seal store     │
├─────────────────────────────────────────────────┤
│  Python layer                                    │
│  18 domain adapters + Kahan certification        │
├─────────────────────────────────────────────────┤
│  Certificate system                              │
│  SHA-256 sealed reduction certificates           │
├─────────────────────────────────────────────────┤
│  Rust engine (nalgebra + LAPACK)                 │
│  SVD, Schur complement, regime classification    │
└─────────────────────────────────────────────────┘
```

## Data flow

```
Any source              Rust engine              Output
──────────             ───────────              ──────
  (H, P, Q)  ───────>  SVD(H_PQ) → lambda
                        SVD(H_QQ) → gamma
                        chi = (lambda/gamma)^2
                        if chi < 1:
                          H_eff = H_PP - H_PQ H_QQ^{-1} H_QP
                                                  ───────>  PhasePoint + report
                                                            ↓
                                                     certify → seal
                                                            ↓
                                                     store.append(cert)
```

The engine is domain-agnostic. It takes a symmetric matrix and a partition. Domain adapters, user-supplied JSON, or programmatic construction — the engine doesn't care where H comes from.

1. **Construct** the matrix H and the partition (P, Q) — from a domain adapter, a file, or your own code.

2. The **Rust engine** computes gamma, lambda, chi via SVD. If chi < 1, it computes the Schur complement H_eff and reports `subcritical`. If chi >= 1, it reports `supercritical`. If H_QQ is singular, it reports `pre_admissible`.

3. The **phase point** wraps (gamma, lambda, chi, beta) into a `PhasePoint`. When beta is known, the full trichotomy applies: A (Cap > 0, beta < 1), B (Cap = 0, V_eff <= 0, 1 <= beta <= 2), C (Cap = 0, V_eff > 0, beta > 2). Determinacy — whether the question has a unique answer — is a derived property.

4. The **Kahan envelope** certifies that chi < 1 survives IEEE 754 rounding.

5. The **certificate** seals the entire result (input hash, partition hash, metrics, beta, regime, determinacy, Kahan status, reduced matrix hash) into a SHA-256 seal.

6. The **store** appends the sealed certificate to a JSONL log. Deduplication is by seal — same inputs always produce the same seal.

## Phase portrait

The phase portrait is the (gamma, lambda) plane. Every certified reduction is a point in this plane. The regime is a consequence of the point's position:

- **chi = 1 boundary**: the curve lambda = gamma divides licensed from unlicensed.
- **beta thresholds**: beta = 1 (Feller) and beta = 2 (Mexican hat) further classify the licensed region into Regimes B and C.

```python
from emet.portrait import PhasePoint, Regime

pt = PhasePoint.from_values(gamma=5.0, lambda_=0.3, beta=3.0)
pt.regime      # Regime.C
pt.determinacy # True
pt.licensed    # True
pt.cap_zero    # True
```

The portrait accumulates points:

```python
from emet.portrait import PhasePortrait

portrait = PhasePortrait()
portrait.add(pt)
portrait.query_neighborhood(gamma=5.0, lambda_=0.3, radius=1.0)
portrait.predict(gamma=5.0, lambda_=0.3, beta=3.0)  # Regime.C
```

## Certificate store

The store is an append-only JSONL file. Each line is a self-contained JSON certificate with its SHA-256 seal as primary key. Content-addressed, deterministic, immutable.

```python
from emet.store import CertificateStore

store = CertificateStore("certs.jsonl")
store.append(cert, domain="yang_mills", params={"g_squared": 8.0}, beta=8.0)
store.query_trajectory("yang_mills", "g_squared")  # ordered by g^2
store.portrait()  # PhasePortrait from all stored certs
```

No database. No server. No trust. Every line is independently verifiable — re-derive the seal, check it matches. The spatial index is built lazily in memory. Even with a million certificates, brute-force scan on a JSONL file takes milliseconds. The portrait has two dimensions.

The hashes are the database. The seal is the primary key. The file is a git-friendly log that anyone can fork, verify, and extend.

## Rust engine

```
src/
  lib.rs          Public API: decide_dense_matrix, DenseDecision
  main.rs         CLI binary (reads JSON, writes JSON)
  input.rs        Input parsing and validation
  partition.rs    Partition construction and validation
  reduction.rs    Core algorithm: SVD, Schur complement, regime classification
  report.rs       Result formatting
  ramanujan.rs    Ramanujan hierarchy builder
```

## Python bindings

```
python/emet/
  __init__.py       PyO3 bridge + phase_point() convenience
  portrait.py       Regime, PhasePoint, PhasePortrait
  store.py          CertificateStore (JSONL + seal index)
  certificate.py    SHA-256 certificate system (v2: beta + determinacy)
  cli.py            CLI entry point (emet tui | emet decide | emet certify)
  domains/
    torus.py              Pinched torus with weighted measure
    yang_mills.py         SU(2) Kogut-Susskind plaquette
    yang_mills_sun.py     SU(N) generalization
    torus_4d.py           4D torus with dimension-dependent pinching
    mexican_hat.py        Symmetry-breaking effective potential
    quantum_channel.py    Choi matrix of CPTP maps
    transformer.py        Attention Gram matrix
    graph_laplacian.py    Two-clique graph Laplacian
    surgery.py            Severed double-well Markov generator
    schumann.py           Earth-ionosphere cavity
    adm_flrw.py           ADM constraint tower on FLRW
    kuramoto.py           Kuramoto oscillator coupling
    lattice.py            Transfer matrix on lattice
    ising_block_spin.py   2D Ising covariance, block-spin RG
    ramanujan.py          Ramanujan hierarchy (via Rust)
    kahan.py              Kahan envelope certification
  tui/
    app.py            EmetApp (Textual application)
    registry.py       DomainDescriptor + ParamSpec for 18 adapters
    workers.py        Threaded sweep and certify workers (beta-aware)
    emet.tcss         Textual CSS
    screens/
      home.py         Main workspace (beta extraction + seal)
      sweep.py        Parameter sweep with live DataTable
      certify.py      Full certification pipeline
      inspect.py      Detail view
    widgets/
      domain_list.py      OptionList of domains
      param_panel.py      Dynamic parameter inputs
      metrics_panel.py    chi/gamma/lambda display
      regime_badge.py     Color-coded regime label
      torus_plot.py       Braille V_eff visualization
      sweep_table.py      DataTable with regime colors
      kahan_panel.py      Kahan certification status
      certificate_view.py SHA-256 seal display
      matrix_view.py      Formatted matrix table
```

## Certificate system

`EmetCertificate` is a frozen dataclass with 21 fields (v2). The seal is a SHA-256 hash of a canonical JSON representation of all fields except timestamp (so the same inputs always produce the same seal).

Fields added in v2:
- `beta: float | None` — weight exponent of the transverse measure
- `determinacy: bool` — whether the question has a unique answer (Cap = 0)

The seal covers: schema, emet_version, domain, input_hash, partition_hash, matrix_dimension, retained_dim, omitted_dim, chi, gamma, lambda, beta, regime, licensed, determinacy, kahan_certified, kahan_margin, reduced_matrix_hash.

`certify(H, retained, omitted, report, beta=...)` produces a certificate.
`verify(cert, H, retained, omitted)` re-derives the seal and checks it matches.

## TUI architecture

The TUI is a Textual application with four screens:

- **Home**: domain input → decide → seal. Extracts beta from the domain's `beta_param` field when sealing. Displays regime, determinacy, and seal hashes.
- **Sweep**: Parameter sweep over a sweepable parameter. DataTable with color-coded regime column.
- **Certify**: Runs decide + certify pipeline. Passes beta from domain registry. Displays certificate with SHA-256 seal, Kahan status, and reduced matrix.
- **Inspect**: Detail view of last decision. Shows H_eff and full H as formatted tables.

The registry normalizes all 18 heterogeneous adapter signatures to `(H, retained, omitted)` via wrapper functions. Workers run in threads and extract beta from the domain's `beta_param` when available.

## Test infrastructure

```
tests/
  cli.rs                  Rust CLI integration test
  fixtures/               JSON test fixtures

python/tests/
  test_core.py            Core API tests
  test_domains.py         Domain adapter tests
  test_portrait.py        Phase portrait + regime classification (28 tests)
  test_store.py           Certificate store (7 tests)
  test_kahan.py           Kahan certification tests
  ...                     (308 tests total)

oracle/
  generate.py             Generate golden test vectors
  check.py                Validate current output against oracle
  manifest.json           Test case metadata
  cases/                  Golden vector files
```

## CI

GitHub Actions runs on every push:
1. `rust`: `cargo build --release && cargo test`
2. `python`: `pytest` on Python 3.10 and 3.12
3. `examples`: all 5 examples
4. `oracle`: golden vector checks
