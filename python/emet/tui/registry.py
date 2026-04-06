"""Domain adapter registry for the TUI.

Each domain module gets a DomainDescriptor that normalizes its build
function to always return (H, retained, omitted). Adapters with
heterogeneous signatures are wrapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ParamSpec:
    name: str
    label: str
    type: type
    default: Any
    min_val: float | None = None
    max_val: float | None = None
    sweepable: bool = True
    choices: list[Any] | None = None


@dataclass
class DomainDescriptor:
    key: str
    name: str
    description: str
    params: list[ParamSpec]
    build_fn: Callable[..., tuple[np.ndarray, list[int], list[int]]]
    has_torus_plot: bool = False
    beta_param: str | None = None
    extra_metrics: list[str] = field(default_factory=list)


def _build_yang_mills(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.yang_mills import build_plaquette_blocks
    H, ret, omit, _j_values = build_plaquette_blocks(
        g_squared=kw["g_squared"], j_max=kw["j_max"], j_cut=kw["j_cut"],
    )
    return H, ret, omit


def _build_yang_mills_sun(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.yang_mills_sun import build_sun_plaquette
    H, ret, omit, _info = build_sun_plaquette(
        n=kw["gauge_n"], g_squared=kw["g_squared"],
        max_irrep=kw["max_irrep"], cut_index=kw["cut_index"],
    )
    return H, ret, omit


def _build_torus(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.torus import build_torus_operator
    return build_torus_operator(
        beta=kw["beta"], n_valley=kw["n_valley"], n_barrier=kw["n_barrier"],
    )


def _build_torus_4d(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.torus_4d import build_4d_torus_operator
    H, ret, omit, _params = build_4d_torus_operator(
        g_squared=kw["g_squared"], d=kw["d"],
        n_valley=kw["n_valley"], n_barrier=kw["n_barrier"],
    )
    return H, ret, omit


def _build_mexican_hat(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.torus import build_torus_operator
    return build_torus_operator(
        beta=kw["beta"], n_valley=kw["n_valley"], n_barrier=kw["n_barrier"],
    )


def _build_graph_laplacian(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.graph_laplacian import build_two_clique_laplacian
    return build_two_clique_laplacian(
        n_retained=kw["n_retained"], n_omitted=kw["n_omitted"],
        w_ret=kw["w_ret"], w_omit=kw["w_omit"], w_cross=kw["w_cross"],
    )


def _build_surgery(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.surgery import build_severed_double_well
    return build_severed_double_well(
        barrier_height=kw["barrier_height"], valley_energy=kw["valley_energy"],
        n_left=kw["n_left"], n_right=kw["n_right"],
    )


def _build_schumann(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.schumann import build_schumann_blocks
    H, ret, omit, _labels = build_schumann_blocks(
        delta_over_R=kw["delta_over_R"], n_angular=kw["n_angular"],
        n_radial=kw["n_radial"],
    )
    return H, ret, omit


def _build_adm_flrw(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.adm_flrw import build_adm_flrw_tower
    return build_adm_flrw_tower(
        a=kw["a"], ell_max=kw["ell_max"], lambda_coupling=kw["lambda_coupling"],
    )


def _build_kuramoto(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.kuramoto import build_generator, hermite_partition
    H = build_generator(
        n_theta=kw["n_theta"], n_hermite=kw["n_hermite"],
        mass_diag=kw["mass_diag"], gamma_excited=kw["gamma_excited"],
        theta_coupling=kw["theta_coupling"], streaming_scale=kw["streaming_scale"],
    )
    ret, omit = hermite_partition(kw["n_theta"], kw["n_hermite"])
    return H, ret, omit


def _build_lattice(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.lattice import build_transfer_matrix
    from emet.domains.yang_mills import partition_by_representation
    T = build_transfer_matrix(
        g_squared=kw["g_squared"], j_max=kw["j_max"], j_cut=kw["j_cut"],
    )
    j_values = []
    j = 0.0
    while j <= kw["j_max"] + 1e-10:
        j_values.append(j)
        j += 0.5
    ret, omit = partition_by_representation(j_values, kw["j_cut"])
    return T, ret, omit


def _build_transformer(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.transformer import build_gram
    rng = np.random.default_rng(kw.get("seed", 42))
    n = kw["n_tokens"]
    d = kw["d_head"]
    K = rng.standard_normal((n, d))
    H = build_gram(K, eta=kw["eta"])
    n_ret = min(d, n - 1)
    diag = np.abs(np.diag(H))
    order = np.argsort(-diag)
    ret = sorted(order[:n_ret].tolist())
    omit = sorted(order[n_ret:].tolist())
    return H, ret, omit


def _build_quantum_channel(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains import quantum_channel as qc
    channel_type = kw["channel_type"]
    p = kw["p"]
    builders = {
        "depolarizing": qc.choi_depolarizing,
        "dephasing": qc.choi_dephasing,
        "bit_flip": qc.choi_bit_flip,
        "amplitude_damping": qc.choi_amplitude_damping,
        "misaligned": qc.choi_misaligned,
    }
    if channel_type == "identity":
        choi = qc.choi_identity()
    elif channel_type in builders:
        choi = builders[channel_type](p)
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")
    return choi, qc.SIGNAL_INDICES, qc.ERROR_INDICES


def _build_ising(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.ising_block_spin import block_spin_partition
    result = block_spin_partition(
        Lx=kw["Lx"], Ly=kw["Ly"], bx=kw["bx"], by=kw["by"],
        beta=1.0 / kw["T"], J=kw.get("J", 1.0),
    )
    n = kw["Lx"] * kw["Ly"]
    from emet.domains.ising_block_spin import ising_energies
    configs, energies = ising_energies(kw["Lx"], kw["Ly"], kw.get("J", 1.0))
    log_w = -(1.0 / kw["T"]) * energies
    log_w -= np.max(log_w)
    weights = np.exp(log_w)
    probs = weights / np.sum(weights)
    mean_sigma = configs.T @ probs
    weighted = configs * np.sqrt(probs)[:, None]
    cov = weighted.T @ weighted - np.outer(mean_sigma, mean_sigma)
    cov = 0.5 * (cov + cov.T)
    return cov, result["retained"], result["omitted"]


def _build_random_hermitian(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    """Generate a random Hermitian matrix with a diagonal-dominant retained block."""
    n = kw["dim"]
    n_ret = kw["n_retained"]
    coupling = kw["coupling"]
    seed = kw.get("seed", 42)
    rng = np.random.default_rng(seed)

    # Random symmetric matrix
    A = rng.standard_normal((n, n))
    H = (A + A.T) / 2.0

    # Make diagonal dominant so H_QQ is invertible
    H += np.diag(np.full(n, 5.0))

    # Scale off-diagonal coupling between retained and omitted
    ret = list(range(n_ret))
    omit = list(range(n_ret, n))
    H[np.ix_(ret, omit)] *= coupling
    H[np.ix_(omit, ret)] *= coupling

    return H, ret, omit


def _build_from_file(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    """Load a matrix from a JSON file.

    Expected JSON format (same as CLI):
      {"matrix": [[...], ...], "retained": [...], "omitted": [...]}
    or sparse:
      {"dimension": N, "entries": [{"row":i,"col":j,"value":v},...],
       "retained": [...], "omitted": [...]}
    """
    import json
    from pathlib import Path

    path = Path(kw["file_path"])
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    data = json.loads(path.read_text())

    if "entries" in data:
        dim = data["dimension"]
        H = np.zeros((dim, dim))
        for e in data["entries"]:
            H[e["row"], e["col"]] = e["value"]
    elif "matrix" in data:
        H = np.array(data["matrix"], dtype=float)
    else:
        raise ValueError("JSON must have 'matrix' or 'entries' key")

    retained = data["retained"]
    omitted = data["omitted"]
    return H, retained, omitted


def _build_covariance(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    """Generate a sample covariance matrix — generic statistical reduction.

    Models: you have p observed variables and q latent/nuisance variables.
    The question is whether the latent sector can be reduced away.
    """
    p = kw["n_observed"]
    q = kw["n_latent"]
    n_samples = kw["n_samples"]
    signal_strength = kw["signal_strength"]
    seed = kw.get("seed", 42)
    rng = np.random.default_rng(seed)

    n = p + q
    # Generate data with signal in the observed block
    X = rng.standard_normal((n_samples, n))
    # Add correlated signal to observed variables
    signal = rng.standard_normal((n_samples, 1)) * signal_strength
    X[:, :p] += signal

    cov = (X.T @ X) / n_samples
    cov = 0.5 * (cov + cov.T)

    ret = list(range(p))
    omit = list(range(p, n))
    return cov, ret, omit


def _build_ramanujan(**kw: Any) -> tuple[np.ndarray, list[int], list[int]]:
    from emet.domains.ramanujan import build_hierarchy
    h = build_hierarchy(
        n_tokens=kw["n_tokens"], cluster_size=kw["cluster_size"],
        inter_degree=kw["inter_degree"], seed=kw.get("seed", 42),
    )
    mask = h["attention_mask"]
    n = h["n_tokens"]
    cs = kw["cluster_size"]
    degree_vec = mask.sum(axis=1)
    laplacian = np.diag(degree_vec) - mask
    ret = list(range(cs))
    omit = list(range(cs, min(2 * cs, n)))
    block_idx = ret + omit
    block = laplacian[np.ix_(block_idx, block_idx)]
    local_ret = list(range(len(ret)))
    local_omit = list(range(len(ret), len(block_idx)))
    return block, local_ret, local_omit


DOMAINS: list[DomainDescriptor] = [
    DomainDescriptor(
        key="torus",
        name="Pinched Torus",
        description="Centrifugal barrier, Feller threshold, valley confinement",
        has_torus_plot=True,
        beta_param="beta",
        params=[
            ParamSpec("beta", "Weight exponent beta", float, 3.0, 0.1, 10.0),
            ParamSpec("n_valley", "Valley modes", int, 6, 2, 20, sweepable=False),
            ParamSpec("n_barrier", "Barrier modes", int, 4, 2, 20, sweepable=False),
        ],
        build_fn=_build_torus,
    ),
    DomainDescriptor(
        key="yang_mills",
        name="Yang-Mills SU(2)",
        description="Kogut-Susskind plaquette, Casimir barrier, confinement",
        has_torus_plot=True,
        beta_param="g_squared",
        extra_metrics=["mass_gap"],
        params=[
            ParamSpec("g_squared", "Coupling g^2", float, 8.0, 0.1, 100.0),
            ParamSpec("j_max", "Truncation j_max", float, 4.0, 1.0, 20.0),
            ParamSpec("j_cut", "Partition cut j_cut", float, 1.0, 0.5, 10.0),
        ],
        build_fn=_build_yang_mills,
    ),
    DomainDescriptor(
        key="yang_mills_sun",
        name="Yang-Mills SU(N)",
        description="SU(2) and SU(3) Kogut-Susskind with Casimir ordering",
        extra_metrics=["mass_gap"],
        params=[
            ParamSpec("gauge_n", "Gauge group SU(N)", int, 2, 2, 3, choices=[2, 3]),
            ParamSpec("g_squared", "Coupling g^2", float, 8.0, 0.1, 100.0),
            ParamSpec("max_irrep", "Max irrep label", int, 4, 2, 10, sweepable=False),
            ParamSpec("cut_index", "Partition cut index", int, 3, 1, 8, sweepable=False),
        ],
        build_fn=_build_yang_mills_sun,
    ),
    DomainDescriptor(
        key="torus_4d",
        name="4D Yang-Mills Torus",
        description="Dimension-dependent pinching, alpha = g^2/d",
        has_torus_plot=True,
        beta_param="g_squared",
        params=[
            ParamSpec("g_squared", "Coupling g^2", float, 3.0, 0.1, 20.0),
            ParamSpec("d", "Spacetime dimension", int, 4, 2, 6, choices=[2, 3, 4, 5, 6]),
            ParamSpec("n_valley", "Valley modes", int, 6, 2, 20, sweepable=False),
            ParamSpec("n_barrier", "Barrier modes", int, 4, 2, 20, sweepable=False),
        ],
        build_fn=_build_torus_4d,
    ),
    DomainDescriptor(
        key="mexican_hat",
        name="Mexican Hat",
        description="Licensed reduction forces centrifugal potential for beta > 2",
        has_torus_plot=True,
        beta_param="beta",
        params=[
            ParamSpec("beta", "Weight exponent beta", float, 3.0, 0.1, 10.0),
            ParamSpec("n_valley", "Valley modes", int, 6, 2, 20, sweepable=False),
            ParamSpec("n_barrier", "Barrier modes", int, 4, 2, 20, sweepable=False),
        ],
        build_fn=_build_mexican_hat,
    ),
    DomainDescriptor(
        key="quantum_channel",
        name="Quantum Channel",
        description="Choi matrix, signal/error partition, Shor-Preskill rate",
        extra_metrics=["qber", "key_rate"],
        params=[
            ParamSpec("channel_type", "Channel type", str, "depolarizing",
                      choices=["identity", "depolarizing", "dephasing", "bit_flip",
                               "amplitude_damping", "misaligned"]),
            ParamSpec("p", "Error rate p", float, 0.1, 0.0, 1.0),
        ],
        build_fn=_build_quantum_channel,
    ),
    DomainDescriptor(
        key="transformer",
        name="Transformer Attention",
        description="Key cache Gram matrix, PCA eviction partition",
        params=[
            ParamSpec("n_tokens", "Sequence length", int, 64, 4, 512),
            ParamSpec("d_head", "Head dimension", int, 32, 4, 256),
            ParamSpec("eta", "Regularization eta", float, 1.0, 0.01, 10.0),
            ParamSpec("seed", "Random seed", int, 42, 0, 9999, sweepable=False),
        ],
        build_fn=_build_transformer,
    ),
    DomainDescriptor(
        key="graph_laplacian",
        name="Graph Laplacian",
        description="Two-clique Laplacian, tunable cross-coupling",
        params=[
            ParamSpec("n_retained", "Retained clique size", int, 5, 2, 20, sweepable=False),
            ParamSpec("n_omitted", "Omitted clique size", int, 5, 2, 20, sweepable=False),
            ParamSpec("w_ret", "Retained intra-weight", float, 1.0, 0.01, 10.0),
            ParamSpec("w_omit", "Omitted intra-weight", float, 2.0, 0.01, 10.0),
            ParamSpec("w_cross", "Cross-coupling weight", float, 1e-12, 1e-15, 1.0),
        ],
        build_fn=_build_graph_laplacian,
    ),
    DomainDescriptor(
        key="surgery",
        name="Surgery",
        description="Severed double-well, post-fracture reconstruction",
        params=[
            ParamSpec("barrier_height", "Barrier height", float, 4.0, 0.1, 20.0),
            ParamSpec("valley_energy", "Valley energy", float, 1.0, 0.1, 10.0),
            ParamSpec("n_left", "Left well modes", int, 3, 2, 10, sweepable=False),
            ParamSpec("n_right", "Right well modes", int, 3, 2, 10, sweepable=False),
        ],
        build_fn=_build_surgery,
    ),
    DomainDescriptor(
        key="schumann",
        name="Schumann Cavity",
        description="Earth-ionosphere shell, thin-shell confinement",
        params=[
            ParamSpec("delta_over_R", "Thickness ratio delta/R", float, 0.01, 0.001, 0.1),
            ParamSpec("n_angular", "Angular modes", int, 8, 2, 16, sweepable=False),
            ParamSpec("n_radial", "Radial modes", int, 5, 2, 10, sweepable=False),
        ],
        build_fn=_build_schumann,
    ),
    DomainDescriptor(
        key="adm_flrw",
        name="ADM / FLRW",
        description="ADM constraint tower on S^3 with dust, TT/constraint partition",
        params=[
            ParamSpec("a", "Scale factor a", float, 2.0, 0.5, 5.0),
            ParamSpec("ell_max", "Max angular mode", int, 10, 2, 20, sweepable=False),
            ParamSpec("lambda_coupling", "Coupling lambda", float, 1.0, 0.01, 10.0),
        ],
        build_fn=_build_adm_flrw,
    ),
    DomainDescriptor(
        key="kuramoto",
        name="Kuramoto Oscillators",
        description="Kramers-Moyal generator, Hermite mode partition",
        params=[
            ParamSpec("n_theta", "Phase sites", int, 4, 2, 16),
            ParamSpec("n_hermite", "Hermite modes per site", int, 3, 2, 8),
            ParamSpec("mass_diag", "Mass diagonal", float, 2.0, 0.1, 10.0),
            ParamSpec("gamma_excited", "Excited mode decay", float, 5.0, 0.1, 20.0),
            ParamSpec("theta_coupling", "Phase coupling", float, 0.1, 0.0, 5.0),
            ParamSpec("streaming_scale", "Streaming scale", float, 0.05, 0.0, 2.0),
        ],
        build_fn=_build_kuramoto,
    ),
    DomainDescriptor(
        key="lattice",
        name="Lattice Transfer Matrix",
        description="Transfer matrix T = exp(-H), thermodynamic limit",
        extra_metrics=["transfer_gap"],
        params=[
            ParamSpec("g_squared", "Coupling g^2", float, 8.0, 0.1, 100.0),
            ParamSpec("j_max", "Truncation j_max", float, 3.0, 1.0, 10.0),
            ParamSpec("j_cut", "Partition cut j_cut", float, 1.0, 0.5, 5.0),
        ],
        build_fn=_build_lattice,
    ),
    DomainDescriptor(
        key="ising",
        name="Ising Block-Spin",
        description="2D Ising covariance, block-spin RG, capacity transition at T_c",
        params=[
            ParamSpec("Lx", "Lattice width", int, 4, 2, 4, choices=[2, 4], sweepable=False),
            ParamSpec("Ly", "Lattice height", int, 4, 2, 4, choices=[2, 4], sweepable=False),
            ParamSpec("bx", "Block width", int, 2, 2, 2, sweepable=False),
            ParamSpec("by", "Block height", int, 2, 2, 2, sweepable=False),
            ParamSpec("T", "Temperature", float, 2.27, 0.5, 5.0),
        ],
        build_fn=_build_ising,
    ),
    DomainDescriptor(
        key="ramanujan",
        name="Ramanujan Hierarchy",
        description="Spectral gap guarantee, chi bound, attention mask",
        params=[
            ParamSpec("n_tokens", "Number of tokens", int, 64, 8, 256),
            ParamSpec("cluster_size", "Cluster size", int, 16, 4, 64),
            ParamSpec("inter_degree", "Inter-cluster degree", int, 1, 0, 10),
            ParamSpec("seed", "Random seed", int, 42, 0, 9999, sweepable=False),
        ],
        build_fn=_build_ramanujan,
    ),
    # --- General-purpose domains (not physics-specific) ---
    DomainDescriptor(
        key="random_hermitian",
        name="Random Hermitian",
        description="Random symmetric matrix with tunable coupling between retained/omitted blocks",
        params=[
            ParamSpec("dim", "Matrix dimension", int, 20, 4, 200, sweepable=False),
            ParamSpec("n_retained", "Retained indices", int, 8, 1, 100, sweepable=False),
            ParamSpec("coupling", "Cross-block coupling", float, 0.1, 0.001, 5.0),
            ParamSpec("seed", "Random seed", int, 42, 0, 9999, sweepable=False),
        ],
        build_fn=_build_random_hermitian,
    ),
    DomainDescriptor(
        key="covariance",
        name="Sample Covariance",
        description="Observed/latent covariance — can the latent sector be reduced?",
        params=[
            ParamSpec("n_observed", "Observed variables", int, 8, 2, 50, sweepable=False),
            ParamSpec("n_latent", "Latent variables", int, 12, 2, 50, sweepable=False),
            ParamSpec("n_samples", "Sample count", int, 200, 20, 10000),
            ParamSpec("signal_strength", "Signal strength", float, 2.0, 0.0, 10.0),
            ParamSpec("seed", "Random seed", int, 42, 0, 9999, sweepable=False),
        ],
        build_fn=_build_covariance,
    ),
    DomainDescriptor(
        key="from_file",
        name="Load from File",
        description="Load H, retained, omitted from a JSON file (same format as CLI)",
        params=[
            ParamSpec("file_path", "Path to JSON file", str, "problem.json",
                      sweepable=False),
        ],
        build_fn=_build_from_file,
    ),
]

DOMAIN_MAP: dict[str, DomainDescriptor] = {d.key: d for d in DOMAINS}


def get_domain(key: str) -> DomainDescriptor:
    return DOMAIN_MAP[key]


def domain_keys() -> list[str]:
    return [d.key for d in DOMAINS]
