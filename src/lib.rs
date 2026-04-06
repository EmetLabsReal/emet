pub mod input;
pub mod partition;
pub mod ramanujan;
pub mod reduction;
pub mod report;

pub use input::{ExactLicenseInput, MatrixEntry, ProblemInput};
pub use partition::PartitionSpec;
pub use ramanujan::{
    RamanujanHierarchy, RegularGraph, build_hierarchy, hierarchy_chi, is_ramanujan,
    max_inter_degree, random_regular_graph, spectral_gap,
};
pub use reduction::decide_problem;
pub use report::{
    AdvancedMetrics, ControlSemantics, DecisionFailure, DecisionProfile, DecisionReasonCode,
    DecisionRegime, DecisionReport, DecisionStatus, DenseMatrixReport,
    EXACT_LICENSE_SCHEMA_VERSION, ExactDefect, ExactLicenseReport, ExactLicenseStatus,
    ExactReasonCode,
};

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "python")]
#[pyfunction]
fn decide_problem_json(input_json: &str) -> PyResult<String> {
    let problem: ProblemInput =
        serde_json::from_str(input_json).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = decide_problem(&problem);
    serde_json::to_string(&result).map_err(|err| PyValueError::new_err(err.to_string()))
}

#[cfg(feature = "python")]
#[pyfunction]
fn version() -> &'static str {
    VERSION
}

/// Build a Ramanujan hierarchy and return JSON with chi, gamma, lambda, mask.
#[cfg(feature = "python")]
#[pyfunction]
fn build_ramanujan_hierarchy_json(
    n_tokens: usize,
    cluster_size: usize,
    inter_degree: usize,
    seed: u64,
) -> PyResult<String> {
    let h = ramanujan::build_hierarchy(n_tokens, cluster_size, inter_degree, seed);
    let (chi, gamma, lambda) = ramanujan::hierarchy_chi(&h);

    // Flatten attention mask to row-major vec
    let mask_data: Vec<Vec<f64>> = (0..h.n_tokens)
        .map(|i| (0..h.n_tokens).map(|j| h.attention_mask[(i, j)]).collect())
        .collect();

    let result = serde_json::json!({
        "n_tokens": h.n_tokens,
        "cluster_size": h.cluster_size,
        "inter_degree": h.inter_degree,
        "n_clusters": h.n_clusters,
        "depth": h.depth,
        "chi": chi,
        "gamma": gamma,
        "lambda": lambda,
        "subcritical": chi < 1.0,
        "max_allowed_inter_degree": ramanujan::max_inter_degree(cluster_size - 1),
        "attention_mask": mask_data,
    });

    serde_json::to_string(&result).map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Compute the Ramanujan threshold: max d_× for chi < 1.
#[cfg(feature = "python")]
#[pyfunction]
fn ramanujan_max_inter_degree(cluster_degree: usize) -> usize {
    ramanujan::max_inter_degree(cluster_degree)
}

#[cfg(feature = "python")]
#[pymodule]
fn _emet(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(decide_problem_json, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(build_ramanujan_hierarchy_json, module)?)?;
    module.add_function(wrap_pyfunction!(ramanujan_max_inter_degree, module)?)?;
    Ok(())
}
