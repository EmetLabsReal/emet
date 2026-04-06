use serde::{Deserialize, Serialize};
use sprs::{CsMat, TriMat};

use crate::partition::PartitionSpec;
use crate::report::{ExactDefect, ExactLicenseStatus, ExactReasonCode};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct MatrixEntry {
    pub row: usize,
    pub col: usize,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ExactLicenseInput {
    #[serde(default)]
    pub schema_version: Option<String>,
    #[serde(default)]
    pub status: Option<ExactLicenseStatus>,
    #[serde(default)]
    pub reason_code: Option<ExactReasonCode>,
    #[serde(default)]
    pub invariant_kind: Option<String>,
    #[serde(default)]
    pub provenance: Option<String>,
    #[serde(default)]
    pub conserved_quantities: Vec<String>,
    #[serde(default)]
    pub exact_defects: Vec<ExactDefect>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProblemInput {
    pub dimension: usize,
    pub retained: Vec<usize>,
    pub omitted: Vec<usize>,
    pub entries: Vec<MatrixEntry>,
    #[serde(default)]
    pub exact_license: Option<ExactLicenseInput>,
}

impl ProblemInput {
    pub fn partition(&self) -> PartitionSpec {
        PartitionSpec::new(self.retained.clone(), self.omitted.clone())
    }

    pub fn to_sparse_matrix(&self) -> CsMat<f64> {
        let mut tri = TriMat::new((self.dimension, self.dimension));
        for entry in &self.entries {
            tri.add_triplet(entry.row, entry.col, entry.value);
        }
        tri.to_csr()
    }
}
