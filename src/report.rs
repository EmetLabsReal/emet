use serde::{Deserialize, Serialize};

pub const EXACT_LICENSE_SCHEMA_VERSION: &str = "v1";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionStatus {
    Valid,
    Invalid,
}

impl DecisionStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Valid => "valid",
            Self::Invalid => "invalid",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionRegime {
    ScreeningFailure,
    PreAdmissible,
    Supercritical,
    Subcritical,
}

impl DecisionRegime {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ScreeningFailure => "screening_failure",
            Self::PreAdmissible => "pre_admissible",
            Self::Supercritical => "supercritical",
            Self::Subcritical => "subcritical",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionReasonCode {
    ValidReduction,
    InvalidInput,
    InvalidPartition,
    MatrixTooLarge,
    InsufficientOmittedBlockControl,
    OmittedBlockNumericallySingular,
    CouplingTooStrong,
}

impl DecisionReasonCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ValidReduction => "valid_reduction",
            Self::InvalidInput => "invalid_input",
            Self::InvalidPartition => "invalid_partition",
            Self::MatrixTooLarge => "matrix_too_large",
            Self::InsufficientOmittedBlockControl => "insufficient_omitted_block_control",
            Self::OmittedBlockNumericallySingular => "omitted_block_numerically_singular",
            Self::CouplingTooStrong => "coupling_too_strong",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExactLicenseStatus {
    LicensedQuotient,
    ObstructedQuotient,
    External,
    NotEvaluated,
}

impl ExactLicenseStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LicensedQuotient => "licensed_quotient",
            Self::ObstructedQuotient => "obstructed_quotient",
            Self::External => "external",
            Self::NotEvaluated => "not_evaluated",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExactReasonCode {
    CertifiedUpstream,
    ExternalCertificateRequired,
    QuotientMapMissing,
    PullbackDefectTooLarge,
    InvariantDegeneracy,
    ClosureUnverified,
    ConservationDefectTooLarge,
}

impl ExactReasonCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CertifiedUpstream => "certified_upstream",
            Self::ExternalCertificateRequired => "external_certificate_required",
            Self::QuotientMapMissing => "quotient_map_missing",
            Self::PullbackDefectTooLarge => "pullback_defect_too_large",
            Self::InvariantDegeneracy => "invariant_degeneracy",
            Self::ClosureUnverified => "closure_unverified",
            Self::ConservationDefectTooLarge => "conservation_defect_too_large",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControlSemantics {
    CoercivityAlignedSpd,
    InvertibilityProxy,
}

impl ControlSemantics {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CoercivityAlignedSpd => "coercivity_aligned_spd",
            Self::InvertibilityProxy => "invertibility_proxy",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecisionProfile {
    pub retained_dimension: usize,
    pub omitted_dimension: usize,
    pub cross_nnz: usize,
    pub omitted_nnz: usize,
    pub coupling_strength: Option<f64>,
    pub control_strength: Option<f64>,
    pub control_semantics: Option<ControlSemantics>,
    pub control_margin: Option<f64>,
    pub omitted_condition_number: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdvancedMetrics {
    pub lambda: f64,
    pub gamma: f64,
    pub sigma_min_qq: f64,
    pub delta: Option<f64>,
    pub chi: Option<f64>,
    pub epsilon: Option<f64>,
    pub security_margin: Option<f64>,
    pub q_inverse_bound: Option<f64>,
    pub residual_window: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DenseMatrixReport {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalStatus {
    NotEvaluated,
    UniqueCanonical,
    ContinuousOrbit,
    SymmetryTied,
    Dominated,
    Indeterminate,
}

impl CanonicalStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotEvaluated => "not_evaluated",
            Self::UniqueCanonical => "unique_canonical",
            Self::ContinuousOrbit => "continuous_orbit",
            Self::SymmetryTied => "symmetry_tied",
            Self::Dominated => "dominated",
            Self::Indeterminate => "indeterminate",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CanonicalityReport {
    pub status: CanonicalStatus,
    pub symmetry_quotiented: bool,
    pub orbit_size: Option<usize>,
    pub orbit_dimension: Option<usize>,
    pub symmetry_kind: Option<String>,
    pub competitor_count: usize,
    pub representative_retained: Option<Vec<usize>>,
    pub representative_omitted: Option<Vec<usize>>,
    pub dominance_margin: Option<f64>,
    pub summary: String,
}

impl CanonicalityReport {
    pub fn not_evaluated_single_cut() -> Self {
        Self {
            status: CanonicalStatus::NotEvaluated,
            symmetry_quotiented: false,
            orbit_size: None,
            orbit_dimension: None,
            symmetry_kind: None,
            competitor_count: 0,
            representative_retained: None,
            representative_omitted: None,
            dominance_margin: None,
            summary: "Canonicality was not evaluated: the current decision path certifies one reported cut, not a competing cut family.".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExactDefect {
    pub kind: String,
    pub value: Option<f64>,
    pub tolerance: Option<f64>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExactLicenseReport {
    pub schema_version: String,
    pub status: ExactLicenseStatus,
    pub reason_code: ExactReasonCode,
    pub invariant_kind: Option<String>,
    pub provenance: Option<String>,
    pub conserved_quantities: Vec<String>,
    pub exact_defects: Vec<ExactDefect>,
}

impl ExactLicenseReport {
    pub fn external_default() -> Self {
        Self {
            schema_version: EXACT_LICENSE_SCHEMA_VERSION.to_string(),
            status: ExactLicenseStatus::External,
            reason_code: ExactReasonCode::ExternalCertificateRequired,
            invariant_kind: None,
            provenance: None,
            conserved_quantities: Vec::new(),
            exact_defects: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecisionReport {
    pub exact_license: ExactLicenseReport,
    pub regime: DecisionRegime,
    pub valid: bool,
    pub status: DecisionStatus,
    pub summary: String,
    pub reason_code: DecisionReasonCode,
    pub action_items: Vec<String>,
    pub partition_profile: DecisionProfile,
    pub reduced_matrix: Option<DenseMatrixReport>,
    pub advanced_metrics: Option<AdvancedMetrics>,
    pub canonicality: CanonicalityReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionFailure {
    pub reason_code: DecisionReasonCode,
    pub summary: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_exact_license_default_is_stable() {
        let exact = ExactLicenseReport::external_default();
        assert_eq!(exact.schema_version, EXACT_LICENSE_SCHEMA_VERSION);
        assert_eq!(exact.status, ExactLicenseStatus::External);
        assert_eq!(
            exact.reason_code,
            ExactReasonCode::ExternalCertificateRequired
        );
        assert!(exact.conserved_quantities.is_empty());
        assert!(exact.exact_defects.is_empty());
    }

    #[test]
    fn exact_license_round_trips_through_json() {
        let report = ExactLicenseReport {
            schema_version: EXACT_LICENSE_SCHEMA_VERSION.to_string(),
            status: ExactLicenseStatus::LicensedQuotient,
            reason_code: ExactReasonCode::CertifiedUpstream,
            invariant_kind: Some("dirichlet_form".to_string()),
            provenance: Some("operator_estimates".to_string()),
            conserved_quantities: vec!["energy".to_string()],
            exact_defects: vec![ExactDefect {
                kind: "pullback_residual".to_string(),
                value: Some(1.0e-12),
                tolerance: Some(1.0e-9),
                summary: "Residual stays below the supplied tolerance.".to_string(),
            }],
        };

        let encoded = serde_json::to_string(&report).expect("serialize exact license");
        let decoded: ExactLicenseReport =
            serde_json::from_str(&encoded).expect("deserialize exact license");
        assert_eq!(decoded, report);
    }

    #[test]
    fn canonicality_default_round_trips_through_json() {
        let report = CanonicalityReport::not_evaluated_single_cut();
        let encoded = serde_json::to_string(&report).expect("serialize canonicality");
        let decoded: CanonicalityReport =
            serde_json::from_str(&encoded).expect("deserialize canonicality");
        assert_eq!(decoded, report);
    }
}
