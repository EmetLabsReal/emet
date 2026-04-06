use nalgebra::DMatrix;
use sprs::CsMat;

use crate::input::{ExactLicenseInput, ProblemInput};
use crate::partition::PartitionSpec;
use crate::report::{
    AdvancedMetrics, CanonicalityReport, ControlSemantics, DecisionFailure, DecisionProfile,
    DecisionReasonCode, DecisionRegime, DecisionReport, DecisionStatus, DenseMatrixReport,
    EXACT_LICENSE_SCHEMA_VERSION, ExactLicenseReport, ExactLicenseStatus, ExactReasonCode,
};

const EPSILON: f64 = 1.0e-12;
const MAX_DENSE_ENTRIES: usize = 25_000_000;
const SYMMETRY_TOLERANCE: f64 = 1.0e-9;

pub fn decide_problem(problem: &ProblemInput) -> DecisionReport {
    match decide_validated(problem) {
        Ok(report) => report,
        Err(failure) => invalid_report(problem, failure),
    }
}

fn decide_validated(problem: &ProblemInput) -> Result<DecisionReport, DecisionFailure> {
    let exact_license = exact_license_report(problem);

    if problem
        .entries
        .iter()
        .any(|entry| entry.row >= problem.dimension || entry.col >= problem.dimension)
    {
        return Err(DecisionFailure {
            reason_code: DecisionReasonCode::InvalidInput,
            summary: "Matrix entries must stay within the declared dimension.".to_string(),
        });
    }

    let partition = problem.partition();
    partition.validate(problem.dimension)?;

    let dense_entries = problem
        .dimension
        .checked_mul(problem.dimension)
        .ok_or_else(|| DecisionFailure {
            reason_code: DecisionReasonCode::MatrixTooLarge,
            summary: "Matrix dimension is too large for the dense working-set check.".to_string(),
        })?;

    if dense_entries > MAX_DENSE_ENTRIES {
        return Err(DecisionFailure {
            reason_code: DecisionReasonCode::MatrixTooLarge,
            summary: format!(
                "Matrix size exceeds the dense working-set ceiling of {} entries.",
                MAX_DENSE_ENTRIES
            ),
        });
    }

    let matrix = dense_from_sparse(&problem.to_sparse_matrix(), problem.dimension);
    let blocks = ProblemBlocks::from_dense(&matrix, &partition);

    let coupling_strength = blocks.cross_norm();
    let control_strength = smallest_singular_value(&blocks.qq);
    let control_semantics = classify_control_semantics(&blocks.qq);
    let mut partition_profile = base_partition_profile(
        &blocks,
        coupling_strength,
        control_strength,
        control_semantics,
    );
    let advanced_metrics = build_advanced_metrics(coupling_strength, control_strength);

    if control_strength <= EPSILON {
        return Ok(invalid_with_blocks(
            exact_license,
            partition_profile,
            DecisionReasonCode::InsufficientOmittedBlockControl,
            "The omitted block is pre-admissible only: sigma_min(QQ) is too small to enter the reduction classification domain.",
            Some(advanced_metrics),
        ));
    }

    let Some(q_inverse) = blocks.qq.clone().try_inverse() else {
        return Ok(invalid_with_blocks(
            exact_license,
            partition_profile,
            DecisionReasonCode::OmittedBlockNumericallySingular,
            "The omitted block remains pre-admissible because QQ is numerically singular for the proposed partition.",
            Some(advanced_metrics),
        ));
    };

    let effective = &blocks.pp - (&blocks.pq * &q_inverse * &blocks.qp);
    let q_condition_number = condition_number(&blocks.qq, control_strength);
    partition_profile.control_margin = advanced_metrics.residual_window;
    partition_profile.omitted_condition_number = Some(q_condition_number);

    if advanced_metrics.chi.unwrap_or(f64::INFINITY) >= 1.0 {
        return Ok(DecisionReport {
            exact_license,
            regime: DecisionRegime::Supercritical,
            valid: false,
            status: DecisionStatus::Invalid,
            summary:
                "The effective matrix was formed, but the split is supercritical: cross-block coupling exceeds the reduction margin."
                    .to_string(),
            reason_code: DecisionReasonCode::CouplingTooStrong,
            action_items: action_items(DecisionReasonCode::CouplingTooStrong),
            partition_profile,
            reduced_matrix: Some(DenseMatrixReport::from_matrix(&effective)),
            advanced_metrics: Some(advanced_metrics),
            canonicality: CanonicalityReport::not_evaluated_single_cut(),
        });
    }

    Ok(DecisionReport {
        exact_license,
        regime: DecisionRegime::Subcritical,
        valid: true,
        status: DecisionStatus::Valid,
        summary: "Reduction is numerically subcritical and licensed for the proposed partition."
            .to_string(),
        reason_code: DecisionReasonCode::ValidReduction,
        action_items: action_items(DecisionReasonCode::ValidReduction),
        partition_profile,
        reduced_matrix: Some(DenseMatrixReport::from_matrix(&effective)),
        advanced_metrics: Some(advanced_metrics),
        canonicality: CanonicalityReport::not_evaluated_single_cut(),
    })
}

fn invalid_report(problem: &ProblemInput, failure: DecisionFailure) -> DecisionReport {
    DecisionReport {
        exact_license: exact_license_report(problem),
        regime: regime_from_reason_code(failure.reason_code),
        valid: false,
        status: DecisionStatus::Invalid,
        summary: failure.summary,
        reason_code: failure.reason_code,
        action_items: action_items(failure.reason_code),
        partition_profile: DecisionProfile {
            retained_dimension: problem.retained.len(),
            omitted_dimension: problem.omitted.len(),
            cross_nnz: 0,
            omitted_nnz: 0,
            coupling_strength: None,
            control_strength: None,
            control_semantics: None,
            control_margin: None,
            omitted_condition_number: None,
        },
        reduced_matrix: None,
        advanced_metrics: None,
        canonicality: CanonicalityReport::not_evaluated_single_cut(),
    }
}

fn invalid_with_blocks(
    exact_license: ExactLicenseReport,
    partition_profile: DecisionProfile,
    reason_code: DecisionReasonCode,
    summary: &str,
    advanced_metrics: Option<AdvancedMetrics>,
) -> DecisionReport {
    DecisionReport {
        exact_license,
        regime: regime_from_reason_code(reason_code),
        valid: false,
        status: DecisionStatus::Invalid,
        summary: summary.to_string(),
        reason_code,
        action_items: action_items(reason_code),
        partition_profile,
        reduced_matrix: None,
        advanced_metrics,
        canonicality: CanonicalityReport::not_evaluated_single_cut(),
    }
}

fn exact_license_report(problem: &ProblemInput) -> ExactLicenseReport {
    match problem.exact_license.as_ref() {
        Some(input) => normalize_exact_license(input),
        None => ExactLicenseReport::external_default(),
    }
}

fn normalize_exact_license(input: &ExactLicenseInput) -> ExactLicenseReport {
    let status = input.status.unwrap_or(ExactLicenseStatus::External);
    let reason_code = input
        .reason_code
        .unwrap_or_else(|| default_exact_reason_code(status));

    ExactLicenseReport {
        schema_version: input
            .schema_version
            .clone()
            .unwrap_or_else(|| EXACT_LICENSE_SCHEMA_VERSION.to_string()),
        status,
        reason_code,
        invariant_kind: input.invariant_kind.clone(),
        provenance: input.provenance.clone(),
        conserved_quantities: input.conserved_quantities.clone(),
        exact_defects: input.exact_defects.clone(),
    }
}

fn default_exact_reason_code(status: ExactLicenseStatus) -> ExactReasonCode {
    match status {
        ExactLicenseStatus::LicensedQuotient => ExactReasonCode::CertifiedUpstream,
        ExactLicenseStatus::ObstructedQuotient => ExactReasonCode::ClosureUnverified,
        ExactLicenseStatus::External => ExactReasonCode::ExternalCertificateRequired,
        ExactLicenseStatus::NotEvaluated => ExactReasonCode::QuotientMapMissing,
    }
}

fn action_items(reason_code: DecisionReasonCode) -> Vec<String> {
    match reason_code {
        DecisionReasonCode::ValidReduction => vec![
            "Use the effective matrix for the next RG step or downstream analysis.".to_string(),
            "Track the residual window; if it shrinks toward zero, revisit the partition."
                .to_string(),
        ],
        DecisionReasonCode::InvalidInput => vec![
            "Check the declared dimension and confirm every matrix entry lies within it."
                .to_string(),
        ],
        DecisionReasonCode::InvalidPartition => vec![
            "Make retained and omitted indices disjoint and unique.".to_string(),
            "Cover the intended full matrix dimension with the two index sets.".to_string(),
        ],
        DecisionReasonCode::MatrixTooLarge => {
            vec!["Analyze a smaller extracted matrix or a smaller partitioned block.".to_string()]
        }
        DecisionReasonCode::InsufficientOmittedBlockControl => vec![
            "Treat this as a pre-admissible step: the omitted block never enters the classification domain."
                .to_string(),
            "Move weak omitted coordinates into the retained set or strengthen the omitted block before retrying."
                .to_string(),
        ],
        DecisionReasonCode::OmittedBlockNumericallySingular => vec![
            "Treat this as a pre-admissible step: the omitted block is numerically singular under the reported cut."
                .to_string(),
            "Move near-singular omitted directions into the retained set or revise the partition until QQ is invertible."
                .to_string(),
        ],
        DecisionReasonCode::CouplingTooStrong => vec![
            "Treat this as a supercritical step: the effective matrix is reported for inspection but is not licensed."
                .to_string(),
            "Widen the retained set to absorb heavily coupled coordinates.".to_string(),
            "If the split must stay fixed, use the full matrix instead of a reduced model."
                .to_string(),
        ],
    }
}

fn base_partition_profile(
    blocks: &ProblemBlocks,
    coupling_strength: f64,
    control_strength: f64,
    control_semantics: ControlSemantics,
) -> DecisionProfile {
    DecisionProfile {
        retained_dimension: blocks.pp.nrows(),
        omitted_dimension: blocks.qq.nrows(),
        cross_nnz: nnz_dense(&blocks.pq) + nnz_dense(&blocks.qp),
        omitted_nnz: nnz_dense(&blocks.qq),
        coupling_strength: Some(coupling_strength),
        control_strength: Some(control_strength),
        control_semantics: Some(control_semantics),
        control_margin: None,
        omitted_condition_number: None,
    }
}

fn regime_from_reason_code(reason_code: DecisionReasonCode) -> DecisionRegime {
    match reason_code {
        DecisionReasonCode::InvalidInput
        | DecisionReasonCode::InvalidPartition
        | DecisionReasonCode::MatrixTooLarge => DecisionRegime::ScreeningFailure,
        DecisionReasonCode::InsufficientOmittedBlockControl
        | DecisionReasonCode::OmittedBlockNumericallySingular => DecisionRegime::PreAdmissible,
        DecisionReasonCode::CouplingTooStrong => DecisionRegime::Supercritical,
        DecisionReasonCode::ValidReduction => DecisionRegime::Subcritical,
    }
}

fn build_advanced_metrics(coupling_strength: f64, control_strength: f64) -> AdvancedMetrics {
    let q_inverse_bound = if control_strength > EPSILON {
        Some(1.0 / control_strength)
    } else {
        None
    };
    let delta = if control_strength > EPSILON {
        Some((coupling_strength * coupling_strength) / control_strength)
    } else {
        None
    };
    let chi = if control_strength > EPSILON {
        let ratio = coupling_strength / control_strength;
        Some(ratio * ratio)
    } else {
        None
    };
    let residual_window = delta.map(|value| control_strength - value);
    let security_margin = chi.map(|c| control_strength * (1.0 - c));
    let epsilon = match (delta, security_margin) {
        (Some(d), Some(m)) if m > EPSILON => Some(d / (1.0 - chi.unwrap())),
        _ => None,
    };

    AdvancedMetrics {
        lambda: coupling_strength,
        gamma: control_strength,
        sigma_min_qq: control_strength,
        delta,
        chi,
        epsilon,
        security_margin,
        q_inverse_bound,
        residual_window,
    }
}

fn classify_control_semantics(matrix: &DMatrix<f64>) -> ControlSemantics {
    if is_nearly_symmetric(matrix) && min_symmetric_eigenvalue(matrix) > EPSILON {
        ControlSemantics::CoercivityAlignedSpd
    } else {
        ControlSemantics::InvertibilityProxy
    }
}

struct ProblemBlocks {
    pp: DMatrix<f64>,
    pq: DMatrix<f64>,
    qp: DMatrix<f64>,
    qq: DMatrix<f64>,
}

impl ProblemBlocks {
    fn from_dense(matrix: &DMatrix<f64>, partition: &PartitionSpec) -> Self {
        Self {
            pp: extract_block(matrix, &partition.retained, &partition.retained),
            pq: extract_block(matrix, &partition.retained, &partition.omitted),
            qp: extract_block(matrix, &partition.omitted, &partition.retained),
            qq: extract_block(matrix, &partition.omitted, &partition.omitted),
        }
    }

    fn cross_norm(&self) -> f64 {
        spectral_norm(&self.pq).max(spectral_norm(&self.qp))
    }
}

impl DenseMatrixReport {
    fn from_matrix(matrix: &DMatrix<f64>) -> Self {
        let data = (0..matrix.nrows())
            .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
            .collect();
        Self {
            rows: matrix.nrows(),
            cols: matrix.ncols(),
            data,
        }
    }
}

fn dense_from_sparse(matrix: &CsMat<f64>, dimension: usize) -> DMatrix<f64> {
    let mut dense = DMatrix::zeros(dimension, dimension);
    for (row, vec) in matrix.outer_iterator().enumerate() {
        for (col, value) in vec.iter() {
            dense[(row, col)] = *value;
        }
    }
    dense
}

fn extract_block(matrix: &DMatrix<f64>, rows: &[usize], cols: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(rows.len(), cols.len(), |i, j| matrix[(rows[i], cols[j])])
}

fn spectral_norm(matrix: &DMatrix<f64>) -> f64 {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return 0.0;
    }
    matrix
        .clone()
        .svd(false, false)
        .singular_values
        .iter()
        .copied()
        .fold(0.0, f64::max)
}

fn smallest_singular_value(matrix: &DMatrix<f64>) -> f64 {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return 0.0;
    }
    matrix
        .clone()
        .svd(false, false)
        .singular_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
}

fn is_nearly_symmetric(matrix: &DMatrix<f64>) -> bool {
    if matrix.nrows() != matrix.ncols() {
        return false;
    }

    let scale = matrix
        .iter()
        .fold(1.0_f64, |acc, value| acc.max(value.abs()));
    for row in 0..matrix.nrows() {
        for col in 0..row {
            if (matrix[(row, col)] - matrix[(col, row)]).abs() > SYMMETRY_TOLERANCE * scale {
                return false;
            }
        }
    }
    true
}

fn min_symmetric_eigenvalue(matrix: &DMatrix<f64>) -> f64 {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return 0.0;
    }
    matrix
        .clone()
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
}

fn condition_number(matrix: &DMatrix<f64>, sigma_min: f64) -> f64 {
    if matrix.nrows() == 0 || matrix.ncols() == 0 || sigma_min <= EPSILON {
        return f64::INFINITY;
    }
    spectral_norm(matrix) / sigma_min
}

fn nnz_dense(matrix: &DMatrix<f64>) -> usize {
    matrix.iter().filter(|value| value.abs() > EPSILON).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{ExactLicenseInput, MatrixEntry};

    fn problem(
        dimension: usize,
        retained: &[usize],
        omitted: &[usize],
        entries: &[(usize, usize, f64)],
    ) -> ProblemInput {
        ProblemInput {
            dimension,
            retained: retained.to_vec(),
            omitted: omitted.to_vec(),
            entries: entries
                .iter()
                .map(|(row, col, value)| MatrixEntry {
                    row: *row,
                    col: *col,
                    value: *value,
                })
                .collect(),
            exact_license: None,
        }
    }

    #[test]
    fn quarter_turn_is_invalid_due_to_missing_control() {
        let result = decide_problem(&problem(2, &[0], &[1], &[(0, 1, -1.0), (1, 0, 1.0)]));

        assert!(!result.valid);
        assert_eq!(result.status, DecisionStatus::Invalid);
        assert_eq!(result.regime, DecisionRegime::PreAdmissible);
        assert_eq!(
            result.reason_code,
            DecisionReasonCode::InsufficientOmittedBlockControl
        );
        assert!(result.reduced_matrix.is_none());
        let advanced_metrics = result
            .advanced_metrics
            .as_ref()
            .expect("expected advanced metrics");
        assert_eq!(advanced_metrics.lambda, 1.0);
        assert_eq!(advanced_metrics.gamma, 0.0);
        assert!(advanced_metrics.chi.is_none());
        assert_eq!(
            result.partition_profile.control_semantics,
            Some(ControlSemantics::InvertibilityProxy)
        );
        assert_eq!(result.exact_license.status, ExactLicenseStatus::External);
        assert_eq!(
            result.exact_license.reason_code,
            ExactReasonCode::ExternalCertificateRequired
        );
    }

    #[test]
    fn same_family_splits_invalid_and_valid() {
        let supercritical_result = decide_problem(&problem(
            3,
            &[0, 1],
            &[2],
            &[
                (0, 0, 1.0),
                (1, 1, 2.0),
                (2, 2, 1.0),
                (0, 2, 2.0),
                (2, 0, 2.0),
            ],
        ));
        let valid_result = decide_problem(&problem(
            3,
            &[0, 1],
            &[2],
            &[
                (0, 0, 1.0),
                (1, 1, 2.0),
                (2, 2, 2.0),
                (0, 2, 0.25),
                (2, 0, 0.25),
            ],
        ));

        assert_eq!(supercritical_result.status, DecisionStatus::Invalid);
        assert_eq!(supercritical_result.regime, DecisionRegime::Supercritical);
        assert_eq!(
            supercritical_result.reason_code,
            DecisionReasonCode::CouplingTooStrong
        );
        assert!(supercritical_result.reduced_matrix.is_some());
        assert!(supercritical_result.advanced_metrics.is_some());

        assert_eq!(valid_result.status, DecisionStatus::Valid);
        assert_eq!(valid_result.regime, DecisionRegime::Subcritical);
        assert_eq!(valid_result.reason_code, DecisionReasonCode::ValidReduction);
        assert!(valid_result.reduced_matrix.is_some());
        assert!(valid_result.advanced_metrics.is_some());
        assert!(
            valid_result
                .partition_profile
                .control_margin
                .expect("expected control margin")
                > 0.0
        );
        assert_eq!(
            valid_result.partition_profile.control_semantics,
            Some(ControlSemantics::CoercivityAlignedSpd)
        );
        assert_eq!(
            valid_result.exact_license.status,
            ExactLicenseStatus::External
        );
    }

    #[test]
    fn large_dense_fallback_requests_are_rejected() {
        let result = decide_problem(&problem(
            5_001,
            &[0],
            &(1..5_001).collect::<Vec<_>>(),
            &[(0, 0, 1.0)],
        ));

        assert_eq!(result.status, DecisionStatus::Invalid);
        assert_eq!(result.regime, DecisionRegime::ScreeningFailure);
        assert_eq!(result.reason_code, DecisionReasonCode::MatrixTooLarge);
        assert!(result.summary.contains("dense working-set ceiling"));
        assert_eq!(result.exact_license.status, ExactLicenseStatus::External);
    }

    #[test]
    fn supplied_exact_license_is_normalized_and_preserved() {
        let mut problem = problem(
            3,
            &[0, 1],
            &[2],
            &[
                (0, 0, 1.0),
                (1, 1, 2.0),
                (2, 2, 2.0),
                (0, 2, 0.25),
                (2, 0, 0.25),
            ],
        );
        problem.exact_license = Some(ExactLicenseInput {
            schema_version: None,
            status: Some(ExactLicenseStatus::LicensedQuotient),
            reason_code: None,
            invariant_kind: Some("dirichlet_form".to_string()),
            provenance: Some("operator_estimates".to_string()),
            conserved_quantities: vec!["energy".to_string()],
            exact_defects: Vec::new(),
        });

        let result = decide_problem(&problem);
        assert_eq!(
            result.exact_license.status,
            ExactLicenseStatus::LicensedQuotient
        );
        assert_eq!(
            result.exact_license.schema_version,
            EXACT_LICENSE_SCHEMA_VERSION
        );
        assert_eq!(
            result.exact_license.reason_code,
            ExactReasonCode::CertifiedUpstream
        );
        assert_eq!(
            result.exact_license.invariant_kind.as_deref(),
            Some("dirichlet_form")
        );
        assert_eq!(
            result.exact_license.provenance.as_deref(),
            Some("operator_estimates")
        );
        assert_eq!(result.exact_license.conserved_quantities, vec!["energy"]);
    }

    #[test]
    fn explicit_schema_version_is_preserved() {
        let mut problem = problem(
            3,
            &[0, 1],
            &[2],
            &[
                (0, 0, 1.0),
                (1, 1, 2.0),
                (2, 2, 2.0),
                (0, 2, 0.25),
                (2, 0, 0.25),
            ],
        );
        problem.exact_license = Some(ExactLicenseInput {
            schema_version: Some("v1".to_string()),
            status: Some(ExactLicenseStatus::ObstructedQuotient),
            reason_code: Some(ExactReasonCode::ClosureUnverified),
            invariant_kind: Some("dirichlet_form".to_string()),
            provenance: Some("forced_carrier_bridge".to_string()),
            conserved_quantities: Vec::new(),
            exact_defects: Vec::new(),
        });

        let result = decide_problem(&problem);
        assert_eq!(result.exact_license.schema_version, "v1");
        assert_eq!(
            result.exact_license.status,
            ExactLicenseStatus::ObstructedQuotient
        );
    }
}
