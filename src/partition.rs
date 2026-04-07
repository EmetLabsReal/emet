use std::collections::BTreeSet;

use crate::report::{DecisionFailure, DecisionReasonCode};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionSpec {
    pub retained: Vec<usize>,
    pub omitted: Vec<usize>,
}

impl PartitionSpec {
    pub fn new(retained: Vec<usize>, omitted: Vec<usize>) -> Self {
        Self { retained, omitted }
    }

    pub fn validate(&self, dimension: usize) -> Result<(), DecisionFailure> {
        if dimension == 0 {
            return Err(DecisionFailure {
                reason_code: DecisionReasonCode::InvalidInput,
                summary: "Dimension must be positive.".to_string(),
            });
        }
        if self.retained.is_empty() {
            return Err(DecisionFailure {
                reason_code: DecisionReasonCode::InvalidPartition,
                summary: "Retained indices must be nonempty.".to_string(),
            });
        }
        if self.omitted.is_empty() {
            return Err(DecisionFailure {
                reason_code: DecisionReasonCode::InvalidPartition,
                summary: "Omitted indices must be nonempty.".to_string(),
            });
        }

        let retained: BTreeSet<_> = self.retained.iter().copied().collect();
        let omitted: BTreeSet<_> = self.omitted.iter().copied().collect();

        if retained.len() != self.retained.len() || omitted.len() != self.omitted.len() {
            return Err(DecisionFailure {
                reason_code: DecisionReasonCode::InvalidPartition,
                summary: "Partition indices must be unique.".to_string(),
            });
        }

        if retained.iter().any(|&idx| idx >= dimension)
            || omitted.iter().any(|&idx| idx >= dimension)
        {
            return Err(DecisionFailure {
                reason_code: DecisionReasonCode::InvalidPartition,
                summary: "Partition indices must stay within the matrix dimension.".to_string(),
            });
        }

        if retained.intersection(&omitted).next().is_some() {
            return Err(DecisionFailure {
                reason_code: DecisionReasonCode::InvalidPartition,
                summary: "Retained and omitted indices must be disjoint.".to_string(),
            });
        }

        if retained.len() + omitted.len() != dimension {
            return Err(DecisionFailure {
                reason_code: DecisionReasonCode::InvalidPartition,
                summary: "Retained and omitted indices must cover the full matrix dimension."
                    .to_string(),
            });
        }

        Ok(())
    }
}
