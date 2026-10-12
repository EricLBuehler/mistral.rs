use llm_watermarking::{textgrain as library, trace::TraceStatus};
use serde::{Deserialize, Serialize};

use super::{GENERATION_RNG_VERSION, MAX_OBSERVATIONS};

const CANDIDATE_FIELDS: usize = 12;
const BLOCK_FIELDS: usize = 3;
const ITERATION_FIELDS: usize = 6;
const SUMMARY_FIELDS: usize = 32;
const TRANSPORT_FIELDS: usize = 2;
const SHARED_RNG_VERSION: &str = "rand_isaac/0.4.0/Isaac64Rng/shared_host_stream";

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct TextGrainTraceConfig {
    #[cfg_attr(feature = "utoipa", schema(minimum = 0, maximum = 10000))]
    pub max_iterations: usize,
    pub transport: bool,
}

impl TextGrainTraceConfig {
    pub(crate) fn options(self) -> library::TextGrainTraceOptions {
        library::TextGrainTraceOptions {
            max_iterations: self.max_iterations,
            transport: self.transport,
        }
    }

    pub(crate) fn records(
        self,
        candidates: usize,
        blocks: usize,
        columns: usize,
        iterations: usize,
    ) -> anyhow::Result<usize> {
        self.options().validate()?;
        Ok(candidates * CANDIDATE_FIELDS
            + blocks * BLOCK_FIELDS
            + self.max_iterations.min(iterations) * ITERATION_FIELDS
            + SUMMARY_FIELDS
            + if self.transport {
                blocks * columns * TRANSPORT_FIELDS
            } else {
                0
            })
    }

    pub(crate) fn view(self, max_rows: usize) -> library::TextGrainTraceView {
        library::TextGrainTraceView {
            max_rows,
            max_elements: MAX_OBSERVATIONS,
            transport: self.transport,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TextGrainTrace {
    pub status: String,
    pub token_ids: Vec<u32>,
    pub input_probabilities: Vec<f64>,
    pub output_probabilities: Vec<f64>,
    pub output_log_probabilities: Vec<Option<f64>>,
    pub blocks: Option<Vec<usize>>,
    pub selected_column: Option<usize>,
    pub block_masses: Vec<f64>,
    pub conditional_block_probabilities: Vec<f64>,
    pub selected_costs: Option<Vec<f64>>,
    pub detection_scores: Option<Vec<f64>>,
    pub solver: Option<TextGrainSolver>,
    pub iterations: Vec<TextGrainIteration>,
    pub omitted_iterations: usize,
    pub transport: Option<TextGrainTransport>,
    pub generation: Option<TextGrainGeneration>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TextGrainSolver {
    pub status: String,
    pub iterations: usize,
    pub token_entropy: f64,
    pub block_entropy: f64,
    pub requested_fraction: f64,
    pub target_fraction: f64,
    pub achieved_fraction: f64,
    pub entropy_loss: f64,
    pub regularization: Option<f64>,
    pub row_residual: f64,
    pub column_residual: f64,
    pub budget_satisfied: bool,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TextGrainIteration {
    pub iteration: usize,
    pub regularization: f64,
    pub achieved_fraction: f64,
    pub row_residual: f64,
    pub column_residual: f64,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TextGrainTransport {
    pub block_count: usize,
    pub column_count: usize,
    pub costs: Vec<f64>,
    pub coupling: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TextGrainGeneration {
    pub origin: String,
    pub used_for_generation: bool,
    pub selected_token_id: u32,
    pub selected_block: Option<usize>,
    pub block_draw: Option<TextGrainDraw>,
    pub token_draw: TextGrainDraw,
    pub rng_draws: usize,
    pub sampling_version: String,
    pub rng_version: String,
    pub effective_seed: Option<String>,
    pub rng_provenance: String,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TextGrainDraw {
    pub uniform: f64,
    pub cdf_lower: f64,
    pub cdf_upper: f64,
}

impl From<library::trace::CategoricalDraw> for TextGrainDraw {
    fn from(draw: library::trace::CategoricalDraw) -> Self {
        Self {
            uniform: draw.uniform,
            cdf_lower: draw.cdf_lower,
            cdf_upper: draw.cdf_upper,
        }
    }
}

impl TextGrainGeneration {
    pub(crate) fn from_library(
        trace: &library::trace::TextGrainGenerationTrace,
        seed: Option<u64>,
    ) -> Self {
        Self {
            origin: "production".into(),
            used_for_generation: trace.used_for_generation,
            selected_token_id: trace.token_id,
            selected_block: trace.selected_block,
            block_draw: trace.block_draw.clone().map(Into::into),
            token_draw: trace.token_draw.clone().into(),
            rng_draws: trace.rng_draws,
            sampling_version: trace.sampling_version.into(),
            rng_version: if seed.is_some() {
                GENERATION_RNG_VERSION
            } else {
                SHARED_RNG_VERSION
            }
            .into(),
            effective_seed: seed.map(|s| s.to_string()),
            rng_provenance: if seed.is_some() {
                "sequence_seed"
            } else {
                "shared_host_stream"
            }
            .into(),
        }
    }
}

impl TextGrainTrace {
    pub(crate) fn from_library(
        s: library::trace::TextGrainSnapshot,
        generation: Option<TextGrainGeneration>,
    ) -> Self {
        Self {
            status: status(s.status).into(),
            token_ids: s.token_ids,
            input_probabilities: s.input_probabilities,
            output_probabilities: s.output_probabilities,
            output_log_probabilities: s
                .output_log_probabilities
                .into_iter()
                .map(|p| p.is_finite().then_some(p))
                .collect(),
            blocks: s.blocks,
            selected_column: s.selected_column,
            block_masses: s.block_masses,
            conditional_block_probabilities: s.conditional_block_probabilities,
            selected_costs: s.selected_costs,
            detection_scores: s.detection_scores,
            solver: s.solver.map(|r| TextGrainSolver {
                status: match r.status {
                    library::SolverStatus::Independent => "independent",
                    library::SolverStatus::Converged => "converged",
                    library::SolverStatus::IterationLimit => "iteration_limit",
                }
                .into(),
                iterations: r.iterations,
                token_entropy: r.token_entropy,
                block_entropy: r.block_entropy,
                requested_fraction: r.requested_fraction,
                target_fraction: r.target_fraction,
                achieved_fraction: r.achieved_fraction,
                entropy_loss: r.entropy_loss,
                regularization: r.regularization,
                row_residual: r.row_residual,
                column_residual: r.column_residual,
                budget_satisfied: r.budget_satisfied,
            }),
            iterations: s
                .iterations
                .into_iter()
                .map(|i| TextGrainIteration {
                    iteration: i.iteration,
                    regularization: i.regularization,
                    achieved_fraction: i.achieved_fraction,
                    row_residual: i.row_residual,
                    column_residual: i.column_residual,
                })
                .collect(),
            omitted_iterations: s.omitted_iterations,
            transport: s.transport.map(|t| TextGrainTransport {
                block_count: t.block_count,
                column_count: t.column_count,
                costs: t.costs,
                coupling: t.coupling,
            }),
            generation,
        }
    }
}

pub(crate) fn status(status: TraceStatus) -> &'static str {
    match status {
        TraceStatus::Applied => "applied",
        TraceStatus::Warmup => "warmup",
        TraceStatus::RepeatedContext => "repeated_context",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sampling_trace::SamplingTraceConfig, WatermarkConfig};

    #[test]
    fn textgrain_capture_limits_account_for_transport_and_solver_iterations() -> anyhow::Result<()>
    {
        let watermark: WatermarkConfig = serde_json::from_value(serde_json::json!({
            "scheme": "textgrain", "key": "42".repeat(32)
        }))?;
        let config = SamplingTraceConfig {
            textgrain: Some(TextGrainTraceConfig {
                max_iterations: 8,
                transport: true,
            }),
            ..Default::default()
        };
        config.validate(true, 1)?;
        config.validate_watermark(Some(&watermark))?;
        assert!(config.validate_watermark(None).is_err());
        let oversized = SamplingTraceConfig {
            textgrain: Some(TextGrainTraceConfig {
                max_iterations: 10_001,
                transport: false,
            }),
            ..config
        };
        assert!(oversized.validate(true, 1).is_err());
        let huge: WatermarkConfig = serde_json::from_value(serde_json::json!({
            "scheme": "textgrain", "key": "42".repeat(32), "block_count": 256, "column_count": 256
        }))?;
        assert!(SamplingTraceConfig {
            max_steps: 1,
            ..config
        }
        .validate_watermark(Some(&huge))
        .is_err());
        assert!(SamplingTraceConfig {
            textgrain: None,
            ..config
        }
        .validate_watermark(Some(&huge))
        .is_ok());
        Ok(())
    }
}
