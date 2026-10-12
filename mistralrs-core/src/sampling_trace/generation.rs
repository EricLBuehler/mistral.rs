use llm_watermarking::synthid::{
    generation_tournament as library,
    tournament::{TournamentSide, TournamentWinReason},
};
use serde::{Deserialize, Serialize};

use super::TeachingWinner;

const DEFAULT_MATCHES: usize = 4095;
const MAX_TEXT_BYTES: usize = 65_536;
pub(crate) const MIN_GENERATION_STEP_BYTES: usize = 4096;
pub(crate) const GENERATION_RNG_VERSION: &str =
    "rand_isaac/0.4.0/Isaac64Rng/seed_from_u64/sequence_v1";
const SHARED_RNG_VERSION: &str = "rand_isaac/0.4.0/Isaac64Rng/shared_host_stream";

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct GenerationTournamentConfig {
    #[cfg_attr(feature = "utoipa", schema(minimum = 0, maximum = 65535))]
    pub max_matches: usize,
}

impl Default for GenerationTournamentConfig {
    fn default() -> Self {
        Self {
            max_matches: DEFAULT_MATCHES,
        }
    }
}

impl GenerationTournamentConfig {
    pub(crate) fn options(self, max_layers: usize) -> library::GenerationTournamentOptions {
        library::GenerationTournamentOptions {
            max_matches: self.max_matches,
            max_layers,
        }
    }

    pub(crate) fn records(self, max_layers: usize) -> anyhow::Result<usize> {
        self.options(max_layers).validate()?;
        Ok(if max_layers == 0 {
            2
        } else {
            self.max_matches * 4 + 3
        })
    }
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct GenerationTournament {
    pub origin: String,
    pub used_for_generation: bool,
    pub status: String,
    pub configured_depth: Option<usize>,
    pub rounds: usize,
    pub total_draws: usize,
    pub total_matches: usize,
    pub draws: Vec<GenerationDraw>,
    pub matches: Vec<GenerationMatch>,
    pub collapsed_subtrees: Vec<CollapsedSubtree>,
    pub winner: Option<TeachingWinner>,
    pub rng_version: String,
    pub effective_seed: Option<String>,
    pub rng_provenance: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_version: Option<String>,
    pub truncated: bool,
    pub truncation_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct GenerationDraw {
    pub draw_id: usize,
    pub token_id: u32,
    pub text: Option<String>,
    pub text_status: String,
    pub probability: f64,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct GenerationSource {
    pub kind: String,
    pub id: usize,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct GenerationEntrant {
    pub source: GenerationSource,
    pub draw_id: usize,
    pub token_id: u32,
    pub g_value: u8,
}

impl From<library::GenerationTournamentEntrant> for GenerationEntrant {
    fn from(entrant: library::GenerationTournamentEntrant) -> Self {
        let (kind, id) = match entrant.source {
            library::GenerationTournamentSource::Draw(id) => ("draw", id),
            library::GenerationTournamentSource::Match(id) => ("match", id),
            library::GenerationTournamentSource::CollapsedSubtree(id) => ("collapsed_subtree", id),
        };
        Self {
            source: GenerationSource {
                kind: kind.into(),
                id,
            },
            draw_id: entrant.draw_id,
            token_id: entrant.token_id,
            g_value: entrant.g_value,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct GenerationMatch {
    pub match_id: usize,
    pub round: usize,
    pub left: GenerationEntrant,
    pub right: GenerationEntrant,
    pub winner: String,
    pub winner_draw_id: usize,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct CollapsedSubtree {
    pub root_match_id: usize,
    pub round: usize,
    pub first_draw_id: usize,
    pub draw_count: usize,
    pub match_count: usize,
    pub winner_draw_id: usize,
    pub token_id: u32,
}

impl GenerationTournament {
    pub(crate) fn not_run(
        depth: Option<usize>,
        reason: library::NoTournamentReason,
        seed: Option<u64>,
    ) -> Self {
        let status = match reason {
            library::NoTournamentReason::ProbabilityUpdates => "no_production_bracket",
            library::NoTournamentReason::WatermarkDisabled => "watermark_disabled",
            library::NoTournamentReason::Greedy => "greedy",
            library::NoTournamentReason::UnsupportedSampling => "unsupported_sampling",
            library::NoTournamentReason::UnsupportedDepth => "unsupported_depth",
        };
        Self {
            origin: "production".into(),
            used_for_generation: false,
            status: status.into(),
            configured_depth: depth,
            rounds: 0,
            total_draws: 0,
            total_matches: 0,
            draws: Vec::new(),
            matches: Vec::new(),
            collapsed_subtrees: Vec::new(),
            winner: None,
            rng_version: if seed.is_some() {
                GENERATION_RNG_VERSION
            } else {
                SHARED_RNG_VERSION
            }
            .into(),
            effective_seed: seed.map(|seed| seed.to_string()),
            rng_provenance: if seed.is_some() {
                "sequence_seed"
            } else {
                "unavailable_shared_stream"
            }
            .into(),
            sampling_version: None,
            truncated: false,
            truncation_reason: None,
        }
    }

    pub(crate) fn from_library(
        trace: library::GenerationTournament,
        mut decode: impl FnMut(u32) -> candle_core::Result<Option<String>>,
    ) -> Self {
        let mut report = Self {
            origin: trace.origin.into(),
            used_for_generation: trace.used_for_generation,
            status: trace.status.as_str().into(),
            configured_depth: trace.configured_depth,
            rounds: trace.rounds,
            total_draws: trace.total_draws,
            total_matches: trace.total_matches,
            draws: trace
                .draws
                .into_iter()
                .map(|draw| GenerationDraw {
                    draw_id: draw.draw_id,
                    token_id: draw.token_id,
                    probability: draw.probability,
                    text: None,
                    text_status: "unavailable".into(),
                })
                .collect(),
            matches: trace
                .matches
                .into_iter()
                .map(|m| GenerationMatch {
                    match_id: m.match_id,
                    round: m.round,
                    left: m.left.into(),
                    right: m.right.into(),
                    winner: match m.winner {
                        TournamentSide::Left => "left",
                        TournamentSide::Right => "right",
                    }
                    .into(),
                    winner_draw_id: m.winner_draw_id,
                    reason: match m.reason {
                        TournamentWinReason::HigherScore => "higher_g",
                        TournamentWinReason::RandomTieBreak => "random_tie",
                    }
                    .into(),
                })
                .collect(),
            collapsed_subtrees: trace
                .collapsed_subtrees
                .into_iter()
                .map(|s| CollapsedSubtree {
                    root_match_id: s.root_match_id,
                    round: s.round,
                    first_draw_id: s.first_draw_id,
                    draw_count: s.draw_count,
                    match_count: s.match_count,
                    winner_draw_id: s.winner_draw_id,
                    token_id: s.token_id,
                })
                .collect(),
            winner: trace.winner.map(|w| TeachingWinner {
                match_id: w.match_id,
                draw_id: w.draw_id,
                token_id: w.token_id,
            }),
            rng_version: trace.rng_version,
            effective_seed: Some(trace.effective_seed),
            rng_provenance: "sequence_seed".into(),
            sampling_version: trace.sampling_version.map(str::to_owned),
            truncated: trace.truncated,
            truncation_reason: trace.truncation_reason.map(|r| r.as_str().into()),
        };
        let mut text_bytes = 0;
        for draw in &mut report.draws {
            match decode(draw.token_id) {
                Ok(Some(text)) if text_bytes + text.len() <= MAX_TEXT_BYTES => {
                    text_bytes += text.len();
                    draw.text = Some(text);
                    draw.text_status = "decoded".into();
                }
                Ok(Some(_)) => {
                    report.collapse("max_text_bytes");
                    break;
                }
                Ok(None) => {}
                Err(_) => draw.text_status = "decode_error".into(),
            }
        }
        report
    }

    pub(crate) fn collapse(&mut self, reason: &str) {
        let Some(winner) = &self.winner else { return };
        self.matches.clear();
        self.draws.retain(|draw| draw.draw_id == winner.draw_id);
        for draw in &mut self.draws {
            draw.text = None;
            draw.text_status = reason.into();
        }
        self.collapsed_subtrees = vec![CollapsedSubtree {
            root_match_id: winner.match_id,
            round: self.rounds - 1,
            first_draw_id: 0,
            draw_count: self.total_draws,
            match_count: self.total_matches,
            winner_draw_id: winner.draw_id,
            token_id: winner.token_id,
        }];
        self.truncated = true;
        self.truncation_reason = Some(match &self.truncation_reason {
            Some(previous) => format!("{previous},{reason}"),
            None => reason.into(),
        });
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use llm_watermarking::synthid::{SynthIdConfig, SynthIdText};
    use rand::{RngCore, SeedableRng};
    use rand_isaac::Isaac64Rng;

    fn raw(max_matches: usize, max_layers: usize) -> library::GenerationTournament {
        let watermark = SynthIdText::new(&SynthIdConfig {
            key: [42; 32],
            ngram_len: 2,
            depth: 4,
        })
        .unwrap();
        let mut rng = Isaac64Rng::seed_from_u64(42);
        let (token, report) = watermark
            .tournament_sampler()
            .unwrap()
            .sample_traced(
                &[0.0, 1.0],
                &[0],
                1,
                &mut || rng.next_u64(),
                &library::GenerationTournamentOptions {
                    max_matches,
                    max_layers,
                },
                library::GenerationRngInfo {
                    rng_version: GENERATION_RNG_VERSION,
                    effective_seed: "42",
                },
            )
            .unwrap();
        assert_eq!(token, 1);
        report
    }

    pub(crate) fn report() -> GenerationTournament {
        GenerationTournament::from_library(raw(15, 32), |id| Ok(Some(id.to_string())))
    }

    #[test]
    fn generation_tournament_wire_preserves_duplicate_draws_and_sparse_lineage() {
        let report = report();
        assert_eq!(report.draws.len(), 16);
        for (id, draw) in report.draws.iter().enumerate() {
            assert_eq!(draw.draw_id, id);
            assert_eq!(draw.token_id, 1);
            assert_eq!(draw.probability, 1.0);
        }
        let mut report = GenerationTournament::from_library(raw(1, 32), |_| Ok(None));
        assert_eq!(report.matches.len(), 1);
        assert_eq!(report.matches[0].match_id, 14);
        assert_eq!(report.matches[0].left.source.kind, "collapsed_subtree");
        assert_eq!(report.collapsed_subtrees.len(), 2);
        let winner = serde_json::to_value(&report.winner).unwrap();
        report.collapse("max_bytes");
        assert_eq!(serde_json::to_value(&report.winner).unwrap(), winner);
        assert_eq!(report.collapsed_subtrees[0].root_match_id, 14);
        assert_eq!(report.collapsed_subtrees[0].round, 3);
        assert_eq!(report.draws.len(), 1);
        assert_eq!(report.total_draws, 16);
        assert_eq!(report.total_matches, 15);
        assert_eq!(
            report.truncation_reason.as_deref(),
            Some("max_matches,max_bytes")
        );
    }

    #[test]
    fn generation_tournament_text_limits_and_decode_failures_preserve_winner() {
        let failed = GenerationTournament::from_library(raw(15, 32), |_| {
            candle_core::bail!("decode failed")
        });
        assert!(failed
            .draws
            .iter()
            .all(|d| d.text.is_none() && d.text_status == "decode_error"));
        let mut calls = 0;
        let limited = GenerationTournament::from_library(raw(15, 32), |_| {
            calls += 1;
            Ok(Some("x".repeat(MAX_TEXT_BYTES + 1)))
        });
        assert_eq!(calls, 1);
        assert_eq!(limited.winner.unwrap().token_id, 1);
        assert_eq!(limited.collapsed_subtrees.len(), 1);
        assert_eq!(limited.truncation_reason.as_deref(), Some("max_text_bytes"));
        assert!(limited.draws[0].text.is_none());
    }
}
