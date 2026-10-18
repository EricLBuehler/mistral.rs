use llm_watermarking::{
    synthid::tournament::{
        TournamentDemo, TournamentEntrant, TournamentOptions, TournamentOrigin, TournamentSide,
        TournamentSource, TournamentWinReason, MAX_TOURNAMENT_ROUNDS,
    },
    trace::TraceStatus,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const DEFAULT_ROUNDS: usize = 2;
const SEED_DOMAIN: &[u8] = b"mistralrs-synthid-teaching-seed-v1\0";
const RNG_VERSION: &str = "sha256_tournament_demo_v1";
const ORIGIN: &str = "teaching_simulation";

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct TeachingTournamentConfig {
    #[cfg_attr(feature = "utoipa", schema(minimum = 1, maximum = 4))]
    pub rounds: usize,
    pub seed: u64,
}

impl Default for TeachingTournamentConfig {
    fn default() -> Self {
        Self {
            rounds: DEFAULT_ROUNDS,
            seed: 0,
        }
    }
}

impl TeachingTournamentConfig {
    pub(crate) fn records(&self) -> anyhow::Result<usize> {
        anyhow::ensure!(
            (1..=MAX_TOURNAMENT_ROUNDS).contains(&self.rounds),
            "teaching_tournament rounds must be 1..4"
        );
        Ok((1 << (self.rounds + 1)) - 1)
    }

    pub(crate) fn validate_watermark(
        &self,
        watermark: Option<&crate::WatermarkConfig>,
    ) -> anyhow::Result<()> {
        self.records()?;
        let Some(crate::WatermarkConfig::Synthid { depth, .. }) = watermark else {
            anyhow::bail!("teaching_tournament requires a SynthID watermark");
        };
        anyhow::ensure!(
            self.rounds <= *depth,
            "teaching_tournament rounds must not exceed SynthID depth"
        );
        Ok(())
    }

    pub(crate) fn options(&self, generated_index: usize) -> TournamentOptions {
        let mut hash = Sha256::new();
        hash.update(SEED_DOMAIN);
        hash.update(self.seed.to_le_bytes());
        hash.update((generated_index as u64).to_le_bytes());
        let digest = hash.finalize();
        TournamentOptions {
            rounds: self.rounds,
            seed: u64::from_le_bytes(digest[..8].try_into().unwrap()),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TeachingTournament {
    pub origin: String,
    pub used_for_generation: bool,
    pub rng_version: String,
    pub effective_seed: String,
    pub status: String,
    pub requested_rounds: usize,
    pub configured_depth: usize,
    pub rounds: usize,
    pub layer_indices: Vec<usize>,
    pub draws: Vec<TeachingDraw>,
    pub matches: Vec<TeachingMatch>,
    pub winner: Option<TeachingWinner>,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TeachingDraw {
    pub draw_id: usize,
    pub token_id: u32,
    pub text: Option<String>,
    pub probability: f64,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TeachingSource {
    #[serde(rename = "type")]
    pub kind: String,
    pub id: usize,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TeachingEntrant {
    pub source: TeachingSource,
    pub draw_id: usize,
    pub g_value: u8,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TeachingMatch {
    pub match_id: usize,
    pub round: usize,
    pub left: TeachingEntrant,
    pub right: TeachingEntrant,
    pub winner: String,
    pub winner_draw_id: usize,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(get_all))]
pub struct TeachingWinner {
    pub match_id: usize,
    pub draw_id: usize,
    pub token_id: u32,
}

impl TeachingTournament {
    pub(crate) fn greedy(options: TournamentOptions, depth: usize) -> Self {
        Self {
            origin: ORIGIN.into(),
            used_for_generation: false,
            rng_version: RNG_VERSION.into(),
            effective_seed: options.seed.to_string(),
            status: "greedy".into(),
            requested_rounds: options.rounds,
            configured_depth: depth,
            rounds: 0,
            layer_indices: Vec::new(),
            draws: Vec::new(),
            matches: Vec::new(),
            winner: None,
        }
    }

    pub(crate) fn from_library(
        demo: TournamentDemo,
        mut decode: impl FnMut(u32) -> candle_core::Result<Option<String>>,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            origin: match demo.origin {
                TournamentOrigin::Demonstration => ORIGIN,
            }
            .into(),
            used_for_generation: false,
            rng_version: RNG_VERSION.into(),
            effective_seed: demo.seed.to_string(),
            status: match demo.status {
                TraceStatus::Applied => "demonstrated",
                TraceStatus::Warmup => "warmup",
                TraceStatus::RepeatedContext => "repeated_context",
            }
            .into(),
            requested_rounds: demo.requested_rounds,
            configured_depth: demo.configured_depth,
            rounds: demo.rounds,
            layer_indices: (0..demo.rounds).collect(),
            draws: demo
                .draws
                .into_iter()
                .map(|d| {
                    Ok(TeachingDraw {
                        draw_id: d.draw_id,
                        token_id: d.token_id,
                        text: decode(d.token_id)?,
                        probability: d.probability,
                    })
                })
                .collect::<candle_core::Result<_>>()?,
            matches: demo
                .matches
                .into_iter()
                .map(|m| TeachingMatch {
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
            winner: demo.winner.map(|w| TeachingWinner {
                match_id: w.match_id,
                draw_id: w.draw_id,
                token_id: w.token_id,
            }),
        })
    }
}

impl From<TournamentEntrant> for TeachingEntrant {
    fn from(entrant: TournamentEntrant) -> Self {
        let (kind, id) = match entrant.source {
            TournamentSource::Draw(id) => ("draw", id),
            TournamentSource::Match(id) => ("match", id),
        };
        Self {
            source: TeachingSource {
                kind: kind.into(),
                id,
            },
            draw_id: entrant.draw_id,
            g_value: entrant.g_value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sampling_trace::SamplingTraceConfig, WatermarkConfig};

    #[test]
    fn teaching_tournament_validates_depth_defaults_and_observation_budget() {
        let config: SamplingTraceConfig =
            serde_json::from_str(r#"{"teaching_tournament":{}}"#).unwrap();
        let teaching = config.teaching_tournament.unwrap();
        assert_eq!((teaching.rounds, teaching.seed), (2, 0));
        assert!(config.validate(true, 1).is_ok());
        assert!(config.validate_watermark(None).is_err());
        for scheme in crate::watermark::token_configs(17) {
            assert_eq!(
                config.validate_watermark(Some(&scheme)).is_ok(),
                scheme.scheme() == "synthid"
            );
        }
        let shallow = WatermarkConfig::Synthid {
            generation_policy: Default::default(),
            key: "42".repeat(32),
            ngram_len: 5,
            depth: 1,
        };
        assert!(config.validate_watermark(Some(&shallow)).is_err());
        for rounds in [0, 5, usize::MAX] {
            let invalid = SamplingTraceConfig {
                teaching_tournament: Some(TeachingTournamentConfig { rounds, seed: 0 }),
                ..config
            };
            assert!(invalid.validate(true, 1).is_err());
        }
        let at_limit = SamplingTraceConfig {
            generation_tournament: None,
            textgrain: None,
            max_steps: 256,
            max_candidates: 128,
            max_layers: 2,
            teaching_tournament: None,
        };
        assert!(at_limit.validate(true, 1).is_ok());
        assert!(SamplingTraceConfig {
            teaching_tournament: Some(teaching),
            ..at_limit
        }
        .validate(true, 1)
        .is_err());
        assert!(SamplingTraceConfig {
            max_layers: 0,
            ..config
        }
        .validate(true, 1)
        .is_ok());
        assert!(serde_json::from_str::<SamplingTraceConfig>(
            r#"{"teaching_tournament":{"unknown":1}}"#
        )
        .is_err());
        assert!(serde_json::to_value(SamplingTraceConfig::default())
            .unwrap()
            .get("teaching_tournament")
            .is_none());
    }

    #[test]
    fn teaching_tournament_seed_has_stable_position_domain() {
        for (seed, position, expected) in [
            (0, 0, 3_008_017_655_017_818_559),
            (42, 4, 11_240_688_699_727_373_584),
            (u64::MAX, 17, 6_666_083_765_865_951_009),
        ] {
            assert_eq!(
                TeachingTournamentConfig { rounds: 2, seed }
                    .options(position)
                    .seed,
                expected
            );
        }
    }
}
