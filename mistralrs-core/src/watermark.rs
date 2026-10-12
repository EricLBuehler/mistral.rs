use std::fmt;

mod schemes;
#[cfg(test)]
pub(crate) use schemes::tests::token_configs;
pub(crate) use schemes::RequestWatermark;
pub use schemes::{
    SynthIdGenerationPolicy, TextGrainGenerationPolicy, Watermark, WatermarkConfig,
    WatermarkEvidence, WatermarkTensor,
};

use llm_watermarking::synthid::{
    SynthIdConfig, SynthIdText, DEFAULT_DEPTH, DEFAULT_NGRAM_LEN, KEY_BYTES,
};
use serde::{Deserialize, Serialize};

const HASH_DOMAIN: &[u8] = b"mistralrs-synthid-text-v1\0";

/// Opt-in SynthID-Text tournament sampling with an independent, secret deployment key.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct SynthIdTextWatermarkConfig {
    /// A random 32-byte key encoded as 64 hexadecimal characters.
    pub key: String,
    #[serde(default = "default_ngram_len")]
    pub ngram_len: usize,
    #[serde(default = "default_depth")]
    pub depth: usize,
}

fn default_ngram_len() -> usize {
    DEFAULT_NGRAM_LEN
}

fn default_depth() -> usize {
    DEFAULT_DEPTH
}

impl fmt::Debug for SynthIdTextWatermarkConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SynthIdTextWatermarkConfig")
            .field("key", &"[redacted]")
            .field("ngram_len", &self.ngram_len)
            .field("depth", &self.depth)
            .finish()
    }
}

impl SynthIdTextWatermarkConfig {
    pub fn new(key: String) -> anyhow::Result<Self> {
        let config = Self {
            key,
            ngram_len: DEFAULT_NGRAM_LEN,
            depth: DEFAULT_DEPTH,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        self.algorithm_config().map(|_| ())
    }

    fn algorithm_config(&self) -> anyhow::Result<SynthIdConfig> {
        anyhow::ensure!(
            self.key.len() == KEY_BYTES * 2
                && self.key.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "watermark key must contain exactly 64 hexadecimal characters"
        );
        let mut key = [0; KEY_BYTES];
        for (index, byte) in key.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&self.key[index * 2..index * 2 + 2], 16)?;
        }
        let config = SynthIdConfig {
            key,
            ngram_len: self.ngram_len,
            depth: self.depth,
        };
        config.validate()?;
        Ok(config)
    }
}

/// Mean g-value evidence, without a calibrated classification or probability of authorship.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WatermarkDetection {
    pub tokens_scored: usize,
    pub mean_g_value: Option<f64>,
}

/// SynthID-Text with a versioned keyed SHA-256 g-function, independent of Claude and Gemini keys.
#[derive(Clone)]
pub struct SynthIdTextWatermark {
    inner: SynthIdText,
}

impl SynthIdTextWatermark {
    pub fn new(config: &SynthIdTextWatermarkConfig) -> anyhow::Result<Self> {
        Ok(Self {
            inner: SynthIdText::with_domain(&config.algorithm_config()?, HASH_DOMAIN)?,
        })
    }

    /// Score generated token IDs, excluding the prompt, EOS and all repeated contexts.
    pub fn detect(
        &self,
        tokens: &[u32],
        prompt_len: usize,
        eos_token_ids: &[u32],
    ) -> anyhow::Result<WatermarkDetection> {
        let evidence = self.inner.detect(tokens, prompt_len, eos_token_ids)?;
        Ok(WatermarkDetection {
            tokens_scored: evidence.tokens_scored,
            mean_g_value: evidence.mean_g_value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_watermarking::synthid::{MAX_DEPTH, MAX_NGRAM_LEN};

    fn config() -> SynthIdTextWatermarkConfig {
        SynthIdTextWatermarkConfig::new(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f".into(),
        )
        .unwrap()
    }

    #[test]
    fn watermark_config_validation_and_redaction() {
        let mut config = config();
        assert!(!format!("{config:?}").contains(&config.key));
        for key in ["", "abc", &"g".repeat(64), &"\u{e9}".repeat(32)] {
            assert!(SynthIdTextWatermarkConfig::new(key.into()).is_err());
        }
        for depth in [0, MAX_DEPTH + 1] {
            config.depth = depth;
            assert!(config.validate().is_err());
        }
        config.depth = DEFAULT_DEPTH;
        for ngram_len in [0, 1, MAX_NGRAM_LEN + 1] {
            config.ngram_len = ngram_len;
            assert!(config.validate().is_err());
        }
        let decoded: SynthIdTextWatermarkConfig =
            serde_json::from_value(serde_json::json!({"key": config.key})).unwrap();
        assert_eq!(decoded.ngram_len, DEFAULT_NGRAM_LEN);
        assert_eq!(decoded.depth, DEFAULT_DEPTH);
        assert!(serde_json::from_value::<SynthIdTextWatermarkConfig>(
            serde_json::json!({"key": decoded.key, "unknown": true})
        )
        .is_err());
    }

    #[test]
    fn watermark_dependency_preserves_existing_format() {
        let mut one_layer = config();
        one_layer.depth = 1;
        let watermark = SynthIdTextWatermark::new(&one_layer).unwrap();
        let mut probs = [0.125; 8];
        watermark.inner.apply(&mut probs, &[1, 2, 3, 4], 4).unwrap();
        assert_eq!(
            probs,
            [0.171875, 0.171875, 0.171875, 0.171875, 0.046875, 0.171875, 0.046875, 0.046875]
        );
        let watermark = SynthIdTextWatermark::new(&config()).unwrap();
        let evidence = watermark.detect(&[1, 2, 3, 4, 7, 99, 8], 4, &[99]).unwrap();
        assert_eq!(evidence.tokens_scored, 1);
        assert_eq!(evidence.mean_g_value, Some(0.2));
        let serialized = serde_json::to_value(&evidence).unwrap();
        assert_eq!(
            serialized,
            serde_json::json!({"tokens_scored": 1, "mean_g_value": 0.2})
        );
    }
}
