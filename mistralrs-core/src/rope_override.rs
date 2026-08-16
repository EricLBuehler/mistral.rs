use serde::{Deserialize, Serialize};

/// YaRN rope override applied when synthesizing a GGUF model config.
///
/// Mirrors llama.cpp's `--rope-scaling yarn --rope-scale N --yarn-orig-ctx M`
/// semantics: `scale` is the positional extension factor, `orig_ctx` is the
/// pretraining context the model was trained with (defaults to the GGUF
/// `context_length` when unset), and `target_ctx`, when set, forces the
/// synthesized `max_position_embeddings` instead of `round(orig_ctx * scale)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RopeOverride {
    pub scale: f32,
    pub orig_ctx: Option<usize>,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub target_ctx: Option<usize>,
}

impl RopeOverride {
    pub fn yarn(scale: f32, orig_ctx: Option<usize>, target_ctx: Option<usize>) -> Self {
        Self {
            scale,
            orig_ctx,
            beta_fast: 32.0,
            beta_slow: 1.0,
            target_ctx,
        }
    }

    pub fn resolved_orig_ctx(&self, native_ctx: usize) -> usize {
        self.orig_ctx.unwrap_or(native_ctx)
    }

    pub fn resolved_target_ctx(&self, native_ctx: usize) -> usize {
        self.target_ctx
            .unwrap_or_else(|| (self.resolved_orig_ctx(native_ctx) as f32 * self.scale).round() as usize)
    }
}
