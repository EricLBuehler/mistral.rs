//! Block-diffusion text generation support (e.g. DiffusionGemma): models that commit a
//! whole denoised block of tokens per engine step instead of sampling one token from logits.

/// Mixin for block-diffusion models. Defaults describe an ordinary autoregressive model;
/// diffusion models override all three.
pub trait BlockDiffusionMixin {
    /// When true, `forward` returns committed block token ids as a u32 tensor
    /// [bs, block_len] rather than logits.
    fn is_block_diffusion(&self) -> bool {
        false
    }

    /// Hand the model the checkpoint's raw `generation_config.json` (the source of truth
    /// for denoising parameters). No-op for other models.
    fn configure_block_diffusion(&self, _generation_config_json: &str) {}

    /// Time the last forward spent in the denoising loop (vs encoding); lets the engine
    /// book that share as completion time rather than prompt time.
    fn take_block_denoise_time(&self) -> Option<std::time::Duration> {
        None
    }
}
