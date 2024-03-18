mod classifier;
mod config;
mod gemma;
mod llama;
mod mistral;
mod mixtral;
mod quantized_llama;

use candle_core::{DType, Device, Result, Tensor};
pub use config::XLoraConfig;
pub use gemma::XLoraModel as XLoraGemma;
pub use llama::XLoraLlama;
pub use mistral::XLoraModel as XLoraMistral;
pub use mixtral::XLoraModel as XLoraMixtral;
pub use quantized_llama::ModelWeights as XLoraModelWeights;

use crate::models::Cache;

use self::classifier::XLoraClassifier;

trait ScalingsMaker {
    fn get_classifier(&self) -> &XLoraClassifier;
    /// For dummy scalings
    fn dtype(&self) -> DType;
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        scalings: Tensor,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor>;
    fn get_cache(&self) -> &Cache;

    #[allow(clippy::too_many_arguments)]
    fn get_scalings(
        &mut self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        start_offsets_kernel: &Tensor,
        start_offsets_kernel_full: &Tensor,
        no_kv_cache: bool,
    ) -> Result<Tensor> {
        let (b_size, _) = input_ids_full.dims2()?;
        let (_, seq_len) = input_ids.dims2()?;

        let dummy_scalings = self.get_classifier().get_dummy_scalings(
            b_size,
            seq_len,
            input_ids.device(),
            self.dtype(),
        )?;
        // Using X-LoRA cache here
        let hidden_states = if no_kv_cache {
            let res = self.forward(
                input_ids_full,
                seqlen_offsets_full,
                start_offsets_kernel_full.clone(),
                dummy_scalings,
                true,
                no_kv_cache,
                Some(self.get_classifier().config.scaling_pass_value),
            )?;

            let mut new_cache = Vec::new();
            for _ in 0..self.get_cache().xlora_lock().len() {
                new_cache.push(Some((
                    Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
                    Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
                )));
            }
            *self.get_cache().lock() = new_cache.clone();

            res
        } else {
            self.forward(
                input_ids,
                seqlen_offsets,
                start_offsets_kernel.clone(),
                dummy_scalings,
                false,
                no_kv_cache,
                Some(self.get_classifier().config.scaling_pass_value),
            )?
        };

        let scalings = self.get_classifier().forward(hidden_states)?;
        Ok(scalings)
    }
}
